"""Immutable committed-epoch metrics and an atomic recoverable JSONL journal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Protocol, runtime_checkable

from refsite_mlip._atomic import commit_temporary_file

from .checkpoint import CheckpointMetadata, TrainingCheckpoint
from .checkpoint_manager import ManagedCheckpointResult
from .fit import FitEpochRecord
from .run_directory import ResumeRunLock, TrainingRunDirectory, canonical_runtime_json
from .resume_fit import validate_checkpoint_history


COMMITTED_EPOCH_METRICS_SCHEMA_VERSION = "refsite_training_metrics_v1"
COMMITTED_EPOCH_METRICS_CONVENTION_VERSION = "committed_epoch_metrics_v1"
METRICS_JOURNAL_CONFIG_SCHEMA_VERSION = (
    "refsite_training_metrics_journal_config_v1"
)
METRICS_JOURNAL_FILENAME = "metrics.jsonl"
EPOCH_COMMITTED_EVENT = "epoch_committed"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EPOCH_BASENAME_PATTERN = re.compile(r"^epoch_([0-9]+)\.pt$")


def _strict_keys(values: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{path} keys are invalid; missing={missing}, unknown={unknown}")


def _sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _positive_integer(name: str, value: Any) -> int:
    result = _nonnegative_integer(name, value)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_runtime_json(value).encode("utf-8") + b"\n"


@dataclass(frozen=True)
class CommittedEpochProvenance:
    """Path- and source-independent provenance shared by journal events."""

    initial_bundle_fingerprint: str
    training_configuration_fingerprint: str
    train_data_fingerprint: str
    validation_data_fingerprint: str
    template_fingerprints: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in (
            "initial_bundle_fingerprint",
            "training_configuration_fingerprint",
            "train_data_fingerprint",
            "validation_data_fingerprint",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if not isinstance(self.template_fingerprints, (tuple, list)):
            raise TypeError("template_fingerprints must be a sequence of pairs")
        canonical: list[tuple[str, str]] = []
        for index, item in enumerate(self.template_fingerprints):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(f"template_fingerprints[{index}] must be a pair")
            template_id, fingerprint = item
            if type(template_id) is not str or not template_id:
                raise ValueError("template IDs must be nonempty strings")
            canonical.append(
                (template_id, _sha256(fingerprint, name=f"template {template_id}"))
            )
        canonical.sort()
        if not canonical:
            raise ValueError("template_fingerprints must not be empty")
        if len({key for key, _ in canonical}) != len(canonical):
            raise ValueError("template_fingerprints contains duplicate template IDs")
        object.__setattr__(self, "template_fingerprints", tuple(canonical))

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_bundle_fingerprint": self.initial_bundle_fingerprint,
            "training_configuration_fingerprint": (
                self.training_configuration_fingerprint
            ),
            "train_data_fingerprint": self.train_data_fingerprint,
            "validation_data_fingerprint": self.validation_data_fingerprint,
            "template_fingerprints": {
                template_id: fingerprint
                for template_id, fingerprint in self.template_fingerprints
            },
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CommittedEpochProvenance":
        if not isinstance(values, Mapping):
            raise TypeError("committed epoch provenance requires a mapping")
        expected = {
            "initial_bundle_fingerprint",
            "training_configuration_fingerprint",
            "train_data_fingerprint",
            "validation_data_fingerprint",
            "template_fingerprints",
        }
        _strict_keys(values, expected, path="committed_epoch_provenance")
        templates = values["template_fingerprints"]
        if not isinstance(templates, Mapping):
            raise TypeError("template_fingerprints must be a mapping")
        return cls(
            initial_bundle_fingerprint=values["initial_bundle_fingerprint"],
            training_configuration_fingerprint=values[
                "training_configuration_fingerprint"
            ],
            train_data_fingerprint=values["train_data_fingerprint"],
            validation_data_fingerprint=values["validation_data_fingerprint"],
            template_fingerprints=tuple(templates.items()),
        )


def committed_epoch_provenance_from_checkpoint_metadata(
    metadata: CheckpointMetadata,
    *,
    initial_bundle_fingerprint: str,
) -> CommittedEpochProvenance:
    """Derive stable provenance without changing checkpoint schema."""

    if not isinstance(metadata, CheckpointMetadata):
        raise TypeError("metadata must be CheckpointMetadata")
    configuration = json.loads(canonical_runtime_json(metadata.resolved_configuration))
    fit = configuration.get("fit")
    if not isinstance(fit, dict) or "max_epochs" not in fit:
        raise ValueError("checkpoint resolved fit config is missing max_epochs")
    # This is the sole supported resume override and therefore not part of the
    # invariant per-epoch trajectory contract.
    del fit["max_epochs"]
    baseline = metadata.baseline_fit_metadata
    if baseline is not None:
        baseline = json.loads(canonical_runtime_json(baseline))
        # These two values identify the enclosing run/source rather than the
        # training trajectory.  Keeping them would make equivalent scratch
        # and bundle-source runs, or runs in different output directories,
        # produce different epoch events.
        baseline.pop("training_run_config_fingerprint", None)
        baseline.pop("initial_bundle_fingerprint", None)
    trajectory_configuration = {
        "checkpoint_configuration": configuration,
        "baseline_fit": baseline,
    }
    encoded = canonical_runtime_json(trajectory_configuration).encode("utf-8")
    configuration_fingerprint = hashlib.sha256(
        COMMITTED_EPOCH_METRICS_CONVENTION_VERSION.encode("ascii")
        + b"\n"
        + encoded
    ).hexdigest()
    return CommittedEpochProvenance(
        initial_bundle_fingerprint=initial_bundle_fingerprint,
        training_configuration_fingerprint=configuration_fingerprint,
        train_data_fingerprint=metadata.training_data.fingerprint,
        validation_data_fingerprint=metadata.validation_data.fingerprint,
        template_fingerprints=tuple(metadata.template_fingerprints.items()),
    )


@dataclass(frozen=True)
class MetricsJournalConfig:
    schema_version: str = METRICS_JOURNAL_CONFIG_SCHEMA_VERSION
    filename: str = METRICS_JOURNAL_FILENAME
    epoch_filename_width: int = 6

    def __post_init__(self) -> None:
        if self.schema_version != METRICS_JOURNAL_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported metrics journal config schema")
        if self.filename != METRICS_JOURNAL_FILENAME:
            raise ValueError("v1 metrics journal filename must be 'metrics.jsonl'")
        object.__setattr__(
            self,
            "epoch_filename_width",
            _positive_integer("epoch_filename_width", self.epoch_filename_width),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "filename": self.filename,
            "epoch_filename_width": self.epoch_filename_width,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MetricsJournalConfig":
        if not isinstance(values, Mapping):
            raise TypeError("metrics journal config requires a mapping")
        _strict_keys(
            values,
            {"schema_version", "filename", "epoch_filename_width"},
            path="metrics_journal_config",
        )
        return cls(**dict(values))


TermMetrics = tuple[float, float, float, int]


def _term_tuple(value: Any, *, path: str) -> TermMetrics:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise TypeError(
            f"{path} must contain numerator, denominator, mean, and valid_count"
        )
    numerator = _finite_nonnegative(f"{path}.numerator", value[0])
    denominator = _finite_nonnegative(f"{path}.denominator", value[1])
    mean = _finite_nonnegative(f"{path}.mean", value[2])
    valid_count = _nonnegative_integer(f"{path}.valid_count", value[3])
    if valid_count == 0:
        if (numerator, denominator, mean) != (0.0, 0.0, 0.0):
            raise ValueError(f"{path} without valid values must be exactly zero")
    else:
        if denominator <= 0.0:
            raise ValueError(f"{path} with valid values requires positive denominator")
        if mean != numerator / denominator:
            raise ValueError(f"{path}.mean must equal numerator / denominator")
    return numerator, denominator, mean, valid_count


def _term_to_dict(value: TermMetrics) -> dict[str, Any]:
    return {
        "numerator": value[0],
        "denominator": value[1],
        "mean": value[2],
        "valid_count": value[3],
    }


def _term_from_dict(values: Any, *, path: str) -> TermMetrics:
    if not isinstance(values, Mapping):
        raise TypeError(f"{path} must be a mapping")
    _strict_keys(
        values,
        {"numerator", "denominator", "mean", "valid_count"},
        path=path,
    )
    return _term_tuple(
        (
            values["numerator"],
            values["denominator"],
            values["mean"],
            values["valid_count"],
        ),
        path=path,
    )


@dataclass(frozen=True)
class CommittedEpochMetrics:
    """Tensor-free immutable projection of one fully committed fit epoch."""

    epoch_index: int
    global_step_start: int
    global_step_end: int
    successful_optimizer_steps: int
    training_metric_semantics: str
    validation_metric_semantics: str
    training_total_loss: float
    validation_total_loss: float
    training_energy: TermMetrics
    training_force: TermMetrics
    training_stress: TermMetrics
    validation_energy: TermMetrics
    validation_force: TermMetrics
    validation_stress: TermMetrics
    learning_rates_before_scheduler: tuple[float, ...]
    learning_rates_after_scheduler: tuple[float, ...]
    monitored_metric_name: str
    monitored_metric_value: float
    monitored_metric_mode: str
    is_best: bool
    should_stop: bool
    best_epoch: int
    best_value: float
    bad_validation_count: int
    epoch_checkpoint_basename: str
    latest_checkpoint_basename: str
    best_checkpoint_basename: str | None
    provenance: CommittedEpochProvenance
    schema_version: str = field(
        default=COMMITTED_EPOCH_METRICS_SCHEMA_VERSION, init=False
    )
    convention_version: str = field(
        default=COMMITTED_EPOCH_METRICS_CONVENTION_VERSION, init=False
    )
    event: str = field(default=EPOCH_COMMITTED_EVENT, init=False)

    def __post_init__(self) -> None:
        epoch = _nonnegative_integer("epoch_index", self.epoch_index)
        start = _nonnegative_integer("global_step_start", self.global_step_start)
        end = _nonnegative_integer("global_step_end", self.global_step_end)
        steps = _nonnegative_integer(
            "successful_optimizer_steps", self.successful_optimizer_steps
        )
        if end < start or end - start != steps:
            raise ValueError(
                "global step interval must equal successful optimizer steps"
            )
        object.__setattr__(self, "epoch_index", epoch)
        object.__setattr__(self, "global_step_start", start)
        object.__setattr__(self, "global_step_end", end)
        object.__setattr__(self, "successful_optimizer_steps", steps)
        for name in ("training_metric_semantics", "validation_metric_semantics"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("training_total_loss", "validation_total_loss"):
            object.__setattr__(
                self, name, _finite_nonnegative(name, getattr(self, name))
            )
        for name in (
            "training_energy",
            "training_force",
            "training_stress",
            "validation_energy",
            "validation_force",
            "validation_stress",
        ):
            object.__setattr__(
                self, name, _term_tuple(getattr(self, name), path=name)
            )
        for name in (
            "learning_rates_before_scheduler",
            "learning_rates_after_scheduler",
        ):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"{name} must be a nonempty sequence")
            object.__setattr__(
                self,
                name,
                tuple(
                    _finite_nonnegative(f"{name}[]", value) for value in values
                ),
            )
        if len(self.learning_rates_before_scheduler) != len(
            self.learning_rates_after_scheduler
        ):
            raise ValueError("learning-rate tuples must have equal length")
        if (
            type(self.monitored_metric_name) is not str
            or not self.monitored_metric_name
        ):
            raise ValueError("monitored_metric_name must be a nonempty string")
        object.__setattr__(
            self,
            "monitored_metric_value",
            _finite("monitored_metric_value", self.monitored_metric_value),
        )
        if self.monitored_metric_mode not in ("min", "max"):
            raise ValueError("monitored_metric_mode must be 'min' or 'max'")
        for name in ("is_best", "should_stop"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(
            self, "best_epoch", _nonnegative_integer("best_epoch", self.best_epoch)
        )
        object.__setattr__(
            self, "best_value", _finite("best_value", self.best_value)
        )
        object.__setattr__(
            self,
            "bad_validation_count",
            _nonnegative_integer(
                "bad_validation_count", self.bad_validation_count
            ),
        )
        if self.is_best and self.best_epoch != self.epoch_index:
            raise ValueError("is_best requires best_epoch to equal epoch_index")
        if self.best_epoch > self.epoch_index:
            raise ValueError("best_epoch must not be in the future")
        monitored_value = (
            self.validation_total_loss
            if self.monitored_metric_name == "total_loss"
            else {
                "energy": self.validation_energy,
                "force": self.validation_force,
                "stress": self.validation_stress,
            }.get(self.monitored_metric_name, (None, None, None, None))[2]
        )
        if monitored_value is None:
            raise ValueError(
                "monitored_metric_name must be total_loss, energy, force, or stress"
            )
        if self.monitored_metric_value != monitored_value:
            raise ValueError(
                "monitored_metric_value differs from validation metrics"
            )
        if self.is_best:
            if self.best_value != self.monitored_metric_value:
                raise ValueError("a best event must store its monitored value")
            if self.bad_validation_count != 0:
                raise ValueError("a best event must reset bad_validation_count")
            if self.should_stop:
                raise ValueError("a best event cannot request early stopping")
        self._validate_checkpoint_basenames()
        if not isinstance(self.provenance, CommittedEpochProvenance):
            raise TypeError("provenance must be CommittedEpochProvenance")

    def _validate_checkpoint_basenames(self) -> None:
        epoch_name = self.epoch_checkpoint_basename
        if type(epoch_name) is not str:
            raise TypeError("epoch_checkpoint_basename must be a string")
        match = _EPOCH_BASENAME_PATTERN.fullmatch(epoch_name)
        if match is None or int(match.group(1)) != self.epoch_index:
            raise ValueError(
                "epoch checkpoint basename does not match epoch_index"
            )
        if self.latest_checkpoint_basename != "latest.pt":
            raise ValueError("latest checkpoint basename must be 'latest.pt'")
        expected_best = "best.pt" if self.is_best else None
        if self.best_checkpoint_basename != expected_best:
            raise ValueError("best checkpoint basename differs from is_best")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "convention_version": self.convention_version,
            "event": self.event,
            "epoch_index": self.epoch_index,
            "global_step_start": self.global_step_start,
            "global_step_end": self.global_step_end,
            "successful_optimizer_steps": self.successful_optimizer_steps,
            "training": {
                "metric_semantics": self.training_metric_semantics,
                "total_loss": self.training_total_loss,
                "energy": _term_to_dict(self.training_energy),
                "force": _term_to_dict(self.training_force),
                "stress": _term_to_dict(self.training_stress),
            },
            "validation": {
                "metric_semantics": self.validation_metric_semantics,
                "total_loss": self.validation_total_loss,
                "energy": _term_to_dict(self.validation_energy),
                "force": _term_to_dict(self.validation_force),
                "stress": _term_to_dict(self.validation_stress),
            },
            "learning_rates_before_scheduler": list(
                self.learning_rates_before_scheduler
            ),
            "learning_rates_after_scheduler": list(
                self.learning_rates_after_scheduler
            ),
            "monitored_metric_name": self.monitored_metric_name,
            "monitored_metric_value": self.monitored_metric_value,
            "monitored_metric_mode": self.monitored_metric_mode,
            "is_best": self.is_best,
            "should_stop": self.should_stop,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "bad_validation_count": self.bad_validation_count,
            "epoch_checkpoint_basename": self.epoch_checkpoint_basename,
            "latest_checkpoint_basename": self.latest_checkpoint_basename,
            "best_checkpoint_basename": self.best_checkpoint_basename,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CommittedEpochMetrics":
        if not isinstance(values, Mapping):
            raise TypeError("committed epoch metrics requires a mapping")
        expected = {
            "schema_version",
            "convention_version",
            "event",
            "epoch_index",
            "global_step_start",
            "global_step_end",
            "successful_optimizer_steps",
            "training",
            "validation",
            "learning_rates_before_scheduler",
            "learning_rates_after_scheduler",
            "monitored_metric_name",
            "monitored_metric_value",
            "monitored_metric_mode",
            "is_best",
            "should_stop",
            "best_epoch",
            "best_value",
            "bad_validation_count",
            "epoch_checkpoint_basename",
            "latest_checkpoint_basename",
            "best_checkpoint_basename",
            "provenance",
        }
        _strict_keys(values, expected, path="committed_epoch_metrics")
        if values["schema_version"] != COMMITTED_EPOCH_METRICS_SCHEMA_VERSION:
            raise ValueError("unsupported committed epoch metrics schema")
        if (
            values["convention_version"]
            != COMMITTED_EPOCH_METRICS_CONVENTION_VERSION
        ):
            raise ValueError("unsupported committed epoch metrics convention")
        if values["event"] != EPOCH_COMMITTED_EVENT:
            raise ValueError("committed epoch event must be 'epoch_committed'")
        phases: dict[str, Mapping[str, Any]] = {}
        for phase in ("training", "validation"):
            phase_value = values[phase]
            if not isinstance(phase_value, Mapping):
                raise TypeError(f"{phase} metrics must be a mapping")
            _strict_keys(
                phase_value,
                {
                    "metric_semantics",
                    "total_loss",
                    "energy",
                    "force",
                    "stress",
                },
                path=phase,
            )
            phases[phase] = phase_value
        training = phases["training"]
        validation = phases["validation"]
        return cls(
            epoch_index=values["epoch_index"],
            global_step_start=values["global_step_start"],
            global_step_end=values["global_step_end"],
            successful_optimizer_steps=values["successful_optimizer_steps"],
            training_metric_semantics=training["metric_semantics"],
            validation_metric_semantics=validation["metric_semantics"],
            training_total_loss=training["total_loss"],
            validation_total_loss=validation["total_loss"],
            training_energy=_term_from_dict(
                training["energy"], path="training.energy"
            ),
            training_force=_term_from_dict(
                training["force"], path="training.force"
            ),
            training_stress=_term_from_dict(
                training["stress"], path="training.stress"
            ),
            validation_energy=_term_from_dict(
                validation["energy"], path="validation.energy"
            ),
            validation_force=_term_from_dict(
                validation["force"], path="validation.force"
            ),
            validation_stress=_term_from_dict(
                validation["stress"], path="validation.stress"
            ),
            learning_rates_before_scheduler=tuple(
                values["learning_rates_before_scheduler"]
            ),
            learning_rates_after_scheduler=tuple(
                values["learning_rates_after_scheduler"]
            ),
            monitored_metric_name=values["monitored_metric_name"],
            monitored_metric_value=values["monitored_metric_value"],
            monitored_metric_mode=values["monitored_metric_mode"],
            is_best=values["is_best"],
            should_stop=values["should_stop"],
            best_epoch=values["best_epoch"],
            best_value=values["best_value"],
            bad_validation_count=values["bad_validation_count"],
            epoch_checkpoint_basename=values["epoch_checkpoint_basename"],
            latest_checkpoint_basename=values["latest_checkpoint_basename"],
            best_checkpoint_basename=values["best_checkpoint_basename"],
            provenance=CommittedEpochProvenance.from_dict(values["provenance"]),
        )


@runtime_checkable
class EpochMetricsObserver(Protocol):
    def __call__(self, event: CommittedEpochMetrics) -> None: ...


def _epoch_term(value: Any) -> TermMetrics:
    return _term_tuple(
        (value.numerator, value.denominator, value.mean, value.valid_count),
        path="epoch_term",
    )


def _basename(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError(f"{name} path must be a nonempty string or None")
    return Path(value).name


def committed_epoch_metrics_from_record(
    record: FitEpochRecord,
    managed: ManagedCheckpointResult | None,
    *,
    selection_mode: str,
    provenance: CommittedEpochProvenance,
    epoch_filename_width: int = 6,
) -> CommittedEpochMetrics:
    """Project one committed checkpoint record into an immutable plain event."""

    if not isinstance(record, FitEpochRecord):
        raise TypeError("record must be a FitEpochRecord")
    if not isinstance(provenance, CommittedEpochProvenance):
        raise TypeError("provenance must be CommittedEpochProvenance")
    width = _positive_integer("epoch_filename_width", epoch_filename_width)
    if managed is None:
        epoch_name = f"epoch_{record.epoch_index:0{width}d}.pt"
        latest_name = "latest.pt"
        best_name = "best.pt" if record.decision.is_best else None
    else:
        if not isinstance(managed, ManagedCheckpointResult):
            raise TypeError("managed must be a ManagedCheckpointResult or None")
        if managed.epoch_index != record.epoch_index:
            raise ValueError("managed checkpoint epoch differs from record")
        if managed.global_step != record.training.global_step_end:
            raise ValueError("managed checkpoint global step differs from record")
        if managed.is_best != record.decision.is_best:
            raise ValueError("managed checkpoint is_best differs from record")
        if not managed.epoch_written or not managed.latest_written:
            raise ValueError(
                "observer requires committed epoch and latest checkpoints"
            )
        if managed.best_written != record.decision.is_best:
            raise ValueError("managed best write differs from selection decision")
        epoch_name = _basename(managed.epoch_path, name="epoch")
        latest_name = _basename(managed.latest_path, name="latest")
        best_name = _basename(managed.best_path, name="best")
        assert epoch_name is not None and latest_name is not None
    training = record.training
    validation = record.validation
    decision = record.decision
    state = record.selection_state_after_epoch
    if training.phase != "train" or validation.phase != "validation":
        raise ValueError("record must contain training and validation results")
    if (
        training.epoch_index != record.epoch_index
        or validation.epoch_index != record.epoch_index
    ):
        raise ValueError("record epoch indices are inconsistent")
    if (
        validation.global_step_start != training.global_step_end
        or validation.global_step_end != training.global_step_end
    ):
        raise ValueError(
            "validation global step must follow training without updates"
        )
    if validation.successful_optimizer_steps != 0:
        raise ValueError("validation must not contain optimizer steps")
    if tuple(record.learning_rates_used_for_training) != tuple(
        decision.learning_rates_before
    ):
        raise ValueError("training and pre-scheduler learning rates differ")
    if tuple(record.learning_rates_after_validation) != tuple(
        decision.learning_rates_after
    ):
        raise ValueError("record and decision post-scheduler learning rates differ")
    if (
        state.best_metric != decision.best_metric
        or state.best_epoch != decision.best_epoch
    ):
        raise ValueError("selection state and decision best values differ")
    if state.epochs_since_improvement != decision.epochs_since_improvement:
        raise ValueError(
            "selection state and decision bad validation counts differ"
        )
    if state.stopped_early != decision.should_stop:
        raise ValueError("selection state and decision stop values differ")
    if decision.metric_name == "total_loss":
        monitored_value = validation.total_loss
    else:
        monitored_term = getattr(validation, decision.metric_name, None)
        if monitored_term is None:
            raise ValueError(
                "selection decision names an unknown validation metric"
            )
        monitored_value = monitored_term.mean
    if decision.metric_value != monitored_value:
        raise ValueError(
            "selection decision metric differs from validation result"
        )
    if state.best_global_step != decision.best_global_step:
        raise ValueError(
            "selection state and decision best global steps differ"
        )
    if state.validation_events != decision.validation_events:
        raise ValueError(
            "selection state and decision validation event counts differ"
        )
    return CommittedEpochMetrics(
        epoch_index=record.epoch_index,
        global_step_start=training.global_step_start,
        global_step_end=training.global_step_end,
        successful_optimizer_steps=training.successful_optimizer_steps,
        training_metric_semantics=training.metric_semantics,
        validation_metric_semantics=validation.metric_semantics,
        training_total_loss=training.total_loss,
        validation_total_loss=validation.total_loss,
        training_energy=_epoch_term(training.energy),
        training_force=_epoch_term(training.force),
        training_stress=_epoch_term(training.stress),
        validation_energy=_epoch_term(validation.energy),
        validation_force=_epoch_term(validation.force),
        validation_stress=_epoch_term(validation.stress),
        learning_rates_before_scheduler=tuple(decision.learning_rates_before),
        learning_rates_after_scheduler=tuple(decision.learning_rates_after),
        monitored_metric_name=decision.metric_name,
        monitored_metric_value=decision.metric_value,
        monitored_metric_mode=selection_mode,
        is_best=decision.is_best,
        should_stop=decision.should_stop,
        best_epoch=decision.best_epoch,
        best_value=decision.best_metric,
        bad_validation_count=decision.epochs_since_improvement,
        epoch_checkpoint_basename=epoch_name,
        latest_checkpoint_basename=latest_name,
        best_checkpoint_basename=best_name,
        provenance=provenance,
    )


class MetricsJournalError(RuntimeError):
    """Structured journal failure with explicit pre/post-commit state."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        path: str | os.PathLike[str],
        epoch_index: int | None = None,
        last_valid_epoch: int | None = None,
        last_valid_event_count: int | None = None,
        last_valid_semantic_sha256: str | None = None,
        commit_completed: bool = False,
        original_error: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.stage = stage
        self.path = str(path)
        self.epoch_index = epoch_index
        self.last_valid_epoch = last_valid_epoch
        self.last_valid_event_count = last_valid_event_count
        self.last_valid_semantic_sha256 = last_valid_semantic_sha256
        self.commit_completed = commit_completed
        self.rollback_performed = False
        self.original_error = original_error
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        cause_text = (
            ""
            if original_error is None
            else "; cause="
            f"{self.original_exception_type}: {self.original_exception_message}"
        )
        super().__init__(
            f"[{reason_code}] stage={stage!r} path={self.path!r} "
            f"epoch_index={epoch_index!r} last_valid_epoch={last_valid_epoch!r} "
            f"commit_completed={commit_completed}: {message}{cause_text}"
        )


@dataclass(frozen=True)
class MetricsJournalSummary:
    metrics_journal: str
    metrics_event_count: int
    metrics_last_epoch: int | None
    metrics_semantic_sha256: str

    def __post_init__(self) -> None:
        if self.metrics_journal != METRICS_JOURNAL_FILENAME:
            raise ValueError("metrics_journal must be the relative v1 filename")
        object.__setattr__(
            self,
            "metrics_event_count",
            _nonnegative_integer(
                "metrics_event_count", self.metrics_event_count
            ),
        )
        if self.metrics_last_epoch is not None:
            object.__setattr__(
                self,
                "metrics_last_epoch",
                _nonnegative_integer(
                    "metrics_last_epoch", self.metrics_last_epoch
                ),
            )
        if self.metrics_event_count == 0:
            if self.metrics_last_epoch is not None:
                raise ValueError("empty journal must not have a last epoch")
        elif self.metrics_last_epoch != self.metrics_event_count - 1:
            raise ValueError("metrics event count and last epoch are inconsistent")
        object.__setattr__(
            self,
            "metrics_semantic_sha256",
            _sha256(
                self.metrics_semantic_sha256,
                name="metrics_semantic_sha256",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics_journal": self.metrics_journal,
            "metrics_event_count": self.metrics_event_count,
            "metrics_last_epoch": self.metrics_last_epoch,
            "metrics_semantic_sha256": self.metrics_semantic_sha256,
        }


def _strict_json_object(encoded: str, *, line_number: int) -> dict[str, Any]:
    def reject_constant(value: str):
        raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")

    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        encoded,
        parse_constant=reject_constant,
        object_pairs_hook=strict_object,
    )
    if not isinstance(value, dict):
        raise TypeError(
            f"metrics journal line {line_number} must be a JSON object"
        )
    return value


class MetricsJournal:
    """Strict append-only semantics implemented by atomic full-file rewrites."""

    def __init__(
        self,
        directory: TrainingRunDirectory,
        lock: ResumeRunLock | None,
        provenance: CommittedEpochProvenance,
        config: MetricsJournalConfig = MetricsJournalConfig(),
    ) -> None:
        if not isinstance(directory, TrainingRunDirectory):
            raise TypeError("directory must be a TrainingRunDirectory")
        if lock is not None and not isinstance(lock, ResumeRunLock):
            raise TypeError("lock must be a ResumeRunLock or None")
        if not isinstance(provenance, CommittedEpochProvenance):
            raise TypeError("provenance must be CommittedEpochProvenance")
        if not isinstance(config, MetricsJournalConfig):
            raise TypeError("config must be a MetricsJournalConfig")
        self.directory = directory
        self.lock = lock
        self.provenance = provenance
        self.config = config
        self.path = directory.root / config.filename

    def _require_owned_lock(self) -> ResumeRunLock:
        if self.lock is None:
            raise MetricsJournalError(
                "METRICS_JOURNAL_LOCK_REQUIRED",
                "a live owned run lock is required for journal mutation",
                stage="metrics_journal.lock",
                path=self.path,
            )
        try:
            self.lock.validate_owned(self.directory.resume_lock_path)
        except Exception as error:
            raise MetricsJournalError(
                "METRICS_JOURNAL_LOCK_NOT_OWNED",
                "journal mutation requires ownership of the run lock",
                stage="metrics_journal.lock",
                path=self.path,
                original_error=error,
            ) from error
        return self.lock

    def _read(
        self,
    ) -> tuple[
        tuple[CommittedEpochMetrics, ...],
        bytes,
        tuple[int, int] | None,
    ]:
        path = self.path
        if path.is_symlink():
            raise MetricsJournalError(
                "METRICS_JOURNAL_SYMLINK_REJECTED",
                "metrics journal must not be a symbolic link",
                stage="metrics_journal.load",
                path=path,
            )
        if not path.exists():
            return (), b"", None
        if not path.is_file():
            raise MetricsJournalError(
                "INVALID_METRICS_JOURNAL_SOURCE",
                "metrics journal must be a regular file",
                stage="metrics_journal.load",
                path=path,
            )
        events: list[CommittedEpochMetrics] = []
        raw = b""
        valid_prefix_length = 0
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("metrics journal is not a regular file")
                chunks = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
            finally:
                os.close(descriptor)
            raw = b"".join(chunks)
            if not raw:
                raise ValueError("an existing metrics journal must not be empty")
            lines = raw.splitlines(keepends=True)
            for index, line in enumerate(lines, start=1):
                if not line.endswith(b"\n") or line.endswith(b"\r\n"):
                    raise ValueError(
                        "metrics journal lines must end with exactly one LF"
                    )
                payload = line[:-1].decode("utf-8", errors="strict")
                if not payload:
                    raise ValueError("metrics journal contains an empty line")
                value = _strict_json_object(payload, line_number=index)
                event = CommittedEpochMetrics.from_dict(value)
                if _canonical_bytes(event.to_dict()) != line:
                    raise ValueError(
                        "metrics journal line is not canonical JSON"
                    )
                self._validate_sequence(tuple(events) + (event,))
                events.append(event)
                valid_prefix_length += len(line)
            return (
                tuple(events),
                raw,
                (int(metadata.st_dev), int(metadata.st_ino)),
            )
        except MetricsJournalError:
            raise
        except Exception as error:
            last_valid_epoch = self._last_valid_epoch(tuple(events))
            valid_prefix = raw[:valid_prefix_length]
            raise MetricsJournalError(
                "METRICS_JOURNAL_LOAD_FAILED",
                "metrics journal could not be parsed and validated",
                stage="metrics_journal.load",
                path=path,
                last_valid_epoch=last_valid_epoch,
                last_valid_event_count=len(events),
                last_valid_semantic_sha256=hashlib.sha256(
                    valid_prefix
                ).hexdigest(),
                original_error=error,
            ) from error

    def _last_valid_epoch(
        self, events: tuple[CommittedEpochMetrics, ...]
    ) -> int | None:
        valid: list[CommittedEpochMetrics] = []
        for event in events:
            try:
                self._validate_sequence(tuple(valid) + (event,))
            except Exception:
                break
            valid.append(event)
        return None if not valid else valid[-1].epoch_index

    def _validate_sequence(
        self, events: tuple[CommittedEpochMetrics, ...]
    ) -> None:
        previous: CommittedEpochMetrics | None = None
        for event in events:
            if event.provenance != self.provenance:
                raise ValueError(
                    "metrics journal provenance diverges from this run"
                )
            if previous is None:
                if event.epoch_index != 0 or event.global_step_start != 0:
                    raise ValueError(
                        "metrics journal must begin at epoch/global step zero"
                    )
                if not event.is_best:
                    raise ValueError(
                        "the first committed validation must establish the best metric"
                    )
            else:
                if previous.should_stop:
                    raise ValueError(
                        "metrics journal must terminate after should_stop"
                    )
                if event.epoch_index != previous.epoch_index + 1:
                    raise ValueError(
                        "metrics journal epoch is duplicate, stale, or has a gap"
                    )
                if event.global_step_start != previous.global_step_end:
                    raise ValueError(
                        "metrics journal global-step continuity mismatch"
                    )
                if (
                    event.learning_rates_before_scheduler
                    != previous.learning_rates_after_scheduler
                ):
                    raise ValueError(
                        "metrics journal learning-rate continuity mismatch"
                    )
                if (
                    event.training_metric_semantics
                    != previous.training_metric_semantics
                    or event.validation_metric_semantics
                    != previous.validation_metric_semantics
                    or event.monitored_metric_name
                    != previous.monitored_metric_name
                    or event.monitored_metric_mode
                    != previous.monitored_metric_mode
                ):
                    raise ValueError(
                        "metrics journal metric conventions changed within the run"
                    )
                if not event.is_best:
                    if (
                        event.best_epoch != previous.best_epoch
                        or event.best_value != previous.best_value
                    ):
                        raise ValueError(
                            "non-best event changed the retained best state"
                        )
                    if (
                        event.bad_validation_count
                        != previous.bad_validation_count + 1
                    ):
                        raise ValueError(
                            "non-best event bad-validation count is not contiguous"
                        )
            previous = event

    def append(self, event: CommittedEpochMetrics) -> MetricsJournalSummary:
        if not isinstance(event, CommittedEpochMetrics):
            raise TypeError("event must be CommittedEpochMetrics")
        if event.provenance != self.provenance:
            raise MetricsJournalError(
                "METRICS_JOURNAL_PROVENANCE_MISMATCH",
                "event provenance differs from journal provenance",
                stage="metrics_journal.validate",
                path=self.path,
                epoch_index=event.epoch_index,
            )
        self._require_owned_lock()
        existing, prefix, identity = self._read()
        try:
            self._validate_sequence(existing + (event,))
        except Exception as error:
            raise MetricsJournalError(
                "METRICS_JOURNAL_SEQUENCE_MISMATCH",
                "event is not the exact next committed epoch",
                stage="metrics_journal.validate",
                path=self.path,
                epoch_index=event.epoch_index,
                last_valid_epoch=(
                    None if not existing else existing[-1].epoch_index
                ),
                last_valid_event_count=len(existing),
                last_valid_semantic_sha256=hashlib.sha256(prefix).hexdigest(),
                original_error=error,
            ) from error
        encoded = prefix + _canonical_bytes(event.to_dict())
        self._atomic_rewrite(
            encoded,
            expected_prefix=prefix,
            # Bind the commit mode to the journal that was actually opened,
            # not to a second racy exists() observation.  If an existing
            # journal disappears after _read(), the identity recheck below
            # must reject the mutation instead of silently recreating it.
            target_existed=identity is not None,
            expected_identity=identity,
            epoch_index=event.epoch_index,
            last_valid_epoch=(
                None if not existing else existing[-1].epoch_index
            ),
        )
        return self.summary()

    def _atomic_rewrite(
        self,
        encoded: bytes,
        *,
        expected_prefix: bytes,
        target_existed: bool,
        expected_identity: tuple[int, int] | None,
        epoch_index: int,
        last_valid_epoch: int | None,
    ) -> None:
        descriptor: int | None = None
        temporary: Path | None = None
        committed = False
        try:
            descriptor, name = tempfile.mkstemp(
                dir=str(self.directory.root),
                prefix=f".{self.config.filename}.",
                suffix=".tmp",
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._require_owned_lock()
            if self.path.is_symlink():
                raise ValueError(
                    "metrics journal target became a symbolic link"
                )
            if target_existed:
                _, current, current_identity = self._read()
                if (
                    current != expected_prefix
                    or current_identity != expected_identity
                ):
                    raise ValueError(
                        "metrics journal changed before atomic commit"
                    )
            elif self.path.exists():
                raise FileExistsError(
                    "metrics journal appeared before initial commit"
                )
            commit_temporary_file(
                temporary, self.path, overwrite=target_existed
            )
            committed = True
            temporary = None
            directory_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            directory_descriptor = os.open(
                self.directory.root, directory_flags
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except MetricsJournalError:
            raise
        except Exception as error:
            raise MetricsJournalError(
                "METRICS_JOURNAL_ATOMIC_WRITE_FAILED",
                "same-directory atomic journal rewrite failed",
                stage=(
                    "metrics_journal.directory_fsync"
                    if committed
                    else "metrics_journal.commit"
                ),
                path=self.path,
                epoch_index=epoch_index,
                last_valid_epoch=last_valid_epoch,
                last_valid_event_count=(
                    epoch_index + 1
                    if committed
                    else (0 if last_valid_epoch is None else last_valid_epoch + 1)
                ),
                last_valid_semantic_sha256=hashlib.sha256(
                    encoded if committed else expected_prefix
                ).hexdigest(),
                commit_completed=committed,
                original_error=error,
            ) from error
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def inspect_checkpoint(
        self, checkpoint: TrainingCheckpoint
    ) -> tuple[CommittedEpochMetrics, ...]:
        """Return the missing history suffix without changing any file."""

        if not isinstance(checkpoint, TrainingCheckpoint):
            raise TypeError("checkpoint must be a TrainingCheckpoint")
        expected_provenance = committed_epoch_provenance_from_checkpoint_metadata(
            checkpoint.metadata,
            initial_bundle_fingerprint=(
                self.provenance.initial_bundle_fingerprint
            ),
        )
        if expected_provenance != self.provenance:
            raise MetricsJournalError(
                "METRICS_JOURNAL_CHECKPOINT_PROVENANCE_MISMATCH",
                "checkpoint metadata differs from journal provenance",
                stage="metrics_journal.reconcile",
                path=self.path,
            )
        actual: tuple[CommittedEpochMetrics, ...] = ()
        matched_last: int | None = None
        try:
            history = validate_checkpoint_history(
                checkpoint, allow_stopped_early=True
            )
            selection = checkpoint.metadata.resolved_configuration[
                "model_selection"
            ]
            if not isinstance(selection, Mapping):
                raise TypeError(
                    "checkpoint model_selection config must be a mapping"
                )
            mode = selection.get("mode")
            expected = tuple(
                committed_epoch_metrics_from_record(
                    record,
                    None,
                    selection_mode=mode,
                    provenance=self.provenance,
                    epoch_filename_width=self.config.epoch_filename_width,
                )
                for record in history
            )
            actual, _, _ = self._read()
            for index, event in enumerate(actual[: len(expected)]):
                if event != expected[index]:
                    raise ValueError(
                        "metrics journal diverges from checkpoint history at "
                        f"epoch {index}"
                    )
                matched_last = event.epoch_index
            if len(actual) > len(expected):
                raise ValueError(
                    "metrics journal contains future events beyond checkpoint"
                )
            return expected[len(actual) :]
        except MetricsJournalError:
            raise
        except Exception as error:
            raise MetricsJournalError(
                "METRICS_JOURNAL_HISTORY_DIVERGENCE",
                "journal is not an exact prefix of checkpoint history",
                stage="metrics_journal.reconcile",
                path=self.path,
                last_valid_epoch=matched_last,
                original_error=error,
            ) from error

    def reconcile_checkpoint(
        self, checkpoint: TrainingCheckpoint
    ) -> MetricsJournalSummary:
        """Atomically append only the checkpoint history suffix."""

        self._require_owned_lock()
        for event in self.inspect_checkpoint(checkpoint):
            self.append(event)
        return self.summary()

    def summary(self) -> MetricsJournalSummary:
        events, encoded, _ = self._read()
        return MetricsJournalSummary(
            metrics_journal=self.config.filename,
            metrics_event_count=len(events),
            metrics_last_epoch=(
                None if not events else events[-1].epoch_index
            ),
            metrics_semantic_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def __call__(self, event: CommittedEpochMetrics) -> None:
        self.append(event)


__all__ = [
    "COMMITTED_EPOCH_METRICS_CONVENTION_VERSION",
    "COMMITTED_EPOCH_METRICS_SCHEMA_VERSION",
    "EPOCH_COMMITTED_EVENT",
    "METRICS_JOURNAL_CONFIG_SCHEMA_VERSION",
    "METRICS_JOURNAL_FILENAME",
    "CommittedEpochMetrics",
    "CommittedEpochProvenance",
    "EpochMetricsObserver",
    "MetricsJournal",
    "MetricsJournalConfig",
    "MetricsJournalError",
    "MetricsJournalSummary",
    "committed_epoch_metrics_from_record",
    "committed_epoch_provenance_from_checkpoint_metadata",
]
