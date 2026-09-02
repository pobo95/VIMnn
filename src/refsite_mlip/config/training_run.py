"""Canonical, read-only preflight configuration for fresh training runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, replace
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import torch

from refsite_mlip.data import (
    EXTXYZ_LOADER_CONVENTION_VERSION,
    EXTXYZ_UNIT_CONVENTION_VERSION,
    ExtXYZLoadConfig,
    ExtXYZLoadError,
    StructureSample,
    TemplateRegistry,
    load_extxyz_samples,
)
from refsite_mlip.data.extxyz import (
    _extract_component_mask,
    _extract_label,
    _independent_stress_mask,
)
from refsite_mlip.models import (
    ModelBundleError,
    PotentialConfig,
    load_reference_site_model_bundle,
)
from refsite_mlip.training import (
    AtomicBaselineConfig,
    CheckpointedFitConfig,
    FitConfig,
    LossConfig,
    ModelSelectionConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    fit_atomic_baseline,
)
from refsite_mlip.transport import TRAIN_FIXED

from .radii import (
    DerivedInteractionRadii,
    InteractionRadiusConfig,
    RadiusConfigError,
    validate_radius_artifact_compatibility,
    validate_radius_model_compatibility,
)


TRAINING_RUN_CONFIG_SCHEMA_VERSION = "refsite_training_run_config_v1"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "initial_bundle",
        "radii",
        "data",
        "runtime",
        "loss",
        "baseline",
        "optimizer",
        "train_step",
        "validation_step",
        "scheduler",
        "selection",
        "fit",
        "checkpointed_fit",
        "output_directory",
    }
)
_RADIUS_SHORT_KEYS = frozenset(
    {"r_ot", "r_mp", "ot_switch_width", "ot_skin", "mp_skin"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEVICE = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")


class TrainingRunConfigError(ValueError):
    """Structured schema, path, data, or compatibility preflight failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        config_path: str | None = None,
        field: str | None = None,
        split: str | None = None,
        frame_index: int | None = None,
        sample_id: str | None = None,
        template_id: str | None = None,
        expected: Any = None,
        actual: Any = None,
        original_reason_code: str | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("reason_code must be a nonempty string")
        if not isinstance(message, str) or not message:
            raise ValueError("message must be a nonempty string")
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a nonempty string")
        self.reason_code = reason_code
        self.stage = stage
        self.config_path = config_path
        self.field = field
        self.split = split
        self.frame_index = frame_index
        self.sample_id = sample_id
        self.template_id = template_id
        self.expected = expected
        self.actual = actual
        self.original_reason_code = original_reason_code
        self.original_error = original_error
        self.message = message
        context = []
        for name in (
            "config_path",
            "field",
            "split",
            "frame_index",
            "sample_id",
            "template_id",
            "expected",
            "actual",
            "original_reason_code",
        ):
            value = getattr(self, name)
            if value is not None:
                context.append(f"{name}={value!r}")
        suffix = "" if not context else " " + " ".join(context)
        super().__init__(f"[{reason_code}] stage={stage!r}{suffix} {message}")


def _error(
    reason_code: str,
    message: str,
    *,
    stage: str,
    config_path: str | None = None,
    field: str | None = None,
    split: str | None = None,
    frame_index: int | None = None,
    sample_id: str | None = None,
    template_id: str | None = None,
    expected: Any = None,
    actual: Any = None,
    original_reason_code: str | None = None,
    original_error: BaseException | None = None,
) -> TrainingRunConfigError:
    return TrainingRunConfigError(
        reason_code,
        message,
        stage=stage,
        config_path=config_path,
        field=field,
        split=split,
        frame_index=frame_index,
        sample_id=sample_id,
        template_id=template_id,
        expected=expected,
        actual=actual,
        original_reason_code=original_reason_code,
        original_error=original_error,
    )


def _strict_keys(
    value: Any,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(
            "INVALID_CONFIG_SECTION",
            f"{name} must be a JSON object",
            stage="config.schema",
            field=name,
            actual=type(value).__name__,
        )
    keys = frozenset(value)
    unknown = keys - expected
    missing = expected - keys
    if unknown:
        raise _error(
            "UNKNOWN_CONFIG_KEY",
            f"{name} contains unknown keys",
            stage="config.schema",
            field=", ".join(sorted(repr(key) for key in unknown)),
        )
    if missing:
        raise _error(
            "MISSING_CONFIG_KEY",
            f"{name} is missing required keys",
            stage="config.schema",
            field=", ".join(sorted(missing)),
        )
    return value


def _path_text(value: Any, *, field_name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise _error(
            "INVALID_CONFIG_PATH",
            "path must be a nonempty JSON string without NUL",
            stage="config.validation",
            field=field_name,
            actual=value,
        )
    return value


@dataclass(frozen=True)
class TrainingDataSourceConfig:
    """One ordered extxyz source with an explicit template selection rule."""

    path: str
    template_id: str | None = None
    template_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path_text(self.path, field_name="path"))
        for name in ("template_id", "template_key"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise _error(
                    "INVALID_TEMPLATE_SELECTOR",
                    f"{name} must be a nonempty string or null",
                    stage="config.validation",
                    field=name,
                    actual=value,
                )
        if (self.template_id is None) == (self.template_key is None):
            raise _error(
                "CONFLICTING_TEMPLATE_SELECTOR",
                "exactly one of template_id or template_key is required",
                stage="config.validation",
                field="template_id,template_key",
            )

    def to_dict(self) -> dict[str, Any]:
        result = {"path": self.path}
        if self.template_id is not None:
            result["template_id"] = self.template_id
        else:
            result["template_key"] = self.template_key
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingDataSourceConfig":
        if not isinstance(value, Mapping):
            raise _error(
                "INVALID_DATA_SOURCE",
                "data source must be a JSON object",
                stage="config.schema",
                actual=type(value).__name__,
            )
        keys = frozenset(value)
        allowed = frozenset({"path", "template_id", "template_key"})
        unknown = keys - allowed
        if unknown:
            raise _error(
                "UNKNOWN_CONFIG_KEY",
                "data source contains unknown keys",
                stage="config.schema",
                field=", ".join(sorted(repr(key) for key in unknown)),
            )
        if "path" not in keys:
            raise _error(
                "MISSING_CONFIG_KEY",
                "data source is missing path",
                stage="config.schema",
                field="path",
            )
        selectors = keys & {"template_id", "template_key"}
        if len(selectors) != 1:
            raise _error(
                "CONFLICTING_TEMPLATE_SELECTOR",
                "data source requires exactly one template selector",
                stage="config.schema",
                field="template_id,template_key",
            )
        return cls(
            path=value["path"],
            template_id=value.get("template_id"),
            template_key=value.get("template_key"),
        )


def _source_sequence(value: Any, *, split: str) -> tuple[TrainingDataSourceConfig, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _error(
            "INVALID_DATA_SEQUENCE",
            f"data.{split} must be a deterministic JSON array",
            stage="config.schema",
            field=f"data.{split}",
            split=split,
        )
    result = tuple(TrainingDataSourceConfig.from_dict(item) for item in value)
    if not result:
        raise _error(
            "EMPTY_DATA_SPLIT",
            f"data.{split} must contain at least one source",
            stage="config.validation",
            field=f"data.{split}",
            split=split,
        )
    return result


@dataclass(frozen=True)
class TrainingDataConfig:
    train: tuple[TrainingDataSourceConfig, ...]
    validation: tuple[TrainingDataSourceConfig, ...]
    batch_size: int = 4
    shuffle: bool = False

    def __post_init__(self) -> None:
        def sources(values):
            if not isinstance(values, Sequence) or isinstance(
                values, (str, bytes, bytearray)
            ):
                raise _error(
                    "INVALID_DATA_SEQUENCE",
                    "training data sources must be a deterministic sequence",
                    stage="config.validation",
                    field="data",
                )
            return tuple(
                item
                if isinstance(item, TrainingDataSourceConfig)
                else TrainingDataSourceConfig.from_dict(item)
                for item in values
            )

        train = sources(self.train)
        validation = sources(self.validation)
        if not train or not validation:
            raise _error(
                "EMPTY_DATA_SPLIT",
                "train and validation source sequences must be nonempty",
                stage="config.validation",
                field="data",
            )
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "validation", validation)
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, Integral)
            or int(self.batch_size) <= 0
        ):
            raise _error(
                "INVALID_BATCH_SIZE",
                "batch_size must be a positive integer and bool is not accepted",
                stage="config.validation",
                field="data.batch_size",
                actual=self.batch_size,
            )
        object.__setattr__(self, "batch_size", int(self.batch_size))
        if type(self.shuffle) is not bool:
            raise _error(
                "INVALID_SHUFFLE",
                "shuffle must be a bool",
                stage="config.validation",
                field="data.shuffle",
                actual=self.shuffle,
            )
        if self.shuffle:
            raise _error(
                "UNSUPPORTED_SHUFFLE",
                "schema v1 requires shuffle=false because no deterministic epoch plan exists",
                stage="config.validation",
                field="data.shuffle",
                actual=True,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": [item.to_dict() for item in self.train],
            "validation": [item.to_dict() for item in self.validation],
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingDataConfig":
        payload = _strict_keys(
            value,
            frozenset({"train", "validation", "batch_size", "shuffle"}),
            name="data",
        )
        return cls(
            train=_source_sequence(payload["train"], split="train"),
            validation=_source_sequence(payload["validation"], split="validation"),
            batch_size=payload["batch_size"],
            shuffle=payload["shuffle"],
        )


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if type(self.device) is not str or _DEVICE.fullmatch(self.device) is None:
            raise _error(
                "INVALID_RUNTIME_DEVICE",
                "device must be cpu, cuda, or cuda:N",
                stage="config.validation",
                field="runtime.device",
                actual=self.device,
            )
        if self.dtype not in ("float32", "float64"):
            raise _error(
                "INVALID_RUNTIME_DTYPE",
                "dtype must be float32 or float64",
                stage="config.validation",
                field="runtime.dtype",
                actual=self.dtype,
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == "float32" else torch.float64

    def to_dict(self) -> dict[str, str]:
        return {"device": self.device, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRuntimeConfig":
        payload = _strict_keys(
            value, frozenset({"device", "dtype"}), name="runtime"
        )
        return cls(device=payload["device"], dtype=payload["dtype"])


def _parse_existing_config(name: str, cls, value: Any):
    expected = frozenset(item.name for item in dataclass_fields(cls))
    payload = _strict_keys(value, expected, name=name)
    if name in ("train_step", "validation_step"):
        solver_path = payload["solver_path"]
        if solver_path != TRAIN_FIXED:
            reason = (
                "UNSUPPORTED_TRAINING_SOLVER"
                if name == "train_step"
                else "UNSUPPORTED_VALIDATION_SOLVER"
            )
            raise _error(
                reason,
                f"{name} solver_path must be TRAIN_FIXED",
                stage="config.cross_validation",
                field=f"{name}.solver_path",
                expected=TRAIN_FIXED,
                actual=solver_path,
            )
    try:
        return cls.from_dict(payload)
    except TrainingRunConfigError:
        raise
    except Exception as error:
        raise _error(
            "INVALID_TRAINING_CONFIG",
            f"existing {cls.__name__} validation failed",
            stage="config.validation",
            field=name,
            original_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error


def _parse_radii(value: Any) -> InteractionRadiusConfig:
    if not isinstance(value, Mapping):
        raise _error(
            "INVALID_CONFIG_SECTION",
            "radii must be a JSON object",
            stage="config.schema",
            field="radii",
            actual=type(value).__name__,
        )
    try:
        metadata = {
            "schema_version",
            "convention_version",
            "length_unit",
        }
        if frozenset(value) & metadata:
            return InteractionRadiusConfig.from_dict(value)
        unknown = frozenset(value) - _RADIUS_SHORT_KEYS
        missing = {"r_ot", "r_mp"} - frozenset(value)
        if unknown:
            raise _error(
                "UNKNOWN_CONFIG_KEY",
                "radii contains unknown or derived keys",
                stage="config.schema",
                field=", ".join(sorted(repr(key) for key in unknown)),
            )
        if missing:
            raise _error(
                "MISSING_CONFIG_KEY",
                "radii shorthand requires r_ot and r_mp",
                stage="config.schema",
                field=", ".join(sorted(missing)),
            )
        defaults = InteractionRadiusConfig()
        return InteractionRadiusConfig(
            r_ot=value["r_ot"],
            r_mp=value["r_mp"],
            ot_switch_width=value.get(
                "ot_switch_width", defaults.ot_switch_width
            ),
            ot_skin=value.get("ot_skin", defaults.ot_skin),
            mp_skin=value.get("mp_skin", defaults.mp_skin),
        )
    except TrainingRunConfigError:
        raise
    except RadiusConfigError as error:
        raise _error(
            error.reason_code,
            "InteractionRadiusConfig validation failed",
            stage="config.radii",
            field=error.field,
            expected=error.expected,
            actual=error.actual,
            original_reason_code=error.reason_code,
            original_error=error,
        ) from error


@dataclass(frozen=True)
class TrainingRunConfig:
    schema_version: str
    initial_bundle: str
    radii: InteractionRadiusConfig
    data: TrainingDataConfig
    runtime: TrainingRuntimeConfig
    loss: LossConfig
    baseline: AtomicBaselineConfig
    optimizer: OptimizerConfig
    train_step: TrainStepConfig
    validation_step: ValidationStepConfig
    scheduler: SchedulerConfig
    selection: ModelSelectionConfig
    fit: FitConfig
    checkpointed_fit: CheckpointedFitConfig
    output_directory: str
    source_path: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_RUN_CONFIG_SCHEMA_VERSION:
            raise _error(
                "UNSUPPORTED_TRAINING_RUN_SCHEMA",
                "unsupported training-run config schema",
                stage="config.schema",
                field="schema_version",
                expected=TRAINING_RUN_CONFIG_SCHEMA_VERSION,
                actual=self.schema_version,
            )
        object.__setattr__(
            self,
            "initial_bundle",
            _path_text(self.initial_bundle, field_name="initial_bundle"),
        )
        object.__setattr__(
            self,
            "output_directory",
            _path_text(self.output_directory, field_name="output_directory"),
        )
        for name, cls in (
            ("radii", InteractionRadiusConfig),
            ("data", TrainingDataConfig),
            ("runtime", TrainingRuntimeConfig),
            ("loss", LossConfig),
            ("baseline", AtomicBaselineConfig),
            ("optimizer", OptimizerConfig),
            ("train_step", TrainStepConfig),
            ("validation_step", ValidationStepConfig),
            ("scheduler", SchedulerConfig),
            ("selection", ModelSelectionConfig),
            ("fit", FitConfig),
            ("checkpointed_fit", CheckpointedFitConfig),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be a {cls.__name__}")
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))
        validate_training_run_config(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "initial_bundle": self.initial_bundle,
            "radii": self.radii.to_dict(),
            "data": self.data.to_dict(),
            "runtime": self.runtime.to_dict(),
            "loss": self.loss.to_dict(),
            "baseline": self.baseline.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "train_step": self.train_step.to_dict(),
            "validation_step": self.validation_step.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "selection": self.selection.to_dict(),
            "fit": self.fit.to_dict(),
            "checkpointed_fit": self.checkpointed_fit.to_dict(),
            "output_directory": self.output_directory,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRunConfig":
        payload = _strict_keys(value, _TOP_LEVEL_KEYS, name="training_run")
        return cls(
            schema_version=payload["schema_version"],
            initial_bundle=payload["initial_bundle"],
            radii=_parse_radii(payload["radii"]),
            data=TrainingDataConfig.from_dict(payload["data"]),
            runtime=TrainingRuntimeConfig.from_dict(payload["runtime"]),
            loss=_parse_existing_config("loss", LossConfig, payload["loss"]),
            baseline=_parse_existing_config(
                "baseline", AtomicBaselineConfig, payload["baseline"]
            ),
            optimizer=_parse_existing_config(
                "optimizer", OptimizerConfig, payload["optimizer"]
            ),
            train_step=_parse_existing_config(
                "train_step", TrainStepConfig, payload["train_step"]
            ),
            validation_step=_parse_existing_config(
                "validation_step",
                ValidationStepConfig,
                payload["validation_step"],
            ),
            scheduler=_parse_existing_config(
                "scheduler", SchedulerConfig, payload["scheduler"]
            ),
            selection=_parse_existing_config(
                "selection", ModelSelectionConfig, payload["selection"]
            ),
            fit=_parse_existing_config("fit", FitConfig, payload["fit"]),
            checkpointed_fit=_parse_existing_config(
                "checkpointed_fit",
                CheckpointedFitConfig,
                payload["checkpointed_fit"],
            ),
            output_directory=payload["output_directory"],
        )

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_json(self) -> str:
        return self.canonical_json()

    @classmethod
    def from_json(cls, value: str) -> "TrainingRunConfig":
        if type(value) is not str:
            raise _error(
                "INVALID_CONFIG_JSON",
                "training-run JSON must be a string",
                stage="config.parse",
                actual=type(value).__name__,
            )

        def reject_constant(constant: str):
            raise _error(
                "NONFINITE_CONFIG_VALUE",
                "NaN and Infinity are forbidden in training-run JSON",
                stage="config.parse",
                actual=constant,
            )

        def strict_object(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise _error(
                        "CONFLICTING_CONFIG_KEY",
                        "duplicate JSON object key is forbidden",
                        stage="config.parse",
                        field=key,
                    )
                result[key] = item
            return result

        try:
            payload = json.loads(
                value,
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
        except TrainingRunConfigError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise _error(
                "INVALID_CONFIG_JSON",
                "training-run JSON could not be decoded",
                stage="config.parse",
                original_error=error,
            ) from error
        return cls.from_dict(payload)

    @property
    def config_fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self.config_fingerprint

    @property
    def content_fingerprint(self) -> str:
        """Compatibility alias for the canonical config SHA-256."""

        return self.config_fingerprint


def validate_training_run_config(config: TrainingRunConfig) -> TrainingRunConfig:
    """Validate cross-config contracts without touching external resources."""

    if not isinstance(config, TrainingRunConfig):
        raise TypeError("config must be a TrainingRunConfig")
    if config.train_step.solver_path != TRAIN_FIXED:
        raise _error(
            "UNSUPPORTED_TRAINING_SOLVER",
            "training solver_path must be TRAIN_FIXED",
            stage="config.cross_validation",
            field="train_step.solver_path",
            expected=TRAIN_FIXED,
            actual=config.train_step.solver_path,
        )
    if config.validation_step.solver_path != TRAIN_FIXED:
        raise _error(
            "UNSUPPORTED_VALIDATION_SOLVER",
            "validation solver_path must be TRAIN_FIXED",
            stage="config.cross_validation",
            field="validation_step.solver_path",
            expected=TRAIN_FIXED,
            actual=config.validation_step.solver_path,
        )
    if config.scheduler.monitor != config.selection.monitor:
        raise _error(
            "MONITOR_MISMATCH",
            "scheduler and model selection must monitor the same metric",
            stage="config.cross_validation",
            field="scheduler.monitor,selection.monitor",
            expected=config.selection.monitor,
            actual=config.scheduler.monitor,
        )
    if config.scheduler.mode != config.selection.mode:
        raise _error(
            "MONITOR_MODE_MISMATCH",
            "scheduler and model selection must use the same mode",
            stage="config.cross_validation",
            field="scheduler.mode,selection.mode",
            expected=config.selection.mode,
            actual=config.scheduler.mode,
        )
    weights = {
        "energy": config.loss.energy_weight,
        "force": config.loss.force_weight,
        "stress": config.loss.stress_weight,
    }
    if not any(value > 0.0 for value in weights.values()):
        raise _error(
            "NO_ACTIVE_LOSS_TERM",
            "at least one loss term must have positive weight",
            stage="config.cross_validation",
            field="loss",
        )
    monitor = config.selection.monitor
    if monitor != "total_loss" and weights[monitor] <= 0.0:
        raise _error(
            "INACTIVE_MONITORED_TERM",
            "the monitored term must have positive loss weight",
            stage="config.cross_validation",
            field=f"loss.{monitor}_weight",
            expected="> 0",
            actual=weights[monitor],
        )
    if config.fit.start_epoch != 0 or config.fit.global_step_start != 0:
        raise _error(
            "FRESH_RUN_PROGRESS_REQUIRED",
            "initial_bundle config describes a fresh run; resume progress is unsupported",
            stage="config.cross_validation",
            field="fit.start_epoch,fit.global_step_start",
            expected=(0, 0),
            actual=(config.fit.start_epoch, config.fit.global_step_start),
        )
    return config


def load_training_run_config(path: str | os.PathLike[str]) -> TrainingRunConfig:
    """Load strict JSON while preserving its semantic relative path strings."""

    raw_path = Path(path)
    try:
        resolved = raw_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise _error(
            "CONFIG_NOT_FOUND",
            "training-run config file does not exist",
            stage="config.path",
            config_path=str(raw_path),
            original_error=error,
        ) from error
    except OSError as error:
        raise _error(
            "CONFIG_PATH_ERROR",
            "training-run config path could not be resolved",
            stage="config.path",
            config_path=str(raw_path),
            original_error=error,
        ) from error
    if not resolved.is_file():
        raise _error(
            "INVALID_CONFIG_PATH",
            "training-run config path must be a regular file",
            stage="config.path",
            config_path=str(resolved),
        )
    try:
        encoded = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _error(
            "CONFIG_READ_FAILED",
            "training-run config could not be read as UTF-8",
            stage="config.read",
            config_path=str(resolved),
            original_error=error,
        ) from error
    try:
        config = TrainingRunConfig.from_json(encoded)
    except TrainingRunConfigError as error:
        if error.config_path is None:
            error.config_path = str(resolved)
        raise
    return replace(config, source_path=str(resolved))


def _freeze_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_plain(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_plain(item) for item in value)
    if type(value) is float and not math.isfinite(value):
        raise ValueError("resolved metadata contains NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"resolved metadata contains non-plain {type(value).__name__}")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("resolved metadata contains NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"resolved metadata contains non-plain {type(value).__name__}")


@dataclass(frozen=True)
class ResolvedTrainingRun:
    config_fingerprint: str
    bundle_fingerprint: str
    train_semantic_digest: str
    validation_semantic_digest: str
    train_frame_count: int
    validation_frame_count: int
    train_batch_count: int
    validation_batch_count: int
    resolved_device: str
    resolved_dtype: str
    radius_config: InteractionRadiusConfig
    radii: DerivedInteractionRadii
    species_vocabulary: tuple[int, ...]
    template_fingerprints: Mapping[str, Mapping[str, Any]]
    train_template_frame_counts: Mapping[str, int]
    validation_template_frame_counts: Mapping[str, int]
    train_composition_statistics: tuple[Mapping[str, Any], ...]
    validation_composition_statistics: tuple[Mapping[str, Any], ...]
    train_label_statistics: Mapping[str, Mapping[str, int]]
    validation_label_statistics: Mapping[str, Mapping[str, int]]
    baseline_preflight: Mapping[str, Any]
    configured_paths: Mapping[str, Any]
    runtime_paths: Mapping[str, Any]
    expected_paths: Mapping[str, str]
    training_configuration: Mapping[str, Any]
    training_executed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "config_fingerprint",
            "bundle_fingerprint",
            "train_semantic_digest",
            "validation_semantic_digest",
        ):
            if type(getattr(self, name)) is not str or _SHA256.fullmatch(
                getattr(self, name)
            ) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 string")
        for name in (
            "train_frame_count",
            "validation_frame_count",
            "train_batch_count",
            "validation_batch_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.resolved_dtype not in ("float32", "float64"):
            raise ValueError("resolved_dtype must be float32 or float64")
        if type(self.resolved_device) is not str or _DEVICE.fullmatch(
            self.resolved_device
        ) is None:
            raise ValueError("resolved_device must be cpu, cuda, or cuda:N")
        if not isinstance(self.radius_config, InteractionRadiusConfig):
            raise TypeError("radius_config must be InteractionRadiusConfig")
        if not isinstance(self.radii, DerivedInteractionRadii):
            raise TypeError("radii must be DerivedInteractionRadii")
        if self.radius_config.derived != self.radii:
            raise ValueError("radius_config and derived radii disagree")
        if type(self.training_executed) is not bool or self.training_executed:
            raise ValueError("preflight metadata requires training_executed=False")
        species = tuple(self.species_vocabulary)
        if (
            not species
            or any(isinstance(value, bool) or not isinstance(value, Integral) for value in species)
            or any(int(value) <= 0 for value in species)
            or len(set(int(value) for value in species)) != len(species)
        ):
            raise ValueError("species_vocabulary must contain unique positive integers")
        object.__setattr__(self, "species_vocabulary", tuple(int(v) for v in species))
        for name in (
            "template_fingerprints",
            "train_template_frame_counts",
            "validation_template_frame_counts",
            "train_label_statistics",
            "validation_label_statistics",
            "baseline_preflight",
            "configured_paths",
            "runtime_paths",
            "expected_paths",
            "training_configuration",
        ):
            object.__setattr__(self, name, _freeze_plain(getattr(self, name)))
        for name in (
            "train_composition_statistics",
            "validation_composition_statistics",
        ):
            object.__setattr__(self, name, _freeze_plain(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "preflight_ready",
            "training_executed": False,
            "schema_version": TRAINING_RUN_CONFIG_SCHEMA_VERSION,
            "config_fingerprint": self.config_fingerprint,
            "bundle_fingerprint": self.bundle_fingerprint,
            "data": {
                "train": {
                    "semantic_digest": self.train_semantic_digest,
                    "frame_count": self.train_frame_count,
                    "batch_count": self.train_batch_count,
                    "template_frame_counts": _plain(
                        self.train_template_frame_counts
                    ),
                    "composition_statistics": _plain(
                        self.train_composition_statistics
                    ),
                    "label_statistics": _plain(self.train_label_statistics),
                },
                "validation": {
                    "semantic_digest": self.validation_semantic_digest,
                    "frame_count": self.validation_frame_count,
                    "batch_count": self.validation_batch_count,
                    "template_frame_counts": _plain(
                        self.validation_template_frame_counts
                    ),
                    "composition_statistics": _plain(
                        self.validation_composition_statistics
                    ),
                    "label_statistics": _plain(
                        self.validation_label_statistics
                    ),
                },
            },
            "runtime": {
                "device": self.resolved_device,
                "dtype": self.resolved_dtype,
                "configured_paths": _plain(self.configured_paths),
                "paths": _plain(self.runtime_paths),
            },
            "radii": {
                "user": {
                    "r_ot": self.radius_config.r_ot,
                    "r_mp": self.radius_config.r_mp,
                },
                "advanced": {
                    "ot_switch_width": self.radius_config.ot_switch_width,
                    "ot_skin": self.radius_config.ot_skin,
                    "mp_skin": self.radius_config.mp_skin,
                },
                "derived": self.radii.to_dict(),
                "diagnostics": self.radii.to_diagnostics_dict(),
            },
            "species_vocabulary": list(self.species_vocabulary),
            "template_fingerprints": _plain(self.template_fingerprints),
            "baseline_preflight": _plain(self.baseline_preflight),
            "expected_paths": _plain(self.expected_paths),
            "training_configuration": _plain(self.training_configuration),
        }


def _base_directory(
    config: TrainingRunConfig,
    base_directory: str | os.PathLike[str] | None,
) -> tuple[Path, Path | None]:
    if base_directory is not None:
        try:
            base = Path(base_directory).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise _error(
                "INVALID_BASE_DIRECTORY",
                "base_directory could not be resolved",
                stage="paths.base",
                actual=str(base_directory),
                original_error=error,
            ) from error
        if not base.is_dir():
            raise _error(
                "INVALID_BASE_DIRECTORY",
                "base_directory must resolve to a directory",
                stage="paths.base",
                actual=str(base),
            )
        config_path = (
            None if config.source_path is None else Path(config.source_path)
        )
        return base, config_path
    if config.source_path is not None:
        try:
            config_path = Path(config.source_path).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise _error(
                "CONFIG_PATH_ERROR",
                "loaded config runtime path could not be resolved",
                stage="paths.base",
                config_path=config.source_path,
                original_error=error,
            ) from error
        return config_path.parent, config_path
    return Path.cwd().resolve(strict=True), None


def _resolve_existing_file(
    text: str,
    *,
    base: Path,
    field_name: str,
    config_path: Path | None,
) -> Path:
    candidate = Path(text)
    unresolved = candidate if candidate.is_absolute() else base / candidate
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise _error(
            "INPUT_NOT_FOUND",
            "configured input file does not exist; paths are not shell-expanded",
            stage="paths.input",
            config_path=None if config_path is None else str(config_path),
            field=field_name,
            actual=text,
            original_error=error,
        ) from error
    except OSError as error:
        raise _error(
            "INPUT_PATH_ERROR",
            "configured input path could not be resolved",
            stage="paths.input",
            config_path=None if config_path is None else str(config_path),
            field=field_name,
            actual=text,
            original_error=error,
        ) from error
    if not resolved.is_file():
        raise _error(
            "INVALID_INPUT_PATH",
            "configured input must be a regular file",
            stage="paths.input",
            config_path=None if config_path is None else str(config_path),
            field=field_name,
            actual=str(resolved),
        )
    return resolved


def _same_path(first: Path, second: Path) -> bool:
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            return False
    return False


def _resolve_output_directory(
    text: str,
    *,
    base: Path,
    config_path: Path | None,
    protected: tuple[tuple[str, Path], ...],
) -> Path:
    candidate = Path(text)
    unresolved = candidate if candidate.is_absolute() else base / candidate
    if unresolved.is_symlink():
        raise _error(
            "OUTPUT_SYMLINK_REJECTED",
            "output_directory must not be a symbolic link",
            stage="paths.output",
            config_path=None if config_path is None else str(config_path),
            field="output_directory",
            actual=str(unresolved),
        )
    resolved = unresolved.resolve(strict=False)
    for protected_name, protected_path in protected:
        if _same_path(resolved, protected_path):
            raise _error(
                "OUTPUT_PATH_COLLISION",
                "output_directory collides with a read-only input",
                stage="paths.output",
                config_path=None if config_path is None else str(config_path),
                field="output_directory",
                expected=f"distinct from {protected_name}",
                actual=str(resolved),
            )
    if resolved.exists():
        raise _error(
            "OUTPUT_ALREADY_EXISTS",
            "output_directory already exists; fresh-run preflight refuses it",
            stage="paths.output",
            config_path=None if config_path is None else str(config_path),
            field="output_directory",
            actual=str(resolved),
        )
    if not resolved.parent.exists() or not resolved.parent.is_dir():
        raise _error(
            "OUTPUT_PARENT_MISSING",
            "output_directory parent must already be a directory",
            stage="paths.output",
            config_path=None if config_path is None else str(config_path),
            field="output_directory",
            actual=str(resolved.parent),
        )
    return resolved


def _preflight_device(runtime: TrainingRuntimeConfig) -> str:
    device = torch.device(runtime.device)
    if device.type != "cuda":
        return str(device)
    try:
        available = torch.cuda.is_available()
    except Exception as error:
        raise _error(
            "CUDA_PREFLIGHT_FAILED",
            "CUDA availability could not be established",
            stage="runtime.device",
            field="runtime.device",
            actual=runtime.device,
            original_error=error,
        ) from error
    if not available:
        raise _error(
            "CUDA_UNAVAILABLE",
            "CUDA was requested but is unavailable",
            stage="runtime.device",
            field="runtime.device",
            actual=runtime.device,
        )
    if device.index is not None:
        try:
            count = torch.cuda.device_count()
        except Exception as error:
            raise _error(
                "CUDA_PREFLIGHT_FAILED",
                "CUDA device count could not be established",
                stage="runtime.device",
                field="runtime.device",
                actual=runtime.device,
                original_error=error,
            ) from error
        if device.index >= count:
            raise _error(
                "CUDA_DEVICE_INDEX_INVALID",
                "requested CUDA device index is unavailable",
                stage="runtime.device",
                field="runtime.device",
                expected=f"index < {count}",
                actual=device.index,
            )
    return str(device)


def _registry_from_bundle(bundle, *, bundle_path: Path) -> tuple[TemplateRegistry, dict[str, Any]]:
    try:
        templates = bundle.validate(bundle_path=str(bundle_path))
    except ModelBundleError:
        raise
    registry = TemplateRegistry()
    for template_id in sorted(templates):
        registry.add(templates[template_id])
    return registry, templates


def _geometry_sample_from_atoms(
    atoms: Any,
    *,
    template_id: str,
    registry: TemplateRegistry,
    dtype: torch.dtype,
    frame_index: int,
    sample_id: str,
) -> StructureSample:
    positions = torch.tensor(atoms.get_positions().copy(), dtype=torch.float64)
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers().copy(), dtype=torch.long
    )
    cell = torch.tensor(atoms.cell.array.copy(), dtype=torch.float64)
    pbc = torch.tensor(atoms.get_pbc().copy(), dtype=torch.bool)
    if positions.shape != (len(atoms), 3) or atomic_numbers.shape != (len(atoms),):
        raise ExtXYZLoadError(
            "MALFORMED_GEOMETRY",
            "positions/numbers have invalid shape",
            frame_index=frame_index,
            sample_id=sample_id,
        )
    if not bool(torch.all(torch.isfinite(positions))) or not bool(
        torch.all(torch.isfinite(cell))
    ):
        raise ExtXYZLoadError(
            "NONFINITE_GEOMETRY",
            "positions or cell contain NaN or Inf",
            frame_index=frame_index,
            sample_id=sample_id,
        )
    if pbc.shape != (3,) or not bool(torch.all(pbc)):
        raise ExtXYZLoadError(
            "NONPERIODIC_STRUCTURE",
            "only full PBC extxyz frames are supported",
            frame_index=frame_index,
            sample_id=sample_id,
        )
    if cell.shape != (3, 3) or bool(
        torch.linalg.svdvals(cell)[-1] <= torch.finfo(torch.float64).eps
    ):
        raise ExtXYZLoadError(
            "MALFORMED_GEOMETRY",
            "cell must be nonsingular [3,3]",
            frame_index=frame_index,
            sample_id=sample_id,
        )
    try:
        template = registry.resolve(template_id)
    except KeyError as error:
        raise ExtXYZLoadError(
            "UNKNOWN_TEMPLATE",
            f"unknown exact template_id {template_id!r}",
            frame_index=frame_index,
            sample_id=sample_id,
        ) from error
    if template.strict_domain is None:
        raise ExtXYZLoadError(
            "MISSING_STRICT_DOMAIN",
            "training preflight requires a strict template domain",
            frame_index=frame_index,
            sample_id=sample_id,
        )
    labels = {
        name: _extract_label(
            atoms,
            name,
            required=False,
            frame_index=frame_index,
            sample_id=sample_id,
        )
        for name in ("energy", "forces", "stress")
    }
    masks = {
        term: _extract_component_mask(
            atoms,
            term=term,
            label_present=labels[term] is not None,
            frame_index=frame_index,
            sample_id=sample_id,
        )
        for term in ("forces", "stress")
    }
    try:
        template.validate_structure(
            atomic_numbers,
            cell=cell,
            pbc=pbc,
            sample_id=sample_id,
        )
    except ValueError as error:
        raise ExtXYZLoadError(
            "TEMPLATE_DOMAIN_REJECTION",
            str(error),
            frame_index=frame_index,
            sample_id=sample_id,
        ) from error

    def floating(value):
        return None if value is None else value.detach().clone().to(dtype=dtype)

    def boolean(value):
        return None if value is None else value.detach().clone().to(dtype=torch.bool)

    return StructureSample(
        sample_id=sample_id,
        positions=positions.to(dtype=dtype),
        atomic_numbers=atomic_numbers,
        cell=cell.to(dtype=dtype),
        pbc=pbc,
        origin=torch.zeros(3, dtype=dtype),
        template_id=template_id,
        energy=floating(labels["energy"]),
        forces=floating(labels["forces"]),
        stress=floating(labels["stress"]),
        force_mask=boolean(masks["forces"]),
        stress_mask=boolean(masks["stress"]),
    )


def _load_template_key_source(
    path: Path,
    source: TrainingDataSourceConfig,
    *,
    split: str,
    source_index: int,
    registry: TemplateRegistry,
    dtype: torch.dtype,
) -> tuple[StructureSample, ...]:
    try:
        from ase.io import iread
    except ImportError as error:  # pragma: no cover - optional dependency
        raise _error(
            "ASE_UNAVAILABLE",
            "ASE is required for extxyz training preflight",
            stage="data.parse",
            split=split,
            field=f"data.{split}[{source_index}].path",
            original_error=error,
        ) from error
    samples = []
    try:
        for frame_index, atoms in enumerate(
            iread(str(path), index=":", format="extxyz")
        ):
            sample_id = f"{split}.{source_index:04d}:{frame_index:06d}"
            key = source.template_key
            if key not in atoms.info:
                raise ExtXYZLoadError(
                    "MISSING_TEMPLATE_KEY",
                    f"Atoms.info is missing template key {key!r}",
                    frame_index=frame_index,
                    sample_id=sample_id,
                )
            template_id = atoms.info[key]
            if type(template_id) is not str or not template_id:
                raise ExtXYZLoadError(
                    "INVALID_TEMPLATE_ID",
                    "template-key value must be a nonempty exact string",
                    frame_index=frame_index,
                    sample_id=sample_id,
                )
            try:
                samples.append(
                    _geometry_sample_from_atoms(
                        atoms,
                        template_id=template_id,
                        registry=registry,
                        dtype=dtype,
                        frame_index=frame_index,
                        sample_id=sample_id,
                    )
                )
            except ExtXYZLoadError as error:
                error.template_id = template_id
                raise
    except ExtXYZLoadError:
        raise
    except Exception as error:
        raise ExtXYZLoadError(
            "ASE_PARSE_FAILURE", f"ASE extxyz parse failed: {error}"
        ) from error
    if not samples:
        raise ExtXYZLoadError("EMPTY_SOURCE", "extxyz source contains no frames")
    return tuple(samples)


def _load_split(
    sources: tuple[TrainingDataSourceConfig, ...],
    paths: tuple[Path, ...],
    *,
    split: str,
    registry: TemplateRegistry,
    dtype: torch.dtype,
    config_path: Path | None,
) -> tuple[StructureSample, ...]:
    samples = []
    for source_index, (source, path) in enumerate(zip(sources, paths)):
        if source.template_id is not None and source.template_id not in registry:
            raise _error(
                "UNKNOWN_TEMPLATE",
                "configured exact template_id is absent from the initial bundle",
                stage="data.template_selection",
                config_path=None if config_path is None else str(config_path),
                field=f"data.{split}[{source_index}].template_id",
                split=split,
                template_id=source.template_id,
            )
        if source.template_id is not None:
            selected_template = registry.resolve(source.template_id)
            if selected_template.strict_domain is None:
                raise _error(
                    "MISSING_STRICT_DOMAIN",
                    "training data requires a strict template domain",
                    stage="data.template_selection",
                    config_path=(
                        None if config_path is None else str(config_path)
                    ),
                    field=f"data.{split}[{source_index}].template_id",
                    split=split,
                    template_id=source.template_id,
                )
        try:
            if source.template_id is not None:
                loaded = load_extxyz_samples(
                    ExtXYZLoadConfig(
                        source_path=str(path),
                        sample_id_prefix=f"{split}.{source_index:04d}",
                        template_id=source.template_id,
                        require_energy=False,
                        require_forces=False,
                        require_stress=False,
                        dtype=dtype,
                        device="cpu",
                    ),
                    registry,
                ).samples
            else:
                loaded = _load_template_key_source(
                    path,
                    source,
                    split=split,
                    source_index=source_index,
                    registry=registry,
                    dtype=dtype,
                )
        except ExtXYZLoadError as error:
            raise _error(
                error.reason_code,
                f"extxyz data preflight failed: {error}",
                stage="data.load",
                config_path=None if config_path is None else str(config_path),
                field=f"data.{split}[{source_index}]",
                split=split,
                frame_index=error.frame_index,
                sample_id=error.sample_id,
                template_id=getattr(error, "template_id", source.template_id),
                original_reason_code=error.reason_code,
                original_error=error,
            ) from error
        except Exception as error:
            raise _error(
                getattr(error, "reason_code", "DATA_LOAD_FAILED"),
                f"extxyz data preflight failed: {error}",
                stage="data.load",
                config_path=None if config_path is None else str(config_path),
                field=f"data.{split}[{source_index}]",
                split=split,
                template_id=source.template_id,
                original_reason_code=getattr(error, "reason_code", None),
                original_error=error,
            ) from error
        samples.extend(loaded)
    result = tuple(samples)
    ids = tuple(sample.sample_id for sample in result)
    if len(set(ids)) != len(ids):
        raise _error(
            "DUPLICATE_SAMPLE_ID",
            "generated sample IDs collide within a split",
            stage="data.identity",
            split=split,
        )
    return result


def _hash_text(digest: Any, value: str) -> None:
    raw = value.encode("utf-8")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _hash_tensor(digest: Any, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    _hash_text(digest, name)
    _hash_text(digest, str(tensor.dtype))
    _hash_text(digest, ",".join(str(size) for size in tensor.shape))
    raw = tensor.numpy().tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _split_digest(
    samples: tuple[StructureSample, ...],
    templates: Mapping[str, Any],
    *,
    split: str,
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "training_run_extxyz_split_v1")
    _hash_text(digest, EXTXYZ_LOADER_CONVENTION_VERSION)
    _hash_text(digest, EXTXYZ_UNIT_CONVENTION_VERSION)
    _hash_text(digest, split)
    _hash_text(digest, str(len(samples)))
    for frame_index, sample in enumerate(samples):
        _hash_text(digest, f"frame:{frame_index}")
        _hash_text(digest, sample.sample_id)
        _hash_text(digest, sample.template_id)
        _hash_text(digest, templates[sample.template_id].fingerprint)
        for name in ("positions", "atomic_numbers", "cell", "pbc", "origin"):
            _hash_tensor(digest, name, getattr(sample, name))
        for name in ("energy", "forces", "stress"):
            value = getattr(sample, name)
            _hash_text(digest, f"{name}_present:{value is not None}")
            if value is not None:
                _hash_tensor(digest, name, value)
        for name in ("force_mask", "stress_mask"):
            value = getattr(sample, name)
            _hash_text(digest, f"{name}_explicit:{value is not None}")
            if value is not None:
                _hash_tensor(digest, name, value)
    return digest.hexdigest()


def _phase_specification_fingerprint(phase_specification: Any) -> str:
    payload = {
        "scope": "reference_site_phase_specification_inspection_v1",
        "value": phase_specification.to_dict(),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _label_statistics(samples: tuple[StructureSample, ...]) -> dict[str, Any]:
    result = {}
    for term in ("energy", "forces", "stress"):
        present = sum(getattr(sample, term) is not None for sample in samples)
        if term == "energy":
            valid = present
        elif term == "forces":
            valid = sum(
                (
                    sample.num_atoms * 3
                    if sample.force_mask is None
                    else int(torch.count_nonzero(sample.force_mask))
                )
                for sample in samples
                if sample.forces is not None
            )
        else:
            valid = sum(
                (
                    6
                    if sample.stress_mask is None
                    else int(
                        torch.count_nonzero(
                            _independent_stress_mask(sample.stress_mask)
                        )
                    )
                )
                for sample in samples
                if sample.stress is not None
            )
        result[term] = {
            "present_frames": present,
            "missing_frames": len(samples) - present,
            "valid_count": valid,
        }
    return result


def _composition_statistics(samples: tuple[StructureSample, ...]):
    histogram = Counter()
    for sample in samples:
        counts = Counter(int(value) for value in sample.atomic_numbers.tolist())
        histogram[tuple(sorted(counts.items()))] += 1
    return tuple(
        {
            "frame_count": histogram[composition],
            "species": [
                {"atomic_number": atomic_number, "count": count}
                for atomic_number, count in composition
            ],
        }
        for composition in sorted(histogram)
    )


def _active_loss_terms(config: LossConfig) -> tuple[str, ...]:
    return tuple(
        name
        for name, weight in (
            ("energy", config.energy_weight),
            ("forces", config.force_weight),
            ("stress", config.stress_weight),
        )
        if weight > 0.0
    )


def _sample_has_term(sample: StructureSample, term: str) -> bool:
    value = getattr(sample, term)
    if value is None:
        return False
    if term == "forces" and sample.force_mask is not None:
        return bool(torch.any(sample.force_mask))
    if term == "stress" and sample.stress_mask is not None:
        return bool(torch.any(_independent_stress_mask(sample.stress_mask)))
    return value.numel() > 0


def _validate_supervision(
    train_samples: tuple[StructureSample, ...],
    validation_samples: tuple[StructureSample, ...],
    config: TrainingRunConfig,
) -> None:
    active = _active_loss_terms(config.loss)
    for term in active:
        if not any(_sample_has_term(sample, term) for sample in train_samples):
            raise _error(
                "MISSING_TRAIN_SUPERVISION",
                "active loss term has no valid training labels",
                stage="data.supervision",
                field=f"loss.{term}",
                split="train",
            )
    batch_size = config.data.batch_size
    for start in range(0, len(train_samples), batch_size):
        chunk = train_samples[start : start + batch_size]
        if not any(
            _sample_has_term(sample, term) for sample in chunk for term in active
        ):
            first = chunk[0]
            raise _error(
                "UNSUPERVISED_TRAIN_BATCH",
                "deterministic training batch has no active weighted supervision",
                stage="data.batch_plan",
                split="train",
                frame_index=start,
                sample_id=first.sample_id,
                template_id=first.template_id,
            )
    monitor = config.selection.monitor
    monitored_terms = (
        active
        if monitor == "total_loss"
        else (monitor + "s" if monitor == "force" else monitor,)
    )
    if not any(
        _sample_has_term(sample, term)
        for sample in validation_samples
        for term in monitored_terms
    ):
        raise _error(
            "MISSING_MONITORED_SUPERVISION",
            "validation split has no supervision for the monitored metric",
            stage="data.supervision",
            field=f"selection.monitor={monitor}",
            split="validation",
        )


def _baseline_preflight(
    samples: tuple[StructureSample, ...],
    species_vocabulary: tuple[int, ...],
    config: AtomicBaselineConfig,
) -> dict[str, Any]:
    try:
        fitted = fit_atomic_baseline(
            samples,
            range(len(samples)),
            species_vocabulary,
            config,
        )
    except Exception as error:
        raise _error(
            "BASELINE_PREFLIGHT_FAILED",
            "training energies/compositions cannot satisfy atomic baseline fitting: "
            f"{error}",
            stage="baseline.preflight",
            field="baseline",
            split="train",
            original_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error
    return {
        "num_valid_energy_structures": fitted.num_valid_energy_structures,
        "rank": fitted.rank,
        "required_rank": len(species_vocabulary),
        "rank_deficient": fitted.rank_deficient,
        "condition_number": (
            None
            if not math.isfinite(fitted.condition_number)
            else fitted.condition_number
        ),
        "rank_policy": config.rank_policy,
        "species_occurrence_counts": fitted.species_occurrence_counts.tolist(),
        "residual_rmse": fitted.residual_rmse,
        "residual_mae": fitted.residual_mae,
        "weighted_objective": fitted.weighted_objective,
        "parameter_update_applied": False,
    }


def resolve_training_run(
    config: TrainingRunConfig,
    *,
    base_directory: str | os.PathLike[str] | None = None,
) -> ResolvedTrainingRun:
    """Perform complete read-only preflight without model execution or writes."""

    validate_training_run_config(config)
    base, config_path = _base_directory(config, base_directory)
    resolved_device = _preflight_device(config.runtime)
    bundle_path = _resolve_existing_file(
        config.initial_bundle,
        base=base,
        field_name="initial_bundle",
        config_path=config_path,
    )
    train_paths = tuple(
        _resolve_existing_file(
            source.path,
            base=base,
            field_name=f"data.train[{index}].path",
            config_path=config_path,
        )
        for index, source in enumerate(config.data.train)
    )
    validation_paths = tuple(
        _resolve_existing_file(
            source.path,
            base=base,
            field_name=f"data.validation[{index}].path",
            config_path=config_path,
        )
        for index, source in enumerate(config.data.validation)
    )
    protected = [("initial_bundle", bundle_path)]
    protected.extend(
        (f"data.train[{index}]", path) for index, path in enumerate(train_paths)
    )
    protected.extend(
        (f"data.validation[{index}]", path)
        for index, path in enumerate(validation_paths)
    )
    if config_path is not None:
        protected.append(("config", config_path))
    output_path = _resolve_output_directory(
        config.output_directory,
        base=base,
        config_path=config_path,
        protected=tuple(protected),
    )

    try:
        bundle = load_reference_site_model_bundle(bundle_path, map_location="cpu")
        registry, templates = _registry_from_bundle(bundle, bundle_path=bundle_path)
        potential_config = PotentialConfig.from_dict(bundle.model_config)
    except ModelBundleError as error:
        raise _error(
            error.reason_code,
            "initial portable bundle validation failed",
            stage=error.validation_stage or "bundle.load",
            config_path=None if config_path is None else str(config_path),
            field="initial_bundle",
            template_id=error.template_id,
            expected=error.expected_fingerprint,
            actual=error.actual_fingerprint,
            original_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise _error(
            getattr(error, "reason_code", "BUNDLE_LOAD_FAILED"),
            "initial portable bundle validation failed",
            stage="bundle.load",
            config_path=None if config_path is None else str(config_path),
            field="initial_bundle",
            original_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error

    try:
        validate_radius_model_compatibility(config.radii, potential_config)
    except RadiusConfigError as error:
        first = error.mismatches[0] if error.mismatches else (None, None, None)
        raise _error(
            error.reason_code,
            str(error),
            stage="radii.model_compatibility",
            config_path=None if config_path is None else str(config_path),
            field=first[0],
            expected=first[1],
            actual=first[2],
            original_reason_code=error.reason_code,
            original_error=error,
        ) from error
    for binding in bundle.template_bindings:
        try:
            validate_radius_artifact_compatibility(
                config.radii, binding.structural_artifact
            )
        except RadiusConfigError as error:
            first = error.mismatches[0] if error.mismatches else (None, None, None)
            raise _error(
                error.reason_code,
                str(error),
                stage="radii.artifact_compatibility",
                config_path=None if config_path is None else str(config_path),
                field=first[0],
                template_id=binding.template_id,
                expected=first[1],
                actual=first[2],
                original_reason_code=error.reason_code,
                original_error=error,
            ) from error

    train_samples = _load_split(
        config.data.train,
        train_paths,
        split="train",
        registry=registry,
        dtype=torch.float64,
        config_path=config_path,
    )
    validation_samples = _load_split(
        config.data.validation,
        validation_paths,
        split="validation",
        registry=registry,
        dtype=torch.float64,
        config_path=config_path,
    )
    if set(sample.sample_id for sample in train_samples) & set(
        sample.sample_id for sample in validation_samples
    ):
        raise _error(
            "CROSS_SPLIT_SAMPLE_ID_COLLISION",
            "train and validation sample ID namespaces overlap",
            stage="data.identity",
        )
    _validate_supervision(train_samples, validation_samples, config)
    baseline = _baseline_preflight(
        train_samples, tuple(bundle.species_vocabulary), config.baseline
    )

    batch_size = config.data.batch_size
    train_template_counts = dict(
        sorted(Counter(sample.template_id for sample in train_samples).items())
    )
    validation_template_counts = dict(
        sorted(
            Counter(sample.template_id for sample in validation_samples).items()
        )
    )
    bindings = {
        binding.template_id: binding for binding in bundle.template_bindings
    }
    template_fingerprints = {
        template_id: {
            "structural_artifact_fingerprint": (
                bindings[template_id].structural_artifact.structural_fingerprint
            ),
            "full_template_fingerprint": (
                bindings[template_id].full_template_fingerprint
            ),
            "phase_specification_fingerprint": (
                _phase_specification_fingerprint(
                    bindings[template_id].phase_specification
                )
            ),
            "binding_fingerprint": bindings[template_id].binding_fingerprint,
            "evaluation_policy_fingerprint": (
                None
                if bindings[template_id].evaluation_policy is None
                else bindings[template_id].evaluation_policy.content_fingerprint
            ),
        }
        for template_id in sorted(templates)
    }
    runtime_paths = {
        "config": None if config_path is None else str(config_path),
        "initial_bundle": str(bundle_path),
        "output_directory": str(output_path),
        "train_inputs": [str(path) for path in train_paths],
        "validation_inputs": [str(path) for path in validation_paths],
        "path_kind": "runtime_location_not_semantic_fingerprint",
    }
    configured_paths = {
        "initial_bundle": config.initial_bundle,
        "output_directory": config.output_directory,
        "train_inputs": [source.path for source in config.data.train],
        "validation_inputs": [
            source.path for source in config.data.validation
        ],
        "path_kind": "original_config_expression_in_semantic_fingerprint",
    }
    expected_paths = {
        "output_directory": str(output_path),
        "latest_checkpoint": str(output_path / "latest.pt"),
        "best_checkpoint": str(output_path / "best.pt"),
        "epoch_checkpoint_pattern": str(output_path / "epoch-XXXXXX.pt"),
    }
    training_configuration = {
        "loss": config.loss.to_dict(),
        "baseline": config.baseline.to_dict(),
        "optimizer": config.optimizer.to_dict(),
        "train_step": config.train_step.to_dict(),
        "validation_step": config.validation_step.to_dict(),
        "scheduler": config.scheduler.to_dict(),
        "selection": config.selection.to_dict(),
        "fit": config.fit.to_dict(),
        "checkpointed_fit": config.checkpointed_fit.to_dict(),
        "batch_size": batch_size,
        "shuffle": False,
    }
    return ResolvedTrainingRun(
        config_fingerprint=config.config_fingerprint,
        bundle_fingerprint=bundle.bundle_fingerprint,
        train_semantic_digest=_split_digest(
            train_samples, templates, split="train"
        ),
        validation_semantic_digest=_split_digest(
            validation_samples, templates, split="validation"
        ),
        train_frame_count=len(train_samples),
        validation_frame_count=len(validation_samples),
        train_batch_count=math.ceil(len(train_samples) / batch_size),
        validation_batch_count=math.ceil(len(validation_samples) / batch_size),
        resolved_device=resolved_device,
        resolved_dtype=config.runtime.dtype,
        radius_config=config.radii,
        radii=config.radii.derived,
        species_vocabulary=tuple(bundle.species_vocabulary),
        template_fingerprints=template_fingerprints,
        train_template_frame_counts=train_template_counts,
        validation_template_frame_counts=validation_template_counts,
        train_composition_statistics=_composition_statistics(train_samples),
        validation_composition_statistics=_composition_statistics(
            validation_samples
        ),
        train_label_statistics=_label_statistics(train_samples),
        validation_label_statistics=_label_statistics(validation_samples),
        baseline_preflight=baseline,
        configured_paths=configured_paths,
        runtime_paths=runtime_paths,
        expected_paths=expected_paths,
        training_configuration=training_configuration,
        training_executed=False,
    )


__all__ = [
    "TRAINING_RUN_CONFIG_SCHEMA_VERSION",
    "ResolvedTrainingRun",
    "TrainingDataConfig",
    "TrainingRuntimeConfig",
    "TrainingRunConfig",
    "TrainingRunConfigError",
    "load_training_run_config",
    "resolve_training_run",
    "validate_training_run_config",
]
