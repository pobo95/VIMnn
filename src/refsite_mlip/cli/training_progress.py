"""Newline-oriented, presentation-only training progress rendering.

This module owns no live training state. It formats immutable startup snapshots
and committed-epoch events for human-facing stderr logs without taking part in
the training transaction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
import sys
import time
from typing import TextIO

from refsite_mlip.training.metrics_journal import (
    CommittedEpochMetrics,
    EpochMetricsObserver,
)


TRAINING_PROGRESS_CONFIG_VERSION = "refsite_training_progress_config_v1"


def _bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_integer(name: str, value: object | None) -> int | None:
    return None if value is None else _integer(name, value)


def _signed_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _optional_signed_integer(name: str, value: object | None) -> int | None:
    return None if value is None else _signed_integer(name, value)


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_finite(name: str, value: object | None) -> float | None:
    return None if value is None else _finite(name, value)


def _sequence(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _fingerprint(name: str, value: object) -> str:
    result = _string(name, value)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return result


@dataclass(frozen=True)
class TrainingProgressConfig:
    """Non-semantic controls for newline-based console presentation."""

    enabled: bool = True
    program_name: str = "refsite-mlip"
    float_precision: int = 6
    duration_precision: int = 1
    schema_version: str = field(
        default=TRAINING_PROGRESS_CONFIG_VERSION, init=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _bool("enabled", self.enabled))
        object.__setattr__(
            self, "program_name", _string("program_name", self.program_name)
        )
        precision = _integer("float_precision", self.float_precision, minimum=1)
        if precision > 17:
            raise ValueError("float_precision must not exceed 17")
        object.__setattr__(self, "float_precision", precision)
        duration = _integer(
            "duration_precision", self.duration_precision, minimum=0
        )
        if duration > 6:
            raise ValueError("duration_precision must not exceed 6")
        object.__setattr__(self, "duration_precision", duration)


LabelPresence = tuple[str, int, int, int, int]
TemplateSummary = tuple[str, int]
TemplateFrameCounts = tuple[str, int, int]
CompositionSummary = tuple[str, int, int]
LossTermSummary = tuple[str, float, float]


@dataclass(frozen=True)
class TrainingStartSummary:
    """Immutable effective-runtime snapshot made before the first update."""

    run_name: str
    source_kind: str
    device: str
    dtype: str
    training_seed: int
    initialization_seed: int | None
    parameter_tensor_count: int
    parameter_element_count: int
    species_vocabulary: tuple[int, ...]
    templates: tuple[TemplateSummary, ...]
    default_template_id: str | None
    train_frame_count: int
    validation_frame_count: int
    train_batch_count: int
    validation_batch_count: int
    train_batch_size: int
    validation_batch_size: int
    template_frame_counts: tuple[TemplateFrameCounts, ...]
    composition_summary: tuple[CompositionSummary, ...]
    label_presence: tuple[LabelPresence, ...]
    r_ot: float
    r_mp: float
    r_candidate_ot: float
    r_candidate_mp: float
    ot_backend: str
    solver_path: str
    baseline_enabled: bool
    baseline_values: tuple[float, ...]
    baseline_rank_policy: str
    baseline_status: str
    loss_terms: tuple[LossTermSummary, ...]
    optimizer_kind: str
    initial_learning_rate: float
    weight_decay: float
    scheduler_kind: str
    scheduler_monitor: str
    scheduler_mode: str
    max_epochs: int
    early_stop_patience: int | None
    output_directory: str
    initial_bundle_fingerprint: str
    train_semantic_digest: str
    validation_semantic_digest: str
    resumed: bool = False
    resume_checkpoint_epoch: int | None = None
    resume_global_step: int | None = None
    existing_best_epoch: int | None = None
    existing_best_value: float | None = None
    recovered_journal_event_count: int = 0

    def __post_init__(self) -> None:
        string_fields = (
            "run_name",
            "source_kind",
            "device",
            "dtype",
            "ot_backend",
            "solver_path",
            "baseline_rank_policy",
            "baseline_status",
            "optimizer_kind",
            "scheduler_kind",
            "scheduler_monitor",
            "scheduler_mode",
            "output_directory",
        )
        for name in string_fields:
            object.__setattr__(self, name, _string(name, getattr(self, name)))
        if self.default_template_id is not None:
            object.__setattr__(
                self,
                "default_template_id",
                _string("default_template_id", self.default_template_id),
            )
        if self.source_kind not in ("scratch", "bundle"):
            raise ValueError("source_kind must be 'scratch' or 'bundle'")
        if self.solver_path not in ("train-fixed", "TRAIN_FIXED"):
            raise ValueError("solver_path must be TRAIN_FIXED")
        if self.scheduler_mode not in ("min", "max"):
            raise ValueError("scheduler_mode must be 'min' or 'max'")

        object.__setattr__(
            self,
            "training_seed",
            _signed_integer("training_seed", self.training_seed),
        )
        object.__setattr__(
            self,
            "initialization_seed",
            _optional_signed_integer(
                "initialization_seed", self.initialization_seed
            ),
        )
        positive_counts = (
            "parameter_tensor_count",
            "parameter_element_count",
            "train_frame_count",
            "validation_frame_count",
            "train_batch_count",
            "validation_batch_count",
            "train_batch_size",
            "validation_batch_size",
            "max_epochs",
        )
        for name in positive_counts:
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), minimum=1)
            )

        species = tuple(
            _integer(f"species_vocabulary[{index}]", item, minimum=1)
            for index, item in enumerate(
                _sequence("species_vocabulary", self.species_vocabulary)
            )
        )
        if not species or len(species) != len(set(species)):
            raise ValueError("species_vocabulary must be nonempty and unique")
        object.__setattr__(self, "species_vocabulary", tuple(sorted(species)))

        templates = self._count_pairs(self.templates, name="templates", minimum=1)
        templates = tuple(sorted(templates))
        if not templates:
            raise ValueError("templates must not be empty")
        if (
            self.default_template_id is not None
            and self.default_template_id not in {item[0] for item in templates}
        ):
            raise ValueError("default_template_id is absent from templates")
        object.__setattr__(self, "templates", templates)

        template_counts = self._three_column_counts(
            self.template_frame_counts, name="template_frame_counts"
        )
        if {item[0] for item in template_counts} != {item[0] for item in templates}:
            raise ValueError("template_frame_counts must cover every template")
        if sum(item[1] for item in template_counts) != self.train_frame_count:
            raise ValueError("train template-frame counts differ from frame count")
        if sum(item[2] for item in template_counts) != self.validation_frame_count:
            raise ValueError("validation template-frame counts differ from frame count")
        object.__setattr__(
            self, "template_frame_counts", tuple(sorted(template_counts))
        )
        object.__setattr__(
            self,
            "composition_summary",
            tuple(
                sorted(
                    self._three_column_counts(
                        self.composition_summary, name="composition_summary"
                    )
                )
            ),
        )
        object.__setattr__(
            self, "label_presence", self._canonical_label_presence()
        )

        for name in ("r_ot", "r_mp", "r_candidate_ot", "r_candidate_mp"):
            object.__setattr__(
                self, name, _finite(name, getattr(self, name), minimum=0.0)
            )
        if self.r_ot <= 0.0 or self.r_mp <= 0.0:
            raise ValueError("r_ot and r_mp must be positive")
        if self.r_candidate_ot < self.r_ot or self.r_candidate_mp < self.r_mp:
            raise ValueError("candidate radii must not be below interaction radii")

        object.__setattr__(
            self, "baseline_enabled", _bool("baseline_enabled", self.baseline_enabled)
        )
        values = tuple(
            _finite(f"baseline_values[{index}]", value)
            for index, value in enumerate(
                _sequence("baseline_values", self.baseline_values)
            )
        )
        if self.baseline_enabled and len(values) != len(species):
            raise ValueError("baseline values must match species vocabulary")
        object.__setattr__(self, "baseline_values", values)
        object.__setattr__(self, "loss_terms", self._canonical_loss_terms())
        object.__setattr__(
            self,
            "initial_learning_rate",
            _finite("initial_learning_rate", self.initial_learning_rate, minimum=0.0),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite("weight_decay", self.weight_decay, minimum=0.0),
        )
        object.__setattr__(
            self,
            "early_stop_patience",
            _optional_integer("early_stop_patience", self.early_stop_patience),
        )
        for name in (
            "initial_bundle_fingerprint",
            "train_semantic_digest",
            "validation_semantic_digest",
        ):
            object.__setattr__(self, name, _fingerprint(name, getattr(self, name)))

        object.__setattr__(self, "resumed", _bool("resumed", self.resumed))
        for name in (
            "resume_checkpoint_epoch",
            "resume_global_step",
            "existing_best_epoch",
        ):
            object.__setattr__(
                self, name, _optional_integer(name, getattr(self, name))
            )
        object.__setattr__(
            self,
            "existing_best_value",
            _optional_finite("existing_best_value", self.existing_best_value),
        )
        object.__setattr__(
            self,
            "recovered_journal_event_count",
            _integer(
                "recovered_journal_event_count",
                self.recovered_journal_event_count,
            ),
        )
        if self.resumed:
            if self.resume_checkpoint_epoch is None or self.resume_global_step is None:
                raise ValueError("resumed summary requires checkpoint epoch and step")
        elif (
            self.resume_checkpoint_epoch is not None
            or self.resume_global_step is not None
            or self.recovered_journal_event_count != 0
        ):
            raise ValueError("fresh summary cannot contain resume progress")

    @staticmethod
    def _count_pairs(
        values: object, *, name: str, minimum: int = 0
    ) -> tuple[tuple[str, int], ...]:
        result: list[tuple[str, int]] = []
        for index, value in enumerate(_sequence(name, values)):
            row = _sequence(f"{name}[{index}]", value)
            if len(row) != 2:
                raise ValueError(f"{name}[{index}] must contain two values")
            result.append(
                (
                    _string(f"{name}[{index}][0]", row[0]),
                    _integer(f"{name}[{index}][1]", row[1], minimum=minimum),
                )
            )
        if len({item[0] for item in result}) != len(result):
            raise ValueError(f"{name} contains duplicate keys")
        return tuple(result)

    @staticmethod
    def _three_column_counts(
        values: object, *, name: str
    ) -> tuple[tuple[str, int, int], ...]:
        result: list[tuple[str, int, int]] = []
        for index, value in enumerate(_sequence(name, values)):
            row = _sequence(f"{name}[{index}]", value)
            if len(row) != 3:
                raise ValueError(f"{name}[{index}] must contain three values")
            result.append(
                (
                    _string(f"{name}[{index}][0]", row[0]),
                    _integer(f"{name}[{index}][1]", row[1]),
                    _integer(f"{name}[{index}][2]", row[2]),
                )
            )
        if len({item[0] for item in result}) != len(result):
            raise ValueError(f"{name} contains duplicate keys")
        return tuple(result)

    def _canonical_label_presence(self) -> tuple[LabelPresence, ...]:
        by_term: dict[str, LabelPresence] = {}
        for index, value in enumerate(_sequence("label_presence", self.label_presence)):
            row = _sequence(f"label_presence[{index}]", value)
            if len(row) != 5:
                raise ValueError(
                    f"label_presence[{index}] must contain five values"
                )
            term = _string(f"label_presence[{index}][0]", row[0])
            if term not in ("energy", "force", "stress") or term in by_term:
                raise ValueError("label terms must be unique energy, force, stress")
            by_term[term] = (
                term,
                _integer(f"label_presence[{index}][1]", row[1]),
                _integer(f"label_presence[{index}][2]", row[2]),
                _integer(f"label_presence[{index}][3]", row[3]),
                _integer(f"label_presence[{index}][4]", row[4]),
            )
        order = ("energy", "force", "stress")
        if set(by_term) != set(order):
            raise ValueError("label_presence must contain energy, force, and stress")
        return tuple(by_term[term] for term in order)

    def _canonical_loss_terms(self) -> tuple[LossTermSummary, ...]:
        by_term: dict[str, LossTermSummary] = {}
        for index, value in enumerate(_sequence("loss_terms", self.loss_terms)):
            row = _sequence(f"loss_terms[{index}]", value)
            if len(row) != 3:
                raise ValueError(f"loss_terms[{index}] must contain three values")
            term = _string(f"loss_terms[{index}][0]", row[0])
            if term not in ("energy", "force", "stress") or term in by_term:
                raise ValueError("loss terms must be unique energy, force, stress")
            by_term[term] = (
                term,
                _finite(f"loss_terms[{index}].weight", row[1], minimum=0.0),
                _finite(f"loss_terms[{index}].scale", row[2], minimum=0.0),
            )
        order = ("energy", "force", "stress")
        if set(by_term) != set(order):
            raise ValueError("loss_terms must contain energy, force, and stress")
        return tuple(by_term[term] for term in order)


class TrainingProgressError(RuntimeError):
    """Captured presentation failure which never becomes a training failure."""

    def __init__(self, message: str, *, original_error: BaseException) -> None:
        super().__init__(message)
        self.reason_code = "TRAINING_PROGRESS_WRITE_FAILED"
        self.original_error = original_error
        self.original_exception_type = type(original_error).__name__
        self.original_exception_message = str(original_error)


class TrainingProgressRenderer:
    """Render deterministic newline blocks from immutable metric snapshots."""

    def __init__(
        self,
        config: TrainingProgressConfig | None = None,
        *,
        stream: TextIO | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if config is None:
            config = TrainingProgressConfig()
        if not isinstance(config, TrainingProgressConfig):
            raise TypeError("config must be a TrainingProgressConfig")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self._config = config
        self._stream = sys.stderr if stream is None else stream
        self._monotonic = monotonic
        self._io_available = True
        self._presentation_error: TrainingProgressError | None = None
        self._summary: TrainingStartSummary | None = None
        self._session_started_at: float | None = None
        self._session_event_count = 0

    @property
    def config(self) -> TrainingProgressConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._io_available

    @property
    def presentation_error(self) -> TrainingProgressError | None:
        return self._presentation_error

    @property
    def session_event_count(self) -> int:
        return self._session_event_count

    def render_stage(self, message: str) -> None:
        if not self.enabled:
            return
        try:
            self._write(f"{self._config.program_name}: {_single_line(message)}\n")
        except Exception as error:
            self._disable(error)

    def render_start(self, summary: TrainingStartSummary) -> None:
        if not isinstance(summary, TrainingStartSummary):
            raise TypeError("summary must be a TrainingStartSummary")
        self._summary = summary
        if not self.enabled:
            return
        try:
            self._session_started_at = _finite(
                "monotonic clock", self._monotonic()
            )
            self._write(self._start_block(summary))
        except Exception as error:
            self._disable(error)

    def render_start_from(
        self, factory: Callable[[], TrainingStartSummary]
    ) -> None:
        """Build and render a summary entirely inside the presentation boundary.

        Quiet and already-disabled renderers do not evaluate ``factory``.  A
        projection/formatting error is presentation-only and therefore disables
        subsequent progress without affecting the training transaction.
        """

        if not callable(factory):
            raise TypeError("factory must be callable")
        if not self.enabled:
            return
        try:
            summary = factory()
        except Exception as error:
            self._disable(error)
            return
        self.render_start(summary)

    def render_epoch(self, event: CommittedEpochMetrics) -> None:
        if not isinstance(event, CommittedEpochMetrics):
            raise TypeError("event must be CommittedEpochMetrics")
        if not self.enabled:
            return
        try:
            now = _finite("monotonic clock", self._monotonic())
            if self._session_started_at is None:
                self._session_started_at = now
            elapsed = max(0.0, now - self._session_started_at)
            count = self._session_event_count + 1
            self._write(self._epoch_block(event, elapsed=elapsed, count=count))
            self._session_event_count = count
        except Exception as error:
            self._disable(error)

    def __call__(self, event: CommittedEpochMetrics) -> None:
        self.render_epoch(event)

    def render_terminal(
        self,
        status: str,
        *,
        epochs: int,
        global_step: int,
        best_epoch: int | None = None,
        best_value: float | None = None,
        latest_checkpoint: str | None = None,
        reason: str | None = None,
        phase: str | None = None,
        recoverable: str | None = None,
    ) -> None:
        # --quiet suppresses progress only; terminal information remains. A
        # failed stream, however, is never written again.
        if not self._io_available:
            return
        try:
            self._write(
                self._terminal_line(
                    status,
                    epochs=epochs,
                    global_step=global_step,
                    best_epoch=best_epoch,
                    best_value=best_value,
                    latest_checkpoint=latest_checkpoint,
                    reason=reason,
                    phase=phase,
                    recoverable=recoverable,
                )
            )
        except Exception as error:
            self._disable(error)

    def _write(self, value: str) -> None:
        self._stream.write(value)
        self._stream.flush()

    def _disable(self, error: Exception) -> None:
        if self._presentation_error is None:
            self._presentation_error = TrainingProgressError(
                "training console progress was disabled after a presentation failure",
                original_error=error,
            )
        self._io_available = False

    def _float(self, value: float) -> str:
        return format(_finite("display value", value), f".{self._config.float_precision}g")

    def _duration(self, value: float) -> str:
        seconds = _finite("duration", value, minimum=0.0)
        return f"{seconds:.{self._config.duration_precision}f}s"

    def _start_block(self, summary: TrainingStartSummary) -> str:
        source = summary.source_kind + (" (resumed)" if summary.resumed else "")
        init_seed = (
            "n/a"
            if summary.initialization_seed is None
            else str(summary.initialization_seed)
        )
        species = ", ".join(
            _species_text(value) for value in summary.species_vocabulary
        )
        templates = ", ".join(
            f"{template_id} (M={sites}"
            + (", default)" if template_id == summary.default_template_id else ")")
            for template_id, sites in summary.templates
        )
        labels = ", ".join(
            f"{_term_abbreviation(term)}="
            f"{_presence_text(train_present, train_missing, valid_present, valid_missing)}"
            for term, train_present, train_missing, valid_present, valid_missing
            in summary.label_presence
        )
        baseline = "disabled"
        if summary.baseline_enabled:
            values = ", ".join(self._float(value) for value in summary.baseline_values)
            baseline = (
                f"{summary.baseline_rank_policy}/{summary.baseline_status} [{values}]"
            )
        loss = ", ".join(
            f"{_term_abbreviation(term)}={self._float(weight)}"
            f" (scale={self._float(scale)})"
            for term, weight, scale in summary.loss_terms
        )
        lines = [
            "Reference-site MLIP training",
            f"  Run: {summary.run_name}",
            f"  Source: {source}",
            f"  Device: {summary.device}, dtype={summary.dtype}",
            f"  Seeds: training={summary.training_seed}, initialization={init_seed}",
            f"  Model: {summary.parameter_element_count} parameters "
            f"({summary.parameter_tensor_count} tensors)",
            f"  Species: {species}",
            f"  Templates: {templates}",
            f"  Data: train={summary.train_frame_count} frames/"
            f"{summary.train_batch_count} batches (batch={summary.train_batch_size}), "
            f"validation={summary.validation_frame_count} frames/"
            f"{summary.validation_batch_count} batches "
            f"(batch={summary.validation_batch_size})",
            f"  Labels: {labels}",
            f"  Radii: r_ot={self._float(summary.r_ot)} A, "
            f"r_mp={self._float(summary.r_mp)} A, candidates=("
            f"{self._float(summary.r_candidate_ot)}, "
            f"{self._float(summary.r_candidate_mp)}) A",
            f"  Transport: backend={summary.ot_backend}, solver={summary.solver_path}",
            f"  Baseline: {baseline}",
            f"  Loss weights: {loss}",
            f"  Optimizer: {summary.optimizer_kind}, "
            f"lr={self._float(summary.initial_learning_rate)}, "
            f"weight_decay={self._float(summary.weight_decay)}",
            f"  Scheduler: {summary.scheduler_kind}, "
            f"monitor={summary.scheduler_monitor}/{summary.scheduler_mode}",
            f"  Epochs: {summary.max_epochs}, early-stop patience="
            f"{_optional_text(summary.early_stop_patience)}",
            f"  Output: {summary.output_directory}",
            f"  Initial bundle: {summary.initial_bundle_fingerprint}",
            f"  Data digests: train={summary.train_semantic_digest}, "
            f"validation={summary.validation_semantic_digest}",
        ]
        if summary.template_frame_counts:
            lines.append(
                "  Template frames: "
                + ", ".join(
                    f"{template_id}=train:{train}/validation:{validation}"
                    for template_id, train, validation
                    in summary.template_frame_counts
                )
            )
        if summary.composition_summary:
            lines.append(
                "  Compositions: "
                + ", ".join(
                    f"{composition}=train:{train}/validation:{validation}"
                    for composition, train, validation in summary.composition_summary
                )
            )
        if summary.resumed:
            lines.append(
                "  Resume: checkpoint_epoch="
                f"{_human_epoch(summary.resume_checkpoint_epoch)}, "
                f"step={summary.resume_global_step}, "
                f"requested_epochs={summary.max_epochs}, "
                f"best_epoch={_human_epoch(summary.existing_best_epoch)}, "
                f"best={_optional_float(summary.existing_best_value, self._float)}, "
                f"journal_recovered={summary.recovered_journal_event_count}"
            )
        return "\n".join(lines) + "\n"

    def _epoch_block(
        self,
        event: CommittedEpochMetrics,
        *,
        elapsed: float,
        count: int,
    ) -> str:
        maximum = (
            self._summary.max_epochs
            if self._summary is not None
            else event.epoch_index + 1
        )
        width = max(3, len(str(maximum)))
        human_epoch = event.epoch_index + 1
        remaining = max(0, maximum - human_epoch)
        # A committed early-stop decision is terminal: reporting time for
        # epochs which will not run would contradict the same line's
        # ``stop=yes`` state.
        eta = 0.0 if event.should_stop else elapsed / count * remaining
        return (
            f"Epoch {human_epoch:0{width}d}/{maximum} | step={event.global_step_end}\n"
            f"  train [{event.training_metric_semantics}] "
            f"total={self._float(event.training_total_loss)} "
            f"E={_term_mean(event.training_energy, self._float)} "
            f"F={_term_mean(event.training_force, self._float)} "
            f"S={_term_mean(event.training_stress, self._float)}\n"
            f"  valid [{event.validation_metric_semantics}] "
            f"total={self._float(event.validation_total_loss)} "
            f"E={_term_mean(event.validation_energy, self._float)} "
            f"F={_term_mean(event.validation_force, self._float)} "
            f"S={_term_mean(event.validation_stress, self._float)}\n"
            f"  lr(before)="
            f"{_learning_rates(event.learning_rates_before_scheduler, self._float)} "
            f"lr(next)="
            f"{_learning_rates(event.learning_rates_after_scheduler, self._float)} "
            f"best={'yes' if event.is_best else 'no'} "
            f"stop={'yes' if event.should_stop else 'no'} "
            f"checkpoint={event.epoch_checkpoint_basename} "
            f"elapsed={self._duration(elapsed)} eta={self._duration(eta)}\n"
        )

    def _terminal_line(
        self,
        status: str,
        *,
        epochs: int,
        global_step: int,
        best_epoch: int | None,
        best_value: float | None,
        latest_checkpoint: str | None,
        reason: str | None,
        phase: str | None,
        recoverable: str | None,
    ) -> str:
        status = _string("status", status)
        epochs = _integer("epochs", epochs)
        global_step = _integer("global_step", global_step)
        best_epoch = _optional_integer("best_epoch", best_epoch)
        best_value = _optional_finite("best_value", best_value)
        latest_checkpoint = _optional_single_line(latest_checkpoint)
        reason = _optional_single_line(reason)
        phase = _optional_single_line(phase)
        recoverable = _optional_single_line(recoverable)
        if status == "completed":
            return (
                f"Training completed | epochs={epochs} step={global_step} "
                f"best_epoch={_human_epoch(best_epoch)} "
                f"best={_optional_float(best_value, self._float)} "
                f"latest={_optional_text(latest_checkpoint)}\n"
            )
        if status == "early_stopped":
            return (
                f"Training early-stopped | epoch={epochs} step={global_step} "
                f"best_epoch={_human_epoch(best_epoch)} "
                f"reason={_optional_text(reason)}\n"
            )
        if status == "failed":
            return (
                f"Training failed | phase={_optional_text(phase)} epoch={epochs} "
                f"step={global_step} recoverable={_optional_text(recoverable)}\n"
            )
        if status == "interrupted":
            return (
                f"Training interrupted | epoch={epochs} step={global_step} "
                f"recoverable={_optional_text(recoverable)}\n"
            )
        raise ValueError(
            "status must be completed, early_stopped, failed, or interrupted"
        )


@dataclass(frozen=True)
class _JournalThenProgressObserver:
    journal: EpochMetricsObserver
    renderer: TrainingProgressRenderer

    def __post_init__(self) -> None:
        if not callable(self.journal):
            raise TypeError("journal must be callable")
        if not isinstance(self.renderer, TrainingProgressRenderer):
            raise TypeError("renderer must be a TrainingProgressRenderer")

    def __call__(self, event: CommittedEpochMetrics) -> None:
        # Persistent truth must exist before the presentation reports it. A
        # journal error intentionally prevents the renderer call; renderer I/O
        # errors are captured internally and never escape this observer.
        self.journal(event)
        self.renderer.render_epoch(event)


def journal_then_progress_observer(
    journal: EpochMetricsObserver,
    renderer: TrainingProgressRenderer,
) -> EpochMetricsObserver:
    """Compose journal persistence followed by best-effort presentation."""

    return _JournalThenProgressObserver(journal=journal, renderer=renderer)


def _single_line(value: object) -> str:
    return " ".join(_string("message", value).splitlines())


def _optional_single_line(value: object | None) -> str | None:
    return None if value is None else _single_line(value)


def _optional_text(value: object | None) -> str:
    return "n/a" if value is None else _single_line(str(value))


def _optional_float(
    value: float | None, formatter: Callable[[float], str]
) -> str:
    return "n/a" if value is None else formatter(value)


def _human_epoch(value: int | None) -> str:
    return "n/a" if value is None else str(value + 1)


def _term_abbreviation(term: str) -> str:
    return {"energy": "E", "force": "F", "stress": "S"}[term]


def _species_text(atomic_number: int) -> str:
    # ASE is a required runtime dependency. Import lazily so importing the CLI
    # presentation API itself remains lightweight.
    from ase.data import chemical_symbols

    if atomic_number < len(chemical_symbols) and chemical_symbols[atomic_number]:
        return f"{chemical_symbols[atomic_number]}({atomic_number})"
    return f"Z={atomic_number}"


def _presence_text(
    train_present: int,
    train_missing: int,
    validation_present: int,
    validation_missing: int,
) -> str:
    def part(present: int, missing: int) -> str:
        if present == 0:
            return "no"
        if missing == 0:
            return "yes"
        return f"partial({present}/{present + missing})"

    return (
        f"train:{part(train_present, train_missing)}/"
        f"validation:{part(validation_present, validation_missing)}"
    )


def _term_mean(
    metrics: tuple[float, float, float, int],
    formatter: Callable[[float], str],
) -> str:
    return "n/a" if metrics[1] == 0.0 else formatter(metrics[2])


def _learning_rates(
    values: tuple[float, ...], formatter: Callable[[float], str]
) -> str:
    if len(values) == 1:
        return formatter(values[0])
    return "[" + ",".join(formatter(value) for value in values) + "]"


__all__ = [
    "TRAINING_PROGRESS_CONFIG_VERSION",
    "TrainingProgressConfig",
    "TrainingProgressError",
    "TrainingProgressRenderer",
    "TrainingStartSummary",
    "journal_then_progress_observer",
]
