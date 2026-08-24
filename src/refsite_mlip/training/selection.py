"""Primary-validation model-selection and early-stopping state transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping

import torch

from .epoch import EpochResult, VALIDATION_METRIC_SEMANTICS
from .scheduler import SchedulerConfig, _validate_scheduler_binding


_MONITORS = ("total_loss", "energy", "force", "stress")


def _finite_nonnegative(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _optional_nonnegative_integer(name: str, value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer or None")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")
    return result


@dataclass(frozen=True)
class ModelSelectionConfig:
    monitor: str = "total_loss"
    mode: str = "min"
    min_delta: float = 0.0
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if self.monitor not in _MONITORS:
            raise ValueError(
                "monitor must be 'total_loss', 'energy', 'force', or 'stress'"
            )
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        object.__setattr__(
            self, "min_delta", _finite_nonnegative("min_delta", self.min_delta)
        )
        object.__setattr__(
            self,
            "early_stopping_patience",
            _optional_nonnegative_integer(
                "early_stopping_patience", self.early_stopping_patience
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelSelectionConfig":
        if not isinstance(values, Mapping):
            raise TypeError("selection config must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class ModelSelectionState:
    best_metric: float | None = None
    best_epoch: int | None = None
    best_global_step: int | None = None
    epochs_since_improvement: int = 0
    validation_events: int = 0
    last_validation_epoch: int | None = None
    last_validation_global_step: int | None = None
    stopped_early: bool = False
    stop_epoch: int | None = None
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.best_metric is not None and not math.isfinite(float(self.best_metric)):
            raise ValueError("best_metric must be finite or None")
        for name in (
            "best_epoch",
            "best_global_step",
            "last_validation_epoch",
            "last_validation_global_step",
            "stop_epoch",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _optional_nonnegative_integer(name, value)
                )
        for name in ("epochs_since_improvement", "validation_events"):
            value = _optional_nonnegative_integer(name, getattr(self, name))
            object.__setattr__(self, name, value)
        if not isinstance(self.stopped_early, bool):
            raise TypeError("stopped_early must be a bool")
        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise TypeError("stop_reason must be a string or None")
        empty = self.validation_events == 0
        if empty != (self.last_validation_epoch is None):
            raise ValueError("validation event count and last epoch are inconsistent")
        if empty != (self.last_validation_global_step is None):
            raise ValueError("validation event count and last global step are inconsistent")
        if (self.best_metric is None) != (self.best_epoch is None):
            raise ValueError("best metric and best epoch must be set together")
        if (self.best_metric is None) != (self.best_global_step is None):
            raise ValueError("best metric and best global step must be set together")
        if self.stopped_early != (self.stop_epoch is not None):
            raise ValueError("stopped_early and stop_epoch are inconsistent")
        if self.stopped_early != (self.stop_reason is not None):
            raise ValueError("stopped_early and stop_reason are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelSelectionState":
        if not isinstance(values, Mapping):
            raise TypeError("selection state must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class ValidationDecision:
    metric_name: str
    metric_value: float
    is_best: bool
    should_stop: bool
    best_metric: float
    best_epoch: int
    best_global_step: int
    epochs_since_improvement: int
    validation_events: int
    learning_rates_before: tuple[float, ...]
    learning_rates_after: tuple[float, ...]
    scheduler_stepped: bool
    learning_rate_changed: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["learning_rates_before"] = list(self.learning_rates_before)
        result["learning_rates_after"] = list(self.learning_rates_after)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ValidationDecision":
        if not isinstance(values, Mapping):
            raise TypeError("validation decision must be reconstructed from a mapping")
        data = dict(values)
        data["learning_rates_before"] = tuple(data["learning_rates_before"])
        data["learning_rates_after"] = tuple(data["learning_rates_after"])
        return cls(**data)


def _extract_metric(result: EpochResult, monitor: str) -> float:
    if monitor == "total_loss":
        value = float(result.total_loss)
    else:
        term = getattr(result, monitor)
        if not math.isfinite(float(term.denominator)) or term.denominator <= 0.0:
            raise ValueError(
                f"monitored validation term {monitor!r} has no valid denominator"
            )
        value = float(term.mean)
    if not math.isfinite(value):
        raise ValueError(f"monitored validation metric {monitor!r} must be finite")
    return value


def _validate_event(result: EpochResult, state: ModelSelectionState) -> None:
    if not isinstance(result, EpochResult):
        raise TypeError("validation_epoch_result must be an EpochResult")
    if result.phase != "validation":
        raise ValueError("primary model selection requires a validation EpochResult")
    if result.metric_semantics != VALIDATION_METRIC_SEMANTICS:
        raise ValueError(
            "primary model selection requires fixed_model_validation semantics "
            "from the TRAIN_FIXED validation path"
        )
    if result.global_step_start != result.global_step_end:
        raise ValueError("validation must not change the global optimizer step")
    if result.successful_optimizer_steps != 0:
        raise ValueError("validation must not contain optimizer steps")
    if not result.has_supervision:
        raise ValueError("primary validation must contain valid supervision")
    if state.stopped_early:
        raise ValueError("model selection has already stopped early")
    if state.validation_events > 0:
        if result.epoch_index <= state.last_validation_epoch:
            raise ValueError(
                "validation event epoch is duplicate, stale, or out of order"
            )
        if result.global_step_end <= state.last_validation_global_step:
            raise ValueError(
                "validation event global step is duplicate, stale, or out of order"
            )


def process_primary_validation(
    optimizer: torch.optim.Optimizer,
    scheduler,
    validation_epoch_result: EpochResult,
    scheduler_config: SchedulerConfig,
    selection_config: ModelSelectionConfig,
    selection_state: ModelSelectionState,
) -> tuple[ModelSelectionState, ValidationDecision]:
    """Validate, schedule once, and return an immutable successor state.

    All event and compatibility checks precede scheduler.step. Thus an invalid
    event cannot mutate optimizer learning rates, scheduler state, or state.
    """

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(scheduler_config, SchedulerConfig):
        raise TypeError("scheduler_config must be a SchedulerConfig")
    if not isinstance(selection_config, ModelSelectionConfig):
        raise TypeError("selection_config must be a ModelSelectionConfig")
    if not isinstance(selection_state, ModelSelectionState):
        raise TypeError("selection_state must be a ModelSelectionState")
    if scheduler_config.monitor != selection_config.monitor:
        raise ValueError("scheduler and model selection must monitor the same metric")
    if scheduler_config.mode != selection_config.mode:
        raise ValueError("scheduler and model selection must use the same mode")
    _validate_scheduler_binding(optimizer, scheduler, scheduler_config)
    _validate_event(validation_epoch_result, selection_state)
    metric = _extract_metric(validation_epoch_result, selection_config.monitor)

    first = selection_state.validation_events == 0
    if first:
        improved = True
    elif selection_config.mode == "min":
        improved = metric < selection_state.best_metric - selection_config.min_delta
    else:
        improved = metric > selection_state.best_metric + selection_config.min_delta

    if improved:
        best_metric = metric
        best_epoch = validation_epoch_result.epoch_index
        best_global_step = validation_epoch_result.global_step_end
        epochs_since_improvement = 0
    else:
        best_metric = selection_state.best_metric
        best_epoch = selection_state.best_epoch
        best_global_step = selection_state.best_global_step
        epochs_since_improvement = selection_state.epochs_since_improvement + 1

    should_stop = (
        not first
        and selection_config.early_stopping_patience is not None
        and epochs_since_improvement
        >= selection_config.early_stopping_patience
    )
    learning_rates_before = tuple(
        float(group["lr"]) for group in optimizer.param_groups
    )
    scheduler.step(metric)
    learning_rates_after = tuple(
        float(group["lr"]) for group in optimizer.param_groups
    )
    validation_events = selection_state.validation_events + 1
    new_state = ModelSelectionState(
        best_metric=best_metric,
        best_epoch=best_epoch,
        best_global_step=best_global_step,
        epochs_since_improvement=epochs_since_improvement,
        validation_events=validation_events,
        last_validation_epoch=validation_epoch_result.epoch_index,
        last_validation_global_step=validation_epoch_result.global_step_end,
        stopped_early=should_stop,
        stop_epoch=(validation_epoch_result.epoch_index if should_stop else None),
        stop_reason=(
            "early-stopping patience exhausted without metric improvement"
            if should_stop
            else None
        ),
    )
    decision = ValidationDecision(
        metric_name=selection_config.monitor,
        metric_value=metric,
        is_best=improved,
        should_stop=should_stop,
        best_metric=best_metric,
        best_epoch=best_epoch,
        best_global_step=best_global_step,
        epochs_since_improvement=epochs_since_improvement,
        validation_events=validation_events,
        learning_rates_before=learning_rates_before,
        learning_rates_after=learning_rates_after,
        scheduler_stepped=True,
        learning_rate_changed=learning_rates_before != learning_rates_after,
    )
    return new_state, decision
