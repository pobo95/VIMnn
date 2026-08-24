"""Schedulers driven exclusively by primary fixed-path validation events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Literal, Mapping

import torch


SchedulerKind = Literal["none", "reduce_on_plateau"]
MonitorName = Literal["total_loss", "energy", "force", "stress"]
SelectionMode = Literal["min", "max"]
ThresholdMode = Literal["abs", "rel"]


def _finite_real(name: str, value: Real, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    invalid = result <= 0.0 if positive else result < 0.0
    if not math.isfinite(result) or invalid:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _nonnegative_integer(name: str, value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for one scheduler step per valid primary validation."""

    kind: SchedulerKind = "none"
    monitor: MonitorName = "total_loss"
    mode: SelectionMode = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1.0e-4
    threshold_mode: ThresholdMode = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.kind not in ("none", "reduce_on_plateau"):
            raise ValueError("kind must be 'none' or 'reduce_on_plateau'")
        if self.monitor not in ("total_loss", "energy", "force", "stress"):
            raise ValueError(
                "monitor must be 'total_loss', 'energy', 'force', or 'stress'"
            )
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        factor = _finite_real("factor", self.factor, positive=True)
        if factor >= 1.0:
            raise ValueError("factor must satisfy 0 < factor < 1")
        object.__setattr__(self, "factor", factor)
        object.__setattr__(
            self, "patience", _nonnegative_integer("patience", self.patience)
        )
        object.__setattr__(
            self,
            "threshold",
            _finite_real("threshold", self.threshold, positive=False),
        )
        if self.threshold_mode not in ("abs", "rel"):
            raise ValueError("threshold_mode must be 'abs' or 'rel'")
        object.__setattr__(
            self, "cooldown", _nonnegative_integer("cooldown", self.cooldown)
        )
        object.__setattr__(
            self, "min_lr", _finite_real("min_lr", self.min_lr, positive=False)
        )
        object.__setattr__(self, "eps", _finite_real("eps", self.eps, positive=True))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SchedulerConfig":
        if not isinstance(values, Mapping):
            raise TypeError("scheduler config must be reconstructed from a mapping")
        return cls(**dict(values))


class _NoOpValidationScheduler:
    """Stateful no-op used to count successful validation-driven steps."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.validation_steps = 0

    def step(self, metric: float) -> None:
        if not math.isfinite(float(metric)):
            raise ValueError("scheduler metric must be finite")
        self.validation_steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"validation_steps": self.validation_steps}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("scheduler state must be a mapping")
        if set(state_dict) != {"validation_steps"}:
            raise ValueError("invalid no-op scheduler state")
        self.validation_steps = _nonnegative_integer(
            "validation_steps", state_dict["validation_steps"]
        )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
):
    """Build a scheduler without changing optimizer parameters or RNG state."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(config, SchedulerConfig):
        raise TypeError("config must be a SchedulerConfig")
    if len(optimizer.param_groups) != 1:
        raise ValueError("v1 schedulers require exactly one optimizer parameter group")
    if config.kind == "none":
        return _NoOpValidationScheduler(optimizer)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.mode,
        factor=config.factor,
        patience=config.patience,
        threshold=config.threshold,
        threshold_mode=config.threshold_mode,
        cooldown=config.cooldown,
        min_lr=config.min_lr,
        eps=config.eps,
    )


def _validate_scheduler_binding(
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: SchedulerConfig,
) -> None:
    """Internal validation performed before any scheduler mutation."""

    if len(optimizer.param_groups) != 1:
        raise ValueError("v1 schedulers require exactly one optimizer parameter group")
    expected_type = (
        _NoOpValidationScheduler
        if config.kind == "none"
        else torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    if not isinstance(scheduler, expected_type):
        raise TypeError(f"scheduler does not match configured kind={config.kind!r}")
    if scheduler.optimizer is not optimizer:
        raise ValueError("scheduler is bound to a different optimizer")
