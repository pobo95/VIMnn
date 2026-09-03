"""Optimizer construction for deterministic reference-site training steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Literal, Mapping

import torch


OptimizerName = Literal["adamw"]


def _finite_real(name: str, value: Real, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    invalid = result <= 0.0 if positive else result < 0.0
    if not math.isfinite(result) or invalid:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True)
class OptimizerConfig:
    optimizer: OptimizerName = "adamw"
    learning_rate: float = 1.0e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.0
    amsgrad: bool = False

    def __post_init__(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError("optimizer must be 'adamw'")
        object.__setattr__(
            self,
            "learning_rate",
            _finite_real("learning_rate", self.learning_rate, positive=True),
        )
        if not isinstance(self.betas, (tuple, list)) or len(self.betas) != 2:
            raise TypeError("betas must contain exactly two real values")
        betas = tuple(
            _finite_real(f"betas[{index}]", value, positive=False)
            for index, value in enumerate(self.betas)
        )
        if any(value >= 1.0 for value in betas):
            raise ValueError("AdamW betas must satisfy 0 <= beta < 1")
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "eps", _finite_real("eps", self.eps, positive=True))
        object.__setattr__(
            self,
            "weight_decay",
            _finite_real("weight_decay", self.weight_decay, positive=False),
        )
        if not isinstance(self.amsgrad, bool):
            raise TypeError("amsgrad must be a bool")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["betas"] = list(self.betas)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "OptimizerConfig":
        if not isinstance(values, Mapping):
            raise TypeError("optimizer config must be reconstructed from a mapping")
        return cls(**dict(values))


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.AdamW:
    """Build one explicit AdamW group containing each trainable parameter once."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(config, OptimizerConfig):
        raise TypeError("config must be an OptimizerConfig")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    identities = [id(parameter) for parameter in parameters]
    if len(set(identities)) != len(identities):
        raise ValueError("a trainable parameter appears more than once")
    return torch.optim.AdamW(
        [{"params": parameters}],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    )


def optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Parameter, ...]:
    """Return optimizer parameters in persisted parameter-group order."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    parameters = []
    for group_index, group in enumerate(optimizer.param_groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise ValueError(
                f"optimizer parameter group {group_index} has no params sequence"
            )
        for parameter_index, parameter in enumerate(group["params"]):
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(
                    "optimizer parameters must be torch.nn.Parameter objects; "
                    f"group={group_index}, index={parameter_index}"
                )
            parameters.append(parameter)
    return tuple(parameters)


def validate_optimizer_binding(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Require one exact ordered identity binding to every trainable parameter.

    Ordering is part of the checkpoint optimizer-state contract.  The function
    is read-only and can therefore be called before any mode, gradient, state,
    or RNG mutation at every parameter-update and checkpoint boundary.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    actual = optimizer_parameters(optimizer)
    expected_named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    expected = tuple(parameter for _, parameter in expected_named)
    if not expected:
        raise ValueError("model has no trainable parameters")

    actual_ids = tuple(id(parameter) for parameter in actual)
    duplicate_count = len(actual_ids) - len(set(actual_ids))
    if duplicate_count:
        raise ValueError(
            "optimizer binding contains duplicate parameter identities; "
            f"duplicate_count={duplicate_count}"
        )

    expected_ids = tuple(id(parameter) for parameter in expected)
    actual_set = set(actual_ids)
    expected_set = set(expected_ids)
    missing = tuple(
        name
        for name, parameter in expected_named
        if id(parameter) not in actual_set
    )
    additional = sum(identity not in expected_set for identity in actual_ids)
    if missing or additional:
        raise ValueError(
            "optimizer parameters do not exactly match current model trainable "
            f"parameters by identity; missing={missing!r}, additional={additional}"
        )
    if actual_ids != expected_ids:
        raise ValueError(
            "optimizer parameters match by identity but not in current model "
            "trainable parameter order"
        )
