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
