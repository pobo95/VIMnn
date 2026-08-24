"""Training-time objectives for reference-site potentials."""

from .losses import (
    LossConfig,
    LossTerm,
    PotentialLossOutput,
    compute_potential_loss,
)

__all__ = [
    "LossConfig",
    "LossTerm",
    "PotentialLossOutput",
    "compute_potential_loss",
]
