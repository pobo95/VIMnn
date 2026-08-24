"""Training-time objectives for reference-site potentials."""

from .baseline import (
    AtomicBaselineConfig,
    AtomicBaselineFit,
    apply_atomic_baseline_,
    fit_atomic_baseline,
)
from .losses import (
    LossConfig,
    LossTerm,
    PotentialLossOutput,
    compute_potential_loss,
)

__all__ = [
    "AtomicBaselineConfig",
    "AtomicBaselineFit",
    "LossConfig",
    "LossTerm",
    "PotentialLossOutput",
    "apply_atomic_baseline_",
    "compute_potential_loss",
    "fit_atomic_baseline",
]
