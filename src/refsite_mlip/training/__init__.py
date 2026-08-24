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
from .optimizer import OptimizerConfig, build_optimizer
from .step import (
    TrainStepConfig,
    TrainStepResult,
    TrainStepTermResult,
    train_step,
)
from .validation import (
    ValidationStepConfig,
    ValidationStepResult,
    ValidationTermResult,
    validation_step,
)

__all__ = [
    "AtomicBaselineConfig",
    "AtomicBaselineFit",
    "LossConfig",
    "LossTerm",
    "OptimizerConfig",
    "PotentialLossOutput",
    "TrainStepConfig",
    "TrainStepResult",
    "TrainStepTermResult",
    "ValidationStepConfig",
    "ValidationStepResult",
    "ValidationTermResult",
    "apply_atomic_baseline_",
    "build_optimizer",
    "compute_potential_loss",
    "fit_atomic_baseline",
    "train_step",
    "validation_step",
]
