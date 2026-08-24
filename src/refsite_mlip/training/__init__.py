"""Training-time objectives for reference-site potentials."""

from .baseline import (
    AtomicBaselineConfig,
    AtomicBaselineFit,
    apply_atomic_baseline_,
    fit_atomic_baseline,
)
from .epoch import (
    EpochMetrics,
    EpochResult,
    EpochTermMetrics,
    run_training_epoch,
    run_validation_epoch,
)
from .fit import (
    FitConfig,
    FitEpochRecord,
    FitExecutionError,
    FitResult,
    run_fit,
)
from .losses import (
    LossConfig,
    LossTerm,
    PotentialLossOutput,
    compute_potential_loss,
)
from .optimizer import OptimizerConfig, build_optimizer
from .scheduler import SchedulerConfig, build_scheduler
from .selection import (
    ModelSelectionConfig,
    ModelSelectionState,
    ValidationDecision,
    process_primary_validation,
)
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
    "EpochMetrics",
    "EpochResult",
    "EpochTermMetrics",
    "FitConfig",
    "FitEpochRecord",
    "FitExecutionError",
    "FitResult",
    "LossConfig",
    "LossTerm",
    "ModelSelectionConfig",
    "ModelSelectionState",
    "OptimizerConfig",
    "PotentialLossOutput",
    "SchedulerConfig",
    "TrainStepConfig",
    "TrainStepResult",
    "TrainStepTermResult",
    "ValidationDecision",
    "ValidationStepConfig",
    "ValidationStepResult",
    "ValidationTermResult",
    "apply_atomic_baseline_",
    "build_optimizer",
    "build_scheduler",
    "compute_potential_loss",
    "fit_atomic_baseline",
    "process_primary_validation",
    "run_fit",
    "run_training_epoch",
    "run_validation_epoch",
    "train_step",
    "validation_step",
]
