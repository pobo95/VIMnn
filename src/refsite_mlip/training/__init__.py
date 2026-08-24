"""Training-time objectives for reference-site potentials."""

from .baseline import (
    AtomicBaselineConfig,
    AtomicBaselineFit,
    apply_atomic_baseline_,
    fit_atomic_baseline,
)
from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetadata,
    FitProgress,
    TrainingCheckpoint,
    TrainingDataManifest,
    capture_training_checkpoint,
    fingerprint_batch_sequence,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .checkpoint_manager import (
    CheckpointManager,
    CheckpointManagerConfig,
    CheckpointManagerError,
    ManagedCheckpointResult,
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
from .resume import (
    CheckpointCompatibilityError,
    CheckpointRestoreError,
    ResumePolicy,
    ResumeState,
    restore_training_checkpoint_,
    validate_checkpoint_compatibility,
)
from .resume_fit import (
    ResumedFitExecutionError,
    ResumedFitResult,
    compose_resumed_fit_result,
    run_resumed_fit,
    validate_checkpoint_history,
)
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
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMetadata",
    "CheckpointManager",
    "CheckpointManagerConfig",
    "CheckpointManagerError",
    "CheckpointCompatibilityError",
    "CheckpointRestoreError",
    "EpochMetrics",
    "EpochResult",
    "EpochTermMetrics",
    "FitConfig",
    "FitEpochRecord",
    "FitExecutionError",
    "FitProgress",
    "FitResult",
    "LossConfig",
    "LossTerm",
    "ModelSelectionConfig",
    "ModelSelectionState",
    "ManagedCheckpointResult",
    "OptimizerConfig",
    "PotentialLossOutput",
    "ResumePolicy",
    "ResumeState",
    "ResumedFitExecutionError",
    "ResumedFitResult",
    "SchedulerConfig",
    "TrainStepConfig",
    "TrainStepResult",
    "TrainStepTermResult",
    "TrainingCheckpoint",
    "TrainingDataManifest",
    "ValidationDecision",
    "ValidationStepConfig",
    "ValidationStepResult",
    "ValidationTermResult",
    "apply_atomic_baseline_",
    "build_optimizer",
    "build_scheduler",
    "capture_training_checkpoint",
    "compute_potential_loss",
    "fingerprint_batch_sequence",
    "fit_atomic_baseline",
    "load_training_checkpoint",
    "process_primary_validation",
    "run_fit",
    "run_training_epoch",
    "run_validation_epoch",
    "save_training_checkpoint",
    "restore_training_checkpoint_",
    "run_resumed_fit",
    "train_step",
    "validation_step",
    "validate_checkpoint_compatibility",
    "validate_checkpoint_history",
    "compose_resumed_fit_result",
]
