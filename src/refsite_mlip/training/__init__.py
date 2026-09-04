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
from .checkpointed_fit import (
    CheckpointedFitConfig,
    CheckpointedFitExecutionError,
    CheckpointedFitResult,
    run_checkpointed_fit,
)
from .checkpointed_resume import (
    CheckpointedResumeResult,
    run_checkpointed_resumed_fit,
    validate_managed_checkpoint_history,
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
from .optimizer import (
    OptimizerConfig,
    build_optimizer,
    optimizer_parameters,
    validate_optimizer_binding,
)
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
from .scratch_preparation import (
    SCRATCH_DATA_MANIFEST_CONVENTION_VERSION,
    SCRATCH_PREPARATION_CONVENTION_VERSION,
    ScratchTrainingPreparation,
    prepare_scratch_training_run,
)
from .selection import (
    ModelSelectionConfig,
    ModelSelectionState,
    ValidationDecision,
    process_primary_validation,
)
from .run_directory import (
    RESUME_LOCK_FILENAME,
    RUN_STATUS_SCHEMA_VERSION,
    ResumeRunLock,
    RunDirectoryError,
    TrainingRunDirectory,
    canonical_runtime_json,
    load_runtime_json,
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
    "CheckpointedFitConfig",
    "CheckpointedFitExecutionError",
    "CheckpointedFitResult",
    "CheckpointedResumeResult",
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
    "RUN_STATUS_SCHEMA_VERSION",
    "RESUME_LOCK_FILENAME",
    "ResumeRunLock",
    "ResumePolicy",
    "ResumeState",
    "ResumedFitExecutionError",
    "ResumedFitResult",
    "SchedulerConfig",
    "SCRATCH_DATA_MANIFEST_CONVENTION_VERSION",
    "SCRATCH_PREPARATION_CONVENTION_VERSION",
    "ScratchTrainingPreparation",
    "RunDirectoryError",
    "TrainStepConfig",
    "TrainStepResult",
    "TrainStepTermResult",
    "TrainingCheckpoint",
    "TrainingRunDirectory",
    "TrainingDataManifest",
    "ValidationDecision",
    "ValidationStepConfig",
    "ValidationStepResult",
    "ValidationTermResult",
    "apply_atomic_baseline_",
    "build_optimizer",
    "build_scheduler",
    "canonical_runtime_json",
    "capture_training_checkpoint",
    "compute_potential_loss",
    "fingerprint_batch_sequence",
    "fit_atomic_baseline",
    "load_training_checkpoint",
    "load_runtime_json",
    "process_primary_validation",
    "prepare_scratch_training_run",
    "run_fit",
    "run_checkpointed_fit",
    "run_checkpointed_resumed_fit",
    "run_training_epoch",
    "run_validation_epoch",
    "save_training_checkpoint",
    "restore_training_checkpoint_",
    "run_resumed_fit",
    "train_step",
    "validation_step",
    "validate_checkpoint_compatibility",
    "validate_checkpoint_history",
    "validate_managed_checkpoint_history",
    "validate_optimizer_binding",
    "optimizer_parameters",
    "compose_resumed_fit_result",
]
