"""Exact epoch-boundary continuation and full-history result composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import math
from numbers import Integral
from typing import Any

import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.models.template_context import TemplateExecutionContext

from .checkpoint import TrainingCheckpoint, _plain
from .fit import FitConfig, FitEpochRecord, FitExecutionError, FitResult, run_fit
from .losses import LossConfig
from .resume import ResumePolicy, ResumeState, restore_training_checkpoint_
from .scheduler import SchedulerConfig
from .selection import ModelSelectionConfig, ModelSelectionState
from .step import TrainStepConfig
from .validation import ValidationStepConfig


def _nonnegative_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _finite_rates(name: str, values) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"{name} must contain at least one learning rate")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} must contain finite nonnegative rates")
    return result


def _record_selection_matches(record: FitEpochRecord) -> None:
    state = record.selection_state_after_epoch
    decision = record.decision
    fields = (
        ("best_metric", decision.best_metric, state.best_metric),
        ("best_epoch", decision.best_epoch, state.best_epoch),
        ("best_global_step", decision.best_global_step, state.best_global_step),
        (
            "epochs_since_improvement",
            decision.epochs_since_improvement,
            state.epochs_since_improvement,
        ),
        ("validation_events", decision.validation_events, state.validation_events),
    )
    for name, left, right in fields:
        if left != right:
            raise ValueError(
                f"history epoch {record.epoch_index} decision/state {name} mismatch"
            )
    if decision.should_stop != state.stopped_early:
        raise ValueError(
            f"history epoch {record.epoch_index} stop decision/state mismatch"
        )


def _history_records(checkpoint: TrainingCheckpoint) -> tuple[FitEpochRecord, ...]:
    history = checkpoint.fit_history
    if history is None:
        raise ValueError("exact resumed fitting requires full checkpoint fit_history")
    if not isinstance(history, (tuple, list)):
        raise TypeError("checkpoint fit_history must be a tuple or list")
    records = []
    for index, value in enumerate(history):
        if isinstance(value, FitEpochRecord):
            record = FitEpochRecord.from_dict(value.to_dict())
        elif isinstance(value, Mapping):
            try:
                record = FitEpochRecord.from_dict(value)
            except Exception as error:
                raise ValueError(
                    f"checkpoint fit_history[{index}] is invalid: {error}"
                ) from error
        else:
            raise TypeError(
                f"checkpoint fit_history[{index}] must be a FitEpochRecord mapping"
            )
        records.append(record)
    if not records:
        raise ValueError("exact resumed fitting requires nonempty fit_history")
    return tuple(records)


def validate_checkpoint_history(
    checkpoint: TrainingCheckpoint,
) -> tuple[FitEpochRecord, ...]:
    """Return owned records after strict history/progress/LR continuity checks.

    This function is read-only.  It never mutates the checkpoint or any live
    model, optimizer, scheduler, or RNG state.
    """

    if not isinstance(checkpoint, TrainingCheckpoint):
        raise TypeError("checkpoint must be a TrainingCheckpoint")
    progress = checkpoint.progress
    selection = checkpoint.selection_state
    if progress.stopped_early or selection.stopped_early:
        raise ValueError("an already early-stopped checkpoint cannot be resumed")
    if progress.next_batch_index != 0:
        raise ValueError("exact resumed fitting only supports epoch boundaries")
    records = _history_records(checkpoint)
    if len(records) != progress.completed_epochs:
        raise ValueError(
            "fit_history length does not match checkpoint completed_epochs"
        )
    saved_fit = FitConfig.from_dict(
        checkpoint.metadata.resolved_configuration["fit"]
    )
    expected_epochs = tuple(
        range(saved_fit.start_epoch, saved_fit.start_epoch + len(records))
    )
    actual_epochs = tuple(record.epoch_index for record in records)
    if actual_epochs != expected_epochs:
        raise ValueError(
            "fit_history epoch indices must be contiguous without duplicates or gaps"
        )
    if progress.last_completed_epoch != records[-1].epoch_index:
        raise ValueError(
            "last history epoch does not match progress.last_completed_epoch"
        )
    if progress.next_epoch != records[-1].epoch_index + 1:
        raise ValueError("progress.next_epoch must follow the last history epoch")

    previous_global_step = saved_fit.global_step_start
    previous_rates_after = None
    previous_selection_events = 0
    for index, record in enumerate(records):
        epoch = record.epoch_index
        if record.training.phase != "train" or record.validation.phase != "validation":
            raise ValueError(f"history epoch {epoch} has invalid train/validation phases")
        if (
            record.training.epoch_index != epoch
            or record.validation.epoch_index != epoch
        ):
            raise ValueError(f"history epoch {epoch} contains mismatched epoch metadata")
        if record.training.global_step_start != previous_global_step:
            raise ValueError(
                f"history epoch {epoch} training global-step continuity mismatch"
            )
        if (
            record.validation.global_step_start
            != record.training.global_step_end
            or record.validation.global_step_end
            != record.training.global_step_end
        ):
            raise ValueError(
                f"history epoch {epoch} validation global step does not follow training"
            )
        rates_used = _finite_rates(
            f"history epoch {epoch} training learning rates",
            record.learning_rates_used_for_training,
        )
        rates_after = _finite_rates(
            f"history epoch {epoch} post-validation learning rates",
            record.learning_rates_after_validation,
        )
        if rates_after != tuple(record.decision.learning_rates_after):
            raise ValueError(
                f"history epoch {epoch} decision/post-validation LR mismatch"
            )
        if rates_used != tuple(record.decision.learning_rates_before):
            raise ValueError(
                f"history epoch {epoch} training/validation-before learning rate mismatch"
            )
        if previous_rates_after is not None and rates_used != previous_rates_after:
            raise ValueError(
                f"history epoch {epoch} learning rate does not continue the prior event"
            )
        _record_selection_matches(record)
        state = record.selection_state_after_epoch
        if state.validation_events != previous_selection_events + 1:
            raise ValueError(
                f"history epoch {epoch} validation-event count is not continuous"
            )
        if state.last_validation_epoch != epoch:
            raise ValueError(
                f"history epoch {epoch} selection last epoch is inconsistent"
            )
        if state.last_validation_global_step != record.validation.global_step_end:
            raise ValueError(
                f"history epoch {epoch} selection global step is inconsistent"
            )
        previous_global_step = record.training.global_step_end
        previous_rates_after = rates_after
        previous_selection_events = state.validation_events

    if records[-1].training.global_step_end != progress.global_step:
        raise ValueError("last training global step does not match checkpoint progress")
    if records[-1].validation.global_step_end != progress.global_step:
        raise ValueError("last validation global step does not match checkpoint progress")
    if records[-1].selection_state_after_epoch != selection:
        raise ValueError(
            "last history selection state does not match checkpoint selection state"
        )
    if progress.best_epoch != selection.best_epoch:
        raise ValueError("progress best_epoch does not match selection state")
    if progress.best_global_step != selection.best_global_step:
        raise ValueError("progress best_global_step does not match selection state")
    saved_groups = checkpoint.optimizer_state_dict.get("param_groups", ())
    saved_rates = tuple(float(group["lr"]) for group in saved_groups)
    if previous_rates_after != saved_rates:
        raise ValueError(
            "last history learning rate does not match checkpoint optimizer state"
        )
    initial_lr = checkpoint.metadata.resolved_configuration["optimizer"].get(
        "learning_rate"
    )
    if initial_lr is not None and records[0].learning_rates_used_for_training != (
        float(initial_lr),
    ):
        raise ValueError(
            "first history learning rate does not match optimizer configuration"
        )
    return records


def _validate_execution_configs(
    checkpoint: TrainingCheckpoint,
    resolved_configs: Mapping[str, Any],
    loss_config: LossConfig,
    train_step_config: TrainStepConfig,
    validation_step_config: ValidationStepConfig,
    scheduler_config: SchedulerConfig,
    selection_config: ModelSelectionConfig,
    policy: ResumePolicy,
) -> None:
    if not isinstance(resolved_configs, Mapping):
        raise TypeError("resolved_configs must be a mapping")
    values = (
        ("loss", loss_config, LossConfig),
        ("train_step", train_step_config, TrainStepConfig),
        ("validation_step", validation_step_config, ValidationStepConfig),
        ("scheduler", scheduler_config, SchedulerConfig),
        ("model_selection", selection_config, ModelSelectionConfig),
    )
    for key, value, expected_type in values:
        if not isinstance(value, expected_type):
            raise TypeError(f"{key} config must be a {expected_type.__name__}")
        if key not in resolved_configs:
            raise ValueError(f"resolved_configs is missing {key!r}")
        public = _plain(value, path=f"resume_fit.{key}")
        resolved = _plain(resolved_configs[key], path=f"resolved_configs.{key}")
        if public != resolved:
            raise ValueError(
                f"public {key} config does not match resolved_configs before restore"
            )
    if not policy.restore_python_rng:
        raise ValueError("exact resumed fitting requires Python RNG restoration")
    if not policy.restore_numpy_rng:
        raise ValueError("exact resumed fitting requires NumPy RNG restoration")
    if not policy.restore_torch_cpu_rng:
        raise ValueError("exact resumed fitting requires Torch CPU RNG restoration")
    if checkpoint.cuda_device_count and not policy.restore_cuda_rng:
        raise ValueError("exact resumed fitting requires CUDA RNG restoration")


def _validate_continuation(
    checkpoint: TrainingCheckpoint,
    continuation: FitResult,
    resumed_max_epochs: int,
    history: tuple[FitEpochRecord, ...],
) -> None:
    progress = checkpoint.progress
    expected_config = FitConfig(
        max_epochs=resumed_max_epochs,
        start_epoch=progress.next_epoch,
        global_step_start=progress.global_step,
    )
    if continuation.config != expected_config:
        raise ValueError("continuation FitConfig is not checkpoint-derived")
    if continuation.start_epoch != progress.next_epoch:
        raise ValueError("continuation start_epoch does not match checkpoint")
    if continuation.global_step_start != progress.global_step:
        raise ValueError("continuation global_step_start does not match checkpoint")
    if not continuation.records:
        raise ValueError("continuation must contain at least one completed epoch")
    if continuation.records[0].epoch_index != progress.next_epoch:
        raise ValueError("continuation record does not begin at checkpoint next_epoch")
    expected_epochs = tuple(
        range(progress.next_epoch, progress.next_epoch + len(continuation.records))
    )
    if tuple(record.epoch_index for record in continuation.records) != expected_epochs:
        raise ValueError("continuation record epochs are not contiguous")
    prior_rates = history[-1].learning_rates_after_validation
    if continuation.records[0].learning_rates_used_for_training != prior_rates:
        raise ValueError("continuation does not use checkpoint post-validation LR")
    if (
        continuation.records[0].selection_state_after_epoch.validation_events
        != checkpoint.selection_state.validation_events + 1
    ):
        raise ValueError("continuation selection event count is not continuous")


def compose_resumed_fit_result(
    checkpoint: TrainingCheckpoint,
    continuation_fit_result: FitResult,
    *,
    resumed_max_epochs: int,
) -> FitResult:
    """Compose checkpoint history and continuation into full-fit semantics."""

    history = validate_checkpoint_history(checkpoint)
    if isinstance(resumed_max_epochs, bool) or not isinstance(
        resumed_max_epochs, Integral
    ):
        raise TypeError("resumed_max_epochs must be a positive integer")
    resumed_max_epochs = int(resumed_max_epochs)
    if resumed_max_epochs <= checkpoint.progress.next_epoch:
        raise ValueError("resumed_max_epochs must exceed checkpoint next_epoch")
    if not isinstance(continuation_fit_result, FitResult):
        raise TypeError("continuation_fit_result must be a FitResult")
    _validate_continuation(
        checkpoint, continuation_fit_result, resumed_max_epochs, history
    )
    original_fit = FitConfig.from_dict(
        checkpoint.metadata.resolved_configuration["fit"]
    )
    if resumed_max_epochs < original_fit.max_epochs:
        raise ValueError("resumed_max_epochs cannot decrease checkpoint max_epochs")
    combined_records = history + tuple(continuation_fit_result.records)
    combined_config = FitConfig(
        max_epochs=resumed_max_epochs,
        start_epoch=original_fit.start_epoch,
        global_step_start=original_fit.global_step_start,
    )
    final_state = continuation_fit_result.final_selection_state
    return FitResult(
        config=combined_config,
        records=combined_records,
        epochs_requested=resumed_max_epochs - original_fit.start_epoch,
        epochs_completed=len(combined_records),
        start_epoch=original_fit.start_epoch,
        next_epoch=continuation_fit_result.next_epoch,
        global_step_start=original_fit.global_step_start,
        global_step_end=continuation_fit_result.global_step_end,
        stopped_early=continuation_fit_result.stopped_early,
        stop_epoch=continuation_fit_result.stop_epoch,
        stop_reason=continuation_fit_result.stop_reason,
        best_metric=continuation_fit_result.best_metric,
        best_epoch=continuation_fit_result.best_epoch,
        best_global_step=continuation_fit_result.best_global_step,
        final_selection_state=final_state,
        final_learning_rates=continuation_fit_result.final_learning_rates,
        terminal_model_is_best=(
            final_state.best_epoch == continuation_fit_result.records[-1].epoch_index
        ),
    )


@dataclass(frozen=True)
class ResumedFitResult:
    combined_fit_result: FitResult
    continuation_fit_result: FitResult
    resume_state: ResumeState
    checkpoint_next_epoch: int
    resumed_epochs_completed: int
    exact_resume_conditions: tuple[str, ...]
    checkpoint_data_fingerprints: dict[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.combined_fit_result, FitResult):
            raise TypeError("combined_fit_result must be a FitResult")
        if not isinstance(self.continuation_fit_result, FitResult):
            raise TypeError("continuation_fit_result must be a FitResult")
        if not isinstance(self.resume_state, ResumeState):
            raise TypeError("resume_state must be a ResumeState")
        object.__setattr__(
            self,
            "checkpoint_next_epoch",
            _nonnegative_integer("checkpoint_next_epoch", self.checkpoint_next_epoch),
        )
        object.__setattr__(
            self,
            "resumed_epochs_completed",
            _nonnegative_integer(
                "resumed_epochs_completed", self.resumed_epochs_completed
            ),
        )
        if set(self.checkpoint_data_fingerprints) != {"train", "validation"}:
            raise ValueError("checkpoint data fingerprints require train/validation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "combined_fit_result": self.combined_fit_result.to_dict(),
            "continuation_fit_result": self.continuation_fit_result.to_dict(),
            "resume_state": self.resume_state.to_dict(),
            "checkpoint_next_epoch": self.checkpoint_next_epoch,
            "resumed_epochs_completed": self.resumed_epochs_completed,
            "exact_resume_conditions": list(self.exact_resume_conditions),
            "checkpoint_data_fingerprints": dict(
                self.checkpoint_data_fingerprints
            ),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ResumedFitResult":
        if not isinstance(values, Mapping):
            raise TypeError("resumed fit result must be reconstructed from a mapping")
        data = dict(values)
        data["combined_fit_result"] = FitResult.from_dict(
            data["combined_fit_result"]
        )
        data["continuation_fit_result"] = FitResult.from_dict(
            data["continuation_fit_result"]
        )
        data["resume_state"] = ResumeState.from_dict(data["resume_state"])
        data["exact_resume_conditions"] = tuple(data["exact_resume_conditions"])
        return cls(**data)


class ResumedFitExecutionError(RuntimeError):
    """Continuation failure after a successful transactional restore."""

    def __init__(
        self,
        *,
        checkpoint_next_epoch: int,
        checkpoint_global_step: int,
        cause: BaseException,
    ) -> None:
        self.checkpoint_next_epoch = checkpoint_next_epoch
        self.checkpoint_global_step = checkpoint_global_step
        self.failure_epoch = getattr(cause, "epoch_index", checkpoint_next_epoch)
        self.failure_phase = getattr(cause, "phase", "composition")
        self.continuation_completed_epochs = getattr(cause, "completed_epochs", 0)
        self.current_global_step = getattr(
            cause, "current_global_step", checkpoint_global_step
        )
        self.original_exception_type = type(cause).__name__
        self.original_exception_message = str(cause)
        self.rollback_performed = False
        super().__init__(
            "resumed fit continuation failed after checkpoint restore: "
            f"checkpoint_next_epoch={checkpoint_next_epoch}, "
            f"checkpoint_global_step={checkpoint_global_step}, "
            f"failure_epoch={self.failure_epoch}, phase={self.failure_phase}, "
            f"continuation_completed_epochs={self.continuation_completed_epochs}, "
            f"current_global_step={self.current_global_step}; "
            f"cause={self.original_exception_type}: "
            f"{self.original_exception_message}; completed continuation updates "
            "are retained and are not rolled back"
        )


def run_resumed_fit(
    checkpoint: TrainingCheckpoint,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_batches: Sequence[StructureBatch],
    validation_batches: Sequence[StructureBatch],
    template_contexts: Mapping[str, TemplateExecutionContext],
    loss_config: LossConfig,
    train_step_config: TrainStepConfig,
    validation_step_config: ValidationStepConfig,
    scheduler_config: SchedulerConfig,
    selection_config: ModelSelectionConfig,
    resolved_configs: Mapping[str, Any],
    *,
    resumed_max_epochs: int,
    policy: ResumePolicy | None = None,
    current_source_git_commit: str | None = None,
) -> ResumedFitResult:
    """Restore an exact boundary checkpoint and continue through max_epochs."""

    policy = ResumePolicy() if policy is None else policy
    if not isinstance(policy, ResumePolicy):
        raise TypeError("policy must be a ResumePolicy")
    history = validate_checkpoint_history(checkpoint)
    del history
    _validate_execution_configs(
        checkpoint,
        resolved_configs,
        loss_config,
        train_step_config,
        validation_step_config,
        scheduler_config,
        selection_config,
        policy,
    )
    resume_state = restore_training_checkpoint_(
        checkpoint,
        model,
        optimizer,
        scheduler,
        train_batches,
        validation_batches,
        template_contexts,
        resolved_configs,
        resumed_max_epochs=resumed_max_epochs,
        policy=policy,
        current_source_git_commit=current_source_git_commit,
    )
    try:
        continuation = run_fit(
            model,
            optimizer,
            scheduler,
            train_batches,
            validation_batches,
            template_contexts,
            loss_config,
            train_step_config,
            validation_step_config,
            scheduler_config,
            selection_config,
            resume_state.selection_state,
            resume_state.resumed_fit_config,
        )
        combined = compose_resumed_fit_result(
            checkpoint,
            continuation,
            resumed_max_epochs=resumed_max_epochs,
        )
    except Exception as error:
        raise ResumedFitExecutionError(
            checkpoint_next_epoch=checkpoint.progress.next_epoch,
            checkpoint_global_step=checkpoint.progress.global_step,
            cause=error,
        ) from error
    return ResumedFitResult(
        combined_fit_result=combined,
        continuation_fit_result=continuation,
        resume_state=resume_state,
        checkpoint_next_epoch=checkpoint.progress.next_epoch,
        resumed_epochs_completed=continuation.epochs_completed,
        exact_resume_conditions=(
            "full checkpoint history validated",
            "epoch-boundary progress is authoritative",
            "data/template/config compatibility is exact",
            "model/optimizer/scheduler/RNG restored transactionally",
            "continuation records are contiguous with checkpoint history",
        ),
        checkpoint_data_fingerprints={
            "train": checkpoint.metadata.training_data.fingerprint,
            "validation": checkpoint.metadata.validation_data.fingerprint,
        },
    )


__all__ = [
    "ResumedFitExecutionError",
    "ResumedFitResult",
    "compose_resumed_fit_result",
    "run_resumed_fit",
    "validate_checkpoint_history",
]
