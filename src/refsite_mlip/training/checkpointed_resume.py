"""Exact checkpointed continuation in an existing managed run directory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.models.template_context import TemplateExecutionContext

from .checkpoint import CheckpointMetadata, TrainingCheckpoint
from .checkpoint_manager import CheckpointManager
from .checkpointed_fit import (
    CheckpointedFitConfig,
    CheckpointedFitResult,
    _run_checkpointed_epochs,
    _validate_manager_preflight,
    _validate_static_metadata,
)
from .fit import FitConfig, FitResult
from .losses import LossConfig
from .resume import (
    CheckpointCompatibilityError,
    ResumePolicy,
    ResumeState,
    restore_training_checkpoint_,
)
from .resume_fit import compose_resumed_fit_result, validate_checkpoint_history
from .scheduler import SchedulerConfig
from .selection import ModelSelectionConfig
from .step import TrainStepConfig
from .validation import ValidationStepConfig


def _tree_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.shape == second.shape
            and first.dtype == second.dtype
            and torch.equal(first.detach().cpu(), second.detach().cpu())
        )
    if isinstance(first, Mapping):
        return (
            isinstance(second, Mapping)
            and first.keys() == second.keys()
            and all(_tree_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (tuple, list)):
        return (
            isinstance(second, (tuple, list))
            and len(first) == len(second)
            and all(_tree_equal(left, right) for left, right in zip(first, second))
        )
    return first == second


def _compatibility_error(message: str) -> None:
    raise CheckpointCompatibilityError(message)


def validate_managed_checkpoint_history(
    manager: CheckpointManager,
    latest: TrainingCheckpoint,
) -> tuple[int, ...]:
    """Validate immutable epoch/latest/best continuity without writing files."""

    if not isinstance(manager, CheckpointManager):
        raise TypeError("manager must be a CheckpointManager")
    if not isinstance(latest, TrainingCheckpoint):
        raise TypeError("latest must be a TrainingCheckpoint")
    latest_history = validate_checkpoint_history(latest)
    saved_fit = FitConfig.from_dict(
        latest.metadata.resolved_configuration["fit"]
    )
    expected_epochs = tuple(range(saved_fit.start_epoch, latest.progress.next_epoch))
    actual_epochs = manager.list_epochs()
    if actual_epochs != expected_epochs:
        _compatibility_error(
            "managed epoch files must be contiguous, unique, and end at latest; "
            f"expected={expected_epochs}, actual={actual_epochs}"
        )

    for offset, epoch_index in enumerate(actual_epochs):
        snapshot = manager.load_epoch(epoch_index)
        snapshot_history = validate_checkpoint_history(snapshot)
        if snapshot.progress.last_completed_epoch != epoch_index:
            _compatibility_error(
                f"managed epoch file {epoch_index} contains different progress"
            )
        expected_prefix = latest_history[: offset + 1]
        if snapshot_history != expected_prefix:
            _compatibility_error(
                f"managed epoch file {epoch_index} history is not a latest-history prefix"
            )

    terminal = manager.load_epoch(actual_epochs[-1])
    if not _tree_equal(terminal.to_dict(), latest.to_dict()):
        _compatibility_error(
            "latest checkpoint does not equal the terminal immutable epoch snapshot"
        )

    best = manager.load_best()
    best_history = validate_checkpoint_history(best)
    best_epoch = latest.selection_state.best_epoch
    if best_epoch is None or best.progress.last_completed_epoch != best_epoch:
        _compatibility_error(
            "best checkpoint does not match latest selection best_epoch"
        )
    prefix_length = best_epoch - saved_fit.start_epoch + 1
    if prefix_length <= 0 or best_history != latest_history[:prefix_length]:
        _compatibility_error(
            "best checkpoint history is not the selected latest-history prefix"
        )
    return actual_epochs


def _positive_extension(checkpoint: TrainingCheckpoint, resumed_max_epochs: int) -> int:
    if isinstance(resumed_max_epochs, bool) or not isinstance(
        resumed_max_epochs, Integral
    ):
        raise TypeError("resumed_max_epochs must be a positive integer")
    result = int(resumed_max_epochs)
    saved_fit = FitConfig.from_dict(
        checkpoint.metadata.resolved_configuration["fit"]
    )
    if result <= saved_fit.max_epochs:
        _compatibility_error(
            "resumed max_epochs must strictly increase the latest checkpoint value; "
            f"checkpoint={saved_fit.max_epochs}, requested={result}"
        )
    return result


def _resumed_metadata(
    checkpoint: TrainingCheckpoint,
    full_fit_config: FitConfig,
) -> CheckpointMetadata:
    payload = checkpoint.metadata.to_dict()
    payload["resolved_configuration"]["fit"] = full_fit_config.to_dict()
    return CheckpointMetadata.from_dict(payload)


@dataclass(frozen=True)
class CheckpointedResumeResult:
    """Full-history fit result and files produced by one resume invocation."""

    fit_result: FitResult
    continuation_fit_result: FitResult
    resume_state: ResumeState
    checkpointed_fit_result: CheckpointedFitResult
    previous_epochs: tuple[int, ...]
    resumed_max_epochs: int

    def __post_init__(self) -> None:
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be a FitResult")
        if not isinstance(self.continuation_fit_result, FitResult):
            raise TypeError("continuation_fit_result must be a FitResult")
        if not isinstance(self.resume_state, ResumeState):
            raise TypeError("resume_state must be a ResumeState")
        if not isinstance(self.checkpointed_fit_result, CheckpointedFitResult):
            raise TypeError(
                "checkpointed_fit_result must be a CheckpointedFitResult"
            )
        if isinstance(self.resumed_max_epochs, bool) or not isinstance(
            self.resumed_max_epochs, Integral
        ):
            raise TypeError("resumed_max_epochs must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_result": self.fit_result.to_dict(),
            "continuation_fit_result": self.continuation_fit_result.to_dict(),
            "resume_state": self.resume_state.to_dict(),
            "checkpointed_fit_result": self.checkpointed_fit_result.to_dict(),
            "previous_epochs": list(self.previous_epochs),
            "resumed_max_epochs": int(self.resumed_max_epochs),
        }


def run_checkpointed_resumed_fit(
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
    checkpoint_manager: CheckpointManager,
    checkpoint_config: CheckpointedFitConfig,
    *,
    resumed_max_epochs: int,
    policy: ResumePolicy | None = None,
    current_source_git_commit: str | None = None,
) -> CheckpointedResumeResult:
    """Transactionally restore latest and continue the shared checkpoint loop."""

    policy = ResumePolicy() if policy is None else policy
    if not isinstance(policy, ResumePolicy):
        raise TypeError("policy must be a ResumePolicy")
    if not isinstance(checkpoint_config, CheckpointedFitConfig):
        raise TypeError("checkpoint_config must be a CheckpointedFitConfig")
    resumed_max_epochs = _positive_extension(checkpoint, resumed_max_epochs)
    history = validate_checkpoint_history(checkpoint)
    previous_epochs = validate_managed_checkpoint_history(
        checkpoint_manager, checkpoint
    )

    saved_fit = FitConfig.from_dict(
        checkpoint.metadata.resolved_configuration["fit"]
    )
    full_fit_config = FitConfig(
        max_epochs=resumed_max_epochs,
        start_epoch=saved_fit.start_epoch,
        global_step_start=saved_fit.global_step_start,
    )
    metadata = _resumed_metadata(checkpoint, full_fit_config)
    optimizer_config = _validate_static_metadata(
        model,
        optimizer,
        train_batches,
        validation_batches,
        loss_config,
        train_step_config,
        validation_step_config,
        scheduler_config,
        selection_config,
        full_fit_config,
        metadata,
    )
    continuation_config = FitConfig(
        max_epochs=resumed_max_epochs,
        start_epoch=checkpoint.progress.next_epoch,
        global_step_start=checkpoint.progress.global_step,
    )
    runtime_checkpoint_config = CheckpointedFitConfig(
        save_every_epoch=checkpoint_config.save_every_epoch,
        require_empty_manager=False,
    )
    _validate_manager_preflight(
        checkpoint_manager, continuation_config, runtime_checkpoint_config
    )

    current_configs = dict(resolved_configs)
    current_configs["fit"] = full_fit_config
    resume_state = restore_training_checkpoint_(
        checkpoint,
        model,
        optimizer,
        scheduler,
        train_batches,
        validation_batches,
        template_contexts,
        current_configs,
        resumed_max_epochs=resumed_max_epochs,
        policy=policy,
        current_source_git_commit=current_source_git_commit,
    )
    continued = _run_checkpointed_epochs(
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
        checkpoint_manager,
        metadata,
        optimizer_config,
        history_prefix=history,
        checkpoint_fit_config=full_fit_config,
        existing_best_path=str(checkpoint_manager.root / "best.pt"),
    )
    combined = compose_resumed_fit_result(
        checkpoint,
        continued.fit_result,
        resumed_max_epochs=resumed_max_epochs,
    )
    return CheckpointedResumeResult(
        fit_result=combined,
        continuation_fit_result=continued.fit_result,
        resume_state=resume_state,
        checkpointed_fit_result=continued,
        previous_epochs=previous_epochs,
        resumed_max_epochs=resumed_max_epochs,
    )


__all__ = [
    "CheckpointedResumeResult",
    "run_checkpointed_resumed_fit",
    "validate_managed_checkpoint_history",
]
