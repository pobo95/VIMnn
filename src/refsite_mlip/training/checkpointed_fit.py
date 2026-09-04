"""Automatic atomic checkpointing at deterministic fit epoch boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from refsite_mlip.data import StructureBatch

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_SCOPE,
    CheckpointMetadata,
    FitProgress,
    _data_manifest,
    _package_versions,
    _plain,
    _template_fingerprint_mapping,
    _unit_conventions,
    capture_training_checkpoint,
)
from .checkpoint_manager import (
    CheckpointManager,
    CheckpointManagerError,
    ManagedCheckpointResult,
)
from .fit import (
    FitConfig,
    FitEpochRecord,
    FitExecutionError,
    FitResult,
    _partial_training_steps,
    _preflight,
)
from .losses import LossConfig
from .optimizer import OptimizerConfig
from .scheduler import SchedulerConfig
from .selection import (
    ModelSelectionConfig,
    ModelSelectionState,
    process_primary_validation,
)
from .step import TrainStepConfig
from .validation import ValidationStepConfig
from .epoch import run_training_epoch, run_validation_epoch


if TYPE_CHECKING:
    from .metrics_journal import (
        CommittedEpochProvenance,
        EpochMetricsObserver,
    )


CheckpointFailureStage = Literal["capture", "manager", "epoch_observer"]


@dataclass(frozen=True)
class CheckpointedFitConfig:
    """Policy for fresh-directory, every-epoch checkpointed fitting."""

    save_every_epoch: bool = True
    require_empty_manager: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.save_every_epoch, bool):
            raise TypeError("save_every_epoch must be a bool")
        if not self.save_every_epoch:
            raise ValueError("v1 checkpointed fitting requires save_every_epoch=True")
        if not isinstance(self.require_empty_manager, bool):
            raise TypeError("require_empty_manager must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CheckpointedFitConfig":
        if not isinstance(values, Mapping):
            raise TypeError("checkpointed fit config requires a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class CheckpointedFitResult:
    """Fit result plus lightweight paths/results for each completed checkpoint."""

    fit_result: FitResult
    managed_checkpoint_results: tuple[ManagedCheckpointResult, ...]
    epoch_paths: tuple[str, ...]
    latest_path: str
    best_path: str
    epochs_checkpointed: int
    terminal_checkpoint_epoch: int
    terminal_checkpoint_global_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be a FitResult")
        if not self.managed_checkpoint_results:
            raise ValueError("checkpointed fit must contain at least one checkpoint")
        if len(self.managed_checkpoint_results) != self.epochs_checkpointed:
            raise ValueError("checkpoint result count differs from epochs_checkpointed")
        if len(self.epoch_paths) != self.epochs_checkpointed:
            raise ValueError("epoch path count differs from epochs_checkpointed")
        if self.epochs_checkpointed != self.fit_result.epochs_completed:
            raise ValueError("each completed fit epoch must have one checkpoint")
        terminal = self.managed_checkpoint_results[-1]
        if self.terminal_checkpoint_epoch != terminal.epoch_index:
            raise ValueError("terminal checkpoint epoch is inconsistent")
        if self.terminal_checkpoint_global_step != terminal.global_step:
            raise ValueError("terminal checkpoint global step is inconsistent")
        if self.terminal_checkpoint_epoch != self.fit_result.next_epoch - 1:
            raise ValueError("terminal checkpoint does not match fit terminal epoch")
        if self.terminal_checkpoint_global_step != self.fit_result.global_step_end:
            raise ValueError("terminal checkpoint does not match fit global step")
        if not self.latest_path or not self.best_path:
            raise ValueError("latest_path and best_path must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_result": self.fit_result.to_dict(),
            "managed_checkpoint_results": [
                result.to_dict() for result in self.managed_checkpoint_results
            ],
            "epoch_paths": list(self.epoch_paths),
            "latest_path": self.latest_path,
            "best_path": self.best_path,
            "epochs_checkpointed": self.epochs_checkpointed,
            "terminal_checkpoint_epoch": self.terminal_checkpoint_epoch,
            "terminal_checkpoint_global_step": self.terminal_checkpoint_global_step,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CheckpointedFitResult":
        if not isinstance(values, Mapping):
            raise TypeError("checkpointed fit result requires a mapping")
        data = dict(values)
        data["fit_result"] = FitResult.from_dict(data["fit_result"])
        data["managed_checkpoint_results"] = tuple(
            ManagedCheckpointResult.from_dict(value)
            for value in data["managed_checkpoint_results"]
        )
        data["epoch_paths"] = tuple(data["epoch_paths"])
        return cls(**data)


class CheckpointedFitExecutionError(RuntimeError):
    """Post-update checkpoint/observer failure with recoverable progress."""

    def __init__(
        self,
        *,
        failure_stage: CheckpointFailureStage,
        epoch_record: FitEpochRecord,
        completed_checkpoint_results: tuple[ManagedCheckpointResult, ...],
        cause: BaseException,
    ) -> None:
        self.failure_stage = failure_stage
        self.epoch_index = epoch_record.epoch_index
        self.global_step = epoch_record.training.global_step_end
        self.epoch_record = epoch_record
        self.completed_checkpoint_results = completed_checkpoint_results
        self.epochs_checkpointed = len(completed_checkpoint_results)
        self.manager_stage = (
            cause.stage if isinstance(cause, CheckpointManagerError) else None
        )
        self.manager_completed_stages = (
            cause.completed_stages
            if isinstance(cause, CheckpointManagerError)
            else ()
        )
        self.orphan_epoch_snapshot = (
            cause.orphan_epoch_snapshot
            if isinstance(cause, CheckpointManagerError)
            else False
        )
        self.original_error = cause
        self.original_exception_type = type(cause).__name__
        self.original_exception_message = str(cause)
        self.rollback_performed = False
        super().__init__(
            f"checkpointed fit failed: failure_stage={failure_stage!r}, "
            f"epoch_index={self.epoch_index}, global_step={self.global_step}, "
            f"epochs_checkpointed={self.epochs_checkpointed}, "
            f"manager_stage={self.manager_stage!r}, "
            f"manager_completed_stages={self.manager_completed_stages}, "
            f"orphan_epoch_snapshot={self.orphan_epoch_snapshot}; "
            f"cause={self.original_exception_type}: "
            f"{self.original_exception_message}; current epoch updates are retained "
            "and are not rolled back"
        )


def _validate_epoch_observer(
    epoch_metrics_provenance: "CommittedEpochProvenance | None",
    epoch_metrics_observer: "EpochMetricsObserver | None",
) -> None:
    """Validate the optional committed-epoch observation pair before updates."""

    if epoch_metrics_observer is None:
        if epoch_metrics_provenance is not None:
            raise ValueError(
                "epoch_metrics_provenance requires epoch_metrics_observer"
            )
        return
    if not callable(epoch_metrics_observer):
        raise TypeError("epoch_metrics_observer must be callable or None")
    if epoch_metrics_provenance is None:
        raise ValueError(
            "epoch_metrics_observer requires epoch_metrics_provenance"
        )
    # Import lazily so the observer-free checkpointed engine retains its
    # existing import and execution boundary.
    from .metrics_journal import CommittedEpochProvenance

    if not isinstance(epoch_metrics_provenance, CommittedEpochProvenance):
        raise TypeError(
            "epoch_metrics_provenance must be a CommittedEpochProvenance"
        )


def _validate_epoch_provenance_metadata(
    provenance: "CommittedEpochProvenance | None",
    checkpoint_metadata: CheckpointMetadata,
) -> None:
    """Bind observable data/template provenance to checkpoint source-of-truth."""

    if provenance is None:
        return
    if not isinstance(checkpoint_metadata, CheckpointMetadata):
        raise TypeError("checkpoint_metadata must be a CheckpointMetadata")
    from .metrics_journal import committed_epoch_provenance_from_checkpoint_metadata

    expected_provenance = committed_epoch_provenance_from_checkpoint_metadata(
        checkpoint_metadata,
        initial_bundle_fingerprint=provenance.initial_bundle_fingerprint,
    )
    comparisons = (
        (
            "training_configuration_fingerprint",
            provenance.training_configuration_fingerprint,
            expected_provenance.training_configuration_fingerprint,
        ),
        (
            "train_data_fingerprint",
            provenance.train_data_fingerprint,
            expected_provenance.train_data_fingerprint,
        ),
        (
            "validation_data_fingerprint",
            provenance.validation_data_fingerprint,
            expected_provenance.validation_data_fingerprint,
        ),
        (
            "template_fingerprints",
            provenance.template_fingerprints,
            expected_provenance.template_fingerprints,
        ),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(
                f"epoch metrics provenance {name} differs from checkpoint metadata"
            )


def _observe_committed_epoch(
    epoch_record: FitEpochRecord,
    managed_result: ManagedCheckpointResult,
    *,
    selection_mode: str,
    provenance: "CommittedEpochProvenance",
    observer: "EpochMetricsObserver",
) -> None:
    """Project and publish one event containing no live runtime references."""

    from .metrics_journal import committed_epoch_metrics_from_record

    event = committed_epoch_metrics_from_record(
        epoch_record,
        managed_result,
        selection_mode=selection_mode,
        provenance=provenance,
    )
    observer(event)


def _validate_optimizer_config(
    optimizer: torch.optim.Optimizer, config: OptimizerConfig
) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("resolved optimizer config requires an AdamW optimizer")
    if len(optimizer.param_groups) != 1:
        raise ValueError("checkpointed fitting requires one optimizer parameter group")
    group = optimizer.param_groups[0]
    comparisons = {
        "lr": config.learning_rate,
        "betas": config.betas,
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "amsgrad": config.amsgrad,
    }
    for key, expected in comparisons.items():
        actual = group[key]
        mismatch = (
            tuple(actual) != tuple(expected)
            if key == "betas"
            else actual != expected
        )
        if mismatch:
            raise ValueError(f"optimizer state differs from resolved {key} config")


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current


def _validate_manager_preflight(
    manager: CheckpointManager,
    fit_config: FitConfig,
    config: CheckpointedFitConfig,
) -> None:
    if not isinstance(manager, CheckpointManager):
        raise TypeError("checkpoint_manager must be a CheckpointManager")
    if not manager.config.save_epoch_snapshots:
        raise ValueError("checkpointed fitting requires immutable epoch snapshots")
    last_epoch = fit_config.max_epochs - 1
    if last_epoch >= 10**manager.config.epoch_filename_width:
        raise ValueError("future epoch index does not fit manager filename width")
    root = manager.root
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"checkpoint manager root is not a directory: {root}")
    access_target = root if root.exists() else _nearest_existing_parent(root)
    if not access_target.is_dir() or not os.access(access_target, os.W_OK | os.X_OK):
        raise PermissionError(f"checkpoint manager root is not writable: {root}")
    if config.require_empty_manager and root.exists():
        managed = sorted(
            entry.name
            for entry in root.iterdir()
            if entry.name in ("latest.pt", "best.pt")
            or (entry.name.startswith("epoch_") and entry.name.endswith(".pt"))
        )
        if managed:
            raise FileExistsError(
                "checkpoint manager must be empty; existing managed files="
                + repr(managed)
            )


def _trees_equal(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, Mapping):
        return (
            isinstance(second, Mapping)
            and first.keys() == second.keys()
            and all(_trees_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (tuple, list)):
        return (
            isinstance(second, (tuple, list))
            and len(first) == len(second)
            and all(_trees_equal(a, b) for a, b in zip(first, second))
        )
    return first == second


def _validate_static_metadata(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batches: Sequence[StructureBatch],
    validation_batches: Sequence[StructureBatch],
    loss_config: LossConfig,
    train_step_config: TrainStepConfig,
    validation_step_config: ValidationStepConfig,
    scheduler_config: SchedulerConfig,
    selection_config: ModelSelectionConfig,
    fit_config: FitConfig,
    metadata: CheckpointMetadata,
) -> OptimizerConfig:
    if not isinstance(metadata, CheckpointMetadata):
        raise TypeError("checkpoint_metadata must be CheckpointMetadata")
    expected_configs = {
        "loss": _plain(loss_config, path="loss_config"),
        "train_step": _plain(train_step_config, path="train_step_config"),
        "validation_step": _plain(
            validation_step_config, path="validation_step_config"
        ),
        "scheduler": _plain(scheduler_config, path="scheduler_config"),
        "model_selection": _plain(
            selection_config, path="model_selection_config"
        ),
        "fit": _plain(fit_config, path="fit_config"),
    }
    for key, expected in expected_configs.items():
        if not _trees_equal(metadata.resolved_configuration[key], expected):
            raise ValueError(f"resolved {key} config does not match runtime config")
    if hasattr(model, "config"):
        model_config = _plain(model.config, path="model.config")
        if not _trees_equal(metadata.resolved_configuration["model"], model_config):
            raise ValueError("resolved model config does not match model.config")
    optimizer_config = OptimizerConfig.from_dict(
        metadata.resolved_configuration["optimizer"]
    )
    _validate_optimizer_config(optimizer, optimizer_config)
    if hasattr(model, "config") and hasattr(model.config, "species_vocabulary"):
        if tuple(model.config.species_vocabulary) != metadata.species_vocabulary:
            raise ValueError("checkpoint species vocabulary differs from model")
    if metadata.unit_conventions != _unit_conventions():
        raise ValueError("checkpoint unit conventions differ from runtime conventions")
    if metadata.package_versions != _package_versions():
        raise ValueError("checkpoint package/version metadata differs from runtime")
    if metadata.template_fingerprints != _template_fingerprint_mapping(
        train_batches, validation_batches
    ):
        raise ValueError("checkpoint template fingerprint mapping differs from batches")
    if metadata.training_data != _data_manifest(train_batches, split_name="train"):
        raise ValueError("checkpoint training manifest differs from batches")
    if metadata.validation_data != _data_manifest(
        validation_batches, split_name="validation"
    ):
        raise ValueError("checkpoint validation manifest differs from batches")
    return optimizer_config


def _validate_checkpoint_preflight(
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
    selection_state,
    fit_config,
    checkpoint_manager,
    checkpoint_metadata,
    checkpoint_config,
):
    if not isinstance(checkpoint_config, CheckpointedFitConfig):
        raise TypeError("checkpoint_config must be a CheckpointedFitConfig")
    if (
        fit_config.start_epoch != 0
        or fit_config.global_step_start != 0
        or selection_state.validation_events != 0
    ):
        raise ValueError(
            "v1 checkpointed fitting requires fresh progress: start_epoch=0, "
            "global_step_start=0, and an empty selection state; resume into an "
            "existing checkpoint directory is outside this milestone"
        )
    train_batches, validation_batches = _preflight(
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
        selection_state,
        fit_config,
    )
    _validate_manager_preflight(checkpoint_manager, fit_config, checkpoint_config)
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
        fit_config,
        checkpoint_metadata,
    )
    if CHECKPOINT_SCHEMA_VERSION != "refsite_training_checkpoint_v1":
        raise ValueError("unsupported checkpoint schema for epoch-boundary capture")
    if CHECKPOINT_SCOPE != "epoch_boundary":
        raise ValueError("checkpointed fitting requires epoch-boundary schema")
    return train_batches, validation_batches, optimizer_config


def _fit_result(
    fit_config: FitConfig,
    records: list[FitEpochRecord],
    state: ModelSelectionState,
    optimizer: torch.optim.Optimizer,
    current_global_step: int,
) -> FitResult:
    records_tuple = tuple(records)
    last_epoch = records_tuple[-1].epoch_index
    return FitResult(
        config=fit_config,
        records=records_tuple,
        epochs_requested=fit_config.max_epochs - fit_config.start_epoch,
        epochs_completed=len(records_tuple),
        start_epoch=fit_config.start_epoch,
        next_epoch=last_epoch + 1,
        global_step_start=fit_config.global_step_start,
        global_step_end=current_global_step,
        stopped_early=state.stopped_early,
        stop_epoch=state.stop_epoch,
        stop_reason=state.stop_reason,
        best_metric=state.best_metric,
        best_epoch=state.best_epoch,
        best_global_step=state.best_global_step,
        final_selection_state=state,
        final_learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
        terminal_model_is_best=state.best_epoch == last_epoch,
    )


def _run_checkpointed_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_batches: Sequence[StructureBatch],
    validation_batches: Sequence[StructureBatch],
    template_contexts,
    loss_config: LossConfig,
    train_step_config: TrainStepConfig,
    validation_step_config: ValidationStepConfig,
    scheduler_config: SchedulerConfig,
    selection_config: ModelSelectionConfig,
    selection_state: ModelSelectionState,
    fit_config: FitConfig,
    checkpoint_manager: CheckpointManager,
    checkpoint_metadata: CheckpointMetadata,
    optimizer_config: OptimizerConfig,
    *,
    history_prefix: Sequence[FitEpochRecord] = (),
    checkpoint_fit_config: FitConfig | None = None,
    existing_best_path: str | None = None,
    epoch_metrics_provenance: "CommittedEpochProvenance | None" = None,
    epoch_metrics_observer: "EpochMetricsObserver | None" = None,
) -> CheckpointedFitResult:
    """Execute the shared epoch/checkpoint loop for fresh and resumed fits."""

    _validate_epoch_observer(epoch_metrics_provenance, epoch_metrics_observer)
    _validate_epoch_provenance_metadata(
        epoch_metrics_provenance, checkpoint_metadata
    )
    prefix = tuple(history_prefix)
    if any(not isinstance(record, FitEpochRecord) for record in prefix):
        raise TypeError("history_prefix entries must be FitEpochRecord objects")
    persisted_fit_config = (
        fit_config if checkpoint_fit_config is None else checkpoint_fit_config
    )
    if not isinstance(persisted_fit_config, FitConfig):
        raise TypeError("checkpoint_fit_config must be a FitConfig")

    records: list[FitEpochRecord] = []
    managed_results: list[ManagedCheckpointResult] = []
    state = selection_state
    current_global_step = fit_config.global_step_start

    for epoch_index in range(fit_config.start_epoch, fit_config.max_epochs):
        learning_rates_used = tuple(
            float(group["lr"]) for group in optimizer.param_groups
        )
        epoch_global_step_start = current_global_step
        try:
            training = run_training_epoch(
                model,
                optimizer,
                train_batches,
                template_contexts,
                loss_config,
                train_step_config,
                epoch_index=epoch_index,
                global_step_start=current_global_step,
            )
        except Exception as error:
            current_global_step = epoch_global_step_start + _partial_training_steps(error)
            raise FitExecutionError(
                phase="train",
                epoch_index=epoch_index,
                current_global_step=current_global_step,
                completed_epochs=len(records),
                training_update_completed=False,
                cause=error,
            ) from error
        current_global_step = training.global_step_end

        try:
            validation = run_validation_epoch(
                model,
                validation_batches,
                template_contexts,
                loss_config,
                validation_step_config,
                epoch_index=epoch_index,
                global_step=current_global_step,
            )
        except Exception as error:
            raise FitExecutionError(
                phase="validation",
                epoch_index=epoch_index,
                current_global_step=current_global_step,
                completed_epochs=len(records),
                training_update_completed=True,
                cause=error,
            ) from error

        try:
            state, decision = process_primary_validation(
                optimizer,
                scheduler,
                validation,
                scheduler_config,
                selection_config,
                state,
            )
        except Exception as error:
            raise FitExecutionError(
                phase="selection",
                epoch_index=epoch_index,
                current_global_step=current_global_step,
                completed_epochs=len(records),
                training_update_completed=True,
                cause=error,
            ) from error

        record = FitEpochRecord(
            epoch_index=epoch_index,
            training=training,
            validation=validation,
            decision=decision,
            selection_state_after_epoch=state,
            learning_rates_used_for_training=learning_rates_used,
            learning_rates_after_validation=decision.learning_rates_after,
        )
        records.append(record)
        full_history = prefix + tuple(records)
        progress = FitProgress(
            next_epoch=epoch_index + 1,
            global_step=current_global_step,
            completed_epochs=len(full_history),
            next_batch_index=0,
            last_completed_epoch=epoch_index,
            stopped_early=decision.should_stop,
            best_epoch=state.best_epoch,
            best_global_step=state.best_global_step,
        )
        try:
            checkpoint = capture_training_checkpoint(
                model,
                optimizer,
                scheduler,
                state,
                progress,
                train_batches,
                validation_batches,
                model_config=checkpoint_metadata.resolved_configuration["model"],
                loss_config=loss_config,
                optimizer_config=optimizer_config,
                train_step_config=train_step_config,
                validation_step_config=validation_step_config,
                scheduler_config=scheduler_config,
                model_selection_config=selection_config,
                fit_config=persisted_fit_config,
                species_vocabulary=checkpoint_metadata.species_vocabulary,
                fit_history=full_history,
                baseline_fit_metadata=checkpoint_metadata.baseline_fit_metadata,
                source_git_commit=checkpoint_metadata.source_git_commit,
            )
            if not _trees_equal(
                checkpoint.metadata.to_dict(), checkpoint_metadata.to_dict()
            ):
                raise ValueError(
                    "captured checkpoint metadata differs from preflight metadata"
                )
        except Exception as error:
            raise CheckpointedFitExecutionError(
                failure_stage="capture",
                epoch_record=record,
                completed_checkpoint_results=tuple(managed_results),
                cause=error,
            ) from error
        try:
            managed = checkpoint_manager.save_epoch(checkpoint, record)
        except Exception as error:
            raise CheckpointedFitExecutionError(
                failure_stage="manager",
                epoch_record=record,
                completed_checkpoint_results=tuple(managed_results),
                cause=error,
            ) from error
        managed_results.append(managed)
        if epoch_metrics_observer is not None:
            assert epoch_metrics_provenance is not None
            try:
                _observe_committed_epoch(
                    record,
                    managed,
                    selection_mode=selection_config.mode,
                    provenance=epoch_metrics_provenance,
                    observer=epoch_metrics_observer,
                )
            except Exception as error:
                raise CheckpointedFitExecutionError(
                    failure_stage="epoch_observer",
                    epoch_record=record,
                    completed_checkpoint_results=tuple(managed_results),
                    cause=error,
                ) from error
        if decision.should_stop:
            break

    fit_result = _fit_result(
        fit_config, records, state, optimizer, current_global_step
    )
    epoch_paths = tuple(
        result.epoch_path for result in managed_results if result.epoch_path is not None
    )
    best_paths = tuple(
        result.best_path for result in managed_results if result.best_path is not None
    )
    best_path = best_paths[-1] if best_paths else existing_best_path
    if best_path is None:
        raise ValueError("checkpointed fit did not establish a best checkpoint")
    resolved_best = Path(best_path)
    if resolved_best.is_symlink() or not resolved_best.is_file():
        raise ValueError("checkpointed fit best checkpoint is not a regular file")
    return CheckpointedFitResult(
        fit_result=fit_result,
        managed_checkpoint_results=tuple(managed_results),
        epoch_paths=epoch_paths,
        latest_path=managed_results[-1].latest_path,
        best_path=best_path,
        epochs_checkpointed=len(managed_results),
        terminal_checkpoint_epoch=managed_results[-1].epoch_index,
        terminal_checkpoint_global_step=managed_results[-1].global_step,
    )


def run_checkpointed_fit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_batches: Sequence[StructureBatch],
    validation_batches: Sequence[StructureBatch],
    template_contexts,
    loss_config: LossConfig,
    train_step_config: TrainStepConfig,
    validation_step_config: ValidationStepConfig,
    scheduler_config: SchedulerConfig,
    selection_config: ModelSelectionConfig,
    selection_state: ModelSelectionState,
    fit_config: FitConfig,
    checkpoint_manager: CheckpointManager,
    checkpoint_metadata: CheckpointMetadata,
    checkpoint_config: CheckpointedFitConfig = CheckpointedFitConfig(),
    *,
    epoch_metrics_provenance: "CommittedEpochProvenance | None" = None,
    epoch_metrics_observer: "EpochMetricsObserver | None" = None,
) -> CheckpointedFitResult:
    """Run a fresh deterministic fit and atomically checkpoint every epoch.

    The checkpoint is captured only after validation, scheduler/model selection,
    and the immutable epoch record are complete. A successful checkpoint write
    precedes both early-stop termination and the next training epoch.
    """

    _validate_epoch_observer(epoch_metrics_provenance, epoch_metrics_observer)
    train_batches, validation_batches, optimizer_config = _validate_checkpoint_preflight(
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
        selection_state,
        fit_config,
        checkpoint_manager,
        checkpoint_metadata,
        checkpoint_config,
    )
    _validate_epoch_provenance_metadata(
        epoch_metrics_provenance, checkpoint_metadata
    )
    return _run_checkpointed_epochs(
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
        selection_state,
        fit_config,
        checkpoint_manager,
        checkpoint_metadata,
        optimizer_config,
        epoch_metrics_provenance=epoch_metrics_provenance,
        epoch_metrics_observer=epoch_metrics_observer,
    )


__all__ = [
    "CheckpointedFitConfig",
    "CheckpointedFitExecutionError",
    "CheckpointedFitResult",
    "run_checkpointed_fit",
]
