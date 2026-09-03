"""Deterministic multi-epoch controller over prepared StructureBatch sequences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
import re
from typing import Any, Literal

import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.models.batch_executor import _validated_context

from .epoch import EpochResult, run_training_epoch, run_validation_epoch
from .losses import LossConfig
from .optimizer import validate_optimizer_binding
from .scheduler import SchedulerConfig, _validate_scheduler_binding
from .selection import (
    ModelSelectionConfig,
    ModelSelectionState,
    ValidationDecision,
    process_primary_validation,
)
from .step import TrainStepConfig
from .validation import ValidationStepConfig


FitFailurePhase = Literal["train", "validation", "selection"]


def _integer(name: str, value, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        qualifier = "positive" if positive else "nonnegative"
        raise TypeError(f"{name} must be a {qualifier} integer")
    result = int(value)
    if (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return result


@dataclass(frozen=True)
class FitConfig:
    max_epochs: int
    start_epoch: int = 0
    global_step_start: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_epochs", _integer("max_epochs", self.max_epochs, positive=True)
        )
        object.__setattr__(
            self,
            "start_epoch",
            _integer("start_epoch", self.start_epoch, positive=False),
        )
        object.__setattr__(
            self,
            "global_step_start",
            _integer("global_step_start", self.global_step_start, positive=False),
        )
        if self.start_epoch >= self.max_epochs:
            raise ValueError("start_epoch must be smaller than max_epochs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FitConfig":
        if not isinstance(values, Mapping):
            raise TypeError("fit config must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class FitEpochRecord:
    epoch_index: int
    training: EpochResult
    validation: EpochResult
    decision: ValidationDecision
    selection_state_after_epoch: ModelSelectionState
    learning_rates_used_for_training: tuple[float, ...]
    learning_rates_after_validation: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch_index": self.epoch_index,
            "training": self.training.to_dict(),
            "validation": self.validation.to_dict(),
            "decision": self.decision.to_dict(),
            "selection_state_after_epoch": self.selection_state_after_epoch.to_dict(),
            "learning_rates_used_for_training": list(
                self.learning_rates_used_for_training
            ),
            "learning_rates_after_validation": list(
                self.learning_rates_after_validation
            ),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FitEpochRecord":
        if not isinstance(values, Mapping):
            raise TypeError("fit epoch record must be reconstructed from a mapping")
        data = dict(values)
        data["training"] = EpochResult.from_dict(data["training"])
        data["validation"] = EpochResult.from_dict(data["validation"])
        data["decision"] = ValidationDecision.from_dict(data["decision"])
        data["selection_state_after_epoch"] = ModelSelectionState.from_dict(
            data["selection_state_after_epoch"]
        )
        data["learning_rates_used_for_training"] = tuple(
            data["learning_rates_used_for_training"]
        )
        data["learning_rates_after_validation"] = tuple(
            data["learning_rates_after_validation"]
        )
        return cls(**data)


@dataclass(frozen=True)
class FitResult:
    config: FitConfig
    records: tuple[FitEpochRecord, ...]
    epochs_requested: int
    epochs_completed: int
    start_epoch: int
    next_epoch: int
    global_step_start: int
    global_step_end: int
    stopped_early: bool
    stop_epoch: int | None
    stop_reason: str | None
    best_metric: float
    best_epoch: int
    best_global_step: int
    final_selection_state: ModelSelectionState
    final_learning_rates: tuple[float, ...]
    terminal_model_is_best: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "records": [record.to_dict() for record in self.records],
            "epochs_requested": self.epochs_requested,
            "epochs_completed": self.epochs_completed,
            "start_epoch": self.start_epoch,
            "next_epoch": self.next_epoch,
            "global_step_start": self.global_step_start,
            "global_step_end": self.global_step_end,
            "stopped_early": self.stopped_early,
            "stop_epoch": self.stop_epoch,
            "stop_reason": self.stop_reason,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_global_step": self.best_global_step,
            "final_selection_state": self.final_selection_state.to_dict(),
            "final_learning_rates": list(self.final_learning_rates),
            "terminal_model_is_best": self.terminal_model_is_best,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FitResult":
        if not isinstance(values, Mapping):
            raise TypeError("fit result must be reconstructed from a mapping")
        data = dict(values)
        data["config"] = FitConfig.from_dict(data["config"])
        data["records"] = tuple(
            FitEpochRecord.from_dict(record) for record in data["records"]
        )
        data["final_selection_state"] = ModelSelectionState.from_dict(
            data["final_selection_state"]
        )
        data["final_learning_rates"] = tuple(data["final_learning_rates"])
        return cls(**data)


class FitExecutionError(RuntimeError):
    """Failure with explicit retained partial-progress metadata."""

    def __init__(
        self,
        *,
        phase: FitFailurePhase,
        epoch_index: int,
        current_global_step: int,
        completed_epochs: int,
        training_update_completed: bool,
        cause: Exception,
    ) -> None:
        self.phase = phase
        self.epoch_index = epoch_index
        self.current_global_step = current_global_step
        self.completed_epochs = completed_epochs
        self.training_update_completed = training_update_completed
        self.original_exception_type = type(cause).__name__
        self.original_exception_message = str(cause)
        self.rollback_performed = False
        super().__init__(
            f"fit execution failed: phase={phase}, epoch_index={epoch_index}, "
            f"current_global_step={current_global_step}, "
            f"completed_epochs={completed_epochs}, "
            f"training_update_completed={training_update_completed}; "
            f"cause={self.original_exception_type}: {self.original_exception_message}; "
            "completed parameter/optimizer updates are retained and are not rolled back"
        )


def _deterministic_batches(name: str, batches) -> Sequence[StructureBatch]:
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError(f"{name} must be a deterministic Sequence[StructureBatch]")
    if len(batches) == 0:
        raise ValueError(f"{name} must not be empty")
    for index, batch in enumerate(batches):
        if not isinstance(batch, StructureBatch):
            raise TypeError(f"{name}[{index}] must be a StructureBatch")
        batch.validate()
    return batches


def _validate_batch_contexts(batches, template_contexts) -> None:
    if not isinstance(template_contexts, Mapping):
        raise TypeError("template_contexts must be a mapping")
    for batch in batches:
        for group in batch.template_groups:
            _validated_context(
                group.template_id,
                group.template_fingerprint,
                template_contexts,
            )


def _has_valid_label(batch: StructureBatch, term: str) -> bool:
    if term == "energy":
        return bool(torch.any(batch.energy_mask))
    if term == "force":
        valid = batch.force_mask & batch.force_present[batch.atom_batch, None]
        return bool(torch.any(valid))
    if term == "stress":
        valid = batch.stress_mask & batch.stress_present[:, None, None]
        diagonal = torch.diagonal(valid, dim1=-2, dim2=-1)
        off_diagonal = torch.stack(
            (valid[:, 0, 1], valid[:, 0, 2], valid[:, 1, 2]), dim=1
        )
        return bool(torch.any(diagonal)) or bool(torch.any(off_diagonal))
    raise ValueError(f"unsupported validation monitor term: {term}")


def _validate_monitor_supervision(
    validation_batches,
    loss_config: LossConfig,
    selection_config: ModelSelectionConfig,
) -> None:
    weights = {
        "energy": loss_config.energy_weight,
        "force": loss_config.force_weight,
        "stress": loss_config.stress_weight,
    }
    if selection_config.monitor == "total_loss":
        active_terms = tuple(name for name, weight in weights.items() if weight > 0.0)
        if not active_terms:
            raise ValueError("total_loss monitor requires a positive loss weight")
        if not any(
            _has_valid_label(batch, term)
            for batch in validation_batches
            for term in active_terms
        ):
            raise ValueError(
                "total_loss monitor has no positive-weight validation labels"
            )
        return
    monitor = selection_config.monitor
    if weights[monitor] <= 0.0:
        raise ValueError(f"{monitor} monitor requires {monitor}_weight > 0")
    if not any(_has_valid_label(batch, monitor) for batch in validation_batches):
        raise ValueError(f"{monitor} monitor has no valid validation labels")


def _preflight(
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
) -> tuple[Sequence[StructureBatch], Sequence[StructureBatch]]:
    validate_optimizer_binding(model, optimizer)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    for value, cls, name in (
        (loss_config, LossConfig, "loss_config"),
        (train_step_config, TrainStepConfig, "train_step_config"),
        (validation_step_config, ValidationStepConfig, "validation_step_config"),
        (scheduler_config, SchedulerConfig, "scheduler_config"),
        (selection_config, ModelSelectionConfig, "selection_config"),
        (selection_state, ModelSelectionState, "selection_state"),
        (fit_config, FitConfig, "fit_config"),
    ):
        if not isinstance(value, cls):
            raise TypeError(f"{name} must be a {cls.__name__}")
    train_batches = _deterministic_batches("train_batches", train_batches)
    validation_batches = _deterministic_batches(
        "validation_batches", validation_batches
    )
    _validate_scheduler_binding(optimizer, scheduler, scheduler_config)
    if scheduler_config.monitor != selection_config.monitor:
        raise ValueError("scheduler and model selection must monitor the same metric")
    if scheduler_config.mode != selection_config.mode:
        raise ValueError("scheduler and model selection must use the same mode")
    if selection_state.stopped_early:
        raise ValueError("initial selection state is already stopped")
    if selection_state.validation_events > 0:
        if fit_config.start_epoch <= selection_state.last_validation_epoch:
            raise ValueError(
                "start_epoch must be after the last processed validation epoch"
            )
        if fit_config.global_step_start != selection_state.last_validation_global_step:
            raise ValueError(
                "global_step_start must equal the last validation global step"
            )
    _validate_batch_contexts(
        tuple(train_batches) + tuple(validation_batches), template_contexts
    )
    _validate_monitor_supervision(
        validation_batches, loss_config, selection_config
    )
    return train_batches, validation_batches


def _partial_training_steps(error: Exception) -> int:
    match = re.search(r"successful_optimizer_steps=(\d+)", str(error))
    return int(match.group(1)) if match is not None else 0


def run_fit(
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
) -> FitResult:
    """Run ordered train/validation/selection events until max epoch or stop."""

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
    records = []
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
        if decision.should_stop:
            break

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
        final_learning_rates=tuple(
            float(group["lr"]) for group in optimizer.param_groups
        ),
        terminal_model_is_best=state.best_epoch == last_epoch,
    )
