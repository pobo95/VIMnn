"""Deterministic single-epoch runners over prepared ragged batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Integral
from typing import Any, Literal

import torch

from refsite_mlip.data import StructureBatch

from .losses import LossConfig
from .step import TrainStepConfig, train_step
from .validation import ValidationStepConfig, validation_step


EpochPhase = Literal["train", "validation"]
TRAIN_METRIC_SEMANTICS = "pre_update_batch_observations"
VALIDATION_METRIC_SEMANTICS = "fixed_model_validation"


@dataclass(frozen=True)
class EpochTermMetrics:
    numerator: float
    denominator: float
    mean: float
    valid_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EpochTermMetrics":
        if not isinstance(values, Mapping):
            raise TypeError("epoch term metrics must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class EpochMetrics:
    energy: EpochTermMetrics
    force: EpochTermMetrics
    stress: EpochTermMetrics
    total_loss: float
    has_supervision: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EpochMetrics":
        if not isinstance(values, Mapping):
            raise TypeError("epoch metrics must be reconstructed from a mapping")
        data = dict(values)
        for name in ("energy", "force", "stress"):
            data[name] = EpochTermMetrics.from_dict(data[name])
        return cls(**data)


@dataclass(frozen=True)
class EpochResult(EpochMetrics):
    phase: EpochPhase
    epoch_index: int
    global_step_start: int
    global_step_end: int
    number_of_batches: int
    number_of_supervised_batches: int
    number_of_structures: int
    number_of_atoms: int
    successful_optimizer_steps: int
    ordered_batch_sample_ids: tuple[tuple[str, ...], ...]
    metric_semantics: str

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EpochResult":
        if not isinstance(values, Mapping):
            raise TypeError("epoch result must be reconstructed from a mapping")
        data = dict(values)
        for name in ("energy", "force", "stress"):
            data[name] = EpochTermMetrics.from_dict(data[name])
        data["ordered_batch_sample_ids"] = tuple(
            tuple(sample_ids) for sample_ids in data["ordered_batch_sample_ids"]
        )
        return cls(**data)


def _nonnegative_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _validated_batches(batches) -> Sequence[StructureBatch]:
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError("batches must be a deterministic Sequence[StructureBatch]")
    if len(batches) == 0:
        raise ValueError("epoch batch sequence must not be empty")
    return batches


def _term_metrics(results, name: str) -> EpochTermMetrics:
    terms = tuple(getattr(result, name) for result in results)
    numerator = math.fsum(float(term.numerator) for term in terms)
    denominator = math.fsum(float(term.denominator) for term in terms)
    valid_count = sum(int(term.valid_count) for term in terms)
    mean = numerator / denominator if denominator > 0.0 else 0.0
    return EpochTermMetrics(numerator, denominator, mean, valid_count)


def _is_supervised(result) -> bool:
    explicit = getattr(result, "has_supervision", None)
    if explicit is not None:
        return bool(explicit)
    return any(
        int(getattr(result, name).valid_count) > 0
        for name in ("energy", "force", "stress")
    )


def _epoch_result(
    *,
    phase: EpochPhase,
    epoch_index: int,
    global_step_start: int,
    global_step_end: int,
    batches: Sequence[StructureBatch],
    results,
    loss_config: LossConfig,
    successful_optimizer_steps: int,
) -> EpochResult:
    energy = _term_metrics(results, "energy")
    force = _term_metrics(results, "force")
    stress = _term_metrics(results, "stress")
    total_loss = math.fsum(
        (
            loss_config.energy_weight * energy.mean,
            loss_config.force_weight * force.mean,
            loss_config.stress_weight * stress.mean,
        )
    )
    supervised_batches = sum(_is_supervised(result) for result in results)
    return EpochResult(
        energy=energy,
        force=force,
        stress=stress,
        total_loss=total_loss,
        has_supervision=supervised_batches > 0,
        phase=phase,
        epoch_index=epoch_index,
        global_step_start=global_step_start,
        global_step_end=global_step_end,
        number_of_batches=len(batches),
        number_of_supervised_batches=supervised_batches,
        number_of_structures=sum(batch.num_structures for batch in batches),
        number_of_atoms=sum(batch.num_atoms for batch in batches),
        successful_optimizer_steps=successful_optimizer_steps,
        ordered_batch_sample_ids=tuple(batch.sample_ids for batch in batches),
        metric_semantics=(
            TRAIN_METRIC_SEMANTICS
            if phase == "train"
            else VALIDATION_METRIC_SEMANTICS
        ),
    )


def _batch_failure(
    *,
    phase: EpochPhase,
    epoch_index: int,
    batch_index: int,
    batch,
    successful_optimizer_steps: int,
    error: Exception,
) -> RuntimeError:
    sample_ids = getattr(batch, "sample_ids", ("<unavailable>",))
    return RuntimeError(
        f"{phase} epoch batch failed: epoch_index={epoch_index}, "
        f"batch_index={batch_index}, sample_ids={tuple(sample_ids)}, "
        f"successful_optimizer_steps={successful_optimizer_steps}; "
        "completed updates are retained and are not rolled back; "
        f"cause={type(error).__name__}: {error}"
    )


def run_training_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[StructureBatch],
    template_contexts,
    loss_config: LossConfig,
    train_step_config: TrainStepConfig,
    *,
    epoch_index: int,
    global_step_start: int,
) -> EpochResult:
    """Aggregate pre-update batch observations while the model is updated.

    These metrics are not the loss of the final model evaluated over the full
    training set.  Use ``run_validation_epoch`` explicitly for that quantity.
    """

    batches = _validated_batches(batches)
    epoch_index = _nonnegative_integer("epoch_index", epoch_index)
    global_step_start = _nonnegative_integer("global_step_start", global_step_start)
    results = []
    successful_steps = 0
    for batch_index in range(len(batches)):
        batch = batches[batch_index]
        try:
            if not isinstance(batch, StructureBatch):
                raise TypeError("epoch entries must be StructureBatch objects")
            result = train_step(
                model,
                optimizer,
                batch,
                template_contexts,
                loss_config,
                train_step_config,
            )
        except Exception as error:
            raise _batch_failure(
                phase="train",
                epoch_index=epoch_index,
                batch_index=batch_index,
                batch=batch,
                successful_optimizer_steps=successful_steps,
                error=error,
            ) from error
        results.append(result)
        successful_steps += 1
    return _epoch_result(
        phase="train",
        epoch_index=epoch_index,
        global_step_start=global_step_start,
        global_step_end=global_step_start + successful_steps,
        batches=batches,
        results=tuple(results),
        loss_config=loss_config,
        successful_optimizer_steps=successful_steps,
    )


def run_validation_epoch(
    model: torch.nn.Module,
    batches: Sequence[StructureBatch],
    template_contexts,
    loss_config: LossConfig,
    validation_step_config: ValidationStepConfig,
    *,
    epoch_index: int,
    global_step: int,
) -> EpochResult:
    """Evaluate prepared batches without changing optimizer or global step."""

    batches = _validated_batches(batches)
    epoch_index = _nonnegative_integer("epoch_index", epoch_index)
    global_step = _nonnegative_integer("global_step", global_step)
    results = []
    for batch_index in range(len(batches)):
        batch = batches[batch_index]
        try:
            if not isinstance(batch, StructureBatch):
                raise TypeError("epoch entries must be StructureBatch objects")
            result = validation_step(
                model,
                batch,
                template_contexts,
                loss_config,
                validation_step_config,
            )
        except Exception as error:
            raise _batch_failure(
                phase="validation",
                epoch_index=epoch_index,
                batch_index=batch_index,
                batch=batch,
                successful_optimizer_steps=0,
                error=error,
            ) from error
        results.append(result)
    return _epoch_result(
        phase="validation",
        epoch_index=epoch_index,
        global_step_start=global_step,
        global_step_end=global_step,
        batches=batches,
        results=tuple(results),
        loss_config=loss_config,
        successful_optimizer_steps=0,
    )
