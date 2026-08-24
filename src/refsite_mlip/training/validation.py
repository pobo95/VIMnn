"""Deterministic single-batch TRAIN_FIXED validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.models import evaluate_structure_batch
from refsite_mlip.transport import TRAIN_FIXED

from .losses import LossConfig, LossTerm, compute_potential_loss
from .step import (
    _active_terms,
    _has_force_supervision,
    _has_stress_supervision,
    _step_batch,
)


@dataclass(frozen=True)
class ValidationStepConfig:
    solver_path: str = TRAIN_FIXED

    def __post_init__(self) -> None:
        if self.solver_path != TRAIN_FIXED:
            raise ValueError("validation solver_path must be TRAIN_FIXED")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ValidationStepConfig":
        if not isinstance(values, Mapping):
            raise TypeError("validation config must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class ValidationTermResult:
    numerator: float
    denominator: float
    mean: float
    valid_count: int


@dataclass(frozen=True)
class ValidationStepResult:
    total_loss: float
    energy_loss: float
    force_loss: float
    stress_loss: float
    energy: ValidationTermResult
    force: ValidationTermResult
    stress: ValidationTermResult
    has_supervision: bool
    need_forces: bool
    need_stress: bool
    sample_ids: tuple[str, ...]
    solver_path: str

    def __getitem__(self, key):
        return getattr(self, key)


def _term_result(term: LossTerm) -> ValidationTermResult:
    return ValidationTermResult(
        numerator=float(term.numerator.detach().cpu()),
        denominator=float(term.denominator.detach().cpu()),
        mean=float(term.mean.detach().cpu()),
        valid_count=int(term.valid_count.detach().cpu()),
    )


def _require_prediction_finite(
    prediction,
    *,
    need_forces: bool,
    need_stress: bool,
    sample_ids: tuple[str, ...],
) -> None:
    identifiers = ",".join(sample_ids)
    for name, required in (
        ("energy", True),
        ("forces", need_forces),
        ("stress", need_stress),
    ):
        if not required:
            continue
        tensor = getattr(prediction, name, None)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"missing requested {name} prediction for sample_id(s): {identifiers}"
            )
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(
                f"nonfinite {name} prediction for sample_id(s): {identifiers}"
            )


def _require_loss_finite(loss, config: LossConfig, sample_ids: tuple[str, ...]) -> None:
    identifiers = ",".join(sample_ids)
    for name, _, term in _active_terms(loss, config):
        for quantity in ("numerator", "mean"):
            value = getattr(term, quantity)
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"nonfinite {name} {quantity} for sample_id(s): {identifiers}"
                )
    if not bool(torch.all(torch.isfinite(loss.total))):
        raise ValueError(f"nonfinite total loss for sample_id(s): {identifiers}")


def _capture_rng_state():
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_state = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return cpu_state, cuda_state


def _restore_rng_state(cpu_state, cuda_state) -> None:
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def validation_step(
    model: torch.nn.Module,
    batch: StructureBatch,
    template_contexts,
    loss_config: LossConfig,
    config: ValidationStepConfig,
) -> ValidationStepResult:
    """Evaluate one batch without mutating optimization or autograd state."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(batch, StructureBatch):
        raise TypeError("batch must be a StructureBatch")
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be a LossConfig")
    if not isinstance(config, ValidationStepConfig):
        raise TypeError("config must be a ValidationStepConfig")

    previous_training = model.training
    cpu_rng_state, cuda_rng_state = _capture_rng_state()
    model.eval()
    try:
        need_forces = _has_force_supervision(batch, loss_config)
        need_stress = _has_stress_supervision(batch, loss_config)
        derivatives_required = need_forces or need_stress
        if derivatives_required and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "force/stress validation requires geometry derivatives and cannot "
                "run inside torch.inference_mode(); use torch.no_grad() or ordinary "
                "execution instead"
            )
        local_batch = _step_batch(batch, need_forces=need_forces)
        grad_context = torch.enable_grad() if derivatives_required else torch.no_grad()
        with grad_context:
            prediction = evaluate_structure_batch(
                model,
                local_batch,
                template_contexts,
                solver_path=config.solver_path,
                compute_forces=need_forces,
                compute_stress=need_stress,
                create_graph=False,
                return_aux=False,
            )
            _require_prediction_finite(
                prediction,
                need_forces=need_forces,
                need_stress=need_stress,
                sample_ids=batch.sample_ids,
            )
            loss = compute_potential_loss(prediction, local_batch, loss_config)
            _require_loss_finite(loss, loss_config, batch.sample_ids)

        energy = _term_result(loss.energy)
        force = _term_result(loss.force)
        stress = _term_result(loss.stress)
        has_supervision = any(
            term.valid_count > 0 for term in (energy, force, stress)
        )
        return ValidationStepResult(
            total_loss=float(loss.total.detach().cpu()),
            energy_loss=energy.mean,
            force_loss=force.mean,
            stress_loss=stress.mean,
            energy=energy,
            force=force,
            stress=stress,
            has_supervision=has_supervision,
            need_forces=need_forces,
            need_stress=need_stress,
            sample_ids=batch.sample_ids,
            solver_path=config.solver_path,
        )
    finally:
        try:
            _restore_rng_state(cpu_rng_state, cuda_rng_state)
        finally:
            model.train(previous_training)
