"""One deterministic TRAIN_FIXED optimization step."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Real
from typing import Any, Mapping

import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.models import evaluate_structure_batch
from refsite_mlip.transport import TRAIN_FIXED

from .losses import LossConfig, LossTerm, compute_potential_loss


def _positive_optional_real(name: str, value: Real | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be None or a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be None or a finite positive real number")
    return result


@dataclass(frozen=True)
class TrainStepConfig:
    gradient_clip_norm: float | None = None
    fail_on_nonfinite: bool = True
    solver_path: str = TRAIN_FIXED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _positive_optional_real("gradient_clip_norm", self.gradient_clip_norm),
        )
        if not isinstance(self.fail_on_nonfinite, bool):
            raise TypeError("fail_on_nonfinite must be a bool")
        if not self.fail_on_nonfinite:
            raise ValueError("only fail_on_nonfinite=True is supported")
        if self.solver_path != TRAIN_FIXED:
            raise ValueError("training solver_path must be TRAIN_FIXED")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainStepConfig":
        if not isinstance(values, Mapping):
            raise TypeError("train-step config must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class TrainStepTermResult:
    numerator: float
    denominator: float
    mean: float
    valid_count: int


@dataclass(frozen=True)
class TrainStepResult:
    total_loss: float
    energy_loss: float
    force_loss: float
    stress_loss: float
    energy: TrainStepTermResult
    force: TrainStepTermResult
    stress: TrainStepTermResult
    pre_clip_grad_norm: float
    post_clip_grad_norm: float
    clipping_applied: bool
    number_of_parameters_with_grad: int
    need_forces: bool
    need_stress: bool
    sample_ids: tuple[str, ...]

    def __getitem__(self, key):
        return getattr(self, key)


def _has_force_supervision(batch: StructureBatch, config: LossConfig) -> bool:
    if config.force_weight == 0.0:
        return False
    valid = batch.force_mask & batch.force_present[batch.atom_batch, None]
    return bool(torch.any(valid))


def _has_stress_supervision(batch: StructureBatch, config: LossConfig) -> bool:
    if config.stress_weight == 0.0:
        return False
    valid = batch.stress_mask & batch.stress_present[:, None, None]
    return bool(torch.any(valid))


def _step_batch(batch: StructureBatch, *, need_forces: bool) -> StructureBatch:
    positions = batch.positions.detach().clone()
    if need_forces:
        positions.requires_grad_(True)
    return replace(batch, positions=positions)


def _active_terms(loss, config: LossConfig):
    return tuple(
        (name, weight, getattr(loss, name))
        for name, weight in (
            ("energy", config.energy_weight),
            ("force", config.force_weight),
            ("stress", config.stress_weight),
        )
        if weight > 0.0
    )


def _require_finite_loss(loss, config: LossConfig, sample_ids: tuple[str, ...]) -> None:
    for name, _, term in _active_terms(loss, config):
        for quantity in ("numerator", "mean"):
            value = getattr(term, quantity)
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"nonfinite {name} {quantity} for sample_id(s): "
                    + ",".join(sample_ids)
                )
    if not bool(torch.all(torch.isfinite(loss.total))):
        raise ValueError(
            "nonfinite total loss for sample_id(s): " + ",".join(sample_ids)
        )


def _named_gradients(model: torch.nn.Module):
    return tuple(
        (name, parameter, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    )


def _require_finite_gradients(named_gradients, sample_ids: tuple[str, ...]) -> None:
    for name, _, gradient in named_gradients:
        if gradient.is_sparse:
            values = gradient.coalesce().values()
        else:
            values = gradient
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError(
                f"nonfinite gradient for parameter {name}; sample_id(s): "
                + ",".join(sample_ids)
            )


def _global_gradient_norm(named_gradients) -> torch.Tensor:
    if not named_gradients:
        return torch.zeros((), dtype=torch.float64)
    norms = []
    for _, parameter, gradient in named_gradients:
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        norms.append(torch.linalg.vector_norm(values.detach()).to(torch.float64))
    return torch.linalg.vector_norm(torch.stack(norms))


def _term_result(term: LossTerm) -> TrainStepTermResult:
    return TrainStepTermResult(
        numerator=float(term.numerator.detach().cpu()),
        denominator=float(term.denominator.detach().cpu()),
        mean=float(term.mean.detach().cpu()),
        valid_count=int(term.valid_count.detach().cpu()),
    )


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: StructureBatch,
    template_contexts,
    loss_config: LossConfig,
    step_config: TrainStepConfig,
) -> TrainStepResult:
    """Run exactly one guarded AdamW update through the TRAIN_FIXED graph."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("optimizer must be torch.optim.AdamW")
    if not isinstance(batch, StructureBatch):
        raise TypeError("batch must be a StructureBatch")
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be a LossConfig")
    if not isinstance(step_config, TrainStepConfig):
        raise TypeError("step_config must be a TrainStepConfig")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    need_forces = _has_force_supervision(batch, loss_config)
    need_stress = _has_stress_supervision(batch, loss_config)
    step_batch = _step_batch(batch, need_forces=need_forces)
    prediction = evaluate_structure_batch(
        model,
        step_batch,
        template_contexts,
        solver_path=step_config.solver_path,
        compute_forces=need_forces,
        compute_stress=need_stress,
        create_graph=need_forces or need_stress,
        return_aux=False,
    )
    loss = compute_potential_loss(prediction, step_batch, loss_config)
    active = _active_terms(loss, loss_config)
    supervised = tuple(name for name, _, term in active if bool(term.valid_count))
    if not supervised:
        requested = ",".join(name for name, _, _ in active) or "none"
        raise ValueError(
            f"no weighted valid supervision (active terms: {requested}); "
            "sample_id(s): " + ",".join(batch.sample_ids)
        )
    _require_finite_loss(loss, loss_config, batch.sample_ids)

    loss.total.backward()
    named_gradients = _named_gradients(model)
    _require_finite_gradients(named_gradients, batch.sample_ids)
    pre_norm_tensor = _global_gradient_norm(named_gradients)
    if not bool(torch.isfinite(pre_norm_tensor)):
        raise ValueError(
            "nonfinite global gradient norm for sample_id(s): "
            + ",".join(batch.sample_ids)
        )
    pre_norm = float(pre_norm_tensor.cpu())

    clipping_applied = False
    if step_config.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter, _ in named_gradients],
            step_config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        clipping_applied = pre_norm > step_config.gradient_clip_norm
    clipped_gradients = _named_gradients(model)
    _require_finite_gradients(clipped_gradients, batch.sample_ids)
    post_norm_tensor = _global_gradient_norm(clipped_gradients)
    if not bool(torch.isfinite(post_norm_tensor)):
        raise ValueError(
            "nonfinite clipped gradient norm for sample_id(s): "
            + ",".join(batch.sample_ids)
        )
    post_norm = float(post_norm_tensor.cpu())
    optimizer.step()

    energy = _term_result(loss.energy)
    force = _term_result(loss.force)
    stress = _term_result(loss.stress)
    return TrainStepResult(
        total_loss=float(loss.total.detach().cpu()),
        energy_loss=energy.mean,
        force_loss=force.mean,
        stress_loss=stress.mean,
        energy=energy,
        force=force,
        stress=stress,
        pre_clip_grad_norm=pre_norm,
        post_clip_grad_norm=post_norm,
        clipping_applied=clipping_applied,
        number_of_parameters_with_grad=len(named_gradients),
        need_forces=need_forces,
        need_stress=need_stress,
        sample_ids=batch.sample_ids,
    )
