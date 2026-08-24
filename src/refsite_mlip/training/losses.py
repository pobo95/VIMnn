"""Masked energy, force, and symmetric-stress objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Callable, Literal, Mapping

import torch

from refsite_mlip.data import StructureBatch


EnergyNormalization = Literal["per_structure", "per_atom"]


def _finite_real(name: str, value: Real, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True)
class LossConfig:
    energy_weight: float = 1.0
    force_weight: float = 0.0
    stress_weight: float = 0.0
    energy_scale: float = 1.0
    force_scale: float = 1.0
    stress_scale: float = 1.0
    energy_normalization: EnergyNormalization = "per_structure"
    stress_symmetry_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        for name in ("energy_weight", "force_weight", "stress_weight"):
            object.__setattr__(
                self, name, _finite_real(name, getattr(self, name), positive=False)
            )
        for name in ("energy_scale", "force_scale", "stress_scale"):
            object.__setattr__(
                self, name, _finite_real(name, getattr(self, name), positive=True)
            )
        object.__setattr__(
            self,
            "stress_symmetry_tolerance",
            _finite_real(
                "stress_symmetry_tolerance",
                self.stress_symmetry_tolerance,
                positive=False,
            ),
        )
        if self.energy_normalization not in ("per_structure", "per_atom"):
            raise ValueError(
                "energy_normalization must be 'per_structure' or 'per_atom'"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "LossConfig":
        if not isinstance(values, Mapping):
            raise TypeError("loss config must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class LossTerm:
    numerator: torch.Tensor
    denominator: torch.Tensor
    mean: torch.Tensor
    valid_count: torch.Tensor


@dataclass(frozen=True)
class PotentialLossOutput:
    total: torch.Tensor
    energy: LossTerm
    force: LossTerm
    stress: LossTerm

    def __getitem__(self, key):
        return getattr(self, key)


def _zero_term(anchor: torch.Tensor) -> LossTerm:
    zero = anchor * 0.0
    denominator = anchor.new_zeros(())
    valid_count = torch.zeros((), dtype=torch.long, device=anchor.device)
    return LossTerm(zero, denominator, zero, valid_count)


def _term(numerator: torch.Tensor, valid_count: torch.Tensor) -> LossTerm:
    denominator = valid_count.to(dtype=numerator.dtype)
    mean = numerator / denominator
    return LossTerm(numerator, denominator, mean, valid_count)


def _check_prediction_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    batch: StructureBatch,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"prediction.{name} must be a tensor")
    if tensor.shape != shape:
        raise ValueError(
            f"prediction.{name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != batch.dtype:
        raise ValueError(f"prediction.{name} dtype does not match targets")
    if tensor.device != batch.device:
        raise ValueError(f"prediction.{name} device does not match targets")


def _structure_ids(batch: StructureBatch, mask: torch.Tensor) -> str:
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    return ",".join(batch.sample_ids[index] for index in indices)


def _atom_structure_mask(batch: StructureBatch, component_mask: torch.Tensor) -> torch.Tensor:
    structures = torch.zeros(
        batch.num_structures, dtype=torch.bool, device=batch.device
    )
    if bool(torch.any(component_mask)):
        atom_mask = torch.any(component_mask, dim=1)
        structures[batch.atom_batch[atom_mask]] = True
    return structures


def _require_finite(
    values: torch.Tensor,
    *,
    term_name: str,
    sample_ids: str | Callable[[], str],
    quantity: str,
) -> None:
    if not bool(torch.all(torch.isfinite(values))):
        resolved_ids = sample_ids() if callable(sample_ids) else sample_ids
        raise ValueError(
            f"nonfinite {term_name} {quantity} for sample_id(s): {resolved_ids}"
        )


def _energy_term(prediction, batch: StructureBatch, config: LossConfig, anchor):
    if config.energy_weight == 0.0:
        return _zero_term(anchor)
    valid = batch.energy_mask
    valid_count = torch.count_nonzero(valid)
    if not bool(valid_count):
        return _zero_term(anchor)
    predicted = prediction.energy[valid]
    target = batch.energy[valid]
    sample_ids = lambda: _structure_ids(batch, valid)
    _require_finite(predicted, term_name="energy", sample_ids=sample_ids, quantity="prediction")
    _require_finite(target, term_name="energy", sample_ids=sample_ids, quantity="target")
    if config.energy_normalization == "per_atom":
        atom_counts = (batch.atom_ptr[1:] - batch.atom_ptr[:-1]).to(
            dtype=predicted.dtype
        )[valid]
        residual = (predicted - target) / (atom_counts * config.energy_scale)
    else:
        residual = (predicted - target) / config.energy_scale
    numerator = torch.sum(residual.square())
    _require_finite(numerator, term_name="energy", sample_ids=sample_ids, quantity="loss")
    return _term(numerator, valid_count)


def _force_term(prediction, batch: StructureBatch, config: LossConfig, anchor):
    if config.force_weight == 0.0:
        return _zero_term(anchor)
    valid = batch.force_mask & batch.force_present[batch.atom_batch, None]
    valid_count = torch.count_nonzero(valid)
    if not bool(valid_count):
        return _zero_term(anchor)
    predicted_forces = getattr(prediction, "forces", None)
    if predicted_forces is None:
        sample_ids = _structure_ids(batch, _atom_structure_mask(batch, valid))
        raise ValueError(
            f"force prediction is required for valid sample_id(s): {sample_ids}"
        )
    _check_prediction_tensor(
        predicted_forces,
        name="forces",
        shape=(batch.num_atoms, 3),
        batch=batch,
    )
    predicted = predicted_forces[valid]
    target = batch.forces[valid]
    sample_ids = lambda: _structure_ids(
        batch, _atom_structure_mask(batch, valid)
    )
    _require_finite(predicted, term_name="force", sample_ids=sample_ids, quantity="prediction")
    _require_finite(target, term_name="force", sample_ids=sample_ids, quantity="target")
    numerator = torch.sum(((predicted - target) / config.force_scale).square())
    _require_finite(numerator, term_name="force", sample_ids=sample_ids, quantity="loss")
    return _term(numerator, valid_count)


def _stress_term(prediction, batch: StructureBatch, config: LossConfig, anchor):
    if config.stress_weight == 0.0:
        return _zero_term(anchor)
    valid_mask = batch.stress_mask & batch.stress_present[:, None, None]
    mask_mismatch = torch.any(valid_mask != valid_mask.transpose(-1, -2), dim=(-2, -1))
    if bool(torch.any(mask_mismatch)):
        raise ValueError(
            "asymmetric stress mask for sample_id(s): "
            + _structure_ids(batch, mask_mismatch)
        )

    diagonal_indices = torch.arange(3, device=batch.device)
    diagonal_valid = valid_mask[:, diagonal_indices, diagonal_indices]
    pairs = ((0, 1), (0, 2), (1, 2))
    off_diagonal_valid = torch.stack(
        [valid_mask[:, first, second] for first, second in pairs], dim=1
    )
    valid_count = torch.count_nonzero(diagonal_valid) + torch.count_nonzero(
        off_diagonal_valid
    )
    if not bool(valid_count):
        return _zero_term(anchor)

    target = batch.stress
    tolerance = config.stress_symmetry_tolerance
    target_asymmetric = torch.zeros(
        batch.num_structures, dtype=torch.bool, device=batch.device
    )
    for pair_index, (first, second) in enumerate(pairs):
        target_asymmetric |= off_diagonal_valid[:, pair_index] & (
            torch.abs(target[:, first, second] - target[:, second, first])
            > tolerance
        )
    if bool(torch.any(target_asymmetric)):
        raise ValueError(
            "asymmetric stress target for sample_id(s): "
            + _structure_ids(batch, target_asymmetric)
        )

    predicted_stress = getattr(prediction, "stress", None)
    if predicted_stress is None:
        structures = torch.any(diagonal_valid, dim=1) | torch.any(
            off_diagonal_valid, dim=1
        )
        raise ValueError(
            "stress prediction is required for valid sample_id(s): "
            + _structure_ids(batch, structures)
        )
    _check_prediction_tensor(
        predicted_stress,
        name="stress",
        shape=(batch.num_structures, 3, 3),
        batch=batch,
    )

    prediction_asymmetric = torch.zeros_like(target_asymmetric)
    for pair_index, (first, second) in enumerate(pairs):
        prediction_asymmetric |= off_diagonal_valid[:, pair_index] & (
            torch.abs(
                predicted_stress[:, first, second]
                - predicted_stress[:, second, first]
            )
            > tolerance
        )
    if bool(torch.any(prediction_asymmetric)):
        raise ValueError(
            "asymmetric stress prediction for sample_id(s): "
            + _structure_ids(batch, prediction_asymmetric)
        )

    structure_valid = torch.any(diagonal_valid, dim=1) | torch.any(
        off_diagonal_valid, dim=1
    )
    sample_ids = lambda: _structure_ids(batch, structure_valid)
    diagonal_prediction = predicted_stress[
        :, diagonal_indices, diagonal_indices
    ][diagonal_valid]
    diagonal_target = target[:, diagonal_indices, diagonal_indices][diagonal_valid]
    off_prediction = torch.stack(
        [predicted_stress[:, first, second] for first, second in pairs], dim=1
    )[off_diagonal_valid]
    off_target = torch.stack(
        [target[:, first, second] for first, second in pairs], dim=1
    )[off_diagonal_valid]
    _require_finite(
        diagonal_prediction,
        term_name="stress",
        sample_ids=sample_ids,
        quantity="prediction",
    )
    _require_finite(
        diagonal_target,
        term_name="stress",
        sample_ids=sample_ids,
        quantity="target",
    )
    _require_finite(
        off_prediction,
        term_name="stress",
        sample_ids=sample_ids,
        quantity="prediction",
    )
    _require_finite(
        off_target,
        term_name="stress",
        sample_ids=sample_ids,
        quantity="target",
    )
    diagonal_residual = (diagonal_prediction - diagonal_target) / config.stress_scale
    off_residual = (off_prediction - off_target) / config.stress_scale
    numerator = torch.sum(diagonal_residual.square()) + 2.0 * torch.sum(
        off_residual.square()
    )
    _require_finite(numerator, term_name="stress", sample_ids=sample_ids, quantity="loss")
    return _term(numerator, valid_count)


def compute_potential_loss(
    prediction,
    batch: StructureBatch,
    config: LossConfig,
) -> PotentialLossOutput:
    """Compute mask-first losses without unit conversion or implicit casting."""

    if not isinstance(batch, StructureBatch):
        raise TypeError("batch must be a StructureBatch")
    if not isinstance(config, LossConfig):
        raise TypeError("config must be a LossConfig")
    predicted_energy = getattr(prediction, "energy", None)
    _check_prediction_tensor(
        predicted_energy,
        name="energy",
        shape=(batch.num_structures,),
        batch=batch,
    )
    finite_energy = torch.where(
        torch.isfinite(predicted_energy),
        predicted_energy,
        torch.zeros_like(predicted_energy),
    )
    anchor = finite_energy.sum() * 0.0
    energy = _energy_term(prediction, batch, config, anchor)
    force = _force_term(prediction, batch, config, anchor)
    stress = _stress_term(prediction, batch, config, anchor)
    total = (
        config.energy_weight * energy.mean
        + config.force_weight * force.mean
        + config.stress_weight * stress.mean
    )
    _require_finite(
        total,
        term_name="total",
        sample_ids=",".join(batch.sample_ids),
        quantity="loss",
    )
    return PotentialLossOutput(total, energy, force, stress)
