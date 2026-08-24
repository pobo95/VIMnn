"""Deterministic elemental atomic-energy baseline fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Literal

import torch

from refsite_mlip.data import StructureSample


BaselineWeighting = Literal["per_structure", "per_atom"]
RankPolicy = Literal["error", "minimum_norm"]


def _nonnegative_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _species_vocabulary(values) -> tuple[int, ...]:
    vocabulary = tuple(values)
    if (
        not vocabulary
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in vocabulary)
        or any(int(value) <= 0 for value in vocabulary)
        or len(set(int(value) for value in vocabulary)) != len(vocabulary)
    ):
        raise ValueError("species_vocabulary must contain unique positive integers")
    return tuple(int(value) for value in vocabulary)


@dataclass(frozen=True)
class AtomicBaselineConfig:
    weighting: BaselineWeighting = "per_structure"
    rcond: float | None = None
    ridge: float = 0.0
    rank_policy: RankPolicy = "error"

    def __post_init__(self) -> None:
        if self.weighting not in ("per_structure", "per_atom"):
            raise ValueError("weighting must be 'per_structure' or 'per_atom'")
        if self.rank_policy not in ("error", "minimum_norm"):
            raise ValueError("rank_policy must be 'error' or 'minimum_norm'")
        if self.rcond is not None:
            object.__setattr__(
                self, "rcond", _nonnegative_real("rcond", self.rcond)
            )
        object.__setattr__(self, "ridge", _nonnegative_real("ridge", self.ridge))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AtomicBaselineConfig":
        if not isinstance(values, Mapping):
            raise TypeError("baseline config must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class AtomicBaselineFit:
    species_vocabulary: tuple[int, ...]
    baseline_energies: torch.Tensor
    config: AtomicBaselineConfig
    training_sample_ids: tuple[str, ...]
    num_valid_energy_structures: int
    rank: int
    singular_values: torch.Tensor
    condition_number: float
    species_occurrence_counts: torch.Tensor
    residual_rmse: float
    residual_mae: float
    weighted_objective: float

    def __post_init__(self) -> None:
        vocabulary = _species_vocabulary(self.species_vocabulary)
        object.__setattr__(self, "species_vocabulary", vocabulary)
        if not isinstance(self.config, AtomicBaselineConfig):
            raise TypeError("config must be an AtomicBaselineConfig")
        if (
            isinstance(self.num_valid_energy_structures, bool)
            or not isinstance(self.num_valid_energy_structures, Integral)
            or int(self.num_valid_energy_structures) <= 0
        ):
            raise ValueError("num_valid_energy_structures must be positive")
        object.__setattr__(
            self,
            "num_valid_energy_structures",
            int(self.num_valid_energy_structures),
        )
        if len(self.training_sample_ids) != self.num_valid_energy_structures:
            raise ValueError("training sample ID count does not match valid structures")
        if any(not isinstance(value, str) or not value for value in self.training_sample_ids):
            raise ValueError("training sample IDs must be nonempty strings")
        if isinstance(self.rank, bool) or not isinstance(self.rank, Integral):
            raise TypeError("rank must be an integer")
        rank = int(self.rank)
        if rank < 0 or rank > len(vocabulary):
            raise ValueError("rank is outside the species dimension")
        object.__setattr__(self, "rank", rank)

        baseline = torch.as_tensor(self.baseline_energies).detach().cpu().to(torch.float64).contiguous().clone()
        singular = torch.as_tensor(self.singular_values).detach().cpu().to(torch.float64).contiguous().clone()
        occurrences = torch.as_tensor(self.species_occurrence_counts).detach().cpu().to(torch.long).contiguous().clone()
        if baseline.shape != (len(vocabulary),):
            raise ValueError("baseline energies must have shape [A]")
        if singular.ndim != 1:
            raise ValueError("singular values must be one-dimensional")
        if occurrences.shape != (len(vocabulary),):
            raise ValueError("species occurrence counts must have shape [A]")
        if not bool(torch.all(torch.isfinite(baseline))) or not bool(
            torch.all(torch.isfinite(singular))
        ):
            raise ValueError("baseline fit tensors must be finite")
        if bool(torch.any(singular < 0)) or bool(torch.any(occurrences < 0)):
            raise ValueError("singular values and occurrences must be nonnegative")
        object.__setattr__(self, "baseline_energies", baseline)
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "species_occurrence_counts", occurrences)

        condition = float(self.condition_number)
        if math.isnan(condition) or condition < 0.0:
            raise ValueError("condition number must be nonnegative or infinity")
        object.__setattr__(self, "condition_number", condition)
        for name in ("residual_rmse", "residual_mae", "weighted_objective"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)

    @property
    def rank_deficient(self) -> bool:
        return self.rank < len(self.species_vocabulary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_vocabulary": list(self.species_vocabulary),
            "baseline_energies": self.baseline_energies.tolist(),
            "config": self.config.to_dict(),
            "training_sample_ids": list(self.training_sample_ids),
            "num_valid_energy_structures": self.num_valid_energy_structures,
            "rank": self.rank,
            "singular_values": self.singular_values.tolist(),
            "condition_number": self.condition_number,
            "species_occurrence_counts": self.species_occurrence_counts.tolist(),
            "residual_rmse": self.residual_rmse,
            "residual_mae": self.residual_mae,
            "weighted_objective": self.weighted_objective,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AtomicBaselineFit":
        if not isinstance(values, Mapping):
            raise TypeError("baseline fit must be reconstructed from a mapping")
        data = dict(values)
        data["species_vocabulary"] = tuple(data["species_vocabulary"])
        data["training_sample_ids"] = tuple(data["training_sample_ids"])
        data["config"] = AtomicBaselineConfig.from_dict(data["config"])
        data["baseline_energies"] = torch.tensor(
            data["baseline_energies"], dtype=torch.float64
        )
        data["singular_values"] = torch.tensor(
            data["singular_values"], dtype=torch.float64
        )
        data["species_occurrence_counts"] = torch.tensor(
            data["species_occurrence_counts"], dtype=torch.long
        )
        return cls(**data)


def _validated_indices(dataset: Sequence, train_indices) -> tuple[int, ...]:
    if not isinstance(dataset, Sequence):
        raise TypeError("dataset must implement the Sequence contract")
    indices = tuple(train_indices)
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in indices):
        raise TypeError("training indices must be integers")
    indices = tuple(int(index) for index in indices)
    if len(set(indices)) != len(indices):
        raise ValueError("training indices must not contain duplicates")
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError("training index is out of range")
    return indices


def fit_atomic_baseline(
    dataset: Sequence,
    train_indices,
    species_vocabulary,
    config: AtomicBaselineConfig,
) -> AtomicBaselineFit:
    """Fit elemental E0 values using only explicitly selected training samples."""

    if not isinstance(config, AtomicBaselineConfig):
        raise TypeError("config must be an AtomicBaselineConfig")
    vocabulary = _species_vocabulary(species_vocabulary)
    species_to_index = {species: index for index, species in enumerate(vocabulary)}
    indices = _validated_indices(dataset, train_indices)
    rows = []
    targets = []
    sample_ids = []
    occurrence_counts = torch.zeros(len(vocabulary), dtype=torch.long)

    for dataset_index in indices:
        sample = dataset[dataset_index]
        if not isinstance(sample, StructureSample):
            raise TypeError("dataset entries must be StructureSample objects")
        sample.validate()
        atomic_numbers = sample.atomic_numbers.detach().cpu().to(torch.long)
        unknown = sorted(set(atomic_numbers.tolist()) - set(vocabulary))
        if unknown:
            raise ValueError(
                f"unknown species {unknown} in training sample {sample.sample_id}"
            )
        if sample.energy is None:
            continue
        composition = torch.zeros(len(vocabulary), dtype=torch.float64)
        for species, species_index in species_to_index.items():
            composition[species_index] = torch.count_nonzero(
                atomic_numbers == species
            )
        occurrence_counts += composition.to(torch.long)
        rows.append(composition)
        targets.append(sample.energy.detach().cpu().to(torch.float64))
        sample_ids.append(sample.sample_id)

    if not rows:
        raise ValueError("no valid energy labels in the selected training split")
    absent = [
        vocabulary[index]
        for index, count in enumerate(occurrence_counts.tolist())
        if count == 0
    ]
    if absent:
        raise ValueError(
            f"species absent from energy-labeled training structures: {absent}"
        )

    composition_matrix = torch.stack(rows)
    target_energy = torch.stack(targets)
    if config.weighting == "per_atom":
        atom_counts = composition_matrix.sum(dim=1)
        if bool(torch.any(atom_counts <= 0)):
            raise ValueError("per_atom weighting requires nonempty structures")
        design_matrix = composition_matrix / atom_counts[:, None]
        weighted_target = target_energy / atom_counts
    else:
        design_matrix = composition_matrix
        weighted_target = target_energy

    left, singular_values, right_h = torch.linalg.svd(
        design_matrix, full_matrices=False
    )
    rcond = (
        config.rcond
        if config.rcond is not None
        else torch.finfo(torch.float64).eps * max(design_matrix.shape)
    )
    cutoff = float(rcond) * float(singular_values.max())
    retained = singular_values > cutoff
    rank = int(torch.count_nonzero(retained))
    if rank < len(vocabulary) and config.rank_policy == "error":
        raise ValueError(
            "atomic baseline composition matrix is rank deficient: "
            f"rank={rank}, species={len(vocabulary)}; add independent training "
            "compositions or set rank_policy='minimum_norm' explicitly"
        )

    projected_target = left.transpose(0, 1) @ weighted_target
    if config.ridge > 0.0:
        filter_values = singular_values / (
            singular_values.square() + config.ridge
        )
    else:
        filter_values = torch.zeros_like(singular_values)
        filter_values[retained] = torch.reciprocal(singular_values[retained])
    baseline = right_h.transpose(0, 1) @ (filter_values * projected_target)

    physical_residual = composition_matrix @ baseline - target_energy
    weighted_residual = design_matrix @ baseline - weighted_target
    rmse = float(torch.sqrt(torch.mean(physical_residual.square())))
    mae = float(torch.mean(torch.abs(physical_residual)))
    objective = float(
        torch.sum(weighted_residual.square())
        + config.ridge * torch.sum(baseline.square())
    )
    condition = (
        math.inf
        if rank < len(vocabulary)
        else float(singular_values[0] / singular_values[len(vocabulary) - 1])
    )
    return AtomicBaselineFit(
        species_vocabulary=vocabulary,
        baseline_energies=baseline,
        config=config,
        training_sample_ids=tuple(sample_ids),
        num_valid_energy_structures=len(rows),
        rank=rank,
        singular_values=singular_values,
        condition_number=condition,
        species_occurrence_counts=occurrence_counts,
        residual_rmse=rmse,
        residual_mae=mae,
        weighted_objective=objective,
    )


def apply_atomic_baseline_(model, fit: AtomicBaselineFit):
    """Copy a canonical fit into an existing model buffer in place."""

    if not isinstance(fit, AtomicBaselineFit):
        raise TypeError("fit must be an AtomicBaselineFit")
    config = getattr(model, "config", None)
    model_vocabulary = getattr(config, "species_vocabulary", None)
    if tuple(model_vocabulary or ()) != fit.species_vocabulary:
        raise ValueError("model and baseline species vocabulary/order do not match")
    buffers = getattr(model, "_buffers", {})
    if "atomic_baseline" not in buffers:
        raise ValueError("model does not expose atomic_baseline as a buffer")
    baseline = model.atomic_baseline
    if not isinstance(baseline, torch.Tensor) or baseline.shape != fit.baseline_energies.shape:
        raise ValueError("model atomic_baseline buffer shape mismatch")
    if isinstance(baseline, torch.nn.Parameter):
        raise ValueError("atomic_baseline must not be a trainable Parameter")
    with torch.no_grad():
        baseline.copy_(fit.baseline_energies.to(device=baseline.device, dtype=baseline.dtype))
    baseline.requires_grad_(False)
    return model
