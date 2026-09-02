"""Detached, ragged prediction snapshots for production inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from refsite_mlip.transport import (
    CandidateReuseDecision,
    CompactCandidateNeighborState,
)


def _require_detached(name: str, value: torch.Tensor) -> None:
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be a detached prediction snapshot")


@dataclass(frozen=True)
class StructurePrediction:
    """One structure view of a :class:`BatchPrediction`.

    Stress remains tensile-positive in eV/angstrom^3 and uses Voigt ordering
    ``[xx, yy, zz, yz, xz, xy]``. No target label is copied into this record.
    """

    energy: torch.Tensor
    baseline_energy: torch.Tensor
    residual_energy: torch.Tensor
    forces: torch.Tensor | None
    stress: torch.Tensor | None
    stress_voigt: torch.Tensor | None
    site_energy: torch.Tensor
    sample_id: str
    template_id: str
    diagnostics: Mapping[str, Any] | None = None
    candidate_neighbor_state: CompactCandidateNeighborState | None = None
    candidate_reuse_decision: CandidateReuseDecision | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class BatchPrediction:
    """Padding-free predictions in original structure and atom ordering."""

    energy: torch.Tensor
    baseline_energy: torch.Tensor
    residual_energy: torch.Tensor
    forces: torch.Tensor | None
    stress: torch.Tensor | None
    stress_voigt: torch.Tensor | None
    site_energy: torch.Tensor
    atom_ptr: torch.Tensor
    site_ptr: torch.Tensor
    sample_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    diagnostics: tuple[Mapping[str, Any] | None, ...]
    candidate_neighbor_states: Mapping[
        str, CompactCandidateNeighborState
    ] | None = None
    candidate_reuse_decisions: Mapping[str, CandidateReuseDecision] | None = None

    def __post_init__(self) -> None:
        batch_size = len(self.sample_ids)
        if batch_size == 0:
            raise ValueError("BatchPrediction must contain at least one structure")
        if len(set(self.sample_ids)) != batch_size:
            raise ValueError("BatchPrediction sample_ids must be unique")
        if len(self.template_ids) != batch_size or len(self.diagnostics) != batch_size:
            raise ValueError("prediction metadata must have length B")
        if self.energy.shape != (batch_size,):
            raise ValueError("energy must have shape [B]")
        for name, value in (
            ("baseline_energy", self.baseline_energy),
            ("residual_energy", self.residual_energy),
        ):
            if value.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
        if self.atom_ptr.shape != (batch_size + 1,) or self.atom_ptr.dtype != torch.long:
            raise ValueError("atom_ptr must be torch.long [B+1]")
        if self.site_ptr.shape != (batch_size + 1,) or self.site_ptr.dtype != torch.long:
            raise ValueError("site_ptr must be torch.long [B+1]")
        if int(self.atom_ptr[0].detach().cpu()) != 0 or int(
            self.site_ptr[0].detach().cpu()
        ) != 0:
            raise ValueError("ragged pointers must start at zero")
        if bool(torch.any(self.atom_ptr[1:] < self.atom_ptr[:-1])) or bool(
            torch.any(self.site_ptr[1:] < self.site_ptr[:-1])
        ):
            raise ValueError("ragged pointers must be nondecreasing")
        num_atoms = int(self.atom_ptr[-1].detach().cpu())
        num_sites = int(self.site_ptr[-1].detach().cpu())
        if self.site_energy.shape != (num_sites,):
            raise ValueError("site_energy shape is inconsistent with site_ptr")
        if self.forces is not None and self.forces.shape != (num_atoms, 3):
            raise ValueError("forces must have shape [N_total,3]")
        if self.stress is not None and self.stress.shape != (batch_size, 3, 3):
            raise ValueError("stress must have shape [B,3,3]")
        if self.stress_voigt is not None and self.stress_voigt.shape != (batch_size, 6):
            raise ValueError("stress_voigt must have shape [B,6]")
        if (self.stress is None) != (self.stress_voigt is None):
            raise ValueError("stress and stress_voigt must be present together")

        floating = [
            self.energy,
            self.baseline_energy,
            self.residual_energy,
            self.site_energy,
        ]
        if self.forces is not None:
            floating.append(self.forces)
        if self.stress is not None:
            assert self.stress_voigt is not None
            floating.extend((self.stress, self.stress_voigt))
        dtype = self.energy.dtype
        device = self.energy.device
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("prediction floating dtype must be float32 or float64")
        for index, value in enumerate(floating):
            if value.dtype != dtype or value.device != device:
                raise ValueError("prediction floating tensors must share dtype/device")
            _require_detached(f"prediction tensor {index}", value)
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError("prediction contains NaN or Inf")
        for name, value in (("atom_ptr", self.atom_ptr), ("site_ptr", self.site_ptr)):
            _require_detached(name, value)
            if value.device != device:
                raise ValueError("prediction pointers must share output device")

        for mapping, name in (
            (self.candidate_neighbor_states, "candidate_neighbor_states"),
            (self.candidate_reuse_decisions, "candidate_reuse_decisions"),
        ):
            if mapping is not None and tuple(mapping) != self.sample_ids:
                raise ValueError(f"{name} must follow original sample order")

    @property
    def num_structures(self) -> int:
        return len(self.sample_ids)

    @property
    def num_atoms(self) -> int:
        return int(self.atom_ptr[-1].detach().cpu())

    @property
    def num_sites(self) -> int:
        return int(self.site_ptr[-1].detach().cpu())

    @property
    def dtype(self) -> torch.dtype:
        return self.energy.dtype

    @property
    def device(self) -> torch.device:
        return self.energy.device

    @property
    def auxiliary(self) -> tuple[Mapping[str, Any] | None, ...]:
        """Compatibility alias for structure-ordered diagnostics."""

        return self.diagnostics

    def structure(self, index: int) -> StructurePrediction:
        if not isinstance(index, int):
            raise TypeError("structure index must be an integer")
        normalized = index + self.num_structures if index < 0 else index
        if normalized < 0 or normalized >= self.num_structures:
            raise IndexError("structure index out of range")
        atom_start = int(self.atom_ptr[normalized].detach().cpu())
        atom_stop = int(self.atom_ptr[normalized + 1].detach().cpu())
        site_start = int(self.site_ptr[normalized].detach().cpu())
        site_stop = int(self.site_ptr[normalized + 1].detach().cpu())
        sample_id = self.sample_ids[normalized]
        return StructurePrediction(
            energy=self.energy[normalized],
            baseline_energy=self.baseline_energy[normalized],
            residual_energy=self.residual_energy[normalized],
            forces=(
                None if self.forces is None else self.forces[atom_start:atom_stop]
            ),
            stress=None if self.stress is None else self.stress[normalized],
            stress_voigt=(
                None
                if self.stress_voigt is None
                else self.stress_voigt[normalized]
            ),
            site_energy=self.site_energy[site_start:site_stop],
            sample_id=sample_id,
            template_id=self.template_ids[normalized],
            diagnostics=self.diagnostics[normalized],
            candidate_neighbor_state=(
                None
                if self.candidate_neighbor_states is None
                else self.candidate_neighbor_states[sample_id]
            ),
            candidate_reuse_decision=(
                None
                if self.candidate_reuse_decisions is None
                else self.candidate_reuse_decisions[sample_id]
            ),
        )

    @property
    def structures(self) -> tuple[StructurePrediction, ...]:
        return tuple(self.structure(index) for index in range(self.num_structures))

    def __len__(self) -> int:
        return self.num_structures

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        return self.structure(key)


__all__ = ["BatchPrediction", "StructurePrediction"]
