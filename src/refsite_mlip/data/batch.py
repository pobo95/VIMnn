"""Padding-free ragged batches of :class:`StructureSample` objects."""

from __future__ import annotations

from dataclasses import dataclass
import string
from typing import Sequence

import torch

from .schema import StructureSample
from .templates import TemplateRegistry


@dataclass(frozen=True)
class TemplateGroup:
    """Deterministic structure and atom membership for one template ID."""

    template_id: str
    template_fingerprint: str
    structure_indices: torch.Tensor
    atom_slices: tuple[slice, ...]
    atom_indices: torch.Tensor


@dataclass(frozen=True)
class StructureBatch:
    """Flattened ragged structures and masked optional supervision.

    Missing targets use zero-valued placeholders and false masks.  The
    structure-level presence flags preserve the distinction between a missing
    target and a supplied target whose component mask is entirely false.
    """

    sample_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    template_fingerprints: tuple[str, ...]
    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    cells: torch.Tensor
    origins: torch.Tensor
    pbc: torch.Tensor
    atom_ptr: torch.Tensor
    atom_batch: torch.Tensor
    energy: torch.Tensor
    energy_mask: torch.Tensor
    forces: torch.Tensor
    force_mask: torch.Tensor
    stress: torch.Tensor
    stress_mask: torch.Tensor
    force_present: torch.Tensor
    stress_present: torch.Tensor
    force_mask_provided: torch.Tensor
    stress_mask_provided: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_structures(self) -> int:
        return len(self.sample_ids)

    @property
    def num_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        return self.positions.dtype

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def validate(self) -> None:
        batch_size = self.num_structures
        if batch_size == 0:
            raise ValueError("StructureBatch must contain at least one structure")
        if len(set(self.sample_ids)) != batch_size:
            raise ValueError("StructureBatch sample_ids must be unique")
        if (
            len(self.template_ids) != batch_size
            or len(self.template_fingerprints) != batch_size
        ):
            raise ValueError("template metadata must have length B")
        if any(not isinstance(value, str) or not value for value in self.sample_ids):
            raise ValueError("sample_ids must be nonempty strings")
        if any(not isinstance(value, str) or not value for value in self.template_ids):
            raise ValueError("template_ids must be nonempty strings")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in string.hexdigits for character in value)
            for value in self.template_fingerprints
        ):
            raise ValueError("template fingerprints must be SHA-256 hex strings")
        fingerprints_by_id = {}
        for template_id, fingerprint in zip(
            self.template_ids, self.template_fingerprints
        ):
            previous = fingerprints_by_id.setdefault(template_id, fingerprint)
            if previous != fingerprint:
                raise ValueError("one template ID has inconsistent fingerprints")

        if self.positions.shape != (self.num_atoms, 3):
            raise ValueError("positions must have shape [N_total,3]")
        if self.positions.dtype not in (torch.float32, torch.float64):
            raise ValueError("batch floating dtype must be float32 or float64")
        expected_shapes = (
            (self.atomic_numbers, (self.num_atoms,), torch.long, "atomic_numbers"),
            (self.cells, (batch_size, 3, 3), self.dtype, "cells"),
            (self.origins, (batch_size, 3), self.dtype, "origins"),
            (self.pbc, (batch_size, 3), torch.bool, "pbc"),
            (self.atom_ptr, (batch_size + 1,), torch.long, "atom_ptr"),
            (self.atom_batch, (self.num_atoms,), torch.long, "atom_batch"),
            (self.energy, (batch_size,), self.dtype, "energy"),
            (self.energy_mask, (batch_size,), torch.bool, "energy_mask"),
            (self.forces, (self.num_atoms, 3), self.dtype, "forces"),
            (self.force_mask, (self.num_atoms, 3), torch.bool, "force_mask"),
            (self.stress, (batch_size, 3, 3), self.dtype, "stress"),
            (self.stress_mask, (batch_size, 3, 3), torch.bool, "stress_mask"),
            (self.force_present, (batch_size,), torch.bool, "force_present"),
            (self.stress_present, (batch_size,), torch.bool, "stress_present"),
            (
                self.force_mask_provided,
                (batch_size,),
                torch.bool,
                "force_mask_provided",
            ),
            (
                self.stress_mask_provided,
                (batch_size,),
                torch.bool,
                "stress_mask_provided",
            ),
        )
        for tensor, shape, dtype, name in expected_shapes:
            if tensor.shape != shape or tensor.dtype != dtype:
                raise ValueError(f"{name} shape/dtype mismatch")
            if tensor.device != self.device:
                raise ValueError("all StructureBatch tensors must share a device")

        floating = (self.positions, self.cells, self.origins, self.energy, self.forces, self.stress)
        if any(not bool(torch.all(torch.isfinite(value))) for value in floating):
            raise ValueError("StructureBatch contains NaN or Inf")
        if self.num_atoms and bool(torch.any(self.atomic_numbers <= 0)):
            raise ValueError("atomic numbers must be positive")
        if not bool(torch.all(self.pbc)):
            raise ValueError("only pbc=[True,True,True] is supported")
        singular_values = torch.linalg.svdvals(self.cells)
        if bool(torch.any(singular_values[..., -1] <= torch.finfo(self.dtype).eps)):
            raise ValueError("batch cells must be nonsingular")

        if int(self.atom_ptr[0]) != 0 or int(self.atom_ptr[-1]) != self.num_atoms:
            raise ValueError("atom_ptr endpoints are inconsistent")
        if bool(torch.any(self.atom_ptr[1:] < self.atom_ptr[:-1])):
            raise ValueError("atom_ptr must be nondecreasing")
        expected_atom_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device, dtype=torch.long),
            self.atom_ptr[1:] - self.atom_ptr[:-1],
        )
        if not torch.equal(self.atom_batch, expected_atom_batch):
            raise ValueError("atom_batch is inconsistent with atom_ptr")

        if bool(torch.any(self.energy[~self.energy_mask] != 0)):
            raise ValueError("missing energy placeholders must be zero")
        if bool(torch.any(self.force_mask & ~self.force_present[self.atom_batch, None])):
            raise ValueError("missing force targets must have false masks")
        if bool(torch.any(self.stress_mask & ~self.stress_present[:, None, None])):
            raise ValueError("missing stress targets must have false masks")
        if bool(torch.any(self.force_mask_provided & ~self.force_present)):
            raise ValueError("force mask cannot be provided without force target")
        if bool(torch.any(self.stress_mask_provided & ~self.stress_present)):
            raise ValueError("stress mask cannot be provided without stress target")

        missing_force_atoms = ~self.force_present[self.atom_batch]
        if self.num_atoms and bool(torch.any(self.forces[missing_force_atoms] != 0)):
            raise ValueError("missing force placeholders must be zero")
        if bool(torch.any(self.stress[~self.stress_present] != 0)):
            raise ValueError("missing stress placeholders must be zero")
        for structure_index in range(batch_size):
            atom_slice = self._atom_slice(structure_index)
            if (
                bool(self.force_present[structure_index])
                and not bool(self.force_mask_provided[structure_index])
                and not bool(torch.all(self.force_mask[atom_slice]))
            ):
                raise ValueError("implicit force masks must select every component")
            if (
                bool(self.stress_present[structure_index])
                and not bool(self.stress_mask_provided[structure_index])
                and not bool(torch.all(self.stress_mask[structure_index]))
            ):
                raise ValueError("implicit stress masks must select every component")

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("structure index must be an integer")
        normalized = index + self.num_structures if index < 0 else index
        if normalized < 0 or normalized >= self.num_structures:
            raise IndexError("structure index out of range")
        return normalized

    def _atom_slice(self, index: int) -> slice:
        start = int(self.atom_ptr[index])
        stop = int(self.atom_ptr[index + 1])
        return slice(start, stop)

    @property
    def template_groups(self) -> tuple[TemplateGroup, ...]:
        groups = []
        for template_id in sorted(set(self.template_ids)):
            indices = [
                index
                for index, value in enumerate(self.template_ids)
                if value == template_id
            ]
            fingerprints = {self.template_fingerprints[index] for index in indices}
            if len(fingerprints) != 1:
                raise ValueError("one template ID has inconsistent fingerprints")
            structure_indices = torch.tensor(
                indices, dtype=torch.long, device=self.device
            )
            atom_slices = tuple(self._atom_slice(index) for index in indices)
            atom_indices = torch.cat(
                [
                    torch.arange(
                        atom_slice.start,
                        atom_slice.stop,
                        dtype=torch.long,
                        device=self.device,
                    )
                    for atom_slice in atom_slices
                ]
            )
            groups.append(
                TemplateGroup(
                    template_id=template_id,
                    template_fingerprint=fingerprints.pop(),
                    structure_indices=structure_indices,
                    atom_slices=atom_slices,
                    atom_indices=atom_indices,
                )
            )
        return tuple(groups)

    def structure_slice(self, index: int) -> StructureSample:
        """Reconstruct one sample without sharing mutable tensor storage."""

        index = self._normalize_index(index)
        atom_slice = self._atom_slice(index)
        has_forces = bool(self.force_present[index])
        has_stress = bool(self.stress_present[index])
        return StructureSample(
            sample_id=self.sample_ids[index],
            positions=self.positions[atom_slice].clone(),
            atomic_numbers=self.atomic_numbers[atom_slice].clone(),
            cell=self.cells[index].clone(),
            pbc=self.pbc[index].clone(),
            origin=self.origins[index].clone(),
            template_id=self.template_ids[index],
            energy=self.energy[index].clone() if bool(self.energy_mask[index]) else None,
            forces=self.forces[atom_slice].clone() if has_forces else None,
            stress=self.stress[index].clone() if has_stress else None,
            force_mask=(
                self.force_mask[atom_slice].clone()
                if has_forces and bool(self.force_mask_provided[index])
                else None
            ),
            stress_mask=(
                self.stress_mask[index].clone()
                if has_stress and bool(self.stress_mask_provided[index])
                else None
            ),
        )

    def unbind(self) -> tuple[StructureSample, ...]:
        return tuple(self.structure_slice(index) for index in range(self.num_structures))

    def to(self, device=None, dtype=None) -> "StructureBatch":
        """Move tensors while casting only floating tensors."""

        target_dtype = self.dtype if dtype is None else dtype
        if target_dtype not in (torch.float32, torch.float64):
            raise ValueError("batch floating dtype must be float32 or float64")

        def floating(tensor):
            return tensor.to(device=device, dtype=target_dtype)

        def fixed(tensor):
            return tensor.to(device=device)

        return StructureBatch(
            sample_ids=self.sample_ids,
            template_ids=self.template_ids,
            template_fingerprints=self.template_fingerprints,
            positions=floating(self.positions),
            atomic_numbers=fixed(self.atomic_numbers),
            cells=floating(self.cells),
            origins=floating(self.origins),
            pbc=fixed(self.pbc),
            atom_ptr=fixed(self.atom_ptr),
            atom_batch=fixed(self.atom_batch),
            energy=floating(self.energy),
            energy_mask=fixed(self.energy_mask),
            forces=floating(self.forces),
            force_mask=fixed(self.force_mask),
            stress=floating(self.stress),
            stress_mask=fixed(self.stress_mask),
            force_present=fixed(self.force_present),
            stress_present=fixed(self.stress_present),
            force_mask_provided=fixed(self.force_mask_provided),
            stress_mask_provided=fixed(self.stress_mask_provided),
        )


def collate_structure_samples(
    samples: Sequence[StructureSample],
    template_registry: TemplateRegistry,
) -> StructureBatch:
    """Collate validated samples without padding or implicit conversion."""

    if not isinstance(template_registry, TemplateRegistry):
        raise TypeError("template_registry must be a TemplateRegistry")
    samples = tuple(samples)
    if not samples:
        raise ValueError("cannot collate an empty batch")
    if any(not isinstance(sample, StructureSample) for sample in samples):
        raise TypeError("all entries must be StructureSample objects")
    for sample in samples:
        sample.validate()
    sample_ids = tuple(sample.sample_id for sample in samples)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample_id in batch")

    dtype = samples[0].positions.dtype
    device = samples[0].positions.device
    for sample in samples[1:]:
        if sample.positions.dtype != dtype or sample.positions.device != device:
            raise ValueError("batch samples must share floating dtype and device")

    templates = [template_registry.resolve(sample.template_id) for sample in samples]
    for sample, template in zip(samples, templates):
        if sample.num_atoms > template.topology.num_sites:
            raise ValueError(f"N > M for sample {sample.sample_id}")
        species = set(sample.atomic_numbers.detach().cpu().tolist())
        if not species.issubset(set(template.supported_species)):
            raise ValueError(f"unknown species for template {sample.template_id}")

    atom_counts = torch.tensor(
        [sample.num_atoms for sample in samples], dtype=torch.long, device=device
    )
    atom_ptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(atom_counts, dim=0),
        )
    )
    atom_batch = torch.repeat_interleave(
        torch.arange(len(samples), dtype=torch.long, device=device), atom_counts
    )

    energy_mask = torch.tensor(
        [sample.energy is not None for sample in samples],
        dtype=torch.bool,
        device=device,
    )
    force_present = torch.tensor(
        [sample.forces is not None for sample in samples],
        dtype=torch.bool,
        device=device,
    )
    stress_present = torch.tensor(
        [sample.stress is not None for sample in samples],
        dtype=torch.bool,
        device=device,
    )
    force_mask_provided = torch.tensor(
        [sample.force_mask is not None for sample in samples],
        dtype=torch.bool,
        device=device,
    )
    stress_mask_provided = torch.tensor(
        [sample.stress_mask is not None for sample in samples],
        dtype=torch.bool,
        device=device,
    )

    energies = torch.stack(
        [
            sample.energy
            if sample.energy is not None
            else sample.positions.new_zeros(())
            for sample in samples
        ]
    )
    forces = torch.cat(
        [
            sample.forces
            if sample.forces is not None
            else sample.positions.new_zeros((sample.num_atoms, 3))
            for sample in samples
        ],
        dim=0,
    )
    force_masks = torch.cat(
        [
            sample.force_mask
            if sample.force_mask is not None
            else torch.full(
                (sample.num_atoms, 3),
                sample.forces is not None,
                dtype=torch.bool,
                device=device,
            )
            for sample in samples
        ],
        dim=0,
    )
    stresses = torch.stack(
        [
            sample.stress
            if sample.stress is not None
            else sample.positions.new_zeros((3, 3))
            for sample in samples
        ]
    )
    stress_masks = torch.stack(
        [
            sample.stress_mask
            if sample.stress_mask is not None
            else torch.full(
                (3, 3),
                sample.stress is not None,
                dtype=torch.bool,
                device=device,
            )
            for sample in samples
        ]
    )

    return StructureBatch(
        sample_ids=sample_ids,
        template_ids=tuple(sample.template_id for sample in samples),
        template_fingerprints=tuple(template.fingerprint for template in templates),
        positions=torch.cat([sample.positions for sample in samples], dim=0),
        atomic_numbers=torch.cat(
            [sample.atomic_numbers for sample in samples], dim=0
        ),
        cells=torch.stack([sample.cell for sample in samples]),
        origins=torch.stack([sample.origin for sample in samples]),
        pbc=torch.stack([sample.pbc for sample in samples]),
        atom_ptr=atom_ptr,
        atom_batch=atom_batch,
        energy=energies,
        energy_mask=energy_mask,
        forces=forces,
        force_mask=force_masks,
        stress=stresses,
        stress_mask=stress_masks,
        force_present=force_present,
        stress_present=stress_present,
        force_mask_provided=force_mask_provided,
        stress_mask_provided=stress_mask_provided,
    )
