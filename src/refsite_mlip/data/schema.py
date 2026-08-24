"""Typed single-structure schema in fixed internal units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

LENGTH_UNIT = "angstrom"
ENERGY_UNIT = "eV"
FORCE_UNIT = "eV/angstrom"
STRESS_UNIT = "eV/angstrom^3"
STRESS_SIGN = "tensile_positive"
STRESS_VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")

@dataclass(frozen=True)
class StructureSample:
    """One fully periodic structure in the package's fixed unit convention.

    Masks use the exact label shape: ``force_mask`` is ``[N, 3]`` and
    ``stress_mask`` is ``[3, 3]``.
    """

    sample_id: str
    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    cell: torch.Tensor
    pbc: torch.Tensor
    origin: torch.Tensor
    template_id: str
    energy: Optional[torch.Tensor] = None
    forces: Optional[torch.Tensor] = None
    stress: Optional[torch.Tensor] = None
    force_mask: Optional[torch.Tensor] = None
    stress_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_atoms(self) -> int:
        return int(self.positions.shape[0])

    def validate(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a nonempty string")
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be a nonempty string")
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape [N,3]")
        if (
            self.atomic_numbers.shape != (self.num_atoms,)
            or self.atomic_numbers.dtype != torch.long
        ):
            raise ValueError("atomic_numbers must be torch.long [N]")
        if self.cell.shape != (3, 3) or self.origin.shape != (3,):
            raise ValueError("cell/origin shape mismatch")
        if (
            self.pbc.shape != (3,)
            or self.pbc.dtype != torch.bool
            or not bool(torch.all(self.pbc))
        ):
            raise ValueError("only pbc=[True,True,True] is supported")

        geometry = (self.positions, self.cell, self.origin)
        if self.positions.dtype not in (torch.float32, torch.float64) or any(
            value.dtype != self.positions.dtype
            or value.device != self.positions.device
            for value in geometry
        ):
            raise ValueError(
                "geometry must share float32/float64 dtype and device"
            )
        if (
            self.atomic_numbers.device != self.positions.device
            or self.pbc.device != self.positions.device
        ):
            raise ValueError("all sample tensors must share device")
        if self.num_atoms and bool(torch.any(self.atomic_numbers <= 0)):
            raise ValueError("atomic numbers must be positive")
        if any(not bool(torch.all(torch.isfinite(value))) for value in geometry):
            raise ValueError("geometry contains NaN or Inf")
        singular_values = torch.linalg.svdvals(self.cell)
        if bool(singular_values.min() <= torch.finfo(self.cell.dtype).eps):
            raise ValueError("cell must be nonsingular")

        labels = (
            (self.energy, (), "energy"),
            (self.forces, (self.num_atoms, 3), "forces"),
            (self.stress, (3, 3), "stress"),
        )
        for value, shape, name in labels:
            if value is None:
                continue
            if (
                value.shape != shape
                or value.dtype != self.positions.dtype
                or value.device != self.positions.device
            ):
                raise ValueError(f"{name} shape/dtype/device mismatch")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{name} contains NaN or Inf")

        masks = (
            (self.force_mask, (self.num_atoms, 3), self.forces, "force_mask"),
            (self.stress_mask, (3, 3), self.stress, "stress_mask"),
        )
        for mask, shape, label, name in masks:
            if mask is None:
                continue
            if label is None:
                raise ValueError(f"{name} requires its label")
            if (
                mask.shape != shape
                or mask.dtype != torch.bool
                or mask.device != self.positions.device
            ):
                raise ValueError(f"{name} must be bool with label shape/device")

    def to(self, device=None, dtype=None) -> "StructureSample":
        """Return a validated copy on ``device`` with floating tensors cast."""

        target_dtype = self.positions.dtype if dtype is None else dtype
        if target_dtype not in (torch.float32, torch.float64):
            raise ValueError("sample floating dtype must be float32 or float64")

        def floating(value):
            return (
                None
                if value is None
                else value.to(device=device, dtype=target_dtype)
            )

        def fixed(value):
            return None if value is None else value.to(device=device)

        return StructureSample(
            sample_id=self.sample_id,
            positions=floating(self.positions),
            atomic_numbers=fixed(self.atomic_numbers),
            cell=floating(self.cell),
            pbc=fixed(self.pbc),
            origin=floating(self.origin),
            template_id=self.template_id,
            energy=floating(self.energy),
            forces=floating(self.forces),
            stress=floating(self.stress),
            force_mask=fixed(self.force_mask),
            stress_mask=fixed(self.stress_mask),
        )
