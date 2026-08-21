"""Differentiable row-vector cell operations."""

from __future__ import annotations

from typing import Tuple

import torch


def _validate_geometry(
    positions: torch.Tensor, origin: torch.Tensor, cell: torch.Tensor
) -> None:
    if (
        not positions.is_floating_point()
        or not origin.is_floating_point()
        or not cell.is_floating_point()
    ):
        raise ValueError("positions, origin, and cell must be floating tensors")
    if (
        positions.shape[-1] != 3
        or origin.shape[-1] != 3
        or cell.shape[-2:] != (3, 3)
    ):
        raise ValueError(
            "expected positions [..., N, 3], origin [..., 3], "
            "cell [..., 3, 3]"
        )
    if positions.dtype != origin.dtype or positions.dtype != cell.dtype:
        raise ValueError("positions, origin, and cell must have the same dtype")
    if positions.device != origin.device or positions.device != cell.device:
        raise ValueError("positions, origin, and cell must be on the same device")


def fractional_coordinates(
    positions: torch.Tensor, origin: torch.Tensor, cell: torch.Tensor
) -> torch.Tensor:
    """Return ``(positions-origin) @ inverse(cell)`` without forming an inverse."""

    _validate_geometry(positions, origin, cell)
    relative = positions - origin.unsqueeze(-2)
    return torch.linalg.solve(
        cell.transpose(-1, -2), relative.transpose(-1, -2)
    ).transpose(-1, -2)


def affine_deform(
    positions: torch.Tensor,
    origin: torch.Tensor,
    cell: torch.Tensor,
    deformation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the row-vector affine convention ``rF, oF, HF``."""

    _validate_geometry(positions, origin, cell)
    if deformation.shape[-2:] != (3, 3):
        raise ValueError("deformation must have shape [..., 3, 3]")
    deformed_positions = positions @ deformation
    deformed_origin = (origin.unsqueeze(-2) @ deformation).squeeze(-2)
    deformed_cell = cell @ deformation
    return deformed_positions, deformed_origin, deformed_cell
