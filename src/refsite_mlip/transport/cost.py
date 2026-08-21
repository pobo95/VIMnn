"""Differentiable atom-reference geometry and dense OT costs."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import torch


def _pbc_tuple(pbc: Sequence[bool] | torch.Tensor) -> tuple[bool, bool, bool]:
    if isinstance(pbc, torch.Tensor):
        if pbc.shape != (3,) or pbc.dtype != torch.bool:
            raise ValueError("pbc tensor must be bool with shape [3]")
        values = pbc.detach().cpu().tolist()
    else:
        values = list(pbc)
    if len(values) != 3 or any(not isinstance(value, bool) for value in values):
        raise ValueError("pbc must contain exactly three booleans")
    return tuple(values)


@dataclass(frozen=True)
class MICImageDiagnostics:
    """Nearest-image diagnostics; distances are not used to alter features."""

    displacement: torch.Tensor
    nearest_distance: torch.Tensor
    second_nearest_distance: torch.Tensor
    unique_image_gap: torch.Tensor


def minimum_image_diagnostics(
    displacement: torch.Tensor,
    cell: torch.Tensor,
    pbc: Sequence[bool] | torch.Tensor,
    *,
    image_range: int = 2,
    minimum_unique_gap: Real | None = None,
) -> MICImageDiagnostics:
    """Return the certified nearest image and an independently bounded runner-up.

    The gap is ``second_nearest_distance - nearest_distance`` in Cartesian
    length units.  It is a diagnostic of the accepted MIC branch only: no plan,
    displacement, or feature is clipped or thresholded.
    """

    nearest = minimum_image_displacement(
        displacement, cell, pbc, image_range=image_range
    )
    periodic = _pbc_tuple(pbc)
    if minimum_unique_gap is not None and (
        isinstance(minimum_unique_gap, bool)
        or not isinstance(minimum_unique_gap, Real)
        or not math.isfinite(float(minimum_unique_gap))
        or float(minimum_unique_gap) < 0.0
    ):
        raise ValueError("minimum_unique_gap must be finite and nonnegative")
    nearest_distance = torch.linalg.vector_norm(nearest, dim=-1)
    if not any(periodic):
        second = torch.full_like(nearest_distance, torch.inf)
        gap = second.clone()
    elif displacement.numel() == 0:
        second = nearest_distance.clone()
        gap = nearest_distance.clone()
    else:
        fractional = torch.linalg.solve(
            cell.T, displacement.reshape(-1, 3).T
        ).T.reshape(displacement.shape)
        mask = displacement.new_tensor(periodic)
        base = torch.round(fractional) * mask
        base_residual = fractional - base
        smallest_singular_value = torch.linalg.svdvals(cell).min()

        # A radius-one periodic stencil supplies an upper bound U on the
        # runner-up.  Any omitted integer offset k obeys
        # ||(x-k)H|| >= sigma_min(H) (||k||-||x||), so the radius below
        # certifies that no omitted image can improve on U.
        seed_axes = [range(-1, 2) if enabled else (0,) for enabled in periodic]
        seed_offsets = torch.tensor(
            list(itertools.product(*seed_axes)),
            dtype=displacement.dtype,
            device=displacement.device,
        )
        seed_candidates = (
            base_residual.unsqueeze(-2) - seed_offsets
        ) @ cell
        seed_squared = seed_candidates.square().sum(dim=-1)
        seed_second = torch.topk(
            seed_squared, k=2, dim=-1, largest=False
        ).values[..., 1].sqrt()
        required = (
            seed_second / smallest_singular_value
            + torch.linalg.vector_norm(base_residual, dim=-1)
        ).max()
        certified_radius = math.ceil(float(required.detach().cpu())) + 1
        search_radius = max(int(image_range), certified_radius)
        axes = [
            range(-search_radius, search_radius + 1) if enabled else (0,)
            for enabled in periodic
        ]
        offsets = torch.tensor(
            list(itertools.product(*axes)),
            dtype=displacement.dtype,
            device=displacement.device,
        )
        candidates = (base_residual.unsqueeze(-2) - offsets) @ cell
        distances = candidates.square().sum(dim=-1)
        best_two = torch.topk(distances, k=2, dim=-1, largest=False).values.sqrt()
        nearest_distance = best_two[..., 0]
        second = best_two[..., 1]
        gap = second - nearest_distance

    if minimum_unique_gap is not None and bool(
        torch.any(gap <= float(minimum_unique_gap))
    ):
        raise ValueError(
            "MIC nearest-image branch is not unique within minimum_unique_gap"
        )
    return MICImageDiagnostics(
        displacement=nearest,
        nearest_distance=nearest_distance,
        second_nearest_distance=second,
        unique_image_gap=gap,
    )


def minimum_image_displacement(
    displacement: torch.Tensor,
    cell: torch.Tensor,
    pbc: Sequence[bool] | torch.Tensor,
    *,
    image_range: int = 2,
) -> torch.Tensor:
    """Return the exact triclinic image using a certified finite search."""

    if displacement.shape[-1] != 3 or cell.shape != (3, 3):
        raise ValueError("expected displacement [...,3] and cell [3,3]")
    if displacement.dtype != cell.dtype or displacement.device != cell.device:
        raise ValueError("displacement and cell must share dtype and device")
    if displacement.dtype not in (torch.float32, torch.float64):
        raise ValueError("MIC supports float32 and float64")
    if not bool(torch.all(torch.isfinite(displacement))) or not bool(
        torch.all(torch.isfinite(cell))
    ):
        raise ValueError("MIC inputs contain NaN or Inf")
    determinant = torch.linalg.det(cell)
    if not bool(torch.isfinite(determinant)) or float(
        determinant.abs().detach().cpu()
    ) <= torch.finfo(cell.dtype).eps:
        raise ValueError("physical cell is singular")
    if (
        isinstance(image_range, bool)
        or not isinstance(image_range, Integral)
        or image_range < 1
    ):
        raise ValueError("image_range must be a positive integer")
    periodic = _pbc_tuple(pbc)
    if not any(periodic):
        return displacement

    if displacement.numel() == 0:
        return displacement
    fractional = torch.linalg.solve(
        cell.T, displacement.reshape(-1, 3).T
    ).T.reshape(displacement.shape)
    mask = displacement.new_tensor(periodic)
    base = torch.round(fractional) * mask

    base_cartesian = (fractional - base) @ cell
    current_bound = torch.linalg.vector_norm(base_cartesian, dim=-1).max()
    smallest_singular_value = torch.linalg.svdvals(cell).min()
    certified_radius = math.ceil(
        float((current_bound / smallest_singular_value).detach().cpu()) + 0.5
    )
    search_radius = max(int(image_range), certified_radius)
    axes = [
        range(-search_radius, search_radius + 1) if enabled else (0,)
        for enabled in periodic
    ]
    offsets = torch.tensor(
        list(itertools.product(*axes)),
        dtype=displacement.dtype,
        device=displacement.device,
    )
    shifts = base.unsqueeze(-2) + offsets
    candidate_fractional = fractional.unsqueeze(-2) - shifts
    candidates = candidate_fractional @ cell
    squared = candidates.square().sum(dim=-1)
    selected = torch.argmin(squared, dim=-1)
    gather = selected.unsqueeze(-1).unsqueeze(-1).expand(
        selected.shape + (1, 3)
    )
    return torch.gather(candidates, -2, gather).squeeze(-2)


def atom_site_displacements(
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    pbc: Sequence[bool] | torch.Tensor,
    *,
    image_range: int = 2,
) -> torch.Tensor:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")
    if reference_sites.ndim != 2 or reference_sites.shape[1] != 3:
        raise ValueError("reference_sites must have shape [M,3]")
    if positions.dtype != reference_sites.dtype or positions.device != reference_sites.device:
        raise ValueError("positions and reference_sites must share dtype and device")
    raw = positions.unsqueeze(0) - reference_sites.unsqueeze(1)
    return minimum_image_displacement(raw, cell, pbc, image_range=image_range)


def atom_site_cost(
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    pbc: Sequence[bool] | torch.Tensor,
    ell_ot: Real,
    *,
    image_range: int = 2,
) -> torch.Tensor:
    if (
        isinstance(ell_ot, bool)
        or not isinstance(ell_ot, Real)
        or not math.isfinite(float(ell_ot))
        or float(ell_ot) <= 0.0
    ):
        raise ValueError("ell_ot must be a finite positive fixed real")
    displacement = atom_site_displacements(
        positions,
        reference_sites,
        cell,
        pbc,
        image_range=image_range,
    )
    return displacement.square().sum(dim=-1) / (2.0 * float(ell_ot) ** 2)
