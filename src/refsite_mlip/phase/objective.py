"""Typed reciprocal fields and their periodic alignment objective."""

from __future__ import annotations

import math
from typing import Tuple

import torch

from refsite_mlip.geometry.cell import fractional_coordinates


def _validate_modes(modes: torch.Tensor, device: torch.device) -> None:
    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("modes must have shape [G, 3]")
    if modes.dtype != torch.long:
        raise ValueError("reciprocal modes must have dtype torch.long")
    if modes.device != device:
        raise ValueError("modes must be on the input device")


def typed_reciprocal_fields(
    positions: torch.Tensor,
    origin: torch.Tensor,
    cell: torch.Tensor,
    sites_fractional: torch.Tensor,
    atom_channel_weights: torch.Tensor,
    site_channel_weights: torch.Tensor,
    modes: torch.Tensor,
    channel_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return atomic fields, static reference fields, and typed cross fields."""

    _validate_modes(modes, positions.device)
    if sites_fractional.ndim != 2 or sites_fractional.shape[1] != 3:
        raise ValueError("sites_fractional must have shape [M, 3]")
    if atom_channel_weights.shape != (positions.shape[-2], site_channel_weights.shape[1]):
        raise ValueError("atom channel weights must have shape [N, C]")
    if site_channel_weights.shape[0] != sites_fractional.shape[0]:
        raise ValueError("site channel weights must have shape [M, C]")
    if channel_weights.shape != (site_channel_weights.shape[1],):
        raise ValueError("channel_weights must have shape [C]")
    for value in (sites_fractional, atom_channel_weights, site_channel_weights, channel_weights):
        if value.dtype != positions.dtype or value.device != positions.device:
            raise ValueError("all floating phase inputs must share dtype and device")

    fractional = fractional_coordinates(positions, origin, cell)
    real_modes = modes.to(dtype=positions.dtype)
    atom_angles = 2.0 * math.pi * torch.einsum("...ni,gi->...ng", fractional, real_modes)
    site_angles = 2.0 * math.pi * torch.einsum("mi,gi->mg", sites_fractional, real_modes)
    atom_phases = torch.polar(torch.ones_like(atom_angles), atom_angles)
    site_phases = torch.polar(torch.ones_like(site_angles), site_angles)
    complex_atom_weights = atom_channel_weights.to(dtype=atom_phases.dtype)
    complex_site_weights = site_channel_weights.to(dtype=site_phases.dtype)
    complex_channel_weights = channel_weights.to(dtype=atom_phases.dtype)
    atomic = torch.einsum("nc,...ng->...gc", complex_atom_weights, atom_phases)
    reference = torch.einsum("mc,mg->gc", complex_site_weights, site_phases)
    cross = torch.einsum(
        "c,...gc,gc->...g",
        complex_channel_weights,
        atomic,
        reference.conj(),
    )
    return atomic, reference, cross


def phase_terms(
    phase: torch.Tensor, cross: torch.Tensor, modes: torch.Tensor
) -> torch.Tensor:
    _validate_modes(modes, phase.device)
    if cross.shape[-1] != modes.shape[0]:
        raise ValueError("cross and modes disagree on mode count")
    if not torch.is_complex(cross):
        raise ValueError("cross must be a complex tensor")
    if cross.device != phase.device:
        raise ValueError("cross, phase, and modes must share device")
    expected_dtype = (
        torch.complex128 if phase.dtype == torch.float64 else torch.complex64
    )
    if cross.dtype != expected_dtype:
        raise ValueError("cross precision must match the real phase dtype")
    angles = -2.0 * math.pi * torch.einsum(
        "...i,gi->...g", phase, modes.to(dtype=phase.dtype)
    )
    return cross * torch.polar(torch.ones_like(angles), angles)


def phase_objective(
    phase: torch.Tensor,
    cross: torch.Tensor,
    modes: torch.Tensor,
    mode_weights: torch.Tensor,
) -> torch.Tensor:
    if mode_weights.shape != (modes.shape[0],):
        raise ValueError("mode_weights must have shape [G]")
    if mode_weights.dtype != phase.dtype or mode_weights.device != phase.device:
        raise ValueError("mode_weights must share phase dtype and device")
    return torch.sum(mode_weights * phase_terms(phase, cross, modes).real, dim=-1)


def phase_gradient_hessian(
    phase: torch.Tensor,
    cross: torch.Tensor,
    modes: torch.Tensor,
    mode_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic gradient and Hessian of the periodic phase objective."""

    if mode_weights.shape != (modes.shape[0],):
        raise ValueError("mode_weights must have shape [G]")
    if mode_weights.dtype != phase.dtype or mode_weights.device != phase.device:
        raise ValueError("mode_weights must share phase dtype and device")
    terms = phase_terms(phase, cross, modes)
    real_modes = modes.to(dtype=phase.dtype)
    gradient = 2.0 * math.pi * torch.einsum(
        "...g,gi->...i", mode_weights * terms.imag, real_modes
    )
    hessian = -(2.0 * math.pi) ** 2 * torch.einsum(
        "...g,gi,gj->...ij", mode_weights * terms.real, real_modes, real_modes
    )
    return gradient, hessian
