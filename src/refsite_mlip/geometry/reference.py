"""Reference-lattice reconstruction."""

from __future__ import annotations

import torch


def aligned_reference_sites(
    sites_fractional: torch.Tensor,
    phase: torch.Tensor,
    origin: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Return ``origin + (sites_fractional + phase) @ cell``."""

    if sites_fractional.shape[-1] != 3 or phase.shape[-1] != 3:
        raise ValueError("fractional sites and phase must end in dimension 3")
    shifted = sites_fractional + phase.unsqueeze(-2)
    return origin.unsqueeze(-2) + shifted @ cell
