"""Translation-covariant primary-mode initialization."""

from __future__ import annotations

import math

import torch


def primary_phase_initialization(
    primary_cross: torch.Tensor, primary_modes: torch.Tensor
) -> torch.Tensor:
    """Construct a torus representative from exactly three primary modes.

    This solves the integer primary-mode coordinate map directly.  It is not
    ordinary least-squares phase unwrapping.  Branch jumps belong to the mode
    alias group, which static validation requires to equal the typed stabilizer.
    """

    if primary_modes.shape != (3, 3) or primary_modes.dtype != torch.long:
        raise ValueError("primary_modes must have shape [3,3] and dtype long")
    if primary_cross.shape[-1] != 3:
        raise ValueError("primary_cross must end in three primary modes")
    if not torch.is_complex(primary_cross):
        raise ValueError("primary_cross must be complex")
    if primary_cross.device != primary_modes.device:
        raise ValueError("primary_cross and primary_modes must share device")
    phases = torch.atan2(primary_cross.imag, primary_cross.real) / (2.0 * math.pi)
    matrix = primary_modes.to(dtype=phases.dtype)
    matrix = matrix.expand(phases.shape[:-1] + (3, 3))
    return torch.linalg.solve(matrix, phases.unsqueeze(-1)).squeeze(-1)
