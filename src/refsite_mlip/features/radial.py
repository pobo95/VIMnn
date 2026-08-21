"""C2 compact radial basis expressed only through squared distance xi."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch


RADIAL_BASIS_VERSION = "c2_xi_polynomial_v1"


def _positive_fixed_real(value: Real, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive fixed real")
    return float(value)


def c2_envelope(u: torch.Tensor) -> torch.Tensor:
    if not u.is_floating_point():
        raise ValueError("radial coordinate u must be floating point")
    polynomial = 1.0 - 10.0 * u.pow(3) + 15.0 * u.pow(4) - 6.0 * u.pow(5)
    return torch.where(u < 1.0, polynomial, torch.zeros_like(u))


def compact_radial_basis(
    xi: torch.Tensor,
    *,
    n_radial: int,
    ell_feature: Real,
    r_cut: Real,
) -> torch.Tensor:
    if (
        isinstance(n_radial, bool)
        or not isinstance(n_radial, Integral)
        or int(n_radial) <= 0
    ):
        raise ValueError("n_radial must be a positive integer")
    ell = _positive_fixed_real(ell_feature, "ell_feature")
    cutoff = _positive_fixed_real(r_cut, "r_cut")
    if not xi.is_floating_point() or xi.dtype not in (torch.float32, torch.float64):
        raise ValueError("xi must use float32 or float64")
    if not bool(torch.all(torch.isfinite(xi))):
        raise ValueError("xi contains NaN or Inf")
    if bool(torch.any(xi < 0.0)):
        raise ValueError("xi must be nonnegative")
    xi_cut = (cutoff / ell) ** 2
    u = xi / xi.new_tensor(xi_cut)
    polynomial_channels = [torch.ones_like(u)]
    for _ in range(1, int(n_radial)):
        polynomial_channels.append(polynomial_channels[-1] * u)
    stacked = torch.stack(polynomial_channels, dim=-1)
    return stacked * c2_envelope(u).unsqueeze(-1)
