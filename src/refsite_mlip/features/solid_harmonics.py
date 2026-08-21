"""e3nn 0.4.4 Cartesian regular solid harmonics."""

from __future__ import annotations

import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4


def regular_solid_harmonics(
    y: torch.Tensor, lmax: int = 2
) -> tuple[torch.Tensor, object]:
    if y.shape[-1] != 3 or y.dtype not in (torch.float32, torch.float64):
        raise ValueError("solid-harmonic input must have shape [...,3] and float32/64")
    if not bool(torch.all(torch.isfinite(y))):
        raise ValueError("solid-harmonic input contains NaN or Inf")
    if isinstance(lmax, bool) or not isinstance(lmax, int) or lmax < 0:
        raise ValueError("lmax must be a nonnegative integer")
    _, o3 = import_e3nn_0_4_4()
    irreps_sh = o3.Irreps.spherical_harmonics(lmax)
    values = o3.spherical_harmonics(
        irreps_sh,
        y,
        normalize=False,
        normalization="component",
    )
    return values, irreps_sh


def harmonic_slice(l: int) -> slice:
    if isinstance(l, bool) or not isinstance(l, int) or l < 0:
        raise ValueError("l must be a nonnegative integer")
    start = l * l
    return slice(start, (l + 1) * (l + 1))
