from __future__ import annotations

import math

import pytest
import torch

from refsite_mlip.phase.objective import typed_reciprocal_fields


@pytest.fixture
def typed_crystal():
    dtype = torch.float64
    cell = torch.tensor(
        [[4.1, 0.2, -0.1], [0.4, 3.7, 0.3], [-0.2, 0.5, 3.5]],
        dtype=dtype,
    )
    origin = torch.tensor([0.73, -0.41, 0.29], dtype=dtype)
    sites = torch.tensor(
        [
            [0.03, 0.07, 0.11],
            [0.31, 0.19, 0.43],
            [0.57, 0.37, 0.23],
            [0.79, 0.71, 0.61],
            [0.17, 0.83, 0.47],
            [0.68, 0.12, 0.88],
        ],
        dtype=dtype,
    )
    site_types = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    phase = torch.tensor([0.173, -0.219, 0.137], dtype=dtype)
    displacement = torch.tensor(
        [
            [0.008, -0.013, 0.004],
            [-0.011, 0.006, 0.009],
            [0.004, 0.010, -0.007],
            [-0.006, -0.005, 0.012],
            [0.009, 0.002, -0.010],
            [-0.004, -0.008, -0.003],
        ],
        dtype=dtype,
    )
    positions = origin + (sites + phase) @ cell + displacement
    atom_weights = torch.nn.functional.one_hot(site_types, num_classes=2).to(dtype)
    site_weights = atom_weights.clone()
    channel_weights = torch.tensor([1.0, 1.3], dtype=dtype)
    modes = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]],
        dtype=torch.long,
    )
    mode_weights = torch.tensor([1.0, 1.1, 0.9, 0.4, 0.35], dtype=dtype)
    return {
        "cell": cell,
        "origin": origin,
        "sites": sites,
        "site_types": site_types,
        "phase": phase,
        "positions": positions,
        "atom_weights": atom_weights,
        "site_weights": site_weights,
        "channel_weights": channel_weights,
        "modes": modes,
        "mode_weights": mode_weights,
    }


def reciprocal_fields(data, positions=None, origin=None, cell=None):
    return typed_reciprocal_fields(
        data["positions"] if positions is None else positions,
        data["origin"] if origin is None else origin,
        data["cell"] if cell is None else cell,
        data["sites"],
        data["atom_weights"],
        data["site_weights"],
        data["modes"],
        data["channel_weights"],
    )


def phase_factor(modes, shift, dtype):
    angles = 2.0 * math.pi * (modes.to(dtype=dtype) @ shift)
    return torch.polar(torch.ones_like(angles), angles)
