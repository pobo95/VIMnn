from __future__ import annotations

import torch

from conftest import reciprocal_fields
from refsite_mlip.geometry.cell import affine_deform, fractional_coordinates
from refsite_mlip.geometry.reference import aligned_reference_sites
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase


def _strained_energy(strain, data):
    identity = torch.eye(3, dtype=strain.dtype, device=strain.device)
    positions, origin, cell = affine_deform(
        data["positions"], data["origin"], data["cell"], identity + strain
    )
    _, _, cross = reciprocal_fields(
        data, positions=positions, origin=origin, cell=cell
    )
    initial = primary_phase_initialization(cross[:3], data["modes"][:3])
    phase = solve_training_phase(
        cross,
        data["modes"],
        data["mode_weights"],
        initial,
        (0.7, 0.8, 0.9),
        (2.0, 1.0, 0.5),
    ).phase
    references = aligned_reference_sites(data["sites"], phase, origin, cell)
    return 0.5 * (positions - references).square().sum()


def _symmetric_directions(dtype):
    directions = []
    for axis in range(3):
        value = torch.zeros((3, 3), dtype=dtype)
        value[axis, axis] = 1.0
        directions.append(value)
    for left, right in ((1, 2), (0, 2), (0, 1)):
        value = torch.zeros((3, 3), dtype=dtype)
        value[left, right] = 0.5
        value[right, left] = 0.5
        directions.append(value)
    return directions


def test_triclinic_affine_strain_preserves_fractional_coordinates(typed_crystal):
    strain = torch.tensor(
        [[0.03, 0.01, -0.02], [0.01, -0.02, 0.015], [-0.02, 0.015, 0.01]],
        dtype=torch.float64,
    )
    identity = torch.eye(3, dtype=torch.float64)
    positions, origin, cell = affine_deform(
        typed_crystal["positions"],
        typed_crystal["origin"],
        typed_crystal["cell"],
        identity + strain,
    )
    before = fractional_coordinates(
        typed_crystal["positions"], typed_crystal["origin"], typed_crystal["cell"]
    )
    after = fractional_coordinates(positions, origin, cell)
    torch.testing.assert_close(after, before, atol=3.0e-13, rtol=3.0e-13)


def test_six_symmetric_strain_derivatives_match_central_difference(typed_crystal):
    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    energy = _strained_energy(strain, typed_crystal)
    derivative = torch.autograd.grad(energy, strain)[0]
    step = 2.0e-6
    for direction in _symmetric_directions(torch.float64):
        plus = _strained_energy(step * direction, typed_crystal)
        minus = _strained_energy(-step * direction, typed_crystal)
        finite_difference = (plus - minus) / (2.0 * step)
        automatic = torch.sum(derivative * direction)
        torch.testing.assert_close(
            automatic, finite_difference, atol=2.0e-7, rtol=2.0e-6
        )
