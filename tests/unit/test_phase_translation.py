from __future__ import annotations

import torch

from conftest import reciprocal_fields
from refsite_mlip.geometry.reference import aligned_reference_sites
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.stabilizer import torus_difference


def _energy_and_phase(data, positions):
    _, _, cross = reciprocal_fields(data, positions=positions)
    initial = primary_phase_initialization(cross[:3], data["modes"][:3])
    result = solve_training_phase(
        cross,
        data["modes"],
        data["mode_weights"],
        initial,
        (0.7, 0.8, 0.9, 1.0),
        (2.0, 1.0, 0.5, 0.2),
    )
    references = aligned_reference_sites(
        data["sites"], result.phase, data["origin"], data["cell"]
    )
    energy = 0.5 * (positions - references).square().sum()
    return energy, result.phase


def test_atom_only_translation_energy_force_and_zero_net_force(typed_crystal):
    positions = typed_crystal["positions"].clone().requires_grad_(True)
    energy, phase = _energy_and_phase(typed_crystal, positions)
    force = -torch.autograd.grad(energy, positions, create_graph=True)[0]

    translation = torch.tensor([0.57, -0.83, 0.36], dtype=torch.float64)
    moved = (positions.detach() + translation).requires_grad_(True)
    moved_energy, moved_phase = _energy_and_phase(typed_crystal, moved)
    moved_force = -torch.autograd.grad(moved_energy, moved)[0]

    fractional_translation = torch.linalg.solve(
        typed_crystal["cell"].T, translation
    )
    torch.testing.assert_close(energy, moved_energy, atol=2.0e-11, rtol=2.0e-11)
    torch.testing.assert_close(force, moved_force, atol=3.0e-10, rtol=3.0e-10)
    torch.testing.assert_close(
        torus_difference(moved_phase - phase, fractional_translation),
        torch.zeros(3, dtype=torch.float64),
        atol=4.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        force.sum(dim=0),
        torch.zeros(3, dtype=torch.float64),
        atol=4.0e-10,
        rtol=0.0,
    )
