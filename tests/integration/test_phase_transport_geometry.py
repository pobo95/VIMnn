from __future__ import annotations

import itertools

import torch

from refsite_mlip.geometry.cell import affine_deform
from refsite_mlip.geometry.reference import aligned_reference_sites
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.transport import (
    TRAIN_FIXED,
    TrainSinkhornConfig,
    atom_site_cost,
    minimum_image_displacement,
    solve_atom_vacancy_ot,
)


PHASE_STEPS = (0.7, 0.8, 0.9, 1.0)
PHASE_DAMPING = (2.0, 1.0, 0.5, 0.2)
OT_CONFIG = TrainSinkhornConfig(iterations=96, diagnostic_tolerance=1.0e-7)


def _solve_phase_ot(
    data,
    positions,
    atom_weights,
    *,
    origin=None,
    cell=None,
    sites=None,
    site_weights=None,
    pbc=(True, True, True),
):
    origin = data["origin"] if origin is None else origin
    cell = data["cell"] if cell is None else cell
    sites = data["sites"] if sites is None else sites
    site_weights = data["site_weights"] if site_weights is None else site_weights
    _, _, cross = typed_reciprocal_fields(
        positions,
        origin,
        cell,
        sites,
        atom_weights,
        site_weights,
        data["modes"],
        data["channel_weights"],
    )
    initial = primary_phase_initialization(cross[:3], data["modes"][:3])
    phase = solve_training_phase(
        cross,
        data["modes"],
        data["mode_weights"],
        initial,
        PHASE_STEPS,
        PHASE_DAMPING,
    ).phase
    references = aligned_reference_sites(sites, phase, origin, cell)
    cost = atom_site_cost(positions, references, cell, pbc, 0.85)
    result = solve_atom_vacancy_ot(
        cost, 0.38, TRAIN_FIXED, "sinkhorn", OT_CONFIG
    )
    energy = torch.sum(result.P * cost.square()) + 0.27 * result.q.square().sum()
    return energy, result, phase, cost


def _vacancy_inputs(data):
    return data["positions"][:5], data["atom_weights"][:5]


def test_phase_cost_ot_joint_and_atom_only_translation(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    energy, result, phase, _ = _solve_phase_ot(typed_crystal, positions, weights)
    translation = torch.tensor([0.77, -1.03, 0.46], dtype=torch.float64)

    joint_energy, joint_result, joint_phase, _ = _solve_phase_ot(
        typed_crystal,
        positions + translation,
        weights,
        origin=typed_crystal["origin"] + translation,
    )
    moved_energy, moved_result, _, _ = _solve_phase_ot(
        typed_crystal, positions + translation, weights
    )
    torch.testing.assert_close(joint_energy, energy, atol=2.0e-11, rtol=2.0e-11)
    torch.testing.assert_close(moved_energy, energy, atol=2.0e-10, rtol=2.0e-10)
    torch.testing.assert_close(joint_result.P, result.P, atol=2.0e-11, rtol=2.0e-11)
    torch.testing.assert_close(moved_result.P, result.P, atol=2.0e-10, rtol=2.0e-10)
    torch.testing.assert_close(joint_phase, phase, atol=2.0e-12, rtol=0.0)


def test_lattice_wrapping_and_partial_no_pbc(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    energy, result, _, _ = _solve_phase_ot(typed_crystal, positions, weights)
    wrapped = positions.clone()
    wrapped[0] = wrapped[0] + torch.tensor([2.0, -1.0, 1.0], dtype=torch.float64) @ typed_crystal["cell"]
    wrapped_energy, wrapped_result, _, _ = _solve_phase_ot(
        typed_crystal, wrapped, weights
    )
    torch.testing.assert_close(wrapped_energy, energy, atol=3.0e-10, rtol=3.0e-10)
    torch.testing.assert_close(wrapped_result.P, result.P, atol=3.0e-10, rtol=3.0e-10)

    displacement = torch.tensor([[3.9, -0.2, 0.1]], dtype=torch.float64)
    partial = minimum_image_displacement(
        displacement, typed_crystal["cell"], (True, False, False)
    )
    none = minimum_image_displacement(
        displacement, typed_crystal["cell"], (False, False, False)
    )
    torch.testing.assert_close(none, displacement, atol=0, rtol=0)
    assert torch.linalg.vector_norm(partial) < torch.linalg.vector_norm(displacement)


def test_atom_and_site_permutation(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    energy, result, _, _ = _solve_phase_ot(typed_crystal, positions, weights)

    atom_permutation = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
    atom_energy, atom_result, _, _ = _solve_phase_ot(
        typed_crystal,
        positions[atom_permutation],
        weights[atom_permutation],
    )
    torch.testing.assert_close(atom_energy, energy, atol=2.0e-11, rtol=2.0e-11)
    torch.testing.assert_close(
        atom_result.P, result.P[:, atom_permutation], atol=2.0e-11, rtol=2.0e-11
    )
    torch.testing.assert_close(atom_result.q, result.q, atol=2.0e-11, rtol=2.0e-11)

    site_permutation = torch.tensor([4, 1, 5, 0, 3, 2], dtype=torch.long)
    site_energy, site_result, _, _ = _solve_phase_ot(
        typed_crystal,
        positions,
        weights,
        sites=typed_crystal["sites"][site_permutation],
        site_weights=typed_crystal["site_weights"][site_permutation],
    )
    torch.testing.assert_close(site_energy, energy, atol=2.0e-11, rtol=2.0e-11)
    torch.testing.assert_close(
        site_result.P, result.P[site_permutation], atol=2.0e-11, rtol=2.0e-11
    )
    torch.testing.assert_close(
        site_result.q, result.q[site_permutation], atol=2.0e-11, rtol=2.0e-11
    )


def _rotation(dtype):
    axis = torch.tensor([0.3, -0.5, 0.8], dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.71, dtype=dtype)
    cross = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=dtype,
    )
    identity = torch.eye(3, dtype=dtype)
    return identity + torch.sin(angle) * cross + (1.0 - torch.cos(angle)) * (cross @ cross)


def test_rotation_energy_plan_and_force_covariance(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    positions = positions.clone().requires_grad_(True)
    energy, result, _, _ = _solve_phase_ot(typed_crystal, positions, weights)
    force = -torch.autograd.grad(energy, positions)[0]

    rotation = _rotation(torch.float64)
    rotated_positions = (positions.detach() @ rotation.T).requires_grad_(True)
    rotated_origin = typed_crystal["origin"] @ rotation.T
    rotated_cell = typed_crystal["cell"] @ rotation.T
    rotated_energy, rotated_result, _, _ = _solve_phase_ot(
        typed_crystal,
        rotated_positions,
        weights,
        origin=rotated_origin,
        cell=rotated_cell,
    )
    rotated_force = -torch.autograd.grad(rotated_energy, rotated_positions)[0]
    torch.testing.assert_close(rotated_energy, energy, atol=3.0e-10, rtol=3.0e-10)
    torch.testing.assert_close(rotated_result.P, result.P, atol=3.0e-10, rtol=3.0e-10)
    torch.testing.assert_close(rotated_force, force @ rotation.T, atol=2.0e-8, rtol=2.0e-8)


def test_zero_net_force(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    positions = positions.clone().requires_grad_(True)
    energy, _, _, _ = _solve_phase_ot(typed_crystal, positions, weights)
    force = -torch.autograd.grad(energy, positions)[0]
    torch.testing.assert_close(
        force.sum(dim=0),
        torch.zeros(3, dtype=torch.float64),
        atol=3.0e-9,
        rtol=0.0,
    )


def _independent_mic_oracle(displacement, cell, image_range=4):
    candidates = []
    for integer in itertools.product(
        range(-image_range, image_range + 1), repeat=3
    ):
        shift = torch.tensor(integer, dtype=cell.dtype) @ cell
        candidates.append(displacement - shift)
    stacked = torch.stack(candidates)
    return stacked[torch.argmin(stacked.square().sum(dim=-1))]


def test_triclinic_mic_matches_integer_image_enumeration_oracle(typed_crystal):
    displacement = torch.tensor([5.73, -3.81, 4.29], dtype=torch.float64)
    production = minimum_image_displacement(
        displacement, typed_crystal["cell"], (True, True, True), image_range=2
    )
    oracle = _independent_mic_oracle(displacement, typed_crystal["cell"])
    torch.testing.assert_close(production, oracle, atol=2.0e-13, rtol=0.0)


def _strained_energy(strain, data):
    identity = torch.eye(3, dtype=torch.float64)
    positions, origin, cell = affine_deform(
        data["positions"][:5], data["origin"], data["cell"], identity + strain
    )
    energy, _, _, _ = _solve_phase_ot(
        data,
        positions,
        data["atom_weights"][:5],
        origin=origin,
        cell=cell,
    )
    return energy


def _symmetric_directions():
    directions = []
    for axis in range(3):
        value = torch.zeros((3, 3), dtype=torch.float64)
        value[axis, axis] = 1.0
        directions.append(value)
    for left, right in ((1, 2), (0, 2), (0, 1)):
        value = torch.zeros((3, 3), dtype=torch.float64)
        value[left, right] = value[right, left] = 0.5
        directions.append(value)
    return directions


def test_phase_ot_affine_strain_finite_difference(typed_crystal):
    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(_strained_energy(strain, typed_crystal), strain)[0]
    step = 2.0e-6
    for direction in _symmetric_directions():
        finite = (
            _strained_energy(step * direction, typed_crystal)
            - _strained_energy(-step * direction, typed_crystal)
        ) / (2.0 * step)
        automatic = torch.sum(derivative * direction)
        torch.testing.assert_close(automatic, finite, atol=3.0e-7, rtol=3.0e-6)


def test_typed_stabilizer_translation_induces_site_permutation():
    cell = torch.diag(torch.tensor([2.0, 3.0, 3.0], dtype=torch.float64))
    sites = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    atom = torch.tensor([[0.12, 0.0, 0.0]], dtype=torch.float64)
    first_cost = atom_site_cost(atom, sites, cell, (True, True, True), 0.8)
    shifted_sites = sites[torch.tensor([1, 0])]
    shifted_cost = atom_site_cost(atom, shifted_sites, cell, (True, True, True), 0.8)
    first = solve_atom_vacancy_ot(
        first_cost, 0.3, TRAIN_FIXED, "sinkhorn", OT_CONFIG
    )
    shifted = solve_atom_vacancy_ot(
        shifted_cost, 0.3, TRAIN_FIXED, "sinkhorn", OT_CONFIG
    )
    torch.testing.assert_close(shifted.P, first.P[[1, 0]], atol=2.0e-13, rtol=0.0)
    torch.testing.assert_close(shifted.q, first.q[[1, 0]], atol=2.0e-13, rtol=0.0)


def test_phase_ot_force_matches_central_finite_difference(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    positions = positions.clone().requires_grad_(True)
    energy, _, _, _ = _solve_phase_ot(typed_crystal, positions, weights)
    force = -torch.autograd.grad(energy, positions)[0]
    step = 2.0e-6
    indices = ((0, 0), (2, 1), (4, 2))
    for atom, component in indices:
        direction = torch.zeros_like(positions)
        direction[atom, component] = step
        plus = _solve_phase_ot(
            typed_crystal, positions.detach() + direction, weights
        )[0]
        minus = _solve_phase_ot(
            typed_crystal, positions.detach() - direction, weights
        )[0]
        finite_force = -(plus - minus) / (2.0 * step)
        torch.testing.assert_close(
            force[atom, component],
            finite_force,
            atol=3.0e-7,
            rtol=3.0e-6,
        )


def test_certified_mic_handles_pathological_typed_cell_beyond_radius_two():
    cell = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8944192358949739, 0.05269652753075747, 0.0],
            [0.9584232970442724, 0.06409389957653025, 0.044633321049509775],
        ],
        dtype=torch.float64,
    )
    fractional = torch.tensor(
        [-0.13734369100154487, 1.389928358159995, -2.6546577769633917],
        dtype=torch.float64,
    )
    displacement = fractional @ cell
    production = minimum_image_displacement(
        displacement, cell, (True, True, True), image_range=2
    )
    oracle = _independent_mic_oracle(displacement, cell, image_range=8)
    naive = (fractional - torch.round(fractional)) @ cell
    assert torch.linalg.vector_norm(production) < 0.3 * torch.linalg.vector_norm(naive)
    torch.testing.assert_close(production, oracle, atol=3.0e-13, rtol=0.0)
