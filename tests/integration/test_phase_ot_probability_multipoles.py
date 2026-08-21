from __future__ import annotations

import math

import pytest
import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.features import ProbabilityMultipoleConfig, build_probability_multipoles
from refsite_mlip.geometry.cell import affine_deform
from refsite_mlip.geometry.reference import aligned_reference_sites
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.phase.stabilizer import find_typed_stabilizer, permutation_for_translation
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    EvalOTConfig,
    TrainSinkhornConfig,
    atom_site_cost,
    atom_site_displacements,
    minimum_image_diagnostics,
    solve_atom_vacancy_ot,
)
from refsite_mlip.transport.operating_point import OTOperatingDomain, audit_train_fixed_operating_point


PHASE_STEPS = (0.7, 0.8, 0.9, 1.0)
PHASE_DAMPING = (2.0, 1.0, 0.5, 0.2)
DOMAIN = OTOperatingDomain(0.5, 1.5, "float64", 256, 1.0e-7)
TRAIN_CONFIG = TrainSinkhornConfig(256, 1.0e-7)
EVAL_CONFIG = EvalOTConfig(sinkhorn_iterations=16, convergence_tolerance=1.0e-12)
FEATURE_CONFIG = ProbabilityMultipoleConfig(
    species_vocabulary=(6, 41),
    n_radial=2,
    lmax=2,
    ell_feature=1.0,
    r_cut=3.0,
    site_type_vocabulary=(0, 1),
)


def _numbers(weights):
    vocabulary = torch.tensor([6, 41], dtype=torch.long, device=weights.device)
    return vocabulary[torch.argmax(weights, dim=-1)]


def _pipeline(
    data,
    positions,
    atom_weights,
    *,
    origin=None,
    cell=None,
    sites=None,
    site_weights=None,
    site_types=None,
    path=TRAIN_FIXED,
    scale=None,
):
    origin = data["origin"] if origin is None else origin
    cell = data["cell"] if cell is None else cell
    sites = data["sites"] if sites is None else sites
    site_weights = data["site_weights"] if site_weights is None else site_weights
    site_types = data["site_types"] if site_types is None else site_types
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
    displacements = atom_site_displacements(
        positions, references, cell, (True, True, True)
    )
    cost = displacements.square().sum(dim=-1) / (2.0 * DOMAIN.ell_ot**2)
    if path == TRAIN_FIXED:
        ot = solve_atom_vacancy_ot(
            cost, DOMAIN.epsilon_ot, path, "sinkhorn", TRAIN_CONFIG
        )
    else:
        ot = solve_atom_vacancy_ot(
            cost, DOMAIN.epsilon_ot, path, "hybrid", EVAL_CONFIG
        )
    features = build_probability_multipoles(
        ot.P,
        ot.q,
        _numbers(atom_weights),
        displacements,
        FEATURE_CONFIG,
        site_types,
    )
    if scale is None:
        scale = positions.new_ones(())
    energy = 0.37 * ot.q.square().sum()
    species_coefficients = positions.new_tensor([0.19, 0.31])
    energy = energy + torch.sum(
        features.species_probabilities.square() * species_coefficients
    )
    for channel, metadata in enumerate(features.channel_metadata):
        if metadata.exact_occupancy:
            continue
        start, stop = metadata.component_slice
        coefficient = positions.new_tensor(0.007 * (channel + 1))
        energy = energy + coefficient * features.equivariant_features[:, start:stop].square().sum()
    energy = scale * energy
    return {
        "energy": energy,
        "ot": ot,
        "features": features,
        "phase": phase,
        "references": references,
        "displacements": displacements,
        "cost": cost,
    }


def _vacancy_inputs(data):
    return data["positions"][:5], data["atom_weights"][:5]


def _rotation(dtype=torch.float64):
    axis = torch.tensor([0.3, -0.5, 0.8], dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.71, dtype=dtype)
    cross = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=dtype,
    )
    return torch.eye(3, dtype=dtype) + torch.sin(angle) * cross + (1.0 - torch.cos(angle)) * (cross @ cross)


def _strain_energy(data, strain, *, path=TRAIN_FIXED):
    deformation = torch.eye(3, dtype=strain.dtype, device=strain.device) + strain
    positions, origin, cell = affine_deform(
        data["positions"][:5], data["origin"], data["cell"], deformation
    )
    return _pipeline(
        data,
        positions,
        data["atom_weights"][:5],
        origin=origin,
        cell=cell,
        path=path,
    )["energy"]


def _stress(data, path=TRAIN_FIXED):
    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    return torch.autograd.grad(_strain_energy(data, strain, path=path), strain)[0]


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


def test_operating_domain_train_eval_feature_force_stress_parity(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    train_positions = positions.clone().requires_grad_(True)
    train = _pipeline(typed_crystal, train_positions, weights)
    audit = audit_train_fixed_operating_point(
        train["ot"], train["cost"], DOMAIN, structure_id="typed_vacancy"
    )
    eval_positions = positions.clone().requires_grad_(True)
    adaptive = _pipeline(
        typed_crystal, eval_positions, weights, path=EVAL_ADAPTIVE
    )
    train_force = -torch.autograd.grad(train["energy"], train_positions)[0]
    eval_force = -torch.autograd.grad(adaptive["energy"], eval_positions)[0]
    assert audit.residual <= 1.0e-7
    torch.testing.assert_close(train["ot"].P, adaptive["ot"].P, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(train["ot"].q, adaptive["ot"].q, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(train["features"].equivariant_features, adaptive["features"].equivariant_features, atol=8e-12, rtol=8e-12)
    torch.testing.assert_close(train["energy"], adaptive["energy"], atol=4e-12, rtol=4e-12)
    torch.testing.assert_close(train_force, eval_force, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(_stress(typed_crystal), _stress(typed_crystal, EVAL_ADAPTIVE), atol=2e-9, rtol=2e-9)


def test_full_path_translation_wrapping_and_permutations(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    baseline = _pipeline(typed_crystal, positions, weights)
    translation = torch.tensor([0.77, -1.03, 0.46], dtype=torch.float64)
    joint = _pipeline(
        typed_crystal,
        positions + translation,
        weights,
        origin=typed_crystal["origin"] + translation,
    )
    moved = _pipeline(typed_crystal, positions + translation, weights)
    lattice = torch.tensor([2.0, -1.0, 1.0], dtype=torch.float64) @ typed_crystal["cell"]
    lattice_moved = _pipeline(typed_crystal, positions + lattice, weights)
    wrapped_positions = positions.clone()
    wrapped_positions[0] = wrapped_positions[0] + lattice
    wrapped = _pipeline(typed_crystal, wrapped_positions, weights)
    for result in (joint, moved, lattice_moved, wrapped):
        torch.testing.assert_close(result["energy"], baseline["energy"], atol=8e-10, rtol=8e-10)
        torch.testing.assert_close(result["features"].equivariant_features, baseline["features"].equivariant_features, atol=3e-9, rtol=3e-9)

    atom_order = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
    atoms = _pipeline(typed_crystal, positions[atom_order], weights[atom_order])
    torch.testing.assert_close(atoms["features"].equivariant_features, baseline["features"].equivariant_features, atol=3e-12, rtol=3e-12)
    site_order = torch.tensor([4, 1, 5, 0, 3, 2], dtype=torch.long)
    sites = _pipeline(
        typed_crystal,
        positions,
        weights,
        sites=typed_crystal["sites"][site_order],
        site_weights=typed_crystal["site_weights"][site_order],
        site_types=typed_crystal["site_types"][site_order],
    )
    torch.testing.assert_close(sites["features"].equivariant_features, baseline["features"].equivariant_features[site_order], atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize(
    "matrix",
    [
        _rotation(),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
    ],
)
def test_full_multipole_o3_equivariance_energy_and_force(matrix, typed_crystal):
    _, o3 = import_e3nn_0_4_4()
    positions, weights = _vacancy_inputs(typed_crystal)
    positions = positions.clone().requires_grad_(True)
    baseline = _pipeline(typed_crystal, positions, weights)
    force = -torch.autograd.grad(baseline["energy"], positions)[0]
    transformed_positions = (positions.detach() @ matrix.T).requires_grad_(True)
    transformed = _pipeline(
        typed_crystal,
        transformed_positions,
        weights,
        origin=typed_crystal["origin"] @ matrix.T,
        cell=typed_crystal["cell"] @ matrix.T,
    )
    transformed_force = -torch.autograd.grad(transformed["energy"], transformed_positions)[0]
    representation = baseline["features"].irreps_out.D_from_matrix(matrix)
    expected = baseline["features"].equivariant_features @ representation.T
    torch.testing.assert_close(transformed["features"].equivariant_features, expected, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(transformed["energy"], baseline["energy"], atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(transformed_force, force @ matrix.T, atol=3e-8, rtol=3e-8)


def test_full_path_force_finite_difference_zero_net_and_double_backward(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    positions = positions.clone().requires_grad_(True)
    scale = torch.tensor(1.17, dtype=torch.float64, requires_grad=True)
    result = _pipeline(typed_crystal, positions, weights, scale=scale)
    force = -torch.autograd.grad(result["energy"], positions, create_graph=True)[0]
    torch.testing.assert_close(force.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=8e-10, rtol=0)
    step = 2e-6
    for atom, component in ((0, 0), (2, 1), (4, 2)):
        direction = torch.zeros_like(positions)
        direction[atom, component] = step
        plus = _pipeline(typed_crystal, positions.detach() + direction, weights, scale=scale.detach())["energy"]
        minus = _pipeline(typed_crystal, positions.detach() - direction, weights, scale=scale.detach())["energy"]
        finite = -(plus - minus) / (2.0 * step)
        torch.testing.assert_close(force[atom, component], finite, atol=8e-7, rtol=8e-6)
    loss = force.square().sum()
    parameter_gradient = torch.autograd.grad(loss, scale)[0]
    assert torch.isfinite(parameter_gradient) and parameter_gradient.abs() > 0


def test_full_phase_ot_feature_position_gradcheck_gradgradcheck(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    positions = positions.clone().requires_grad_(True)
    function = lambda value: _pipeline(typed_crystal, value, weights)["energy"]
    assert torch.autograd.gradcheck(function, (positions,), eps=1e-6, atol=3e-5, rtol=3e-4)
    assert torch.autograd.gradgradcheck(function, (positions,), eps=1e-6, atol=8e-5, rtol=8e-4)


def test_stress_symmetric_strain_finite_difference_and_rotation(typed_crystal):
    derivative = _stress(typed_crystal)
    step = 2e-6
    for direction in _symmetric_directions():
        finite = (_strain_energy(typed_crystal, step * direction) - _strain_energy(typed_crystal, -step * direction)) / (2.0 * step)
        automatic = torch.sum(derivative * direction)
        torch.testing.assert_close(automatic, finite, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(derivative, derivative.T, atol=2e-8, rtol=2e-8)

    rotation = _rotation()
    rotated = dict(typed_crystal)
    rotated["positions"] = typed_crystal["positions"] @ rotation.T
    rotated["origin"] = typed_crystal["origin"] @ rotation.T
    rotated["cell"] = typed_crystal["cell"] @ rotation.T
    rotated_stress = _stress(rotated)
    torch.testing.assert_close(rotated_stress, rotation @ derivative @ rotation.T, atol=3e-8, rtol=3e-8)


def test_mic_unique_image_gap_on_feature_fixture(typed_crystal):
    positions, weights = _vacancy_inputs(typed_crystal)
    result = _pipeline(typed_crystal, positions, weights)
    raw = positions.unsqueeze(0) - result["references"].unsqueeze(1)
    diagnostics = minimum_image_diagnostics(raw, typed_crystal["cell"], (True, True, True), minimum_unique_gap=1e-4)
    assert torch.all(diagnostics.unique_image_gap > 1e-4)
    torch.testing.assert_close(diagnostics.displacement, result["displacements"], atol=2e-13, rtol=0)


def test_typed_stabilizer_translation_permuted_ot_multipoles():
    sites = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.25, 0.1], [0.5, 0.25, 0.1]],
        dtype=torch.float64,
    )
    site_types = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    cell = torch.diag(torch.tensor([4.0, 4.5, 5.0], dtype=torch.float64))
    references = sites @ cell
    positions = references[[0, 1, 2]] + torch.tensor(
        [[0.04, -0.02, 0.01], [-0.03, 0.01, 0.02], [0.02, 0.03, -0.01]],
        dtype=torch.float64,
    )
    numbers = torch.tensor([6, 41, 6], dtype=torch.long)
    stabilizer = find_typed_stabilizer(sites, site_types)
    tau = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    permutation = permutation_for_translation(tau, stabilizer)

    def solve(reference):
        displacement = atom_site_displacements(positions, reference, cell, (True, True, True))
        cost = displacement.square().sum(dim=-1) / (2.0 * DOMAIN.ell_ot**2)
        ot = solve_atom_vacancy_ot(cost, DOMAIN.epsilon_ot, TRAIN_FIXED, "sinkhorn", TRAIN_CONFIG)
        feature = build_probability_multipoles(ot.P, ot.q, numbers, displacement, FEATURE_CONFIG, site_types)
        return ot, feature

    baseline_ot, baseline = solve(references)
    shifted_ot, shifted = solve((sites + tau) @ cell)
    torch.testing.assert_close(shifted_ot.P, baseline_ot.P[permutation], atol=3e-13, rtol=0)
    torch.testing.assert_close(shifted_ot.q, baseline_ot.q[permutation], atol=3e-13, rtol=0)
    torch.testing.assert_close(shifted.equivariant_features, baseline.equivariant_features[permutation], atol=3e-13, rtol=0)
