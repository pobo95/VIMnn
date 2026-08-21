from __future__ import annotations

import torch

from conftest import reciprocal_fields
from refsite_mlip.phase.evaluation import solve_evaluation_phase
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.stabilizer import find_typed_stabilizer


def _phase_from_positions(positions, data):
    _, _, cross = reciprocal_fields(data, positions=positions)
    initial = primary_phase_initialization(cross[:3], data["modes"][:3])
    return solve_training_phase(
        cross,
        data["modes"],
        data["mode_weights"],
        initial,
        (0.7, 0.8, 0.9),
        (2.0, 1.0, 0.5),
    ).phase


def test_training_solver_gradcheck_and_gradgradcheck(typed_crystal):
    positions = typed_crystal["positions"].clone().requires_grad_(True)
    function = lambda value: _phase_from_positions(value, typed_crystal)
    assert torch.autograd.gradcheck(
        function, (positions,), eps=1.0e-6, atol=2.0e-5, rtol=2.0e-4
    )
    assert torch.autograd.gradgradcheck(
        function, (positions,), eps=1.0e-6, atol=4.0e-5, rtol=4.0e-4
    )


def test_stable_evaluation_branch_preserves_selected_gradient(typed_crystal):
    positions_train = typed_crystal["positions"].clone().requires_grad_(True)
    _, _, cross_train = reciprocal_fields(typed_crystal, positions=positions_train)
    initial_train = primary_phase_initialization(
        cross_train[:3], typed_crystal["modes"][:3]
    )
    schedules = ((0.7, 0.8, 0.9, 1.0), (2.0, 1.0, 0.5, 0.2))
    training = solve_training_phase(
        cross_train,
        typed_crystal["modes"],
        typed_crystal["mode_weights"],
        initial_train,
        *schedules,
    )
    loss_train = training.phase.square().sum()
    gradient_train = torch.autograd.grad(loss_train, positions_train)[0]

    positions_eval = typed_crystal["positions"].clone().requires_grad_(True)
    _, _, cross_eval = reciprocal_fields(typed_crystal, positions=positions_eval)
    initial_eval = primary_phase_initialization(
        cross_eval[:3], typed_crystal["modes"][:3]
    )
    stabilizer = find_typed_stabilizer(
        typed_crystal["sites"], typed_crystal["site_types"]
    )
    evaluation = solve_evaluation_phase(
        cross_eval,
        typed_crystal["modes"],
        typed_crystal["mode_weights"],
        initial_eval,
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=torch.float64,
        ),
        stabilizer,
        *schedules,
        minimum_gap=0.05,
        minimum_curvature=1.0,
        maximum_condition=1.0e5,
        maximum_gradient_norm=2.0e-4,
        minimum_cross_amplitude=1.0e-12,
    )
    loss_eval = evaluation.refined.phase.square().sum()
    gradient_eval = torch.autograd.grad(loss_eval, positions_eval)[0]
    torch.testing.assert_close(
        evaluation.refined.phase, training.phase, atol=2.0e-11, rtol=2.0e-11
    )
    torch.testing.assert_close(gradient_eval, gradient_train, atol=2.0e-9, rtol=2.0e-9)
