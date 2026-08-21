from __future__ import annotations

import torch

from refsite_mlip.transport import (
    TRAIN_FIXED,
    TrainSinkhornConfig,
    atom_site_cost,
    solve_atom_vacancy_ot,
)


CONFIG = TrainSinkhornConfig(iterations=80, diagnostic_tolerance=1.0e-7)


def _solve(cost):
    result = solve_atom_vacancy_ot(
        cost, 0.42, TRAIN_FIXED, "sinkhorn", CONFIG
    )
    return result.P, result.q


def test_cost_to_plan_gradcheck_and_gradgradcheck():
    cost = torch.tensor(
        [[0.12, 0.91], [0.73, 0.18], [0.41, 0.56]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        _solve, (cost,), eps=1.0e-6, atol=2.0e-6, rtol=2.0e-5
    )
    assert torch.autograd.gradgradcheck(
        _solve, (cost,), eps=1.0e-6, atol=5.0e-6, rtol=5.0e-5
    )


def _position_to_plan(positions):
    references = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.1, 0.3]],
        dtype=torch.float64,
    )
    cell = torch.tensor(
        [[4.0, 0.1, 0.0], [0.2, 3.8, 0.1], [0.0, 0.3, 4.2]],
        dtype=torch.float64,
    )
    cost = atom_site_cost(
        positions, references, cell, (False, False, False), 0.9
    )
    return _solve(cost)


def test_position_to_plan_gradcheck_and_gradgradcheck():
    positions = torch.tensor(
        [[0.13, 0.07, -0.02], [1.08, 0.16, 0.04]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        _position_to_plan,
        (positions,),
        eps=1.0e-6,
        atol=3.0e-6,
        rtol=3.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        _position_to_plan,
        (positions,),
        eps=1.0e-6,
        atol=8.0e-6,
        rtol=8.0e-5,
    )


def test_force_create_graph_and_force_loss_parameter_backward():
    positions = torch.tensor(
        [[0.13, 0.07, -0.02], [1.08, 0.16, 0.04]],
        dtype=torch.float64,
        requires_grad=True,
    )
    references = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.1, 0.3]],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 4.0
    cost = atom_site_cost(
        positions, references, cell, (False, False, False), 0.9
    )
    result = solve_atom_vacancy_ot(
        cost, 0.42, TRAIN_FIXED, "sinkhorn", CONFIG
    )
    beta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    energy = torch.sum(result.P * cost.square()) + beta * result.q.square().sum()
    force = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    force_loss = force.square().mean()
    beta_gradient = torch.autograd.grad(force_loss, beta)[0]
    assert torch.isfinite(beta_gradient)
    assert beta_gradient.abs() > 1.0e-10


def test_fixed_training_is_deterministically_repeatable():
    cost = torch.tensor(
        [[0.12, 0.91], [0.73, 0.18], [0.41, 0.56]],
        dtype=torch.float64,
    )
    first = solve_atom_vacancy_ot(
        cost, 0.42, TRAIN_FIXED, "sinkhorn", CONFIG
    )
    second = solve_atom_vacancy_ot(
        cost, 0.42, TRAIN_FIXED, "sinkhorn", CONFIG
    )
    torch.testing.assert_close(first.gamma, second.gamma, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.f, second.f, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.g, second.g, atol=0.0, rtol=0.0)


def test_training_factory_rejects_newton_and_initial_duals():
    cost = torch.tensor([[0.1], [0.3]], dtype=torch.float64)
    import pytest
    from refsite_mlip.transport import DualVariables

    with pytest.raises(ValueError, match="only fixed-unrolled"):
        solve_atom_vacancy_ot(
            cost, 0.3, TRAIN_FIXED, "newton_krylov", CONFIG
        )
    duals = DualVariables(
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="zero dual"):
        solve_atom_vacancy_ot(
            cost,
            0.3,
            TRAIN_FIXED,
            "sinkhorn",
            CONFIG,
            init_duals=duals,
        )
