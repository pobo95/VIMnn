from __future__ import annotations

import math

import pytest
import torch

from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    EvalOTConfig,
    TrainSinkhornConfig,
    solve_atom_vacancy_ot,
)
from refsite_mlip.transport.diagnostics import primal_objective
from refsite_mlip.transport.marginals import species_probability_field


def _cost(M, N, dtype=torch.float64, device="cpu"):
    rows = torch.arange(M, dtype=dtype, device=device).unsqueeze(1)
    columns = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
    return 0.07 + (rows - 1.17 * columns).square() / max(M, 1)


@pytest.mark.parametrize("M,N", [(4, 4), (4, 3), (5, 2)])
def test_marginals_and_vacancy_edge_cases(M, N):
    result = solve_atom_vacancy_ot(
        _cost(M, N),
        0.35,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations=256),
    )
    K = M - N
    torch.testing.assert_close(
        result.P.sum(dim=1) + result.q,
        torch.ones(M, dtype=torch.float64),
        atol=2.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.P.sum(dim=0),
        torch.ones(N, dtype=torch.float64),
        atol=2.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.q.sum(), torch.tensor(float(K), dtype=torch.float64), atol=2.0e-12, rtol=0.0
    )
    assert torch.all(result.P >= 0)
    assert torch.all(result.q >= 0)
    assert torch.all(result.q <= 1.0 + 2.0e-12)
    if K == 0:
        assert result.gamma.shape == (M, N)
        torch.testing.assert_close(result.q, torch.zeros_like(result.q), atol=0, rtol=0)
    else:
        assert result.gamma.shape == (M, N + 1)
        assert result.q.untyped_storage().data_ptr() == result.gamma.untyped_storage().data_ptr()


def test_empty_atom_problem_is_analytic():
    result = solve_atom_vacancy_ot(
        torch.empty((4, 0), dtype=torch.float64),
        0.3,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations=32),
    )
    assert result.P.shape == (4, 0)
    torch.testing.assert_close(result.q, torch.ones(4, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(result.gamma, torch.ones((4, 1), dtype=torch.float64), atol=0, rtol=0)
    assert result.sinkhorn_iterations == 0
    assert result.solver_name == "analytic_empty_atoms"


def test_more_atoms_than_sites_fails_fast():
    with pytest.raises(ValueError, match="exceeds"):
        solve_atom_vacancy_ot(
            torch.zeros((2, 3), dtype=torch.float64),
            0.3,
            TRAIN_FIXED,
            "sinkhorn",
            TrainSinkhornConfig(),
        )


def test_species_probability_simplex_and_number_conservation():
    result = solve_atom_vacancy_ot(
        _cost(5, 3),
        0.4,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations=256),
    )
    numbers = torch.tensor([6, 41, 6], dtype=torch.long)
    species = torch.tensor([6, 41], dtype=torch.long)
    field = species_probability_field(result.P, numbers, species)
    torch.testing.assert_close(
        field.sum(dim=1) + result.q,
        torch.ones(5, dtype=torch.float64),
        atol=2.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        field.sum(dim=0),
        torch.tensor([2.0, 1.0], dtype=torch.float64),
        atol=2.0e-12,
        rtol=0.0,
    )


def _generic_balanced_log_sinkhorn(cost, rows, columns, epsilon, iterations=6000):
    f = torch.zeros_like(rows)
    g = torch.zeros_like(columns)
    for _ in range(iterations):
        f = epsilon * (
            torch.log(rows)
            - torch.logsumexp((g.unsqueeze(0) - cost) / epsilon, dim=1)
        )
        g = epsilon * (
            torch.log(columns)
            - torch.logsumexp((f.unsqueeze(1) - cost) / epsilon, dim=0)
        )
    return torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - cost) / epsilon)


@pytest.mark.parametrize("K", [2, 3])
def test_aggregate_vacancy_matches_expanded_identical_dummies(K):
    M = 5
    N = M - K
    epsilon = 0.37
    atom_cost = _cost(M, N)
    aggregate = solve_atom_vacancy_ot(
        atom_cost,
        epsilon,
        EVAL_ADAPTIVE,
        "sinkhorn",
        EvalOTConfig(sinkhorn_iterations=6000, convergence_tolerance=1.0e-13),
    )
    expanded_cost = torch.cat(
        (atom_cost, torch.zeros((M, K), dtype=torch.float64)), dim=1
    )
    expanded = _generic_balanced_log_sinkhorn(
        expanded_cost,
        torch.ones(M, dtype=torch.float64),
        torch.ones(N + K, dtype=torch.float64),
        epsilon,
    )
    torch.testing.assert_close(expanded[:, :N], aggregate.P, atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(expanded[:, N:].sum(dim=1), aggregate.q, atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(
        expanded[:, N:],
        (aggregate.q / K).unsqueeze(1).expand(M, K),
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    expanded_objective = primal_objective(expanded, expanded_cost, epsilon)
    aggregate_objective = primal_objective(aggregate.gamma, torch.cat(
        (atom_cost, torch.zeros((M, 1), dtype=torch.float64)), dim=1
    ), epsilon)
    expected = aggregate_objective - epsilon * K * math.log(K)
    torch.testing.assert_close(expanded_objective, expected, atol=3.0e-12, rtol=2.0e-12)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_fixed_epsilon_fails_fast(epsilon):
    with pytest.raises(ValueError, match="epsilon_ot"):
        solve_atom_vacancy_ot(
            _cost(3, 2),
            epsilon,
            TRAIN_FIXED,
            "sinkhorn",
            TrainSinkhornConfig(),
        )
