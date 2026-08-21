from __future__ import annotations

from dataclasses import replace

import torch

from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    EvalOTConfig,
    solve_atom_vacancy_ot,
)
from refsite_mlip.transport.dual import (
    gauge_fixed_operator,
    jacobi_inverse,
    jacobian_vector_product,
    marginal_residuals,
    residual_vector,
    transport_plan,
)
from refsite_mlip.transport.gauge import (
    gauge_vector,
    project_gauge,
)
from refsite_mlip.transport.krylov import projected_pcg
from refsite_mlip.transport.newton_krylov import solve_newton_krylov
from refsite_mlip.transport.problem import build_ot_problem
from refsite_mlip.transport.result import DualVariables


def _problem_and_solution():
    cost = torch.tensor(
        [[0.1, 1.1], [0.8, 0.2], [0.45, 0.63]], dtype=torch.float64
    )
    problem = build_ot_problem(cost, 0.31)
    result = solve_atom_vacancy_ot(
        cost,
        0.31,
        EVAL_ADAPTIVE,
        "sinkhorn",
        EvalOTConfig(sinkhorn_iterations=3000, convergence_tolerance=1.0e-13),
    )
    return problem, result


def test_projection_idempotence_and_orthogonality():
    problem, _ = _problem_and_solution()
    value = torch.tensor([0.3, -0.7, 1.2, 0.8, -0.4, 0.9], dtype=torch.float64)
    projected = project_gauge(value, problem.num_sites, problem.num_columns)
    twice = project_gauge(projected, problem.num_sites, problem.num_columns)
    null = gauge_vector(problem.num_sites, problem.num_columns, value)
    torch.testing.assert_close(twice, projected, atol=2.0e-15, rtol=0.0)
    torch.testing.assert_close(torch.dot(null, projected), torch.zeros((), dtype=torch.float64), atol=2.0e-15, rtol=0.0)


def test_gauge_shift_leaves_plan_invariant():
    problem, result = _problem_and_solution()
    alpha = torch.tensor(1.37, dtype=torch.float64)
    shifted = transport_plan(problem, result.f + alpha, result.g - alpha)
    torch.testing.assert_close(shifted, result.gamma, atol=2.0e-14, rtol=2.0e-14)


def test_matrix_free_jvp_matches_finite_difference_and_autograd():
    problem, result = _problem_and_solution()
    vector = torch.tensor([0.2, -0.3, 0.5, -0.7, 0.4, 0.1], dtype=torch.float64)
    analytic = jacobian_vector_product(problem, result.gamma, vector)
    h = 1.0e-6
    plus = transport_plan(
        problem,
        result.f + h * vector[: problem.num_sites],
        result.g + h * vector[problem.num_sites :],
    )
    minus = transport_plan(
        problem,
        result.f - h * vector[: problem.num_sites],
        result.g - h * vector[problem.num_sites :],
    )
    finite = (residual_vector(problem, plus) - residual_vector(problem, minus)) / (2.0 * h)
    torch.testing.assert_close(analytic, finite, atol=2.0e-9, rtol=2.0e-9)

    dual = torch.cat((result.f, result.g)).detach().requires_grad_(True)
    autograd = torch.autograd.functional.jvp(
        lambda value: residual_vector(
            problem,
            transport_plan(
                problem,
                value[: problem.num_sites],
                value[problem.num_sites :],
            ),
        ),
        dual,
        vector,
    )[1]
    torch.testing.assert_close(analytic, autograd, atol=3.0e-14, rtol=3.0e-14)


def test_jacobian_symmetry_psd_and_null_vector():
    problem, result = _problem_and_solution()
    left = torch.tensor([0.2, -0.1, 0.7, -0.3, 0.4, 0.9], dtype=torch.float64)
    right = torch.tensor([-0.8, 0.5, 0.1, 0.6, -0.2, 0.3], dtype=torch.float64)
    j_left = jacobian_vector_product(problem, result.gamma, left)
    j_right = jacobian_vector_product(problem, result.gamma, right)
    torch.testing.assert_close(torch.dot(left, j_right), torch.dot(j_left, right), atol=2.0e-14, rtol=2.0e-14)
    assert torch.dot(left, j_left) >= -2.0e-14
    null = gauge_vector(problem.num_sites, problem.num_columns, left)
    torch.testing.assert_close(
        jacobian_vector_product(problem, result.gamma, null),
        torch.zeros_like(null),
        atol=2.0e-14,
        rtol=0.0,
    )


def test_projected_pcg_matches_explicit_dense_gauge_fixed_solve():
    problem, result = _problem_and_solution()
    size = problem.num_sites + problem.num_columns
    projector = lambda value: project_gauge(value, problem.num_sites, problem.num_columns)
    operator = lambda value: gauge_fixed_operator(problem, result.gamma, value, 1.0)
    basis = torch.eye(size, dtype=torch.float64)
    dense = torch.stack([operator(column) for column in basis], dim=1)
    rhs = projector(torch.tensor([0.4, -0.7, 0.2, 0.6, -0.1, 0.3], dtype=torch.float64))
    oracle = torch.linalg.solve(dense, rhs)
    pcg = projected_pcg(
        operator,
        rhs,
        jacobi_inverse(problem, result.gamma),
        projector,
        maximum_iterations=100,
        absolute_tolerance=1.0e-13,
        relative_tolerance=1.0e-13,
    )
    assert pcg.converged
    torch.testing.assert_close(pcg.solution, oracle, atol=2.0e-12, rtol=2.0e-12)


def test_newton_entry_reduces_residual():
    problem, _ = _problem_and_solution()
    zero = DualVariables(
        torch.zeros_like(problem.row_marginal),
        torch.zeros_like(problem.column_marginal),
    )
    initial_gamma = transport_plan(problem, zero.f, zero.g)
    initial = residual_vector(problem, initial_gamma).abs().max()
    config = EvalOTConfig(
        sinkhorn_iterations=0,
        max_newton_iterations=1,
        convergence_tolerance=1.0e-15,
        pcg_relative_tolerance=1.0e-12,
    )
    outcome = solve_newton_krylov(problem, config, zero)
    final = residual_vector(
        problem, transport_plan(problem, outcome.f, outcome.g)
    ).abs().max()
    assert final < initial


def test_hybrid_matches_high_accuracy_sinkhorn():
    cost = torch.tensor(
        [[0.1, 1.1], [0.8, 0.2], [0.45, 0.63]], dtype=torch.float64
    )
    reference = solve_atom_vacancy_ot(
        cost,
        0.31,
        EVAL_ADAPTIVE,
        "sinkhorn",
        EvalOTConfig(sinkhorn_iterations=3000, convergence_tolerance=1.0e-13),
    )
    hybrid = solve_atom_vacancy_ot(
        cost,
        0.31,
        EVAL_ADAPTIVE,
        "hybrid",
        EvalOTConfig(
            sinkhorn_iterations=3,
            convergence_tolerance=1.0e-12,
            pcg_relative_tolerance=1.0e-12,
        ),
    )
    torch.testing.assert_close(hybrid.P, reference.P, atol=2.0e-11, rtol=2.0e-11)
    torch.testing.assert_close(hybrid.q, reference.q, atol=2.0e-11, rtol=2.0e-11)
    assert hybrid.newton_iterations > 0


def test_newton_krylov_accepts_explicit_initial_duals():
    from refsite_mlip.transport.sinkhorn import fixed_sinkhorn_updates

    problem, reference = _problem_and_solution()
    initial = fixed_sinkhorn_updates(problem, 2)
    result = solve_atom_vacancy_ot(
        problem.atom_cost,
        float(problem.epsilon),
        EVAL_ADAPTIVE,
        "newton_krylov",
        EvalOTConfig(
            max_newton_iterations=20,
            convergence_tolerance=1.0e-12,
            pcg_relative_tolerance=1.0e-12,
        ),
        init_duals=initial,
    )
    torch.testing.assert_close(result.P, reference.P, atol=2.0e-11, rtol=2.0e-11)
