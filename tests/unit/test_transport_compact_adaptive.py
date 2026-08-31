from __future__ import annotations

import torch

from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    EvalOTConfig,
    TrainSinkhornConfig,
    TransportSupportConfig,
    solve_atom_vacancy_ot,
)
from refsite_mlip.transport.dual import (
    dual_objective,
    jacobian_vector_product,
    residual_vector,
    transport_plan,
)
from refsite_mlip.transport.gauge import gauge_vector, project_gauge
from refsite_mlip.transport.problem import build_ot_problem


def _fixture():
    distances = torch.tensor(
        [[0.40, 0.80], [0.70, 2.60], [2.65, 0.60]],
        dtype=torch.float64,
    )
    support = TransportSupportConfig("compact_c2", 2.5, 0.5, 0.2)
    cost = distances.square() / (2.0 * 1.5**2)
    return distances, cost, support


def _adaptive(distances, cost, support, **changes):
    values = dict(
        sinkhorn_iterations=3,
        max_newton_iterations=20,
        convergence_tolerance=1.0e-12,
        pcg_max_iterations=256,
        pcg_absolute_tolerance=1.0e-12,
        pcg_relative_tolerance=1.0e-10,
        fallback_sinkhorn_iterations=4096,
    )
    values.update(changes)
    return solve_atom_vacancy_ot(
        cost,
        0.5,
        EVAL_ADAPTIVE,
        "hybrid",
        EvalOTConfig(**values),
        support_config=support,
        atom_distances=distances,
        template_id="compact-template",
        sample_id="compact-sample",
    )


def test_compact_hybrid_matches_fixed_and_keeps_mask_exactly_zero():
    distances, cost, support = _fixture()
    fixed = solve_atom_vacancy_ot(
        cost,
        0.5,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(256),
        support_config=support,
        atom_distances=distances,
    )
    adaptive = _adaptive(distances, cost, support)
    assert adaptive.converged and not adaptive.fallback_used
    assert adaptive.newton_iterations > 0 and adaptive.cg_iterations > 0
    assert adaptive.warmup_sinkhorn_iterations == 3
    assert adaptive.fallback_sinkhorn_iterations == 0
    torch.testing.assert_close(adaptive.P, fixed.P, atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(adaptive.q, fixed.q, atol=2.0e-12, rtol=2.0e-12)
    mask = distances >= support.cutoff
    assert torch.equal(adaptive.P[mask], torch.zeros_like(adaptive.P[mask]))
    assert torch.equal(adaptive.P == 0.0, fixed.P == 0.0)
    assert adaptive.support_diagnostics.maximum_atom_matching_size == 2
    assert adaptive.support_diagnostics.total_support_feasible
    assert adaptive.support_diagnostics.effective_diagnostic_tolerance == 1.0e-12
    assert max(float(adaptive.row_residual), float(adaptive.column_residual)) <= 1.0e-12
    torch.testing.assert_close(
        adaptive.q.sum(), torch.tensor(1.0, dtype=torch.float64), atol=2e-13, rtol=0
    )


def test_compact_dual_jvp_matches_autograd_is_symmetric_psd_and_gauge_fixed():
    distances, cost, support = _fixture()
    problem = build_ot_problem(
        cost, 0.5, support_config=support, atom_distances=distances
    )
    result = _adaptive(distances, cost, support)
    dual = torch.cat((result.f, result.g)).detach().requires_grad_(True)
    size = dual.numel()
    vector = torch.linspace(-0.7, 0.9, size, dtype=torch.float64)
    gamma = transport_plan(
        problem, dual[: problem.num_sites], dual[problem.num_sites :]
    )
    automatic_gradient = torch.autograd.grad(
        dual_objective(
            problem,
            dual[: problem.num_sites],
            dual[problem.num_sites :],
        ),
        dual,
        create_graph=True,
    )[0]
    torch.testing.assert_close(
        automatic_gradient,
        residual_vector(problem, gamma),
        atol=3.0e-14,
        rtol=3.0e-14,
    )
    analytic = jacobian_vector_product(problem, gamma, vector)
    automatic = torch.autograd.functional.jvp(
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
    torch.testing.assert_close(analytic, automatic, atol=3.0e-14, rtol=3.0e-14)

    basis = torch.eye(size, dtype=torch.float64)
    hessian = torch.stack(
        [jacobian_vector_product(problem, gamma, column) for column in basis],
        dim=1,
    )
    torch.testing.assert_close(hessian, hessian.T, atol=2.0e-14, rtol=0.0)
    projected = project_gauge(vector, problem.num_sites, problem.num_columns)
    assert torch.dot(projected, hessian @ projected) >= -2.0e-14
    null = gauge_vector(problem.num_sites, problem.num_columns, vector)
    torch.testing.assert_close(hessian @ null, torch.zeros_like(null), atol=2e-14, rtol=0)
    assert gamma[1, 1] == 0.0 and gamma[2, 0] == 0.0
    assert torch.all(gamma[:, -1] > 0.0), "vacancy coupling must remain dense"


def test_compact_hybrid_fallback_uses_the_same_masked_kernel():
    distances, cost, support = _fixture()
    result = _adaptive(
        distances,
        cost,
        support,
        sinkhorn_iterations=0,
        max_newton_iterations=1,
        pcg_max_iterations=1,
        fallback_sinkhorn_iterations=4096,
    )
    assert result.fallback_used
    assert result.warmup_sinkhorn_iterations == 0
    assert result.fallback_sinkhorn_iterations > 0
    assert result.failure_reason is not None
    mask = distances >= support.cutoff
    assert torch.equal(result.P[mask], torch.zeros_like(result.P[mask]))
    assert result.support_diagnostics.kind == "compact_c2"
    assert max(float(result.row_residual), float(result.column_residual)) <= 1.0e-12


def test_compact_adaptive_selected_arithmetic_is_position_connected():
    distances, _, support = _fixture()
    live = distances.clone().requires_grad_(True)
    cost = live.square() / (2.0 * 1.5**2)
    result = _adaptive(live, cost, support)
    weights = live.new_tensor([[0.7, -0.2], [0.1, 0.4], [-0.3, 0.8]])
    value = (result.P * weights).sum() + 0.23 * result.q.square().sum()
    gradient = torch.autograd.grad(value, live)[0]
    assert torch.isfinite(gradient).all()
    step = 1.0e-6
    direction = torch.zeros_like(live)
    direction[0, 1] = step

    def evaluate(argument):
        output = _adaptive(
            argument,
            argument.square() / (2.0 * 1.5**2),
            support,
        )
        return (output.P * weights).sum() + 0.23 * output.q.square().sum()

    finite = (evaluate(live.detach() + direction) - evaluate(live.detach() - direction)) / (
        2.0 * step
    )
    torch.testing.assert_close(gradient[0, 1], finite, atol=3.0e-6, rtol=3.0e-5)


@torch.no_grad()
def test_compact_adaptive_cpu_float32_and_float64_effective_tolerance():
    for dtype, tolerance in ((torch.float32, 1.0e-6), (torch.float64, 1.0e-12)):
        distances, _, support = _fixture()
        distances = distances.to(dtype=dtype)
        result = solve_atom_vacancy_ot(
            distances.square() / (2.0 * 1.5**2),
            0.5,
            EVAL_ADAPTIVE,
            "hybrid",
            EvalOTConfig(
                sinkhorn_iterations=16,
                convergence_tolerance=tolerance,
                fallback_sinkhorn_iterations=4096,
            ),
            support_config=support,
            atom_distances=distances,
        )
        assert result.P.dtype == dtype
        assert torch.isfinite(result.gamma).all()
        assert result.support_diagnostics.effective_diagnostic_tolerance == tolerance
        assert torch.equal(
            result.P[distances >= support.cutoff],
            torch.zeros_like(result.P[distances >= support.cutoff]),
        )
