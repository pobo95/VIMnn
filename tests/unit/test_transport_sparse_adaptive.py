from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    DualVariables,
    EvalOTConfig,
    SparseAdaptiveTransportError,
    TrainSinkhornConfig,
    TransportSupportConfig,
    TransportSupportError,
    atom_site_displacements,
    build_compact_transport_edges,
    materialize_dense_plan,
    solve_atom_vacancy_ot,
    solve_sparse_hybrid_eval,
    solve_sparse_newton_krylov,
    solve_sparse_sinkhorn_train_fixed,
    sparse_dual_objective,
    sparse_fixed_sinkhorn_updates,
    sparse_gauge_fixed_operator,
    sparse_jacobi_inverse,
    sparse_jacobian_vector_product,
    sparse_projected_pcg,
    sparse_residual_vector,
    sparse_transport_plan,
)
from refsite_mlip.transport.dual import (
    dual_objective,
    gauge_fixed_operator,
    jacobi_inverse,
    jacobian_vector_product,
    residual_vector,
    transport_plan,
)
from refsite_mlip.transport.gauge import gauge_vector, project_gauge
from refsite_mlip.transport.krylov import projected_pcg
from refsite_mlip.transport.problem import build_ot_problem


def _support(backend: str) -> TransportSupportConfig:
    return TransportSupportConfig(
        "compact_c2", 2.5, 0.5, 0.2, backend=backend
    )


def _config(**changes) -> EvalOTConfig:
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
    return EvalOTConfig(**values)


def _one_vacancy_distances(dtype=torch.float64, device="cpu") -> torch.Tensor:
    return torch.tensor(
        [[0.40, 0.80], [0.70, 2.60], [2.65, 0.60]],
        dtype=dtype,
        device=device,
    )


def _edges(distances: torch.Tensor):
    displacements = torch.stack(
        (distances, torch.zeros_like(distances), torch.zeros_like(distances)),
        dim=-1,
    )
    return build_compact_transport_edges(
        displacements,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_support("edge_list"),
        template_id="sparse-adaptive-fixture",
        sample_id="sample-0",
    )


def _dense_problem(distances: torch.Tensor):
    cost = distances.square() / (2.0 * 1.5**2)
    return build_ot_problem(
        cost,
        0.5,
        support_config=_support("dense"),
        atom_distances=distances,
    )


def _dense_hybrid(distances: torch.Tensor, config: EvalOTConfig):
    return solve_atom_vacancy_ot(
        distances.square() / (2.0 * 1.5**2),
        0.5,
        EVAL_ADAPTIVE,
        "hybrid",
        config,
        support_config=_support("dense"),
        atom_distances=distances,
    )


def _objective_roundoff_fixture():
    # The two final entries differ from the integration geometry by only a
    # handful of binary64 ULPs.  The full Newton step is converged, while its
    # independently reduced objective rounds one ULP above the current value.
    cost = torch.tensor(
        [
            [0.012091520574566365, 0.7848798290018676, 1.214882844255246,
             1.3106785466903754, 0.4691935469130355],
            [0.5688401746050944, 0.014368208552289072, 0.49366771412150584,
             1.1557863959948065, 0.43457966357254146],
            [0.8999197912427667, 0.5349493270400782, 0.01328062291002436,
             1.3342427470886042, 1.6522795897557845],
            [1.1960797643540755, 1.4388525269574766, 0.9561440724622954,
             0.013565543941512536, 0.6822482212144998],
            [0.43502876229173043, 0.6208988299628464, 1.1771523918796816,
             0.7450037551396463, 0.012328841244243528],
            [0.583267730836448, 0.8861251606879396, 0.8774240763459057,
             0.6736118787571361, 1.5076794123617066],
        ],
        dtype=torch.float64,
    )
    return torch.sqrt(cost * (2.0 * 1.5**2)), cost


def test_sparse_hybrid_one_vacancy_matches_dense_and_fixed_oracles():
    distances = _one_vacancy_distances()
    edges = _edges(distances)
    config = _config()
    sparse = solve_sparse_hybrid_eval(edges, config)
    dense = _dense_hybrid(distances, config)
    fixed = solve_sparse_sinkhorn_train_fixed(edges, TrainSinkhornConfig(256))
    plan = materialize_dense_plan(sparse).plan
    fixed_plan = materialize_dense_plan(fixed).plan

    assert sparse.converged and not sparse.fallback_used
    assert sparse.solver_name == "edge_list_hybrid"
    assert sparse.path_name == EVAL_ADAPTIVE
    assert sparse.warmup_sinkhorn_iterations == 3
    assert sparse.newton_iterations == dense.newton_iterations == 3
    assert sparse.cg_iterations == dense.cg_iterations == 12
    assert not sparse.dense_plan_materialized
    assert not sparse.adaptive_diagnostics.dense_plan_materialized
    assert len(sparse.adaptive_diagnostics.support_fingerprint) == 64
    torch.testing.assert_close(plan, dense.P, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(sparse.q, dense.q, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(plan, fixed_plan, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(sparse.q, fixed.q, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        sparse_dual_objective(edges, sparse.f, sparse.g),
        dual_objective(_dense_problem(distances), dense.f, dense.g),
        atol=2e-14,
        rtol=2e-14,
    )
    assert torch.equal(sparse.edge_plan[~edges.active], torch.zeros_like(sparse.edge_plan[~edges.active]))
    assert max(float(sparse.row_residual), float(sparse.column_residual)) <= 1e-12
    assert float(sparse.vacancy_residual.abs()) <= 1e-12
    assert float((sparse.q.sum() - 1.0).abs()) <= 1e-12

    repeated = solve_sparse_hybrid_eval(_edges(distances), config)
    for name in ("edge_plan", "q", "f", "g", "row_residual", "column_residual"):
        assert torch.equal(getattr(sparse, name), getattr(repeated, name))


def test_converged_newton_step_below_objective_resolution_avoids_fallback():
    distances, cost = _objective_roundoff_fixture()
    config = _config(sinkhorn_iterations=16)
    dense_support = TransportSupportConfig("compact_c2", 2.6, 0.5, 0.2)
    edge_support = TransportSupportConfig(
        "compact_c2", 2.6, 0.5, 0.2, backend="edge_list"
    )
    dense = solve_atom_vacancy_ot(
        cost,
        0.5,
        EVAL_ADAPTIVE,
        "hybrid",
        config,
        support_config=dense_support,
        atom_distances=distances,
    )
    displacements = torch.stack(
        (distances, torch.zeros_like(distances), torch.zeros_like(distances)),
        dim=-1,
    )
    edges = build_compact_transport_edges(
        displacements,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=edge_support,
    )
    sparse = solve_sparse_hybrid_eval(edges, config)

    for result in (dense, sparse):
        assert result.converged and not result.fallback_used
        assert result.newton_iterations == 1
        assert result.line_search_reductions == 0
        residual = max(float(result.row_residual), float(result.column_residual))
        assert residual <= 1e-12
    step = sparse.adaptive_diagnostics.line_search_steps[0]
    assert step.accepted_damping == 1.0 and step.failure_reason is None
    assert step.objective_after > step.objective_before
    roundoff = torch.finfo(torch.float64).eps * (
        step.objective_before.abs() + step.objective_after.abs() + 1.0
    )
    assert step.objective_after - step.objective_before <= roundoff

    sparse_plan = materialize_dense_plan(sparse).plan
    torch.testing.assert_close(sparse_plan, dense.P, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(sparse.q, dense.q, atol=2e-14, rtol=2e-14)
    dense_fixed = solve_atom_vacancy_ot(
        cost,
        0.5,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(512),
        support_config=dense_support,
        atom_distances=distances,
    )
    sparse_fixed = solve_sparse_sinkhorn_train_fixed(
        edges, TrainSinkhornConfig(512)
    )
    torch.testing.assert_close(dense.P, dense_fixed.P, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(dense.q, dense_fixed.q, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        sparse_plan,
        materialize_dense_plan(sparse_fixed).plan,
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(sparse.q, sparse_fixed.q, atol=2e-12, rtol=2e-12)


def test_sparse_objective_gradient_hvp_gauge_psd_and_newton_direction_oracle():
    distances = _one_vacancy_distances()
    edges = _edges(distances)
    problem = _dense_problem(distances)
    result = solve_sparse_hybrid_eval(edges, _config())
    dual = torch.cat((result.f, result.g)).detach().requires_grad_(True)
    rows = edges.num_sites
    edge_plan, q = sparse_transport_plan(edges, dual[:rows], dual[rows:])
    automatic_gradient = torch.autograd.grad(
        sparse_dual_objective(edges, dual[:rows], dual[rows:]),
        dual,
        create_graph=True,
    )[0]
    analytic_gradient = sparse_residual_vector(edges, edge_plan, q)
    torch.testing.assert_close(
        automatic_gradient, analytic_gradient, atol=3e-14, rtol=3e-14
    )

    vector = torch.linspace(-0.7, 0.9, dual.numel(), dtype=torch.float64)
    sparse_hvp = sparse_jacobian_vector_product(
        edges, edge_plan, q, vector
    )
    automatic_hvp = torch.autograd.functional.jvp(
        lambda value: sparse_residual_vector(
            edges,
            *sparse_transport_plan(edges, value[:rows], value[rows:]),
        ),
        dual,
        vector,
    )[1]
    dense_gamma = transport_plan(problem, dual[:rows], dual[rows:])
    dense_hvp = jacobian_vector_product(problem, dense_gamma, vector)
    torch.testing.assert_close(sparse_hvp, automatic_hvp, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(sparse_hvp, dense_hvp, atol=3e-14, rtol=3e-14)

    basis = torch.eye(dual.numel(), dtype=torch.float64)
    hessian = torch.stack(
        [
            sparse_jacobian_vector_product(edges, edge_plan, q, column)
            for column in basis
        ],
        dim=1,
    )
    torch.testing.assert_close(hessian, hessian.T, atol=2e-14, rtol=0)
    null = gauge_vector(rows, result.g.numel(), vector)
    torch.testing.assert_close(hessian @ null, torch.zeros_like(null), atol=2e-14, rtol=0)
    projected = project_gauge(vector, rows, result.g.numel())
    assert torch.dot(projected, hessian @ projected) >= -2e-14

    warm = sparse_fixed_sinkhorn_updates(edges, 3)
    warm_edge_plan, warm_q = sparse_transport_plan(edges, warm.f, warm.g)
    sparse_residual = sparse_residual_vector(edges, warm_edge_plan, warm_q)
    sparse_rhs = -project_gauge(sparse_residual, rows, warm.g.numel())
    sparse_pcg = sparse_projected_pcg(
        edges,
        warm_edge_plan,
        warm_q,
        sparse_rhs,
        sparse_jacobi_inverse(edges, warm_edge_plan, warm_q),
        gauge_rho=1.0,
        maximum_iterations=256,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )
    dense_warm_gamma = transport_plan(problem, warm.f, warm.g)
    dense_rhs = -project_gauge(
        residual_vector(problem, dense_warm_gamma), rows, warm.g.numel()
    )
    dense_pcg = projected_pcg(
        lambda value: gauge_fixed_operator(problem, dense_warm_gamma, value, 1.0),
        dense_rhs,
        jacobi_inverse(problem, dense_warm_gamma),
        lambda value: project_gauge(value, rows, warm.g.numel()),
        maximum_iterations=256,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
    )
    assert sparse_pcg.diagnostics.converged and dense_pcg.converged
    torch.testing.assert_close(
        sparse_pcg.solution, dense_pcg.solution, atol=2e-13, rtol=2e-13
    )
    sparse_directional = torch.dot(sparse_residual, sparse_pcg.solution)
    dense_directional = torch.dot(
        residual_vector(problem, dense_warm_gamma), dense_pcg.solution
    )
    torch.testing.assert_close(sparse_directional, dense_directional, atol=2e-14, rtol=2e-14)
    assert sparse_directional < 0.0


@pytest.mark.parametrize(
    "distances, vacancy_mass, oracle_tolerance",
    [
        (
            torch.tensor(
                [[0.2, 1.0, 1.3], [1.1, 0.3, 0.9], [0.8, 1.2, 0.4]],
                dtype=torch.float64,
            ),
            0,
            8e-14,
        ),
        (
            torch.tensor(
                [[0.2, 0.8], [0.7, 0.4], [1.1, 0.6], [0.9, 1.2]],
                dtype=torch.float64,
            ),
            2,
            2e-14,
        ),
    ],
)
def test_sparse_adaptive_pristine_and_multiple_vacancy(
    distances, vacancy_mass, oracle_tolerance
):
    sparse = solve_sparse_hybrid_eval(_edges(distances), _config())
    dense = _dense_hybrid(distances, _config())
    torch.testing.assert_close(
        materialize_dense_plan(sparse).plan,
        dense.P,
        atol=oracle_tolerance,
        rtol=oracle_tolerance,
    )
    torch.testing.assert_close(
        sparse.q, dense.q, atol=oracle_tolerance, rtol=oracle_tolerance
    )
    torch.testing.assert_close(
        sparse.q.sum(),
        sparse.q.new_tensor(float(vacancy_mass)),
        atol=1e-12,
        rtol=0,
    )
    assert sparse.g.numel() == distances.shape[1] + int(vacancy_mass > 0)
    if vacancy_mass == 0:
        assert torch.equal(sparse.q, torch.zeros_like(sparse.q))
        assert sparse.vacancy_residual == 0.0


def test_sparse_edge_preflight_retains_matching_and_total_support_failures():
    atomless = torch.tensor([[0.2, 3.0], [0.3, 3.1]], dtype=torch.float64)
    with pytest.raises(TransportSupportError) as failure:
        _edges(atomless)
    assert failure.value.reason_code == "ATOM_WITHOUT_SUPPORT"

    incomplete = torch.tensor(
        [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
        dtype=torch.float64,
    )
    with pytest.raises(TransportSupportError) as failure:
        _edges(incomplete)
    assert failure.value.reason_code == "INCOMPLETE_ATOM_MATCHING"

    no_total_support = torch.tensor(
        [[0.2, 3.0], [3.0, 0.2], [3.0, 0.3]], dtype=torch.float64
    )
    with pytest.raises(TransportSupportError) as failure:
        _edges(no_total_support)
    assert failure.value.reason_code == "NO_TOTAL_SUPPORT"


@pytest.mark.parametrize(
    "diagnostic_changes, reason_code",
    [
        ({"atom_active_degrees": (2, 0)}, "ATOM_WITHOUT_SUPPORT"),
        ({"maximum_atom_matching_size": 1}, "INCOMPLETE_ATOM_MATCHING"),
        ({"total_support_feasible": False}, "NO_TOTAL_SUPPORT"),
    ],
)
def test_sparse_solver_rechecks_support_certificate(
    diagnostic_changes, reason_code
):
    edges = _edges(_one_vacancy_distances())
    corrupted = replace(
        edges,
        support_diagnostics=replace(
            edges.support_diagnostics, **diagnostic_changes
        ),
    )
    with pytest.raises(SparseAdaptiveTransportError) as failure:
        solve_sparse_hybrid_eval(corrupted, _config())
    assert failure.value.reason_code == reason_code
    assert failure.value.stage == "support_preflight"


def test_sparse_newton_armijo_fallback_and_structured_fallback_failure():
    edges = _edges(_one_vacancy_distances())
    converged = solve_sparse_newton_krylov(
        edges,
        _config(),
        sparse_fixed_sinkhorn_updates(edges, 3),
    )
    assert converged.converged and converged.iterations == 3
    assert converged.pcg_steps and all(step.converged for step in converged.pcg_steps)
    assert all(step.preconditioner_min > 0.0 for step in converged.pcg_steps)

    reduced = solve_sparse_hybrid_eval(
        edges,
        _config(
            sinkhorn_iterations=0,
            max_newton_iterations=80,
            armijo_coefficient=0.51,
        ),
    )
    assert not reduced.fallback_used and reduced.line_search_reductions > 0
    reduced_steps = reduced.adaptive_diagnostics.line_search_steps
    assert any(step.accepted_damping < 1.0 for step in reduced_steps)
    assert all(
        step.objective_after <= step.objective_before
        for step in reduced_steps
        if step.objective_after is not None
    )

    armijo_fallback = solve_sparse_hybrid_eval(
        edges,
        _config(
            sinkhorn_iterations=0,
            armijo_coefficient=0.99,
            max_line_search_reductions=0,
        ),
    )
    assert armijo_fallback.fallback_used
    assert armijo_fallback.failure_reason == "ARMIJO_FAILURE"
    assert armijo_fallback.adaptive_diagnostics.line_search_steps[0].failure_reason == "ARMIJO_FAILURE"

    pcg_fallback_config = _config(
        sinkhorn_iterations=0,
        max_newton_iterations=1,
        pcg_max_iterations=1,
    )
    sparse_fallback = solve_sparse_hybrid_eval(edges, pcg_fallback_config)
    dense_fallback = _dense_hybrid(_one_vacancy_distances(), pcg_fallback_config)
    assert sparse_fallback.fallback_used
    assert sparse_fallback.failure_reason.startswith("PCG_BREAKDOWN")
    assert sparse_fallback.fallback_sinkhorn_iterations > 0
    torch.testing.assert_close(
        materialize_dense_plan(sparse_fallback).plan,
        dense_fallback.P,
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        sparse_fallback.q, dense_fallback.q, atol=2e-12, rtol=2e-12
    )
    assert torch.equal(
        sparse_fallback.edge_plan[~edges.active],
        torch.zeros_like(sparse_fallback.edge_plan[~edges.active]),
    )

    maximum = solve_sparse_newton_krylov(
        edges,
        _config(sinkhorn_iterations=0, max_newton_iterations=1),
        DualVariables(
            torch.zeros(edges.num_sites, dtype=torch.float64),
            torch.zeros(edges.num_atoms + 1, dtype=torch.float64),
        ),
    )
    assert not maximum.converged
    assert maximum.failure_reason == "MAXIMUM_NEWTON_ITERATIONS"

    with pytest.raises(SparseAdaptiveTransportError) as failure:
        solve_sparse_hybrid_eval(
            edges,
            _config(
                sinkhorn_iterations=0,
                max_newton_iterations=1,
                pcg_max_iterations=1,
                fallback_sinkhorn_iterations=1,
            ),
        )
    assert failure.value.reason_code == "FALLBACK_CONVERGENCE_FAILURE"
    assert failure.value.stage == "fallback"


def _geometric_observable(
    positions: torch.Tensor,
    references: torch.Tensor,
    cell: torch.Tensor,
):
    displacements = atom_site_displacements(
        positions, references, cell, (True, True, True)
    )
    edges = build_compact_transport_edges(
        displacements,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_support("edge_list"),
    )
    result = solve_sparse_hybrid_eval(edges, _config())
    weights = torch.linspace(
        -0.3,
        0.7,
        result.edge_plan.numel(),
        dtype=positions.dtype,
        device=positions.device,
    )
    value = (result.edge_plan * weights).sum() + 0.23 * result.q.square().sum()
    return value, result


def test_sparse_adaptive_selected_branch_first_derivative_and_translation():
    positions = torch.tensor(
        [[0.4, 0.1, 0.0], [1.1, 0.5, 0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    references = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.2, 0.1], [0.2, 1.2, 0.3]],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 6.0
    value, result = _geometric_observable(positions, references, cell)
    gradient = torch.autograd.grad(value, positions)[0]
    assert not result.fallback_used
    assert result.edge_plan.requires_grad and result.q.requires_grad
    assert torch.isfinite(gradient).all()

    step = 1e-6
    delta = torch.zeros_like(positions)
    delta[0, 1] = step
    plus = _geometric_observable(positions.detach() + delta, references, cell)[0]
    minus = _geometric_observable(positions.detach() - delta, references, cell)[0]
    finite = (plus - minus) / (2.0 * step)
    torch.testing.assert_close(gradient[0, 1], finite, atol=3e-7, rtol=3e-6)

    shift = torch.tensor([0.7, -0.3, 0.9], dtype=torch.float64)
    translated, translated_result = _geometric_observable(
        positions.detach() + shift, references + shift, cell
    )
    torch.testing.assert_close(translated, value.detach(), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(
        translated_result.edge_plan, result.edge_plan.detach(), atol=2e-14, rtol=2e-14
    )


def _walk_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _walk_tensors(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensors(item)


@pytest.mark.parametrize(
    "dtype, tolerance",
    [(torch.float32, 1e-6), (torch.float64, 1e-12)],
)
def test_sparse_adaptive_dtype_and_solver_core_non_densification(
    dtype, tolerance, monkeypatch
):
    import refsite_mlip.transport.dual as dense_dual
    import refsite_mlip.transport.edge_list as edge_module
    import refsite_mlip.transport.problem as dense_problem

    edges = _edges(_one_vacancy_distances(dtype=dtype))

    def forbidden(*args, **kwargs):
        raise AssertionError("dense transport path was called")

    monkeypatch.setattr(dense_problem, "build_ot_problem", forbidden)
    monkeypatch.setattr(dense_dual, "transport_plan", forbidden)
    monkeypatch.setattr(edge_module, "materialize_dense_plan", forbidden)
    result = solve_sparse_hybrid_eval(
        edges,
        _config(
            sinkhorn_iterations=16,
            convergence_tolerance=tolerance,
            pcg_absolute_tolerance=tolerance * 0.1,
            pcg_relative_tolerance=tolerance * 0.1,
        ),
    )
    assert result.edge_plan.dtype == dtype and result.q.dtype == dtype
    assert result.edge_plan.device == edges.distances.device
    assert torch.isfinite(result.edge_plan).all() and torch.isfinite(result.q).all()
    assert not result.dense_plan_materialized
    assert result.adaptive_diagnostics.dense_plan_materialized is False
    forbidden_shapes = {
        (edges.num_sites, edges.num_atoms),
        (
            edges.num_sites + edges.num_atoms + 1,
            edges.num_sites + edges.num_atoms + 1,
        ),
    }
    assert all(tuple(tensor.shape) not in forbidden_shapes for tensor in _walk_tensors(result))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "dtype, tolerance",
    [(torch.float32, 1e-6), (torch.float64, 1e-12)],
)
def test_sparse_adaptive_cuda(dtype, tolerance):
    edges = _edges(_one_vacancy_distances(dtype=dtype, device="cuda"))
    result = solve_sparse_hybrid_eval(
        edges,
        _config(
            sinkhorn_iterations=16,
            convergence_tolerance=tolerance,
            pcg_absolute_tolerance=tolerance * 0.1,
            pcg_relative_tolerance=tolerance * 0.1,
        ),
    )
    assert result.edge_plan.device.type == "cuda" and result.edge_plan.dtype == dtype
    assert result.q.device.type == "cuda" and result.q.dtype == dtype
    assert torch.isfinite(result.edge_plan).all() and torch.isfinite(result.q).all()
    assert max(float(result.row_residual), float(result.column_residual)) <= tolerance
    assert not result.dense_plan_materialized
