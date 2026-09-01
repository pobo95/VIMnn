from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import refsite_mlip.models.potential as potential_module
from refsite_mlip.models import PotentialConfig, ReferenceSitePotential
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TransportSupportConfig,
    TransportSupportError,
    materialize_dense_plan,
)

from test_compact_support_potential import _compact, _model, _numbers
from test_edge_list_compact_potential import _edge_clone
from test_evaluation_phase_derivatives import _accepted_positions
from test_evaluation_phase_potential import _policy
from test_runtime_template_context import make_context, make_template


@pytest.fixture(autouse=True)
def _preserve_global_rng():
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20260803)
        yield


def _case(data, atom_count=5, *, template_id="edge-adaptive", warmup=32):
    base = _model(data, _compact())
    dense = ReferenceSitePotential(
        replace(base.config, eval_sinkhorn_warmup_iterations=warmup),
        base.topology,
        base.phase_modes,
        base.phase_mode_weights,
        base.species_alignment_weights,
        base.site_alignment_weights,
        base.phase_channel_weights,
        base.atomic_baseline,
    ).to(base.atomic_baseline)
    dense.load_state_dict(base.state_dict(), strict=True)
    edge = _edge_clone(dense)
    template = make_template(data, template_id=template_id)
    context = make_context(template)
    policy = _policy(template)
    positions = _accepted_positions(data, atom_count)
    arguments = (
        positions,
        _numbers(data, atom_count),
        data["cell"],
        data["origin"],
    )
    keywords = {
        "solver_path": EVAL_ADAPTIVE,
        "template_context": context,
        "evaluation_policy": policy,
    }
    return dense, edge, context, policy, arguments, keywords


def test_direct_edge_adaptive_executes_sparse_newton_krylov(typed_crystal):
    _, edge, _, _, arguments, keywords = _case(
        typed_crystal,
        5,
        template_id="edge-adaptive-newton",
        warmup=20,
    )
    output = edge(
        *arguments,
        **keywords,
        compute_forces=True,
        return_aux=True,
    )
    diagnostics = output.auxiliary["evaluation_diagnostics"]
    assert diagnostics.transport_newton_iterations > 0
    assert diagnostics.transport_cg_iterations > 0
    assert not diagnostics.transport_fallback_used
    assert output.auxiliary["ot"].edge_plan.requires_grad
    assert torch.all(torch.isfinite(output.forces))


def _branch(output):
    diagnostics = output.auxiliary["evaluation_diagnostics"]
    return (
        diagnostics.selected_grouped_index,
        diagnostics.transport_support_fingerprint,
        diagnostics.transport_candidate_edge_count,
        diagnostics.transport_active_edge_count,
        diagnostics.transport_fallback_used,
    )


@pytest.mark.parametrize("atom_count", [6, 5], ids=["pristine", "vacancy"])
def test_direct_edge_adaptive_matches_dense_compact_and_preserves_state(
    typed_crystal, atom_count
):
    dense, edge, context, policy, arguments, keywords = _case(
        typed_crystal, atom_count, template_id=f"edge-direct-{atom_count}"
    )
    parameter_ids = tuple(id(parameter) for parameter in edge.parameters())
    state = {key: value.clone() for key, value in edge.state_dict().items()}
    cpu_rng = torch.get_rng_state().clone()
    first_parameter = next(edge.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.125)
    gradient_identity = first_parameter.grad
    gradient_value = first_parameter.grad.clone()
    edge.train()

    dense_output = dense(
        *arguments,
        **keywords,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    edge_output = edge(
        *arguments,
        **keywords,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    dense_ot = dense_output.auxiliary["ot"]
    edge_ot = edge_output.auxiliary["ot"]
    edge_plan = materialize_dense_plan(edge_ot).plan

    assert not hasattr(edge_ot, "P") and not hasattr(edge_ot, "gamma")
    assert not edge_ot.dense_plan_materialized
    assert edge_ot.edges.num_candidate_edges > edge_ot.edges.num_active_edges
    torch.testing.assert_close(edge_plan, dense_ot.P, atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(edge_ot.q, dense_ot.q, atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(
        edge_output.auxiliary["phase"],
        dense_output.auxiliary["phase"],
        atol=0.0,
        rtol=0.0,
    )
    assert (
        edge_output.auxiliary["evaluation_diagnostics"].selected_grouped_index
        == dense_output.auxiliary["evaluation_diagnostics"].selected_grouped_index
    )
    for left, right, tolerance in (
        (edge_output.raw_c, dense_output.raw_c, 4e-15),
        (edge_output.site_features, dense_output.site_features, 4e-14),
        (edge_output.site_energy, dense_output.site_energy, 4e-14),
        (edge_output.energy, dense_output.energy, 4e-14),
        (edge_output.forces, dense_output.forces, 4e-13),
        (edge_output.stress, dense_output.stress, 4e-13),
        (edge_output.stress_voigt, dense_output.stress_voigt, 4e-13),
        (
            edge_output.auxiliary["multipoles"].equivariant_features,
            dense_output.auxiliary["multipoles"].equivariant_features,
            4e-14,
        ),
    ):
        torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)

    diagnostics = edge_output.auxiliary["evaluation_diagnostics"]
    assert diagnostics.transport_backend == "edge_list"
    assert diagnostics.transport_solver_name == "edge_list_hybrid"
    assert diagnostics.transport_support_fingerprint
    assert diagnostics.transport_active_dense_ratio < 1.0
    assert diagnostics.transport_candidate_dense_ratio <= 1.0
    assert diagnostics.transport_maximum_matching_size == atom_count
    assert diagnostics.transport_total_support_feasible
    assert not diagnostics.transport_dense_plan_materialized
    assert diagnostics.differentiability_scope == "selected_branch_first_order"
    assert diagnostics.hard_branch_frozen and diagnostics.derivative_order == 1
    assert not diagnostics.transport_fallback_used
    assert edge_output["energy"] is edge_output.energy

    assert edge.training
    assert tuple(id(parameter) for parameter in edge.parameters()) == parameter_ids
    assert tuple(edge.state_dict()) == tuple(state)
    assert all(torch.equal(edge.state_dict()[key], value) for key, value in state.items())
    assert first_parameter.grad is gradient_identity
    assert torch.equal(first_parameter.grad, gradient_value)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert keywords["template_context"] is context
    assert PotentialConfig.from_dict(edge.config.to_dict()) == edge.config


def _symmetric_directions(reference):
    directions = []
    for axis in range(3):
        direction = torch.zeros((3, 3), dtype=reference.dtype, device=reference.device)
        direction[axis, axis] = 1.0
        directions.append(direction)
    for left, right in ((1, 2), (0, 2), (0, 1)):
        direction = torch.zeros((3, 3), dtype=reference.dtype, device=reference.device)
        direction[left, right] = direction[right, left] = 0.5
        directions.append(direction)
    return directions


def test_edge_adaptive_all_force_and_stress_fd_and_single_autograd_call(
    typed_crystal, monkeypatch
):
    _, model, _, _, arguments, keywords = _case(
        typed_crystal, 5, template_id="edge-adaptive-fd"
    )
    force_only = model(*arguments, **keywords, compute_forces=True, return_aux=True)
    stress_only = model(*arguments, **keywords, compute_stress=True, return_aux=True)
    original_grad = torch.autograd.grad
    calls = []

    def counted_grad(*args, **kwargs):
        calls.append(1)
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    with torch.no_grad():
        baseline = model(
            *arguments,
            **keywords,
            compute_forces=True,
            compute_stress=True,
            return_aux=True,
        )
    assert len(calls) == 1
    monkeypatch.setattr(torch.autograd, "grad", original_grad)
    energy_only = model(*arguments, **keywords, return_aux=True)
    no_aux = model(
        *arguments,
        **keywords,
        compute_forces=True,
        compute_stress=True,
        return_aux=False,
    )
    torch.testing.assert_close(force_only.forces, baseline.forces, atol=0.0, rtol=0.0)
    torch.testing.assert_close(stress_only.stress, baseline.stress, atol=0.0, rtol=0.0)
    torch.testing.assert_close(energy_only.energy, baseline.energy, atol=0.0, rtol=0.0)
    assert no_aux.auxiliary is None
    assert baseline.forces.grad_fn is None and not baseline.forces.requires_grad
    assert baseline.stress.grad_fn is None and not baseline.stress.requires_grad
    assert torch.equal(baseline.stress, baseline.stress.T)
    assert torch.equal(
        baseline.stress_voigt,
        baseline.stress[(0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)],
    )

    expected_branch = _branch(baseline)
    positions, numbers, cell, origin = arguments
    force_step = 1.0e-6
    maximum_force_error = 0.0
    maximum_force_relative = 0.0
    for atom in range(positions.shape[0]):
        for component in range(3):
            delta = torch.zeros_like(positions)
            delta[atom, component] = force_step
            plus = model(
                positions + delta, numbers, cell, origin, **keywords, return_aux=True
            )
            minus = model(
                positions - delta, numbers, cell, origin, **keywords, return_aux=True
            )
            assert _branch(plus) == expected_branch == _branch(minus)
            finite = -(plus.energy - minus.energy) / (2.0 * force_step)
            automatic = baseline.forces[atom, component]
            absolute = float((automatic - finite).abs())
            relative = absolute / max(float(automatic.abs()), float(finite.abs()), 1e-12)
            maximum_force_error = max(maximum_force_error, absolute)
            maximum_force_relative = max(maximum_force_relative, relative)
    assert maximum_force_error <= 5.0e-6
    assert maximum_force_relative <= 5.0e-4

    volume = torch.linalg.det(cell).abs()
    stress_step = 1.0e-4
    maximum_stress_error = 0.0
    maximum_stress_relative = 0.0
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
    for direction in _symmetric_directions(positions):
        plus_deformation = identity + stress_step * direction
        minus_deformation = identity - stress_step * direction
        plus = model(
            positions @ plus_deformation,
            numbers,
            cell @ plus_deformation,
            origin @ plus_deformation,
            **keywords,
            return_aux=True,
        )
        minus = model(
            positions @ minus_deformation,
            numbers,
            cell @ minus_deformation,
            origin @ minus_deformation,
            **keywords,
            return_aux=True,
        )
        assert _branch(plus) == expected_branch == _branch(minus)
        finite = (plus.energy - minus.energy) / (2.0 * stress_step * volume)
        automatic = torch.sum(baseline.stress * direction)
        absolute = float((automatic - finite).abs())
        relative = absolute / max(float(automatic.abs()), float(finite.abs()), 1e-12)
        maximum_stress_error = max(maximum_stress_error, absolute)
        maximum_stress_relative = max(maximum_stress_relative, relative)
    assert maximum_stress_error <= 5.0e-6
    assert maximum_stress_relative <= 5.0e-4


def test_edge_adaptive_structured_failures_and_sparse_fallback_contract(
    typed_crystal, monkeypatch
):
    _, model, context, policy, arguments, keywords = _case(
        typed_crystal, 5, template_id="edge-adaptive-failures"
    )
    with pytest.raises(ValueError, match="evaluation_policy"):
        model(*arguments, solver_path=EVAL_ADAPTIVE, template_context=context)
    wrong_template = make_template(typed_crystal, template_id="wrong-edge-template")
    with pytest.raises(EvaluationPhaseError) as caught:
        model(
            *arguments,
            solver_path=EVAL_ADAPTIVE,
            template_context=context,
            evaluation_policy=_policy(wrong_template),
        )
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    with pytest.raises(EvaluationPhaseError) as caught:
        model(*arguments, **keywords, compute_forces=True, create_graph=True)
    assert caught.value.reason_code == "CREATE_GRAPH_UNSUPPORTED"
    with torch.inference_mode(), pytest.raises(EvaluationPhaseError) as caught:
        model(*arguments, **keywords, compute_forces=True)
    assert caught.value.reason_code == "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED"

    original_solver = potential_module.solve_sparse_hybrid_eval

    def fallback_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        diagnostics = replace(
            result.adaptive_diagnostics,
            fallback_reason="forced fallback",
            fallback_residual=result.row_residual,
        )
        return replace(
            result,
            fallback_used=True,
            failure_reason="forced fallback",
            adaptive_diagnostics=diagnostics,
        )

    monkeypatch.setattr(potential_module, "solve_sparse_hybrid_eval", fallback_solver)
    fallback_energy = model(*arguments, **keywords, return_aux=True)
    assert fallback_energy.auxiliary["ot"].fallback_used
    with pytest.raises(EvaluationPhaseError) as caught:
        model(*arguments, **keywords, compute_forces=True)
    assert caught.value.reason_code == "DERIVATIVE_FALLBACK_UNSUPPORTED"
    message = str(caught.value)
    assert "forced fallback" in message
    assert "residual=" in message and "support_fingerprint=" in message

    def disconnected_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        return replace(result, edge_plan=result.edge_plan.detach())

    monkeypatch.setattr(potential_module, "solve_sparse_hybrid_eval", disconnected_solver)
    with pytest.raises(EvaluationPhaseError) as caught:
        model(*arguments, **keywords, compute_forces=True)
    assert caught.value.reason_code == "GRAPH_DISCONNECTED"

    def nonfinite_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        bad = result.edge_plan.clone()
        bad[0] = torch.nan
        return replace(result, edge_plan=bad)

    monkeypatch.setattr(potential_module, "solve_sparse_hybrid_eval", nonfinite_solver)
    with pytest.raises(EvaluationPhaseError) as caught:
        model(*arguments, **keywords)
    assert caught.value.reason_code == "NONFINITE_OUTPUT"

    tiny_support = TransportSupportConfig(
        "compact_c2", 0.1, 0.05, 0.01, backend="edge_list"
    )
    infeasible = ReferenceSitePotential(
        replace(model.config, transport_support=tiny_support),
        model.topology,
        model.phase_modes,
        model.phase_mode_weights,
        model.species_alignment_weights,
        model.site_alignment_weights,
        model.phase_channel_weights,
        model.atomic_baseline,
    ).to(model.atomic_baseline)
    infeasible.load_state_dict(model.state_dict(), strict=True)
    monkeypatch.setattr(potential_module, "solve_sparse_hybrid_eval", original_solver)
    with pytest.raises(TransportSupportError) as caught:
        infeasible(*arguments, **keywords)
    assert caught.value.reason_code in ("ATOM_WITHOUT_SUPPORT", "NO_TOTAL_SUPPORT")


def test_direct_edge_adaptive_never_calls_dense_transport_or_features(
    typed_crystal, monkeypatch
):
    import refsite_mlip.transport.dual as dense_dual
    import refsite_mlip.transport.edge_list as edge_module
    import refsite_mlip.transport.factory as dense_factory

    _, model, _, _, arguments, keywords = _case(
        typed_crystal, 5, template_id="edge-adaptive-no-dense"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dense transport/feature path was called")

    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", forbidden)
    monkeypatch.setattr(potential_module, "build_probability_multipoles", forbidden)
    monkeypatch.setattr(dense_factory, "build_ot_problem", forbidden)
    monkeypatch.setattr(dense_dual, "transport_plan", forbidden)
    monkeypatch.setattr(edge_module, "materialize_dense_plan", forbidden)
    for force, stress in ((False, False), (True, False), (False, True), (True, True)):
        output = model(
            *arguments,
            **keywords,
            compute_forces=force,
            compute_stress=stress,
            return_aux=True,
        )
        ot = output.auxiliary["ot"]
        assert not hasattr(ot, "P") and not ot.dense_plan_materialized
        assert torch.isfinite(output.energy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_edge_adaptive_cuda_energy_force_stress(typed_crystal, dtype):
    data = {
        key: value.to(device="cuda", dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value.to(device="cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in typed_crystal.items()
    }
    _, model, _, _, arguments, keywords = _case(
        data, 5, template_id=f"edge-adaptive-cuda-{dtype}"
    )
    output = model(
        *arguments,
        **keywords,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert output.energy.dtype == dtype and output.energy.device.type == "cuda"
    assert output.forces.dtype == dtype and output.forces.device.type == "cuda"
    assert output.stress.dtype == dtype and output.stress.device.type == "cuda"
    assert torch.isfinite(output.energy)
    assert torch.all(torch.isfinite(output.forces))
    assert torch.all(torch.isfinite(output.stress))
    assert not output.auxiliary["ot"].dense_plan_materialized
    expected = 1.0e-6 if dtype == torch.float32 else 1.0e-12
    assert (
        output.auxiliary["evaluation_diagnostics"].effective_transport_tolerance
        == expected
    )
