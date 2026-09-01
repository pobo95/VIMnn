from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import refsite_mlip.models.potential as potential_module
from refsite_mlip.models import ReferenceSitePotential, evaluate_structure_batch
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import EVAL_ADAPTIVE, materialize_dense_plan

from test_compact_support_potential import _compact
from test_edge_list_compact_potential import _edge_clone
from test_grouped_evaluation_phase_batch import _adaptive_case, _individual


@pytest.fixture(autouse=True)
def _preserve_global_rng():
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20260803)
        yield


def _configured(model, *, support):
    configured = ReferenceSitePotential(
        replace(
            model.config,
            transport_support=support,
            eval_sinkhorn_warmup_iterations=20,
        ),
        model.topology,
        model.phase_modes,
        model.phase_mode_weights,
        model.species_alignment_weights,
        model.site_alignment_weights,
        model.phase_channel_weights,
        model.atomic_baseline,
    ).to(model.atomic_baseline)
    configured.load_state_dict(model.state_dict(), strict=True)
    return configured


def _case(data, *, dtype=torch.float64, device="cpu"):
    values = _adaptive_case(data, dtype=dtype, device=device)
    _, model, registry, samples, batch, contexts, policies = values
    dense = _configured(model, support=_compact())
    edge = _edge_clone(dense)
    return dense, edge, registry, samples, batch, contexts, policies


def _assert_grouped_close(grouped, individual, *, tolerance):
    torch.testing.assert_close(
        grouped.energy,
        torch.stack([item.energy for item in individual]),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        grouped.site_energy,
        torch.cat([item.site_energy for item in individual]),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        grouped.forces,
        torch.cat([item.forces for item in individual]),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        grouped.stress,
        torch.stack([item.stress for item in individual]),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        grouped.stress_voigt,
        torch.stack([item.stress_voigt for item in individual]),
        atol=tolerance,
        rtol=tolerance,
    )


def test_grouped_edge_adaptive_individual_dense_parity_and_order(
    typed_crystal, monkeypatch
):
    dense, edge, _, _, batch, contexts, policies = _case(typed_crystal)
    batch.positions.requires_grad_(True)
    execution_order = []
    original_forward = edge.forward

    def counted_forward(*args, **kwargs):
        execution_order.append(kwargs["template_context"].template_id)
        return original_forward(*args, **kwargs)

    original_grad = torch.autograd.grad
    gradient_calls = []

    def counted_grad(*args, **kwargs):
        gradient_calls.append(1)
        return original_grad(*args, **kwargs)

    edge.forward = counted_forward
    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    grouped = evaluate_structure_batch(
        edge,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    edge.forward = original_forward
    monkeypatch.setattr(torch.autograd, "grad", original_grad)
    individual = _individual(
        edge,
        batch,
        contexts,
        policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    dense_grouped = evaluate_structure_batch(
        dense,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )

    assert execution_order == ["alpha", "zeta", "zeta"]
    assert len(gradient_calls) == batch.num_structures
    assert grouped.sample_ids == batch.sample_ids
    assert grouped.template_ids == ("zeta", "alpha", "zeta")
    assert grouped.site_ptr.tolist() == [0, 6, 10, 16]
    assert grouped.site_batch.tolist() == [0] * 6 + [1] * 4 + [2] * 6
    _assert_grouped_close(grouped, individual, tolerance=4.0e-13)
    grouped_diagnostics = []
    for sparse_aux, dense_aux in zip(grouped.auxiliary, dense_grouped.auxiliary):
        sparse_ot = sparse_aux["ot"]
        dense_ot = dense_aux["ot"]
        torch.testing.assert_close(
            materialize_dense_plan(sparse_ot).plan,
            dense_ot.P,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        torch.testing.assert_close(
            sparse_ot.q, dense_ot.q, atol=1.0e-12, rtol=1.0e-12
        )
        torch.testing.assert_close(
            sparse_aux["multipoles"].equivariant_features,
            dense_aux["multipoles"].equivariant_features,
            atol=1.0e-12,
            rtol=1.0e-11,
        )
        sparse_diagnostics = sparse_aux["evaluation_diagnostics"]
        grouped_diagnostics.append(sparse_diagnostics)
        dense_diagnostics = dense_aux["evaluation_diagnostics"]
        assert sparse_diagnostics.transport_backend == "edge_list"
        assert not sparse_diagnostics.transport_dense_plan_materialized
        assert sparse_diagnostics.transport_support_fingerprint
        assert not sparse_diagnostics.transport_fallback_used
        assert (
            sparse_diagnostics.selected_grouped_index
            == dense_diagnostics.selected_grouped_index
        )
        assert torch.equal(sparse_aux["phase"], dense_aux["phase"])
    assert any(item.transport_newton_iterations > 0 for item in grouped_diagnostics)
    assert any(item.transport_cg_iterations > 0 for item in grouped_diagnostics)
    torch.testing.assert_close(grouped.energy, dense_grouped.energy, atol=5e-14, rtol=5e-14)
    torch.testing.assert_close(
        grouped.forces, dense_grouped.forces, atol=2e-11, rtol=1e-8
    )
    torch.testing.assert_close(
        grouped.stress, dense_grouped.stress, atol=2e-11, rtol=1e-8
    )

    no_aux = evaluate_structure_batch(
        edge,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
    )
    assert no_aux.auxiliary is None


def test_grouped_edge_adaptive_preflight_and_contextual_fallback(
    typed_crystal, monkeypatch
):
    _, edge, _, _, batch, contexts, policies = _case(typed_crystal)
    calls = []
    original_forward = edge.forward

    def counted_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(edge, "forward", counted_forward)
    with pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            edge,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies={"zeta": policies["zeta"]},
        )
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    assert calls == []

    batch.positions.requires_grad_(True)
    with torch.inference_mode(), pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            edge,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
            compute_forces=True,
        )
    assert caught.value.reason_code == "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED"
    assert calls == []

    monkeypatch.setattr(edge, "forward", original_forward)
    original_solver = potential_module.solve_sparse_hybrid_eval

    def forced_fallback(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        diagnostics = replace(
            result.adaptive_diagnostics,
            fallback_reason="grouped forced fallback",
            fallback_residual=result.row_residual,
        )
        return replace(
            result,
            fallback_used=True,
            failure_reason="grouped forced fallback",
            adaptive_diagnostics=diagnostics,
        )

    monkeypatch.setattr(
        potential_module, "solve_sparse_hybrid_eval", forced_fallback
    )
    energy_only = evaluate_structure_batch(
        edge,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        return_aux=True,
    )
    assert all(item["ot"].fallback_used for item in energy_only.auxiliary)
    with pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            edge,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
            compute_forces=True,
        )
    assert caught.value.reason_code == "DERIVATIVE_FALLBACK_UNSUPPORTED"
    message = str(caught.value)
    assert "structure_index=1" in message
    assert "sample_id='alpha-pristine'" in message
    assert "template_id='alpha'" in message
    assert "backend='edge_list'" in message
    assert "grouped forced fallback" in message
    assert "support_fingerprint=" in message


def test_grouped_edge_adaptive_never_densifies(typed_crystal, monkeypatch):
    import refsite_mlip.transport.dual as dense_dual
    import refsite_mlip.transport.edge_list as edge_module
    import refsite_mlip.transport.factory as dense_factory

    _, edge, _, _, batch, contexts, policies = _case(typed_crystal)
    batch.positions.requires_grad_(True)

    def forbidden(*args, **kwargs):
        raise AssertionError("grouped sparse adaptive path invoked dense arithmetic")

    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", forbidden)
    monkeypatch.setattr(potential_module, "build_probability_multipoles", forbidden)
    monkeypatch.setattr(dense_factory, "build_ot_problem", forbidden)
    monkeypatch.setattr(dense_dual, "transport_plan", forbidden)
    monkeypatch.setattr(edge_module, "materialize_dense_plan", forbidden)
    output = evaluate_structure_batch(
        edge,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert torch.all(torch.isfinite(output.energy))
    assert torch.all(torch.isfinite(output.forces))
    assert torch.all(torch.isfinite(output.stress))
    assert all(not item["ot"].dense_plan_materialized for item in output.auxiliary)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_edge_adaptive_cuda(typed_crystal, dtype):
    _, edge, _, _, batch, contexts, policies = _case(
        typed_crystal, dtype=dtype, device="cuda"
    )
    batch.positions.requires_grad_(True)
    output = evaluate_structure_batch(
        edge,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    for tensor in (output.energy, output.forces, output.stress):
        assert tensor.dtype == dtype and tensor.device.type == "cuda"
        assert torch.all(torch.isfinite(tensor))
    assert all(not item["ot"].dense_plan_materialized for item in output.auxiliary)
