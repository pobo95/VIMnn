from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import refsite_mlip.models.potential as potential_module
import refsite_mlip.transport.candidate_state as candidate_state_module
from refsite_mlip.data import collate_structure_samples
from refsite_mlip.models import evaluate_structure_batch
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import EVAL_ADAPTIVE, TransportSupportError

from test_blocked_candidate_potential import (
    _blocked_clone,
    _configured,
    _grouped_case,
)
from test_compact_support_potential import _model, _numbers
from test_edge_list_adaptive_potential import _case as _adaptive_case
from test_edge_list_compact_potential import _edge_support


def _fixed_case(data):
    model = _blocked_clone(_model(data, _edge_support()), 2, 2)
    return model, (
        data["positions"][:5],
        _numbers(data),
        data["cell"],
        data["origin"],
    )


def _assert_output_close(left, right, *, derivative=False):
    names = ["energy", "site_energy", "raw_c", "site_features"]
    if derivative:
        names += ["forces", "stress", "stress_voigt"]
    tolerance = 2.0e-11 if derivative else 3.0e-13
    for name in names:
        torch.testing.assert_close(
            getattr(left, name),
            getattr(right, name),
            atol=tolerance,
            rtol=tolerance,
        )


def test_direct_fixed_candidate_state_sequence_autograd_and_stateless_contract(
    typed_crystal,
):
    model, arguments = _fixed_case(typed_crystal)
    stateless = model(*arguments, return_aux=True)
    assert stateless.candidate_neighbor_state is None
    assert stateless.candidate_reuse_decision is None
    assert "candidate_neighbor_diagnostics" not in stateless.auxiliary

    initial = model(
        *arguments,
        return_aux=True,
        return_candidate_neighbor_state=True,
    )
    state = initial.candidate_neighbor_state
    assert state is not None
    assert initial.candidate_reuse_decision.reason_code == "INITIAL_BUILD"
    assert initial.candidate_reuse_decision.rebuilt
    assert not state.build_positions.requires_grad
    assert state.build_positions.data_ptr() != arguments[0].data_ptr()
    assert initial.auxiliary["candidate_neighbor_diagnostics"][
        "state_provided"
    ] is False
    _assert_output_close(initial, stateless)

    positions = arguments[0].clone().requires_grad_(True)
    reused = model(
        positions,
        *arguments[1:],
        candidate_neighbor_state=state,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    fresh = model(
        arguments[0].clone().requires_grad_(True),
        *arguments[1:],
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    assert reused.candidate_reuse_decision.reason_code == "REUSED"
    assert reused.candidate_reuse_decision.reused
    assert reused.candidate_neighbor_state is not state
    assert reused.candidate_neighbor_state.reuse_count == 1
    _assert_output_close(reused, fresh, derivative=True)
    assert reused.forces.requires_grad and reused.stress.requires_grad
    mixed = reused.energy + 0.02 * reused.forces.square().sum()
    gradient = torch.autograd.grad(mixed, model.readout.mlp[-1].weight)[0]
    assert torch.all(torch.isfinite(gradient))
    assert torch.count_nonzero(gradient) > 0

    translation = arguments[0].new_tensor([0.19, -0.11, 0.07])
    translated = model(
        arguments[0] + translation,
        arguments[1],
        arguments[2],
        arguments[3] + translation,
        candidate_neighbor_state=state,
    )
    assert translated.candidate_reuse_decision.reused
    assert translated.candidate_reuse_decision.delta_pair_bound == pytest.approx(
        0.0, abs=3.0e-14
    )

    wrapped_positions = arguments[0].clone()
    wrapped_positions[0] += arguments[2][0]
    wrapped = model(
        wrapped_positions,
        *arguments[1:],
        candidate_neighbor_state=state,
    )
    assert wrapped.candidate_reuse_decision.reused
    _assert_output_close(wrapped, initial)

    changed_cell = arguments[2].clone()
    changed_cell[0, 0] *= 1.0001
    cell_rebuild = model(
        *arguments[:2],
        changed_cell,
        arguments[3],
        candidate_neighbor_state=state,
    )
    assert cell_rebuild.candidate_reuse_decision.reason_code == "CELL_CHANGED"
    assert cell_rebuild.candidate_reuse_decision.rebuilt

    moved_positions = arguments[0].clone()
    moved_positions[0, 0] += 0.205
    skin_rebuild = model(
        moved_positions,
        *arguments[1:],
        candidate_neighbor_state=state,
    )
    assert skin_rebuild.candidate_reuse_decision.rebuilt
    assert skin_rebuild.candidate_reuse_decision.reason_code == "SKIN_EXHAUSTED"


def test_direct_adaptive_candidate_state_branch_failure_and_single_grad(
    typed_crystal, monkeypatch
):
    _, edge, _, _, arguments, keywords = _adaptive_case(
        typed_crystal, 5, template_id="candidate-state-adaptive", warmup=20
    )
    model = _blocked_clone(edge, 2, 2)
    initial = model(
        *arguments,
        **keywords,
        return_aux=True,
        return_candidate_neighbor_state=True,
    )
    state = initial.candidate_neighbor_state
    assert state is not None
    assert initial.candidate_reuse_decision.reason_code == "INITIAL_BUILD"

    calls = []
    original_grad = torch.autograd.grad

    def counted_grad(*args, **kwargs):
        calls.append(1)
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    reused = model(
        arguments[0].clone().requires_grad_(True),
        *arguments[1:],
        **keywords,
        candidate_neighbor_state=state,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    monkeypatch.setattr(torch.autograd, "grad", original_grad)
    fresh = model(
        *arguments,
        **keywords,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert calls == [1]
    assert reused.candidate_reuse_decision.reused
    assert reused.candidate_reuse_decision.reason_code == "REUSED"
    assert (
        reused.auxiliary["evaluation_diagnostics"].selected_grouped_index
        == fresh.auxiliary["evaluation_diagnostics"].selected_grouped_index
    )
    _assert_output_close(reused, fresh, derivative=True)
    assert not reused.forces.requires_grad and not reused.stress.requires_grad
    assert state.reuse_count == 0
    state.validate_integrity()

    with torch.no_grad():
        no_grad = model(
            *arguments,
            **keywords,
            candidate_neighbor_state=state,
            compute_forces=True,
            compute_stress=True,
        )
    _assert_output_close(no_grad, reused, derivative=True)
    assert no_grad.candidate_reuse_decision.reused

    changed_branch = replace(
        state,
        phase_site_branch_fingerprint="0" * 64,
        integrity_fingerprint=None,
    )
    branch_rebuild = model(
        *arguments,
        **keywords,
        candidate_neighbor_state=changed_branch,
    )
    assert branch_rebuild.candidate_reuse_decision.reason_code == (
        "PHASE_SITE_BRANCH_CHANGED"
    )
    assert branch_rebuild.candidate_reuse_decision.rebuilt

    with pytest.raises(EvaluationPhaseError) as create_graph_error:
        model(
            *arguments,
            **keywords,
            candidate_neighbor_state=state,
            compute_forces=True,
            create_graph=True,
        )
    assert create_graph_error.value.reason_code == "CREATE_GRAPH_UNSUPPORTED"

    with torch.inference_mode(), pytest.raises(
        EvaluationPhaseError
    ) as inference_error:
        model(
            *arguments,
            **keywords,
            candidate_neighbor_state=state,
            compute_forces=True,
        )
    assert inference_error.value.reason_code == (
        "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED"
    )

    wrong_template = replace(
        state, template_fingerprint="1" * 64, integrity_fingerprint=None
    )
    with pytest.raises(TransportSupportError) as mismatch:
        model(*arguments, **keywords, candidate_neighbor_state=wrong_template)
    assert mismatch.value.reason_code == "TEMPLATE_MISMATCH"

    corrupted = state.to(device=state.device, dtype=state.dtype)
    corrupted.site_index[0] = (corrupted.site_index[0] + 1) % corrupted.num_sites
    with pytest.raises(TransportSupportError) as integrity:
        model(*arguments, **keywords, candidate_neighbor_state=corrupted)
    assert integrity.value.reason_code == "STATE_INTEGRITY_MISMATCH"


def test_grouped_candidate_state_original_sample_order_mixed_decisions_and_parity(
    typed_crystal,
):
    _, dense_candidate, registry, samples, batch, contexts, policies = _grouped_case(
        typed_crystal
    )
    model = _blocked_clone(dense_candidate, 2, 2)
    initial = evaluate_structure_batch(
        model,
        batch,
        contexts,
        return_aux=True,
        return_candidate_neighbor_states=True,
    )
    assert tuple(initial.candidate_neighbor_states) == batch.sample_ids
    assert tuple(initial.candidate_reuse_decisions) == batch.sample_ids
    assert all(
        decision.reason_code == "INITIAL_BUILD"
        for decision in initial.candidate_reuse_decisions.values()
    )

    reused = evaluate_structure_batch(
        model,
        batch,
        contexts,
        candidate_neighbor_states=initial.candidate_neighbor_states,
    )
    assert reused.candidate_neighbor_states is not None
    assert all(
        decision.reason_code == "REUSED"
        for decision in reused.candidate_reuse_decisions.values()
    )
    fresh = evaluate_structure_batch(model, batch, contexts)
    torch.testing.assert_close(reused.energy, fresh.energy, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(
        reused.site_energy, fresh.site_energy, atol=3e-13, rtol=3e-13
    )
    assert fresh.candidate_neighbor_states is None

    permutation = (2, 0, 1)
    permuted_batch = collate_structure_samples(
        [samples[index] for index in permutation], registry
    )
    permuted = evaluate_structure_batch(
        model,
        permuted_batch,
        contexts,
        candidate_neighbor_states=reused.candidate_neighbor_states,
    )
    assert tuple(permuted.candidate_neighbor_states) == permuted_batch.sample_ids
    assert all(
        decision.reason_code == "REUSED"
        for decision in permuted.candidate_reuse_decisions.values()
    )
    torch.testing.assert_close(
        permuted.energy,
        reused.energy[list(permutation)],
        atol=3e-13,
        rtol=3e-13,
    )

    cells = batch.cells.clone()
    cells[1, 0, 0] *= 1.0001
    changed_batch = replace(batch, cells=cells)
    partial_mapping = {
        batch.sample_ids[0]: reused.candidate_neighbor_states[batch.sample_ids[0]],
        batch.sample_ids[1]: reused.candidate_neighbor_states[batch.sample_ids[1]],
        "unused:state": object(),
    }
    mixed = evaluate_structure_batch(
        model,
        changed_batch,
        contexts,
        candidate_neighbor_states=partial_mapping,
        return_aux=True,
    )
    decisions = mixed.candidate_reuse_decisions
    assert decisions[batch.sample_ids[0]].reason_code == "REUSED"
    assert decisions[batch.sample_ids[1]].reason_code == "CELL_CHANGED"
    assert decisions[batch.sample_ids[2]].reason_code == "INITIAL_BUILD"
    assert "unused:state" not in mixed.candidate_neighbor_states
    assert tuple(mixed.candidate_neighbor_states) == batch.sample_ids

    adaptive_initial = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        return_candidate_neighbor_states=True,
    )
    adaptive_reuse = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        candidate_neighbor_states=adaptive_initial.candidate_neighbor_states,
    )
    assert all(
        decision.reason_code == "REUSED"
        for decision in adaptive_reuse.candidate_reuse_decisions.values()
    )
    adaptive_fresh = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
    )
    torch.testing.assert_close(
        adaptive_reuse.energy,
        adaptive_fresh.energy,
        atol=4e-13,
        rtol=4e-13,
    )

    fixed_to_adaptive = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        candidate_neighbor_states=initial.candidate_neighbor_states,
    )
    assert all(
        decision.reason_code == "PHASE_SITE_BRANCH_CHANGED"
        for decision in fixed_to_adaptive.candidate_reuse_decisions.values()
    )


def test_grouped_candidate_state_derivatives_double_backward_and_no_hidden_cache(
    typed_crystal, monkeypatch
):
    _, dense_candidate, _, _, batch, contexts, policies = _grouped_case(
        typed_crystal
    )
    model = _blocked_clone(dense_candidate, 2, 2)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    state_keys = tuple(model.state_dict())
    model_attributes = set(vars(model))

    initial = evaluate_structure_batch(
        model,
        batch,
        contexts,
        return_candidate_neighbor_states=True,
    )
    batch.positions.requires_grad_(True)
    fixed = evaluate_structure_batch(
        model,
        batch,
        contexts,
        candidate_neighbor_states=initial.candidate_neighbor_states,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
    )
    assert all(
        decision.reused for decision in fixed.candidate_reuse_decisions.values()
    )
    assert fixed.forces.requires_grad and fixed.stress.requires_grad
    force_loss = fixed.energy.sum() + 0.01 * fixed.forces.square().sum()
    mixed_gradient = torch.autograd.grad(
        force_loss, model.readout.mlp[-1].weight
    )[0]
    assert torch.all(torch.isfinite(mixed_gradient))
    assert torch.count_nonzero(mixed_gradient) > 0

    adaptive_initial = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        return_candidate_neighbor_states=True,
    )
    original_grad = torch.autograd.grad
    calls = []

    def counted_grad(*args, **kwargs):
        calls.append(1)
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    adaptive = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        candidate_neighbor_states=adaptive_initial.candidate_neighbor_states,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    monkeypatch.setattr(torch.autograd, "grad", original_grad)
    assert len(calls) == batch.num_structures
    assert all(
        decision.reused
        for decision in adaptive.candidate_reuse_decisions.values()
    )
    assert not adaptive.forces.requires_grad and not adaptive.stress.requires_grad
    fresh = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
    )
    torch.testing.assert_close(adaptive.energy, fresh.energy, atol=4e-13, rtol=4e-13)
    torch.testing.assert_close(
        adaptive.forces, fresh.forces, atol=3e-11, rtol=3e-11
    )
    torch.testing.assert_close(
        adaptive.stress, fresh.stress, atol=3e-11, rtol=3e-11
    )
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert tuple(model.state_dict()) == state_keys
    assert set(vars(model)) == model_attributes
    assert not any("candidate" in name and "state" in name for name in vars(model))


def test_reuse_direct_and_grouped_do_not_traverse_or_densify(
    typed_crystal, monkeypatch
):
    model, arguments = _fixed_case(typed_crystal)
    direct_state = model(
        *arguments, return_candidate_neighbor_state=True
    ).candidate_neighbor_state

    def forbidden(*args, **kwargs):
        raise AssertionError("stateless/dense candidate path was called during reuse")

    monkeypatch.setattr(
        candidate_state_module, "build_periodic_compact_transport_edges", forbidden
    )
    monkeypatch.setattr(
        potential_module, "build_periodic_compact_transport_edges", forbidden
    )
    monkeypatch.setattr(potential_module, "atom_site_displacements", forbidden)
    reused = model(
        arguments[0].clone().requires_grad_(True),
        *arguments[1:],
        candidate_neighbor_state=direct_state,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
    )
    assert reused.candidate_reuse_decision.reused
    assert not reused.candidate_reuse_decision.dense_allocation_observed

    _, dense_grouped, _, _, batch, contexts, policies = _grouped_case(typed_crystal)
    grouped_model = _blocked_clone(dense_grouped, 2, 2)
    # Temporarily restore the build function for the explicit initial build.
    monkeypatch.undo()
    initial = evaluate_structure_batch(
        grouped_model,
        batch,
        contexts,
        return_candidate_neighbor_states=True,
    )
    monkeypatch.setattr(
        candidate_state_module, "build_periodic_compact_transport_edges", forbidden
    )
    monkeypatch.setattr(
        potential_module, "build_periodic_compact_transport_edges", forbidden
    )
    monkeypatch.setattr(potential_module, "atom_site_displacements", forbidden)
    grouped = evaluate_structure_batch(
        grouped_model,
        batch,
        contexts,
        candidate_neighbor_states=initial.candidate_neighbor_states,
    )
    assert all(
        decision.reused for decision in grouped.candidate_reuse_decisions.values()
    )
    assert all(
        not decision.dense_allocation_observed
        for decision in grouped.candidate_reuse_decisions.values()
    )


def test_candidate_state_backend_misuse_and_grouped_contextual_corruption(
    typed_crystal,
):
    blocked, arguments = _fixed_case(typed_crystal)
    state = blocked(
        *arguments, return_candidate_neighbor_state=True
    ).candidate_neighbor_state
    dense_candidate = _model(typed_crystal, _edge_support())
    with pytest.raises(TransportSupportError) as invalid:
        dense_candidate(*arguments, candidate_neighbor_state=state)
    assert invalid.value.reason_code == "INVALID_SUPPORT_CONFIG"

    changed_support = replace(
        blocked.config.transport_support,
        cutoff=blocked.config.transport_support.cutoff + 0.1,
    )
    support_mismatch_model = _configured(blocked, changed_support)
    with pytest.raises(TransportSupportError) as support_mismatch:
        support_mismatch_model(*arguments, candidate_neighbor_state=state)
    assert support_mismatch.value.reason_code == "SUPPORT_CONFIG_MISMATCH"

    reordered_numbers = arguments[1].clone()
    reordered_numbers[[0, 1]] = reordered_numbers[[1, 0]]
    assert not torch.equal(reordered_numbers, arguments[1])
    with pytest.raises(TransportSupportError) as atom_order:
        blocked(
            arguments[0],
            reordered_numbers,
            *arguments[2:],
            candidate_neighbor_state=state,
        )
    assert atom_order.value.reason_code == "ATOM_ORDER_CHANGED"

    wrong_schema = state.to(device=state.device, dtype=state.dtype)
    object.__setattr__(wrong_schema, "schema_version", "wrong-state-schema")
    with pytest.raises(TransportSupportError) as schema_error:
        blocked(*arguments, candidate_neighbor_state=wrong_schema)
    assert schema_error.value.reason_code == "STATE_SCHEMA_MISMATCH"

    _, dense_grouped, _, _, batch, contexts, policies = _grouped_case(
        typed_crystal
    )
    grouped_model = _blocked_clone(dense_grouped, 2, 2)
    built = evaluate_structure_batch(
        grouped_model,
        batch,
        contexts,
        return_candidate_neighbor_states=True,
    )
    sample_id = batch.sample_ids[1]
    corrupted = built.candidate_neighbor_states[sample_id].to(
        device="cpu", dtype=torch.float64
    )
    corrupted.atom_index[0] = (
        corrupted.atom_index[0] + 1
    ) % corrupted.num_atoms
    mapping = dict(built.candidate_neighbor_states)
    mapping[sample_id] = corrupted
    with pytest.raises(TransportSupportError) as caught:
        evaluate_structure_batch(
            grouped_model,
            batch,
            contexts,
            candidate_neighbor_states=mapping,
        )
    assert caught.value.reason_code == "STATE_INTEGRITY_MISMATCH"
    message = str(caught.value)
    assert "structure_index=1" in message
    assert f"sample_id={sample_id!r}" in message
    assert "rebuild_stage='state_preflight'" in message

    with pytest.raises(EvaluationPhaseError) as adaptive_caught:
        evaluate_structure_batch(
            grouped_model,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
            candidate_neighbor_states=mapping,
        )
    assert adaptive_caught.value.reason_code == "STATE_INTEGRITY_MISMATCH"
    adaptive_message = str(adaptive_caught.value)
    assert "structure_index=1" in adaptive_message
    assert f"sample_id={sample_id!r}" in adaptive_message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_candidate_state_cpu_to_cuda_materialization(typed_crystal, dtype):
    cpu_data = {
        key: value.to(dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value.clone()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in typed_crystal.items()
    }
    _, edge, context, policy, arguments, keywords = _adaptive_case(
        cpu_data, 5, template_id="candidate-state-cuda", warmup=20
    )
    model = _blocked_clone(edge, 2, 2)
    initial = model(
        *arguments,
        **keywords,
        return_candidate_neighbor_state=True,
    )
    state = initial.candidate_neighbor_state

    model = model.to(device="cuda", dtype=dtype)
    cuda_arguments = tuple(
        value.to(
            device="cuda",
            dtype=dtype if value.is_floating_point() else value.dtype,
        )
        for value in arguments
    )
    output = model(
        *cuda_arguments,
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        candidate_neighbor_state=state,
        compute_forces=True,
        compute_stress=True,
    )
    assert output.candidate_reuse_decision.state_materialized
    assert output.candidate_reuse_decision.reason_code == (
        "STATE_DEVICE_MATERIALIZATION"
    )
    assert output.candidate_neighbor_state.device.type == "cuda"
    assert output.candidate_neighbor_state.dtype == dtype
    for tensor in (output.energy, output.forces, output.stress):
        assert tensor.device.type == "cuda" and tensor.dtype == dtype
        assert torch.all(torch.isfinite(tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_candidate_state_cuda_materialization_and_derivatives(
    typed_crystal, dtype
):
    _, dense_candidate, _, _, batch, contexts, policies = _grouped_case(
        typed_crystal, dtype=dtype, device="cuda"
    )
    model = _blocked_clone(dense_candidate, 2, 2)
    initial = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        return_candidate_neighbor_states=True,
    )
    cpu_states = {
        sample_id: state.to(device="cpu", dtype=torch.float64)
        for sample_id, state in initial.candidate_neighbor_states.items()
    }
    batch.positions.requires_grad_(True)
    output = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        candidate_neighbor_states=cpu_states,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert tuple(output.candidate_neighbor_states) == batch.sample_ids
    assert all(
        decision.reused and decision.state_materialized
        for decision in output.candidate_reuse_decisions.values()
    )
    assert all(
        decision.reason_code == "STATE_DEVICE_MATERIALIZATION"
        for decision in output.candidate_reuse_decisions.values()
    )
    for tensor in (output.energy, output.forces, output.stress):
        assert tensor.device.type == "cuda" and tensor.dtype == dtype
        assert torch.all(torch.isfinite(tensor))
    assert all(
        not auxiliary["transport_support"].dense_candidate_allocation_observed
        for auxiliary in output.auxiliary
    )
