from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import refsite_mlip.models.potential as potential_module
from refsite_mlip.data import collate_structure_samples
from refsite_mlip.models import evaluate_structure_batch
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from test_evaluation_phase_potential import _policy
from test_grouped_template_batch import _case


def _adaptive_case(typed_crystal, *, dtype=torch.float64, device="cpu"):
    data, model, registry, samples, batch, contexts = _case(
        typed_crystal, dtype=dtype, device=device
    )
    policies = {
        template_id: _policy(registry.resolve(template_id))
        for template_id in ("alpha", "zeta")
    }
    return data, model, registry, samples, batch, contexts, policies


def _individual(model, batch, contexts, policies, **kwargs):
    outputs = []
    for index, template_id in enumerate(batch.template_ids):
        atom_slice = slice(
            int(batch.atom_ptr[index]), int(batch.atom_ptr[index + 1])
        )
        outputs.append(
            model(
                batch.positions[atom_slice],
                batch.atomic_numbers[atom_slice],
                batch.cells[index],
                batch.origins[index],
                solver_path=EVAL_ADAPTIVE,
                template_context=contexts[template_id],
                evaluation_policy=policies[template_id],
                **kwargs,
            )
        )
    return tuple(outputs)


def _assert_auxiliary_equal(grouped, individual):
    assert grouped.auxiliary is not None
    for auxiliary, single in zip(grouped.auxiliary, individual):
        assert auxiliary is not None and single.auxiliary is not None
        assert torch.equal(auxiliary["phase"], single.auxiliary["phase"])
        assert torch.equal(auxiliary["ot"].P, single.auxiliary["ot"].P)
        assert torch.equal(auxiliary["ot"].q, single.auxiliary["ot"].q)
        assert torch.equal(
            auxiliary["multipoles"].equivariant_features,
            single.auxiliary["multipoles"].equivariant_features,
        )
        left = auxiliary["evaluation_diagnostics"]
        right = single.auxiliary["evaluation_diagnostics"]
        assert left.selected_original_candidate_index == right.selected_original_candidate_index
        assert left.selected_grouped_index == right.selected_grouped_index
        assert left.absolute_objective_gap == right.absolute_objective_gap
        assert left.transport_solver_name == right.transport_solver_name
        assert left.transport_sinkhorn_iterations == right.transport_sinkhorn_iterations
        assert left.transport_newton_iterations == right.transport_newton_iterations
        assert left.transport_cg_iterations == right.transport_cg_iterations
        assert left.transport_fallback_used == right.transport_fallback_used
        assert torch.equal(left.selected_pre_refinement_phase, right.selected_pre_refinement_phase)
        assert torch.equal(left.refined_phase, right.refined_phase)
        assert left.template_fingerprint == left.context_fingerprint
        assert left.policy_template_fingerprint == left.context_fingerprint
        assert left.policy_content_fingerprint


def test_grouped_adaptive_individual_parity_order_and_diagnostics(typed_crystal):
    _, model, _, _, batch, contexts, policies = _adaptive_case(typed_crystal)
    batch.positions.requires_grad_(True)
    execution_order = []
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        execution_order.append(kwargs["template_context"].template_id)
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    grouped = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    model.forward = original_forward
    individual = _individual(
        model,
        batch,
        contexts,
        policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )

    assert execution_order == ["alpha", "zeta", "zeta"]
    assert grouped.sample_ids == batch.sample_ids
    assert grouped.template_ids == ("zeta", "alpha", "zeta")
    assert torch.equal(grouped.energy, torch.stack([item.energy for item in individual]))
    assert torch.equal(
        grouped.baseline_energy,
        torch.stack([item.baseline_energy for item in individual]),
    )
    assert torch.equal(
        grouped.residual_energy,
        torch.stack([item.residual_energy for item in individual]),
    )
    assert torch.equal(
        grouped.site_energy, torch.cat([item.site_energy for item in individual])
    )
    torch.testing.assert_close(
        grouped.forces,
        torch.cat([item.forces for item in individual]),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    torch.testing.assert_close(
        grouped.stress,
        torch.stack([item.stress for item in individual]),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    torch.testing.assert_close(
        grouped.stress_voigt,
        torch.stack([item.stress_voigt for item in individual]),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    assert grouped.site_ptr.tolist() == [0, 6, 10, 16]
    assert grouped.site_batch.tolist() == [0] * 6 + [1] * 4 + [2] * 6
    _assert_auxiliary_equal(grouped, individual)
    assert not grouped.forces.requires_grad and grouped.forces.grad_fn is None
    assert not grouped.stress.requires_grad and grouped.stress.grad_fn is None


def test_grouped_adaptive_permutation_split_independence_and_no_aux(typed_crystal):
    _, model, registry, samples, batch, contexts, policies = _adaptive_case(
        typed_crystal
    )
    full = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
    )
    assert full.auxiliary is None

    order = (2, 0, 1)
    permuted = evaluate_structure_batch(
        model,
        collate_structure_samples(tuple(samples[index] for index in order), registry),
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
    )
    assert torch.equal(permuted.energy, full.energy[list(order)])

    split = tuple(
        evaluate_structure_batch(
            model,
            collate_structure_samples((sample,), registry),
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
        )
        for sample in samples
    )
    assert torch.equal(full.energy, torch.cat([item.energy for item in split]))
    assert torch.equal(full.site_energy, torch.cat([item.site_energy for item in split]))

    moved = replace(
        samples[0],
        positions=samples[0].positions
        + torch.tensor([0.011, -0.007, 0.005], dtype=torch.float64),
    )
    perturbed = evaluate_structure_batch(
        model,
        collate_structure_samples((moved, samples[1], samples[2]), registry),
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
    )
    assert torch.equal(perturbed.energy[1:], full.energy[1:])


def test_grouped_adaptive_preflight_is_complete_before_forward(
    typed_crystal, monkeypatch
):
    _, model, _, _, batch, contexts, policies = _adaptive_case(typed_crystal)
    calls = []
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", counted_forward)
    with pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            model,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies={"zeta": policies["zeta"]},
        )
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    assert calls == []

    mutated_policy = policies["alpha"]
    mutated_policy.candidate_offsets[0, 0] = 0.125
    with pytest.raises(ValueError, match="content fingerprint"):
        evaluate_structure_batch(
            model,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
        )
    assert calls == []

    swapped = {"alpha": policies["zeta"], "zeta": policies["alpha"]}
    with pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            model,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=swapped,
        )
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    assert calls == []

    with pytest.raises(ValueError, match="TRAIN_FIXED"):
        evaluate_structure_batch(
            model,
            batch,
            contexts,
            solver_path=TRAIN_FIXED,
            evaluation_policies=policies,
        )
    assert calls == []


def test_grouped_adaptive_invalid_solver_combinations_fail_before_forward(
    typed_crystal, monkeypatch
):
    _, model, _, _, batch, contexts, policies = _adaptive_case(typed_crystal)
    calls = []
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", counted_forward)
    with pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            model,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
            create_graph=True,
        )
    assert caught.value.reason_code == "CREATE_GRAPH_UNSUPPORTED"
    assert calls == []

    batch.positions.requires_grad_(True)
    with torch.inference_mode(), pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            model,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
            compute_forces=True,
        )
    assert caught.value.reason_code == "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED"
    assert calls == []


def test_grouped_adaptive_runtime_error_preserves_reason_and_structure_context(
    typed_crystal,
):
    _, model, registry, _, batch, contexts, policies = _adaptive_case(typed_crystal)
    policies = dict(policies)
    policies["alpha"] = _policy(
        registry.resolve("alpha"), minimum_objective_gap_absolute=1.0e6
    )
    with pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            model,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
        )
    assert caught.value.reason_code == "NON_EQUIVALENT_GAP_TOO_SMALL"
    message = str(caught.value)
    assert "structure_index=1" in message
    assert "sample_id='alpha-pristine'" in message
    assert "template_id='alpha'" in message
    assert "stage=single_structure_evaluation" in message


def test_grouped_adaptive_derivative_fallback_preserves_reason_and_context(
    typed_crystal, monkeypatch
):
    _, model, _, _, batch, contexts, policies = _adaptive_case(typed_crystal)
    batch.positions.requires_grad_(True)
    original_solver = potential_module.solve_atom_vacancy_ot

    def fallback_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        return replace(result, fallback_used=True)

    monkeypatch.setattr(
        potential_module, "solve_atom_vacancy_ot", fallback_solver
    )
    with pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            model,
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
    assert "stage=single_structure_evaluation" in message


def test_grouped_adaptive_unused_bindings_do_not_affect_results(typed_crystal):
    _, model, _, _, batch, contexts, policies = _adaptive_case(typed_crystal)
    baseline = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
    )
    extra_contexts = dict(contexts)
    extra_contexts["unused"] = contexts["alpha"]
    extra_policies = dict(policies)
    extra_policies["unused"] = policies["alpha"]
    extra = evaluate_structure_batch(
        model,
        batch,
        extra_contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=extra_policies,
    )
    assert torch.equal(extra.energy, baseline.energy)
    assert torch.equal(extra.site_energy, baseline.site_energy)


def test_grouped_adaptive_representative_force_and_stress_finite_difference(
    typed_crystal,
):
    _, model, _, _, batch, contexts, policies = _adaptive_case(typed_crystal)
    batch.positions.requires_grad_(True)
    baseline = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    force_step = 2.0e-6
    for structure_index, atom_index, component in ((0, 0, 0), (1, 5, 1)):
        displacement = torch.zeros_like(batch.positions)
        displacement[atom_index, component] = force_step
        plus = evaluate_structure_batch(
            model,
            replace(batch, positions=batch.positions.detach() + displacement),
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
            return_aux=True,
        )
        minus = evaluate_structure_batch(
            model,
            replace(batch, positions=batch.positions.detach() - displacement),
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
            return_aux=True,
        )
        finite = -(plus.energy[structure_index] - minus.energy[structure_index]) / (
            2.0 * force_step
        )
        torch.testing.assert_close(
            baseline.forces[atom_index, component], finite, atol=5.0e-6, rtol=5.0e-4
        )
        assert torch.equal(plus.energy[1 - structure_index], baseline.energy[1 - structure_index])
        for perturbed in (plus, minus):
            assert (
                perturbed.auxiliary[structure_index]["evaluation_diagnostics"]
                .selected_grouped_index
                == baseline.auxiliary[structure_index]["evaluation_diagnostics"]
                .selected_grouped_index
            )

    strain_step = 1.0e-4
    identity = torch.eye(3, dtype=batch.dtype, device=batch.device)
    direction = torch.zeros_like(identity)
    direction[0, 1] = direction[1, 0] = 0.5
    for structure_index in (0, 1):
        plus_positions = batch.positions.detach().clone()
        minus_positions = batch.positions.detach().clone()
        atom_slice = slice(
            int(batch.atom_ptr[structure_index]),
            int(batch.atom_ptr[structure_index + 1]),
        )
        plus_deformation = identity + strain_step * direction
        minus_deformation = identity - strain_step * direction
        plus_positions[atom_slice] = plus_positions[atom_slice] @ plus_deformation
        minus_positions[atom_slice] = minus_positions[atom_slice] @ minus_deformation
        plus_cells = batch.cells.clone()
        minus_cells = batch.cells.clone()
        plus_origins = batch.origins.clone()
        minus_origins = batch.origins.clone()
        plus_cells[structure_index] = plus_cells[structure_index] @ plus_deformation
        minus_cells[structure_index] = minus_cells[structure_index] @ minus_deformation
        plus_origins[structure_index] = plus_origins[structure_index] @ plus_deformation
        minus_origins[structure_index] = minus_origins[structure_index] @ minus_deformation
        plus = evaluate_structure_batch(
            model,
            replace(
                batch,
                positions=plus_positions,
                cells=plus_cells,
                origins=plus_origins,
            ),
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
        )
        minus = evaluate_structure_batch(
            model,
            replace(
                batch,
                positions=minus_positions,
                cells=minus_cells,
                origins=minus_origins,
            ),
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
        )
        volume = torch.linalg.det(batch.cells[structure_index]).abs()
        finite = (plus.energy[structure_index] - minus.energy[structure_index]) / (
            2.0 * strain_step * volume
        )
        automatic = torch.sum(baseline.stress[structure_index] * direction)
        torch.testing.assert_close(automatic, finite, atol=5.0e-6, rtol=5.0e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_adaptive_cuda_energy_force_stress_smoke(typed_crystal, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _, model, _, _, batch, contexts, policies = _adaptive_case(
        typed_crystal, dtype=dtype, device="cuda"
    )
    batch.positions.requires_grad_(True)
    output = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert output.energy.dtype == dtype and output.energy.device.type == "cuda"
    assert output.forces.dtype == dtype and output.forces.device.type == "cuda"
    assert output.stress.dtype == dtype and output.stress.device.type == "cuda"
    assert bool(torch.all(torch.isfinite(output.energy)))
    assert bool(torch.all(torch.isfinite(output.forces)))
    assert bool(torch.all(torch.isfinite(output.stress)))
    expected_tolerance = 1.0e-6 if dtype == torch.float32 else 1.0e-12
    assert output.auxiliary is not None
    for auxiliary in output.auxiliary:
        assert (
            auxiliary["evaluation_diagnostics"].effective_transport_tolerance
            == expected_tolerance
        )
