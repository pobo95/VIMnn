from __future__ import annotations

import pytest
import torch

from refsite_mlip.data import collate_structure_samples
from refsite_mlip.models import evaluate_structure_batch
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from symmetric_potential_helpers import v2_grouped_case
from test_evaluation_phase_potential import _policy


def _individual(model, batch, contexts, *, solver_path, policies=None, **kwargs):
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
                solver_path=solver_path,
                template_context=contexts[template_id],
                evaluation_policy=(
                    None if policies is None else policies[template_id]
                ),
                **kwargs,
            )
        )
    return tuple(outputs)


def _assert_grouped_parity(grouped, individual):
    assert torch.equal(grouped.energy, torch.stack([value.energy for value in individual]))
    assert torch.equal(
        grouped.site_energy,
        torch.cat([value.site_energy for value in individual]),
    )
    assert torch.equal(
        grouped.baseline_energy,
        torch.stack([value.baseline_energy for value in individual]),
    )
    assert torch.equal(
        grouped.residual_energy,
        torch.stack([value.residual_energy for value in individual]),
    )
    if grouped.forces is not None:
        torch.testing.assert_close(
            grouped.forces,
            torch.cat([value.forces for value in individual]),
            atol=2.0e-14,
            rtol=2.0e-14,
        )
    if grouped.stress is not None:
        torch.testing.assert_close(
            grouped.stress,
            torch.stack([value.stress for value in individual]),
            atol=2.0e-14,
            rtol=2.0e-14,
        )
        torch.testing.assert_close(
            grouped.stress_voigt,
            torch.stack([value.stress_voigt for value in individual]),
            atol=2.0e-14,
            rtol=2.0e-14,
        )


def test_v2_grouped_fixed_mixed_template_direct_parity(typed_crystal):
    _, model, _, _, batch, contexts = v2_grouped_case(
        typed_crystal, layers=2
    )
    batch.positions.requires_grad_(True)
    grouped = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    individual = _individual(
        model,
        batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    _assert_grouped_parity(grouped, individual)
    assert grouped.sample_ids == batch.sample_ids
    assert grouped.template_ids == ("zeta", "alpha", "zeta")
    assert grouped.site_ptr.tolist() == [0, 6, 10, 16]
    assert grouped.auxiliary is not None
    for index, (auxiliary, direct) in enumerate(
        zip(grouped.auxiliary, individual)
    ):
        assert auxiliary is not None and direct.auxiliary is not None
        dense_plan = auxiliary["ot"].P
        assert torch.equal(dense_plan, direct.auxiliary["ot"].P)
        assert torch.equal(auxiliary["ot"].q, direct.auxiliary["ot"].q)
        assert torch.equal(
            auxiliary["multipoles"].equivariant_features,
            direct.auxiliary["multipoles"].equivariant_features,
        )
        site_count = int(grouped.site_ptr[index + 1] - grouped.site_ptr[index])
        atom_count = int(batch.atom_ptr[index + 1] - batch.atom_ptr[index])
        torch.testing.assert_close(
            auxiliary["ot"].q.sum(),
            auxiliary["ot"].q.new_tensor(site_count - atom_count),
            atol=2.0e-12,
            rtol=0.0,
        )


def test_v2_grouped_fixed_permutation_and_split_parity(typed_crystal):
    _, model, registry, samples, batch, contexts = v2_grouped_case(
        typed_crystal, layers=1
    )
    full = evaluate_structure_batch(
        model, batch, contexts, solver_path=TRAIN_FIXED
    )
    permutation = (2, 0, 1)
    permuted = evaluate_structure_batch(
        model,
        collate_structure_samples(
            tuple(samples[index] for index in permutation), registry
        ),
        contexts,
        solver_path=TRAIN_FIXED,
    )
    assert torch.equal(permuted.energy, full.energy[list(permutation)])
    split = tuple(
        evaluate_structure_batch(
            model,
            collate_structure_samples((sample,), registry),
            contexts,
            solver_path=TRAIN_FIXED,
        )
        for sample in samples
    )
    assert torch.equal(full.energy, torch.cat([value.energy for value in split]))
    assert torch.equal(
        full.site_energy, torch.cat([value.site_energy for value in split])
    )


def test_v2_grouped_adaptive_direct_parity_and_branch_contract(typed_crystal):
    _, model, registry, _, batch, contexts = v2_grouped_case(
        typed_crystal, layers=1
    )
    policies = {
        template_id: _policy(registry.resolve(template_id))
        for template_id in ("alpha", "zeta")
    }
    batch.positions.requires_grad_(True)
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
    individual = _individual(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    _assert_grouped_parity(grouped, individual)
    assert grouped.auxiliary is not None
    for auxiliary in grouped.auxiliary:
        diagnostics = auxiliary["evaluation_diagnostics"]
        assert diagnostics.selected_grouped_index == 0
        assert not diagnostics.transport_fallback_used
        assert diagnostics.derivative_order == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v2_grouped_cuda_fixed_smoke(typed_crystal, dtype):
    _, model, _, _, batch, contexts = v2_grouped_case(
        typed_crystal,
        layers=1,
        dtype=dtype,
        device="cuda:0",
    )
    batch.positions.requires_grad_(True)
    output = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
    )
    torch.cuda.synchronize()
    assert output.energy.device.type == "cuda" and output.energy.dtype == dtype
    assert bool(torch.all(torch.isfinite(output.energy)))
    assert bool(torch.all(torch.isfinite(output.forces)))
    assert bool(torch.all(torch.isfinite(output.stress)))
