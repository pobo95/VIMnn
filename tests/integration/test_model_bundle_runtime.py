from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from refsite_mlip.data import PhaseSpecification, capture_reference_structure_artifact
from refsite_mlip.models import (
    capture_reference_site_model_bundle,
    evaluate_structure_batch,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED, TransportSupportConfig

from test_grouped_evaluation_phase_batch import _adaptive_case


def _phase_from_template(template):
    site_type_count = len(template.supported_species)
    rows = []
    for site_type in range(site_type_count):
        matches = torch.nonzero(
            template.topology.site_types == site_type, as_tuple=False
        ).flatten()
        assert matches.numel() > 0
        row = template.site_alignment_weights[int(matches[0])]
        assert torch.all(
            template.site_alignment_weights[matches] == row.unsqueeze(0)
        )
        rows.append(row)
    return PhaseSpecification(
        modes=template.phase_modes,
        mode_weights=template.phase_mode_weights,
        site_type_alignment_weights=torch.stack(rows),
        channel_weights=template.phase_channel_weights,
        approval_status="provisional",
        convention_version="bundle_runtime_phase_v1",
    )


def _capture_case(typed_crystal, *, edge_backend=False):
    data, model, registry, samples, batch, contexts, policies = _adaptive_case(
        typed_crystal
    )
    if edge_backend:
        support = TransportSupportConfig(
            kind="compact_c2",
            cutoff=2.6,
            switch_width=0.5,
            candidate_skin=0.2,
            backend="edge_list",
            candidate_backend="blocked",
            site_block_size=2,
            atom_block_size=3,
        )
        configured = type(model)(
            replace(
                model.config,
                transport_support=support,
                feature=replace(
                    model.config.feature, probability_tolerance=1.0e-6
                ),
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
        model = configured
    artifacts = {}
    phases = {}
    for template_id in ("alpha", "zeta"):
        template = registry.resolve(template_id)
        artifacts[template_id] = capture_reference_structure_artifact(
            template, avg_num_neighbors=6.0
        )
        phases[template_id] = _phase_from_template(template)
    bundle = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts={"zeta": artifacts["zeta"], "alpha": artifacts["alpha"]},
        phase_specifications={"alpha": phases["alpha"], "zeta": phases["zeta"]},
        evaluation_policies={"zeta": policies["zeta"], "alpha": policies["alpha"]},
        default_template_id="zeta",
        provenance={"purpose": "synthetic_bundle_runtime_parity"},
    )
    return data, model, registry, samples, batch, contexts, policies, bundle


def _assert_grouped(left, right, *, tolerance):
    for first, second in (
        (left.energy, right.energy),
        (left.baseline_energy, right.baseline_energy),
        (left.residual_energy, right.residual_energy),
        (left.site_energy, right.site_energy),
    ):
        torch.testing.assert_close(first, second, atol=tolerance, rtol=tolerance)
    if left.forces is not None:
        torch.testing.assert_close(left.forces, right.forces, atol=tolerance, rtol=tolerance)
    if left.stress is not None:
        torch.testing.assert_close(left.stress, right.stress, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(left.stress_voigt, right.stress_voigt, atol=tolerance, rtol=tolerance)
    assert left.sample_ids == right.sample_ids
    assert left.template_ids == right.template_ids
    assert torch.equal(left.site_ptr, right.site_ptr)
    assert torch.equal(left.site_batch, right.site_batch)


def test_mixed_template_bundle_fixed_and_adaptive_runtime_parity(typed_crystal, tmp_path):
    _, model, registry, _, batch, contexts, policies, bundle = _capture_case(
        typed_crystal
    )
    path = tmp_path / "mixed.pt"
    save_reference_site_model_bundle(path, bundle)
    runtime = instantiate_reference_site_model_bundle(
        load_reference_site_model_bundle(path)
    )
    by_id = {binding.template_id: binding for binding in bundle.template_bindings}
    reordered = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts={
            "zeta": by_id["zeta"].structural_artifact,
            "alpha": by_id["alpha"].structural_artifact,
        },
        phase_specifications={
            "zeta": by_id["zeta"].phase_specification,
            "alpha": by_id["alpha"].phase_specification,
        },
        evaluation_policies={
            "zeta": policies["zeta"],
            "alpha": policies["alpha"],
        },
        default_template_id="zeta",
        provenance={"purpose": "synthetic_bundle_runtime_parity"},
    )
    assert reordered.bundle_fingerprint == bundle.bundle_fingerprint
    assert runtime.registry.fingerprint == registry.fingerprint
    assert tuple(runtime.template_contexts) == ("alpha", "zeta")
    assert tuple(runtime.evaluation_policies) == ("alpha", "zeta")

    fixed_batch_left = replace(
        batch,
        positions=batch.positions.detach().clone().requires_grad_(True),
    )
    fixed_batch_right = replace(
        batch,
        positions=batch.positions.detach().clone().requires_grad_(True),
    )
    original_fixed = evaluate_structure_batch(
        model,
        fixed_batch_left,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    restored_fixed = evaluate_structure_batch(
        runtime.model,
        fixed_batch_right,
        runtime.template_contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    _assert_grouped(original_fixed, restored_fixed, tolerance=2e-14)
    for left, right in zip(original_fixed.auxiliary, restored_fixed.auxiliary):
        assert torch.equal(left["phase"], right["phase"])
        assert torch.equal(left["ot"].P, right["ot"].P)
        assert torch.equal(left["ot"].q, right["ot"].q)
        assert torch.equal(
            left["multipoles"].equivariant_features,
            right["multipoles"].equivariant_features,
        )

    adaptive_batch_left = replace(
        batch,
        positions=batch.positions.detach().clone().requires_grad_(True),
    )
    adaptive_batch_right = replace(
        batch,
        positions=batch.positions.detach().clone().requires_grad_(True),
    )
    original_adaptive = evaluate_structure_batch(
        model,
        adaptive_batch_left,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    restored_adaptive = evaluate_structure_batch(
        runtime.model,
        adaptive_batch_right,
        runtime.template_contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=runtime.evaluation_policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    _assert_grouped(original_adaptive, restored_adaptive, tolerance=3e-13)
    for left, right in zip(original_adaptive.auxiliary, restored_adaptive.auxiliary):
        assert torch.equal(left["phase"], right["phase"])
        assert torch.equal(left["ot"].P, right["ot"].P)
        assert torch.equal(left["ot"].q, right["ot"].q)
        assert (
            left["evaluation_diagnostics"].selected_grouped_index
            == right["evaluation_diagnostics"].selected_grouped_index
        )


def test_edge_blocked_config_roundtrip_and_first_call_has_no_restored_candidate_state(
    typed_crystal,
):
    _, _, _, _, batch, _, _, bundle = _capture_case(
        typed_crystal, edge_backend=True
    )
    runtime = instantiate_reference_site_model_bundle(bundle)
    support = runtime.model.config.transport_support
    assert support.backend == "edge_list"
    assert support.candidate_backend == "blocked"
    assert (support.site_block_size, support.atom_block_size) == (2, 3)
    atom_slice = slice(int(batch.atom_ptr[0]), int(batch.atom_ptr[1]))
    output = runtime.model(
        batch.positions[atom_slice],
        batch.atomic_numbers[atom_slice],
        batch.cells[0],
        batch.origins[0],
        template_context=runtime.template_contexts[batch.template_ids[0]],
        return_aux=True,
        return_candidate_neighbor_state=True,
    )
    assert output.candidate_neighbor_state is not None
    assert output.candidate_reuse_decision.reason_code == "INITIAL_BUILD"
    assert not output.auxiliary["ot"].dense_plan_materialized


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_mixed_bundle_fixed_adaptive_smoke(typed_crystal, dtype):
    _, _, registry, samples, _, _, _, bundle = _capture_case(
        typed_crystal, edge_backend=True
    )
    runtime = instantiate_reference_site_model_bundle(
        bundle, device="cuda", dtype=dtype
    )
    moved = tuple(sample.to(device="cuda", dtype=dtype) for sample in samples)
    from refsite_mlip.data import collate_structure_samples

    batch = collate_structure_samples(moved, registry)
    batch = replace(batch, positions=batch.positions.requires_grad_(True))
    for solver_path in (TRAIN_FIXED, EVAL_ADAPTIVE):
        output = evaluate_structure_batch(
            runtime.model,
            batch,
            runtime.template_contexts,
            solver_path=solver_path,
            evaluation_policies=(
                runtime.evaluation_policies if solver_path == EVAL_ADAPTIVE else None
            ),
            compute_forces=True,
            compute_stress=True,
        )
        assert output.energy.dtype == dtype and output.energy.device.type == "cuda"
        assert torch.all(torch.isfinite(output.energy))
        assert torch.all(torch.isfinite(output.forces))
        assert torch.all(torch.isfinite(output.stress))
