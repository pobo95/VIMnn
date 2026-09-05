from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from refsite_mlip.cli.export_bundle import export_bundle
from refsite_mlip.cli.train import run_training
from refsite_mlip.data import (
    ReferenceTemplate,
    StrictTemplateDomain,
    capture_reference_structure_artifact,
    collate_structure_samples,
)
from refsite_mlip.models import (
    ReferenceSitePotential,
    capture_reference_site_model_bundle,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.training import load_training_checkpoint
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED, TransportSupportConfig

from test_cli_inspect_bundle import _typed_crystal_data
from test_symmetric_correlation_bundle import (
    _assert_output_equal,
    _assert_single_equal,
    _capture_v2,
    _geometry,
    _grouped,
    _single,
)
from test_validate_train_config_cli import _simple_case


def _strict_training_bundle(tmp_path, *, edge_backend):
    _, model, registry, samples, _, _, policies, original = _capture_v2(
        _typed_crystal_data()
    )
    support = TransportSupportConfig(
        kind="compact_c2",
        cutoff=4.0,
        switch_width=0.5,
        candidate_skin=0.2,
        backend="edge_list" if edge_backend else "dense",
        candidate_backend="blocked" if edge_backend else "dense",
        site_block_size=2,
        atom_block_size=3,
    )
    configured = ReferenceSitePotential(
        replace(model.config, transport_support=support),
        model.topology,
        model.phase_modes,
        model.phase_mode_weights,
        model.species_alignment_weights,
        model.site_alignment_weights,
        model.phase_channel_weights,
        model.atomic_baseline,
    ).to(model.atomic_baseline)
    configured.load_state_dict(model.state_dict(), strict=True)

    original_bindings = {
        binding.template_id: binding for binding in original.template_bindings
    }
    artifacts = {}
    phases = {}
    for template_id in ("alpha", "zeta"):
        template = registry.resolve(template_id)
        reference_composition = tuple(
            int(torch.count_nonzero(template.topology.site_types == index))
            for index in range(len(template.supported_species))
        )
        compositions = []
        for sample in samples:
            if sample.template_id != template_id:
                continue
            composition = tuple(
                int(torch.count_nonzero(sample.atomic_numbers == species))
                for species in template.supported_species
            )
            if composition not in compositions:
                compositions.append(composition)
        strict = ReferenceTemplate.snapshot(
            template.template_id,
            template.topology,
            template.phase_modes,
            template.phase_mode_weights,
            template.site_alignment_weights,
            template.phase_channel_weights,
            template.stabilizer,
            template.supported_species,
            convention_version=template.convention_version,
            strict_domain=StrictTemplateDomain(
                reference_site_count=template.topology.num_sites,
                supercell_shape=(1, 1, 1),
                species_vocabulary=template.supported_species,
                reference_composition=reference_composition,
                allowed_compositions=tuple(compositions),
                allowed_num_atoms=tuple(sum(value) for value in compositions),
                allowed_vacancy_masses=tuple(
                    template.topology.num_sites - sum(value)
                    for value in compositions
                ),
            ),
        )
        artifacts[template_id] = capture_reference_structure_artifact(
            strict, avg_num_neighbors=6.0
        )
        phases[template_id] = original_bindings[template_id].phase_specification
    provisional = capture_reference_site_model_bundle(
        model=configured,
        structural_artifacts=artifacts,
        phase_specifications=phases,
        evaluation_policies=None,
        default_template_id="zeta",
        provenance={"purpose": "v2_adaptive_export_closure"},
    )
    provisional_bindings = {
        binding.template_id: binding for binding in provisional.template_bindings
    }
    rebound_policies = {
        template_id: replace(
            policies[template_id],
            template_fingerprint=provisional_bindings[
                template_id
            ].full_template_fingerprint,
            content_fingerprint=None,
        )
        for template_id in policies
    }
    bundle = capture_reference_site_model_bundle(
        model=configured,
        structural_artifacts=artifacts,
        phase_specifications=phases,
        evaluation_policies=rebound_policies,
        default_template_id="zeta",
        provenance={"purpose": "v2_adaptive_export_closure"},
    )
    path = tmp_path / "initial-v2.pt"
    save_reference_site_model_bundle(path, bundle)
    return {"path": path, "bundle": bundle, "samples": samples}


@pytest.mark.parametrize("edge_backend", [False, True], ids=["dense_masked", "edge_list"])
def test_v2_best_latest_export_fixed_and_adaptive_direct_grouped_exact(
    tmp_path, edge_backend
):
    fixture = _strict_training_bundle(tmp_path, edge_backend=edge_backend)
    assert fixture["bundle"].model_config["transport_support"]["kind"] == (
        "compact_c2"
    )
    assert fixture["bundle"].model_config["transport_support"]["backend"] == (
        "edge_list" if edge_backend else "dense"
    )
    config_path, payload = _simple_case(tmp_path, fixture)
    payload["fit"]["max_epochs"] = 2
    payload["selection"]["mode"] = "max"
    payload["scheduler"]["mode"] = "max"
    payload["output_directory"] = "run"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    run_training(config_path)
    run = tmp_path / "run"
    best = load_training_checkpoint(run / "checkpoints" / "best.pt")
    latest = load_training_checkpoint(run / "checkpoints" / "latest.pt")
    assert best.progress.last_completed_epoch == 0
    assert latest.progress.last_completed_epoch == 1
    assert any(
        not torch.equal(best.model_state_dict[key], latest.model_state_dict[key])
        for key in best.model_state_dict
        if ".symmetric_contraction.weight_" in key
    )

    geometries = tuple(_geometry(sample) for sample in fixture["samples"])
    for source, checkpoint in (("best", best), ("latest", latest)):
        exported_path = tmp_path / f"{source}-{edge_backend}.pt"
        export_bundle(run, source=source, output_path=exported_path)
        exported = load_reference_site_model_bundle(exported_path)
        source_runtime = instantiate_reference_site_model_bundle(
            fixture["bundle"], device="cpu", dtype=torch.float64
        )
        source_runtime.model.load_state_dict(checkpoint.model_state_dict, strict=True)
        target_runtime = instantiate_reference_site_model_bundle(
            exported, device="cpu", dtype=torch.float64
        )
        source_batch = collate_structure_samples(geometries, source_runtime.registry)
        target_batch = collate_structure_samples(geometries, target_runtime.registry)
        for solver in (TRAIN_FIXED, EVAL_ADAPTIVE):
            expected_grouped = _grouped(
                source_runtime.model,
                source_batch,
                source_runtime.template_contexts,
                source_runtime.evaluation_policies,
                solver,
            )
            actual_grouped = _grouped(
                target_runtime.model,
                target_batch,
                target_runtime.template_contexts,
                target_runtime.evaluation_policies,
                solver,
            )
            _assert_output_equal(expected_grouped, actual_grouped)
            for sample in geometries:
                expected = _single(
                    source_runtime.model,
                    sample,
                    source_runtime.template_contexts[sample.template_id],
                    source_runtime.evaluation_policies.get(sample.template_id),
                    solver,
                )
                actual = _single(
                    target_runtime.model,
                    sample,
                    target_runtime.template_contexts[sample.template_id],
                    target_runtime.evaluation_policies.get(sample.template_id),
                    solver,
                )
                _assert_single_equal(expected, actual)
            if solver == EVAL_ADAPTIVE:
                for expected_aux, actual_aux in zip(
                    expected_grouped.auxiliary, actual_grouped.auxiliary
                ):
                    left = expected_aux["evaluation_diagnostics"]
                    right = actual_aux["evaluation_diagnostics"]
                    assert left.selected_grouped_index == right.selected_grouped_index
                    assert left.transport_support_fingerprint == (
                        right.transport_support_fingerprint
                    )
                    assert left.transport_fallback_used is False
                    assert right.transport_fallback_used is False
                    if edge_backend:
                        assert left.transport_backend == right.transport_backend == "edge_list"
                        assert not left.transport_dense_plan_materialized
                        assert not right.transport_dense_plan_materialized
                    else:
                        assert left.transport_backend == right.transport_backend == "dense"
