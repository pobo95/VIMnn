from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import textwrap
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("ase")

from refsite_mlip.cli.main import build_parser, main
from refsite_mlip.cli.errors import CLIError
from refsite_mlip.cli.export_bundle import export_bundle
from refsite_mlip.cli.resolve_train_config import render_resolution_human
from refsite_mlip.cli.resume import resume_training
from refsite_mlip.config import (
    AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V1,
    AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V2,
    AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION,
    AUTOMATIC_EVALUATION_SEMANTIC_PROJECTION_VERSION,
    AutomaticEvaluationPolicyAuditError,
    automatic_evaluation_certificate_semantic_identity,
    qualify_automatic_evaluation_policies,
    resolve_training_recipe,
    validate_automatic_evaluation_certificate,
)
import refsite_mlip.config.automatic_evaluation as automatic_evaluation_module
from refsite_mlip.models import load_reference_site_model_bundle
from refsite_mlip.training import (
    canonical_runtime_json,
    load_training_checkpoint,
    prepare_scratch_training_run,
)

from test_scratch_training_preparation import _atoms, _labeled, _partially_labeled
from test_bundle_predictor_runtime import _save_case
from test_training_recipe_cli import _write_automatic_recipe


_V1_EVALUATION_CERTIFICATE_SHA256 = (
    "31709e0c31c25f012c91dd5c57dce85fb92a0924f41d9162ec3e023677be8dea"
)
_V1_HISTORICAL_POSITION_RELATIVE_DIAGNOSTIC = 8.51007437861865e-08
_PRE_12E1C_V2_EVALUATION_SEMANTIC_SHA256 = (
    "d0d79b4022d2977b6e58c390a4305b92f913df6b693b8cf5c63c464705a2f905"
)
_PRE_12E1C_V2_AUTOMATIC_PREPARATION_SHA256 = (
    "f4e95ac28666a64fa3f418b29ac0f87242570fc5055e889c534a57fe81d39719"
)
_PRE_12E1C_NK_CONFIG_SHA256 = (
    "44857e1e0e66b1a9ecdf785e72a185e6b7d1e5f664ef17d47801d07ea49a7d7c"
)
_V2_EVALUATION_SEMANTIC_SHA256 = (
    "8e6b688f092f150502119315accdb819d3ede5a9f5d8aca7623bd080b0134be5"
)
_V2_AUTOMATIC_PREPARATION_SHA256 = (
    "8793f1113f59ea6ccb492e84d254827b51c4891f5f5ac9e3085f8694fafac498"
)
_NK_CONFIG_SHA256 = (
    "c2a7f9aaf1df383199b0e04713e0bef7506840073f15fd8082028fc524089c65"
)


def _stable_reference():
    reference = _atoms(1)
    # Preserve the builder's cubic reference-cell contract while keeping the
    # transport fixture away from exact inter-sublattice MIC ties.
    fractional = reference.get_scaled_positions(wrap=True)
    fractional[reference.numbers == 6] += np.array([0.017, 0.011, 0.007])
    reference.set_scaled_positions(fractional)
    return reference


def _nk_recipe(
    directory: Path, *, inference: str = "sinkhorn_newton_krylov"
) -> Path:
    reference = _stable_reference()
    vacancy = reference.copy()
    del vacancy[0]
    recipe = _write_automatic_recipe(
        directory,
        references=(("POSCAR", reference, "alpha", "auto"),),
        train=(("train.xyz", (_labeled(reference, -8.0),), "alpha"),),
        validation=(("validation.xyz", (_partially_labeled(vacancy, -6.4),), "alpha"),),
        inference=inference,
    )
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    payload["radii"]["r_ot"] = 2.9
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return recipe


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert tuple(left) == tuple(right)
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_tree_equal(left_item, right_item)
    else:
        assert left == right


def _plain_copy(value):
    return json.loads(canonical_runtime_json(value))


def _rehash_evaluation_certificate(certificate):
    result = _plain_copy(certificate)
    result.pop("evaluation_certificate_sha256", None)
    result["evaluation_certificate_sha256"] = hashlib.sha256(
        canonical_runtime_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _as_pre_12e1c_v2_evaluation_certificate(certificate, structural):
    result = _plain_copy(certificate)
    result["audit_convention_version"] = "automatic_evaluation_policy_audit_v1"
    result["profile"] = _plain_copy(
        automatic_evaluation_module.automatic_evaluation_policy_profile(
            audit_version="automatic_evaluation_policy_audit_v1"
        )
    )
    for records in result["dtype_diagnostics"].values():
        for record in records:
            record.pop("frozen_float64_oracle_maximum_errors", None)
            record.pop("frozen_float64_oracle_residuals", None)
            record.pop("fixed_sinkhorn_float32_diagnostic", None)
    result["observed_extrema"].pop("maximum_oracle_residual", None)
    result["observed_extrema"]["maximum_oracle_error"] = max(
        max(item["oracle_maximum_errors"].values())
        for item in result["dtype_diagnostics"]["float64"]
    )
    historical_structural = _plain_copy(structural)
    historical_structural["evaluation_policy"]["audit_profile"] = (
        "automatic_evaluation_policy_audit_v1"
    )
    historical_structural.pop("certificate_sha256", None)
    historical_structural["certificate_sha256"] = hashlib.sha256(
        canonical_runtime_json(historical_structural).encode("utf-8")
    ).hexdigest()
    result["structural_certificate_sha256"] = historical_structural[
        "certificate_sha256"
    ]
    for key in (
        "normative_outcomes",
        "evaluation_semantic_fingerprint_sha256",
        "evaluation_certificate_sha256",
    ):
        result.pop(key, None)
    outcomes = automatic_evaluation_module._normative_outcomes(result)
    automatic_evaluation_module._assert_qualified_outcomes(outcomes)
    result["normative_outcomes"] = outcomes
    result["evaluation_semantic_fingerprint_sha256"] = hashlib.sha256(
        canonical_runtime_json(
            automatic_evaluation_module._semantic_projection(result, outcomes)
        ).encode("utf-8")
    ).hexdigest()
    return _rehash_evaluation_certificate(result)


def _as_legacy_v1_evaluation_certificate(certificate, structural):
    result = _as_pre_12e1c_v2_evaluation_certificate(
        certificate, structural
    )
    result["schema_version"] = (
        AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V1
    )
    for key in (
        "semantic_projection_version",
        "normative_outcomes",
        "evaluation_semantic_fingerprint_sha256",
        "evaluation_certificate_sha256",
    ):
        result.pop(key, None)
    for probe in result["derivative_probes"]:
        probe.pop("branch_agreement", None)
    # Reconstruct the already-released cold-process v1 artifact.  This is a
    # test fixture value, not production quantization: v1 remains a byte-exact
    # integrity format and is never rewritten by the decoder.
    result["derivative_probes"][2][
        "position_maximum_relative_error"
    ] = _V1_HISTORICAL_POSITION_RELATIVE_DIAGNOSTIC
    return _rehash_evaluation_certificate(result)


def test_poscar_only_newton_krylov_policy_audit_is_deterministic_and_geometry_only(
    tmp_path,
):
    recipe = _nk_recipe(tmp_path)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    grad_enabled = torch.is_grad_enabled()
    inference_enabled = torch.is_inference_mode_enabled()
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()

    first = resolve_training_recipe(recipe)
    with torch.no_grad():
        second = resolve_training_recipe(recipe)
    assert first.config.to_dict() == second.config.to_dict()
    assert first.automatic_reference_preparation.to_dict() == (
        second.automatic_reference_preparation.to_dict()
    )
    result = first.automatic_reference_preparation.results[0]
    assert result.specification.evaluation_policy is not None
    assert result.specification.evaluation_policy.content_fingerprint == (
        "91813ca851baec9cd98b8f6a2cb4b8a9020c8c92669d97e8b7621a8b355e02ca"
    )
    certificate = result.evaluation_certificate
    assert certificate is not None
    assert certificate["schema_version"] == (
        AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V2
    )
    assert certificate["semantic_projection_version"] == (
        AUTOMATIC_EVALUATION_SEMANTIC_PROJECTION_VERSION
    )
    assert certificate["evaluation_semantic_fingerprint_sha256"] == (
        _V2_EVALUATION_SEMANTIC_SHA256
    )
    assert certificate["evaluation_certificate_sha256"] != (
        certificate["evaluation_semantic_fingerprint_sha256"]
    )
    assert validate_automatic_evaluation_certificate(certificate) == (
        _plain_copy(certificate)
    )
    assert first.automatic_reference_preparation.content_fingerprint == (
        _V2_AUTOMATIC_PREPARATION_SHA256
    )
    assert first.config.content_fingerprint == _NK_CONFIG_SHA256
    manifest_reference = dict(
        first.manifest.automatic_reference_certificates
    )[result.template_id]
    assert manifest_reference["evaluation_certificate"] == (
        automatic_evaluation_certificate_semantic_identity(certificate)
    )
    assert certificate["audit_convention_version"] == (
        AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION
    )
    assert certificate["status"] == "qualified"
    assert certificate["scope"] == "assigned_dataset_local_neighborhood"
    assert certificate["phase_approval"] == "provisional"
    assert certificate["differentiability_scope"] == (
        "selected_branch_first_order"
    )
    assert certificate["fallback_count"] == 0
    assert set(certificate["dtype_diagnostics"]) == {"float32", "float64"}
    assert all(
        not item["fallback_used"]
        for values in certificate["dtype_diagnostics"].values()
        for item in values
    )
    tolerances = {"float64": 1e-12, "float32": 1e-6}
    records = certificate["dtype_diagnostics"]
    for dtype_name, values in records.items():
        assert all(item["backend"] == "dense" for item in values)
        assert all(item["dense_plan_materialized"] for item in values)
        assert all(
            max(
                item["transport_row_residual"],
                item["transport_column_residual"],
                item["q_mass_error"],
            )
            <= tolerances[dtype_name]
            for item in values
        )
        assert all(
            max(item["frozen_float64_oracle_residuals"].values()) <= 1e-12
            for item in values
        )
        assert all(
            max(item["frozen_float64_oracle_maximum_errors"].values())
            <= (1e-5 if dtype_name == "float32" else 1e-10)
            for item in values
        )
    assert all(
        item["fixed_sinkhorn_float32_diagnostic"]["normative"] is False
        for item in records["float32"]
    )
    assert all(
        item["fixed_sinkhorn_float32_diagnostic"] is None
        for item in records["float64"]
    )
    f64 = {
        (item["split"], item["sample_id"]): item
        for item in records["float64"]
    }
    for item in records["float32"]:
        other = f64[(item["split"], item["sample_id"])]
        assert item["selected_group"] == other["selected_group"]
        assert item["semantic_support_fingerprint"] == other[
            "semantic_support_fingerprint"
        ]
        assert item["mic_branch_fingerprint"] == other[
            "mic_branch_fingerprint"
        ]
    assert certificate["observed_extrema"]["maximum_position_fd_error"] <= 5e-6
    assert certificate["observed_extrema"]["maximum_strain_fd_error"] <= 5e-6
    rendered = render_resolution_human(first)
    for expected in (
        "Evaluation solver: sinkhorn_newton_krylov",
        "Policy status: qualified",
        "Scope: assigned_dataset_local_neighborhood",
        "Phase approval: provisional",
        "Fallback count: 0",
        "Production/MD guarantee: no",
        "No training was executed.",
    ):
        assert expected in rendered

    original_data = prepare_scratch_training_run(
        first.config,
        automatic_reference_preparation=(
            first.automatic_reference_preparation.to_dict()
        ),
    )

    # Labels and masks are deliberately outside the automatic policy audit.
    from ase.io import read, write

    frames = read(tmp_path / "train.xyz", index=":")
    relabeled = _partially_labeled(frames[0], 12345.0)
    relabeled.arrays["force_mask"][:] = False
    relabeled.info["stress_mask"] = np.ones(6, dtype=bool)
    write(tmp_path / "train.xyz", [relabeled], format="extxyz")
    changed_labels = resolve_training_recipe(recipe)
    changed = changed_labels.automatic_reference_preparation.results[0]
    assert changed.specification.evaluation_policy.content_fingerprint == (
        result.specification.evaluation_policy.content_fingerprint
    )
    assert changed.evaluation_certificate[
        "evaluation_semantic_fingerprint_sha256"
    ] == certificate["evaluation_semantic_fingerprint_sha256"]
    assert changed_labels.automatic_reference_preparation.content_fingerprint == (
        first.automatic_reference_preparation.content_fingerprint
    )
    assert changed_labels.manifest.content_fingerprint == (
        first.manifest.content_fingerprint
    )
    changed_data = prepare_scratch_training_run(
        changed_labels.config,
        automatic_reference_preparation=(
            changed_labels.automatic_reference_preparation.to_dict()
        ),
    )
    assert changed_data.train_semantic_digest != original_data.train_semantic_digest
    assert changed_data.validation_semantic_digest == (
        original_data.validation_semantic_digest
    )

    # Source locations and authored path spelling are provenance, not policy
    # semantics.  Move every audited input and resolve the same geometry again.
    relocated_poscar = tmp_path / "relocated-reference.POSCAR"
    relocated_train = tmp_path / "relocated-train.xyz"
    relocated_validation = tmp_path / "relocated-validation.xyz"
    (tmp_path / "POSCAR").rename(relocated_poscar)
    (tmp_path / "train.xyz").rename(relocated_train)
    (tmp_path / "validation.xyz").rename(relocated_validation)
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    payload["reference"]["sources"][0]["poscar"] = relocated_poscar.name
    payload["data"]["train"][0]["path"] = relocated_train.name
    payload["data"]["validation"][0]["path"] = relocated_validation.name
    relocated_recipe = tmp_path / "relocated.yaml"
    relocated_recipe.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    relocated = resolve_training_recipe(relocated_recipe)
    relocated_preparation = relocated.automatic_reference_preparation
    relocated_result = relocated_preparation.results[0]
    assert relocated_preparation.content_fingerprint == (
        first.automatic_reference_preparation.content_fingerprint
    )
    assert relocated_result.specification.evaluation_policy.content_fingerprint == (
        result.specification.evaluation_policy.content_fingerprint
    )
    assert relocated_result.evaluation_certificate[
        "evaluation_semantic_fingerprint_sha256"
    ] == certificate["evaluation_semantic_fingerprint_sha256"]

    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_default_dtype() == default_dtype
    assert torch.is_grad_enabled() == grad_enabled
    assert torch.is_inference_mode_enabled() == inference_enabled
    assert torch.are_deterministic_algorithms_enabled() == deterministic_enabled
    assert not (tmp_path / "runs" / "automatic-run").exists()

def test_mixed_template_policy_audit_records_k0_k1_k2_and_is_order_independent(
    tmp_path,
):
    alpha = _stable_reference()
    zeta = alpha.repeat((2, 1, 1))
    alpha_k1 = alpha.copy()
    del alpha_k1[0]
    zeta_k2 = zeta.copy()
    del zeta_k2[:2]
    recipe = _write_automatic_recipe(
        tmp_path,
        references=(
            ("POSCAR_a", alpha, "alpha", "auto"),
            ("POSCAR_z", zeta, "zeta", "auto"),
        ),
        train=(
            ("train_a.xyz", (_labeled(alpha, -8.0), _labeled(alpha_k1, -6.5)), "alpha"),
            ("train_z.xyz", (_labeled(zeta, -16.0), _labeled(zeta_k2, -13.0)), "zeta"),
        ),
        validation=(
            ("valid_a.xyz", (_labeled(alpha_k1, -6.4),), "alpha"),
            ("valid_z.xyz", (_labeled(zeta, -15.8),), "zeta"),
        ),
        inference="sinkhorn_newton_krylov",
    )
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    payload["radii"]["r_ot"] = 2.9
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    first = resolve_training_recipe(recipe)
    automatic = first.automatic_reference_preparation
    assert [result.template_id for result in automatic.results] == ["alpha", "zeta"]
    by_id = {result.template_id: result for result in automatic.results}
    assert by_id["alpha"].certificate["vacancies"]["train"]["observed_K_values"] == (0, 1)
    assert by_id["zeta"].certificate["vacancies"]["train"]["observed_K_values"] == (0, 2)
    assert all(result.evaluation_certificate["fallback_count"] == 0 for result in automatic.results)

    payload["reference"]["sources"].reverse()
    reversed_recipe = tmp_path / "reversed.yaml"
    reversed_recipe.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    second = resolve_training_recipe(reversed_recipe)
    assert first.config.canonical_json() == second.config.canonical_json()
    assert automatic.to_dict() == second.automatic_reference_preparation.to_dict()


def test_policy_certificate_semantic_identity_is_exact_across_thread_and_process_boundaries(
    tmp_path,
):
    recipe = _nk_recipe(tmp_path)

    def resolve_identity() -> dict[str, str]:
        result = resolve_training_recipe(recipe)
        certificate = result.automatic_reference_preparation.to_dict()[
            "references"
        ][0]["evaluation_certificate"]
        return {
            "semantic": certificate[
                "evaluation_semantic_fingerprint_sha256"
            ],
            "preparation": result.automatic_reference_preparation.content_fingerprint,
            "config": result.config.content_fingerprint,
            "status": certificate["status"],
        }

    expected = resolve_identity()
    assert expected == {
        "semantic": _V2_EVALUATION_SEMANTIC_SHA256,
        "preparation": _V2_AUTOMATIC_PREPARATION_SHA256,
        "config": _NK_CONFIG_SHA256,
        "status": "qualified",
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(resolve_identity).result() == expected

    command = (
        "import json,sys; "
        "from refsite_mlip.config import resolve_training_recipe; "
        "r=resolve_training_recipe(sys.argv[1]); "
        "c=r.automatic_reference_preparation.to_dict()['references'][0]"
        "['evaluation_certificate']; "
        "print(json.dumps({'semantic':c['evaluation_semantic_fingerprint_sha256'],"
        "'preparation':r.automatic_reference_preparation.content_fingerprint,"
        "'config':r.config.content_fingerprint,'status':c['status']},"
        "sort_keys=True,separators=(',',':')))"
    )
    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", command, str(recipe)],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads(completed.stdout.strip()))
    assert outputs == [expected, expected]


def test_v2_certificate_separates_semantic_identity_from_raw_integrity(tmp_path):
    resolution = resolve_training_recipe(_nk_recipe(tmp_path))
    certificate = resolution.automatic_reference_preparation.to_dict()[
        "references"
    ][0]["evaluation_certificate"]
    assert automatic_evaluation_certificate_semantic_identity(certificate) == {
        "schema_version": AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V2,
        "audit_convention_version": AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION,
        "semantic_projection_version": (
            AUTOMATIC_EVALUATION_SEMANTIC_PROJECTION_VERSION
        ),
        "evaluation_semantic_fingerprint_sha256": (
            _V2_EVALUATION_SEMANTIC_SHA256
        ),
    }

    raw_only = _plain_copy(certificate)
    raw_only["derivative_probes"][0][
        "position_maximum_relative_error"
    ] += 1.0e-9
    raw_only = _rehash_evaluation_certificate(raw_only)
    assert raw_only["evaluation_certificate_sha256"] != (
        certificate["evaluation_certificate_sha256"]
    )
    assert validate_automatic_evaluation_certificate(raw_only)[
        "evaluation_semantic_fingerprint_sha256"
    ] == certificate["evaluation_semantic_fingerprint_sha256"]
    original_result = resolution.automatic_reference_preparation.results[0]
    raw_only_result = replace(
        original_result, evaluation_certificate=raw_only
    )
    assert raw_only_result.semantic_dict() == original_result.semantic_dict()

    fixed_cycle_only = _plain_copy(certificate)
    fixed_cycle_only["dtype_diagnostics"]["float32"][0][
        "fixed_sinkhorn_float32_diagnostic"
    ]["terminal_residuals"]["row"] += 1.0e-7
    fixed_cycle_only = _rehash_evaluation_certificate(fixed_cycle_only)
    assert validate_automatic_evaluation_certificate(fixed_cycle_only)[
        "evaluation_semantic_fingerprint_sha256"
    ] == certificate["evaluation_semantic_fingerprint_sha256"]

    same_outcome = _plain_copy(certificate)
    objective_record = same_outcome["dtype_diagnostics"]["float64"][0]
    objective_record["objective_gap"] *= 1.0000001
    same_outcome = _rehash_evaluation_certificate(same_outcome)
    assert validate_automatic_evaluation_certificate(same_outcome)[
        "evaluation_semantic_fingerprint_sha256"
    ] == certificate["evaluation_semantic_fingerprint_sha256"]

    integrity_corruption = _plain_copy(certificate)
    integrity_corruption["derivative_probes"][0][
        "position_maximum_relative_error"
    ] += 1.0e-9
    with pytest.raises(AutomaticEvaluationPolicyAuditError) as integrity:
        validate_automatic_evaluation_certificate(integrity_corruption)
    assert integrity.value.reason_code == (
        "EVALUATION_CERTIFICATE_INTEGRITY_MISMATCH"
    )

    threshold_corruption = _plain_copy(certificate)
    threshold_corruption["dtype_diagnostics"]["float64"][0][
        "objective_gap"
    ] = 0.0
    threshold_corruption = _rehash_evaluation_certificate(
        threshold_corruption
    )
    with pytest.raises(AutomaticEvaluationPolicyAuditError) as qualification:
        validate_automatic_evaluation_certificate(threshold_corruption)
    assert qualification.value.reason_code == (
        "EVALUATION_CERTIFICATE_QUALIFICATION_FAILED"
    )

    for path, value in (
        (("transport_row_residual",), 1.1e-6),
        (("frozen_float64_oracle_residuals", "row"), 1.1e-12),
        (("frozen_float64_oracle_maximum_errors", "plan"), 1.1e-5),
        (("frozen_float64_oracle_maximum_errors", "q"), 1.1e-5),
        (("frozen_float64_oracle_maximum_errors", "multipoles"), 1.1e-5),
    ):
        corrupted = _plain_copy(certificate)
        record = corrupted["dtype_diagnostics"]["float32"][0]
        if len(path) == 1:
            record[path[0]] = value
        else:
            record[path[0]][path[1]] = value
        corrupted = _rehash_evaluation_certificate(corrupted)
        with pytest.raises(AutomaticEvaluationPolicyAuditError) as caught:
            validate_automatic_evaluation_certificate(corrupted)
        assert caught.value.reason_code == (
            "EVALUATION_CERTIFICATE_QUALIFICATION_FAILED"
        )

    for field, value in (
        ("selected_group", 999),
        ("semantic_support_fingerprint", "0" * 64),
        ("fallback_used", True),
    ):
        categorical = _plain_copy(certificate)
        categorical["dtype_diagnostics"]["float64"][0][field] = value
        categorical = _rehash_evaluation_certificate(categorical)
        with pytest.raises(AutomaticEvaluationPolicyAuditError):
            validate_automatic_evaluation_certificate(categorical)


def test_legacy_v1_evaluation_certificate_sha_and_decoder_are_preserved(tmp_path):
    result = resolve_training_recipe(
        _nk_recipe(tmp_path)
    ).automatic_reference_preparation.to_dict()["references"][0]
    structural = {
        key: value
        for key, value in result.items()
        if key not in {"poscar", "evaluation_certificate"}
    }
    legacy = _as_legacy_v1_evaluation_certificate(
        result["evaluation_certificate"], structural
    )
    assert legacy["evaluation_certificate_sha256"] == (
        _V1_EVALUATION_CERTIFICATE_SHA256
    )
    assert validate_automatic_evaluation_certificate(legacy) == legacy
    assert automatic_evaluation_certificate_semantic_identity(legacy) == legacy


def test_pre_12e1c_v2_certificate_loads_without_migration_or_relabeling(
    tmp_path,
):
    result = resolve_training_recipe(
        _nk_recipe(tmp_path)
    ).automatic_reference_preparation.to_dict()["references"][0]
    structural = {
        key: value
        for key, value in result.items()
        if key not in {"poscar", "evaluation_certificate"}
    }
    legacy = _as_pre_12e1c_v2_evaluation_certificate(
        result["evaluation_certificate"], structural
    )
    assert legacy["audit_convention_version"] == (
        "automatic_evaluation_policy_audit_v1"
    )
    assert legacy["evaluation_semantic_fingerprint_sha256"] == (
        _PRE_12E1C_V2_EVALUATION_SEMANTIC_SHA256
    )
    assert validate_automatic_evaluation_certificate(legacy) == legacy
    identity = automatic_evaluation_certificate_semantic_identity(legacy)
    assert identity["audit_convention_version"] == (
        "automatic_evaluation_policy_audit_v1"
    )
    assert identity["evaluation_semantic_fingerprint_sha256"] == (
        _PRE_12E1C_V2_EVALUATION_SEMANTIC_SHA256
    )


def test_frozen_float64_oracle_residual_failure_is_immediate(
    tmp_path, monkeypatch
):
    resolution = resolve_training_recipe(_nk_recipe(tmp_path))
    preparation = resolution.automatic_reference_preparation
    audit_input = preparation._audit_inputs[0]
    geometry = audit_input.train_geometries[0]
    policy = preparation.results[0].specification.evaluation_policy
    potential = resolution.config.model_source.potential
    original = automatic_evaluation_module._frozen_float64_dense_oracle

    def failed(problem):
        promoted, result, residuals = original(problem)
        return promoted, result, {**residuals, "row": 1.1e-12}

    monkeypatch.setattr(
        automatic_evaluation_module,
        "_frozen_float64_dense_oracle",
        failed,
    )
    with pytest.raises(AutomaticEvaluationPolicyAuditError) as caught:
        automatic_evaluation_module._evaluate(
            audit_input,
            geometry,
            policy,
            potential,
            torch.float32,
        )
    assert caught.value.reason_code == (
        "TRANSPORT_ORACLE_CONVERGENCE_FAILURE"
    )
    assert caught.value.observed == pytest.approx(1.1e-12)
    assert caught.value.required == "<= 1e-12"


def test_production_float32_nk_fallback_residual_and_oracle_mismatch_fail_closed(
    tmp_path, monkeypatch
):
    resolution = resolve_training_recipe(_nk_recipe(tmp_path))
    preparation = resolution.automatic_reference_preparation
    audit_input = preparation._audit_inputs[0]
    geometry = audit_input.train_geometries[0]
    policy = preparation.results[0].specification.evaluation_policy
    potential = resolution.config.model_source.potential
    original_solve = automatic_evaluation_module.solve_atom_vacancy_ot

    def evaluate(*, oracle=True):
        return automatic_evaluation_module._evaluate(
            audit_input,
            geometry,
            policy,
            potential,
            torch.float32,
            oracle=oracle,
        )

    with monkeypatch.context() as current:
        def fallback(*args, **kwargs):
            result = original_solve(*args, **kwargs)
            return replace(result, fallback_used=True)

        current.setattr(
            automatic_evaluation_module, "solve_atom_vacancy_ot", fallback
        )
        with pytest.raises(AutomaticEvaluationPolicyAuditError) as caught:
            evaluate(oracle=False)
        assert caught.value.reason_code == "ADAPTIVE_TRANSPORT_FALLBACK"

    with monkeypatch.context() as current:
        def unconverged(*args, **kwargs):
            result = original_solve(*args, **kwargs)
            return replace(
                result,
                row_residual=result.row_residual.new_tensor(1.1e-6),
            )

        current.setattr(
            automatic_evaluation_module,
            "solve_atom_vacancy_ot",
            unconverged,
        )
        with pytest.raises(AutomaticEvaluationPolicyAuditError) as caught:
            evaluate(oracle=False)
        assert caught.value.reason_code == "TRANSPORT_CONVERGENCE_FAILURE"

    with monkeypatch.context() as current:
        current.setattr(
            automatic_evaluation_module,
            "_tensor_maximum_error",
            lambda _left, _right: 1.1e-5,
        )
        with pytest.raises(AutomaticEvaluationPolicyAuditError) as caught:
            evaluate()
        assert caught.value.reason_code == "TRANSPORT_ORACLE_MISMATCH"


def test_cold_and_ase_first_share_v2_semantic_identity(
    typed_crystal, tmp_path
):
    recipe_root = tmp_path / "recipe"
    recipe_root.mkdir()
    recipe = _nk_recipe(recipe_root)
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    *_, bundle_path = _save_case(typed_crystal, bundle_root)
    program = textwrap.dedent(
        """
        import json
        import sys
        import numpy as np
        import torch
        from ase import Atoms
        from refsite_mlip.config import resolve_training_recipe
        from refsite_mlip.interfaces import ReferenceSiteASECalculator

        recipe, bundle, order = sys.argv[1:]

        def certificate():
            resolution = resolve_training_recipe(recipe)
            cert = resolution.automatic_reference_preparation.to_dict()[\
                "references"][0]["evaluation_certificate"]
            return {
                "status": cert["status"],
                "semantic": cert["evaluation_semantic_fingerprint_sha256"],
                "content": cert["evaluation_certificate_sha256"],
                "preparation": resolution.automatic_reference_preparation.content_fingerprint,
                "config": resolution.config.content_fingerprint,
                "policy": cert["policy_fingerprint"],
                "outcomes": cert["normative_outcomes"],
                "position_relative": [
                    item["position_maximum_relative_error"]
                    for item in cert["derivative_probes"]
                ],
            }

        def ase_force_stress():
            calculator = ReferenceSiteASECalculator(
                bundle,
                template_id="zeta",
                dtype=torch.float64,
                solver_path="train_fixed",
            )
            template = calculator.predictor.registry.resolve("zeta")
            site_types = template.topology.site_types.detach().cpu().numpy()
            vocabulary = np.asarray(template.supported_species, dtype=np.int64)
            atoms = Atoms(
                numbers=vocabulary[site_types],
                positions=(
                    template.topology.reference_fractional
                    @ template.topology.reference_cell
                ).detach().cpu().numpy(),
                cell=template.topology.reference_cell.detach().cpu().numpy(),
                pbc=True,
            )
            atoms.calc = calculator
            assert np.isfinite(atoms.get_forces()).all()
            assert np.isfinite(atoms.get_stress()).all()

        if order == "cold":
            values = [certificate()]
        elif order == "certificate-ase-certificate":
            values = [certificate()]
            ase_force_stress()
            values.append(certificate())
        else:
            ase_force_stress()
            values = [certificate()]
        print(json.dumps(values, sort_keys=True, separators=(",", ":")))
        """
    )

    def run(order):
        completed = subprocess.run(
            [sys.executable, "-c", program, str(recipe), str(bundle_path), order],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout.strip())

    cold = run("cold")[0]
    around = run("certificate-ase-certificate")
    ase_first = run("ase-certificate")[0]
    for value in (cold, *around, ase_first):
        assert value["status"] == "qualified"
        assert value["semantic"] == _V2_EVALUATION_SEMANTIC_SHA256
        assert value["preparation"] == _V2_AUTOMATIC_PREPARATION_SHA256
        assert value["config"] == _NK_CONFIG_SHA256
        assert value["policy"] == (
            "91813ca851baec9cd98b8f6a2cb4b8a9020c8c92669d97e8b7621a8b355e02ca"
        )
        assert value["outcomes"] == cold["outcomes"]
    if cold["content"] != ase_first["content"]:
        assert cold["position_relative"] != ase_first["position_relative"]


def test_full_base_census_exceeds_315_while_only_witnesses_are_budgeted(
    monkeypatch,
):
    import refsite_mlip.config.automatic_evaluation as audit_module

    def geometry(sample_id, split):
        digest = hashlib.sha256(
            f"{split}:{sample_id}".encode("utf-8")
        ).hexdigest()
        return SimpleNamespace(
            sample_id=sample_id,
            split=split,
            semantic_digest=digest,
        )

    reference = geometry("ideal-pristine", "reference")
    train = tuple(geometry(f"train-{index:04d}", "train") for index in range(316))
    validation = tuple(
        geometry(f"validation-{index:04d}", "validation")
        for index in range(2)
    )
    candidate = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    coverage = audit_module._CandidateCoverage(
        runtime_candidates=candidate,
        audit_candidates=candidate,
        metadata=MappingProxyType(
            {
                "runtime_candidate_fingerprint": "runtime",
                "broader_audit_candidate_fingerprint": "audit",
            }
        ),
    )
    policy = SimpleNamespace(
        candidate_offsets=candidate,
        content_fingerprint="policy-fingerprint",
    )
    support = SimpleNamespace(backend="dense", to_dict=lambda: {"backend": "dense"})
    config = SimpleNamespace(
        transport_support=support,
        eval_sinkhorn_warmup_iterations=8,
    )
    audit_input = SimpleNamespace(
        template_id="census-template",
        context=SimpleNamespace(fingerprint="template-fingerprint"),
        reference_geometry=reference,
        train_geometries=train,
        validation_geometries=validation,
        ideal_k1_geometries=(),
    )
    visits = []

    def fake_evaluate(_audit_input, item, _policy, _config, dtype, **_kwargs):
        visits.append((str(dtype), item.split, item.sample_id))
        diagnostics = {
            "sample_id": item.sample_id,
            "geometry_digest": item.semantic_digest,
            "split": item.split,
            "K": 0,
            "composition": [2],
            "dtype": str(dtype).removeprefix("torch."),
            "backend": "dense",
            "selected_group": 0,
            "selected_candidate": 0,
            "objective_gap": 1.0,
            "minimum_atomic_amplitude": 1.0,
            "minimum_reference_amplitude": 1.0,
            "minimum_cross_amplitude": 1.0,
            "hessian_minimum_curvature": 1.0,
            "hessian_condition": 1.0,
            "phase_residual": 0.0,
            "support_margin": 1.0,
            "mic_margin": 1.0,
            "production_support_fingerprint": "support",
            "semantic_support_fingerprint": "support",
            "mic_branch_fingerprint": "mic",
            "transport_row_residual": 0.0,
            "transport_column_residual": 0.0,
            "q_mass_error": 0.0,
            "warmup_sinkhorn_iterations": 8,
            "newton_iterations": 1,
            "cg_iterations": 1,
            "line_search_reductions": 0,
            "total_transport_work": 10,
            "fallback_used": False,
            "observed_strain": 0.0,
            "oracle_maximum_errors": {
                "plan": 0.0,
                "q": 0.0,
                "multipoles": 0.0,
            },
            "frozen_float64_oracle_maximum_errors": {
                "plan": 0.0,
                "q": 0.0,
                "multipoles": 0.0,
            },
            "frozen_float64_oracle_residuals": {
                "row": 0.0,
                "column": 0.0,
                "q_mass": 0.0,
            },
            "fixed_sinkhorn_float32_diagnostic": None,
            "dense_plan_materialized": True,
        }
        zero = torch.zeros((), dtype=dtype)
        return audit_module._Outcome(
            scalar=zero,
            phase=torch.zeros(3, dtype=dtype),
            edge_or_plan=zero,
            q=zero,
            multipoles=zero,
            diagnostics=MappingProxyType(diagnostics),
            branch_signature=(0, "support", "mic", "dense", False),
        )

    def fake_probe(_audit_input, item, _policy, _config):
        return {
            "sample_id": item.sample_id,
            "geometry_digest": item.semantic_digest,
            "position_directions": 1,
            "strain_directions": 6,
            "position_maximum_absolute_error": 0.0,
            "position_maximum_relative_error": 0.0,
            "strain_maximum_absolute_error": 0.0,
            "strain_maximum_relative_error": 0.0,
            "symmetry_probes": {},
        }

    def fake_coverage(_audit_input, item, _policy, _coverage):
        return {
            "sample_id": item.sample_id,
            "geometry_digest": item.semantic_digest,
            "runtime_selected_group": 0,
            "selected_canonical_group": [0.0, 0.0, 0.0],
            "broader_selected_candidate": 0,
            "broader_non_equivalent_gap": None,
            "broader_minus_runtime_objective": 0.0,
            "same_stabilizer_group": True,
        }

    monkeypatch.setattr(audit_module, "_candidate_coverage", lambda _context: coverage)
    monkeypatch.setattr(audit_module, "_evaluate", fake_evaluate)
    monkeypatch.setattr(audit_module, "_probe_witness", fake_probe)
    monkeypatch.setattr(audit_module, "_phase_candidate_coverage", fake_coverage)

    first = audit_module._audit_one(audit_input, policy, config)
    assert len(visits) == 2 * (316 + 2 + 1)
    counts = first["audit_input"]["census_counts"]
    assert counts == {
        "total_train_census_count": 316,
        "total_validation_census_count": 2,
        "ideal_reference_census_count": 1,
        "ideal_pristine_census_count": 1,
        "ideal_k1_census_count": 0,
        "total_base_census": 319,
        "unique_semantic_geometry_count": 319,
    }
    witnesses = first["witness_selection"]
    assert witnesses["selected_witness_count"] == 1
    assert witnesses["max_witness_geometries"] == 256
    assert witnesses["executed_position_probe_count"] == 1
    assert witnesses["executed_strain_probe_count"] == 6

    visits.clear()
    reordered = SimpleNamespace(
        **{
            **audit_input.__dict__,
            "train_geometries": tuple(reversed(train)),
            "validation_geometries": tuple(reversed(validation)),
        }
    )
    second = audit_module._audit_one(reordered, policy, config)
    assert len(visits) == 2 * (316 + 2 + 1)
    assert second == first

    monkeypatch.setattr(
        audit_module,
        "_limiting_witnesses",
        lambda _records: tuple(f"digest-{index:03d}" for index in range(257)),
    )
    with pytest.raises(AutomaticEvaluationPolicyAuditError) as caught:
        audit_module._audit_one(audit_input, policy, config)
    assert caught.value.reason_code == "AUDIT_BUDGET_EXCEEDED"
    assert caught.value.observed == 257
    assert caught.value.required == "<= 256"


def test_newton_krylov_policy_is_materialized_and_public_solver_names_are_canonical(
    tmp_path, capsys, monkeypatch
):
    recipe = _nk_recipe(tmp_path)
    import refsite_mlip.config.automatic_evaluation as audit_module

    original_qualifier = audit_module.qualify_automatic_evaluation_policies
    qualifier_calls = []

    def counted_qualifier(*args, **kwargs):
        qualifier_calls.append(1)
        return original_qualifier(*args, **kwargs)

    monkeypatch.setattr(
        audit_module, "qualify_automatic_evaluation_policies", counted_qualifier
    )
    assert main(["train", str(recipe), "--json", "--quiet"]) == 0
    assert len(qualifier_calls) == 1
    terminal = json.loads(capsys.readouterr().out)
    assert terminal["status"] == "completed"
    output = tmp_path / "runs" / "automatic-run"
    references = output / "references"
    assert {path.name for path in references.iterdir()} == {
        "alpha.reference.json",
        "alpha.certificate.json",
        "alpha.evaluation-certificate.json",
    }
    specification = json.loads(
        (references / "alpha.reference.json").read_text(encoding="utf-8")
    )
    structural = json.loads(
        (references / "alpha.certificate.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (references / "alpha.evaluation-certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert specification["evaluation_policy"] is not None
    assert structural["evaluation_policy"]["status"] == "qualified"
    assert evaluation["structural_certificate_sha256"] == (
        structural["certificate_sha256"]
    )
    assert evaluation["specification_sha256"] == specification["content_fingerprint"]
    assert evaluation["policy_fingerprint"] == (
        specification["evaluation_policy"]["content_fingerprint"]
    )
    bundle = load_reference_site_model_bundle(output / "initial_bundle.pt")
    assert bundle.template_bindings[0].evaluation_policy is not None
    assert bundle.template_bindings[0].evaluation_policy.content_fingerprint == (
        evaluation["policy_fingerprint"]
    )

    # Continuation and export are bound to the materialized reference/bundle;
    # neither may recertify or reread the original POSCAR.
    (tmp_path / "POSCAR").unlink()
    import refsite_mlip.config.automatic_reference as reference_module

    def forbidden(*args, **kwargs):
        raise AssertionError("automatic reference/policy rebuild is forbidden")

    monkeypatch.setattr(
        audit_module, "qualify_automatic_evaluation_policies", forbidden
    )
    monkeypatch.setattr(reference_module, "prepare_automatic_references", forbidden)
    resumed = resume_training(output, max_epochs=2)
    assert resumed["status"] == "completed"
    exported_path = tmp_path / "latest.pt"
    exported_report = export_bundle(
        output, source="latest", output_path=exported_path
    )
    assert exported_report["status"] == "completed"
    best_path = tmp_path / "best.pt"
    best_report = export_bundle(output, source="best", output_path=best_path)
    assert best_report["status"] == "completed"
    exported = load_reference_site_model_bundle(exported_path)
    exported_best = load_reference_site_model_bundle(best_path)
    assert exported.template_bindings[0].evaluation_policy is not None
    assert exported_best.template_bindings[0].evaluation_policy is not None
    predicted = tmp_path / "predicted.xyz"
    assert main(
        [
            "predict",
            "--bundle",
            str(exported_path),
            "--input",
            str(tmp_path / "validation.xyz"),
            "--output",
            str(predicted),
            "--template-id",
            "alpha",
            "--solver",
            "sinkhorn_newton_krylov",
            "--properties",
            "energy,forces,stress",
            "--json",
        ]
    ) == 0
    prediction_report = json.loads(capsys.readouterr().out)
    assert prediction_report["solver"] == "sinkhorn_newton_krylov"
    assert predicted.is_file()
    assert main(
        [
            "evaluate",
            "--bundle",
            str(exported_path),
            "--input",
            str(tmp_path / "validation.xyz"),
            "--template-id",
            "alpha",
            "--solver",
            "sinkhorn_newton_krylov",
            "--terms",
            "energy,forces,stress",
            "--json",
        ]
    ) == 0
    evaluation_report = json.loads(capsys.readouterr().out)
    assert evaluation_report["solver"] == "sinkhorn_newton_krylov"

    checkpoint_bytes = (output / "checkpoints" / "latest.pt").read_bytes()
    journal_bytes = (output / "metrics.jsonl").read_bytes()
    corrupted = dict(evaluation)
    corrupted["policy_fingerprint"] = "0" * 64
    corrupted.pop("evaluation_certificate_sha256")
    corrupted["evaluation_certificate_sha256"] = hashlib.sha256(
        canonical_runtime_json(corrupted).encode("utf-8")
    ).hexdigest()
    (references / "alpha.evaluation-certificate.json").write_text(
        canonical_runtime_json(corrupted) + "\n", encoding="utf-8"
    )
    with pytest.raises(CLIError) as mismatch:
        resume_training(output, max_epochs=3)
    assert mismatch.value.reason_code == "EVALUATION_CERTIFICATE_BINDING_MISMATCH"
    assert (output / "checkpoints" / "latest.pt").read_bytes() == checkpoint_bytes
    assert (output / "metrics.jsonl").read_bytes() == journal_bytes

    parser = build_parser()
    canonical = parser.parse_args(
        [
            "predict", "--bundle", "b.pt", "--input", "i.xyz",
            "--output", "o.xyz", "--solver", "sinkhorn_newton_krylov",
        ]
    )
    deprecated = parser.parse_args(
        [
            "predict", "--bundle", "b.pt", "--input", "i.xyz",
            "--output", "o.xyz", "--solver", "eval-adaptive",
        ]
    )
    assert canonical.solver == deprecated.solver == "sinkhorn_newton_krylov"
    evaluated = parser.parse_args(
        [
            "evaluate", "--bundle", "b.pt", "--input", "i.xyz",
            "--solver", "sinkhorn_newton_krylov",
        ]
    )
    evaluated_deprecated = parser.parse_args(
        [
            "evaluate", "--bundle", "b.pt", "--input", "i.xyz",
            "--solver", "eval-adaptive",
        ]
    )
    assert evaluated.solver == evaluated_deprecated.solver == (
        "sinkhorn_newton_krylov"
    )


def test_sinkhorn_only_automatic_reference_has_no_policy_or_evaluation_certificate(
    tmp_path,
):
    reference = _atoms(1)
    recipe = _write_automatic_recipe(
        tmp_path,
        references=(("POSCAR", reference, "alpha", "auto"),),
        train=(("train.xyz", (_labeled(reference, -8.0),), "alpha"),),
        validation=(("validation.xyz", (_labeled(reference, -7.9),), "alpha"),),
        inference="sinkhorn",
    )
    result = resolve_training_recipe(recipe).automatic_reference_preparation.results[0]
    assert result.specification.evaluation_policy is None
    assert result.evaluation_certificate is None
    assert result.to_dict()["evaluation_policy"] is None
    assert "evaluation_certificate" not in result.to_dict()


def test_inference_policy_selection_does_not_change_fixed_training_trajectory(
    tmp_path, capsys
):
    sinkhorn_root = tmp_path / "sinkhorn"
    adaptive_root = tmp_path / "adaptive"
    sinkhorn_root.mkdir()
    adaptive_root.mkdir()
    sinkhorn_recipe = _nk_recipe(sinkhorn_root, inference="sinkhorn")
    adaptive_recipe = _nk_recipe(adaptive_root)
    assert main(["train", str(sinkhorn_recipe), "--json", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert main(["train", str(adaptive_recipe), "--json", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    sinkhorn_run = sinkhorn_root / "runs" / "automatic-run"
    adaptive_run = adaptive_root / "runs" / "automatic-run"
    sinkhorn_certificate = json.loads(
        (sinkhorn_run / "references" / "alpha.certificate.json").read_text(
            encoding="utf-8"
        )
    )
    # Closure hardening is strictly NK-only.  Lock the pre-existing
    # Sinkhorn-only structural/specification semantic bytes for this fixture.
    assert sinkhorn_certificate["certificate_sha256"] == (
        "ada69da9a9afcafeb8f06c9cb94821531c06984a8230fb0eb50d83a2db0827c5"
    )
    assert sinkhorn_certificate["specification_sha256"] == (
        "7918f8c686c484c08dc1ceea8ed6cf0921d373ec726f0115ac33d4a863a6c16a"
    )
    assert hashlib.sha256(
        (
            sinkhorn_run / "references" / "alpha.certificate.json"
        ).read_bytes()
    ).hexdigest() == (
        "c4f4386c3ca88bb8f1163c1eb13b6252f37d040505ec1ece846a769e3d1bdcf4"
    )
    assert hashlib.sha256(
        (
            sinkhorn_run / "references" / "alpha.reference.json"
        ).read_bytes()
    ).hexdigest() == (
        "395455f2b22d288729d4b9084f1339307fa8c409f72dc27ec3b79ea605008bac"
    )
    fixed = load_training_checkpoint(sinkhorn_run / "checkpoints" / "latest.pt")
    qualified = load_training_checkpoint(adaptive_run / "checkpoints" / "latest.pt")
    _assert_tree_equal(fixed.model_state_dict, qualified.model_state_dict)
    _assert_tree_equal(fixed.optimizer_state_dict, qualified.optimizer_state_dict)
    _assert_tree_equal(fixed.scheduler_state_dict, qualified.scheduler_state_dict)
    assert fixed.selection_state == qualified.selection_state
    assert fixed.progress == qualified.progress
    assert fixed.fit_history == qualified.fit_history
    fixed_event = json.loads((sinkhorn_run / "metrics.jsonl").read_text())
    qualified_event = json.loads((adaptive_run / "metrics.jsonl").read_text())
    fixed_event.pop("provenance")
    qualified_event.pop("provenance")
    assert fixed_event == qualified_event


def test_qualified_policy_continuous_and_resumed_fixed_training_are_exact(
    tmp_path, capsys
):
    continuous_root = tmp_path / "continuous"
    resumed_root = tmp_path / "resumed"
    continuous_root.mkdir()
    resumed_root.mkdir()

    def recipe(root: Path, epochs: int) -> Path:
        path = _nk_recipe(root)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["training"]["max_epochs"] = epochs
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    continuous_recipe = recipe(continuous_root, 2)
    resumed_recipe = recipe(resumed_root, 1)
    assert main(["train", str(continuous_recipe), "--json", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    continuous_run = continuous_root / "runs" / "automatic-run"
    continuous_checkpoint = load_training_checkpoint(
        continuous_run / "checkpoints" / "latest.pt"
    )
    continuous_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    assert main(["train", str(resumed_recipe), "--json", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    resumed_run = resumed_root / "runs" / "automatic-run"
    epoch_zero = resumed_run / "checkpoints" / "epoch_000000.pt"
    epoch_zero_bytes = epoch_zero.read_bytes()
    resumed = resume_training(resumed_run, max_epochs=2)
    assert resumed["status"] == "completed"
    resumed_checkpoint = load_training_checkpoint(
        resumed_run / "checkpoints" / "latest.pt"
    )
    resumed_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    _assert_tree_equal(
        continuous_checkpoint.model_state_dict,
        resumed_checkpoint.model_state_dict,
    )
    _assert_tree_equal(
        continuous_checkpoint.optimizer_state_dict,
        resumed_checkpoint.optimizer_state_dict,
    )
    _assert_tree_equal(
        continuous_checkpoint.scheduler_state_dict,
        resumed_checkpoint.scheduler_state_dict,
    )
    assert continuous_checkpoint.selection_state == resumed_checkpoint.selection_state
    assert continuous_checkpoint.progress == resumed_checkpoint.progress
    assert continuous_checkpoint.fit_history == resumed_checkpoint.fit_history
    assert (continuous_run / "metrics.jsonl").read_bytes() == (
        resumed_run / "metrics.jsonl"
    ).read_bytes()
    assert epoch_zero.read_bytes() == epoch_zero_bytes
    assert continuous_draws[:2] == resumed_draws[:2]
    assert torch.equal(continuous_draws[2], resumed_draws[2])


def test_automatic_policy_audit_exercises_sparse_edge_list_without_densification(
    tmp_path,
):
    resolution = resolve_training_recipe(_nk_recipe(tmp_path))
    preparation = resolution.automatic_reference_preparation
    potential = resolution.config.model_source.potential
    edge_list = replace(potential.transport_support, backend="edge_list")
    sparse = qualify_automatic_evaluation_policies(
        preparation, replace(potential, transport_support=edge_list)
    )
    certificate = sparse.results[0].evaluation_certificate
    assert certificate is not None
    assert certificate["fallback_count"] == 0
    for records in certificate["dtype_diagnostics"].values():
        assert records
        assert all(record["backend"] == "edge_list" for record in records)
        assert all(not record["dense_plan_materialized"] for record in records)
        assert all(record["support_fingerprint"] for record in records)
        assert all(
            max(record["frozen_float64_oracle_residuals"].values())
            <= 1e-12
            for record in records
        )
    for dtype_name, records in certificate["dtype_diagnostics"].items():
        tolerance = 1e-5 if dtype_name == "float32" else 1e-10
        assert all(
            max(record["frozen_float64_oracle_maximum_errors"].values())
            <= tolerance
            for record in records
        )


def test_resolve_validate_and_dry_run_share_the_qualified_policy_snapshot(
    tmp_path, capsys
):
    recipe = _nk_recipe(tmp_path)
    resolved_output = tmp_path / "resolved.json"
    manifest_output = tmp_path / "resolution-manifest.json"
    assert main(
        [
            "resolve-train-config",
            str(recipe),
            "--output",
            str(resolved_output),
            "--manifest",
            str(manifest_output),
            "--dry-run",
            "--json",
        ]
    ) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert main(["validate-train-config", str(recipe), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert main(["train", str(recipe), "--dry-run", "--json", "--quiet"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert resolved["reference_preparation"] == validated["reference_preparation"]
    assert validated == dry_run
    certificate = validated["reference_preparation"]["references"][0][
        "evaluation_certificate"
    ]
    assert certificate["status"] == "qualified"
    assert certificate["fallback_count"] == 0
    assert not resolved_output.exists()
    assert not manifest_output.exists()
    assert not (tmp_path / "runs" / "automatic-run").exists()
