from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("ase")

from refsite_mlip.cli.main import main
from refsite_mlip.config import (
    ReferenceSpecificationConfig,
    load_training_run_config,
    load_reference_specification,
    resolve_training_recipe,
)
from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceTemplateBuilderConfig,
    build_reference_template_from_poscar,
)
from refsite_mlip.models import EvaluationPolicy
from refsite_mlip.training import prepare_scratch_training_run

from test_scratch_training_preparation import _atoms, _case, _labeled


def _write_recipe_case(
    directory: Path, *, yaml_recipe: bool = True, frame_count: int = 1
):
    (directory / "runs").mkdir(exist_ok=True)
    reference = _atoms(1)
    _, poscar, _, canonical = _case(
        directory,
        train_frames=tuple(
            _labeled(reference, -8.0 + 0.01 * index)
            for index in range(frame_count)
        ),
        validation_frames=tuple(
            _labeled(reference, -7.75 + 0.01 * index)
            for index in range(frame_count)
        ),
        selector={"template_id": "scratch-111-a"},
    )
    template = canonical["model_source"]["reference_templates"][0]
    builder = ReferenceTemplateBuilderConfig.from_dict(template["builder"])
    phase = PhaseSpecification.from_dict(template["phase_specification"])
    built = build_reference_template_from_poscar(
        poscar,
        config=builder,
        phase_specification=phase,
    )
    policy = EvaluationPolicy(
        template_id=builder.template_id,
        template_fingerprint=built.template.fingerprint,
        candidate_offsets=torch.tensor(
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=torch.float64
        ),
        phase_step_schedule=(0.5,),
        phase_damping_schedule=(1.0,),
        minimum_objective_gap_absolute=1.0e-8,
        minimum_cross_amplitude_absolute=1.0e-8,
        minimum_atomic_amplitude_absolute=1.0e-8,
        minimum_reference_amplitude_absolute=1.0e-8,
        minimum_curvature=1.0e-8,
        maximum_condition=1.0e6,
        maximum_gradient_norm=1.0e-6,
        equivalence_tolerance=1.0e-10,
    )
    specification = ReferenceSpecificationConfig(
        builder=builder,
        phase_specification=phase,
        evaluation_policy=policy,
        species_alignment_weights=tuple(
            tuple(row) for row in canonical["model_source"]["species_alignment_weights"]
        ),
        poscar_sha256=hashlib.sha256(poscar.read_bytes()).hexdigest(),
    )
    specification_path = directory / "reference-specification.yaml"
    specification_path.write_text(
        yaml.safe_dump(specification.to_dict(), sort_keys=False), encoding="utf-8"
    )
    recipe = {
        "schema_version": "refsite_training_recipe_v1",
        "name": "recipe-run",
        "model": {
            "num_interactions": 1,
            "correlation": 1,
            "hidden_channels": 1,
            "max_L": 2,
            "initialization_seed": 19,
        },
        "reference": {
            "specification": specification_path.name,
            "poscar": poscar.name,
            "allow_provisional_phase": True,
        },
        "data": {"train": "train.xyz", "validation": "validation.xyz"},
        "ot_solver": {
            "training": "sinkhorn",
            "inference": "sinkhorn_newton_krylov",
        },
        "loss": {"energy_weight": 1.0, "forces_weight": 0.0, "stress_weight": 0.0},
        "baseline": "zero",
        "radii": {"r_ot": 4.0, "r_mp": 3.0},
        "training": {
            "batch_size": 1,
            "validation_batch_size": 2,
            "max_epochs": 1,
            "learning_rate": 0.001,
        },
        "runtime": {"device": "cpu", "dtype": "float64", "seed": 23},
    }
    suffix = ".yaml" if yaml_recipe else ".json"
    recipe_path = directory / f"recipe{suffix}"
    recipe_path.write_text(
        yaml.safe_dump(recipe, sort_keys=False) if yaml_recipe else json.dumps(recipe),
        encoding="utf-8",
    )
    return recipe_path, specification_path, poscar


def test_resolve_validate_and_train_dry_run_share_canonical_resolution(tmp_path, capsys):
    recipe_path, _, _ = _write_recipe_case(tmp_path)
    output = tmp_path / "resolved.json"
    manifest = tmp_path / "resolution-manifest.json"

    assert main([
        "resolve-train-config", str(recipe_path), "--output", str(output),
        "--manifest", str(manifest), "--json",
    ]) == 0
    resolution_stdout = capsys.readouterr().out
    report = json.loads(resolution_stdout)
    assert report["compiled_config"]["schema_version"] == "refsite_training_run_config_v2"
    assert report["recipe_summary"] == {
        "correlation_method": "symmetric",
        "maximum_correlation_order": 1,
        "training_ot_solver": "sinkhorn",
        "inference_ot_solver": "sinkhorn_newton_krylov",
        "inference_application": "prediction/evaluation call-time preference",
        "training_batch_size": 1,
        "validation_batch_size": 2,
    }
    assert output.exists() and manifest.exists()
    loaded = load_training_run_config(output)
    direct = resolve_training_recipe(recipe_path)
    assert loaded.to_dict() == direct.config.to_dict()
    assert loaded.config_fingerprint == direct.config.config_fingerprint
    assert json.loads(manifest.read_text())["content_fingerprint"] == direct.manifest.content_fingerprint

    assert main(["validate-train-config", str(recipe_path), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "scratch_preflight_ready"
    assert validated["training_executed"] is False
    assert main(["train", str(recipe_path), "--dry-run", "--json", "--quiet"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run == validated
    assert not (tmp_path / "runs" / "recipe-run").exists()


def test_recipe_normal_train_is_rejected_after_preflight_without_side_effects(tmp_path, capsys):
    recipe_path, _, _ = _write_recipe_case(tmp_path)
    py = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    assert main(["train", str(recipe_path), "--quiet"]) == 1
    captured = capsys.readouterr()
    assert "RECIPE_EXECUTION_NOT_INTEGRATED" in captured.err
    assert not (tmp_path / "runs" / "recipe-run").exists()
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_resolve_dry_run_writes_nothing_and_cli_overrides_are_manifested(tmp_path, capsys):
    recipe_path, _, _ = _write_recipe_case(tmp_path)
    output = tmp_path / "never.json"
    manifest = tmp_path / "never-manifest.json"
    assert main([
        "resolve-train-config", str(recipe_path), "--output", str(output),
        "--manifest", str(manifest), "--dry-run", "--json",
        "--batch-size", "3", "--validation-batch-size", "5",
        "--max-epochs", "2", "--r-ot", "4.5",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not output.exists() and not manifest.exists()
    assert report["compiled_config"]["data"]["batch_size"] == 3
    assert report["compiled_config"]["data"]["validation_batch_size"] == 5
    assert report["compiled_config"]["fit"]["max_epochs"] == 2
    assert report["compiled_config"]["radii"]["r_ot"] == 4.5
    origins = report["resolution_manifest"]["field_origins"]
    assert origins["data.batch_size"] == "CLI"
    assert origins["data.validation_batch_size"] == "CLI"
    assert origins["fit.max_epochs"] == origins["radii.r_ot"] == "CLI"


def test_human_resolution_uses_beginner_vocabulary_only(tmp_path, capsys):
    recipe_path, _, _ = _write_recipe_case(tmp_path)
    assert main([
        "resolve-train-config",
        str(recipe_path),
        "--output",
        str(tmp_path / "resolved.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--dry-run",
    ]) == 0
    output = capsys.readouterr().out
    assert "Correlation method: symmetric" in output
    assert "Maximum correlation order: 1" in output
    assert "Training OT solver: sinkhorn" in output
    assert "Inference OT solver: sinkhorn_newton_krylov" in output
    assert "Application: prediction/evaluation call-time preference" in output
    assert "Sinkhorn iterations: 256" in output
    assert "Residual tolerance: 1e-07" in output
    assert "Nonconvergence policy: fail-fast" in output
    assert "Training batch size: 1" in output
    assert "Validation batch size: 2" in output
    assert "TRAIN_FIXED" not in output
    assert "EVAL_ADAPTIVE" not in output
    assert "symmetric_power_v2" not in output
    assert "central_conditioned" not in output


def test_batch_sizes_change_only_the_corresponding_manifest_segmentation(tmp_path):
    recipe_path, _, _ = _write_recipe_case(tmp_path, frame_count=3)
    base_payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

    def prepare(name, *, train_batch, validation_batch):
        payload = json.loads(json.dumps(base_payload))
        payload["training"]["batch_size"] = train_batch
        payload["training"]["validation_batch_size"] = validation_batch
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        resolved = resolve_training_recipe(path)
        return resolved, prepare_scratch_training_run(resolved.config)

    base, base_preparation = prepare("base", train_batch=1, validation_batch=1)
    train_changed, train_preparation = prepare(
        "train-batch", train_batch=2, validation_batch=1
    )
    validation_changed, validation_preparation = prepare(
        "validation-batch", train_batch=1, validation_batch=2
    )

    assert base_preparation.data_manifest["train"]["batch_count"] == 3
    assert train_preparation.data_manifest["train"]["batch_count"] == 2
    assert base_preparation.data_manifest["validation"]["batch_count"] == 3
    assert validation_preparation.data_manifest["validation"]["batch_count"] == 2
    assert train_preparation.data_manifest["validation"]["batch_count"] == 3
    assert validation_preparation.data_manifest["train"]["batch_count"] == 3
    assert base.config.config_fingerprint != train_changed.config.config_fingerprint
    assert base.config.config_fingerprint != validation_changed.config.config_fingerprint
    for preparation in (train_preparation, validation_preparation):
        assert preparation.train_semantic_digest == base_preparation.train_semantic_digest
        assert (
            preparation.validation_semantic_digest
            == base_preparation.validation_semantic_digest
        )
        assert tuple(sample.sample_id for sample in preparation.train_samples) == tuple(
            sample.sample_id for sample in base_preparation.train_samples
        )
        assert tuple(
            sample.sample_id for sample in preparation.validation_samples
        ) == tuple(
            sample.sample_id for sample in base_preparation.validation_samples
        )
        assert (
            preparation.train_label_statistics
            == base_preparation.train_label_statistics
        )
        assert (
            preparation.validation_label_statistics
            == base_preparation.validation_label_statistics
        )


def test_newton_krylov_policy_fingerprint_binding_is_checked_in_full_preflight(
    tmp_path, capsys
):
    recipe_path, specification_path, _ = _write_recipe_case(tmp_path)
    specification = load_reference_specification(specification_path)
    wrong_policy = replace(
        specification.evaluation_policy,
        template_fingerprint="0" * 64,
        content_fingerprint=None,
    )
    wrong_specification = ReferenceSpecificationConfig(
        builder=specification.builder,
        phase_specification=specification.phase_specification,
        evaluation_policy=wrong_policy,
        species_alignment_weights=specification.species_alignment_weights,
        poscar_sha256=specification.poscar_sha256,
    )
    specification_path.write_text(
        yaml.safe_dump(wrong_specification.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    assert main(["validate-train-config", str(recipe_path)]) == 2
    assert "POLICY_TEMPLATE_FINGERPRINT_MISMATCH" in capsys.readouterr().err


def test_safe_recipe_input_and_output_protection(tmp_path, capsys):
    recipe_path, _, poscar = _write_recipe_case(tmp_path)
    output = tmp_path / "resolved.json"
    manifest = tmp_path / "manifest.json"
    assert main(["resolve-train-config", str(recipe_path), "--output", str(output), "--manifest", str(manifest)]) == 0
    capsys.readouterr()
    before = output.read_bytes()
    assert main(["resolve-train-config", str(recipe_path), "--output", str(output), "--manifest", str(manifest)]) == 2
    assert output.read_bytes() == before
    capsys.readouterr()

    symlink = tmp_path / "linked.json"
    symlink.symlink_to(output)
    assert main(["resolve-train-config", str(recipe_path), "--output", str(symlink), "--manifest", str(tmp_path / "m2.json")]) == 2
    assert symlink.is_symlink()
    capsys.readouterr()

    changed = poscar.read_bytes() + b"\n"
    poscar.write_bytes(changed)
    assert main(["validate-train-config", str(recipe_path)]) == 2
    assert "POSCAR_FINGERPRINT_MISMATCH" in capsys.readouterr().err


def test_atomic_resolution_commit_never_clobbers_a_racing_writer(
    monkeypatch, tmp_path, capsys
):
    import refsite_mlip.cli.resolve_train_config as module

    recipe_path, _, _ = _write_recipe_case(tmp_path)
    output = tmp_path / "racing.json"
    manifest = tmp_path / "manifest.json"
    original = module.commit_temporary_file

    def race(temporary, target, *, overwrite):
        if target == output and not target.exists():
            target.write_bytes(b"competitor\n")
        return original(temporary, target, overwrite=overwrite)

    monkeypatch.setattr(module, "commit_temporary_file", race)
    assert main([
        "resolve-train-config", str(recipe_path), "--output", str(output),
        "--manifest", str(manifest),
    ]) == 2
    assert output.read_bytes() == b"competitor\n"
    assert not manifest.exists()
    assert "OUTPUT_EXISTS" in capsys.readouterr().err


@pytest.mark.parametrize(
    "encoded,reason",
    (
        ("schema_version: refsite_training_recipe_v1\nname: a\nname: b\n", "DUPLICATE_RECIPE_KEY"),
        ("base: &base {x: 1}\ncopy: *base\n", "UNSUPPORTED_YAML_ALIAS"),
        ("!!python/object:builtins.object {}\n", "INVALID_RECIPE_DOCUMENT"),
        ("schema_version: refsite_training_recipe_v1\nname: .nan\n", "NONFINITE_RECIPE_VALUE"),
    ),
)
def test_unsafe_yaml_is_rejected(encoded, reason, tmp_path, capsys):
    path = tmp_path / "unsafe.yaml"
    path.write_text(encoded, encoding="utf-8")
    assert main([
        "resolve-train-config", str(path),
        "--output", str(tmp_path / "resolved.json"),
        "--manifest", str(tmp_path / "manifest.json"),
    ]) == 2
    assert reason in capsys.readouterr().err
