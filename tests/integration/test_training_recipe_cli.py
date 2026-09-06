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
from refsite_mlip.cli.errors import CLIError
from refsite_mlip.cli.export_bundle import export_bundle
from refsite_mlip.cli.resume import resume_training
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
from refsite_mlip.inference import ReferenceSitePredictor
from refsite_mlip.models import EvaluationPolicy
from refsite_mlip.models import (
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
)
from refsite_mlip.training import (
    ScratchCheckpointedTrainingError,
    TrainingRunDirectory,
    canonical_runtime_json,
    load_training_checkpoint,
    materialize_automatic_references,
    prepare_scratch_training_run,
    run_scratch_checkpointed_training,
)
from refsite_mlip.transport import TRAIN_FIXED

from test_scratch_training_preparation import _atoms, _case, _labeled


def _write_automatic_recipe(
    directory: Path,
    *,
    references,
    train,
    validation,
    inference="sinkhorn",
):
    from ase.io import write

    (directory / "runs").mkdir(exist_ok=True)
    for filename, atoms, _, _ in references:
        write(directory / filename, atoms, format="vasp", direct=True)
    for filename, frames, _ in train:
        write(directory / filename, list(frames), format="extxyz")
    for filename, frames, _ in validation:
        write(directory / filename, list(frames), format="extxyz")
    payload = {
        "schema_version": "refsite_training_recipe_v1",
        "name": "automatic-run",
        "model": {
            "num_interactions": 1,
            "correlation": 1,
            "hidden_channels": 1,
            "max_L": 2,
            "initialization_seed": 19,
        },
        "reference": {
            "sources": [
                {
                    "poscar": filename,
                    "template_id": template_id,
                    "maximum_strain": maximum_strain,
                }
                for filename, _, template_id, maximum_strain in references
            ]
        },
        "data": {
            "train": [
                {"path": filename, "template_id": template_id}
                for filename, _, template_id in train
            ],
            "validation": [
                {"path": filename, "template_id": template_id}
                for filename, _, template_id in validation
            ],
        },
        "ot_solver": {"training": "sinkhorn", "inference": inference},
        "loss": {
            "energy_weight": 1.0,
            "forces_weight": 0.0,
            "stress_weight": 0.0,
        },
        "baseline": "zero",
        "radii": {"r_ot": 4.0, "r_mp": 3.0},
        "training": {
            "batch_size": 2,
            "validation_batch_size": 2,
            "max_epochs": 1,
            "learning_rate": 0.001,
        },
        "runtime": {"device": "cpu", "dtype": "float64", "seed": 23},
    }
    recipe = directory / "automatic.yaml"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return recipe


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


def test_poscar_only_recipe_resolves_and_full_preflight_is_shared(tmp_path, capsys):
    recipe_path, specification_path, poscar = _write_recipe_case(tmp_path)
    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    payload["reference"] = poscar.name
    payload["ot_solver"]["inference"] = "sinkhorn"
    recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    specification_path.unlink()

    resolved = resolve_training_recipe(recipe_path)
    automatic = resolved.automatic_reference_preparation
    assert automatic is not None
    assert len(automatic.results) == 1
    certificate = automatic.results[0].to_dict()
    assert certificate["approval_status"] == "provisional"
    assert certificate["scope"] == "dataset_bounded"
    assert certificate["evaluation_policy"] is None
    assert certificate["strain"]["resolved_maximum_strain"] == 0.01
    assert certificate["vacancies"]["train"]["observed_K_values"] == [0]

    output = tmp_path / "automatic-resolved.json"
    manifest = tmp_path / "automatic-manifest.json"
    assert main(
        [
            "resolve-train-config",
            str(recipe_path),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--dry-run",
            "--json",
        ]
    ) == 0
    resolution_report = json.loads(capsys.readouterr().out)
    assert resolution_report["reference_preparation"] == automatic.to_dict()
    assert not output.exists() and not manifest.exists()

    assert main(
        ["resolve-train-config", str(recipe_path), "--dry-run", "--json"]
    ) == 0
    output_free_report = json.loads(capsys.readouterr().out)
    assert output_free_report == resolution_report

    assert main(["validate-train-config", str(recipe_path), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "scratch_preflight_ready"
    assert validated["reference_preparation"] == automatic.to_dict()
    assert (
        validated["reference_preparation"]["content_fingerprint"]
        == resolved.manifest.automatic_reference_fingerprint
    )
    assert main(["train", str(recipe_path), "--dry-run", "--json", "--quiet"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run == validated
    assert not (tmp_path / "runs" / "recipe-run").exists()


def test_poscar_only_recipe_materializes_trains_resumes_and_exports_without_rebuild(
    tmp_path, capsys, monkeypatch
):
    reference = _atoms(1)
    vacancy = reference.copy()
    del vacancy[0]
    recipe_path = _write_automatic_recipe(
        tmp_path,
        references=(("POSCAR", reference, "alpha", "auto"),),
        train=((
            "train.xyz",
            (_labeled(reference, -8.0), _labeled(vacancy, -6.5)),
            "alpha",
        ),),
        validation=(("validation.xyz", (_labeled(vacancy, -6.4),), "alpha"),),
    )
    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    payload["baseline"] = "minimum_norm"
    recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    resolved = resolve_training_recipe(recipe_path)
    automatic = resolved.automatic_reference_preparation
    assert automatic is not None

    assert main(["train", str(recipe_path), "--json", "--quiet"]) == 0
    terminal = json.loads(capsys.readouterr().out)
    assert terminal["status"] == "completed"
    output = tmp_path / "runs" / "automatic-run"
    expected = {
        "resolved_config.json",
        "preflight.json",
        "data_manifest.json",
        "run_status.json",
        "initial_bundle.pt",
        "metrics.jsonl",
        "references",
        "checkpoints",
        "training.log",
    }
    assert {path.name for path in output.iterdir()} == expected
    references = output / "references"
    assert {path.name for path in references.iterdir()} == {
        "alpha.reference.json",
        "alpha.certificate.json",
    }
    saved_specification = load_reference_specification(
        references / "alpha.reference.json"
    )
    generated = automatic.results[0]
    assert saved_specification.to_dict() == generated.specification.to_dict()
    saved_certificate = json.loads(
        (references / "alpha.certificate.json").read_text(encoding="utf-8")
    )
    expected_certificate = generated.to_dict()
    expected_certificate.pop("poscar")
    assert saved_certificate == expected_certificate

    initial = load_reference_site_model_bundle(output / "initial_bundle.pt")
    latest = load_training_checkpoint(output / "checkpoints" / "latest.pt")
    assert int(torch.count_nonzero(initial.model_state["atomic_baseline"])) == 0
    assert bool(torch.any(latest.model_state_dict["atomic_baseline"] != 0.0))
    binding = initial.template_bindings[0]
    assert binding.structural_artifact.structural_fingerprint == saved_certificate[
        "artifact_sha256"
    ]
    assert binding.full_template_fingerprint == saved_certificate["template_sha256"]
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    assert status["runtime"]["solver_path"] == "sinkhorn"
    assert "train_fixed" not in canonical_runtime_json(status)
    assert "TRAIN_FIXED" not in canonical_runtime_json(status)
    materialized = status["reference_materialization"]
    assert materialized["reference_resolution_mode"] == "materialized"
    assert materialized["default_template_id"] == "alpha"
    assert materialized["templates"]["alpha"]["specification_fingerprint"] == (
        saved_specification.content_fingerprint
    )
    assert len((output / "metrics.jsonl").read_bytes().splitlines()) == 1

    # Once initial_bundle.pt exists, continuation and export must not invoke
    # any POSCAR/builder/automatic-search path.  Removing the source provides
    # a black-box guard in addition to the call traps.
    (tmp_path / "POSCAR").unlink()
    import refsite_mlip.config.automatic_reference as automatic_module
    import refsite_mlip.training.scratch_preparation as preparation_module

    def forbidden(*args, **kwargs):
        raise AssertionError("automatic reference rebuild is forbidden")

    monkeypatch.setattr(automatic_module, "prepare_automatic_references", forbidden)
    monkeypatch.setattr(
        preparation_module, "build_reference_template_from_poscar", forbidden
    )
    resumed = resume_training(output, max_epochs=2)
    assert resumed["status"] == "completed"
    assert (output / "checkpoints" / "epoch_000001.pt").is_file()
    assert len((output / "metrics.jsonl").read_bytes().splitlines()) == 2
    exported_path = tmp_path / "exported.pt"
    report = export_bundle(
        output, source="latest", output_path=exported_path
    )
    assert report["status"] == "completed"
    exported = load_reference_site_model_bundle(exported_path)
    resumed_latest = load_training_checkpoint(output / "checkpoints" / "latest.pt")
    assert set(exported.model_state) == set(resumed_latest.model_state_dict)
    assert all(
        torch.equal(exported.model_state[key], resumed_latest.model_state_dict[key])
        for key in exported.model_state
    )

    # Even if an attacker recomputes the certificate's own outer hash, the
    # persisted bundle/status binding must reject changed artifact semantics
    # without modifying the last committed checkpoint or journal.
    checkpoint_bytes = (output / "checkpoints" / "latest.pt").read_bytes()
    journal_bytes = (output / "metrics.jsonl").read_bytes()
    corrupted = dict(saved_certificate)
    corrupted["artifact_sha256"] = "0" * 64
    corrupted.pop("certificate_sha256")
    corrupted["certificate_sha256"] = hashlib.sha256(
        canonical_runtime_json(corrupted).encode("utf-8")
    ).hexdigest()
    (references / "alpha.certificate.json").write_text(
        canonical_runtime_json(corrupted) + "\n", encoding="utf-8"
    )
    with pytest.raises(CLIError) as mismatch:
        resume_training(output, max_epochs=3)
    assert mismatch.value.reason_code == "MATERIALIZED_REFERENCE_FINGERPRINT_MISMATCH"
    assert (output / "checkpoints" / "latest.pt").read_bytes() == checkpoint_bytes
    assert (output / "metrics.jsonl").read_bytes() == journal_bytes


def test_automatic_mixed_templates_assignment_manifest_and_order_are_deterministic(tmp_path):
    reference_a = _atoms(1)
    reference_b = _atoms(1).repeat((2, 1, 1))
    vacancy_a = reference_a.copy()
    del vacancy_a[0]
    vacancy_b = reference_b.copy()
    del vacancy_b[:2]
    references = (
        ("POSCAR_a", reference_a, "alpha", "auto"),
        ("POSCAR_b", reference_b, "zeta", "auto"),
    )
    train = (
        ("train_a.xyz", (_labeled(reference_a, -8.0), _labeled(vacancy_a, -7.0)), "alpha"),
        ("train_b.xyz", (_labeled(reference_b, -16.0), _labeled(vacancy_b, -14.0)), "zeta"),
    )
    validation = (
        ("validation_a.xyz", (_labeled(vacancy_a, -7.1),), "alpha"),
        ("validation_b.xyz", (_labeled(reference_b, -15.8),), "zeta"),
    )
    first_path = _write_automatic_recipe(
        tmp_path, references=references, train=train, validation=validation
    )
    first = resolve_training_recipe(first_path)
    automatic = first.automatic_reference_preparation
    assert automatic is not None
    assert [result.template_id for result in automatic.results] == ["alpha", "zeta"]
    assert first.config.model_source.default_template_id == "alpha"
    by_id = {result.template_id: result.to_dict() for result in automatic.results}
    assert by_id["alpha"]["vacancies"]["train"]["observed_K_values"] == [0, 1]
    assert by_id["zeta"]["vacancies"]["train"]["observed_K_values"] == [0, 2]
    assert by_id["alpha"]["evaluation_policy"] is None
    assert by_id["zeta"]["approval_status"] == "provisional"

    payload = yaml.safe_load(first_path.read_text(encoding="utf-8"))
    payload["reference"]["sources"].reverse()
    second_path = tmp_path / "automatic-reversed.yaml"
    second_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    second = resolve_training_recipe(second_path)
    assert second.config.model_source.default_template_id == "alpha"
    assert first.config.canonical_json() == second.config.canonical_json()
    assert first.config.config_fingerprint == second.config.config_fingerprint
    assert automatic.content_fingerprint == second.automatic_reference_preparation.content_fingerprint

    shorthand_payload = yaml.safe_load(first_path.read_text(encoding="utf-8"))
    shorthand_payload["reference"] = ["POSCAR_a", "POSCAR_b"]
    shorthand_payload["data"] = {
        "train": ["train_a.xyz", "train_b.xyz"],
        "validation": ["validation_a.xyz", "validation_b.xyz"],
    }
    shorthand_path = tmp_path / "automatic-list-shorthand.yaml"
    shorthand_path.write_text(
        yaml.safe_dump(shorthand_payload, sort_keys=False), encoding="utf-8"
    )
    shorthand = resolve_training_recipe(shorthand_path)
    shorthand_auto = shorthand.automatic_reference_preparation
    assert shorthand_auto is not None
    assert len(shorthand_auto.results) == 2
    assert all(result.template_id.startswith("ref_m") for result in shorthand_auto.results)
    assert len(shorthand_auto.train_assignments) == 4
    assert len({template_id for _, template_id in shorthand_auto.train_assignments}) == 2


def test_automatic_mixed_templates_train_with_k0_k1_and_k2(tmp_path, capsys):
    reference_a = _atoms(1)
    reference_b = _atoms(1).repeat((2, 1, 1))
    vacancy_a = reference_a.copy()
    del vacancy_a[0]
    vacancy_b = reference_b.copy()
    del vacancy_b[:2]
    recipe = _write_automatic_recipe(
        tmp_path,
        references=(
            ("POSCAR_a", reference_a, "alpha", "auto"),
            ("POSCAR_b", reference_b, "zeta", "auto"),
        ),
        train=(
            (
                "train_a.xyz",
                (_labeled(reference_a, -8.0), _labeled(vacancy_a, -7.0)),
                "alpha",
            ),
            (
                "train_b.xyz",
                (_labeled(reference_b, -16.0), _labeled(vacancy_b, -14.0)),
                "zeta",
            ),
        ),
        validation=(
            ("validation_a.xyz", (_labeled(vacancy_a, -7.1),), "alpha"),
            ("validation_b.xyz", (_labeled(reference_b, -15.8),), "zeta"),
        ),
    )
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    payload["training"]["batch_size"] = 3
    payload["training"]["validation_batch_size"] = 1
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    assert main(["train", str(recipe), "--json", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    output = tmp_path / "runs" / "automatic-run"
    bundle = load_reference_site_model_bundle(output / "initial_bundle.pt")
    checkpoint = load_training_checkpoint(output / "checkpoints" / "latest.pt")
    assert bundle.binding_ids == ("alpha", "zeta")
    assert bundle.default_template_id == "alpha"
    assert {
        binding.template_id: binding.structural_artifact.diagnostics.num_sites
        for binding in bundle.template_bindings
    } == {"alpha": 8, "zeta": 16}
    certificate_a = json.loads(
        (output / "references" / "alpha.certificate.json").read_text()
    )
    certificate_b = json.loads(
        (output / "references" / "zeta.certificate.json").read_text()
    )
    assert certificate_a["vacancies"]["train"]["observed_K_values"] == [0, 1]
    assert certificate_b["vacancies"]["train"]["observed_K_values"] == [0, 2]
    manifest = json.loads((output / "data_manifest.json").read_text())
    status = json.loads((output / "run_status.json").read_text())
    assert status["reference_materialization"]["default_template_id"] == "alpha"
    assert [
        template_id
        for batch in manifest["train"]["batches"]
        for template_id in batch["template_ids"]
    ] == ["alpha", "alpha", "zeta", "zeta"]
    assert manifest["train"]["batch_count"] == 2
    assert manifest["validation"]["batch_count"] == 2
    assert [
        batch["stop"] - batch["start"] for batch in manifest["train"]["batches"]
    ] == [
        3,
        1,
    ]
    assert [
        batch["stop"] - batch["start"]
        for batch in manifest["validation"]["batches"]
    ] == [1, 1]
    assert set(checkpoint.metadata.template_fingerprints) == {"alpha", "zeta"}
    assert checkpoint.progress.completed_epochs == 1
    assert checkpoint.progress.global_step == 2


def test_automatic_reference_continuous_and_resumed_trajectory_are_exact(tmp_path):
    continuous_root = tmp_path / "continuous"
    split_root = tmp_path / "split"
    continuous_root.mkdir()
    split_root.mkdir()
    reference = _atoms(1)
    vacancy = reference.copy()
    del vacancy[0]

    def prepare(root: Path, *, max_epochs: int):
        recipe = _write_automatic_recipe(
            root,
            references=(("POSCAR", reference, "alpha", "auto"),),
            train=((
                "train.xyz",
                (_labeled(reference, -8.0), _labeled(vacancy, -6.5)),
                "alpha",
            ),),
            validation=((
                "validation.xyz",
                (_labeled(vacancy, -6.4),),
                "alpha",
            ),),
        )
        payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        payload["baseline"] = "minimum_norm"
        payload["training"]["max_epochs"] = max_epochs
        recipe.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        resolution = resolve_training_recipe(recipe)
        prepared = prepare_scratch_training_run(
            resolution.config,
            automatic_reference_preparation=(
                resolution.automatic_reference_preparation.to_dict()
            ),
        )
        return resolution.config, prepared

    continuous_config, continuous_preparation = prepare(
        continuous_root, max_epochs=2
    )
    split_config, split_preparation = prepare(split_root, max_epochs=1)
    continuous = run_scratch_checkpointed_training(
        continuous_config, continuous_preparation
    )
    continuous_checkpoint = load_training_checkpoint(continuous.latest_path)
    continuous_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    first = run_scratch_checkpointed_training(split_config, split_preparation)
    epoch_zero = Path(first.latest_path).with_name("epoch_000000.pt")
    epoch_zero_bytes = epoch_zero.read_bytes()
    resumed = resume_training(first.run_directory, max_epochs=2)
    resumed_checkpoint = load_training_checkpoint(resumed["latest_checkpoint"])
    resumed_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    def assert_tree_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor)
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert isinstance(right, dict)
            assert tuple(left) == tuple(right)
            for key in left:
                assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert type(left) is type(right)
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                assert_tree_equal(left_item, right_item)
        else:
            assert left == right

    assert_tree_equal(
        continuous_checkpoint.model_state_dict,
        resumed_checkpoint.model_state_dict,
    )
    assert_tree_equal(
        continuous_checkpoint.optimizer_state_dict,
        resumed_checkpoint.optimizer_state_dict,
    )
    assert_tree_equal(
        continuous_checkpoint.scheduler_state_dict,
        resumed_checkpoint.scheduler_state_dict,
    )
    assert continuous_checkpoint.selection_state == resumed_checkpoint.selection_state
    assert continuous_checkpoint.progress == resumed_checkpoint.progress
    assert continuous_checkpoint.fit_history == resumed_checkpoint.fit_history
    assert continuous_draws[0] == resumed_draws[0]
    assert continuous_draws[1] == resumed_draws[1]
    assert torch.equal(continuous_draws[2], resumed_draws[2])
    assert epoch_zero.read_bytes() == epoch_zero_bytes
    assert (Path(continuous.run_directory) / "metrics.jsonl").read_bytes() == (
        Path(first.run_directory) / "metrics.jsonl"
    ).read_bytes()
    assert (
        continuous.startup.initial_bundle_fingerprint
        == first.startup.initial_bundle_fingerprint
    )


def test_explicit_reference_recipe_cli_training_matches_automatic_source(
    tmp_path, capsys
):
    reference = _atoms(1)
    vacancy = reference.copy()
    del vacancy[0]
    recipe = _write_automatic_recipe(
        tmp_path,
        references=(("POSCAR", reference, "alpha", "auto"),),
        train=((
            "train.xyz",
            (_labeled(reference, -8.0), _labeled(vacancy, -6.5)),
            "alpha",
        ),),
        validation=((
            "validation.xyz",
            (_labeled(vacancy, -6.4),),
            "alpha",
        ),),
    )
    resolution = resolve_training_recipe(recipe)
    automatic_preparation = prepare_scratch_training_run(
        resolution.config,
        automatic_reference_preparation=(
            resolution.automatic_reference_preparation.to_dict()
        ),
    )
    assert main(["train", str(recipe), "--json", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    automatic_directory = tmp_path / "runs" / "automatic-run"
    specification_path = (
        automatic_directory / "references" / "alpha.reference.json"
    )
    assert specification_path.is_file()
    explicit_payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    explicit_payload["name"] = "explicit-run"
    explicit_payload["reference"] = {
        "specification": "runs/automatic-run/references/alpha.reference.json",
        "poscar": "POSCAR",
        "allow_provisional_phase": True,
    }
    explicit_recipe = tmp_path / "explicit-recipe.yaml"
    explicit_recipe.write_text(
        yaml.safe_dump(explicit_payload, sort_keys=False), encoding="utf-8"
    )

    assert main(
        ["train", str(explicit_recipe), "--json", "--quiet"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    explicit_directory = tmp_path / "runs" / "explicit-run"
    for directory in (automatic_directory, explicit_directory):
        assert (directory / "checkpoints" / "epoch_000000.pt").is_file()
        assert (directory / "checkpoints" / "latest.pt").is_file()
        assert (directory / "checkpoints" / "best.pt").is_file()
        assert len((directory / "metrics.jsonl").read_bytes().splitlines()) == 1

    automatic_bundle = load_reference_site_model_bundle(
        automatic_directory / "initial_bundle.pt"
    )
    explicit_bundle = load_reference_site_model_bundle(
        explicit_directory / "initial_bundle.pt"
    )
    assert automatic_bundle.architecture_fingerprint == (
        explicit_bundle.architecture_fingerprint
    )
    assert tuple(automatic_bundle.model_state) == tuple(explicit_bundle.model_state)
    for key in automatic_bundle.model_state:
        assert torch.equal(
            automatic_bundle.model_state[key], explicit_bundle.model_state[key]
        )
    assert automatic_bundle.binding_ids == explicit_bundle.binding_ids == ("alpha",)
    automatic_binding = automatic_bundle.template_bindings[0]
    explicit_binding = explicit_bundle.template_bindings[0]
    assert automatic_binding.full_template_fingerprint == (
        explicit_binding.full_template_fingerprint
    )
    assert automatic_binding.structural_artifact.structural_fingerprint == (
        explicit_binding.structural_artifact.structural_fingerprint
    )
    assert automatic_binding.phase_specification.to_dict() == (
        explicit_binding.phase_specification.to_dict()
    )

    automatic_runtime = instantiate_reference_site_model_bundle(
        automatic_bundle, device="cpu", dtype=torch.float64
    )
    explicit_runtime = instantiate_reference_site_model_bundle(
        explicit_bundle, device="cpu", dtype=torch.float64
    )
    assert automatic_runtime.template_fingerprints == (
        explicit_runtime.template_fingerprints
    )
    assert automatic_runtime.structural_fingerprints == (
        explicit_runtime.structural_fingerprints
    )
    automatic_context = automatic_runtime.template_contexts["alpha"]
    explicit_context = explicit_runtime.template_contexts["alpha"]
    assert automatic_context.fingerprint == explicit_context.fingerprint
    for field_name in (
        "reference_fractional",
        "site_types",
        "reference_cell",
        "edge_index",
        "shifts",
        "phase_modes",
        "phase_mode_weights",
        "site_alignment_weights",
        "phase_channel_weights",
    ):
        assert torch.equal(
            getattr(automatic_context, field_name),
            getattr(explicit_context, field_name),
        )

    automatic_checkpoint = load_training_checkpoint(
        automatic_directory / "checkpoints" / "latest.pt"
    )
    explicit_checkpoint = load_training_checkpoint(
        explicit_directory / "checkpoints" / "latest.pt"
    )
    assert automatic_checkpoint.progress == explicit_checkpoint.progress
    assert automatic_checkpoint.selection_state == explicit_checkpoint.selection_state
    assert automatic_checkpoint.fit_history == explicit_checkpoint.fit_history
    for key in automatic_checkpoint.model_state_dict:
        assert torch.equal(
            automatic_checkpoint.model_state_dict[key],
            explicit_checkpoint.model_state_dict[key],
        )
    assert automatic_checkpoint.optimizer_state_dict.keys() == (
        explicit_checkpoint.optimizer_state_dict.keys()
    )
    for left_group, right_group in zip(
        automatic_checkpoint.optimizer_state_dict["param_groups"],
        explicit_checkpoint.optimizer_state_dict["param_groups"],
    ):
        assert left_group == right_group
    for parameter_id, left_state in automatic_checkpoint.optimizer_state_dict[
        "state"
    ].items():
        right_state = explicit_checkpoint.optimizer_state_dict["state"][parameter_id]
        for name, left_value in left_state.items():
            right_value = right_state[name]
            if isinstance(left_value, torch.Tensor):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value

    automatic_runtime.model.load_state_dict(
        automatic_checkpoint.model_state_dict, strict=True
    )
    explicit_runtime.model.load_state_dict(
        explicit_checkpoint.model_state_dict, strict=True
    )
    automatic_prediction = ReferenceSitePredictor(
        automatic_runtime
    ).predict_sample(
        automatic_preparation.train_samples[0],
        solver_path=TRAIN_FIXED,
        compute_forces=False,
        compute_stress=False,
    )
    explicit_prediction = ReferenceSitePredictor(explicit_runtime).predict_sample(
        automatic_preparation.train_samples[0],
        solver_path=TRAIN_FIXED,
        compute_forces=False,
        compute_stress=False,
    )
    for field_name in (
        "energy",
        "baseline_energy",
        "residual_energy",
        "site_energy",
    ):
        assert torch.equal(
            getattr(automatic_prediction, field_name),
            getattr(explicit_prediction, field_name),
        )


def test_automatic_reference_persistence_failures_preserve_committed_files(
    tmp_path, monkeypatch
):
    reference_a = _atoms(1)
    reference_b = _atoms(1).repeat((2, 1, 1))
    recipe = _write_automatic_recipe(
        tmp_path,
        references=(
            ("POSCAR_a", reference_a, "alpha", "auto"),
            ("POSCAR_b", reference_b, "zeta", "auto"),
        ),
        train=(
            ("train_a.xyz", (_labeled(reference_a, -8.0),), "alpha"),
            ("train_b.xyz", (_labeled(reference_b, -16.0),), "zeta"),
        ),
        validation=(
            ("validation_a.xyz", (_labeled(reference_a, -7.9),), "alpha"),
            ("validation_b.xyz", (_labeled(reference_b, -15.9),), "zeta"),
        ),
    )
    resolution = resolve_training_recipe(recipe)
    preparation = prepare_scratch_training_run(
        resolution.config,
        automatic_reference_preparation=(
            resolution.automatic_reference_preparation.to_dict()
        ),
    )
    import refsite_mlip.training.automatic_reference_materialization as module

    original_write = module._write_json
    expected_committed = {
        1: set(),
        2: {"alpha.reference.json"},
        4: {
            "alpha.reference.json",
            "alpha.certificate.json",
            "zeta.reference.json",
        },
    }
    for fail_at in (1, 2, 4):
        call_count = 0

        def injected(path, value, *, stage):
            nonlocal call_count
            call_count += 1
            if call_count == fail_at:
                raise OSError(f"injected reference write {fail_at}")
            return original_write(path, value, stage=stage)

        monkeypatch.setattr(module, "_write_json", injected)
        directory = TrainingRunDirectory.create(tmp_path / f"failure-{fail_at}")
        lock = directory.acquire_resume_lock()
        with pytest.raises(Exception) as caught:
            materialize_automatic_references(preparation, directory, lock)
        assert getattr(caught.value, "rollback_performed", None) is False
        assert set(path.name for path in directory.references.iterdir()) == (
            expected_committed[fail_at]
        )
        assert not list(directory.references.glob("*.tmp"))
        lock.release()


def test_automatic_reference_reload_failure_records_no_update_status(
    tmp_path, monkeypatch
):
    reference = _atoms(1)
    recipe = _write_automatic_recipe(
        tmp_path,
        references=(("POSCAR", reference, "alpha", "auto"),),
        train=(("train.xyz", (_labeled(reference, -8.0),), "alpha"),),
        validation=(("validation.xyz", (_labeled(reference, -7.9),), "alpha"),),
    )
    resolution = resolve_training_recipe(recipe)
    preparation = prepare_scratch_training_run(
        resolution.config,
        automatic_reference_preparation=(
            resolution.automatic_reference_preparation.to_dict()
        ),
    )
    import refsite_mlip.training.automatic_reference_materialization as module

    def fail_reload(path):
        raise OSError("injected strict reference reload failure")

    monkeypatch.setattr(module, "_load_reference", fail_reload)
    with pytest.raises(ScratchCheckpointedTrainingError) as caught:
        run_scratch_checkpointed_training(resolution.config, preparation)
    output = tmp_path / "runs" / "automatic-run"
    status = json.loads((output / "run_status.json").read_text())
    assert caught.value.reason_code == "AUTOMATIC_REFERENCE_MATERIALIZATION_FAILED"
    assert status["status"] == "failed"
    assert status["first_optimizer_update_executed"] is False
    assert status["global_step"] == 0
    assert not (output / "initial_bundle.pt").exists()
    assert not (output / "checkpoints").exists()
    assert not (output / ".resume.lock").exists()
    assert {path.name for path in (output / "references").iterdir()} == {
        "alpha.reference.json",
        "alpha.certificate.json",
    }
    assert status["error"]["completed_persistence_stages"] == [
        "references/",
        "references/alpha.reference.json",
        "references/alpha.certificate.json",
    ]


def test_automatic_reference_rejects_ambiguity_unused_source_and_unstable_newton_policy(tmp_path):
    reference = _atoms(1)
    shifted = reference.copy()
    shifted.positions[0, 0] += 0.01
    common_train = (("train.xyz", (_labeled(reference, -8.0),), "alpha"),)
    common_validation = (("validation.xyz", (_labeled(reference, -7.9),), "alpha"),)
    recipe = _write_automatic_recipe(
        tmp_path,
        references=(
            ("POSCAR_a", reference, "alpha", "auto"),
            ("POSCAR_b", shifted, "zeta", "auto"),
        ),
        train=common_train,
        validation=common_validation,
    )
    # Remove the exact selector so both content-distinct, same-domain templates
    # are genuinely eligible.  No volume/filename heuristic may choose one.
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    payload["data"]["train"] = "train.xyz"
    payload["data"]["validation"] = "validation.xyz"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="AMBIGUOUS_TEMPLATE_ASSIGNMENT"):
        resolve_training_recipe(recipe)

    payload["data"]["train"] = [{"path": "train.xyz", "template_id": "alpha"}]
    payload["data"]["validation"] = [{"path": "validation.xyz", "template_id": "alpha"}]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="UNUSED_AUTOMATIC_REFERENCE"):
        resolve_training_recipe(recipe)

    payload["reference"]["sources"] = [payload["reference"]["sources"][0]]
    payload["ot_solver"]["inference"] = "sinkhorn_newton_krylov"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="AUTOMATIC_EVALUATION_POLICY_AUDIT_FAILED") as caught:
        resolve_training_recipe(recipe)
    assert "SUPPORT_BRANCH_UNSTABLE" in str(caught.value)
    assert not (tmp_path / "runs" / "automatic-run").exists()


def test_automatic_reference_is_geometry_only_path_independent_and_rng_free(tmp_path):
    from ase.io import write

    reference = _atoms(1)
    recipe = _write_automatic_recipe(
        tmp_path,
        references=(("POSCAR_original", reference, "alpha", "auto"),),
        train=(("train.xyz", (_labeled(reference, -8.0),), "alpha"),),
        validation=(("validation.xyz", (_labeled(reference, -7.9),), "alpha"),),
    )
    generated_payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    generated_payload["reference"]["sources"][0].pop("template_id")
    generated_payload["data"] = {
        "train": "train.xyz",
        "validation": "validation.xyz",
    }
    recipe.write_text(
        yaml.safe_dump(generated_payload, sort_keys=False), encoding="utf-8"
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    grad_enabled = torch.is_grad_enabled()
    first = resolve_training_recipe(recipe)
    first_result = first.automatic_reference_preparation.results[0]

    # Neither source filename nor labels participate in automatic reference
    # identity.  The canonical training config still preserves the user's path
    # expression under its established path contract.
    renamed = tmp_path / "renamed-POSCAR"
    renamed.write_bytes((tmp_path / "POSCAR_original").read_bytes())
    write(tmp_path / "train.xyz", [_labeled(reference, 1234.5)], format="extxyz")
    write(tmp_path / "validation.xyz", [_labeled(reference, -999.0)], format="extxyz")
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    payload["reference"]["sources"][0]["poscar"] = renamed.name
    moved_recipe = tmp_path / "moved.yaml"
    moved_recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    second = resolve_training_recipe(moved_recipe)
    second_result = second.automatic_reference_preparation.results[0]

    assert first_result.template_id == second_result.template_id
    assert first_result.template_id.startswith("ref_m8_")
    assert first_result.specification.content_fingerprint == second_result.specification.content_fingerprint
    assert first_result.to_dict()["artifact_sha256"] == second_result.to_dict()["artifact_sha256"]
    assert first.automatic_reference_preparation.content_fingerprint == second.automatic_reference_preparation.content_fingerprint
    assert first_result.specification.phase_specification.to_dict() == second_result.specification.phase_specification.to_dict()
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_default_dtype() == default_dtype
    assert torch.is_grad_enabled() == grad_enabled


def test_automatic_reference_strain_content_and_species_failures_are_structured(tmp_path):
    reference = _atoms(1)
    duplicate_recipe = _write_automatic_recipe(
        tmp_path,
        references=(
            ("POSCAR_a", reference, "alpha", "auto"),
            ("POSCAR_b", reference, "zeta", "auto"),
        ),
        train=(("train.xyz", (_labeled(reference, -8.0),), "alpha"),),
        validation=(("validation.xyz", (_labeled(reference, -7.9),), "alpha"),),
    )
    with pytest.raises(Exception, match="DUPLICATE_REFERENCE_CONTENT"):
        resolve_training_recipe(duplicate_recipe)

    strained = reference.copy()
    strained.set_cell(reference.cell.array * 1.046, scale_atoms=True)
    ceiling_dir = tmp_path / "ceiling"
    ceiling_dir.mkdir()
    ceiling_recipe = _write_automatic_recipe(
        ceiling_dir,
        references=(("POSCAR", reference, "alpha", "auto"),),
        train=(("train.xyz", (_labeled(strained, -8.0),), "alpha"),),
        validation=(("validation.xyz", (_labeled(reference, -7.9),), "alpha"),),
    )
    with pytest.raises(Exception, match="AUTO_MAXIMUM_STRAIN_LIMIT_EXCEEDED"):
        resolve_training_recipe(ceiling_recipe)

    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    explicit_recipe = _write_automatic_recipe(
        explicit_dir,
        references=(("POSCAR", reference, "alpha", 0.01),),
        train=(("train.xyz", (_labeled(strained, -8.0),), "alpha"),),
        validation=(("validation.xyz", (_labeled(reference, -7.9),), "alpha"),),
    )
    with pytest.raises(Exception, match="EXPLICIT_MAXIMUM_STRAIN_TOO_SMALL"):
        resolve_training_recipe(explicit_recipe)

    unsupported = reference.copy()
    unsupported.numbers[0] = 14
    species_dir = tmp_path / "species"
    species_dir.mkdir()
    species_recipe = _write_automatic_recipe(
        species_dir,
        references=(("POSCAR", reference, "alpha", "auto"),),
        train=(("train.xyz", (_labeled(reference, -8.0),), "alpha"),),
        validation=(("validation.xyz", (_labeled(unsupported, -7.9),), "alpha"),),
    )
    with pytest.raises(Exception, match="VALIDATION_ONLY_SPECIES"):
        resolve_training_recipe(species_recipe)


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
