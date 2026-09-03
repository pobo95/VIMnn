"""Integration coverage for portable checkpoint bundle export."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch

pytest.importorskip("ase")

from refsite_mlip.cli.errors import CLIError
from refsite_mlip.cli.export_bundle import (
    export_bundle,
    render_export_bundle_json,
)
from refsite_mlip.cli.main import main
from refsite_mlip.cli.resume import resume_training
from refsite_mlip.inference import (
    ReferenceSitePredictor,
    load_reference_site_predictor,
)
from refsite_mlip.models import (
    capture_reference_site_model_bundle,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.training import load_training_checkpoint
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from test_cli_train_bundle import _set_epochs
from test_cli_inspect_bundle import _typed_crystal_data
from test_model_bundle_runtime import _capture_case
from test_validate_train_config_cli import _simple_case, training_bundle


def _tree_snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): (None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


def _tree_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _numpy_rng_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


@pytest.fixture(scope="module")
def synthetic_run(training_bundle, tmp_path_factory):
    root = tmp_path_factory.mktemp("export-bundle-run")
    config_path, _ = _simple_case(root, training_bundle)
    _set_epochs(config_path, 1)
    train_module = importlib.import_module("refsite_mlip.cli.train")
    result = train_module.run_training(config_path)
    return {
        "root": root,
        "run": root / "run-output",
        "result": result,
        "config": config_path,
        "bundle": training_bundle,
    }


def test_latest_export_exact_state_baseline_bindings_and_payload_exclusions(
    synthetic_run, tmp_path
):
    run = synthetic_run["run"]
    output = tmp_path / "latest-model.pt"
    report = export_bundle(
        run,
        source="latest",
        output_path=output,
    )
    assert report["status"] == "completed"
    assert report["output_written"] is True
    assert output.is_file()
    checkpoint = load_training_checkpoint(run / "checkpoints" / "latest.pt")
    parent = load_reference_site_model_bundle(synthetic_run["bundle"]["path"])
    exported = load_reference_site_model_bundle(output)
    assert exported.bundle_fingerprint == report["bundle_sha256"]
    assert tuple(exported.model_state_keys) == tuple(checkpoint.model_state_dict)
    for key in exported.model_state_keys:
        assert torch.equal(exported.model_state[key], checkpoint.model_state_dict[key])
    assert torch.equal(
        exported.model_state["atomic_baseline"],
        checkpoint.model_state_dict["atomic_baseline"],
    )
    assert exported.default_template_id == parent.default_template_id
    assert exported.conventions.keys() == parent.conventions.keys()
    for key in exported.conventions:
        assert _tree_equal(exported.conventions[key], parent.conventions[key])
    before = {binding.template_id: binding for binding in parent.template_bindings}
    after = {binding.template_id: binding for binding in exported.template_bindings}
    assert set(before) == set(after)
    for template_id in before:
        assert _tree_equal(
            before[template_id].structural_artifact.to_payload(),
            after[template_id].structural_artifact.to_payload(),
        )
        assert (
            before[template_id].phase_specification.to_dict()
            == after[template_id].phase_specification.to_dict()
        )
        assert before[template_id].evaluation_policy is None
        assert after[template_id].evaluation_policy is None
    payload = exported.to_payload()
    forbidden = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "selection_state",
        "fit_history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "cuda_rng_states",
        "training_data",
        "validation_data",
    }
    assert forbidden.isdisjoint(payload)
    provenance = exported.provenance
    assert provenance["source"] == "latest"
    assert provenance["checkpoint_epoch"] == checkpoint.progress.last_completed_epoch
    assert provenance["global_step"] == checkpoint.progress.global_step
    assert provenance["parent_initial_bundle_sha256"] == parent.bundle_fingerprint
    assert provenance["training_config_sha256"] == report["training_config_sha256"]
    assert not any("/" in str(value) for key, value in provenance.items() if "path" in key)
    predictor = load_reference_site_predictor(output, device="cpu", dtype=torch.float64)
    assert predictor.runtime.bundle_fingerprint == exported.bundle_fingerprint
    assert predictor.model.training is False


def test_dry_run_is_fully_read_only_and_preserves_all_rng(
    synthetic_run, tmp_path, monkeypatch
):
    run = synthetic_run["run"]
    output = tmp_path / "dry-run.pt"
    run_before = _tree_snapshot(run)
    bundle_before = synthetic_run["bundle"]["path"].read_bytes()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    safe_before = tuple(torch.serialization.get_safe_globals())
    potential = importlib.import_module("refsite_mlip.models.potential")

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("export must not run model.forward")

    monkeypatch.setattr(potential.ReferenceSitePotential, "forward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    report = export_bundle(
        run,
        source="latest",
        output_path=output,
        dry_run=True,
    )
    assert report["status"] == "dry_run_ready"
    assert report["output_written"] is False
    assert not output.exists()
    assert _tree_snapshot(run) == run_before
    assert synthetic_run["bundle"]["path"].read_bytes() == bundle_before
    assert random.getstate() == python_before
    assert _numpy_rng_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert tuple(torch.serialization.get_safe_globals()) == safe_before


def test_repeated_exports_have_identical_semantic_sha_and_override_works(
    synthetic_run, tmp_path
):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    replacement = tmp_path / "moved-initial.pt"
    replacement.write_bytes(synthetic_run["bundle"]["path"].read_bytes())
    first_report = export_bundle(
        synthetic_run["run"], source="latest", output_path=first
    )
    second_report = export_bundle(
        synthetic_run["run"],
        source="latest",
        output_path=second,
        initial_bundle_path=replacement,
    )
    assert first_report["bundle_sha256"] == second_report["bundle_sha256"]
    assert load_reference_site_model_bundle(first).bundle_fingerprint == (
        load_reference_site_model_bundle(second).bundle_fingerprint
    )
    assert render_export_bundle_json(first_report) == render_export_bundle_json(
        dict(reversed(tuple(first_report.items())))
    )


def test_cli_json_human_overwrite_and_output_collision(synthetic_run, tmp_path, capsys):
    output = tmp_path / "cli.pt"
    base = [
        "export-bundle",
        str(synthetic_run["run"]),
        "--source",
        "best",
        "--output",
        str(output),
    ]
    assert main(base + ["--json"]) == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["source"]["kind"] == "best"
    assert captured.err == ""
    assert main(base) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "reason='OUTPUT_EXISTS'" in captured.err
    assert main(base + ["--overwrite"]) == 0
    captured = capsys.readouterr()
    assert "Optimizer/training state excluded." in captured.out
    assert captured.err == ""
    collision = synthetic_run["run"] / "checkpoints" / "latest.pt"
    assert (
        main(
            [
                "export-bundle",
                str(synthetic_run["run"]),
                "--source",
                "latest",
                "--output",
                str(collision),
                "--overwrite",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "OUTPUT_INSIDE_RUN_DIRECTORY" in captured.err


def test_active_lock_wrong_override_symlink_and_atomic_failure_preserve_target(
    synthetic_run, tmp_path, monkeypatch
):
    run = synthetic_run["run"]
    lock = run / ".resume.lock"
    lock.write_text("foreign", encoding="utf-8")
    try:
        with pytest.raises(CLIError) as caught:
            export_bundle(run, source="latest", output_path=tmp_path / "locked.pt")
        assert caught.value.reason_code == "RESUME_LOCK_EXISTS"
    finally:
        lock.unlink()

    wrong = tmp_path / "wrong.pt"
    first_export = tmp_path / "different-semantic-bundle.pt"
    export_bundle(run, source="latest", output_path=first_export)
    wrong.write_bytes(first_export.read_bytes())
    with pytest.raises(CLIError) as caught:
        export_bundle(
            run,
            source="latest",
            output_path=tmp_path / "wrong-output.pt",
            initial_bundle_path=wrong,
        )
    assert caught.value.reason_code == "INITIAL_BUNDLE_FINGERPRINT_MISMATCH"

    symlink = tmp_path / "symlink.pt"
    symlink.symlink_to(tmp_path / "target.pt")
    with pytest.raises(CLIError) as caught:
        export_bundle(run, source="latest", output_path=symlink)
    assert caught.value.reason_code == "OUTPUT_SYMLINK_REJECTED"

    existing = tmp_path / "preserved.pt"
    existing.write_bytes(b"original")
    module = importlib.import_module("refsite_mlip.cli.export_bundle")

    def fail_save(*args, **kwargs):
        del args, kwargs
        raise OSError("injected atomic save failure")

    monkeypatch.setattr(module, "save_reference_site_model_bundle", fail_save)
    with pytest.raises(CLIError) as caught:
        export_bundle(
            run,
            source="latest",
            output_path=existing,
            overwrite=True,
        )
    assert caught.value.reason_code == "BUNDLE_SAVE_FAILED"
    assert existing.read_bytes() == b"original"
    assert not list(tmp_path.glob(".preserved.pt.*.tmp"))


def test_checkpoint_model_can_be_instantiated_without_training_state(
    synthetic_run, tmp_path
):
    output = tmp_path / "runtime.pt"
    export_bundle(synthetic_run["run"], source="best", output_path=output)
    bundle = load_reference_site_model_bundle(output)
    loaded = instantiate_reference_site_model_bundle(
        bundle, device="cpu", dtype=torch.float64
    )
    checkpoint = load_training_checkpoint(
        synthetic_run["run"] / "checkpoints" / "best.pt"
    )
    assert loaded.model.training is False
    for key, value in loaded.model.state_dict().items():
        assert torch.equal(value.detach().cpu(), checkpoint.model_state_dict[key])


def test_exported_predictor_matches_checkpoint_restored_model_exactly(
    synthetic_run, tmp_path
):
    output = tmp_path / "prediction-parity.pt"
    export_bundle(synthetic_run["run"], source="latest", output_path=output)
    checkpoint = load_training_checkpoint(
        synthetic_run["run"] / "checkpoints" / "latest.pt"
    )
    parent = load_reference_site_model_bundle(synthetic_run["bundle"]["path"])
    restored_runtime = instantiate_reference_site_model_bundle(
        parent, device="cpu", dtype=torch.float64
    )
    restored_runtime.model.load_state_dict(checkpoint.model_state_dict, strict=True)
    restored_runtime.model.eval()
    direct = ReferenceSitePredictor(restored_runtime)
    portable = load_reference_site_predictor(
        output, device="cpu", dtype=torch.float64
    )
    sample = synthetic_run["bundle"]["samples"][2]
    solvers = [TRAIN_FIXED]
    if sample.template_id in portable.runtime.evaluation_policies:
        solvers.append(EVAL_ADAPTIVE)
    for solver in solvers:
        expected = direct.predict_sample(
            sample,
            solver_path=solver,
            compute_forces=True,
            compute_stress=True,
        )
        actual = portable.predict_sample(
            sample,
            solver_path=solver,
            compute_forces=True,
            compute_stress=True,
        )
        for name in ("energy", "forces", "stress", "stress_voigt"):
            assert torch.equal(getattr(actual, name), getattr(expected, name))


def test_best_and_latest_export_distinct_managed_epochs(training_bundle, tmp_path):
    config_path, payload = _simple_case(tmp_path, training_bundle)
    payload["fit"]["max_epochs"] = 2
    # This synthetic fixture improves by decreasing validation loss.  Selecting
    # the maximum therefore keeps epoch zero as best while latest reaches one.
    payload["selection"]["mode"] = "max"
    payload["scheduler"]["mode"] = "max"
    payload["output_directory"] = "different-best-run"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    train_module = importlib.import_module("refsite_mlip.cli.train")
    train_module.run_training(config_path)
    run = tmp_path / "different-best-run"
    best_checkpoint = load_training_checkpoint(run / "checkpoints" / "best.pt")
    latest_checkpoint = load_training_checkpoint(run / "checkpoints" / "latest.pt")
    assert best_checkpoint.progress.last_completed_epoch == 0
    assert latest_checkpoint.progress.last_completed_epoch == 1

    best_path = tmp_path / "best-export.pt"
    latest_path = tmp_path / "latest-export.pt"
    best_report = export_bundle(run, source="best", output_path=best_path)
    latest_report = export_bundle(run, source="latest", output_path=latest_path)
    assert best_report["source"]["epoch"] == 0
    assert latest_report["source"]["epoch"] == 1
    best_bundle = load_reference_site_model_bundle(best_path)
    latest_bundle = load_reference_site_model_bundle(latest_path)
    for key in best_checkpoint.model_state_dict:
        assert torch.equal(
            best_bundle.model_state[key], best_checkpoint.model_state_dict[key]
        )
        assert torch.equal(
            latest_bundle.model_state[key], latest_checkpoint.model_state_dict[key]
        )
    assert best_bundle.bundle_fingerprint != latest_bundle.bundle_fingerprint


@pytest.mark.parametrize("status_name", ["failed", "interrupted"])
def test_recoverable_checkpoint_exports_from_terminal_failure_status(
    synthetic_run, tmp_path, status_name
):
    status_path = synthetic_run["run"] / "run_status.json"
    before = status_path.read_bytes()
    try:
        status = json.loads(before)
        status["status"] = status_name
        status["failure_phase"] = "injected_fixture_failure"
        status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
        report = export_bundle(
            synthetic_run["run"],
            source="latest",
            output_path=tmp_path / f"{status_name}.pt",
        )
        assert report["status"] == "completed"
    finally:
        status_path.write_bytes(before)


def test_export_does_not_require_training_data_to_remain_present(
    synthetic_run, tmp_path
):
    train = synthetic_run["root"] / "train.xyz"
    validation = synthetic_run["root"] / "validation.xyz"
    moved_train = synthetic_run["root"] / "train.xyz.moved"
    moved_validation = synthetic_run["root"] / "validation.xyz.moved"
    train.rename(moved_train)
    validation.rename(moved_validation)
    try:
        report = export_bundle(
            synthetic_run["run"],
            source="latest",
            output_path=tmp_path / "without-data.pt",
        )
        assert report["output_written"] is True
    finally:
        moved_train.rename(train)
        moved_validation.rename(validation)


def test_evaluation_policy_preserved_and_adaptive_prediction_exact(
    training_bundle, tmp_path
):
    parent = training_bundle["bundle"]
    parent_bindings = {
        binding.template_id: binding for binding in parent.template_bindings
    }
    *_, original_policies, _ = _capture_case(_typed_crystal_data())
    policies = {
        template_id: replace(
            original_policies[template_id],
            template_fingerprint=binding.full_template_fingerprint,
            content_fingerprint=None,
        )
        for template_id, binding in parent_bindings.items()
    }
    loaded_parent = instantiate_reference_site_model_bundle(
        parent, device="cpu", dtype=torch.float64
    )
    policy_bundle = capture_reference_site_model_bundle(
        model=loaded_parent.model,
        structural_artifacts={
            key: value.structural_artifact for key, value in parent_bindings.items()
        },
        phase_specifications={
            key: value.phase_specification for key, value in parent_bindings.items()
        },
        evaluation_policies=policies,
        default_template_id=parent.default_template_id,
        provenance={"purpose": "policy-export-test"},
    )
    policy_path = tmp_path / "policy-initial.pt"
    save_reference_site_model_bundle(policy_path, policy_bundle)
    policy_fixture = {
        "path": policy_path,
        "bundle": policy_bundle,
        "samples": training_bundle["samples"],
    }
    config_path, payload = _simple_case(tmp_path, policy_fixture)
    payload["fit"]["max_epochs"] = 1
    payload["output_directory"] = "policy-run"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    train_module = importlib.import_module("refsite_mlip.cli.train")
    train_module.run_training(config_path)
    run = tmp_path / "policy-run"
    exported_path = tmp_path / "policy-export.pt"
    export_bundle(run, source="latest", output_path=exported_path)
    exported = load_reference_site_model_bundle(exported_path)
    exported_bindings = {
        binding.template_id: binding for binding in exported.template_bindings
    }
    for template_id, expected in policies.items():
        assert exported_bindings[template_id].evaluation_policy is not None
        assert (
            exported_bindings[template_id].evaluation_policy.to_dict()
            == expected.to_dict()
        )

    checkpoint = load_training_checkpoint(run / "checkpoints" / "latest.pt")
    restored_runtime = instantiate_reference_site_model_bundle(
        policy_bundle, device="cpu", dtype=torch.float64
    )
    restored_runtime.model.load_state_dict(checkpoint.model_state_dict, strict=True)
    direct = ReferenceSitePredictor(restored_runtime)
    portable = load_reference_site_predictor(exported_path)
    sample = training_bundle["samples"][2]
    expected = direct.predict_sample(
        sample,
        solver_path=EVAL_ADAPTIVE,
        compute_forces=True,
        compute_stress=True,
    )
    actual = portable.predict_sample(
        sample,
        solver_path=EVAL_ADAPTIVE,
        compute_forces=True,
        compute_stress=True,
    )
    for name in ("energy", "forces", "stress", "stress_voigt"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))


def test_missing_stored_initial_bundle_requires_exact_override(
    synthetic_run, tmp_path
):
    initial = synthetic_run["bundle"]["path"]
    moved = initial.with_name(initial.name + ".temporarily-moved")
    initial.rename(moved)
    try:
        with pytest.raises(CLIError) as caught:
            export_bundle(
                synthetic_run["run"],
                source="latest",
                output_path=tmp_path / "missing-default.pt",
            )
        assert caught.value.reason_code == "INITIAL_BUNDLE_NOT_FOUND"
        report = export_bundle(
            synthetic_run["run"],
            source="latest",
            output_path=tmp_path / "override.pt",
            initial_bundle_path=moved,
        )
        assert report["parent_initial_bundle_sha256"] == (
            synthetic_run["bundle"]["bundle"].bundle_fingerprint
        )
    finally:
        moved.rename(initial)


def test_corrupt_checkpoint_history_and_nonfinite_state_are_structured(
    synthetic_run, tmp_path
):
    source = synthetic_run["run"] / "checkpoints" / "latest.pt"
    before = source.read_bytes()
    try:
        source.write_bytes(b"truncated")
        with pytest.raises(CLIError) as caught:
            export_bundle(
                synthetic_run["run"],
                source="latest",
                output_path=tmp_path / "truncated.pt",
            )
        assert caught.value.reason_code == "SOURCE_CHECKPOINT_LOAD_FAILED"
        assert caught.value.checkpoint_stage == "weights_only_load"
    finally:
        source.write_bytes(before)

    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["fit_history"][0]["epoch_index"] = 7
    try:
        torch.save(payload, source)
        with pytest.raises(CLIError) as caught:
            export_bundle(
                synthetic_run["run"],
                source="latest",
                output_path=tmp_path / "history.pt",
            )
        assert caught.value.reason_code == "CHECKPOINT_HISTORY_INVALID"
        assert caught.value.checkpoint_stage == "history_progress"
    finally:
        source.write_bytes(before)

    payload = torch.load(source, map_location="cpu", weights_only=True)
    key = next(
        name
        for name, value in payload["model_state_dict"].items()
        if value.is_floating_point()
    )
    payload["model_state_dict"][key] = payload["model_state_dict"][key].clone()
    payload["model_state_dict"][key].reshape(-1)[0] = float("nan")
    try:
        torch.save(payload, source)
        with pytest.raises(CLIError) as caught:
            export_bundle(
                synthetic_run["run"],
                source="latest",
                output_path=tmp_path / "nonfinite.pt",
            )
        assert caught.value.reason_code == "NONFINITE_CHECKPOINT_MODEL_STATE"
        assert caught.value.config_field == key
    finally:
        source.write_bytes(before)


def test_running_status_and_checkpoint_symlink_are_rejected(
    synthetic_run, tmp_path
):
    status_path = synthetic_run["run"] / "run_status.json"
    status_before = status_path.read_bytes()
    try:
        status = json.loads(status_before)
        status["status"] = "running"
        status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
        with pytest.raises(CLIError) as caught:
            export_bundle(
                synthetic_run["run"],
                source="latest",
                output_path=tmp_path / "running.pt",
            )
        assert caught.value.reason_code == "ACTIVE_RUN_REJECTED"
    finally:
        status_path.write_bytes(status_before)

    checkpoint = synthetic_run["run"] / "checkpoints" / "latest.pt"
    backup = synthetic_run["run"] / "checkpoints" / "latest.backup"
    checkpoint.rename(backup)
    checkpoint.symlink_to(backup.name)
    try:
        with pytest.raises(CLIError) as caught:
            export_bundle(
                synthetic_run["run"],
                source="latest",
                output_path=tmp_path / "source-symlink.pt",
            )
        assert caught.value.reason_code == "SOURCE_CHECKPOINT_LOAD_FAILED"
    finally:
        checkpoint.unlink()
        backup.rename(checkpoint)


def test_resumed_run_latest_and_preserved_best_are_exportable(
    training_bundle, tmp_path
):
    config_path, payload = _simple_case(tmp_path, training_bundle)
    payload["fit"]["max_epochs"] = 1
    payload["output_directory"] = "resumed-export-run"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    train_module = importlib.import_module("refsite_mlip.cli.train")
    train_module.run_training(config_path)
    run = tmp_path / "resumed-export-run"
    resume_training(run, max_epochs=2)
    latest = load_training_checkpoint(run / "checkpoints" / "latest.pt")
    best = load_training_checkpoint(run / "checkpoints" / "best.pt")
    latest_report = export_bundle(
        run, source="latest", output_path=tmp_path / "resumed-latest.pt"
    )
    best_report = export_bundle(
        run, source="best", output_path=tmp_path / "resumed-best.pt"
    )
    assert latest_report["source"]["epoch"] == latest.progress.last_completed_epoch
    assert best_report["source"]["epoch"] == best.progress.last_completed_epoch
    assert latest_report["source"]["global_step"] == latest.progress.global_step
