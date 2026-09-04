"""Black-box CPU workflow exercised only through an installed project wheel."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import pytest

from test_wheel_installation import (
    InstalledWheelEnvironment,
    installed_wheel_environment,
)


def _json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    return json.loads(text, parse_constant=reject_constant)


def _terminal_json(result: subprocess.CompletedProcess[str]) -> Mapping[str, Any]:
    assert result.returncode == 0
    assert result.stdout.endswith("\n")
    assert result.stdout.count("\n") == 1
    value = _json_loads(result.stdout)
    assert isinstance(value, Mapping)
    assert "Traceback" not in result.stderr
    return value


def _failure(
    result: subprocess.CompletedProcess[str],
    *,
    exit_code: int,
    reason: str | None = None,
) -> None:
    assert result.returncode == exit_code
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    if reason is not None:
        assert reason in result.stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal(path: Path) -> tuple[dict[str, Any], ...]:
    data = path.read_bytes()
    assert data.endswith(b"\n")
    lines = data.splitlines()
    assert lines
    return tuple(_json_loads(line.decode("utf-8")) for line in lines)


def _assert_no_presentation_time(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert key not in {
                "elapsed",
                "elapsed_seconds",
                "eta",
                "eta_seconds",
                "wall_clock",
                "timestamp",
            }
            _assert_no_presentation_time(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_presentation_time(item)


def _copy_support(installed: InstalledWheelEnvironment, name: str) -> Path:
    source = installed.repository_root / "tests" / "integration" / name
    assert source.is_file()
    support = installed.root / "workflow-support"
    support.mkdir(exist_ok=True)
    target = support / name
    shutil.copyfile(source, target)
    assert not target.is_relative_to(installed.repository_root)
    return target


def _critical_run_bytes(run: Path) -> dict[str, bytes]:
    candidates = (
        "metrics.jsonl",
        "checkpoints/epoch_000000.pt",
        "checkpoints/epoch_000001.pt",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
    )
    return {
        relative: (run / relative).read_bytes()
        for relative in candidates
        if (run / relative).is_file()
    }


@dataclass(frozen=True)
class InstalledWorkflow:
    installed: InstalledWheelEnvironment
    manifest: Mapping[str, Any]
    split_run: Path
    continuous_run: Path
    best_bundle: Path
    latest_bundle: Path
    predictions: Path
    evaluation_report: Path
    split_train: subprocess.CompletedProcess[str]
    split_resume: subprocess.CompletedProcess[str]
    continuous_train: subprocess.CompletedProcess[str]
    journal_before_resume: bytes
    epoch_zero_before_resume: bytes
    best_before_resume: bytes
    probe: Mapping[str, Any]


@pytest.fixture(scope="module")
def installed_workflow(
    installed_wheel_environment: InstalledWheelEnvironment,
) -> InstalledWorkflow:
    installed = installed_wheel_environment
    assert installed.wheel.suffix == ".whl"
    assert not installed.wheel.is_relative_to(installed.repository_root)
    assert "PYTHONPATH" not in installed.environment

    fixture_script = _copy_support(installed, "installed_workflow_fixture.py")
    probe_script = _copy_support(installed, "installed_workflow_probe.py")
    fixture_root = installed.root / "workflow-fixture"
    generated = installed.run(
        (installed.python, fixture_script, fixture_root),
        cwd=installed.root / "work",
    )
    manifest = _terminal_json(generated)
    assert manifest["schema_version"] == "refsite_installed_workflow_fixture_v1"
    assert manifest["device"] == "cpu"
    assert manifest["dtype"] == "float64"
    assert manifest["species_vocabulary"] == [6, 41]
    assert len(manifest["template_ids"]) == 2
    assert set(manifest["template_site_counts"].values()) == {8, 16}
    assert {item["K"] for item in manifest["train_structures"]} == {0, 1}
    assert {item["K"] for item in manifest["validation_structures"]} == {0, 1}

    # Both installed entry points must resolve from site-packages while the
    # subprocess working directory remains outside the checkout.
    import_probe = installed.run(
        (
            installed.python,
            "-c",
            "import json, pathlib, refsite_mlip; "
            "print(json.dumps({'module': str(pathlib.Path(refsite_mlip.__file__).resolve())}, sort_keys=True))",
        )
    )
    module_path = Path(_terminal_json(import_probe)["module"])
    assert "site-packages" in module_path.parts
    assert not module_path.is_relative_to(installed.repository_root / "src")

    console_version = installed.run_console("version")
    module_version = installed.run_module("version")
    assert console_version.stdout == module_version.stdout
    # Dependency import warnings are allowed on stderr; both entry points must
    # still have identical behavior and never emit a traceback.
    assert console_version.stderr == module_version.stderr
    assert "Traceback" not in console_version.stderr

    split = manifest["cases"]["split"]
    continuous = manifest["cases"]["continuous"]
    split_config = str(split["config"])
    continuous_config = str(continuous["config"])
    split_run = Path(split["output_directory"])
    continuous_run = Path(continuous["output_directory"])

    validate_console = installed.run_console(
        "validate-train-config", split_config, "--json"
    )
    validate_module = installed.run_module(
        "validate-train-config", "--config", split_config, "--json"
    )
    assert _terminal_json(validate_console) == _terminal_json(validate_module)
    assert validate_console.stdout == validate_module.stdout

    dry_run = installed.run_module(
        "train", "--config", split_config, "--dry-run", "--json"
    )
    assert _terminal_json(dry_run) == _terminal_json(validate_console)
    assert not split_run.exists()
    assert "loading training configuration" in dry_run.stderr
    assert "preparing data and reference templates" in dry_run.stderr
    assert "training started" not in dry_run.stderr

    split_train = installed.run_console(
        "train", "--config", split_config, "--json"
    )
    split_result = _terminal_json(split_train)
    assert split_result["status"] == "completed"
    for text in (
        "refsite-mlip: loading training configuration",
        "refsite-mlip: preparing data and reference templates",
        "refsite-mlip: initializing training run",
        "refsite-mlip: training started",
        "Reference-site MLIP training",
        "Epoch 001/1",
        "elapsed=",
        "eta=",
        "Training completed",
    ):
        assert text in split_train.stderr

    required = {
        "resolved_config.json",
        "preflight.json",
        "data_manifest.json",
        "initial_bundle.pt",
        "run_status.json",
        "metrics.jsonl",
        "checkpoints/epoch_000000.pt",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
    }
    assert all((split_run / relative).is_file() for relative in required)
    before = _critical_run_bytes(split_run)
    journal_before = before["metrics.jsonl"]
    first_events = _journal(split_run / "metrics.jsonl")
    assert len(first_events) == 1
    assert first_events[0]["epoch_index"] == 0

    invalid_resume = installed.run_console(
        "resume", str(split_run), "--max-epochs", "1", "--json", check=False
    )
    _failure(invalid_resume, exit_code=1, reason="MAX_EPOCHS_NOT_INCREASED")
    assert _critical_run_bytes(split_run) == before

    # At one epoch best.pt and latest.pt are aliases for identical semantic
    # checkpoint state.  Their user-facing source kinds remain different, but
    # portable bundle provenance and SHA must be alias-neutral.
    pre_best = installed.root / "work" / "pre-resume-best.pt"
    pre_latest = installed.root / "work" / "pre-resume-latest.pt"
    pre_best_result = installed.run_console(
        "export-bundle",
        str(split_run),
        "--source",
        "best",
        "--output",
        str(pre_best),
        "--json",
    )
    pre_latest_result = installed.run_module(
        "export-bundle",
        str(split_run),
        "--source",
        "latest",
        "--output",
        str(pre_latest),
        "--json",
    )
    pre_best_report = _terminal_json(pre_best_result)
    pre_latest_report = _terminal_json(pre_latest_result)
    assert pre_best_report["source"]["kind"] == "best"
    assert pre_latest_report["source"]["kind"] == "latest"
    assert pre_best_report["source"]["epoch"] == 0
    assert pre_latest_report["source"]["epoch"] == 0
    assert pre_best_report["bundle_sha256"] == pre_latest_report["bundle_sha256"]

    existing_run = installed.run_console(
        "train", "--config", split_config, "--json", check=False
    )
    _failure(existing_run, exit_code=2, reason="OUTPUT_ALREADY_EXISTS")
    assert _critical_run_bytes(split_run) == before

    split_resume = installed.run_module(
        "resume", str(split_run), "--max-epochs", "2", "--json"
    )
    resume_result = _terminal_json(split_resume)
    assert resume_result["status"] == "completed"
    assert "Epoch 001/2" not in split_resume.stderr
    assert "Epoch 002/2" in split_resume.stderr
    assert "Training completed" in split_resume.stderr

    assert (split_run / "checkpoints" / "epoch_000001.pt").is_file()
    assert (split_run / "checkpoints" / "epoch_000000.pt").read_bytes() == before[
        "checkpoints/epoch_000000.pt"
    ]
    assert (split_run / "checkpoints" / "best.pt").read_bytes() == before[
        "checkpoints/best.pt"
    ]
    journal_after = (split_run / "metrics.jsonl").read_bytes()
    assert journal_after.startswith(journal_before)
    events = _journal(split_run / "metrics.jsonl")
    assert [event["epoch_index"] for event in events] == [0, 1]
    assert events[0]["global_step_end"] == events[1]["global_step_start"]
    assert events[0]["learning_rates_after_scheduler"] == events[1][
        "learning_rates_before_scheduler"
    ]
    assert events[0]["is_best"] is True
    assert events[1]["best_epoch"] == 0
    status = _json_loads((split_run / "run_status.json").read_text("utf-8"))
    assert status["status"] == "completed"
    assert status["completed_epochs"] == 2
    assert status["global_step"] == events[-1]["global_step_end"]
    assert status["metrics_event_count"] == 2
    assert status["metrics_last_epoch"] == 1

    best_bundle = installed.root / "work" / "best-model.pt"
    latest_bundle = installed.root / "work" / "latest-model.pt"
    best_export = installed.run_console(
        "export-bundle",
        str(split_run),
        "--source",
        "best",
        "--output",
        str(best_bundle),
        "--json",
    )
    latest_export = installed.run_module(
        "export-bundle",
        str(split_run),
        "--source",
        "latest",
        "--output",
        str(latest_bundle),
        "--json",
    )
    assert _terminal_json(best_export)["source"]["kind"] == "best"
    assert _terminal_json(latest_export)["source"]["kind"] == "latest"
    assert best_bundle.is_file() and latest_bundle.is_file()

    inspect_best = installed.run_console("inspect-bundle", str(best_bundle), "--json")
    inspect_latest = installed.run_module(
        "inspect-bundle", str(latest_bundle), "--json"
    )
    for report in (_terminal_json(inspect_best), _terminal_json(inspect_latest)):
        assert report["schema_version"] == "reference_site_model_bundle_v1"
        assert report["template_ids"] == sorted(manifest["template_ids"])
        assert report["species_vocabulary"] == [6, 41]

    predictions = installed.root / "work" / "predictions.xyz"
    predict_arguments = (
        "predict",
        "--bundle",
        str(latest_bundle),
        "--input",
        str(split["mixed_labeled"]),
        "--output",
        str(predictions),
        "--template-key",
        manifest["template_key"],
        "--solver",
        "train-fixed",
        "--properties",
        "energy,forces,stress",
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--batch-size",
        "2",
        "--json",
    )
    prediction_result = installed.run_console(*predict_arguments)
    prediction_report = _terminal_json(prediction_result)
    assert prediction_report["frame_count"] == 4
    assert predictions.is_file()

    evaluation_report = installed.root / "work" / "evaluation.json"
    evaluation_arguments = (
        "evaluate",
        "--bundle",
        str(latest_bundle),
        "--input",
        str(split["mixed_labeled"]),
        "--template-key",
        manifest["template_key"],
        "--solver",
        "train-fixed",
        "--terms",
        "energy,forces,stress",
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--batch-size",
        "2",
        "--energy-mode",
        "per-atom",
        "--energy-scale",
        "2",
        "--force-scale",
        "3",
        "--stress-scale",
        "4",
        "--energy-weight",
        "1.5",
        "--force-weight",
        "2.5",
        "--stress-weight",
        "3.5",
    )
    evaluation_stdout = installed.run_module(*evaluation_arguments, "--json")
    evaluation_payload = _terminal_json(evaluation_stdout)
    assert evaluation_payload["frame_count"] == 4
    evaluation_file = installed.run_console(
        *evaluation_arguments, "--output", str(evaluation_report)
    )
    assert evaluation_file.returncode == 0
    assert evaluation_file.stdout == ""
    assert "Traceback" not in evaluation_file.stderr
    assert _json_loads(evaluation_report.read_text("utf-8")) == evaluation_payload

    # A quiet run is trajectory-equivalent but suppresses stage/start/epoch
    # presentation.  The terminal outcome remains visible on stderr.
    continuous_train = installed.run_console(
        "train", continuous_config, "--quiet", "--json"
    )
    continuous_result = _terminal_json(continuous_train)
    assert continuous_result["status"] == "completed"
    assert "refsite-mlip:" not in continuous_train.stderr
    assert "Reference-site MLIP training" not in continuous_train.stderr
    assert "Epoch " not in continuous_train.stderr
    assert "Training completed |" in continuous_train.stderr
    assert (continuous_run / "metrics.jsonl").read_bytes() == (
        split_run / "metrics.jsonl"
    ).read_bytes()

    # Installed-package-only probe performs strict weights-only checkpoint and
    # bundle loads plus direct Predictor/loss/ASE and exact trajectory checks.
    probed = installed.run(
        (
            installed.python,
            probe_script,
            "--fixture-manifest",
            manifest["manifest_path"],
            "--continuous-run",
            continuous["output_directory"],
            "--split-run",
            split["output_directory"],
            "--best-bundle",
            best_bundle,
            "--latest-bundle",
            latest_bundle,
            "--predictions",
            predictions,
            "--evaluation-report",
            evaluation_report,
        )
    )
    probe = _terminal_json(probed)

    # No-overwrite, symlink, corrupt input, runtime validation, and argparse
    # exit-code contracts are checked through installed subprocesses.
    prediction_before = predictions.read_bytes()
    no_overwrite = installed.run_console(*predict_arguments, check=False)
    _failure(no_overwrite, exit_code=1, reason="OUTPUT_EXISTS")
    assert predictions.read_bytes() == prediction_before

    symlink_target = installed.root / "work" / "symlink-target.xyz"
    symlink_target.write_bytes(b"preserved-target")
    symlink_output = installed.root / "work" / "prediction-link.xyz"
    symlink_output.symlink_to(symlink_target)
    symlink_arguments = list(predict_arguments)
    symlink_arguments[symlink_arguments.index(str(predictions))] = str(symlink_output)
    symlink_failure = installed.run_console(*symlink_arguments, check=False)
    _failure(symlink_failure, exit_code=1, reason="OUTPUT_SYMLINK_REJECTED")
    assert symlink_output.is_symlink()
    assert symlink_target.read_bytes() == b"preserved-target"

    corrupt_bundle = installed.root / "work" / "corrupt.pt"
    corrupt_bundle.write_bytes(b"not a torch archive")
    corrupt_bundle_result = installed.run_module(
        "inspect-bundle", str(corrupt_bundle), "--json", check=False
    )
    _failure(corrupt_bundle_result, exit_code=1, reason="SAFE_LOAD_FAILURE")

    critical_before_corrupt = _critical_run_bytes(split_run)
    latest_path = split_run / "checkpoints" / "latest.pt"
    temporary = latest_path.parent / ".latest.workflow-corrupt.tmp"
    temporary.write_bytes(b"truncated checkpoint")
    os.replace(temporary, latest_path)
    try:
        corrupt_checkpoint_result = installed.run_console(
            "export-bundle",
            str(split_run),
            "--source",
            "latest",
            "--output",
            str(installed.root / "work" / "corrupt-source-export.pt"),
            "--json",
            check=False,
        )
        _failure(
            corrupt_checkpoint_result,
            exit_code=1,
            reason="SOURCE_CHECKPOINT_LOAD_FAILED",
        )
        assert latest_path.read_bytes() == b"truncated checkpoint"
        assert (split_run / "metrics.jsonl").read_bytes() == critical_before_corrupt[
            "metrics.jsonl"
        ]
        assert (split_run / "checkpoints" / "epoch_000000.pt").read_bytes() == (
            critical_before_corrupt["checkpoints/epoch_000000.pt"]
        )
        assert (split_run / "checkpoints" / "best.pt").read_bytes() == (
            critical_before_corrupt["checkpoints/best.pt"]
        )
        assert (split_run / "checkpoints" / "epoch_000001.pt").read_bytes() == (
            critical_before_corrupt["checkpoints/epoch_000001.pt"]
        )
    finally:
        temporary.write_bytes(critical_before_corrupt["checkpoints/latest.pt"])
        os.replace(temporary, latest_path)
    assert _critical_run_bytes(split_run) == critical_before_corrupt

    usage = installed.run_console("predict", "--bundle", str(latest_bundle), check=False)
    _failure(usage, exit_code=2)

    # Presentation timing is intentionally absent from every canonical JSON
    # artifact.  Binary checkpoint/bundle payloads are checked by the probe.
    for run in (split_run, continuous_run):
        for name in (
            "resolved_config.json",
            "preflight.json",
            "data_manifest.json",
            "run_status.json",
        ):
            _assert_no_presentation_time(
                _json_loads((run / name).read_text(encoding="utf-8"))
            )
        for event in _journal(run / "metrics.jsonl"):
            _assert_no_presentation_time(event)

    # Input/reference/config files remain byte-identical.
    for case in manifest["cases"].values():
        directory = Path(case["directory"])
        for basename, expected in case["input_sha256"].items():
            assert _sha256(directory / basename) == expected

    return InstalledWorkflow(
        installed=installed,
        manifest=manifest,
        split_run=split_run,
        continuous_run=continuous_run,
        best_bundle=best_bundle,
        latest_bundle=latest_bundle,
        predictions=predictions,
        evaluation_report=evaluation_report,
        split_train=split_train,
        split_resume=split_resume,
        continuous_train=continuous_train,
        journal_before_resume=journal_before,
        epoch_zero_before_resume=before["checkpoints/epoch_000000.pt"],
        best_before_resume=before["checkpoints/best.pt"],
        probe=probe,
    )


def test_installed_wheel_cli_workflow_and_filesystem_contracts(
    installed_workflow: InstalledWorkflow,
) -> None:
    result = installed_workflow
    assert result.best_bundle.is_file()
    assert result.latest_bundle.is_file()
    assert result.predictions.is_file()
    assert result.evaluation_report.is_file()
    assert (result.split_run / "metrics.jsonl").read_bytes().startswith(
        result.journal_before_resume
    )
    assert (result.split_run / "checkpoints" / "epoch_000000.pt").read_bytes() == (
        result.epoch_zero_before_resume
    )
    assert (result.split_run / "checkpoints" / "best.pt").read_bytes() == (
        result.best_before_resume
    )


def test_installed_wheel_runtime_probe_contract(
    installed_workflow: InstalledWorkflow,
) -> None:
    report = installed_workflow.probe
    assert report["schema_version"] == "refsite_installed_workflow_probe_v1"
    assert report["status"] == "passed"
