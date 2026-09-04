from __future__ import annotations

import importlib
import json
import random

import numpy as np
import pytest
import torch

from refsite_mlip.cli.errors import CLIError, CLIInterruptedError
from refsite_mlip.cli.main import build_parser, main
from refsite_mlip.cli.train import seed_training_runtime
from refsite_mlip.training.run_directory import (
    RunDirectoryError,
    TrainingRunDirectory,
    canonical_runtime_json,
)


def test_seed_training_runtime_is_strict_and_reproducible():
    seed_training_runtime(12345)
    first = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )
    seed_training_runtime(12345)
    second = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    for invalid in (True, 1.5, "17"):
        with pytest.raises(ValueError):
            seed_training_runtime(invalid)

    seed_training_runtime(-1)
    negative = torch.rand(2)
    seed_training_runtime(-1)
    assert torch.equal(negative, torch.rand(2))


def test_run_directory_is_exclusive_atomic_and_strict_json(tmp_path, monkeypatch):
    root = tmp_path / "run"
    directory = TrainingRunDirectory.create(root)
    directory.write_resolved_config({"z": 2, "a": 1})
    directory.write_preflight({"ready": True})
    directory.write_status({"status": "running"})
    directory.write_status({"status": "completed"})

    assert root.is_dir()
    assert root.joinpath("resolved_config.json").read_text() == '{"a":1,"z":2}\n'
    assert json.loads(root.joinpath("run_status.json").read_text()) == {
        "status": "completed"
    }
    assert canonical_runtime_json({"b": 1, "a": [True, None]}) == (
        '{"a":[true,null],"b":1}'
    )
    with pytest.raises(RunDirectoryError, match="already exists"):
        TrainingRunDirectory.create(root)
    with pytest.raises(ValueError, match="NaN or Infinity"):
        directory.write_status({"value": float("nan")})

    original = root.joinpath("run_status.json").read_bytes()
    module = importlib.import_module("refsite_mlip.training.run_directory")
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("injected replace failure")),
    )
    with pytest.raises(RunDirectoryError) as caught:
        directory.write_status({"status": "failed"})
    assert caught.value.reason_code == "ATOMIC_RUNTIME_WRITE_FAILED"
    assert root.joinpath("run_status.json").read_bytes() == original
    assert not list(root.glob(".run_status.json.*.tmp"))


def test_run_directory_rejects_output_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "run-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RunDirectoryError) as caught:
        TrainingRunDirectory.create(link)
    assert caught.value.reason_code == "OUTPUT_SYMLINK_REJECTED"


def test_train_cli_json_progress_and_dry_run_wiring(monkeypatch, capsys):
    module = importlib.import_module("refsite_mlip.cli.train")
    calls = []

    def fake_run(path, *, dry_run, progress, overrides):
        calls.append((path, dry_run, progress is not None, overrides))
        if progress is not None:
            progress("synthetic progress")
        return {"status": "completed"}

    monkeypatch.setattr(module, "run_training", fake_run)
    monkeypatch.setattr(
        module, "render_train_result_json", lambda result: '{"status":"completed"}'
    )
    monkeypatch.setattr(module, "render_train_result_human", lambda result: "done")

    assert main(["train", "run.json", "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"status":"completed"}\n'
    assert "synthetic progress" in captured.err
    assert calls[-1][:3] == ("run.json", False, True)
    assert calls[-1][3].device is None

    assert main(["train", "run.json", "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "done\n"
    assert captured.err == ""
    assert calls[-1][:3] == ("run.json", True, False)

    assert main(
        [
            "train",
            "--config",
            "run.yaml",
            "--dry-run",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-epochs",
            "7",
            "--batch-size",
            "3",
            "--validation-batch-size",
            "2",
            "--learning-rate",
            "0.0003",
            "--r-ot",
            "4.5",
            "--r-mp",
            "3.25",
            "--output-directory",
            "override-output",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out == "done\n" and captured.err == ""
    path, dry_run, progress_enabled, overrides = calls[-1]
    assert (path, dry_run, progress_enabled) == ("run.yaml", True, False)
    assert overrides.dtype == "float32"
    assert overrides.max_epochs == 7
    assert overrides.validation_batch_size == 2
    assert overrides.output_directory == "override-output"


def test_training_config_positional_alias_ambiguity_is_usage_error():
    parser = build_parser()
    assert parser.parse_args(
        ["validate-train-config", "run.json"]
    ).config_path == "run.json"
    assert parser.parse_args(
        ["validate-train-config", "--config", "run.yaml"]
    ).config_option == "run.yaml"
    with pytest.raises(SystemExit) as missing:
        parser.parse_args(["validate-train-config"])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as conflicting:
        parser.parse_args(
            ["validate-train-config", "run.json", "--config", "other.yaml"]
        )
    assert conflicting.value.code == 2


def test_scratch_train_dry_run_stops_before_seed_model_optimizer_or_output(
    tmp_path, monkeypatch, capsys
):
    from test_training_run_config import _v2_payload

    payload = _v2_payload()
    output = tmp_path / "scratch-output"
    payload["output_directory"] = str(output)
    path = tmp_path / "scratch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    module = importlib.import_module("refsite_mlip.cli.train")

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("execution API must not be called")

    monkeypatch.setattr(module, "seed_training_runtime", forbidden)
    monkeypatch.setattr(module, "_prepare_training_runtime", forbidden)
    rng = torch.get_rng_state().clone()

    assert main(["validate-train-config", str(path), "--json"]) == 0
    captured = capsys.readouterr()
    validation_report = json.loads(captured.out)
    assert validation_report["status"] == "scratch_config_ready"
    assert validation_report["training_executed"] is False
    assert captured.err == ""

    assert main(["train", str(path), "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "Status: config ready" in captured.out
    assert "Scratch execution: not implemented" in captured.out
    assert captured.err == ""
    assert not output.exists()
    assert torch.equal(torch.get_rng_state(), rng)

    assert main(["train", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SCRATCH_EXECUTION_NOT_IMPLEMENTED" in captured.err
    assert not output.exists()
    assert torch.equal(torch.get_rng_state(), rng)


def test_locked_toctou_reload_reapplies_identical_effective_overrides(tmp_path):
    from test_training_run_config import _payload

    from refsite_mlip.config import (
        TrainingRunConfigOverrides,
        load_effective_training_run_config,
    )

    path = tmp_path / "run.json"
    payload = _payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    overrides = TrainingRunConfigOverrides(
        max_epochs=7,
        validation_batch_size=2,
        output_directory="cli-output",
    )
    config = load_effective_training_run_config(
        path, overrides, cli_cwd=tmp_path
    )
    module = importlib.import_module("refsite_mlip.cli.train")
    reloaded = module._reload_effective_config_for_toctou(
        config,
        overrides=overrides,
        cli_cwd=tmp_path,
    )
    assert reloaded.config_fingerprint == config.config_fingerprint

    payload["runtime"]["dtype"] = "float32"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CLIError) as caught:
        module._reload_effective_config_for_toctou(
            config,
            overrides=overrides,
            cli_cwd=tmp_path,
        )
    assert caught.value.reason_code == "TRAIN_CONFIG_TOCTOU_MISMATCH"


def test_train_cli_interrupt_exit_code_and_debug_traceback(monkeypatch, capsys):
    module = importlib.import_module("refsite_mlip.cli.train")

    def interrupted(*args, **kwargs):
        del args, kwargs
        raise CLIInterruptedError(
            "TRAINING_INTERRUPTED",
            "training interrupted",
            stage="training.fit",
            failure_phase="fit",
        )

    monkeypatch.setattr(module, "run_training", interrupted)
    assert main(["train", "run.json"]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "reason='TRAINING_INTERRUPTED'" in captured.err
    assert "Traceback" not in captured.err

    assert main(["train", "run.json", "--debug"]) == 130
    captured = capsys.readouterr()
    assert "Traceback" in captured.err
