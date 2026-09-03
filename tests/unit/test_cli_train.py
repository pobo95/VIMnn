from __future__ import annotations

import importlib
import json
import random

import numpy as np
import pytest
import torch

from refsite_mlip.cli.errors import CLIInterruptedError
from refsite_mlip.cli.main import main
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

    def fake_run(path, *, dry_run, progress):
        calls.append((path, dry_run, progress is not None))
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
    assert calls[-1] == ("run.json", False, True)

    assert main(["train", "run.json", "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "done\n"
    assert captured.err == ""
    assert calls[-1] == ("run.json", True, False)


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
