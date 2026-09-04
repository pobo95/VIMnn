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

    def fake_run(path, *, dry_run, progress_renderer, overrides):
        calls.append((path, dry_run, progress_renderer.enabled, overrides))
        progress_renderer.render_stage("synthetic progress")
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
    assert "synthetic progress" in captured.err
    assert calls[-1][:3] == ("run.json", True, True)

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
    assert captured.out == "done\n"
    assert "synthetic progress" in captured.err
    path, dry_run, progress_enabled, overrides = calls[-1]
    assert (path, dry_run, progress_enabled) == ("run.yaml", True, True)
    assert overrides.dtype == "float32"
    assert overrides.max_epochs == 7
    assert overrides.validation_batch_size == 2
    assert overrides.output_directory == "override-output"

    assert main(["train", "run.json", "--dry-run", "--quiet"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "done\n"
    assert captured.err == ""
    assert calls[-1][:3] == ("run.json", True, False)


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


def test_scratch_cli_requires_full_inputs_before_seed_model_optimizer_or_output(
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

    assert main(["validate-train-config", str(path), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "INPUT_NOT_FOUND" in captured.err

    assert main(["train", str(path), "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "INPUT_NOT_FOUND" in captured.err
    assert not output.exists()
    assert torch.equal(torch.get_rng_state(), rng)

    assert main(["train", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "INPUT_NOT_FOUND" in captured.err
    assert not output.exists()
    assert torch.equal(torch.get_rng_state(), rng)

    assert main(["train", str(path), "--debug"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" in captured.err
    assert "INPUT_NOT_FOUND" in captured.err
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


def test_scratch_training_branch_uses_checkpointed_orchestrator_without_reseeding(
    monkeypatch,
):
    module = importlib.import_module("refsite_mlip.cli.train")
    training = importlib.import_module("refsite_mlip.training")

    class FakePreparation:
        pass

    class FakeConfig:
        source_path = "/synthetic/run.yaml"

    class FakeResult:
        def to_dict(self):
            return {
                "checkpointed_fit_result": {},
                "startup": {},
                "terminal_status": {"z": 2, "status": "completed", "a": 1},
            }

    preparation = FakePreparation()
    config = FakeConfig()
    calls = []

    class FakeScratchError(RuntimeError):
        pass

    def execute(observed_config, observed_preparation, *, progress):
        calls.append((observed_config, observed_preparation, progress))
        return FakeResult()

    monkeypatch.setattr(module, "ScratchTrainingPreparation", FakePreparation)
    monkeypatch.setattr(
        module, "_load_preflight", lambda *args, **kwargs: (config, preparation)
    )
    monkeypatch.setattr(
        module,
        "seed_training_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("scratch startup owns training seeding")
        ),
    )
    monkeypatch.setattr(
        training, "ScratchCheckpointedTrainingError", FakeScratchError, raising=False
    )
    monkeypatch.setattr(
        training, "run_scratch_checkpointed_training", execute, raising=False
    )
    progress = lambda message: None

    report = module.run_training("run.yaml", progress=progress)
    assert calls == [(config, preparation, progress)]
    assert tuple(report) == ("a", "status", "z")
    assert report["status"] == "completed"


def test_scratch_structured_interrupt_maps_to_cli_interrupt_context(monkeypatch):
    module = importlib.import_module("refsite_mlip.cli.train")
    training = importlib.import_module("refsite_mlip.training")

    class FakePreparation:
        pass

    class FakeConfig:
        source_path = "/synthetic/run.json"

    class FakeScratchError(RuntimeError):
        def __init__(self):
            self.reason_code = "SCRATCH_CHECKPOINTED_TRAINING_INTERRUPTED"
            self.message = "synthetic interruption"
            self.stage = "fit"
            self.failure_phase = "fit"
            self.output_path = "/synthetic/output"
            self.template_id = "alpha"
            self.sample_id = "train:000001"
            self.completed_epochs = 1
            self.batch_index = 2
            self.global_step = 3
            self.rollback_performed = False
            self.bundle_fingerprint = "a" * 64
            self.config_fingerprint = "b" * 64
            self.original_reason_code = "KEYBOARD_INTERRUPT"
            self.original_error = KeyboardInterrupt()
            super().__init__(self.message)

    preparation = FakePreparation()
    config = FakeConfig()
    monkeypatch.setattr(module, "ScratchTrainingPreparation", FakePreparation)
    monkeypatch.setattr(
        module, "_load_preflight", lambda *args, **kwargs: (config, preparation)
    )
    monkeypatch.setattr(
        training, "ScratchCheckpointedTrainingError", FakeScratchError, raising=False
    )
    def interrupt_with_retained_context(*args, **kwargs):
        del args, kwargs
        interrupted = KeyboardInterrupt()
        interrupted.scratch_training_error = FakeScratchError()
        raise interrupted

    monkeypatch.setattr(
        training,
        "run_scratch_checkpointed_training",
        interrupt_with_retained_context,
        raising=False,
    )

    with pytest.raises(CLIInterruptedError) as caught:
        module.run_training("run.json")
    error = caught.value
    assert error.reason_code == "SCRATCH_CHECKPOINTED_TRAINING_INTERRUPTED"
    assert error.path == "/synthetic/output"
    assert error.source_path == "/synthetic/run.json"
    assert error.run_directory == "/synthetic/output"
    assert error.failure_phase == "fit"
    assert error.template_id == "alpha"
    assert error.sample_id == "train:000001"
    assert error.epoch_index == 1
    assert error.batch_index == 2
    assert error.global_step == 3
    assert error.source_kind == "scratch"
    assert error.underlying_reason_code == "KEYBOARD_INTERRUPT"


def test_scratch_nested_result_has_human_terminal_summary():
    module = importlib.import_module("refsite_mlip.cli.train")
    terminal = {
        "status": "completed",
        "source_kind": "scratch",
        "config_fingerprint": "a" * 64,
        "bundle_fingerprint": "b" * 64,
        "train_semantic_digest": "c" * 64,
        "validation_semantic_digest": "d" * 64,
        "seed": 17,
        "runtime": {
            "device": "cpu",
            "dtype": "float64",
            "solver_path": "train-fixed",
        },
        "completed_epochs": 2,
        "global_step": 4,
        "fit_result": {
            "stopped_early": False,
            "best_epoch": 1,
            "best_metric": 0.25,
        },
        "baseline": {"parameter_update_applied": True},
        "latest_checkpoint": "/run/checkpoints/latest.pt",
        "best_checkpoint": "/run/checkpoints/best.pt",
    }
    report = {
        "checkpointed_fit_result": {},
        "startup": {},
        "terminal_status": terminal,
    }

    rendered = module.render_train_result_human(report)
    assert "Status: completed" in rendered
    assert "Epochs completed: 2" in rendered
    assert "Global step: 4" in rendered


def test_training_terminal_presentation_preserves_primary_outcome(monkeypatch):
    module = importlib.import_module("refsite_mlip.cli.train")

    class Recorder:
        def __init__(self):
            self.calls = []

        def render_terminal(self, status, **values):
            self.calls.append((status, values))

    success = {
        "status": "completed",
        "completed_epochs": 1,
        "global_step": 3,
        "latest_checkpoint": "/display/checkpoints/latest.pt",
        "recoverable_checkpoint": "/display/checkpoints/latest.pt",
        "fit_result": {
            "stopped_early": False,
            "best_epoch": 0,
            "best_metric": 1.25,
            "stop_reason": None,
        },
    }
    renderer = Recorder()
    monkeypatch.setattr(module, "_run_training_impl", lambda *args, **kwargs: success)
    assert module.run_training("run.json", progress_renderer=renderer) is success
    assert renderer.calls == [
        (
            "completed",
            {
                "epochs": 1,
                "global_step": 3,
                "best_epoch": 0,
                "best_value": 1.25,
                "latest_checkpoint": "latest.pt",
                "reason": None,
                "recoverable": "latest.pt",
            },
        )
    ]

    failure = CLIError(
        "SYNTHETIC_FAILURE",
        "synthetic failure",
        stage="training.fit",
        failure_phase="fit",
        epoch_index=1,
        global_step=4,
    )
    failure.completed_epochs = 1
    failure.recoverable_checkpoint = "/display/checkpoints/latest.pt"
    renderer.calls.clear()

    def fail(*args, **kwargs):
        del args, kwargs
        raise failure

    monkeypatch.setattr(module, "_run_training_impl", fail)
    with pytest.raises(CLIError) as caught:
        module.run_training("run.json", progress_renderer=renderer)
    assert caught.value is failure
    assert renderer.calls == [
        (
            "failed",
            {
                "epochs": 2,
                "global_step": 4,
                "phase": "fit",
                "recoverable": "latest.pt",
            },
        )
    ]

    interrupt = CLIInterruptedError(
        "TRAINING_INTERRUPTED",
        "synthetic interrupt",
        stage="training.fit",
        global_step=5,
    )
    interrupt.completed_epochs = 2
    renderer.calls.clear()

    def interrupted(*args, **kwargs):
        del args, kwargs
        raise interrupt

    monkeypatch.setattr(module, "_run_training_impl", interrupted)
    with pytest.raises(CLIInterruptedError) as caught:
        module.run_training("run.json", progress_renderer=renderer)
    assert caught.value is interrupt
    assert renderer.calls[0][0] == "interrupted"
    assert renderer.calls[0][1]["epochs"] == 2
    assert renderer.calls[0][1]["global_step"] == 5
