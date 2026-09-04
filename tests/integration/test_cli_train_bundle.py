from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

pytest.importorskip("ase")

from refsite_mlip.cli.errors import CLIError, CLIInterruptedError
from refsite_mlip.cli.main import main
from refsite_mlip.cli.train import (
    render_train_result_human,
    render_train_result_json,
    run_training,
    seed_training_runtime,
)
from refsite_mlip.cli.training_progress import (
    TrainingProgressConfig,
    TrainingProgressRenderer,
)
from refsite_mlip.cli.validate_train_config import validate_train_config
from refsite_mlip.config import (
    TrainingRunConfigOverrides,
    load_training_run_config,
    resolve_training_run,
)
from refsite_mlip.training import (
    CheckpointManager,
    CheckpointManagerConfig,
    FitExecutionError,
    FitProgress,
    MetricsJournal,
    MetricsJournalError,
    ModelSelectionState,
    TrainingRunDirectory,
    build_optimizer,
    build_scheduler,
    capture_training_checkpoint,
    load_training_checkpoint,
    run_checkpointed_fit,
)

from test_validate_train_config_cli import (
    _atoms,
    _base_payload,
    _simple_case,
    _write_frames,
    training_bundle,
)


def _set_epochs(config_path: Path, epochs: int, *, output: str | None = None) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["fit"]["max_epochs"] = epochs
    if output is not None:
        payload["output_directory"] = output
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def _tree_equal(first, second) -> bool:
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            _tree_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return isinstance(second, (tuple, list)) and len(first) == len(second) and all(
            _tree_equal(left, right) for left, right in zip(first, second)
        )
    return first == second


def test_train_dry_run_exact_validate_parity_and_no_execution(
    training_bundle, tmp_path, monkeypatch, capsys
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    module = importlib.import_module("refsite_mlip.models.potential")

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dry-run must not execute model.forward")

    monkeypatch.setattr(module.ReferenceSitePotential, "forward", forbidden)
    assert main(["validate-train-config", str(config_path), "--json"]) == 0
    validated = capsys.readouterr()
    assert main(["train", str(config_path), "--dry-run", "--json"]) == 0
    trained = capsys.readouterr()
    assert trained.out == validated.out
    assert trained.err == (
        "refsite-mlip: loading training configuration\n"
        "refsite-mlip: preparing data and model bundle\n"
    )
    assert json.loads(trained.out)["runtime"]["seed"] == 17
    assert not (tmp_path / "run-output").exists()


def test_synthetic_cpu_float64_one_epoch_writes_recoverable_state(
    training_bundle, tmp_path, capsys
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 1)
    config_before = config_path.read_bytes()
    bundle_before = training_bundle["path"].read_bytes()
    train_path = tmp_path / "train.xyz"
    validation_path = tmp_path / "validation.xyz"
    inputs_before = (train_path.read_bytes(), validation_path.read_bytes())

    assert main(["train", str(config_path), "--json"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert "training started" in captured.err
    assert "Training completed" in captured.err
    assert report["status"] == "completed"
    assert report["training_executed"] is True
    assert report["seed"] == 17
    assert report["runtime"] == {
        "device": "cpu",
        "dtype": "float64",
        "solver_path": "train_fixed",
    }
    assert report["completed_epochs"] == 1
    assert report["global_step"] == 1
    output = tmp_path / "run-output"
    assert sorted(path.name for path in output.iterdir()) == [
        "checkpoints",
        "metrics.jsonl",
        "preflight.json",
        "resolved_config.json",
        "run_status.json",
    ]
    checkpoints = output / "checkpoints"
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "best.pt",
        "epoch_000000.pt",
        "latest.pt",
    ]
    assert json.loads((output / "run_status.json").read_text()) == report
    assert json.loads((output / "resolved_config.json").read_text())["runtime"][
        "seed"
    ] == 17
    checkpoint = load_training_checkpoint(checkpoints / "latest.pt")
    baseline = checkpoint.metadata.baseline_fit_metadata
    assert baseline["seed"] == 17
    assert baseline["parameter_update_applied"] is True
    assert checkpoint.progress.completed_epochs == 1
    assert checkpoint.progress.global_step == 1
    assert report["metrics_journal"] == "metrics.jsonl"
    assert report["metrics_event_count"] == 1
    assert report["metrics_last_epoch"] == 0
    assert len(report["metrics_semantic_sha256"]) == 64
    metric_lines = (output / "metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(metric_lines) == 1
    metric = json.loads(metric_lines[0])
    assert metric["event"] == "epoch_committed"
    assert metric["epoch_index"] == 0
    assert render_train_result_json(report) == render_train_result_json(
        dict(reversed(tuple(report.items())))
    )
    human = render_train_result_human(report)
    assert "Status: completed" in human
    assert "No portable prediction bundle was exported." in human
    assert config_path.read_bytes() == config_before
    assert training_bundle["path"].read_bytes() == bundle_before
    assert (train_path.read_bytes(), validation_path.read_bytes()) == inputs_before


def test_bundle_progress_summary_and_quiet_are_trajectory_neutral(
    training_bundle, tmp_path
):
    visible_root = tmp_path / "visible"
    quiet_root = tmp_path / "quiet"
    visible_root.mkdir()
    quiet_root.mkdir()
    config_path, _ = _simple_case(visible_root, training_bundle)
    _set_epochs(config_path, 1)
    visible_stream = io.StringIO()
    times = iter((10.0, 12.5))
    visible = TrainingProgressRenderer(
        stream=visible_stream, monotonic=lambda: next(times)
    )
    first = run_training(config_path, progress_renderer=visible)
    first_journal = (visible_root / "run-output" / "metrics.jsonl").read_bytes()
    first_checkpoint = load_training_checkpoint(first["latest_checkpoint"])
    first_draws = (random.random(), float(np.random.random()), torch.rand(()).item())

    quiet_path, _ = _simple_case(quiet_root, training_bundle)
    _set_epochs(quiet_path, 1)
    quiet_stream = io.StringIO()
    quiet = TrainingProgressRenderer(
        TrainingProgressConfig(enabled=False), stream=quiet_stream
    )
    second = run_training(quiet_path, progress_renderer=quiet)
    second_journal = (quiet_root / "run-output" / "metrics.jsonl").read_bytes()
    second_checkpoint = load_training_checkpoint(second["latest_checkpoint"])
    second_draws = (random.random(), float(np.random.random()), torch.rand(()).item())

    rendered = visible_stream.getvalue()
    assert rendered.count("refsite-mlip: loading training configuration") == 1
    assert rendered.count("refsite-mlip: preparing data and model bundle") == 1
    assert rendered.count("refsite-mlip: initializing training run") == 1
    assert rendered.count("refsite-mlip: training started") == 1
    assert "Reference-site MLIP training\n" in rendered
    assert "Source: bundle" in rendered
    assert "Device: cpu, dtype=float64" in rendered
    assert "Epoch 001/1" in rendered
    assert "checkpoint=epoch_000000.pt" in rendered
    assert "elapsed=2.5s eta=0.0s" in rendered
    assert rendered.rstrip().endswith("latest=latest.pt")
    assert quiet_stream.getvalue().startswith("Training completed |")
    assert "Epoch " not in quiet_stream.getvalue()
    assert first_journal == second_journal
    assert first_draws == second_draws
    assert first["config_fingerprint"] == second["config_fingerprint"]
    assert first["metrics_semantic_sha256"] == second["metrics_semantic_sha256"]
    assert first["fit_result"] == second["fit_result"]
    assert _tree_equal(first_checkpoint.to_dict(), second_checkpoint.to_dict())
    assert _tree_equal(
        first_checkpoint.model_state_dict, second_checkpoint.model_state_dict
    )
    assert _tree_equal(
        first_checkpoint.optimizer_state_dict,
        second_checkpoint.optimizer_state_dict,
    )
    assert _tree_equal(
        first_checkpoint.scheduler_state_dict,
        second_checkpoint.scheduler_state_dict,
    )
    assert first_checkpoint.selection_state == second_checkpoint.selection_state
    runtime_path_fields = {
        "latest_checkpoint",
        "best_checkpoint",
        "recoverable_checkpoint",
    }
    assert {
        key: value for key, value in first.items() if key not in runtime_path_fields
    } == {
        key: value for key, value in second.items() if key not in runtime_path_fields
    }


def test_v2_bundle_source_dry_run_uses_existing_preflight_contract(
    training_bundle, tmp_path
):
    config_path, payload = _simple_case(tmp_path, training_bundle)
    bundle_path = payload.pop("initial_bundle")
    payload["schema_version"] = "refsite_training_run_config_v2"
    payload["model_source"] = {"kind": "bundle", "path": bundle_path}
    payload["data"]["validation_batch_size"] = 1
    payload["fit"]["max_epochs"] = 1
    payload["output_directory"] = "v2-output"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    overrides = TrainingRunConfigOverrides(
        max_epochs=3,
        batch_size=1,
        validation_batch_size=1,
        learning_rate=2.0e-4,
    )
    report = run_training(config_path, dry_run=True, overrides=overrides)
    validated = validate_train_config(config_path, overrides=overrides)
    assert report.to_dict() == validated.to_dict()
    assert report.config_schema_version == "refsite_training_run_config_v2"
    assert report.training_configuration["validation_batch_size"] == 1
    output = tmp_path / "v2-output"
    assert not output.exists()


def test_fresh_training_holds_common_run_lock_for_entire_fit(
    training_bundle, tmp_path, monkeypatch
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 1)
    module = importlib.import_module("refsite_mlip.cli.train")
    original = module.run_checkpointed_fit
    observed = []

    def inspect_lock(*args, **kwargs):
        lock = tmp_path / "run-output" / ".resume.lock"
        status = json.loads(
            (tmp_path / "run-output" / "run_status.json").read_text(
                encoding="utf-8"
            )
        )
        observed.append((lock.is_file(), status["status"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_checkpointed_fit", inspect_lock)
    result = run_training(config_path)
    assert result["status"] == "completed"
    assert observed == [(True, "running")]
    assert not (tmp_path / "run-output" / ".resume.lock").exists()


def test_fresh_training_rechecks_config_after_lock_acquisition(
    training_bundle, tmp_path, monkeypatch
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 1)
    original = TrainingRunDirectory.acquire_resume_lock

    def mutate_then_acquire(directory):
        lock = original(directory)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["runtime"]["seed"] += 1
        config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return lock

    monkeypatch.setattr(
        TrainingRunDirectory, "acquire_resume_lock", mutate_then_acquire
    )
    with pytest.raises(Exception) as caught:
        run_training(config_path)
    assert getattr(caught.value, "reason_code", None) == (
        "TRAIN_CONFIG_TOCTOU_MISMATCH"
    )
    output = tmp_path / "run-output"
    assert not (output / ".resume.lock").exists()
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["training_executed"] is False


def test_cli_path_matches_direct_checkpointed_fit_payload_and_result(
    training_bundle, tmp_path
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 1)
    train_module = importlib.import_module("refsite_mlip.cli.train")
    config = load_training_run_config(config_path)
    resolved = resolve_training_run(config)

    seed_training_runtime(config.runtime.seed)
    prepared = train_module._prepare_training_runtime(config, resolved)
    baseline = train_module._baseline_metadata(config, resolved, prepared)
    optimizer = build_optimizer(prepared.loaded.model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler)
    selection = ModelSelectionState()
    metadata = capture_training_checkpoint(
        prepared.loaded.model,
        optimizer,
        scheduler,
        selection,
        FitProgress(next_epoch=0, global_step=0, completed_epochs=0),
        prepared.train_batches,
        prepared.validation_batches,
        model_config=prepared.loaded.model.config,
        loss_config=config.loss,
        optimizer_config=config.optimizer,
        train_step_config=config.train_step,
        validation_step_config=config.validation_step,
        scheduler_config=config.scheduler,
        model_selection_config=config.selection,
        fit_config=config.fit,
        species_vocabulary=resolved.species_vocabulary,
        fit_history=(),
        baseline_fit_metadata=baseline,
    ).metadata
    direct_manager = CheckpointManager(
        CheckpointManagerConfig(tmp_path / "direct-checkpoints")
    )
    direct = run_checkpointed_fit(
        prepared.loaded.model,
        optimizer,
        scheduler,
        prepared.train_batches,
        prepared.validation_batches,
        prepared.loaded.template_contexts,
        config.loss,
        config.train_step,
        config.validation_step,
        config.scheduler,
        config.selection,
        selection,
        config.fit,
        direct_manager,
        metadata,
        config.checkpointed_fit,
    )

    cli_report = run_training(config_path)
    direct_checkpoint = direct_manager.load_latest()
    cli_checkpoint = load_training_checkpoint(cli_report["latest_checkpoint"])
    assert _tree_equal(direct_checkpoint.to_dict(), cli_checkpoint.to_dict())
    assert json.loads(json.dumps(direct.fit_result.to_dict())) == cli_report[
        "fit_result"
    ]


def test_mixed_template_two_epoch_batch_plan_and_seed_repetition(
    training_bundle, tmp_path
):
    samples = training_bundle["samples"]
    train = tmp_path / "mixed-train.xyz"
    validation = tmp_path / "mixed-validation.xyz"
    _write_frames(
        train,
        (
            _atoms(samples[0], energy=5.0, template_key="template"),
            _atoms(samples[1], energy=4.0, template_key="template"),
            _atoms(samples[2], energy=6.0, template_key="template"),
        ),
    )
    _write_frames(
        validation,
        (
            _atoms(samples[1], energy=4.25, template_key="template"),
            _atoms(samples[2], energy=6.25, template_key="template"),
        ),
    )

    def write_config(name: str, output: str) -> Path:
        payload = _base_payload(
            bundle=str(training_bundle["path"]),
            train_sources=[{"path": train.name, "template_key": "template"}],
            validation_sources=[
                {"path": validation.name, "template_key": "template"}
            ],
            output=output,
        )
        payload["data"]["batch_size"] = 2
        payload["fit"]["max_epochs"] = 2
        path = tmp_path / name
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    first = run_training(write_config("first.json", "first-output"))
    second = run_training(write_config("second.json", "second-output"))
    assert first["completed_epochs"] == second["completed_epochs"] == 2
    assert first["global_step"] == second["global_step"] == 4
    first_checkpoint = load_training_checkpoint(first["latest_checkpoint"])
    second_checkpoint = load_training_checkpoint(second["latest_checkpoint"])
    assert first_checkpoint.progress.to_dict() == second_checkpoint.progress.to_dict()
    for key in first_checkpoint.model_state_dict:
        assert torch.equal(
            first_checkpoint.model_state_dict[key],
            second_checkpoint.model_state_dict[key],
        )
    assert _tree_equal(
        first_checkpoint.optimizer_state_dict,
        second_checkpoint.optimizer_state_dict,
    )
    assert _tree_equal(
        first_checkpoint.scheduler_state_dict,
        second_checkpoint.scheduler_state_dict,
    )
    assert first_checkpoint.fit_history == second_checkpoint.fit_history
    assert first_checkpoint.python_rng_state == second_checkpoint.python_rng_state
    assert _tree_equal(
        first_checkpoint.numpy_rng_state, second_checkpoint.numpy_rng_state
    )
    assert torch.equal(
        first_checkpoint.torch_cpu_rng_state,
        second_checkpoint.torch_cpu_rng_state,
    )


def test_force_and_stress_supervision_runs_without_energy_baseline_refit(
    training_bundle, tmp_path
):
    samples = training_bundle["samples"]
    train = tmp_path / "derivative-train.xyz"
    validation = tmp_path / "derivative-validation.xyz"
    _write_frames(
        train,
        (
            _atoms(
                samples[0],
                energy=None,
                forces=True,
                stress=True,
                partial_masks=True,
            ),
            _atoms(samples[2], energy=None, forces=True, stress=True),
        ),
    )
    _write_frames(
        validation,
        (_atoms(samples[2], energy=None, forces=True, stress=True),),
    )
    payload = _base_payload(
        bundle=str(training_bundle["path"]),
        train_sources=[{"path": train.name, "template_id": "zeta"}],
        validation_sources=[
            {"path": validation.name, "template_id": "zeta"}
        ],
        output="derivative-output",
    )
    payload["loss"].update(
        energy_weight=0.0,
        force_weight=1.0,
        stress_weight=1.0,
    )
    payload["baseline"] = None
    payload["fit"]["max_epochs"] = 1
    config_path = tmp_path / "derivative.json"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    report = run_training(config_path)
    assert report["status"] == "completed"
    assert report["baseline"] == {
        "enabled": False,
        "initial_bundle_fingerprint": report["bundle_fingerprint"],
        "parameter_update_applied": False,
        "reason": "baseline config is null",
        "seed": 17,
        "training_run_config_fingerprint": report["config_fingerprint"],
    }
    history = report["fit_result"]["records"]
    assert history[0]["training"]["force"]["valid_count"] > 0
    assert history[0]["training"]["stress"]["valid_count"] > 0


def test_toctou_failure_records_status_without_checkpoint(
    training_bundle, tmp_path, monkeypatch
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 1)
    module = importlib.import_module("refsite_mlip.cli.train")
    monkeypatch.setattr(module, "_split_digest", lambda *args, **kwargs: "0" * 64)
    with pytest.raises(Exception) as caught:
        run_training(config_path)
    assert getattr(caught.value, "reason_code", None) == "TRAIN_DATA_TOCTOU_MISMATCH"
    output = tmp_path / "run-output"
    status = json.loads((output / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["failure_phase"] == "toctou.data"
    assert status["training_executed"] is False
    assert status["rollback_performed"] is False
    assert not (output / "checkpoints").exists()
    assert not list(output.glob(".*.tmp"))


def test_fit_failure_and_interrupt_status_are_recoverable(
    training_bundle, tmp_path, monkeypatch
):
    module = importlib.import_module("refsite_mlip.cli.train")
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 1)

    def fail(*args, **kwargs):
        del args, kwargs
        raise FitExecutionError(
            phase="train",
            epoch_index=0,
            current_global_step=0,
            completed_epochs=0,
            training_update_completed=False,
            cause=RuntimeError("injected first epoch failure"),
        )

    monkeypatch.setattr(module, "run_checkpointed_fit", fail)
    with pytest.raises(Exception) as caught:
        run_training(config_path)
    assert getattr(caught.value, "failure_phase", None) == "train"
    failed = json.loads((tmp_path / "run-output" / "run_status.json").read_text())
    assert failed["status"] == "failed"
    assert failed["completed_epochs"] == 0
    assert failed["global_step"] == 0
    assert failed["rollback_performed"] is False

    interrupt_directory = tmp_path / "interrupt"
    interrupt_directory.mkdir()
    interrupt_path, _ = _simple_case(interrupt_directory, training_bundle)
    _set_epochs(interrupt_path, 1)
    monkeypatch.setattr(
        module,
        "run_checkpointed_fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(CLIInterruptedError):
        run_training(interrupt_path)
    interrupted = json.loads(
        (tmp_path / "interrupt" / "run-output" / "run_status.json").read_text()
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["recoverable_checkpoint"] is None
    assert interrupted["rollback_performed"] is False


def test_interrupt_after_completed_epoch_records_recoverable_latest(
    training_bundle, tmp_path, monkeypatch
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 2)
    original_save = CheckpointManager.save_epoch

    def save_then_interrupt(self, checkpoint, record):
        managed = original_save(self, checkpoint, record)
        if record.epoch_index == 0:
            raise KeyboardInterrupt()
        return managed

    monkeypatch.setattr(CheckpointManager, "save_epoch", save_then_interrupt)
    with pytest.raises(CLIInterruptedError):
        run_training(config_path)
    status = json.loads(
        (tmp_path / "run-output" / "run_status.json").read_text()
    )
    assert status["status"] == "interrupted"
    assert status["completed_epochs"] == 1
    assert status["global_step"] == 1
    assert status["recoverable_checkpoint"].endswith("checkpoints/latest.pt")
    assert status["best_checkpoint"].endswith("checkpoints/best.pt")
    checkpoint = load_training_checkpoint(status["recoverable_checkpoint"])
    assert checkpoint.progress.completed_epochs == 1
    assert checkpoint.progress.next_epoch == 1
    assert status["terminal_selection_state"] == (
        checkpoint.selection_state.to_dict()
    )


def test_middle_epoch_checkpoint_failure_preserves_previous_epoch(
    training_bundle, tmp_path, monkeypatch
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 2)
    original_save = CheckpointManager.save_epoch

    def fail_second_epoch(self, checkpoint, record):
        if record.epoch_index == 1:
            raise OSError("injected second-epoch checkpoint failure")
        return original_save(self, checkpoint, record)

    monkeypatch.setattr(CheckpointManager, "save_epoch", fail_second_epoch)
    with pytest.raises(Exception) as caught:
        run_training(config_path)
    assert getattr(caught.value, "failure_phase", None) == "checkpoint.manager"
    status = json.loads(
        (tmp_path / "run-output" / "run_status.json").read_text()
    )
    assert status["status"] == "failed"
    assert status["completed_epochs"] == 1
    assert status["global_step"] == 2
    assert status["rollback_performed"] is False
    assert status["recoverable_checkpoint"].endswith("checkpoints/latest.pt")
    latest = load_training_checkpoint(status["recoverable_checkpoint"])
    assert latest.progress.completed_epochs == 1
    assert latest.progress.global_step == 1
    assert sorted(
        path.name for path in (tmp_path / "run-output" / "checkpoints").iterdir()
    ) == ["best.pt", "epoch_000000.pt", "latest.pt"]


def test_metrics_journal_failure_preserves_committed_checkpoint_and_status(
    training_bundle, tmp_path, monkeypatch
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 2)

    def fail_observer(self, event):
        raise MetricsJournalError(
            "INJECTED_METRICS_JOURNAL_FAILURE",
            "injected metrics journal failure",
            stage="metrics_journal.commit",
            path=self.path,
            epoch_index=event.epoch_index,
            original_error=OSError("injected journal write failure"),
        )

    monkeypatch.setattr(MetricsJournal, "__call__", fail_observer)
    with pytest.raises(CLIError) as caught:
        run_training(config_path)
    assert getattr(caught.value, "failure_phase", None) == "metrics_journal"

    output = tmp_path / "run-output"
    latest_path = output / "checkpoints" / "latest.pt"
    checkpoint = load_training_checkpoint(latest_path)
    assert checkpoint.progress.completed_epochs == 1
    assert not (output / "checkpoints" / "epoch_000001.pt").exists()
    assert not (output / "metrics.jsonl").exists()
    status = json.loads((output / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["failure_phase"] == "metrics_journal"
    assert status["completed_epochs"] == 1
    assert status["recoverable_checkpoint"] == str(latest_path)
    assert status["metrics_event_count"] == 0
    assert status["metrics_last_epoch"] is None
    assert status["error"] == {
        "message": "injected journal write failure",
        "reason_code": "INJECTED_METRICS_JOURNAL_FAILURE",
        "type": "OSError",
    }


def test_metrics_journal_corrupt_suffix_status_reports_last_valid_prefix(
    training_bundle, tmp_path, monkeypatch
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    _set_epochs(config_path, 2)
    real_observer = MetricsJournal.__call__

    def corrupt_second_event(self, event):
        if event.epoch_index == 0:
            return real_observer(self, event)
        valid_prefix = self.path.read_bytes()
        self.path.write_bytes(valid_prefix + b'{"truncated":')
        raise MetricsJournalError(
            "INJECTED_METRICS_JOURNAL_FAILURE",
            "injected corrupt suffix",
            stage="metrics_journal.commit",
            path=self.path,
            epoch_index=event.epoch_index,
            original_error=OSError("injected corrupt suffix"),
        )

    monkeypatch.setattr(MetricsJournal, "__call__", corrupt_second_event)
    with pytest.raises(CLIError):
        run_training(config_path)

    output = tmp_path / "run-output"
    latest = load_training_checkpoint(output / "checkpoints" / "latest.pt")
    assert latest.progress.completed_epochs == 2
    journal_bytes = (output / "metrics.jsonl").read_bytes()
    valid_prefix = journal_bytes.splitlines(keepends=True)[0]
    status = json.loads((output / "run_status.json").read_text())
    assert status["failure_phase"] == "metrics_journal"
    assert status["completed_epochs"] == 2
    assert status["metrics_event_count"] == 1
    assert status["metrics_last_epoch"] == 0
    assert status["metrics_semantic_sha256"] == hashlib.sha256(
        valid_prefix
    ).hexdigest()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="9D CUDA gate: unavailable")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_one_epoch_smoke(training_bundle, tmp_path, dtype):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    payload = _set_epochs(config_path, 1)
    payload["runtime"].update(device="cuda", dtype=dtype)
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    report = run_training(config_path)
    assert report["status"] == "completed"
    assert report["runtime"]["device"].startswith("cuda")
    assert report["runtime"]["dtype"] == dtype
    checkpoint = load_training_checkpoint(report["latest_checkpoint"])
    assert checkpoint.cuda_device_count == torch.cuda.device_count()
