from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("ase")

from refsite_mlip.cli.errors import CLIError
from refsite_mlip.cli.export_bundle import ExportBundleConfig, export_bundle
from refsite_mlip.cli.main import main
from refsite_mlip.cli.resume import resume_training
from refsite_mlip.cli.train import run_training
from refsite_mlip.cli.training_progress import TrainingProgressRenderer
from refsite_mlip.config import load_training_run_config
from refsite_mlip.models import load_reference_site_model_bundle
from refsite_mlip.training import (
    CheckpointManager,
    MetricsJournal,
    MetricsJournalError,
    ResumeRunLock,
    RunDirectoryError,
    ScratchCheckpointedTrainingError,
    TrainingRunDirectory,
    load_training_checkpoint,
    prepare_scratch_training_run,
    run_scratch_checkpointed_training,
)

from test_scratch_training_preparation import (
    _atoms,
    _case,
    _partially_labeled,
    _vacancy,
)


def _prepared(
    directory: Path,
    *,
    max_epochs: int = 1,
    baseline: bool = True,
    mixed_loss: bool = False,
    scheduler: dict | None = None,
    selection: dict | None = None,
):
    reference = _atoms(1)
    config_path, _, _, payload = _case(
        directory,
        train_frames=(
            _partially_labeled(reference, -8.0),
            _partially_labeled(_vacancy(reference), -6.5),
        ),
        validation_frames=(_partially_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
        baseline=baseline,
    )
    payload["fit"]["max_epochs"] = max_epochs
    if mixed_loss:
        payload["loss"].update(
            {
                "energy_weight": 1.0,
                "force_weight": 0.05,
                "stress_weight": 0.02,
            }
        )
    if scheduler is not None:
        payload["scheduler"] = scheduler
    if selection is not None:
        payload["selection"] = selection
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    config = load_training_run_config(config_path)
    return config, prepare_scratch_training_run(config)


def _tree_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _next_rng_draws():
    return random.random(), float(np.random.random()), torch.rand(4)


def test_one_epoch_mixed_loss_writes_safe_checkpoints_and_holds_common_lock(
    tmp_path,
):
    config, preparation = _prepared(
        tmp_path, max_epochs=1, baseline=True, mixed_loss=True
    )
    output = Path(preparation.runtime_paths["output_directory"])
    events: list[str] = []

    def observe(event: str) -> None:
        events.append(event)
        lock_path = output / ".resume.lock"
        assert lock_path.is_file() and not lock_path.is_symlink()
        observed_status = json.loads(
            output.joinpath("run_status.json").read_text(encoding="utf-8")
        )["status"]
        assert observed_status == {
            "lock_acquired": "initializing",
            "startup_ready": "startup_ready",
            "before_fit": "running",
            "after_fit": "running",
            "terminal_status_written": "completed",
        }[event]
        if event == "before_fit":
            directory = TrainingRunDirectory.open_existing(output)
            with pytest.raises(Exception) as second:
                directory.acquire_resume_lock()
            assert getattr(second.value, "reason_code", None) == "RESUME_LOCK_EXISTS"
            with pytest.raises(CLIError) as active_export:
                export_bundle(
                    ExportBundleConfig(
                        run_directory=output,
                        source="latest",
                        output_path=tmp_path / "forbidden.pt",
                        dry_run=True,
                    )
                )
            assert active_export.value.reason_code == "RESUME_LOCK_EXISTS"
            with pytest.raises(CLIError) as active_resume:
                resume_training(output, max_epochs=2, dry_run=True)
            assert active_resume.value.reason_code == "RESUME_LOCK_EXISTS"

    result = run_scratch_checkpointed_training(
        config, preparation, event_callback=observe
    )
    assert events == [
        "lock_acquired",
        "startup_ready",
        "before_fit",
        "after_fit",
        "terminal_status_written",
    ]
    assert result.status == "completed"
    assert result.completed_epochs == 1
    assert result.global_step == 1
    assert result.final_progress.completed_epochs == 1
    assert result.final_selection == result.fit_result.final_selection_state
    assert result.initial_bundle_fingerprint == (
        result.startup.initial_bundle_fingerprint
    )
    assert result.train_semantic_digest == preparation.train_semantic_digest
    assert result.validation_semantic_digest == (
        preparation.validation_semantic_digest
    )
    assert result.baseline_fit_metadata == result.startup.baseline_metadata
    assert result.recoverability["kind"] == "latest_checkpoint"
    assert not (output / ".resume.lock").exists()

    checkpoints = output / "checkpoints"
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "best.pt",
        "epoch_000000.pt",
        "latest.pt",
    ]
    initial = load_reference_site_model_bundle(output / "initial_bundle.pt")
    latest = load_training_checkpoint(checkpoints / "latest.pt")
    assert torch.equal(
        initial.model_state["atomic_baseline"],
        torch.zeros_like(initial.model_state["atomic_baseline"]),
    )
    assert torch.equal(
        latest.model_state_dict["atomic_baseline"],
        result.startup.model.atomic_baseline.detach().cpu(),
    )
    assert bool(torch.any(latest.model_state_dict["atomic_baseline"] != 0.0))
    parameter_names = tuple(dict(result.startup.model.named_parameters()))
    assert any(
        not torch.equal(initial.model_state[name], latest.model_state_dict[name])
        for name in parameter_names
    )
    assert latest.progress.completed_epochs == 1
    assert latest.progress.global_step == 1
    assert latest.metadata.baseline_fit_metadata == result.to_dict()["startup"][
        "baseline"
    ]
    assert result.fit_result.records[0].training.force.valid_count > 0
    assert result.fit_result.records[0].training.stress.valid_count > 0

    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    assert status == result.to_dict()["terminal_status"]
    assert status["status"] == "completed"
    assert status["first_optimizer_update_executed"] is True
    assert status["recoverable_checkpoint"] == str(checkpoints / "latest.pt")
    assert status["rollback_performed"] is False
    journal_bytes = (output / "metrics.jsonl").read_bytes()
    journal_lines = journal_bytes.splitlines()
    assert len(journal_lines) == 1
    committed_event = json.loads(journal_lines[0])
    assert committed_event["schema_version"] == "refsite_training_metrics_v1"
    assert committed_event["event"] == "epoch_committed"
    assert committed_event["epoch_index"] == 0
    assert status["metrics_journal"] == "metrics.jsonl"
    assert status["metrics_event_count"] == 1
    assert status["metrics_last_epoch"] == 0
    assert status["metrics_semantic_sha256"] == hashlib.sha256(
        journal_bytes
    ).hexdigest()

    # The generated initial bundle and bundle-compatible metadata are already
    # sufficient for resume/export; neither path rebuilds the scratch POSCAR.
    source_poscar = Path(
        preparation.runtime_paths["reference_poscars"][0]["path"]
    )
    source_poscar.unlink()
    assert not source_poscar.exists()
    resume_report = resume_training(output, max_epochs=2, dry_run=True)
    assert resume_report["status"] == "resume_preflight_ready"
    export_report = export_bundle(
        ExportBundleConfig(
            run_directory=output,
            source="latest",
            output_path=tmp_path / "unused.pt",
            dry_run=True,
        )
    )
    assert export_report["dry_run"] is True
    exported_path = tmp_path / "exported-latest.pt"
    saved_export = export_bundle(
        ExportBundleConfig(
            run_directory=output,
            source="latest",
            output_path=exported_path,
        )
    )
    exported = load_reference_site_model_bundle(exported_path)
    assert saved_export["dry_run"] is False
    assert exported.bundle_fingerprint == saved_export["bundle_sha256"]
    assert _tree_equal(exported.model_state, latest.model_state_dict)

    status_path = output / "run_status.json"
    original_status = status_path.read_text(encoding="utf-8")
    missing_initial_recovery = json.loads(original_status)
    missing_initial_recovery["recoverable_initial_bundle"] = None
    status_path.write_text(
        json.dumps(missing_initial_recovery, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(CLIError) as missing_recovery:
        export_bundle(
            ExportBundleConfig(
                run_directory=output,
                source="latest",
                output_path=tmp_path / "missing-recovery.pt",
                dry_run=True,
            )
        )
    assert missing_recovery.value.reason_code == (
        "RUN_STATUS_CHECKPOINT_PATH_MISMATCH"
    )
    status_path.write_text(original_status, encoding="utf-8")

    # Scratch-only recovery metadata is integrity checked rather than silently
    # ignored by resume.  Restore the first injected file before checking the
    # independent status invariant.
    manifest_path = output / "data_manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original_manifest)
    manifest["train_semantic_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(CLIError) as bad_manifest:
        resume_training(output, max_epochs=2, dry_run=True)
    assert bad_manifest.value.reason_code == "DATA_MANIFEST_FINGERPRINT_MISMATCH"
    with pytest.raises(CLIError) as bad_manifest_export:
        export_bundle(
            ExportBundleConfig(
                run_directory=output,
                source="latest",
                output_path=tmp_path / "tampered.pt",
                dry_run=True,
            )
        )
    assert bad_manifest_export.value.reason_code == (
        "DATA_MANIFEST_FINGERPRINT_MISMATCH"
    )
    manifest_path.write_text(original_manifest, encoding="utf-8")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["first_optimizer_update_executed"] = False
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    with pytest.raises(CLIError) as bad_status:
        resume_training(output, max_epochs=2, dry_run=True)
    assert bad_status.value.reason_code == "RUN_STATUS_IDENTITY_MISMATCH"


def test_scratch_continuous_two_epochs_equals_one_plus_exact_resume(tmp_path):
    continuous_config, continuous_preparation = _prepared(
        tmp_path / "continuous", max_epochs=2, baseline=True
    )
    split_config, split_preparation = _prepared(
        tmp_path / "split", max_epochs=1, baseline=True
    )

    continuous = run_scratch_checkpointed_training(
        continuous_config, continuous_preparation
    )
    continuous_checkpoint = load_training_checkpoint(continuous.latest_path)
    continuous_draws = _next_rng_draws()

    first = run_scratch_checkpointed_training(split_config, split_preparation)
    first_epoch = Path(first.latest_path).with_name("epoch_000000.pt")
    first_epoch_bytes = first_epoch.read_bytes()
    resume_stream = io.StringIO()
    resume_clock = iter((100.0, 102.0))
    resumed = resume_training(
        split_preparation.runtime_paths["output_directory"],
        max_epochs=2,
        progress_renderer=TrainingProgressRenderer(
            stream=resume_stream,
            monotonic=lambda: next(resume_clock),
        ),
    )
    resumed_checkpoint = load_training_checkpoint(resumed["latest_checkpoint"])
    resumed_draws = _next_rng_draws()

    assert resumed["source_kind"] == "scratch"
    resume_output = resume_stream.getvalue()
    assert "Source: scratch (resumed)" in resume_output
    assert "Epoch 002/2" in resume_output
    assert "Epoch 001/2" not in resume_output
    assert resumed["initial_bundle_fingerprint"] == (
        first.startup.initial_bundle_fingerprint
    )
    assert resumed["initialization_seed"] == first.startup.initialization_seed
    assert resumed["preparation_fingerprint"] == (
        split_preparation.preparation_fingerprint
    )
    assert resumed["data_manifest_fingerprint"] == (
        split_preparation.data_manifest["fingerprint"]
    )
    assert resumed["recovery"] == {
        "kind": "latest_checkpoint",
        "path": resumed["latest_checkpoint"],
    }

    assert _tree_equal(
        continuous_checkpoint.model_state_dict,
        resumed_checkpoint.model_state_dict,
    )
    assert _tree_equal(
        continuous_checkpoint.optimizer_state_dict,
        resumed_checkpoint.optimizer_state_dict,
    )
    assert _tree_equal(
        continuous_checkpoint.scheduler_state_dict,
        resumed_checkpoint.scheduler_state_dict,
    )
    assert continuous_checkpoint.selection_state == resumed_checkpoint.selection_state
    assert continuous_checkpoint.progress == resumed_checkpoint.progress
    assert continuous_checkpoint.fit_history == resumed_checkpoint.fit_history
    assert _tree_equal(
        continuous.to_dict()["terminal_status"]["fit_result"],
        resumed["fit_result"],
    )
    assert continuous_draws[0] == resumed_draws[0]
    assert continuous_draws[1] == resumed_draws[1]
    assert torch.equal(continuous_draws[2], resumed_draws[2])
    assert first_epoch.read_bytes() == first_epoch_bytes
    continuous_journal = Path(continuous.run_directory) / "metrics.jsonl"
    resumed_journal = Path(first.run_directory) / "metrics.jsonl"
    assert continuous_journal.read_bytes() == resumed_journal.read_bytes()
    assert sorted(
        path.name
        for path in Path(first.run_directory).joinpath("checkpoints").glob(
            "epoch_*.pt"
        )
    ) == ["epoch_000000.pt", "epoch_000001.pt"]
    assert not Path(first.run_directory).joinpath(".resume.lock").exists()


def test_failure_and_interrupt_retain_recovery_and_release_lock(
    tmp_path, monkeypatch
):
    import refsite_mlip.training.scratch_checkpointed_training as module

    for name, failure in (
        ("failure", RuntimeError("injected training failure")),
        ("interrupt", KeyboardInterrupt()),
    ):
        config, preparation = _prepared(
            tmp_path / name, max_epochs=1, baseline=False
        )
        output = Path(preparation.runtime_paths["output_directory"])

        def fail(*args, _failure=failure, **kwargs):
            del args, kwargs
            raise _failure

        monkeypatch.setattr(module, "run_checkpointed_fit", fail)
        if isinstance(failure, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                run_scratch_checkpointed_training(config, preparation)
            expected_status = "interrupted"
        else:
            with pytest.raises(ScratchCheckpointedTrainingError) as caught:
                run_scratch_checkpointed_training(config, preparation)
            assert caught.value.rollback_performed is False
            assert caught.value.recoverable_initial_bundle == str(
                output / "initial_bundle.pt"
            )
            expected_status = "failed"
        status = json.loads(
            (output / "run_status.json").read_text(encoding="utf-8")
        )
        assert status["status"] == expected_status
        assert status["completed_epochs"] == 0
        assert status["recoverable_checkpoint"] is None
        assert status["recoverable_initial_bundle"] == str(
            output / "initial_bundle.pt"
        )
        assert status["rollback_performed"] is False
        assert not output.joinpath(".resume.lock").exists()
        assert not list(output.joinpath("checkpoints").iterdir())


@pytest.mark.parametrize("failure", [RuntimeError("late failure"), KeyboardInterrupt()])
def test_failure_or_interrupt_after_completed_epoch_preserves_latest(
    tmp_path, monkeypatch, failure
):
    checkpointed = importlib.import_module(
        "refsite_mlip.training.checkpointed_fit"
    )
    config, preparation = _prepared(
        tmp_path / type(failure).__name__, max_epochs=2, baseline=False
    )
    output = Path(preparation.runtime_paths["output_directory"])
    original_epoch = checkpointed.run_training_epoch
    calls = 0

    def fail_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(checkpointed, "run_training_epoch", fail_second_epoch)
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt) as caught:
            run_scratch_checkpointed_training(config, preparation)
        structured = caught.value.scratch_training_error
        assert structured.interrupted is True
    else:
        with pytest.raises(ScratchCheckpointedTrainingError) as caught:
            run_scratch_checkpointed_training(config, preparation)
        structured = caught.value
    latest = output / "checkpoints" / "latest.pt"
    assert latest.is_file()
    checkpoint = load_training_checkpoint(latest)
    assert checkpoint.progress.completed_epochs == 1
    assert checkpoint.progress.global_step == 1
    assert structured.completed_epochs == 1
    assert structured.global_step == 1
    assert structured.recoverable_checkpoint == str(latest)
    status = json.loads(
        output.joinpath("run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == (
        "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
    )
    assert status["completed_epochs"] == 1
    assert status["global_step"] == 1
    assert status["recoverable_checkpoint"] == str(latest)
    assert not output.joinpath(".resume.lock").exists()


def test_body_and_lock_release_double_failure_preserves_primary_error(
    tmp_path, monkeypatch
):
    import refsite_mlip.training.scratch_checkpointed_training as module

    config, preparation = _prepared(
        tmp_path, max_epochs=1, baseline=False
    )
    output = Path(preparation.runtime_paths["output_directory"])
    primary_failure = RuntimeError("primary training failure")

    def fail_fit(*args, **kwargs):
        del args, kwargs
        raise primary_failure

    original_release = ResumeRunLock.release

    def release_then_report_failure(self):
        original_release(self)
        raise RunDirectoryError(
            "INJECTED_LOCK_RELEASE_FAILURE",
            "injected failure after owned lock cleanup",
            stage="run_directory.resume_lock.release",
            path=self.path,
        )

    monkeypatch.setattr(module, "run_checkpointed_fit", fail_fit)
    monkeypatch.setattr(ResumeRunLock, "release", release_then_report_failure)
    with pytest.raises(ScratchCheckpointedTrainingError) as caught:
        run_scratch_checkpointed_training(config, preparation)
    error = caught.value
    assert error.original_error is primary_failure
    assert error.lock_release_exception_type == "RunDirectoryError"
    assert "INJECTED_LOCK_RELEASE_FAILURE" in error.lock_release_exception_message
    assert isinstance(error.__cause__, RunDirectoryError)
    assert "primary training failure" in str(error)
    assert not output.joinpath(".resume.lock").exists()


def test_replaced_common_lock_is_detected_before_running_status_or_fit(tmp_path):
    config, preparation = _prepared(
        tmp_path, max_epochs=1, baseline=False
    )
    output = Path(preparation.runtime_paths["output_directory"])
    foreign_bytes = b"foreign lock must be preserved\n"

    def replace_lock(event: str) -> None:
        if event != "startup_ready":
            return
        lock_path = output / ".resume.lock"
        lock_path.unlink()
        lock_path.write_bytes(foreign_bytes)

    with pytest.raises(ScratchCheckpointedTrainingError) as caught:
        run_scratch_checkpointed_training(
            config,
            preparation,
            event_callback=replace_lock,
        )

    error = caught.value
    assert error.reason_code == "RESUME_LOCK_OWNERSHIP_LOST"
    assert error.stage == "event.startup_ready"
    assert error.global_step == 0
    assert error.completed_epochs == 0
    assert error.lock_release_exception_type == "RunDirectoryError"
    assert error.status_write_exception_type == "RunDirectoryError"
    lock_path = output / ".resume.lock"
    assert lock_path.read_bytes() == foreign_bytes
    status = json.loads(
        output.joinpath("run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "startup_ready"
    assert status["training_executed"] is False
    assert not any(output.joinpath("checkpoints").iterdir())


def test_reduce_on_plateau_early_stop_checkpoints_terminal_epoch(tmp_path):
    scheduler = {
        "kind": "reduce_on_plateau",
        "monitor": "total_loss",
        "mode": "min",
        "factor": 0.5,
        "patience": 0,
        "threshold": 1.0e6,
        "threshold_mode": "abs",
        "cooldown": 0,
        "min_lr": 0.0,
        "eps": 1.0e-8,
    }
    selection = {
        "monitor": "total_loss",
        "mode": "min",
        "min_delta": 1.0e6,
        "early_stopping_patience": 0,
    }
    config, preparation = _prepared(
        tmp_path,
        max_epochs=3,
        baseline=False,
        scheduler=scheduler,
        selection=selection,
    )
    result = run_scratch_checkpointed_training(config, preparation)
    assert result.status == "early_stopped"
    assert result.stopped_early is True
    assert result.completed_epochs == 2
    assert result.fit_result.stop_epoch == 1
    assert result.fit_result.best_epoch == 0
    assert result.terminal_model_is_best is False
    assert result.fit_result.final_learning_rates == (0.0005,)
    root = Path(result.run_directory) / "checkpoints"
    assert sorted(path.name for path in root.glob("epoch_*.pt")) == [
        "epoch_000000.pt",
        "epoch_000001.pt",
    ]
    latest = load_training_checkpoint(root / "latest.pt")
    best = load_training_checkpoint(root / "best.pt")
    assert latest.progress.last_completed_epoch == 1
    assert latest.progress.stopped_early is True
    assert best.progress.last_completed_epoch == 0
    status = json.loads(
        Path(result.run_directory, "run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "early_stopped"
    journal = Path(result.run_directory, "metrics.jsonl")
    journal_events = tuple(
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
    )
    assert tuple(event["epoch_index"] for event in journal_events) == (0, 1)
    assert journal_events[0]["is_best"] is True
    assert journal_events[0]["should_stop"] is False
    assert journal_events[1]["is_best"] is False
    assert journal_events[1]["should_stop"] is True
    assert journal_events[1]["best_epoch"] == 0
    assert journal_events[1]["best_checkpoint_basename"] is None
    assert status["metrics_event_count"] == 2
    assert status["metrics_last_epoch"] == 1
    assert status["metrics_semantic_sha256"] == hashlib.sha256(
        journal.read_bytes()
    ).hexdigest()
    exported = export_bundle(
        ExportBundleConfig(
            run_directory=result.run_directory,
            source="best",
            output_path=tmp_path / "early-best.pt",
            dry_run=True,
        )
    )
    assert exported["source"]["kind"] == "best"
    assert exported["source"]["epoch"] == 0


def test_metrics_journal_failure_preserves_committed_checkpoint_and_status(
    tmp_path, monkeypatch
):
    config, preparation = _prepared(
        tmp_path, max_epochs=2, baseline=False
    )
    output = Path(preparation.runtime_paths["output_directory"])
    observed_epochs: list[int] = []
    rendered_epochs: list[int] = []

    def fail_after_checkpoint(self, event):
        observed_epochs.append(event.epoch_index)
        assert output.joinpath("checkpoints", "epoch_000000.pt").is_file()
        assert output.joinpath("checkpoints", "latest.pt").is_file()
        raise MetricsJournalError(
            "INJECTED_METRICS_JOURNAL_FAILURE",
            "injected journal failure after checkpoint commit",
            stage="metrics_journal.commit",
            path=self.path,
            epoch_index=event.epoch_index,
            last_valid_epoch=None,
            original_error=OSError("injected atomic rewrite failure"),
        )

    monkeypatch.setattr(MetricsJournal, "append", fail_after_checkpoint)
    with pytest.raises(ScratchCheckpointedTrainingError) as caught:
        run_scratch_checkpointed_training(
            config,
            preparation,
            committed_epoch_observer=(
                lambda event: rendered_epochs.append(event.epoch_index)
            ),
        )

    error = caught.value
    assert observed_epochs == [0]
    assert rendered_epochs == []
    assert error.stage == "metrics_journal"
    assert error.reason_code == "INJECTED_METRICS_JOURNAL_FAILURE"
    assert error.completed_epochs == 1
    assert error.global_step == 1
    latest = output / "checkpoints" / "latest.pt"
    assert error.recoverable_checkpoint == str(latest)
    checkpoint = load_training_checkpoint(latest)
    assert checkpoint.progress.completed_epochs == 1
    assert checkpoint.progress.global_step == 1
    assert sorted(
        path.name for path in output.joinpath("checkpoints").glob("epoch_*.pt")
    ) == ["epoch_000000.pt"]
    assert not output.joinpath("metrics.jsonl").exists()

    status = json.loads(
        output.joinpath("run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["failure_phase"] == "metrics_journal"
    assert status["completed_epochs"] == 1
    assert status["global_step"] == 1
    assert status["recoverable_checkpoint"] == str(latest)
    assert status["rollback_performed"] is False
    assert status["metrics_journal"] == "metrics.jsonl"
    assert status["metrics_event_count"] == 0
    assert status["metrics_last_epoch"] is None
    assert status["metrics_semantic_sha256"] == hashlib.sha256(b"").hexdigest()
    assert status["error"]["original_reason_code"] == (
        "INJECTED_METRICS_JOURNAL_FAILURE"
    )
    assert not output.joinpath(".resume.lock").exists()


def test_checkpoint_failure_records_retained_update_and_initial_recovery(
    tmp_path, monkeypatch
):
    config, preparation = _prepared(
        tmp_path, max_epochs=1, baseline=False
    )
    output = Path(preparation.runtime_paths["output_directory"])

    def fail_save(*args, **kwargs):
        del args, kwargs
        raise OSError("injected checkpoint commit failure")

    monkeypatch.setattr(CheckpointManager, "save_epoch", fail_save)
    with pytest.raises(ScratchCheckpointedTrainingError) as caught:
        run_scratch_checkpointed_training(config, preparation)
    assert caught.value.stage == "fit"
    assert caught.value.global_step == 1
    assert caught.value.completed_epochs == 0
    assert caught.value.rollback_performed is False
    status = json.loads(
        output.joinpath("run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["first_optimizer_update_executed"] is True
    assert status["global_step"] == 1
    assert status["completed_epochs"] == 0
    assert status["recoverable_checkpoint"] is None
    assert status["recovery"] == {
        "kind": "initial_bundle",
        "path": str(output / "initial_bundle.pt"),
    }
    assert not output.joinpath(".resume.lock").exists()


def test_terminal_status_failure_preserves_latest_checkpoint_and_marks_failed(
    tmp_path, monkeypatch
):
    config, preparation = _prepared(
        tmp_path, max_epochs=1, baseline=False
    )
    output = Path(preparation.runtime_paths["output_directory"])
    original = TrainingRunDirectory.write_status
    injected = {"raised": False}

    def fail_terminal_once(self, payload):
        if payload.get("status") == "completed" and not injected["raised"]:
            injected["raised"] = True
            raise OSError("injected terminal status failure")
        return original(self, payload)

    monkeypatch.setattr(
        TrainingRunDirectory, "write_status", fail_terminal_once
    )
    with pytest.raises(ScratchCheckpointedTrainingError) as caught:
        run_scratch_checkpointed_training(config, preparation)
    assert caught.value.stage == "status.terminal"
    latest = output / "checkpoints" / "latest.pt"
    assert latest.is_file()
    status = json.loads(
        output.joinpath("run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["completed_epochs"] == 1
    assert status["global_step"] == 1
    assert status["recoverable_checkpoint"] == str(latest)
    assert status["rollback_performed"] is False
    assert not output.joinpath(".resume.lock").exists()


def test_cli_positional_and_config_alias_execute_scratch_training(tmp_path, capsys):
    first_config, first_preparation = _prepared(
        tmp_path / "positional", max_epochs=1, baseline=False
    )
    assert main(["train", str(first_config.source_path), "--json"]) == 0
    positional = capsys.readouterr()
    positional_report = json.loads(positional.out)
    assert positional_report["status"] == "completed"
    assert "refsite-mlip: training started" in positional.err
    assert "Reference-site MLIP training" in positional.err
    assert "Source: scratch" in positional.err
    assert "Epoch 001/1" in positional.err
    first_checkpoint = load_training_checkpoint(
        positional_report["latest_checkpoint"]
    )
    first_draws = _next_rng_draws()

    second_config, second_preparation = _prepared(
        tmp_path / "option", max_epochs=1, baseline=False
    )
    assert main(
        ["train", "--config", str(second_config.source_path), "--json"]
    ) == 0
    option = capsys.readouterr()
    option_report = json.loads(option.out)
    assert option_report["status"] == "completed"
    assert "refsite-mlip: training started" in option.err
    second_checkpoint = load_training_checkpoint(
        option_report["latest_checkpoint"]
    )
    second_draws = _next_rng_draws()
    assert first_config.config_fingerprint == second_config.config_fingerprint
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
    assert first_checkpoint.fit_history == second_checkpoint.fit_history
    assert first_draws[0] == second_draws[0]
    assert first_draws[1] == second_draws[1]
    assert torch.equal(first_draws[2], second_draws[2])
    assert not Path(first_preparation.runtime_paths["output_directory"]).joinpath(
        ".resume.lock"
    ).exists()
    assert not Path(second_preparation.runtime_paths["output_directory"]).joinpath(
        ".resume.lock"
    ).exists()


def test_python_module_config_alias_executes_scratch_training(tmp_path):
    config, preparation = _prepared(
        tmp_path / "python-module", max_epochs=1, baseline=False
    )
    project_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(project_root / "src") + (
        "" if not existing else os.pathsep + existing
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "refsite_mlip",
            "train",
            "--config",
            str(config.source_path),
            "--json",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "completed"
    assert "refsite-mlip: training started" in completed.stderr
    assert Path(report["latest_checkpoint"]).is_file()
    assert not Path(preparation.runtime_paths["output_directory"]).joinpath(
        ".resume.lock"
    ).exists()


def test_scratch_and_equivalent_bundle_source_have_exact_trajectory(tmp_path):
    config, preparation = _prepared(
        tmp_path, max_epochs=1, baseline=True
    )
    scratch = run_scratch_checkpointed_training(config, preparation)
    scratch_checkpoint = load_training_checkpoint(scratch.latest_path)

    config_path = Path(config.source_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["model_source"] = {
        "kind": "bundle",
        "path": "scratch-output/initial_bundle.pt",
    }
    payload["output_directory"] = "bundle-output"
    bundle_config = tmp_path / "bundle-run.json"
    bundle_config.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    bundle_report = run_training(bundle_config)
    bundle_checkpoint = load_training_checkpoint(
        bundle_report["latest_checkpoint"]
    )

    assert torch.equal(
        scratch_checkpoint.model_state_dict["atomic_baseline"],
        bundle_checkpoint.model_state_dict["atomic_baseline"],
    )
    assert _tree_equal(
        scratch_checkpoint.model_state_dict,
        bundle_checkpoint.model_state_dict,
    )
    assert _tree_equal(
        scratch_checkpoint.optimizer_state_dict,
        bundle_checkpoint.optimizer_state_dict,
    )
    assert _tree_equal(
        scratch_checkpoint.scheduler_state_dict,
        bundle_checkpoint.scheduler_state_dict,
    )
    assert scratch_checkpoint.selection_state == bundle_checkpoint.selection_state
    assert scratch_checkpoint.progress == bundle_checkpoint.progress
    assert scratch_checkpoint.fit_history == bundle_checkpoint.fit_history
    assert _tree_equal(scratch.fit_result.to_dict(), bundle_report["fit_result"])
    assert Path(scratch.run_directory).joinpath("metrics.jsonl").read_bytes() == (
        tmp_path / "bundle-output" / "metrics.jsonl"
    ).read_bytes()
