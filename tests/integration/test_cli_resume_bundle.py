from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase.io import read, write

from refsite_mlip.cli.main import main
from refsite_mlip.cli.errors import CLIError, CLIInterruptedError
from refsite_mlip.cli.resume import resume_training
from refsite_mlip.cli.train import run_training
from refsite_mlip.training import (
    CheckpointManager,
    FitExecutionError,
    MetricsJournal,
    MetricsJournalError,
    RunDirectoryError,
    TrainingRunDirectory,
    load_training_checkpoint,
    save_training_checkpoint,
)

from test_validate_train_config_cli import _simple_case, training_bundle


def _write_variant(source: Path, target: Path, *, output: str, epochs: int, **updates):
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["output_directory"] = output
    payload["fit"]["max_epochs"] = epochs
    for section, values in updates.items():
        payload[section].update(values)
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return target


def _tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and torch.equal(first.detach().cpu(), second.detach().cpu())
        )
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            _tree_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return isinstance(second, (tuple, list)) and len(first) == len(second) and all(
            _tree_equal(left, right) for left, right in zip(first, second)
        )
    return first == second


def _rng_snapshot():
    numpy = np.random.get_state()
    return (
        random.getstate(),
        (numpy[0], numpy[1].copy(), *numpy[2:]),
        torch.get_rng_state().clone(),
        tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    )


def _rng_equal(first, second):
    return (
        first[0] == second[0]
        and first[1][0] == second[1][0]
        and np.array_equal(first[1][1], second[1][1])
        and first[1][2:] == second[1][2:]
        and torch.equal(first[2], second[2])
        and len(first[3]) == len(second[3])
        and all(torch.equal(left, right) for left, right in zip(first[3], second[3]))
    )


def _file_snapshot(root: Path):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _fresh_one_epoch(tmp_path, training_bundle, *, output="split-output"):
    source, _ = _simple_case(tmp_path, training_bundle)
    config = _write_variant(
        source,
        tmp_path / f"{output}.json",
        output=output,
        epochs=1,
    )
    report = run_training(config)
    return config, tmp_path / output, report


def test_resume_dry_run_is_fully_read_only_and_cwd_independent(
    training_bundle, tmp_path, monkeypatch
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    before = _file_snapshot(run_directory)
    rng_before = _rng_snapshot()
    resume_module = __import__("refsite_mlip.cli.resume", fromlist=["resume"])

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dry-run must not instantiate model/optimizer")

    monkeypatch.setattr(resume_module, "_prepare_training_runtime", forbidden)
    other = tmp_path / "other-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    report = resume_training(run_directory, max_epochs=3, dry_run=True)
    assert report["status"] == "resume_preflight_ready"
    assert report["message"] == "no training was executed"
    assert report["checkpoint"]["managed_epochs"] == [0]
    assert _file_snapshot(run_directory) == before
    assert not (run_directory / ".resume.lock").exists()
    assert _rng_equal(rng_before, _rng_snapshot())


@pytest.mark.parametrize(
    "scheduler",
    [
        None,
        {
            "kind": "reduce_on_plateau",
            "patience": 0,
            "factor": 0.5,
        },
    ],
)
def test_cpu_float64_continuous_three_epoch_equals_train_one_resume_two(
    training_bundle, tmp_path, scheduler
):
    source, _ = _simple_case(tmp_path, training_bundle)
    updates = {} if scheduler is None else {"scheduler": scheduler}
    continuous_config = _write_variant(
        source,
        tmp_path / "continuous.json",
        output="continuous-output",
        epochs=3,
        **updates,
    )
    split_config = _write_variant(
        source,
        tmp_path / "split.json",
        output="split-output",
        epochs=1,
        **updates,
    )

    continuous = run_training(continuous_config)
    continuous_checkpoint = load_training_checkpoint(
        continuous["latest_checkpoint"]
    )
    continuous_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    split = run_training(split_config)
    epoch_zero = Path(split["latest_checkpoint"]).with_name("epoch_000000.pt")
    epoch_zero_before = epoch_zero.read_bytes()
    read_only_paths = (
        split_config,
        training_bundle["path"],
        tmp_path / "train.xyz",
        tmp_path / "validation.xyz",
    )
    read_only_before = tuple(path.read_bytes() for path in read_only_paths)
    resumed = resume_training(tmp_path / "split-output", max_epochs=3)
    resumed_checkpoint = load_training_checkpoint(resumed["latest_checkpoint"])
    resumed_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

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
    assert resumed["fit_result"] == continuous["fit_result"]
    assert resumed["resumed_epochs_completed"] == 2
    assert continuous_draws[0] == resumed_draws[0]
    assert continuous_draws[1] == resumed_draws[1]
    assert torch.equal(continuous_draws[2], resumed_draws[2])
    assert epoch_zero.read_bytes() == epoch_zero_before
    assert tuple(path.read_bytes() for path in read_only_paths) == read_only_before
    assert sorted(
        path.name
        for path in (tmp_path / "split-output" / "checkpoints").glob("epoch_*.pt")
    ) == ["epoch_000000.pt", "epoch_000001.pt", "epoch_000002.pt"]
    assert (tmp_path / "continuous-output" / "metrics.jsonl").read_bytes() == (
        tmp_path / "split-output" / "metrics.jsonl"
    ).read_bytes()
    assert not (tmp_path / "split-output" / ".resume.lock").exists()


def test_resume_recovers_a_missing_committed_journal_suffix_before_continuing(
    training_bundle, tmp_path
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    journal_path = run_directory / "metrics.jsonl"
    journal_path.unlink()
    status_path = run_directory / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "metrics_journal": "metrics.jsonl",
            "metrics_event_count": 0,
            "metrics_last_epoch": None,
            "metrics_semantic_sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    status_before_dry_run = status_path.read_bytes()

    dry_run = resume_training(run_directory, max_epochs=2, dry_run=True)
    assert dry_run["message"] == "no training was executed"
    assert not journal_path.exists()
    assert status_path.read_bytes() == status_before_dry_run

    resumed = resume_training(run_directory, max_epochs=2)

    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["epoch_index"] for event in events] == [0, 1]
    assert resumed["metrics_event_count"] == 2
    assert resumed["metrics_last_epoch"] == 1
    assert resumed["metrics_semantic_sha256"] == hashlib.sha256(
        journal_path.read_bytes()
    ).hexdigest()


def test_resume_journal_failure_preserves_checkpoint_and_next_resume_recovers(
    training_bundle, tmp_path, monkeypatch
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    real_observer = MetricsJournal.__call__

    def fail_resumed_epoch(self, event):
        if event.epoch_index == 1:
            raise MetricsJournalError(
                "INJECTED_RESUME_JOURNAL_FAILURE",
                "injected resumed journal failure",
                stage="metrics_journal.commit",
                path=self.path,
                epoch_index=event.epoch_index,
                last_valid_epoch=0,
                original_error=OSError("injected resumed journal failure"),
            )
        return real_observer(self, event)

    monkeypatch.setattr(MetricsJournal, "__call__", fail_resumed_epoch)
    with pytest.raises(CLIError) as caught:
        resume_training(run_directory, max_epochs=2)
    assert caught.value.failure_phase == "metrics_journal"

    latest = load_training_checkpoint(run_directory / "checkpoints" / "latest.pt")
    assert latest.progress.completed_epochs == 2
    assert latest.progress.last_completed_epoch == 1
    status = json.loads((run_directory / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["failure_phase"] == "metrics_journal"
    assert status["completed_epochs"] == 2
    assert status["metrics_event_count"] == 1
    assert status["metrics_last_epoch"] == 0

    monkeypatch.setattr(MetricsJournal, "__call__", real_observer)
    resumed = resume_training(run_directory, max_epochs=3)
    events = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["epoch_index"] for event in events] == [0, 1, 2]
    assert resumed["completed_epochs"] == 3
    assert resumed["metrics_event_count"] == 3


def test_resume_rejects_nonincrease_lock_and_data_change(
    training_bundle, tmp_path, capsys
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    status_before = (run_directory / "run_status.json").read_bytes()
    assert main(
        ["resume", str(run_directory), "--max-epochs", "1", "--json"]
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RESUME_MAX_EPOCHS_NOT_INCREASED" in captured.err
    assert (run_directory / "run_status.json").read_bytes() == status_before

    lock = run_directory / ".resume.lock"
    lock.write_text("foreign", encoding="utf-8")
    assert main(["resume", str(run_directory), "--max-epochs", "3"]) == 1
    captured = capsys.readouterr()
    assert "RESUME_LOCK_EXISTS" in captured.err
    assert lock.read_text(encoding="utf-8") == "foreign"
    lock.unlink()

    train_path = tmp_path / "train.xyz"
    frames = read(train_path, index=":", format="extxyz")
    frames[0].positions[0, 0] += 0.01
    write(train_path, frames, format="extxyz")
    assert main(["resume", str(run_directory), "--max-epochs", "3"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "TRAIN_DATA_DIGEST_MISMATCH" in captured.err
    assert (run_directory / "run_status.json").read_bytes() == status_before


def test_resume_rejects_unowned_running_status(
    training_bundle, tmp_path, capsys
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    status_path = run_directory / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "running"
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    before = _file_snapshot(run_directory)

    assert main(
        ["resume", str(run_directory), "--max-epochs", "3", "--dry-run"]
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RUN_STATUS_ACTIVE_OR_UNCERTAIN" in captured.err
    assert _file_snapshot(run_directory) == before
    assert not (run_directory / ".resume.lock").exists()


def test_resume_rechecks_status_after_lock_without_clobbering_foreign_change(
    training_bundle, tmp_path, monkeypatch
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    status_path = run_directory / "run_status.json"
    original_acquire = TrainingRunDirectory.acquire_resume_lock
    changed = {}

    def acquire_then_change(directory):
        lock = original_acquire(directory)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["operation_phase"] = "foreign-change"
        status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
        changed["bytes"] = status_path.read_bytes()
        return lock

    monkeypatch.setattr(
        TrainingRunDirectory, "acquire_resume_lock", acquire_then_change
    )
    with pytest.raises(Exception) as caught:
        resume_training(run_directory, max_epochs=3)
    assert getattr(caught.value, "reason_code", None) == (
        "RUN_STATUS_TOCTOU_MISMATCH"
    )
    assert status_path.read_bytes() == changed["bytes"]
    assert not (run_directory / ".resume.lock").exists()


def test_resume_rechecks_metrics_journal_after_lock_without_repairing_race(
    training_bundle, tmp_path, monkeypatch
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    journal_path = run_directory / "metrics.jsonl"
    original_acquire = TrainingRunDirectory.acquire_resume_lock

    def acquire_then_remove_prefix(directory):
        lock = original_acquire(directory)
        journal_path.unlink()
        return lock

    monkeypatch.setattr(
        TrainingRunDirectory,
        "acquire_resume_lock",
        acquire_then_remove_prefix,
    )
    with pytest.raises(CLIError) as caught:
        resume_training(run_directory, max_epochs=2)
    assert getattr(caught.value, "reason_code", None) == (
        "METRICS_JOURNAL_TOCTOU_MISMATCH"
    )
    assert not journal_path.exists()
    assert not (run_directory / ".resume.lock").exists()


def test_resume_rejects_epoch_gap_and_arbitrary_checkpoint_path(
    training_bundle, tmp_path, capsys
):
    _, run_directory, report = _fresh_one_epoch(tmp_path, training_bundle)
    epoch = run_directory / "checkpoints" / "epoch_000000.pt"
    moved = run_directory / "checkpoints" / "epoch_000001.pt"
    epoch.rename(moved)
    assert main(["resume", str(run_directory), "--max-epochs", "3"]) == 1
    captured = capsys.readouterr()
    assert "CHECKPOINT_HISTORY_INVALID" in captured.err

    assert main(
        ["resume", report["best_checkpoint"], "--max-epochs", "3"]
    ) == 1
    captured = capsys.readouterr()
    assert "INVALID_RUN_DIRECTORY" in captured.err


def test_resume_rejects_config_radius_dtype_seed_and_bundle_changes(
    training_bundle, tmp_path, capsys
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    resolved_path = run_directory / "resolved_config.json"
    original_config = resolved_path.read_bytes()
    original_status = (run_directory / "run_status.json").read_bytes()

    for section, field, value in (
        ("runtime", "seed", 99),
        ("runtime", "dtype", "float32"),
        ("radii", "r_ot", 4.1),
        ("data", "batch_size", 1),
    ):
        payload = json.loads(original_config)
        payload[section][field] = value
        resolved_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        assert main(["resume", str(run_directory), "--max-epochs", "3"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "CONFIG_FINGERPRINT_MISMATCH" in captured.err
        resolved_path.write_bytes(original_config)
        assert (run_directory / "run_status.json").read_bytes() == original_status

    bundle_path = training_bundle["path"]
    original_bundle = bundle_path.read_bytes()
    try:
        bundle_path.write_bytes(original_bundle[: max(1, len(original_bundle) // 2)])
        assert main(["resume", str(run_directory), "--max-epochs", "3"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "stage=" in captured.err and "reason=" in captured.err
        assert (run_directory / "run_status.json").read_bytes() == original_status
    finally:
        bundle_path.write_bytes(original_bundle)


def test_repeated_resume_epoch_continuity_and_best_update_policy(
    training_bundle, tmp_path, capsys
):
    _, run_directory, first = _fresh_one_epoch(tmp_path, training_bundle)
    checkpoint_root = run_directory / "checkpoints"
    epoch_zero_before = (checkpoint_root / "epoch_000000.pt").read_bytes()
    best_before = (checkpoint_root / "best.pt").read_bytes()

    assert main(
        ["resume", str(run_directory), "--max-epochs", "2", "--json"]
    ) == 0
    captured = capsys.readouterr()
    second = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert "resume started" in captured.err and "resume completed" in captured.err
    second_checkpoint = load_training_checkpoint(second["latest_checkpoint"])
    second_is_best = second_checkpoint.fit_history[-1]["decision"]["is_best"]
    if not second_is_best:
        assert (checkpoint_root / "best.pt").read_bytes() == best_before
    best_before_third = (checkpoint_root / "best.pt").read_bytes()

    third = resume_training(run_directory, max_epochs=3)
    third_checkpoint = load_training_checkpoint(third["latest_checkpoint"])
    third_is_best = third_checkpoint.fit_history[-1]["decision"]["is_best"]
    if not third_is_best:
        assert (checkpoint_root / "best.pt").read_bytes() == best_before_third
    assert third["resume_from_epoch"] == 2
    assert third["resumed_epochs_completed"] == 1
    assert third_checkpoint.metadata.resolved_configuration["fit"][
        "max_epochs"
    ] == 3
    assert third_checkpoint.progress.completed_epochs == 3
    assert len(third_checkpoint.fit_history) == 3
    assert (checkpoint_root / "epoch_000000.pt").read_bytes() == epoch_zero_before
    assert sorted(path.name for path in checkpoint_root.glob("epoch_*.pt")) == [
        "epoch_000000.pt",
        "epoch_000001.pt",
        "epoch_000002.pt",
    ]
    assert Path(first["latest_checkpoint"]) == Path(third["latest_checkpoint"])


def test_resume_failure_and_interrupt_preserve_recoverable_latest_and_unlock(
    training_bundle, tmp_path, monkeypatch
):
    _, run_directory, first = _fresh_one_epoch(tmp_path, training_bundle)
    latest = Path(first["latest_checkpoint"])
    latest_before = latest.read_bytes()
    module = __import__("refsite_mlip.cli.resume", fromlist=["resume"])

    def failed(*args, **kwargs):
        del args, kwargs
        raise FitExecutionError(
            phase="train",
            epoch_index=1,
            current_global_step=1,
            completed_epochs=0,
            training_update_completed=False,
            cause=RuntimeError("injected resumed training failure"),
        )

    monkeypatch.setattr(module, "run_checkpointed_resumed_fit", failed)
    with pytest.raises(Exception) as caught:
        resume_training(run_directory, max_epochs=3)
    assert getattr(caught.value, "failure_phase", None) == "train"
    status = json.loads((run_directory / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_epochs"] == 1
    assert status["recoverable_global_step"] == 1
    assert status["rollback_performed"] is False
    assert latest.read_bytes() == latest_before
    assert not (run_directory / ".resume.lock").exists()

    monkeypatch.setattr(
        module,
        "run_checkpointed_resumed_fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(CLIInterruptedError):
        resume_training(run_directory, max_epochs=3)
    status = json.loads((run_directory / "run_status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["recoverable_checkpoint"] == str(latest)
    assert latest.read_bytes() == latest_before
    assert not (run_directory / ".resume.lock").exists()


def test_resume_interrupt_after_new_epoch_reports_new_recoverable_latest(
    training_bundle, tmp_path, monkeypatch
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    original_save = CheckpointManager.save_epoch

    def save_then_interrupt(self, checkpoint, record):
        managed = original_save(self, checkpoint, record)
        if record.epoch_index == 1:
            raise KeyboardInterrupt()
        return managed

    monkeypatch.setattr(CheckpointManager, "save_epoch", save_then_interrupt)
    with pytest.raises(CLIInterruptedError):
        resume_training(run_directory, max_epochs=3)
    status = json.loads((run_directory / "run_status.json").read_text())
    latest = load_training_checkpoint(run_directory / "checkpoints" / "latest.pt")
    assert status["status"] == "interrupted"
    assert status["completed_epochs"] == 2
    assert status["resumed_epochs_completed"] == 1
    assert status["recoverable_global_step"] == latest.progress.global_step == 2
    assert latest.progress.next_epoch == 2
    assert (run_directory / "checkpoints" / "epoch_000001.pt").is_file()
    assert not (run_directory / ".resume.lock").exists()

def test_resume_rejects_early_stopped_checkpoint_and_checkpoint_symlink(
    training_bundle, tmp_path, capsys
):
    _, run_directory, first = _fresh_one_epoch(tmp_path, training_bundle)
    latest_path = Path(first["latest_checkpoint"])
    checkpoint = load_training_checkpoint(latest_path)
    stopped_selection = replace(
        checkpoint.selection_state,
        stopped_early=True,
        stop_epoch=checkpoint.progress.last_completed_epoch,
        stop_reason="injected",
    )
    stopped_progress = replace(checkpoint.progress, stopped_early=True)
    stopped = replace(
        checkpoint,
        selection_state=stopped_selection,
        progress=stopped_progress,
    )
    save_training_checkpoint(stopped, latest_path, overwrite=True)
    assert main(["resume", str(run_directory), "--max-epochs", "3"]) == 1
    captured = capsys.readouterr()
    assert "CHECKPOINT_HISTORY_INVALID" in captured.err

    other_root = tmp_path / "symlink-case"
    other_root.mkdir()
    _, other_directory, _ = _fresh_one_epoch(
        other_root, training_bundle, output="output"
    )
    checkpoints = other_directory / "checkpoints"
    foreign = other_root / "foreign-checkpoints"
    checkpoints.rename(foreign)
    checkpoints.symlink_to(foreign, target_is_directory=True)
    assert main(["resume", str(other_directory), "--max-epochs", "3"]) == 1
    captured = capsys.readouterr()
    assert "CHECKPOINT_DIRECTORY_SYMLINK_REJECTED" in captured.err
    assert foreign.is_dir()


def test_resume_status_write_failure_keeps_new_checkpoint_recoverable(
    training_bundle, tmp_path, monkeypatch
):
    _, run_directory, _ = _fresh_one_epoch(tmp_path, training_bundle)
    module = __import__("refsite_mlip.cli.resume", fromlist=["resume"])
    original = TrainingRunDirectory.write_status

    def fail_completed(self, value):
        if value.get("status") == "completed":
            raise RunDirectoryError(
                "INJECTED_STATUS_FAILURE",
                "injected completed-status write failure",
                stage="run_directory.status",
                path=self.status_path,
            )
        return original(self, value)

    monkeypatch.setattr(TrainingRunDirectory, "write_status", fail_completed)
    with pytest.raises(Exception):
        module.resume_training(run_directory, max_epochs=2)
    latest = load_training_checkpoint(run_directory / "checkpoints" / "latest.pt")
    status = json.loads((run_directory / "run_status.json").read_text())
    assert latest.progress.completed_epochs == 2
    assert status["status"] == "failed"
    assert status["recoverable_global_step"] == latest.progress.global_step
    assert not (run_directory / ".resume.lock").exists()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_one_epoch_resume_smoke_when_available(
    training_bundle, tmp_path, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("9D CUDA gate: unavailable")
    source, _ = _simple_case(tmp_path, training_bundle)
    config = _write_variant(
        source,
        tmp_path / f"cuda-{dtype}.json",
        output=f"cuda-{dtype}-output",
        epochs=1,
        runtime={"device": "cuda", "dtype": dtype},
    )
    run_training(config)
    report = resume_training(tmp_path / f"cuda-{dtype}-output", max_epochs=2)
    checkpoint = load_training_checkpoint(report["latest_checkpoint"])
    assert report["completed_epochs"] == 2
    assert checkpoint.progress.next_epoch == 2
    assert checkpoint.cuda_device_count == torch.cuda.device_count()
