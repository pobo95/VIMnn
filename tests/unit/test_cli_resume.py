from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from refsite_mlip.cli.errors import CLIInterruptedError
from refsite_mlip.cli.main import build_parser, main
from refsite_mlip.cli.resume import render_resume_human, render_resume_json
from refsite_mlip.training import RunDirectoryError, TrainingRunDirectory


def _ready_report():
    return {
        "schema_version": "refsite_training_resume_preflight_v1",
        "status": "resume_preflight_ready",
        "training_executed": False,
        "mutation_performed": False,
        "run_directory": "/runtime/run",
        "path_kind": "runtime_location_not_semantic_fingerprint",
        "config_fingerprint": "1" * 64,
        "bundle_fingerprint": "2" * 64,
        "train_semantic_digest": "3" * 64,
        "validation_semantic_digest": "4" * 64,
        "seed": 17,
        "runtime": {
            "device": "cpu",
            "dtype": "float64",
            "solver_path": "train_fixed",
        },
        "checkpoint": {
            "source": "/runtime/run/checkpoints/latest.pt",
            "scope": "epoch_boundary",
            "completed_epochs": 1,
            "next_epoch": 1,
            "global_step": 2,
            "max_epochs": 1,
            "best_epoch": 0,
            "best_global_step": 2,
            "managed_epochs": [0],
        },
        "requested_max_epochs": 3,
        "continuation_epoch_count": 2,
        "train_batch_count": 2,
        "validation_batch_count": 1,
        "template_ids": ["template"],
        "resume_policy": {},
        "exact_rng_restore_required": True,
        "lock_state": "available_not_acquired",
        "message": "no training was executed",
    }


def test_resume_parser_requires_positive_max_epochs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "resume",
            "run",
            "--max-epochs",
            "20",
            "--dry-run",
            "--json",
            "--quiet",
        ]
    )
    assert args.command == "resume"
    assert args.run_directory == "run"
    assert args.max_epochs == 20
    assert args.dry_run and args.json_output and args.quiet
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["resume", "run", "--max-epochs", "0"])
    assert caught.value.code == 2


def test_resume_lock_is_exclusive_owned_and_cleaned(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    directory = TrainingRunDirectory.open_existing(root)
    lock = directory.acquire_resume_lock()
    assert lock.owned and directory.resume_lock_path.is_file()
    with pytest.raises(RunDirectoryError, match="RESUME_LOCK_EXISTS"):
        directory.acquire_resume_lock()
    lock.release()
    assert not lock.owned and not directory.resume_lock_path.exists()
    lock.release()


def test_resume_lock_validate_owned_is_read_only_and_enter_rejects_release(
    tmp_path,
):
    root = tmp_path / "run"
    root.mkdir()
    directory = TrainingRunDirectory.open_existing(root)
    lock = directory.acquire_resume_lock()
    before = directory.resume_lock_path.read_bytes()
    identity = directory.resume_lock_path.lstat()

    assert lock.validate_owned(directory.resume_lock_path) is None
    assert lock.owned
    assert directory.resume_lock_path.read_bytes() == before
    after = directory.resume_lock_path.lstat()
    assert (after.st_dev, after.st_ino) == (identity.st_dev, identity.st_ino)

    with pytest.raises(RunDirectoryError) as mismatch:
        lock.validate_owned(root / ".different.lock")
    assert mismatch.value.reason_code == "RESUME_LOCK_PATH_MISMATCH"
    assert lock.owned and directory.resume_lock_path.read_bytes() == before

    lock.release()
    with pytest.raises(RunDirectoryError) as released:
        lock.validate_owned(directory.resume_lock_path)
    assert released.value.reason_code == "RESUME_LOCK_NOT_OWNED"
    with pytest.raises(RunDirectoryError, match="RESUME_LOCK_NOT_OWNED"):
        with lock:
            raise AssertionError("released lock body must not execute")


@pytest.mark.parametrize("replacement_kind", ["file", "symlink"])
def test_resume_lock_validate_owned_preserves_replacement(
    tmp_path, replacement_kind
):
    root = tmp_path / "run"
    root.mkdir()
    directory = TrainingRunDirectory.open_existing(root)
    lock = directory.acquire_resume_lock()
    owned = root / ".owned-lock"
    directory.resume_lock_path.rename(owned)
    if replacement_kind == "file":
        directory.resume_lock_path.write_bytes(b"foreign")
    else:
        foreign = tmp_path / "foreign"
        foreign.write_bytes(b"foreign")
        directory.resume_lock_path.symlink_to(foreign)

    with pytest.raises(RunDirectoryError) as caught:
        lock.validate_owned(directory.resume_lock_path)
    assert caught.value.reason_code == "RESUME_LOCK_OWNERSHIP_LOST"
    assert lock.owned
    assert owned.is_file()
    if replacement_kind == "file":
        assert directory.resume_lock_path.read_bytes() == b"foreign"
    else:
        assert directory.resume_lock_path.is_symlink()
        assert directory.resume_lock_path.read_bytes() == b"foreign"

    # Explicit test teardown restores the inode owned by this lock; production
    # validation never removes a replacement on the caller's behalf.
    directory.resume_lock_path.unlink()
    owned.rename(directory.resume_lock_path)
    lock.release()


def test_resume_lock_never_removes_a_replacement(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    directory = TrainingRunDirectory.open_existing(root)
    lock = directory.acquire_resume_lock()
    owned = root / ".owned-lock"
    directory.resume_lock_path.rename(owned)
    directory.resume_lock_path.write_text("foreign", encoding="utf-8")
    with pytest.raises(RunDirectoryError, match="RESUME_LOCK_OWNERSHIP_LOST"):
        lock.release()
    assert directory.resume_lock_path.read_text(encoding="utf-8") == "foreign"
    assert owned.is_file()


def test_resume_lock_symlink_is_rejected_without_removing_target(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    target = tmp_path / "foreign-lock"
    target.write_text("foreign", encoding="utf-8")
    (root / ".resume.lock").symlink_to(target)
    directory = TrainingRunDirectory.open_existing(root)
    with pytest.raises(RunDirectoryError, match="RESUME_LOCK_SYMLINK_REJECTED"):
        directory.validate_resume_lock_available()
    assert target.read_text(encoding="utf-8") == "foreign"

    real_run = tmp_path / "real-run"
    real_run.mkdir()
    linked_run = tmp_path / "linked-run"
    linked_run.symlink_to(real_run, target_is_directory=True)
    with pytest.raises(RunDirectoryError, match="RUN_DIRECTORY_SYMLINK_REJECTED"):
        TrainingRunDirectory.open_existing(linked_run)


def test_resume_lock_context_preserves_body_and_release_failures(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    root.mkdir()
    directory = TrainingRunDirectory.open_existing(root)

    with directory.acquire_resume_lock() as lock:
        assert lock.owned
    assert not directory.resume_lock_path.exists()

    body_error = RuntimeError("body failure")
    with pytest.raises(RuntimeError) as caught:
        with directory.acquire_resume_lock():
            raise body_error
    assert caught.value is body_error
    assert caught.value.__cause__ is None
    assert not directory.resume_lock_path.exists()

    original_unlink = Path.unlink

    def fail_lock_unlink(path, *args, **kwargs):
        if path == directory.resume_lock_path:
            raise OSError("injected release failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_lock_unlink)
    with pytest.raises(RunDirectoryError, match="RESUME_LOCK_RELEASE_FAILED"):
        with directory.acquire_resume_lock():
            pass
    assert directory.resume_lock_path.exists()

    # The failed release left an owned/stale lock by contract; remove it only
    # as explicit test teardown, then exercise the double-failure path.
    monkeypatch.setattr(Path, "unlink", original_unlink)
    directory.resume_lock_path.unlink()
    monkeypatch.setattr(Path, "unlink", fail_lock_unlink)
    second_body_error = ValueError("second body failure")
    with pytest.raises(ValueError) as double:
        with directory.acquire_resume_lock():
            raise second_body_error
    assert double.value is second_body_error
    assert isinstance(double.value.__cause__, RunDirectoryError)
    assert double.value.__cause__.reason_code == "RESUME_LOCK_RELEASE_FAILED"
    assert directory.resume_lock_path.exists()


def test_resume_rendering_is_deterministic_plain_json():
    report = _ready_report()
    reversed_report = dict(reversed(tuple(report.items())))
    assert render_resume_json(report) == render_resume_json(reversed_report)
    assert json.loads(render_resume_json(report)) == report
    human = render_resume_human(report)
    assert "Status: ready" in human
    assert "No training was executed" in human


def test_resume_cli_routes_json_and_interrupt(monkeypatch, capsys):
    module = importlib.import_module("refsite_mlip.cli.resume")
    report = _ready_report()
    calls = []

    def fake(path, *, max_epochs, dry_run, progress_renderer):
        calls.append((path, max_epochs, dry_run, progress_renderer))
        return report

    monkeypatch.setattr(module, "resume_training", fake)
    assert main(
        [
            "resume",
            "run",
            "--max-epochs",
            "3",
            "--dry-run",
            "--json",
            "--quiet",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == report and captured.err == ""
    assert calls[0][:3] == ("run", 3, True)
    assert calls[0][3].config.enabled is False

    def interrupted(*args, **kwargs):
        del args, kwargs
        raise CLIInterruptedError(
            "RESUME_INTERRUPTED",
            "interrupted",
            stage="resume.fit",
            path="run",
        )

    monkeypatch.setattr(module, "resume_training", interrupted)
    assert main(["resume", "run", "--max-epochs", "3"]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RESUME_INTERRUPTED" in captured.err
