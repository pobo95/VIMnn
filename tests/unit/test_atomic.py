from __future__ import annotations

import os
from pathlib import Path

import pytest

import refsite_mlip._atomic as atomic_module
from refsite_mlip._atomic import commit_temporary_file


def test_no_clobber_commit_and_competing_target_contract(tmp_path):
    temporary = tmp_path / ".artifact.tmp"
    target = tmp_path / "artifact.pt"
    temporary.write_bytes(b"new")

    result = commit_temporary_file(temporary, target, overwrite=False)
    assert target.read_bytes() == b"new"
    assert not temporary.exists()
    assert result.temporary_cleanup_succeeded
    assert result.orphaned_temporary is None

    competing_temporary = tmp_path / ".competing.tmp"
    competing_temporary.write_bytes(b"loser")
    target.write_bytes(b"winner")
    with pytest.raises(FileExistsError):
        commit_temporary_file(competing_temporary, target, overwrite=False)
    assert target.read_bytes() == b"winner"
    assert competing_temporary.read_bytes() == b"loser"


def test_post_commit_cleanup_failure_is_success_with_orphan(
    tmp_path, monkeypatch
):
    temporary = tmp_path / ".artifact.tmp"
    target = tmp_path / "artifact.pt"
    temporary.write_bytes(b"committed bytes")
    original_unlink = Path.unlink

    def fail_temporary_unlink(path, *args, **kwargs):
        if path == temporary:
            raise OSError("injected post-commit cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)
    result = commit_temporary_file(temporary, target, overwrite=False)

    assert target.read_bytes() == b"committed bytes"
    assert temporary.read_bytes() == b"committed bytes"
    assert not result.temporary_cleanup_succeeded
    assert result.orphaned_temporary == temporary


def test_overwrite_commit_and_precommit_failure_preserve_bytes(
    tmp_path, monkeypatch
):
    target = tmp_path / "artifact.pt"
    target.write_bytes(b"old")
    temporary = tmp_path / ".artifact.tmp"
    temporary.write_bytes(b"new")
    result = commit_temporary_file(temporary, target, overwrite=True)
    assert target.read_bytes() == b"new"
    assert not temporary.exists()
    assert result.temporary_cleanup_succeeded

    failed_temporary = tmp_path / ".failed.tmp"
    failed_temporary.write_bytes(b"uncommitted")

    def fail_replace(*args, **kwargs):
        del args, kwargs
        raise OSError("injected pre-commit failure")

    monkeypatch.setattr(atomic_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="pre-commit"):
        commit_temporary_file(failed_temporary, target, overwrite=True)
    assert target.read_bytes() == b"new"
    assert failed_temporary.read_bytes() == b"uncommitted"
