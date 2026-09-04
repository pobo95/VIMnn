from __future__ import annotations

import json

import pytest

from refsite_mlip.training.run_directory import (
    RunDirectoryError,
    TrainingRunDirectory,
)


def test_startup_paths_and_immutable_data_manifest(tmp_path):
    root = tmp_path / "run"
    directory = TrainingRunDirectory.create(root)

    assert directory.data_manifest_path == root / "data_manifest.json"
    assert directory.initial_bundle_path == root / "initial_bundle.pt"
    assert not directory.initial_bundle_path.exists()

    directory.write_data_manifest({"z": 2, "samples": ["b", "a"]})
    assert directory.data_manifest_path.read_text(encoding="utf-8") == (
        '{"samples":["b","a"],"z":2}\n'
    )
    assert json.loads(directory.data_manifest_path.read_text(encoding="utf-8")) == {
        "samples": ["b", "a"],
        "z": 2,
    }

    original = directory.data_manifest_path.read_bytes()
    with pytest.raises(RunDirectoryError) as caught:
        directory.write_data_manifest({"replacement": True})
    assert caught.value.reason_code == "RUNTIME_FILE_ALREADY_EXISTS"
    assert caught.value.stage == "run_directory.data_manifest"
    assert directory.data_manifest_path.read_bytes() == original
    assert not list(root.glob(".data_manifest.json.*.tmp"))


def test_checkpoint_directory_is_created_exclusively_and_empty(tmp_path):
    directory = TrainingRunDirectory.create(tmp_path / "run")

    created = directory.create_checkpoints_directory()
    assert created == directory.checkpoints
    assert created.is_dir()
    assert list(created.iterdir()) == []

    marker = created / "foreign.pt"
    marker.write_bytes(b"preserve")
    with pytest.raises(RunDirectoryError) as caught:
        directory.create_checkpoints_directory()
    assert caught.value.reason_code == "CHECKPOINT_DIRECTORY_ALREADY_EXISTS"
    assert caught.value.stage == "run_directory.checkpoints.create"
    assert marker.read_bytes() == b"preserve"


@pytest.mark.parametrize("existing_kind", ["file", "symlink"])
def test_checkpoint_directory_rejects_existing_path_without_mutation(
    tmp_path, existing_kind
):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    checkpoint_path = directory.checkpoints

    if existing_kind == "file":
        checkpoint_path.write_bytes(b"foreign-file")
        expected_reason = "CHECKPOINT_DIRECTORY_ALREADY_EXISTS"
    else:
        target = tmp_path / "foreign-directory"
        target.mkdir()
        (target / "marker").write_bytes(b"foreign-directory")
        checkpoint_path.symlink_to(target, target_is_directory=True)
        expected_reason = "CHECKPOINT_DIRECTORY_SYMLINK_REJECTED"

    with pytest.raises(RunDirectoryError) as caught:
        directory.create_checkpoints_directory()
    assert caught.value.reason_code == expected_reason

    if existing_kind == "file":
        assert checkpoint_path.read_bytes() == b"foreign-file"
    else:
        assert checkpoint_path.is_symlink()
        assert checkpoint_path.joinpath("marker").read_bytes() == b"foreign-directory"


def test_checkpoint_directory_wraps_creation_failure(tmp_path, monkeypatch):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    path_type = type(directory.checkpoints)
    original_mkdir = path_type.mkdir

    def fail_checkpoint_mkdir(path, *args, **kwargs):
        if path == directory.checkpoints:
            raise OSError("injected checkpoint mkdir failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "mkdir", fail_checkpoint_mkdir)
    with pytest.raises(RunDirectoryError) as caught:
        directory.create_checkpoints_directory()
    assert caught.value.reason_code == "CHECKPOINT_DIRECTORY_CREATE_FAILED"
    assert caught.value.original_exception_type == "OSError"
    assert "injected checkpoint mkdir failure" in caught.value.original_exception_message
    assert not directory.checkpoints.exists()
