from __future__ import annotations

import copy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import (
    CHECKPOINT_SCHEMA_VERSION,
    FitConfig,
    FitProgress,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    TrainingCheckpoint,
    ValidationStepConfig,
    build_scheduler,
    capture_training_checkpoint,
    fingerprint_batch_sequence,
    load_training_checkpoint,
    save_training_checkpoint,
)


checkpoint_module = __import__(
    "refsite_mlip.training.checkpoint", fromlist=["checkpoint"]
)


class TinyModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([1.0, -0.5], dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "atomic_baseline", torch.tensor([0.25], dtype=torch.float64, device=device)
        )


def _batch(
    sample_ids=("sample",),
    *,
    dtype=torch.float64,
    device="cpu",
    position_shift=0.0,
    energy_shift=0.0,
    fingerprint="2" * 64,
):
    count = len(sample_ids)
    positions = torch.arange(count * 3, dtype=dtype, device=device).reshape(count, 3)
    positions = positions * 0.01 + position_shift
    ptr = torch.arange(count + 1, dtype=torch.long, device=device)
    return StructureBatch(
        sample_ids=tuple(sample_ids),
        template_ids=("template",) * count,
        template_fingerprints=(fingerprint,) * count,
        positions=positions,
        atomic_numbers=torch.full((count,), 6, dtype=torch.long, device=device),
        cells=torch.eye(3, dtype=dtype, device=device).repeat(count, 1, 1) * 4.0,
        origins=torch.zeros((count, 3), dtype=dtype, device=device),
        pbc=torch.ones((count, 3), dtype=torch.bool, device=device),
        atom_ptr=ptr,
        atom_batch=torch.arange(count, dtype=torch.long, device=device),
        energy=torch.arange(count, dtype=dtype, device=device) + energy_shift,
        energy_mask=torch.ones(count, dtype=torch.bool, device=device),
        forces=torch.zeros((count, 3), dtype=dtype, device=device),
        force_mask=torch.ones((count, 3), dtype=torch.bool, device=device),
        stress=torch.zeros((count, 3, 3), dtype=dtype, device=device),
        stress_mask=torch.ones((count, 3, 3), dtype=torch.bool, device=device),
        force_present=torch.ones(count, dtype=torch.bool, device=device),
        stress_present=torch.ones(count, dtype=torch.bool, device=device),
        force_mask_provided=torch.ones(count, dtype=torch.bool, device=device),
        stress_mask_provided=torch.ones(count, dtype=torch.bool, device=device),
    )


def _live_state(device="cpu"):
    model = TinyModel(device)
    optimizer_config = OptimizerConfig(learning_rate=0.01, weight_decay=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config.learning_rate, weight_decay=0.0
    )
    optimizer.zero_grad(set_to_none=True)
    model.weight.square().sum().backward()
    optimizer.step()
    scheduler_config = SchedulerConfig(kind="reduce_on_plateau", patience=0)
    scheduler = build_scheduler(optimizer, scheduler_config)
    scheduler.step(1.0)
    selection = ModelSelectionState(
        best_metric=1.0,
        best_epoch=1,
        best_global_step=3,
        validation_events=2,
        last_validation_epoch=1,
        last_validation_global_step=3,
    )
    progress = FitProgress(
        next_epoch=2,
        global_step=3,
        completed_epochs=2,
        last_completed_epoch=1,
        best_epoch=1,
        best_global_step=3,
    )
    return model, optimizer, scheduler, scheduler_config, optimizer_config, selection, progress


def _capture(*, device="cpu", train_batches=None, validation_batches=None):
    model, optimizer, scheduler, scheduler_config, optimizer_config, selection, progress = _live_state(device)
    train_batches = train_batches or (_batch(("train",), device=device),)
    validation_batches = validation_batches or (_batch(("validation",), device=device),)
    checkpoint = capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
        selection,
        progress,
        train_batches,
        validation_batches,
        model_config={"hidden_irreps": "2x0e+2x1o", "dtype": torch.float64},
        loss_config=LossConfig(),
        optimizer_config=optimizer_config,
        train_step_config=TrainStepConfig(),
        validation_step_config=ValidationStepConfig(),
        scheduler_config=scheduler_config,
        model_selection_config=ModelSelectionConfig(),
        fit_config=FitConfig(3),
        species_vocabulary=(6,),
        baseline_fit_metadata={"kind": "explicit", "values": [0.25]},
        source_git_commit="abc123",
    )
    return model, optimizer, scheduler, checkpoint


def _walk_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensors(item)


def _assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_tree_equal(left, right)
    else:
        assert first == second


def _rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
        tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    )


def _assert_rng_equal(first, second):
    assert first[0] == second[0]
    assert first[1][0] == second[1][0]
    assert np.array_equal(first[1][1], second[1][1])
    assert first[1][2:] == second[1][2:]
    assert torch.equal(first[2], second[2])
    assert len(first[3]) == len(second[3])
    for left, right in zip(first[3], second[3]):
        assert torch.equal(left, right)


def test_complete_capture_and_owned_cpu_detached_snapshot():
    model, optimizer, scheduler, checkpoint = _capture()
    payload = checkpoint.to_dict()
    assert set(payload) == {
        "schema_version", "checkpoint_scope", "model_state_dict",
        "optimizer_state_dict", "scheduler_state_dict", "selection_state",
        "progress", "fit_history", "metadata", "python_rng_state",
        "numpy_rng_state", "torch_cpu_rng_state", "cuda_rng_states",
        "cuda_device_count",
    }
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["checkpoint_scope"] == "epoch_boundary"
    tensors = tuple(_walk_tensors(payload))
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in tensors)
    assert checkpoint.model_state_dict["weight"].data_ptr() != model.weight.data_ptr()
    assert checkpoint.metadata.template_fingerprints == {"template": "2" * 64}
    assert checkpoint.metadata.unit_conventions["voigt_order"] == [
        "xx", "yy", "zz", "yz", "xz", "xy"
    ]
    assert checkpoint.metadata.resolved_configuration.keys() == {
        "model", "loss", "optimizer", "train_step", "validation_step",
        "scheduler", "model_selection", "fit",
    }


def test_snapshot_does_not_follow_later_model_or_optimizer_updates():
    model, optimizer, _, checkpoint = _capture()
    model_before = checkpoint.model_state_dict["weight"].clone()
    optimizer_before = copy.deepcopy(checkpoint.optimizer_state_dict)
    optimizer.zero_grad(set_to_none=True)
    (model.weight * 3.0).sum().backward()
    optimizer.step()
    assert torch.equal(checkpoint.model_state_dict["weight"], model_before)
    _assert_tree_equal(checkpoint.optimizer_state_dict, optimizer_before)


def test_semantic_save_load_round_trip_weights_only_and_safe_globals(tmp_path):
    _, _, _, checkpoint = _capture()
    path = tmp_path / "state.pt"
    safe_before = list(torch.serialization.get_safe_globals())
    save_training_checkpoint(checkpoint, path)
    loaded = load_training_checkpoint(path)
    assert list(torch.serialization.get_safe_globals()) == safe_before
    _assert_tree_equal(loaded.model_state_dict, checkpoint.model_state_dict)
    _assert_tree_equal(loaded.optimizer_state_dict, checkpoint.optimizer_state_dict)
    _assert_tree_equal(loaded.scheduler_state_dict, checkpoint.scheduler_state_dict)
    assert loaded.selection_state == checkpoint.selection_state
    assert loaded.progress == checkpoint.progress
    assert loaded.metadata.to_dict() == checkpoint.metadata.to_dict()
    assert loaded.python_rng_state == checkpoint.python_rng_state
    assert loaded.numpy_rng_state == checkpoint.numpy_rng_state
    assert torch.equal(loaded.torch_cpu_rng_state, checkpoint.torch_cpu_rng_state)
    for left, right in zip(loaded.cuda_rng_states, checkpoint.cuda_rng_states):
        assert torch.equal(left, right)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(raw, dict) and raw["schema_version"] == CHECKPOINT_SCHEMA_VERSION


def test_capture_save_and_load_do_not_change_global_rng(tmp_path):
    before = _rng_snapshot()
    _, _, _, checkpoint = _capture()
    after_capture = _rng_snapshot()
    path = tmp_path / "rng.pt"
    save_training_checkpoint(checkpoint, path)
    after_save = _rng_snapshot()
    load_training_checkpoint(path)
    after_load = _rng_snapshot()
    _assert_rng_equal(before, after_capture)
    _assert_rng_equal(before, after_save)
    _assert_rng_equal(before, after_load)


def test_checkpoint_capture_rejects_foreign_optimizer_before_rng_or_state_capture():
    (
        model,
        _optimizer,
        _scheduler,
        scheduler_config,
        optimizer_config,
        selection,
        progress,
    ) = _live_state()
    foreign_model = TinyModel()
    foreign_optimizer = torch.optim.AdamW(
        foreign_model.parameters(), lr=optimizer_config.learning_rate
    )
    foreign_scheduler = build_scheduler(foreign_optimizer, scheduler_config)
    before = _rng_snapshot()
    model_before = copy.deepcopy(model.state_dict())
    gradient_before = model.weight.grad.clone()
    with pytest.raises(ValueError, match="optimizer parameters"):
        capture_training_checkpoint(
            model,
            foreign_optimizer,
            foreign_scheduler,
            selection,
            progress,
            (_batch(("train",)),),
            (_batch(("validation",)),),
            model_config={"kind": "tiny"},
            loss_config=LossConfig(),
            optimizer_config=optimizer_config,
            train_step_config=TrainStepConfig(),
            validation_step_config=ValidationStepConfig(),
            scheduler_config=scheduler_config,
            model_selection_config=ModelSelectionConfig(),
            fit_config=FitConfig(3),
            species_vocabulary=(6,),
        )
    _assert_rng_equal(before, _rng_snapshot())
    _assert_tree_equal(model.state_dict(), model_before)
    assert torch.equal(model.weight.grad, gradient_before)
    assert foreign_optimizer.state == {}


def test_python_numpy_torch_and_cuda_rng_payload_shape():
    _, _, _, checkpoint = _capture()
    assert isinstance(checkpoint.python_rng_state, list)
    assert set(checkpoint.numpy_rng_state) == {
        "bit_generator", "state", "position", "has_gauss", "cached_gaussian"
    }
    assert isinstance(checkpoint.numpy_rng_state["state"], list)
    assert checkpoint.torch_cpu_rng_state.dtype == torch.uint8
    assert checkpoint.cuda_device_count == len(checkpoint.cuda_rng_states)
    assert all(state.dtype == torch.uint8 for state in checkpoint.cuda_rng_states)


def test_fingerprint_device_independence_when_cuda_available():
    cpu = _batch(("a", "b"), dtype=torch.float64)
    cpu_value = fingerprint_batch_sequence((cpu,), split_name="train")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cuda = cpu.to(device="cuda", dtype=torch.float64)
    assert fingerprint_batch_sequence((cuda,), split_name="train") == cpu_value


def test_fingerprint_changes_for_all_semantic_inputs():
    base = _batch(("a", "b"))
    baseline = fingerprint_batch_sequence((base,), split_name="train")
    variants = [
        replace(base, positions=base.positions + 1.0e-3),
        replace(base, atomic_numbers=torch.tensor([6, 8], dtype=torch.long)),
        replace(base, energy=base.energy + 1.0),
        replace(
            base,
            energy=torch.zeros_like(base.energy),
            energy_mask=torch.tensor([True, False]),
        ),
        replace(base, template_fingerprints=("3" * 64,) * 2),
        base.to(dtype=torch.float32),
    ]
    for variant in variants:
        assert fingerprint_batch_sequence((variant,), split_name="train") != baseline
    first = _batch(("a",))
    second = replace(
        _batch(("b",)),
        positions=base.positions[1:2].clone(),
        energy=base.energy[1:2].clone(),
    )
    ordered = fingerprint_batch_sequence((first, second), split_name="train")
    reversed_order = fingerprint_batch_sequence((second, first), split_name="train")
    combined = fingerprint_batch_sequence((base,), split_name="train")
    assert ordered != reversed_order
    assert ordered != combined
    assert fingerprint_batch_sequence((base,), split_name="validation") != baseline


def test_fit_progress_rejects_mid_epoch_and_round_trips():
    progress = FitProgress(2, 4, 2, 0, 1, False, 1, 4)
    assert FitProgress.from_dict(progress.to_dict()) == progress
    with pytest.raises(ValueError, match="next_batch_index=0"):
        FitProgress(2, 4, 2, 1, 1)
    with pytest.raises(ValueError, match="next_epoch"):
        FitProgress(3, 4, 2, 0, 1)


def test_atomic_save_overwrite_contract(tmp_path):
    _, _, _, checkpoint = _capture()
    path = tmp_path / "atomic.pt"
    save_training_checkpoint(checkpoint, path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_training_checkpoint(checkpoint, path)
    assert path.read_bytes() == original
    save_training_checkpoint(checkpoint, path, overwrite=True)
    assert load_training_checkpoint(path).progress == checkpoint.progress
    assert not list(tmp_path.glob(".atomic.pt.*.tmp"))


def test_torch_save_failure_preserves_target_and_cleans_temp(tmp_path, monkeypatch):
    _, _, _, checkpoint = _capture()
    path = tmp_path / "save-failure.pt"
    path.write_bytes(b"original")
    monkeypatch.setattr(
        checkpoint_module.torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("save failed")),
    )
    with pytest.raises(OSError, match="save failed"):
        save_training_checkpoint(checkpoint, path, overwrite=True)
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".save-failure.pt.*.tmp"))


def test_replace_failure_preserves_target_and_cleans_temp(tmp_path, monkeypatch):
    _, _, _, checkpoint = _capture()
    path = tmp_path / "replace-failure.pt"
    path.write_bytes(b"original")
    monkeypatch.setattr(
        checkpoint_module.os,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        save_training_checkpoint(checkpoint, path, overwrite=True)
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".replace-failure.pt.*.tmp"))


def test_no_overwrite_race_never_clobbers_competing_checkpoint(
    tmp_path, monkeypatch
):
    _, _, _, checkpoint = _capture()
    path = tmp_path / "raced.pt"

    def competing_link(source, target, *args, **kwargs):
        del source, args, kwargs
        target_path = type(path)(target)
        target_path.write_bytes(b"competitor")
        raise FileExistsError(f"competing checkpoint won: {target_path}")

    monkeypatch.setattr(checkpoint_module.os, "link", competing_link)
    with pytest.raises(FileExistsError, match="competing checkpoint"):
        save_training_checkpoint(checkpoint, path, overwrite=False)
    assert path.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".raced.pt.*.tmp"))


def test_corrupt_wrong_schema_scope_and_missing_key_fail_fast(tmp_path):
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="safely load"):
        load_training_checkpoint(corrupt)
    _, _, _, checkpoint = _capture()
    payload = checkpoint.to_dict()
    for name, mutation in (
        ("schema", {**payload, "schema_version": "future"}),
        ("scope", {**payload, "checkpoint_scope": "mid_epoch"}),
        ("missing", {key: value for key, value in payload.items() if key != "progress"}),
    ):
        path = tmp_path / f"{name}.pt"
        torch.save(mutation, path)
        with pytest.raises(ValueError, match="invalid training checkpoint"):
            load_training_checkpoint(path)


def test_load_does_not_mutate_live_training_objects(tmp_path):
    model, optimizer, scheduler, checkpoint = _capture()
    path = tmp_path / "no-apply.pt"
    save_training_checkpoint(checkpoint, path)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    load_training_checkpoint(path)
    _assert_tree_equal(model.state_dict(), model_before)
    _assert_tree_equal(optimizer.state_dict(), optimizer_before)
    _assert_tree_equal(scheduler.state_dict(), scheduler_before)


def test_metadata_plain_serialization_round_trip():
    _, _, _, checkpoint = _capture()
    restored = TrainingCheckpoint.from_dict(checkpoint.to_dict())
    assert restored.metadata.to_dict() == checkpoint.metadata.to_dict()
    assert restored.metadata.resolved_configuration["model"]["dtype"] == "torch.float64"
    assert restored.metadata.baseline_fit_metadata == {
        "kind": "explicit", "values": [0.25]
    }
