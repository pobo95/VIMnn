from __future__ import annotations

import copy
from dataclasses import replace
import importlib
import random

import numpy as np
import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import (
    CheckpointManager,
    CheckpointManagerConfig,
    CheckpointManagerError,
    EpochResult,
    EpochTermMetrics,
    FitConfig,
    FitEpochRecord,
    FitProgress,
    LossConfig,
    ManagedCheckpointResult,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationDecision,
    ValidationStepConfig,
    build_scheduler,
    capture_training_checkpoint,
    save_training_checkpoint,
    validate_checkpoint_history,
)


manager_module = importlib.import_module("refsite_mlip.training.checkpoint_manager")


class TinyModel(torch.nn.Module):
    def __init__(self, *, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([1.0, -0.25], dtype=dtype, device=device)
        )
        self.register_buffer(
            "atomic_baseline", torch.tensor([0.5], dtype=dtype, device=device)
        )


def _batch(*, dtype=torch.float64, device="cpu"):
    return StructureBatch(
        sample_ids=("sample",),
        template_ids=("template",),
        template_fingerprints=("2" * 64,),
        positions=torch.tensor([[0.1, 0.2, 0.3]], dtype=dtype, device=device),
        atomic_numbers=torch.tensor([6], dtype=torch.long, device=device),
        cells=torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3) * 4.0,
        origins=torch.zeros((1, 3), dtype=dtype, device=device),
        pbc=torch.ones((1, 3), dtype=torch.bool, device=device),
        atom_ptr=torch.tensor([0, 1], dtype=torch.long, device=device),
        atom_batch=torch.zeros(1, dtype=torch.long, device=device),
        energy=torch.zeros(1, dtype=dtype, device=device),
        energy_mask=torch.ones(1, dtype=torch.bool, device=device),
        forces=torch.zeros((1, 3), dtype=dtype, device=device),
        force_mask=torch.zeros((1, 3), dtype=torch.bool, device=device),
        stress=torch.zeros((1, 3, 3), dtype=dtype, device=device),
        stress_mask=torch.zeros((1, 3, 3), dtype=torch.bool, device=device),
        force_present=torch.zeros(1, dtype=torch.bool, device=device),
        stress_present=torch.zeros(1, dtype=torch.bool, device=device),
        force_mask_provided=torch.zeros(1, dtype=torch.bool, device=device),
        stress_mask_provided=torch.zeros(1, dtype=torch.bool, device=device),
    )


def _term(value):
    return EpochTermMetrics(float(value), 1.0, float(value), 1)


def _epoch_result(phase, epoch, start, end, metric):
    return EpochResult(
        energy=_term(metric),
        force=EpochTermMetrics(0.0, 0.0, 0.0, 0),
        stress=EpochTermMetrics(0.0, 0.0, 0.0, 0),
        total_loss=float(metric),
        has_supervision=True,
        phase=phase,
        epoch_index=epoch,
        global_step_start=start,
        global_step_end=end,
        number_of_batches=1,
        number_of_supervised_batches=1,
        number_of_structures=1,
        number_of_atoms=1,
        successful_optimizer_steps=1 if phase == "train" else 0,
        ordered_batch_sample_ids=(("sample",),),
        metric_semantics=(
            "pre_update_batch_observations"
            if phase == "train"
            else "fixed_model_validation"
        ),
    )


def _records():
    metrics = (2.0, 3.0, 1.0)
    states = (
        ModelSelectionState(
            best_metric=2.0,
            best_epoch=0,
            best_global_step=1,
            validation_events=1,
            last_validation_epoch=0,
            last_validation_global_step=1,
        ),
        ModelSelectionState(
            best_metric=2.0,
            best_epoch=0,
            best_global_step=1,
            epochs_since_improvement=1,
            validation_events=2,
            last_validation_epoch=1,
            last_validation_global_step=2,
        ),
        ModelSelectionState(
            best_metric=1.0,
            best_epoch=2,
            best_global_step=3,
            validation_events=3,
            last_validation_epoch=2,
            last_validation_global_step=3,
        ),
    )
    records = []
    for epoch, (metric, state) in enumerate(zip(metrics, states)):
        is_best = epoch in (0, 2)
        decision = ValidationDecision(
            metric_name="total_loss",
            metric_value=metric,
            is_best=is_best,
            should_stop=False,
            best_metric=state.best_metric,
            best_epoch=state.best_epoch,
            best_global_step=state.best_global_step,
            epochs_since_improvement=state.epochs_since_improvement,
            validation_events=state.validation_events,
            learning_rates_before=(0.01,),
            learning_rates_after=(0.01,),
            scheduler_stepped=True,
            learning_rate_changed=False,
        )
        records.append(
            FitEpochRecord(
                epoch_index=epoch,
                training=_epoch_result("train", epoch, epoch, epoch + 1, metric),
                validation=_epoch_result(
                    "validation", epoch, epoch + 1, epoch + 1, metric
                ),
                decision=decision,
                selection_state_after_epoch=state,
                learning_rates_used_for_training=(0.01,),
                learning_rates_after_validation=(0.01,),
            )
        )
    return tuple(records)


def _checkpoint(
    count,
    *,
    dtype=torch.float64,
    device="cpu",
):
    records = _records()[:count]
    model = TinyModel(dtype=dtype, device=device)
    optimizer_config = OptimizerConfig(learning_rate=0.01, weight_decay=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    for _ in records:
        optimizer.zero_grad(set_to_none=True)
        model.weight.square().sum().backward()
        optimizer.step()
    scheduler_config = SchedulerConfig()
    scheduler = build_scheduler(optimizer, scheduler_config)
    for record in records:
        scheduler.step(record.decision.metric_value)
    last = records[-1]
    progress = FitProgress(
        next_epoch=count,
        global_step=count,
        completed_epochs=count,
        last_completed_epoch=count - 1,
        best_epoch=last.selection_state_after_epoch.best_epoch,
        best_global_step=last.selection_state_after_epoch.best_global_step,
    )
    batch = _batch(dtype=dtype, device=device)
    checkpoint = capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
        last.selection_state_after_epoch,
        progress,
        (batch,),
        (batch,),
        model_config={"kind": "tiny"},
        loss_config=LossConfig(),
        optimizer_config=optimizer_config,
        train_step_config=TrainStepConfig(),
        validation_step_config=ValidationStepConfig(),
        scheduler_config=scheduler_config,
        model_selection_config=ModelSelectionConfig(),
        fit_config=FitConfig(count),
        species_vocabulary=(6,),
        fit_history=records,
    )
    return checkpoint, last


def _tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first.detach().cpu(), second.detach().cpu())
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _tree_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _tree_equal(a, b) for a, b in zip(first, second)
        )
    return first == second


def _assert_checkpoint_equal(first, second):
    assert _tree_equal(first.to_dict(), second.to_dict())


def _rng():
    numpy = np.random.get_state()
    return (
        random.getstate(),
        (numpy[0], numpy[1].copy(), *numpy[2:]),
        torch.get_rng_state().clone(),
        tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    )


def _assert_rng_equal(first, second):
    assert first[0] == second[0]
    assert first[1][0] == second[1][0]
    assert np.array_equal(first[1][1], second[1][1])
    assert first[1][2:] == second[1][2:]
    assert torch.equal(first[2], second[2])
    assert len(first[3]) == len(second[3])
    assert all(torch.equal(a, b) for a, b in zip(first[3], second[3]))


def test_first_best_then_nonbest_then_best_file_semantics(tmp_path):
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    checkpoint0, record0 = _checkpoint(1)
    result0 = manager.save_epoch(checkpoint0, record0)
    assert result0.completed_stages == ("epoch", "latest", "best")
    assert result0.epoch_written and result0.latest_written and result0.best_written
    best0 = (manager.root / "best.pt").read_bytes()

    checkpoint1, record1 = _checkpoint(2)
    result1 = manager.save_epoch(checkpoint1, record1)
    assert result1.completed_stages == ("epoch", "latest")
    assert result1.best_path is None and not result1.best_written
    assert (manager.root / "best.pt").read_bytes() == best0
    _assert_checkpoint_equal(manager.load_latest(), checkpoint1)
    _assert_checkpoint_equal(manager.load_best(), checkpoint0)

    checkpoint2, record2 = _checkpoint(3)
    result2 = manager.save_epoch(checkpoint2, record2)
    assert result2.best_written
    _assert_checkpoint_equal(manager.load_latest(), checkpoint2)
    _assert_checkpoint_equal(manager.load_best(), checkpoint2)
    assert manager.list_epochs() == (0, 1, 2)
    for epoch, checkpoint in enumerate((checkpoint0, checkpoint1, checkpoint2)):
        _assert_checkpoint_equal(manager.load_epoch(epoch), checkpoint)


def test_epoch_overwrite_and_missing_loads_fail_fast(tmp_path):
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    with pytest.raises(FileNotFoundError, match="latest"):
        manager.load_latest()
    with pytest.raises(FileNotFoundError, match="best"):
        manager.load_best()
    checkpoint, record = _checkpoint(1)
    manager.save_epoch(checkpoint, record)
    with pytest.raises(CheckpointManagerError, match="stage='epoch'") as caught:
        manager.save_epoch(checkpoint, record)
    assert caught.value.completed_stages == ()
    assert not caught.value.orphan_epoch_snapshot


def test_list_strict_filtering_sorting_and_config_result_roundtrip(tmp_path):
    config = CheckpointManagerConfig(tmp_path / "managed", epoch_filename_width=6)
    assert CheckpointManagerConfig.from_dict(config.to_dict()) == config
    manager = CheckpointManager(config)
    checkpoint2, record2 = _checkpoint(3)
    checkpoint0, record0 = _checkpoint(1)
    manager.save_epoch(checkpoint2, record2)
    manager.save_epoch(checkpoint0, record0)
    for name in (
        "epoch_1.pt",
        "epoch_000001.pt.bak",
        "epoch_abcdef.pt",
        "other.pt",
    ):
        (manager.root / name).write_text("unrelated")
    assert manager.list_epochs() == (0, 2)
    result = ManagedCheckpointResult(
        2,
        3,
        True,
        str(manager.root / "epoch_000002.pt"),
        str(manager.root / "latest.pt"),
        str(manager.root / "best.pt"),
        True,
        True,
        True,
        ("epoch", "latest", "best"),
    )
    assert ManagedCheckpointResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("kind", ["progress", "selection", "history"])
def test_invalid_input_preflight_does_not_create_or_modify_directory(tmp_path, kind):
    root = tmp_path / "never-created"
    manager = CheckpointManager(CheckpointManagerConfig(root))
    checkpoint, record = _checkpoint(1)
    if kind == "progress":
        checkpoint = replace(
            checkpoint,
            progress=replace(checkpoint.progress, global_step=9),
        )
    elif kind == "selection":
        record = replace(
            record,
            selection_state_after_epoch=ModelSelectionState(),
        )
    else:
        checkpoint = replace(checkpoint, fit_history=None)
    with pytest.raises((TypeError, ValueError)):
        manager.save_epoch(checkpoint, record)
    assert not root.exists()


@pytest.mark.parametrize("failure_stage", ["epoch", "latest", "best"])
def test_multi_file_failure_contract_preserves_existing_files(
    tmp_path, monkeypatch, failure_stage
):
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    checkpoint0, record0 = _checkpoint(1)
    manager.save_epoch(checkpoint0, record0)
    latest_before = (manager.root / "latest.pt").read_bytes()
    best_before = (manager.root / "best.pt").read_bytes()
    checkpoint = _checkpoint(2)[0] if failure_stage != "best" else _checkpoint(3)[0]
    record = _checkpoint(2)[1] if failure_stage != "best" else _checkpoint(3)[1]
    original = manager_module.save_training_checkpoint

    def injected(value, path, *, overwrite=False):
        name = path.name
        if (
            (failure_stage == "epoch" and name.startswith("epoch_"))
            or (failure_stage == "latest" and name == "latest.pt")
            or (failure_stage == "best" and name == "best.pt")
        ):
            raise OSError(f"injected {failure_stage} failure")
        return original(value, path, overwrite=overwrite)

    monkeypatch.setattr(manager_module, "save_training_checkpoint", injected)
    with pytest.raises(CheckpointManagerError) as caught:
        manager.save_epoch(checkpoint, record)
    error = caught.value
    assert error.stage == failure_stage
    expected = {
        "epoch": (),
        "latest": ("epoch",),
        "best": ("epoch", "latest"),
    }[failure_stage]
    assert error.completed_stages == expected
    assert error.orphan_epoch_snapshot == (failure_stage != "epoch")
    if failure_stage in ("epoch", "latest"):
        assert (manager.root / "latest.pt").read_bytes() == latest_before
    assert (manager.root / "best.pt").read_bytes() == best_before
    _assert_checkpoint_equal(manager.load_best(), checkpoint0)
    if failure_stage != "best":
        _assert_checkpoint_equal(manager.load_latest(), checkpoint0)


def test_symlink_outside_root_and_existing_file_root_rejected(tmp_path):
    outside_checkpoint = tmp_path / "outside.pt"
    checkpoint, record = _checkpoint(1)
    save_training_checkpoint(checkpoint, outside_checkpoint)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    manager.root.mkdir()
    (manager.root / "latest.pt").symlink_to(outside_checkpoint)
    with pytest.raises(ValueError, match="symlink"):
        manager.load_latest()
    root_file = tmp_path / "root-file"
    root_file.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        CheckpointManager(CheckpointManagerConfig(root_file))


def test_save_load_preserve_rng_and_inputs_and_latest_history_compatibility(tmp_path):
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    checkpoint, record = _checkpoint(2)
    checkpoint_before = copy.deepcopy(checkpoint.to_dict())
    record_before = record.to_dict()
    rng = _rng()
    manager.save_epoch(checkpoint, record)
    latest = manager.load_latest()
    _assert_rng_equal(rng, _rng())
    assert _tree_equal(checkpoint_before, checkpoint.to_dict())
    assert record_before == record.to_dict()
    assert tuple(item.epoch_index for item in validate_checkpoint_history(latest)) == (
        0,
        1,
    )


def test_epoch_snapshots_can_be_disabled_without_affecting_latest(tmp_path):
    manager = CheckpointManager(
        CheckpointManagerConfig(
            tmp_path / "managed", save_epoch_snapshots=False
        )
    )
    checkpoint, record = _checkpoint(1)
    result = manager.save_epoch(checkpoint, record)
    assert result.epoch_path is None and not result.epoch_written
    assert result.completed_stages == ("latest", "best")
    assert manager.list_epochs() == ()
    _assert_checkpoint_equal(manager.load_latest(), checkpoint)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_checkpoint_state_save_load_smoke_when_available(tmp_path, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / str(dtype)))
    checkpoint, record = _checkpoint(1, dtype=dtype, device="cuda")
    result = manager.save_epoch(checkpoint, record)
    loaded = manager.load_latest()
    assert result.best_written
    assert loaded.cuda_rng_states
    assert loaded.cuda_device_count == torch.cuda.device_count()
    assert any(
        isinstance(value, torch.Tensor)
        for state in loaded.optimizer_state_dict["state"].values()
        for value in state.values()
    )

