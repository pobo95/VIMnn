from __future__ import annotations

import copy
from dataclasses import fields, replace
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
    CheckpointedFitConfig,
    CheckpointedFitExecutionError,
    CheckpointedFitResult,
    EpochResult,
    EpochTermMetrics,
    FitConfig,
    FitExecutionError,
    FitProgress,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_scheduler,
    capture_training_checkpoint,
    run_checkpointed_fit,
    run_fit,
    validate_checkpoint_history,
)


module = importlib.import_module("refsite_mlip.training.checkpointed_fit")
fit_module = importlib.import_module("refsite_mlip.training.fit")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer(
            "atomic_baseline", torch.tensor([0.25], dtype=torch.float64)
        )


def _batch(sample_id="sample", *, labeled=True, fingerprint="2" * 64):
    dtype = torch.float64
    return StructureBatch(
        sample_ids=(sample_id,),
        template_ids=("template",),
        template_fingerprints=(fingerprint,),
        positions=torch.zeros((1, 3), dtype=dtype),
        atomic_numbers=torch.tensor([6], dtype=torch.long),
        cells=torch.eye(3, dtype=dtype).unsqueeze(0),
        origins=torch.zeros((1, 3), dtype=dtype),
        pbc=torch.ones((1, 3), dtype=torch.bool),
        atom_ptr=torch.tensor([0, 1], dtype=torch.long),
        atom_batch=torch.zeros(1, dtype=torch.long),
        energy=torch.zeros(1, dtype=dtype),
        energy_mask=torch.tensor([labeled]),
        forces=torch.zeros((1, 3), dtype=dtype),
        force_mask=torch.zeros((1, 3), dtype=torch.bool),
        stress=torch.zeros((1, 3, 3), dtype=dtype),
        stress_mask=torch.zeros((1, 3, 3), dtype=torch.bool),
        force_present=torch.zeros(1, dtype=torch.bool),
        stress_present=torch.zeros(1, dtype=torch.bool),
        force_mask_provided=torch.zeros(1, dtype=torch.bool),
        stress_mask_provided=torch.zeros(1, dtype=torch.bool),
    )


def _term(value):
    return EpochTermMetrics(float(value), 1.0, float(value), 1)


def _epoch(phase, epoch_index, global_step, metric, batches):
    training = phase == "train"
    end = global_step + len(batches) if training else global_step
    return EpochResult(
        energy=_term(metric),
        force=EpochTermMetrics(0.0, 0.0, 0.0, 0),
        stress=EpochTermMetrics(0.0, 0.0, 0.0, 0),
        total_loss=float(metric),
        has_supervision=True,
        phase=phase,
        epoch_index=epoch_index,
        global_step_start=global_step,
        global_step_end=end,
        number_of_batches=len(batches),
        number_of_supervised_batches=len(batches),
        number_of_structures=sum(batch.num_structures for batch in batches),
        number_of_atoms=sum(batch.num_atoms for batch in batches),
        successful_optimizer_steps=len(batches) if training else 0,
        ordered_batch_sample_ids=tuple(batch.sample_ids for batch in batches),
        metric_semantics=(
            "pre_update_batch_observations"
            if training
            else "fixed_model_validation"
        ),
    )


def _configs(max_epochs=3, *, scheduler=None, selection=None):
    return {
        "model": {"kind": "tiny"},
        "loss": LossConfig(),
        "optimizer": OptimizerConfig(learning_rate=0.1, weight_decay=0.0),
        "train_step": TrainStepConfig(),
        "validation_step": ValidationStepConfig(),
        "scheduler": scheduler or SchedulerConfig(),
        "model_selection": selection or ModelSelectionConfig(),
        "fit": FitConfig(max_epochs),
    }


def _live(configs):
    model = TinyModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configs["optimizer"].learning_rate,
        betas=configs["optimizer"].betas,
        eps=configs["optimizer"].eps,
        weight_decay=configs["optimizer"].weight_decay,
        amsgrad=configs["optimizer"].amsgrad,
    )
    scheduler = build_scheduler(optimizer, configs["scheduler"])
    return model, optimizer, scheduler


def _metadata(model, optimizer, scheduler, batch, configs):
    return capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
        ModelSelectionState(),
        FitProgress(next_epoch=0, global_step=0, completed_epochs=0),
        (batch,),
        (batch,),
        model_config=configs["model"],
        loss_config=configs["loss"],
        optimizer_config=configs["optimizer"],
        train_step_config=configs["train_step"],
        validation_step_config=configs["validation_step"],
        scheduler_config=configs["scheduler"],
        model_selection_config=configs["model_selection"],
        fit_config=configs["fit"],
        species_vocabulary=(6,),
        fit_history=(),
    ).metadata


def _install(monkeypatch, metrics, *, target=module, fail=None, snapshots=None):
    metric_iterator = iter(metrics)

    def training(model, optimizer, batches, *args, epoch_index, global_step_start):
        if fail == ("train", epoch_index):
            raise RuntimeError("broken train")
        for _ in batches:
            optimizer.zero_grad(set_to_none=True)
            model.weight.square().backward()
            optimizer.step()
        if snapshots is not None:
            snapshots.append(copy.deepcopy(model.state_dict()))
        return _epoch("train", epoch_index, global_step_start, 1.0, batches)

    def validation(model, batches, *args, epoch_index, global_step):
        if fail == ("validation", epoch_index):
            raise RuntimeError("broken validation")
        return _epoch(
            "validation", epoch_index, global_step, next(metric_iterator), batches
        )

    monkeypatch.setattr(target, "run_training_epoch", training)
    monkeypatch.setattr(target, "run_validation_epoch", validation)
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)


def _run(monkeypatch, tmp_path, metrics=(1.0, 2.0, 3.0), *, configs=None):
    configs = configs or _configs(len(metrics))
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    _install(monkeypatch, metrics)
    result = run_checkpointed_fit(
        model,
        optimizer,
        scheduler,
        (batch,),
        (batch,),
        {},
        configs["loss"],
        configs["train_step"],
        configs["validation_step"],
        configs["scheduler"],
        configs["model_selection"],
        ModelSelectionState(),
        configs["fit"],
        manager,
        metadata,
    )
    return model, optimizer, scheduler, manager, result, configs, batch


def _tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _tree_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _tree_equal(left, right) for left, right in zip(first, second)
        )
    return first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"save_every_epoch": False},
        {"save_every_epoch": 1},
        {"require_empty_manager": 1},
    ],
)
def test_checkpointed_config_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        CheckpointedFitConfig(**kwargs)
    config = CheckpointedFitConfig()
    assert CheckpointedFitConfig.from_dict(config.to_dict()) == config


def test_every_epoch_latest_best_history_progress_and_serialization(
    monkeypatch, tmp_path
):
    _, _, scheduler, manager, result, _, _ = _run(monkeypatch, tmp_path)
    assert manager.list_epochs() == (0, 1, 2)
    assert result.epoch_paths == tuple(
        str(manager.root / f"epoch_{epoch:06d}.pt") for epoch in range(3)
    )
    assert result.latest_path == str(manager.root / "latest.pt")
    assert result.best_path == str(manager.root / "best.pt")
    latest = manager.load_latest()
    best = manager.load_best()
    assert latest.progress.next_epoch == 3 and latest.progress.global_step == 3
    assert best.progress.next_epoch == 1 and best.progress.best_epoch == 0
    assert tuple(r.epoch_index for r in validate_checkpoint_history(latest)) == (0, 1, 2)
    assert scheduler.state_dict() == {"validation_steps": 3}
    assert CheckpointedFitResult.from_dict(result.to_dict()) == result
    assert all(
        not isinstance(getattr(result, field.name), torch.Tensor)
        for field in fields(result)
    )


def test_best_contains_actual_best_epoch_weights_and_survives_nonbest(
    monkeypatch, tmp_path
):
    snapshots = []
    configs = _configs(3)
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    _install(monkeypatch, (1.0, 2.0, 3.0), snapshots=snapshots)
    run_checkpointed_fit(
        model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
        configs["train_step"], configs["validation_step"], configs["scheduler"],
        configs["model_selection"], ModelSelectionState(), configs["fit"], manager,
        metadata,
    )
    assert _tree_equal(manager.load_best().model_state_dict, snapshots[0])
    assert not _tree_equal(manager.load_latest().model_state_dict, snapshots[0])


def test_early_stop_epoch_is_checkpointed_before_termination(monkeypatch, tmp_path):
    selection = ModelSelectionConfig(early_stopping_patience=1)
    configs = _configs(4, selection=selection)
    _, _, _, manager, result, _, _ = _run(
        monkeypatch, tmp_path, (1.0, 2.0, 3.0, 4.0), configs=configs
    )
    assert result.fit_result.stopped_early
    assert result.fit_result.stop_epoch == 1
    assert result.epochs_checkpointed == 2
    latest = manager.load_latest()
    assert latest.progress.stopped_early and latest.progress.next_epoch == 2


def test_plateau_scheduler_state_is_captured_for_next_epoch(monkeypatch, tmp_path):
    scheduler_config = SchedulerConfig(
        kind="reduce_on_plateau", patience=0, factor=0.5, threshold=0.0
    )
    configs = _configs(3, scheduler=scheduler_config)
    _, _, _, manager, result, _, _ = _run(
        monkeypatch, tmp_path, (1.0, 2.0, 3.0), configs=configs
    )
    assert [r.learning_rates_used_for_training for r in result.fit_result.records] == [
        (0.1,), (0.1,), (0.05,)
    ]
    assert manager.load_epoch(1).optimizer_state_dict["param_groups"][0]["lr"] == 0.05


def test_existing_manager_preflight_is_transactional(monkeypatch, tmp_path):
    configs = _configs(1)
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    manager.root.mkdir()
    existing = manager.root / "latest.pt"
    existing.write_bytes(b"sentinel")
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    with pytest.raises(FileExistsError, match="must be empty"):
        run_checkpointed_fit(
            model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
            configs["train_step"], configs["validation_step"], configs["scheduler"],
            configs["model_selection"], ModelSelectionState(), configs["fit"],
            manager, metadata,
        )
    assert _tree_equal(model.state_dict(), before)
    assert optimizer.state == {} and scheduler.state_dict() == {"validation_steps": 0}
    assert torch.equal(torch.get_rng_state(), rng) and existing.read_bytes() == b"sentinel"


@pytest.mark.parametrize("kind", ["manifest", "template", "config"])
def test_static_metadata_mismatch_is_preflight_and_file_free(
    monkeypatch, tmp_path, kind
):
    configs = _configs(1)
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    if kind == "manifest":
        metadata = replace(
            metadata,
            training_data=replace(metadata.training_data, fingerprint="3" * 64),
        )
    elif kind == "template":
        metadata = replace(metadata, template_fingerprints={"template": "3" * 64})
    else:
        resolved = copy.deepcopy(metadata.resolved_configuration)
        resolved["loss"]["energy_scale"] = 2.0
        metadata = replace(metadata, resolved_configuration=resolved)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    with pytest.raises(ValueError):
        run_checkpointed_fit(
            model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
            configs["train_step"], configs["validation_step"], configs["scheduler"],
            configs["model_selection"], ModelSelectionState(), configs["fit"],
            manager, metadata,
        )
    assert not manager.root.exists() and optimizer.state == {}



def test_nonfresh_progress_and_filename_overflow_fail_before_updates(monkeypatch, tmp_path):
    configs = _configs(2)
    configs["fit"] = FitConfig(2, start_epoch=1, global_step_start=1)
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    state = ModelSelectionState(
        best_metric=1.0, best_epoch=0, best_global_step=1,
        validation_events=1, last_validation_epoch=0,
        last_validation_global_step=1,
    )
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "fresh"))
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="fresh progress"):
        run_checkpointed_fit(
            model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
            configs["train_step"], configs["validation_step"], configs["scheduler"],
            configs["model_selection"], state, configs["fit"], manager, metadata,
        )
    assert torch.equal(model.weight, before) and not manager.root.exists()

    configs = _configs(11)
    model, optimizer, scheduler = _live(configs)
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(
        CheckpointManagerConfig(tmp_path / "width", epoch_filename_width=1)
    )
    with pytest.raises(ValueError, match="filename width"):
        run_checkpointed_fit(
            model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
            configs["train_step"], configs["validation_step"], configs["scheduler"],
            configs["model_selection"], ModelSelectionState(), configs["fit"],
            manager, metadata,
        )
    assert optimizer.state == {} and not manager.root.exists()


def test_selection_failure_creates_no_checkpoint(monkeypatch, tmp_path):
    configs = _configs(1)
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    _install(monkeypatch, (1.0,))
    monkeypatch.setattr(
        module, "process_primary_validation",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("selection broke")),
    )
    with pytest.raises(FitExecutionError) as caught:
        run_checkpointed_fit(
            model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
            configs["train_step"], configs["validation_step"], configs["scheduler"],
            configs["model_selection"], ModelSelectionState(), configs["fit"],
            manager, metadata,
        )
    assert caught.value.phase == "selection"
    assert manager.list_epochs() == ()
    assert scheduler.state_dict() == {"validation_steps": 0}


@pytest.mark.parametrize("phase", ["train", "validation"])
def test_train_or_validation_failure_creates_no_current_checkpoint(
    monkeypatch, tmp_path, phase
):
    configs = _configs(2)
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    _install(monkeypatch, (1.0, 2.0), fail=(phase, 0))
    with pytest.raises(FitExecutionError) as caught:
        run_checkpointed_fit(
            model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
            configs["train_step"], configs["validation_step"], configs["scheduler"],
            configs["model_selection"], ModelSelectionState(), configs["fit"],
            manager, metadata,
        )
    assert caught.value.phase == phase
    assert manager.list_epochs() == ()
    assert not (manager.root / "latest.pt").exists()


def test_capture_failure_stops_before_manager_and_reports_retained_update(
    monkeypatch, tmp_path
):
    configs = _configs(2)
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    _install(monkeypatch, (1.0, 2.0))
    monkeypatch.setattr(
        module,
        "capture_training_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture broke")),
    )
    with pytest.raises(CheckpointedFitExecutionError) as caught:
        run_checkpointed_fit(
            model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
            configs["train_step"], configs["validation_step"], configs["scheduler"],
            configs["model_selection"], ModelSelectionState(), configs["fit"],
            manager, metadata,
        )
    error = caught.value
    assert error.failure_stage == "capture" and error.global_step == 1
    assert error.epochs_checkpointed == 0 and not error.rollback_performed
    assert manager.list_epochs() == ()
    assert float(next(iter(optimizer.state.values()))["step"]) == 1.0


def test_manager_partial_failure_metadata_and_previous_checkpoint_recovery(
    monkeypatch, tmp_path
):
    configs = _configs(2)
    model, optimizer, scheduler = _live(configs)
    batch = _batch()
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    _install(monkeypatch, (1.0, 2.0))
    original = manager.save_epoch

    def injected(checkpoint, record):
        if record.epoch_index == 1:
            raise CheckpointManagerError(
                stage="latest",
                epoch_index=1,
                completed_stages=("epoch",),
                epoch_path=manager.root / "epoch_000001.pt",
                cause=OSError("latest broke"),
            )
        return original(checkpoint, record)

    monkeypatch.setattr(manager, "save_epoch", injected)
    with pytest.raises(CheckpointedFitExecutionError) as caught:
        run_checkpointed_fit(
            model, optimizer, scheduler, (batch,), (batch,), {}, configs["loss"],
            configs["train_step"], configs["validation_step"], configs["scheduler"],
            configs["model_selection"], ModelSelectionState(), configs["fit"],
            manager, metadata,
        )
    error = caught.value
    assert error.failure_stage == "manager" and error.epoch_index == 1
    assert error.manager_completed_stages == ("epoch",)
    assert error.orphan_epoch_snapshot and error.epochs_checkpointed == 1
    assert manager.load_latest().progress.next_epoch == 1


def test_run_fit_and_checkpointed_fit_exact_trajectory_and_rng(
    monkeypatch, tmp_path
):
    configs = _configs(3)
    batch = _batch()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model_a, optimizer_a, scheduler_a = _live(configs)
    _install(monkeypatch, (1.0, 2.0, 3.0), target=fit_module)
    result_a = run_fit(
        model_a, optimizer_a, scheduler_a, (batch,), (batch,), {}, configs["loss"],
        configs["train_step"], configs["validation_step"], configs["scheduler"],
        configs["model_selection"], ModelSelectionState(), configs["fit"],
    )
    states_a = (
        copy.deepcopy(model_a.state_dict()), copy.deepcopy(optimizer_a.state_dict()),
        copy.deepcopy(scheduler_a.state_dict()), random.getstate(), np.random.get_state(),
        torch.get_rng_state().clone(),
    )

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model_b, optimizer_b, scheduler_b = _live(configs)
    metadata = _metadata(model_b, optimizer_b, scheduler_b, batch, configs)
    _install(monkeypatch, (1.0, 2.0, 3.0), target=module)
    result_b = run_checkpointed_fit(
        model_b, optimizer_b, scheduler_b, (batch,), (batch,), {}, configs["loss"],
        configs["train_step"], configs["validation_step"], configs["scheduler"],
        configs["model_selection"], ModelSelectionState(), configs["fit"],
        CheckpointManager(CheckpointManagerConfig(tmp_path / "managed")), metadata,
    )
    assert result_b.fit_result == result_a
    assert _tree_equal(model_b.state_dict(), states_a[0])
    assert _tree_equal(optimizer_b.state_dict(), states_a[1])
    assert _tree_equal(scheduler_b.state_dict(), states_a[2])
    assert random.getstate() == states_a[3]
    numpy_b = np.random.get_state()
    assert numpy_b[0] == states_a[4][0] and np.array_equal(numpy_b[1], states_a[4][1])
    assert numpy_b[2:] == states_a[4][2:]
    assert torch.equal(torch.get_rng_state(), states_a[5])

