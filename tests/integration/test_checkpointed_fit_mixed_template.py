from __future__ import annotations

import copy
from pathlib import Path
import runpy

import pytest
import torch

from refsite_mlip.training import (
    CheckpointManager,
    CheckpointManagerConfig,
    FitConfig,
    FitProgress,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    ResumePolicy,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_optimizer,
    build_scheduler,
    capture_training_checkpoint,
    restore_training_checkpoint_,
    run_checkpointed_fit,
    run_fit,
)


def _mixed_case(typed_crystal, *, dtype, device):
    path = Path(__file__).with_name("test_fit_controller_mixed_template.py")
    return runpy.run_path(str(path))["_mixed_case"](
        typed_crystal, dtype=dtype, device=device
    )


def _configs(model, epochs):
    return {
        "model": model.config,
        "loss": LossConfig(energy_weight=1.0),
        "optimizer": OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0),
        "train_step": TrainStepConfig(),
        "validation_step": ValidationStepConfig(),
        "scheduler": SchedulerConfig(),
        "model_selection": ModelSelectionConfig(),
        "fit": FitConfig(epochs),
    }


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
        species_vocabulary=model.config.species_vocabulary,
        fit_history=(),
    ).metadata


def _tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first.detach().cpu(), second.detach().cpu())
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _tree_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _tree_equal(left, right) for left, right in zip(first, second)
        )
    return first == second


def _run_fit(model, optimizer, scheduler, batch, contexts, configs):
    return run_fit(
        model,
        optimizer,
        scheduler,
        (batch,),
        (batch,),
        contexts,
        configs["loss"],
        configs["train_step"],
        configs["validation_step"],
        configs["scheduler"],
        configs["model_selection"],
        ModelSelectionState(),
        configs["fit"],
    )


def test_actual_mixed_template_two_epoch_checkpointed_fit_exact_trajectory(
    typed_crystal, tmp_path
):
    _, model_a, _, _, batch_a, contexts_a = _mixed_case(
        typed_crystal, dtype=torch.float64, device="cpu"
    )
    _, model_b, _, _, batch_b, contexts_b = _mixed_case(
        typed_crystal, dtype=torch.float64, device="cpu"
    )
    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()), strict=True)
    configs_a = _configs(model_a, 2)
    configs_b = _configs(model_b, 2)
    optimizer_a = build_optimizer(model_a, configs_a["optimizer"])
    optimizer_b = build_optimizer(model_b, configs_b["optimizer"])
    scheduler_a = build_scheduler(optimizer_a, configs_a["scheduler"])
    scheduler_b = build_scheduler(optimizer_b, configs_b["scheduler"])
    metadata = _metadata(model_b, optimizer_b, scheduler_b, batch_b, configs_b)

    plain = _run_fit(
        model_a, optimizer_a, scheduler_a, batch_a, contexts_a, configs_a
    )
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    checkpointed = run_checkpointed_fit(
        model_b,
        optimizer_b,
        scheduler_b,
        (batch_b,),
        (batch_b,),
        contexts_b,
        configs_b["loss"],
        configs_b["train_step"],
        configs_b["validation_step"],
        configs_b["scheduler"],
        configs_b["model_selection"],
        ModelSelectionState(),
        configs_b["fit"],
        manager,
        metadata,
    )
    assert checkpointed.fit_result == plain
    assert _tree_equal(model_b.state_dict(), model_a.state_dict())
    assert _tree_equal(optimizer_b.state_dict(), optimizer_a.state_dict())
    assert _tree_equal(scheduler_b.state_dict(), scheduler_a.state_dict())
    assert manager.list_epochs() == (0, 1)
    latest = manager.load_latest()
    assert latest.progress.next_epoch == 2 and latest.progress.global_step == 2

    _, fresh, _, _, fresh_batch, fresh_contexts = _mixed_case(
        typed_crystal, dtype=torch.float64, device="cpu"
    )
    configs_fresh = _configs(fresh, 3)
    fresh_optimizer = build_optimizer(fresh, configs_fresh["optimizer"])
    fresh_scheduler = build_scheduler(fresh_optimizer, configs_fresh["scheduler"])
    state = restore_training_checkpoint_(
        latest,
        fresh,
        fresh_optimizer,
        fresh_scheduler,
        (fresh_batch,),
        (fresh_batch,),
        fresh_contexts,
        configs_fresh,
        resumed_max_epochs=3,
        policy=ResumePolicy(),
    )
    assert state.next_epoch == 2 and state.global_step == 2
    assert tuple(fresh_optimizer.param_groups[0]["params"]) == tuple(
        fresh.parameters()
    )


    best = manager.load_best()
    _, best_model, _, _, best_batch, best_contexts = _mixed_case(
        typed_crystal, dtype=torch.float64, device="cpu"
    )
    best_configs = _configs(best_model, 3)
    best_optimizer = build_optimizer(best_model, best_configs["optimizer"])
    best_scheduler = build_scheduler(best_optimizer, best_configs["scheduler"])
    restore_training_checkpoint_(
        best, best_model, best_optimizer, best_scheduler,
        (best_batch,), (best_batch,), best_contexts, best_configs,
        resumed_max_epochs=3, policy=ResumePolicy(),
    )
    assert _tree_equal(best_model.state_dict(), best.model_state_dict)
    assert best.progress.best_epoch == checkpointed.fit_result.best_epoch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_cuda_one_epoch_checkpointed_fit_smoke_when_available(
    typed_crystal, tmp_path, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _, model, _, _, batch, contexts = _mixed_case(
        typed_crystal, dtype=dtype, device="cuda"
    )
    configs = _configs(model, 1)
    optimizer = build_optimizer(model, configs["optimizer"])
    scheduler = build_scheduler(optimizer, configs["scheduler"])
    metadata = _metadata(model, optimizer, scheduler, batch, configs)
    manager = CheckpointManager(
        CheckpointManagerConfig(tmp_path / str(dtype).replace("torch.", ""))
    )
    result = run_checkpointed_fit(
        model,
        optimizer,
        scheduler,
        (batch,),
        (batch,),
        contexts,
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
    assert result.epochs_checkpointed == 1
    loaded = manager.load_latest()
    assert loaded.cuda_device_count == torch.cuda.device_count()
    assert loaded.cuda_rng_states
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

