from __future__ import annotations

from pathlib import Path
import runpy

import pytest
import torch

from refsite_mlip.training import (
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
    run_fit,
)


def _mixed_case(typed_crystal, *, dtype, device):
    path = Path(__file__).with_name("test_fit_controller_mixed_template.py")
    return runpy.run_path(str(path))["_mixed_case"](
        typed_crystal, dtype=dtype, device=device
    )


def _configs(model, *, max_epochs):
    optimizer = OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0)
    scheduler = SchedulerConfig()
    values = {
        "model": model.config,
        "loss": LossConfig(energy_weight=1.0),
        "optimizer": optimizer,
        "train_step": TrainStepConfig(),
        "validation_step": ValidationStepConfig(),
        "scheduler": scheduler,
        "model_selection": ModelSelectionConfig(),
        "fit": FitConfig(max_epochs),
    }
    return optimizer, scheduler, values


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


def _actual_restore(typed_crystal, *, dtype, device):
    _, source_model, _, _, batch, contexts = _mixed_case(
        typed_crystal, dtype=dtype, device=device
    )
    optimizer_config, scheduler_config, configs = _configs(
        source_model, max_epochs=2
    )
    source_optimizer = build_optimizer(source_model, optimizer_config)
    source_scheduler = build_scheduler(source_optimizer, scheduler_config)
    fit = run_fit(
        source_model,
        source_optimizer,
        source_scheduler,
        (batch,),
        (batch,),
        contexts,
        configs["loss"],
        configs["train_step"],
        configs["validation_step"],
        scheduler_config,
        configs["model_selection"],
        ModelSelectionState(),
        configs["fit"],
    )
    progress = FitProgress(
        next_epoch=fit.next_epoch,
        global_step=fit.global_step_end,
        completed_epochs=fit.epochs_completed,
        last_completed_epoch=fit.records[-1].epoch_index,
        stopped_early=fit.stopped_early,
        best_epoch=fit.best_epoch,
        best_global_step=fit.best_global_step,
    )
    checkpoint = capture_training_checkpoint(
        source_model,
        source_optimizer,
        source_scheduler,
        fit.final_selection_state,
        progress,
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
        species_vocabulary=source_model.config.species_vocabulary,
        fit_history=fit.records,
    )

    _, target_model, _, _, target_batch, target_contexts = _mixed_case(
        typed_crystal, dtype=dtype, device=device
    )
    target_optimizer = build_optimizer(target_model, optimizer_config)
    target_scheduler = build_scheduler(target_optimizer, scheduler_config)
    parameter_ids = tuple(id(parameter) for parameter in target_model.parameters())
    resolved = dict(configs)
    resolved["model"] = target_model.config
    resolved["fit"] = FitConfig(3)
    resume = restore_training_checkpoint_(
        checkpoint,
        target_model,
        target_optimizer,
        target_scheduler,
        (target_batch,),
        (target_batch,),
        target_contexts,
        resolved,
        resumed_max_epochs=3,
        policy=ResumePolicy(),
    )
    assert resume.next_epoch == 2 and resume.global_step == 2
    assert resume.selection_state == fit.final_selection_state
    assert resume.fit_history == checkpoint.fit_history
    assert tuple(id(parameter) for parameter in target_model.parameters()) == parameter_ids
    assert tuple(target_optimizer.param_groups[0]["params"]) == tuple(
        target_model.parameters()
    )
    assert target_scheduler.optimizer is target_optimizer
    assert all(parameter.grad is None for parameter in target_model.parameters())
    assert _tree_equal(target_model.state_dict(), checkpoint.model_state_dict)
    assert _tree_equal(target_optimizer.state_dict(), checkpoint.optimizer_state_dict)
    assert _tree_equal(target_scheduler.state_dict(), checkpoint.scheduler_state_dict)


def test_actual_mixed_template_cpu_float64_restore(typed_crystal):
    _actual_restore(typed_crystal, dtype=torch.float64, device="cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_mixed_template_cuda_optimizer_restore(typed_crystal, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _actual_restore(typed_crystal, dtype=dtype, device="cuda")

