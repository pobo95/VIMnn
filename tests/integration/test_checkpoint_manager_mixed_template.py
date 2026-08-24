from __future__ import annotations

from pathlib import Path
import runpy

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
    run_fit,
    validate_checkpoint_compatibility,
    validate_checkpoint_history,
)


def _mixed_case(typed_crystal):
    path = Path(__file__).with_name("test_fit_controller_mixed_template.py")
    return runpy.run_path(str(path))["_mixed_case"](
        typed_crystal, dtype=torch.float64, device="cpu"
    )


def _configs(model, max_epochs):
    return {
        "model": model.config,
        "loss": LossConfig(energy_weight=1.0),
        "optimizer": OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0),
        "train_step": TrainStepConfig(),
        "validation_step": ValidationStepConfig(),
        "scheduler": SchedulerConfig(),
        "model_selection": ModelSelectionConfig(),
        "fit": FitConfig(max_epochs),
    }


def _run(model, optimizer, scheduler, batch, contexts, configs, state, fit_config):
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
        state,
        fit_config,
    )


def _capture(model, optimizer, scheduler, batch, configs, records, state):
    last = records[-1]
    progress = FitProgress(
        next_epoch=last.epoch_index + 1,
        global_step=last.training.global_step_end,
        completed_epochs=len(records),
        last_completed_epoch=last.epoch_index,
        best_epoch=state.best_epoch,
        best_global_step=state.best_global_step,
    )
    return capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
        state,
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
        species_vocabulary=model.config.species_vocabulary,
        fit_history=records,
    )


def test_actual_mixed_template_two_epochs_and_latest_restore_compatibility(
    typed_crystal, tmp_path
):
    _, model, _, _, batch, contexts = _mixed_case(typed_crystal)
    configs1 = _configs(model, 1)
    optimizer = build_optimizer(model, configs1["optimizer"])
    scheduler = build_scheduler(optimizer, configs1["scheduler"])
    first = _run(
        model,
        optimizer,
        scheduler,
        batch,
        contexts,
        configs1,
        ModelSelectionState(),
        FitConfig(1),
    )
    checkpoint0 = _capture(
        model,
        optimizer,
        scheduler,
        batch,
        configs1,
        first.records,
        first.final_selection_state,
    )
    manager = CheckpointManager(CheckpointManagerConfig(tmp_path / "managed"))
    manager.save_epoch(checkpoint0, first.records[-1])

    configs2 = _configs(model, 2)
    continuation = _run(
        model,
        optimizer,
        scheduler,
        batch,
        contexts,
        configs2,
        first.final_selection_state,
        FitConfig(2, 1, 1),
    )
    all_records = first.records + continuation.records
    checkpoint1 = _capture(
        model,
        optimizer,
        scheduler,
        batch,
        configs2,
        all_records,
        continuation.final_selection_state,
    )
    manager.save_epoch(checkpoint1, continuation.records[-1])
    assert manager.list_epochs() == (0, 1)
    latest = manager.load_latest()
    assert latest.progress.next_epoch == 2
    assert tuple(record.epoch_index for record in validate_checkpoint_history(latest)) == (
        0,
        1,
    )

    _, fresh, _, _, fresh_batch, fresh_contexts = _mixed_case(typed_crystal)
    configs3 = _configs(fresh, 3)
    fresh_optimizer = build_optimizer(fresh, configs3["optimizer"])
    fresh_scheduler = build_scheduler(fresh_optimizer, configs3["scheduler"])
    diagnostics = validate_checkpoint_compatibility(
        latest,
        fresh,
        fresh_optimizer,
        fresh_scheduler,
        (fresh_batch,),
        (fresh_batch,),
        fresh_contexts,
        configs3,
        resumed_max_epochs=3,
        policy=ResumePolicy(),
    )
    assert diagnostics
    parameter_ids = tuple(id(parameter) for parameter in fresh.parameters())
    state = restore_training_checkpoint_(
        latest,
        fresh,
        fresh_optimizer,
        fresh_scheduler,
        (fresh_batch,),
        (fresh_batch,),
        fresh_contexts,
        configs3,
        resumed_max_epochs=3,
        policy=ResumePolicy(),
    )
    assert state.next_epoch == 2 and state.global_step == 2
    assert tuple(id(parameter) for parameter in fresh.parameters()) == parameter_ids
    assert tuple(fresh_optimizer.param_groups[0]["params"]) == tuple(
        fresh.parameters()
    )

