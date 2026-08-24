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
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    capture_training_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _fit_case(typed_crystal, *, dtype, device, epochs):
    path = Path(__file__).with_name("test_fit_controller_mixed_template.py")
    runner = runpy.run_path(str(path))["_run_actual_fit"]
    return runner(typed_crystal, dtype=dtype, device=device, epochs=epochs)


def _capture_fit(model, optimizer, scheduler, batch, result):
    progress = FitProgress(
        next_epoch=result.next_epoch,
        global_step=result.global_step_end,
        completed_epochs=result.epochs_completed,
        last_completed_epoch=result.records[-1].epoch_index,
        stopped_early=result.stopped_early,
        best_epoch=result.best_epoch,
        best_global_step=result.best_global_step,
    )
    return capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
        result.final_selection_state,
        progress,
        (batch,),
        (batch,),
        model_config=model.config,
        loss_config=LossConfig(energy_weight=1.0),
        optimizer_config=OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0),
        train_step_config=TrainStepConfig(),
        validation_step_config=ValidationStepConfig(),
        scheduler_config=SchedulerConfig(),
        model_selection_config=ModelSelectionConfig(),
        fit_config=FitConfig(result.config.max_epochs),
        species_vocabulary=model.config.species_vocabulary,
        fit_history=result.records,
    )


def _tensor_values(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_values(item)


def test_actual_mixed_template_fit_checkpoint_smoke(typed_crystal, tmp_path):
    model, optimizer, scheduler, batch, result = _fit_case(
        typed_crystal, dtype=torch.float64, device="cpu", epochs=1
    )
    checkpoint = _capture_fit(model, optimizer, scheduler, batch, result)
    path = tmp_path / "mixed-fit.pt"
    save_training_checkpoint(checkpoint, path)
    loaded = load_training_checkpoint(path)
    assert loaded.progress.global_step == 1
    assert loaded.progress.next_epoch == 1
    assert loaded.selection_state == result.final_selection_state
    assert loaded.fit_history == checkpoint.fit_history
    assert set(loaded.metadata.template_fingerprints) == {"alpha", "zeta"}
    assert loaded.metadata.training_data.number_of_structures == 3
    assert torch.equal(
        loaded.model_state_dict["atomic_baseline"],
        model.atomic_baseline.detach().cpu(),
    )


def test_cuda_rng_and_optimizer_snapshot_when_available(typed_crystal, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model, optimizer, scheduler, batch, result = _fit_case(
        typed_crystal, dtype=torch.float64, device="cuda", epochs=1
    )
    checkpoint = _capture_fit(model, optimizer, scheduler, batch, result)
    assert checkpoint.cuda_device_count == torch.cuda.device_count()
    assert len(checkpoint.cuda_rng_states) == checkpoint.cuda_device_count
    assert checkpoint.cuda_rng_states
    assert all(
        tensor.device.type == "cpu" and not tensor.requires_grad
        for tensor in _tensor_values(checkpoint.optimizer_state_dict)
    )
    path = tmp_path / "cuda-fit.pt"
    save_training_checkpoint(checkpoint, path)
    loaded = load_training_checkpoint(path)
    assert len(loaded.cuda_rng_states) == checkpoint.cuda_device_count
    assert all(state.device.type == "cpu" for state in loaded.cuda_rng_states)
