from __future__ import annotations

from pathlib import Path
import runpy

import pytest
import torch

from refsite_mlip.training import (
    FitConfig,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_optimizer,
    build_scheduler,
    run_fit,
)


def _mixed_case(typed_crystal, *, dtype=torch.float64, device="cpu"):
    path = Path(__file__).with_name("test_grouped_template_batch.py")
    case_builder = runpy.run_path(str(path))["_case"]
    return case_builder(typed_crystal, dtype=dtype, device=device)


def _run_actual_fit(typed_crystal, *, dtype, device, epochs):
    _, model, _, _, batch, contexts = _mixed_case(
        typed_crystal, dtype=dtype, device=device
    )
    optimizer = build_optimizer(
        model,
        OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0),
    )
    scheduler_config = SchedulerConfig()
    scheduler = build_scheduler(optimizer, scheduler_config)
    result = run_fit(
        model,
        optimizer,
        scheduler,
        (batch,),
        (batch,),
        contexts,
        LossConfig(energy_weight=1.0),
        TrainStepConfig(),
        ValidationStepConfig(),
        scheduler_config,
        ModelSelectionConfig(),
        ModelSelectionState(),
        FitConfig(epochs),
    )
    return model, optimizer, scheduler, batch, result


def test_actual_mixed_template_multi_epoch_cpu_float64(typed_crystal):
    model, optimizer, scheduler, batch, result = _run_actual_fit(
        typed_crystal, dtype=torch.float64, device="cpu", epochs=2
    )
    assert result.epochs_completed == 2
    assert result.global_step_end == 2
    assert scheduler.state_dict() == {"validation_steps": 2}
    assert tuple(result.records[0].training.ordered_batch_sample_ids[0]) == batch.sample_ids
    assert len(set(batch.template_ids)) == 2
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_mixed_template_fit_cuda_smoke_when_available(typed_crystal, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model, _, scheduler, _, result = _run_actual_fit(
        typed_crystal, dtype=dtype, device="cuda", epochs=1
    )
    assert result.epochs_completed == 1 and result.global_step_end == 1
    assert scheduler.state_dict() == {"validation_steps": 1}
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
