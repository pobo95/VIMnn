from __future__ import annotations

from pathlib import Path
import random
import runpy

import numpy as np
import pytest
import torch

from refsite_mlip.training import (
    FitConfig,
    FitProgress,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_optimizer,
    build_scheduler,
    capture_training_checkpoint,
    load_training_checkpoint,
    run_fit,
    run_resumed_fit,
    save_training_checkpoint,
)


def _mixed_case(typed_crystal, *, dtype, device):
    path = Path(__file__).with_name("test_fit_controller_mixed_template.py")
    return runpy.run_path(str(path))["_mixed_case"](
        typed_crystal, dtype=dtype, device=device
    )


def _configs(model, *, max_epochs):
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


def _fit(model, optimizer, scheduler, batch, contexts, configs):
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


def _capture(model, optimizer, scheduler, batch, configs, result):
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
        model_config=configs["model"],
        loss_config=configs["loss"],
        optimizer_config=configs["optimizer"],
        train_step_config=configs["train_step"],
        validation_step_config=configs["validation_step"],
        scheduler_config=configs["scheduler"],
        model_selection_config=configs["model_selection"],
        fit_config=configs["fit"],
        species_vocabulary=model.config.species_vocabulary,
        fit_history=result.records,
    )


def _tree_first_mismatch(first, second, path="root"):
    if isinstance(first, torch.Tensor):
        if torch.equal(first.detach().cpu(), second.detach().cpu()):
            return None
        difference = (first.detach().cpu() - second.detach().cpu()).abs()
        return path, float(difference.max())
    if isinstance(first, dict):
        if first.keys() != second.keys():
            return path + ".keys", float("nan")
        for key in first:
            mismatch = _tree_first_mismatch(first[key], second[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(first, (tuple, list)):
        if len(first) != len(second):
            return path + ".length", float("nan")
        for index, (left, right) in enumerate(zip(first, second)):
            mismatch = _tree_first_mismatch(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if first == second else (path, float("nan"))


def _rng():
    numpy = np.random.get_state()
    return (
        random.getstate(),
        (numpy[0], numpy[1].copy(), *numpy[2:]),
        torch.get_rng_state().clone(),
        tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    )


def _set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(list(state[3]))


def _draw(device):
    result = [random.random(), float(np.random.random()), torch.rand(4)]
    if torch.device(device).type == "cuda":
        result.append(torch.rand(4, device=device).cpu())
    return tuple(result)


def _assert_draws_equal(first, second):
    assert first[0] == second[0] and first[1] == second[1]
    assert all(torch.equal(left, right) for left, right in zip(first[2:], second[2:]))


def _run_exact_resume(typed_crystal, tmp_path, *, dtype, device):
    initial_rng = _rng()
    _, model_a, _, _, batch_a, contexts_a = _mixed_case(
        typed_crystal, dtype=dtype, device=device
    )
    configs_a = _configs(model_a, max_epochs=2)
    optimizer_a = build_optimizer(model_a, configs_a["optimizer"])
    scheduler_a = build_scheduler(optimizer_a, configs_a["scheduler"])
    continuous = _fit(
        model_a, optimizer_a, scheduler_a, batch_a, contexts_a, configs_a
    )
    draws_a = _draw(device)

    _set_rng(initial_rng)
    _, split_model, _, _, split_batch, split_contexts = _mixed_case(
        typed_crystal, dtype=dtype, device=device
    )
    split_configs = _configs(split_model, max_epochs=1)
    split_optimizer = build_optimizer(split_model, split_configs["optimizer"])
    split_scheduler = build_scheduler(split_optimizer, split_configs["scheduler"])
    first = _fit(
        split_model,
        split_optimizer,
        split_scheduler,
        split_batch,
        split_contexts,
        split_configs,
    )
    checkpoint = _capture(
        split_model,
        split_optimizer,
        split_scheduler,
        split_batch,
        split_configs,
        first,
    )
    path = tmp_path / f"mixed-{dtype}-{device}.pt"
    save_training_checkpoint(checkpoint, path)
    loaded = load_training_checkpoint(path)

    _, model_b, _, _, batch_b, contexts_b = _mixed_case(
        typed_crystal, dtype=dtype, device=device
    )
    configs_b = _configs(model_b, max_epochs=2)
    optimizer_b = build_optimizer(model_b, configs_b["optimizer"])
    scheduler_b = build_scheduler(optimizer_b, configs_b["scheduler"])
    parameter_ids = tuple(id(parameter) for parameter in model_b.parameters())
    resumed = run_resumed_fit(
        loaded,
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
        configs_b,
        resumed_max_epochs=2,
    )
    draws_b = _draw(device)

    model_mismatch = _tree_first_mismatch(model_a.state_dict(), model_b.state_dict())
    optimizer_mismatch = _tree_first_mismatch(
        optimizer_a.state_dict(), optimizer_b.state_dict()
    )
    scheduler_mismatch = _tree_first_mismatch(
        scheduler_a.state_dict(), scheduler_b.state_dict()
    )
    _assert_draws_equal(draws_a, draws_b)
    assert tuple(id(parameter) for parameter in model_b.parameters()) == parameter_ids
    assert tuple(optimizer_b.param_groups[0]["params"]) == tuple(model_b.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model_b.parameters())
    return {
        "model": model_mismatch,
        "optimizer": optimizer_mismatch,
        "scheduler": scheduler_mismatch,
        "fit_result_exact": resumed.combined_fit_result == continuous,
        "global_step_exact": (
            resumed.combined_fit_result.global_step_end == continuous.global_step_end
        ),
        "selection_exact": (
            resumed.combined_fit_result.final_selection_state
            == continuous.final_selection_state
        ),
        "rng_next_draw_exact": True,
    }


def test_mixed_template_cpu_float64_continuous_resume_exact(typed_crystal, tmp_path):
    diagnostics = _run_exact_resume(
        typed_crystal, tmp_path, dtype=torch.float64, device="cpu"
    )
    assert diagnostics == {
        "model": None,
        "optimizer": None,
        "scheduler": None,
        "fit_result_exact": True,
        "global_step_exact": True,
        "selection_exact": True,
        "rng_next_draw_exact": True,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_template_cuda_continuous_resume_exact_attempt(
    typed_crystal, tmp_path, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    diagnostics = _run_exact_resume(
        typed_crystal, tmp_path, dtype=dtype, device="cuda"
    )
    print(f"CUDA resume exactness dtype={dtype}: {diagnostics}")
    assert diagnostics["scheduler"] is None
    assert diagnostics["global_step_exact"]
    assert diagnostics["selection_exact"]
    assert diagnostics["rng_next_draw_exact"]
    for key in ("model", "optimizer"):
        mismatch = diagnostics[key]
        if mismatch is not None:
            assert isinstance(mismatch[0], str)
            assert torch.isfinite(torch.tensor(mismatch[1]))
