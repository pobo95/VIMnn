from __future__ import annotations

import copy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from refsite_mlip.models import instantiate_reference_site_model_bundle
from refsite_mlip.training import (
    CheckpointCompatibilityError,
    CheckpointRestoreError,
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
    restore_training_checkpoint_,
    run_fit,
    run_resumed_fit,
    save_training_checkpoint,
    train_step,
)

from test_symmetric_correlation_bundle import _capture_v2


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _configs(model, max_epochs, *, scheduler_kind="none"):
    return {
        "model": model.config,
        "loss": LossConfig(energy_weight=1.0),
        "optimizer": OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0),
        "train_step": TrainStepConfig(),
        "validation_step": ValidationStepConfig(),
        "scheduler": SchedulerConfig(kind=scheduler_kind, patience=0),
        "model_selection": ModelSelectionConfig(),
        "fit": FitConfig(max_epochs),
    }


def _train_and_capture(typed_crystal):
    _, model, _, _, batch, contexts, _, bundle = _capture_v2(typed_crystal)
    configs = _configs(model, 1)
    optimizer = build_optimizer(model, configs["optimizer"])
    scheduler = build_scheduler(optimizer, configs["scheduler"])
    result = run_fit(
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
    progress = FitProgress(
        next_epoch=result.next_epoch,
        global_step=result.global_step_end,
        completed_epochs=result.epochs_completed,
        last_completed_epoch=result.records[-1].epoch_index,
        stopped_early=result.stopped_early,
        best_epoch=result.best_epoch,
        best_global_step=result.best_global_step,
    )
    checkpoint = capture_training_checkpoint(
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
    return model, optimizer, checkpoint, bundle, batch, contexts


def _fresh(bundle, batch):
    loaded = instantiate_reference_site_model_bundle(
        bundle, device="cpu", dtype=torch.float64
    )
    model = loaded.model
    configs = _configs(model, 2)
    optimizer = build_optimizer(model, configs["optimizer"])
    scheduler = build_scheduler(optimizer, configs["scheduler"])
    return loaded, model, optimizer, scheduler, configs


def test_v2_checkpoint_preserves_single_basis_and_all_independent_weights(
    typed_crystal,
):
    model, optimizer, checkpoint, _, _, _ = _train_and_capture(typed_crystal)
    state = checkpoint.model_state_dict
    u_keys = [key for key in state if key.startswith("symmetric_cg_basis.")]
    w_keys = [
        key for key in state if ".symmetric_contraction.weight_" in key
    ]
    assert checkpoint.schema_version == "refsite_training_checkpoint_v1"
    assert len(u_keys) == 9
    assert len(w_keys) == 18
    assert not any(key.startswith("layers.") and ".basis" in key for key in state)

    named = dict(model.named_parameters())
    weights = [named[key] for key in w_keys]
    assert len({id(value) for value in weights}) == 18
    assert len({value.untyped_storage().data_ptr() for value in weights}) == 18
    optimized = [value for group in optimizer.param_groups for value in group["params"]]
    assert all(sum(value is candidate for candidate in optimized) == 1 for value in weights)
    assert not any("symmetric_cg_basis" in key for key in named)
    assert all(value in optimizer.state for value in weights)
    assert all(
        torch.isfinite(optimizer.state[value][name]).all()
        for value in weights
        for name in ("exp_avg", "exp_avg_sq")
    )


def test_v2_force_and_stress_double_backward_updates_every_active_weight(
    typed_crystal,
):
    _, model, _, _, batch, contexts, _, _ = _capture_v2(typed_crystal)
    supervised = replace(
        batch,
        energy_mask=torch.ones_like(batch.energy_mask),
        forces=torch.zeros_like(batch.forces),
        force_mask=torch.ones_like(batch.force_mask),
        force_present=torch.ones_like(batch.force_present),
        force_mask_provided=torch.ones_like(batch.force_mask_provided),
        stress=torch.zeros_like(batch.stress),
        stress_mask=torch.ones_like(batch.stress_mask),
        stress_present=torch.ones_like(batch.stress_present),
        stress_mask_provided=torch.ones_like(batch.stress_mask_provided),
    )
    optimizer = build_optimizer(
        model, OptimizerConfig(learning_rate=1.0e-5, weight_decay=0.0)
    )
    before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if ".symmetric_contraction.weight_" in name
    }
    result = train_step(
        model,
        optimizer,
        supervised,
        contexts,
        LossConfig(energy_weight=1.0, force_weight=0.01, stress_weight=0.01),
        TrainStepConfig(),
    )
    assert result.need_forces and result.need_stress
    named = dict(model.named_parameters())
    assert set(before) == {
        name
        for name in named
        if ".symmetric_contraction.weight_" in name
    }
    changed = []
    for name, old in before.items():
        parameter = named[name]
        changed.append(not torch.equal(parameter, old))
        assert parameter in optimizer.state
        assert all(
            torch.isfinite(optimizer.state[parameter][field]).all()
            for field in ("exp_avg", "exp_avg_sq")
        )
    assert any(changed)


def test_v2_immediate_restore_is_exact_and_preserves_parameter_identity(
    typed_crystal,
):
    _, _, checkpoint, bundle, batch, _ = _train_and_capture(typed_crystal)
    loaded, model, optimizer, scheduler, configs = _fresh(bundle, batch)
    parameter_ids = tuple(id(value) for value in model.parameters())
    restore_training_checkpoint_(
        checkpoint,
        model,
        optimizer,
        scheduler,
        (batch,),
        (batch,),
        loaded.template_contexts,
        configs,
        resumed_max_epochs=2,
    )
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert _tree_equal(model.state_dict(), checkpoint.model_state_dict)
    assert _tree_equal(optimizer.state_dict(), checkpoint.optimizer_state_dict)
    assert _tree_equal(scheduler.state_dict(), checkpoint.scheduler_state_dict)
    optimized = tuple(value for group in optimizer.param_groups for value in group["params"])
    assert optimized == tuple(model.parameters())


@pytest.mark.parametrize("corruption", ["basis_value", "optimizer_weight_state"])
def test_v2_semantic_corruption_fails_before_runtime_or_rng_mutation(
    typed_crystal, corruption
):
    trained_model, trained_optimizer, checkpoint, bundle, batch, _ = _train_and_capture(
        typed_crystal
    )
    if corruption == "basis_value":
        state = {key: value.clone() for key, value in checkpoint.model_state_dict.items()}
        key = next(key for key in state if key.startswith("symmetric_cg_basis."))
        state[key].view(-1)[0] += 1.0
        corrupt = replace(checkpoint, model_state_dict=state)
    else:
        optimizer_state = copy.deepcopy(checkpoint.optimizer_state_dict)
        first_weight = next(
            value
            for name, value in trained_model.named_parameters()
            if ".symmetric_contraction.weight_" in name
        )
        parameter_index = next(
            index
            for index, value in enumerate(trained_optimizer.param_groups[0]["params"])
            if value is first_weight
        )
        first_slot = optimizer_state["param_groups"][0]["params"][parameter_index]
        optimizer_state["state"].pop(first_slot)
        corrupt = replace(checkpoint, optimizer_state_dict=optimizer_state)

    loaded, model, optimizer, scheduler, configs = _fresh(bundle, batch)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    parameter_ids = tuple(id(value) for value in model.parameters())
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    with pytest.raises(CheckpointCompatibilityError):
        restore_training_checkpoint_(
            corrupt,
            model,
            optimizer,
            scheduler,
            (batch,),
            (batch,),
            loaded.template_contexts,
            configs,
            resumed_max_epochs=2,
        )
    assert _tree_equal(model.state_dict(), model_before)
    assert _tree_equal(optimizer.state_dict(), optimizer_before)
    assert _tree_equal(scheduler.state_dict(), scheduler_before)
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert random.getstate() == python_before
    after_numpy = np.random.get_state()
    assert numpy_before[0] == after_numpy[0]
    assert np.array_equal(numpy_before[1], after_numpy[1])
    assert numpy_before[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize("scheduler_kind", ["none", "reduce_on_plateau"])
def test_v2_cpu_float64_continuous_three_epochs_equals_one_plus_resume(
    typed_crystal, scheduler_kind
):
    rng_entry = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    _, continuous_model, _, _, continuous_batch, continuous_contexts, _, _ = (
        _capture_v2(typed_crystal)
    )
    continuous_configs = _configs(
        continuous_model, 3, scheduler_kind=scheduler_kind
    )
    continuous_optimizer = build_optimizer(
        continuous_model, continuous_configs["optimizer"]
    )
    continuous_scheduler = build_scheduler(
        continuous_optimizer, continuous_configs["scheduler"]
    )
    continuous = run_fit(
        continuous_model,
        continuous_optimizer,
        continuous_scheduler,
        (continuous_batch,),
        (continuous_batch,),
        continuous_contexts,
        continuous_configs["loss"],
        continuous_configs["train_step"],
        continuous_configs["validation_step"],
        continuous_configs["scheduler"],
        continuous_configs["model_selection"],
        ModelSelectionState(),
        continuous_configs["fit"],
    )
    continuous_draw = (random.random(), float(np.random.random()), torch.rand(4))

    random.setstate(rng_entry[0])
    np.random.set_state(rng_entry[1])
    torch.set_rng_state(rng_entry[2])
    _, split_model, _, _, split_batch, split_contexts, _, split_bundle = _capture_v2(
        typed_crystal
    )
    split_configs = _configs(split_model, 1, scheduler_kind=scheduler_kind)
    split_optimizer = build_optimizer(split_model, split_configs["optimizer"])
    split_scheduler = build_scheduler(split_optimizer, split_configs["scheduler"])
    first = run_fit(
        split_model,
        split_optimizer,
        split_scheduler,
        (split_batch,),
        (split_batch,),
        split_contexts,
        split_configs["loss"],
        split_configs["train_step"],
        split_configs["validation_step"],
        split_configs["scheduler"],
        split_configs["model_selection"],
        ModelSelectionState(),
        split_configs["fit"],
    )
    progress = FitProgress(
        next_epoch=first.next_epoch,
        global_step=first.global_step_end,
        completed_epochs=first.epochs_completed,
        last_completed_epoch=first.records[-1].epoch_index,
        stopped_early=first.stopped_early,
        best_epoch=first.best_epoch,
        best_global_step=first.best_global_step,
    )
    checkpoint = capture_training_checkpoint(
        split_model,
        split_optimizer,
        split_scheduler,
        first.final_selection_state,
        progress,
        (split_batch,),
        (split_batch,),
        model_config=split_configs["model"],
        loss_config=split_configs["loss"],
        optimizer_config=split_configs["optimizer"],
        train_step_config=split_configs["train_step"],
        validation_step_config=split_configs["validation_step"],
        scheduler_config=split_configs["scheduler"],
        model_selection_config=split_configs["model_selection"],
        fit_config=split_configs["fit"],
        species_vocabulary=split_model.config.species_vocabulary,
        fit_history=first.records,
    )
    loaded = instantiate_reference_site_model_bundle(
        split_bundle, device="cpu", dtype=torch.float64
    )
    resumed_model = loaded.model
    resumed_configs = _configs(
        resumed_model, 3, scheduler_kind=scheduler_kind
    )
    resumed_optimizer = build_optimizer(
        resumed_model, resumed_configs["optimizer"]
    )
    resumed_scheduler = build_scheduler(
        resumed_optimizer, resumed_configs["scheduler"]
    )
    resumed = run_resumed_fit(
        checkpoint,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        (split_batch,),
        (split_batch,),
        loaded.template_contexts,
        resumed_configs["loss"],
        resumed_configs["train_step"],
        resumed_configs["validation_step"],
        resumed_configs["scheduler"],
        resumed_configs["model_selection"],
        resumed_configs,
        resumed_max_epochs=3,
    )
    resumed_draw = (random.random(), float(np.random.random()), torch.rand(4))

    assert _tree_equal(continuous_model.state_dict(), resumed_model.state_dict())
    assert _tree_equal(
        continuous_optimizer.state_dict(), resumed_optimizer.state_dict()
    )
    assert _tree_equal(
        continuous_scheduler.state_dict(), resumed_scheduler.state_dict()
    )
    assert resumed.combined_fit_result == continuous
    assert continuous_draw[:2] == resumed_draw[:2]
    assert torch.equal(continuous_draw[2], resumed_draw[2])


@pytest.mark.parametrize("failure_stage", ["scheduler", "rng"])
def test_v2_post_mutation_restore_failure_rolls_back_every_live_state(
    typed_crystal, tmp_path, monkeypatch, failure_stage
):
    import refsite_mlip.training.resume as resume_module

    _, _, checkpoint, bundle, batch, _ = _train_and_capture(typed_crystal)
    checkpoint_path = tmp_path / "committed.pt"
    save_training_checkpoint(checkpoint, checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    journal_path = tmp_path / "metrics.jsonl"
    journal_path.write_bytes(b'{"committed":true}\n')
    journal_bytes = journal_path.read_bytes()

    loaded, model, optimizer, scheduler, configs = _fresh(bundle, batch)
    model.eval()
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.125)
    gradient_object = first_parameter.grad
    gradient_value = gradient_object.clone()
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    parameter_ids = tuple(id(value) for value in model.parameters())
    checkpoint_before = copy.deepcopy(checkpoint.to_dict())
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    dtype_before = torch.get_default_dtype()
    grad_mode_before = torch.is_grad_enabled()

    if failure_stage == "scheduler":
        original_load = scheduler.load_state_dict
        calls = 0

        def fail_once(state):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected scheduler restore failure")
            return original_load(state)

        monkeypatch.setattr(scheduler, "load_state_dict", fail_once)
    else:
        def fail_rng(*args, **kwargs):
            del args, kwargs
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            raise RuntimeError("injected RNG restore failure")

        monkeypatch.setattr(resume_module, "_restore_checkpoint_rng", fail_rng)

    with pytest.raises(CheckpointRestoreError) as caught:
        restore_training_checkpoint_(
            checkpoint,
            model,
            optimizer,
            scheduler,
            (batch,),
            (batch,),
            loaded.template_contexts,
            configs,
            resumed_max_epochs=2,
        )
    assert caught.value.rollback_succeeded
    assert _tree_equal(model.state_dict(), model_before)
    assert _tree_equal(optimizer.state_dict(), optimizer_before)
    assert _tree_equal(scheduler.state_dict(), scheduler_before)
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert first_parameter.grad is gradient_object
    assert torch.equal(first_parameter.grad, gradient_value)
    assert all(value.grad is None for value in tuple(model.parameters())[1:])
    assert model.training is False
    assert _tree_equal(checkpoint.to_dict(), checkpoint_before)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert torch.get_default_dtype() == dtype_before
    assert torch.is_grad_enabled() == grad_mode_before
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert journal_path.read_bytes() == journal_bytes
