from __future__ import annotations

import copy
from dataclasses import replace
import importlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import (
    EpochResult,
    EpochTermMetrics,
    FitConfig,
    FitExecutionError,
    FitProgress,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    ResumedFitExecutionError,
    ResumedFitResult,
    ResumePolicy,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_scheduler,
    capture_training_checkpoint,
    compose_resumed_fit_result,
    load_training_checkpoint,
    run_fit,
    run_resumed_fit,
    save_training_checkpoint,
    validate_checkpoint_history,
)


fit_module = importlib.import_module("refsite_mlip.training.fit")
resume_module = importlib.import_module("refsite_mlip.training.resume")
resume_fit_module = importlib.import_module("refsite_mlip.training.resume_fit")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
        self.register_buffer(
            "atomic_baseline", torch.tensor([0.5], dtype=torch.float64)
        )
        self.config = SimpleNamespace(species_vocabulary=(6,))


def _batch():
    return StructureBatch(
        sample_ids=("sample",),
        template_ids=("template",),
        template_fingerprints=("2" * 64,),
        positions=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64),
        atomic_numbers=torch.tensor([6], dtype=torch.long),
        cells=torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * 4.0,
        origins=torch.zeros((1, 3), dtype=torch.float64),
        pbc=torch.ones((1, 3), dtype=torch.bool),
        atom_ptr=torch.tensor([0, 1], dtype=torch.long),
        atom_batch=torch.zeros(1, dtype=torch.long),
        energy=torch.zeros(1, dtype=torch.float64),
        energy_mask=torch.ones(1, dtype=torch.bool),
        forces=torch.zeros((1, 3), dtype=torch.float64),
        force_mask=torch.zeros((1, 3), dtype=torch.bool),
        stress=torch.zeros((1, 3, 3), dtype=torch.float64),
        stress_mask=torch.zeros((1, 3, 3), dtype=torch.bool),
        force_present=torch.zeros(1, dtype=torch.bool),
        stress_present=torch.zeros(1, dtype=torch.bool),
        force_mask_provided=torch.zeros(1, dtype=torch.bool),
        stress_mask_provided=torch.zeros(1, dtype=torch.bool),
    )


def _term(value):
    return EpochTermMetrics(float(value), 1.0, float(value), 1)


def _epoch(phase, epoch_index, start, end, metric):
    training = phase == "train"
    return EpochResult(
        energy=_term(metric),
        force=EpochTermMetrics(0.0, 0.0, 0.0, 0),
        stress=EpochTermMetrics(0.0, 0.0, 0.0, 0),
        total_loss=float(metric),
        has_supervision=True,
        phase=phase,
        epoch_index=epoch_index,
        global_step_start=start,
        global_step_end=end,
        number_of_batches=1,
        number_of_supervised_batches=1,
        number_of_structures=1,
        number_of_atoms=1,
        successful_optimizer_steps=1 if training else 0,
        ordered_batch_sample_ids=(("sample",),),
        metric_semantics=(
            "pre_update_batch_observations"
            if training
            else "fixed_model_validation"
        ),
    )


def _install_deterministic_fit(monkeypatch, metrics):
    def training(
        model,
        optimizer,
        batches,
        *args,
        epoch_index,
        global_step_start,
    ):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (model.weight - 0.125).square()
        observed = float(loss.detach())
        loss.backward()
        optimizer.step()
        return _epoch(
            "train", epoch_index, global_step_start, global_step_start + 1, observed
        )

    def validation(model, batches, *args, epoch_index, global_step):
        return _epoch(
            "validation",
            epoch_index,
            global_step,
            global_step,
            metrics[epoch_index],
        )

    monkeypatch.setattr(fit_module, "run_training_epoch", training)
    monkeypatch.setattr(fit_module, "run_validation_epoch", validation)
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    monkeypatch.setattr(
        resume_module,
        "_validated_context",
        lambda template_id, fingerprint, mapping: mapping[template_id],
    )


def _configs(*, max_epochs, scheduler_config, selection_config):
    optimizer_config = OptimizerConfig(learning_rate=0.05, weight_decay=0.0)
    return {
        "model": {"kind": "tiny", "version": 1},
        "loss": LossConfig(energy_weight=1.0),
        "optimizer": optimizer_config,
        "train_step": TrainStepConfig(),
        "validation_step": ValidationStepConfig(),
        "scheduler": scheduler_config,
        "model_selection": selection_config,
        "fit": FitConfig(max_epochs),
    }


def _live(configs):
    model = TinyModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configs["optimizer"].learning_rate,
        weight_decay=0.0,
    )
    scheduler = build_scheduler(optimizer, configs["scheduler"])
    return model, optimizer, scheduler


def _fit(model, optimizer, scheduler, batch, configs, state=None):
    return run_fit(
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
        ModelSelectionState() if state is None else state,
        configs["fit"],
    )


def _capture(model, optimizer, scheduler, batch, configs, fit):
    progress = FitProgress(
        next_epoch=fit.next_epoch,
        global_step=fit.global_step_end,
        completed_epochs=fit.epochs_completed,
        last_completed_epoch=fit.records[-1].epoch_index,
        stopped_early=fit.stopped_early,
        best_epoch=fit.best_epoch,
        best_global_step=fit.best_global_step,
    )
    return capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
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
        species_vocabulary=(6,),
        fit_history=fit.records,
    )


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


def _rng_snapshot():
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


def _draws():
    return (random.random(), float(np.random.random()), torch.rand(4))


def _continuous_and_resumed(
    monkeypatch,
    tmp_path,
    *,
    metrics,
    total_epochs,
    split_epoch,
    scheduler_config=None,
    selection_config=None,
):
    scheduler_config = scheduler_config or SchedulerConfig()
    selection_config = selection_config or ModelSelectionConfig()
    _install_deterministic_fit(monkeypatch, metrics)
    batch = _batch()
    contexts = {"template": object()}
    initial_rng = _rng_snapshot()

    continuous_configs = _configs(
        max_epochs=total_epochs,
        scheduler_config=scheduler_config,
        selection_config=selection_config,
    )
    continuous_model, continuous_optimizer, continuous_scheduler = _live(
        continuous_configs
    )
    continuous = _fit(
        continuous_model,
        continuous_optimizer,
        continuous_scheduler,
        batch,
        continuous_configs,
    )
    continuous_draws = _draws()

    _set_rng(initial_rng)
    split_configs = _configs(
        max_epochs=split_epoch,
        scheduler_config=scheduler_config,
        selection_config=selection_config,
    )
    split_model, split_optimizer, split_scheduler = _live(split_configs)
    first = _fit(
        split_model, split_optimizer, split_scheduler, batch, split_configs
    )
    checkpoint = _capture(
        split_model, split_optimizer, split_scheduler, batch, split_configs, first
    )
    path = tmp_path / "resume.pt"
    save_training_checkpoint(checkpoint, path)
    loaded = load_training_checkpoint(path)

    resumed_configs = _configs(
        max_epochs=total_epochs,
        scheduler_config=scheduler_config,
        selection_config=selection_config,
    )
    resumed_model, resumed_optimizer, resumed_scheduler = _live(resumed_configs)
    resumed_parameter_ids = tuple(id(p) for p in resumed_model.parameters())
    resumed = run_resumed_fit(
        loaded,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        (batch,),
        (batch,),
        contexts,
        resumed_configs["loss"],
        resumed_configs["train_step"],
        resumed_configs["validation_step"],
        scheduler_config,
        selection_config,
        resumed_configs,
        resumed_max_epochs=total_epochs,
    )
    resumed_draws = _draws()
    assert tuple(id(p) for p in resumed_model.parameters()) == resumed_parameter_ids
    assert tuple(resumed_optimizer.param_groups[0]["params"]) == tuple(
        resumed_model.parameters()
    )
    return (
        continuous_model,
        continuous_optimizer,
        continuous_scheduler,
        continuous,
        continuous_draws,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed,
        resumed_draws,
        loaded,
        batch,
        contexts,
        resumed_configs,
    )


def test_history_continuity_and_missing_duplicate_gap_rejection(monkeypatch, tmp_path):
    values = _continuous_and_resumed(
        monkeypatch, tmp_path, metrics=(3.0, 2.0, 1.0), total_epochs=3, split_epoch=2
    )
    checkpoint = values[10]
    records = validate_checkpoint_history(checkpoint)
    assert tuple(record.epoch_index for record in records) == (0, 1)
    with pytest.raises(ValueError, match="full checkpoint fit_history"):
        validate_checkpoint_history(replace(checkpoint, fit_history=None))
    duplicate = copy.deepcopy(list(checkpoint.fit_history))
    duplicate[1]["epoch_index"] = 0
    with pytest.raises(ValueError, match="contiguous"):
        validate_checkpoint_history(replace(checkpoint, fit_history=tuple(duplicate)))
    gap = copy.deepcopy(list(checkpoint.fit_history))
    gap[1]["epoch_index"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        validate_checkpoint_history(replace(checkpoint, fit_history=tuple(gap)))


def test_history_progress_global_step_and_lr_mismatch(monkeypatch, tmp_path):
    values = _continuous_and_resumed(
        monkeypatch, tmp_path, metrics=(3.0, 2.0, 1.0), total_epochs=3, split_epoch=2
    )
    checkpoint = values[10]
    wrong_progress = replace(checkpoint.progress, global_step=9)
    with pytest.raises(ValueError, match="global step"):
        validate_checkpoint_history(replace(checkpoint, progress=wrong_progress))
    changed = copy.deepcopy(list(checkpoint.fit_history))
    changed[1]["learning_rates_used_for_training"] = [0.7]
    with pytest.raises(ValueError, match="learning rate"):
        validate_checkpoint_history(replace(checkpoint, fit_history=tuple(changed)))


@pytest.mark.parametrize(
    "scheduler_config",
    [
        SchedulerConfig(),
        SchedulerConfig(
            kind="reduce_on_plateau",
            factor=0.5,
            patience=0,
            threshold=0.0,
        ),
    ],
)
def test_continuous_vs_resumed_exact_state_result_and_rng(
    monkeypatch, tmp_path, scheduler_config
):
    values = _continuous_and_resumed(
        monkeypatch,
        tmp_path,
        metrics=(1.0, 2.0, 3.0),
        total_epochs=3,
        split_epoch=1,
        scheduler_config=scheduler_config,
    )
    (
        model_a,
        optimizer_a,
        scheduler_a,
        fit_a,
        draws_a,
        model_b,
        optimizer_b,
        scheduler_b,
        resumed,
        draws_b,
        *_rest,
    ) = values
    assert _tree_equal(model_a.state_dict(), model_b.state_dict())
    assert _tree_equal(optimizer_a.state_dict(), optimizer_b.state_dict())
    assert _tree_equal(scheduler_a.state_dict(), scheduler_b.state_dict())
    assert resumed.combined_fit_result == fit_a
    assert resumed.combined_fit_result.final_selection_state == fit_a.final_selection_state
    assert draws_a[0] == draws_b[0] and draws_a[1] == draws_b[1]
    assert torch.equal(draws_a[2], draws_b[2])
    assert ResumedFitResult.from_dict(resumed.to_dict()) == resumed
    assert resumed.resume_state.resumed_fit_config == FitConfig(3, 1, 1)


def test_scheduler_patience_and_early_stopping_cross_checkpoint(monkeypatch, tmp_path):
    scheduler = SchedulerConfig(
        kind="reduce_on_plateau",
        factor=0.5,
        patience=1,
        threshold=0.0,
        cooldown=1,
    )
    selection = ModelSelectionConfig(early_stopping_patience=2)
    values = _continuous_and_resumed(
        monkeypatch,
        tmp_path,
        metrics=(1.0, 2.0, 3.0, 4.0),
        total_epochs=4,
        split_epoch=1,
        scheduler_config=scheduler,
        selection_config=selection,
    )
    fit_a = values[3]
    resumed = values[8]
    assert fit_a.stopped_early and fit_a.stop_epoch == 2
    assert resumed.combined_fit_result == fit_a
    assert resumed.continuation_fit_result.stopped_early
    assert resumed.combined_fit_result.records[2].learning_rates_after_validation == (
        0.025,
    )


def test_composition_and_continuation_config(monkeypatch, tmp_path):
    values = _continuous_and_resumed(
        monkeypatch, tmp_path, metrics=(3.0, 2.0), total_epochs=2, split_epoch=1
    )
    fit_a, resumed, checkpoint = values[3], values[8], values[10]
    composed = compose_resumed_fit_result(
        checkpoint,
        resumed.continuation_fit_result,
        resumed_max_epochs=2,
    )
    assert composed == fit_a
    assert resumed.continuation_fit_result.config == FitConfig(2, 1, 1)
    assert resumed.checkpoint_next_epoch == 1
    assert resumed.resumed_epochs_completed == 1
    assert set(resumed.checkpoint_data_fingerprints) == {"train", "validation"}


def test_invalid_history_fails_before_restore_without_mutation(monkeypatch, tmp_path):
    values = _continuous_and_resumed(
        monkeypatch, tmp_path, metrics=(3.0, 2.0), total_epochs=2, split_epoch=1
    )
    checkpoint, batch, contexts, configs = values[10:14]
    invalid = replace(checkpoint, fit_history=None)
    model, optimizer, scheduler = _live(configs)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    rng = _rng_snapshot()
    with pytest.raises(ValueError, match="fit_history"):
        run_resumed_fit(
            invalid,
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
            configs,
            resumed_max_epochs=2,
        )
    assert _tree_equal(before_model, model.state_dict())
    assert _tree_equal(before_optimizer, optimizer.state_dict())
    assert _tree_equal(before_scheduler, scheduler.state_dict())
    after_rng = _rng_snapshot()
    assert rng[0] == after_rng[0] and np.array_equal(rng[1][1], after_rng[1][1])
    assert torch.equal(rng[2], after_rng[2])


@pytest.mark.parametrize("failure", ["public_config", "rng_policy"])
def test_invalid_execution_contract_fails_before_restore(
    monkeypatch, tmp_path, failure
):
    values = _continuous_and_resumed(
        monkeypatch, tmp_path, metrics=(3.0, 2.0), total_epochs=2, split_epoch=1
    )
    checkpoint, batch, contexts, configs = values[10:14]
    model, optimizer, scheduler = _live(configs)
    with torch.no_grad():
        model.weight.add_(9.0)
    before = model.weight.detach().clone()
    loss = configs["loss"]
    policy = ResumePolicy()
    if failure == "public_config":
        loss = LossConfig(energy_weight=2.0)
    else:
        policy = ResumePolicy(restore_numpy_rng=False)
    with pytest.raises(ValueError, match="config|RNG restoration"):
        run_resumed_fit(
            checkpoint,
            model,
            optimizer,
            scheduler,
            (batch,),
            (batch,),
            contexts,
            loss,
            configs["train_step"],
            configs["validation_step"],
            configs["scheduler"],
            configs["model_selection"],
            configs,
            resumed_max_epochs=2,
            policy=policy,
        )
    assert torch.equal(model.weight, before)


def test_continuation_failure_reports_partial_progress_without_rollback(
    monkeypatch, tmp_path
):
    values = _continuous_and_resumed(
        monkeypatch, tmp_path, metrics=(3.0, 2.0), total_epochs=2, split_epoch=1
    )
    checkpoint, batch, contexts, configs = values[10:14]
    model, optimizer, scheduler = _live(configs)
    checkpoint_weight = checkpoint.model_state_dict["weight"].clone()

    def fail(*args, **kwargs):
        with torch.no_grad():
            model.weight.add_(0.75)
        raise FitExecutionError(
            phase="train",
            epoch_index=1,
            current_global_step=2,
            completed_epochs=1,
            training_update_completed=False,
            cause=RuntimeError("injected continuation failure"),
        )

    monkeypatch.setattr(resume_fit_module, "run_fit", fail)
    with pytest.raises(ResumedFitExecutionError, match="not rolled back") as caught:
        run_resumed_fit(
            checkpoint,
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
            configs,
            resumed_max_epochs=2,
        )
    assert caught.value.failure_phase == "train"
    assert caught.value.continuation_completed_epochs == 1
    assert not caught.value.rollback_performed
    assert torch.equal(model.weight, checkpoint_weight + 0.75)
