from __future__ import annotations

import copy
from dataclasses import fields, replace
import importlib

import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import (
    EpochResult,
    EpochTermMetrics,
    FitConfig,
    FitExecutionError,
    FitResult,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_scheduler,
    run_fit,
)


fit_module = importlib.import_module("refsite_mlip.training.fit")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))


def _batch(sample_id="sample", *, labeled=True):
    dtype = torch.float64
    return StructureBatch(
        sample_ids=(sample_id,),
        template_ids=("template",),
        template_fingerprints=("2" * 64,),
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


def _install_runners(monkeypatch, validation_metrics, *, calls=None):
    calls = [] if calls is None else calls
    metrics = iter(validation_metrics)

    def training(model, optimizer, batches, *args, epoch_index, global_step_start):
        calls.append(("train", epoch_index, global_step_start, optimizer.param_groups[0]["lr"]))
        for _ in batches:
            optimizer.zero_grad(set_to_none=True)
            model.weight.square().backward()
            optimizer.step()
        return _epoch("train", epoch_index, global_step_start, 1.0, batches)

    def validation(model, batches, *args, epoch_index, global_step):
        calls.append(("validation", epoch_index, global_step, optimizer_lr(model)))
        return _epoch("validation", epoch_index, global_step, next(metrics), batches)

    def optimizer_lr(model):
        return calls[-1][3] if calls and calls[-1][0] == "train" else 0.0

    monkeypatch.setattr(fit_module, "run_training_epoch", training)
    monkeypatch.setattr(fit_module, "run_validation_epoch", validation)
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    return calls


def _run(monkeypatch, metrics, *, fit_config=None, scheduler_config=None, selection_config=None, state=None, batches=None):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    scheduler_config = scheduler_config or SchedulerConfig()
    selection_config = selection_config or ModelSelectionConfig()
    scheduler = build_scheduler(optimizer, scheduler_config)
    batches = batches or (_batch(),)
    calls = _install_runners(monkeypatch, metrics)
    result = run_fit(
        model,
        optimizer,
        scheduler,
        batches,
        batches,
        {},
        LossConfig(),
        TrainStepConfig(),
        ValidationStepConfig(),
        scheduler_config,
        selection_config,
        state or ModelSelectionState(),
        fit_config or FitConfig(len(metrics)),
    )
    return model, optimizer, scheduler, result, calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_epochs": 0},
        {"max_epochs": True},
        {"max_epochs": 2, "start_epoch": -1},
        {"max_epochs": 2, "global_step_start": -1},
        {"max_epochs": 2, "start_epoch": 2},
    ],
)
def test_fit_config_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FitConfig(**kwargs)


def test_three_epoch_order_global_steps_and_validation_once(monkeypatch):
    batches = (_batch("a"), _batch("b"))
    _, optimizer, scheduler, result, calls = _run(
        monkeypatch, (3.0, 2.0, 1.0), batches=batches
    )
    assert [(call[0], call[1], call[2]) for call in calls] == [
        ("train", 0, 0),
        ("validation", 0, 2),
        ("train", 1, 2),
        ("validation", 1, 4),
        ("train", 2, 4),
        ("validation", 2, 6),
    ]
    assert result.epochs_requested == result.epochs_completed == 3
    assert result.global_step_start == 0 and result.global_step_end == 6
    assert result.next_epoch == 3
    assert result.best_epoch == 2 and result.best_global_step == 6
    assert result.best_metric == 1.0 and result.terminal_model_is_best
    assert scheduler.state_dict() == {"validation_steps": 3}
    assert float(next(iter(optimizer.state.values()))["step"]) == 6.0


def test_plateau_lr_change_is_used_by_next_training_epoch(monkeypatch):
    scheduler_config = SchedulerConfig(
        kind="reduce_on_plateau", factor=0.5, patience=0, threshold=0.0
    )
    _, _, _, result, _ = _run(
        monkeypatch,
        (1.0, 2.0, 3.0),
        scheduler_config=scheduler_config,
    )
    assert [record.learning_rates_used_for_training for record in result.records] == [
        (0.1,),
        (0.1,),
        (0.05,),
    ]
    assert [record.learning_rates_after_validation for record in result.records] == [
        (0.1,),
        (0.05,),
        (0.025,),
    ]


def test_early_stop_and_terminal_best_semantics(monkeypatch):
    selection = ModelSelectionConfig(early_stopping_patience=2)
    _, _, scheduler, stopped, _ = _run(
        monkeypatch,
        (1.0, 2.0, 3.0, 4.0),
        selection_config=selection,
        fit_config=FitConfig(4),
    )
    assert stopped.epochs_completed == 3
    assert stopped.stopped_early and stopped.stop_epoch == 2
    assert not stopped.terminal_model_is_best
    assert scheduler.state_dict() == {"validation_steps": 3}

    _, _, _, full, _ = _run(monkeypatch, (3.0, 2.0, 1.0))
    assert not full.stopped_early and full.epochs_completed == 3
    assert full.terminal_model_is_best


def test_continuation_from_initial_selection_state(monkeypatch):
    initial = ModelSelectionState(
        best_metric=2.0,
        best_epoch=1,
        best_global_step=5,
        validation_events=2,
        last_validation_epoch=1,
        last_validation_global_step=5,
    )
    _, _, _, result, calls = _run(
        monkeypatch,
        (1.5, 1.0),
        state=initial,
        fit_config=FitConfig(4, start_epoch=2, global_step_start=5),
    )
    assert result.start_epoch == 2 and result.next_epoch == 4
    assert result.global_step_start == 5 and result.global_step_end == 7
    assert [record.epoch_index for record in result.records] == [2, 3]
    assert result.final_selection_state.validation_events == 4
    assert calls[0][:3] == ("train", 2, 5)


def test_invalid_start_state_fails_before_parameter_change(monkeypatch):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler_config = SchedulerConfig()
    scheduler = build_scheduler(optimizer, scheduler_config)
    state = ModelSelectionState(
        best_metric=1.0,
        best_epoch=2,
        best_global_step=5,
        validation_events=1,
        last_validation_epoch=2,
        last_validation_global_step=5,
    )
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="start_epoch"):
        run_fit(
            model, optimizer, scheduler, (_batch(),), (_batch(),), {},
            LossConfig(), TrainStepConfig(), ValidationStepConfig(),
            scheduler_config, ModelSelectionConfig(), state,
            FitConfig(4, start_epoch=2, global_step_start=5),
        )
    assert torch.equal(model.weight, before)
    assert optimizer.state == {}
    assert scheduler.state_dict() == {"validation_steps": 0}


def test_fit_rejects_optimizer_from_another_model_before_epoch_or_mode_change(
    monkeypatch,
):
    model = TinyModel()
    foreign = TinyModel()
    optimizer = torch.optim.AdamW(foreign.parameters(), lr=0.1)
    config = SchedulerConfig()
    scheduler = build_scheduler(optimizer, config)
    model.eval()
    model.weight.grad = torch.tensor(9.0, dtype=torch.float64)
    before = model.weight.grad.clone()
    monkeypatch.setattr(
        fit_module,
        "run_training_epoch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("optimizer preflight must precede epochs")
        ),
    )
    with pytest.raises(ValueError, match="optimizer parameters"):
        run_fit(
            model,
            optimizer,
            scheduler,
            (_batch(),),
            (_batch(),),
            {},
            LossConfig(),
            TrainStepConfig(),
            ValidationStepConfig(),
            config,
            ModelSelectionConfig(),
            ModelSelectionState(),
            FitConfig(1),
        )
    assert not model.training
    assert torch.equal(model.weight.grad, before)
    assert optimizer.state == {}


def test_empty_generator_and_missing_context_preflight():
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    config = SchedulerConfig()
    scheduler = build_scheduler(optimizer, config)
    common = (
        model, optimizer, scheduler, (_batch(),), (_batch(),), {}, LossConfig(),
        TrainStepConfig(), ValidationStepConfig(), config,
        ModelSelectionConfig(), ModelSelectionState(), FitConfig(1),
    )
    with pytest.raises(ValueError, match="must not be empty"):
        run_fit(*common[:3], (), *common[4:])
    with pytest.raises(TypeError, match="deterministic Sequence"):
        run_fit(*common[:3], iter((_batch(),)), *common[4:])
    with pytest.raises(KeyError, match="missing TemplateExecutionContext"):
        run_fit(*common)


def test_fingerprint_failure_is_preflight_and_transactional(monkeypatch):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    config = SchedulerConfig()
    scheduler = build_scheduler(optimizer, config)
    monkeypatch.setattr(
        fit_module,
        "_validated_context",
        lambda *args: (_ for _ in ()).throw(ValueError("fingerprint mismatch")),
    )
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="fingerprint"):
        run_fit(
            model, optimizer, scheduler, (_batch(),), (_batch(),), {"template": object()},
            LossConfig(), TrainStepConfig(), ValidationStepConfig(), config,
            ModelSelectionConfig(), ModelSelectionState(), FitConfig(1),
        )
    assert torch.equal(model.weight, before) and optimizer.state == {}
    assert scheduler.state_dict() == {"validation_steps": 0}


@pytest.mark.parametrize(
    ("loss_config", "selection_config", "message"),
    [
        (LossConfig(energy_weight=0.0), ModelSelectionConfig(), "positive loss weight"),
        (LossConfig(), ModelSelectionConfig(monitor="force"), "force_weight"),
    ],
)
def test_monitored_weight_preflight(monkeypatch, loss_config, selection_config, message):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    config = SchedulerConfig(monitor=selection_config.monitor)
    scheduler = build_scheduler(optimizer, config)
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    with pytest.raises(ValueError, match=message):
        run_fit(
            model, optimizer, scheduler, (_batch(),), (_batch(),), {}, loss_config,
            TrainStepConfig(), ValidationStepConfig(), config, selection_config,
            ModelSelectionState(), FitConfig(1),
        )


def test_missing_monitored_validation_label_preflight(monkeypatch):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    config = SchedulerConfig()
    scheduler = build_scheduler(optimizer, config)
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    with pytest.raises(ValueError, match="no positive-weight validation labels"):
        run_fit(
            model, optimizer, scheduler, (_batch(),), (_batch(labeled=False),), {},
            LossConfig(), TrainStepConfig(), ValidationStepConfig(), config,
            ModelSelectionConfig(), ModelSelectionState(), FitConfig(1),
        )


def _base_for_failure(monkeypatch):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    config = SchedulerConfig()
    scheduler = build_scheduler(optimizer, config)
    monkeypatch.setattr(fit_module, "_validate_batch_contexts", lambda *args: None)
    return model, optimizer, scheduler, config


def test_training_failure_skips_validation_and_reports_partial_progress(monkeypatch):
    model, optimizer, scheduler, config = _base_for_failure(monkeypatch)
    validation_calls = []
    monkeypatch.setattr(
        fit_module,
        "run_training_epoch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("successful_optimizer_steps=1; broken train")
        ),
    )
    monkeypatch.setattr(
        fit_module, "run_validation_epoch", lambda *args, **kwargs: validation_calls.append(1)
    )
    with pytest.raises(FitExecutionError) as caught:
        run_fit(
            model, optimizer, scheduler, (_batch(),), (_batch(),), {}, LossConfig(),
            TrainStepConfig(), ValidationStepConfig(), config,
            ModelSelectionConfig(), ModelSelectionState(), FitConfig(2, global_step_start=4),
        )
    error = caught.value
    assert error.phase == "train" and error.current_global_step == 5
    assert error.completed_epochs == 0 and not error.training_update_completed
    assert not error.rollback_performed and validation_calls == []
    assert scheduler.state_dict() == {"validation_steps": 0}


def test_validation_failure_retains_training_update_and_skips_selection(monkeypatch):
    model, optimizer, scheduler, config = _base_for_failure(monkeypatch)
    before = model.weight.detach().clone()

    def training(model, optimizer, batches, *args, epoch_index, global_step_start):
        optimizer.zero_grad(set_to_none=True)
        model.weight.square().backward()
        optimizer.step()
        return _epoch("train", epoch_index, global_step_start, 1.0, batches)

    monkeypatch.setattr(fit_module, "run_training_epoch", training)
    monkeypatch.setattr(
        fit_module,
        "run_validation_epoch",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("broken validation")),
    )
    with pytest.raises(FitExecutionError) as caught:
        run_fit(
            model, optimizer, scheduler, (_batch(),), (_batch(),), {}, LossConfig(),
            TrainStepConfig(), ValidationStepConfig(), config,
            ModelSelectionConfig(), ModelSelectionState(), FitConfig(1),
        )
    assert caught.value.phase == "validation"
    assert caught.value.training_update_completed
    assert caught.value.current_global_step == 1
    assert not torch.equal(model.weight, before)
    assert scheduler.state_dict() == {"validation_steps": 0}


def test_selection_failure_does_not_mutate_scheduler(monkeypatch):
    model, optimizer, scheduler, config = _base_for_failure(monkeypatch)
    calls = _install_runners(monkeypatch, (1.0,))
    original_validation = fit_module.run_validation_epoch

    def invalid_validation(*args, **kwargs):
        return replace(original_validation(*args, **kwargs), metric_semantics="invalid")

    monkeypatch.setattr(fit_module, "run_validation_epoch", invalid_validation)
    with pytest.raises(FitExecutionError) as caught:
        run_fit(
            model, optimizer, scheduler, (_batch(),), (_batch(),), {}, LossConfig(),
            TrainStepConfig(), ValidationStepConfig(), config,
            ModelSelectionConfig(), ModelSelectionState(), FitConfig(1),
        )
    assert caught.value.phase == "selection"
    assert caught.value.training_update_completed
    assert scheduler.state_dict() == {"validation_steps": 0}
    assert [call[0] for call in calls] == ["train", "validation"]


def _assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_deterministic_repeat_and_serialization(monkeypatch):
    snapshots = []
    for _ in range(2):
        model, optimizer, scheduler, result, calls = _run(
            monkeypatch, (2.0, 1.5, 1.0)
        )
        snapshots.append(
            (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(scheduler.state_dict()),
                result,
                calls,
            )
        )
    _assert_nested_equal(snapshots[0], snapshots[1])
    result = snapshots[0][3]
    assert FitConfig.from_dict(result.config.to_dict()) == result.config
    assert FitResult.from_dict(result.to_dict()) == result
    assert all(
        not isinstance(getattr(result, field.name), torch.Tensor)
        for field in fields(result)
    )
