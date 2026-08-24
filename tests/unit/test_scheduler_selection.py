from __future__ import annotations

import copy
from dataclasses import replace
import math

import pytest
import torch

from refsite_mlip.training import (
    EpochResult,
    EpochTermMetrics,
    ModelSelectionConfig,
    ModelSelectionState,
    SchedulerConfig,
    ValidationDecision,
    build_scheduler,
    process_primary_validation,
)


def _term(value: float, denominator: float = 1.0) -> EpochTermMetrics:
    numerator = value * denominator if denominator > 0 else 0.0
    return EpochTermMetrics(numerator, denominator, value, int(denominator))


def _event(
    metric: float,
    *,
    epoch: int = 0,
    step: int = 1,
    phase: str = "validation",
    semantics: str = "fixed_model_validation",
    supervised: bool = True,
    energy_denominator: float = 1.0,
) -> EpochResult:
    return EpochResult(
        energy=_term(metric, energy_denominator),
        force=_term(metric + 1.0),
        stress=_term(metric + 2.0),
        total_loss=metric + 3.0,
        has_supervision=supervised,
        phase=phase,
        epoch_index=epoch,
        global_step_start=step,
        global_step_end=step,
        number_of_batches=1,
        number_of_supervised_batches=int(supervised),
        number_of_structures=1,
        number_of_atoms=2,
        successful_optimizer_steps=0,
        ordered_batch_sample_ids=(("sample",),),
        metric_semantics=semantics,
    )


def _objects(scheduler_config=None, selection_config=None):
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler_config = scheduler_config or SchedulerConfig()
    selection_config = selection_config or ModelSelectionConfig()
    scheduler = build_scheduler(optimizer, scheduler_config)
    return parameter, optimizer, scheduler, scheduler_config, selection_config


def _process(objects, state, metric, epoch):
    _, optimizer, scheduler, scheduler_config, selection_config = objects
    return process_primary_validation(
        optimizer,
        scheduler,
        _event(metric, epoch=epoch, step=epoch + 1),
        scheduler_config,
        selection_config,
        state,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "bad"},
        {"factor": 0.0},
        {"factor": 1.0},
        {"patience": -1},
        {"patience": True},
        {"threshold": -1.0},
        {"threshold_mode": "bad"},
        {"cooldown": -1},
        {"min_lr": -1.0},
        {"eps": 0.0},
    ],
)
def test_scheduler_config_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SchedulerConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"monitor": "bad"},
        {"mode": "bad"},
        {"min_delta": -1.0},
        {"early_stopping_patience": -1},
        {"early_stopping_patience": True},
    ],
)
def test_selection_config_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ModelSelectionConfig(**kwargs)


def test_config_state_and_decision_serialization():
    scheduler_config = SchedulerConfig(
        "reduce_on_plateau", "force", "max", 0.5, 2, 0.01, "abs", 1, 0.001
    )
    selection_config = ModelSelectionConfig("force", "max", 0.1, 3)
    assert SchedulerConfig.from_dict(scheduler_config.to_dict()) == scheduler_config
    assert ModelSelectionConfig.from_dict(selection_config.to_dict()) == selection_config
    objects = _objects(scheduler_config, selection_config)
    state, decision = _process(objects, ModelSelectionState(), 2.0, 0)
    assert ModelSelectionState.from_dict(state.to_dict()) == state
    assert ValidationDecision.from_dict(decision.to_dict()) == decision


@pytest.mark.parametrize(
    ("monitor", "expected"),
    [("total_loss", 5.0), ("energy", 2.0), ("force", 3.0), ("stress", 4.0)],
)
def test_monitored_metric_extraction(monitor, expected):
    scheduler_config = SchedulerConfig(monitor=monitor)
    selection_config = ModelSelectionConfig(monitor=monitor)
    objects = _objects(scheduler_config, selection_config)
    state, decision = _process(objects, ModelSelectionState(), 2.0, 0)
    assert decision.metric_value == expected
    assert decision.is_best
    assert not decision.should_stop
    assert state.best_metric == expected


def test_min_and_max_improvement_and_absolute_delta_boundary():
    minimum = _objects(
        SchedulerConfig(mode="min"), ModelSelectionConfig(mode="min", min_delta=0.1)
    )
    state, _ = _process(minimum, ModelSelectionState(), 2.0, 0)
    state, tie = _process(minimum, state, 1.9, 1)
    assert not tie.is_best
    state, improved = _process(minimum, state, 1.899, 2)
    assert improved.is_best and state.epochs_since_improvement == 0

    maximum = _objects(
        SchedulerConfig(mode="max"), ModelSelectionConfig(mode="max", min_delta=0.1)
    )
    state, _ = _process(maximum, ModelSelectionState(), 2.0, 0)
    state, tie = _process(maximum, state, 2.1, 1)
    assert not tie.is_best
    state, improved = _process(maximum, state, 2.101, 2)
    assert improved.is_best and state.best_metric == pytest.approx(5.101)


@pytest.mark.parametrize("patience, stop_bad_event", [(0, 1), (1, 1), (2, 2)])
def test_early_stopping_patience_semantics(patience, stop_bad_event):
    objects = _objects(
        SchedulerConfig(), ModelSelectionConfig(early_stopping_patience=patience)
    )
    state, decision = _process(objects, ModelSelectionState(), 1.0, 0)
    assert decision.is_best and not decision.should_stop
    for bad_index in range(1, stop_bad_event + 1):
        state, decision = _process(objects, state, 2.0, bad_index)
        assert decision.should_stop == (bad_index == stop_bad_event)
    assert state.stop_epoch == stop_bad_event


def test_improvement_resets_counter():
    objects = _objects(
        SchedulerConfig(), ModelSelectionConfig(early_stopping_patience=3)
    )
    state, _ = _process(objects, ModelSelectionState(), 2.0, 0)
    state, _ = _process(objects, state, 3.0, 1)
    assert state.epochs_since_improvement == 1
    state, decision = _process(objects, state, 1.0, 2)
    assert decision.is_best and state.epochs_since_improvement == 0


@pytest.mark.parametrize(
    "event",
    [
        _event(1.0, supervised=False),
        _event(1.0, phase="train", semantics="pre_update_batch_observations"),
        _event(float("nan")),
        _event(1.0, energy_denominator=0.0),
    ],
)
def test_invalid_event_is_transactional(event):
    scheduler_config = SchedulerConfig(
        kind="reduce_on_plateau", monitor="energy", patience=0
    )
    selection_config = ModelSelectionConfig(monitor="energy")
    parameter, optimizer, scheduler, _, _ = _objects(
        scheduler_config, selection_config
    )
    state = ModelSelectionState()
    parameter_before = parameter.detach().clone()
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError):
        process_primary_validation(
            optimizer,
            scheduler,
            event,
            scheduler_config,
            selection_config,
            state,
        )
    assert torch.equal(parameter, parameter_before)
    assert scheduler.state_dict() == scheduler_before
    assert optimizer.state_dict() == optimizer_before
    assert state == ModelSelectionState()


def test_duplicate_and_stale_events_are_rejected_before_step():
    objects = _objects()
    state, _ = _process(objects, ModelSelectionState(), 1.0, 2)
    _, optimizer, scheduler, scheduler_config, selection_config = objects
    before = copy.deepcopy(scheduler.state_dict())
    for event in (_event(0.5, epoch=2, step=3), _event(0.5, epoch=3, step=2)):
        with pytest.raises(ValueError, match="duplicate|stale|order"):
            process_primary_validation(
                optimizer,
                scheduler,
                event,
                scheduler_config,
                selection_config,
                state,
            )
        assert scheduler.state_dict() == before


def test_none_scheduler_steps_once_per_event_without_lr_change():
    objects = _objects()
    state = ModelSelectionState()
    decisions = []
    for epoch, metric in enumerate((1.0, 0.9, 1.1)):
        state, decision = _process(objects, state, metric, epoch)
        decisions.append(decision)
    _, optimizer, scheduler, _, _ = objects
    assert scheduler.state_dict() == {"validation_steps": 3}
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert all(item.scheduler_stepped for item in decisions)
    assert not any(item.learning_rate_changed for item in decisions)


def test_reduce_on_plateau_factor_patience_and_min_lr():
    scheduler_config = SchedulerConfig(
        kind="reduce_on_plateau",
        factor=0.5,
        patience=0,
        threshold=0.0,
        min_lr=0.025,
    )
    objects = _objects(scheduler_config, ModelSelectionConfig())
    state = ModelSelectionState()
    rates = []
    changes = []
    for epoch, metric in enumerate((1.0, 2.0, 3.0, 4.0)):
        state, decision = _process(objects, state, metric, epoch)
        rates.append(decision.learning_rates_after[0])
        changes.append(decision.learning_rate_changed)
    assert rates == pytest.approx([0.1, 0.05, 0.025, 0.025])
    assert changes == [False, True, True, False]


def test_parameter_and_rng_are_unchanged():
    objects = _objects(SchedulerConfig(kind="reduce_on_plateau"))
    parameter = objects[0]
    parameter_before = parameter.detach().clone()
    rng_before = torch.random.get_rng_state().clone()
    _process(objects, ModelSelectionState(), 1.0, 0)
    assert torch.equal(parameter, parameter_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def _run_sequence(metrics):
    config = SchedulerConfig(
        kind="reduce_on_plateau", factor=0.5, patience=1, threshold=0.0
    )
    objects = _objects(config, ModelSelectionConfig())
    state = ModelSelectionState()
    decisions = []
    for epoch, metric in enumerate(metrics):
        state, decision = _process(objects, state, metric, epoch)
        decisions.append(decision)
    return objects[1].state_dict(), objects[2].state_dict(), state, decisions


def test_deterministic_metric_sequence_exact_repeat():
    first = _run_sequence((1.0, 1.1, 1.2, 0.8))
    second = _run_sequence((1.0, 1.1, 1.2, 0.8))
    assert first == second


def test_scheduler_and_selection_mid_sequence_round_trip():
    config = SchedulerConfig(
        kind="reduce_on_plateau", factor=0.5, patience=1, threshold=0.0
    )
    continuous = _objects(config, ModelSelectionConfig())
    state = ModelSelectionState()
    for epoch, metric in enumerate((1.0, 1.1)):
        state, _ = _process(continuous, state, metric, epoch)
    saved_scheduler = copy.deepcopy(continuous[2].state_dict())
    saved_state = state.to_dict()
    saved_lr = continuous[1].param_groups[0]["lr"]

    resumed = _objects(config, ModelSelectionConfig())
    resumed[1].param_groups[0]["lr"] = saved_lr
    resumed[2].load_state_dict(saved_scheduler)
    resumed_state = ModelSelectionState.from_dict(saved_state)
    continuous_decisions = []
    resumed_decisions = []
    for epoch, metric in ((2, 1.2), (3, 0.8)):
        state, decision = _process(continuous, state, metric, epoch)
        continuous_decisions.append(decision)
        resumed_state, decision = _process(resumed, resumed_state, metric, epoch)
        resumed_decisions.append(decision)
    assert resumed_state == state
    assert resumed_decisions == continuous_decisions
    assert resumed[1].param_groups[0]["lr"] == continuous[1].param_groups[0]["lr"]
    assert resumed[2].state_dict() == continuous[2].state_dict()


def test_scheduler_and_selection_monitor_mode_must_match():
    objects = _objects()
    with pytest.raises(ValueError, match="same metric"):
        process_primary_validation(
            objects[1],
            objects[2],
            _event(1.0),
            objects[3],
            ModelSelectionConfig(monitor="energy"),
            ModelSelectionState(),
        )
    with pytest.raises(ValueError, match="same mode"):
        process_primary_validation(
            objects[1],
            objects[2],
            _event(1.0),
            objects[3],
            ModelSelectionConfig(mode="max"),
            ModelSelectionState(),
        )
