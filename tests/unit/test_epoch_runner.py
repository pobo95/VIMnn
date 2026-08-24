from __future__ import annotations

from dataclasses import fields
import importlib

import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import (
    EpochResult,
    LossConfig,
    TrainStepConfig,
    TrainStepResult,
    TrainStepTermResult,
    ValidationStepConfig,
    ValidationStepResult,
    ValidationTermResult,
    run_training_epoch,
    run_validation_epoch,
)


epoch_module = importlib.import_module("refsite_mlip.training.epoch")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))


def _batch(sample_id, *, atoms=1):
    dtype = torch.float64
    return StructureBatch(
        sample_ids=(sample_id,),
        template_ids=("template",),
        template_fingerprints=("2" * 64,),
        positions=torch.zeros((atoms, 3), dtype=dtype),
        atomic_numbers=torch.full((atoms,), 6, dtype=torch.long),
        cells=torch.eye(3, dtype=dtype).unsqueeze(0),
        origins=torch.zeros((1, 3), dtype=dtype),
        pbc=torch.ones((1, 3), dtype=torch.bool),
        atom_ptr=torch.tensor([0, atoms], dtype=torch.long),
        atom_batch=torch.zeros(atoms, dtype=torch.long),
        energy=torch.zeros(1, dtype=dtype),
        energy_mask=torch.ones(1, dtype=torch.bool),
        forces=torch.zeros((atoms, 3), dtype=dtype),
        force_mask=torch.zeros((atoms, 3), dtype=torch.bool),
        stress=torch.zeros((1, 3, 3), dtype=dtype),
        stress_mask=torch.zeros((1, 3, 3), dtype=torch.bool),
        force_present=torch.zeros(1, dtype=torch.bool),
        stress_present=torch.zeros(1, dtype=torch.bool),
        force_mask_provided=torch.zeros(1, dtype=torch.bool),
        stress_mask_provided=torch.zeros(1, dtype=torch.bool),
    )


def _term(kind, numerator, denominator, valid_count):
    mean = numerator / denominator if denominator else 0.0
    cls = TrainStepTermResult if kind == "train" else ValidationTermResult
    return cls(float(numerator), float(denominator), float(mean), int(valid_count))


def _result(kind, batch, energy, force=(0, 0, 0), stress=(0, 0, 0), supervised=True):
    e = _term(kind, *energy)
    f = _term(kind, *force)
    s = _term(kind, *stress)
    common = dict(
        total_loss=e.mean + f.mean + s.mean,
        energy_loss=e.mean,
        force_loss=f.mean,
        stress_loss=s.mean,
        energy=e,
        force=f,
        stress=s,
        sample_ids=batch.sample_ids,
    )
    if kind == "train":
        return TrainStepResult(
            **common,
            pre_clip_grad_norm=1.0,
            post_clip_grad_norm=1.0,
            clipping_applied=False,
            number_of_parameters_with_grad=1,
            need_forces=force[2] > 0,
            need_stress=stress[2] > 0,
        )
    return ValidationStepResult(
        **common,
        has_supervision=supervised,
        need_forces=force[2] > 0,
        need_stress=stress[2] > 0,
        solver_path="train_fixed",
    )


def test_unequal_denominator_aggregation_is_not_batch_mean(monkeypatch):
    batches = (_batch("first", atoms=1), _batch("second", atoms=2))
    results = {
        "first": _result("validation", batches[0], (1, 1, 1), (4, 2, 2), (9, 3, 3)),
        "second": _result("validation", batches[1], (9, 3, 3), (20, 4, 4), (42, 6, 6)),
    }
    monkeypatch.setattr(
        epoch_module,
        "validation_step",
        lambda model, batch, *args: results[batch.sample_ids[0]],
    )
    config = LossConfig(energy_weight=1.0, force_weight=2.0, stress_weight=3.0)
    epoch = run_validation_epoch(
        TinyModel(), batches, {}, config, ValidationStepConfig(), epoch_index=4, global_step=11
    )
    assert epoch.energy.numerator == 10.0 and epoch.energy.denominator == 4.0
    assert epoch.energy.mean == 2.5
    assert epoch.energy.mean != (1.0 + 3.0) / 2.0
    assert epoch.force.mean == 24.0 / 6.0
    assert epoch.stress.mean == 51.0 / 9.0
    assert epoch.total_loss == pytest.approx(2.5 + 2.0 * 4.0 + 3.0 * 51.0 / 9.0)
    assert epoch.global_step_start == epoch.global_step_end == 11
    assert epoch.number_of_structures == 2 and epoch.number_of_atoms == 3
    assert epoch.ordered_batch_sample_ids == (("first",), ("second",))


def test_full_vs_split_validation_and_unlabeled_exclusion(monkeypatch):
    full = _batch("full", atoms=3)
    first = _batch("part-a", atoms=1)
    second = _batch("part-b", atoms=2)
    unlabeled = _batch("unlabeled", atoms=4)
    results = {
        "full": _result("validation", full, (10, 4, 4), (6, 3, 3)),
        "part-a": _result("validation", first, (1, 1, 1), (2, 1, 1)),
        "part-b": _result("validation", second, (9, 3, 3), (4, 2, 2)),
        "unlabeled": _result("validation", unlabeled, (0, 0, 0), supervised=False),
    }
    monkeypatch.setattr(
        epoch_module,
        "validation_step",
        lambda model, batch, *args: results[batch.sample_ids[0]],
    )
    config = LossConfig(energy_weight=1.0, force_weight=1.0)
    one = run_validation_epoch(
        TinyModel(), (full,), {}, config, ValidationStepConfig(), epoch_index=0, global_step=0
    )
    split = run_validation_epoch(
        TinyModel(), (first, unlabeled, second), {}, config, ValidationStepConfig(), epoch_index=0, global_step=0
    )
    assert one.energy == split.energy and one.force == split.force
    assert one.total_loss == split.total_loss
    assert split.number_of_supervised_batches == 2
    empty = run_validation_epoch(
        TinyModel(), (unlabeled,), {}, config, ValidationStepConfig(), epoch_index=0, global_step=0
    )
    assert not empty.has_supervision and empty.total_loss == 0.0
    assert empty.energy.denominator == empty.force.denominator == 0.0


def test_training_calls_each_batch_once_and_tracks_global_step(monkeypatch):
    batches = (_batch("a"), _batch("b"), _batch("c"))
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    calls = []

    def step(model, optimizer, batch, *args):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model.weight.square().backward()
        optimizer.step()
        calls.append(batch.sample_ids)
        return _result("train", batch, (1, 1, 1))

    monkeypatch.setattr(epoch_module, "train_step", step)
    result = run_training_epoch(
        model,
        optimizer,
        batches,
        {},
        LossConfig(),
        TrainStepConfig(),
        epoch_index=2,
        global_step_start=7,
    )
    assert calls == [("a",), ("b",), ("c",)]
    assert result.global_step_start == 7 and result.global_step_end == 10
    assert result.successful_optimizer_steps == 3
    assert result.number_of_batches == result.number_of_supervised_batches == 3
    assert float(optimizer.state[model.weight]["step"]) == 3.0
    assert model.training
    assert result.metric_semantics == "pre_update_batch_observations"


def test_empty_or_nonsequence_batches_fail_fast():
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="must not be empty"):
        run_training_epoch(model, optimizer, (), {}, LossConfig(), TrainStepConfig(), epoch_index=0, global_step_start=0)
    with pytest.raises(ValueError, match="must not be empty"):
        run_validation_epoch(model, [], {}, LossConfig(), ValidationStepConfig(), epoch_index=0, global_step=0)
    with pytest.raises(TypeError, match="deterministic Sequence"):
        run_validation_epoch(model, iter((_batch("a"),)), {}, LossConfig(), ValidationStepConfig(), epoch_index=0, global_step=0)


def test_training_failure_reports_partial_progress_without_rollback(monkeypatch):
    batches = (_batch("completed"), _batch("failing"), _batch("unreached"))
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    before = model.weight.detach().clone()

    def step(model, optimizer, batch, *args):
        if batch.sample_ids == ("failing",):
            raise ValueError("no weighted valid supervision")
        optimizer.zero_grad(set_to_none=True)
        model.weight.square().backward()
        optimizer.step()
        return _result("train", batch, (1, 1, 1))

    monkeypatch.setattr(epoch_module, "train_step", step)
    with pytest.raises(
        RuntimeError,
        match=r"epoch_index=3, batch_index=1, sample_ids=\('failing',\), successful_optimizer_steps=1",
    ):
        run_training_epoch(model, optimizer, batches, {}, LossConfig(), TrainStepConfig(), epoch_index=3, global_step_start=9)
    assert not torch.equal(model.weight.detach(), before)
    assert float(optimizer.state[model.weight]["step"]) == 1.0


def test_epoch_result_serialization_and_graph_free(monkeypatch):
    batch = _batch("serial")
    result = _result("validation", batch, (2, 4, 4))
    monkeypatch.setattr(epoch_module, "validation_step", lambda *args: result)
    epoch = run_validation_epoch(
        TinyModel(), (batch,), {}, LossConfig(), ValidationStepConfig(), epoch_index=5, global_step=13
    )
    assert EpochResult.from_dict(epoch.to_dict()) == epoch
    assert all(not isinstance(getattr(epoch, field.name), torch.Tensor) for field in fields(epoch))
    assert epoch.metric_semantics == "fixed_model_validation"
