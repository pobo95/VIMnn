from __future__ import annotations

import copy
from dataclasses import fields, replace
import importlib
import random

import numpy as np
import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import (
    LossConfig,
    OptimizerConfig,
    TrainStepConfig,
    build_optimizer,
    train_step,
    validate_optimizer_binding,
)
from refsite_mlip.transport import TRAIN_FIXED


step_module = importlib.import_module("refsite_mlip.training.step")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.register_buffer(
            "atomic_baseline", torch.tensor([7.0], dtype=torch.float64)
        )


def _batch(*, energy=True, forces=False, stress=False):
    dtype = torch.float64
    return StructureBatch(
        sample_ids=("tiny",),
        template_ids=("template",),
        template_fingerprints=("0" * 64,),
        positions=torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype),
        atomic_numbers=torch.tensor([6], dtype=torch.long),
        cells=torch.eye(3, dtype=dtype).unsqueeze(0),
        origins=torch.zeros((1, 3), dtype=dtype),
        pbc=torch.ones((1, 3), dtype=torch.bool),
        atom_ptr=torch.tensor([0, 1], dtype=torch.long),
        atom_batch=torch.tensor([0], dtype=torch.long),
        energy=torch.tensor([0.0], dtype=dtype),
        energy_mask=torch.tensor([energy], dtype=torch.bool),
        forces=torch.zeros((1, 3), dtype=dtype),
        force_mask=torch.full((1, 3), forces, dtype=torch.bool),
        stress=torch.zeros((1, 3, 3), dtype=dtype),
        stress_mask=torch.full((1, 3, 3), stress, dtype=torch.bool),
        force_present=torch.tensor([forces], dtype=torch.bool),
        stress_present=torch.tensor([stress], dtype=torch.bool),
        force_mask_provided=torch.tensor([False], dtype=torch.bool),
        stress_mask_provided=torch.tensor([False], dtype=torch.bool),
    )


def _executor(model, batch, template_contexts, **kwargs):
    assert kwargs["solver_path"] == TRAIN_FIXED
    assert kwargs["create_graph"] == (
        kwargs["compute_forces"] or kwargs["compute_stress"]
    )
    if kwargs["compute_forces"]:
        assert batch.positions.is_leaf and batch.positions.requires_grad
    energy = (model.weight * batch.positions.sum()).reshape(1)
    forces = model.weight * batch.positions if kwargs["compute_forces"] else None
    stress = None
    if kwargs["compute_stress"]:
        stress = model.weight * torch.eye(
            3, dtype=batch.dtype, device=batch.device
        ).unsqueeze(0)
    return type("Prediction", (), {"energy": energy, "forces": forces, "stress": stress})()


def _patch_executor(monkeypatch):
    monkeypatch.setattr(step_module, "evaluate_structure_batch", _executor)


def _parameters(optimizer):
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def test_optimizer_contract_config_round_trip_and_rng_preservation():
    model = TinyModel()
    config = OptimizerConfig(
        learning_rate=0.02,
        betas=(0.8, 0.95),
        eps=1.0e-7,
        weight_decay=0.1,
        amsgrad=True,
    )
    assert OptimizerConfig.from_dict(config.to_dict()) == config
    before = torch.random.get_rng_state().clone()
    optimizer = build_optimizer(model, config)
    assert torch.equal(torch.random.get_rng_state(), before)
    parameters = _parameters(optimizer)
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(optimizer.param_groups) == 1
    assert [id(value) for value in parameters] == [id(value) for value in expected]
    assert len({id(value) for value in parameters}) == len(parameters)
    assert all(value is not model.atomic_baseline for value in parameters)
    assert not model.atomic_baseline.requires_grad
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(torch.nn.Identity(), OptimizerConfig())
    with pytest.raises(ValueError, match="optimizer"):
        OptimizerConfig(optimizer="sgd")
    with pytest.raises(ValueError, match="learning_rate"):
        OptimizerConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="betas"):
        OptimizerConfig(betas=(0.9, 1.0))


@pytest.mark.parametrize("mutation", ("missing", "additional", "duplicate", "other"))
def test_optimizer_binding_mismatch_is_rejected_before_any_step_mutation(
    monkeypatch, mutation
):
    model = TinyModel()
    optimizer = build_optimizer(model, OptimizerConfig(weight_decay=0.0))
    foreign = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    parameters = optimizer.param_groups[0]["params"]
    if mutation == "missing":
        parameters.clear()
    elif mutation == "additional":
        parameters.append(foreign)
    elif mutation == "duplicate":
        parameters.append(model.weight)
    else:
        parameters[0] = foreign

    model.eval()
    model.weight.grad = torch.tensor(17.0, dtype=torch.float64)
    gradient_before = model.weight.grad.clone()
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(
        step_module,
        "evaluate_structure_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("binding failure must precede model evaluation")
        ),
    )

    with pytest.raises(ValueError, match="optimizer"):
        train_step(
            model,
            optimizer,
            _batch(),
            {},
            LossConfig(),
            TrainStepConfig(),
        )
    assert not model.training
    assert torch.equal(model.weight.grad, gradient_before)
    assert optimizer.state_dict() == optimizer_before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_optimizer_binding_accepts_exact_current_trainable_identity_order():
    model = TinyModel()
    optimizer = build_optimizer(model, OptimizerConfig())
    assert validate_optimizer_binding(model, optimizer) is None


def test_energy_step_zeroes_stale_gradient_updates_once_and_detaches_result(monkeypatch):
    _patch_executor(monkeypatch)
    model = TinyModel()
    optimizer = build_optimizer(
        model, OptimizerConfig(learning_rate=0.01, weight_decay=0.0)
    )
    batch = _batch(energy=True)
    original_positions = batch.positions.clone()
    original_baseline = model.atomic_baseline.clone()
    model.weight.grad = torch.tensor(999.0, dtype=torch.float64)
    before = model.weight.detach().clone()
    result = train_step(
        model,
        optimizer,
        batch,
        {},
        LossConfig(energy_weight=1.0),
        TrainStepConfig(),
    )
    assert result.pre_clip_grad_norm == pytest.approx(144.0)
    assert result.pre_clip_grad_norm == result.post_clip_grad_norm
    assert not result.clipping_applied and not result.need_forces and not result.need_stress
    assert result.energy.valid_count == 1 and result.force.valid_count == 0
    assert not torch.equal(model.weight.detach(), before)
    assert torch.equal(model.atomic_baseline, original_baseline)
    assert float(optimizer.state[model.weight]["step"]) == 1.0
    assert not batch.positions.requires_grad
    assert torch.equal(batch.positions, original_positions)
    assert all(not isinstance(getattr(result, field.name), torch.Tensor) for field in fields(result))


def test_force_and_stress_demand_are_mask_driven(monkeypatch):
    _patch_executor(monkeypatch)
    model = TinyModel()
    force_optimizer = build_optimizer(model, OptimizerConfig(weight_decay=0.0))
    force_batch = _batch(energy=False, forces=True)
    force = train_step(
        model,
        force_optimizer,
        force_batch,
        {},
        LossConfig(energy_weight=0.0, force_weight=1.0),
        TrainStepConfig(),
    )
    assert force.need_forces and not force.need_stress
    assert force.force.valid_count == 3 and force.number_of_parameters_with_grad == 1
    assert not force_batch.positions.requires_grad

    stress_optimizer = build_optimizer(model, OptimizerConfig(weight_decay=0.0))
    stress = train_step(
        model,
        stress_optimizer,
        _batch(energy=False, stress=True),
        {},
        LossConfig(energy_weight=0.0, stress_weight=1.0),
        TrainStepConfig(),
    )
    assert not stress.need_forces and stress.need_stress
    assert stress.stress.valid_count == 6


def test_empty_supervision_does_not_step_or_create_optimizer_state(monkeypatch):
    _patch_executor(monkeypatch)
    model = TinyModel()
    optimizer = build_optimizer(
        model, OptimizerConfig(learning_rate=0.1, weight_decay=0.5)
    )
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="no weighted valid supervision.*tiny"):
        train_step(
            model,
            optimizer,
            _batch(energy=False),
            {},
            LossConfig(energy_weight=1.0),
            TrainStepConfig(),
        )
    assert torch.equal(model.weight.detach(), before)
    assert optimizer.state == {}


def test_nonfinite_loss_does_not_step(monkeypatch):
    model = TinyModel()
    optimizer = build_optimizer(model, OptimizerConfig(weight_decay=0.0))
    before = model.weight.detach().clone()

    def nonfinite(*args, **kwargs):
        value = model.weight * model.weight.new_tensor(float("nan"))
        return type("Prediction", (), {"energy": value.reshape(1), "forces": None, "stress": None})()

    monkeypatch.setattr(step_module, "evaluate_structure_batch", nonfinite)
    with pytest.raises(ValueError, match="nonfinite energy prediction.*tiny"):
        train_step(model, optimizer, _batch(), {}, LossConfig(), TrainStepConfig())
    assert torch.equal(model.weight.detach(), before) and optimizer.state == {}


def test_nonfinite_gradient_names_parameter_and_does_not_step(monkeypatch):
    _patch_executor(monkeypatch)
    model = TinyModel()
    optimizer = build_optimizer(model, OptimizerConfig(weight_decay=0.0))
    before = model.weight.detach().clone()
    handle = model.weight.register_hook(lambda gradient: gradient * float("nan"))
    try:
        with pytest.raises(ValueError, match="parameter weight.*tiny"):
            train_step(model, optimizer, _batch(), {}, LossConfig(), TrainStepConfig())
    finally:
        handle.remove()
    assert torch.equal(model.weight.detach(), before) and optimizer.state == {}


def test_global_norm_clipping_and_disabled_parity(monkeypatch):
    _patch_executor(monkeypatch)
    clipped_model = TinyModel()
    clipped = train_step(
        clipped_model,
        build_optimizer(clipped_model, OptimizerConfig(weight_decay=0.0)),
        _batch(),
        {},
        LossConfig(),
        TrainStepConfig(gradient_clip_norm=0.25),
    )
    assert clipped.clipping_applied
    assert clipped.pre_clip_grad_norm == pytest.approx(144.0)
    assert clipped.post_clip_grad_norm <= 0.25 + 1.0e-12

    plain_model = TinyModel()
    plain = train_step(
        plain_model,
        build_optimizer(plain_model, OptimizerConfig(weight_decay=0.0)),
        _batch(),
        {},
        LossConfig(),
        TrainStepConfig(),
    )
    assert not plain.clipping_applied
    assert plain.pre_clip_grad_norm == plain.post_clip_grad_norm
    assert TrainStepConfig.from_dict(TrainStepConfig(0.5).to_dict()) == TrainStepConfig(0.5)
    with pytest.raises(ValueError, match="fail_on_nonfinite"):
        TrainStepConfig(fail_on_nonfinite=False)
    with pytest.raises(ValueError, match="TRAIN_FIXED"):
        TrainStepConfig(solver_path="eval_adaptive")
