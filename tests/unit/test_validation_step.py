from __future__ import annotations

from dataclasses import fields
import importlib

import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import (
    LossConfig,
    ValidationStepConfig,
    validation_step,
)
from refsite_mlip.transport import TRAIN_FIXED


validation_module = importlib.import_module("refsite_mlip.training.validation")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("fixed", torch.tensor([3.0], dtype=torch.float64))


def _batch(*, energy=True, forces=False, stress=False):
    dtype = torch.float64
    return StructureBatch(
        sample_ids=("validation-tiny",),
        template_ids=("template",),
        template_fingerprints=("1" * 64,),
        positions=torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype),
        atomic_numbers=torch.tensor([6], dtype=torch.long),
        cells=torch.eye(3, dtype=dtype).unsqueeze(0),
        origins=torch.zeros((1, 3), dtype=dtype),
        pbc=torch.ones((1, 3), dtype=torch.bool),
        atom_ptr=torch.tensor([0, 1], dtype=torch.long),
        atom_batch=torch.tensor([0], dtype=torch.long),
        energy=torch.zeros(1, dtype=dtype),
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
    assert not model.training
    assert kwargs["solver_path"] == TRAIN_FIXED
    assert kwargs["create_graph"] is False
    derivatives = kwargs["compute_forces"] or kwargs["compute_stress"]
    assert torch.is_grad_enabled() is derivatives
    if kwargs["compute_forces"]:
        assert batch.positions.is_leaf and batch.positions.requires_grad
    torch.rand((), device=batch.device)
    energy = (model.weight * batch.positions.sum()).reshape(1)
    forces = model.weight * batch.positions if kwargs["compute_forces"] else None
    stress = None
    if kwargs["compute_stress"]:
        stress = model.weight * torch.eye(
            3, dtype=batch.dtype, device=batch.device
        ).unsqueeze(0)
    return type("Prediction", (), {"energy": energy, "forces": forces, "stress": stress})()


def _patch(monkeypatch, function=_executor):
    monkeypatch.setattr(validation_module, "evaluate_structure_batch", function)


def _tensor_snapshot(batch):
    return {
        name: (id(value), value.clone(), value.requires_grad)
        for name, value in vars(batch).items()
        if isinstance(value, torch.Tensor)
    }


def _assert_batch_unchanged(batch, snapshot):
    for name, (identity, value, requires_grad) in snapshot.items():
        current = getattr(batch, name)
        assert id(current) == identity
        assert torch.equal(current, value)
        assert current.requires_grad == requires_grad


def test_energy_only_is_deterministic_and_preserves_mode_state_grad_rng_and_input(monkeypatch):
    _patch(monkeypatch)
    model = TinyModel()
    model.train()
    existing_gradient = torch.tensor(5.0, dtype=torch.float64)
    model.weight.grad = existing_gradient
    parameter = model.weight.detach().clone()
    buffer = model.fixed.clone()
    state_keys = tuple(model.state_dict())
    batch = _batch()
    batch_snapshot = _tensor_snapshot(batch)
    contexts = {"template": object()}
    context_identity = id(contexts["template"])
    rng = torch.random.get_rng_state().clone()

    def forbidden_backward(*args, **kwargs):
        raise AssertionError("validation must not call backward")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    first = validation_step(
        model, batch, contexts, LossConfig(), ValidationStepConfig()
    )
    second = validation_step(
        model, batch, contexts, LossConfig(), ValidationStepConfig()
    )
    assert first == second
    assert first.total_loss == 144.0 and first.energy.valid_count == 1
    assert first.has_supervision and not first.need_forces and not first.need_stress
    assert model.training and tuple(model.state_dict()) == state_keys
    assert torch.equal(model.weight.detach(), parameter)
    assert torch.equal(model.unused.detach(), torch.tensor(1.0))
    assert torch.equal(model.fixed, buffer)
    assert model.weight.grad is existing_gradient and torch.equal(existing_gradient, torch.tensor(5.0))
    assert model.unused.grad is None
    assert torch.equal(torch.random.get_rng_state(), rng)
    _assert_batch_unchanged(batch, batch_snapshot)
    assert id(contexts["template"]) == context_identity
    assert all(not isinstance(getattr(first, field.name), torch.Tensor) for field in fields(first))


def test_force_inside_outer_no_grad_and_eval_mode_restoration(monkeypatch):
    _patch(monkeypatch)
    model = TinyModel()
    model.eval()
    batch = _batch(energy=False, forces=True)
    with torch.no_grad():
        result = validation_step(
            model,
            batch,
            {},
            LossConfig(energy_weight=0.0, force_weight=1.0),
            ValidationStepConfig(),
        )
    assert not model.training
    assert result.need_forces and not result.need_stress
    assert result.force.numerator == 56.0
    assert result.force.denominator == 3.0
    assert result.force.mean == pytest.approx(56.0 / 3.0)


def test_mixed_energy_force_stress_statistics(monkeypatch):
    _patch(monkeypatch)
    result = validation_step(
        TinyModel(),
        _batch(energy=True, forces=True, stress=True),
        {},
        LossConfig(energy_weight=0.5, force_weight=0.25, stress_weight=2.0),
        ValidationStepConfig(),
    )
    assert result.energy.numerator == 144.0
    assert result.force.numerator == 56.0
    assert result.stress.numerator == 12.0
    assert result.stress.denominator == 6.0
    assert result.total_loss == pytest.approx(0.5 * 144.0 + 0.25 * 56.0 / 3.0 + 4.0)


def test_empty_supervision_returns_explicit_zeros(monkeypatch):
    _patch(monkeypatch)
    result = validation_step(
        TinyModel(), _batch(energy=False), {}, LossConfig(), ValidationStepConfig()
    )
    assert not result.has_supervision
    assert result.total_loss == 0.0
    for term in (result.energy, result.force, result.stress):
        assert term.numerator == term.denominator == term.mean == 0.0
        assert term.valid_count == 0
    assert result.sample_ids == ("validation-tiny",)
    assert result.solver_path == TRAIN_FIXED


def test_inference_mode_derivative_request_is_actionable_and_restores_mode(monkeypatch):
    _patch(monkeypatch)
    model = TinyModel()
    model.train()
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="geometry derivatives.*inference_mode"):
            validation_step(
                model,
                _batch(energy=False, forces=True),
                {},
                LossConfig(energy_weight=0.0, force_weight=1.0),
                ValidationStepConfig(),
            )
    assert model.training


def test_nonfinite_and_executor_exception_restore_mode_gradient_and_rng(monkeypatch):
    model = TinyModel()
    model.train()
    gradient = torch.tensor(8.0, dtype=torch.float64)
    model.weight.grad = gradient
    rng = torch.random.get_rng_state().clone()

    def nonfinite(model, batch, template_contexts, **kwargs):
        torch.rand(())
        value = model.weight * model.weight.new_tensor(float("nan"))
        return type("Prediction", (), {"energy": value.reshape(1), "forces": None, "stress": None})()

    _patch(monkeypatch, nonfinite)
    with pytest.raises(ValueError, match="nonfinite energy prediction.*validation-tiny"):
        validation_step(model, _batch(), {}, LossConfig(), ValidationStepConfig())
    assert model.training and model.weight.grad is gradient
    assert torch.equal(torch.random.get_rng_state(), rng)

    def failure(*args, **kwargs):
        torch.rand(())
        raise RuntimeError("executor failed")

    _patch(monkeypatch, failure)
    with pytest.raises(RuntimeError, match="executor failed"):
        validation_step(model, _batch(), {}, LossConfig(), ValidationStepConfig())
    assert model.training and model.weight.grad is gradient
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_validation_config_round_trip_and_rejects_adaptive():
    config = ValidationStepConfig()
    assert ValidationStepConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="TRAIN_FIXED"):
        ValidationStepConfig(solver_path="eval_adaptive")
