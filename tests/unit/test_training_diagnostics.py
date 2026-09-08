from types import SimpleNamespace

import pytest
import torch
from torch import nn

from refsite_mlip.training import diagnostics
from refsite_mlip.training.losses import LossConfig, LossTerm, PotentialLossOutput


def setup_case(monkeypatch):
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(1, 1, bias=False).double()])
    model.unused = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    with torch.no_grad():
        model.layers[0].weight.fill_(2.0)
    model.eval()
    model.layers[0].train()
    batch = SimpleNamespace(sample_ids=('test',))
    monkeypatch.setattr(diagnostics, '_has_force_supervision', lambda *a: True)
    monkeypatch.setattr(diagnostics, '_has_stress_supervision', lambda *a: True)
    monkeypatch.setattr(diagnostics, '_step_batch', lambda batch, **kw: batch)

    def evaluate(model, *args, **kwargs):
        assert kwargs['create_graph']
        torch.rand(2)  # diagnostics must restore the caller's RNG
        return model.layers[0](torch.tensor([[3.0]], dtype=torch.float64)).sum()

    def loss(prediction, *args):
        def term(value):
            return LossTerm(value, torch.tensor(1), value, torch.tensor(1))
        e, f, s = prediction.square(), -prediction, 2 * prediction
        return PotentialLossOutput(e + 2*f + 3*s, term(e), term(f), term(s))

    monkeypatch.setattr(diagnostics, 'evaluate_structure_batch', evaluate)
    monkeypatch.setattr(diagnostics, 'compute_potential_loss', loss)
    return model, batch


def test_weighted_gradients_activations_and_preservation(monkeypatch):
    model, batch = setup_case(monkeypatch)
    model.layers[0].weight.grad = torch.full_like(model.layers[0].weight, 7)
    grad = model.layers[0].weight.grad
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    modes = [m.training for m in model.modules()]
    report = diagnostics.diagnose_training_batch(
        model, batch, {}, loss_config=LossConfig(force_weight=2, stress_weight=3)
    )
    assert report['terms']['energy']['gradient_norm'] == pytest.approx(36)
    assert report['terms']['force']['gradient_norm'] == pytest.approx(6)
    assert report['terms']['stress']['gradient_norm'] == pytest.approx(18)
    assert report['total']['gradient_norm'] == pytest.approx(48)
    assert report['terms']['energy']['parameters_without_grad'] == 1
    assert report['activations']['layers.0.input']['rms'] == 3
    assert report['activations']['layers.0.output']['rms'] == 6
    assert model.layers[0].weight.grad is grad
    assert torch.equal(grad, torch.full_like(grad, 7))
    assert model.unused.grad is None
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert not model.layers[0]._forward_hooks


def test_exception_cleans_hooks_modes_and_rng(monkeypatch):
    model, batch = setup_case(monkeypatch)
    rng = torch.get_rng_state().clone()
    modes = [m.training for m in model.modules()]
    def fail(*a, **kw):
        torch.rand(1)
        raise RuntimeError('evaluation failed')
    monkeypatch.setattr(diagnostics, 'evaluate_structure_batch', fail)
    with pytest.raises(RuntimeError, match='evaluation failed'):
        diagnostics.diagnose_training_batch(model, batch, {}, loss_config=LossConfig())
    assert not model.layers[0]._forward_hooks
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())


def test_no_trainable_parameters(monkeypatch):
    model, batch = setup_case(monkeypatch)
    model.requires_grad_(False)
    result = diagnostics.diagnose_training_batch(model, batch, {}, loss_config=LossConfig())
    assert result['total']['gradient_norm'] == 0
    assert result['total']['parameters_with_grad'] == 0


def test_missing_supervision_is_rejected_and_hooks_removed(monkeypatch):
    model, batch = setup_case(monkeypatch)
    with pytest.raises(ValueError, match='no weighted valid supervision'):
        diagnostics.diagnose_training_batch(
            model, batch, {}, loss_config=LossConfig(energy_weight=0)
        )
    assert not model.layers[0]._forward_hooks


def test_nonfinite_is_reported_as_null_not_zero(monkeypatch):
    model, batch = setup_case(monkeypatch)
    with torch.no_grad():
        model.layers[0].weight.fill_(float('inf'))
    report = diagnostics.diagnose_training_batch(model, batch, {}, loss_config=LossConfig())
    assert not report['terms']['energy']['loss_finite']
    assert not report['terms']['energy']['gradients_finite']
    assert report['terms']['energy']['gradient_norm'] is None
    assert report['activations']['layers.0.output']['rms'] is None
    assert report['activations']['layers.0.output']['nonfinite_count'] == 1
