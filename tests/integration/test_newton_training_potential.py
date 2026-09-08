from dataclasses import replace
import pytest
import torch
from refsite_mlip.models import instantiate_reference_site_model_bundle
from refsite_mlip.transport import TrainNewtonConfig
from test_symmetric_correlation_bundle import _capture_v2, _single, _assert_single_equal
from refsite_mlip.transport import TRAIN_FIXED


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_newton_bundle_force_stress_backward_and_reload(typed_crystal, dtype):
    _, model, _, samples, _, contexts, _, bundle = _capture_v2(
        typed_crystal, dtype=dtype, train_ot_solver='newton_krylov')
    restored = instantiate_reference_site_model_bundle(bundle, dtype=dtype)
    assert restored.model.config.train_ot_solver == 'newton_krylov'
    sample = samples[0]
    sample = replace(sample, **{
        name: getattr(sample, name).to(dtype=dtype)
        for name in ('positions', 'cell', 'origin', 'energy', 'forces', 'stress')
        if getattr(sample, name) is not None
    })
    context = contexts[sample.template_id]
    before = _single(model, sample, context, None, TRAIN_FIXED)
    after = _single(restored.model, sample, restored.template_contexts[sample.template_id], None, TRAIN_FIXED)
    _assert_single_equal(before, after)
    p = sample.positions.detach().clone().requires_grad_()
    out = model(p, sample.atomic_numbers, sample.cell, sample.origin,
        template_context=context, compute_forces=True, compute_stress=True, create_graph=True, return_aux=True)
    assert out.auxiliary['ot'].solver_name == 'newton_krylov'
    weights = tuple(p for name, p in model.named_parameters() if '.symmetric_contraction.weight_' in name)
    for loss in (out.forces.square().sum(), out.stress.square().sum()):
        grads = torch.autograd.grad(loss, weights, retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().max() > 0 for g in grads)
    if dtype == torch.float64:
        h = 1e-5
        direction = torch.zeros_like(p); direction[0, 0] = 1
        def energy(pos, cell):
            return model(pos, sample.atomic_numbers, cell, sample.origin, template_context=context).energy
        force_fd = -(energy(p+h*direction, sample.cell)-energy(p-h*direction, sample.cell))/(2*h)
        torch.testing.assert_close(force_fd, out.forces[0,0], atol=1e-5, rtol=1e-4)
        deformation = torch.eye(3, dtype=dtype)
        strain = torch.zeros_like(deformation); strain[0,0] = h
        stress_fd = (energy(p@(deformation+strain),sample.cell@(deformation+strain))-
                     energy(p@(deformation-strain),sample.cell@(deformation-strain)))/(2*h*sample.cell.det())
        torch.testing.assert_close(stress_fd, out.stress[0,0], atol=1e-5, rtol=1e-4)
