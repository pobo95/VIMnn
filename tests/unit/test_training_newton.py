from dataclasses import replace
import pytest
import torch
from refsite_mlip.transport import TRAIN_FIXED, TrainNewtonConfig, TrainSinkhornConfig, solve_atom_vacancy_ot


def _cost():
    return torch.tensor([[0.1, 1.2, 0.7], [0.8, 0.2, 1.1], [0.3, 0.7, 0.1], [1.3, 0.6, 0.8]], dtype=torch.float64)


def test_newton_training_matches_converged_sinkhorn_and_second_derivatives():
    cost = _cost().requires_grad_()
    cfg = TrainNewtonConfig(warmup_iterations=1, convergence_tolerance=1e-12)
    def solve(x):
        return solve_atom_vacancy_ot(x, 0.2, TRAIN_FIXED, 'newton_krylov', cfg)
    out = solve(cost)
    assert out.newton_iterations > 0 and out.cg_iterations > 0
    assert out.solver_name == 'newton_krylov' and not out.fallback_used
    ref = solve_atom_vacancy_ot(cost, 0.2, TRAIN_FIXED, 'sinkhorn', TrainSinkhornConfig(2048))
    torch.testing.assert_close(out.gamma, ref.gamma, atol=1e-10, rtol=1e-9)
    weights = torch.arange(out.gamma.numel(), dtype=cost.dtype).reshape_as(out.gamma).square()
    def value(x): return (solve(x).gamma * weights).sum()
    assert torch.autograd.gradcheck(value, (cost,), eps=1e-5, atol=2e-6, rtol=2e-4)
    assert torch.autograd.gradgradcheck(value, (cost,), eps=1e-5, atol=2e-5, rtol=2e-3)


def test_newton_training_failure_is_explicit():
    cfg = TrainNewtonConfig(warmup_iterations=0, max_newton_iterations=1, convergence_tolerance=1e-14)
    with pytest.raises(ValueError, match='without fallback'):
        solve_atom_vacancy_ot(_cost(), 0.05, TRAIN_FIXED, 'newton_krylov', cfg)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('pristine', [False, True])
def test_newton_dtype_and_pristine_vacancy(dtype, pristine):
    cost = _cost().to(dtype)
    if pristine: cost = cost[:3]
    cost.requires_grad_()
    out = solve_atom_vacancy_ot(cost, 0.2, TRAIN_FIXED, 'newton_krylov', TrainNewtonConfig(warmup_iterations=2))
    assert out.converged
    assert float(out.row_residual) < (2e-6 if dtype == torch.float32 else 2e-11)
    gradient, = torch.autograd.grad(out.P.square().sum(), cost, create_graph=True)
    second, = torch.autograd.grad(gradient.square().sum(), cost)
    assert torch.isfinite(gradient).all() and torch.isfinite(second).all()


@pytest.mark.parametrize('kwargs', [{'warmup_iterations': -1}, {'warmup_iterations': True},
    {'max_newton_iterations': 0}, {'convergence_tolerance': 0},
    {'pcg_max_iterations': False}, {'pcg_relative_tolerance': float('nan')}])
def test_invalid_newton_config(kwargs):
    with pytest.raises((TypeError, ValueError)): TrainNewtonConfig(**kwargs)


def test_already_converged_zero_warmup_has_correct_derivatives():
    cost = torch.zeros((1,1), dtype=torch.float64, requires_grad=True)
    cfg = TrainNewtonConfig(warmup_iterations=0)
    def value(x): return solve_atom_vacancy_ot(x, .2, TRAIN_FIXED, 'newton_krylov', cfg).gamma.sum()
    first, = torch.autograd.grad(value(cost), cost, create_graph=True)
    second, = torch.autograd.grad(first.sum(), cost)
    torch.testing.assert_close(first, torch.zeros_like(first), atol=1e-12, rtol=0)
    torch.testing.assert_close(second, torch.zeros_like(second), atol=1e-12, rtol=0)
    # Numerical perturbations must stay inside the nonnegative cost domain.
    interior = torch.full_like(cost, 0.1, requires_grad=True)
    assert torch.autograd.gradcheck(value, (interior,))
    assert torch.autograd.gradgradcheck(value, (interior,))


def test_compact_support_second_derivatives():
    from refsite_mlip.transport import TransportSupportConfig
    support = TransportSupportConfig('compact_c2', 2., .5, .2)
    distance = torch.tensor([[.4,1.8,2.3],[1.5,.6,1.7],[.7,1.6,.3],[2.3,1.2,1.4]],dtype=torch.float64,requires_grad=True)
    cfg = TrainNewtonConfig(warmup_iterations=2)
    def value(d):
        out = solve_atom_vacancy_ot(d.square()/2, .3, TRAIN_FIXED, 'newton_krylov', cfg,
                                   support_config=support, atom_distances=d)
        return (out.gamma * torch.arange(out.gamma.numel(), dtype=d.dtype).reshape_as(out.gamma).square()).sum()
    assert torch.autograd.gradcheck(value, (distance,), eps=1e-5, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(value, (distance,), eps=1e-5, atol=2e-4, rtol=1e-3)


def test_disconnected_pristine_support_gauges():
    from refsite_mlip.transport import TransportSupportConfig
    support = TransportSupportConfig('compact_c2', 2., .5, .2)
    distance = torch.tensor([[.3,3.],[3.,.5]],dtype=torch.float64,requires_grad=True)
    out = solve_atom_vacancy_ot(distance.square()/2,.3,TRAIN_FIXED,'newton_krylov',TrainNewtonConfig(),
                               support_config=support,atom_distances=distance)
    torch.testing.assert_close(out.P, torch.eye(2,dtype=distance.dtype), atol=1e-12, rtol=0)
    gradient, = torch.autograd.grad(out.P.sum(), distance, create_graph=True)
    second, = torch.autograd.grad(gradient.sum(), distance)
    assert torch.isfinite(second).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_newton_cuda_double_backward_matches_cpu(dtype):
    cfg = TrainNewtonConfig(warmup_iterations=2)
    results=[]
    for device in ('cpu','cuda'):
        cost=_cost().to(device=device,dtype=dtype).requires_grad_()
        out=solve_atom_vacancy_ot(cost,.2,TRAIN_FIXED,'newton_krylov',cfg)
        first,=torch.autograd.grad(out.gamma.square().sum(),cost,create_graph=True)
        second,=torch.autograd.grad(first.square().sum(),cost)
        assert torch.isfinite(second).all()
        results.append((out.gamma.detach().cpu(),first.detach().cpu(),second.detach().cpu()))
    tol=2e-4 if dtype==torch.float32 else 1e-8
    for left,right in zip(*results):
        torch.testing.assert_close(left,right,atol=tol,rtol=tol)
