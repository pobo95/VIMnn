from __future__ import annotations

import pytest
import torch

from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    EvalOTConfig,
    TrainSinkhornConfig,
    solve_atom_vacancy_ot,
    solve_ragged_atom_vacancy_ot,
)


def _ragged_costs(dtype=torch.float64, device="cpu"):
    return [
        torch.tensor(
            [[0.1, 0.8, 1.1], [0.7, 0.2, 0.9], [1.0, 0.6, 0.15]],
            dtype=dtype,
            device=device,
        ),
        torch.tensor(
            [[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]],
            dtype=dtype,
            device=device,
        ),
        torch.tensor(
            [[0.1, 0.8], [0.7, 0.2], [0.4, 0.6], [0.9, 0.3]],
            dtype=dtype,
            device=device,
        ),
    ]


def test_mixed_ragged_batch_matches_independent_solves_and_order():
    costs = _ragged_costs()
    config = TrainSinkhornConfig(iterations=160)
    batch = solve_ragged_atom_vacancy_ot(
        costs, 0.34, TRAIN_FIXED, "sinkhorn", config
    )
    independent = tuple(
        solve_atom_vacancy_ot(
            cost, 0.34, TRAIN_FIXED, "sinkhorn", config
        )
        for cost in costs
    )
    assert [value.q.sum().round().item() for value in batch] == [0.0, 1.0, 2.0]
    for batched, single in zip(batch, independent):
        torch.testing.assert_close(batched.gamma, single.gamma, atol=0, rtol=0)

    permutation = [2, 0, 1]
    permuted = solve_ragged_atom_vacancy_ot(
        [costs[index] for index in permutation],
        0.34,
        TRAIN_FIXED,
        "sinkhorn",
        config,
    )
    for value, index in zip(permuted, permutation):
        torch.testing.assert_close(value.gamma, batch[index].gamma, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_dtype_preservation(dtype):
    cost = _ragged_costs(dtype=dtype)[1]
    result = solve_atom_vacancy_ot(
        cost,
        0.34,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations=128),
    )
    assert result.gamma.dtype == dtype
    assert result.gamma.device.type == "cpu"
    assert torch.all(torch.isfinite(result.gamma))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_dtype_device_preservation(dtype):
    cost = _ragged_costs(dtype=dtype, device="cuda")[1]
    result = solve_atom_vacancy_ot(
        cost,
        0.34,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations=128),
    )
    assert result.gamma.dtype == dtype
    assert result.gamma.device.type == "cuda"
    assert torch.all(torch.isfinite(result.gamma))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_float64_parity():
    cpu_cost = _ragged_costs()[1]
    gpu_cost = cpu_cost.cuda()
    config = TrainSinkhornConfig(iterations=160)
    cpu = solve_atom_vacancy_ot(
        cpu_cost, 0.34, TRAIN_FIXED, "sinkhorn", config
    )
    gpu = solve_atom_vacancy_ot(
        gpu_cost, 0.34, TRAIN_FIXED, "sinkhorn", config
    )
    torch.testing.assert_close(gpu.P.cpu(), cpu.P, atol=2.0e-13, rtol=2.0e-13)
    torch.testing.assert_close(gpu.q.cpu(), cpu.q, atol=2.0e-13, rtol=2.0e-13)


def test_pcg_failure_triggers_only_explicit_hybrid_fallback():
    rows = torch.arange(6, dtype=torch.float64).unsqueeze(1)
    columns = torch.arange(4, dtype=torch.float64).unsqueeze(0)
    cost = 0.1 + (rows - 1.3 * columns).square() / 3.0
    config = EvalOTConfig(
        sinkhorn_iterations=0,
        pcg_max_iterations=1,
        fallback_sinkhorn_iterations=2000,
        convergence_tolerance=1.0e-11,
    )
    hybrid = solve_atom_vacancy_ot(
        cost, 0.3, EVAL_ADAPTIVE, "hybrid", config
    )
    assert hybrid.fallback_used
    assert "PCG failure" in hybrid.failure_reason
    assert hybrid.row_residual <= 1.0e-11
    with pytest.raises(ValueError, match="PCG failure"):
        solve_atom_vacancy_ot(
            cost, 0.3, EVAL_ADAPTIVE, "newton_krylov", config
        )


def test_line_search_failure_triggers_hybrid_fallback(monkeypatch):
    import refsite_mlip.transport.newton_krylov as newton_module

    cost = _ragged_costs()[1]
    original = newton_module.dual_objective
    calls = {"count": 0}

    def injected_objective(problem, f, g):
        calls["count"] += 1
        value = original(problem, f, g)
        if calls["count"] > 1:
            return value * value.new_tensor(float("nan"))
        return value

    monkeypatch.setattr(newton_module, "dual_objective", injected_objective)
    result = solve_atom_vacancy_ot(
        cost,
        0.34,
        EVAL_ADAPTIVE,
        "hybrid",
        EvalOTConfig(
            sinkhorn_iterations=0,
            fallback_sinkhorn_iterations=2000,
            convergence_tolerance=1.0e-11,
        ),
    )
    assert result.fallback_used
    assert result.failure_reason == "Armijo line search failed"
    assert result.row_residual <= 1.0e-11


def test_extreme_cost_and_small_epsilon_remain_finite_with_recorded_underflow():
    cost = torch.tensor(
        [[0.0, 100.0], [100.0, 0.0], [45.0, 55.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = solve_atom_vacancy_ot(
        cost,
        0.02,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations=512, diagnostic_tolerance=1.0e-7),
    )
    assert torch.all(torch.isfinite(result.gamma))
    assert torch.count_nonzero(result.gamma == 0.0) > 0
    assert result.row_residual < 2.0e-3
    assert not bool(result.converged)
    adaptive = solve_atom_vacancy_ot(
        cost.detach(),
        0.02,
        EVAL_ADAPTIVE,
        "hybrid",
        EvalOTConfig(
            sinkhorn_iterations=32,
            convergence_tolerance=1.0e-10,
            fallback_sinkhorn_iterations=20000,
        ),
    )
    assert adaptive.row_residual <= 1.0e-10
    assert adaptive.column_residual <= 1.0e-10
    weighted = torch.sum(
        result.gamma
        * torch.arange(
            result.gamma.numel(), dtype=torch.float64
        ).reshape_as(result.gamma)
    )
    first = torch.autograd.grad(weighted, cost, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), cost)[0]
    assert torch.all(torch.isfinite(first))
    assert torch.all(torch.isfinite(second))


@pytest.mark.parametrize(
    "bad_cost",
    [
        torch.tensor([[float("nan")]], dtype=torch.float64),
        torch.tensor([[float("inf")]], dtype=torch.float64),
        torch.tensor([[-1.0]], dtype=torch.float64),
    ],
)
def test_invalid_cost_fails_fast(bad_cost):
    with pytest.raises(ValueError):
        solve_atom_vacancy_ot(
            bad_cost,
            0.3,
            TRAIN_FIXED,
            "sinkhorn",
            TrainSinkhornConfig(),
        )
