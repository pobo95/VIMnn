"""Differentiable Newton-Krylov OT on the fixed training phase branch.

Accepted Sinkhorn, PCG and Newton updates remain in the autograd graph. Scalar
stopping/line-search decisions select a branch; they are not differentiated.
No inference phase selection, detached warm starts or silent fallback is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Mapping

import torch

from .diagnostics import build_result
from .newton_krylov import solve_newton_krylov, validate_eval_config
from .result import EvalOTConfig
from .sinkhorn import fixed_sinkhorn_updates
from .dual import transport_plan, residual_vector, marginal_residuals


def _gauge_penalty(problem):
    """Fix one null direction per connected support component."""
    rows, columns = problem.num_sites, problem.num_columns
    parent = list(range(rows + columns))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    active = (torch.ones_like(problem.cost, dtype=torch.bool)
              if problem.log_kernel is None else torch.isfinite(problem.log_kernel))
    for row, column in active.nonzero().tolist():
        parent[root(row)] = root(rows + column)
    components = {}
    for i in range(rows + columns):
        components.setdefault(root(i), []).append(i)
    penalty = problem.cost.new_zeros((rows + columns, rows + columns))
    for members in components.values():
        null = problem.cost.new_zeros(rows + columns)
        for i in members:
            null[i] = 1 if i < rows else -1
        penalty = penalty + torch.outer(null, null) / len(members)
    return penalty


def _polish(problem, f, g):
    """Two live Newton corrections recover first/second root sensitivities.

    Unlike tolerance-stopped PCG, a differentiable dense solve retains the RHS
    derivative even when its value is exactly zero. This is why the training
    implementation currently requires the dense transport backend.
    """
    penalty = _gauge_penalty(problem)
    for _ in range(2):
        gamma = transport_plan(problem, f, g)
        jacobian = torch.cat((
            torch.cat((torch.diag(gamma.sum(1)), gamma), dim=1),
            torch.cat((gamma.T, torch.diag(gamma.sum(0))), dim=1),
        ), dim=0) / problem.epsilon
        correction = torch.linalg.solve(jacobian + penalty, -residual_vector(problem, gamma))
        f = f + correction[:problem.num_sites]
        g = g + correction[problem.num_sites:]
    return f, g


@dataclass(frozen=True)
class TrainNewtonConfig:
    warmup_iterations: int = 16
    max_newton_iterations: int = 30
    convergence_tolerance: float | None = None
    pcg_max_iterations: int = 256
    pcg_absolute_tolerance: float | None = None
    pcg_relative_tolerance: float | None = None

    def __post_init__(self):
        validate_eval_config(self.runtime_config(torch.float64))
        for name in ('warmup_iterations', 'max_newton_iterations', 'pcg_max_iterations'):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ('convergence_tolerance', 'pcg_absolute_tolerance', 'pcg_relative_tolerance'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, float(getattr(self, name)))

    def runtime_config(self, dtype):
        single = dtype == torch.float32
        return EvalOTConfig(
            sinkhorn_iterations=self.warmup_iterations,
            max_newton_iterations=self.max_newton_iterations,
            convergence_tolerance=(1e-6 if single else 1e-11)
                if self.convergence_tolerance is None else self.convergence_tolerance,
            pcg_max_iterations=self.pcg_max_iterations,
            pcg_absolute_tolerance=(1e-8 if single else 1e-13)
                if self.pcg_absolute_tolerance is None else self.pcg_absolute_tolerance,
            pcg_relative_tolerance=(1e-6 if single else 1e-11)
                if self.pcg_relative_tolerance is None else self.pcg_relative_tolerance,
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, Mapping):
            raise TypeError("newton config must be a mapping")
        return cls(**dict(values))


def solve_newton_train(problem, config: TrainNewtonConfig):
    runtime = config.runtime_config(problem.cost.dtype)
    original = problem
    # OT dual solves are small dense systems. Float64 avoids float32 loss of
    # precision near a converged root; casts preserve the autograd graph.
    problem = replace(problem, **{
        field.name: getattr(problem, field.name).double()
        for field in fields(problem)
        if isinstance(getattr(problem, field.name), torch.Tensor)
        and getattr(problem, field.name).is_floating_point()
    })
    with torch.autocast(device_type=problem.cost.device.type, enabled=False):
        warm = fixed_sinkhorn_updates(problem, config.warmup_iterations)
        outcome = solve_newton_krylov(problem, runtime, warm)
        if not outcome.converged:
            raise ValueError(
                "training Newton-Krylov failed without fallback: "
                + str(outcome.failure_reason)
            )
        f, g = _polish(problem, outcome.f, outcome.g)
        result = build_result(
            problem, f, g, converged=True,
            sinkhorn_iterations=config.warmup_iterations,
            newton_iterations=outcome.iterations + 2, cg_iterations=outcome.cg_iterations,
            line_search_reductions=outcome.line_search_reductions,
            fallback_used=False, solver_name="newton_krylov", path_name="train_fixed",
            final_linear_residual=outcome.final_linear_residual,
            accepted_damping=1.0,
            warmup_sinkhorn_iterations=config.warmup_iterations,
            effective_diagnostic_tolerance=runtime.convergence_tolerance,
        )
        result = replace(result, **{
            field.name: getattr(result, field.name).to(dtype=original.cost.dtype)
            for field in fields(result)
            if isinstance(getattr(result, field.name), torch.Tensor)
        })
        row, column = marginal_residuals(original, result.gamma)
        result = replace(result, row_residual=row.abs().max(), column_residual=column.abs().max())
        if not all(bool(torch.isfinite(value).all()) for value in (result.gamma, row, column)):
            raise ValueError("training Newton-Krylov produced nonfinite transport")
        if max(float(result.row_residual.detach()), float(result.column_residual.detach())) > runtime.convergence_tolerance:
            raise ValueError("training Newton-Krylov marginal residual exceeds tolerance")
        return result
