"""Log-domain fixed training and adaptive evaluation Sinkhorn solvers."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Optional

import torch

from .diagnostics import build_result
from .dual import marginal_residuals, transport_plan
from .gauge import project_duals
from .problem import OTProblem
from .result import DualVariables, OTResult, TrainSinkhornConfig


def _validate_iterations(iterations: int, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, Integral)
        or int(iterations) < minimum
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"Sinkhorn iterations must be a {qualifier} integer")
    return int(iterations)


def zero_duals(problem: OTProblem) -> DualVariables:
    return DualVariables(
        f=torch.zeros_like(problem.row_marginal),
        g=torch.zeros_like(problem.column_marginal),
    )


def _validate_duals(
    problem: OTProblem, initial: Optional[DualVariables]
) -> DualVariables:
    if initial is None:
        return zero_duals(problem)
    if initial.f.shape != problem.row_marginal.shape:
        raise ValueError("initial row dual has incorrect shape")
    if initial.g.shape != problem.column_marginal.shape:
        raise ValueError("initial column dual has incorrect shape")
    for dual in (initial.f, initial.g):
        if dual.dtype != problem.cost.dtype or dual.device != problem.cost.device:
            raise ValueError("initial dual dtype/device must match atom_cost")
        if not bool(torch.all(torch.isfinite(dual))):
            raise ValueError("initial dual contains NaN or Inf")
    return initial


def sinkhorn_full_update(
    problem: OTProblem, f: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    log_rows = torch.log(problem.row_marginal)
    updated_f = problem.epsilon * (
        log_rows
        - torch.logsumexp(
            (g.unsqueeze(0) - problem.cost) / problem.epsilon, dim=1
        )
    )
    log_columns = torch.log(problem.column_marginal)
    updated_g = problem.epsilon * (
        log_columns
        - torch.logsumexp(
            (updated_f.unsqueeze(1) - problem.cost) / problem.epsilon,
            dim=0,
        )
    )
    return project_duals(updated_f, updated_g)


def fixed_sinkhorn_updates(
    problem: OTProblem,
    iterations: int,
    initial: Optional[DualVariables] = None,
) -> DualVariables:
    """Pure fixed-count unrolled updates used by TRAIN_FIXED."""

    count = _validate_iterations(iterations, allow_zero=True)
    duals = _validate_duals(problem, initial)
    f, g = duals.f, duals.g
    with torch.autocast(device_type=problem.cost.device.type, enabled=False):
        for _ in range(count):
            f, g = sinkhorn_full_update(problem, f, g)
    return DualVariables(f=f, g=g)


def solve_sinkhorn_train_fixed(
    problem: OTProblem, config: TrainSinkhornConfig
) -> OTResult:
    iterations = _validate_iterations(config.iterations)
    tolerance = float(config.diagnostic_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("diagnostic_tolerance must be finite and positive")
    duals = fixed_sinkhorn_updates(problem, iterations)
    gamma = transport_plan(problem, duals.f, duals.g)
    row, column = marginal_residuals(problem, gamma)
    converged = torch.maximum(row.abs().max(), column.abs().max()) <= tolerance
    return build_result(
        problem,
        duals.f,
        duals.g,
        converged=converged,
        sinkhorn_iterations=iterations,
        newton_iterations=0,
        cg_iterations=0,
        line_search_reductions=0,
        fallback_used=False,
        solver_name="sinkhorn",
        path_name="train_fixed",
    )


def solve_sinkhorn_eval_adaptive(
    problem: OTProblem,
    *,
    maximum_iterations: int,
    tolerance: float,
    initial: Optional[DualVariables] = None,
    solver_name: str = "sinkhorn",
    fallback_used: bool = False,
    previous_sinkhorn_iterations: int = 0,
    newton_iterations: int = 0,
    cg_iterations: int = 0,
    line_search_reductions: int = 0,
    final_linear_residual=None,
    failure_reason=None,
) -> OTResult:
    maximum = _validate_iterations(maximum_iterations)
    if not isinstance(tolerance, Real) or not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError("evaluation Sinkhorn tolerance must be finite and positive")
    duals = _validate_duals(problem, initial)
    f, g = duals.f, duals.g
    used = 0
    converged = False
    with torch.autocast(device_type=problem.cost.device.type, enabled=False):
        for index in range(maximum):
            f, g = sinkhorn_full_update(problem, f, g)
            used = index + 1
            gamma = transport_plan(problem, f, g)
            row, column = marginal_residuals(problem, gamma)
            residual = torch.maximum(row.abs().max(), column.abs().max())
            if float(residual.detach().cpu()) <= float(tolerance):
                converged = True
                break
    if not converged:
        gamma = transport_plan(problem, f, g)
        row, column = marginal_residuals(problem, gamma)
        residual = torch.maximum(row.abs().max(), column.abs().max())
        raise ValueError(
            "adaptive log-Sinkhorn did not converge: "
            f"iterations={used}/{maximum}, residual={float(residual.detach().cpu()):.9e}, "
            f"tolerance={float(tolerance):.9e}"
        )
    return build_result(
        problem,
        f,
        g,
        converged=True,
        sinkhorn_iterations=previous_sinkhorn_iterations + used,
        newton_iterations=newton_iterations,
        cg_iterations=cg_iterations,
        line_search_reductions=line_search_reductions,
        fallback_used=fallback_used,
        solver_name=solver_name,
        path_name="eval_adaptive",
        final_linear_residual=final_linear_residual,
        failure_reason=failure_reason,
    )
