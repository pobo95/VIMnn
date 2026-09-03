"""Transport result construction and storage-roundoff diagnostics."""

from __future__ import annotations

import torch

from .dual import marginal_residuals, transport_plan
from .marginals import split_atom_vacancy_plan
from .problem import OTProblem
from .result import OTResult


def _converged_value(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu())
    return bool(value)


def _stabilize_float32_marginals(
    problem: OTProblem,
    gamma: torch.Tensor,
    *,
    converged,
) -> torch.Tensor:
    """Remove only float32 storage roundoff from an already-converged plan.

    A log-domain solve can satisfy its native marginal check while independently
    rounded float32 entries miss the same invariant when accumulated in float64.
    One fixed row/column multiplicative projection in float64 restores the stored
    plan to the requested marginals.  The operation remains differentiable and
    preserves exact zeros.  It is deliberately disabled for float64,
    non-converged results, and residuals larger than a dimension-aware roundoff
    bound, so it cannot serve as an extra solver iteration or convergence retry.
    """

    if gamma.dtype != torch.float32 or not _converged_value(converged):
        return gamma

    validation = gamma.detach().to(torch.float64)
    rows = problem.row_marginal.detach().to(torch.float64)
    columns = problem.column_marginal.detach().to(torch.float64)
    row_sums = validation.sum(dim=1)
    column_sums = validation.sum(dim=0)
    if not bool(torch.all(torch.isfinite(row_sums))) or not bool(
        torch.all(torch.isfinite(column_sums))
    ):
        return gamma
    if bool(torch.any(row_sums <= 0.0)) or bool(torch.any(column_sums <= 0.0)):
        return gamma

    residual = torch.maximum(
        (row_sums - rows).abs().max(),
        (column_sums - columns).abs().max(),
    )
    reduction_terms = max(problem.num_sites, problem.num_columns) + 2
    marginal_scale = float(max(1, problem.num_vacancies))
    roundoff_bound = (
        reduction_terms * torch.finfo(torch.float32).eps * marginal_scale
    )
    if float(residual.cpu()) > roundoff_bound:
        return gamma

    working = gamma.to(torch.float64)
    working = working * (rows / working.sum(dim=1)).unsqueeze(1)
    working = working * (columns / working.sum(dim=0)).unsqueeze(0)
    return working.to(gamma.dtype)


def build_result(
    problem: OTProblem,
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    converged,
    sinkhorn_iterations: int,
    newton_iterations: int,
    cg_iterations: int,
    line_search_reductions: int,
    fallback_used: bool,
    solver_name: str,
    path_name: str,
    final_linear_residual=None,
    failure_reason=None,
    effective_diagnostic_tolerance=None,
    accepted_damping=None,
    warmup_sinkhorn_iterations=0,
    fallback_sinkhorn_iterations=0,
) -> OTResult:
    gamma = transport_plan(problem, f, g)
    gamma = _stabilize_float32_marginals(
        problem,
        gamma,
        converged=converged,
    )
    row, column = marginal_residuals(problem, gamma)
    P, q = split_atom_vacancy_plan(problem, gamma)
    support_diagnostics = problem.support_diagnostics
    if support_diagnostics is not None and effective_diagnostic_tolerance is not None:
        support_diagnostics = support_diagnostics.with_effective_tolerance(
            effective_diagnostic_tolerance
        )
    return OTResult(
        gamma=gamma,
        P=P,
        q=q,
        f=f,
        g=g,
        row_residual=row.abs().max(),
        column_residual=column.abs().max(),
        converged=converged,
        sinkhorn_iterations=sinkhorn_iterations,
        newton_iterations=newton_iterations,
        cg_iterations=cg_iterations,
        line_search_reductions=line_search_reductions,
        fallback_used=fallback_used,
        solver_name=solver_name,
        path_name=path_name,
        final_linear_residual=final_linear_residual,
        failure_reason=failure_reason,
        support_diagnostics=support_diagnostics,
        effective_diagnostic_tolerance=effective_diagnostic_tolerance,
        accepted_damping=accepted_damping,
        warmup_sinkhorn_iterations=warmup_sinkhorn_iterations,
        fallback_sinkhorn_iterations=fallback_sinkhorn_iterations,
    )


def primal_objective(
    gamma: torch.Tensor, cost: torch.Tensor, epsilon: float
) -> torch.Tensor:
    entropy_terms = torch.xlogy(gamma, gamma) - gamma
    return torch.sum(gamma * cost) + gamma.new_tensor(epsilon) * entropy_terms.sum()
