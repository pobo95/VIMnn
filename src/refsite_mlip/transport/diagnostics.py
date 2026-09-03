"""Transport result construction without post-solve plan repair."""

from __future__ import annotations

import torch

from .dual import marginal_residuals, transport_plan
from .marginals import split_atom_vacancy_plan
from .problem import OTProblem
from .result import OTResult

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
