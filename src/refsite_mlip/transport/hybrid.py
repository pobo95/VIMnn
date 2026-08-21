"""Sinkhorn-warm-started Newton-PCG with explicit evaluation fallback."""

from __future__ import annotations

from typing import Optional

from .diagnostics import build_result
from .newton_krylov import solve_newton_krylov, validate_eval_config
from .problem import OTProblem
from .result import DualVariables, EvalOTConfig, OTResult
from .sinkhorn import fixed_sinkhorn_updates, solve_sinkhorn_eval_adaptive


def solve_hybrid_eval(
    problem: OTProblem,
    config: EvalOTConfig,
    initial: Optional[DualVariables] = None,
) -> OTResult:
    validate_eval_config(config)
    warm = fixed_sinkhorn_updates(
        problem, config.sinkhorn_iterations, initial=initial
    )
    outcome = solve_newton_krylov(problem, config, warm)
    if outcome.converged:
        return build_result(
            problem,
            outcome.f,
            outcome.g,
            converged=True,
            sinkhorn_iterations=config.sinkhorn_iterations,
            newton_iterations=outcome.iterations,
            cg_iterations=outcome.cg_iterations,
            line_search_reductions=outcome.line_search_reductions,
            fallback_used=False,
            solver_name="hybrid",
            path_name="eval_adaptive",
            final_linear_residual=outcome.final_linear_residual,
        )
    fallback_initial = DualVariables(outcome.f, outcome.g)
    return solve_sinkhorn_eval_adaptive(
        problem,
        maximum_iterations=config.fallback_sinkhorn_iterations,
        tolerance=config.convergence_tolerance,
        initial=fallback_initial,
        solver_name="hybrid",
        fallback_used=True,
        previous_sinkhorn_iterations=config.sinkhorn_iterations,
        newton_iterations=outcome.iterations,
        cg_iterations=outcome.cg_iterations,
        line_search_reductions=outcome.line_search_reductions,
        final_linear_residual=outcome.final_linear_residual,
        failure_reason=outcome.failure_reason,
    )
