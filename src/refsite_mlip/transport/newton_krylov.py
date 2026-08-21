"""Adaptive matrix-free Newton-PCG evaluation solver."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Optional

import torch

from .dual import (
    dual_objective,
    gauge_fixed_operator,
    jacobi_inverse,
    residual_vector,
    transport_plan,
)
from .gauge import project_duals, project_gauge
from .krylov import projected_pcg
from .problem import OTProblem
from .result import DualVariables, EvalOTConfig, NewtonOutcome
from .sinkhorn import _validate_duals


def validate_eval_config(config: EvalOTConfig) -> None:
    integer_fields = (
        ("sinkhorn_iterations", config.sinkhorn_iterations, True),
        ("max_newton_iterations", config.max_newton_iterations, False),
        ("pcg_max_iterations", config.pcg_max_iterations, False),
        (
            "max_line_search_reductions",
            config.max_line_search_reductions,
            True,
        ),
        (
            "fallback_sinkhorn_iterations",
            config.fallback_sinkhorn_iterations,
            False,
        ),
    )
    for name, value, allow_zero in integer_fields:
        lower = 0 if allow_zero else 1
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < lower
        ):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
    positive_fields = (
        ("convergence_tolerance", config.convergence_tolerance),
        ("pcg_absolute_tolerance", config.pcg_absolute_tolerance),
        ("pcg_relative_tolerance", config.pcg_relative_tolerance),
        ("gauge_rho", config.gauge_rho),
        ("armijo_coefficient", config.armijo_coefficient),
        ("line_search_reduction", config.line_search_reduction),
    )
    for name, value in positive_fields:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if config.armijo_coefficient >= 1.0:
        raise ValueError("armijo_coefficient must be less than one")
    if config.line_search_reduction >= 1.0:
        raise ValueError("line_search_reduction must be less than one")


def solve_newton_krylov(
    problem: OTProblem,
    config: EvalOTConfig,
    initial: Optional[DualVariables] = None,
) -> NewtonOutcome:
    """Solve on the gauge-fixed subspace with adaptive PCG and Armijo."""

    validate_eval_config(config)
    duals = _validate_duals(problem, initial)
    f, g = project_duals(duals.f, duals.g)
    total_cg = 0
    total_reductions = 0
    final_linear_residual = None

    for newton_index in range(config.max_newton_iterations):
        gamma = transport_plan(problem, f, g)
        if not bool(torch.all(torch.isfinite(gamma))):
            return NewtonOutcome(
                f,
                g,
                False,
                newton_index,
                total_cg,
                total_reductions,
                final_linear_residual,
                "non-finite transport plan",
            )
        residual = residual_vector(problem, gamma)
        projected_residual = project_gauge(
            residual, problem.num_sites, problem.num_columns
        )
        residual_max = projected_residual.abs().max()
        if float(residual_max.detach().cpu()) <= config.convergence_tolerance:
            return NewtonOutcome(
                f,
                g,
                True,
                newton_index,
                total_cg,
                total_reductions,
                final_linear_residual,
                None,
            )

        inverse_diagonal = jacobi_inverse(problem, gamma)
        if not bool(torch.all(torch.isfinite(inverse_diagonal))):
            return NewtonOutcome(
                f,
                g,
                False,
                newton_index,
                total_cg,
                total_reductions,
                final_linear_residual,
                "non-finite Jacobi preconditioner",
            )
        projector = lambda value: project_gauge(
            value, problem.num_sites, problem.num_columns
        )
        operator = lambda value: gauge_fixed_operator(
            problem, gamma, value, config.gauge_rho
        )
        pcg = projected_pcg(
            operator,
            -projected_residual,
            inverse_diagonal,
            projector,
            maximum_iterations=config.pcg_max_iterations,
            absolute_tolerance=config.pcg_absolute_tolerance,
            relative_tolerance=config.pcg_relative_tolerance,
        )
        total_cg += pcg.iterations
        final_linear_residual = pcg.final_residual
        if not pcg.converged:
            return NewtonOutcome(
                f,
                g,
                False,
                newton_index,
                total_cg,
                total_reductions,
                final_linear_residual,
                f"PCG failure: {pcg.breakdown}",
            )

        correction = pcg.solution
        directional_derivative = torch.dot(residual, correction)
        if (
            not bool(torch.isfinite(directional_derivative))
            or float(directional_derivative.detach().cpu()) >= 0.0
        ):
            return NewtonOutcome(
                f,
                g,
                False,
                newton_index,
                total_cg,
                total_reductions,
                final_linear_residual,
                "Newton direction is not a descent direction",
            )
        objective = dual_objective(problem, f, g)
        accepted = False
        accepted_f, accepted_g = f, g
        step = 1.0
        reductions_this_step = 0
        for reduction in range(config.max_line_search_reductions + 1):
            candidate_f = f + f.new_tensor(step) * correction[: problem.num_sites]
            candidate_g = g + g.new_tensor(step) * correction[problem.num_sites :]
            candidate_f, candidate_g = project_duals(candidate_f, candidate_g)
            candidate_objective = dual_objective(
                problem, candidate_f, candidate_g
            )
            bound = objective + objective.new_tensor(
                config.armijo_coefficient * step
            ) * directional_derivative
            if bool(torch.isfinite(candidate_objective)) and float(
                (candidate_objective - bound).detach().cpu()
            ) <= 0.0:
                accepted = True
                accepted_f, accepted_g = candidate_f, candidate_g
                reductions_this_step = reduction
                break
            step *= config.line_search_reduction
        if not accepted:
            return NewtonOutcome(
                f,
                g,
                False,
                newton_index,
                total_cg,
                total_reductions,
                final_linear_residual,
                "Armijo line search failed",
            )
        f, g = accepted_f, accepted_g
        total_reductions += reductions_this_step

    final_gamma = transport_plan(problem, f, g)
    final_residual = project_gauge(
        residual_vector(problem, final_gamma),
        problem.num_sites,
        problem.num_columns,
    ).abs().max()
    converged = (
        bool(torch.isfinite(final_residual))
        and float(final_residual.detach().cpu()) <= config.convergence_tolerance
    )
    return NewtonOutcome(
        f,
        g,
        converged,
        config.max_newton_iterations,
        total_cg,
        total_reductions,
        final_linear_residual,
        None if converged else "maximum Newton iterations reached",
    )
