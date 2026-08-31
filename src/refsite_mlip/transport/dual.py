"""Dense plan arithmetic and matrix-free dual Jacobian products."""

from __future__ import annotations

import torch

from .gauge import project_gauge
from .problem import OTProblem


def transport_plan(
    problem: OTProblem, f: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    if problem.log_kernel is not None:
        live_log_gamma = (
            f.unsqueeze(-1) / problem.epsilon
            + g.unsqueeze(-2) / problem.epsilon
            + problem.log_kernel
        )
        # Avoid the indeterminate ``-inf + inf``/``exp(nan)`` arithmetic that
        # can otherwise occur on an exactly masked entry while an adaptive
        # dual iterate is large.  The support is a discrete, prevalidated
        # control decision; arithmetic on every active entry remains live.
        active = torch.isfinite(problem.log_kernel)
        safe_log_gamma = torch.where(
            active, live_log_gamma, torch.zeros_like(live_log_gamma)
        )
        return torch.where(
            active, torch.exp(safe_log_gamma), torch.zeros_like(safe_log_gamma)
        )
    log_gamma = (
        f.unsqueeze(-1) + g.unsqueeze(-2) - problem.cost
    ) / problem.epsilon
    gamma = torch.exp(log_gamma)
    return gamma


def marginal_residuals(
    problem: OTProblem, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        gamma.sum(dim=1) - problem.row_marginal,
        gamma.sum(dim=0) - problem.column_marginal,
    )


def residual_vector(problem: OTProblem, gamma: torch.Tensor) -> torch.Tensor:
    row, column = marginal_residuals(problem, gamma)
    return torch.cat((row, column))


def dual_objective(
    problem: OTProblem, f: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    gamma = transport_plan(problem, f, g)
    return (
        problem.epsilon * gamma.sum()
        - torch.dot(problem.row_marginal, f)
        - torch.dot(problem.column_marginal, g)
    )


def jacobian_vector_product(
    problem: OTProblem,
    gamma: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    rows = problem.num_sites
    columns = problem.num_columns
    if vector.shape != (rows + columns,):
        raise ValueError("dual Jacobian vector has incorrect shape")
    u = vector[:rows]
    v = vector[rows:]
    weighted = gamma * (u.unsqueeze(1) + v.unsqueeze(0)) / problem.epsilon
    return torch.cat((weighted.sum(dim=1), weighted.sum(dim=0)))


def gauge_fixed_operator(
    problem: OTProblem,
    gamma: torch.Tensor,
    vector: torch.Tensor,
    rho_gauge: float,
) -> torch.Tensor:
    projected = project_gauge(
        vector, problem.num_sites, problem.num_columns
    )
    jacobian_part = project_gauge(
        jacobian_vector_product(problem, gamma, projected),
        problem.num_sites,
        problem.num_columns,
    )
    null_part = vector - projected
    return jacobian_part + vector.new_tensor(rho_gauge) * null_part


def jacobi_inverse(
    problem: OTProblem, gamma: torch.Tensor
) -> torch.Tensor:
    diagonal = torch.cat((gamma.sum(dim=1), gamma.sum(dim=0))) / problem.epsilon
    return torch.reciprocal(diagonal)
