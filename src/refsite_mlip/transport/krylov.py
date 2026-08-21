"""Projected matrix-free preconditioned conjugate gradients."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Callable

import torch

from .result import PCGResult


def projected_pcg(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    inverse_diagonal: torch.Tensor,
    projector: Callable[[torch.Tensor], torch.Tensor],
    *,
    maximum_iterations: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> PCGResult:
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, Integral)
        or maximum_iterations <= 0
    ):
        raise ValueError("PCG maximum_iterations must be a positive integer")
    for name, value in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
    ):
        if not isinstance(value, Real) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"PCG {name} must be finite and positive")
    if rhs.shape != inverse_diagonal.shape:
        raise ValueError("PCG preconditioner shape mismatch")

    solution = torch.zeros_like(rhs)
    residual = projector(rhs - operator(solution))
    initial_norm = torch.linalg.vector_norm(residual)
    threshold = max(
        float(absolute_tolerance),
        float(relative_tolerance) * float(initial_norm.detach().cpu()),
    )
    if not bool(torch.isfinite(initial_norm)):
        return PCGResult(
            solution, False, 0, initial_norm, initial_norm, "non-finite residual"
        )
    if float(initial_norm.detach().cpu()) <= threshold:
        return PCGResult(solution, True, 0, initial_norm, initial_norm, None)

    preconditioned = projector(inverse_diagonal * residual)
    direction = preconditioned
    residual_preconditioned = torch.dot(residual, preconditioned)
    final_norm = initial_norm
    used = 0
    for iteration in range(int(maximum_iterations)):
        applied = operator(direction)
        curvature = torch.dot(direction, applied)
        if not bool(torch.isfinite(curvature)) or float(curvature.detach().cpu()) <= 0.0:
            return PCGResult(
                solution,
                False,
                iteration,
                initial_norm,
                final_norm,
                "non-finite or non-positive curvature",
            )
        alpha = residual_preconditioned / curvature
        solution = projector(solution + alpha * direction)
        residual = projector(residual - alpha * applied)
        final_norm = torch.linalg.vector_norm(residual)
        used = iteration + 1
        if not bool(torch.isfinite(final_norm)):
            return PCGResult(
                solution,
                False,
                used,
                initial_norm,
                final_norm,
                "non-finite residual",
            )
        if float(final_norm.detach().cpu()) <= threshold:
            return PCGResult(
                solution, True, used, initial_norm, final_norm, None
            )
        next_preconditioned = projector(inverse_diagonal * residual)
        next_scalar = torch.dot(residual, next_preconditioned)
        if not bool(torch.isfinite(next_scalar)) or float(
            next_scalar.detach().cpu()
        ) <= 0.0:
            return PCGResult(
                solution,
                False,
                used,
                initial_norm,
                final_norm,
                "preconditioner breakdown",
            )
        beta = next_scalar / residual_preconditioned
        direction = projector(next_preconditioned + beta * direction)
        preconditioned = next_preconditioned
        residual_preconditioned = next_scalar
    return PCGResult(
        solution,
        False,
        used,
        initial_norm,
        final_norm,
        "maximum iterations reached",
    )
