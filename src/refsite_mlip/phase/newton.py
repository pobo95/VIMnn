"""Fixed-schedule unrolled damped Newton phase solver."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import torch

from .objective import phase_gradient_hessian, phase_objective
from .types import PhaseResult


def solve_training_phase(
    cross: torch.Tensor,
    modes: torch.Tensor,
    mode_weights: torch.Tensor,
    initial_phase: torch.Tensor,
    step_schedule: Sequence[float],
    damping_schedule: Sequence[float],
) -> PhaseResult:
    """Run a branch-free fixed number of unwrapped Newton iterations."""

    if len(step_schedule) == 0 or len(step_schedule) != len(damping_schedule):
        raise ValueError("step and damping schedules must have equal positive length")
    for name, schedule in (
        ("step", step_schedule),
        ("damping", damping_schedule),
    ):
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in schedule
        ):
            raise ValueError(f"{name} schedule values must be finite positive reals")
    if initial_phase.shape[-1] != 3:
        raise ValueError("initial_phase must end in dimension 3")
    if not initial_phase.is_floating_point():
        raise ValueError("initial_phase must be floating point")
    if cross.shape[:-1] != initial_phase.shape[:-1]:
        raise ValueError("cross and initial_phase batch shapes must match")
    phase = initial_phase
    identity = torch.eye(3, dtype=phase.dtype, device=phase.device)
    minimum_eigenvalues = []
    for step_value, damping_value in zip(step_schedule, damping_schedule):
        gradient, hessian = phase_gradient_hessian(
            phase, cross, modes, mode_weights
        )
        damping = phase.new_tensor(damping_value)
        curvature = -hessian + damping * identity
        minimum_eigenvalues.append(torch.linalg.eigvalsh(curvature)[..., 0])
        update = torch.linalg.solve(curvature, gradient.unsqueeze(-1)).squeeze(-1)
        phase = phase + phase.new_tensor(step_value) * update
    final_gradient, final_hessian = phase_gradient_hessian(
        phase, cross, modes, mode_weights
    )
    return PhaseResult(
        phase=phase,
        objective=phase_objective(phase, cross, modes, mode_weights),
        gradient=final_gradient,
        hessian=final_hessian,
        regularized_min_eigenvalues=torch.stack(minimum_eigenvalues, dim=-1),
    )


def validate_training_result(
    result: PhaseResult,
    minimum_regularized_curvature: float,
    maximum_gradient_norm: float,
) -> None:
    """Detached control-plane validation, never used inside training iterations."""

    tensors = (
        result.phase,
        result.objective,
        result.gradient,
        result.hessian,
        result.regularized_min_eigenvalues,
    )
    if not all(bool(torch.all(torch.isfinite(value))) for value in tensors):
        raise ValueError("phase solver produced non-finite values")
    if bool(
        torch.any(
            result.regularized_min_eigenvalues <= minimum_regularized_curvature
        )
    ):
        raise ValueError("fixed damping does not provide accepted curvature")
    if bool(
        torch.any(
            torch.linalg.vector_norm(result.gradient, dim=-1)
            > maximum_gradient_norm
        )
    ):
        raise ValueError("final phase-gradient residual is too large")
