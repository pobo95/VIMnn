"""Stable-branch evaluation phase search and local refinement."""

from __future__ import annotations

from typing import Sequence

import torch

from .newton import solve_training_phase
from .objective import phase_objective
from .stabilizer import stabilizer_equivalent
from .types import EvaluationPhaseResult, TypedStabilizer


def _group_non_equivalent_offsets(
    offsets: torch.Tensor, stabilizer: TypedStabilizer, tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one static representative per stabilizer-equivalent group."""

    representatives = []
    for candidate in range(offsets.shape[0]):
        if not any(
            stabilizer_equivalent(
                offsets[candidate], offsets[known], stabilizer, tolerance
            )
            for known in representatives
        ):
            representatives.append(candidate)
    indices = torch.tensor(representatives, dtype=torch.long, device=offsets.device)
    return offsets.index_select(0, indices), indices


def solve_evaluation_phase(
    cross: torch.Tensor,
    modes: torch.Tensor,
    mode_weights: torch.Tensor,
    covariant_initial_phase: torch.Tensor,
    candidate_offsets: torch.Tensor,
    stabilizer: TypedStabilizer,
    step_schedule: Sequence[float],
    damping_schedule: Sequence[float],
    *,
    minimum_gap: float,
    minimum_curvature: float,
    maximum_condition: float,
    maximum_gradient_norm: float,
    minimum_cross_amplitude: float,
    equivalence_tolerance: float = 1.0e-8,
) -> EvaluationPhaseResult:
    """Search covariant candidates, select a stable branch, and refine it."""

    if candidate_offsets.ndim != 2 or candidate_offsets.shape[1] != 3:
        raise ValueError("candidate_offsets must have shape [J,3]")
    if candidate_offsets.dtype != covariant_initial_phase.dtype or (
        candidate_offsets.device != covariant_initial_phase.device
    ):
        raise ValueError("candidate offsets must share phase dtype and device")
    if not bool(torch.all(torch.isfinite(cross.abs()))) or bool(
        torch.any(cross.abs() <= minimum_cross_amplitude)
    ):
        raise ValueError("runtime typed cross amplitude collapsed")
    grouped_offsets, representative_indices = _group_non_equivalent_offsets(
        candidate_offsets, stabilizer, equivalence_tolerance
    )
    if grouped_offsets.shape[0] < 2:
        raise ValueError("evaluation requires at least two non-equivalent candidates")
    candidates = covariant_initial_phase.unsqueeze(-2) + grouped_offsets
    values = phase_objective(
        candidates,
        cross.unsqueeze(-2),
        modes,
        mode_weights,
    )
    best_two = torch.topk(values, k=2, dim=-1)
    gap = best_two.values[..., 0] - best_two.values[..., 1]
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("evaluation phase candidate objective is non-finite")
    if bool(torch.any(gap <= minimum_gap)):
        raise ValueError("best/second-best non-equivalent phase gap is too small")
    grouped_index = best_two.indices[..., 0]
    selected_index = representative_indices[grouped_index]
    gather_index = grouped_index.unsqueeze(-1).unsqueeze(-1).expand(
        grouped_index.shape + (1, 3)
    )
    selected = torch.gather(candidates, -2, gather_index).squeeze(-2)
    refined = solve_training_phase(
        cross,
        modes,
        mode_weights,
        selected,
        step_schedule,
        damping_schedule,
    )
    negative_hessian = -refined.hessian
    eigenvalues = torch.linalg.eigvalsh(negative_hessian)
    condition = eigenvalues[..., -1] / eigenvalues[..., 0]
    gradient_norm = torch.linalg.vector_norm(refined.gradient, dim=-1)
    if not bool(torch.all(torch.isfinite(eigenvalues))) or bool(
        torch.any(eigenvalues[..., 0] <= minimum_curvature)
    ):
        raise ValueError("refined phase Hessian is singular or insufficiently curved")
    if not bool(torch.all(torch.isfinite(condition))) or bool(
        torch.any(condition >= maximum_condition)
    ):
        raise ValueError("refined phase Hessian condition is unacceptable")
    if not bool(torch.all(torch.isfinite(gradient_norm))) or bool(
        torch.any(gradient_norm >= maximum_gradient_norm)
    ):
        raise ValueError("final phase-gradient residual is too large")
    return EvaluationPhaseResult(refined, selected, selected_index, gap)
