"""Stable-branch evaluation phase search and local refinement."""

from __future__ import annotations

from typing import Sequence

import torch

from .newton import solve_training_phase
from .objective import phase_objective
from .stabilizer import stabilizer_equivalent
from .types import EvaluationPhaseError, EvaluationPhaseResult, TypedStabilizer


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
        raise EvaluationPhaseError(
            "INVALID_CANDIDATES", "candidate_offsets must have shape [J,3]"
        )
    if candidate_offsets.dtype != covariant_initial_phase.dtype or (
        candidate_offsets.device != covariant_initial_phase.device
    ):
        raise EvaluationPhaseError(
            "INVALID_CANDIDATES",
            "candidate offsets must share phase dtype and device",
        )
    if not bool(torch.all(torch.isfinite(candidate_offsets))):
        raise EvaluationPhaseError("INVALID_CANDIDATES", "candidate offsets are non-finite")
    cross_minimum = cross.abs().min()
    if not bool(torch.all(torch.isfinite(cross.abs()))) or bool(
        torch.any(cross.abs() <= minimum_cross_amplitude)
    ):
        raise EvaluationPhaseError(
            "CROSS_AMPLITUDE_TOO_SMALL",
            "runtime typed cross amplitude collapsed",
            observed=float(cross_minimum.detach().cpu()),
            threshold=minimum_cross_amplitude,
        )
    grouped_offsets, representative_indices = _group_non_equivalent_offsets(
        candidate_offsets, stabilizer, equivalence_tolerance
    )
    if grouped_offsets.shape[0] < 2:
        raise EvaluationPhaseError(
            "INVALID_CANDIDATES",
            "evaluation requires at least two non-equivalent candidates",
        )
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
        raise EvaluationPhaseError(
            "INVALID_CANDIDATES", "evaluation phase candidate objective is non-finite"
        )
    if bool(torch.any(gap <= minimum_gap)):
        raise EvaluationPhaseError(
            "NON_EQUIVALENT_GAP_TOO_SMALL",
            "best/second-best non-equivalent phase gap is too small",
            observed=float(gap.detach().min().cpu()),
            threshold=minimum_gap,
        )
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
        raise EvaluationPhaseError(
            "HESSIAN_CURVATURE_FAILURE",
            "refined phase Hessian is singular or insufficiently curved",
            observed=float(eigenvalues[..., 0].detach().min().cpu()),
            threshold=minimum_curvature,
        )
    if not bool(torch.all(torch.isfinite(condition))) or bool(
        torch.any(condition >= maximum_condition)
    ):
        raise EvaluationPhaseError(
            "HESSIAN_CONDITION_FAILURE",
            "refined phase Hessian condition is unacceptable",
            observed=float(condition.detach().max().cpu()),
            threshold=maximum_condition,
        )
    if not bool(torch.all(torch.isfinite(gradient_norm))) or bool(
        torch.any(gradient_norm >= maximum_gradient_norm)
    ):
        raise EvaluationPhaseError(
            "PHASE_RESIDUAL_TOO_LARGE",
            "final phase-gradient residual is too large",
            observed=float(gradient_norm.detach().max().cpu()),
            threshold=maximum_gradient_norm,
        )
    return EvaluationPhaseResult(
        refined=refined,
        selected_candidate=selected,
        selected_index=selected_index,
        non_equivalent_gap=gap,
        input_candidate_count=int(candidate_offsets.shape[0]),
        non_equivalent_group_count=int(grouped_offsets.shape[0]),
        selected_grouped_index=grouped_index,
        best_raw_score=best_two.values[..., 0],
        second_best_raw_score=best_two.values[..., 1],
    )
