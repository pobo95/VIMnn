"""Typed results for phase solvers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhaseResult:
    phase: torch.Tensor
    objective: torch.Tensor
    gradient: torch.Tensor
    hessian: torch.Tensor
    regularized_min_eigenvalues: torch.Tensor


@dataclass(frozen=True)
class EvaluationPhaseResult:
    refined: PhaseResult
    selected_candidate: torch.Tensor
    selected_index: torch.Tensor
    non_equivalent_gap: torch.Tensor


@dataclass(frozen=True)
class TypedStabilizer:
    translations: torch.Tensor
    permutations: torch.Tensor
