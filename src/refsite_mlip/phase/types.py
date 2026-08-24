"""Typed results for phase solvers."""

from dataclasses import dataclass
from typing import Any

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
    input_candidate_count: int = 0
    non_equivalent_group_count: int = 0
    selected_grouped_index: torch.Tensor | None = None
    best_raw_score: torch.Tensor | None = None
    second_best_raw_score: torch.Tensor | None = None


class EvaluationPhaseError(ValueError):
    """Structured evaluation-domain failure that remains ValueError-compatible."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        template_id: str | None = None,
        observed: Any = None,
        threshold: Any = None,
    ) -> None:
        self.reason_code = reason_code
        self.template_id = template_id
        self.observed = observed
        self.threshold = threshold
        context = f" template_id={template_id}" if template_id is not None else ""
        values = ""
        if observed is not None:
            values += f" observed={observed}"
        if threshold is not None:
            values += f" threshold={threshold}"
        super().__init__(f"[{reason_code}]{context} {message}{values}")


@dataclass(frozen=True)
class TypedStabilizer:
    translations: torch.Tensor
    permutations: torch.Tensor
