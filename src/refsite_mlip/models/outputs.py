from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True)
class EvaluationDiagnostics:
    template_id: str
    template_fingerprint: str
    input_candidate_count: int
    non_equivalent_group_count: int
    selected_original_candidate_index: int
    selected_grouped_index: int
    best_raw_score: float
    second_best_non_equivalent_raw_score: float | None
    absolute_objective_gap: float
    selected_pre_refinement_phase: torch.Tensor
    refined_phase: torch.Tensor
    minimum_atomic_amplitude: float
    minimum_cross_amplitude: float
    minimum_reference_amplitude: float
    final_gradient_norm: float
    hessian_minimum_curvature: float
    hessian_maximum_curvature: float
    hessian_condition_number: float
    stabilizer_size: int
    alias_stabilizer_validated: bool
    phase_failure_reason: str | None
    transport_path: str
    transport_solver_name: str
    transport_row_residual: float
    transport_column_residual: float
    transport_sinkhorn_iterations: int
    transport_newton_iterations: int
    transport_cg_iterations: int
    transport_fallback_used: bool


@dataclass(frozen=True)
class PotentialOutput:
    def __getitem__(self,key): return getattr(self,key)
    energy:torch.Tensor; site_energy:torch.Tensor; baseline_energy:torch.Tensor; residual_energy:torch.Tensor; site_features:torch.Tensor; raw_c:torch.Tensor; forces:torch.Tensor|None=None; stress:torch.Tensor|None=None; stress_voigt:torch.Tensor|None=None; auxiliary:dict[str,Any]|None=None


@dataclass(frozen=True)
class BatchedPotentialOutput:
    """Ragged model outputs restored to the input structure/atom order."""

    energy: torch.Tensor
    baseline_energy: torch.Tensor
    residual_energy: torch.Tensor
    site_energy: torch.Tensor
    site_ptr: torch.Tensor
    site_batch: torch.Tensor
    forces: torch.Tensor | None
    stress: torch.Tensor | None
    stress_voigt: torch.Tensor | None
    sample_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    auxiliary: tuple[dict[str, Any] | None, ...] | None = None

    def __getitem__(self, key):
        return getattr(self, key)
