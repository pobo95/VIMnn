from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any
import torch

from refsite_mlip.transport import (
    CandidateReuseDecision,
    CompactCandidateNeighborState,
)


@dataclass(frozen=True)
class EvaluationDiagnostics:
    template_id: str
    template_fingerprint: str
    context_fingerprint: str
    policy_template_fingerprint: str
    policy_content_fingerprint: str
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
    transport_sinkhorn_warmup_iterations: int
    transport_fallback_sinkhorn_iterations: int
    transport_newton_iterations: int
    transport_cg_iterations: int
    transport_fallback_used: bool
    transport_fallback_reason: str | None
    transport_kind: str
    transport_r_on: float | None
    transport_r_off: float | None
    transport_r_candidate: float | None
    transport_core_edge_count: int | None
    transport_active_edge_count: int | None
    transport_candidate_edge_count: int | None
    transport_maximum_matching_size: int | None
    transport_total_support_feasible: bool | None
    transport_switch_on_boundary_gap: float | None
    transport_cutoff_boundary_gap: float | None
    transport_candidate_boundary_gap: float | None
    transport_line_search_reductions: int
    transport_accepted_damping: float | None
    transport_q_mass_error: float
    effective_transport_tolerance: float
    differentiability_scope: str
    hard_branch_frozen: bool
    derivative_order: int
    forces_requested: bool
    stress_requested: bool
    transport_backend: str = "dense"
    transport_active_dense_ratio: float | None = None
    transport_candidate_dense_ratio: float | None = None
    transport_support_fingerprint: str | None = None
    transport_dense_plan_materialized: bool = True
    transport_candidate_backend: str = "dense"
    transport_site_block_size: int | None = None
    transport_atom_block_size: int | None = None
    transport_processed_block_count: int = 0
    transport_maximum_pair_block_elements: int = 0
    transport_theoretical_full_pair_elements: int = 0
    transport_peak_temporary_geometry_elements: int = 0
    transport_dense_candidate_allocation_observed: bool = True
    transport_mic_image_gap: float = float("inf")
    transport_candidate_fingerprint: str | None = None


@dataclass(frozen=True)
class PotentialOutput:
    def __getitem__(self,key): return getattr(self,key)
    energy:torch.Tensor; site_energy:torch.Tensor; baseline_energy:torch.Tensor; residual_energy:torch.Tensor; site_features:torch.Tensor; raw_c:torch.Tensor; forces:torch.Tensor|None=None; stress:torch.Tensor|None=None; stress_voigt:torch.Tensor|None=None; auxiliary:dict[str,Any]|None=None; candidate_neighbor_state:CompactCandidateNeighborState|None=None; candidate_reuse_decision:CandidateReuseDecision|None=None


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
    candidate_neighbor_states: Mapping[
        str, CompactCandidateNeighborState
    ] | None = None
    candidate_reuse_decisions: Mapping[str, CandidateReuseDecision] | None = None

    def __getitem__(self, key):
        return getattr(self, key)
