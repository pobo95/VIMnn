"""Deterministic, geometry-only qualification of automatic evaluation policy.

The audit is intentionally upstream of model construction.  It exercises the
existing phase, compact-support, adaptive-transport, and probability-feature
implementations on CPU, without reading labels or fitting thresholds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import torch

from refsite_mlip.features import (
    build_probability_multipoles,
    build_sparse_probability_multipoles,
)
from refsite_mlip.geometry.reference import aligned_reference_sites
from refsite_mlip.models import (
    PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1,
    EvaluationPolicy,
)
from refsite_mlip.phase import solve_evaluation_phase
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.modes import (
    validate_runtime_amplitudes,
    validate_static_mode_amplitudes,
)
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.stabilizer import (
    canonical_phase,
    integer_alias_order,
    stabilizer_equivalent,
    validate_alias_matches_stabilizer,
)
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    EvalOTConfig,
    atom_site_displacements,
    build_periodic_compact_transport_edges,
    minimum_image_diagnostics,
    solve_atom_vacancy_ot,
    solve_sparse_hybrid_eval,
    sparse_fixed_sinkhorn_updates,
    sparse_support_fingerprint,
    sparse_transport_plan,
)


AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION = (
    "automatic_evaluation_policy_audit_v1"
)
AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION = (
    "refsite_automatic_evaluation_certificate_v1"
)
AUTOMATIC_EVALUATION_SCOPE = "assigned_dataset_local_neighborhood"
AUTOMATIC_EVALUATION_CANDIDATE_GENERATION_VERSION = (
    "primary_mode_alias_torus_coverage_v1"
)

# The phase acceptance values come from the 9D production certificate and
# must never be weakened by automatic qualification.
_PRODUCTION_ACCEPTANCE = PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1
_PHASE_STEPS = _PRODUCTION_ACCEPTANCE.phase_step_schedule
_PHASE_DAMPING = _PRODUCTION_ACCEPTANCE.phase_damping_schedule
_MINIMUM_OBJECTIVE_GAP = _PRODUCTION_ACCEPTANCE.minimum_objective_gap_absolute
_MINIMUM_ATOMIC_AMPLITUDE = (
    _PRODUCTION_ACCEPTANCE.minimum_atomic_amplitude_absolute
)
_MINIMUM_REFERENCE_AMPLITUDE = (
    _PRODUCTION_ACCEPTANCE.minimum_reference_amplitude_absolute
)
_MINIMUM_CROSS_AMPLITUDE = (
    _PRODUCTION_ACCEPTANCE.minimum_cross_amplitude_absolute
)
_MINIMUM_CURVATURE = _PRODUCTION_ACCEPTANCE.minimum_curvature
_MAXIMUM_CONDITION = _PRODUCTION_ACCEPTANCE.maximum_condition
_MAXIMUM_GRADIENT_NORM = _PRODUCTION_ACCEPTANCE.maximum_gradient_norm
_EQUIVALENCE_TOLERANCE = _PRODUCTION_ACCEPTANCE.equivalence_tolerance
# The fixed production Newton schedule may leave separately seeded copies of
# the same stationary basin a few 1e-7 apart.  Permit their curvature/residual
# error bounds to overlap, but cap that numerical identity envelope at 1e-5 so
# it cannot stand in for candidate-basin coverage.
_REFINED_BASIN_EQUIVALENCE_MULTIPLIER = 1.0e3
_FLOAT64_TRANSPORT_TOLERANCE = 1.0e-12
_FLOAT32_TRANSPORT_TOLERANCE = 1.0e-6
_FLOAT64_ORACLE_TOLERANCE = 1.0e-10
_FLOAT32_ORACLE_TOLERANCE = 1.0e-5
_FLOAT32_ORACLE_RESIDUAL_TARGET = 2.0 * torch.finfo(torch.float32).eps
_FIRST_DERIVATIVE_FD_TOLERANCE = 5.0e-6
_POSITION_FD_STEP = 2.0e-6
_STRAIN_FD_STEP = 1.0e-4
_MAX_WITNESS_GEOMETRIES = 256
_MAX_PRIMARY_COORDINATE_BANDLIMIT = 7
_MAX_AUDIT_CANDIDATE_GROUPS = 4096
_ORACLE_SINKHORN_ITERATIONS = 1024
_MAX_NEWTON_ITERATIONS = 20
_PCG_MAX_ITERATIONS = 256
_PCG_ABSOLUTE_TOLERANCE = 1.0e-12
_PCG_RELATIVE_TOLERANCE = 1.0e-10
_GAUGE_RHO = 1.0
_ARMIJO_COEFFICIENT = 1.0e-4
_LINE_SEARCH_REDUCTION = 0.5
_MAX_LINE_SEARCH_REDUCTIONS = 12
_FALLBACK_SINKHORN_ITERATIONS = 1024
# Dimensionless normalization for the fixed feature-norm audit functional.
# It is part of the versioned profile (not inferred from the audited data).
_DERIVATIVE_SCALAR_SCALE = 1.0e-3


class AutomaticEvaluationPolicyAuditError(ValueError):
    """Structured all-or-nothing automatic policy qualification failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        template_id: str | None = None,
        sample_id: str | None = None,
        geometry_digest: str | None = None,
        probe: str | None = None,
        dtype: str | None = None,
        backend: str | None = None,
        observed: Any = None,
        required: Any = None,
        diagnostics: Any = None,
        original_error: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.message = message
        self.stage = "automatic_evaluation_policy.audit"
        self.template_id = template_id
        self.sample_id = sample_id
        self.geometry_digest = geometry_digest
        self.probe = probe
        self.dtype = dtype
        self.backend = backend
        self.observed = observed
        self.required = required
        self.diagnostics = diagnostics
        self.original_error = original_error
        context = " ".join(
            f"{key}={value!r}"
            for key, value in (
                ("template_id", template_id),
                ("sample_id", sample_id),
                ("geometry_digest", geometry_digest),
                ("probe", probe),
                ("dtype", dtype),
                ("backend", backend),
                ("observed", observed),
                ("required", required),
            )
            if value is not None
        )
        super().__init__(
            f"[{reason_code}] {message}" + ("" if not context else f"; {context}")
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise AutomaticEvaluationPolicyAuditError(
                "NONFINITE_AUDIT_RESULT", "audit metadata is nonfinite"
            )
        return value
    raise TypeError(f"audit metadata contains {type(value).__name__}")


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_fingerprint(value: torch.Tensor) -> str:
    snapshot = value.detach().cpu().contiguous()
    return _fingerprint(
        {
            "dtype": str(snapshot.dtype).removeprefix("torch."),
            "shape": list(snapshot.shape),
            "values": snapshot.tolist(),
        }
    )


@dataclass(frozen=True)
class _CandidateCoverage:
    runtime_candidates: torch.Tensor
    audit_candidates: torch.Tensor
    metadata: Mapping[str, Any]


def _canonical_candidate_groups(
    candidates: torch.Tensor,
    stabilizer: Any,
) -> torch.Tensor:
    canonical = canonical_phase(candidates.to(dtype=torch.float64, device="cpu"))
    translations = stabilizer.translations.detach().cpu().to(torch.float64)
    # A phase orbit is represented by the lexicographically smallest torus
    # point obtained by subtracting every exact typed-stabilizer translation.
    # This is O(candidate_count * stabilizer_size), independent of insertion
    # order, rather than a quadratic representative scan.
    orbits = canonical_phase(
        canonical[:, None, :] - translations[None, :, :]
    )
    # Stabilizer translations originate from canonical structural artifacts.
    # Snap only representation noise before lexicographic ordering; this does
    # not change the phase solver's acceptance tolerances.
    tick = 1.0e-12
    orbits = canonical_phase(torch.round(orbits / tick) * tick)
    representatives = sorted(
        {
            min(tuple(float(item) for item in member) for member in orbit)
            for orbit in orbits.tolist()
        }
    )
    if len(representatives) < 2:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "template-derived candidate generation produced fewer than two non-equivalent groups",
        )
    return torch.tensor(representatives, dtype=torch.float64)


def _candidate_grid(
    primary_modes: torch.Tensor,
    stabilizer: Any,
    resolution: tuple[int, int, int],
) -> tuple[torch.Tensor, int]:
    coordinates = torch.tensor(
        list(itertools.product(*(range(value) for value in resolution))),
        dtype=torch.float64,
    ) / torch.tensor(resolution, dtype=torch.float64)
    inverse = torch.linalg.inv(primary_modes.to(torch.float64))
    fundamental = coordinates @ inverse.T
    translations = stabilizer.translations.detach().cpu().to(torch.float64)
    raw = (
        fundamental[:, None, :] + translations[None, :, :]
    ).reshape(-1, 3)
    return _canonical_candidate_groups(raw, stabilizer), int(raw.shape[0])


def _candidate_coverage(template: Any) -> _CandidateCoverage:
    modes = template.phase_modes.detach().cpu().to(torch.long)
    stabilizer = template.stabilizer
    if modes.ndim != 2 or modes.shape[1] != 3 or modes.shape[0] < 3:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "automatic candidate generation requires at least three integer phase modes",
            template_id=getattr(template, "template_id", None),
        )
    primary = modes[:3].contiguous()
    try:
        validate_alias_matches_stabilizer(
            primary, stabilizer, tolerance=_EQUIVALENCE_TOLERANCE
        )
        alias_order = integer_alias_order(primary)
    except Exception as error:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "primary integer alias kernel does not equal the typed stabilizer",
            template_id=getattr(template, "template_id", None),
            original_error=error,
        ) from error
    inverse = torch.linalg.inv(primary.to(torch.float64))
    primary_coordinates = modes.to(torch.float64) @ inverse
    rounded = torch.round(primary_coordinates)
    coordinate_error = float(torch.max(torch.abs(primary_coordinates - rounded)))
    if coordinate_error > _EQUIVALENCE_TOLERANCE:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "phase modes are outside the exact primary-mode integer lattice",
            template_id=getattr(template, "template_id", None),
            observed=coordinate_error,
            required=f"<= {_EQUIVALENCE_TOLERANCE}",
        )
    integer_coordinates = rounded.to(torch.long)
    coordinate_bandlimits = torch.max(
        torch.abs(integer_coordinates), dim=0
    ).values
    bandlimit = int(coordinate_bandlimits.max())
    l1_bandlimit = int(
        torch.sum(torch.abs(integer_coordinates), dim=1).max()
    )
    if bandlimit > _MAX_PRIMARY_COORDINATE_BANDLIMIT:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "primary-coordinate phase bandlimit exceeds the versioned automatic coverage domain",
            template_id=getattr(template, "template_id", None),
            observed=bandlimit,
            required=f"<= {_MAX_PRIMARY_COORDINATE_BANDLIMIT}",
        )
    # The phase objective is a trigonometric polynomial in the primary-mode
    # torus coordinates.  Use the odd Nyquist grid for its maximum coordinate
    # bandlimit, then audit against twice that resolution.
    runtime_resolution = tuple(
        max(3, 2 * int(value) + 1)
        for value in coordinate_bandlimits.tolist()
    )
    audit_resolution = tuple(2 * value for value in runtime_resolution)
    audit_group_bound = math.prod(audit_resolution)
    if audit_group_bound > _MAX_AUDIT_CANDIDATE_GROUPS:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "template-derived broader candidate grid exceeds the fixed coverage bound",
            template_id=getattr(template, "template_id", None),
            observed=audit_group_bound,
            required=f"<= {_MAX_AUDIT_CANDIDATE_GROUPS}",
        )
    runtime, runtime_raw = _candidate_grid(
        primary, stabilizer, runtime_resolution
    )
    broader, broader_raw = _candidate_grid(
        primary, stabilizer, audit_resolution
    )
    metadata = {
        "convention_version": AUTOMATIC_EVALUATION_CANDIDATE_GENERATION_VERSION,
        "primary_mode_matrix": primary.tolist(),
        "primary_coordinate_mode_matrix": integer_coordinates.tolist(),
        "primary_coordinate_bandlimit": bandlimit,
        "primary_coordinate_l1_bandlimit": l1_bandlimit,
        "primary_coordinate_axis_bandlimits": (
            coordinate_bandlimits.tolist()
        ),
        "maximum_supported_primary_coordinate_bandlimit": (
            _MAX_PRIMARY_COORDINATE_BANDLIMIT
        ),
        "alias_kernel_order": alias_order,
        "alias_kernel_fingerprint": _fingerprint(
            {"primary_mode_matrix": primary.tolist(), "order": alias_order}
        ),
        "typed_stabilizer_fingerprint": _fingerprint(
            {
                "elements": sorted(
                    (
                        [float(value) for value in translation],
                        [int(value) for value in permutation],
                    )
                    for translation, permutation in zip(
                        stabilizer.translations.detach().cpu().tolist(),
                        stabilizer.permutations.detach().cpu().tolist(),
                    )
                )
            }
        ),
        "runtime_grid_resolution": list(runtime_resolution),
        "broader_audit_grid_resolution": list(audit_resolution),
        "runtime_raw_candidate_count": runtime_raw,
        "broader_audit_raw_candidate_count": broader_raw,
        "runtime_candidate_count": int(runtime.shape[0]),
        "broader_audit_candidate_count": int(broader.shape[0]),
        "runtime_candidate_fingerprint": _tensor_fingerprint(runtime),
        "broader_audit_candidate_fingerprint": _tensor_fingerprint(broader),
        "stabilizer_reduction_rule": (
            "canonical torus lexicographic representatives under typed translations"
        ),
    }
    return _CandidateCoverage(runtime, broader, MappingProxyType(metadata))


def automatic_evaluation_policy_profile() -> Mapping[str, Any]:
    """Return the immutable fixed profile used by automatic qualification."""

    return MappingProxyType(
        {
            "convention_version": AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION,
            "production_acceptance_convention_version": (
                _PRODUCTION_ACCEPTANCE.convention_version
            ),
            "candidate_generation_convention_version": (
                AUTOMATIC_EVALUATION_CANDIDATE_GENERATION_VERSION
            ),
            "runtime_candidate_grid": (
                "axiswise primary-coordinate odd Nyquist resolution=max(3,2*b_j+1)"
            ),
            "broader_audit_candidate_grid": "2*runtime resolution",
            "maximum_primary_coordinate_bandlimit": (
                _MAX_PRIMARY_COORDINATE_BANDLIMIT
            ),
            "maximum_audit_candidate_groups": _MAX_AUDIT_CANDIDATE_GROUPS,
            "phase_step_schedule": _PHASE_STEPS,
            "phase_damping_schedule": _PHASE_DAMPING,
            "minimum_objective_gap_absolute": _MINIMUM_OBJECTIVE_GAP,
            "minimum_cross_amplitude_absolute": _MINIMUM_CROSS_AMPLITUDE,
            "minimum_atomic_amplitude_absolute": _MINIMUM_ATOMIC_AMPLITUDE,
            "minimum_reference_amplitude_absolute": _MINIMUM_REFERENCE_AMPLITUDE,
            "minimum_curvature": _MINIMUM_CURVATURE,
            "maximum_condition": _MAXIMUM_CONDITION,
            "maximum_gradient_norm": _MAXIMUM_GRADIENT_NORM,
            "equivalence_tolerance": _EQUIVALENCE_TOLERANCE,
            "refined_basin_equivalence_maximum": (
                _REFINED_BASIN_EQUIVALENCE_MULTIPLIER
                * _EQUIVALENCE_TOLERANCE
            ),
            "transport_tolerance_float64": _FLOAT64_TRANSPORT_TOLERANCE,
            "transport_tolerance_float32": _FLOAT32_TRANSPORT_TOLERANCE,
            "oracle_sinkhorn_iterations": _ORACLE_SINKHORN_ITERATIONS,
            "oracle_tolerance_float64": _FLOAT64_ORACLE_TOLERANCE,
            "oracle_tolerance_float32": _FLOAT32_ORACLE_TOLERANCE,
            "oracle_residual_target_float32": _FLOAT32_ORACLE_RESIDUAL_TARGET,
            "max_newton_iterations": _MAX_NEWTON_ITERATIONS,
            "pcg_max_iterations": _PCG_MAX_ITERATIONS,
            "pcg_absolute_tolerance": _PCG_ABSOLUTE_TOLERANCE,
            "pcg_relative_tolerance": _PCG_RELATIVE_TOLERANCE,
            "gauge_rho": _GAUGE_RHO,
            "armijo_coefficient": _ARMIJO_COEFFICIENT,
            "line_search_reduction": _LINE_SEARCH_REDUCTION,
            "max_line_search_reductions": _MAX_LINE_SEARCH_REDUCTIONS,
            "fallback_sinkhorn_iterations": _FALLBACK_SINKHORN_ITERATIONS,
            "first_derivative_fd_tolerance": _FIRST_DERIVATIVE_FD_TOLERANCE,
            "position_fd_step": _POSITION_FD_STEP,
            "strain_fd_step": _STRAIN_FD_STEP,
            "max_witness_geometries": _MAX_WITNESS_GEOMETRIES,
            "derivative_scalar_scale": _DERIVATIVE_SCALAR_SCALE,
        }
    )


def build_automatic_evaluation_policy(template: Any) -> EvaluationPolicy:
    """Bind the fixed candidate policy to one already-built template."""

    coverage = _candidate_coverage(template)
    return EvaluationPolicy(
        template_id=template.template_id,
        template_fingerprint=template.fingerprint,
        candidate_offsets=coverage.runtime_candidates,
        phase_step_schedule=_PHASE_STEPS,
        phase_damping_schedule=_PHASE_DAMPING,
        minimum_objective_gap_absolute=_MINIMUM_OBJECTIVE_GAP,
        minimum_cross_amplitude_absolute=_MINIMUM_CROSS_AMPLITUDE,
        minimum_atomic_amplitude_absolute=_MINIMUM_ATOMIC_AMPLITUDE,
        minimum_reference_amplitude_absolute=_MINIMUM_REFERENCE_AMPLITUDE,
        minimum_curvature=_MINIMUM_CURVATURE,
        maximum_condition=_MAXIMUM_CONDITION,
        maximum_gradient_norm=_MAXIMUM_GRADIENT_NORM,
        equivalence_tolerance=_EQUIVALENCE_TOLERANCE,
    )


@dataclass(frozen=True)
class _Outcome:
    scalar: torch.Tensor
    phase: torch.Tensor
    edge_or_plan: torch.Tensor
    q: torch.Tensor
    multipoles: torch.Tensor
    diagnostics: Mapping[str, Any]
    branch_signature: tuple[Any, ...]


def _transport_config(config: Any, dtype: torch.dtype) -> EvalOTConfig:
    return EvalOTConfig(
        sinkhorn_iterations=config.eval_sinkhorn_warmup_iterations,
        max_newton_iterations=_MAX_NEWTON_ITERATIONS,
        convergence_tolerance=(
            _FLOAT32_TRANSPORT_TOLERANCE
            if dtype == torch.float32
            else _FLOAT64_TRANSPORT_TOLERANCE
        ),
        pcg_max_iterations=_PCG_MAX_ITERATIONS,
        pcg_absolute_tolerance=_PCG_ABSOLUTE_TOLERANCE,
        pcg_relative_tolerance=_PCG_RELATIVE_TOLERANCE,
        gauge_rho=_GAUGE_RHO,
        armijo_coefficient=_ARMIJO_COEFFICIENT,
        line_search_reduction=_LINE_SEARCH_REDUCTION,
        max_line_search_reductions=_MAX_LINE_SEARCH_REDUCTIONS,
        fallback_sinkhorn_iterations=_FALLBACK_SINKHORN_ITERATIONS,
    )


def _tensor_maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return math.inf
    if not left.numel():
        return 0.0
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _mic_branch_fingerprint(
    *,
    site_index: torch.Tensor,
    atom_identity_per_edge: torch.Tensor,
    periodic_shift: torch.Tensor,
) -> str:
    records = sorted(
        (
            int(site),
            int(atom_identity),
            *(int(value) for value in shift),
        )
        for site, atom_identity, shift in zip(
            site_index.detach().cpu().tolist(),
            atom_identity_per_edge.detach().cpu().tolist(),
            periodic_shift.detach().cpu().to(torch.long).tolist(),
        )
    )
    return hashlib.sha256(repr(records).encode("utf-8")).hexdigest()


def _semantic_support_fingerprint(
    *,
    site_index: torch.Tensor,
    atom_identity_per_edge: torch.Tensor,
    active: torch.Tensor,
) -> str:
    records = sorted(
        (
            int(site),
            int(atom_identity),
            bool(is_active),
        )
        for site, atom_identity, is_active in zip(
            site_index.detach().cpu().tolist(),
            atom_identity_per_edge.detach().cpu().tolist(),
            active.detach().cpu().tolist(),
        )
    )
    return hashlib.sha256(repr(records).encode("utf-8")).hexdigest()


def _evaluate(
    audit_input: Any,
    geometry: Any,
    policy: EvaluationPolicy,
    config: Any,
    dtype: torch.dtype,
    *,
    positions: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    origin: torch.Tensor | None = None,
    atomic_numbers: torch.Tensor | None = None,
    atom_identity: torch.Tensor | None = None,
    oracle: bool = True,
) -> _Outcome:
    runtime = audit_input.context.materialize(device="cpu", dtype=dtype)
    positions = (
        geometry.positions_tensor().to(dtype=dtype)
        if positions is None
        else positions
    )
    cell = geometry.cell_tensor().to(dtype=dtype) if cell is None else cell
    origin = torch.zeros(3, dtype=dtype) if origin is None else origin
    atomic_numbers = (
        torch.tensor(geometry.atomic_numbers, dtype=torch.long)
        if atomic_numbers is None
        else atomic_numbers
    )
    atom_identity = (
        torch.arange(atomic_numbers.shape[0], dtype=torch.long)
        if atom_identity is None
        else atom_identity
    )
    vocabulary = torch.tensor(config.species_vocabulary, dtype=torch.long)
    matches = atomic_numbers[:, None] == vocabulary[None, :]
    if bool(torch.any(matches.sum(-1) != 1)):
        raise ValueError("audit geometry contains an unsupported species")
    species_indices = torch.argmax(matches.to(torch.long), dim=-1)
    alignment = torch.tensor(
        audit_input.species_alignment_weights, dtype=dtype
    )
    atom_weights = alignment[species_indices]
    atomic, reference, cross = typed_reciprocal_fields(
        positions,
        origin,
        cell,
        runtime.topology.reference_fractional,
        atom_weights,
        runtime.site_alignment_weights,
        runtime.phase_modes,
        runtime.phase_channel_weights,
    )
    validate_alias_matches_stabilizer(
        runtime.phase_modes[:3], runtime.stabilizer, policy.equivalence_tolerance
    )
    validate_alias_matches_stabilizer(
        runtime.phase_modes, runtime.stabilizer, policy.equivalence_tolerance
    )
    reference_amplitude = validate_static_mode_amplitudes(
        reference,
        runtime.phase_channel_weights,
        policy.minimum_reference_amplitude_absolute,
    )
    atomic_amplitude, cross_amplitude = validate_runtime_amplitudes(
        atomic,
        cross,
        runtime.phase_channel_weights,
        policy.minimum_atomic_amplitude_absolute,
        policy.minimum_cross_amplitude_absolute,
    )
    initial = primary_phase_initialization(
        cross[:3], runtime.phase_modes[:3]
    )
    evaluation = solve_evaluation_phase(
        cross,
        runtime.phase_modes,
        runtime.phase_mode_weights,
        initial,
        policy.materialize_candidate_offsets(device="cpu", dtype=dtype),
        runtime.stabilizer,
        policy.phase_step_schedule,
        policy.phase_damping_schedule,
        minimum_gap=policy.minimum_objective_gap_absolute,
        minimum_curvature=policy.minimum_curvature,
        maximum_condition=policy.maximum_condition,
        maximum_gradient_norm=policy.maximum_gradient_norm,
        minimum_cross_amplitude=policy.minimum_cross_amplitude_absolute,
        equivalence_tolerance=policy.equivalence_tolerance,
    )
    phase = evaluation.refined.phase
    references = aligned_reference_sites(
        runtime.topology.reference_fractional, phase, origin, cell
    )
    support_config = config.transport_support
    eval_config = _transport_config(config, dtype)
    sparse = support_config.backend == "edge_list"
    if sparse:
        edges = build_periodic_compact_transport_edges(
            positions,
            references,
            cell,
            runtime.topology.pbc,
            origin=origin,
            epsilon_ot=config.epsilon_ot,
            ell_ot=config.ell_ot,
            config=support_config,
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
        )
        ot = solve_sparse_hybrid_eval(edges, eval_config)
        if ot.fallback_used:
            raise AutomaticEvaluationPolicyAuditError(
                "ADAPTIVE_TRANSPORT_FALLBACK",
                "adaptive edge-list transport used fallback",
                template_id=audit_input.template_id,
                sample_id=geometry.sample_id,
                geometry_digest=geometry.semantic_digest,
                dtype=str(dtype).removeprefix("torch."),
                backend="edge_list",
                diagnostics={"failure_reason": ot.failure_reason},
            )
        feature = build_sparse_probability_multipoles(
            ot.edge_plan,
            ot.q,
            ot.edges,
            atomic_numbers,
            config.feature,
            runtime.topology.site_types,
        )
        production_support_fingerprint = sparse_support_fingerprint(edges)
        semantic_support_fingerprint = _semantic_support_fingerprint(
            site_index=edges.site_index,
            atom_identity_per_edge=atom_identity[edges.atom_index],
            active=edges.active,
        )
        mic_branch_fingerprint = _mic_branch_fingerprint(
            site_index=edges.site_index,
            atom_identity_per_edge=atom_identity[edges.atom_index],
            periodic_shift=edges.periodic_shift,
        )
        plan = ot.edge_plan
        if oracle:
            duals = sparse_fixed_sinkhorn_updates(
                edges, _ORACLE_SINKHORN_ITERATIONS
            )
            oracle_plan, oracle_q = sparse_transport_plan(
                edges, duals.f, duals.g
            )
            oracle_feature = build_sparse_probability_multipoles(
                oracle_plan,
                oracle_q,
                edges,
                atomic_numbers,
                config.feature,
                runtime.topology.site_types,
            )
            oracle_errors = {
                "plan": _tensor_maximum_error(plan, oracle_plan),
                "q": _tensor_maximum_error(ot.q, oracle_q),
                "multipoles": _tensor_maximum_error(
                    feature.equivariant_features,
                    oracle_feature.equivariant_features,
                ),
            }
        else:
            oracle_errors = {"plan": 0.0, "q": 0.0, "multipoles": 0.0}
        support = edges.support_diagnostics
        dense_plan_materialized = ot.dense_plan_materialized
    else:
        displacements = atom_site_displacements(
            positions, references, cell, runtime.topology.pbc
        )
        distances = torch.linalg.vector_norm(displacements, dim=-1)
        cost = displacements.square().sum(-1) / (2.0 * config.ell_ot**2)
        ot = solve_atom_vacancy_ot(
            cost,
            config.epsilon_ot,
            EVAL_ADAPTIVE,
            "hybrid",
            eval_config,
            support_config=support_config,
            atom_distances=distances,
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
        )
        if ot.fallback_used:
            raise AutomaticEvaluationPolicyAuditError(
                "ADAPTIVE_TRANSPORT_FALLBACK",
                "adaptive dense transport used fallback",
                template_id=audit_input.template_id,
                sample_id=geometry.sample_id,
                geometry_digest=geometry.semantic_digest,
                dtype=str(dtype).removeprefix("torch."),
                backend="dense",
                diagnostics={"failure_reason": ot.failure_reason},
            )
        feature = build_probability_multipoles(
            ot.P,
            ot.q,
            atomic_numbers,
            displacements,
            config.feature,
            runtime.topology.site_types,
        )
        support = ot.support_diagnostics
        active = distances < distances.new_tensor(support_config.cutoff)
        site_index = torch.arange(
            runtime.topology.num_sites, dtype=torch.long
        )[:, None].expand_as(active).reshape(-1)
        atom_index = torch.arange(
            atomic_numbers.shape[0], dtype=torch.long
        )[None, :].expand_as(active).reshape(-1)
        semantic_support_fingerprint = _semantic_support_fingerprint(
            site_index=site_index,
            atom_identity_per_edge=atom_identity[atom_index],
            active=active.reshape(-1),
        )
        production_support_fingerprint = semantic_support_fingerprint
        raw = positions.unsqueeze(0) - references.unsqueeze(1)
        mic = minimum_image_diagnostics(raw, cell, runtime.topology.pbc)
        # Dense minimum-image diagnostics include every site/atom pair, while
        # the certified transport branch only owns pairs inside its support.
        # Hashing excluded pairs would make harmless, exactly tied remote
        # images part of the policy architecture and has no sparse analogue.
        mic_branch_fingerprint = _mic_branch_fingerprint(
            site_index=site_index[active.reshape(-1)],
            atom_identity_per_edge=atom_identity[atom_index][
                active.reshape(-1)
            ],
            periodic_shift=mic.periodic_shift[active],
        )
        plan = ot.P
        dense_plan_materialized = True
        if oracle:
            oracle_config = replace(
                eval_config,
                sinkhorn_iterations=_ORACLE_SINKHORN_ITERATIONS,
                convergence_tolerance=(
                    _FLOAT32_ORACLE_RESIDUAL_TARGET
                    if dtype == torch.float32
                    else _FLOAT64_TRANSPORT_TOLERANCE
                ),
            )
            oracle_ot = solve_atom_vacancy_ot(
                cost,
                config.epsilon_ot,
                EVAL_ADAPTIVE,
                "sinkhorn",
                oracle_config,
                support_config=support_config,
                atom_distances=distances,
                template_id=audit_input.template_id,
                sample_id=geometry.sample_id,
            )
            oracle_feature = build_probability_multipoles(
                oracle_ot.P,
                oracle_ot.q,
                atomic_numbers,
                displacements,
                config.feature,
                runtime.topology.site_types,
            )
            oracle_errors = {
                "plan": _tensor_maximum_error(plan, oracle_ot.P),
                "q": _tensor_maximum_error(ot.q, oracle_ot.q),
                "multipoles": _tensor_maximum_error(
                    feature.equivariant_features,
                    oracle_feature.equivariant_features,
                ),
            }
        else:
            oracle_errors = {"plan": 0.0, "q": 0.0, "multipoles": 0.0}

    tolerance = (
        _FLOAT32_ORACLE_TOLERANCE
        if dtype == torch.float32
        else _FLOAT64_ORACLE_TOLERANCE
    )
    worst_oracle = max(oracle_errors.values())
    if worst_oracle > tolerance:
        raise AutomaticEvaluationPolicyAuditError(
            "TRANSPORT_ORACLE_MISMATCH",
            "adaptive transport differs from the fixed high-accuracy Sinkhorn oracle",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            geometry_digest=geometry.semantic_digest,
            dtype=str(dtype).removeprefix("torch."),
            backend=support_config.backend,
            observed=worst_oracle,
            required=f"<= {tolerance}",
            diagnostics=oracle_errors,
        )
    curvature = torch.linalg.eigvalsh(-evaluation.refined.hessian)
    condition = curvature[-1] / curvature[0]
    residual = torch.linalg.vector_norm(evaluation.refined.gradient)
    transport_residual = torch.maximum(ot.row_residual, ot.column_residual)
    required_transport = eval_config.convergence_tolerance
    q_mass_error = torch.abs(
        ot.q.sum() - ot.q.new_tensor(runtime.topology.num_sites - positions.shape[0])
    )
    finite_tensors = (phase, plan, ot.q, feature.equivariant_features)
    if any(not bool(torch.all(torch.isfinite(value)).detach()) for value in finite_tensors):
        raise AutomaticEvaluationPolicyAuditError(
            "NONFINITE_AUDIT_RESULT",
            "phase/transport/probability audit produced a nonfinite tensor",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            dtype=str(dtype).removeprefix("torch."),
            backend=support_config.backend,
        )
    if (
        float(transport_residual.detach()) > required_transport
        or float(q_mass_error.detach()) > required_transport
    ):
        raise AutomaticEvaluationPolicyAuditError(
            "TRANSPORT_CONVERGENCE_FAILURE",
            "adaptive transport residual exceeds the existing dtype tolerance",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            dtype=str(dtype).removeprefix("torch."),
            backend=support_config.backend,
            observed=max(float(transport_residual.detach()), float(q_mass_error.detach())),
            required=f"<= {required_transport}",
        )
    scalar = _DERIVATIVE_SCALAR_SCALE * (
        feature.equivariant_features.square().mean()
        + ot.q.square().mean()
        + 0.01 * torch.cos(2.0 * math.pi * phase).mean()
    )
    support_margin = math.inf
    mic_margin = math.inf
    if support is not None:
        support_margin = min(
            support.cutoff_boundary_gap,
            support.switch_on_boundary_gap,
            support.candidate_boundary_gap,
        )
        mic_margin = support.mic_image_gap
    if not sparse:
        mic_margin = float(mic.unique_image_gap[active].min().detach())
    if not math.isfinite(support_margin) or not math.isfinite(mic_margin):
        raise AutomaticEvaluationPolicyAuditError(
            "NONFINITE_AUDIT_RESULT",
            "support/MIC branch margin is nonfinite",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            geometry_digest=geometry.semantic_digest,
            dtype=str(dtype).removeprefix("torch."),
            backend=support_config.backend,
            diagnostics={
                "support_margin": support_margin,
                "mic_margin": mic_margin,
            },
        )
    # The dense backend makes a single MIC argmin choice and therefore needs
    # a strict positive image gap.  Sparse periodic enumeration owns every
    # explicit candidate image; a zero nearest-image gap is not an unresolved
    # branch when the complete tied image set and its fingerprint are stable.
    if support_margin <= 0.0 or (not sparse and mic_margin <= 0.0):
        raise AutomaticEvaluationPolicyAuditError(
            "SUPPORT_BRANCH_UNSTABLE",
            "support or minimum-image branch lies on its decision boundary",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            geometry_digest=geometry.semantic_digest,
            dtype=str(dtype).removeprefix("torch."),
            backend=support_config.backend,
            observed=min(support_margin, mic_margin),
            required="> 0",
            diagnostics={
                "support_margin": support_margin,
                "mic_margin": mic_margin,
            },
        )
    from .automatic_reference import _strain as production_row_vector_strain

    observed_strain = production_row_vector_strain(
        audit_input.context.topology.reference_cell,
        geometry.cell_tensor(),
    )
    total_transport_work = (
        ot.warmup_sinkhorn_iterations
        + ot.newton_iterations
        + ot.cg_iterations
        + ot.line_search_reductions
    )
    diagnostics = {
        "sample_id": geometry.sample_id,
        "geometry_digest": geometry.semantic_digest,
        "split": geometry.split,
        "K": runtime.topology.num_sites - positions.shape[0],
        "composition": [
            int((atomic_numbers == value).sum()) for value in vocabulary
        ],
        "dtype": str(dtype).removeprefix("torch."),
        "backend": support_config.backend,
        "selected_group": int(evaluation.selected_grouped_index.detach()),
        "selected_candidate": int(evaluation.selected_index.detach()),
        "objective_gap": float(evaluation.non_equivalent_gap.detach()),
        "minimum_atomic_amplitude": float(atomic_amplitude.min().detach()),
        "minimum_reference_amplitude": float(reference_amplitude.min().detach()),
        "minimum_cross_amplitude": float(cross_amplitude.min().detach()),
        "hessian_minimum_curvature": float(curvature[0].detach()),
        "hessian_condition": float(condition.detach()),
        "phase_residual": float(residual.detach()),
        "support_margin": float(support_margin),
        "mic_margin": float(mic_margin),
        "support_fingerprint": production_support_fingerprint,
        "semantic_support_fingerprint": semantic_support_fingerprint,
        "mic_branch_fingerprint": mic_branch_fingerprint,
        "transport_row_residual": float(ot.row_residual.detach()),
        "transport_column_residual": float(ot.column_residual.detach()),
        "q_mass_error": float(q_mass_error.detach()),
        "observed_strain": observed_strain,
        "sinkhorn_warmup_iterations": ot.warmup_sinkhorn_iterations,
        "newton_iterations": ot.newton_iterations,
        "cg_iterations": ot.cg_iterations,
        "line_search_reductions": ot.line_search_reductions,
        "total_transport_work": total_transport_work,
        "fallback_used": ot.fallback_used,
        "oracle_maximum_errors": oracle_errors,
        "dense_plan_materialized": dense_plan_materialized,
    }
    branch = (
        diagnostics["selected_group"],
        diagnostics["semantic_support_fingerprint"],
        diagnostics["mic_branch_fingerprint"],
        support_config.backend,
        False,
    )
    return _Outcome(
        scalar=scalar,
        phase=phase,
        edge_or_plan=plan,
        q=ot.q,
        multipoles=feature.equivariant_features,
        diagnostics=diagnostics,
        branch_signature=branch,
    )


def _symmetric_directions(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    values = []
    for axis in range(3):
        value = torch.zeros((3, 3), dtype=dtype)
        value[axis, axis] = 1.0
        values.append(value)
    for left, right in ((1, 2), (0, 2), (0, 1)):
        value = torch.zeros((3, 3), dtype=dtype)
        value[left, right] = value[right, left] = 0.5
        values.append(value)
    return tuple(values)


def _position_directions(positions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    index = torch.arange(positions.numel(), dtype=positions.dtype).reshape_as(positions)
    first = torch.sin(index + 1.0)
    second = torch.cos(0.5 * index + 0.25)
    values = []
    for direction in (first, second):
        direction = direction - direction.mean(dim=0, keepdim=True)
        norm = torch.linalg.vector_norm(direction)
        if float(norm) > 0.0:
            values.append(direction / norm)
    return tuple(values)


def _canonical_phase_group(
    phase: torch.Tensor,
    stabilizer: Any,
    tolerance: float = _EQUIVALENCE_TOLERANCE,
) -> tuple[float, float, float]:
    orbit = canonical_phase(
        phase.detach().cpu().to(torch.float64).unsqueeze(0)
        + stabilizer.translations.detach().cpu().to(torch.float64)
    )
    orbit = canonical_phase(torch.round(orbit / tolerance) * tolerance)
    return min(tuple(float(item) for item in row) for row in orbit.tolist())


def _refined_stabilizer_equivalent(
    left: torch.Tensor,
    right: torch.Tensor,
    stabilizer: Any,
    tolerance: float,
    *,
    left_stationarity_bound: float = 0.0,
    right_stationarity_bound: float = 0.0,
) -> bool:
    stationarity_allowance = min(
        left_stationarity_bound + right_stationarity_bound,
        (_REFINED_BASIN_EQUIVALENCE_MULTIPLIER - 1.0) * tolerance,
    )
    effective_tolerance = tolerance + stationarity_allowance
    left_canonical = canonical_phase(
        torch.round(canonical_phase(left) / tolerance) * tolerance
    )
    right_canonical = canonical_phase(
        torch.round(canonical_phase(right) / tolerance) * tolerance
    )
    return stabilizer_equivalent(
        left_canonical,
        right_canonical,
        stabilizer,
        effective_tolerance,
    )


def _compare_phase_candidate_sets(
    *,
    cross: torch.Tensor,
    modes: torch.Tensor,
    mode_weights: torch.Tensor,
    initial: torch.Tensor,
    stabilizer: Any,
    policy: EvaluationPolicy,
    runtime_candidates: torch.Tensor,
    audit_candidates: torch.Tensor,
    template_id: str,
    sample_id: str,
    geometry_digest: str,
) -> dict[str, Any]:
    """Require runtime seeds to cover the broader accepted optimum basin."""

    runtime_result = solve_evaluation_phase(
        cross,
        modes,
        mode_weights,
        initial,
        runtime_candidates,
        stabilizer,
        policy.phase_step_schedule,
        policy.phase_damping_schedule,
        minimum_gap=policy.minimum_objective_gap_absolute,
        minimum_curvature=policy.minimum_curvature,
        maximum_condition=policy.maximum_condition,
        maximum_gradient_norm=policy.maximum_gradient_norm,
        minimum_cross_amplitude=policy.minimum_cross_amplitude_absolute,
        equivalence_tolerance=policy.equivalence_tolerance,
    )
    candidates = initial.unsqueeze(0) + audit_candidates
    repeated_cross = cross.unsqueeze(0).expand(candidates.shape[0], -1)
    refined = solve_training_phase(
        repeated_cross,
        modes,
        mode_weights,
        candidates,
        policy.phase_step_schedule,
        policy.phase_damping_schedule,
    )
    curvature = torch.linalg.eigvalsh(-refined.hessian)
    condition = curvature[:, -1] / curvature[:, 0]
    residual = torch.linalg.vector_norm(refined.gradient, dim=-1)
    finite = (
        torch.isfinite(refined.objective)
        & torch.all(torch.isfinite(refined.phase), dim=-1)
        & torch.all(torch.isfinite(curvature), dim=-1)
        & torch.isfinite(condition)
        & torch.isfinite(residual)
    )
    accepted = (
        finite
        & (curvature[:, 0] > policy.minimum_curvature)
        & (condition < policy.maximum_condition)
        & (residual < policy.maximum_gradient_norm)
    )
    indices = torch.nonzero(accepted).reshape(-1).tolist()
    if not indices:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "broader audit candidate set found no phase satisfying production acceptance",
            template_id=template_id,
            sample_id=sample_id,
            geometry_digest=geometry_digest,
        )
    indices.sort(
        key=lambda index: (
            -float(refined.objective[index].detach()),
            _canonical_phase_group(refined.phase[index], stabilizer),
            index,
        )
    )
    best = indices[0]
    second = next(
        (
            index
            for index in indices[1:]
            if not _refined_stabilizer_equivalent(
                refined.phase[best],
                refined.phase[index],
                stabilizer,
                policy.equivalence_tolerance,
                left_stationarity_bound=float(
                    residual[best] / curvature[best, 0]
                ),
                right_stationarity_bound=float(
                    residual[index] / curvature[index, 0]
                ),
            )
        ),
        None,
    )
    # A single accepted basin is a valid (and especially strong) coverage
    # result: every broader audit seed converged to the basin available to the
    # runtime candidate set.  A gap is meaningful only when a second accepted
    # non-equivalent basin exists.
    broader_gap = (
        None
        if second is None
        else float(
            (refined.objective[best] - refined.objective[second]).detach()
        )
    )
    runtime_objective = float(runtime_result.refined.objective.detach())
    broader_objective = float(refined.objective[best].detach())
    coverage_gap = broader_objective - runtime_objective
    runtime_curvature = torch.linalg.eigvalsh(
        -runtime_result.refined.hessian
    )[0]
    runtime_residual = torch.linalg.vector_norm(
        runtime_result.refined.gradient
    )
    same_group = _refined_stabilizer_equivalent(
        refined.phase[best],
        runtime_result.refined.phase,
        stabilizer,
        policy.equivalence_tolerance,
        left_stationarity_bound=float(residual[best] / curvature[best, 0]),
        right_stationarity_bound=float(runtime_residual / runtime_curvature),
    )
    unresolved_alternate = (
        broader_gap is not None
        and broader_gap <= policy.minimum_objective_gap_absolute
    )
    if unresolved_alternate or not same_group:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "runtime candidates do not uniquely cover the best basin found by the broader audit set",
            template_id=template_id,
            sample_id=sample_id,
            geometry_digest=geometry_digest,
            dtype="float64",
            observed={
                "broader_non_equivalent_gap": broader_gap,
                "broader_minus_runtime_objective": coverage_gap,
                "same_stabilizer_group": same_group,
            },
            required={
                "broader_non_equivalent_gap": (
                    f"> {policy.minimum_objective_gap_absolute}"
                ),
                "same_stabilizer_group": True,
            },
            diagnostics={
                "runtime_selected_group": int(
                    runtime_result.selected_grouped_index.detach()
                ),
                "broader_selected_candidate": best,
                "runtime_canonical_group": _canonical_phase_group(
                    runtime_result.refined.phase, stabilizer
                ),
                "broader_canonical_group": _canonical_phase_group(
                    refined.phase[best], stabilizer
                ),
                "second_broader_canonical_group": (
                    None
                    if second is None
                    else _canonical_phase_group(
                        refined.phase[second], stabilizer
                    )
                ),
            },
        )
    return {
        "sample_id": sample_id,
        "geometry_digest": geometry_digest,
        "runtime_selected_group": int(
            runtime_result.selected_grouped_index.detach()
        ),
        "selected_canonical_group": list(
            _canonical_phase_group(refined.phase[best], stabilizer)
        ),
        "broader_selected_candidate": best,
        "broader_non_equivalent_gap": broader_gap,
        "broader_minus_runtime_objective": coverage_gap,
        "same_stabilizer_group": same_group,
    }


def _phase_candidate_coverage(
    audit_input: Any,
    geometry: Any,
    policy: EvaluationPolicy,
    coverage: _CandidateCoverage,
) -> dict[str, Any]:
    dtype = torch.float64
    runtime = audit_input.context.materialize(device="cpu", dtype=dtype)
    positions = geometry.positions_tensor()
    cell = geometry.cell_tensor()
    atomic_numbers = torch.tensor(geometry.atomic_numbers, dtype=torch.long)
    vocabulary = torch.tensor(audit_input.context.supported_species, dtype=torch.long)
    matches = atomic_numbers[:, None] == vocabulary[None, :]
    if bool(torch.any(matches.sum(-1) != 1)):
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "candidate coverage geometry contains an unsupported species",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            geometry_digest=geometry.semantic_digest,
        )
    species_indices = torch.argmax(matches.to(torch.long), dim=-1)
    alignment = torch.tensor(
        audit_input.species_alignment_weights, dtype=dtype
    )
    atomic, reference, cross = typed_reciprocal_fields(
        positions,
        torch.zeros(3, dtype=dtype),
        cell,
        runtime.topology.reference_fractional,
        alignment[species_indices],
        runtime.site_alignment_weights,
        runtime.phase_modes,
        runtime.phase_channel_weights,
    )
    validate_static_mode_amplitudes(
        reference,
        runtime.phase_channel_weights,
        policy.minimum_reference_amplitude_absolute,
    )
    validate_runtime_amplitudes(
        atomic,
        cross,
        runtime.phase_channel_weights,
        policy.minimum_atomic_amplitude_absolute,
        policy.minimum_cross_amplitude_absolute,
    )
    initial = primary_phase_initialization(
        cross[:3], runtime.phase_modes[:3]
    )
    return _compare_phase_candidate_sets(
        cross=cross,
        modes=runtime.phase_modes,
        mode_weights=runtime.phase_mode_weights,
        initial=initial,
        stabilizer=runtime.stabilizer,
        policy=policy,
        runtime_candidates=policy.materialize_candidate_offsets(
            device="cpu", dtype=dtype
        ),
        audit_candidates=coverage.audit_candidates,
        template_id=audit_input.template_id,
        sample_id=geometry.sample_id,
        geometry_digest=geometry.semantic_digest,
    )


def _probe_witness(
    audit_input: Any,
    geometry: Any,
    policy: EvaluationPolicy,
    config: Any,
) -> dict[str, Any]:
    dtype = torch.float64
    base_positions = geometry.positions_tensor()
    base_cell = geometry.cell_tensor()
    base = _evaluate(
        audit_input, geometry, policy, config, dtype, oracle=False
    )
    symmetry = {}
    translation = torch.tensor([0.137, -0.211, 0.083], dtype=dtype)
    translated = _evaluate(
        audit_input,
        geometry,
        policy,
        config,
        dtype,
        positions=base_positions + translation,
        origin=translation,
        oracle=False,
    )
    symmetry["joint_translation_absolute_error"] = float(
        torch.abs(translated.scalar - base.scalar).detach()
    )
    if translated.branch_signature != base.branch_signature:
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_BRANCH_UNSTABLE",
            "joint translation changed the selected phase/support/MIC branch",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            geometry_digest=geometry.semantic_digest,
            probe="joint_translation",
            dtype="float64",
            backend=config.transport_support.backend,
        )
    if base_positions.shape[0]:
        wrapped_positions = base_positions.clone()
        wrapped_positions[0] = wrapped_positions[0] + base_cell[0]
        wrapped = _evaluate(
            audit_input,
            geometry,
            policy,
            config,
            dtype,
            positions=wrapped_positions,
            oracle=False,
        )
        symmetry["periodic_wrap_absolute_error"] = float(
            torch.abs(wrapped.scalar - base.scalar).detach()
        )
        # A lattice wrap necessarily changes the raw integer image label.  The
        # canonical phase group, physical support, backend and fallback branch
        # must nevertheless remain unchanged.
        if (
            wrapped.branch_signature[0] != base.branch_signature[0]
            or wrapped.branch_signature[1] != base.branch_signature[1]
            or wrapped.branch_signature[3:] != base.branch_signature[3:]
        ):
            raise AutomaticEvaluationPolicyAuditError(
                "SUPPORT_BRANCH_UNSTABLE",
                "periodic lattice wrap changed the physical selected branch",
                template_id=audit_input.template_id,
                sample_id=geometry.sample_id,
                geometry_digest=geometry.semantic_digest,
                probe="periodic_lattice_wrap",
                dtype="float64",
                backend=config.transport_support.backend,
            )
        permutation = torch.arange(
            base_positions.shape[0] - 1, -1, -1, dtype=torch.long
        )
        permuted = _evaluate(
            audit_input,
            geometry,
            policy,
            config,
            dtype,
            positions=base_positions[permutation],
            atomic_numbers=torch.tensor(
                geometry.atomic_numbers, dtype=torch.long
            )[permutation],
            atom_identity=permutation,
            oracle=False,
        )
        symmetry["atom_permutation_absolute_error"] = float(
            torch.abs(permuted.scalar - base.scalar).detach()
        )
        if permuted.branch_signature != base.branch_signature:
            raise AutomaticEvaluationPolicyAuditError(
                "SUPPORT_BRANCH_UNSTABLE",
                "atom permutation changed the physical selected branch",
                template_id=audit_input.template_id,
                sample_id=geometry.sample_id,
                geometry_digest=geometry.semantic_digest,
                probe="atom_permutation",
                dtype="float64",
                backend=config.transport_support.backend,
            )
    position_errors = []
    position_relative_errors = []
    for index, direction in enumerate(_position_directions(base_positions)):
        positions = base_positions.clone().requires_grad_(True)
        live = _evaluate(
            audit_input,
            geometry,
            policy,
            config,
            dtype,
            positions=positions,
            oracle=False,
        )
        gradient = torch.autograd.grad(live.scalar, positions)[0]
        automatic = torch.sum(gradient * direction)
        plus = _evaluate(
            audit_input,
            geometry,
            policy,
            config,
            dtype,
            positions=base_positions + _POSITION_FD_STEP * direction,
            oracle=False,
        )
        minus = _evaluate(
            audit_input,
            geometry,
            policy,
            config,
            dtype,
            positions=base_positions - _POSITION_FD_STEP * direction,
            oracle=False,
        )
        if plus.branch_signature != base.branch_signature or minus.branch_signature != base.branch_signature:
            raise AutomaticEvaluationPolicyAuditError(
                "PHASE_BRANCH_UNSTABLE",
                "position finite-difference probe changed the selected phase/support branch",
                template_id=audit_input.template_id,
                sample_id=geometry.sample_id,
                geometry_digest=geometry.semantic_digest,
                probe=f"position_direction_{index}",
                dtype="float64",
                backend=config.transport_support.backend,
            )
        finite = (plus.scalar - minus.scalar) / (2.0 * _POSITION_FD_STEP)
        error = float(torch.abs(automatic - finite).detach())
        relative = error / max(float(torch.abs(automatic)), float(torch.abs(finite)), 1.0e-15)
        position_errors.append(error)
        position_relative_errors.append(relative)
    strain_errors = []
    strain_relative_errors = []
    identity = torch.eye(3, dtype=dtype)
    for index, direction in enumerate(_symmetric_directions(dtype)):
        parameter = torch.zeros((), dtype=dtype, requires_grad=True)
        deformation = identity + parameter * direction
        live = _evaluate(
            audit_input,
            geometry,
            policy,
            config,
            dtype,
            positions=base_positions @ deformation,
            cell=base_cell @ deformation,
            oracle=False,
        )
        automatic = torch.autograd.grad(live.scalar, parameter)[0]
        plus_deformation = identity + _STRAIN_FD_STEP * direction
        minus_deformation = identity - _STRAIN_FD_STEP * direction
        plus = _evaluate(
            audit_input,
            geometry,
            policy,
            config,
            dtype,
            positions=base_positions @ plus_deformation,
            cell=base_cell @ plus_deformation,
            oracle=False,
        )
        minus = _evaluate(
            audit_input,
            geometry,
            policy,
            config,
            dtype,
            positions=base_positions @ minus_deformation,
            cell=base_cell @ minus_deformation,
            oracle=False,
        )
        if plus.branch_signature != base.branch_signature or minus.branch_signature != base.branch_signature:
            raise AutomaticEvaluationPolicyAuditError(
                "SUPPORT_BRANCH_UNSTABLE",
                "strain finite-difference probe changed the selected phase/support branch",
                template_id=audit_input.template_id,
                sample_id=geometry.sample_id,
                geometry_digest=geometry.semantic_digest,
                probe=f"strain_direction_{index}",
                dtype="float64",
                backend=config.transport_support.backend,
            )
        finite = (plus.scalar - minus.scalar) / (2.0 * _STRAIN_FD_STEP)
        error = float(torch.abs(automatic - finite).detach())
        relative = error / max(float(torch.abs(automatic)), float(torch.abs(finite)), 1.0e-15)
        strain_errors.append(error)
        strain_relative_errors.append(relative)
    maximum_position = max(position_errors, default=0.0)
    maximum_strain = max(strain_errors, default=0.0)
    if maximum_position > _FIRST_DERIVATIVE_FD_TOLERANCE or maximum_strain > _FIRST_DERIVATIVE_FD_TOLERANCE:
        raise AutomaticEvaluationPolicyAuditError(
            "FIRST_DERIVATIVE_FD_FAILED",
            "selected phase/transport/probability derivative differs from central finite difference",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            geometry_digest=geometry.semantic_digest,
            dtype="float64",
            backend=config.transport_support.backend,
            observed=max(maximum_position, maximum_strain),
            required=f"<= {_FIRST_DERIVATIVE_FD_TOLERANCE}",
            diagnostics={
                "position_errors": position_errors,
                "strain_errors": strain_errors,
            },
        )
    return {
        "sample_id": geometry.sample_id,
        "geometry_digest": geometry.semantic_digest,
        "position_directions": len(position_errors),
        "strain_directions": len(strain_errors),
        "position_maximum_absolute_error": maximum_position,
        "position_maximum_relative_error": max(position_relative_errors, default=0.0),
        "strain_maximum_absolute_error": maximum_strain,
        "strain_maximum_relative_error": max(strain_relative_errors, default=0.0),
        "symmetry_probes": symmetry,
    }


def _limiting_witnesses(records: list[Mapping[str, Any]]) -> tuple[str, ...]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["K"]), []).append(record)
    selected: set[str] = set()
    extrema = (
        ("objective_gap", min),
        ("minimum_atomic_amplitude", min),
        ("minimum_reference_amplitude", min),
        ("minimum_cross_amplitude", min),
        ("hessian_minimum_curvature", min),
        ("hessian_condition", max),
        ("phase_residual", max),
        ("support_margin", min),
        ("mic_margin", min),
        ("transport_row_residual", max),
        ("transport_column_residual", max),
        ("newton_iterations", max),
        ("cg_iterations", max),
        ("line_search_reductions", max),
        ("total_transport_work", max),
        ("observed_strain", min),
        ("observed_strain", max),
    )
    for values in grouped.values():
        for field, operation in extrema:
            target = operation(float(item[field]) for item in values)
            candidates = [
                item for item in values if float(item[field]) == target
            ]
            chosen = min(candidates, key=lambda item: item["geometry_digest"])
            selected.add(str(chosen["geometry_digest"]))
    return tuple(sorted(selected))


def _audit_one(audit_input: Any, policy: EvaluationPolicy, config: Any) -> dict[str, Any]:
    coverage = _candidate_coverage(audit_input.context)
    if not torch.equal(
        policy.candidate_offsets,
        coverage.runtime_candidates,
    ):
        raise AutomaticEvaluationPolicyAuditError(
            "PHASE_CANDIDATE_COVERAGE_FAILED",
            "candidate EvaluationPolicy does not use the canonical template-derived runtime seeds",
            template_id=audit_input.template_id,
            diagnostics={
                "expected_fingerprint": coverage.metadata[
                    "runtime_candidate_fingerprint"
                ],
                "actual_fingerprint": _tensor_fingerprint(
                    policy.candidate_offsets
                ),
            },
        )
    geometries = (
        (audit_input.reference_geometry,)
        + audit_input.train_geometries
        + audit_input.validation_geometries
        + audit_input.ideal_k1_geometries
    )
    by_digest = {}
    for geometry in sorted(
        geometries,
        key=lambda value: (
            value.semantic_digest,
            value.split,
            value.sample_id,
        ),
    ):
        by_digest.setdefault(geometry.semantic_digest, geometry)
    records_by_dtype: dict[str, list[Mapping[str, Any]]] = {}
    for dtype in (torch.float64, torch.float32):
        name = str(dtype).removeprefix("torch.")
        records = []
        for geometry in sorted(
            geometries,
            key=lambda value: (
                value.semantic_digest,
                value.split,
                value.sample_id,
            ),
        ):
            try:
                outcome = _evaluate(
                    audit_input, geometry, policy, config, dtype
                )
            except AutomaticEvaluationPolicyAuditError:
                raise
            except Exception as error:
                underlying_reason = getattr(error, "reason_code", None)
                reason = (
                    "PHASE_CANDIDATE_COVERAGE_FAILED"
                    if underlying_reason == "INVALID_CANDIDATES"
                    else underlying_reason
                ) or (
                    "TRANSPORT_CONVERGENCE_FAILURE"
                    if "transport" in str(error).lower()
                    else "AUTOMATIC_EVALUATION_POLICY_AUDIT_FAILED"
                )
                composition = sorted(Counter(geometry.atomic_numbers).items())
                raise AutomaticEvaluationPolicyAuditError(
                    reason,
                    f"automatic EvaluationPolicy audit failed: {type(error).__name__}: {error}",
                    template_id=audit_input.template_id,
                    sample_id=geometry.sample_id,
                    geometry_digest=geometry.semantic_digest,
                    dtype=name,
                    backend=config.transport_support.backend,
                    observed=getattr(error, "observed", None),
                    required=getattr(error, "threshold", None),
                    diagnostics={
                        "underlying_reason_code": underlying_reason,
                        "K": audit_input.context.topology.num_sites
                        - geometry.num_atoms,
                        "composition": composition,
                    },
                    original_error=error,
                ) from error
            diagnostics = dict(outcome.diagnostics)
            diagnostics["sample_id"] = geometry.sample_id
            diagnostics["split"] = geometry.split
            records.append(diagnostics)
        records_by_dtype[name] = records
    f64_by_sample = {
        (item["split"], item["sample_id"], item["geometry_digest"]): item
        for item in records_by_dtype["float64"]
    }
    for item in records_by_dtype["float32"]:
        counterpart = f64_by_sample[
            (item["split"], item["sample_id"], item["geometry_digest"])
        ]
        for field in (
            "selected_group",
            "semantic_support_fingerprint",
            "mic_branch_fingerprint",
            "backend",
            "fallback_used",
        ):
            if item[field] != counterpart[field]:
                raise AutomaticEvaluationPolicyAuditError(
                    "PHASE_BRANCH_UNSTABLE" if field == "selected_group" else "SUPPORT_BRANCH_UNSTABLE",
                    f"float32 and float64 selected different certified {field}",
                    template_id=audit_input.template_id,
                    sample_id=item["sample_id"],
                    geometry_digest=item["geometry_digest"],
                    dtype="float32",
                    backend=config.transport_support.backend,
                    diagnostics={"field": field, "float32": item[field], "float64": counterpart[field]},
                )
    witness_digests = _limiting_witnesses(records_by_dtype["float64"])
    if len(witness_digests) > _MAX_WITNESS_GEOMETRIES:
        raise AutomaticEvaluationPolicyAuditError(
            "AUDIT_BUDGET_EXCEEDED",
            "the deterministic extrema witness union exceeds the fixed probe budget; no sampling was performed",
            template_id=audit_input.template_id,
            observed=len(witness_digests),
            required=f"<= {_MAX_WITNESS_GEOMETRIES}",
            diagnostics={"witness_digests": list(witness_digests)},
        )
    probes = []
    # Qualification is a CPU first-derivative audit even when a caller enters
    # through ``torch.no_grad`` or ``torch.inference_mode``.  Both context
    # managers restore their caller-owned state on exit.
    with torch.inference_mode(False), torch.enable_grad():
        for digest in witness_digests:
            probes.append(
                _probe_witness(
                    audit_input, by_digest[digest], policy, config
                )
            )
    coverage_digests = tuple(
        sorted(
            set(witness_digests)
            | {audit_input.reference_geometry.semantic_digest}
        )
    )
    coverage_diagnostics = [
        _phase_candidate_coverage(
            audit_input,
            by_digest[digest],
            policy,
            coverage,
        )
        for digest in coverage_digests
    ]
    splits = {
        "reference": len({audit_input.reference_geometry.semantic_digest}),
        "train": len(audit_input.train_geometries),
        "validation": len(audit_input.validation_geometries),
        "ideal_k1": len(audit_input.ideal_k1_geometries),
    }
    all_records = records_by_dtype["float64"]
    certificate = {
        "schema_version": AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION,
        "audit_convention_version": AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION,
        "status": "qualified",
        "scope": AUTOMATIC_EVALUATION_SCOPE,
        "phase_approval": "provisional",
        "differentiability_scope": "selected_branch_first_order",
        "hard_branch_frozen": True,
        "future_md_guarantee": False,
        "cuda_qualified": False,
        "create_graph_supported": False,
        "force_loss_double_backward_supported": False,
        "inference_mode_derivative_supported": False,
        "derivative_fallback_supported": False,
        "adaptive_stopping_differentiated": False,
        "hard_candidate_group_indices_differentiated": False,
        "template_id": audit_input.template_id,
        "policy_fingerprint": policy.content_fingerprint,
        "template_fingerprint": audit_input.context.fingerprint,
        "audit_input": {
            "kind": "transductive_geometry_only_preflight",
            "labels_used": False,
            "split_counts": splits,
            "geometry_digests": sorted(by_digest),
            "geometry_manifest": [
                {
                    "sample_id": geometry.sample_id,
                    "split": geometry.split,
                    "geometry_digest": geometry.semantic_digest,
                }
                for geometry in sorted(
                    geometries,
                    key=lambda value: (
                        value.semantic_digest,
                        value.split,
                        value.sample_id,
                    ),
                )
            ],
            "census_counts": {
                "total_train_census_count": len(
                    audit_input.train_geometries
                ),
                "total_validation_census_count": len(
                    audit_input.validation_geometries
                ),
                "ideal_reference_census_count": (
                    1 + len(audit_input.ideal_k1_geometries)
                ),
                "ideal_pristine_census_count": 1,
                "ideal_k1_census_count": len(
                    audit_input.ideal_k1_geometries
                ),
                "total_base_census": len(geometries),
                "unique_semantic_geometry_count": len(by_digest),
            },
        },
        "profile": _plain(automatic_evaluation_policy_profile()),
        "effective_transport": {
            "backend": config.transport_support.backend,
            "support_fingerprint": _fingerprint(
                config.transport_support.to_dict()
            ),
            "sinkhorn_warmup_iterations": config.eval_sinkhorn_warmup_iterations,
            "convergence_tolerance_float64": _FLOAT64_TRANSPORT_TOLERANCE,
            "convergence_tolerance_float32": _FLOAT32_TRANSPORT_TOLERANCE,
        },
        "candidate_group_manifest": {
            **_plain(coverage.metadata),
            "selected_groups_by_geometry": {
                item["geometry_digest"]: item["selected_group"]
                for item in all_records
            },
            "coverage_diagnostics": coverage_diagnostics,
            "limiting_coverage_gap": max(
                (
                    item["broader_minus_runtime_objective"]
                    for item in coverage_diagnostics
                ),
                default=0.0,
            ),
        },
        "dtype_diagnostics": records_by_dtype,
        "witness_selection": {
            "rule": "per-K extrema union; semantic geometry digest tie-break",
            "geometry_digests": list(witness_digests),
            "selected_witness_count": len(witness_digests),
            "executed_position_probe_count": sum(
                item["position_directions"] for item in probes
            ),
            "executed_strain_probe_count": sum(
                item["strain_directions"] for item in probes
            ),
            "max_witness_geometries": _MAX_WITNESS_GEOMETRIES,
        },
        "derivative_probes": probes,
        "fallback_count": sum(
            int(item["fallback_used"])
            for records in records_by_dtype.values()
            for item in records
        ),
        "observed_extrema": {
            "minimum_objective_gap": min(item["objective_gap"] for item in all_records),
            "minimum_atomic_amplitude": min(item["minimum_atomic_amplitude"] for item in all_records),
            "minimum_reference_amplitude": min(item["minimum_reference_amplitude"] for item in all_records),
            "minimum_cross_amplitude": min(item["minimum_cross_amplitude"] for item in all_records),
            "minimum_hessian_curvature": min(item["hessian_minimum_curvature"] for item in all_records),
            "maximum_hessian_condition": max(item["hessian_condition"] for item in all_records),
            "maximum_phase_residual": max(item["phase_residual"] for item in all_records),
            "minimum_support_margin": min(item["support_margin"] for item in all_records),
            "minimum_mic_margin": min(item["mic_margin"] for item in all_records),
            "maximum_transport_residual": max(
                max(item["transport_row_residual"], item["transport_column_residual"])
                for item in all_records
            ),
            "maximum_oracle_error": max(
                max(item["oracle_maximum_errors"].values()) for item in all_records
            ),
            "maximum_transport_work": max(
                item["total_transport_work"] for item in all_records
            ),
            "minimum_observed_strain": min(
                item["observed_strain"] for item in all_records
            ),
            "maximum_observed_strain": max(
                item["observed_strain"] for item in all_records
            ),
            "maximum_position_fd_error": max(
                (item["position_maximum_absolute_error"] for item in probes), default=0.0
            ),
            "maximum_strain_fd_error": max(
                (item["strain_maximum_absolute_error"] for item in probes), default=0.0
            ),
        },
    }
    return certificate


def qualify_automatic_evaluation_policies(
    preparation: Any, potential_config: Any
) -> Any:
    """Audit every template, then atomically return a fully qualified snapshot."""

    from .automatic_reference import (
        AutomaticReferencePreparation,
        AutomaticReferenceResult,
        _fingerprint as reference_fingerprint,
        _plain as reference_plain,
    )

    if not isinstance(preparation, AutomaticReferencePreparation):
        raise TypeError("preparation must be AutomaticReferencePreparation")
    inputs = {item.template_id: item for item in preparation._audit_inputs}
    if set(inputs) != {item.template_id for item in preparation.results}:
        raise AutomaticEvaluationPolicyAuditError(
            "AUTOMATIC_EVALUATION_POLICY_AUDIT_FAILED",
            "automatic evaluation audit inputs do not match reference templates",
        )
    audited = []
    for result in preparation.results:
        policy = result.specification.evaluation_policy
        if policy is None:
            raise AutomaticEvaluationPolicyAuditError(
                "AUTOMATIC_EVALUATION_POLICY_AUDIT_FAILED",
                "candidate EvaluationPolicy is missing before audit",
                template_id=result.template_id,
            )
        certificate = _audit_one(inputs[result.template_id], policy, potential_config)
        structural = reference_plain(result.certificate)
        structural.pop("certificate_sha256", None)
        structural["evaluation_policy"] = {
            "audit_profile": AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION,
            "status": "qualified",
            "scope": AUTOMATIC_EVALUATION_SCOPE,
            "content_fingerprint": policy.content_fingerprint,
            "convention_version": policy.convention_version,
        }
        structural["certificate_sha256"] = reference_fingerprint(structural)
        certificate.update(
            {
                "structural_certificate_sha256": structural["certificate_sha256"],
                "specification_sha256": result.specification.content_fingerprint,
                "artifact_sha256": structural["artifact_sha256"],
                "phase_specification_sha256": structural["phase_specification_sha256"],
                "radius_fingerprint": structural["radius_fingerprint"],
            }
        )
        certificate["evaluation_certificate_sha256"] = reference_fingerprint(
            certificate
        )
        audited.append(
            AutomaticReferenceResult(
                source_index=result.source_index,
                original_poscar=result.original_poscar,
                resolved_poscar=result.resolved_poscar,
                specification=result.specification,
                certificate=structural,
                evaluation_certificate=certificate,
            )
        )
    audited.sort(key=lambda value: value.template_id)
    semantic = {
        "convention_version": preparation.convention_version,
        "scope": "dataset_bounded",
        "species_vocabulary": list(preparation.species_vocabulary),
        "references": [item.semantic_dict() for item in audited],
        "train_assignments": list(preparation.train_assignments),
        "validation_assignments": list(preparation.validation_assignments),
    }
    return AutomaticReferencePreparation(
        results=tuple(audited),
        train_assignments=preparation.train_assignments,
        validation_assignments=preparation.validation_assignments,
        species_vocabulary=preparation.species_vocabulary,
        content_fingerprint=reference_fingerprint(semantic),
        convention_version=preparation.convention_version,
        _audit_inputs=preparation._audit_inputs,
    )


__all__ = [
    "AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION",
    "AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION",
    "AUTOMATIC_EVALUATION_SCOPE",
    "AutomaticEvaluationPolicyAuditError",
    "automatic_evaluation_policy_profile",
    "build_automatic_evaluation_policy",
    "qualify_automatic_evaluation_policies",
]
