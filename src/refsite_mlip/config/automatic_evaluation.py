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
from refsite_mlip.features.probability_multipoles import (
    _assemble_dense_probability_multipoles,
    _assemble_sparse_probability_multipoles,
    _site_segment_sum,
)
from refsite_mlip.features.species import (
    species_indicator,
    species_probabilities,
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
    sparse_marginal_residual_components,
    sparse_support_fingerprint,
    sparse_transport_plan,
)
from refsite_mlip.transport.dual import marginal_residuals, transport_plan
from refsite_mlip.transport.edge_list import CompactTransportEdges
from refsite_mlip.transport.marginals import split_atom_vacancy_plan
from refsite_mlip.transport.problem import OTProblem, build_ot_problem
from refsite_mlip.transport.sinkhorn import (
    masked_sinkhorn_full_update,
    sinkhorn_full_update,
    solve_sinkhorn_eval_adaptive,
    zero_duals,
)


_AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION_V1 = (
    "automatic_evaluation_policy_audit_v1"
)
AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION = (
    "automatic_evaluation_policy_audit_v2"
)
_SUPPORTED_AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSIONS = frozenset(
    {
        _AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION_V1,
        AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION,
    }
)
AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V1 = (
    "refsite_automatic_evaluation_certificate_v1"
)
AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V2 = (
    "refsite_automatic_evaluation_certificate_v2"
)
AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION = (
    AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V2
)
AUTOMATIC_EVALUATION_SEMANTIC_PROJECTION_VERSION = (
    "automatic_evaluation_certificate_semantic_projection_v1"
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
_FROZEN_FLOAT64_ORACLE_RESIDUAL_TOLERANCE = 1.0e-12
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


_QUALIFICATION_FIELDS = (
    "status",
    "scope",
    "phase_approval",
    "differentiability_scope",
    "hard_branch_frozen",
    "future_md_guarantee",
    "cuda_qualified",
    "create_graph_supported",
    "force_loss_double_backward_supported",
    "inference_mode_derivative_supported",
    "derivative_fallback_supported",
    "adaptive_stopping_differentiated",
    "hard_candidate_group_indices_differentiated",
)
_BINDING_FIELDS = (
    "template_id",
    "policy_fingerprint",
    "template_fingerprint",
    "structural_certificate_sha256",
    "specification_sha256",
    "artifact_sha256",
    "phase_specification_sha256",
    "radius_fingerprint",
)
_CANDIDATE_DEFINITION_FIELDS = (
    "convention_version",
    "primary_mode_matrix",
    "primary_coordinate_mode_matrix",
    "primary_coordinate_bandlimit",
    "primary_coordinate_l1_bandlimit",
    "primary_coordinate_axis_bandlimits",
    "maximum_supported_primary_coordinate_bandlimit",
    "alias_kernel_order",
    "alias_kernel_fingerprint",
    "typed_stabilizer_fingerprint",
    "runtime_grid_resolution",
    "broader_audit_grid_resolution",
    "runtime_raw_candidate_count",
    "broader_audit_raw_candidate_count",
    "runtime_candidate_count",
    "broader_audit_candidate_count",
    "runtime_candidate_fingerprint",
    "broader_audit_candidate_fingerprint",
    "stabilizer_reduction_rule",
    "selected_groups_by_geometry",
)
_NORMATIVE_COMPARATORS_V1 = {
    "objective_gap_passed": "> minimum_objective_gap_absolute",
    "atomic_amplitude_passed": "> minimum_atomic_amplitude_absolute",
    "reference_amplitude_passed": "> minimum_reference_amplitude_absolute",
    "cross_amplitude_passed": "> minimum_cross_amplitude_absolute",
    "hessian_curvature_passed": "> minimum_curvature",
    "hessian_condition_passed": "< maximum_condition",
    "phase_residual_passed": "< maximum_gradient_norm",
    "transport_row_residual_passed": "<= dtype transport_tolerance",
    "transport_column_residual_passed": "<= dtype transport_tolerance",
    "vacancy_mass_residual_passed": "<= dtype transport_tolerance",
    "transport_oracle_passed": "<= dtype oracle_tolerance",
    "support_margin_passed": "> 0",
    "mic_margin_passed": "> 0 for dense; not normative for edge_list",
    "fallback_free": "is true",
    "runtime_sparse_non_densified": "is true for edge_list",
    "same_stabilizer_group": "is true",
    "alternate_basin_gap_passed": (
        "is absent or > minimum_objective_gap_absolute"
    ),
    "position_absolute_fd_passed": "<= first_derivative_fd_tolerance",
    "strain_absolute_fd_passed": "<= first_derivative_fd_tolerance",
    "all_probe_branches_agree": "is true",
    "cross_dtype_branch_agreement": "is true",
    "fallback_count_matches": "is true",
}
_NORMATIVE_COMPARATORS_V2 = {
    **_NORMATIVE_COMPARATORS_V1,
    "transport_oracle_residual_passed": (
        "<= frozen CPU float64 oracle residual tolerance"
    ),
}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticEvaluationPolicyAuditError(
            "INVALID_EVALUATION_CERTIFICATE",
            f"{name} must be a mapping",
        )
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AutomaticEvaluationPolicyAuditError(
            "INVALID_EVALUATION_CERTIFICATE",
            f"{name} must be a list",
        )
    return value


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AutomaticEvaluationPolicyAuditError(
            "INVALID_EVALUATION_CERTIFICATE",
            f"{name} must be a finite number",
        )
    result = float(value)
    if not math.isfinite(result):
        raise AutomaticEvaluationPolicyAuditError(
            "NONFINITE_AUDIT_RESULT",
            f"{name} is nonfinite",
        )
    return result


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "split": record["split"],
        "geometry_digest": record["geometry_digest"],
        "K": record["K"],
        "composition": record["composition"],
        "dtype": record["dtype"],
        "backend": record["backend"],
    }


def _normative_outcomes(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute pass/fail claims from unrounded full-precision telemetry.

    Relative derivative errors and other continuous diagnostics are deliberately
    informational.  Qualification continues to use the existing absolute
    comparators and production thresholds before this projection is created.
    """

    audit_version = certificate.get("audit_convention_version")
    if audit_version not in _SUPPORTED_AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSIONS:
        raise AutomaticEvaluationPolicyAuditError(
            "UNSUPPORTED_EVALUATION_CERTIFICATE_AUDIT",
            "evaluation certificate audit convention is unsupported",
        )
    profile = _require_mapping(certificate.get("profile"), "profile")
    expected_profile = _plain(
        automatic_evaluation_policy_profile(audit_version=audit_version)
    )
    if _plain(profile) != expected_profile:
        raise AutomaticEvaluationPolicyAuditError(
            "EVALUATION_CERTIFICATE_PROFILE_MISMATCH",
            "certificate profile differs from the versioned audit profile",
        )
    diagnostics = _require_mapping(
        certificate.get("dtype_diagnostics"), "dtype_diagnostics"
    )
    if set(diagnostics) != {"float32", "float64"}:
        raise AutomaticEvaluationPolicyAuditError(
            "INVALID_EVALUATION_CERTIFICATE",
            "certificate must contain float32 and float64 diagnostics",
        )
    record_checks: dict[str, list[dict[str, Any]]] = {}
    by_dtype: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    fallback_count = 0
    for dtype_name in ("float32", "float64"):
        records = _require_list(diagnostics[dtype_name], dtype_name)
        if not records:
            raise AutomaticEvaluationPolicyAuditError(
                "INVALID_EVALUATION_CERTIFICATE",
                f"{dtype_name} diagnostics must not be empty",
            )
        tolerance = _numeric(
            profile[f"transport_tolerance_{dtype_name}"],
            f"transport_tolerance_{dtype_name}",
        )
        oracle_tolerance = _numeric(
            profile[f"oracle_tolerance_{dtype_name}"],
            f"oracle_tolerance_{dtype_name}",
        )
        values: list[dict[str, Any]] = []
        identities: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for index, raw in enumerate(records):
            record = _require_mapping(raw, f"{dtype_name}[{index}]")
            if record.get("dtype") != dtype_name:
                raise AutomaticEvaluationPolicyAuditError(
                    "INVALID_EVALUATION_CERTIFICATE",
                    "diagnostic dtype key and record disagree",
                )
            identity = _record_identity(record)
            key = (
                identity["split"],
                identity["sample_id"],
                identity["geometry_digest"],
            )
            if key in identities:
                raise AutomaticEvaluationPolicyAuditError(
                    "INVALID_EVALUATION_CERTIFICATE",
                    "duplicate dtype diagnostic identity",
                )
            identities[key] = record
            oracle_field = (
                "oracle_maximum_errors"
                if audit_version
                == _AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION_V1
                else "frozen_float64_oracle_maximum_errors"
            )
            oracle = _require_mapping(
                record.get(oracle_field),
                oracle_field,
            )
            fallback = record.get("fallback_used") is True
            fallback_count += int(fallback)
            backend = record.get("backend")
            checks = {
                "objective_gap_passed": _numeric(
                    record.get("objective_gap"), "objective_gap"
                )
                > _numeric(
                    profile["minimum_objective_gap_absolute"],
                    "minimum_objective_gap_absolute",
                ),
                "atomic_amplitude_passed": _numeric(
                    record.get("minimum_atomic_amplitude"),
                    "minimum_atomic_amplitude",
                )
                > _numeric(
                    profile["minimum_atomic_amplitude_absolute"],
                    "minimum_atomic_amplitude_absolute",
                ),
                "reference_amplitude_passed": _numeric(
                    record.get("minimum_reference_amplitude"),
                    "minimum_reference_amplitude",
                )
                > _numeric(
                    profile["minimum_reference_amplitude_absolute"],
                    "minimum_reference_amplitude_absolute",
                ),
                "cross_amplitude_passed": _numeric(
                    record.get("minimum_cross_amplitude"),
                    "minimum_cross_amplitude",
                )
                > _numeric(
                    profile["minimum_cross_amplitude_absolute"],
                    "minimum_cross_amplitude_absolute",
                ),
                "hessian_curvature_passed": _numeric(
                    record.get("hessian_minimum_curvature"),
                    "hessian_minimum_curvature",
                )
                > _numeric(profile["minimum_curvature"], "minimum_curvature"),
                "hessian_condition_passed": _numeric(
                    record.get("hessian_condition"), "hessian_condition"
                )
                < _numeric(profile["maximum_condition"], "maximum_condition"),
                "phase_residual_passed": _numeric(
                    record.get("phase_residual"), "phase_residual"
                )
                < _numeric(
                    profile["maximum_gradient_norm"],
                    "maximum_gradient_norm",
                ),
                "transport_row_residual_passed": _numeric(
                    record.get("transport_row_residual"),
                    "transport_row_residual",
                )
                <= tolerance,
                "transport_column_residual_passed": _numeric(
                    record.get("transport_column_residual"),
                    "transport_column_residual",
                )
                <= tolerance,
                "vacancy_mass_residual_passed": _numeric(
                    record.get("q_mass_error"), "q_mass_error"
                )
                <= tolerance,
                "transport_oracle_passed": max(
                    _numeric(value, f"oracle_maximum_errors.{name}")
                    for name, value in oracle.items()
                )
                <= oracle_tolerance,
                "support_margin_passed": _numeric(
                    record.get("support_margin"), "support_margin"
                )
                > 0.0,
                "mic_margin_passed": (
                    backend == "edge_list"
                    or _numeric(record.get("mic_margin"), "mic_margin") > 0.0
                ),
                "fallback_free": not fallback,
                "runtime_sparse_non_densified": (
                    backend != "edge_list"
                    or record.get("dense_plan_materialized") is False
                ),
            }
            if audit_version == AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION:
                oracle_residuals = _require_mapping(
                    record.get("frozen_float64_oracle_residuals"),
                    "frozen_float64_oracle_residuals",
                )
                checks["transport_oracle_residual_passed"] = max(
                    _numeric(
                        value,
                        f"frozen_float64_oracle_residuals.{name}",
                    )
                    for name, value in oracle_residuals.items()
                ) <= _numeric(
                    profile["oracle_residual_tolerance"],
                    "oracle_residual_tolerance",
                )
            values.append({**identity, "checks": checks})
        by_dtype[dtype_name] = identities
        record_checks[dtype_name] = values

    shared = set(by_dtype["float64"])
    cross_dtype = shared == set(by_dtype["float32"])
    if cross_dtype:
        for key in sorted(shared):
            left = by_dtype["float64"][key]
            right = by_dtype["float32"][key]
            for field in (
                "selected_group",
                "semantic_support_fingerprint",
                "mic_branch_fingerprint",
                "backend",
                "fallback_used",
            ):
                cross_dtype = cross_dtype and left.get(field) == right.get(field)

    candidate = _require_mapping(
        certificate.get("candidate_group_manifest"),
        "candidate_group_manifest",
    )
    coverage_checks = []
    for index, raw in enumerate(
        _require_list(candidate.get("coverage_diagnostics"), "coverage_diagnostics")
    ):
        item = _require_mapping(raw, f"coverage_diagnostics[{index}]")
        gap = item.get("broader_non_equivalent_gap")
        coverage_checks.append(
            {
                "sample_id": item["sample_id"],
                "geometry_digest": item["geometry_digest"],
                "runtime_selected_group": item["runtime_selected_group"],
                "checks": {
                    "same_stabilizer_group": item.get("same_stabilizer_group")
                    is True,
                    "alternate_basin_gap_passed": gap is None
                    or _numeric(gap, "broader_non_equivalent_gap")
                    > _numeric(
                        profile["minimum_objective_gap_absolute"],
                        "minimum_objective_gap_absolute",
                    ),
                },
            }
        )

    probe_checks = []
    for index, raw in enumerate(
        _require_list(certificate.get("derivative_probes"), "derivative_probes")
    ):
        probe = _require_mapping(raw, f"derivative_probes[{index}]")
        branch = _require_mapping(
            probe.get("branch_agreement"), "branch_agreement"
        )
        probe_checks.append(
            {
                "sample_id": probe["sample_id"],
                "geometry_digest": probe["geometry_digest"],
                "position_directions": probe["position_directions"],
                "strain_directions": probe["strain_directions"],
                "branch_agreement": _plain(branch),
                "checks": {
                    "position_absolute_fd_passed": _numeric(
                        probe.get("position_maximum_absolute_error"),
                        "position_maximum_absolute_error",
                    )
                    <= _numeric(
                        profile["first_derivative_fd_tolerance"],
                        "first_derivative_fd_tolerance",
                    ),
                    "strain_absolute_fd_passed": _numeric(
                        probe.get("strain_maximum_absolute_error"),
                        "strain_maximum_absolute_error",
                    )
                    <= _numeric(
                        profile["first_derivative_fd_tolerance"],
                        "first_derivative_fd_tolerance",
                    ),
                    "all_probe_branches_agree": bool(branch)
                    and all(value is True for value in branch.values()),
                },
            }
        )
    return {
        "record_checks": record_checks,
        "cross_dtype_branch_agreement": cross_dtype,
        "candidate_coverage_checks": coverage_checks,
        "probe_checks": probe_checks,
        "fallback_count_matches": certificate.get("fallback_count")
        == fallback_count,
        "fallback_free": fallback_count == 0,
    }


def _assert_qualified_outcomes(outcomes: Mapping[str, Any]) -> None:
    booleans = [
        outcomes.get("cross_dtype_branch_agreement"),
        outcomes.get("fallback_count_matches"),
        outcomes.get("fallback_free"),
    ]
    for records in _require_mapping(
        outcomes.get("record_checks"), "record_checks"
    ).values():
        for record in _require_list(records, "record_checks records"):
            booleans.extend(
                _require_mapping(record, "record check")["checks"].values()
            )
    for key in ("candidate_coverage_checks", "probe_checks"):
        for item in _require_list(outcomes.get(key), key):
            current = _require_mapping(item, key)
            booleans.extend(
                _require_mapping(current.get("checks"), f"{key}.checks").values()
            )
            if key == "probe_checks":
                booleans.extend(
                    _require_mapping(
                        current.get("branch_agreement"),
                        "probe branch agreement",
                    ).values()
                )
    if not booleans or any(value is not True for value in booleans):
        raise AutomaticEvaluationPolicyAuditError(
            "EVALUATION_CERTIFICATE_QUALIFICATION_FAILED",
            "full-precision certificate telemetry does not satisfy its semantic qualification claim",
            diagnostics=_plain(outcomes),
        )


def _semantic_projection(
    certificate: Mapping[str, Any], outcomes: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = _require_mapping(
        certificate.get("candidate_group_manifest"),
        "candidate_group_manifest",
    )
    diagnostics = _require_mapping(
        certificate.get("dtype_diagnostics"), "dtype_diagnostics"
    )
    dtype_branches = {}
    for dtype_name in sorted(diagnostics):
        dtype_branches[dtype_name] = [
            {
                **_record_identity(record),
                "selected_group": record["selected_group"],
                "semantic_support_fingerprint": record[
                    "semantic_support_fingerprint"
                ],
                "mic_branch_fingerprint": record["mic_branch_fingerprint"],
                "fallback_used": record["fallback_used"],
                "dense_plan_materialized": record[
                    "dense_plan_materialized"
                ],
            }
            for record in diagnostics[dtype_name]
        ]
    comparators = (
        _NORMATIVE_COMPARATORS_V1
        if certificate["audit_convention_version"]
        == _AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION_V1
        else _NORMATIVE_COMPARATORS_V2
    )
    return {
        "schema_version": certificate["schema_version"],
        "audit_convention_version": certificate["audit_convention_version"],
        "semantic_projection_version": certificate[
            "semantic_projection_version"
        ],
        "qualification": {
            key: certificate[key] for key in _QUALIFICATION_FIELDS
        },
        "bindings": {key: certificate[key] for key in _BINDING_FIELDS},
        "audit_domain": certificate["audit_input"],
        "normative_profile": certificate["profile"],
        "normative_comparators": _plain(comparators),
        "effective_transport": certificate["effective_transport"],
        "candidate_definition": {
            key: candidate[key] for key in _CANDIDATE_DEFINITION_FIELDS
        },
        "dtype_branch_manifest": dtype_branches,
        "witness_and_probe_definition": certificate["witness_selection"],
        "normative_outcomes": _plain(outcomes),
    }


def _finalize_evaluation_certificate(
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    result = _plain(certificate)
    result["semantic_projection_version"] = (
        AUTOMATIC_EVALUATION_SEMANTIC_PROJECTION_VERSION
    )
    outcomes = _normative_outcomes(result)
    _assert_qualified_outcomes(outcomes)
    result["normative_outcomes"] = outcomes
    result["evaluation_semantic_fingerprint_sha256"] = _fingerprint(
        _semantic_projection(result, outcomes)
    )
    result["evaluation_certificate_sha256"] = _fingerprint(result)
    return result


def validate_automatic_evaluation_certificate(
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate v1 integrity or v2 integrity plus full semantic qualification."""

    result = _plain(_require_mapping(certificate, "certificate"))
    declared = result.pop("evaluation_certificate_sha256", None)
    if declared != _fingerprint(result):
        raise AutomaticEvaluationPolicyAuditError(
            "EVALUATION_CERTIFICATE_INTEGRITY_MISMATCH",
            "evaluation certificate content SHA-256 differs from its payload",
        )
    result["evaluation_certificate_sha256"] = declared
    schema = result.get("schema_version")
    if schema == AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V1:
        return result
    if schema != AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V2:
        raise AutomaticEvaluationPolicyAuditError(
            "UNSUPPORTED_EVALUATION_CERTIFICATE_SCHEMA",
            "evaluation certificate schema is unsupported",
        )
    if result.get("semantic_projection_version") != (
        AUTOMATIC_EVALUATION_SEMANTIC_PROJECTION_VERSION
    ):
        raise AutomaticEvaluationPolicyAuditError(
            "UNSUPPORTED_EVALUATION_CERTIFICATE_SEMANTIC_PROJECTION",
            "evaluation certificate semantic projection is unsupported",
        )
    if result.get("audit_convention_version") not in (
        _SUPPORTED_AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSIONS
    ):
        raise AutomaticEvaluationPolicyAuditError(
            "UNSUPPORTED_EVALUATION_CERTIFICATE_AUDIT",
            "evaluation certificate audit convention is unsupported",
        )
    outcomes = _normative_outcomes(result)
    _assert_qualified_outcomes(outcomes)
    if result.get("normative_outcomes") != outcomes:
        raise AutomaticEvaluationPolicyAuditError(
            "EVALUATION_CERTIFICATE_OUTCOME_MISMATCH",
            "stored semantic outcomes differ from full-precision telemetry",
        )
    actual_semantic = _fingerprint(_semantic_projection(result, outcomes))
    if result.get("evaluation_semantic_fingerprint_sha256") != actual_semantic:
        raise AutomaticEvaluationPolicyAuditError(
            "EVALUATION_CERTIFICATE_SEMANTIC_FINGERPRINT_MISMATCH",
            "evaluation certificate semantic fingerprint differs from its allowlist projection",
        )
    if result.get("status") != "qualified" or result.get("scope") != (
        AUTOMATIC_EVALUATION_SCOPE
    ):
        raise AutomaticEvaluationPolicyAuditError(
            "EVALUATION_CERTIFICATE_QUALIFICATION_FAILED",
            "evaluation certificate does not make the supported qualified claim",
        )
    return result


def automatic_evaluation_certificate_semantic_identity(
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the v2 semantic identity or the exact legacy v1 payload."""

    validated = validate_automatic_evaluation_certificate(certificate)
    if validated["schema_version"] == (
        AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V1
    ):
        return validated
    return {
        "schema_version": validated["schema_version"],
        "audit_convention_version": validated["audit_convention_version"],
        "semantic_projection_version": validated[
            "semantic_projection_version"
        ],
        "evaluation_semantic_fingerprint_sha256": validated[
            "evaluation_semantic_fingerprint_sha256"
        ],
    }


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


def automatic_evaluation_policy_profile(
    *, audit_version: str | None = None
) -> Mapping[str, Any]:
    """Return one immutable, versioned automatic-qualification profile.

    The historical v1 profile remains available solely to validate already
    persisted v2 certificates.  New qualification always writes v2.
    """

    version = (
        AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION
        if audit_version is None
        else audit_version
    )
    if version not in _SUPPORTED_AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSIONS:
        raise ValueError("unsupported automatic evaluation audit version")
    profile = {
        "convention_version": version,
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
    if version == _AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION_V1:
        profile["oracle_residual_target_float32"] = (
            _FLOAT32_ORACLE_RESIDUAL_TARGET
        )
    else:
        profile.update(
            {
                "oracle_problem_float32": (
                    "CPU float64 promotion of the frozen float32 OT problem"
                ),
                "oracle_problem_float64": (
                    "CPU float64 frozen runtime OT problem"
                ),
                "oracle_residual_tolerance": (
                    _FROZEN_FLOAT64_ORACLE_RESIDUAL_TOLERANCE
                ),
                "fixed_sinkhorn_float32_residual_normative": False,
                "float32_multipole_comparison": (
                    "production feature arithmetic after the 1e-6 NK residual gate"
                ),
            }
        )
    return MappingProxyType(profile)


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


def _audit_dense_probability_multipoles(
    P: torch.Tensor,
    q: torch.Tensor,
    atomic_numbers: torch.Tensor,
    displacements: torch.Tensor,
    config: Any,
    site_types: torch.Tensor,
):
    """Use exact production feature arithmetic after the audit's NK gate.

    The public feature builder retains its own validation contract.  Audit v2
    has already required the actual float32 NK row, column, and vacancy
    residuals to be at most 1e-6, so it must not substitute the unrelated
    same-dtype global species reduction as a second transport convergence gate.
    """

    finite = (P, q, displacements)
    if any(not bool(torch.all(torch.isfinite(value)).detach()) for value in finite):
        raise AutomaticEvaluationPolicyAuditError(
            "NONFINITE_AUDIT_RESULT",
            "production NK probability feature input is nonfinite",
        )
    tolerance = (
        _FLOAT32_TRANSPORT_TOLERANCE
        if P.dtype == torch.float32
        else _FLOAT64_TRANSPORT_TOLERANCE
    )
    if bool(torch.any(P < 0.0).detach()) or bool(
        torch.any((q < -tolerance) | (q > 1.0 + tolerance)).detach()
    ):
        raise AutomaticEvaluationPolicyAuditError(
            "TRANSPORT_CONVERGENCE_FAILURE",
            "production NK P/q lies outside the physical probability range",
        )
    probabilities, indicator = species_probabilities(
        P, atomic_numbers, config.species_vocabulary
    )
    return _assemble_dense_probability_multipoles(
        P,
        q,
        displacements,
        config,
        site_types,
        probabilities,
        indicator,
    )


def _audit_sparse_probability_multipoles(
    edge_plan: torch.Tensor,
    q: torch.Tensor,
    edges: CompactTransportEdges,
    atomic_numbers: torch.Tensor,
    config: Any,
    site_types: torch.Tensor,
):
    """Use exact sparse feature arithmetic after the audit's NK gate."""

    finite = (edge_plan, q, edges.displacements)
    if any(not bool(torch.all(torch.isfinite(value)).detach()) for value in finite):
        raise AutomaticEvaluationPolicyAuditError(
            "NONFINITE_AUDIT_RESULT",
            "production sparse NK probability feature input is nonfinite",
        )
    tolerance = (
        _FLOAT32_TRANSPORT_TOLERANCE
        if edge_plan.dtype == torch.float32
        else _FLOAT64_TRANSPORT_TOLERANCE
    )
    if bool(torch.any(edge_plan < 0.0).detach()) or bool(
        torch.any((q < -tolerance) | (q > 1.0 + tolerance)).detach()
    ):
        raise AutomaticEvaluationPolicyAuditError(
            "TRANSPORT_CONVERGENCE_FAILURE",
            "production sparse NK edge-plan/q lies outside the physical probability range",
        )
    indicator = species_indicator(
        atomic_numbers, config.species_vocabulary, dtype=edge_plan.dtype
    )
    edge_indicator = indicator[edges.atom_index]
    probabilities = _site_segment_sum(
        edge_plan[:, None] * edge_indicator,
        edges.site_index,
        edges.num_sites,
    )
    return _assemble_sparse_probability_multipoles(
        edge_plan,
        q,
        edges,
        config,
        site_types,
        indicator,
        edge_indicator,
        probabilities,
        {
            "simplex": tolerance,
            "species_count": tolerance,
            "vacancy_mass": tolerance,
        },
    )


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


def _promote_frozen_dense_problem(problem: OTProblem) -> OTProblem:
    """Promote one already-realized OT problem without reselecting support."""

    def floating(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu", dtype=torch.float64).clone()

    return OTProblem(
        atom_cost=floating(problem.atom_cost),
        cost=floating(problem.cost),
        row_marginal=floating(problem.row_marginal),
        column_marginal=floating(problem.column_marginal),
        epsilon=floating(problem.epsilon),
        num_sites=problem.num_sites,
        num_atoms=problem.num_atoms,
        num_vacancies=problem.num_vacancies,
        log_kernel=(
            None if problem.log_kernel is None else floating(problem.log_kernel)
        ),
        support_diagnostics=problem.support_diagnostics,
    )


def _promote_frozen_sparse_edges(
    edges: CompactTransportEdges,
) -> CompactTransportEdges:
    """Promote edge arithmetic while retaining exact indices, mask, and MIC."""

    def index(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu").clone()

    def floating(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu", dtype=torch.float64).clone()

    return replace(
        edges,
        site_index=index(edges.site_index),
        atom_index=index(edges.atom_index),
        displacements=floating(edges.displacements),
        distances=floating(edges.distances),
        switch=floating(edges.switch),
        log_kernel=floating(edges.log_kernel),
        active=index(edges.active),
        atom_major_permutation=index(edges.atom_major_permutation),
        site_ptr=index(edges.site_ptr),
        atom_ptr=index(edges.atom_ptr),
        epsilon=floating(edges.epsilon),
        periodic_shift=index(edges.periodic_shift),
    )


def _dense_residual_diagnostics(
    problem: OTProblem, gamma: torch.Tensor, q: torch.Tensor
) -> dict[str, float]:
    row, column = marginal_residuals(problem, gamma)
    q_mass = torch.abs(
        q.sum() - q.new_tensor(float(problem.num_vacancies))
    )
    return {
        "row": float(row.abs().max().detach().cpu()),
        "column": float(column.abs().max().detach().cpu()),
        "q_mass": float(q_mass.detach().cpu()),
    }


def _sparse_residual_diagnostics(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
) -> dict[str, float]:
    row, atomic_column, vacancy = sparse_marginal_residual_components(
        edges, edge_plan, q
    )
    return {
        "row": float(row.abs().max().detach().cpu()),
        "column": float(atomic_column.abs().max().detach().cpu()),
        "q_mass": float(vacancy.abs().detach().cpu()),
    }


def _frozen_float64_dense_oracle(problem: OTProblem):
    promoted = _promote_frozen_dense_problem(problem)
    result = solve_sinkhorn_eval_adaptive(
        promoted,
        maximum_iterations=_ORACLE_SINKHORN_ITERATIONS,
        tolerance=_FROZEN_FLOAT64_ORACLE_RESIDUAL_TOLERANCE,
    )
    residuals = _dense_residual_diagnostics(promoted, result.gamma, result.q)
    return promoted, result, residuals


def _frozen_float64_sparse_oracle(edges: CompactTransportEdges):
    promoted = _promote_frozen_sparse_edges(edges)
    duals = sparse_fixed_sinkhorn_updates(
        promoted, _ORACLE_SINKHORN_ITERATIONS
    )
    edge_plan, q = sparse_transport_plan(promoted, duals.f, duals.g)
    residuals = _sparse_residual_diagnostics(promoted, edge_plan, q)
    return promoted, edge_plan, q, residuals


def _float32_dense_fixed_sinkhorn_diagnostic(
    problem: OTProblem,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Retain the historical float32 oracle and terminal telemetry, non-normatively.

    The first plan meeting the former 2-epsilon target reproduces the v1 audit
    comparison when such an iterate exists.  Qualification never consumes this
    result; the full 1024-update terminal residual is raw diagnostic telemetry.
    """

    duals = zero_duals(problem)
    f, g = duals.f, duals.g
    first_plan = None
    first_q = None
    first_iteration = None
    final_gamma = None
    final_q = None
    with torch.autocast(device_type=problem.cost.device.type, enabled=False):
        for index in range(_ORACLE_SINKHORN_ITERATIONS):
            if problem.log_kernel is None:
                f, g = sinkhorn_full_update(problem, f, g)
            else:
                f, g = masked_sinkhorn_full_update(problem, f, g)
            gamma = transport_plan(problem, f, g)
            _, q = split_atom_vacancy_plan(problem, gamma)
            residuals = _dense_residual_diagnostics(problem, gamma, q)
            if (
                first_plan is None
                and max(residuals.values())
                <= _FLOAT32_ORACLE_RESIDUAL_TARGET
            ):
                first_plan = gamma[:, : problem.num_atoms]
                first_q = q
                first_iteration = index + 1
            final_gamma = gamma
            final_q = q
            if first_plan is not None:
                break
    assert final_gamma is not None and final_q is not None
    final_residuals = _dense_residual_diagnostics(
        problem, final_gamma, final_q
    )
    legacy_plan = (
        final_gamma[:, : problem.num_atoms]
        if first_plan is None
        else first_plan
    )
    legacy_q = final_q if first_q is None else first_q
    return legacy_plan, legacy_q, {
        "iterations": (
            _ORACLE_SINKHORN_ITERATIONS
            if first_iteration is None
            else first_iteration
        ),
        "maximum_iterations": _ORACLE_SINKHORN_ITERATIONS,
        "former_target": _FLOAT32_ORACLE_RESIDUAL_TARGET,
        "first_passing_iteration": first_iteration,
        "terminal_residuals": final_residuals,
        "normative": False,
    }


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
        if dtype == torch.float32:
            early_row, early_column, early_vacancy = (
                sparse_marginal_residual_components(
                    ot.edges, ot.edge_plan, ot.q
                )
            )
            early_residual = max(
                float(early_row.abs().max()),
                float(early_column.abs().max()),
                float(early_vacancy.abs()),
            )
            if early_residual > eval_config.convergence_tolerance:
                raise AutomaticEvaluationPolicyAuditError(
                    "TRANSPORT_CONVERGENCE_FAILURE",
                    "adaptive sparse transport residual exceeds the existing dtype tolerance",
                    template_id=audit_input.template_id,
                    sample_id=geometry.sample_id,
                    dtype="float32",
                    backend="edge_list",
                    observed=early_residual,
                    required=f"<= {eval_config.convergence_tolerance}",
                )
            feature = _audit_sparse_probability_multipoles(
                ot.edge_plan,
                ot.q,
                ot.edges,
                atomic_numbers,
                config.feature,
                runtime.topology.site_types,
            )
        else:
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
            if dtype == torch.float32:
                legacy_duals = sparse_fixed_sinkhorn_updates(
                    edges, _ORACLE_SINKHORN_ITERATIONS
                )
                legacy_plan, legacy_q = sparse_transport_plan(
                    edges, legacy_duals.f, legacy_duals.g
                )
                legacy_errors = {
                    "plan": _tensor_maximum_error(plan, legacy_plan),
                    "q": _tensor_maximum_error(ot.q, legacy_q),
                }
                legacy_feature_error = None
                try:
                    legacy_feature = build_sparse_probability_multipoles(
                        legacy_plan,
                        legacy_q,
                        edges,
                        atomic_numbers,
                        config.feature,
                        runtime.topology.site_types,
                    )
                    legacy_errors["multipoles"] = _tensor_maximum_error(
                        feature.equivariant_features,
                        legacy_feature.equivariant_features,
                    )
                except ValueError as error:
                    # This same-dtype fixed solve is telemetry in audit v2.
                    # Never let its representational floor veto a qualified
                    # production NK result and frozen float64 oracle.
                    legacy_errors["multipoles"] = None
                    legacy_feature_error = f"{type(error).__name__}: {error}"
                fixed_float32_diagnostic = {
                    "iterations": _ORACLE_SINKHORN_ITERATIONS,
                    "former_target": _FLOAT32_ORACLE_RESIDUAL_TARGET,
                    "first_passing_iteration": None,
                    "terminal_residuals": _sparse_residual_diagnostics(
                        edges, legacy_plan, legacy_q
                    ),
                    "maximum_errors": legacy_errors,
                    "feature_validation_error": legacy_feature_error,
                    "normative": False,
                }
            else:
                legacy_errors = None
                fixed_float32_diagnostic = None
            try:
                (
                    oracle_edges,
                    oracle_plan,
                    oracle_q,
                    oracle_residuals,
                ) = _frozen_float64_sparse_oracle(edges)
            except Exception as error:
                raise AutomaticEvaluationPolicyAuditError(
                    "TRANSPORT_ORACLE_CONVERGENCE_FAILURE",
                    "frozen-support CPU float64 sparse oracle did not converge",
                    template_id=audit_input.template_id,
                    sample_id=geometry.sample_id,
                    geometry_digest=geometry.semantic_digest,
                    dtype=str(dtype).removeprefix("torch."),
                    backend="edge_list",
                    original_error=error,
                ) from error
            oracle_feature = build_sparse_probability_multipoles(
                oracle_plan,
                oracle_q,
                oracle_edges,
                atomic_numbers.detach().cpu(),
                config.feature,
                runtime.topology.site_types.detach().cpu(),
            )
            frozen_oracle_errors = {
                "plan": _tensor_maximum_error(plan, oracle_plan),
                "q": _tensor_maximum_error(ot.q, oracle_q),
                "multipoles": _tensor_maximum_error(
                    feature.equivariant_features,
                    oracle_feature.equivariant_features,
                ),
            }
            if legacy_errors is None:
                legacy_errors = frozen_oracle_errors
        else:
            legacy_errors = {"plan": 0.0, "q": 0.0, "multipoles": 0.0}
            frozen_oracle_errors = legacy_errors
            oracle_residuals = {"row": 0.0, "column": 0.0, "q_mass": 0.0}
            fixed_float32_diagnostic = None
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
        if dtype == torch.float32:
            early_q_mass_error = torch.abs(
                ot.q.sum()
                - ot.q.new_tensor(
                    runtime.topology.num_sites - positions.shape[0]
                )
            )
            if (
                float(torch.maximum(ot.row_residual, ot.column_residual))
                > eval_config.convergence_tolerance
                or float(early_q_mass_error)
                > eval_config.convergence_tolerance
            ):
                raise AutomaticEvaluationPolicyAuditError(
                    "TRANSPORT_CONVERGENCE_FAILURE",
                    "adaptive transport residual exceeds the existing dtype tolerance",
                    template_id=audit_input.template_id,
                    sample_id=geometry.sample_id,
                    dtype="float32",
                    backend="dense",
                    observed=max(
                        float(ot.row_residual),
                        float(ot.column_residual),
                        float(early_q_mass_error),
                    ),
                    required=f"<= {eval_config.convergence_tolerance}",
                )
            feature = _audit_dense_probability_multipoles(
                ot.P,
                ot.q,
                atomic_numbers,
                displacements,
                config.feature,
                runtime.topology.site_types,
            )
        else:
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
            problem = build_ot_problem(
                cost,
                config.epsilon_ot,
                support_config=support_config,
                atom_distances=distances,
                template_id=audit_input.template_id,
                sample_id=geometry.sample_id,
            )
            if dtype == torch.float32:
                (
                    legacy_plan,
                    legacy_q,
                    fixed_float32_diagnostic,
                ) = _float32_dense_fixed_sinkhorn_diagnostic(problem)
                legacy_errors = {
                    "plan": _tensor_maximum_error(plan, legacy_plan),
                    "q": _tensor_maximum_error(ot.q, legacy_q),
                }
                legacy_feature_error = None
                try:
                    legacy_feature = build_probability_multipoles(
                        legacy_plan,
                        legacy_q,
                        atomic_numbers,
                        displacements,
                        config.feature,
                        runtime.topology.site_types,
                    )
                    legacy_errors["multipoles"] = _tensor_maximum_error(
                        feature.equivariant_features,
                        legacy_feature.equivariant_features,
                    )
                except ValueError as error:
                    # Historical float32 fixed-Sinkhorn telemetry can sit on a
                    # quantization floor.  It is intentionally non-normative.
                    legacy_errors["multipoles"] = None
                    legacy_feature_error = f"{type(error).__name__}: {error}"
                fixed_float32_diagnostic["maximum_errors"] = legacy_errors
                fixed_float32_diagnostic["feature_validation_error"] = (
                    legacy_feature_error
                )
            else:
                legacy_errors = None
                fixed_float32_diagnostic = None
            try:
                (
                    _oracle_problem,
                    oracle_ot,
                    oracle_residuals,
                ) = _frozen_float64_dense_oracle(problem)
            except Exception as error:
                raise AutomaticEvaluationPolicyAuditError(
                    "TRANSPORT_ORACLE_CONVERGENCE_FAILURE",
                    "frozen-support CPU float64 dense oracle did not converge",
                    template_id=audit_input.template_id,
                    sample_id=geometry.sample_id,
                    geometry_digest=geometry.semantic_digest,
                    dtype=str(dtype).removeprefix("torch."),
                    backend="dense",
                    original_error=error,
                ) from error
            oracle_feature = build_probability_multipoles(
                oracle_ot.P,
                oracle_ot.q,
                atomic_numbers.detach().cpu(),
                displacements.detach().cpu().to(torch.float64),
                config.feature,
                runtime.topology.site_types.detach().cpu(),
            )
            frozen_oracle_errors = {
                "plan": _tensor_maximum_error(plan, oracle_ot.P),
                "q": _tensor_maximum_error(ot.q, oracle_ot.q),
                "multipoles": _tensor_maximum_error(
                    feature.equivariant_features,
                    oracle_feature.equivariant_features,
                ),
            }
            if legacy_errors is None:
                legacy_errors = frozen_oracle_errors
        else:
            legacy_errors = {"plan": 0.0, "q": 0.0, "multipoles": 0.0}
            frozen_oracle_errors = legacy_errors
            oracle_residuals = {"row": 0.0, "column": 0.0, "q_mass": 0.0}
            fixed_float32_diagnostic = None

    tolerance = (
        _FLOAT32_ORACLE_TOLERANCE
        if dtype == torch.float32
        else _FLOAT64_ORACLE_TOLERANCE
    )
    worst_oracle_residual = max(oracle_residuals.values())
    if worst_oracle_residual > _FROZEN_FLOAT64_ORACLE_RESIDUAL_TOLERANCE:
        raise AutomaticEvaluationPolicyAuditError(
            "TRANSPORT_ORACLE_CONVERGENCE_FAILURE",
            "frozen-support CPU float64 oracle residual exceeds its fixed tolerance",
            template_id=audit_input.template_id,
            sample_id=geometry.sample_id,
            geometry_digest=geometry.semantic_digest,
            dtype=str(dtype).removeprefix("torch."),
            backend=support_config.backend,
            observed=worst_oracle_residual,
            required=f"<= {_FROZEN_FLOAT64_ORACLE_RESIDUAL_TOLERANCE}",
            diagnostics=oracle_residuals,
        )
    worst_oracle = max(frozen_oracle_errors.values())
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
            diagnostics=frozen_oracle_errors,
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
        # Historical same-dtype comparison remains raw compatibility telemetry.
        # Audit v2 qualification uses only the branch-frozen CPU float64 oracle
        # fields below; the float32 fixed-Sinkhorn terminal cycle is explicitly
        # non-normative and excluded from semantic identity.
        "oracle_maximum_errors": legacy_errors,
        "frozen_float64_oracle_maximum_errors": frozen_oracle_errors,
        "frozen_float64_oracle_residuals": oracle_residuals,
        "fixed_sinkhorn_float32_diagnostic": fixed_float32_diagnostic,
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
        "branch_agreement": {
            "joint_translation": True,
            "periodic_lattice_wrap": True,
            "atom_permutation": True,
            "position_finite_difference": True,
            "strain_finite_difference": True,
        },
        "position_maximum_absolute_error": maximum_position,
        # Relative errors are diagnostic-only because near-zero directional
        # derivatives make them backend/allocation-history sensitive.  They
        # are retained at full precision in the integrity certificate, but
        # never participate in qualification or semantic identity.
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
                max(
                    item["frozen_float64_oracle_maximum_errors"].values()
                )
                for item in all_records
            ),
            "maximum_oracle_residual": max(
                max(item["frozen_float64_oracle_residuals"].values())
                for item in all_records
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
        AUTO_REFERENCE_CONVENTION_VERSION_V2,
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
        certificate = _finalize_evaluation_certificate(certificate)
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
        "convention_version": AUTO_REFERENCE_CONVENTION_VERSION_V2,
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
        convention_version=AUTO_REFERENCE_CONVENTION_VERSION_V2,
        _audit_inputs=preparation._audit_inputs,
    )


__all__ = [
    "AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION",
    "AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V1",
    "AUTOMATIC_EVALUATION_CERTIFICATE_SCHEMA_VERSION_V2",
    "AUTOMATIC_EVALUATION_POLICY_AUDIT_VERSION",
    "AUTOMATIC_EVALUATION_SEMANTIC_PROJECTION_VERSION",
    "AUTOMATIC_EVALUATION_SCOPE",
    "AutomaticEvaluationPolicyAuditError",
    "automatic_evaluation_certificate_semantic_identity",
    "automatic_evaluation_policy_profile",
    "build_automatic_evaluation_policy",
    "qualify_automatic_evaluation_policies",
    "validate_automatic_evaluation_certificate",
]
