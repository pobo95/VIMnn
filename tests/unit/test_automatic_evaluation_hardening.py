from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from refsite_mlip.config.automatic_evaluation import (
    _candidate_coverage,
    _compare_phase_candidate_sets,
    automatic_evaluation_policy_profile,
    build_automatic_evaluation_policy,
)
from refsite_mlip.models import (
    PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1,
    EvaluationPolicy,
)
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.evaluation import (
    _group_non_equivalent_offsets,
    solve_evaluation_phase,
)
from refsite_mlip.phase.modes import validate_runtime_amplitudes
from refsite_mlip.phase.types import EvaluationPhaseError, TypedStabilizer


def _trivial_stabilizer() -> TypedStabilizer:
    return TypedStabilizer(
        translations=torch.zeros((1, 3), dtype=torch.float64),
        permutations=torch.zeros((1, 1), dtype=torch.long),
    )


def _solve_production(cross: torch.Tensor):
    acceptance = PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1
    return solve_evaluation_phase(
        cross,
        torch.eye(3, dtype=torch.long),
        torch.ones(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        _trivial_stabilizer(),
        step_schedule=acceptance.phase_step_schedule,
        damping_schedule=acceptance.phase_damping_schedule,
        minimum_gap=acceptance.minimum_objective_gap_absolute,
        minimum_curvature=acceptance.minimum_curvature,
        maximum_condition=acceptance.maximum_condition,
        maximum_gradient_norm=acceptance.maximum_gradient_norm,
        minimum_cross_amplitude=(
            acceptance.minimum_cross_amplitude_absolute
        ),
    )


def test_automatic_profile_cannot_weaken_production_phase_acceptance():
    production = PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1
    profile = automatic_evaluation_policy_profile()
    assert profile["minimum_objective_gap_absolute"] == (
        production.minimum_objective_gap_absolute
    )
    assert profile["minimum_curvature"] == production.minimum_curvature
    assert profile["maximum_condition"] == production.maximum_condition
    assert profile["minimum_cross_amplitude_absolute"] == (
        production.minimum_cross_amplitude_absolute
    )
    assert profile["minimum_atomic_amplitude_absolute"] == (
        production.minimum_atomic_amplitude_absolute
    )
    assert profile["minimum_reference_amplitude_absolute"] == (
        production.minimum_reference_amplitude_absolute
    )
    assert profile["maximum_gradient_norm"] == (
        production.maximum_gradient_norm
    )
    assert profile["equivalence_tolerance"] == (
        production.equivalence_tolerance
    )
    assert profile["phase_step_schedule"] == production.phase_step_schedule
    assert profile["phase_damping_schedule"] == (
        production.phase_damping_schedule
    )
    assert profile["max_witness_geometries"] == 256
    assert "audit_budget" not in profile


def test_refined_basin_envelope_is_audit_only_and_certificate_visible():
    production = PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1
    profile = automatic_evaluation_policy_profile()
    assert not hasattr(production, "refined_basin_equivalence_maximum")
    assert profile["refined_basin_equivalence_maximum"] == 1.0e-5
    assert profile["refined_basin_equivalence_maximum"] > (
        production.equivalence_tolerance
    )


def test_float32_candidate_grouping_uses_production_equivalence_tolerance():
    production = PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1
    stabilizer = TypedStabilizer(
        torch.tensor(
            [[0.0, 0.0, 0.0], [5.0e-9, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        torch.zeros((2, 1), dtype=torch.long),
    )
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [5.0e-9, 0.0, 0.0], [5.0e-6, 0.0, 0.0]],
        dtype=torch.float32,
    )
    grouped, representative_indices = _group_non_equivalent_offsets(
        offsets, stabilizer, production.equivalence_tolerance
    )
    assert representative_indices.tolist() == [0, 2]
    assert torch.equal(grouped, offsets[[0, 2]])


def test_production_objective_gap_boundary_is_strict():
    with pytest.raises(EvaluationPhaseError) as caught:
        _solve_production(
            torch.tensor([0.004999, 1.0, 1.0], dtype=torch.complex128)
        )
    assert caught.value.reason_code == "NON_EQUIVALENT_GAP_TOO_SMALL"

    result = _solve_production(
        torch.tensor([0.005001, 1.0, 1.0], dtype=torch.complex128)
    )
    assert float(result.non_equivalent_gap) > 1.0e-2


def test_production_curvature_and_condition_boundaries_are_strict():
    just_below_curvature = 0.009999 / (2.0 * math.pi) ** 2
    with pytest.raises(EvaluationPhaseError) as curvature:
        _solve_production(
            torch.tensor(
                [1.0, just_below_curvature, 1.0], dtype=torch.complex128
            )
        )
    assert curvature.value.reason_code == "HESSIAN_CURVATURE_FAILURE"

    acceptable_minimum = 0.010001 / (2.0 * math.pi) ** 2
    with pytest.raises(EvaluationPhaseError) as condition:
        _solve_production(
            torch.tensor(
                [1.0, acceptable_minimum, 1.1e6], dtype=torch.complex128
            )
        )
    assert condition.value.reason_code == "HESSIAN_CONDITION_FAILURE"


def test_production_amplitude_boundary_remains_one_e_minus_twelve():
    threshold = (
        PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1.minimum_cross_amplitude_absolute
    )
    atomic = torch.ones(3, dtype=torch.complex128)
    weights = torch.ones(3, dtype=torch.float64)
    at_boundary = torch.tensor(
        [threshold, 1.0, 1.0], dtype=torch.complex128
    )
    with pytest.raises(ValueError, match="cross amplitude"):
        validate_runtime_amplitudes(
            atomic, at_boundary, weights, threshold, threshold
        )
    above = math.nextafter(threshold, math.inf)
    validate_runtime_amplitudes(
        atomic,
        torch.tensor([above, 1.0, 1.0], dtype=torch.complex128),
        weights,
        threshold,
        threshold,
    )


def _candidate_template(
    modes: torch.Tensor,
    translations: torch.Tensor,
    permutations: torch.Tensor,
):
    return SimpleNamespace(
        template_id="candidate-test",
        fingerprint="f" * 64,
        phase_modes=modes,
        stabilizer=TypedStabilizer(translations, permutations),
    )


def test_built_automatic_policy_common_fields_match_production_profile():
    production = PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1
    policy = build_automatic_evaluation_policy(
        _candidate_template(
            torch.eye(3, dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.long),
        )
    )
    assert policy.minimum_objective_gap_absolute == (
        production.minimum_objective_gap_absolute
    )
    assert policy.minimum_atomic_amplitude_absolute == (
        production.minimum_atomic_amplitude_absolute
    )
    assert policy.minimum_reference_amplitude_absolute == (
        production.minimum_reference_amplitude_absolute
    )
    assert policy.minimum_cross_amplitude_absolute == (
        production.minimum_cross_amplitude_absolute
    )
    assert policy.minimum_curvature == production.minimum_curvature
    assert policy.maximum_condition == production.maximum_condition
    assert policy.maximum_gradient_norm == production.maximum_gradient_norm
    assert policy.equivalence_tolerance == production.equivalence_tolerance
    assert policy.phase_step_schedule == production.phase_step_schedule
    assert policy.phase_damping_schedule == production.phase_damping_schedule


def test_general_candidate_generation_is_not_the_legacy_fixed_four():
    template = _candidate_template(
        torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
            dtype=torch.long,
        ),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.long),
    )
    coverage = _candidate_coverage(template)
    legacy = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=torch.float64,
    )
    assert coverage.runtime_candidates.shape == (27, 3)
    assert coverage.audit_candidates.shape == (216, 3)
    assert not torch.equal(coverage.runtime_candidates, legacy)
    assert coverage.metadata["primary_mode_matrix"] == (
        template.phase_modes[:3].tolist()
    )
    assert coverage.metadata["primary_coordinate_bandlimit"] == 1
    assert coverage.metadata["primary_coordinate_l1_bandlimit"] == 2
    assert coverage.metadata["primary_coordinate_axis_bandlimits"] == [1, 1, 1]


def test_broader_coverage_detects_basin_missed_by_legacy_fixed_four():
    modes = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0]],
        dtype=torch.long,
    )
    stabilizer = _trivial_stabilizer()
    coverage = _candidate_coverage(
        _candidate_template(
            modes,
            stabilizer.translations,
            stabilizer.permutations,
        )
    )
    legacy = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=torch.float64,
    )
    cross = torch.ones(4, dtype=torch.complex128)
    cross[3] = torch.polar(
        torch.tensor(2.0, dtype=torch.float64),
        torch.tensor(-0.6 * math.pi, dtype=torch.float64),
    )
    initial = primary_phase_initialization(cross[:3], modes[:3])
    acceptance = PRODUCTION_EVALUATION_POLICY_ACCEPTANCE_V1

    def policy(candidates):
        return EvaluationPolicy(
            template_id="candidate-test",
            template_fingerprint="f" * 64,
            candidate_offsets=candidates,
            phase_step_schedule=acceptance.phase_step_schedule,
            phase_damping_schedule=acceptance.phase_damping_schedule,
            minimum_objective_gap_absolute=(
                acceptance.minimum_objective_gap_absolute
            ),
            minimum_cross_amplitude_absolute=(
                acceptance.minimum_cross_amplitude_absolute
            ),
            minimum_atomic_amplitude_absolute=(
                acceptance.minimum_atomic_amplitude_absolute
            ),
            minimum_reference_amplitude_absolute=(
                acceptance.minimum_reference_amplitude_absolute
            ),
            minimum_curvature=acceptance.minimum_curvature,
            maximum_condition=acceptance.maximum_condition,
            maximum_gradient_norm=acceptance.maximum_gradient_norm,
            equivalence_tolerance=acceptance.equivalence_tolerance,
        )

    with pytest.raises(ValueError) as caught:
        _compare_phase_candidate_sets(
            cross=cross,
            modes=modes,
            mode_weights=torch.ones(4, dtype=torch.float64),
            initial=initial,
            stabilizer=stabilizer,
            policy=policy(legacy),
            runtime_candidates=legacy,
            audit_candidates=coverage.audit_candidates,
            template_id="candidate-test",
            sample_id="missed-basin",
            geometry_digest="a" * 64,
        )
    assert caught.value.reason_code == "PHASE_CANDIDATE_COVERAGE_FAILED"
    assert caught.value.observed["same_stabilizer_group"] is False
    assert caught.value.observed["broader_minus_runtime_objective"] > 1.0

    accepted = _compare_phase_candidate_sets(
        cross=cross,
        modes=modes,
        mode_weights=torch.ones(4, dtype=torch.float64),
        initial=initial,
        stabilizer=stabilizer,
        policy=policy(coverage.runtime_candidates),
        runtime_candidates=coverage.runtime_candidates,
        audit_candidates=coverage.audit_candidates,
        template_id="candidate-test",
        sample_id="covered-basin",
        geometry_digest="b" * 64,
    )
    assert accepted["same_stabilizer_group"] is True
    assert abs(accepted["broader_minus_runtime_objective"]) <= 1.0e-10


def test_rocksalt_alias_candidates_reduce_stabilizer_duplicates_deterministically():
    modes = torch.tensor(
        [
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
            [0, 0, 2],
            [0, 2, 0],
        ],
        dtype=torch.long,
    )
    translations = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float64,
    )
    permutations = torch.arange(4, dtype=torch.long).repeat(4, 1)
    forward = _candidate_coverage(
        _candidate_template(modes, translations, permutations)
    )
    repeated_template = _candidate_template(modes, translations, permutations)
    repeated = _candidate_coverage(repeated_template)
    order = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    reordered = _candidate_coverage(
        _candidate_template(
            modes,
            translations[order],
            permutations[order],
        )
    )
    assert forward.metadata["alias_kernel_order"] == 4
    assert forward.metadata["runtime_raw_candidate_count"] == 108
    assert forward.metadata["runtime_candidate_count"] == 27
    assert forward.metadata["broader_audit_raw_candidate_count"] == 864
    assert forward.metadata["broader_audit_candidate_count"] == 216
    assert torch.equal(forward.runtime_candidates, reordered.runtime_candidates)
    assert torch.equal(forward.audit_candidates, reordered.audit_candidates)
    assert dict(forward.metadata) == dict(reordered.metadata)
    assert torch.equal(forward.runtime_candidates, repeated.runtime_candidates)
    assert torch.equal(forward.audit_candidates, repeated.audit_candidates)
    assert dict(forward.metadata) == dict(repeated.metadata)
    assert (
        build_automatic_evaluation_policy(repeated_template).content_fingerprint
        == build_automatic_evaluation_policy(repeated_template).content_fingerprint
        == build_automatic_evaluation_policy(
            _candidate_template(modes, translations[order], permutations[order])
        ).content_fingerprint
    )


def test_candidate_generation_rejects_modes_outside_supported_general_domain():
    template = _candidate_template(
        torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [8, 0, 0]],
            dtype=torch.long,
        ),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.long),
    )
    with pytest.raises(ValueError, match="bandlimit"):
        _candidate_coverage(template)
