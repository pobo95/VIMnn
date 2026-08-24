from __future__ import annotations

import math

import pytest
import torch

from conftest import reciprocal_fields
from refsite_mlip.phase.evaluation import solve_evaluation_phase
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase, validate_training_result
from refsite_mlip.phase.stabilizer import (
    find_typed_stabilizer,
    torus_difference,
)
from refsite_mlip.phase.types import TypedStabilizer


STEPS = (0.65, 0.75, 0.85, 0.95, 1.0, 1.0)
DAMPING = (4.0, 2.0, 1.0, 0.5, 0.2, 0.1)


def _solve(data, positions=None):
    _, _, cross = reciprocal_fields(data, positions=positions)
    initial = primary_phase_initialization(cross[:3], data["modes"][:3])
    result = solve_training_phase(
        cross,
        data["modes"],
        data["mode_weights"],
        initial,
        STEPS,
        DAMPING,
    )
    return cross, result


def test_fixed_finite_newton_iterations_are_translation_covariant(typed_crystal):
    _, baseline = _solve(typed_crystal)
    translation = torch.tensor([0.63, -0.91, 1.13], dtype=torch.float64)
    _, translated = _solve(
        typed_crystal, positions=typed_crystal["positions"] + translation
    )
    fractional_translation = torch.linalg.solve(
        typed_crystal["cell"].T, translation
    )
    error = torus_difference(
        translated.phase - baseline.phase, fractional_translation
    )
    torch.testing.assert_close(error, torch.zeros_like(error), atol=4.0e-12, rtol=0.0)
    validate_training_result(
        baseline,
        minimum_regularized_curvature=1.0e-8,
        maximum_gradient_norm=2.0e-4,
    )


def test_hessian_regularization_sensitivity_is_explicit(typed_crystal):
    _, _, cross = reciprocal_fields(typed_crystal)
    initial = primary_phase_initialization(cross[:3], typed_crystal["modes"][:3])
    light = solve_training_phase(
        cross,
        typed_crystal["modes"],
        typed_crystal["mode_weights"],
        initial,
        (0.8, 0.8),
        (0.1, 0.1),
    )
    heavy = solve_training_phase(
        cross,
        typed_crystal["modes"],
        typed_crystal["mode_weights"],
        initial,
        (0.8, 0.8),
        (100.0, 100.0),
    )
    assert torch.linalg.vector_norm(light.phase - heavy.phase) > 1.0e-8
    assert torch.all(torch.isfinite(light.phase))
    assert torch.all(torch.isfinite(heavy.phase))


def _trivial_stabilizer(dtype=torch.float64):
    return TypedStabilizer(
        translations=torch.zeros((1, 3), dtype=dtype),
        permutations=torch.zeros((1, 1), dtype=torch.long),
    )


def _evaluation_kwargs():
    return dict(
        step_schedule=(1.0, 1.0, 1.0),
        damping_schedule=(0.1, 0.05, 0.01),
        minimum_gap=0.1,
        minimum_curvature=1.0,
        maximum_condition=1.0e5,
        maximum_gradient_norm=1.0e-7,
        minimum_cross_amplitude=1.0e-12,
    )


def test_evaluation_solver_selects_stable_covariant_branch():
    dtype = torch.float64
    modes = torch.eye(3, dtype=torch.long)
    weights = torch.ones(3, dtype=dtype)
    cross = torch.ones(3, dtype=torch.complex128)
    initial = torch.zeros(3, dtype=dtype)
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        dtype=dtype,
    )
    result = solve_evaluation_phase(
        cross,
        modes,
        weights,
        initial,
        offsets,
        _trivial_stabilizer(),
        **_evaluation_kwargs(),
    )
    torch.testing.assert_close(result.refined.phase, initial, atol=1.0e-14, rtol=0.0)
    assert int(result.selected_index) == 0


def test_evaluation_groups_stabilizer_equivalent_candidates():
    dtype = torch.float64
    modes = torch.tensor([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.long)
    weights = torch.ones(3, dtype=dtype)
    cross = torch.ones(3, dtype=torch.complex128)
    sites = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
    stabilizer = find_typed_stabilizer(
        sites, torch.tensor([0, 0], dtype=torch.long)
    )
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.0, 0.0]],
        dtype=dtype,
    )
    initial = torch.zeros(3, dtype=dtype)
    training = solve_training_phase(
        cross, modes, weights, initial, (1.0, 1.0, 1.0), (0.1, 0.05, 0.01)
    )
    result = solve_evaluation_phase(
        cross,
        modes,
        weights,
        initial,
        offsets,
        stabilizer,
        **_evaluation_kwargs(),
    )
    assert int(result.selected_index) == 0
    from refsite_mlip.phase.stabilizer import stabilizer_equivalent

    assert stabilizer_equivalent(result.refined.phase, training.phase, stabilizer)
    assert result.input_candidate_count == 3
    assert result.non_equivalent_group_count == 2
    assert int(result.selected_grouped_index) == 0
    torch.testing.assert_close(
        result.best_raw_score - result.second_best_raw_score,
        result.non_equivalent_gap,
    )


def test_evaluation_phase_candidate_switching_boundary_fails_gap():
    dtype = torch.float64
    modes = torch.eye(3, dtype=torch.long)
    cross = torch.tensor([1.0j, 1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex128)
    offsets = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
    kwargs = _evaluation_kwargs()
    with pytest.raises(ValueError, match="gap"):
        solve_evaluation_phase(
            cross,
            modes,
            torch.ones(3, dtype=dtype),
            torch.zeros(3, dtype=dtype),
            offsets,
            _trivial_stabilizer(),
            **kwargs,
        )


def test_evaluation_hessian_condition_and_final_residual_fail_fast():
    dtype = torch.float64
    modes = torch.eye(3, dtype=torch.long)
    offsets = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
    kwargs = _evaluation_kwargs()

    poorly_curved = torch.tensor([1.0, 1.0e-8, 1.0e-8], dtype=torch.complex128)
    with pytest.raises(ValueError, match="Hessian"):
        solve_evaluation_phase(
            poorly_curved,
            modes,
            torch.ones(3, dtype=dtype),
            torch.zeros(3, dtype=dtype),
            offsets,
            _trivial_stabilizer(),
            **{**kwargs, "minimum_curvature": 1.0e-5},
        )

    angle = torch.tensor(0.2 * 2.0 * math.pi, dtype=dtype)
    cross = torch.polar(torch.ones(3, dtype=dtype), torch.tensor([angle, 0.0, 0.0]))
    with pytest.raises(ValueError, match="gradient"):
        solve_evaluation_phase(
            cross,
            modes,
            torch.ones(3, dtype=dtype),
            torch.zeros(3, dtype=dtype),
            offsets,
            _trivial_stabilizer(),
            step_schedule=(0.001,),
            damping_schedule=(1.0,),
            minimum_gap=0.1,
            minimum_curvature=1.0,
            maximum_condition=1.0e5,
            maximum_gradient_norm=1.0e-10,
            minimum_cross_amplitude=1.0e-12,
        )


def test_evaluation_runtime_amplitude_collapse_fails_fast():
    dtype = torch.float64
    with pytest.raises(ValueError, match="amplitude"):
        solve_evaluation_phase(
            torch.tensor([0.0, 1.0, 1.0], dtype=torch.complex128),
            torch.eye(3, dtype=torch.long),
            torch.ones(3, dtype=dtype),
            torch.zeros(3, dtype=dtype),
            torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype),
            _trivial_stabilizer(),
            **_evaluation_kwargs(),
        )


def test_newton_loop_does_not_wrap_and_supports_batches():
    dtype = torch.float64
    modes = torch.eye(3, dtype=torch.long)
    weights = torch.ones(3, dtype=dtype)
    angles = 2.0 * math.pi * torch.tensor(
        [[0.11, -0.07, 0.13], [0.21, 0.18, -0.16]], dtype=dtype
    )
    cross = torch.polar(torch.ones_like(angles), angles)
    initial = torch.tensor(
        [[2.05, -3.04, 1.08], [-1.9, 2.1, 3.2]], dtype=dtype
    )
    result = solve_training_phase(
        cross, modes, weights, initial, (0.8, 1.0), (1.0, 0.5)
    )
    shifted = solve_training_phase(
        cross,
        modes,
        weights,
        initial + torch.tensor([3.0, -2.0, 4.0], dtype=dtype),
        (0.8, 1.0),
        (1.0, 0.5),
    )
    assert result.phase.shape == (2, 3)
    torch.testing.assert_close(
        shifted.phase - result.phase,
        torch.tensor([3.0, -2.0, 4.0], dtype=dtype).expand(2, 3),
        atol=2.0e-13,
        rtol=0.0,
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_fixed_schedule_fails_fast(bad):
    with pytest.raises(ValueError, match="schedule"):
        solve_training_phase(
            torch.ones(3, dtype=torch.complex128),
            torch.eye(3, dtype=torch.long),
            torch.ones(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            (bad,),
            (1.0,),
        )
