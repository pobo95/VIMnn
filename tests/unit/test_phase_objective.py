from __future__ import annotations

import pytest
import torch

from conftest import reciprocal_fields
from refsite_mlip.phase.modes import (
    validate_runtime_amplitudes,
    validate_static_mode_amplitudes,
)
from refsite_mlip.phase.objective import (
    phase_gradient_hessian,
    phase_objective,
)


def test_analytic_gradient_and_hessian_match_autograd(typed_crystal):
    _, _, cross = reciprocal_fields(typed_crystal)
    phase = torch.tensor([0.21, -0.17, 0.09], dtype=torch.float64, requires_grad=True)
    modes = typed_crystal["modes"]
    weights = typed_crystal["mode_weights"]

    value = phase_objective(phase, cross, modes, weights)
    gradient = torch.autograd.grad(value, phase, create_graph=True)[0]
    hessian = torch.autograd.functional.jacobian(
        lambda value: torch.autograd.grad(
            phase_objective(value, cross, modes, weights),
            value,
            create_graph=True,
        )[0],
        phase,
        create_graph=True,
    )
    analytic_gradient, analytic_hessian = phase_gradient_hessian(
        phase, cross, modes, weights
    )
    torch.testing.assert_close(analytic_gradient, gradient, atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(analytic_hessian, hessian, atol=2.0e-11, rtol=2.0e-12)


def test_integer_wrapping_preserves_periodic_objective(typed_crystal):
    _, _, cross = reciprocal_fields(typed_crystal)
    phase = torch.tensor([0.19, -0.27, 0.31], dtype=torch.float64)
    integer = torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64)
    first = phase_objective(
        phase, cross, typed_crystal["modes"], typed_crystal["mode_weights"]
    )
    second = phase_objective(
        phase + integer,
        cross,
        typed_crystal["modes"],
        typed_crystal["mode_weights"],
    )
    torch.testing.assert_close(first, second, atol=2.0e-13, rtol=2.0e-13)


def test_static_mode_extinction_and_runtime_collapse_fail_fast(typed_crystal):
    atomic, reference, cross = reciprocal_fields(typed_crystal)
    validate_static_mode_amplitudes(
        reference,
        typed_crystal["channel_weights"],
        minimum_amplitude=1.0e-12,
    )
    validate_runtime_amplitudes(
        atomic,
        cross,
        typed_crystal["channel_weights"],
        minimum_atomic_amplitude=1.0e-12,
        minimum_cross_amplitude=1.0e-12,
    )

    with pytest.raises(ValueError, match="static typed reciprocal mode"):
        validate_static_mode_amplitudes(
            torch.zeros_like(reference),
            typed_crystal["channel_weights"],
            minimum_amplitude=1.0e-12,
        )
    with pytest.raises(ValueError, match="runtime atomic"):
        validate_runtime_amplitudes(
            torch.zeros_like(atomic),
            cross,
            typed_crystal["channel_weights"],
            minimum_atomic_amplitude=1.0e-12,
            minimum_cross_amplitude=1.0e-12,
        )
