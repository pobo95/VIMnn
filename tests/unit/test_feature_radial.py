from __future__ import annotations

import torch

from refsite_mlip.features.radial import compact_radial_basis
from refsite_mlip.features.solid_harmonics import regular_solid_harmonics


def _value_gradient_hessian(xi_value):
    xi = torch.tensor(xi_value, dtype=torch.float64, requires_grad=True)
    value = compact_radial_basis(
        xi, n_radial=3, ell_feature=1.0, r_cut=2.0
    )
    gradients = []
    hessians = []
    for channel in range(3):
        gradient = torch.autograd.grad(
            value[channel], xi, create_graph=True, retain_graph=True
        )[0]
        hessian = torch.autograd.grad(
            gradient, xi, retain_graph=True
        )[0]
        gradients.append(gradient)
        hessians.append(hessian)
    return value, torch.stack(gradients), torch.stack(hessians)


def test_cutoff_value_first_and_second_derivative_continuity():
    delta = 1.0e-5
    below = _value_gradient_hessian(4.0 - delta)
    at = _value_gradient_hessian(4.0)
    above = _value_gradient_hessian(4.0 + delta)
    torch.testing.assert_close(at[0], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(at[1], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(at[2], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(above[0], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(above[1], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(above[2], torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
    assert below[0].abs().max() < 2.0e-13
    assert below[1].abs().max() < 5.0e-8
    assert below[2].abs().max() < 1.0e-2


def test_cutoff_outside_is_exact_zero():
    xi = torch.tensor([4.0, 4.1, 20.0], dtype=torch.float64)
    values = compact_radial_basis(
        xi, n_radial=4, ell_feature=1.0, r_cut=2.0
    )
    torch.testing.assert_close(values, torch.zeros_like(values), atol=0, rtol=0)


def test_radial_times_solid_harmonic_gradcheck_and_gradgradcheck():
    y = torch.tensor(
        [[0.21, -0.17, 0.13], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def function(value):
        xi = torch.sum(value * value, dim=-1)
        radial = compact_radial_basis(
            xi, n_radial=3, ell_feature=1.0, r_cut=2.0
        )
        solid = regular_solid_harmonics(value)[0]
        return radial.unsqueeze(-1) * solid.unsqueeze(-2)

    assert torch.autograd.gradcheck(
        function, (y,), eps=1.0e-6, atol=4.0e-6, rtol=4.0e-5
    )
    assert torch.autograd.gradgradcheck(
        function, (y,), eps=1.0e-6, atol=8.0e-6, rtol=8.0e-5
    )
