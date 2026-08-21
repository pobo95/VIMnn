from __future__ import annotations

import math

import pytest
import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.features.solid_harmonics import (
    harmonic_slice,
    regular_solid_harmonics,
)


def _rotation(dtype=torch.float64):
    axis = torch.tensor([0.3, -0.5, 0.8], dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.71, dtype=dtype)
    cross = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=dtype,
    )
    identity = torch.eye(3, dtype=dtype)
    return identity + torch.sin(angle) * cross + (1.0 - torch.cos(angle)) * (cross @ cross)


def test_origin_and_analytic_l012_convention():
    zero = torch.zeros(3, dtype=torch.float64)
    origin = regular_solid_harmonics(zero)[0]
    torch.testing.assert_close(origin[0], torch.ones((), dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(origin[1:], torch.zeros(8, dtype=torch.float64), atol=0, rtol=0)

    x, y, z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    expected = torch.stack(
        (
            x.new_ones(()),
            math.sqrt(3.0) * x,
            math.sqrt(3.0) * y,
            math.sqrt(3.0) * z,
            math.sqrt(15.0) * x * z,
            math.sqrt(15.0) * x * y,
            math.sqrt(5.0) / 2.0 * (2.0 * y * y - x * x - z * z),
            math.sqrt(15.0) * y * z,
            math.sqrt(15.0) / 2.0 * (z * z - x * x),
        )
    )
    actual = regular_solid_harmonics(torch.stack((x, y, z)))[0]
    torch.testing.assert_close(actual, expected, atol=3.0e-14, rtol=3.0e-14)


def test_homogeneity_and_inversion_parity():
    y = torch.tensor([0.31, -0.27, 0.19], dtype=torch.float64)
    scale = 1.7
    base = regular_solid_harmonics(y)[0]
    scaled = regular_solid_harmonics(scale * y)[0]
    inverted = regular_solid_harmonics(-y)[0]
    for l in range(3):
        block = harmonic_slice(l)
        torch.testing.assert_close(
            scaled[block], scale**l * base[block], atol=2.0e-14, rtol=2.0e-14
        )
        torch.testing.assert_close(
            inverted[block], (-1) ** l * base[block], atol=2.0e-14, rtol=2.0e-14
        )


@pytest.mark.parametrize(
    "matrix",
    [
        _rotation(),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
    ],
)
def test_o3_equivariance(matrix):
    _, o3 = import_e3nn_0_4_4()
    y = torch.tensor(
        [[0.31, -0.27, 0.19], [-0.12, 0.41, 0.23]], dtype=torch.float64
    )
    values, irreps = regular_solid_harmonics(y)
    transformed = regular_solid_harmonics(y @ matrix.T)[0]
    representation = irreps.D_from_matrix(matrix)
    expected = values @ representation.T
    torch.testing.assert_close(transformed, expected, atol=3.0e-13, rtol=3.0e-13)


def test_solid_harmonics_gradcheck_gradgradcheck_and_origin_finiteness():
    y = torch.tensor(
        [[0.21, -0.17, 0.13], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    function = lambda value: regular_solid_harmonics(value)[0]
    assert torch.autograd.gradcheck(
        function, (y,), eps=1.0e-6, atol=2.0e-6, rtol=2.0e-5
    )
    assert torch.autograd.gradgradcheck(
        function, (y,), eps=1.0e-6, atol=3.0e-6, rtol=3.0e-5
    )
    coefficients = torch.arange(1, 10, dtype=torch.float64)
    scalar = (function(y) * coefficients).sum()
    first = torch.autograd.grad(scalar, y, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), y)[0]
    assert torch.all(torch.isfinite(first))
    assert torch.all(torch.isfinite(second))
