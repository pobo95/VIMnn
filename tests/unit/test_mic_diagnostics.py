from __future__ import annotations

import itertools

import pytest
import torch

from refsite_mlip.transport import minimum_image_diagnostics


def _oracle(displacement, cell, radius=5):
    values = []
    for shift in itertools.product(range(-radius, radius + 1), repeat=3):
        vector = displacement - torch.tensor(shift, dtype=cell.dtype) @ cell
        values.append((torch.linalg.vector_norm(vector), vector))
    values.sort(key=lambda item: float(item[0]))
    return values[:2]


def _orthogonal(dtype):
    axis = torch.tensor([0.2, -0.7, 0.5], dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.63, dtype=dtype)
    cross = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=dtype)
    rotation = torch.eye(3, dtype=dtype) + torch.sin(angle) * cross + (1.0 - torch.cos(angle)) * (cross @ cross)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=dtype))
    return rotation, reflection


def test_mic_nearest_second_and_unique_gap_match_oracle():
    cell = torch.tensor([[3.2, 0.1, 0.0], [0.8, 2.9, 0.2], [-0.3, 0.6, 3.1]], dtype=torch.float64)
    displacement = torch.tensor([2.6, -1.8, 2.2], dtype=torch.float64)
    diagnostics = minimum_image_diagnostics(displacement, cell, (True, True, True))
    oracle = _oracle(displacement, cell)
    torch.testing.assert_close(diagnostics.displacement, oracle[0][1], atol=2e-13, rtol=0)
    torch.testing.assert_close(diagnostics.nearest_distance, oracle[0][0], atol=2e-13, rtol=0)
    torch.testing.assert_close(diagnostics.second_nearest_distance, oracle[1][0], atol=2e-13, rtol=0)
    torch.testing.assert_close(diagnostics.unique_image_gap, oracle[1][0] - oracle[0][0], atol=2e-13, rtol=0)


def test_mic_gap_is_rotation_and_reflection_invariant():
    cell = torch.tensor([[3.2, 0.1, 0.0], [0.8, 2.9, 0.2], [-0.3, 0.6, 3.1]], dtype=torch.float64)
    displacement = torch.tensor([2.6, -1.8, 2.2], dtype=torch.float64)
    baseline = minimum_image_diagnostics(displacement, cell, (True, True, True))
    for transform in _orthogonal(torch.float64):
        transformed = minimum_image_diagnostics(displacement @ transform.T, cell @ transform.T, (True, True, True))
        torch.testing.assert_close(transformed.displacement, baseline.displacement @ transform.T, atol=3e-13, rtol=0)
        torch.testing.assert_close(transformed.unique_image_gap, baseline.unique_image_gap, atol=3e-13, rtol=0)


def test_mic_low_unique_gap_fails_without_changing_vector():
    cell = torch.eye(3, dtype=torch.float64)
    displacement = torch.tensor([0.5, 0.1, 0.2], dtype=torch.float64)
    baseline = minimum_image_diagnostics(displacement, cell, (True, True, True))
    assert baseline.unique_image_gap == 0.0
    with pytest.raises(ValueError, match="not unique"):
        minimum_image_diagnostics(displacement, cell, (True, True, True), minimum_unique_gap=1e-12)
