from __future__ import annotations

import math

import pytest
import torch

from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    EvalOTConfig,
    TrainSinkhornConfig,
    TransportSupportConfig,
    TransportSupportError,
    atom_site_displacements,
    compact_c2_switch,
    solve_atom_vacancy_ot,
)


def _config(cutoff=2.5, width=0.5, skin=0.2):
    return TransportSupportConfig("compact_c2", cutoff, width, skin)


def _solve_distances(distances, *, iterations=256, config=None, tolerance=None):
    support = _config() if config is None else config
    cost = distances.square() / (2.0 * 1.5**2)
    return solve_atom_vacancy_ot(
        cost,
        0.5,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations, tolerance),
        support_config=support,
        atom_distances=distances,
        template_id="synthetic",
        sample_id="fixture",
    )


def test_support_config_validation_canonical_round_trip_and_dense_default():
    default = TransportSupportConfig.from_dict(None)
    assert default.kind == "dense"
    compact = TransportSupportConfig("compact_c2", 4.0, 0.5, 0.2)
    assert compact.r_on == 3.5 and compact.r_candidate == 4.2
    assert TransportSupportConfig.from_dict(compact.to_dict()) == compact
    assert tuple(compact.to_dict()) == (
        "kind",
        "cutoff",
        "switch_width",
        "candidate_skin",
        "convention_version",
    )
    for kwargs in (
        {"kind": "unknown"},
        {"cutoff": 0.0},
        {"switch_width": 0.0},
        {"cutoff": 1.0, "switch_width": 1.0},
        {"candidate_skin": -0.1},
        {"cutoff": math.inf},
    ):
        with pytest.raises(TransportSupportError) as failure:
            TransportSupportConfig(**kwargs)
        assert failure.value.reason_code == "INVALID_SUPPORT_CONFIG"


def test_dense_default_and_explicit_dense_are_tensor_bitwise_identical():
    cost = torch.tensor(
        [[0.12, 0.91], [0.73, 0.18], [0.41, 0.56]], dtype=torch.float64
    )
    config = TrainSinkhornConfig(256, 1.0e-7)
    legacy = solve_atom_vacancy_ot(cost, 0.5, TRAIN_FIXED, "sinkhorn", config)
    explicit = solve_atom_vacancy_ot(
        cost,
        0.5,
        TRAIN_FIXED,
        "sinkhorn",
        config,
        support_config=TransportSupportConfig("dense", 0.7, 0.2, 9.0),
        atom_distances=torch.full_like(cost, 100.0),
    )
    for name in ("gamma", "P", "q", "f", "g", "row_residual", "column_residual"):
        assert torch.equal(getattr(legacy, name), getattr(explicit, name))
    assert explicit.support_diagnostics is None


def _derivatives_at(value):
    coordinate = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    switch = compact_c2_switch(coordinate, _config(4.0, 0.5, 0.2))
    first = torch.autograd.grad(switch, coordinate, create_graph=True)[0]
    second = torch.autograd.grad(first, coordinate)[0]
    return float(switch), float(first), float(second)


def test_plateaued_c2_value_first_second_derivative_and_exact_mask():
    assert _derivatives_at(3.5) == (1.0, 0.0, 0.0)
    assert _derivatives_at(4.0) == (0.0, 0.0, 0.0)
    values = torch.tensor([3.49, 3.5, 3.75, 4.0, 4.01], dtype=torch.float64)
    switch = compact_c2_switch(values, _config(4.0, 0.5, 0.2))
    torch.testing.assert_close(
        switch, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0], dtype=torch.float64)
    )
    for endpoint, expected in ((3.5, 1.0), (4.0, 0.0)):
        left = _derivatives_at(endpoint - 1.0e-9)
        right = _derivatives_at(endpoint + 1.0e-9)
        assert abs(left[0] - expected) < 2.0e-12
        assert abs(right[0] - expected) < 2.0e-12
        assert max(abs(left[1]), abs(right[1])) < 3.0e-11
        assert max(abs(left[2]), abs(right[2])) < 5.0e-7


def _masked_oracle(log_kernel, rows, columns, epsilon, iterations=5000):
    f = torch.zeros_like(rows)
    g = torch.zeros_like(columns)
    for _ in range(iterations):
        f = epsilon * (
            torch.log(rows)
            - torch.logsumexp(g.unsqueeze(0) / epsilon + log_kernel, dim=1)
        )
        g = epsilon * (
            torch.log(columns)
            - torch.logsumexp(f.unsqueeze(1) / epsilon + log_kernel, dim=0)
        )
    return torch.exp(f.unsqueeze(1) / epsilon + g.unsqueeze(0) / epsilon + log_kernel)


def test_masked_log_sinkhorn_matches_manual_oracle_and_conserves_marginals():
    distances = torch.tensor(
        [[0.4, 0.8], [0.7, 2.6], [2.65, 0.6]], dtype=torch.float64
    )
    result = _solve_distances(distances)
    switch = compact_c2_switch(distances, _config())
    atom_cost = distances.square() / (2.0 * 1.5**2)
    active = distances < 2.5
    safe = torch.where(active, switch, torch.ones_like(switch))
    log_atoms = torch.where(
        active,
        -atom_cost / 0.5 + torch.log(safe),
        torch.full_like(atom_cost, -torch.inf),
    )
    log_kernel = torch.cat((log_atoms, torch.zeros((3, 1), dtype=torch.float64)), 1)
    oracle = _masked_oracle(
        log_kernel,
        torch.ones(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        torch.tensor(0.5, dtype=torch.float64),
    )
    torch.testing.assert_close(result.gamma, oracle, atol=2.0e-14, rtol=2.0e-14)
    assert result.gamma[1, 1] == 0 and result.gamma[2, 0] == 0
    torch.testing.assert_close(result.P.sum(0), torch.ones(2, dtype=torch.float64), atol=2e-14, rtol=0)
    torch.testing.assert_close(result.P.sum(1) + result.q, torch.ones(3, dtype=torch.float64), atol=2e-14, rtol=0)
    torch.testing.assert_close(result.q.sum(), torch.tensor(1.0, dtype=torch.float64), atol=2e-14, rtol=0)
    assert torch.all(result.q > 0), "the aggregate vacancy reservoir remains dense"
    diagnostics = result.support_diagnostics
    assert diagnostics.active_edge_count == 4
    assert diagnostics.candidate_edge_count == 6
    assert diagnostics.maximum_atom_matching_size == 2
    assert diagnostics.total_matching_size == 3
    assert diagnostics.total_support_feasible
    assert diagnostics.duplicate_atom_site_edge_count == 0
    assert diagnostics.template_id == "synthetic" and diagnostics.sample_id == "fixture"


def test_structured_support_failures_precede_sinkhorn():
    with pytest.raises(TransportSupportError) as failure:
        _solve_distances(torch.tensor([[0.2, 3.0], [0.3, 3.1]], dtype=torch.float64))
    assert failure.value.reason_code == "ATOM_WITHOUT_SUPPORT"

    hall_failure = torch.tensor(
        [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
        dtype=torch.float64,
    )
    with pytest.raises(TransportSupportError) as failure:
        _solve_distances(hall_failure)
    assert failure.value.reason_code == "INCOMPLETE_ATOM_MATCHING"

    no_total_support = torch.tensor(
        [[0.2, 3.0], [3.0, 0.2], [3.0, 3.0]], dtype=torch.float64
    )
    with pytest.raises(TransportSupportError) as failure:
        _solve_distances(no_total_support)
    assert failure.value.reason_code == "NO_TOTAL_SUPPORT"

    with pytest.raises(TransportSupportError) as failure:
        _solve_distances(torch.tensor([[0.2], [float("nan")]], dtype=torch.float64))
    assert failure.value.reason_code == "NONFINITE_SUPPORT_GEOMETRY"


@pytest.mark.parametrize("dtype,expected_tolerance", [(torch.float32, 1e-6), (torch.float64, 1e-7)])
def test_dtype_aware_diagnostic_tolerance_and_explicit_stricter_override(dtype, expected_tolerance):
    distances = torch.tensor(
        [[0.4, 0.8], [0.7, 2.7], [2.8, 0.6]], dtype=dtype
    )
    result = _solve_distances(distances)
    assert result.P.dtype == dtype
    assert result.effective_diagnostic_tolerance == expected_tolerance
    assert result.support_diagnostics.effective_diagnostic_tolerance == expected_tolerance
    strict = _solve_distances(distances, tolerance=1.0e-9)
    assert strict.effective_diagnostic_tolerance == 1.0e-9


def _compact_scalar(distances):
    result = _solve_distances(distances, iterations=96)
    weights = distances.new_tensor([[0.7, -0.2], [0.1, 0.4], [-0.3, 0.8]])
    return (result.P * weights).sum() + 0.23 * result.q.square().sum()


def test_compact_open_domain_gradcheck_and_gradgradcheck():
    distances = torch.tensor(
        [[0.45, 2.2], [2.15, 0.55], [2.3, 2.25]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        _compact_scalar, (distances,), eps=1e-6, atol=3e-6, rtol=3e-5
    )
    assert torch.autograd.gradgradcheck(
        _compact_scalar, (distances,), eps=1e-6, atol=8e-6, rtol=8e-5
    )


def _geometry_energy(positions, references, cell):
    displacements = atom_site_displacements(
        positions, references, cell, (True, True, True)
    )
    distances = torch.linalg.vector_norm(displacements, dim=-1)
    result = _solve_distances(distances, iterations=160)
    weights = positions.new_tensor([[0.5, 0.5], [0.1, 0.1], [-0.3, -0.3]])
    return (result.P * weights).sum() + 0.31 * result.q.square().sum()


def _geometry():
    positions = torch.tensor(
        [[0.2, 0.1, 0.05], [1.6, 0.4, 0.1]], dtype=torch.float64
    )
    references = torch.tensor(
        [[0.0, 0.0, 0.0], [1.8, 0.2, 0.0], [0.3, 1.7, 0.4]],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 6.0
    return positions, references, cell


def test_compact_geometry_force_stress_fd_and_symmetries():
    positions, references, cell = _geometry()
    live_positions = positions.clone().requires_grad_(True)
    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    deformation = torch.eye(3, dtype=torch.float64) + strain
    energy = _geometry_energy(
        live_positions @ deformation,
        references @ deformation,
        cell @ deformation,
    )
    position_gradient, strain_gradient = torch.autograd.grad(
        energy, (live_positions, strain), create_graph=True
    )
    assert torch.isfinite(torch.autograd.grad(position_gradient.square().sum(), live_positions)[0]).all()
    h = 1.0e-6
    perturbation = torch.zeros_like(positions)
    perturbation[1, 0] = h
    force_fd = -(
        _geometry_energy(positions + perturbation, references, cell)
        - _geometry_energy(positions - perturbation, references, cell)
    ) / (2.0 * h)
    torch.testing.assert_close(-position_gradient[1, 0], force_fd, atol=3e-6, rtol=3e-5)
    direction = torch.zeros((3, 3), dtype=torch.float64)
    direction[0, 1] = direction[1, 0] = 0.5
    plus = torch.eye(3, dtype=torch.float64) + h * direction
    minus = torch.eye(3, dtype=torch.float64) - h * direction
    stress_fd = (
        _geometry_energy(positions @ plus, references @ plus, cell @ plus)
        - _geometry_energy(positions @ minus, references @ minus, cell @ minus)
    ) / (2.0 * h)
    torch.testing.assert_close((strain_gradient * direction).sum(), stress_fd, atol=3e-6, rtol=3e-5)

    order = torch.tensor([1, 0])
    permuted = _geometry_energy(positions[order], references, cell)
    translated = _geometry_energy(positions + 0.37, references + 0.37, cell)
    wrapped = positions.clone()
    wrapped[0] += cell[0]
    wrapped_energy = _geometry_energy(wrapped, references, cell)
    torch.testing.assert_close(permuted, energy.detach(), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(translated, energy.detach(), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(wrapped_energy, energy.detach(), atol=2e-14, rtol=2e-14)


def test_compact_eval_adaptive_is_actionably_unsupported():
    distances = torch.tensor([[0.2], [0.4]], dtype=torch.float64)
    cost = distances.square() / (2.0 * 1.5**2)
    with pytest.raises(TransportSupportError) as failure:
        solve_atom_vacancy_ot(
            cost,
            0.5,
            EVAL_ADAPTIVE,
            "sinkhorn",
            EvalOTConfig(),
            support_config=_config(),
            atom_distances=distances,
        )
    assert failure.value.reason_code == "COMPACT_EVAL_ADAPTIVE_UNSUPPORTED"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compact_cuda_smoke(dtype):
    distances = torch.tensor(
        [[0.4, 0.8], [0.7, 2.7], [2.8, 0.6]], dtype=dtype, device="cuda"
    )
    result = _solve_distances(distances)
    assert result.P.device.type == "cuda" and result.P.dtype == dtype
    assert torch.isfinite(result.gamma).all()
