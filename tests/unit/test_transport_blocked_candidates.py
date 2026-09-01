from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from refsite_mlip.transport import (
    EvalOTConfig,
    TrainSinkhornConfig,
    TransportSupportConfig,
    TransportSupportError,
    atom_site_displacements,
    build_compact_transport_edges,
    build_periodic_compact_transport_edges,
    compact_c2_switch,
    solve_sparse_hybrid_eval,
    solve_sparse_sinkhorn_train_fixed,
    validate_compact_support_edges,
)
from refsite_mlip.transport.support import validate_compact_support


def _support(candidate_backend="blocked", site_block=1, atom_block=1):
    return TransportSupportConfig(
        kind="compact_c2",
        cutoff=2.0,
        switch_width=0.5,
        candidate_skin=0.4,
        backend="edge_list",
        candidate_backend=candidate_backend,
        site_block_size=site_block,
        atom_block_size=atom_block,
    )


def _geometry(dtype=torch.float64, device="cpu"):
    positions = torch.tensor(
        [[0.4, 0.08, 0.03], [0.8, -0.04, 0.02]],
        dtype=dtype,
        device=device,
    )
    references = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [2.65, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    cell = torch.tensor(
        [[8.0, 0.15, -0.05], [0.2, 7.7, 0.1], [-0.1, 0.25, 8.2]],
        dtype=dtype,
        device=device,
    )
    return positions, references, cell


def _edges(config, *, dtype=torch.float64, device="cpu", positions=None, cell=None):
    base_positions, references, base_cell = _geometry(dtype, device)
    return build_periodic_compact_transport_edges(
        base_positions if positions is None else positions,
        references,
        base_cell if cell is None else cell,
        (True, True, True),
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=config,
        template_id="blocked-fixture",
        sample_id="sample-0",
    )


def _assert_edge_parity(left, right):
    for name in (
        "site_index",
        "atom_index",
        "periodic_shift",
        "displacements",
        "distances",
        "switch",
        "log_kernel",
        "active",
        "atom_major_permutation",
        "site_ptr",
        "atom_ptr",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
    assert (
        left.support_diagnostics.candidate_fingerprint
        == right.support_diagnostics.candidate_fingerprint
    )


def test_blocked_config_round_trip_preserves_legacy_dense_payload():
    legacy = TransportSupportConfig(
        "compact_c2", 2.0, 0.5, 0.4, backend="edge_list"
    )
    assert "candidate_backend" not in legacy.to_dict()
    assert TransportSupportConfig.from_dict(legacy.to_dict()) == legacy
    blocked = _support("blocked", 7, 5)
    assert TransportSupportConfig.from_dict(blocked.to_dict()) == blocked
    assert blocked.to_dict()["site_block_size"] == 7
    for values in (
        {"candidate_backend": "unknown"},
        {"candidate_backend": "blocked", "site_block_size": 0},
        {"candidate_backend": "blocked", "atom_block_size": True},
    ):
        with pytest.raises(TransportSupportError) as failure:
            TransportSupportConfig(
                "compact_c2", 2.0, 0.5, 0.4, backend="edge_list", **values
            )
        assert failure.value.reason_code == "INVALID_SUPPORT_CONFIG"


def test_block_size_independent_exact_dense_candidate_parity_and_diagnostics():
    dense = _edges(_support("dense"))
    one = _edges(_support("blocked", 1, 1))
    mixed = _edges(_support("blocked", 2, 2))
    _assert_edge_parity(dense, one)
    _assert_edge_parity(one, mixed)
    for name in (
        "cutoff_boundary_gap",
        "switch_on_boundary_gap",
        "candidate_boundary_gap",
        "mic_image_gap",
    ):
        assert getattr(dense.support_diagnostics, name) == getattr(
            one.support_diagnostics, name
        )
    diagnostics = one.support_diagnostics
    assert diagnostics.candidate_backend == "blocked"
    assert diagnostics.num_sites == 3 and diagnostics.num_atoms == 2
    assert diagnostics.processed_block_count == 6
    assert diagnostics.maximum_pair_block_elements == 1
    assert diagnostics.theoretical_full_pair_elements == 6
    assert not diagnostics.dense_candidate_allocation_observed
    assert diagnostics.candidate_edge_count == 6
    assert diagnostics.active_edge_count == 5
    assert torch.equal(one.switch[~one.active], torch.zeros_like(one.switch[~one.active]))
    assert torch.isneginf(one.log_kernel[~one.active]).all()
    assert diagnostics.maximum_atom_matching_size == 2
    assert diagnostics.total_matching_size == 3
    assert diagnostics.total_support_feasible
    assert diagnostics.cutoff_boundary_gap > 0.0
    assert diagnostics.switch_on_boundary_gap > 0.0
    assert diagnostics.candidate_boundary_gap > 0.0
    assert diagnostics.mic_image_gap > 0.0


def test_triclinic_wrapped_shift_and_mic_tie_break_match_dense_oracle():
    positions, _, cell = _geometry()
    wrapped = positions.clone()
    wrapped[0] += cell[1]
    dense = _edges(_support("dense"), positions=wrapped)
    blocked = _edges(_support("blocked", 1, 2), positions=wrapped)
    _assert_edge_parity(dense, blocked)
    assert torch.equal(
        blocked.periodic_shift[blocked.atom_index == 0, 1],
        torch.ones(3, dtype=torch.long),
    )

    tie_positions = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
    tie_references = torch.zeros((1, 3), dtype=torch.float64)
    tie_cell = torch.eye(3, dtype=torch.float64)
    base = dict(
        kind="compact_c2",
        cutoff=0.75,
        switch_width=0.25,
        candidate_skin=0.25,
        backend="edge_list",
    )
    dense_tie = build_periodic_compact_transport_edges(
        tie_positions,
        tie_references,
        tie_cell,
        (True,) * 3,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=TransportSupportConfig(**base),
    )
    blocked_tie = build_periodic_compact_transport_edges(
        tie_positions,
        tie_references,
        tie_cell,
        (True,) * 3,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=TransportSupportConfig(**base, candidate_backend="blocked"),
    )
    _assert_edge_parity(dense_tie, blocked_tie)
    assert dense_tie.support_diagnostics.mic_image_gap == 0.0


def test_new_dense_candidate_frontend_matches_legacy_dense_displacement_builder():
    positions, references, cell = _geometry()
    positions = positions.clone()
    positions[0] += cell[2]
    config = _support("dense")
    legacy_displacements = atom_site_displacements(
        positions, references, cell, (True,) * 3
    )
    legacy = build_compact_transport_edges(
        legacy_displacements,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=config,
    )
    frontend = build_periodic_compact_transport_edges(
        positions,
        references,
        cell,
        (True,) * 3,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=config,
    )
    for name in (
        "site_index",
        "atom_index",
        "displacements",
        "distances",
        "switch",
        "log_kernel",
        "active",
        "site_ptr",
        "atom_ptr",
        "atom_major_permutation",
    ):
        assert torch.equal(getattr(legacy, name), getattr(frontend, name)), name


def test_blocked_fixed_and_adaptive_match_dense_candidate_execution():
    dense_edges = _edges(_support("dense"))
    blocked_edges = _edges(_support("blocked", 2, 1))
    dense_fixed = solve_sparse_sinkhorn_train_fixed(
        dense_edges, TrainSinkhornConfig(256)
    )
    blocked_fixed = solve_sparse_sinkhorn_train_fixed(
        blocked_edges, TrainSinkhornConfig(256)
    )
    for name in ("edge_plan", "q", "f", "g", "row_residual", "column_residual"):
        assert torch.equal(getattr(dense_fixed, name), getattr(blocked_fixed, name))
    config = EvalOTConfig(
        sinkhorn_iterations=3,
        max_newton_iterations=20,
        convergence_tolerance=1.0e-12,
        pcg_max_iterations=256,
        fallback_sinkhorn_iterations=4096,
    )
    dense_eval = solve_sparse_hybrid_eval(dense_edges, config)
    blocked_eval = solve_sparse_hybrid_eval(blocked_edges, config)
    for name in ("edge_plan", "q", "f", "g", "row_residual", "column_residual"):
        assert torch.equal(getattr(dense_eval, name), getattr(blocked_eval, name))
    assert dense_eval.fallback_used == blocked_eval.fallback_used


def _fixed_observable(positions, cell, *, adaptive=False):
    edges = _edges(_support("blocked", 2, 1), positions=positions, cell=cell)
    result = (
        solve_sparse_hybrid_eval(
            edges,
            EvalOTConfig(
                sinkhorn_iterations=8,
                max_newton_iterations=20,
                convergence_tolerance=1.0e-11,
                fallback_sinkhorn_iterations=4096,
            ),
        )
        if adaptive
        else solve_sparse_sinkhorn_train_fixed(edges, TrainSinkhornConfig(96))
    )
    weights = torch.linspace(
        -0.3,
        0.4,
        result.edge_plan.numel(),
        dtype=positions.dtype,
        device=positions.device,
    )
    return (result.edge_plan * weights).sum() + 0.17 * result.q.square().sum()


def test_blocked_live_geometry_gradcheck_gradgradcheck_and_adaptive_fd():
    positions, _, cell = _geometry()
    positions = positions.requires_grad_(True)
    cell = cell.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda p, h: _fixed_observable(p, h),
        (positions, cell),
        eps=1.0e-6,
        atol=7.0e-6,
        rtol=7.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        lambda p, h: _fixed_observable(p, h),
        (positions, cell),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )
    energy = _fixed_observable(positions, cell, adaptive=True)
    position_gradient, cell_gradient = torch.autograd.grad(
        energy, (positions, cell)
    )
    step = 1.0e-6
    position_delta = torch.zeros_like(positions)
    position_delta[1, 0] = step
    position_fd = (
        _fixed_observable(positions.detach() + position_delta, cell.detach(), adaptive=True)
        - _fixed_observable(positions.detach() - position_delta, cell.detach(), adaptive=True)
    ) / (2.0 * step)
    torch.testing.assert_close(
        position_gradient[1, 0], position_fd, atol=4.0e-6, rtol=4.0e-5
    )
    cell_delta = torch.zeros_like(cell)
    cell_delta[0, 1] = step
    cell_fd = (
        _fixed_observable(positions.detach(), cell.detach() + cell_delta, adaptive=True)
        - _fixed_observable(positions.detach(), cell.detach() - cell_delta, adaptive=True)
    ) / (2.0 * step)
    torch.testing.assert_close(
        cell_gradient[0, 1], cell_fd, atol=4.0e-6, rtol=4.0e-5
    )


def test_selected_edge_switch_is_c2_at_r_off_without_candidate_change():
    _, references, cell = _geometry()
    transverse = 0.08**2 + 0.03**2
    x_coordinate = 2.65 - (2.0**2 - transverse) ** 0.5
    positions = torch.tensor(
        [[x_coordinate, 0.08, 0.03], [0.8, -0.04, 0.02]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = build_periodic_compact_transport_edges(
        positions,
        references,
        cell,
        (True,) * 3,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_support("blocked", 1, 1),
    )
    target = (edges.site_index == 2) & (edges.atom_index == 0)
    assert int(target.sum()) == 1
    value = edges.switch[target].sum()
    first = torch.autograd.grad(value, positions, create_graph=True)[0]
    second = torch.autograd.grad(first[0, 0], positions)[0]
    assert abs(float(value)) < 2.0e-14
    assert abs(float(first[0, 0])) < 2.0e-12
    assert abs(float(second[0, 0])) < 2.0e-9
    assert edges.support_diagnostics.candidate_edge_count == 6


def test_wrapping_joint_translation_fractional_origin_and_atom_covariance():
    positions, references, cell = _geometry()
    baseline = _edges(_support("blocked", 1, 1))
    wrapped = positions.clone()
    wrapped[0] += cell[1]
    wrapped_edges = _edges(_support("blocked", 2, 1), positions=wrapped)
    torch.testing.assert_close(
        baseline.distances, wrapped_edges.distances, atol=3e-15, rtol=0
    )
    torch.testing.assert_close(
        baseline.displacements, wrapped_edges.displacements, atol=3e-15, rtol=0
    )

    translation = torch.tensor([0.31, -0.27, 0.19], dtype=torch.float64)
    translated = build_periodic_compact_transport_edges(
        positions + translation,
        references + translation,
        cell,
        (True, True, True),
        origin=translation,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_support("blocked", 1, 2),
    )
    torch.testing.assert_close(
        translated.displacements, baseline.displacements, atol=2e-15, rtol=0
    )

    wrapped_references = references.clone()
    wrapped_references[1] -= cell[0]
    reference_wrapped = build_periodic_compact_transport_edges(
        positions,
        wrapped_references,
        cell,
        (True, True, True),
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_support("blocked", 2, 1),
    )
    torch.testing.assert_close(
        reference_wrapped.displacements,
        baseline.displacements,
        atol=3e-15,
        rtol=0,
    )

    origin = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    fractional = (references - origin) @ torch.linalg.inv(cell)
    fractional_edges = build_periodic_compact_transport_edges(
        positions,
        fractional,
        cell,
        (True, True, True),
        origin=origin,
        reference_coordinates="fractional",
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_support("blocked", 2, 2),
    )
    for name in (
        "site_index",
        "atom_index",
        "periodic_shift",
        "active",
        "site_ptr",
        "atom_ptr",
        "atom_major_permutation",
    ):
        assert torch.equal(getattr(baseline, name), getattr(fractional_edges, name))
    torch.testing.assert_close(
        baseline.displacements, fractional_edges.displacements, atol=3e-15, rtol=0
    )
    torch.testing.assert_close(
        baseline.distances, fractional_edges.distances, atol=3e-15, rtol=0
    )

    permutation = torch.tensor([1, 0])
    permuted = _edges(
        _support("blocked", 1, 1), positions=positions[permutation]
    )
    inverse = torch.argsort(permutation)
    mapped_atom = permutation[permuted.atom_index]
    key = permuted.site_index * 2 + mapped_atom
    order = torch.argsort(key)
    assert torch.equal(permuted.site_index[order], baseline.site_index)
    assert torch.equal(mapped_atom[order], baseline.atom_index)
    torch.testing.assert_close(
        permuted.distances[order], baseline.distances, atol=2e-15, rtol=0
    )
    assert inverse.shape == (2,)


def test_structured_geometry_and_boundary_failures():
    positions, references, cell = _geometry()
    kwargs = dict(
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_support("blocked"),
    )
    with pytest.raises(TransportSupportError) as failure:
        build_periodic_compact_transport_edges(
            positions, references, cell, (True, True, False), **kwargs
        )
    assert failure.value.reason_code == "UNSUPPORTED_PBC"
    with pytest.raises(TransportSupportError) as failure:
        build_periodic_compact_transport_edges(
            positions, references, torch.zeros_like(cell), (True,) * 3, **kwargs
        )
    assert failure.value.reason_code == "SINGULAR_CELL"
    with pytest.raises(TransportSupportError) as failure:
        build_periodic_compact_transport_edges(
            positions * torch.nan, references, cell, (True,) * 3, **kwargs
        )
    assert failure.value.reason_code == "NONFINITE_SUPPORT_GEOMETRY"

    tie_positions = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
    tie_references = torch.zeros((1, 3), dtype=torch.float64)
    with pytest.raises(TransportSupportError) as failure:
        build_periodic_compact_transport_edges(
            tie_positions,
            tie_references,
            torch.eye(3, dtype=torch.float64),
            (True,) * 3,
            epsilon_ot=0.5,
            ell_ot=1.5,
            config=TransportSupportConfig(
                "compact_c2",
                0.75,
                0.25,
                0.25,
                backend="edge_list",
                candidate_backend="blocked",
            ),
            minimum_mic_image_gap=0.0,
        )
    assert failure.value.reason_code == "MIC_AMBIGUITY"

    boundary_positions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(TransportSupportError) as failure:
        build_periodic_compact_transport_edges(
            boundary_positions,
            tie_references,
            torch.eye(3, dtype=torch.float64) * 10.0,
            (True,) * 3,
            epsilon_ot=0.5,
            ell_ot=1.5,
            config=TransportSupportConfig(
                "compact_c2",
                0.75,
                0.25,
                0.25,
                backend="edge_list",
                candidate_backend="blocked",
            ),
            minimum_candidate_boundary_gap=0.0,
        )
    assert failure.value.reason_code == "CANDIDATE_BOUNDARY_INSTABILITY"


def test_blocked_path_does_not_call_dense_candidate_or_support(monkeypatch):
    import refsite_mlip.transport.cost as cost_module
    import refsite_mlip.transport.edge_list as edge_module
    import refsite_mlip.transport.factory as factory_module
    import refsite_mlip.transport.problem as problem_module
    import refsite_mlip.transport.support as support_module

    def forbidden(*args, **kwargs):
        raise AssertionError("dense candidate/support path was called")

    monkeypatch.setattr(cost_module, "atom_site_displacements", forbidden)
    monkeypatch.setattr(edge_module, "build_compact_transport_edges", forbidden)
    monkeypatch.setattr(edge_module, "materialize_dense_plan", forbidden)
    monkeypatch.setattr(factory_module, "solve_atom_vacancy_ot", forbidden)
    monkeypatch.setattr(problem_module, "build_ot_problem", forbidden)
    monkeypatch.setattr(support_module, "validate_compact_support", forbidden)
    edges = _edges(_support("blocked", 1, 1))
    assert edges.support_diagnostics.dense_candidate_allocation_observed is False
    assert edges.support_diagnostics.maximum_pair_block_elements == 1
    assert torch.isfinite(
        solve_sparse_sinkhorn_train_fixed(
            edges, TrainSinkhornConfig(32)
        ).edge_plan
    ).all()
    adaptive = solve_sparse_hybrid_eval(
        edges,
        EvalOTConfig(
            sinkhorn_iterations=8,
            convergence_tolerance=1.0e-11,
            fallback_sinkhorn_iterations=4096,
        ),
    )
    assert torch.isfinite(adaptive.edge_plan).all()
    assert not adaptive.dense_plan_materialized


def test_sparse_matching_and_implicit_vacancy_total_support_match_dense_oracle():
    generator = torch.Generator().manual_seed(8123)
    config = TransportSupportConfig(
        "compact_c2",
        1.0,
        0.25,
        3.0,
        backend="edge_list",
        candidate_backend="blocked",
    )
    for num_sites, num_atoms in ((3, 3), (3, 2), (4, 2), (4, 1), (4, 0)):
        for _ in range(60):
            mask = torch.rand(
                (num_sites, num_atoms),
                generator=generator,
                dtype=torch.float64,
            ) < 0.45
            distances = torch.where(
                mask,
                torch.full(mask.shape, 0.5, dtype=torch.float64),
                torch.full(mask.shape, 2.0, dtype=torch.float64),
            )
            switch = compact_c2_switch(distances, config)
            dense_failure = None
            try:
                _, dense_diagnostics = validate_compact_support(
                    distances, switch, config
                )
            except TransportSupportError as error:
                dense_failure = error.reason_code
            pairs = torch.nonzero(
                distances < config.r_candidate, as_tuple=False
            )
            site_index = pairs[:, 0].long()
            atom_index = pairs[:, 1].long()
            sparse_failure = None
            try:
                _, sparse_diagnostics = validate_compact_support_edges(
                    site_index,
                    atom_index,
                    distances[site_index, atom_index],
                    switch[site_index, atom_index],
                    num_sites,
                    num_atoms,
                    config,
                )
            except TransportSupportError as error:
                sparse_failure = error.reason_code
            assert sparse_failure == dense_failure
            if dense_failure is None:
                assert sparse_diagnostics.maximum_atom_matching_size == dense_diagnostics.maximum_atom_matching_size
                assert sparse_diagnostics.total_matching_size == dense_diagnostics.total_matching_size
                assert sparse_diagnostics.total_support_feasible == dense_diagnostics.total_support_feasible


def _periodic_grid(size, *, dtype, device):
    axis = torch.arange(size, dtype=dtype, device=device) * 2.0
    mesh = torch.cartesian_prod(axis, axis, axis)
    references = mesh.reshape(-1, 3)
    positions = references[:-1] + references.new_tensor([0.07, -0.04, 0.03])
    cell = torch.eye(3, dtype=dtype, device=device) * (2.0 * size)
    return positions, references, cell


@pytest.mark.parametrize("size", [4, 6])
def test_synthetic_64_216_bounded_shape_and_fixed_solve(size):
    positions, references, cell = _periodic_grid(
        size, dtype=torch.float64, device="cpu"
    )
    config = TransportSupportConfig(
        "compact_c2",
        2.6,
        0.4,
        0.3,
        backend="edge_list",
        candidate_backend="blocked",
        site_block_size=8,
        atom_block_size=16,
    )
    edges = build_periodic_compact_transport_edges(
        positions,
        references,
        cell,
        (True,) * 3,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=config,
    )
    diagnostics = edges.support_diagnostics
    assert diagnostics.num_sites == size**3
    assert diagnostics.num_atoms == size**3 - 1
    assert diagnostics.maximum_pair_block_elements <= 8 * 16
    assert diagnostics.maximum_pair_block_elements < diagnostics.theoretical_full_pair_elements
    assert not diagnostics.dense_candidate_allocation_observed
    result = solve_sparse_sinkhorn_train_fixed(edges, TrainSinkhornConfig(64))
    assert torch.isfinite(result.edge_plan).all() and torch.isfinite(result.q).all()
    assert result.q.shape == (size**3,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_blocked_cuda_candidate_fixed_adaptive(dtype):
    positions, references, cell = _geometry(dtype=dtype, device="cuda")
    config = _support("blocked", 2, 1)
    edges = build_periodic_compact_transport_edges(
        positions,
        references,
        cell,
        (True,) * 3,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=config,
    )
    fixed = solve_sparse_sinkhorn_train_fixed(edges, TrainSinkhornConfig(256))
    adaptive = solve_sparse_hybrid_eval(
        edges,
        EvalOTConfig(
            sinkhorn_iterations=16,
            convergence_tolerance=1.0e-6 if dtype == torch.float32 else 1.0e-11,
            fallback_sinkhorn_iterations=4096,
        ),
    )
    assert edges.distances.device.type == "cuda" and edges.distances.dtype == dtype
    assert torch.isfinite(fixed.edge_plan).all() and torch.isfinite(adaptive.edge_plan).all()
    assert not edges.support_diagnostics.dense_candidate_allocation_observed
