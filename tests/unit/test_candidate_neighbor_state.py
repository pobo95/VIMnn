from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import refsite_mlip.transport.candidate_state as state_module
from refsite_mlip.transport import (
    CANDIDATE_NEIGHBOR_STATE_SCHEMA_VERSION,
    EvalOTConfig,
    TrainSinkhornConfig,
    TransportSupportConfig,
    TransportSupportError,
    build_candidate_neighbor_state,
    build_periodic_compact_transport_edges,
    materialize_dense_plan,
    solve_sparse_hybrid_eval,
    solve_sparse_sinkhorn_train_fixed,
    update_candidate_neighbor_state,
)


def _config(*, site_block=1, atom_block=1):
    return TransportSupportConfig(
        kind="compact_c2",
        cutoff=2.0,
        switch_width=0.5,
        candidate_skin=0.2,
        backend="edge_list",
        candidate_backend="blocked",
        site_block_size=site_block,
        atom_block_size=atom_block,
    )


def _geometry(dtype=torch.float64, device="cpu"):
    positions = torch.tensor(
        [[0.0, 0.08, 0.03], [0.5, -0.04, 0.02]],
        dtype=dtype,
        device=device,
    )
    references = torch.tensor(
        [[0.1, 0.0, 0.0], [0.4, 0.0, 0.0], [2.15, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    cell = torch.tensor(
        [[8.0, 0.15, -0.05], [0.2, 7.7, 0.1], [-0.1, 0.25, 8.2]],
        dtype=dtype,
        device=device,
    )
    origin = torch.zeros(3, dtype=dtype, device=device)
    numbers = torch.tensor([6, 41], dtype=torch.long, device=device)
    identities = torch.tensor([71, 93], dtype=torch.long, device=device)
    return positions, references, cell, origin, numbers, identities


def _arguments(dtype=torch.float64, device="cpu", *, config=None):
    positions, references, cell, origin, numbers, identities = _geometry(
        dtype, device
    )
    return dict(
        positions=positions,
        reference_sites=references,
        cell=cell,
        pbc=(True, True, True),
        origin=origin,
        atomic_numbers=numbers,
        atom_order_identity=identities,
        template_fingerprint="template-content-fingerprint-a",
        phase_site_branch_fingerprint="phase-group-0",
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_config() if config is None else config,
        sample_id="candidate-state-sample",
    )


def _fresh(arguments):
    return build_periodic_compact_transport_edges(
        arguments["positions"],
        arguments["reference_sites"],
        arguments["cell"],
        arguments["pbc"],
        origin=arguments["origin"],
        epsilon_ot=arguments["epsilon_ot"],
        ell_ot=arguments["ell_ot"],
        config=arguments["config"],
        template_id=arguments["template_fingerprint"],
        sample_id=arguments["sample_id"],
    )


def _assert_live_edge_equal(left, right):
    for name in (
        "site_index",
        "atom_index",
        "periodic_shift",
        "displacements",
        "distances",
        "switch",
        "log_kernel",
        "active",
        "site_ptr",
        "atom_ptr",
        "atom_major_permutation",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name)), name


def _adaptive(edges):
    tolerance = 1.0e-6 if edges.distances.dtype == torch.float32 else 1.0e-11
    return solve_sparse_hybrid_eval(
        edges,
        EvalOTConfig(
            sinkhorn_iterations=16,
            max_newton_iterations=20,
            convergence_tolerance=tolerance,
            fallback_sinkhorn_iterations=4096,
        ),
    )


def test_initial_build_owns_immutable_snapshot_and_zero_motion_reuses():
    arguments = _arguments()
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in arguments.items()
    }
    built = build_candidate_neighbor_state(**arguments)
    via_update = update_candidate_neighbor_state(None, **arguments)
    assert via_update.decision.reason_code == "INITIAL_BUILD"
    _assert_live_edge_equal(via_update.edges, built.edges)
    state = built.state
    assert built.decision.rebuilt and not built.decision.reused
    assert built.decision.reason_code == "INITIAL_BUILD"
    assert built.decision.build_count == 1
    assert state.schema_version == CANDIDATE_NEIGHBOR_STATE_SCHEMA_VERSION
    assert state.build_generation == 1 and state.reuse_count == 0
    assert state.integrity_fingerprint
    assert not state.build_positions.requires_grad
    assert state.build_positions.data_ptr() != arguments["positions"].data_ptr()
    assert state.build_reference_sites.data_ptr() != arguments["reference_sites"].data_ptr()
    assert not built.decision.dense_allocation_observed

    reused = update_candidate_neighbor_state(state, **arguments)
    assert reused.decision.reused and not reused.decision.rebuilt
    assert reused.decision.reason_code == "REUSED"
    assert reused.decision.build_generation == 1
    assert reused.decision.reuse_count == 1
    assert reused.state is not state
    assert torch.equal(reused.state.build_positions, state.build_positions)
    assert torch.equal(reused.state.build_reference_sites, state.build_reference_sites)
    assert reused.decision.delta_pair_bound == 0.0
    assert reused.decision.remaining_skin > 0.0
    assert reused.decision.numerical_guard > 0.0
    assert "torch.finfo" not in reused.decision.numerical_guard_formula
    _assert_live_edge_equal(reused.edges, _fresh(arguments))
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(arguments[name], value), name


def test_small_combined_motion_block_size_translation_and_wrapping_reuse():
    arguments = _arguments()
    built = build_candidate_neighbor_state(**arguments)
    moved = dict(arguments)
    atom_shift = arguments["positions"].new_tensor([0.035, -0.01, 0.005])
    site_shift = arguments["positions"].new_tensor([-0.025, 0.005, 0.0])
    moved["positions"] = arguments["positions"] + atom_shift
    moved["reference_sites"] = arguments["reference_sites"] + site_shift
    moved["config"] = replace(
        arguments["config"], site_block_size=3, atom_block_size=2
    )
    reused = update_candidate_neighbor_state(built.state, **moved)
    assert reused.decision.reused
    assert reused.decision.delta_atom == pytest.approx(float(atom_shift.norm()))
    assert reused.decision.delta_site == pytest.approx(float(site_shift.norm()))
    assert reused.state.support_content_fingerprint == built.state.support_content_fingerprint
    assert reused.state.candidate_pair_set_fingerprint == built.state.candidate_pair_set_fingerprint

    translation = arguments["positions"].new_tensor([0.31, -0.27, 0.19])
    translated = dict(arguments)
    translated["positions"] = arguments["positions"] + translation
    translated["reference_sites"] = arguments["reference_sites"] + translation
    translated["origin"] = arguments["origin"] + translation
    translated_result = update_candidate_neighbor_state(built.state, **translated)
    assert translated_result.decision.reused
    assert translated_result.decision.delta_pair_bound == pytest.approx(0.0, abs=2e-15)
    assert (
        translated_result.decision.current_live_support_fingerprint
        == built.decision.current_live_support_fingerprint
    )
    torch.testing.assert_close(
        translated_result.edges.displacements,
        built.edges.displacements,
        atol=3e-15,
        rtol=0,
    )

    wrapped = dict(arguments)
    wrapped["positions"] = arguments["positions"].clone()
    wrapped["positions"][0] += arguments["cell"][1]
    wrapped["reference_sites"] = arguments["reference_sites"].clone()
    wrapped["reference_sites"][1] -= arguments["cell"][0]
    wrapped_result = update_candidate_neighbor_state(built.state, **wrapped)
    assert wrapped_result.decision.reused
    assert wrapped_result.decision.delta_pair_bound == pytest.approx(0.0, abs=3e-15)
    assert (
        wrapped_result.decision.current_live_support_fingerprint
        == built.decision.current_live_support_fingerprint
    )
    torch.testing.assert_close(
        wrapped_result.edges.distances,
        built.edges.distances,
        atol=4e-15,
        rtol=0,
    )
    assert not torch.equal(
        wrapped_result.edges.periodic_shift, built.edges.periodic_shift
    )


def test_cumulative_skin_uses_original_build_snapshot_and_rebuilds():
    arguments = _arguments()
    built = build_candidate_neighbor_state(**arguments)
    first_arguments = dict(arguments)
    first_arguments["positions"] = arguments["positions"].clone()
    first_arguments["positions"][:, 0] += 0.08
    first = update_candidate_neighbor_state(built.state, **first_arguments)
    assert first.decision.reused
    assert torch.equal(first.state.build_positions, built.state.build_positions)

    second_arguments = dict(arguments)
    second_arguments["positions"] = arguments["positions"].clone()
    second_arguments["positions"][:, 0] += 0.115
    second_arguments["reference_sites"] = arguments["reference_sites"].clone()
    second_arguments["reference_sites"][:, 0] -= 0.09
    second = update_candidate_neighbor_state(first.state, **second_arguments)
    assert second.decision.rebuilt and not second.decision.reused
    assert second.decision.reason_code == "SKIN_EXHAUSTED"
    assert second.decision.delta_pair_bound == pytest.approx(0.205)
    assert second.decision.build_generation == 2
    assert second.decision.reuse_count == 0
    assert torch.equal(second.state.build_positions, second_arguments["positions"])

    boundary_arguments = dict(arguments)
    boundary_arguments["positions"] = arguments["positions"].clone()
    boundary_arguments["positions"][:, 0] += arguments["config"].candidate_skin
    boundary = update_candidate_neighbor_state(built.state, **boundary_arguments)
    assert boundary.decision.reason_code == "SKIN_EXHAUSTED"
    assert boundary.decision.rebuilt


@pytest.mark.parametrize(
    "change, reason",
    [
        ("cell", "CELL_CHANGED"),
        ("atom_order", "ATOM_ORDER_CHANGED"),
        ("species", "ATOM_ORDER_CHANGED"),
        ("template", "TEMPLATE_MISMATCH"),
        ("support", "SUPPORT_CONFIG_MISMATCH"),
        ("phase", "PHASE_SITE_BRANCH_CHANGED"),
        ("explicit", "EXPLICIT_REBUILD"),
    ],
)
def test_compatibility_and_explicit_rebuild_reasons(change, reason):
    arguments = _arguments()
    built = build_candidate_neighbor_state(**arguments)
    current = dict(arguments)
    if change == "cell":
        current["cell"] = arguments["cell"].clone()
        current["cell"][0, 0] += 1.0e-8
    elif change == "atom_order":
        current["atom_order_identity"] = arguments["atom_order_identity"].flip(0)
    elif change == "species":
        current["atomic_numbers"] = arguments["atomic_numbers"].flip(0)
    elif change == "template":
        current["template_fingerprint"] = "template-content-fingerprint-b"
    elif change == "support":
        current["config"] = replace(
            arguments["config"], cutoff=1.95, candidate_skin=0.25
        )
    elif change == "phase":
        current["phase_site_branch_fingerprint"] = "phase-group-1"
    elif change == "explicit":
        current["explicit_rebuild"] = True
    result = update_candidate_neighbor_state(built.state, **current)
    assert result.decision.rebuilt
    assert result.decision.reason_code == reason
    assert result.state.build_generation == 2


def test_atom_count_change_and_dtype_materialization_are_explicit():
    arguments = _arguments()
    built = build_candidate_neighbor_state(**arguments)
    reduced = dict(arguments)
    reduced["positions"] = arguments["positions"][:1]
    reduced["reference_sites"] = arguments["reference_sites"][:2]
    reduced["atomic_numbers"] = arguments["atomic_numbers"][:1]
    reduced["atom_order_identity"] = arguments["atom_order_identity"][:1]
    changed = update_candidate_neighbor_state(built.state, **reduced)
    assert changed.decision.reason_code == "ATOM_COUNT_CHANGED"
    assert changed.state.num_atoms == 1 and changed.state.num_sites == 2

    converted = {
        name: value.to(dtype=torch.float32)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in arguments.items()
    }
    materialized = update_candidate_neighbor_state(built.state, **converted)
    assert materialized.decision.reused
    assert materialized.decision.reason_code == "STATE_DEVICE_MATERIALIZATION"
    assert materialized.state.dtype == torch.float32
    assert materialized.state.site_index.dtype == torch.long
    assert materialized.decision.state_materialized


def test_state_integrity_schema_and_guard_fail_fast_before_geometry():
    arguments = _arguments()
    built = build_candidate_neighbor_state(**arguments)
    built.state.build_positions[0, 0] += 0.125
    with pytest.raises(TransportSupportError) as failure:
        update_candidate_neighbor_state(built.state, **arguments)
    assert failure.value.reason_code == "STATE_INTEGRITY_MISMATCH"

    state = build_candidate_neighbor_state(**arguments).state
    object.__setattr__(state, "integrity_fingerprint", "0" * 64)
    with pytest.raises(TransportSupportError) as failure:
        update_candidate_neighbor_state(state, **arguments)
    assert failure.value.reason_code == "STATE_INTEGRITY_MISMATCH"

    state = build_candidate_neighbor_state(**arguments).state
    object.__setattr__(state, "schema_version", "corrupt-schema")
    with pytest.raises(TransportSupportError) as failure:
        update_candidate_neighbor_state(state, **arguments)
    assert failure.value.reason_code == "STATE_SCHEMA_MISMATCH"

    with pytest.raises(TransportSupportError) as failure:
        build_candidate_neighbor_state(
            **arguments, numerical_guard=arguments["config"].candidate_skin
        )
    assert failure.value.reason_code == "INVALID_SUPPORT_CONFIG"


def test_mic_and_candidate_certificates_are_enforced_on_reuse():
    arguments = _arguments()
    built = build_candidate_neighbor_state(**arguments)
    build_gap = built.state.build_diagnostics.candidate_boundary_gap
    certified = update_candidate_neighbor_state(
        built.state,
        **arguments,
        minimum_candidate_boundary_gap=0.5 * build_gap,
    )
    assert certified.decision.reused
    assert certified.decision.candidate_boundary_lower_bound > 0.5 * build_gap

    with pytest.raises(TransportSupportError) as failure:
        update_candidate_neighbor_state(
            built.state,
            **arguments,
            minimum_candidate_boundary_gap=1.1 * build_gap,
        )
    assert failure.value.reason_code == "CANDIDATE_BOUNDARY_INSTABILITY"

    with pytest.raises(TransportSupportError) as failure:
        update_candidate_neighbor_state(
            built.state,
            **arguments,
            minimum_mic_image_gap=1.0e6,
        )
    assert failure.value.reason_code == "MIC_AMBIGUITY"


def test_adversarial_skin_bound_prevents_missing_new_active_pair():
    arguments = _arguments()
    arguments["reference_sites"] = arguments["reference_sites"].clone()
    arguments["reference_sites"][2] = arguments["reference_sites"].new_tensor(
        [2.201, 0.0, 0.0]
    )
    built = build_candidate_neighbor_state(**arguments)
    missing = (built.state.site_index == 2) & (built.state.atom_index == 0)
    assert not bool(torch.any(missing))

    below = dict(arguments)
    below["positions"] = arguments["positions"].clone()
    below["positions"][0, 0] += 0.199
    reused = update_candidate_neighbor_state(built.state, **below)
    assert reused.decision.reused
    assert reused.decision.remaining_skin > 0.0
    assert reused.decision.fresh_candidate_fingerprint is None
    fresh_below_update = build_candidate_neighbor_state(**below)
    fresh_below = _fresh(below)
    new_candidate = (fresh_below.site_index == 2) & (fresh_below.atom_index == 0)
    assert bool(torch.any(new_candidate))
    assert not bool(torch.any(fresh_below.active[new_candidate]))
    assert (
        reused.decision.current_live_support_fingerprint
        == fresh_below_update.decision.current_live_support_fingerprint
    )
    assert (
        reused.decision.cached_pair_set_fingerprint
        != fresh_below_update.decision.cached_pair_set_fingerprint
    )

    cached_fixed = solve_sparse_sinkhorn_train_fixed(
        reused.edges, TrainSinkhornConfig(256)
    )
    fresh_fixed = solve_sparse_sinkhorn_train_fixed(
        fresh_below, TrainSinkhornConfig(256)
    )
    torch.testing.assert_close(
        materialize_dense_plan(cached_fixed).plan,
        materialize_dense_plan(fresh_fixed).plan,
        atol=2e-14,
        rtol=2e-14,
    )
    torch.testing.assert_close(cached_fixed.q, fresh_fixed.q, atol=2e-14, rtol=2e-14)
    cached_adaptive = _adaptive(reused.edges)
    fresh_adaptive = _adaptive(fresh_below)
    torch.testing.assert_close(
        materialize_dense_plan(cached_adaptive).plan,
        materialize_dense_plan(fresh_adaptive).plan,
        atol=3e-13,
        rtol=3e-13,
    )
    torch.testing.assert_close(
        cached_adaptive.q, fresh_adaptive.q, atol=3e-13, rtol=3e-13
    )

    exhausted = dict(arguments)
    exhausted["positions"] = arguments["positions"].clone()
    exhausted["positions"][0, 0] += 0.205
    rebuilt = update_candidate_neighbor_state(built.state, **exhausted)
    assert rebuilt.decision.rebuilt
    assert rebuilt.decision.reason_code == "SKIN_EXHAUSTED"
    assert rebuilt.decision.cached_candidate_count == built.state.candidate_count
    assert (
        rebuilt.decision.fresh_candidate_fingerprint
        == rebuilt.state.build_candidate_fingerprint
    )
    included = (rebuilt.state.site_index == 2) & (rebuilt.state.atom_index == 0)
    assert bool(torch.any(included))
    assert bool(torch.any(rebuilt.edges.active[included]))
    _assert_live_edge_equal(rebuilt.edges, _fresh(exhausted))


def _reuse_observable(positions, state, arguments, *, adaptive=False):
    current = dict(arguments)
    current["positions"] = positions
    edges = update_candidate_neighbor_state(state, **current).edges
    result = _adaptive(edges) if adaptive else solve_sparse_sinkhorn_train_fixed(
        edges, TrainSinkhornConfig(96)
    )
    weights = torch.linspace(
        -0.3,
        0.4,
        result.edge_plan.numel(),
        dtype=positions.dtype,
        device=positions.device,
    )
    return (weights * result.edge_plan).sum() + 0.17 * result.q.square().sum()


def test_reuse_fixed_gradcheck_gradgradcheck_and_adaptive_first_derivative():
    arguments = _arguments()
    state = build_candidate_neighbor_state(**arguments).state
    positions = arguments["positions"].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda value: _reuse_observable(value, state, arguments),
        (positions,),
        eps=1.0e-6,
        atol=8.0e-6,
        rtol=8.0e-5,
    )
    assert torch.autograd.gradgradcheck(
        lambda value: _reuse_observable(value, state, arguments),
        (positions,),
        eps=1.0e-6,
        atol=3.0e-5,
        rtol=3.0e-4,
    )
    energy = _reuse_observable(positions, state, arguments, adaptive=True)
    gradient = torch.autograd.grad(energy, positions)[0]
    step = 1.0e-6
    delta = torch.zeros_like(positions)
    delta[1, 0] = step
    finite = (
        _reuse_observable(
            positions.detach() + delta, state, arguments, adaptive=True
        )
        - _reuse_observable(
            positions.detach() - delta, state, arguments, adaptive=True
        )
    ) / (2.0 * step)
    torch.testing.assert_close(gradient[1, 0], finite, atol=5e-6, rtol=5e-5)


def test_reuse_does_not_traverse_or_densify_and_retry_is_recorded(monkeypatch):
    arguments = _arguments()
    built = build_candidate_neighbor_state(**arguments)

    def forbidden(*args, **kwargs):
        raise AssertionError("reuse called the full blocked traversal")

    monkeypatch.setattr(
        state_module, "build_periodic_compact_transport_edges", forbidden
    )
    reused = update_candidate_neighbor_state(built.state, **arguments)
    assert reused.decision.reused
    assert reused.decision.processed_block_count == 0
    assert reused.decision.maximum_pair_block_elements == 0
    assert not reused.decision.dense_allocation_observed
    assert torch.isfinite(
        solve_sparse_sinkhorn_train_fixed(
            reused.edges, TrainSinkhornConfig(32)
        ).edge_plan
    ).all()
    assert torch.isfinite(_adaptive(reused.edges).edge_plan).all()

    monkeypatch.undo()
    original = state_module._materialize_cached_edges

    def forced_cached_failure(*args, **kwargs):
        raise TransportSupportError("NO_TOTAL_SUPPORT", "forced cached failure")

    monkeypatch.setattr(
        state_module, "_materialize_cached_edges", forced_cached_failure
    )
    retry = update_candidate_neighbor_state(built.state, **arguments)
    assert retry.decision.rebuilt
    assert retry.decision.reason_code == "REUSE_FEASIBILITY_RETRY"
    assert retry.decision.fresh_retry_performed
    assert retry.decision.fresh_retry_reason == "NO_TOTAL_SUPPORT"
    assert original is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_candidate_state_cuda_build_reuse_rebuild_and_sparse_solve(dtype):
    arguments = _arguments(dtype=dtype, device="cuda")
    built = build_candidate_neighbor_state(**arguments)
    moved = dict(arguments)
    moved["positions"] = arguments["positions"].clone()
    moved["positions"][:, 0] += 0.05
    reused = update_candidate_neighbor_state(built.state, **moved)
    assert reused.decision.reused
    changed = dict(moved)
    changed["cell"] = arguments["cell"].clone()
    changed["cell"][0, 0] += 1.0e-4
    rebuilt = update_candidate_neighbor_state(reused.state, **changed)
    assert rebuilt.decision.reason_code == "CELL_CHANGED"
    for result in (
        solve_sparse_sinkhorn_train_fixed(reused.edges, TrainSinkhornConfig(256)),
        _adaptive(reused.edges),
    ):
        assert result.edge_plan.device.type == "cuda"
        assert result.edge_plan.dtype == dtype
        assert torch.isfinite(result.edge_plan).all()
        assert torch.isfinite(result.q).all()
        assert not result.dense_plan_materialized
    cpu = built.state.to(device="cpu", dtype=torch.float64)
    assert cpu.device.type == "cpu" and cpu.dtype == torch.float64
    cpu.validate_integrity()
