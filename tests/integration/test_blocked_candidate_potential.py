from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import refsite_mlip.models.potential as potential_module
from refsite_mlip.models import ReferenceSitePotential, evaluate_structure_batch
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TransportSupportConfig,
    TransportSupportError,
    sparse_support_fingerprint,
)

from test_compact_support_potential import _model, _numbers
from test_edge_list_adaptive_grouped import _case as _grouped_case
from test_edge_list_adaptive_potential import (
    _case as _direct_adaptive_case,
    _symmetric_directions,
)
from test_edge_list_compact_potential import _edge_support


def _configured(model, support):
    configured = ReferenceSitePotential(
        replace(model.config, transport_support=support),
        model.topology,
        model.phase_modes,
        model.phase_mode_weights,
        model.species_alignment_weights,
        model.site_alignment_weights,
        model.phase_channel_weights,
        model.atomic_baseline,
    ).to(model.atomic_baseline)
    configured.load_state_dict(model.state_dict(), strict=True)
    return configured


def _blocked_support(support, site_block_size=2, atom_block_size=2):
    return replace(
        support,
        candidate_backend="blocked",
        site_block_size=site_block_size,
        atom_block_size=atom_block_size,
    )


def _blocked_clone(model, site_block_size=2, atom_block_size=2):
    return _configured(
        model,
        _blocked_support(
            model.config.transport_support,
            site_block_size,
            atom_block_size,
        ),
    )


def _assert_sparse_transport_close(left, right, tolerance=3.0e-14):
    assert torch.equal(left.edges.site_index, right.edges.site_index)
    assert torch.equal(left.edges.atom_index, right.edges.atom_index)
    assert torch.equal(left.edges.active, right.edges.active)
    assert sparse_support_fingerprint(left.edges) == sparse_support_fingerprint(
        right.edges
    )
    torch.testing.assert_close(
        left.edges.displacements,
        right.edges.displacements,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        left.edge_plan, right.edge_plan, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(left.q, right.q, atol=tolerance, rtol=tolerance)


def _eval_branch(output):
    diagnostics = output.auxiliary["evaluation_diagnostics"]
    support = output.auxiliary["transport_support"]
    return (
        diagnostics.selected_grouped_index,
        diagnostics.transport_support_fingerprint,
        support.candidate_fingerprint,
        support.active_edge_count,
        diagnostics.transport_fallback_used,
    )


def test_blocked_direct_train_fixed_parity_diagnostics_and_double_backward(
    typed_crystal,
):
    dense_candidate = _model(typed_crystal, _edge_support())
    blocked = _blocked_clone(dense_candidate, 2, 3)
    other_blocks = _blocked_clone(dense_candidate, 1, 1)
    numbers = _numbers(typed_crystal)

    dense_positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    blocked_positions = (
        typed_crystal["positions"][:5].clone().requires_grad_(True)
    )
    keywords = dict(
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    dense_output = dense_candidate(
        dense_positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        **keywords,
    )
    blocked_output = blocked(
        blocked_positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        **keywords,
    )
    _assert_sparse_transport_close(
        dense_output.auxiliary["ot"], blocked_output.auxiliary["ot"]
    )
    for name, tolerance in (
        ("raw_c", 4.0e-14),
        ("site_features", 8.0e-14),
        ("site_energy", 8.0e-14),
        ("energy", 8.0e-14),
        ("forces", 8.0e-13),
        ("stress", 8.0e-13),
        ("stress_voigt", 8.0e-13),
    ):
        torch.testing.assert_close(
            getattr(blocked_output, name),
            getattr(dense_output, name),
            atol=tolerance,
            rtol=tolerance,
        )

    diagnostics = blocked_output.auxiliary["transport_support"]
    assert diagnostics.candidate_backend == "blocked"
    assert diagnostics.site_block_size == 2
    assert diagnostics.atom_block_size == 3
    assert diagnostics.processed_block_count == 6
    assert diagnostics.maximum_pair_block_elements <= 6
    assert diagnostics.theoretical_full_pair_elements == 30
    assert diagnostics.peak_temporary_geometry_elements > 0
    assert not diagnostics.dense_candidate_allocation_observed
    assert diagnostics.candidate_fingerprint
    assert not blocked_output.auxiliary["ot"].dense_plan_materialized

    block_size_output = other_blocks(
        typed_crystal["positions"][:5],
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        return_aux=True,
    )
    assert (
        block_size_output.auxiliary["transport_support"].candidate_fingerprint
        == diagnostics.candidate_fingerprint
    )
    assert torch.equal(
        block_size_output.auxiliary["ot"].edge_plan,
        blocked(
            typed_crystal["positions"][:5],
            numbers,
            typed_crystal["cell"],
            typed_crystal["origin"],
            return_aux=True,
        ).auxiliary["ot"].edge_plan,
    )

    dense_parameters = (
        dense_candidate.readout.mlp[-1].weight,
        dense_candidate.layers[0].corr.C2_product.weight,
        dense_candidate.central.embedding.weight,
    )
    blocked_parameters = (
        blocked.readout.mlp[-1].weight,
        blocked.layers[0].corr.C2_product.weight,
        blocked.central.embedding.weight,
    )
    dense_loss = (
        dense_output.energy
        + 0.03 * dense_output.forces.square().sum()
        + 0.01 * dense_output.stress.square().sum()
    )
    blocked_loss = (
        blocked_output.energy
        + 0.03 * blocked_output.forces.square().sum()
        + 0.01 * blocked_output.stress.square().sum()
    )
    dense_gradients = torch.autograd.grad(dense_loss, dense_parameters)
    blocked_gradients = torch.autograd.grad(blocked_loss, blocked_parameters)
    for dense_gradient, blocked_gradient in zip(
        dense_gradients, blocked_gradients
    ):
        assert torch.isfinite(blocked_gradient).all()
        torch.testing.assert_close(
            blocked_gradient,
            dense_gradient,
            atol=3.0e-11,
            rtol=3.0e-10,
        )


def test_blocked_direct_adaptive_parity_fd_and_single_autograd_call(
    typed_crystal, monkeypatch
):
    _, dense_candidate, _, _, arguments, keywords = _direct_adaptive_case(
        typed_crystal,
        5,
        template_id="blocked-direct-adaptive",
        warmup=32,
    )
    blocked = _blocked_clone(dense_candidate, 2, 2)
    dense_output = dense_candidate(
        *arguments,
        **keywords,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )

    original_grad = torch.autograd.grad
    calls = []

    def counted_grad(*args, **kwargs):
        calls.append(1)
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    with torch.no_grad():
        blocked_output = blocked(
            *arguments,
            **keywords,
            compute_forces=True,
            compute_stress=True,
            return_aux=True,
        )
    monkeypatch.setattr(torch.autograd, "grad", original_grad)
    assert len(calls) == 1
    _assert_sparse_transport_close(
        dense_output.auxiliary["ot"],
        blocked_output.auxiliary["ot"],
        tolerance=8.0e-14,
    )
    assert torch.equal(
        blocked_output.auxiliary["phase"], dense_output.auxiliary["phase"]
    )
    for name, tolerance in (
        ("raw_c", 8.0e-14),
        ("site_features", 2.0e-13),
        ("site_energy", 2.0e-13),
        ("energy", 2.0e-13),
        ("forces", 3.0e-11),
        ("stress", 3.0e-11),
    ):
        torch.testing.assert_close(
            getattr(blocked_output, name),
            getattr(dense_output, name),
            atol=tolerance,
            rtol=tolerance,
        )
    diagnostics = blocked_output.auxiliary["evaluation_diagnostics"]
    assert diagnostics.transport_candidate_backend == "blocked"
    assert diagnostics.transport_site_block_size == 2
    assert diagnostics.transport_atom_block_size == 2
    assert diagnostics.transport_processed_block_count == 9
    assert not diagnostics.transport_dense_candidate_allocation_observed
    assert diagnostics.transport_candidate_fingerprint
    assert diagnostics.transport_mic_image_gap > 0.0
    assert not diagnostics.transport_fallback_used
    assert diagnostics.differentiability_scope == "selected_branch_first_order"
    assert blocked_output.forces.grad_fn is None
    assert blocked_output.stress.grad_fn is None

    other_blocks = _blocked_clone(dense_candidate, 1, 3)
    other_output = other_blocks(*arguments, **keywords, return_aux=True)
    assert (
        other_output.auxiliary["transport_support"].candidate_fingerprint
        == diagnostics.transport_candidate_fingerprint
    )
    assert torch.equal(
        other_output.auxiliary["ot"].edge_plan,
        blocked_output.auxiliary["ot"].edge_plan,
    )
    assert torch.equal(other_output.auxiliary["ot"].q, blocked_output.auxiliary["ot"].q)
    assert torch.equal(other_output.energy, blocked_output.energy)

    positions, numbers, cell, origin = arguments
    force_step = 1.0e-6
    delta = torch.zeros_like(positions)
    delta[2, 1] = force_step
    plus = blocked(
        positions + delta, numbers, cell, origin, **keywords, return_aux=True
    )
    minus = blocked(
        positions - delta, numbers, cell, origin, **keywords, return_aux=True
    )
    assert _eval_branch(plus) == _eval_branch(blocked_output) == _eval_branch(minus)
    force_fd = -(plus.energy - minus.energy) / (2.0 * force_step)
    torch.testing.assert_close(
        blocked_output.forces[2, 1], force_fd, atol=5.0e-6, rtol=5.0e-4
    )

    stress_step = 1.0e-4
    volume = torch.linalg.det(cell).abs()
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
    for direction in _symmetric_directions(positions):
        plus_deformation = identity + stress_step * direction
        minus_deformation = identity - stress_step * direction
        plus = blocked(
            positions @ plus_deformation,
            numbers,
            cell @ plus_deformation,
            origin @ plus_deformation,
            **keywords,
            return_aux=True,
        )
        minus = blocked(
            positions @ minus_deformation,
            numbers,
            cell @ minus_deformation,
            origin @ minus_deformation,
            **keywords,
            return_aux=True,
        )
        assert (
            _eval_branch(plus)
            == _eval_branch(blocked_output)
            == _eval_branch(minus)
        )
        stress_fd = (plus.energy - minus.energy) / (
            2.0 * stress_step * volume
        )
        analytic = torch.sum(blocked_output.stress * direction)
        torch.testing.assert_close(
            analytic, stress_fd, atol=5.0e-6, rtol=5.0e-4
        )


def test_blocked_direct_and_grouped_paths_never_call_dense_candidate_or_ot(
    typed_crystal, monkeypatch
):
    _, edge, _, _, arguments, keywords = _direct_adaptive_case(
        typed_crystal, 5, template_id="blocked-no-dense", warmup=20
    )
    blocked = _blocked_clone(edge, 2, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("blocked Potential path invoked dense candidate/OT")

    monkeypatch.setattr(potential_module, "atom_site_displacements", forbidden)
    monkeypatch.setattr(potential_module, "build_compact_transport_edges", forbidden)
    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", forbidden)
    monkeypatch.setattr(potential_module, "build_probability_multipoles", forbidden)

    positions, numbers, cell, origin = arguments
    fixed = blocked(
        positions.clone().requires_grad_(True),
        numbers,
        cell,
        origin,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
    )
    adaptive = blocked(
        *arguments,
        **keywords,
        compute_forces=True,
        compute_stress=True,
    )
    assert fixed.auxiliary is None and adaptive.auxiliary is None
    assert torch.isfinite(fixed.energy) and torch.isfinite(adaptive.energy)

    _, grouped_edge, _, _, batch, contexts, policies = _grouped_case(typed_crystal)
    grouped_blocked = _blocked_clone(grouped_edge, 2, 2)
    batch.positions.requires_grad_(True)
    grouped = evaluate_structure_batch(
        grouped_blocked,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
    )
    assert grouped.auxiliary is None
    assert torch.isfinite(grouped.energy).all()


@pytest.mark.parametrize(
    "orthogonal",
    [
        torch.tensor(
            [[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]],
            dtype=torch.float64,
        ),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
    ],
)
def test_blocked_fixed_symmetry_covariance_and_state(typed_crystal, orthogonal):
    support = _blocked_support(_edge_support(), 2, 2)
    model = _model(typed_crystal, support)
    positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    numbers = _numbers(typed_crystal)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    output = model(
        positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )

    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        positions.detach()[order].requires_grad_(True),
        numbers[order],
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
    )
    torch.testing.assert_close(permuted.energy, output.energy, atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(
        permuted.forces, output.forces[order], atol=5e-12, rtol=5e-12
    )

    translation = torch.tensor([0.7, -0.3, 0.9], dtype=torch.float64)
    translated = model(
        (positions.detach() + translation).requires_grad_(True),
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"] + translation,
        compute_forces=True,
    )
    torch.testing.assert_close(translated.energy, output.energy, atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(
        translated.forces, output.forces, atol=5e-12, rtol=5e-12
    )

    wrapped_positions = positions.detach().clone()
    wrapped_positions[1] += typed_crystal["cell"][0]
    wrapped = model(
        wrapped_positions.requires_grad_(True),
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
    )
    torch.testing.assert_close(wrapped.energy, output.energy, atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(wrapped.forces, output.forces, atol=5e-12, rtol=5e-12)

    rotated_data = dict(typed_crystal)
    rotated_data["positions"] = typed_crystal["positions"] @ orthogonal.T
    rotated_data["origin"] = typed_crystal["origin"] @ orthogonal.T
    rotated_data["cell"] = typed_crystal["cell"] @ orthogonal.T
    rotated_model = _model(rotated_data, support)
    rotated_model.load_state_dict(model.state_dict(), strict=True)
    rotated = rotated_model(
        rotated_data["positions"][:5].clone().requires_grad_(True),
        numbers,
        rotated_data["cell"],
        rotated_data["origin"],
        compute_forces=True,
    )
    torch.testing.assert_close(rotated.energy, output.energy, atol=3e-8, rtol=3e-8)
    torch.testing.assert_close(
        rotated.forces,
        output.forces @ orthogonal.T,
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        output.forces.sum(0), torch.zeros(3, dtype=torch.float64), atol=2e-11, rtol=0
    )
    assert torch.equal(output.stress, output.stress.T)
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())


def test_blocked_grouped_fixed_adaptive_order_and_contextual_failure(
    typed_crystal,
):
    _, dense_candidate, _, _, batch, contexts, policies = _grouped_case(
        typed_crystal
    )
    blocked = _blocked_clone(dense_candidate, 2, 2)
    batch.positions.requires_grad_(True)
    fixed_dense = evaluate_structure_batch(
        dense_candidate,
        batch,
        contexts,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    fixed_blocked = evaluate_structure_batch(
        blocked,
        batch,
        contexts,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    for name, tolerance in (
        ("energy", 3.0e-13),
        ("site_energy", 3.0e-13),
        ("forces", 3.0e-12),
        ("stress", 3.0e-12),
    ):
        torch.testing.assert_close(
            getattr(fixed_blocked, name),
            getattr(fixed_dense, name),
            atol=tolerance,
            rtol=tolerance,
        )

    adaptive_dense = evaluate_structure_batch(
        dense_candidate,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    adaptive_blocked = evaluate_structure_batch(
        blocked,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert adaptive_blocked.sample_ids == batch.sample_ids
    assert adaptive_blocked.template_ids == ("zeta", "alpha", "zeta")
    assert adaptive_blocked.site_ptr.tolist() == [0, 6, 10, 16]
    for name, tolerance in (
        ("energy", 3.0e-13),
        ("site_energy", 3.0e-13),
        ("forces", 3.0e-11),
        ("stress", 3.0e-11),
    ):
        torch.testing.assert_close(
            getattr(adaptive_blocked, name),
            getattr(adaptive_dense, name),
            atol=tolerance,
            rtol=tolerance,
        )
    for auxiliary in adaptive_blocked.auxiliary:
        support = auxiliary["transport_support"]
        diagnostics = auxiliary["evaluation_diagnostics"]
        assert support.candidate_backend == "blocked"
        assert support.candidate_fingerprint
        assert not support.dense_candidate_allocation_observed
        assert diagnostics.transport_candidate_backend == "blocked"
        assert not diagnostics.transport_dense_plan_materialized

    tiny = TransportSupportConfig(
        "compact_c2",
        0.1,
        0.05,
        0.01,
        backend="edge_list",
        candidate_backend="blocked",
        site_block_size=2,
        atom_block_size=2,
    )
    failing = _configured(blocked, tiny)
    with pytest.raises(EvaluationPhaseError) as caught:
        evaluate_structure_batch(
            failing,
            batch,
            contexts,
            solver_path=EVAL_ADAPTIVE,
            evaluation_policies=policies,
        )
    message = str(caught.value)
    assert caught.value.reason_code in ("ATOM_WITHOUT_SUPPORT", "NO_TOTAL_SUPPORT")
    assert "structure_index=" in message
    assert "sample_id=" in message and "template_id=" in message
    assert "candidate_backend='blocked'" in message
    assert "site_block_size=2 atom_block_size=2" in message
    assert "candidate_extraction" in message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_blocked_cuda_direct_and_grouped_fixed_adaptive(typed_crystal, dtype):
    data = {
        key: value.to(device="cuda", dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value.to(device="cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in typed_crystal.items()
    }
    _, edge, _, _, arguments, keywords = _direct_adaptive_case(
        data, 5, template_id=f"blocked-cuda-{dtype}", warmup=20
    )
    blocked = _blocked_clone(edge, 2, 2)
    fixed = blocked(
        arguments[0].clone().requires_grad_(True),
        arguments[1],
        arguments[2],
        arguments[3],
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    adaptive = blocked(
        *arguments,
        **keywords,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    for output in (fixed, adaptive):
        for tensor in (output.energy, output.forces, output.stress):
            assert tensor.dtype == dtype and tensor.device.type == "cuda"
            assert torch.isfinite(tensor).all()
        assert not output.auxiliary["ot"].dense_plan_materialized
        assert not output.auxiliary[
            "transport_support"
        ].dense_candidate_allocation_observed

    _, grouped_edge, _, _, batch, contexts, policies = _grouped_case(
        typed_crystal, dtype=dtype, device="cuda"
    )
    grouped_blocked = _blocked_clone(grouped_edge, 2, 2)
    batch.positions.requires_grad_(True)
    grouped = evaluate_structure_batch(
        grouped_blocked,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    for tensor in (grouped.energy, grouped.forces, grouped.stress):
        assert tensor.dtype == dtype and tensor.device.type == "cuda"
        assert torch.isfinite(tensor).all()
    assert all(
        not auxiliary["transport_support"].dense_candidate_allocation_observed
        for auxiliary in grouped.auxiliary
    )
