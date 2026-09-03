from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import (
    PotentialConfig,
    ReferenceSitePotential,
    evaluate_structure_batch,
)
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TransportSupportConfig,
)
from test_evaluation_phase_potential import _policy
from test_grouped_evaluation_phase_batch import _adaptive_case, _individual
from test_runtime_template_context import make_context, make_template


def _numbers(data, count=5):
    return torch.tensor(
        [6 if int(value) == 0 else 41 for value in data["site_types"][:count]],
        dtype=torch.long,
        device=data["positions"].device,
    )


def _model(data, support=None):
    tolerance = 1e-6 if data["cell"].dtype == torch.float32 else 1e-7
    feature = ProbabilityMultipoleConfig(
        (6, 41), 2, 2, 1.0, 3.0, tolerance, site_type_vocabulary=(0, 1)
    )
    irreps = "2x0e+4x0e+4x1o+4x2e"
    higher = HigherBodyConfig(irreps, 2, 2, 2, 1, 2, 3, (4,), 6.0, 3.0, 1.0)
    config = PotentialConfig(
        (6, 41),
        1,
        feature,
        higher,
        8,
        1.0,
        transport_support=(TransportSupportConfig() if support is None else support),
    )
    topology = build_reference_graph_topology(
        data["sites"],
        data["site_types"],
        data["cell"],
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    return ReferenceSitePotential(
        config,
        topology,
        data["modes"],
        data["mode_weights"],
        torch.eye(2, dtype=data["cell"].dtype, device=data["cell"].device),
        data["site_weights"],
        data["channel_weights"],
        (-1.0, 2.0),
    ).to(data["cell"])


def _compact():
    return TransportSupportConfig("compact_c2", 2.6, 0.5, 0.2)


def _compact_clone(model):
    compact = ReferenceSitePotential(
        replace(model.config, transport_support=_compact()),
        model.topology,
        model.phase_modes,
        model.phase_mode_weights,
        model.species_alignment_weights,
        model.site_alignment_weights,
        model.phase_channel_weights,
        model.atomic_baseline,
    ).to(model.atomic_baseline)
    compact.load_state_dict(model.state_dict(), strict=True)
    return compact


def _with_eval_warmup(model, iterations):
    configured = ReferenceSitePotential(
        replace(model.config, eval_sinkhorn_warmup_iterations=iterations),
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


def test_evaluation_sinkhorn_warmup_is_explicit_config_opt_in(typed_crystal):
    base = _compact_clone(_model(typed_crystal))
    model = _with_eval_warmup(base, 32)
    template = make_template(typed_crystal, template_id="compact-warmup-opt-in")
    output = model(
        typed_crystal["positions"][:5],
        _numbers(typed_crystal),
        typed_crystal["cell"],
        typed_crystal["origin"],
        solver_path=EVAL_ADAPTIVE,
        template_context=make_context(template),
        evaluation_policy=_policy(template),
        return_aux=True,
    )
    diagnostics = output.auxiliary["evaluation_diagnostics"]
    assert diagnostics.transport_sinkhorn_warmup_iterations == 32
    assert model.config.eval_sinkhorn_warmup_iterations == 32
    assert PotentialConfig.from_dict(model.config.to_dict()) == model.config
    assert set(model.state_dict()) == set(base.state_dict())


def _adaptive_energy(model, data, context, policy, positions, *, cell=None, origin=None):
    return model(
        positions,
        _numbers(data),
        data["cell"] if cell is None else cell,
        data["origin"] if origin is None else origin,
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        return_aux=True,
    )


def _strain_energy(model, data, numbers, direction, magnitude):
    deformation = torch.eye(3, dtype=data["cell"].dtype) + magnitude * direction
    return model(
        data["positions"][:5] @ deformation,
        numbers,
        data["cell"] @ deformation,
        data["origin"] @ deformation,
    ).energy


def test_compact_potential_dense_comparison_state_and_mixed_backward(typed_crystal):
    dense = _model(typed_crystal)
    compact = _model(typed_crystal, _compact())
    assert PotentialConfig.from_dict(compact.config.to_dict()) == compact.config
    legacy_config = dense.config.to_dict()
    del legacy_config["transport_support"]
    assert PotentialConfig.from_dict(legacy_config).transport_support.kind == "dense"
    compact.load_state_dict(dense.state_dict(), strict=True)
    assert tuple(compact.state_dict()) == tuple(dense.state_dict())
    state_keys = tuple(compact.state_dict())
    parameter_ids = tuple(id(value) for value in compact.parameters())
    baseline = compact.atomic_baseline.clone()
    positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    numbers = _numbers(typed_crystal)

    dense_output = dense(
        positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        return_aux=True,
    )
    output = compact(
        positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    assert torch.isfinite(output.energy)
    assert torch.isfinite(output.forces).all()
    assert torch.isfinite(output.stress).all()
    torch.testing.assert_close(output.stress, output.stress.T, atol=0, rtol=0)
    diagnostics = output.auxiliary["transport_support"]
    assert diagnostics.maximum_atom_matching_size == 5
    assert diagnostics.total_matching_size == 6
    assert diagnostics.active_edge_count == 28
    assert diagnostics.candidate_edge_count == 30
    assert diagnostics.effective_diagnostic_tolerance == 1.0e-7
    compact_ot = output.auxiliary["ot"]
    dense_ot = dense_output.auxiliary["ot"]
    assert torch.count_nonzero(compact_ot.P != dense_ot.P) > 0
    assert torch.max(torch.abs(compact_ot.P - dense_ot.P)) > 1.0e-5
    assert torch.max(torch.abs(compact_ot.q - dense_ot.q)) > 1.0e-6
    compact_multipoles = output.auxiliary["multipoles"].equivariant_features
    dense_multipoles = dense_output.auxiliary["multipoles"].equivariant_features
    assert torch.max(torch.abs(compact_multipoles - dense_multipoles)) > 1.0e-6
    assert max(float(compact_ot.row_residual), float(compact_ot.column_residual)) < 1e-12
    torch.testing.assert_close(compact_ot.q.sum(), torch.tensor(1.0, dtype=torch.float64), atol=2e-14, rtol=0)

    gradients = torch.autograd.grad(
        output.forces.square().sum(),
        (
            compact.readout.mlp[-1].weight,
            compact.layers[0].corr.C2_product.weight,
            compact.central.embedding.weight,
        ),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    assert tuple(compact.state_dict()) == state_keys
    assert tuple(id(value) for value in compact.parameters()) == parameter_ids
    assert torch.equal(compact.atomic_baseline, baseline)


def test_compact_potential_force_stress_fd_permutation_translation_and_wrapping(typed_crystal):
    model = _model(typed_crystal, _compact())
    positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    numbers = _numbers(typed_crystal)
    output = model(
        positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
    )
    h = 1.0e-6
    displacement = torch.zeros_like(positions)
    displacement[2, 1] = h
    plus = model(
        positions.detach() + displacement,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
    ).energy
    minus = model(
        positions.detach() - displacement,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
    ).energy
    torch.testing.assert_close(
        output.forces[2, 1], -(plus - minus) / (2 * h), atol=5e-6, rtol=5e-5
    )
    direction = torch.zeros((3, 3), dtype=torch.float64)
    direction[0, 1] = direction[1, 0] = 0.5
    stress_fd = (
        _strain_energy(model, typed_crystal, numbers, direction, h)
        - _strain_energy(model, typed_crystal, numbers, direction, -h)
    ) / (2 * h)
    volume = torch.linalg.det(typed_crystal["cell"]).abs()
    torch.testing.assert_close(
        volume * torch.sum(output.stress * direction), stress_fd, atol=5e-6, rtol=5e-5
    )

    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        positions.detach()[order].requires_grad_(True),
        numbers[order],
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
    )
    torch.testing.assert_close(permuted.energy, output.energy, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(permuted.forces, output.forces[order], atol=2e-8, rtol=2e-8)
    shift = torch.tensor([0.7, -0.3, 0.9], dtype=torch.float64)
    translated = model(
        (positions.detach() + shift).requires_grad_(True),
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"] + shift,
        compute_forces=True,
    )
    torch.testing.assert_close(translated.energy, output.energy, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(translated.forces, output.forces, atol=2e-8, rtol=2e-8)
    wrapped_positions = positions.detach().clone()
    wrapped_positions[1] += typed_crystal["cell"][0]
    wrapped = model(
        wrapped_positions.requires_grad_(True),
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
    )
    torch.testing.assert_close(wrapped.energy, output.energy, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(wrapped.forces, output.forces, atol=2e-8, rtol=2e-8)


def test_compact_potential_adaptive_energy_force_stress_and_fixed_parity(typed_crystal):
    compact = _model(typed_crystal, _compact())
    template = make_template(typed_crystal, template_id="compact-eval")
    context = make_context(template)
    policy = _policy(template)
    keys = tuple(compact.state_dict())
    parameters = tuple(id(value) for value in compact.parameters())
    positions = typed_crystal["positions"][:5].clone()
    arguments = (
        positions,
        _numbers(typed_crystal),
        typed_crystal["cell"],
        typed_crystal["origin"],
    )
    fixed = compact(*arguments, template_context=context, return_aux=True)
    adaptive = compact(
        *arguments,
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert torch.isfinite(adaptive.energy)
    assert torch.isfinite(adaptive.forces).all()
    assert torch.isfinite(adaptive.stress).all()
    assert adaptive.forces.grad_fn is None and adaptive.stress.grad_fn is None
    fixed_ot = fixed.auxiliary["ot"]
    adaptive_ot = adaptive.auxiliary["ot"]
    torch.testing.assert_close(adaptive_ot.P, fixed_ot.P, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(adaptive_ot.q, fixed_ot.q, atol=3e-10, rtol=3e-10)
    assert torch.equal(adaptive_ot.P == 0.0, fixed_ot.P == 0.0)
    torch.testing.assert_close(
        adaptive.auxiliary["multipoles"].equivariant_features,
        fixed.auxiliary["multipoles"].equivariant_features,
        atol=3e-9,
        rtol=3e-9,
    )
    diagnostics = adaptive.auxiliary["evaluation_diagnostics"]
    assert diagnostics.transport_kind == "compact_c2"
    assert diagnostics.transport_sinkhorn_warmup_iterations == 16
    assert diagnostics.transport_fallback_sinkhorn_iterations == 0
    assert diagnostics.transport_r_on == 2.1
    assert diagnostics.transport_r_off == 2.6
    assert diagnostics.transport_r_candidate == pytest.approx(2.8)
    assert diagnostics.transport_active_edge_count == 28
    assert diagnostics.transport_maximum_matching_size == 5
    assert diagnostics.transport_total_support_feasible
    assert diagnostics.transport_q_mass_error < 2.0e-12
    assert diagnostics.effective_transport_tolerance == 1.0e-12
    assert diagnostics.differentiability_scope == "selected_branch_first_order"
    assert not diagnostics.transport_fallback_used
    assert tuple(compact.state_dict()) == keys
    assert tuple(id(value) for value in compact.parameters()) == parameters


def test_compact_adaptive_force_and_six_stress_directions_finite_difference(
    typed_crystal,
):
    model = _model(typed_crystal, _compact())
    template = make_template(typed_crystal, template_id="compact-eval-fd")
    context, policy = make_context(template), _policy(template)
    positions = typed_crystal["positions"][:5].clone()
    baseline = model(
        positions,
        _numbers(typed_crystal),
        typed_crystal["cell"],
        typed_crystal["origin"],
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    signature = (
        baseline.auxiliary["evaluation_diagnostics"].selected_grouped_index,
        baseline.auxiliary["ot"].P == 0.0,
        baseline.auxiliary["ot"].fallback_used,
    )
    force_step = 1.0e-6
    maximum_force_error = 0.0
    for atom, component in ((0, 0), (2, 1), (4, 2)):
        displacement = torch.zeros_like(positions)
        displacement[atom, component] = force_step
        plus = _adaptive_energy(
            model, typed_crystal, context, policy, positions + displacement
        )
        minus = _adaptive_energy(
            model, typed_crystal, context, policy, positions - displacement
        )
        for output in (plus, minus):
            diagnostics = output.auxiliary["evaluation_diagnostics"]
            assert diagnostics.selected_grouped_index == signature[0]
            assert torch.equal(output.auxiliary["ot"].P == 0.0, signature[1])
            assert not diagnostics.transport_fallback_used
        finite = -(plus.energy - minus.energy) / (2.0 * force_step)
        error = float(torch.abs(finite - baseline.forces[atom, component]))
        maximum_force_error = max(maximum_force_error, error)
    assert maximum_force_error <= 5.0e-6

    directions = []
    for axis in range(3):
        direction = torch.zeros((3, 3), dtype=torch.float64)
        direction[axis, axis] = 1.0
        directions.append(direction)
    for left, right in ((1, 2), (0, 2), (0, 1)):
        direction = torch.zeros((3, 3), dtype=torch.float64)
        direction[left, right] = direction[right, left] = 0.5
        directions.append(direction)
    identity = torch.eye(3, dtype=torch.float64)
    volume = torch.linalg.det(typed_crystal["cell"]).abs()
    strain_step = 1.0e-4
    maximum_stress_error = 0.0
    for direction in directions:
        plus_deformation = identity + strain_step * direction
        minus_deformation = identity - strain_step * direction
        plus = _adaptive_energy(
            model,
            typed_crystal,
            context,
            policy,
            positions @ plus_deformation,
            cell=typed_crystal["cell"] @ plus_deformation,
            origin=typed_crystal["origin"] @ plus_deformation,
        )
        minus = _adaptive_energy(
            model,
            typed_crystal,
            context,
            policy,
            positions @ minus_deformation,
            cell=typed_crystal["cell"] @ minus_deformation,
            origin=typed_crystal["origin"] @ minus_deformation,
        )
        for output in (plus, minus):
            diagnostics = output.auxiliary["evaluation_diagnostics"]
            assert diagnostics.selected_grouped_index == signature[0]
            assert torch.equal(output.auxiliary["ot"].P == 0.0, signature[1])
            assert not diagnostics.transport_fallback_used
        finite = (plus.energy - minus.energy) / (2.0 * strain_step * volume)
        automatic = torch.sum(baseline.stress * direction)
        maximum_stress_error = max(
            maximum_stress_error, float(torch.abs(finite - automatic))
        )
    assert maximum_stress_error <= 5.0e-6

    force_only = model(
        positions,
        _numbers(typed_crystal),
        typed_crystal["cell"],
        typed_crystal["origin"],
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        compute_forces=True,
        return_aux=True,
    )
    stress_only = model(
        positions,
        _numbers(typed_crystal),
        typed_crystal["cell"],
        typed_crystal["origin"],
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        compute_stress=True,
        return_aux=True,
    )
    for derivative_only in (force_only, stress_only):
        diagnostics = derivative_only.auxiliary["evaluation_diagnostics"]
        assert diagnostics.selected_grouped_index == signature[0]
        assert torch.equal(derivative_only.auxiliary["ot"].P == 0.0, signature[1])
        assert not diagnostics.transport_fallback_used
        torch.testing.assert_close(
            derivative_only.energy, baseline.energy, atol=0.0, rtol=0.0
        )
    # Asking autograd for one input or both inputs can alter the threaded CPU
    # accumulation order by a few floating-point roundoff units.  This is not
    # a solver-branch or FD tolerance: both calls use the same energy and the
    # checks above certify the same selected support/non-fallback branch.
    derivative_roundoff = 64.0 * torch.finfo(positions.dtype).eps
    torch.testing.assert_close(
        force_only.forces,
        baseline.forces,
        atol=derivative_roundoff,
        rtol=derivative_roundoff,
    )
    torch.testing.assert_close(
        stress_only.stress,
        baseline.stress,
        atol=derivative_roundoff,
        rtol=derivative_roundoff,
    )


def test_grouped_mixed_template_compact_adaptive_matches_individual(typed_crystal):
    _, dense_model, _, _, batch, contexts, policies = _adaptive_case(typed_crystal)
    model = _compact_clone(dense_model)
    batch.positions.requires_grad_(True)
    grouped = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=EVAL_ADAPTIVE,
        evaluation_policies=policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    individual = _individual(
        model,
        batch,
        contexts,
        policies,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert torch.equal(grouped.energy, torch.stack([item.energy for item in individual]))
    assert torch.equal(grouped.site_energy, torch.cat([item.site_energy for item in individual]))
    torch.testing.assert_close(
        grouped.forces, torch.cat([item.forces for item in individual]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        grouped.stress, torch.stack([item.stress for item in individual]), atol=0, rtol=0
    )
    for auxiliary in grouped.auxiliary:
        diagnostics = auxiliary["evaluation_diagnostics"]
        assert diagnostics.transport_kind == "compact_c2"
        assert diagnostics.transport_total_support_feasible
        assert not diagnostics.transport_fallback_used


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
def test_compact_potential_o3_energy_and_force_covariance(typed_crystal, orthogonal):
    model = _model(typed_crystal, _compact())
    positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    numbers = _numbers(typed_crystal)
    output = model(
        positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
    )
    rotated_data = dict(typed_crystal)
    rotated_data["positions"] = typed_crystal["positions"] @ orthogonal.T
    rotated_data["origin"] = typed_crystal["origin"] @ orthogonal.T
    rotated_data["cell"] = typed_crystal["cell"] @ orthogonal.T
    rotated = _model(rotated_data, _compact())
    rotated.load_state_dict(model.state_dict(), strict=True)
    rotated_output = rotated(
        rotated_data["positions"][:5].clone().requires_grad_(True),
        numbers,
        rotated_data["cell"],
        rotated_data["origin"],
        compute_forces=True,
    )
    torch.testing.assert_close(rotated_output.energy, output.energy, atol=3e-8, rtol=3e-8)
    torch.testing.assert_close(
        rotated_output.forces, output.forces @ orthogonal.T, atol=3e-6, rtol=3e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compact_potential_cuda_force_stress_smoke(typed_crystal, dtype):
    data = {
        key: (
            value.to(device="cuda", dtype=dtype)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value.to(device="cuda")
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in typed_crystal.items()
    }
    model = _model(data, _compact())
    positions = data["positions"][:5].clone().requires_grad_(True)
    output = model(
        positions,
        _numbers(data),
        data["cell"],
        data["origin"],
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert output.energy.dtype == dtype and output.energy.device.type == "cuda"
    template = make_template(data, template_id=f"compact-cuda-{dtype}")
    adaptive = model(
        positions,
        _numbers(data),
        data["cell"],
        data["origin"],
        solver_path=EVAL_ADAPTIVE,
        template_context=make_context(template),
        evaluation_policy=_policy(template),
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert adaptive.energy.dtype == dtype and adaptive.energy.device.type == "cuda"
    assert torch.isfinite(adaptive.energy)
    assert torch.isfinite(adaptive.forces).all()
    assert torch.isfinite(adaptive.stress).all()
    expected = 1.0e-6 if dtype == torch.float32 else 1.0e-12
    assert (
        adaptive.auxiliary["evaluation_diagnostics"].effective_transport_tolerance
        == expected
    )
    assert torch.isfinite(output.forces).all() and torch.isfinite(output.stress).all()
    expected_tolerance = 1e-6 if dtype == torch.float32 else 1e-7
    assert output.auxiliary["transport_support"].effective_diagnostic_tolerance == expected_tolerance
