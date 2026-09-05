from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

import refsite_mlip.models.potential as potential_module
from refsite_mlip.interactions import HigherBodyArchitectureError
from refsite_mlip.models import ReferenceSitePotential
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from symmetric_potential_helpers import (
    direct_symmetric_message,
    numbers,
    v2_grouped_case,
)
from test_evaluation_phase_potential import _policy
from test_compact_support_potential import _compact
from test_edge_list_compact_potential import _edge_clone
from test_evaluation_phase_derivatives import _accepted_positions


def _default_arguments(data, contexts, count=5):
    return (
        data["positions"][:count],
        numbers(data, count),
        data["cell"],
        data["origin"],
    ), contexts["zeta"]


def _basis_state(model):
    return {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith("symmetric_cg_basis.U_order_")
    }


def test_v2_constructor_has_one_top_level_basis_and_independent_layer_weights(
    typed_crystal,
):
    _, one, _, _, _, _ = v2_grouped_case(typed_crystal, layers=1)
    data, three, _, _, _, contexts = v2_grouped_case(
        typed_crystal, layers=3
    )
    one_basis = _basis_state(one)
    three_basis = _basis_state(three)
    assert len(one_basis) == len(three_basis) == 9
    assert tuple(one_basis) == tuple(three_basis)
    assert sum(value.numel() for value in one_basis.values()) == sum(
        value.numel() for value in three_basis.values()
    )
    assert sum(
        value.numel() * value.element_size() for value in three_basis.values()
    ) == three.symmetric_cg_basis.buffer_byte_count
    assert not any(
        name.startswith("layers.")
        and ("U_order_" in name or "u_output_" in name)
        for name in three.state_dict()
    )
    assert not any(
        name.startswith("symmetric_cg_basis")
        for name, _ in three.named_parameters()
    )
    per_layer = [
        tuple(layer.symmetric_contraction.parameters()) for layer in three.layers
    ]
    assert all(sum(value.numel() for value in group) == 552 for group in per_layer)
    for left, right in zip(per_layer, per_layer[1:]):
        assert all(a is not b and a.data_ptr() != b.data_ptr() for a, b in zip(left, right))
    assert all(tuple(layer.symmetric_contraction.buffers()) == () for layer in three.layers)
    assert all(
        layer.symmetric_basis_fingerprint
        == three.symmetric_cg_basis.basis_fingerprint
        for layer in three.layers
    )

    clone = v2_grouped_case(typed_crystal, layers=3)[1]
    clone.load_state_dict(three.state_dict(), strict=True)
    arguments, context = _default_arguments(data, contexts)
    original = three(*arguments, template_context=context)
    restored = clone(*arguments, template_context=context)
    torch.testing.assert_close(restored.energy, original.energy, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        restored.site_energy, original.site_energy, atol=0.0, rtol=0.0
    )


def test_invalid_v2_basis_fails_before_residual_parameters_and_rng(
    typed_crystal, monkeypatch
):
    data, model, registry, _, _, _ = v2_grouped_case(typed_crystal, layers=1)
    default = registry.resolve("zeta")

    class InjectedBasisError(ValueError):
        reason_code = "INJECTED_BASIS_INVALID"

    def invalid_basis(*args, **kwargs):
        del args, kwargs
        raise InjectedBasisError("incompatible generalized-CG path")

    def forbidden_residual(*args, **kwargs):
        del args, kwargs
        raise AssertionError("residual parameters were partially constructed")

    monkeypatch.setattr(potential_module, "SymmetricCGBasisBank", invalid_basis)
    monkeypatch.setattr(potential_module, "ResidualInteractionBlock", forbidden_residual)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(HigherBodyArchitectureError) as caught:
        ReferenceSitePotential(
            model.config,
            default.topology,
            default.phase_modes,
            default.phase_mode_weights,
            torch.eye(2, dtype=data["cell"].dtype),
            default.site_alignment_weights,
            default.phase_channel_weights,
        )
    assert caught.value.reason_code == "INJECTED_BASIS_INVALID"
    assert "InjectedBasisError" in str(caught.value)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_v2_potential_order_is_exact_cumulative_full_path(typed_crystal, order):
    data, model, _, _, _, contexts = v2_grouped_case(
        typed_crystal, order=order, layers=2
    )
    arguments, context = _default_arguments(data, contexts)
    output = model(*arguments, template_context=context, return_aux=True)
    c_bar = model.central(output.raw_c, context.topology.site_types)
    assert output.auxiliary is not None
    for layer, details in zip(model.layers, output.auxiliary["correlations"]):
        contributions = tuple(
            direct_symmetric_message(
                layer,
                model.symmetric_cg_basis,
                details["A"],
                c_bar,
                order=current,
            )
            for current in range(1, order + 1)
        )
        expected = sum(contributions)
        torch.testing.assert_close(
            details["symmetric_output"], expected, atol=3.0e-15, rtol=3.0e-15
        )
        # The v2 message is the unscaled full-path sum.  The v1-only /sqrt(3)
        # normalization and C/Z modules are absent.
        assert not hasattr(layer, "corr")
        assert not hasattr(layer, "outer")
        assert not hasattr(layer, "contract")
        if order > 1 and bool(torch.any(expected != 0.0)):
            assert not torch.equal(details["symmetric_output"], expected / (3.0**0.5))
    assert bool(torch.isfinite(output.energy))


def test_v2_energy_force_symmetry_and_finite_difference(typed_crystal):
    data, model, _, _, _, contexts = v2_grouped_case(typed_crystal, layers=2)
    arguments, context = _default_arguments(data, contexts)
    positions, atomic_numbers, cell, origin = arguments
    positions = positions.clone().requires_grad_(True)
    output = model(
        positions,
        atomic_numbers,
        cell,
        origin,
        template_context=context,
        compute_forces=True,
    )
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        positions.detach()[order].requires_grad_(True),
        atomic_numbers[order],
        cell,
        origin,
        template_context=context,
        compute_forces=True,
    )
    torch.testing.assert_close(permuted.energy, output.energy, atol=2.0e-10, rtol=2.0e-10)
    torch.testing.assert_close(permuted.forces, output.forces[order], atol=2.0e-8, rtol=2.0e-8)
    shift = torch.tensor([0.7, -0.3, 0.9], dtype=torch.float64)
    translated = model(
        (positions.detach() + shift).requires_grad_(True),
        atomic_numbers,
        cell,
        origin + shift,
        template_context=context,
        compute_forces=True,
    )
    torch.testing.assert_close(translated.energy, output.energy, atol=2.0e-10, rtol=2.0e-10)
    torch.testing.assert_close(translated.forces, output.forces, atol=2.0e-8, rtol=2.0e-8)
    torch.testing.assert_close(
        output.forces.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=2.0e-8, rtol=0.0
    )

    step = 2.0e-6
    delta = torch.zeros_like(positions)
    delta[2, 1] = step
    plus = model(
        positions.detach() + delta,
        atomic_numbers,
        cell,
        origin,
        template_context=context,
    ).energy
    minus = model(
        positions.detach() - delta,
        atomic_numbers,
        cell,
        origin,
        template_context=context,
    ).energy
    finite_difference = -(plus - minus) / (2.0 * step)
    torch.testing.assert_close(
        output.forces[2, 1], finite_difference, atol=5.0e-6, rtol=5.0e-5
    )


@pytest.mark.parametrize(
    "matrix",
    [
        torch.tensor(
            [[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]],
            dtype=torch.float64,
        ),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
    ],
    ids=["proper-rotation", "reflection"],
)
def test_v2_potential_o3_energy_and_force_covariance(typed_crystal, matrix):
    data, model, _, _, _, contexts = v2_grouped_case(
        typed_crystal, layers=2
    )
    arguments, context = _default_arguments(data, contexts)
    original = model(
        arguments[0].clone().requires_grad_(True),
        *arguments[1:],
        template_context=context,
        compute_forces=True,
    )
    transformed_fixture = dict(typed_crystal)
    for key in ("positions", "origin", "cell", "displacement"):
        if key in transformed_fixture:
            transformed_fixture[key] = transformed_fixture[key] @ matrix.T
    transformed_data, transformed_model, _, _, _, transformed_contexts = (
        v2_grouped_case(transformed_fixture, layers=2)
    )
    transformed_model.load_state_dict(model.state_dict(), strict=True)
    transformed_arguments, transformed_context = _default_arguments(
        transformed_data, transformed_contexts
    )
    transformed = transformed_model(
        transformed_arguments[0].clone().requires_grad_(True),
        *transformed_arguments[1:],
        template_context=transformed_context,
        compute_forces=True,
    )
    torch.testing.assert_close(
        transformed.energy, original.energy, atol=3.0e-8, rtol=3.0e-8
    )
    torch.testing.assert_close(
        transformed.forces,
        original.forces @ matrix.T,
        atol=3.0e-6,
        rtol=3.0e-6,
    )


def _strained_energy(model, arguments, context, strain):
    positions, atomic_numbers, cell, origin = arguments
    deformation = torch.eye(3, dtype=strain.dtype) + strain
    return model(
        positions @ deformation,
        atomic_numbers,
        cell @ deformation,
        origin @ deformation,
        template_context=context,
    ).energy


def test_v2_stress_all_six_symmetric_strain_directions(typed_crystal):
    data, model, _, _, _, contexts = v2_grouped_case(typed_crystal, layers=1)
    arguments, context = _default_arguments(data, contexts)
    output = model(*arguments, template_context=context, compute_stress=True)
    assert output.stress is not None and output.stress_voigt is not None
    torch.testing.assert_close(output.stress, output.stress.T, atol=0.0, rtol=0.0)
    directions = []
    for index in range(3):
        value = torch.zeros((3, 3), dtype=torch.float64)
        value[index, index] = 1.0
        directions.append(value)
    for first, second in ((1, 2), (0, 2), (0, 1)):
        value = torch.zeros((3, 3), dtype=torch.float64)
        value[first, second] = value[second, first] = 0.5
        directions.append(value)
    step = 2.0e-6
    volume = torch.linalg.det(arguments[2]).abs()
    for direction in directions:
        finite_difference = (
            _strained_energy(model, arguments, context, step * direction)
            - _strained_energy(model, arguments, context, -step * direction)
        ) / (2.0 * step * volume)
        torch.testing.assert_close(
            torch.sum(output.stress * direction),
            finite_difference,
            atol=5.0e-6,
            rtol=5.0e-5,
        )
    expected_voigt = output.stress[
        (0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)
    ]
    torch.testing.assert_close(output.stress_voigt, expected_voigt, atol=0.0, rtol=0.0)


def test_v2_force_stress_graph_reaches_every_active_layer(typed_crystal):
    data, model, _, _, _, contexts = v2_grouped_case(typed_crystal, layers=2)
    arguments, context = _default_arguments(data, contexts)
    positions = arguments[0].clone().requires_grad_(True)
    output = model(
        positions,
        arguments[1],
        arguments[2],
        arguments[3],
        template_context=context,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
    )
    loss = (
        output.energy.square()
        + output.forces.square().sum()
        + output.stress.square().sum()
    )
    symmetric_weights = tuple(
        (layer_index, name, parameter)
        for layer_index, layer in enumerate(model.layers)
        for name, parameter in layer.symmetric_contraction.named_parameters()
    )
    selected = (
        *(parameter for _, _, parameter in symmetric_weights),
        model.central.embedding.weight,
        model.central_encoder.weight,
        model.probability_encoder.linear.weight,
        model.layers[0].edge.radial_head.network[0].weight,
    )
    gradients = torch.autograd.grad(loss, selected, allow_unused=False)
    assert all(bool(torch.all(torch.isfinite(value))) for value in gradients)
    symmetric_gradients = gradients[: len(symmetric_weights)]
    for layer_index in range(len(model.layers)):
        for order in range(1, 4):
            relevant = tuple(
                gradient
                for (candidate_layer, name, _), gradient in zip(
                    symmetric_weights, symmetric_gradients
                )
                if candidate_layer == layer_index
                and name.endswith(f"order_{order}")
            )
            assert relevant
            assert any(bool(torch.any(value != 0.0)) for value in relevant)
    assert all(
        bool(torch.any(value != 0.0))
        for value in gradients[len(symmetric_weights) :]
    )
    assert all(not value.requires_grad for value in model.symmetric_cg_basis.buffers())


def test_v2_adaptive_direct_first_derivative_contract(typed_crystal):
    data, model, registry, _, _, contexts = v2_grouped_case(
        typed_crystal, layers=1
    )
    arguments, context = _default_arguments(data, contexts)
    policy = _policy(registry.resolve("zeta"))
    output = model(
        *arguments,
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert bool(torch.isfinite(output.energy))
    assert bool(torch.all(torch.isfinite(output.forces)))
    assert bool(torch.all(torch.isfinite(output.stress)))
    diagnostics = output.auxiliary["evaluation_diagnostics"]
    assert not diagnostics.transport_fallback_used
    assert diagnostics.selected_grouped_index == 0
    with pytest.raises(EvaluationPhaseError, match="CREATE_GRAPH_UNSUPPORTED"):
        model(
            *arguments,
            solver_path=EVAL_ADAPTIVE,
            template_context=context,
            evaluation_policy=policy,
            compute_forces=True,
            create_graph=True,
        )


def test_v2_adaptive_sparse_and_dense_backend_parity(typed_crystal):
    data, base, registry, _, _, contexts = v2_grouped_case(
        typed_crystal, layers=1
    )
    dense = ReferenceSitePotential(
        replace(
            base.config,
            transport_support=_compact(),
            eval_sinkhorn_warmup_iterations=32,
        ),
        base.topology,
        base.phase_modes,
        base.phase_mode_weights,
        base.species_alignment_weights,
        base.site_alignment_weights,
        base.phase_channel_weights,
        base.atomic_baseline,
    ).to(base.atomic_baseline)
    dense.load_state_dict(base.state_dict(), strict=True)
    sparse = _edge_clone(dense)
    context = contexts["zeta"]
    policy = _policy(registry.resolve("zeta"))
    arguments = (
        _accepted_positions(data, 5),
        numbers(data, 5),
        data["cell"],
        data["origin"],
    )
    keywords = dict(
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    dense_output = dense(*arguments, **keywords)
    sparse_output = sparse(*arguments, **keywords)
    for left, right, tolerance in (
        (sparse_output.energy, dense_output.energy, 5.0e-14),
        (sparse_output.site_energy, dense_output.site_energy, 5.0e-14),
        (sparse_output.raw_c, dense_output.raw_c, 1.0e-12),
        (sparse_output.forces, dense_output.forces, 5.0e-12),
        (sparse_output.stress, dense_output.stress, 5.0e-12),
    ):
        torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
    diagnostics = sparse_output.auxiliary["evaluation_diagnostics"]
    assert diagnostics.transport_backend == "edge_list"
    assert diagnostics.transport_support_fingerprint
    assert not diagnostics.transport_dense_plan_materialized
    assert not diagnostics.transport_fallback_used
    assert diagnostics.selected_grouped_index == dense_output.auxiliary[
        "evaluation_diagnostics"
    ].selected_grouped_index


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v2_cpu_dtype_and_strict_state_roundtrip(typed_crystal, dtype):
    data, model, _, _, _, contexts = v2_grouped_case(
        typed_crystal, dtype=dtype, layers=1
    )
    arguments, context = _default_arguments(data, contexts)
    output = model(*arguments, template_context=context)
    assert output.energy.device.type == "cpu" and output.energy.dtype == dtype
    assert bool(torch.isfinite(output.energy))
    fingerprint = model.symmetric_cg_basis.basis_fingerprint
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict(), strict=True)
    restored = clone(*arguments, template_context=context)
    torch.testing.assert_close(restored.energy, output.energy, atol=0.0, rtol=0.0)
    assert clone.symmetric_cg_basis.basis_fingerprint == fingerprint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v2_cuda_direct_smoke(typed_crystal, dtype):
    data, model, registry, _, _, contexts = v2_grouped_case(
        typed_crystal, dtype=dtype, device="cuda:0", layers=1
    )
    context = contexts["zeta"]
    arguments = (
        data["positions"][:5].to("cuda:0"),
        numbers(data, 5).to("cuda:0"),
        data["cell"].to("cuda:0"),
        data["origin"].to("cuda:0"),
    )
    output = model(
        *arguments,
        template_context=context,
        compute_forces=True,
        compute_stress=True,
    )
    torch.cuda.synchronize()
    assert output.energy.device.type == "cuda" and output.energy.dtype == dtype
    assert bool(torch.isfinite(output.energy))
    assert bool(torch.all(torch.isfinite(output.forces)))
    assert bool(torch.all(torch.isfinite(output.stress)))
