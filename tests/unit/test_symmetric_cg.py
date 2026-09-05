from __future__ import annotations

from dataclasses import replace
import itertools
import math
import random

import numpy as np
import pytest
import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.interactions.symmetric_cg import (
    GeneralizedCGPath,
    SymmetricCGError,
    generate_generalized_cg,
)


ANGULAR_IRREPS = "0e + 1o + 2e"


def _o3():
    return import_e3nn_0_4_4()[1]


def _contract(coefficient: torch.Tensor, value: torch.Tensor, order: int):
    if order == 1:
        return torch.einsum("oi,i->o", coefficient, value)
    if order == 2:
        return torch.einsum("oij,i,j->o", coefficient, value, value)
    if order == 3:
        return torch.einsum(
            "oijk,i,j,k->o", coefficient, value, value, value
        )
    raise AssertionError("test helper supports orders 1..3")


def _block_slices(result):
    return {
        block.irrep: slice(block.start, block.stop)
        for block in result.input_blocks
    }


def _component_cg(output_irrep, left_irrep, right_irrep):
    o3 = _o3()
    return math.sqrt(output_irrep.dim) * o3.wigner_3j(
        output_irrep.l,
        left_irrep.l,
        right_irrep.l,
        dtype=torch.float64,
        device="cpu",
    )


def _rotation_matrix():
    o3 = _o3()
    return o3.angles_to_matrix(
        torch.tensor(0.31, dtype=torch.float64),
        torch.tensor(0.47, dtype=torch.float64),
        torch.tensor(-0.23, dtype=torch.float64),
    )


def _representation_matrix(irreps, matrix):
    return torch.block_diag(
        *(irrep.D_from_matrix(matrix) for _, irrep in irreps)
    )


def _allowed_path_identities(input_irreps, output_irrep, order):
    """Combinatorial O(3) oracle independent of production path construction."""

    entries = tuple(irrep for _, irrep in input_irreps)
    identities = []
    for selected in itertools.product(entries, repeat=order):
        if order == 1:
            if selected[0] == output_irrep:
                identities.append((tuple(map(str, selected)), ()))
        elif order == 2:
            if (
                selected[0].p * selected[1].p == output_irrep.p
                and abs(selected[0].l - selected[1].l)
                <= output_irrep.l
                <= selected[0].l + selected[1].l
            ):
                identities.append((tuple(map(str, selected)), ()))
        else:
            for intermediate_l in range(
                abs(selected[0].l - selected[1].l),
                selected[0].l + selected[1].l + 1,
            ):
                intermediate_parity = selected[0].p * selected[1].p
                if intermediate_parity * selected[2].p != output_irrep.p:
                    continue
                if not (
                    abs(intermediate_l - selected[2].l)
                    <= output_irrep.l
                    <= intermediate_l + selected[2].l
                ):
                    continue
                intermediate = f"{intermediate_l}{'e' if intermediate_parity == 1 else 'o'}"
                identities.append(
                    (tuple(map(str, selected)), (intermediate,))
                )
    return identities


def test_order_one_is_identity_with_e3nn_component_ordering():
    result = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, 1)
    assert result.path_count == 3
    assert result.input_dimension == 9
    assert tuple(
        (block.irrep, block.start, block.stop) for block in result.input_blocks
    ) == (("0e", 0, 1), ("1o", 1, 4), ("2e", 4, 9))

    value = torch.arange(9, dtype=torch.float64) + 0.25
    slices = _block_slices(result)
    for path in result.paths:
        output = path.metadata.output_irrep
        expected = value[slices[output]]
        assert torch.equal(_contract(path.coefficient, value, 1), expected)
        assert path.metadata.input_irreps == (output,)
        assert path.metadata.intermediate_irreps == ()


def test_order_two_matches_direct_real_wigner_3j_oracle():
    o3 = _o3()
    result = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, 2)
    slices = _block_slices(result)
    maximum_error = 0.0
    for path in result.paths:
        metadata = path.metadata
        output = o3.Irrep(metadata.output_irrep)
        left, right = (o3.Irrep(label) for label in metadata.input_irreps)
        oracle = torch.zeros_like(path.coefficient)
        oracle[
            (slice(None), slices[str(left)], slices[str(right)])
        ] = _component_cg(output, left, right)
        maximum_error = max(
            maximum_error,
            float(torch.max(torch.abs(path.coefficient - oracle))),
        )
    assert maximum_error == 0.0
    assert [(item.output_irrep, item.path_count) for item in result.outputs] == [
        ("0e", 3),
        ("1o", 4),
        ("2e", 4),
    ]


def test_order_two_obeys_angular_momentum_and_o3_parity_selection():
    o3 = _o3()
    result = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, 2)
    for path in result.paths:
        output = o3.Irrep(path.metadata.output_irrep)
        left, right = (o3.Irrep(label) for label in path.metadata.input_irreps)
        assert abs(left.l - right.l) <= output.l <= left.l + right.l
        assert output.p == left.p * right.p

    inputs = o3.Irreps(ANGULAR_IRREPS)
    for _, output in inputs:
        observed = [
            (path.metadata.input_irreps, path.metadata.intermediate_irreps)
            for path in result.paths_for(str(output))
        ]
        assert observed == _allowed_path_identities(inputs, output, 2)

    unavailable = generate_generalized_cg("0e", "1o", 2)
    assert unavailable.path_count == 0
    assert unavailable.outputs[0].path_count == 0
    assert not unavailable.outputs[0].has_nonzero_path


def test_order_three_matches_explicit_two_cg_contraction_oracle():
    o3 = _o3()
    result = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, 3)
    slices = _block_slices(result)
    maximum_error = 0.0
    for path in result.paths:
        metadata = path.metadata
        output = o3.Irrep(metadata.output_irrep)
        left, right, third = (
            o3.Irrep(label) for label in metadata.input_irreps
        )
        intermediate = o3.Irrep(metadata.intermediate_irreps[0])
        first_cg = _component_cg(intermediate, left, right)
        second_cg = _component_cg(output, intermediate, third)

        # Independent orientation oracle: contract with tensordot, then move the
        # third input component behind the first two input components.
        local = torch.tensordot(
            second_cg, first_cg, dims=([1], [0])
        ).permute(0, 2, 3, 1)
        oracle = torch.zeros_like(path.coefficient)
        oracle[
            (
                slice(None),
                slices[str(left)],
                slices[str(right)],
                slices[str(third)],
            )
        ] = local
        maximum_error = max(
            maximum_error,
            float(torch.max(torch.abs(path.coefficient - oracle))),
        )
    assert maximum_error <= 2.0e-16
    assert [(item.output_irrep, item.path_count) for item in result.outputs] == [
        ("0e", 11),
        ("1o", 21),
        ("2e", 23),
    ]
    inputs = o3.Irreps(ANGULAR_IRREPS)
    for _, output in inputs:
        observed = [
            (path.metadata.input_irreps, path.metadata.intermediate_irreps)
            for path in result.paths_for(str(output))
        ]
        assert observed == _allowed_path_identities(inputs, output, 3)


def test_order_three_preserves_intermediate_larger_than_final_lmax():
    result = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, 3)
    matching = [
        path
        for path in result.paths_for("2e")
        if path.metadata.input_irreps == ("2e", "2e", "2e")
        and path.metadata.intermediate_irreps == ("4e",)
    ]
    assert len(matching) == 1
    assert matching[0].metadata.nonzero
    assert bool(torch.any(matching[0].coefficient != 0.0))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_same_input_polynomial_is_invariant_to_factor_axis_permutation(order):
    value = torch.linspace(-0.8, 1.1, 9, dtype=torch.float64)
    result = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, order)
    for path in result.paths:
        coefficient = path.coefficient
        expected = _contract(coefficient, value, order)
        for permutation in itertools.permutations(range(order)):
            axes = (0,) + tuple(index + 1 for index in permutation)
            actual = _contract(coefficient.permute(axes), value, order)
            assert torch.allclose(actual, expected, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize(
    "matrix",
    [_rotation_matrix(), -torch.eye(3, dtype=torch.float64)],
    ids=["proper-rotation", "inversion"],
)
def test_real_o3_equivariance_for_rotation_and_inversion(order, matrix):
    o3 = _o3()
    irreps = o3.Irreps(ANGULAR_IRREPS)
    input_matrix = _representation_matrix(irreps, matrix)
    value = torch.linspace(-0.7, 0.9, irreps.dim, dtype=torch.float64)
    transformed = input_matrix @ value
    maximum_error = 0.0
    for path in generate_generalized_cg(irreps, irreps, order).paths:
        output_irrep = o3.Irrep(path.metadata.output_irrep)
        output_matrix = output_irrep.D_from_matrix(matrix)
        before = _contract(path.coefficient, value, order)
        after = _contract(path.coefficient, transformed, order)
        error = torch.max(torch.abs(after - output_matrix @ before))
        maximum_error = max(maximum_error, float(error))
    assert maximum_error <= 3.0e-14


def test_scalar_output_is_rotation_invariant():
    o3 = _o3()
    irreps = o3.Irreps(ANGULAR_IRREPS)
    rotation = _rotation_matrix()
    transform = _representation_matrix(irreps, rotation)
    value = torch.linspace(-0.9, 0.6, irreps.dim, dtype=torch.float64)
    for order in (1, 2, 3):
        result = generate_generalized_cg(irreps, "0e", order)
        for path in result.paths:
            original = _contract(path.coefficient, value, order)
            rotated = _contract(path.coefficient, transform @ value, order)
            assert torch.allclose(rotated, original, rtol=0.0, atol=2.0e-14)


def test_generation_is_exact_deterministic_and_caller_cannot_mutate_storage():
    first = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, 3)
    second = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, 3)
    assert first.input_blocks == second.input_blocks
    assert first.outputs == second.outputs
    assert [path.metadata for path in first.paths] == [
        path.metadata for path in second.paths
    ]
    assert all(
        torch.equal(left.coefficient, right.coefficient)
        for left, right in zip(first.paths, second.paths)
    )

    snapshot = first.paths[0].coefficient
    exposed = first.paths[0].coefficient
    exposed.add_(100.0)
    assert torch.equal(first.paths[0].coefficient, snapshot)
    materialized = first.paths[0].materialize(dtype=torch.float32)
    assert materialized.dtype == torch.float32
    assert materialized.device.type == "cpu"
    assert torch.equal(
        first.paths[0].coefficient,
        snapshot,
    )

    source = first.paths[0].coefficient
    owned = GeneralizedCGPath(first.paths[0].metadata, source)
    owned_snapshot = owned.coefficient
    source.zero_()
    assert torch.equal(owned.coefficient, owned_snapshot)


def test_generation_preserves_rng_default_dtype_and_execution_modes(monkeypatch):
    o3 = _o3()
    caller_irreps = o3.Irreps(ANGULAR_IRREPS)
    caller_text = str(caller_irreps)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    grad_enabled = torch.is_grad_enabled()
    inference_enabled = torch.is_inference_mode_enabled()

    def forbidden_cuda_rng(*args, **kwargs):
        del args, kwargs
        raise AssertionError("generalized-CG generation touched CUDA RNG")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden_cuda_rng)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden_cuda_rng)
    generate_generalized_cg(caller_irreps, caller_irreps, 3)

    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.get_default_dtype() == default_dtype
    assert torch.is_grad_enabled() == grad_enabled
    assert torch.is_inference_mode_enabled() == inference_enabled
    assert str(caller_irreps) == caller_text


def test_generation_does_not_change_nondefault_dtype_or_inference_context():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.inference_mode():
            assert torch.is_inference_mode_enabled()
            result = generate_generalized_cg(ANGULAR_IRREPS, "0e", 2)
            assert torch.is_inference_mode_enabled()
            assert all(path.coefficient.dtype == torch.float64 for path in result.paths)
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(original)


def test_synthetic_contraction_supports_gradcheck_and_gradgradcheck():
    result = generate_generalized_cg(ANGULAR_IRREPS, "0e", 3)
    path = next(
        path
        for path in result.paths
        if path.metadata.input_irreps == ("0e", "2e", "2e")
    )
    coefficient = path.coefficient
    value = torch.linspace(-0.4, 0.8, 9, dtype=torch.float64).requires_grad_()

    def function(argument):
        return _contract(coefficient, argument, 3)

    output = function(value)
    gradient = torch.autograd.grad(output.sum(), value, create_graph=True)[0]
    assert bool(torch.all(torch.isfinite(gradient)))
    assert torch.autograd.gradcheck(function, (value,), eps=1.0e-6, atol=1.0e-6)
    assert torch.autograd.gradgradcheck(
        function, (value,), eps=1.0e-6, atol=1.0e-6
    )


@pytest.mark.parametrize("order", [True, False, 0, 4, 1.5])
def test_invalid_correlation_order_is_rejected(order):
    with pytest.raises(SymmetricCGError, match="CORRELATION_ORDER"):
        generate_generalized_cg(ANGULAR_IRREPS, "0e", order)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input", ""),
        ("output", ""),
        ("input", "not-an-irrep"),
        ("output", "1q"),
        ("input", "2x0e + 1o"),
        ("output", "0e + 0e"),
    ],
)
def test_invalid_empty_or_multiplicity_bearing_irreps_are_rejected(field, value):
    input_irreps = value if field == "input" else ANGULAR_IRREPS
    output_irreps = value if field == "output" else "0e"
    with pytest.raises(SymmetricCGError):
        generate_generalized_cg(input_irreps, output_irreps, 2)


def test_invalid_normalization_and_canonical_dtype_are_rejected():
    with pytest.raises(SymmetricCGError, match="UNSUPPORTED_NORMALIZATION"):
        generate_generalized_cg(ANGULAR_IRREPS, "0e", 2, normalization="norm")
    with pytest.raises(SymmetricCGError, match="UNSUPPORTED_CANONICAL_DTYPE"):
        generate_generalized_cg(
            ANGULAR_IRREPS, "0e", 2, canonical_dtype=torch.float32
        )


def test_path_rejects_nonfinite_shape_and_zero_status_injection():
    path = generate_generalized_cg(ANGULAR_IRREPS, "0e", 2).paths[0]
    coefficient = path.coefficient
    coefficient.reshape(-1)[0] = float("nan")
    with pytest.raises(SymmetricCGError, match="NONFINITE_COEFFICIENT"):
        GeneralizedCGPath(path.metadata, coefficient)

    with pytest.raises(SymmetricCGError, match="COEFFICIENT_SHAPE_MISMATCH"):
        GeneralizedCGPath(path.metadata, torch.zeros(1, dtype=torch.float64))

    zero = torch.zeros_like(path.coefficient)
    with pytest.raises(SymmetricCGError, match="ZERO_STATUS_MISMATCH"):
        GeneralizedCGPath(path.metadata, zero)

    wrong_metadata = replace(path.metadata, nonzero=False)
    with pytest.raises(SymmetricCGError, match="ZERO_STATUS_MISMATCH"):
        GeneralizedCGPath(wrong_metadata, path.coefficient)


def test_canonical_shapes_counts_and_angular_only_storage():
    expected = {
        1: (3, 648),
        2: (11, 22_680),
        3: (55, 1_102_248),
    }
    for order, (path_count, byte_count) in expected.items():
        result = generate_generalized_cg(ANGULAR_IRREPS, ANGULAR_IRREPS, order)
        assert result.path_count == path_count
        assert result.nonzero_path_count == path_count
        assert result.total_coefficient_bytes == byte_count
        assert all(
            path.metadata.coefficient_shape
            == (int(path.metadata.output_irrep[0]) * 2 + 1,)
            + (9,) * order
            for path in result.paths
        )


def test_path_order_is_explicit_and_deterministic():
    result = generate_generalized_cg(ANGULAR_IRREPS, "2e", 3)
    identities = [
        (
            path.metadata.path_index,
            path.metadata.input_irreps,
            path.metadata.intermediate_irreps,
        )
        for path in result.paths
    ]
    assert [index for index, _, _ in identities] == list(range(len(identities)))
    assert identities == sorted(
        identities,
        key=lambda item: (
            tuple(
                next(
                    block_index
                    for block_index, block in enumerate(result.input_blocks)
                    if block.irrep == label
                )
                for label in item[1]
            ),
            tuple((_o3().Irrep(label).l, _o3().Irrep(label).p) for label in item[2]),
        ),
    )
