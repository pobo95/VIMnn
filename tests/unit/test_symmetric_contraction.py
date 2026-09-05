from __future__ import annotations

from dataclasses import fields
import math

import pytest
import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.interactions.symmetric_contraction import (
    FactorizedSymmetricContraction,
    SymmetricContractionError,
)


ANGULAR = "0e + 1o + 2e"


def _o3():
    return import_e3nn_0_4_4()[1]


def _pack(module, value):
    pieces = []
    start = 0
    for multiplicity, irrep in module.input_irreps:
        stop = start + multiplicity * irrep.dim
        pieces.append(value[:, start:stop].reshape(value.shape[0], multiplicity, irrep.dim))
        start = stop
    return torch.cat(pieces, dim=-1)


def _naive(module, density, central, *, order=None):
    packed = _pack(module, density)
    orders = range(1, module.correlation_order + 1) if order is None else (order,)
    blocks = []
    for output_index in range(len(module.requested_output_irreps)):
        total = None
        for current_order in orders:
            basis = module.basis_tensor(output_index, current_order)
            weight = module.weight_parameter(output_index, current_order)
            if current_order == 1:
                value = torch.einsum(
                    "sq,qpk,poa,ska->sko", central, weight, basis, packed
                )
            elif current_order == 2:
                value = torch.einsum(
                    "sq,qpk,poab,ska,skb->sko",
                    central,
                    weight,
                    basis,
                    packed,
                    packed,
                )
            else:
                value = torch.einsum(
                    "sq,qpk,poabc,ska,skb,skc->sko",
                    central,
                    weight,
                    basis,
                    packed,
                    packed,
                    packed,
                )
            total = value if total is None else total + value
        blocks.append(total.reshape(total.shape[0], -1))
    return torch.cat(blocks, dim=-1)


def _rotation():
    return _o3().angles_to_matrix(
        torch.tensor(0.31, dtype=torch.float64),
        torch.tensor(0.47, dtype=torch.float64),
        torch.tensor(-0.23, dtype=torch.float64),
    )


@pytest.mark.parametrize("correlation_order", [1, 2, 3])
def test_scalar_oracle_includes_every_order_through_requested_correlation(
    correlation_order,
):
    module = FactorizedSymmetricContraction(
        "2x0e",
        "0e",
        correlation_order=correlation_order,
        central_dimension=3,
        dtype=torch.float64,
    )
    density = torch.tensor([[0.2, -0.3], [0.7, 0.4]], dtype=torch.float64)
    central = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.2, 0.6, 1.1]], dtype=torch.float64
    )
    weights = {
        1: torch.tensor(
            [[[0.3, -0.2]], [[0.7, 0.1]], [[-0.4, 0.8]]],
            dtype=torch.float64,
        ),
        2: torch.tensor(
            [[[0.5, 0.4]], [[-0.1, 0.9]], [[0.2, -0.3]]],
            dtype=torch.float64,
        ),
        3: torch.tensor(
            [[[-0.2, 0.6]], [[0.8, -0.7]], [[0.3, 0.5]]],
            dtype=torch.float64,
        ),
    }
    with torch.no_grad():
        for order in range(1, correlation_order + 1):
            module.weight_parameter(0, order).copy_(weights[order])

    result = module(density, central, return_order_contributions=True)
    expected_orders = []
    for order in range(1, correlation_order + 1):
        theta = torch.einsum("sq,qpk->sk", central, weights[order])
        expected_orders.append(theta * density.pow(order))
    assert result.order_contributions is not None
    for actual, expected in zip(result.order_contributions, expected_orders):
        assert torch.allclose(actual, expected, rtol=0.0, atol=2.0e-17)
    assert torch.allclose(result.output, sum(expected_orders), rtol=0.0, atol=8.0e-17)


def test_factorized_horner_matches_explicit_outer_oracle_and_gradients():
    torch.manual_seed(91)
    module = FactorizedSymmetricContraction(
        "2x0e + 2x1o + 2x2e",
        ANGULAR,
        correlation_order=3,
        central_dimension=3,
        dtype=torch.float64,
    )
    density = torch.randn(3, 18, dtype=torch.float64, requires_grad=True)
    central = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    density_snapshot = density.detach().clone()
    central_snapshot = central.detach().clone()

    result = module(density, central, return_order_contributions=True)
    oracle = _naive(module, density, central)
    assert torch.allclose(result.output, oracle, rtol=3.0e-15, atol=3.0e-15)
    assert result.order_contributions is not None
    for order, contribution in enumerate(result.order_contributions, start=1):
        assert torch.allclose(
            contribution,
            _naive(module, density, central, order=order),
            rtol=3.0e-15,
            atol=3.0e-15,
        )

    probe = torch.linspace(-0.4, 0.7, result.output.numel(), dtype=torch.float64).reshape_as(
        result.output
    )
    arguments = (density, central, *tuple(module.parameters()))
    actual_gradients = torch.autograd.grad(
        (result.output * probe).sum(), arguments, retain_graph=True
    )
    oracle_gradients = torch.autograd.grad((oracle * probe).sum(), arguments)
    for actual, expected in zip(actual_gradients, oracle_gradients):
        assert torch.allclose(actual, expected, rtol=3.0e-13, atol=3.0e-13)
        assert bool(torch.all(torch.isfinite(actual)))

    # Reordering equal density factors cannot change the symmetric polynomial.
    packed = _pack(module, density)
    basis = module.basis_tensor(2, 3)
    weight = module.weight_parameter(2, 3)
    original_order = torch.einsum(
        "sq,qpk,poabc,ska,skb,skc->sko",
        central,
        weight,
        basis,
        packed,
        packed,
        packed,
    )
    permuted_order = torch.einsum(
        "sq,qpk,pocba,ska,skb,skc->sko",
        central,
        weight,
        basis,
        packed,
        packed,
        packed,
    )
    assert torch.allclose(original_order, permuted_order, rtol=0.0, atol=8.0e-16)
    assert torch.equal(density.detach(), density_snapshot)
    assert torch.equal(central.detach(), central_snapshot)
    assert not result.diagnostics.dense_A_outer_materialized
    assert max(len(shape) for shape in result.diagnostics.horner_intermediate_shapes) == 5


@pytest.mark.parametrize(
    "matrix",
    [_rotation(), -torch.eye(3, dtype=torch.float64)],
    ids=["proper-rotation", "inversion"],
)
def test_o3_equivariance_and_site_permutation(matrix):
    torch.manual_seed(17)
    module = FactorizedSymmetricContraction(
        "2x0e + 2x1o + 2x2e",
        ANGULAR,
        correlation_order=3,
        central_dimension=2,
        dtype=torch.float64,
    )
    density = torch.randn(4, 18, dtype=torch.float64)
    central = torch.randn(4, 2, dtype=torch.float64)
    input_transform = module.input_irreps.D_from_matrix(matrix)
    output_transform = module.output_irreps.D_from_matrix(matrix)
    original = module(density, central).output
    transformed = module(density @ input_transform.T, central).output
    assert torch.allclose(
        transformed, original @ output_transform.T, rtol=0.0, atol=1.5e-13
    )

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = module(density[permutation], central[permutation]).output
    assert torch.allclose(permuted, original[permutation], rtol=0.0, atol=2.0e-14)

    scalar = FactorizedSymmetricContraction(
        "2x0e + 2x1o + 2x2e",
        "0e",
        correlation_order=3,
        central_dimension=2,
        dtype=torch.float64,
    )
    scalar.load_state_dict(
        {
            key: value
            for key, value in module.state_dict().items()
            if "output_0_" in key
        },
        strict=True,
    )
    before = scalar(density, central).output
    after = scalar(density @ input_transform.T, central).output
    assert torch.allclose(after, before, rtol=0.0, atol=1.5e-13)


def test_correlation_order_path_and_parameter_counts_are_exact():
    expected_paths = {
        1: {"0e": 1, "1o": 1, "2e": 1},
        2: {"0e": 3, "1o": 4, "2e": 4},
        3: {"0e": 11, "1o": 21, "2e": 23},
    }
    for maximum_order in (1, 2, 3):
        module = FactorizedSymmetricContraction(
            "3x0e + 3x1o + 3x2e",
            ANGULAR,
            correlation_order=maximum_order,
            central_dimension=4,
            dtype=torch.float64,
        )
        observed = {
            (item.order, item.output_irrep): item.path_count
            for item in module(
                torch.zeros(1, 27, dtype=torch.float64),
                torch.zeros(1, 4, dtype=torch.float64),
            ).diagnostics.path_counts
        }
        for order in range(1, maximum_order + 1):
            for output, count in expected_paths[order].items():
                assert observed[(order, output)] == count
        expected_parameters = 4 * 3 * sum(
            sum(expected_paths[order].values())
            for order in range(1, maximum_order + 1)
        )
        assert sum(parameter.numel() for parameter in module.parameters()) == expected_parameters
        assert all(
            f"order_{order}" not in name
            for name, _ in module.named_parameters()
            for order in range(maximum_order + 1, 4)
        )


def test_default_forward_does_not_compute_duplicate_order_contributions(monkeypatch):
    module = FactorizedSymmetricContraction(
        ANGULAR,
        ANGULAR,
        correlation_order=3,
        central_dimension=2,
        dtype=torch.float64,
    )

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("pure-order diagnostic contraction was executed")

    monkeypatch.setattr(module, "_pure_order_output", forbidden)
    result = module(
        torch.zeros(1, 9, dtype=torch.float64),
        torch.zeros(1, 2, dtype=torch.float64),
    )
    assert result.order_contributions is None


def test_gradcheck_gradgradcheck_and_active_weight_gradients():
    torch.manual_seed(123)
    module = FactorizedSymmetricContraction(
        ANGULAR,
        "0e + 1o + 2e",
        correlation_order=3,
        central_dimension=2,
        dtype=torch.float64,
    )
    density = (torch.randn(1, 9, dtype=torch.float64) * 0.15).requires_grad_()
    central = torch.tensor([[0.7, -0.4]], dtype=torch.float64, requires_grad=True)

    def function(density_value, central_value):
        return module(density_value, central_value).output

    assert torch.autograd.gradcheck(function, (density, central), eps=1.0e-6, atol=2.0e-6)
    assert torch.autograd.gradgradcheck(
        function, (density, central), eps=1.0e-6, atol=3.0e-6
    )
    loss = function(density, central).square().sum()
    gradients = torch.autograd.grad(loss, (density, central, *tuple(module.parameters())))
    assert all(bool(torch.all(torch.isfinite(gradient))) for gradient in gradients)
    assert bool(torch.any(gradients[0] != 0.0))
    assert bool(torch.any(gradients[1] != 0.0))
    assert all(bool(torch.any(gradient != 0.0)) for gradient in gradients[2:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_state_roundtrip_seeded_determinism_buffers_and_ownership(dtype):
    arguments = dict(
        input_irreps="2x0e + 2x1o + 2x2e",
        output_irreps=ANGULAR,
        correlation_order=3,
        central_dimension=3,
        dtype=dtype,
    )
    torch.manual_seed(44)
    first = FactorizedSymmetricContraction(**arguments)
    torch.manual_seed(44)
    second = FactorizedSymmetricContraction(**arguments)
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(left, second.state_dict()[name])
        for name, left in first.state_dict().items()
    )
    third = FactorizedSymmetricContraction(**arguments)
    third.load_state_dict(first.state_dict(), strict=True)
    assert all(
        torch.equal(left, third.state_dict()[name])
        for name, left in first.state_dict().items()
    )
    assert all(not buffer.requires_grad for buffer in first.buffers())
    parameter_pointers = {parameter.data_ptr() for parameter in first.parameters()}
    assert all(buffer.data_ptr() not in parameter_pointers for buffer in first.buffers())

    density = torch.linspace(-0.4, 0.5, 36, dtype=dtype).reshape(2, 18)
    central = torch.linspace(0.2, 0.8, 6, dtype=dtype).reshape(2, 3)
    density_snapshot = density.clone()
    central_snapshot = central.clone()
    result = first(density, central, return_order_contributions=True)
    assert result.output.dtype == dtype and result.output.device.type == "cpu"
    assert torch.equal(density, density_snapshot)
    assert torch.equal(central, central_snapshot)
    assert all(not isinstance(getattr(result.diagnostics, field.name), torch.Tensor) for field in fields(result.diagnostics))
    assert result.diagnostics.basis_kind == "full_path"
    assert not result.diagnostics.dense_A_outer_materialized


def test_module_to_materializes_all_floating_state_without_changing_layout():
    module = FactorizedSymmetricContraction(
        ANGULAR,
        ANGULAR,
        correlation_order=3,
        central_dimension=2,
        dtype=torch.float64,
    )
    keys = tuple(module.state_dict())
    module.to(dtype=torch.float32)
    assert tuple(module.state_dict()) == keys
    assert all(parameter.dtype == torch.float32 for parameter in module.parameters())
    assert all(buffer.dtype == torch.float32 for buffer in module.buffers())
    result = module(
        torch.zeros(1, 9, dtype=torch.float32),
        torch.zeros(1, 2, dtype=torch.float32),
    )
    assert result.output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_materialization_smoke(dtype):
    module = FactorizedSymmetricContraction(
        "2x0e + 2x1o + 2x2e",
        ANGULAR,
        correlation_order=3,
        central_dimension=2,
        dtype=dtype,
        device="cuda:0",
    )
    density = torch.linspace(-0.2, 0.4, 36, dtype=dtype, device="cuda:0").reshape(2, 18)
    central = torch.tensor([[0.7, -0.3], [0.2, 0.8]], dtype=dtype, device="cuda:0")
    result = module(density, central)
    torch.cuda.synchronize()
    assert result.output.device.type == "cuda" and result.output.dtype == dtype
    assert bool(torch.all(torch.isfinite(result.output)))
    assert all(parameter.device.type == "cuda" and parameter.dtype == dtype for parameter in module.parameters())
    assert all(buffer.device.type == "cuda" and buffer.dtype == dtype for buffer in module.buffers())


@pytest.mark.parametrize("value", [True, False, 0, 4, 1.5])
def test_invalid_correlation_order(value):
    with pytest.raises(SymmetricContractionError, match="CORRELATION_ORDER|INVALID_INTEGER"):
        FactorizedSymmetricContraction(
            "1x0e", "0e", correlation_order=value, central_dimension=1
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"input_irreps": ""}, "INVALID_INPUT_IRREPS"),
        ({"input_irreps": "2x0e + 1x1o"}, "NONUNIFORM"),
        ({"input_irreps": "1x0e + 1x0e"}, "DUPLICATE_INPUT"),
        ({"input_irreps": "1x3o"}, "ANGULAR_MOMENTUM"),
        ({"input_irreps": "1x1e"}, "PARITY_LAYOUT"),
        ({"output_irreps": ""}, "INVALID_OUTPUT_IRREPS"),
        ({"output_irreps": "0e + 0e"}, "DUPLICATE_OUTPUT"),
        ({"output_irreps": "2x0e"}, "OUTPUT_MULTIPLICITY"),
        ({"output_irreps": "3o"}, "ANGULAR_MOMENTUM"),
        ({"output_irreps": "1e"}, "PARITY_LAYOUT"),
        ({"central_dimension": True}, "INVALID_INTEGER"),
        ({"central_dimension": 0}, "INVALID_INTEGER"),
        ({"normalization": "norm"}, "UNSUPPORTED_NORMALIZATION"),
        ({"dtype": torch.float16}, "UNSUPPORTED_DTYPE"),
    ],
)
def test_constructor_validation(kwargs, match):
    values = dict(
        input_irreps="1x0e + 1x1o + 1x2e",
        output_irreps=ANGULAR,
        correlation_order=3,
        central_dimension=2,
        dtype=torch.float64,
    )
    values.update(kwargs)
    with pytest.raises(SymmetricContractionError, match=match):
        FactorizedSymmetricContraction(**values)


def test_runtime_validation_and_corrupted_basis_state():
    module = FactorizedSymmetricContraction(
        ANGULAR,
        ANGULAR,
        correlation_order=2,
        central_dimension=2,
        dtype=torch.float64,
    )
    density = torch.zeros(2, 9, dtype=torch.float64)
    central = torch.zeros(2, 2, dtype=torch.float64)
    invalid = [
        (torch.zeros(2, 8, dtype=torch.float64), central, "DENSITY_SHAPE"),
        (density, torch.zeros(2, 3, dtype=torch.float64), "CENTRAL_SHAPE"),
        (density.float(), central, "DTYPE_MISMATCH"),
        (density, central.float(), "DTYPE_MISMATCH"),
        (density.clone().index_fill(1, torch.tensor([0]), float("nan")), central, "NONFINITE_INPUT"),
    ]
    for left, right, reason in invalid:
        with pytest.raises(SymmetricContractionError, match=reason):
            module(left, right)

    name = module._name(module._basis_names, 0, 2)
    original = module._buffers[name]
    module._buffers[name] = original[..., :-1]
    with pytest.raises(SymmetricContractionError, match="MISMATCHED_CG_PATH"):
        module(density, central)

    module._buffers[name] = original.clone()
    module._buffers[name].reshape(-1)[0] = float("nan")
    with pytest.raises(SymmetricContractionError, match="NONFINITE_CG_BASIS"):
        module(density, central)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_runtime_device_mismatch_is_structured():
    module = FactorizedSymmetricContraction(
        ANGULAR,
        "0e",
        correlation_order=1,
        central_dimension=1,
        dtype=torch.float64,
        device="cuda:0",
    )
    with pytest.raises(SymmetricContractionError, match="DEVICE_MISMATCH"):
        module(
            torch.zeros(1, 9, dtype=torch.float64),
            torch.zeros(1, 1, dtype=torch.float64),
        )
