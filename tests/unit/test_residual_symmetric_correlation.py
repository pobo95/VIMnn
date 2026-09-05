from __future__ import annotations

import pytest
import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.interactions import (
    HigherBodyConfig,
    SymmetricCGBasisBank,
    SymmetricCorrelationConfig,
    squared_edge_radial_basis,
)
from refsite_mlip.interactions.higher_body import (
    SYMMETRIC_POWER_CONTRACT_VERSION,
)
from refsite_mlip.interactions.symmetric_cg import SymmetricCGError
from refsite_mlip.interactions.symmetric_contraction import (
    FactorizedSymmetricContraction,
    SymmetricContractionError,
)
from refsite_mlip.models.residual_block import ResidualInteractionBlock


ANGULAR = "0e + 1o + 2e"
CENTRAL_IRREPS = "8x0e"


def _o3():
    return import_e3nn_0_4_4()[1]


def _config(order: int, *, channels: int = 2) -> HigherBodyConfig:
    hidden = f"{channels}x0e+{channels}x1o+{channels}x2e"
    return HigherBodyConfig(
        hidden,
        2,
        2,
        site_type_embedding_dim=2,
        n_correlation_channels=channels,
        lmax=2,
        radial_feature_dim=3,
        radial_hidden_dims=(5,),
        avg_num_neighbors=2.0,
        cutoff=3.0,
        correlation_mode=None,
        contract_version=SYMMETRIC_POWER_CONTRACT_VERSION,
        symmetric_correlation=SymmetricCorrelationConfig(order),
    )


def _bank(order: int, *, dtype=torch.float64, device="cpu"):
    return SymmetricCGBasisBank(
        ANGULAR,
        ANGULAR,
        order,
        dtype=dtype,
        device=device,
    )


def _block(order: int, *, channels=2, dtype=torch.float64, device="cpu"):
    config = _config(order, channels=channels)
    bank = _bank(order, dtype=dtype, device=device)
    block = ResidualInteractionBlock(
        config.irreps_feature,
        CENTRAL_IRREPS,
        config,
        residual_scale=0.5,
        basis_bank=bank,
    ).to(device=device, dtype=dtype)
    return block, bank


def _inputs(*, channels=2, dtype=torch.float64, device="cpu", requires=False):
    sites = 3
    hidden_dimension = channels * 9
    h = torch.linspace(
        -0.35, 0.55, sites * hidden_dimension, dtype=dtype, device=device
    ).reshape(sites, hidden_dimension)
    c_bar = torch.tensor(
        [
            [1.0, 0.8, 0.2, 0.0, 0.3, -0.4, 0.0, 0.0],
            [1.0, 0.3, 0.7, 0.4, -0.2, 0.6, -0.08, 0.24],
            [1.0, 0.6, 0.4, 0.9, 0.5, 0.2, 0.45, 0.18],
        ],
        dtype=dtype,
        device=device,
    )
    if requires:
        h = h.clone().requires_grad_()
        c_bar = c_bar.clone().requires_grad_()
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
        dtype=torch.long,
        device=device,
    )
    edge_vectors = torch.tensor(
        [
            [1.0, 0.1, 0.2],
            [-1.0, -0.1, -0.2],
            [0.2, 1.1, -0.1],
            [-0.2, -1.1, 0.1],
            [-0.8, 0.3, 1.0],
            [0.8, -0.3, -1.0],
        ],
        dtype=dtype,
        device=device,
    )
    edge_radial = squared_edge_radial_basis(
        torch.sum(edge_vectors * edge_vectors, dim=-1), 3
    )
    edge_cutoff = torch.full((6,), 0.73, dtype=dtype, device=device)
    return h, c_bar, edge_index, edge_vectors, edge_radial, edge_cutoff


def _pack(contraction, density):
    pieces = []
    start = 0
    for multiplicity, irrep in contraction.input_irreps:
        stop = start + multiplicity * irrep.dim
        pieces.append(
            density[:, start:stop].reshape(
                density.shape[0], multiplicity, irrep.dim
            )
        )
        start = stop
    return torch.cat(pieces, dim=-1)


def _direct(contraction, bank, density, central, *, order=None):
    packed = _pack(contraction, density)
    orders = (
        range(1, contraction.correlation_order + 1)
        if order is None
        else (order,)
    )
    blocks = []
    for output_index, (_, output_irrep) in enumerate(
        contraction.requested_output_irreps
    ):
        total = None
        for current in orders:
            basis = bank.basis_tensor(current, str(output_irrep))
            weight = contraction.weight_parameter(output_index, current)
            if current == 1:
                value = torch.einsum(
                    "sq,qpk,poa,ska->sko",
                    central,
                    weight,
                    basis,
                    packed,
                )
            elif current == 2:
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


@pytest.mark.parametrize("order", [1, 2, 3])
def test_v2_residual_includes_every_order_and_matches_direct_oracle(order):
    torch.manual_seed(20 + order)
    block, bank = _block(order)
    inputs = _inputs()
    _, details = block(*inputs, symmetric_cg_basis=bank)
    density = details["A"]
    expected_orders = tuple(
        _direct(block.symmetric_contraction, bank, density, inputs[1], order=value)
        for value in range(1, order + 1)
    )
    expected = sum(expected_orders)
    torch.testing.assert_close(
        details["symmetric_output"], expected, rtol=3.0e-15, atol=3.0e-15
    )
    assert details["correlation_order"] == order
    assert not details["dense_A_outer_materialized"]
    parameter_names = tuple(name for name, _ in block.named_parameters())
    for active in range(1, order + 1):
        assert any(
            f"symmetric_contraction.weight_output_0_order_{active}" == name
            for name in parameter_names
        )
    for inactive in range(order + 1, 4):
        assert not any(
            f"symmetric_contraction.weight_output_0_order_{inactive}" == name
            for name in parameter_names
        )


def test_external_basis_matches_standalone_full_path_forward_and_gradients():
    torch.manual_seed(31)
    block, bank = _block(3)
    external = block.symmetric_contraction
    internal = FactorizedSymmetricContraction(
        external.input_irreps,
        ANGULAR,
        correlation_order=3,
        central_dimension=8,
        dtype=torch.float64,
    )
    with torch.no_grad():
        for output_index in range(3):
            for order in range(1, 4):
                internal.weight_parameter(output_index, order).copy_(
                    external.weight_parameter(output_index, order)
                )
    density = torch.linspace(-0.25, 0.4, 54, dtype=torch.float64).reshape(3, 18)
    central = _inputs()[1]
    density_external = density.clone().requires_grad_()
    central_external = central.clone().requires_grad_()
    density_internal = density.clone().requires_grad_()
    central_internal = central.clone().requires_grad_()
    output_external = external(
        density_external, central_external, basis_bank=bank
    ).output
    output_internal = internal(density_internal, central_internal).output
    torch.testing.assert_close(
        output_external, output_internal, rtol=0.0, atol=0.0
    )
    probe = torch.linspace(
        -0.7, 0.8, output_external.numel(), dtype=torch.float64
    ).reshape_as(output_external)
    external_args = (
        density_external,
        central_external,
        *tuple(external.parameters()),
    )
    internal_args = (
        density_internal,
        central_internal,
        *tuple(internal.parameters()),
    )
    external_gradients = torch.autograd.grad(
        torch.sum(output_external * probe), external_args
    )
    internal_gradients = torch.autograd.grad(
        torch.sum(output_internal * probe), internal_args
    )
    for actual, expected in zip(external_gradients, internal_gradients):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert bool(torch.all(torch.isfinite(actual)))


@pytest.mark.parametrize(
    "matrix",
    [
        _o3().angles_to_matrix(
            torch.tensor(0.31, dtype=torch.float64),
            torch.tensor(0.47, dtype=torch.float64),
            torch.tensor(-0.23, dtype=torch.float64),
        ),
        -torch.eye(3, dtype=torch.float64),
    ],
    ids=["proper-rotation", "inversion"],
)
def test_v2_residual_is_o3_equivariant_and_edge_order_invariant(matrix):
    torch.manual_seed(41)
    block, bank = _block(3)
    inputs = _inputs()
    original, original_details = block(*inputs, symmetric_cg_basis=bank)
    hidden_transform = block.irreps_h.D_from_matrix(matrix)
    transformed_inputs = list(inputs)
    transformed_inputs[0] = inputs[0] @ hidden_transform.T
    transformed_inputs[3] = inputs[3] @ matrix.T
    transformed, transformed_details = block(
        *transformed_inputs, symmetric_cg_basis=bank
    )
    torch.testing.assert_close(
        transformed,
        original @ hidden_transform.T,
        rtol=0.0,
        atol=5.0e-9,
    )
    torch.testing.assert_close(
        transformed_details["symmetric_output"],
        original_details["symmetric_output"] @ hidden_transform.T,
        rtol=0.0,
        atol=5.0e-9,
    )

    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    permuted_inputs = list(inputs)
    permuted_inputs[2] = inputs[2][:, permutation]
    permuted_inputs[3] = inputs[3][permutation]
    permuted_inputs[4] = inputs[4][permutation]
    permuted_inputs[5] = inputs[5][permutation]
    permuted, permuted_details = block(
        *permuted_inputs, symmetric_cg_basis=bank
    )
    torch.testing.assert_close(permuted, original, rtol=0.0, atol=2.0e-14)
    torch.testing.assert_close(
        permuted_details["symmetric_output"],
        original_details["symmetric_output"],
        rtol=0.0,
        atol=2.0e-14,
    )


def test_v2_residual_gradcheck_gradgradcheck_and_all_central_groups_connect():
    torch.manual_seed(52)
    block, bank = _block(3, channels=1)
    base = list(_inputs(channels=1))
    h = (base[0] * 0.2).requires_grad_()
    central = base[1].clone().requires_grad_()

    def function(hidden_value, central_value):
        values = list(base)
        values[0] = hidden_value
        values[1] = central_value
        return block(*values, symmetric_cg_basis=bank)[0]

    assert torch.autograd.gradcheck(
        function, (h, central), eps=1.0e-6, atol=4.0e-6, rtol=4.0e-5
    )
    assert torch.autograd.gradgradcheck(
        function, (h, central), eps=1.0e-6, atol=8.0e-6, rtol=8.0e-5
    )
    output = function(h, central)
    probe = torch.linspace(-0.4, 0.6, output.numel(), dtype=torch.float64).reshape_as(
        output
    )
    gradients = torch.autograd.grad(
        torch.sum(output * probe),
        (h, central, *tuple(block.symmetric_contraction.parameters())),
    )
    assert all(bool(torch.all(torch.isfinite(value))) for value in gradients)
    assert bool(torch.any(gradients[0] != 0.0))
    density = torch.linspace(-0.3, 0.4, 27, dtype=torch.float64).reshape(3, 9)
    conditioned = central.detach().clone().requires_grad_()
    symmetric_output = block.symmetric_contraction(
        density, conditioned, basis_bank=bank
    ).output
    central_gradient = torch.autograd.grad(
        torch.sum(symmetric_output * probe), conditioned
    )[0]
    groups = (
        slice(0, 1),
        slice(1, 3),
        slice(3, 4),
        slice(4, 6),
        slice(6, 8),
    )
    assert all(
        bool(torch.any(central_gradient[:, group] != 0.0)) for group in groups
    )
    assert all(bool(torch.any(value != 0.0)) for value in gradients[2:])


def test_basis_has_single_owner_and_residual_layers_own_distinct_weights():
    torch.manual_seed(63)
    config = _config(3)
    bank = _bank(3)
    first = ResidualInteractionBlock(
        config.irreps_feature, CENTRAL_IRREPS, config, 0.5, basis_bank=bank
    ).double()
    second = ResidualInteractionBlock(
        config.irreps_feature, CENTRAL_IRREPS, config, 0.5, basis_bank=bank
    ).double()
    assert tuple(bank.state_dict())
    for block in (first, second):
        assert not any(
            "symmetric_contraction.u_output_" in name
            or "symmetric_cg_basis" in name
            or name.startswith("U_order_")
            for name in block.state_dict()
        )
        assert tuple(block.symmetric_contraction.buffers()) == ()
        assert block.symmetric_basis_fingerprint == bank.basis_fingerprint
    first_weights = tuple(first.symmetric_contraction.parameters())
    second_weights = tuple(second.symmetric_contraction.parameters())
    assert len(first_weights) == len(second_weights) == 9
    assert all(left is not right for left, right in zip(first_weights, second_weights))
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(first_weights, second_weights)
    )

    clone = ResidualInteractionBlock(
        config.irreps_feature,
        CENTRAL_IRREPS,
        config,
        0.5,
        basis_bank=bank,
    ).double()
    clone.load_state_dict(first.state_dict(), strict=True)
    output, _ = first(*_inputs(), symmetric_cg_basis=bank)
    clone_output, _ = clone(*_inputs(), symmetric_cg_basis=bank)
    torch.testing.assert_close(clone_output, output, rtol=0.0, atol=0.0)

    with pytest.raises(SymmetricContractionError, match="EXTERNAL_CG_BASIS_REQUIRED"):
        first(*_inputs())
    mismatched = _bank(2)
    with pytest.raises(SymmetricContractionError, match="MISMATCHED_CG_BASIS_BANK"):
        first(*_inputs(), symmetric_cg_basis=mismatched)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v2_residual_cpu_dtype_finite_and_basis_fingerprint_stable(dtype):
    torch.manual_seed(74)
    block, bank = _block(3, dtype=dtype)
    fingerprint = bank.basis_fingerprint
    output, details = block(
        *_inputs(dtype=dtype), symmetric_cg_basis=bank
    )
    assert output.dtype == dtype and output.device.type == "cpu"
    assert bool(torch.all(torch.isfinite(output)))
    assert details["basis_fingerprint"] == fingerprint
    bank.validate_integrity()
    other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    other = _bank(3, dtype=other_dtype)
    assert other.basis_fingerprint == fingerprint
    other.validate_integrity()


def test_v2_residual_rejects_wrong_central_layout_and_detects_basis_mutation():
    config = _config(3)
    bank = _bank(3)
    with pytest.raises(ValueError, match="constant, species, vacancy"):
        ResidualInteractionBlock(
            config.irreps_feature,
            "7x0e",
            config,
            0.5,
            basis_bank=bank,
        )
    with torch.no_grad():
        bank.basis_tensor(3, "2e").reshape(-1)[0] += 1.0
    with pytest.raises(SymmetricCGError, match="BASIS_BUFFER_MISMATCH"):
        bank.validate_integrity()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v2_residual_cuda_smoke(dtype):
    torch.manual_seed(85)
    block, bank = _block(3, dtype=dtype, device="cuda:0")
    output, details = block(
        *_inputs(dtype=dtype, device="cuda:0"),
        symmetric_cg_basis=bank,
    )
    torch.cuda.synchronize()
    assert output.device.type == "cuda" and output.dtype == dtype
    assert bool(torch.all(torch.isfinite(output)))
    assert details["basis_fingerprint"] == bank.basis_fingerprint
    assert all(
        parameter.device.type == "cuda" and parameter.dtype == dtype
        for parameter in block.parameters()
    )
    assert all(
        buffer.device.type == "cuda" and buffer.dtype == dtype
        for buffer in bank.buffers()
    )
