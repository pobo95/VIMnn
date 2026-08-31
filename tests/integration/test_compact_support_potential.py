from __future__ import annotations

import copy

import pytest
import torch

from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import PotentialConfig, ReferenceSitePotential
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TransportSupportConfig,
    TransportSupportError,
)


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


def test_compact_potential_rejects_eval_and_preserves_dense_state_contract(typed_crystal):
    compact = _model(typed_crystal, _compact())
    keys = tuple(compact.state_dict())
    parameters = tuple(id(value) for value in compact.parameters())
    with pytest.raises(TransportSupportError) as failure:
        compact(
            typed_crystal["positions"][:5],
            _numbers(typed_crystal),
            typed_crystal["cell"],
            typed_crystal["origin"],
            solver_path=EVAL_ADAPTIVE,
        )
    assert failure.value.reason_code == "COMPACT_EVAL_ADAPTIVE_UNSUPPORTED"
    assert tuple(compact.state_dict()) == keys
    assert tuple(id(value) for value in compact.parameters()) == parameters


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
    assert torch.isfinite(output.forces).all() and torch.isfinite(output.stress).all()
    expected_tolerance = 1e-6 if dtype == torch.float32 else 1e-7
    assert output.auxiliary["transport_support"].effective_diagnostic_tolerance == expected_tolerance
