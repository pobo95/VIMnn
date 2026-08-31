from __future__ import annotations

import pytest
import torch

from refsite_mlip.features import (
    ProbabilityMultipoleConfig,
    build_probability_multipoles,
    build_sparse_probability_multipoles,
)
from refsite_mlip.transport import (
    TRAIN_FIXED,
    TrainSinkhornConfig,
    TransportSupportConfig,
    TransportSupportError,
    atom_site_displacements,
    build_compact_transport_edges,
    materialize_dense_plan,
    solve_atom_vacancy_ot,
    solve_sparse_sinkhorn_train_fixed,
)


def _dense_config():
    return TransportSupportConfig("compact_c2", 2.5, 0.5, 0.2)


def _edge_config():
    return TransportSupportConfig(
        "compact_c2", 2.5, 0.5, 0.2, backend="edge_list"
    )


def _displacements(dtype=torch.float64, device="cpu", requires_grad=False):
    distances = torch.tensor(
        [[0.4, 0.8], [0.7, 2.6], [2.65, 0.6]], dtype=dtype, device=device
    )
    vectors = torch.zeros((3, 2, 3), dtype=dtype, device=device)
    vectors[..., 0] = distances
    return vectors.requires_grad_(requires_grad)


def _solve_sparse(displacements, iterations=256):
    edges = build_compact_transport_edges(
        displacements,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=_edge_config(),
        template_id="edge-fixture",
        sample_id="sample-0",
    )
    return solve_sparse_sinkhorn_train_fixed(edges, TrainSinkhornConfig(iterations))


def _solve_dense(displacements, iterations=256):
    distances = torch.linalg.vector_norm(displacements, dim=-1)
    return solve_atom_vacancy_ot(
        distances.square() / (2.0 * 1.5**2),
        0.5,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations),
        support_config=_dense_config(),
        atom_distances=distances,
    )


def test_edge_backend_config_round_trip_old_payload_and_validation():
    config = _edge_config()
    assert config.backend == "edge_list"
    assert TransportSupportConfig.from_dict(config.to_dict()) == config
    old = config.to_dict()
    del old["backend"]
    assert TransportSupportConfig.from_dict(old).backend == "dense"
    with pytest.raises(TransportSupportError) as failure:
        TransportSupportConfig("dense", backend="edge_list")
    assert failure.value.reason_code == "INVALID_SUPPORT_CONFIG"
    distances = torch.tensor([[0.2], [0.4]], dtype=torch.float64)
    with pytest.raises(TransportSupportError) as failure:
        solve_atom_vacancy_ot(
            distances.square(),
            0.5,
            TRAIN_FIXED,
            "sinkhorn",
            TrainSinkhornConfig(),
            support_config=config,
            atom_distances=distances,
        )
    assert failure.value.reason_code == "EDGE_LIST_REQUIRES_DISPLACEMENTS"


def test_canonical_edges_pointers_exact_mask_and_explicit_materialization():
    result = _solve_sparse(_displacements())
    edges = result.edges
    assert torch.equal(edges.site_index, torch.tensor([0, 0, 1, 1, 2, 2]))
    assert torch.equal(edges.atom_index, torch.tensor([0, 1, 0, 1, 0, 1]))
    assert torch.equal(edges.site_ptr, torch.tensor([0, 2, 4, 6]))
    assert torch.equal(edges.atom_ptr, torch.tensor([0, 3, 6]))
    assert torch.equal(edges.atom_index[edges.atom_major_permutation], torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.equal(edges.active, torch.tensor([True, True, True, False, False, True]))
    assert result.edge_plan[3] == 0.0 and result.edge_plan[4] == 0.0
    assert result.q.shape == (3,) and torch.all(result.q > 0.0)
    assert not result.dense_plan_materialized
    materialized = materialize_dense_plan(result)
    assert materialized.densification_performed
    assert materialized.plan.shape == (3, 2)


def test_edge_builder_reuses_positive_support_feasibility_failures():
    distances = torch.tensor([[0.2, 3.0], [0.3, 3.1]], dtype=torch.float64)
    displacements = torch.zeros((2, 2, 3), dtype=torch.float64)
    displacements[..., 0] = distances
    with pytest.raises(TransportSupportError) as failure:
        build_compact_transport_edges(
            displacements,
            epsilon_ot=0.5,
            ell_ot=1.5,
            config=_edge_config(),
        )
    assert failure.value.reason_code == "ATOM_WITHOUT_SUPPORT"

    hall = torch.tensor(
        [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
        dtype=torch.float64,
    )
    hall_displacements = torch.zeros((4, 3, 3), dtype=torch.float64)
    hall_displacements[..., 0] = hall
    with pytest.raises(TransportSupportError) as failure:
        build_compact_transport_edges(
            hall_displacements,
            epsilon_ot=0.5,
            ell_ot=1.5,
            config=_edge_config(),
        )
    assert failure.value.reason_code == "INCOMPLETE_ATOM_MATCHING"


def test_sparse_fixed_matches_dense_masked_oracle_and_features():
    displacements = _displacements()
    sparse = _solve_sparse(displacements)
    dense = _solve_dense(displacements)
    plan = materialize_dense_plan(sparse).plan
    torch.testing.assert_close(plan, dense.P, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(sparse.q, dense.q, atol=2e-15, rtol=2e-15)
    assert max(float(sparse.row_residual), float(sparse.column_residual)) < 3e-15
    numbers = torch.tensor([6, 41])
    site_types = torch.tensor([0, 1, 0])
    config = ProbabilityMultipoleConfig(
        (6, 41), 2, 2, 1.0, 3.0, site_type_vocabulary=(0, 1)
    )
    dense_features = build_probability_multipoles(
        dense.P, dense.q, numbers, displacements, config, site_types
    )
    sparse_features = build_sparse_probability_multipoles(
        sparse.edge_plan, sparse.q, sparse.edges, numbers, config, site_types
    )
    for name in (
        "species_probabilities",
        "vacancy_probabilities",
        "raw_probability_state",
        "equivariant_features",
    ):
        torch.testing.assert_close(
            getattr(sparse_features, name), getattr(dense_features, name), atol=3e-15, rtol=3e-15
        )
    assert sparse_features.config_metadata["transport_representation"] == "edge_list"
    repeated = _solve_sparse(displacements)
    for name in ("edge_plan", "q", "f", "g", "row_residual", "column_residual"):
        assert torch.equal(getattr(sparse, name), getattr(repeated, name))


def test_edge_list_all_vacancy_reservoir_is_dense_and_analytic():
    displacements = torch.empty((3, 0, 3), dtype=torch.float64)
    result = _solve_sparse(displacements)
    assert result.edge_plan.shape == (0,)
    assert torch.equal(result.q, torch.ones(3, dtype=torch.float64))
    assert result.q.sum() == 3.0
    assert result.row_residual == 0.0 and result.column_residual == 0.0
    features = build_sparse_probability_multipoles(
        result.edge_plan,
        result.q,
        result.edges,
        torch.empty(0, dtype=torch.long),
        ProbabilityMultipoleConfig((6, 41), 1, 2, 1.0, 3.0),
    )
    assert torch.equal(
        features.species_probabilities, torch.zeros((3, 2), dtype=torch.float64)
    )


def _sparse_scalar(displacements):
    result = _solve_sparse(displacements, iterations=96)
    numbers = torch.tensor([6, 41], device=displacements.device)
    config = ProbabilityMultipoleConfig((6, 41), 1, 2, 1.0, 3.0)
    features = build_sparse_probability_multipoles(
        result.edge_plan, result.q, result.edges, numbers, config
    )
    weights = torch.linspace(
        -0.2,
        0.4,
        features.equivariant_features.numel(),
        dtype=displacements.dtype,
        device=displacements.device,
    ).reshape_as(features.equivariant_features)
    return (features.equivariant_features * weights).sum() + 0.17 * result.q.square().sum()


def test_sparse_edge_kernel_sinkhorn_and_features_gradcheck_gradgradcheck():
    displacements = _displacements(requires_grad=True)
    assert torch.autograd.gradcheck(
        _sparse_scalar, (displacements,), eps=1e-6, atol=5e-6, rtol=5e-5
    )
    assert torch.autograd.gradgradcheck(
        _sparse_scalar, (displacements,), eps=1e-6, atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_fixed_dtype_and_marginals(dtype):
    result = _solve_sparse(_displacements(dtype=dtype))
    assert result.edge_plan.dtype == dtype and result.q.dtype == dtype
    assert torch.isfinite(result.edge_plan).all() and torch.isfinite(result.q).all()
    expected = 1e-6 if dtype == torch.float32 else 1e-7
    assert result.effective_diagnostic_tolerance == expected
    torch.testing.assert_close(
        result.q.sum(), result.q.new_tensor(1.0), atol=2e-6 if dtype == torch.float32 else 2e-14, rtol=0
    )
    if dtype == torch.float32:
        with pytest.raises(ValueError, match="balanced probability-field contract"):
            build_sparse_probability_multipoles(
                result.edge_plan,
                result.q,
                result.edges,
                torch.tensor([6, 41]),
                ProbabilityMultipoleConfig(
                    (6, 41), 1, 2, 1.0, 3.0, probability_tolerance=1e-9
                ),
            )


def _geometric_sparse_energy(positions, references, cell):
    displacements = atom_site_displacements(
        positions, references, cell, (True, True, True)
    )
    result = _solve_sparse(displacements, iterations=128)
    weights = positions.new_tensor([[0.7, -0.2], [0.1, 0.4], [-0.3, 0.8]])
    dense_weights = weights[result.edges.site_index, result.edges.atom_index]
    return (result.edge_plan * dense_weights).sum() + 0.23 * result.q.square().sum()


def test_sparse_transport_rotation_reflection_and_force_covariance():
    positions = torch.tensor(
        [[0.2, 0.1, 0.05], [1.6, 0.4, 0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    references = torch.tensor(
        [[0.0, 0.0, 0.0], [1.8, 0.2, 0.0], [0.3, 1.7, 0.4]],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 6.0
    energy = _geometric_sparse_energy(positions, references, cell)
    force = -torch.autograd.grad(energy, positions)[0]
    orthogonal = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    transformed_positions = (positions.detach() @ orthogonal).requires_grad_(True)
    transformed_energy = _geometric_sparse_energy(
        transformed_positions, references @ orthogonal, cell @ orthogonal
    )
    transformed_force = -torch.autograd.grad(
        transformed_energy, transformed_positions
    )[0]
    torch.testing.assert_close(transformed_energy, energy, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(
        transformed_force, force @ orthogonal, atol=3e-13, rtol=3e-13
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_fixed_cuda(dtype):
    result = _solve_sparse(_displacements(dtype=dtype, device="cuda"))
    assert result.edge_plan.device.type == "cuda"
    assert result.q.device.type == "cuda"
    assert torch.isfinite(result.edge_plan).all() and torch.isfinite(result.q).all()
