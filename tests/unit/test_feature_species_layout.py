from __future__ import annotations

import pytest
import torch

from refsite_mlip.features import (
    ProbabilityMultipoleConfig,
    build_probability_multipoles,
)
from refsite_mlip.features.radial import c2_envelope
from refsite_mlip.features.solid_harmonics import regular_solid_harmonics


def _fixture(dtype=torch.float64, device="cpu"):
    P = torch.tensor(
        [[0.8, 0.1], [0.15, 0.7], [0.05, 0.2]],
        dtype=dtype,
        device=device,
    )
    q = torch.tensor([0.1, 0.15, 0.75], dtype=dtype, device=device)
    numbers = torch.tensor([6, 41], dtype=torch.long, device=device)
    displacements = torch.tensor(
        [
            [[0.1, 0.0, 0.0], [0.4, 0.2, 0.0]],
            [[0.2, 0.1, 0.0], [-0.3, 0.1, 0.2]],
            [[0.5, 0.0, 0.1], [0.2, -0.4, 0.1]],
        ],
        dtype=dtype,
        device=device,
    )
    config = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=2,
        ell_feature=1.0,
        r_cut=2.0,
        site_type_vocabulary=(0, 1),
    )
    site_types = torch.tensor([0, 1, 0], dtype=torch.long, device=device)
    return P, q, numbers, displacements, config, site_types


def test_probability_contract_exact_occupancy_and_q_identity():
    P, q, numbers, displacements, config, site_types = _fixture()
    result = build_probability_multipoles(
        P, q, numbers, displacements, config, site_types
    )
    expected = torch.tensor(
        [[0.8, 0.1], [0.15, 0.7], [0.05, 0.2]], dtype=torch.float64
    )
    torch.testing.assert_close(result.species_probabilities, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        result.species_probabilities.sum(dim=1) + result.vacancy_probabilities,
        torch.ones(3, dtype=torch.float64),
        atol=2.0e-16,
        rtol=0,
    )
    torch.testing.assert_close(
        result.species_probabilities.sum(dim=0),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        atol=2.0e-16,
        rtol=0,
    )
    assert result.vacancy_probabilities is q
    torch.testing.assert_close(result.raw_probability_state[:, -1], q, atol=0, rtol=0)
    torch.testing.assert_close(
        result.equivariant_features[:, :2], expected, atol=0, rtol=0
    )


def test_exact_occupancy_has_no_cutoff_or_distance_dependence():
    P, q, numbers, displacements, config, site_types = _fixture()
    far = displacements * 100.0
    first = build_probability_multipoles(
        P, q, numbers, displacements, config, site_types
    )
    second = build_probability_multipoles(P, q, numbers, far, config, site_types)
    torch.testing.assert_close(
        first.equivariant_features[:, :2],
        second.equivariant_features[:, :2],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        second.equivariant_features[:, 2:],
        torch.zeros_like(second.equivariant_features[:, 2:]),
        atol=0,
        rtol=0,
    )


def test_unknown_species_and_invalid_probability_contract_fail_fast():
    P, q, numbers, displacements, config, site_types = _fixture()
    with pytest.raises(ValueError, match="unknown atomic species"):
        build_probability_multipoles(
            P,
            q,
            torch.tensor([6, 8], dtype=torch.long),
            displacements,
            config,
            site_types,
        )
    with pytest.raises(ValueError, match="balanced probability"):
        build_probability_multipoles(
            P,
            q + 0.01,
            numbers,
            displacements,
            config,
            site_types,
        )


def test_atom_permutation_invariance_and_site_permutation_equivariance():
    P, q, numbers, displacements, config, site_types = _fixture()
    reference = build_probability_multipoles(
        P, q, numbers, displacements, config, site_types
    )
    atom_order = torch.tensor([1, 0], dtype=torch.long)
    atoms = build_probability_multipoles(
        P[:, atom_order],
        q,
        numbers[atom_order],
        displacements[:, atom_order],
        config,
        site_types,
    )
    torch.testing.assert_close(
        atoms.equivariant_features,
        reference.equivariant_features,
        atol=2.0e-15,
        rtol=2.0e-15,
    )
    site_order = torch.tensor([2, 0, 1], dtype=torch.long)
    sites = build_probability_multipoles(
        P[site_order],
        q[site_order],
        numbers,
        displacements[site_order],
        config,
        site_types[site_order],
    )
    torch.testing.assert_close(
        sites.equivariant_features,
        reference.equivariant_features[site_order],
        atol=2.0e-15,
        rtol=2.0e-15,
    )


def _brute_force_oracle(P, q, numbers, displacements, config):
    species = config.species_vocabulary
    M, N = P.shape
    occupancy = torch.zeros((M, len(species)), dtype=P.dtype)
    blocks = []
    for s in range(M):
        for i in range(N):
            occupancy[s, species.index(int(numbers[i]))] += P[s, i]
    blocks.append(occupancy)
    xi_cut = (config.r_cut / config.ell_feature) ** 2
    for l in range(3):
        values = torch.zeros(
            (M, len(species), config.n_radial, 2 * l + 1),
            dtype=P.dtype,
        )
        for s in range(M):
            for i in range(N):
                alpha = species.index(int(numbers[i]))
                y = displacements[s, i] / config.ell_feature
                xi = torch.dot(y, y)
                u = xi / xi_cut
                envelope = (
                    1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5
                    if float(u) < 1.0
                    else xi.new_zeros(())
                )
                solid = regular_solid_harmonics(y, 2)[0][l * l : (l + 1) ** 2]
                for radial in range(config.n_radial):
                    values[s, alpha, radial] += (
                        P[s, i] * u**radial * envelope * solid
                    )
        blocks.append(values.reshape(M, -1))
    return torch.cat(blocks, dim=1)


def test_vectorized_matches_dense_python_oracle_and_fixed_layout():
    P, q, numbers, displacements, config, site_types = _fixture()
    result = build_probability_multipoles(
        P, q, numbers, displacements, config, site_types
    )
    oracle = _brute_force_oracle(P, q, numbers, displacements, config)
    torch.testing.assert_close(
        result.equivariant_features, oracle, atol=3.0e-15, rtol=3.0e-15
    )
    assert str(result.irreps_out) == "2x0e+4x0e+4x1o+4x2e"
    assert result.equivariant_features.shape == (3, 38)
    assert [entry.block_name for entry in result.channel_metadata[:2]] == [
        "exact_species_occupancy",
        "exact_species_occupancy",
    ]
    assert result.channel_metadata[2].component_slice == (2, 3)
    assert result.channel_metadata[-1].component_slice == (33, 38)


def test_config_serialization_round_trip_and_layout_version_guard():
    _, _, _, _, config, _ = _fixture()
    restored = ProbabilityMultipoleConfig.from_dict(config.to_dict())
    assert restored == config
    bad = config.to_dict()
    bad["feature_layout_version"] = "changed"
    with pytest.raises(ValueError, match="layout version"):
        ProbabilityMultipoleConfig.from_dict(bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_feature_dtype_device_and_cpu_cuda_parity(dtype, device):
    P, q, numbers, displacements, config, site_types = _fixture(dtype, device)
    result = build_probability_multipoles(
        P, q, numbers, displacements, config, site_types
    )
    assert result.equivariant_features.dtype == dtype
    assert result.equivariant_features.device.type == device
    assert torch.all(torch.isfinite(result.equivariant_features))

    if device == "cuda" and dtype == torch.float64:
        cpu = build_probability_multipoles(*_fixture(torch.float64, "cpu"))
        torch.testing.assert_close(
            result.equivariant_features.cpu(),
            cpu.equivariant_features,
            atol=2.0e-14,
            rtol=2.0e-14,
        )
