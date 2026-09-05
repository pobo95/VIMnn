from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest
import torch

from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import (
    CentralConditionedHigherBody,
    HigherBodyArchitectureError,
    HigherBodyConfig,
    SymmetricCGBasisBank,
    SymmetricCorrelationConfig,
    squared_edge_radial_basis,
)
from refsite_mlip.interactions.higher_body import (
    LEGACY_HIGHER_BODY_CONTRACT_VERSION,
    SYMMETRIC_POWER_CONTRACT_VERSION,
)
from refsite_mlip.interactions.symmetric_cg import (
    GeneralizedCGPath,
    SYMMETRIC_CG_BASIS_VERSION,
    SymmetricCGError,
    fingerprint_generalized_cg_basis,
    generate_generalized_cg,
)
from refsite_mlip.models import PotentialConfig, ReferenceSitePotential


ANGULAR = "0e + 1o + 2e"
LEGACY_CONFIG_SHA256 = "6a8fad4db1a81d30218c4264429e0dc993df5da4321b3a3e80f80e79f915fce9"
LEGACY_STATE_KEY_SHA256 = "041207b743a3efaf43f5a47d6e1d51fc3fa56e058289297146c77b2533bce317"
LEGACY_OUTPUT_SHA256 = "9be0003343d427a72921f93c0d435ba9870268fc5a6925f0571294f76d9d4692"
LEGACY_POTENTIAL_CONFIG_SHA256 = "85a09c62d3bd162f1dbea216f77f245e68b17daee8ae01d29f98a5d3bf131cf7"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _legacy_config():
    return HigherBodyConfig(
        "2x0e+2x1o+2x2e",
        1,
        2,
        site_type_embedding_dim=2,
        n_correlation_channels=2,
        lmax=2,
        radial_feature_dim=3,
        radial_hidden_dims=(5,),
        avg_num_neighbors=2.0,
        correlation_mode="uuu",
    )


def _v2_config(*, order=3, channels=2):
    return HigherBodyConfig(
        "2x0e+2x1o+2x2e",
        1,
        2,
        site_type_embedding_dim=2,
        n_correlation_channels=channels,
        lmax=2,
        radial_feature_dim=3,
        radial_hidden_dims=(5,),
        avg_num_neighbors=2.0,
        cutoff=3.0,
        edge_length_scale=1.0,
        correlation_mode=None,
        contract_version=SYMMETRIC_POWER_CONTRACT_VERSION,
        symmetric_correlation=SymmetricCorrelationConfig(order),
    )


def _potential_config(higher_body):
    feature = ProbabilityMultipoleConfig(
        (6, 41),
        n_radial=2,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        site_type_vocabulary=(0, 1),
    )
    assert higher_body.species_count == 2
    return PotentialConfig(
        (6, 41),
        1,
        feature,
        higher_body,
        readout_hidden=8,
        energy_scale=1.0,
    )


def _potential_higher(*, v2):
    values = dict(
        irreps_feature="2x0e+4x0e+4x1o+4x2e",
        species_count=2,
        site_type_count=2,
        site_type_embedding_dim=2,
        n_correlation_channels=1,
        lmax=2,
        radial_feature_dim=3,
        radial_hidden_dims=(4,),
        avg_num_neighbors=6.0,
        cutoff=3.0,
        edge_length_scale=1.0,
    )
    if v2:
        values.update(
            correlation_mode=None,
            contract_version=SYMMETRIC_POWER_CONTRACT_VERSION,
            symmetric_correlation=SymmetricCorrelationConfig(3),
        )
    return HigherBodyConfig(**values)


def _legacy_inputs(model):
    torch.manual_seed(11)
    features = torch.randn(3, model.irreps_feature.dim, dtype=torch.float64)
    raw = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], dtype=torch.float64)
    site_types = torch.tensor([0, 1, 0], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    vectors = torch.tensor(
        [[1.0, 0.1, 0.2], [-1.0, -0.1, -0.2], [0.2, 1.1, -0.1], [-0.2, -1.1, 0.1], [-0.8, 0.3, 1.0], [0.8, -0.3, -1.0]],
        dtype=torch.float64,
    )
    radial = squared_edge_radial_basis(torch.sum(vectors * vectors, dim=-1), 3)
    cutoff = torch.full((6,), 0.73, dtype=torch.float64)
    return features, raw, site_types, edge_index, vectors, radial, cutoff


def test_legacy_config_dict_json_fingerprint_state_and_forward_are_exact():
    config = _legacy_config()
    expected = {
        "irreps_feature": "2x0e+2x1o+2x2e",
        "species_count": 1,
        "site_type_count": 2,
        "site_type_embedding_dim": 2,
        "n_correlation_channels": 2,
        "lmax": 2,
        "radial_feature_dim": 3,
        "radial_hidden_dims": [5],
        "avg_num_neighbors": 2.0,
        "cutoff": 3.0,
        "edge_length_scale": 1.0,
        "correlation_mode": "uuu",
        "contract_version": LEGACY_HIGHER_BODY_CONTRACT_VERSION,
    }
    assert config.to_dict() == expected
    assert HigherBodyConfig.from_dict(expected) == config
    assert HigherBodyConfig.from_dict(expected).to_dict() == expected
    assert hashlib.sha256(_canonical(expected).encode()).hexdigest() == LEGACY_CONFIG_SHA256
    assert config.content_fingerprint == LEGACY_CONFIG_SHA256

    torch.manual_seed(7)
    model = CentralConditionedHigherBody(config).double()
    key_bytes = json.dumps(list(model.state_dict()), separators=(",", ":")).encode()
    assert hashlib.sha256(key_bytes).hexdigest() == LEGACY_STATE_KEY_SHA256
    assert sum(parameter.numel() for parameter in model.parameters()) == 596
    result = model(*_legacy_inputs(model))
    output = torch.cat((result.Z1.reshape(-1), result.Z2.reshape(-1), result.Z3.reshape(-1)))
    assert hashlib.sha256(output.detach().contiguous().numpy().tobytes()).hexdigest() == LEGACY_OUTPUT_SHA256


def test_legacy_potential_config_serialization_is_exact():
    config = _potential_config(_potential_higher(v2=False))
    payload = config.to_dict()
    assert PotentialConfig.from_dict(payload) == config
    assert hashlib.sha256(_canonical(payload).encode()).hexdigest() == LEGACY_POTENTIAL_CONFIG_SHA256
    assert "symmetric_correlation" not in payload["higher_body"]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_v2_config_roundtrip_is_strict_frozen_and_deterministic(order):
    config = _v2_config(order=order, channels=5)
    payload = config.to_dict()
    assert payload["contract_version"] == SYMMETRIC_POWER_CONTRACT_VERSION
    assert "correlation_mode" not in payload
    assert payload["symmetric_correlation"] == {
        "correlation_order": order,
        "basis_kind": "full_path",
        "normalization": "component",
        "basis_version": SYMMETRIC_CG_BASIS_VERSION,
    }
    restored = HigherBodyConfig.from_dict(dict(reversed(tuple(payload.items()))))
    assert restored == config and restored.to_dict() == payload
    assert restored.canonical_json() == _canonical(payload)
    assert restored.content_fingerprint == config.content_fingerprint
    assert config.n_correlation_channels == 5
    assert config.symmetric_correlation.correlation_order == order
    with pytest.raises(FrozenInstanceError):
        config.symmetric_correlation.correlation_order = 2


@pytest.mark.parametrize("order", [0, 4, True, False, 1.5])
def test_v2_invalid_order_is_rejected(order):
    with pytest.raises((TypeError, ValueError), match="correlation_order"):
        SymmetricCorrelationConfig(order)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"basis_kind": "reduced"}, "basis_kind"),
        ({"normalization": "norm"}, "normalization"),
        ({"basis_version": "other"}, "basis_version"),
    ],
)
def test_v2_fixed_basis_contract_is_rejected_when_changed(overrides, match):
    with pytest.raises(ValueError, match=match):
        SymmetricCorrelationConfig(3, **overrides)


def test_v1_v2_dictionary_keys_cannot_be_mixed_or_ignored():
    v1 = _legacy_config().to_dict()
    v2 = _v2_config().to_dict()
    with pytest.raises(ValueError, match="forbids symmetric_correlation"):
        HigherBodyConfig.from_dict({**v1, "symmetric_correlation": SymmetricCorrelationConfig(3).to_dict()})
    with pytest.raises(ValueError, match="forbids correlation_mode"):
        HigherBodyConfig.from_dict({**v2, "correlation_mode": "uuu"})
    for key in ("symmetric_correlation", "lmax"):
        incomplete = dict(v2)
        incomplete.pop(key)
        with pytest.raises(ValueError, match="missing"):
            HigherBodyConfig.from_dict(incomplete)
    with pytest.raises(ValueError, match="unknown"):
        HigherBodyConfig.from_dict({**v2, "extra": 1})
    symmetric = SymmetricCorrelationConfig(3).to_dict()
    with pytest.raises(ValueError, match="missing"):
        SymmetricCorrelationConfig.from_dict({key: value for key, value in symmetric.items() if key != "basis_kind"})
    with pytest.raises(ValueError, match="unknown"):
        SymmetricCorrelationConfig.from_dict({**symmetric, "extra": 1})


def test_v2_order_channels_and_model_layers_are_independent():
    first = _potential_config(_potential_higher(v2=True))
    second = replace(first, num_layers=3, higher_body=replace(first.higher_body, n_correlation_channels=7, symmetric_correlation=SymmetricCorrelationConfig(1)))
    assert first.num_layers == 1 and first.higher_body.n_correlation_channels == 1
    assert second.num_layers == 3 and second.higher_body.n_correlation_channels == 7
    assert second.higher_body.symmetric_correlation.correlation_order == 1
    assert PotentialConfig.from_dict(second.to_dict()) == second


def test_basis_fingerprint_is_canonical_sensitive_and_caller_owned():
    generated = {order: generate_generalized_cg(ANGULAR, ANGULAR, order) for order in (1, 2, 3)}
    fingerprint = fingerprint_generalized_cg_basis(generated)
    assert fingerprint == fingerprint_generalized_cg_basis({3: generated[3], 1: generated[1], 2: generated[2]})
    assert fingerprint == fingerprint_generalized_cg_basis(tuple(generated.values()))

    exposed = generated[3].paths[0].coefficient
    exposed.add_(1.0)
    assert fingerprint_generalized_cg_basis(generated) == fingerprint

    original = generated[3].paths[0]
    coefficient = original.coefficient
    coefficient.reshape(-1)[0] += 1.0e-9
    changed_path = GeneralizedCGPath(original.metadata, coefficient)
    changed_order = replace(generated[3], paths=(changed_path, *generated[3].paths[1:]))
    assert fingerprint_generalized_cg_basis({1: generated[1], 2: generated[2], 3: changed_order}) != fingerprint

    reordered = replace(generated[3], paths=(generated[3].paths[1], generated[3].paths[0], *generated[3].paths[2:]))
    assert fingerprint_generalized_cg_basis((generated[1], generated[2], reordered)) != fingerprint
    assert fingerprint_generalized_cg_basis((generated[1], generated[2])) != fingerprint
    changed_normalization = tuple(
        replace(generated[order], normalization="norm") for order in (1, 2, 3)
    )
    assert fingerprint_generalized_cg_basis(changed_normalization) != fingerprint
    with pytest.raises(SymmetricCGError, match="BASIS_VERSION"):
        fingerprint_generalized_cg_basis(tuple(generated.values()), basis_version="other")


def test_basis_bank_has_single_persistent_ownership_and_dtype_independent_identity():
    double = SymmetricCGBasisBank(ANGULAR, ANGULAR, 3, dtype=torch.float64)
    single = SymmetricCGBasisBank(ANGULAR, ANGULAR, 3, dtype=torch.float32)
    assert double.basis_fingerprint == single.basis_fingerprint
    assert double.order_fingerprints == single.order_fingerprints
    expected_keys = tuple(f"U_order_{order}_output_{output}" for order in (1, 2, 3) for output in (0, 1, 2))
    assert tuple(double.state_dict()) == expected_keys
    assert len(tuple(double.buffers())) == 9
    assert double.buffer_byte_count == 1_125_576
    assert single.buffer_byte_count == 562_788
    double.validate_integrity(); single.validate_integrity()

    class Prototype(torch.nn.Module):
        def __init__(self, bank):
            super().__init__()
            self.symmetric_cg_basis = bank
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(3)])

    owner = Prototype(double)
    u_keys = tuple(key for key in owner.state_dict() if ".U_order_" in key)
    assert u_keys == tuple(f"symmetric_cg_basis.{key}" for key in expected_keys)
    assert not any("layers" in key and "U_order" in key for key in owner.state_dict())

    clone = Prototype(SymmetricCGBasisBank(ANGULAR, ANGULAR, 3, dtype=torch.float64))
    clone.load_state_dict(owner.state_dict(), strict=True)
    assert clone.symmetric_cg_basis.basis_fingerprint == double.basis_fingerprint
    assert all(torch.equal(value, clone.state_dict()[key]) for key, value in owner.state_dict().items())
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(double.buffers(), clone.symmetric_cg_basis.buffers()))

    double.to(dtype=torch.float32)
    assert double.basis_fingerprint == single.basis_fingerprint
    double.validate_integrity()
    with torch.no_grad():
        double.basis_tensor(3, "2e").reshape(-1)[0] += 1.0
    with pytest.raises(SymmetricCGError, match="BASIS_BUFFER_MISMATCH"):
        double.validate_integrity()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_basis_bank_cuda_float_materialization_preserves_canonical_fingerprint():
    bank = SymmetricCGBasisBank(ANGULAR, ANGULAR, 3, dtype=torch.float64)
    fingerprint = bank.basis_fingerprint
    bank.to(device="cuda:0", dtype=torch.float32)
    bank.validate_integrity()
    assert bank.basis_fingerprint == fingerprint
    assert all(value.device.type == "cuda" and value.dtype == torch.float32 for value in bank.buffers())


def test_v2_execution_fails_before_legacy_parameters_or_rng_are_touched(monkeypatch):
    higher = _v2_config()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(HigherBodyArchitectureError, match="SYMMETRIC_CORRELATION_NOT_INTEGRATED") as caught:
        CentralConditionedHigherBody(higher)
    assert caught.value.reason_code == "SYMMETRIC_CORRELATION_NOT_INTEGRATED"
    assert torch.equal(torch.random.get_rng_state(), rng)

    config = _potential_config(_potential_higher(v2=True))
    import refsite_mlip.models.potential as potential_module

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("legacy residual block was partially created")

    monkeypatch.setattr(potential_module, "ResidualInteractionBlock", forbidden)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(HigherBodyArchitectureError, match="SYMMETRIC_CORRELATION_NOT_INTEGRATED"):
        ReferenceSitePotential(config, None, None, None, None, None, None)
    assert torch.equal(torch.random.get_rng_state(), rng)
