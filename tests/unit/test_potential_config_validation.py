from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import PotentialConfig
from refsite_mlip.transport import TransportSupportConfig


def _parts():
    feature = ProbabilityMultipoleConfig(
        (6, 41),
        n_radial=2,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        site_type_vocabulary=(0, 1),
    )
    higher = HigherBodyConfig(
        "2x0e+4x0e+4x1o+4x2e",
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
    support = TransportSupportConfig(
        kind="compact_c2",
        cutoff=4.0,
        switch_width=0.5,
        candidate_skin=0.2,
    )
    return feature, higher, support


def _config(**overrides):
    feature, higher, support = _parts()
    values = {
        "species_vocabulary": (6, 41),
        "num_layers": 1,
        "feature": feature,
        "higher_body": higher,
        "transport_support": support,
    }
    values.update(overrides)
    return PotentialConfig(**values)


def test_valid_config_round_trip_payload_is_unchanged():
    config = _config()
    payload = config.to_dict()
    assert PotentialConfig.from_dict(payload) == config
    assert PotentialConfig.from_dict(payload).to_dict() == payload


def test_constructor_canonicalizes_numeric_fields_and_owns_schedules():
    steps = [np.float64(0.7), np.float64(1.0)]
    damping = [np.float32(2.0), np.float32(0.5)]
    config = _config(
        species_vocabulary=(np.int64(6), np.int64(41)),
        num_layers=np.int64(1),
        readout_hidden=np.int64(8),
        energy_scale=np.float64(1.0),
        epsilon_ot=np.float64(0.5),
        ell_ot=np.float64(1.5),
        train_sinkhorn_iterations=np.int64(64),
        eval_sinkhorn_warmup_iterations=np.int64(4),
        phase_steps=steps,
        phase_damping=damping,
    )
    steps[0] = 99.0
    damping[0] = 99.0

    assert config.phase_steps == (0.7, 1.0)
    assert config.phase_damping == (2.0, 0.5)
    assert isinstance(config.phase_steps, tuple)
    assert isinstance(config.phase_damping, tuple)
    assert all(type(value) is int for value in config.species_vocabulary)
    assert all(
        type(getattr(config, name)) is int
        for name in (
            "num_layers",
            "readout_hidden",
            "train_sinkhorn_iterations",
            "eval_sinkhorn_warmup_iterations",
        )
    )
    assert all(
        type(getattr(config, name)) is float
        for name in ("energy_scale", "epsilon_ot", "ell_ot")
    )
    assert all(type(value) is float for value in config.phase_steps)
    assert all(type(value) is float for value in config.phase_damping)
    assert PotentialConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("num_layers", True, "num_layers"),
        ("num_layers", 0, "num_layers"),
        ("readout_hidden", 0, "readout_hidden"),
        ("energy_scale", math.nan, "energy_scale"),
        ("epsilon_ot", 0.0, "epsilon_ot"),
        ("ell_ot", math.inf, "ell_ot"),
        ("train_sinkhorn_iterations", False, "train_sinkhorn_iterations"),
        ("eval_sinkhorn_warmup_iterations", -1, "eval_sinkhorn"),
        ("phase_steps", (1.0, 0.0), "phase_steps"),
        ("phase_damping", (1.0,), "equal positive length"),
    ),
)
def test_scalar_layer_and_schedule_contracts_fail_at_construction(
    field, value, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        _config(**{field: value})


@pytest.mark.parametrize("field", ("feature", "higher_body", "transport_support"))
def test_nested_config_types_fail_at_construction(field):
    with pytest.raises(TypeError, match=field):
        _config(**{field: object()})


def test_nested_cutoff_body_order_and_channel_contracts_fail_at_construction():
    feature, higher, support = _parts()
    with pytest.raises(ValueError, match="lmax=2"):
        _config(feature=replace(feature, lmax=1))
    with pytest.raises(ValueError, match="channel dimensions"):
        _config(higher_body=replace(higher, n_correlation_channels=0))
    with pytest.raises(ValueError, match="r_cut"):
        _config(feature=replace(feature, r_cut=float("nan")))
    with pytest.raises(ValueError, match="cutoff"):
        _config(higher_body=replace(higher, cutoff=0.0))
    with pytest.raises(ValueError, match="v1 MP radius"):
        PotentialConfig(
            (6, 41),
            1,
            feature,
            replace(higher, cutoff=3.1),
            transport_support=support,
        )


def test_potential_requires_executable_global_site_type_contract():
    feature, higher, _ = _parts()
    for vocabulary in (None, (), (1, 2)):
        with pytest.raises(ValueError, match="nonempty fixed tuple.*0..A-1"):
            _config(
                feature=replace(
                    feature, site_type_vocabulary=vocabulary
                )
            )
    with pytest.raises(ValueError, match="site_type_count"):
        _config(higher_body=replace(higher, site_type_count=3))
