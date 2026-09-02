from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from refsite_mlip.config import (
    INTERACTION_RADIUS_CONFIG_SCHEMA_VERSION,
    INTERACTION_RADIUS_CONVENTION_VERSION,
    INTERACTION_RADIUS_LENGTH_UNIT,
    DerivedInteractionRadii,
    InteractionRadiusConfig,
    RadiusConfigError,
    derive_interaction_radii,
    transport_support_config_from_radii,
    validate_radius_artifact_compatibility,
    validate_radius_model_compatibility,
)


def _assert_plain_json(value):
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_plain_json(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_plain_json(item)
        return
    assert value is None or type(value) in (str, int, float, bool)


def test_default_nbc_radii_and_exact_derived_values():
    config = InteractionRadiusConfig()
    assert config.r_ot == 4.0
    assert config.r_mp == 3.0
    assert config.ot_switch_width == 0.5
    assert config.ot_skin == 0.2
    assert config.mp_skin == 0.5
    assert config.schema_version == INTERACTION_RADIUS_CONFIG_SCHEMA_VERSION
    assert config.convention_version == INTERACTION_RADIUS_CONVENTION_VERSION
    assert config.length_unit == INTERACTION_RADIUS_LENGTH_UNIT

    derived = derive_interaction_radii(config)
    assert derived == config.derived
    assert derived.r_on_ot == 3.5
    assert derived.r_off_ot == 4.0
    assert derived.r_candidate_ot == 4.2
    assert derived.r_mp == 3.0
    assert derived.r_candidate_mp == 3.5
    assert derived.ot_candidate_reuse_margin == pytest.approx(0.2)
    assert derived.mp_graph_strain_margin == pytest.approx(0.5)


def test_basic_and_advanced_overrides_do_not_order_ot_and_mp():
    config = InteractionRadiusConfig(
        r_ot=2.0,
        r_mp=5.0,
        ot_switch_width=0.25,
        ot_skin=0.75,
        mp_skin=1.25,
    )
    derived = config.derived
    assert derived.r_on_ot == 1.75
    assert derived.r_off_ot == 2.0
    assert derived.r_candidate_ot == 2.75
    assert derived.r_mp == 5.0
    assert derived.r_candidate_mp == 6.25


def test_zero_skins_are_exact_and_diagnostic_without_hidden_minimum():
    config = InteractionRadiusConfig(ot_skin=0.0, mp_skin=-0.0)
    derived = config.derived
    assert config.mp_skin == 0.0
    assert derived.r_candidate_ot == derived.r_off_ot
    assert derived.r_candidate_mp == derived.r_mp
    support = config.to_transport_support_config()
    assert support.candidate_skin == 0.0
    assert support.r_candidate == support.cutoff
    assert not derived.ot_candidate_reuse_margin_available
    assert not derived.mp_graph_strain_margin_available
    diagnostics = dict(derived.diagnostics)
    assert diagnostics["ot_candidate_reuse_margin"] == 0.0
    assert diagnostics["mp_graph_strain_margin"] == 0.0
    assert "ot_skin=0" in diagnostics["ot_candidate_reuse_note"]
    assert "mp_skin=0" in diagnostics["mp_graph_strain_note"]
    with pytest.raises(TypeError):
        derived.diagnostics["hidden_minimum"] = 0.1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("r_ot", 0.0),
        ("r_ot", -1.0),
        ("r_mp", 0.0),
        ("r_mp", -1.0),
        ("ot_switch_width", 0.0),
        ("ot_switch_width", 4.0),
        ("ot_skin", -0.1),
        ("mp_skin", -0.1),
        ("r_ot", math.nan),
        ("r_mp", math.inf),
        ("ot_switch_width", -math.inf),
        ("ot_skin", math.nan),
        ("mp_skin", math.inf),
        ("r_ot", True),
        ("r_mp", False),
        ("ot_switch_width", True),
        ("ot_skin", False),
        ("mp_skin", True),
    ),
)
def test_invalid_nonfinite_and_bool_values_are_rejected(field, value):
    with pytest.raises(RadiusConfigError) as caught:
        InteractionRadiusConfig(**{field: value})
    assert caught.value.field == field
    assert caught.value.reason_code in (
        "INVALID_RADIUS_VALUE",
        "INVALID_RADIUS_RELATION",
    )


def test_derived_radius_validation_is_public_and_structured():
    with pytest.raises(RadiusConfigError, match="r_candidate_ot"):
        DerivedInteractionRadii(3.5, 4.0, 3.9, 3.0, 3.5)
    with pytest.raises(RadiusConfigError, match="r_candidate_mp"):
        DerivedInteractionRadii(3.5, 4.0, 4.2, 3.0, 2.9)
    with pytest.raises(RadiusConfigError, match="r_on_ot"):
        DerivedInteractionRadii(4.1, 4.0, 4.2, 3.0, 3.5)


def test_dict_json_round_trip_canonical_bytes_and_fingerprint():
    config = InteractionRadiusConfig(r_ot=5, r_mp=6, ot_skin=0.3)
    payload = config.to_dict()
    _assert_plain_json(payload)
    reversed_payload = dict(reversed(tuple(payload.items())))
    first = InteractionRadiusConfig.from_dict(payload)
    second = InteractionRadiusConfig.from_dict(reversed_payload)
    assert first == config == second
    assert first.canonical_json() == second.canonical_json()
    assert first.to_json() == second.to_json()
    assert InteractionRadiusConfig.from_json(first.to_json()) == config
    assert first.content_fingerprint == second.content_fingerprint
    assert first.fingerprint == first.content_fingerprint
    assert first.content_fingerprint == hashlib.sha256(
        first.canonical_json().encode("utf-8")
    ).hexdigest()
    assert (
        first.content_fingerprint
        != InteractionRadiusConfig(r_ot=5.1, r_mp=6, ot_skin=0.3).content_fingerprint
    )
    decoded = json.loads(first.canonical_json())
    assert decoded == payload
    with pytest.raises(FrozenInstanceError):
        config.r_ot = 9.0


def test_strict_schema_unknown_missing_conflicting_and_json_constants():
    payload = InteractionRadiusConfig().to_dict()
    for key, reason in (
        ("unknown", "UNKNOWN_RADIUS_KEY"),
        ("r_off_ot", "CONFLICTING_RADIUS_KEY"),
    ):
        invalid = dict(payload)
        invalid[key] = 9.0
        with pytest.raises(RadiusConfigError) as caught:
            InteractionRadiusConfig.from_dict(invalid)
        assert caught.value.reason_code == reason

    missing = dict(payload)
    missing.pop("mp_skin")
    with pytest.raises(RadiusConfigError) as caught:
        InteractionRadiusConfig.from_dict(missing)
    assert caught.value.reason_code == "MISSING_RADIUS_KEY"
    assert caught.value.field == "mp_skin"

    duplicate = InteractionRadiusConfig().canonical_json().replace(
        '"r_ot":4.0', '"r_ot":4.0,"r_ot":5.0'
    )
    with pytest.raises(RadiusConfigError) as caught:
        InteractionRadiusConfig.from_json(duplicate)
    assert caught.value.reason_code == "CONFLICTING_RADIUS_KEY"

    nonfinite = InteractionRadiusConfig().canonical_json().replace(
        '"r_ot":4.0', '"r_ot":NaN'
    )
    with pytest.raises(RadiusConfigError) as caught:
        InteractionRadiusConfig.from_json(nonfinite)
    assert caught.value.reason_code == "NONFINITE_RADIUS_JSON"


def test_transport_conversion_keeps_backend_policy_outside_radius_content():
    config = InteractionRadiusConfig()
    original_json = config.canonical_json()
    dense = config.to_transport_support_config(backend="dense-masked")
    edge = transport_support_config_from_radii(
        config,
        backend="edge-list",
        candidate_backend="blocked",
        site_block_size=7,
        atom_block_size=11,
    )
    assert dense.kind == edge.kind == "compact_c2"
    assert dense.backend == "dense"
    assert edge.backend == "edge_list"
    assert edge.candidate_backend == "blocked"
    assert (dense.r_on, dense.cutoff, dense.r_candidate) == (3.5, 4.0, 4.2)
    assert (edge.r_on, edge.cutoff, edge.r_candidate) == (3.5, 4.0, 4.2)
    assert dense.switch_width == edge.switch_width == 0.5
    assert dense.candidate_skin == edge.candidate_skin == 0.2
    assert config.canonical_json() == original_json


def test_structured_artifact_and_model_compatibility_mapping_contracts():
    config = InteractionRadiusConfig()
    artifact = SimpleNamespace(mp_cutoff=3.0, mp_skin=0.5)
    assert validate_radius_artifact_compatibility(config, artifact) == config.derived

    support = config.to_transport_support_config(backend="edge-list")
    model_config = SimpleNamespace(transport_support=support)
    assert validate_radius_model_compatibility(config, model_config) == config.derived

    with pytest.raises(RadiusConfigError) as caught:
        validate_radius_artifact_compatibility(
            config, SimpleNamespace(mp_cutoff=3.1, mp_skin=0.4)
        )
    assert caught.value.reason_code == "RADIUS_ARTIFACT_MISMATCH"
    assert {entry[0] for entry in caught.value.mismatches} == {
        "mp_cutoff",
        "mp_skin",
    }
    assert "Rebuild the structural artifact" in str(caught.value)

    with pytest.raises(RadiusConfigError) as caught:
        validate_radius_model_compatibility(
            config,
            {
                "transport_support": {
                    "cutoff": 4.1,
                    "switch_width": 0.4,
                    "candidate_skin": 0.3,
                }
            },
        )
    assert caught.value.reason_code == "RADIUS_MODEL_MISMATCH"
    assert {entry[0] for entry in caught.value.mismatches} == {
        "r_on",
        "r_off",
        "r_candidate",
    }
    assert "new model run" in str(caught.value)
