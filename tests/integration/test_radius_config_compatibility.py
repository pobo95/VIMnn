from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from refsite_mlip.config import (
    InteractionRadiusConfig,
    RadiusConfigError,
    validate_radius_artifact_compatibility,
    validate_radius_model_compatibility,
)
from refsite_mlip.data import (
    REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION,
    PhaseSpecification,
    ReferenceTemplate,
    capture_reference_structure_artifact,
)
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import (
    REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
    PotentialConfig,
)
from refsite_mlip.phase import find_typed_stabilizer
from refsite_mlip.training import CHECKPOINT_SCHEMA_VERSION


def _artifact():
    fractional = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    )
    site_types = torch.tensor([0, 1], dtype=torch.long)
    topology = build_reference_graph_topology(
        fractional,
        site_types,
        torch.eye(3, dtype=torch.float64) * 4.0,
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    phase = PhaseSpecification(
        modes=torch.eye(3, dtype=torch.long),
        mode_weights=torch.ones(3, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
        convention_version="radius_config_test_phase_v1",
    )
    template = ReferenceTemplate.snapshot(
        "radius-config-tiny",
        topology,
        phase.modes,
        phase.mode_weights,
        phase.site_type_alignment_weights[topology.site_types],
        phase.channel_weights,
        find_typed_stabilizer(fractional, site_types),
        (6, 41),
    )
    return capture_reference_structure_artifact(template, avg_num_neighbors=6.0)


def _potential_config(radius_config, *, backend):
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=2,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        site_type_vocabulary=(0, 1),
    )
    higher = HigherBodyConfig(
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
    return PotentialConfig(
        species_vocabulary=(6, 41),
        num_layers=1,
        feature=feature,
        higher_body=higher,
        transport_support=radius_config.to_transport_support_config(
            backend=backend
        ),
    )


def test_real_structural_artifact_compatibility_is_read_only():
    config = InteractionRadiusConfig()
    artifact = _artifact()
    fingerprint = artifact.structural_fingerprint
    diagnostics = artifact.diagnostics.to_dict()
    tensors = {
        name: getattr(artifact, name).clone()
        for name in (
            "reference_fractional",
            "site_types",
            "reference_cell",
            "edge_index",
            "periodic_shifts",
            "stabilizer_translations",
            "stabilizer_permutations",
        )
    }

    derived = validate_radius_artifact_compatibility(config, artifact)
    assert (derived.r_mp, derived.r_candidate_mp) == (3.0, 3.5)
    assert artifact.structural_fingerprint == fingerprint
    assert artifact.diagnostics.to_dict() == diagnostics
    for name, before in tensors.items():
        assert torch.equal(getattr(artifact, name), before)


@pytest.mark.parametrize("backend", ("dense-masked", "edge-list"))
def test_actual_potential_config_backend_independence_and_read_only(backend):
    radius_config = InteractionRadiusConfig()
    potential_config = _potential_config(radius_config, backend=backend)
    payload = copy.deepcopy(potential_config.to_dict())
    radius_json = radius_config.canonical_json()

    derived = validate_radius_model_compatibility(
        radius_config, SimpleNamespace(config=potential_config)
    )
    support = potential_config.transport_support
    assert (support.r_on, support.cutoff, support.r_candidate) == (
        derived.r_on_ot,
        derived.r_off_ot,
        derived.r_candidate_ot,
    )
    assert potential_config.to_dict() == payload
    assert radius_config.canonical_json() == radius_json
    assert "interaction_radius_config" not in potential_config.to_dict()


def test_actual_contract_mismatches_are_actionable_and_transactional():
    config = InteractionRadiusConfig()
    artifact = _artifact()
    artifact_before = artifact.structural_fingerprint
    incompatible_artifact = copy.copy(artifact.diagnostics)
    object.__setattr__(incompatible_artifact, "mp_skin", 0.6)
    with pytest.raises(RadiusConfigError) as caught:
        validate_radius_artifact_compatibility(config, incompatible_artifact)
    assert caught.value.reason_code == "RADIUS_ARTIFACT_MISMATCH"
    assert caught.value.mismatches == (("mp_skin", 0.5, 0.6),)
    assert "r_mp or mp_skin" in caught.value.action
    assert artifact.structural_fingerprint == artifact_before

    potential_config = _potential_config(config, backend="edge-list")
    payload = copy.deepcopy(potential_config.to_dict())
    incompatible = InteractionRadiusConfig(r_ot=4.1)
    with pytest.raises(RadiusConfigError) as caught:
        validate_radius_model_compatibility(incompatible, potential_config)
    assert caught.value.reason_code == "RADIUS_MODEL_MISMATCH"
    assert {item[0] for item in caught.value.mismatches} == {
        "r_on",
        "r_off",
        "r_candidate",
    }
    assert "new model run" in caught.value.action
    assert potential_config.to_dict() == payload


def test_existing_artifact_model_bundle_and_checkpoint_schemas_are_unchanged():
    config = InteractionRadiusConfig()
    artifact = _artifact()
    potential_config = _potential_config(config, backend="dense-masked")
    support_payload = potential_config.to_dict()["transport_support"]

    assert (
        REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION
        == "reference_structure_artifact_v1"
    )
    assert REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION == "reference_site_model_bundle_v1"
    assert CHECKPOINT_SCHEMA_VERSION == "refsite_training_checkpoint_v1"
    assert artifact.schema_version == REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION
    assert tuple(support_payload) == (
        "kind",
        "cutoff",
        "switch_width",
        "candidate_skin",
        "backend",
        "convention_version",
    )
    assert set(support_payload).isdisjoint(
        {"r_ot", "r_mp", "ot_skin", "mp_skin", "radius_fingerprint"}
    )

