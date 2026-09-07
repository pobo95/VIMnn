from __future__ import annotations

from copy import deepcopy

import pytest

from refsite_mlip.config import (
    TrainingDataSourceConfig,
    TrainingRecipeConfig,
    TrainingRecipeError,
    TrainingRunConfig,
)


def _payload():
    return {
        "schema_version": "refsite_training_recipe_v1",
        "name": "bound-run",
        "model": {
            "num_interactions": 1,
            "correlation": 1,
            "hidden_channels": 1,
            "max_L": 2,
            "initialization_seed": 3,
        },
        "reference": {
            "sources": {
                "POSCAR_222": {
                    "poscar": "references/POSCAR_222",
                    "maximum_strain": "auto",
                },
                "POSCAR_333": {
                    "poscar": "references/POSCAR_333",
                    "maximum_strain": "auto",
                },
            }
        },
        "data": {
            "train": [
                {"file": "data/pristine.xyz", "reference": "POSCAR_222"},
                {"file": "data/vacancy.xyz", "reference": "POSCAR_222"},
                {"file": "data/large.xyz", "reference": "POSCAR_333"},
            ],
            "validation": [
                {"file": "data/validate.xyz", "reference": "POSCAR_222"},
            ],
        },
        "loss": {
            "energy_weight": 1.0,
            "forces_weight": 0.0,
            "stress_weight": 0.0,
        },
        "baseline": "minimum_norm",
        "training": {"max_epochs": 1},
        "runtime": {"seed": 5},
    }


def test_alias_reference_and_file_binding_round_trip_is_exact():
    payload = _payload()
    recipe = TrainingRecipeConfig.from_dict(payload)
    canonical = recipe.to_dict()
    assert canonical["reference"] == {
        "sources": {
            "POSCAR_222": {"poscar": "references/POSCAR_222"},
            "POSCAR_333": {"poscar": "references/POSCAR_333"},
        }
    }
    assert canonical["data"] == payload["data"]
    assert tuple(source.authoring_alias for source in recipe.reference.sources) == (
        "POSCAR_222",
        "POSCAR_333",
    )
    assert tuple(source.reference_alias for source in recipe.data.train) == (
        "POSCAR_222",
        "POSCAR_222",
        "POSCAR_333",
    )
    assert TrainingRecipeConfig.from_dict(recipe.to_dict()).to_dict() == canonical


@pytest.mark.parametrize(
    "alias",
    ["", ".", "..", "../POSCAR", "dir/POSCAR", "dir\\POSCAR", "bad\x01name", True],
)
def test_reference_alias_is_a_safe_exact_case_sensitive_name(alias):
    payload = _payload()
    source = payload["reference"]["sources"].pop("POSCAR_222")
    payload["reference"]["sources"][alias] = source
    for item in payload["data"]["train"]:
        if item["reference"] == "POSCAR_222":
            item["reference"] = alias
    for item in payload["data"]["validation"]:
        item["reference"] = alias
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(payload)
    assert caught.value.reason_code in {
        "INVALID_REFERENCE_ALIAS",
        "UNKNOWN_RECIPE_KEY",
    }


def test_unknown_unused_and_mixed_alias_bindings_are_rejected():
    unknown = _payload()
    unknown["data"]["train"][0]["reference"] = "poscar_222"
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(unknown)
    assert caught.value.reason_code == "UNKNOWN_REFERENCE_ALIAS"

    unused = _payload()
    unused["data"]["train"] = unused["data"]["train"][:2]
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(unused)
    assert caught.value.reason_code == "UNUSED_REFERENCE_ALIAS"

    mixed = _payload()
    mixed["data"]["train"][1] = "data/vacancy.xyz"
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(mixed)
    assert caught.value.reason_code == "MIXED_DATA_BINDING_MODE"

    legacy_references = _payload()
    legacy_references["reference"] = [
        "references/POSCAR_222",
        "references/POSCAR_333",
    ]
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(legacy_references)
    assert caught.value.reason_code == "UNKNOWN_REFERENCE_ALIAS"


@pytest.mark.parametrize(
    "entry",
    [
        {"file": "data/a.xyz"},
        {"reference": "POSCAR_222"},
        {"file": "data/a.xyz", "reference": "POSCAR_222", "path": "x"},
        {"file": "data/a.xyz", "reference": "POSCAR_222", "unknown": 1},
    ],
)
def test_file_binding_mapping_is_strict(entry):
    payload = _payload()
    payload["data"]["train"][0] = entry
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(payload)
    assert caught.value.reason_code in {
        "MISSING_RECIPE_KEY",
        "UNKNOWN_RECIPE_KEY",
        "MIXED_DATA_BINDING_MODE",
    }


def test_canonical_v2_alias_provenance_is_additive_and_v1_rejects_it():
    source = TrainingDataSourceConfig(
        path="train.xyz",
        template_id="ref_m8_abc",
        reference_alias="POSCAR_222",
    )
    assert source.to_dict() == {
        "path": "train.xyz",
        "template_id": "ref_m8_abc",
        "reference_alias": "POSCAR_222",
    }
    assert TrainingDataSourceConfig.from_dict(source.to_dict()) == source

    from test_training_run_config import _payload as legacy_payload

    payload = deepcopy(legacy_payload())
    payload["data"]["train"][0]["reference_alias"] = "POSCAR_222"
    with pytest.raises(Exception) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "UNKNOWN_CONFIG_KEY"


def test_legacy_recipe_serialization_has_no_alias_fields():
    payload = _payload()
    payload["reference"] = "references/POSCAR_222"
    payload["data"] = {
        "train": "data/pristine.xyz",
        "validation": "data/validate.xyz",
    }
    recipe = TrainingRecipeConfig.from_dict(payload)
    canonical = recipe.to_dict()
    assert canonical["reference"] == "references/POSCAR_222"
    assert canonical["data"] == payload["data"]
    assert "reference_alias" not in repr(canonical)
    assert "'file'" not in repr(canonical)
