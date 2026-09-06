from __future__ import annotations

import math

import pytest

from refsite_mlip.config import (
    RecipeReferenceConfig,
    TrainingRecipeConfig,
    TrainingRecipeError,
    resolve_auto_maximum_strain,
)


def test_reference_scalar_list_and_advanced_forms_are_strict():
    scalar = RecipeReferenceConfig.from_dict("pos/POSCAR_222")
    assert scalar.to_dict() == "pos/POSCAR_222"
    assert scalar.sources[0].is_automatic
    assert scalar.sources[0].maximum_strain == "auto"

    listed = RecipeReferenceConfig.from_dict(
        ["pos/POSCAR_222", "pos/POSCAR_333"]
    )
    assert listed.to_dict() == ["pos/POSCAR_222", "pos/POSCAR_333"]

    advanced = RecipeReferenceConfig.from_dict(
        {
            "sources": [
                {
                    "poscar": "pos/POSCAR_222",
                    "template_id": "alpha",
                    "maximum_strain": "auto",
                },
                {
                    "poscar": "pos/POSCAR_333",
                    "template_id": "zeta",
                    "maximum_strain": 0.03,
                },
            ]
        }
    )
    assert advanced.to_dict() == {
        "sources": [
            {"poscar": "pos/POSCAR_222", "template_id": "alpha"},
            {
                "poscar": "pos/POSCAR_333",
                "template_id": "zeta",
                "maximum_strain": 0.03,
            },
        ]
    }


def test_explicit_reference_serialization_is_unchanged():
    payload = {
        "specification": "reference-specification.yaml",
        "poscar": "POSCAR",
        "allow_provisional_phase": True,
    }
    assert RecipeReferenceConfig.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize(
    "value, reason",
    [
        ([], "EMPTY_REFERENCE_SEQUENCE"),
        (True, "INVALID_RECIPE_SECTION"),
        ({"sources": []}, "EMPTY_REFERENCE_SEQUENCE"),
        ({"sources": [{"poscar": "P", "unexpected": 1}]}, "UNKNOWN_RECIPE_KEY"),
        (
            {"specification": "s.yaml", "poscar": "P", "maximum_strain": 0.02},
            "UNKNOWN_RECIPE_KEY",
        ),
        ({"sources": [{"poscar": "P", "maximum_strain": True}]}, "INVALID_MAXIMUM_STRAIN"),
        ({"sources": [{"poscar": "P", "maximum_strain": math.nan}]}, "INVALID_MAXIMUM_STRAIN"),
        ({"sources": [{"poscar": "P", "maximum_strain": math.inf}]}, "INVALID_MAXIMUM_STRAIN"),
    ],
)
def test_invalid_automatic_reference_inputs_are_rejected(value, reason):
    with pytest.raises(TrainingRecipeError) as captured:
        RecipeReferenceConfig.from_dict(value)
    assert captured.value.reason_code == reason


def test_auto_maximum_strain_uses_versioned_integer_tick_policy():
    assert resolve_auto_maximum_strain(0.0) == 0.01
    assert resolve_auto_maximum_strain(0.0175207027) == 0.025
    assert resolve_auto_maximum_strain(0.020000000000000000) == 0.025
    assert resolve_auto_maximum_strain(0.020000000000000004) == 0.025
    assert resolve_auto_maximum_strain(0.045) == 0.05
    assert resolve_auto_maximum_strain(0.0450000001) > 0.05
    for invalid in (True, -0.1, math.nan, math.inf):
        with pytest.raises((TypeError, ValueError)):
            resolve_auto_maximum_strain(invalid)


def test_poscar_only_recipe_defaults_inference_to_sinkhorn():
    payload = {
        "schema_version": "refsite_training_recipe_v1",
        "name": "automatic",
        "model": {},
        "reference": "POSCAR",
        "data": {"train": "train.xyz", "validation": "validation.xyz"},
        "loss": {"energy_weight": 1.0},
        "baseline": "zero",
        "training": {"max_epochs": 1},
        "runtime": {"seed": 1},
    }
    recipe = TrainingRecipeConfig.from_dict(payload)
    assert recipe.ot_solver.to_dict() == {
        "training": "sinkhorn",
        "inference": "sinkhorn",
    }
    with pytest.raises(TrainingRecipeError) as captured:
        TrainingRecipeConfig.from_dict(
            {
                **payload,
                "reference": {
                    "specification": "reference.yaml",
                    "poscar": "POSCAR",
                },
            }
        )
    assert captured.value.reason_code == "MISSING_RECIPE_KEY"
