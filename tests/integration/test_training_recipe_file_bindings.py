from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("ase")

from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from refsite_mlip.cli.main import main
from refsite_mlip.config import (
    TrainingDataConfig,
    TrainingDataSourceConfig,
    TrainingRecipeError,
    resolve_training_recipe,
)
from refsite_mlip.training import prepare_scratch_training_run


_LATTICE = 4.482314244155584


def _reference(*, repeated: bool = False):
    atoms = bulk("NbC", "rocksalt", a=_LATTICE, cubic=True)
    fractional = atoms.get_scaled_positions(wrap=True)
    fractional[atoms.numbers == 6] += np.array([0.017, 0.011, 0.007])
    atoms.set_scaled_positions(fractional)
    return atoms.repeat((2, 1, 1)) if repeated else atoms


def _vacancy(atoms, *, species: int, count: int = 1):
    result = atoms.copy()
    indices = np.flatnonzero(result.numbers == species)[:count]
    assert len(indices) == count
    del result[indices.tolist()]
    return result


def _labeled(atoms, energy: float):
    result = atoms.copy()
    result.calc = SinglePointCalculator(result, energy=float(energy))
    return result


def _payload(*, alias_small="POSCAR_222", alias_large="POSCAR_333"):
    return {
        "schema_version": "refsite_training_recipe_v1",
        "name": "bound-run",
        "model": {
            "num_interactions": 1,
            "correlation": 1,
            "hidden_channels": 1,
            "max_L": 2,
            "initialization_seed": 7,
        },
        "reference": {
            "sources": {
                alias_small: {
                    "poscar": "references/POSCAR-small",
                    "maximum_strain": "auto",
                },
                alias_large: {
                    "poscar": "references/POSCAR-large",
                    "maximum_strain": "auto",
                },
            }
        },
        "data": {
            "train": [
                {"file": "data/small-pristine.xyz", "reference": alias_small},
                {"file": "data/small-vacancy.xyz", "reference": alias_small},
                {"file": "data/large-k2.xyz", "reference": alias_large},
            ],
            "validation": [
                {"file": "data/small-validation.xyz", "reference": alias_small},
                {"file": "data/large-validation.xyz", "reference": alias_large},
            ],
        },
        "loss": {
            "energy_weight": 1.0,
            "forces_weight": 0.0,
            "stress_weight": 0.0,
        },
        "baseline": "minimum_norm",
        "radii": {"r_ot": 4.0, "r_mp": 3.0},
        "training": {
            "batch_size": 2,
            "validation_batch_size": 1,
            "max_epochs": 1,
            "learning_rate": 0.001,
        },
        "runtime": {"device": "cpu", "dtype": "float64", "seed": 11},
    }


def _write_case(tmp_path: Path, *, payload=None) -> Path:
    (tmp_path / "references").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    small = _reference()
    large = _reference(repeated=True)
    write(
        tmp_path / "references" / "POSCAR-small",
        small,
        format="vasp",
        direct=True,
    )
    write(
        tmp_path / "references" / "POSCAR-large",
        large,
        format="vasp",
        direct=True,
    )
    frames = {
        "small-pristine.xyz": [_labeled(small, -8.0)],
        "small-vacancy.xyz": [_labeled(_vacancy(small, species=6), -6.4)],
        "large-k2.xyz": [_labeled(_vacancy(large, species=41, count=2), -12.8)],
        "small-validation.xyz": [_labeled(_vacancy(small, species=41), -6.3)],
        "large-validation.xyz": [_labeled(large, -15.8)],
    }
    for filename, values in frames.items():
        write(tmp_path / "data" / filename, values, format="extxyz")
    recipe = tmp_path / "bound.yaml"
    recipe.write_text(
        yaml.safe_dump(_payload() if payload is None else payload, sort_keys=False),
        encoding="utf-8",
    )
    return recipe


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_explicit_file_bindings_preserve_source_frame_and_template_identity(tmp_path):
    recipe = _write_case(tmp_path)
    resolved = resolve_training_recipe(recipe)
    automatic = resolved.automatic_reference_preparation
    assert automatic is not None
    by_sites = {
        result.certificate["num_reference_sites"]: result.template_id
        for result in automatic.results
    }
    assert set(by_sites) == {8, 16}

    config = resolved.config
    assert tuple(source.reference_alias for source in config.data.train) == (
        "POSCAR_222",
        "POSCAR_222",
        "POSCAR_333",
    )
    assert tuple(source.template_id for source in config.data.train) == (
        by_sites[8],
        by_sites[8],
        by_sites[16],
    )
    assert config.model_source.default_template_id == min(by_sites.values())
    prepared = prepare_scratch_training_run(
        config, automatic_reference_preparation=automatic.to_dict()
    )
    assert tuple(sample.sample_id for sample in prepared.train_samples) == (
        "train.0000:000000",
        "train.0001:000000",
        "train.0002:000000",
    )
    assert tuple(sample.template_id for sample in prepared.train_samples) == (
        by_sites[8],
        by_sites[8],
        by_sites[16],
    )
    assert tuple(
        prepared.registry.resolve(sample.template_id).topology.num_sites
        - sample.num_atoms
        for sample in prepared.train_samples
    ) == (0, 1, 2)
    assert tuple(sample.sample_id for sample in prepared.validation_samples) == (
        "validation.0000:000000",
        "validation.0001:000000",
    )

    manifest = prepared.data_manifest
    assert [item["source_index"] for item in manifest["train"]["sources"]] == [
        0,
        1,
        2,
    ]
    assert [item["reference_alias"] for item in manifest["train"]["sources"]] == [
        "POSCAR_222",
        "POSCAR_222",
        "POSCAR_333",
    ]
    assert [
        (item["global_frame_start"], item["global_frame_end_exclusive"])
        for item in manifest["train"]["sources"]
    ] == [(0, 1), (1, 2), (2, 3)]
    for split in ("train", "validation"):
        for item in manifest[split]["sources"]:
            assert item["raw_sha256"] == _sha256(Path(item["resolved_path"]))
            assert len(item["semantic_digest"]) == 64
            assert item["template_fingerprint"] == prepared.registry.resolve(
                item["template_id"]
            ).fingerprint

    assert prepared.baseline_preflight["num_valid_energy_structures"] == 3
    assert prepared.baseline_preflight["rank"] == 2
    assert prepared.validation_label_statistics["energy"]["present_frames"] == 2
    assert not (tmp_path / "runs" / "bound-run").exists()


def test_alias_spelling_is_provenance_while_bound_data_semantics_are_content_based(
    tmp_path,
):
    recipe = _write_case(tmp_path)
    first = resolve_training_recipe(recipe)
    first_prepared = prepare_scratch_training_run(
        first.config,
        automatic_reference_preparation=first.automatic_reference_preparation.to_dict(),
    )

    renamed_recipe = tmp_path / "renamed-aliases.yaml"
    renamed_recipe.write_text(
        yaml.safe_dump(
            _payload(alias_small="small_ref", alias_large="large_ref"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    renamed = resolve_training_recipe(renamed_recipe)
    renamed_prepared = prepare_scratch_training_run(
        renamed.config,
        automatic_reference_preparation=renamed.automatic_reference_preparation.to_dict(),
    )
    assert renamed.automatic_reference_preparation.content_fingerprint == (
        first.automatic_reference_preparation.content_fingerprint
    )
    assert renamed.config.model_source.to_dict() == first.config.model_source.to_dict()
    assert renamed.config.config_fingerprint != first.config.config_fingerprint
    assert renamed_prepared.train_semantic_digest == first_prepared.train_semantic_digest
    assert renamed_prepared.validation_semantic_digest == (
        first_prepared.validation_semantic_digest
    )
    assert renamed_prepared.registry.fingerprint == first_prepared.registry.fingerprint
    assert renamed_prepared.data_manifest["fingerprint"] != (
        first_prepared.data_manifest["fingerprint"]
    )


def test_source_order_boundary_content_and_wrong_binding_are_detected(tmp_path):
    recipe = _write_case(tmp_path)
    resolved = resolve_training_recipe(recipe)
    automatic = resolved.automatic_reference_preparation.to_dict()
    original = prepare_scratch_training_run(
        resolved.config, automatic_reference_preparation=automatic
    )

    reordered_data = replace(
        resolved.config.data,
        train=(
            resolved.config.data.train[1],
            resolved.config.data.train[0],
            resolved.config.data.train[2],
        ),
    )
    reordered = prepare_scratch_training_run(
        replace(resolved.config, data=reordered_data),
        automatic_reference_preparation=automatic,
    )
    assert reordered.train_semantic_digest != original.train_semantic_digest
    assert reordered.data_manifest["fingerprint"] != original.data_manifest["fingerprint"]

    combined_path = tmp_path / "data" / "small-combined.xyz"
    combined = read(tmp_path / "data" / "small-pristine.xyz", index=":") + read(
        tmp_path / "data" / "small-vacancy.xyz", index=":"
    )
    write(combined_path, combined, format="extxyz")
    combined_data = replace(
        resolved.config.data,
        train=(
            TrainingDataSourceConfig(
                path="data/small-combined.xyz",
                template_id=resolved.config.data.train[0].template_id,
                reference_alias="POSCAR_222",
            ),
            resolved.config.data.train[2],
        ),
    )
    combined_prepared = prepare_scratch_training_run(
        replace(resolved.config, data=combined_data),
        automatic_reference_preparation=automatic,
    )
    assert combined_prepared.train_semantic_digest != original.train_semantic_digest
    assert [item["frame_count"] for item in combined_prepared.data_manifest["train"]["sources"]] == [2, 1]

    changed_path = tmp_path / "data" / "small-pristine-changed.xyz"
    changed = read(tmp_path / "data" / "small-pristine.xyz", index=0)
    changed.calc = SinglePointCalculator(changed, energy=123.0)
    write(changed_path, changed, format="extxyz")
    changed_sources = (
        replace(resolved.config.data.train[0], path="data/small-pristine-changed.xyz"),
        *resolved.config.data.train[1:],
    )
    changed_prepared = prepare_scratch_training_run(
        replace(
            resolved.config,
            data=replace(resolved.config.data, train=changed_sources),
        ),
        automatic_reference_preparation=automatic,
    )
    assert changed_prepared.train_semantic_digest != original.train_semantic_digest

    wrong = _payload()
    wrong["data"]["train"][2]["reference"] = "POSCAR_222"
    wrong_recipe = tmp_path / "wrong.yaml"
    wrong_recipe.write_text(yaml.safe_dump(wrong, sort_keys=False), encoding="utf-8")
    with pytest.raises(TrainingRecipeError) as caught:
        resolve_training_recipe(wrong_recipe)
    assert caught.value.reason_code == "NO_COMPATIBLE_TEMPLATE"
    assert caught.value.original_error.sample_id == "train.0002:000000"
    diagnostics = caught.value.original_error.diagnostics
    assert next(
        item for item in diagnostics if item["reference_alias"] == "POSCAR_222"
    )["approved"] is False


def test_duplicate_data_leakage_and_duplicate_reference_content_fail_early(tmp_path):
    payload = _payload()
    payload["data"]["train"][1]["file"] = payload["data"]["train"][0]["file"]
    recipe = tmp_path / "duplicate-data.yaml"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(TrainingRecipeError) as caught:
        resolve_training_recipe(recipe)
    assert caught.value.reason_code == "DUPLICATE_DATA_SOURCE"

    payload = _payload()
    payload["data"]["validation"][0]["file"] = payload["data"]["train"][0]["file"]
    recipe = tmp_path / "leakage.yaml"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(TrainingRecipeError) as caught:
        resolve_training_recipe(recipe)
    assert caught.value.reason_code == "TRAIN_VALIDATION_DATA_LEAKAGE"

    recipe = _write_case(tmp_path / "duplicate-reference")
    duplicate_root = recipe.parent
    (duplicate_root / "references" / "POSCAR-large").write_bytes(
        (duplicate_root / "references" / "POSCAR-small").read_bytes()
    )
    with pytest.raises(TrainingRecipeError) as caught:
        resolve_training_recipe(recipe)
    assert caught.value.reason_code == "DUPLICATE_REFERENCE_CONTENT"


def test_validate_and_train_dry_run_share_bound_assignment_manifest(
    tmp_path, capsys
):
    recipe = _write_case(tmp_path)
    assert main(["validate-train-config", str(recipe), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert main(["train", str(recipe), "--dry-run", "--json", "--quiet"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run == validated
    assert len(validated["data_manifest"]["train"]["sources"]) == 3
    assert len(validated["data_manifest"]["validation"]["sources"]) == 2
    assert validated["side_effects"] == {
        "initial_bundle_created": False,
        "model_parameters_created": False,
        "optimizer_created": False,
        "output_directory_created": False,
    }
    assert not (tmp_path / "runs" / "bound-run").exists()
