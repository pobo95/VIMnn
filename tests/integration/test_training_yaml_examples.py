from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("ase")

from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from refsite_mlip.cli.main import main
from refsite_mlip.config import TrainingRecipeConfig, TrainingRunConfig


_REPOSITORY = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPOSITORY / "examples" / "training"
_LATTICE = 5.98


def _reference(*, repeated: bool = False):
    atoms = bulk("NbC", "rocksalt", a=_LATTICE, cubic=True)
    fractional = atoms.get_scaled_positions(wrap=True)
    fractional[atoms.numbers == 6] += np.array([0.017, 0.011, 0.007])
    atoms.set_scaled_positions(fractional)
    return atoms.repeat((2, 1, 1)) if repeated else atoms


def _vacancy(atoms, *, species: int):
    result = atoms.copy()
    index = int(np.flatnonzero(result.numbers == species)[0])
    del result[index]
    return result


def _labeled(atoms, *, energy: float):
    result = atoms.copy()
    count = len(result)
    forces = np.arange(count * 3, dtype=np.float64).reshape(count, 3)
    forces = (forces - forces.mean()) * 1.0e-4
    stress = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])
    result.calc = SinglePointCalculator(
        result,
        energy=float(energy),
        forces=forces,
        stress=stress,
    )
    return result


def _copy_example(tmp_path: Path, filename: str) -> Path:
    (tmp_path / "references").mkdir()
    (tmp_path / "data").mkdir()
    destination = tmp_path / filename
    shutil.copyfile(_EXAMPLES / filename, destination)
    return destination


def _invoke_json(arguments: list[str], capsys):
    exit_code = main(arguments)
    captured = capsys.readouterr()
    assert captured.out, (
        f"CLI produced no JSON: exit_code={exit_code}, stderr={captured.err!r}"
    )
    assert captured.out.count("\n") <= 1
    return exit_code, json.loads(captured.out), captured.err


@pytest.fixture(autouse=True)
def _forbid_training_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("example dry-run must not execute training or optimizer.step")

    monkeypatch.setattr(
        "refsite_mlip.training.run_scratch_checkpointed_training", forbidden
    )
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)


def _run_read_only_cli(recipe: Path, capsys, *overrides: str):
    common = [str(recipe), *overrides, "--json"]
    code, resolution, resolve_stderr = _invoke_json(
        ["resolve-train-config", *common, "--dry-run"], capsys
    )
    assert code == 0 and resolve_stderr == ""
    code, validation, validate_stderr = _invoke_json(
        ["validate-train-config", *common], capsys
    )
    assert code == 0 and validate_stderr == ""
    code, train_dry_run, train_stderr = _invoke_json(
        ["train", *common, "--dry-run", "--quiet"], capsys
    )
    assert code == 0 and train_stderr == ""
    assert validation == train_dry_run
    assert resolution["reference_preparation"] == validation["reference_preparation"]
    assert TrainingRunConfig.from_dict(
        resolution["compiled_config"]
    ).config_fingerprint == validation["config_fingerprint"]
    assert validation["side_effects"] == {
        "initial_bundle_created": False,
        "model_parameters_created": False,
        "optimizer_created": False,
        "output_directory_created": False,
    }
    return resolution, validation


def _assert_explicit_leaf_output(
    recipe: Path,
    resolution: dict,
    *,
    expected_leaf: str,
) -> None:
    authored = TrainingRecipeConfig.from_dict(
        yaml.safe_load(recipe.read_text(encoding="utf-8"))
    )
    assert authored.name is None
    assert authored.output_directory == f"./{expected_leaf}"
    compiled = TrainingRunConfig.from_dict(resolution["compiled_config"])
    assert compiled.output_directory == f"./{expected_leaf}"
    output_path = next(
        item
        for item in resolution["resolution_manifest"]["paths"]
        if item["field"] == "output_directory"
    )
    assert output_path == {
        "field": "output_directory",
        "original": f"./{expected_leaf}",
        "resolved": str(recipe.parent / expected_leaf),
    }
    assert not (recipe.parent / "runs").exists()
    assert not (recipe.parent / expected_leaf).exists()


def test_minimal_example_uses_defaults_and_is_a_deterministic_read_only_dry_run(
    tmp_path, capsys
):
    recipe = _copy_example(tmp_path, "minimal.yaml")
    reference = _reference()
    carbon_vacancy = _vacancy(reference, species=6)
    niobium_vacancy = _vacancy(reference, species=41)
    write(tmp_path / "references" / "POSCAR", reference, format="vasp", direct=True)
    write(
        tmp_path / "data" / "train.xyz",
        [_labeled(reference, energy=-8.0), _labeled(carbon_vacancy, energy=-6.4)],
        format="extxyz",
    )
    write(
        tmp_path / "data" / "validation.xyz",
        [_labeled(niobium_vacancy, energy=-6.2)],
        format="extxyz",
    )

    resolution, validation = _run_read_only_cli(recipe, capsys)
    _assert_explicit_leaf_output(
        recipe, resolution, expected_leaf="my-first-run"
    )
    compiled = TrainingRunConfig.from_dict(resolution["compiled_config"])
    higher = compiled.model_source.potential.higher_body
    assert compiled.model_source.potential.num_layers == 2
    assert higher.n_correlation_channels == 8
    assert higher.lmax == 2
    assert higher.symmetric_correlation.correlation_order == 3
    assert compiled.data.batch_size == 4
    assert compiled.data.effective_validation_batch_size == 4
    assert compiled.loss.energy_weight == 1.0
    assert compiled.loss.force_weight == 100.0
    assert compiled.loss.stress_weight == 0.0
    assert compiled.fit.max_epochs == 100
    assert compiled.optimizer.learning_rate == 1.0e-3
    assert compiled.runtime.device == "cpu"
    assert compiled.runtime.dtype == "float64"
    assert validation["baseline_preflight"]["rank"] == 2
    assert validation["baseline_preflight"]["required_rank"] == 2
    assert validation["data"]["train"]["frame_count"] == 2
    assert validation["data"]["validation"]["frame_count"] == 1
    assert resolution["recipe_summary"] == {
        "correlation_method": "symmetric",
        "maximum_correlation_order": 3,
        "inference_ot_solver": "sinkhorn",
        "inference_application": "prediction/evaluation call-time preference",
        "training_ot_solver": "sinkhorn",
        "training_batch_size": 4,
        "validation_batch_size": 4,
    }
    assert not (tmp_path / "my-first-run").exists()


def test_advanced_example_binds_files_to_aliases_and_qualifies_inference(
    tmp_path, capsys
):
    recipe = _copy_example(tmp_path, "advanced.yaml")
    small = _reference()
    large = _reference(repeated=True)
    small_c_vacancy = _vacancy(small, species=6)
    small_nb_vacancy = _vacancy(small, species=41)
    large_c_vacancy = _vacancy(large, species=6)
    write(
        tmp_path / "references" / "POSCAR_222",
        small,
        format="vasp",
        direct=True,
    )
    write(
        tmp_path / "references" / "POSCAR_333",
        large,
        format="vasp",
        direct=True,
    )
    files = {
        "train_pristine_222.xyz": [_labeled(small, energy=-8.0)],
        "train_vacancy_222.xyz": [_labeled(small_c_vacancy, energy=-6.4)],
        "train_333.xyz": [_labeled(large, energy=-16.0)],
        "validation_222.xyz": [_labeled(small_nb_vacancy, energy=-6.3)],
        "validation_333.xyz": [_labeled(large_c_vacancy, energy=-14.2)],
    }
    for filename, frames in files.items():
        write(tmp_path / "data" / filename, frames, format="extxyz")

    # The source example remains cuda/float32; CPU is a read-only test override.
    resolution, validation = _run_read_only_cli(
        recipe, capsys, "--device", "cpu"
    )
    _assert_explicit_leaf_output(
        recipe, resolution, expected_leaf="advanced-run"
    )
    authored = TrainingRecipeConfig.from_dict(
        yaml.safe_load((_EXAMPLES / "advanced.yaml").read_text(encoding="utf-8"))
    )
    assert authored.runtime.device == "cuda"
    assert authored.runtime.dtype == "float32"

    compiled = TrainingRunConfig.from_dict(resolution["compiled_config"])
    higher = compiled.model_source.potential.higher_body
    assert compiled.model_source.potential.num_layers == 2
    assert higher.n_correlation_channels == 64
    assert higher.lmax == 2
    assert higher.symmetric_correlation.correlation_order == 3
    assert compiled.radii.r_ot == 4.0
    assert compiled.radii.r_mp == 3.0
    assert compiled.loss.energy_weight == 1.0
    assert compiled.loss.force_weight == 100.0
    assert compiled.loss.stress_weight == 10.0
    assert compiled.data.batch_size == 5
    assert compiled.data.effective_validation_batch_size == 5
    assert compiled.fit.max_epochs == 500
    assert compiled.optimizer.learning_rate == 1.0e-3
    assert compiled.runtime.device == "cpu"
    assert compiled.runtime.dtype == "float32"
    assert resolution["recipe_summary"]["training_ot_solver"] == "sinkhorn"
    assert resolution["recipe_summary"]["inference_ot_solver"] == (
        "sinkhorn_newton_krylov"
    )

    references = validation["reference_preparation"]["references"]
    by_sites = {item["num_reference_sites"]: item["template_id"] for item in references}
    assert set(by_sites) == {8, 16}
    assert all(item["evaluation_certificate"]["status"] == "qualified" for item in references)
    train = validation["data_manifest"]["train"]
    valid = validation["data_manifest"]["validation"]
    assert [item["reference_alias"] for item in train["sources"]] == [
        "POSCAR_222",
        "POSCAR_222",
        "POSCAR_333",
    ]
    assert [item["template_id"] for item in train["sources"]] == [
        by_sites[8],
        by_sites[8],
        by_sites[16],
    ]
    assert [item["reference_alias"] for item in valid["sources"]] == [
        "POSCAR_222",
        "POSCAR_333",
    ]
    assert [item["vacancy_mass"] for item in train["samples"]] == [0, 1, 0]
    assert [item["vacancy_mass"] for item in valid["samples"]] == [1, 1]
    assert [item["num_sites"] for item in train["samples"]] == [8, 8, 16]
    assert validation["baseline_preflight"]["rank"] == 2
    assert validation["baseline_preflight"]["required_rank"] == 2
    assert not (tmp_path / "advanced-run").exists()


@pytest.mark.parametrize("filename", ["minimal.yaml", "advanced.yaml"])
def test_example_yaml_uses_only_live_public_recipe_vocabulary(filename):
    path = _EXAMPLES / filename
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert "name" not in payload
    assert set(payload) >= {"schema_version", "output_directory"}
    recipe = TrainingRecipeConfig.from_dict(payload)
    assert recipe.name is None
    assert recipe.output_directory == (
        "./my-first-run" if filename == "minimal.yaml" else "./advanced-run"
    )
    assert TrainingRecipeConfig.from_dict(recipe.to_dict()).to_dict() == recipe.to_dict()

    lower = text.lower()
    for forbidden in (
        "train_fixed",
        "eval_adaptive",
        "train-fixed",
        "eval-adaptive",
        "central_conditioned_higher_body_v1",
        "central_conditioned_symmetric_power_v2",
        "contract_version",
        "phase mode",
        "stabilizer",
        "fingerprint",
        "template_id",
        "architecture:",
    ):
        assert forbidden not in lower
    without_schema = lower.replace("refsite_training_recipe_v1", "")
    assert re.search(r"(?<![a-z0-9_])v[12](?![a-z0-9_])", without_schema) is None
    assert "&" not in text and "*" not in text and "<<:" not in text
    assert "${" not in text
    assert recipe.model.correlation_method == "symmetric"
    assert recipe.ot_solver.training == "sinkhorn"
    assert recipe.ot_solver.inference in {"sinkhorn", "sinkhorn_newton_krylov"}

    if filename == "minimal.yaml":
        assert recipe.model.hidden_channels == 8
        assert recipe.model.num_interactions == 2
        assert recipe.model.max_L == 2
        assert recipe.model.correlation == 3
        assert recipe.loss.energy_weight == 1.0
        assert recipe.loss.forces_weight == 100.0
        assert recipe.loss.stress_weight == 0.0
        assert recipe.runtime.device == "cpu"
        assert recipe.runtime.dtype == "float64"
    else:
        assert len(recipe.reference.sources) == 2
        assert [source.authoring_alias for source in recipe.reference.sources] == [
            "POSCAR_222",
            "POSCAR_333",
        ]
        assert recipe.model.hidden_channels == 64
        assert recipe.model.correlation == 3
        assert recipe.training.batch_size == 5
        assert recipe.training.validation_batch_size == 5
        assert recipe.ot_solver.inference == "sinkhorn_newton_krylov"
