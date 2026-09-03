from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

import refsite_mlip.models.potential as potential_module
import refsite_mlip.training.optimizer as optimizer_module
from refsite_mlip.cli.errors import CLIError
from refsite_mlip.cli.main import main
from refsite_mlip.cli.validate_train_config import (
    render_train_config_human,
    render_train_config_json,
    validate_train_config,
)
from refsite_mlip.config import load_training_run_config, resolve_training_run
from refsite_mlip.data import (
    ReferenceTemplate,
    StrictTemplateDomain,
    capture_reference_structure_artifact,
)
from refsite_mlip.models import (
    capture_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.training import (
    AtomicBaselineConfig,
    CheckpointedFitConfig,
    FitConfig,
    LossConfig,
    ModelSelectionConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
)
from refsite_mlip.transport import TransportSupportConfig

from test_cli_inspect_bundle import _typed_crystal_data
from test_model_bundle_runtime import _capture_case


@pytest.fixture(scope="module")
def training_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("training-run-preflight-bundle")
    _, model, registry, samples, _, _, _, original = _capture_case(
        _typed_crystal_data()
    )
    support = TransportSupportConfig(
        kind="compact_c2",
        cutoff=4.0,
        switch_width=0.5,
        candidate_skin=0.2,
    )
    configured = type(model)(
        replace(model.config, transport_support=support),
        model.topology,
        model.phase_modes,
        model.phase_mode_weights,
        model.species_alignment_weights,
        model.site_alignment_weights,
        model.phase_channel_weights,
        model.atomic_baseline,
    ).to(model.atomic_baseline)
    configured.load_state_dict(model.state_dict(), strict=True)
    model = configured
    original_bindings = {
        binding.template_id: binding for binding in original.template_bindings
    }
    artifacts = {}
    phases = {}
    for template_id in ("alpha", "zeta"):
        template = registry.resolve(template_id)
        reference_composition = tuple(
            int(torch.count_nonzero(template.topology.site_types == index))
            for index in range(len(template.supported_species))
        )
        matching = tuple(
            sample for sample in samples if sample.template_id == template_id
        )
        compositions = []
        for sample in matching:
            composition = tuple(
                int(torch.count_nonzero(sample.atomic_numbers == species))
                for species in template.supported_species
            )
            if composition not in compositions:
                compositions.append(composition)
        domain = StrictTemplateDomain(
            reference_site_count=template.topology.num_sites,
            supercell_shape=(1, 1, 1),
            species_vocabulary=template.supported_species,
            reference_composition=reference_composition,
            allowed_compositions=tuple(compositions),
            allowed_num_atoms=tuple(sum(value) for value in compositions),
            allowed_vacancy_masses=tuple(
                template.topology.num_sites - sum(value) for value in compositions
            ),
        )
        strict_template = ReferenceTemplate.snapshot(
            template.template_id,
            template.topology,
            template.phase_modes,
            template.phase_mode_weights,
            template.site_alignment_weights,
            template.phase_channel_weights,
            template.stabilizer,
            template.supported_species,
            convention_version=template.convention_version,
            strict_domain=domain,
        )
        artifacts[template_id] = capture_reference_structure_artifact(
            strict_template, avg_num_neighbors=6.0
        )
        phases[template_id] = original_bindings[template_id].phase_specification
    bundle = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts=artifacts,
        phase_specifications=phases,
        evaluation_policies=None,
        default_template_id=original.default_template_id,
        provenance={"purpose": "training-run-preflight"},
    )
    path = directory / "initial-model.pt"
    save_reference_site_model_bundle(path, bundle)
    return {"path": path, "bundle": bundle, "samples": samples}


def _atoms(
    sample,
    *,
    energy: float | None,
    forces: bool = False,
    stress: bool = False,
    template_key: str | None = None,
    partial_masks: bool = False,
):
    info = {}
    if template_key is not None:
        info[template_key] = sample.template_id
    atoms = Atoms(
        numbers=sample.atomic_numbers.detach().cpu().numpy(),
        positions=sample.positions.detach().cpu().numpy(),
        cell=sample.cell.detach().cpu().numpy(),
        pbc=sample.pbc.detach().cpu().numpy(),
        info=info,
    )
    results = {}
    if energy is not None:
        results["energy"] = energy
    if forces:
        results["forces"] = np.arange(len(atoms) * 3, dtype=float).reshape(-1, 3) / 10
    if stress:
        results["stress"] = np.array([0.1, 0.2, 0.3, 0.04, 0.05, 0.06])
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    if partial_masks and forces:
        mask = np.ones((len(atoms), 3), dtype=bool)
        mask[0, 1] = False
        atoms.arrays["force_mask"] = mask
    if partial_masks and stress:
        atoms.info["stress_mask"] = np.array(
            [True, True, False, True, False, True], dtype=bool
        )
    return atoms


def _write_frames(path: Path, frames) -> None:
    write(path, list(frames), format="extxyz")


def _base_payload(
    *,
    bundle: str,
    train_sources: list[dict],
    validation_sources: list[dict],
    output: str = "run-output",
) -> dict:
    return {
        "schema_version": "refsite_training_run_config_v1",
        "initial_bundle": bundle,
        "radii": {"r_ot": 4.0, "r_mp": 3.0},
        "data": {
            "train": train_sources,
            "validation": validation_sources,
            "batch_size": 2,
            "shuffle": False,
        },
        "runtime": {"device": "cpu", "dtype": "float64", "seed": 17},
        "loss": LossConfig().to_dict(),
        "baseline": AtomicBaselineConfig().to_dict(),
        "optimizer": OptimizerConfig().to_dict(),
        "train_step": TrainStepConfig().to_dict(),
        "validation_step": ValidationStepConfig().to_dict(),
        "scheduler": SchedulerConfig().to_dict(),
        "selection": ModelSelectionConfig().to_dict(),
        "fit": FitConfig(max_epochs=2).to_dict(),
        "checkpointed_fit": CheckpointedFitConfig().to_dict(),
        "output_directory": output,
    }


def _simple_case(tmp_path: Path, training_bundle) -> tuple[Path, dict]:
    samples = training_bundle["samples"]
    train = tmp_path / "train.xyz"
    validation = tmp_path / "validation.xyz"
    _write_frames(
        train,
        (
            _atoms(samples[0], energy=5.0),
            _atoms(samples[2], energy=6.0),
        ),
    )
    _write_frames(validation, (_atoms(samples[2], energy=6.25),))
    payload = _base_payload(
        bundle=str(training_bundle["path"]),
        train_sources=[{"path": train.name, "template_id": "zeta"}],
        validation_sources=[
            {"path": validation.name, "template_id": "zeta"}
        ],
    )
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return config_path, payload


def test_cli_relative_paths_deterministic_output_and_no_execution(
    training_bundle, tmp_path, monkeypatch, capsys
):
    config_path, _ = _simple_case(tmp_path, training_bundle)
    bundle_before = training_bundle["path"].read_bytes()
    input_before = (tmp_path / "train.xyz").read_bytes()
    rng_before = torch.random.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("training/model execution must not be called")

    monkeypatch.setattr(potential_module.ReferenceSitePotential, "forward", forbidden)
    monkeypatch.setattr(optimizer_module, "build_optimizer", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)

    first = validate_train_config(config_path)
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    second = validate_train_config(config_path)
    assert render_train_config_json(first) == render_train_config_json(second)
    report = json.loads(render_train_config_json(first))
    assert report["status"] == "preflight_ready"
    assert report["training_executed"] is False
    assert report["data"]["train"]["frame_count"] == 2
    assert report["data"]["train"]["batch_count"] == 1
    assert report["baseline_preflight"]["rank"] == 2
    assert report["runtime"]["paths"]["config"] == str(config_path.resolve())
    assert report["runtime"]["paths"]["path_kind"].startswith("runtime_location")
    assert report["runtime"]["configured_paths"]["train_inputs"] == [
        "train.xyz"
    ]
    assert not (tmp_path / "run-output").exists()
    assert training_bundle["path"].read_bytes() == bundle_before
    assert (tmp_path / "train.xyz").read_bytes() == input_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    human = render_train_config_human(first)
    assert "No training was executed." in human
    assert "Config SHA-256:" in human
    assert "Train semantic SHA-256:" in human
    assert "r_on_ot=3.5" in human and "r_candidate_mp=3.5" in human

    assert main(["validate-train-config", str(config_path), "--json"]) == 0
    first_cli = capsys.readouterr()
    assert json.loads(first_cli.out)["training_executed"] is False
    assert first_cli.err == ""
    assert main(["validate-train-config", str(config_path)]) == 0
    second_cli = capsys.readouterr()
    assert "No training was executed." in second_cli.out
    assert second_cli.err == ""


def test_mixed_template_key_partial_masks_digests_order_and_plain_metadata(
    training_bundle, tmp_path
):
    samples = training_bundle["samples"]
    train = tmp_path / "mixed-train.xyz"
    validation = tmp_path / "mixed-validation.xyz"
    train_frames = (
        _atoms(
            samples[0],
            energy=5.0,
            forces=True,
            stress=True,
            template_key="template",
            partial_masks=True,
        ),
        _atoms(samples[1], energy=4.0, template_key="template"),
    )
    _write_frames(train, train_frames)
    _write_frames(
        validation,
        (
            _atoms(
                samples[2],
                energy=6.0,
                forces=True,
                stress=True,
                template_key="template",
                partial_masks=True,
            ),
        ),
    )
    payload = _base_payload(
        bundle=str(training_bundle["path"]),
        train_sources=[{"path": train.name, "template_key": "template"}],
        validation_sources=[
            {"path": validation.name, "template_key": "template"}
        ],
        output="mixed-output",
    )
    payload["loss"].update(
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=1.0,
    )
    config_path = tmp_path / "mixed.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    resolved = resolve_training_run(load_training_run_config(config_path))
    report = resolved.to_dict()

    assert report["data"]["train"]["template_frame_counts"] == {
        "alpha": 1,
        "zeta": 1,
    }
    train_labels = report["data"]["train"]["label_statistics"]
    assert train_labels["energy"] == {
        "present_frames": 2,
        "missing_frames": 0,
        "valid_count": 2,
    }
    assert train_labels["forces"]["present_frames"] == 1
    assert train_labels["forces"]["missing_frames"] == 1
    assert train_labels["forces"]["valid_count"] == len(samples[0].positions) * 3 - 1
    assert train_labels["stress"]["valid_count"] == 4
    assert tuple(report["template_fingerprints"]) == ("alpha", "zeta")
    assert set(report["template_fingerprints"]["zeta"]) == {
        "structural_artifact_fingerprint",
        "full_template_fingerprint",
        "phase_specification_fingerprint",
        "binding_fingerprint",
        "evaluation_policy_fingerprint",
    }
    assert all(
        len(value) == 64
        for name, value in report["template_fingerprints"]["zeta"].items()
        if name != "evaluation_policy_fingerprint"
    )
    assert len(report["data"]["train"]["semantic_digest"]) == 64

    reversed_path = tmp_path / "mixed-reversed.xyz"
    _write_frames(reversed_path, tuple(reversed(train_frames)))
    reversed_payload = copy.deepcopy(payload)
    reversed_payload["data"]["train"][0]["path"] = reversed_path.name
    reversed_payload["output_directory"] = "reversed-output"
    reversed_config = tmp_path / "mixed-reversed.json"
    reversed_config.write_text(json.dumps(reversed_payload), encoding="utf-8")
    reversed_report = resolve_training_run(
        load_training_run_config(reversed_config)
    ).to_dict()
    assert (
        reversed_report["data"]["train"]["semantic_digest"]
        != report["data"]["train"]["semantic_digest"]
    )


@pytest.mark.parametrize(
    ("mutation", "reason", "template"),
    [
        (
            lambda payload: payload["radii"].update({"r_ot": 4.1}),
            "RADIUS_MODEL_MISMATCH",
            None,
        ),
        (
            lambda payload: payload["radii"].update({"r_mp": 3.1}),
            "RADIUS_MODEL_MISMATCH",
            None,
        ),
        (
            lambda payload: payload["data"]["train"][0].update(
                {"template_id": "missing-template"}
            ),
            "UNKNOWN_TEMPLATE",
            "missing-template",
        ),
    ],
)
def test_radius_and_template_compatibility_errors_are_contextual(
    training_bundle, tmp_path, mutation, reason, template
):
    config_path, payload = _simple_case(tmp_path, training_bundle)
    mutation(payload)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CLIError) as caught:
        validate_train_config(config_path)
    assert caught.value.reason_code == reason
    assert caught.value.config_field is not None
    assert caught.value.template_id == template
    assert caught.value.underlying_reason_code == reason


def test_supervision_baseline_monitor_and_output_preflight_failures(
    training_bundle, tmp_path
):
    config_path, payload = _simple_case(tmp_path, training_bundle)

    rank_deficient = copy.deepcopy(payload)
    rank_deficient["data"]["train"] = [
        {"path": "validation.xyz", "template_id": "zeta"}
    ]
    rank_deficient["output_directory"] = "rank-output"
    rank_path = tmp_path / "rank.json"
    rank_path.write_text(json.dumps(rank_deficient), encoding="utf-8")
    with pytest.raises(CLIError) as caught:
        validate_train_config(rank_path)
    assert caught.value.reason_code == "BASELINE_PREFLIGHT_FAILED"
    assert "rank deficient" in caught.value.message

    no_force = copy.deepcopy(payload)
    no_force["loss"].update(energy_weight=1.0, force_weight=1.0)
    no_force["scheduler"]["monitor"] = "force"
    no_force["selection"]["monitor"] = "force"
    no_force["output_directory"] = "force-output"
    force_path = tmp_path / "force.json"
    force_path.write_text(json.dumps(no_force), encoding="utf-8")
    with pytest.raises(CLIError) as caught:
        validate_train_config(force_path)
    assert caught.value.reason_code == "MISSING_TRAIN_SUPERVISION"
    assert caught.value.split == "train"

    collision = copy.deepcopy(payload)
    collision["output_directory"] = "train.xyz"
    collision_path = tmp_path / "collision.json"
    collision_path.write_text(json.dumps(collision), encoding="utf-8")
    with pytest.raises(CLIError) as caught:
        validate_train_config(collision_path)
    assert caught.value.reason_code == "OUTPUT_PATH_COLLISION"
    assert (tmp_path / "train.xyz").is_file()

    existing = copy.deepcopy(payload)
    existing["output_directory"] = "existing-output"
    (tmp_path / "existing-output").mkdir()
    existing_path = tmp_path / "existing.json"
    existing_path.write_text(json.dumps(existing), encoding="utf-8")
    with pytest.raises(CLIError) as caught:
        validate_train_config(existing_path)
    assert caught.value.reason_code == "OUTPUT_ALREADY_EXISTS"


def test_cuda_preflight_and_cli_error_exit_contract(
    training_bundle, tmp_path, monkeypatch, capsys
):
    config_path, payload = _simple_case(tmp_path, training_bundle)
    payload["runtime"]["device"] = "cuda"
    payload["output_directory"] = "cuda-output"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["validate-train-config", str(config_path), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "reason='CUDA_UNAVAILABLE'" in captured.err
    assert "Traceback" not in captured.err

    payload["runtime"]["device"] = "cuda:7"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(CLIError) as caught:
        validate_train_config(config_path)
    assert caught.value.reason_code == "CUDA_DEVICE_INDEX_INVALID"

    with pytest.raises(SystemExit) as caught:
        main(["validate-train-config"])
    assert caught.value.code == 2
