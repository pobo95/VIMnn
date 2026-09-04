from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("ase")
from ase import Atom
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from refsite_mlip.cli.main import main
from refsite_mlip.cli.validate_train_config import (
    render_train_config_human,
    render_train_config_json,
)
from refsite_mlip.config import (
    TrainingRunConfigError,
    load_training_run_config,
)
from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceTemplateBuilderConfig,
    StrictTemplateDomain,
    build_reference_template_from_poscar,
    capture_reference_structure_artifact,
)
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import EvaluationPolicy, PotentialConfig
from refsite_mlip.training import (
    AtomicBaselineConfig,
    CheckpointedFitConfig,
    FitConfig,
    LossConfig,
    ModelSelectionConfig,
    OptimizerConfig,
    SchedulerConfig,
    ScratchTrainingPreparation,
    TrainStepConfig,
    ValidationStepConfig,
    prepare_scratch_training_run,
)
from refsite_mlip.training.scratch_preparation import (
    SCRATCH_INPUT_FILE_DIGEST_CONVENTION_VERSION,
    verify_scratch_preparation_input_digests,
)
from refsite_mlip.transport import TransportSupportConfig


LATTICE = 4.482314244155584


def _atoms(size: int):
    return bulk("NbC", "rocksalt", a=LATTICE, cubic=True).repeat(
        (size, size, size)
    )


def _phase(size: int) -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.tensor(
            [
                [-size, size, size],
                [size, -size, size],
                [size, size, -size],
                [2 * size, 0, 0],
                [0, 2 * size, 0],
                [0, 0, 2 * size],
            ],
            dtype=torch.long,
        ),
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )


def _config_111() -> ReferenceTemplateBuilderConfig:
    return ReferenceTemplateBuilderConfig(
        template_id="scratch-111-a",
        strict_domain=StrictTemplateDomain(
            reference_site_count=8,
            supercell_shape=(1, 1, 1),
            species_vocabulary=(6, 41),
            reference_composition=(4, 4),
            allowed_compositions=((4, 4), (3, 4)),
            allowed_num_atoms=(8, 7),
            allowed_vacancy_masses=(0, 1),
        ),
        site_type_ids=(0, 1),
        expected_stabilizer_size=4,
    )


def _config_211() -> ReferenceTemplateBuilderConfig:
    return ReferenceTemplateBuilderConfig(
        template_id="scratch-211",
        strict_domain=StrictTemplateDomain(
            reference_site_count=16,
            supercell_shape=(2, 1, 1),
            species_vocabulary=(6, 41),
            reference_composition=(8, 8),
            allowed_compositions=((8, 8), (7, 8)),
            allowed_num_atoms=(16, 15),
            allowed_vacancy_masses=(0, 1),
        ),
        site_type_ids=(0, 1),
        expected_stabilizer_size=8,
    )


def _phase_211() -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.tensor(
            [
                [-2, 1, 1],
                [2, -1, 1],
                [2, 1, -1],
                [4, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ],
            dtype=torch.long,
        ),
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )


def _potential() -> PotentialConfig:
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=2,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        probability_tolerance=1.0e-8,
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
        transport_support=TransportSupportConfig(
            kind="compact_c2",
            cutoff=4.0,
            switch_width=0.5,
            candidate_skin=0.2,
        ),
    )


def _v2_payload() -> dict:
    return {
        "schema_version": "refsite_training_run_config_v2",
        "model_source": {
            "kind": "scratch",
            "initialization_seed": 20260904,
            "potential": _potential().to_dict(),
            "species_alignment_weights": [[1.0, -0.5], [-1.0, 2.0]],
            "reference_templates": [],
            "default_template_id": "scratch-111-a",
        },
        "radii": {"r_ot": 4.0, "r_mp": 3.0},
        "data": {
            "train": [],
            "validation": [],
            "batch_size": 2,
            "validation_batch_size": 1,
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
        "output_directory": "scratch-output",
    }


def _labeled(atoms, energy: float):
    result = atoms.copy()
    result.calc = SinglePointCalculator(result, energy=float(energy))
    return result


def _partially_labeled(atoms, energy: float):
    result = atoms.copy()
    forces = np.arange(len(result) * 3, dtype=float).reshape(-1, 3) / 10.0
    stress = np.array([0.1, 0.2, 0.3, 0.04, 0.05, 0.06])
    result.calc = SinglePointCalculator(
        result,
        energy=float(energy),
        forces=forces,
        stress=stress,
    )
    force_mask = np.ones((len(result), 3), dtype=bool)
    force_mask[0, 1] = False
    result.arrays["force_mask"] = force_mask
    result.info["stress_mask"] = np.array(
        [True, True, False, True, False, True], dtype=bool
    )
    return result


def _vacancy(atoms):
    result = atoms.copy()
    carbon = next(
        index
        for index, atomic_number in enumerate(result.numbers)
        if int(atomic_number) == 6
    )
    del result[carbon]
    return result


def _two_vacancies(atoms):
    result = atoms.copy()
    carbon = [
        index for index, atomic_number in enumerate(result.numbers)
        if int(atomic_number) == 6
    ]
    del result[carbon[:2]]
    return result


def _only_niobium(atoms):
    result = atoms.copy()
    carbon = [
        index
        for index, atomic_number in enumerate(result.numbers)
        if int(atomic_number) == 6
    ]
    del result[carbon]
    return result


def _write_extxyz(path: Path, frames) -> None:
    write(path, list(frames), format="extxyz")


def _template_payload(
    *,
    template_id: str,
    poscar_path: str,
    pristine_only: bool = False,
    evaluation_policy: dict | None = None,
    builder: ReferenceTemplateBuilderConfig | None = None,
    phase: PhaseSpecification | None = None,
) -> dict:
    builder = replace(
        _config_111() if builder is None else builder,
        template_id=template_id,
    )
    if pristine_only:
        domain = builder.strict_domain
        builder = replace(
            builder,
            strict_domain=StrictTemplateDomain(
                reference_site_count=domain.reference_site_count,
                supercell_shape=domain.supercell_shape,
                species_vocabulary=domain.species_vocabulary,
                reference_composition=domain.reference_composition,
                allowed_compositions=(domain.reference_composition,),
                allowed_num_atoms=(domain.reference_site_count,),
                allowed_vacancy_masses=(0,),
                convention_version=domain.convention_version,
            ),
        )
    return {
        "poscar_path": poscar_path,
        "builder": builder.to_dict(),
        "phase_specification": (_phase(1) if phase is None else phase).to_dict(),
        "evaluation_policy": evaluation_policy,
    }


def _case(
    directory: Path,
    *,
    train_frames,
    validation_frames,
    selector: dict,
    templates: tuple[dict, ...] | None = None,
    baseline: bool = False,
) -> tuple[Path, Path, Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    reference = _atoms(1)
    poscar = directory / "reference.POSCAR"
    write(poscar, reference, format="vasp", direct=True)
    train = directory / "train.xyz"
    validation = directory / "validation.xyz"
    _write_extxyz(train, train_frames)
    _write_extxyz(validation, validation_frames)

    payload = _v2_payload()
    if templates is None:
        templates = (
            _template_payload(
                template_id="scratch-111-a",
                poscar_path=poscar.name,
            ),
        )
    payload["model_source"]["reference_templates"] = list(templates)
    payload["model_source"]["default_template_id"] = templates[0]["builder"][
        "template_id"
    ]
    payload["data"] = {
        "train": [{"path": train.name, **selector}],
        "validation": [{"path": validation.name, **selector}],
        "batch_size": 2,
        "validation_batch_size": 1,
        "shuffle": False,
    }
    if not baseline:
        payload["baseline"] = None
    payload["output_directory"] = "scratch-output"
    config_path = directory / "run.json"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return config_path, poscar, train, payload


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _assert_payload_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_payload_equal(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for first, second in zip(left, right):
            _assert_payload_equal(first, second)
    else:
        assert left == right


def test_raw_input_file_digests_cover_inputs_are_immutable_and_verify(tmp_path):
    reference = _atoms(1)
    config_path, poscar, train_path, _ = _case(
        tmp_path,
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
    )
    validation_path = tmp_path / "validation.xyz"
    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )

    snapshot = prepared.input_file_digests
    assert snapshot["convention_version"] == (
        SCRATCH_INPUT_FILE_DIGEST_CONVENTION_VERSION
    )
    assert snapshot["path_kind"] == (
        "runtime_location_not_semantic_fingerprint"
    )
    files = snapshot["files"]
    assert tuple(files) == (
        "config",
        "reference_poscar[000000]",
        "train[000000]",
        "validation[000000]",
    )
    expected_paths = {
        "config": config_path.resolve(),
        "reference_poscar[000000]": poscar.resolve(),
        "train[000000]": train_path.resolve(),
        "validation[000000]": validation_path.resolve(),
    }
    for label, path in expected_paths.items():
        entry = files[label]
        assert entry["label"] == label
        assert entry["runtime_path"] == str(path)
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert files["reference_poscar[000000]"]["template_id"] == (
        "scratch-111-a"
    )
    assert files["train[000000]"]["split"] == "train"
    assert files["validation[000000]"]["split"] == "validation"
    assert verify_scratch_preparation_input_digests(prepared) is None
    assert prepared.to_dict()["runtime"]["input_file_digests"] == {
        "convention_version": SCRATCH_INPUT_FILE_DIGEST_CONVENTION_VERSION,
        "path_kind": "runtime_location_not_semantic_fingerprint",
        "files": {
            label: dict(entry) for label, entry in files.items()
        },
    }
    with pytest.raises(TypeError):
        files["config"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        files["config"]["sha256"] = "0" * 64  # type: ignore[index]


def test_raw_input_digest_detects_byte_mutation_with_structured_context(tmp_path):
    reference = _atoms(1)
    config_path, _, train_path, _ = _case(
        tmp_path,
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
    )
    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    semantic_fingerprint = prepared.preparation_fingerprint
    train_path.write_bytes(train_path.read_bytes() + b"\n")

    with pytest.raises(TrainingRunConfigError) as caught:
        verify_scratch_preparation_input_digests(prepared)
    assert caught.value.reason_code == "INPUT_DIGEST_MISMATCH"
    assert caught.value.stage == "scratch.input_digest"
    assert caught.value.field == "data.train[0].path"
    assert caught.value.split == "train"
    assert caught.value.source_path == str(train_path.resolve())
    assert len(caught.value.expected) == len(caught.value.actual) == 64
    assert prepared.preparation_fingerprint == semantic_fingerprint


@pytest.mark.parametrize(
    ("replacement", "reason"),
    (
        ("symlink", "INPUT_DIGEST_SYMLINK_REJECTED"),
        ("directory", "INPUT_DIGEST_NOT_REGULAR_FILE"),
    ),
)
def test_raw_input_digest_rejects_replacement_symlink_and_nonfile(
    tmp_path, replacement, reason
):
    reference = _atoms(1)
    config_path, _, train_path, _ = _case(
        tmp_path,
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
    )
    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    original = train_path.read_bytes()
    train_path.unlink()
    if replacement == "symlink":
        backing = tmp_path / "train-backing.xyz"
        backing.write_bytes(original)
        train_path.symlink_to(backing.name)
    else:
        train_path.mkdir()

    with pytest.raises(TrainingRunConfigError) as caught:
        verify_scratch_preparation_input_digests(prepared)
    assert caught.value.reason_code == reason
    assert caught.value.stage == "scratch.input_digest"
    assert caught.value.field == "data.train[0].path"
    assert caught.value.split == "train"


def test_exact_pristine_vacancy_preparation_matches_direct_builder_and_is_stable(
    tmp_path,
):
    reference = _atoms(1)
    config_path, poscar, train_path, payload = _case(
        tmp_path,
        train_frames=(
            _labeled(reference, -8.0),
            _labeled(_vacancy(reference), -6.5),
        ),
        validation_frames=(_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
        baseline=True,
    )
    config = load_training_run_config(config_path)
    config_before = config.to_dict()
    input_bytes = {
        path: path.read_bytes() for path in (config_path, poscar, train_path)
    }
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()

    first = prepare_scratch_training_run(config)
    second = prepare_scratch_training_run(config)
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    from_yaml = prepare_scratch_training_run(
        load_training_run_config(yaml_path)
    )

    assert isinstance(first, ScratchTrainingPreparation)
    assert first.to_dict() == second.to_dict()
    assert config.config_fingerprint == load_training_run_config(
        yaml_path
    ).config_fingerprint
    assert first.config_fingerprint == from_yaml.config_fingerprint
    assert first.preparation_fingerprint == from_yaml.preparation_fingerprint
    assert first.train_semantic_digest == from_yaml.train_semantic_digest
    assert (
        first.validation_semantic_digest
        == from_yaml.validation_semantic_digest
    )
    assert {
        key: value.structural_fingerprint
        for key, value in first.structural_artifacts.items()
    } == {
        key: value.structural_fingerprint
        for key, value in from_yaml.structural_artifacts.items()
    }
    assert first.train_semantic_digest == second.train_semantic_digest
    assert first.validation_semantic_digest == second.validation_semantic_digest
    assert first.data_manifest == second.data_manifest
    assert tuple(sample.template_id for sample in first.train_samples) == (
        "scratch-111-a",
        "scratch-111-a",
    )
    assert tuple(sample.num_atoms for sample in first.train_samples) == (8, 7)
    artifact = first.structural_artifacts["scratch-111-a"]
    assert tuple(artifact.diagnostics.to_dict())
    assert artifact.diagnostics.num_sites == 8
    assert tuple(
        artifact.diagnostics.num_sites - sample.num_atoms
        for sample in first.train_samples
    ) == (0, 1)

    direct = build_reference_template_from_poscar(
        poscar,
        config=replace(_config_111(), template_id="scratch-111-a"),
        phase_specification=_phase(1),
    )
    direct_artifact = capture_reference_structure_artifact(direct)
    _assert_payload_equal(artifact.to_payload(), direct_artifact.to_payload())
    template = first.registry.resolve("scratch-111-a")
    assert template.fingerprint == direct.template.fingerprint
    assert first.template_contexts["scratch-111-a"].fingerprint == (
        direct.template.fingerprint
    )
    assert first.evaluation_policies["scratch-111-a"] is None
    assert first.to_dict()["status"] == "scratch_preflight_ready"
    assert json.loads(render_train_config_json(first)) == first.to_dict()
    human = render_train_config_human(first)
    assert "Full POSCAR/data/domain preflight completed." in human
    assert "scratch-111-a: M=8" in human
    assert "No model parameters, optimizer, initial bundle" in human

    assert config.to_dict() == config_before
    assert not (tmp_path / "scratch-output").exists()
    assert all(path.read_bytes() == content for path, content in input_bytes.items())
    assert random.getstate() == python_rng
    assert _numpy_state_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.random.get_rng_state(), torch_rng)

    # The prepared result owns a canonical snapshot rather than retaining a
    # caller-owned phase tensor through the frozen config container.
    prepared_report = first.to_dict()
    assert config.model_source is not None
    caller_modes = config.model_source.reference_templates[
        0
    ].phase_specification.modes
    caller_modes[0, 0] += 1
    assert first.to_dict() == prepared_report


def test_mixed_different_m_templates_preserve_source_and_frame_order(tmp_path):
    reference_111 = _atoms(1)
    config_path, _, _, payload = _case(
        tmp_path,
        train_frames=(_labeled(reference_111, -8.0),),
        validation_frames=(_labeled(reference_111, -7.75),),
        selector={"template_id": "scratch-111-a"},
    )
    reference_211 = bulk(
        "NbC", "rocksalt", a=LATTICE, cubic=True
    ).repeat((2, 1, 1))
    poscar_211 = tmp_path / "reference-211.POSCAR"
    train_211 = tmp_path / "train-211.xyz"
    validation_211 = tmp_path / "validation-211.xyz"
    write(poscar_211, reference_211, format="vasp", direct=True)
    _write_extxyz(train_211, (_labeled(reference_211, -16.0),))
    _write_extxyz(validation_211, (_labeled(reference_211, -15.5),))
    payload["model_source"]["reference_templates"].append(
        _template_payload(
            template_id="scratch-211",
            poscar_path=poscar_211.name,
            builder=_config_211(),
            phase=_phase_211(),
        )
    )
    payload["data"]["train"].append(
        {"path": train_211.name, "template_id": "scratch-211"}
    )
    payload["data"]["validation"].append(
        {"path": validation_211.name, "template_id": "scratch-211"}
    )
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    assert tuple(sample.template_id for sample in prepared.train_samples) == (
        "scratch-111-a",
        "scratch-211",
    )
    assert tuple(sample.num_atoms for sample in prepared.train_samples) == (8, 16)
    assert {
        template_id: artifact.diagnostics.num_sites
        for template_id, artifact in prepared.structural_artifacts.items()
    } == {"scratch-111-a": 8, "scratch-211": 16}
    assert prepared.to_dict()["data"]["train"]["template_frame_counts"] == {
        "scratch-111-a": 1,
        "scratch-211": 1,
    }


def test_template_key_is_an_exact_assignment_not_an_inference(tmp_path):
    reference = _labeled(_atoms(1), -8.0)
    reference.info["template"] = "scratch-111-a"
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(reference,),
        validation_frames=(reference,),
        selector={"template_key": "template"},
    )

    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    assert prepared.train_samples[0].template_id == "scratch-111-a"
    assignment = prepared.to_dict()["data_manifest"]["train"]["samples"][0][
        "template_assignment"
    ]
    assert assignment["kind"] == "exact_template_key"
    assert assignment["selection_rule"] == "frame_exact_template_key"
    assert assignment["compatible_template_ids"] == ["scratch-111-a"]
    assert assignment["rejected_templates"] == []


def test_unique_automatic_template_assignment_records_only_compatible_domain(
    tmp_path,
):
    reference = _atoms(1)
    poscar_name = "reference.POSCAR"
    templates = (
        _template_payload(template_id="vacancy-ok", poscar_path=poscar_name),
        _template_payload(
            template_id="pristine-only",
            poscar_path=poscar_name,
            pristine_only=True,
        ),
    )
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(_vacancy(reference), -6.5),),
        validation_frames=(_labeled(_vacancy(reference), -6.25),),
        selector={"automatic_template_assignment": True},
        templates=templates,
    )

    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    assert tuple(sample.template_id for sample in prepared.train_samples) == (
        "vacancy-ok",
    )
    manifest = prepared.to_dict()["data_manifest"]
    assignment = manifest["train"]["samples"][0]["template_assignment"]
    assert assignment["kind"] == "unique_automatic"
    assert assignment["compatible_template_ids"] == ["vacancy-ok"]


@pytest.mark.parametrize(
    ("frames", "reason"),
    (
        ("pristine", "AMBIGUOUS_TEMPLATE_ASSIGNMENT"),
        ("two-vacancies", "NO_COMPATIBLE_TEMPLATE"),
    ),
)
def test_automatic_assignment_rejects_ambiguous_and_zero_candidates(
    tmp_path, frames, reason
):
    reference = _atoms(1)
    poscar_name = "reference.POSCAR"
    templates = (
        _template_payload(template_id="same-domain-a", poscar_path=poscar_name),
        _template_payload(template_id="same-domain-b", poscar_path=poscar_name),
    )
    atoms = reference if frames == "pristine" else _two_vacancies(reference)
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(atoms, -5.0),),
        validation_frames=(_labeled(atoms, -5.0),),
        selector={"automatic_template_assignment": True},
        templates=templates,
    )

    with pytest.raises(TrainingRunConfigError) as caught:
        prepare_scratch_training_run(load_training_run_config(config_path))
    assert caught.value.reason_code == reason
    assert caught.value.stage == "data.template_assignment"
    assert caught.value.split == "train"
    assert caught.value.frame_index == 0
    assert caught.value.sample_id == "train.0000:000000"


def test_policy_template_fingerprint_mismatch_is_structured(tmp_path):
    reference = _atoms(1)
    policy = EvaluationPolicy(
        template_id="scratch-111-a",
        template_fingerprint="0" * 64,
        candidate_offsets=torch.tensor(
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            dtype=torch.float64,
        ),
        phase_step_schedule=(0.1,),
        phase_damping_schedule=(1.0,),
        minimum_objective_gap_absolute=1.0e-8,
        minimum_cross_amplitude_absolute=1.0e-8,
        minimum_atomic_amplitude_absolute=1.0e-8,
        minimum_reference_amplitude_absolute=1.0e-8,
        minimum_curvature=1.0e-8,
        maximum_condition=10.0,
        maximum_gradient_norm=1.0e-8,
        equivalence_tolerance=1.0e-10,
    )
    templates = (
        _template_payload(
            template_id="scratch-111-a",
            poscar_path="reference.POSCAR",
            evaluation_policy=policy.to_dict(),
        ),
    )
    config_path, poscar, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.5),),
        selector={"template_id": "scratch-111-a"},
        templates=templates,
    )

    with pytest.raises(TrainingRunConfigError) as caught:
        prepare_scratch_training_run(load_training_run_config(config_path))
    assert caught.value.reason_code == "POLICY_TEMPLATE_FINGERPRINT_MISMATCH"
    assert caught.value.template_id == "scratch-111-a"
    assert caught.value.source_path == str(poscar.resolve())


def test_phase_stabilizer_mismatch_is_structured_before_context_creation(
    tmp_path,
):
    reference = _atoms(1)
    incompatible_phase = PhaseSpecification(
        modes=torch.diag(torch.tensor([2, 1, 1], dtype=torch.long)),
        mode_weights=torch.ones(3, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )
    templates = (
        _template_payload(
            template_id="scratch-111-a",
            poscar_path="reference.POSCAR",
            phase=incompatible_phase,
        ),
    )
    config_path, poscar, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.5),),
        selector={"template_id": "scratch-111-a"},
        templates=templates,
    )

    with pytest.raises(TrainingRunConfigError) as caught:
        prepare_scratch_training_run(load_training_run_config(config_path))
    assert caught.value.reason_code == "REFERENCE_BUILD_FAILED"
    assert caught.value.stage == "scratch.reference.build"
    assert caught.value.template_id == "scratch-111-a"
    assert caught.value.source_path == str(poscar.resolve())
    assert "typed stabilizer" in str(caught.value.original_error)


def test_exact_assignment_rejects_n_greater_than_m_with_frame_context(tmp_path):
    reference = _atoms(1)
    too_many = reference.copy()
    too_many.append(Atom("C", position=(0.25, 0.25, 0.25)))
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(too_many, -9.0),),
        validation_frames=(_labeled(reference, -8.0),),
        selector={"template_id": "scratch-111-a"},
    )

    with pytest.raises(TrainingRunConfigError) as caught:
        prepare_scratch_training_run(load_training_run_config(config_path))
    assert caught.value.reason_code == "TEMPLATE_DOMAIN_REJECTION"
    assert caught.value.split == "train"
    assert caught.value.frame_index == 0
    assert caught.value.sample_id == "train.0000:000000"
    assert caught.value.template_id == "scratch-111-a"
    assert "N > M" in str(caught.value.original_error)


def test_automatic_assignment_rejects_species_unsupported_by_all_templates(
    tmp_path,
):
    unsupported = _atoms(1)
    unsupported.numbers[0] = 8
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(unsupported, -8.0),),
        validation_frames=(_labeled(_atoms(1), -7.5),),
        selector={"automatic_template_assignment": True},
    )

    with pytest.raises(TrainingRunConfigError) as caught:
        prepare_scratch_training_run(load_training_run_config(config_path))
    assert caught.value.reason_code == "UNSUPPORTED_SPECIES"
    assert caught.value.split == "train"
    assert caught.value.frame_index == 0
    assert caught.value.sample_id == "train.0000:000000"


def test_split_species_are_checked_as_train_coverage_and_observed_union(
    tmp_path,
):
    reference = _atoms(1)
    domain = _config_111().strict_domain
    builder = replace(
        _config_111(),
        strict_domain=StrictTemplateDomain(
            reference_site_count=domain.reference_site_count,
            supercell_shape=domain.supercell_shape,
            species_vocabulary=domain.species_vocabulary,
            reference_composition=domain.reference_composition,
            allowed_compositions=((4, 4), (0, 4)),
            allowed_num_atoms=(8, 4),
            allowed_vacancy_masses=(0, 4),
            convention_version=domain.convention_version,
        ),
    )
    templates = (
        _template_payload(
            template_id="scratch-111-a",
            poscar_path="reference.POSCAR",
            builder=builder,
        ),
    )
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(_only_niobium(reference), -4.0),),
        selector={"template_id": "scratch-111-a"},
        templates=templates,
    )
    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    assert prepared.species_vocabulary == (6, 41)
    assert prepared.observed_species_vocabulary == (6, 41)
    observed = prepared.to_dict()["data_manifest"]["observed_species"]
    assert observed == {
        "train": [6, 41],
        "validation": [41],
        "union": [6, 41],
        "configured": [6, 41],
    }

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    train_path = tmp_path / "train-only-nb.xyz"
    validation_path = tmp_path / "validation-pristine.xyz"
    _write_extxyz(train_path, (_labeled(_only_niobium(reference), -4.0),))
    _write_extxyz(validation_path, (_labeled(reference, -8.0),))
    payload["data"]["train"][0]["path"] = train_path.name
    payload["data"]["validation"][0]["path"] = validation_path.name
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(TrainingRunConfigError) as caught:
        prepare_scratch_training_run(load_training_run_config(config_path))
    assert caught.value.reason_code == "VALIDATION_SPECIES_NOT_IN_TRAIN"
    assert caught.value.stage == "data.species"


def test_exact_assignment_rejects_cell_outside_strain_certificate(tmp_path):
    strained = _atoms(1)
    strained.set_cell(strained.cell.array * 1.2, scale_atoms=True)
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(strained, -8.0),),
        validation_frames=(_labeled(_atoms(1), -7.5),),
        selector={"template_id": "scratch-111-a"},
    )

    with pytest.raises(TrainingRunConfigError) as caught:
        prepare_scratch_training_run(load_training_run_config(config_path))
    assert caught.value.reason_code == "TEMPLATE_DOMAIN_REJECTION"
    assert caught.value.split == "train"
    assert caught.value.template_id == "scratch-111-a"
    assert "outside certified graph strain domain" in str(
        caught.value.original_error
    )


def test_missing_and_partial_labels_preserve_masks_and_active_term_counts(
    tmp_path,
):
    reference = _atoms(1)
    config_path, _, _, payload = _case(
        tmp_path,
        train_frames=(
            _partially_labeled(reference, -8.0),
            _labeled(_vacancy(reference), 0.0),
        ),
        validation_frames=(_labeled(reference, 0.0),),
        selector={"template_id": "scratch-111-a"},
        baseline=True,
    )
    payload["loss"].update(
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=1.0,
    )
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    first, second = prepared.train_samples
    assert first.forces is not None and first.force_mask is not None
    assert int(torch.count_nonzero(first.force_mask)) == 8 * 3 - 1
    assert first.stress is not None and first.stress_mask is not None
    assert second.energy is not None and float(second.energy) == 0.0
    assert second.forces is None and second.stress is None
    labels = prepared.to_dict()["data"]["train"]["label_statistics"]
    assert labels["energy"] == {
        "present_frames": 2,
        "missing_frames": 0,
        "valid_count": 2,
    }
    assert labels["forces"] == {
        "present_frames": 1,
        "missing_frames": 1,
        "valid_count": 23,
    }
    assert labels["stress"] == {
        "present_frames": 1,
        "missing_frames": 1,
        "valid_count": 4,
    }


def test_duplicate_scratch_template_ids_fail_at_config_validation(tmp_path):
    reference = _atoms(1)
    config_path, _, _, payload = _case(
        tmp_path,
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.5),),
        selector={"template_id": "scratch-111-a"},
    )
    payload["model_source"]["reference_templates"].append(
        copy.deepcopy(payload["model_source"]["reference_templates"][0])
    )
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(TrainingRunConfigError) as caught:
        load_training_run_config(config_path)
    assert caught.value.reason_code == "DUPLICATE_TEMPLATE_ID"
    assert caught.value.field == "model_source.reference_templates"


def test_json_yaml_preparation_semantics_are_identical(tmp_path):
    reference = _atoms(1)
    json_path, _, _, payload = _case(
        tmp_path,
        train_frames=(
            _labeled(reference, -8.0),
            _labeled(_vacancy(reference), -6.5),
        ),
        validation_frames=(_labeled(reference, -7.5),),
        selector={"template_id": "scratch-111-a"},
        baseline=True,
    )
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    from_json = prepare_scratch_training_run(
        load_training_run_config(json_path)
    )
    from_yaml = prepare_scratch_training_run(
        load_training_run_config(yaml_path)
    )
    assert from_json.config_fingerprint == from_yaml.config_fingerprint
    assert from_json.preparation_fingerprint == from_yaml.preparation_fingerprint
    assert from_json.registry.fingerprint == from_yaml.registry.fingerprint
    assert from_json.train_semantic_digest == from_yaml.train_semantic_digest
    assert (
        from_json.validation_semantic_digest
        == from_yaml.validation_semantic_digest
    )
    assert from_json.data_manifest == from_yaml.data_manifest
    assert from_json.to_dict()["runtime"]["paths"]["config"] == str(
        json_path.resolve()
    )
    assert from_yaml.to_dict()["runtime"]["paths"]["config"] == str(
        yaml_path.resolve()
    )


def test_cli_validate_and_train_dry_run_share_preparation_and_never_execute(
    tmp_path, monkeypatch, capsys
):
    reference = _atoms(1)
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(
            _labeled(reference, -8.0),
            _labeled(_vacancy(reference), -6.5),
        ),
        validation_frames=(_labeled(reference, -7.5),),
        selector={"template_id": "scratch-111-a"},
        baseline=True,
    )
    import refsite_mlip.cli.train as train_module
    import refsite_mlip.models.potential as potential_module
    import refsite_mlip.training.optimizer as optimizer_module
    import refsite_mlip.transport.factory as transport_factory

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model/OT/optimizer/training execution is forbidden")

    monkeypatch.setattr(potential_module.ReferenceSitePotential, "forward", forbidden)
    monkeypatch.setattr(
        potential_module.ReferenceSitePotential, "__init__", forbidden
    )
    monkeypatch.setattr(optimizer_module, "build_optimizer", forbidden)
    monkeypatch.setattr(transport_factory, "solve_atom_vacancy_ot", forbidden)
    monkeypatch.setattr(train_module, "seed_training_runtime", forbidden)
    monkeypatch.setattr(train_module, "_prepare_training_runtime", forbidden)
    monkeypatch.setattr(train_module, "build_optimizer", forbidden)
    monkeypatch.setattr(train_module, "build_scheduler", forbidden)
    monkeypatch.setattr(train_module, "run_checkpointed_fit", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    rng = torch.random.get_rng_state().clone()

    assert main(["validate-train-config", str(config_path), "--json"]) == 0
    validated = capsys.readouterr()
    assert validated.err == ""
    validated_report = json.loads(validated.out)
    assert validated_report["status"] == "scratch_preflight_ready"

    assert main(["train", str(config_path), "--dry-run", "--json"]) == 0
    dry_run = capsys.readouterr()
    assert dry_run.err == ""
    assert json.loads(dry_run.out) == validated_report

    assert main(["train", str(config_path)]) == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    assert "SCRATCH_EXECUTION_NOT_IMPLEMENTED" in failed.err
    assert "Traceback" not in failed.err
    assert not (tmp_path / "scratch-output").exists()
    assert torch.equal(torch.random.get_rng_state(), rng)
