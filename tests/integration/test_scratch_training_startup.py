from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ase")

import refsite_mlip.training.scratch_startup as startup_module
from refsite_mlip.config import load_training_run_config
from refsite_mlip.config import TrainingRunConfigError
from refsite_mlip.models import load_reference_site_model_bundle
from refsite_mlip.training import (
    FitProgress,
    ModelSelectionState,
    TrainingRunDirectory,
    ScratchTrainingStartup,
    ScratchTrainingStartupError,
    SCRATCH_TRAINING_STARTUP_STATUS_SCHEMA_VERSION,
    initialize_scratch_training_startup,
    optimizer_parameters,
    prepare_scratch_training_run,
)

from test_scratch_model_initialization import _mixed_preparation, _state_equal
from test_scratch_training_preparation import (
    _atoms,
    _case,
    _labeled,
    _partially_labeled,
    _vacancy,
)


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _preparation(
    directory: Path,
    *,
    dtype: str = "float64",
    baseline: bool = False,
    validation_energy: float = -7.75,
):
    reference = _atoms(1)
    config_path, poscar, train_path, payload = _case(
        directory,
        train_frames=(
            _partially_labeled(reference, -8.0),
            _partially_labeled(_vacancy(reference), -6.5),
        ),
        validation_frames=(
            _partially_labeled(reference, validation_energy),
        ),
        selector={"template_id": "scratch-111-a"},
        baseline=baseline,
    )
    payload["runtime"]["dtype"] = dtype
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    preparation = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    return preparation, (config_path, poscar, train_path, directory / "validation.xyz")


@pytest.mark.parametrize(
    ("dtype_name", "dtype"),
    (("float32", torch.float32), ("float64", torch.float64)),
)
def test_startup_uses_saved_bundle_and_stops_before_first_update(
    tmp_path, dtype_name, dtype, monkeypatch
):
    preparation, inputs = _preparation(
        tmp_path, dtype=dtype_name, baseline=False
    )
    preparation_before = preparation.to_dict()
    input_before = {path: path.read_bytes() for path in inputs}
    original_load = startup_module.load_reference_site_model_bundle
    load_calls = []

    def checked_load(path, **kwargs):
        load_calls.append((Path(path), dict(kwargs)))
        return original_load(path, **kwargs)

    monkeypatch.setattr(
        startup_module, "load_reference_site_model_bundle", checked_load
    )
    context = torch.inference_mode() if dtype_name == "float32" else nullcontext()
    with context:
        result = initialize_scratch_training_startup(preparation)
        if dtype_name == "float32":
            assert torch.is_inference_mode_enabled()

    assert isinstance(result, ScratchTrainingStartup)
    assert load_calls == [
        (result.run_directory.initial_bundle_path, {"map_location": "cpu"})
    ]
    assert result.run_directory_paths == {
        "checkpoints": str(result.run_directory.checkpoints),
        "data_manifest": str(result.run_directory.data_manifest_path),
        "initial_bundle": str(result.run_directory.initial_bundle_path),
        "preflight": str(result.run_directory.preflight_path),
        "resolved_config": str(result.run_directory.resolved_config_path),
        "root": str(result.run_directory.root),
        "run_status": str(result.run_directory.status_path),
    }
    assert set(path.name for path in result.run_directory.root.iterdir()) == {
        "checkpoints",
        "data_manifest.json",
        "initial_bundle.pt",
        "preflight.json",
        "resolved_config.json",
        "run_status.json",
    }
    assert not list(result.run_directory.checkpoints.iterdir())
    disk_bundle = load_reference_site_model_bundle(
        result.run_directory.initial_bundle_path
    )
    assert disk_bundle.bundle_fingerprint == result.initial_bundle_fingerprint
    assert _state_equal(disk_bundle.model_state, result.initial_bundle.model_state)
    assert _state_equal(disk_bundle.model_state, result.model.state_dict())
    assert torch.equal(
        disk_bundle.model_state["atomic_baseline"],
        torch.zeros_like(disk_bundle.model_state["atomic_baseline"]),
    )
    assert result.model.atomic_baseline.dtype == dtype
    assert result.model.atomic_baseline.device.type == "cpu"
    site_count = result.registry.resolve("scratch-111-a").topology.num_sites
    assert tuple(
        site_count - sample.num_atoms for sample in preparation.train_samples
    ) == (0, 1)
    train_batch = result.train_batches[0]
    assert train_batch.sample_ids == tuple(
        sample.sample_id for sample in preparation.train_samples
    )
    assert torch.equal(
        train_batch.force_mask.cpu(),
        torch.cat(
            [sample.force_mask for sample in preparation.train_samples], dim=0
        ),
    )
    assert torch.equal(
        train_batch.stress_mask.cpu(),
        torch.stack(
            [sample.stress_mask for sample in preparation.train_samples]
        ),
    )
    assert result.baseline_fit is None
    assert result.baseline_metadata["enabled"] is False
    assert result.initial_selection_state == ModelSelectionState()
    assert result.initial_fit_progress == FitProgress(0, 0, 0)
    assert result.optimizer.state == {}
    assert tuple(id(value) for value in optimizer_parameters(result.optimizer)) == tuple(
        id(parameter)
        for parameter in result.model.parameters()
        if parameter.requires_grad
    )
    assert getattr(result.scheduler, "optimizer") is result.optimizer
    assert all(parameter.grad is None for parameter in result.model.parameters())
    assert result.model.training is False
    status = json.loads(result.run_directory.status_path.read_text())
    assert status["schema_version"] == (
        SCRATCH_TRAINING_STARTUP_STATUS_SCHEMA_VERSION
    )
    assert status["status"] == "startup_ready"
    assert status["training_executed"] is False
    assert status["first_optimizer_update_executed"] is False
    assert status["completed_epochs"] == status["global_step"] == 0
    assert status["latest_checkpoint"] is None
    assert status["best_checkpoint"] is None
    assert preparation.to_dict() == preparation_before
    assert all(path.read_bytes() == value for path, value in input_before.items())
    assert all(not value.is_inference() for value in result.model.state_dict().values())


def test_startup_reuses_supplied_owned_directory_and_leaves_lock_owned(tmp_path):
    preparation, _ = _preparation(tmp_path)
    output = Path(preparation.runtime_paths["output_directory"])
    directory = TrainingRunDirectory.create(output)
    lock = directory.acquire_resume_lock()
    identity = directory.resume_lock_path.lstat()
    try:
        result = initialize_scratch_training_startup(
            preparation,
            run_directory=directory,
            run_lock=lock,
        )
        assert result.run_directory is directory
        lock.validate_owned(directory.resume_lock_path)
        current = directory.resume_lock_path.lstat()
        assert (current.st_dev, current.st_ino) == (
            identity.st_dev,
            identity.st_ino,
        )
        assert directory.initial_bundle_path.is_file()
        assert directory.status_path.is_file()
        assert not list(directory.checkpoints.iterdir())
    finally:
        if lock.owned:
            lock.release()


def test_startup_supplied_directory_and_lock_are_all_or_nothing(tmp_path):
    preparation, _ = _preparation(tmp_path)
    output = Path(preparation.runtime_paths["output_directory"])
    directory = TrainingRunDirectory.create(output)

    with pytest.raises(ScratchTrainingStartupError) as missing_lock:
        initialize_scratch_training_startup(
            preparation, run_directory=directory
        )
    assert missing_lock.value.stage == "preflight"
    assert list(directory.root.iterdir()) == []

    lock = directory.acquire_resume_lock()
    try:
        with pytest.raises(ScratchTrainingStartupError) as missing_directory:
            initialize_scratch_training_startup(preparation, run_lock=lock)
        assert missing_directory.value.stage == "preflight"
        lock.validate_owned(directory.resume_lock_path)
        assert set(path.name for path in directory.root.iterdir()) == {
            ".resume.lock"
        }
    finally:
        lock.release()


def test_startup_rejects_wrong_or_released_caller_lock_without_mutation(tmp_path):
    preparation, _ = _preparation(tmp_path / "inputs")
    configured = Path(preparation.runtime_paths["output_directory"])
    configured.mkdir()
    other = TrainingRunDirectory.create(tmp_path / "other-run")
    other_lock = other.acquire_resume_lock()
    try:
        with pytest.raises(ScratchTrainingStartupError) as mismatch:
            initialize_scratch_training_startup(
                preparation,
                run_directory=other,
                run_lock=other_lock,
            )
        assert mismatch.value.stage == "preflight"
        other_lock.validate_owned(other.resume_lock_path)
        assert set(path.name for path in other.root.iterdir()) == {".resume.lock"}
    finally:
        other_lock.release()

    configured_directory = TrainingRunDirectory.open_existing(configured)
    released = configured_directory.acquire_resume_lock()
    released.release()
    with pytest.raises(ScratchTrainingStartupError) as not_owned:
        initialize_scratch_training_startup(
            preparation,
            run_directory=configured_directory,
            run_lock=released,
        )
    assert not_owned.value.reason_code == "RESUME_LOCK_NOT_OWNED"
    assert list(configured.iterdir()) == []


def test_supplied_lock_survives_startup_failure_and_post_init_input_toctou(
    tmp_path, monkeypatch
):
    preparation, inputs = _preparation(tmp_path)
    output = Path(preparation.runtime_paths["output_directory"])
    directory = TrainingRunDirectory.create(output)
    lock = directory.acquire_resume_lock()
    original = startup_module.initialize_scratch_model

    def initialize_then_mutate(value):
        initialized = original(value)
        with inputs[2].open("ab") as stream:
            stream.write(b"\n")
        return initialized

    monkeypatch.setattr(
        startup_module, "initialize_scratch_model", initialize_then_mutate
    )
    try:
        with pytest.raises(ScratchTrainingStartupError) as caught:
            initialize_scratch_training_startup(
                preparation,
                run_directory=directory,
                run_lock=lock,
            )
        assert caught.value.stage == "run_directory_validate"
        assert "INPUT_DIGEST" in caught.value.reason_code
        lock.validate_owned(directory.resume_lock_path)
        assert json.loads(directory.status_path.read_text())["status"] == "failed"
        assert not directory.checkpoints.exists()
    finally:
        lock.release()


def test_baseline_is_train_only_and_not_persisted_in_initial_bundle(tmp_path):
    first, _ = _preparation(
        tmp_path / "first", baseline=True, validation_energy=-7.75
    )
    second, _ = _preparation(
        tmp_path / "second", baseline=True, validation_energy=12345.0
    )
    first_result = initialize_scratch_training_startup(first)
    second_result = initialize_scratch_training_startup(second)

    assert first_result.baseline_fit is not None
    assert second_result.baseline_fit is not None
    assert torch.equal(
        first_result.baseline_fit.baseline_energies,
        second_result.baseline_fit.baseline_energies,
    )
    assert first_result.baseline_fit.training_sample_ids == tuple(
        sample.sample_id for sample in first.train_samples
    )
    assert first_result.initial_bundle_fingerprint == (
        second_result.initial_bundle_fingerprint
    )
    for result in (first_result, second_result):
        persisted = result.initial_bundle.model_state["atomic_baseline"]
        assert torch.equal(persisted, torch.zeros_like(persisted))
        assert torch.equal(
            result.model.atomic_baseline,
            result.baseline_fit.baseline_energies.to(
                result.model.atomic_baseline
            ),
        )
        assert result.model.atomic_baseline.data_ptr() == dict(
            result.model.named_buffers()
        )["atomic_baseline"].data_ptr()
        assert "atomic_baseline" not in dict(result.model.named_parameters())
        assert id(result.model.atomic_baseline) not in {
            id(parameter) for parameter in optimizer_parameters(result.optimizer)
        }


def test_rank_deficient_baseline_requires_explicit_minimum_norm(tmp_path):
    reference = _atoms(1)
    config_path, _, _, payload = _case(
        tmp_path / "error",
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
        baseline=True,
    )
    with pytest.raises(TrainingRunConfigError) as caught:
        prepare_scratch_training_run(load_training_run_config(config_path))
    assert caught.value.reason_code == "BASELINE_PREFLIGHT_FAILED"

    config_path, _, _, payload = _case(
        tmp_path / "minimum-norm",
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
        baseline=True,
    )
    payload["baseline"]["rank_policy"] = "minimum_norm"
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    preparation = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    result = initialize_scratch_training_startup(preparation)
    assert result.baseline_fit is not None
    assert result.baseline_fit.rank_deficient is True
    assert result.baseline_fit.config.rank_policy == "minimum_norm"
    assert result.baseline_metadata["condition_number"] is None


def test_mixed_templates_batch_boundaries_and_runtime_bindings(tmp_path):
    preparation = _mixed_preparation(tmp_path)
    result = initialize_scratch_training_startup(preparation)

    assert result.initial_bundle.binding_ids == ("alpha-111", "zeta-211")
    assert result.initial_bundle.default_template_id == "zeta-211"
    assert tuple(result.template_contexts) == ("alpha-111", "zeta-211")
    assert tuple(batch.sample_ids for batch in result.train_batches) == tuple(
        tuple(plan["sample_ids"])
        for plan in preparation.data_manifest["train"]["batches"]
    )
    assert tuple(batch.template_ids for batch in result.train_batches) == tuple(
        tuple(plan["template_ids"])
        for plan in preparation.data_manifest["train"]["batches"]
    )
    assert result.data_manifest["fingerprint"] == (
        preparation.data_manifest["fingerprint"]
    )
    assert {
        binding.template_id: binding.structural_artifact.diagnostics.num_sites
        for binding in result.initial_bundle.template_bindings
    } == {"alpha-111": 8, "zeta-211": 16}


def test_success_applies_training_seed_only_after_static_startup(tmp_path):
    preparation, _ = _preparation(tmp_path)
    random.seed(991)
    np.random.seed(992)
    torch.manual_seed(993)

    result = initialize_scratch_training_startup(preparation)
    seed = result.training_seed
    expected_python = random.Random(seed).random()
    expected_numpy = np.random.RandomState(seed % (2**32)).random_sample()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**64))
    expected_torch = torch.rand(4, generator=generator)
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)


def test_input_toctou_and_output_collision_are_precreation_transactional(tmp_path):
    stale, inputs = _preparation(tmp_path / "stale")
    input_before = {path: path.read_bytes() for path in inputs}
    with inputs[2].open("ab") as stream:
        stream.write(b"\n")
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    with pytest.raises(ScratchTrainingStartupError) as caught:
        initialize_scratch_training_startup(stale)
    assert caught.value.stage == "preflight"
    assert "INPUT_DIGEST" in caught.value.reason_code
    assert not Path(stale.runtime_paths["output_directory"]).exists()
    assert random.getstate() == python_before
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert input_before[inputs[0]] == inputs[0].read_bytes()

    collision, _ = _preparation(tmp_path / "collision")
    output = Path(collision.runtime_paths["output_directory"])
    output.mkdir()
    marker = output / "foreign"
    marker.write_bytes(b"preserve")
    with pytest.raises(ScratchTrainingStartupError) as caught:
        initialize_scratch_training_startup(collision)
    assert caught.value.stage == "preflight"
    assert caught.value.reason_code == "OUTPUT_ALREADY_EXISTS"
    assert marker.read_bytes() == b"preserve"


def test_mutated_caller_preparation_is_rejected_instead_of_silently_replaced(
    tmp_path,
):
    preparation, _ = _preparation(tmp_path)
    template_id = preparation.model_source.default_template_id
    preparation.template_contexts[template_id].phase_modes[0, 0] += 1
    rng = torch.get_rng_state().clone()

    with pytest.raises(ScratchTrainingStartupError) as caught:
        initialize_scratch_training_startup(preparation)
    assert caught.value.stage == "preflight"
    assert caught.value.reason_code == "PHASE_MISMATCH"
    assert caught.value.template_id == template_id
    assert not Path(preparation.runtime_paths["output_directory"]).exists()
    assert torch.equal(torch.get_rng_state(), rng)


def test_in_memory_config_keeps_explicit_preparation_base_across_cwd(
    tmp_path, monkeypatch
):
    prepared_from_file, _ = _preparation(tmp_path / "inputs")
    in_memory_config = replace(prepared_from_file.config, source_path=None)
    preparation = prepare_scratch_training_run(
        in_memory_config, base_directory=tmp_path / "inputs"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = initialize_scratch_training_startup(preparation)
    assert result.run_directory.root == (
        tmp_path / "inputs" / "scratch-output"
    ).resolve()
    assert result.config.source_path is None


@pytest.mark.parametrize(
    ("target", "phase", "bundle_exists"),
    (
        ("save_reference_site_model_bundle", "initial_bundle_save", False),
        ("load_reference_site_model_bundle", "initial_bundle_reload", True),
    ),
)
def test_bundle_failure_restores_rng_and_records_failed_status(
    tmp_path, monkeypatch, target, phase, bundle_exists
):
    preparation, _ = _preparation(tmp_path)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    def fail(*args, **kwargs):
        del args, kwargs
        raise OSError(f"injected {phase}")

    monkeypatch.setattr(startup_module, target, fail)
    with pytest.raises(ScratchTrainingStartupError) as caught:
        initialize_scratch_training_startup(preparation)
    error = caught.value
    output = Path(preparation.runtime_paths["output_directory"])
    assert error.stage == phase
    assert error.original_exception_type == "OSError"
    assert output.joinpath("initial_bundle.pt").exists() is bundle_exists
    status = json.loads(output.joinpath("run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["failure_phase"] == phase
    assert status["first_optimizer_update_executed"] is False
    assert not list(output.joinpath("checkpoints").iterdir())
    assert random.getstate() == python_before
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_status_failure_keeps_verified_bundle_and_primary_context(
    tmp_path, monkeypatch
):
    preparation, _ = _preparation(tmp_path)

    def fail_status(self, value):
        del self, value
        raise OSError("injected status write failure")

    monkeypatch.setattr(
        startup_module.TrainingRunDirectory, "write_status", fail_status
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(ScratchTrainingStartupError) as caught:
        initialize_scratch_training_startup(preparation)
    error = caught.value
    output = Path(preparation.runtime_paths["output_directory"])
    assert error.stage == "status_save"
    assert error.status_write_exception_type == "OSError"
    assert output.joinpath("initial_bundle.pt").is_file()
    assert not output.joinpath("run_status.json").exists()
    assert load_reference_site_model_bundle(
        output / "initial_bundle.pt"
    ).bundle_fingerprint == error.bundle_fingerprint
    assert not list(output.joinpath("checkpoints").iterdir())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    ("target", "phase"),
    (
        ("instantiate_reference_site_model_bundle", "runtime_materialization"),
        ("collate_structure_samples", "batching"),
        ("build_optimizer", "optimizer"),
        ("build_scheduler", "scheduler"),
    ),
)
def test_post_bundle_stage_failures_keep_recoverable_zero_baseline_bundle(
    tmp_path, monkeypatch, target, phase
):
    preparation, _ = _preparation(tmp_path, baseline=False)

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"injected {phase} failure")

    monkeypatch.setattr(startup_module, target, fail)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ScratchTrainingStartupError) as caught:
        initialize_scratch_training_startup(preparation)
    error = caught.value
    output = Path(preparation.runtime_paths["output_directory"])
    assert error.stage == phase
    assert error.recoverable_initial_bundle == str(output / "initial_bundle.pt")
    persisted = load_reference_site_model_bundle(output / "initial_bundle.pt")
    assert persisted.bundle_fingerprint == error.bundle_fingerprint
    assert torch.count_nonzero(persisted.model_state["atomic_baseline"]) == 0
    status = json.loads((output / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["failure_phase"] == phase
    assert status["recoverable_initial_bundle"] == str(
        output / "initial_bundle.pt"
    )
    assert not list((output / "checkpoints").iterdir())
    assert torch.equal(torch.get_rng_state(), rng)


def test_metadata_failure_is_atomic_and_precedes_initial_bundle(tmp_path, monkeypatch):
    preparation, _ = _preparation(tmp_path)

    def fail_preflight(self, value):
        del self, value
        raise OSError("injected preflight metadata failure")

    monkeypatch.setattr(
        startup_module.TrainingRunDirectory, "write_preflight", fail_preflight
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(ScratchTrainingStartupError) as caught:
        initialize_scratch_training_startup(preparation)
    output = Path(preparation.runtime_paths["output_directory"])
    assert caught.value.stage == "metadata_save"
    assert (output / "resolved_config.json").is_file()
    assert not (output / "preflight.json").exists()
    assert not (output / "data_manifest.json").exists()
    assert not (output / "initial_bundle.pt").exists()
    assert json.loads((output / "run_status.json").read_text())["status"] == "failed"
    assert torch.equal(torch.get_rng_state(), rng)


def test_startup_does_not_execute_model_or_training_update(tmp_path, monkeypatch):
    preparation, _ = _preparation(tmp_path, baseline=False)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model/training execution is forbidden at startup")

    import refsite_mlip.models.batch_executor as batch_executor_module
    import refsite_mlip.models.potential as potential_module
    import refsite_mlip.training.checkpointed_fit as checkpointed_fit_module
    import refsite_mlip.training.fit as fit_module
    import refsite_mlip.training.step as step_module

    monkeypatch.setattr(potential_module.ReferenceSitePotential, "forward", forbidden)
    monkeypatch.setattr(batch_executor_module, "evaluate_structure_batch", forbidden)
    monkeypatch.setattr(step_module, "train_step", forbidden)
    monkeypatch.setattr(fit_module, "run_fit", forbidden)
    monkeypatch.setattr(checkpointed_fit_module, "run_checkpointed_fit", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)

    result = initialize_scratch_training_startup(preparation)
    assert result.optimizer.state == {}
    assert result.initial_fit_progress == FitProgress(0, 0, 0)
    assert not list(result.run_directory.checkpoints.iterdir())
