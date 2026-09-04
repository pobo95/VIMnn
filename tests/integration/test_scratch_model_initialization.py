from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase.build import bulk
from ase.io import write

import refsite_mlip.models.bundle as bundle_module
import refsite_mlip.training.scratch_initialization as initialization_module
from refsite_mlip.config import load_training_run_config
from refsite_mlip.data import build_reference_template_from_poscar
from refsite_mlip.models import (
    EvaluationPolicy,
    ModelBundleError,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.training import (
    ScratchModelInitialization,
    ScratchModelInitializationError,
    initialize_scratch_model,
    prepare_scratch_training_run,
)

from test_scratch_training_preparation import (
    LATTICE,
    _atoms,
    _case,
    _config_111,
    _config_211,
    _labeled,
    _phase_211,
    _phase,
    _template_payload,
    _vacancy,
    _write_extxyz,
)


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _single_preparation(
    directory: Path,
    *,
    initialization_seed: int = 20260904,
    dtype: str = "float64",
):
    reference = _atoms(1)
    config_path, poscar, train, payload = _case(
        directory,
        train_frames=(
            _labeled(reference, -8.0),
            _labeled(_vacancy(reference), -6.5),
        ),
        validation_frames=(_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
    )
    payload["model_source"]["initialization_seed"] = initialization_seed
    payload["runtime"]["dtype"] = dtype
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    inputs = (
        config_path,
        poscar,
        train,
        directory / "validation.xyz",
    )
    return prepared, inputs


def _mixed_preparation(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    reference_111 = _atoms(1)
    reference_211 = bulk(
        "NbC", "rocksalt", a=LATTICE, cubic=True
    ).repeat((2, 1, 1))
    poscar_211 = directory / "reference-211.POSCAR"
    train_211 = directory / "train-211.xyz"
    validation_211 = directory / "validation-211.xyz"
    write(poscar_211, reference_211, format="vasp", direct=True)
    _write_extxyz(train_211, (_labeled(reference_211, -16.0),))
    _write_extxyz(validation_211, (_labeled(reference_211, -15.5),))
    templates = (
        _template_payload(
            template_id="zeta-211",
            poscar_path=poscar_211.name,
            builder=_config_211(),
            phase=_phase_211(),
        ),
        _template_payload(
            template_id="alpha-111",
            poscar_path="reference.POSCAR",
        ),
    )
    config_path, _, _, payload = _case(
        directory,
        train_frames=(_labeled(reference_111, -8.0),),
        validation_frames=(_labeled(reference_111, -7.75),),
        selector={"template_id": "alpha-111"},
        templates=templates,
    )
    payload["data"]["train"] = [
        {"path": train_211.name, "template_id": "zeta-211"},
        {"path": "train.xyz", "template_id": "alpha-111"},
    ]
    payload["data"]["validation"] = [
        {"path": "validation.xyz", "template_id": "alpha-111"},
        {
            "path": validation_211.name,
            "template_id": "zeta-211",
        },
    ]
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return prepare_scratch_training_run(load_training_run_config(config_path))


def _state_equal(left, right) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


def _forbidden(*args, **kwargs):
    del args, kwargs
    raise AssertionError("forward/OT/optimizer/training/builder execution is forbidden")


def test_same_seed_is_bitwise_and_different_seed_changes_only_parameters(
    tmp_path,
):
    first_prepared, _ = _single_preparation(tmp_path / "first")
    second_prepared, _ = _single_preparation(tmp_path / "second")
    different_prepared, _ = _single_preparation(
        tmp_path / "different", initialization_seed=20260905
    )
    site_count = first_prepared.structural_artifacts[
        "scratch-111-a"
    ].diagnostics.num_sites
    assert tuple(
        site_count - sample.num_atoms for sample in first_prepared.train_samples
    ) == (0, 1)

    first = initialize_scratch_model(first_prepared)
    random.random()
    np.random.random()
    torch.rand(3)
    second = initialize_scratch_model(second_prepared)
    different = initialize_scratch_model(different_prepared)
    assert _state_equal(first.bundle.model_state, second.bundle.model_state)
    assert first.model_state_fingerprint == second.model_state_fingerprint
    assert first.architecture_fingerprint == second.architecture_fingerprint
    assert first.bundle_fingerprint == second.bundle_fingerprint

    runtime = instantiate_reference_site_model_bundle(first.bundle)
    parameter_names = set(dict(runtime.model.named_parameters()))
    buffer_names = set(dict(runtime.model.named_buffers()))
    assert parameter_names and buffer_names
    assert any(
        not torch.equal(
            first.bundle.model_state[name], different.bundle.model_state[name]
        )
        for name in parameter_names
    )
    assert all(
        torch.equal(
            first.bundle.model_state[name], different.bundle.model_state[name]
        )
        for name in buffer_names
    )
    assert first.architecture_fingerprint == different.architecture_fingerprint
    assert first.bundle_fingerprint != different.bundle_fingerprint


def test_optional_evaluation_policy_is_bound_and_reconstructed_exactly(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference = _atoms(1)
    poscar = tmp_path / "reference.POSCAR"
    write(poscar, reference, format="vasp", direct=True)
    direct = build_reference_template_from_poscar(
        poscar,
        config=_config_111(),
        phase_specification=_phase(1),
    )
    policy = EvaluationPolicy(
        template_id="scratch-111-a",
        template_fingerprint=direct.template.fingerprint,
        candidate_offsets=torch.tensor(
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            dtype=torch.float64,
        ),
        phase_step_schedule=(0.7, 0.8, 0.9, 1.0),
        phase_damping_schedule=(2.0, 1.0, 0.5, 0.2),
        minimum_objective_gap_absolute=1.0e-8,
        minimum_cross_amplitude_absolute=1.0e-8,
        minimum_atomic_amplitude_absolute=1.0e-8,
        minimum_reference_amplitude_absolute=1.0e-8,
        minimum_curvature=1.0e-8,
        maximum_condition=1.0e8,
        maximum_gradient_norm=1.0e-8,
        equivalence_tolerance=1.0e-10,
    )
    templates = (
        _template_payload(
            template_id="scratch-111-a",
            poscar_path=poscar.name,
            evaluation_policy=policy.to_dict(),
        ),
    )
    config_path, _, _, _ = _case(
        tmp_path,
        train_frames=(_labeled(reference, -8.0),),
        validation_frames=(_labeled(reference, -7.75),),
        selector={"template_id": "scratch-111-a"},
        templates=templates,
    )
    prepared = prepare_scratch_training_run(
        load_training_run_config(config_path)
    )
    initialized = initialize_scratch_model(prepared)
    binding = initialized.bundle.template_bindings[0]
    runtime = instantiate_reference_site_model_bundle(initialized.bundle)
    assert binding.evaluation_policy is not None
    assert binding.evaluation_policy.to_dict() == policy.to_dict()
    assert (
        binding.evaluation_policy.candidate_offsets.data_ptr()
        != prepared.evaluation_policies[
            "scratch-111-a"
        ].candidate_offsets.data_ptr()
    )
    assert runtime.evaluation_policies["scratch-111-a"].to_dict() == policy.to_dict()
    assert initialized.template_fingerprints["scratch-111-a"][
        "evaluation_policy_fingerprint"
    ] == policy.content_fingerprint


@pytest.mark.parametrize(
    ("dtype_name", "dtype"),
    (("float32", torch.float32), ("float64", torch.float64)),
)
def test_dtype_zero_baseline_counts_and_parameter_contract(
    tmp_path, dtype_name, dtype
):
    prepared, _ = _single_preparation(tmp_path, dtype=dtype_name)
    initialized = initialize_scratch_model(prepared)
    runtime = instantiate_reference_site_model_bundle(
        initialized.bundle, device="cpu", dtype=dtype
    )
    model = runtime.model
    parameters = tuple(model.named_parameters())
    buffers = tuple(model.named_buffers())

    assert isinstance(initialized, ScratchModelInitialization)
    assert initialized.bundle.model_floating_dtype == dtype_name
    assert initialized.to_dict()["training_state_included"] is False
    assert json.loads(json.dumps(initialized.to_dict(), allow_nan=False))
    assert initialized.parameter_tensor_count == len(parameters) > 0
    assert initialized.parameter_element_count == sum(
        value.numel() for _, value in parameters
    )
    assert initialized.parameter_byte_count == sum(
        value.numel() * value.element_size() for _, value in parameters
    )
    assert initialized.buffer_tensor_count == len(buffers) > 0
    assert initialized.buffer_element_count == sum(
        value.numel() for _, value in buffers
    )
    assert initialized.buffer_byte_count == sum(
        value.numel() * value.element_size() for _, value in buffers
    )
    assert model.training is False
    assert all(
        value.requires_grad and value.grad is None and not value.is_inference()
        for _, value in parameters
    )
    assert all(not value.requires_grad for _, value in buffers)
    baseline = dict(buffers)["atomic_baseline"]
    assert baseline.dtype == dtype and baseline.device.type == "cpu"
    assert baseline.shape == (2,)
    assert torch.equal(baseline, torch.zeros_like(baseline))
    assert "atomic_baseline" not in dict(parameters)
    assert initialized.baseline_metadata == {
        "buffer_name": "atomic_baseline",
        "device": "cpu",
        "dtype": dtype_name,
        "exact_zero": True,
        "is_parameter": False,
        "requires_grad": False,
        "shape": (2,),
    }
    for value in initialized.bundle.model_state.values():
        assert value.device.type == "cpu"
        assert not value.requires_grad and value.grad_fn is None
        assert not value.is_inference()
        if value.is_floating_point():
            assert value.dtype == dtype
            assert torch.all(torch.isfinite(value))


def test_mixed_template_order_default_topology_and_bindings(tmp_path):
    prepared = _mixed_preparation(tmp_path)
    source_order = tuple(
        value.template_id for value in prepared.model_source.reference_templates
    )
    sample_order = tuple(value.template_id for value in prepared.train_samples)
    initialized = initialize_scratch_model(prepared)
    runtime = instantiate_reference_site_model_bundle(initialized.bundle)

    assert source_order == ("zeta-211", "alpha-111")
    assert sample_order == ("zeta-211", "alpha-111")
    assert initialized.template_ids == ("alpha-111", "zeta-211")
    assert initialized.bundle.binding_ids == ("alpha-111", "zeta-211")
    assert initialized.default_template_id == "zeta-211"
    assert runtime.default_template_id == "zeta-211"
    assert runtime.model.topology.num_sites == 16
    assert {
        key: value.structural_artifact.diagnostics.num_sites
        for key, value in {
            binding.template_id: binding
            for binding in initialized.bundle.template_bindings
        }.items()
    } == {"alpha-111": 8, "zeta-211": 16}
    default = runtime.template_contexts["zeta-211"]
    assert torch.equal(runtime.model.phase_modes, default.phase_modes)
    assert torch.equal(
        runtime.model.phase_mode_weights, default.phase_mode_weights
    )
    assert torch.equal(
        runtime.model.site_alignment_weights, default.site_alignment_weights
    )
    assert torch.equal(
        runtime.model.phase_channel_weights, default.phase_channel_weights
    )
    assert tuple(value.template_id for value in prepared.train_samples) == sample_order


def test_prepared_mapping_insertion_order_does_not_change_initial_bundle(tmp_path):
    prepared = _mixed_preparation(tmp_path)

    def reversed_mapping(value):
        return dict(reversed(tuple(value.items())))

    reordered = replace(
        prepared,
        structural_artifacts=reversed_mapping(prepared.structural_artifacts),
        template_contexts=reversed_mapping(prepared.template_contexts),
        evaluation_policies=reversed_mapping(prepared.evaluation_policies),
        template_fingerprints=reversed_mapping(prepared.template_fingerprints),
    )
    first = initialize_scratch_model(prepared)
    second = initialize_scratch_model(reordered)
    assert first.bundle_fingerprint == second.bundle_fingerprint
    assert first.model_state_fingerprint == second.model_state_fingerprint
    assert _state_equal(first.bundle.model_state, second.bundle.model_state)


def test_save_load_is_weights_only_safe_and_state_exact(
    tmp_path, monkeypatch
):
    prepared, _ = _single_preparation(tmp_path)
    initialized = initialize_scratch_model(prepared)
    path = tmp_path / "initial-model.pt"
    save_reference_site_model_bundle(path, initialized.bundle)
    safe_globals = tuple(torch.serialization.get_safe_globals())
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(bundle_module.torch, "load", checked_load)
    loaded = load_reference_site_model_bundle(path)
    runtime = instantiate_reference_site_model_bundle(loaded)
    assert len(calls) == 1 and calls[0]["weights_only"] is True
    assert tuple(torch.serialization.get_safe_globals()) == safe_globals
    assert loaded.bundle_fingerprint == initialized.bundle_fingerprint
    assert loaded.architecture_fingerprint == initialized.architecture_fingerprint
    assert _state_equal(loaded.model_state, initialized.bundle.model_state)
    assert _state_equal(runtime.model.state_dict(), loaded.model_state)
    assert runtime.default_template_id == initialized.default_template_id
    assert tuple(runtime.template_contexts) == initialized.template_ids
    assert loaded.provenance == {
        "atomic_baseline_initialization": "exact_zero",
        "canonical_device": "cpu",
        "initialization_convention_version": "scratch_model_initialization_v1",
        "initialization_seed": 20260904,
        "model_floating_dtype": "float64",
        "source_kind": "scratch",
    }


def test_initialization_preserves_inputs_preparation_process_state_and_ownership(
    tmp_path,
):
    prepared, inputs = _single_preparation(tmp_path, dtype="float32")
    preparation_before = prepared.to_dict()
    input_before = {path: path.read_bytes() for path in inputs}
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    safe_before = tuple(torch.serialization.get_safe_globals())
    cuda_before = (
        tuple(value.clone() for value in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )
    cuda_initialized_at_call = torch.cuda.is_initialized()
    original_default_dtype = torch.get_default_dtype()
    original_grad = torch.is_grad_enabled()
    original_inference = torch.is_inference_mode_enabled()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    forced_default_dtype = torch.float64
    try:
        torch.set_default_dtype(forced_default_dtype)
        with torch.inference_mode():
            assert torch.is_inference_mode_enabled()
            initialized = initialize_scratch_model(prepared)
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
        assert torch.get_default_dtype() == forced_default_dtype
    finally:
        torch.set_default_dtype(original_default_dtype)

    assert prepared.to_dict() == preparation_before
    assert all(path.read_bytes() == content for path, content in input_before.items())
    assert not (tmp_path / "scratch-output").exists()
    assert random.getstate() == python_before
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert torch.is_grad_enabled() == original_grad
    assert torch.is_inference_mode_enabled() == original_inference
    assert torch.are_deterministic_algorithms_enabled() == original_deterministic
    assert tuple(torch.serialization.get_safe_globals()) == safe_before
    assert torch.cuda.is_initialized() == cuda_initialized_at_call
    if cuda_before is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(torch.cuda.get_rng_state_all(), cuda_before)
        )
    assert all(
        not value.is_inference()
        for value in initialized.bundle.model_state.values()
    )
    binding = initialized.bundle.template_bindings[0]
    prepared_artifact = prepared.structural_artifacts[binding.template_id]
    assert (
        binding.structural_artifact.reference_fractional.data_ptr()
        != prepared_artifact.reference_fractional.data_ptr()
    )
    prepared_phase = prepared.model_source.reference_templates[
        0
    ].phase_specification
    assert (
        binding.phase_specification.modes.data_ptr()
        != prepared_phase.modes.data_ptr()
    )
    assert (
        initialized.bundle.model_state["phase_modes"].data_ptr()
        != prepared.template_contexts[
            binding.template_id
        ].phase_modes.data_ptr()
    )


def test_initialization_does_not_call_forward_ot_optimizer_training_or_builders(
    tmp_path, monkeypatch
):
    prepared, _ = _single_preparation(tmp_path)
    import refsite_mlip.data.reference_builder as reference_builder
    import refsite_mlip.graph as graph_module
    import refsite_mlip.models.batch_executor as batch_executor_module
    import refsite_mlip.models.potential as potential_module
    import refsite_mlip.phase.stabilizer as stabilizer_module
    import refsite_mlip.training.checkpointed_fit as checkpointed_fit_module
    import refsite_mlip.training.baseline as baseline_module
    import refsite_mlip.training.fit as fit_module
    import refsite_mlip.training.optimizer as optimizer_module
    import refsite_mlip.training.scheduler as scheduler_module
    import refsite_mlip.training.step as step_module

    monkeypatch.setattr(potential_module.ReferenceSitePotential, "forward", _forbidden)
    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", _forbidden)
    monkeypatch.setattr(potential_module, "solve_sparse_hybrid_eval", _forbidden)
    monkeypatch.setattr(
        potential_module, "solve_sparse_sinkhorn_train_fixed", _forbidden
    )
    monkeypatch.setattr(batch_executor_module, "evaluate_structure_batch", _forbidden)
    monkeypatch.setattr(reference_builder, "build_reference_template_from_atoms", _forbidden)
    monkeypatch.setattr(reference_builder, "build_reference_template_from_poscar", _forbidden)
    monkeypatch.setattr(reference_builder, "canonicalize_reference_atoms", _forbidden)
    monkeypatch.setattr(graph_module, "build_reference_graph_topology", _forbidden)
    monkeypatch.setattr(stabilizer_module, "find_typed_stabilizer", _forbidden)
    monkeypatch.setattr(baseline_module, "fit_atomic_baseline", _forbidden)
    monkeypatch.setattr(optimizer_module, "build_optimizer", _forbidden)
    monkeypatch.setattr(scheduler_module, "build_scheduler", _forbidden)
    monkeypatch.setattr(step_module, "train_step", _forbidden)
    monkeypatch.setattr(fit_module, "run_fit", _forbidden)
    monkeypatch.setattr(checkpointed_fit_module, "run_checkpointed_fit", _forbidden)
    monkeypatch.setattr(torch.autograd, "backward", _forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", _forbidden)

    initialized = initialize_scratch_model(prepared)
    assert initialized.bundle.validate()


@pytest.mark.parametrize(
    ("mutation", "reason", "stage"),
    (
        (
            lambda value: replace(value, structural_artifacts={}),
            "DEFAULT_TEMPLATE_MISSING",
            "initialization.default_template",
        ),
        (
            lambda value: replace(value, species_vocabulary=(6,)),
            "SPECIES_MISMATCH",
            "initialization.species",
        ),
        (
            lambda value: replace(
                value,
                template_fingerprints={
                    "scratch-111-a": {
                        **dict(value.template_fingerprints["scratch-111-a"]),
                        "full_template_fingerprint": "0" * 64,
                    }
                },
            ),
            "PHASE_MISMATCH",
            "initialization.phase",
        ),
        (
            lambda value: replace(
                value, evaluation_policies={"scratch-111-a": object()}
            ),
            "POLICY_MISMATCH",
            "initialization.policy",
        ),
    ),
)
def test_preparation_mismatches_are_structured(
    tmp_path, mutation, reason, stage
):
    prepared, _ = _single_preparation(tmp_path)
    invalid = mutation(prepared)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ScratchModelInitializationError) as caught:
        initialize_scratch_model(invalid)
    assert caught.value.reason_code == reason
    assert caught.value.stage == stage
    assert caught.value.initialization_seed == 20260904
    assert caught.value.config_fingerprint == prepared.config_fingerprint
    assert caught.value.template_id == "scratch-111-a"
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_site_type_and_config_cross_binding_failures_are_structured(tmp_path):
    prepared, _ = _single_preparation(tmp_path / "site")
    artifact = prepared.structural_artifacts["scratch-111-a"]
    artifact.site_types[0] = len(prepared.model_source.potential.feature.site_type_vocabulary)
    with pytest.raises(ScratchModelInitializationError) as caught:
        initialize_scratch_model(prepared)
    assert caught.value.reason_code == "SITE_TYPE_MISMATCH"
    assert caught.value.stage == "initialization.site_type"

    prepared, _ = _single_preparation(tmp_path / "binding")
    invalid = replace(
        prepared,
        runtime=replace(prepared.runtime, seed=prepared.runtime.seed + 1),
    )
    with pytest.raises(ScratchModelInitializationError) as caught:
        initialize_scratch_model(invalid)
    assert caught.value.reason_code == "INVALID_PREPARATION"
    assert caught.value.stage == "initialization.preparation"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (("nonfinite", "NONFINITE_INITIAL_STATE"), ("baseline", "NONZERO_INITIAL_BASELINE")),
)
def test_invalid_initial_model_state_is_rejected_before_capture(
    tmp_path, monkeypatch, mutation, reason
):
    prepared, _ = _single_preparation(tmp_path)
    original_constructor = initialization_module.ReferenceSitePotential

    def construct(*args, **kwargs):
        model = original_constructor(*args, **kwargs)
        with torch.no_grad():
            if mutation == "nonfinite":
                next(model.parameters()).reshape(-1)[0] = float("nan")
            else:
                model.atomic_baseline[0] = 1.0
        return model

    monkeypatch.setattr(
        initialization_module, "ReferenceSitePotential", construct
    )
    with pytest.raises(ScratchModelInitializationError) as caught:
        initialize_scratch_model(prepared)
    assert caught.value.reason_code == reason
    assert caught.value.stage in {
        "initialization.model_state",
        "initialization.baseline",
    }


@pytest.mark.parametrize(
    ("target", "reason", "stage", "original_reason"),
    (
        (
            "ReferenceSitePotential",
            "MODEL_INITIALIZATION_FAILED",
            "initialization.model",
            None,
        ),
        (
            "capture_reference_site_model_bundle",
            "BUNDLE_CAPTURE_FAILED",
            "initialization.bundle_capture",
            "INJECTED_BUNDLE_FAILURE",
        ),
        (
            "instantiate_reference_site_model_bundle",
            "BUNDLE_RECONSTRUCTION_FAILED",
            "initialization.bundle_reconstruction",
            "INJECTED_BUNDLE_FAILURE",
        ),
    ),
)
def test_model_capture_and_reconstruction_failures_preserve_context_and_rng(
    tmp_path, monkeypatch, target, reason, stage, original_reason
):
    prepared, _ = _single_preparation(tmp_path)
    original = (
        ValueError("injected constructor failure")
        if original_reason is None
        else ModelBundleError(
            original_reason,
            "injected bundle failure",
            validation_stage="injected",
        )
    )

    def fail(*args, **kwargs):
        del args, kwargs
        raise original

    monkeypatch.setattr(initialization_module, target, fail)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    cuda_rng = (
        tuple(value.clone() for value in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )
    default_dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    grad_enabled = torch.is_grad_enabled()
    inference_enabled = torch.is_inference_mode_enabled()
    safe_globals = tuple(torch.serialization.get_safe_globals())
    before = prepared.to_dict()
    with pytest.raises(ScratchModelInitializationError) as caught:
        initialize_scratch_model(prepared)
    error = caught.value
    assert error.reason_code == reason
    assert error.stage == stage
    assert error.original_error is original
    assert error.original_reason_code == original_reason
    assert error.template_id == "scratch-111-a"
    assert error.initialization_seed == 20260904
    assert error.config_fingerprint == prepared.config_fingerprint
    assert prepared.to_dict() == before
    assert random.getstate() == python_rng
    assert _numpy_state_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    if cuda_rng is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(torch.cuda.get_rng_state_all(), cuda_rng)
        )
    assert torch.get_default_dtype() == default_dtype
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_grad_enabled() == grad_enabled
    assert torch.is_inference_mode_enabled() == inference_enabled
    assert tuple(torch.serialization.get_safe_globals()) == safe_globals
