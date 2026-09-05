from __future__ import annotations

import copy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch
from ase import Atoms

import refsite_mlip.models.bundle as bundle_module
from refsite_mlip.data import (
    PhaseSpecification,
    StructureSample,
    capture_reference_structure_artifact,
    collate_structure_samples,
)
from refsite_mlip.inference import ReferenceSitePredictor, load_reference_site_predictor
from refsite_mlip.interfaces import ReferenceSiteASECalculator
from refsite_mlip.models import (
    ModelBundleError,
    ReferenceSitePotential,
    capture_reference_site_model_bundle,
    evaluate_structure_batch,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    TransportSupportConfig,
)

from symmetric_potential_helpers import v2_configuration
from test_grouped_evaluation_phase_batch import _adaptive_case
from test_model_bundle_runtime import _phase_from_template


def _capture_v2(typed_crystal, *, edge_backend=False):
    data, _, registry, samples, batch, contexts, policies = _adaptive_case(
        typed_crystal
    )
    default = registry.resolve("zeta")
    config = v2_configuration(torch.float64, order=3, layers=2)
    if edge_backend:
        config = replace(
            config,
            transport_support=TransportSupportConfig(
                kind="compact_c2",
                cutoff=2.6,
                switch_width=0.5,
                candidate_skin=0.2,
                backend="edge_list",
                candidate_backend="blocked",
                site_block_size=2,
                atom_block_size=3,
            ),
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(117)
        model = ReferenceSitePotential(
            config,
            default.topology,
            default.phase_modes,
            default.phase_mode_weights,
            torch.eye(2, dtype=torch.float64),
            default.site_alignment_weights,
            default.phase_channel_weights,
            (-1.0, 2.0),
        ).to(dtype=torch.float64)
    artifacts = {}
    phases: dict[str, PhaseSpecification] = {}
    for template_id in ("alpha", "zeta"):
        template = registry.resolve(template_id)
        artifacts[template_id] = capture_reference_structure_artifact(
            template, avg_num_neighbors=6.0
        )
        phases[template_id] = _phase_from_template(template)
    bundle = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts=artifacts,
        phase_specifications=phases,
        evaluation_policies=policies,
        default_template_id="zeta",
        provenance={"purpose": "symmetric_v2_bundle_test"},
    )
    return data, model, registry, samples, batch, contexts, policies, bundle


def _geometry(sample: StructureSample) -> StructureSample:
    return StructureSample(
        sample_id=sample.sample_id,
        positions=sample.positions.detach().clone(),
        atomic_numbers=sample.atomic_numbers.detach().clone(),
        cell=sample.cell.detach().clone(),
        pbc=sample.pbc.detach().clone(),
        origin=sample.origin.detach().clone(),
        template_id=sample.template_id,
    )


def _grouped(model, batch, contexts, policies, solver_path):
    prepared = replace(
        batch, positions=batch.positions.detach().clone().requires_grad_(True)
    )
    return evaluate_structure_batch(
        model,
        prepared,
        contexts,
        solver_path=solver_path,
        evaluation_policies=policies if solver_path == EVAL_ADAPTIVE else None,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )


def _single(model, sample, context, policy, solver_path):
    return model(
        sample.positions.detach().clone().requires_grad_(True),
        sample.atomic_numbers,
        sample.cell,
        sample.origin,
        solver_path=solver_path,
        template_context=context,
        evaluation_policy=policy if solver_path == EVAL_ADAPTIVE else None,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )


def _assert_single_equal(left, right):
    for name in (
        "energy",
        "site_energy",
        "baseline_energy",
        "residual_energy",
        "site_features",
        "raw_c",
        "forces",
        "stress",
        "stress_voigt",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name))
    assert torch.equal(left.auxiliary["phase"], right.auxiliary["phase"])
    assert torch.equal(left.auxiliary["ot"].q, right.auxiliary["ot"].q)
    assert torch.equal(
        left.auxiliary["multipoles"].equivariant_features,
        right.auxiliary["multipoles"].equivariant_features,
    )


def _assert_output_equal(left, right):
    for first, second in (
        (left.energy, right.energy),
        (left.baseline_energy, right.baseline_energy),
        (left.residual_energy, right.residual_energy),
        (left.site_energy, right.site_energy),
        (left.forces, right.forces),
        (left.stress, right.stress),
        (left.stress_voigt, right.stress_voigt),
    ):
        assert torch.equal(first, second)
    assert left.sample_ids == right.sample_ids
    assert left.template_ids == right.template_ids
    for first, second in zip(left.auxiliary, right.auxiliary):
        assert torch.equal(first["phase"], second["phase"])
        if hasattr(first["ot"], "P"):
            assert torch.equal(first["ot"].P, second["ot"].P)
        else:
            assert torch.equal(first["ot"].edge_plan, second["ot"].edge_plan)
        assert torch.equal(first["ot"].q, second["ot"].q)
        assert torch.equal(
            first["multipoles"].equivariant_features,
            second["multipoles"].equivariant_features,
        )


def test_v2_bundle_schema_capture_ownership_and_strict_reconstruction(
    typed_crystal, tmp_path
):
    *_, model, registry, samples, batch, contexts, policies, bundle = (
        _capture_v2(typed_crystal)
    )
    assert bundle.schema_version == "reference_site_model_bundle_v1"
    assert (
        bundle.model_config["higher_body"]["contract_version"]
        == "central_conditioned_symmetric_power_v2"
    )
    u_keys = tuple(
        key for key in bundle.model_state_keys if key.startswith("symmetric_cg_basis.")
    )
    w_keys = tuple(
        key
        for key in bundle.model_state_keys
        if ".symmetric_contraction.weight_" in key
    )
    assert len(u_keys) == 9
    assert len(w_keys) == 18
    assert not any(key.startswith("layers.") and ".u_output_" in key for key in bundle.model_state_keys)
    assert sum(bundle.model_state[key].numel() for key in u_keys) == sum(
        value.numel() for value in model.symmetric_cg_basis.buffers()
    )
    architecture = bundle_module._architecture_payload(
        bundle.model_config,
        bundle.model_state,
        bundle.model_state_keys,
        bundle.species_vocabulary,
        bundle.conventions,
    )["symmetric_correlation_contract"]
    assert architecture["correlation_order"] == 3
    assert architecture["basis_kind"] == "full_path"
    assert architecture["normalization"] == "component"
    assert architecture["basis_content_fingerprint"] == model.symmetric_cg_basis.basis_fingerprint
    assert [item["name"] for item in architecture["central_channel_ordering"]] == [
        "constant",
        "species",
        "vacancy",
        "site_type",
        "vacancy_site_type",
    ]
    assert len(architecture["u_tensors"]) == 9
    assert len(architecture["w_tensors"]) == 18
    for key in bundle.model_state_keys:
        if bundle.model_state[key].numel():
            assert bundle.model_state[key].data_ptr() != model.state_dict()[key].data_ptr()

    repeated = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts={
            item.template_id: item.structural_artifact
            for item in bundle.template_bindings
        },
        phase_specifications={
            item.template_id: item.phase_specification
            for item in bundle.template_bindings
        },
        evaluation_policies=policies,
        default_template_id="zeta",
        provenance={"purpose": "symmetric_v2_bundle_test"},
    )
    assert repeated.architecture_fingerprint == bundle.architecture_fingerprint
    assert repeated.bundle_fingerprint == bundle.bundle_fingerprint

    path = tmp_path / "symmetric-v2.pt"
    save_reference_site_model_bundle(path, bundle)
    loaded = load_reference_site_model_bundle(path)
    runtime = instantiate_reference_site_model_bundle(loaded)
    assert runtime.registry.fingerprint == registry.fingerprint
    assert runtime.model.symmetric_cg_basis.basis_fingerprint == model.symmetric_cg_basis.basis_fingerprint
    assert tuple(runtime.model.state_dict()) == bundle.model_state_keys
    for key, value in model.state_dict().items():
        assert torch.equal(runtime.model.state_dict()[key], value)
        if value.numel():
            assert runtime.model.state_dict()[key].data_ptr() != loaded.model_state[key].data_ptr()

    original = _grouped(model, batch, contexts, policies, TRAIN_FIXED)
    reconstructed = _grouped(
        runtime.model,
        batch,
        runtime.template_contexts,
        runtime.evaluation_policies,
        TRAIN_FIXED,
    )
    _assert_output_equal(original, reconstructed)
    sample = samples[0]
    _assert_single_equal(
        _single(
            model,
            sample,
            contexts[sample.template_id],
            policies[sample.template_id],
            TRAIN_FIXED,
        ),
        _single(
            runtime.model,
            sample,
            runtime.template_contexts[sample.template_id],
            runtime.evaluation_policies[sample.template_id],
            TRAIN_FIXED,
        ),
    )
    assert path.stat().st_size > sum(
        value.numel() * value.element_size() for value in bundle.model_state.values()
    )


@pytest.mark.parametrize("edge_backend", [False, True])
def test_v2_bundle_adaptive_grouped_branch_and_sparse_parity(
    typed_crystal, tmp_path, edge_backend
):
    *_, model, _, samples, batch, contexts, policies, bundle = _capture_v2(
        typed_crystal, edge_backend=edge_backend
    )
    runtime = instantiate_reference_site_model_bundle(bundle)
    original = _grouped(model, batch, contexts, policies, EVAL_ADAPTIVE)
    reconstructed = _grouped(
        runtime.model,
        batch,
        runtime.template_contexts,
        runtime.evaluation_policies,
        EVAL_ADAPTIVE,
    )
    _assert_output_equal(original, reconstructed)
    sample = samples[0]
    _assert_single_equal(
        _single(
            model,
            sample,
            contexts[sample.template_id],
            policies[sample.template_id],
            EVAL_ADAPTIVE,
        ),
        _single(
            runtime.model,
            sample,
            runtime.template_contexts[sample.template_id],
            runtime.evaluation_policies[sample.template_id],
            EVAL_ADAPTIVE,
        ),
    )
    for left, right in zip(original.auxiliary, reconstructed.auxiliary):
        first = left["evaluation_diagnostics"]
        second = right["evaluation_diagnostics"]
        assert first.selected_grouped_index == second.selected_grouped_index
        assert first.transport_fallback_used is False
        assert second.transport_fallback_used is False
        if edge_backend:
            assert first.transport_backend == "edge_list"
            assert not first.transport_dense_plan_materialized
            assert (
                first.transport_support_fingerprint
                == second.transport_support_fingerprint
            )
    path = tmp_path / f"adaptive-{edge_backend}.pt"
    save_reference_site_model_bundle(path, bundle)
    predictor = load_reference_site_predictor(path)
    predicted = predictor.predict_samples(
        tuple(_geometry(sample) for sample in samples),
        solver_path=EVAL_ADAPTIVE,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert torch.equal(predicted.energy, reconstructed.energy)
    assert torch.equal(predicted.forces, reconstructed.forces)
    assert torch.equal(predicted.stress, reconstructed.stress)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v2_bundle_cpu_materialization_predictor_and_ase(
    typed_crystal, tmp_path, dtype
):
    *_, samples, _, _, _, bundle = _capture_v2(typed_crystal)
    path = tmp_path / f"symmetric-{dtype}.pt"
    save_reference_site_model_bundle(path, bundle)
    predictor = load_reference_site_predictor(path, device="cpu", dtype=dtype)
    assert all(
        value.dtype == dtype
        for value in predictor.model.state_dict().values()
        if value.is_floating_point()
    )
    geometry = tuple(_geometry(sample) for sample in samples)
    predicted = predictor.predict_samples(
        geometry,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert predicted.template_ids == tuple(sample.template_id for sample in samples)
    for value in (predicted.energy, predicted.forces, predicted.stress):
        assert value.dtype == dtype and bool(torch.all(torch.isfinite(value)))

    sample = geometry[0]
    atoms = Atoms(
        numbers=sample.atomic_numbers.cpu().numpy(),
        positions=sample.positions.cpu().numpy(),
        cell=sample.cell.cpu().numpy(),
        pbc=sample.pbc.cpu().numpy(),
    )
    calculator = ReferenceSiteASECalculator(
        path,
        template_id=sample.template_id,
        device="cpu",
        dtype=dtype,
        solver_path=TRAIN_FIXED,
    )
    direct = calculator.predictor.predict_sample(
        sample,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
    )
    atoms.calc = calculator
    tolerance = 2.0e-5 if dtype == torch.float32 else 4.0e-13
    assert atoms.get_potential_energy() == pytest.approx(
        float(direct.energy), abs=tolerance, rel=tolerance
    )
    np.testing.assert_allclose(
        atoms.get_forces(), direct.forces.cpu().numpy(), atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        atoms.get_stress(),
        direct.stress_voigt.cpu().numpy(),
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        atoms.get_stress(voigt=False),
        direct.stress.cpu().numpy(),
        atol=tolerance,
        rtol=tolerance,
    )
    if dtype == torch.float64:
        adaptive_atoms = atoms.copy()
        adaptive_calculator = ReferenceSiteASECalculator(
            path,
            template_id=sample.template_id,
            device="cpu",
            dtype=dtype,
            solver_path=EVAL_ADAPTIVE,
        )
        adaptive_atoms.calc = adaptive_calculator
        assert np.isfinite(adaptive_atoms.get_potential_energy())
        assert np.all(np.isfinite(adaptive_atoms.get_forces()))
        assert np.all(np.isfinite(adaptive_atoms.get_stress()))


def test_v2_capture_load_instantiate_preserves_process_and_caller_state(
    typed_crystal, tmp_path
):
    *_, model, _, _, _, _, _, bundle = _capture_v2(typed_crystal)
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, 0.125)
    gradient = parameter.grad
    gradient_value = gradient.clone()
    model.train()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_state = (
        tuple(value.clone() for value in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )
    safe_globals = tuple(torch.serialization.get_safe_globals())
    default_dtype = torch.get_default_dtype()
    grad_enabled = torch.is_grad_enabled()
    inference_enabled = torch.is_inference_mode_enabled()
    path = tmp_path / "state-preservation.pt"
    save_reference_site_model_bundle(path, bundle)
    loaded = load_reference_site_model_bundle(path)
    instantiate_reference_site_model_bundle(loaded)
    assert random.getstate() == python_state
    observed_numpy = np.random.get_state()
    assert observed_numpy[0] == numpy_state[0]
    assert np.array_equal(observed_numpy[1], numpy_state[1])
    assert observed_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    if cuda_state is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(cuda_state, torch.cuda.get_rng_state_all())
        )
    assert tuple(torch.serialization.get_safe_globals()) == safe_globals
    assert torch.get_default_dtype() == default_dtype
    assert torch.is_grad_enabled() == grad_enabled
    assert torch.is_inference_mode_enabled() == inference_enabled
    assert model.training
    assert parameter.grad is gradient
    assert torch.equal(parameter.grad, gradient_value)


def test_v2_capture_rejects_mutated_runtime_basis_before_snapshot(typed_crystal):
    *_, model, _, _, _, _, policies, bundle = _capture_v2(typed_crystal)
    first = next(model.symmetric_cg_basis.buffers())
    with torch.no_grad():
        first.reshape(-1)[0] += 1.0
    with pytest.raises(ModelBundleError) as caught:
        capture_reference_site_model_bundle(
            model=model,
            structural_artifacts={
                item.template_id: item.structural_artifact
                for item in bundle.template_bindings
            },
            phase_specifications={
                item.template_id: item.phase_specification
                for item in bundle.template_bindings
            },
            evaluation_policies=policies,
            default_template_id="zeta",
        )
    assert caught.value.reason_code == "SYMMETRIC_BASIS_CONTENT_MISMATCH"


def _rehash_outer(payload):
    payload["bundle_fingerprint"] = bundle_module._fingerprint(
        "reference_site_model_bundle_v1", payload["payload"]
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("u_missing", "SYMMETRIC_BASIS_KEY_MISMATCH"),
        ("u_extra", "SYMMETRIC_BASIS_KEY_MISMATCH"),
        ("u_shape", "SYMMETRIC_BASIS_SHAPE_MISMATCH"),
        ("u_value", "SYMMETRIC_BASIS_CONTENT_MISMATCH"),
        ("w_missing", "SYMMETRIC_WEIGHT_KEY_MISMATCH"),
        ("w_extra", "SYMMETRIC_WEIGHT_KEY_MISMATCH"),
        ("w_shape", "SYMMETRIC_WEIGHT_SHAPE_MISMATCH"),
        ("legacy_state", "SYMMETRIC_LEGACY_STATE_CONTAMINATION"),
    ],
)
def test_v2_bundle_rejects_rehashed_internal_state_corruption(
    typed_crystal, mutation, reason
):
    *_, bundle = _capture_v2(typed_crystal)
    payload = copy.deepcopy(bundle.to_payload())
    body = payload["payload"]
    state = body["model_state"]
    keys = body["model_state_keys"]
    u_key = next(key for key in keys if key.startswith("symmetric_cg_basis."))
    w_key = next(key for key in keys if ".symmetric_contraction.weight_" in key)
    if mutation == "u_missing":
        del state[u_key]
        keys.remove(u_key)
    elif mutation == "u_extra":
        state["symmetric_cg_basis.unexpected"] = torch.zeros(1)
        keys.append("symmetric_cg_basis.unexpected")
    elif mutation == "u_shape":
        state[u_key] = state[u_key].reshape(-1)
    elif mutation == "u_value":
        state[u_key].reshape(-1)[0] += 1.0
    elif mutation == "w_missing":
        del state[w_key]
        keys.remove(w_key)
    elif mutation == "w_extra":
        state["layers.0.symmetric_contraction.weight_unexpected"] = torch.zeros(1)
        keys.append("layers.0.symmetric_contraction.weight_unexpected")
    elif mutation == "legacy_state":
        key = "layers.0.corr.legacy_fallback"
        state[key] = torch.zeros(1, dtype=torch.float64)
        keys.append(key)
    else:
        state[w_key] = state[w_key].reshape(-1)
    _rehash_outer(payload)
    with pytest.raises(ModelBundleError) as caught:
        bundle_module._bundle_from_safe_payload(payload, bundle_path="attacker.pt")
    assert caught.value.reason_code == reason


def test_v1_and_v2_architecture_separation_is_strict(typed_crystal):
    *_, v2_bundle = _capture_v2(typed_crystal)
    payload = copy.deepcopy(v2_bundle.to_payload())
    payload["payload"]["model_config"]["higher_body"]["contract_version"] = (
        "unknown_contract"
    )
    _rehash_outer(payload)
    with pytest.raises(ModelBundleError) as caught:
        bundle_module._bundle_from_safe_payload(payload, bundle_path="unknown.pt")
    assert caught.value.reason_code == "INVALID_MODEL_CONFIG"

    payload = copy.deepcopy(v2_bundle.to_payload())
    payload["payload"]["model_config"]["higher_body"]["correlation_mode"] = "uuu"
    _rehash_outer(payload)
    with pytest.raises(ModelBundleError) as caught:
        bundle_module._bundle_from_safe_payload(payload, bundle_path="mixed.pt")
    assert caught.value.reason_code == "INVALID_PAYLOAD"

    payload = copy.deepcopy(v2_bundle.to_payload())
    payload["payload"]["model_config"]["higher_body"]["symmetric_correlation"][
        "correlation_order"
    ] = 2
    _rehash_outer(payload)
    with pytest.raises(ModelBundleError) as caught:
        bundle_module._bundle_from_safe_payload(payload, bundle_path="order.pt")
    assert caught.value.reason_code == "SYMMETRIC_BASIS_KEY_MISMATCH"


def test_architecture_fingerprint_tamper_is_rejected(typed_crystal):
    *_, bundle = _capture_v2(typed_crystal)
    payload = copy.deepcopy(bundle.to_payload())
    payload["payload"]["architecture_fingerprint"] = "0" * 64
    _rehash_outer(payload)
    with pytest.raises(ModelBundleError) as caught:
        bundle_module._bundle_from_safe_payload(payload, bundle_path="architecture.pt")
    assert caught.value.reason_code == "ARCHITECTURE_FINGERPRINT_MISMATCH"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_v2_bundle_cuda_materialization_smoke(typed_crystal, dtype):
    *_, samples, _, _, _, bundle = _capture_v2(typed_crystal, edge_backend=True)
    runtime = instantiate_reference_site_model_bundle(
        bundle, device="cuda:0", dtype=dtype
    )
    assert all(
        value.device.type == "cuda" and value.dtype == dtype
        for value in runtime.model.state_dict().values()
        if value.is_floating_point()
    )
    predictor = ReferenceSitePredictor(runtime)
    output = predictor.predict_sample(
        _geometry(samples[0]),
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
    )
    torch.cuda.synchronize()
    for value in (output.energy, output.forces, output.stress):
        assert value.device.type == "cuda"
        assert value.dtype == dtype
        assert bool(torch.all(torch.isfinite(value)))
