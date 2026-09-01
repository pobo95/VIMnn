from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch

import refsite_mlip.models.bundle as bundle_module
from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceTemplate,
    capture_reference_structure_artifact,
)
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import (
    EvaluationPolicy,
    ModelBundleError,
    PotentialConfig,
    ReferenceSitePotential,
    TemplateExecutionContext,
    capture_reference_site_model_bundle,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.phase import find_typed_stabilizer
from refsite_mlip.transport import EVAL_ADAPTIVE


def _phase(data, *, mode_count=5, scale=1.0):
    return PhaseSpecification(
        modes=data["modes"][:mode_count],
        mode_weights=scale * data["mode_weights"][:mode_count],
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=data["channel_weights"],
        approval_status="provisional",
        convention_version="bundle_test_phase_v1",
    )


def _template(data, template_id="bundle-zeta", *, site_count=6, mode_count=5):
    topology = build_reference_graph_topology(
        data["sites"][:site_count],
        data["site_types"][:site_count],
        data["cell"],
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    phase = _phase(data, mode_count=mode_count)
    template = ReferenceTemplate.snapshot(
        template_id,
        topology,
        phase.modes,
        phase.mode_weights,
        phase.site_type_alignment_weights[topology.site_types],
        phase.channel_weights,
        find_typed_stabilizer(
            topology.reference_fractional, topology.site_types
        ),
        (6, 41),
    )
    return template, phase


def _config():
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=2,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        # The same portable bundle is intentionally materialized as float32
        # in the CUDA smoke, so its explicit validation contract must support
        # legitimate float32 fixed-Sinkhorn accumulation.
        probability_tolerance=1.0e-6,
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
        readout_hidden=8,
        energy_scale=1.0,
    )


def _model(template):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260901)
        return ReferenceSitePotential(
            _config(),
            template.topology,
            template.phase_modes,
            template.phase_mode_weights,
            torch.eye(2, dtype=torch.float64),
            template.site_alignment_weights,
            template.phase_channel_weights,
            (-1.0, 2.0),
        ).to(dtype=torch.float64)


def _policy(template):
    return EvaluationPolicy(
        template_id=template.template_id,
        template_fingerprint=template.fingerprint,
        candidate_offsets=torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=torch.float64,
        ),
        phase_step_schedule=(0.7, 0.8, 0.9, 1.0),
        phase_damping_schedule=(2.0, 1.0, 0.5, 0.2),
        minimum_objective_gap_absolute=1.0e-2,
        minimum_cross_amplitude_absolute=1.0e-12,
        minimum_atomic_amplitude_absolute=1.0e-12,
        minimum_reference_amplitude_absolute=1.0e-12,
        minimum_curvature=1.0e-2,
        maximum_condition=1.0e8,
        maximum_gradient_norm=2.0e-4,
        equivalence_tolerance=1.0e-8,
    )


def _capture(data, *, include_policy=True, provenance=None):
    template, phase = _template(data)
    model = _model(template)
    artifact = capture_reference_structure_artifact(
        template, avg_num_neighbors=6.0
    )
    policy = _policy(template)
    bundle = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts={template.template_id: artifact},
        phase_specifications={template.template_id: phase},
        evaluation_policies=(
            {template.template_id: policy} if include_policy else None
        ),
        default_template_id=template.template_id,
        provenance={} if provenance is None else provenance,
    )
    return model, template, phase, policy, bundle


def _assert_safe(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
        assert not value.requires_grad and value.grad_fn is None
        return
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_safe(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_safe(item)
        return
    assert value is None or type(value) in (str, bool, int, float)


def test_capture_safe_owned_deterministic_and_preserves_model_state(typed_crystal):
    model, _, _, _, bundle = _capture(
        typed_crystal,
        provenance={"z": [3, 2], "a": {"note": "synthetic"}},
    )
    first = next(model.parameters())
    first.grad = torch.full_like(first, 0.125)
    gradient = first.grad
    gradient_value = gradient.clone()
    model.train()
    parameter_ids = tuple(id(value) for value in model.parameters())
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    cpu_rng = torch.get_rng_state().clone()
    binding = bundle.template_bindings[0]
    repeated = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts={
            binding.template_id: binding.structural_artifact
        },
        phase_specifications={
            binding.template_id: binding.phase_specification
        },
        evaluation_policies={
            binding.template_id: binding.evaluation_policy
        },
        default_template_id=binding.template_id,
        provenance={"a": {"note": "synthetic"}, "z": [3, 2]},
    )
    assert repeated.bundle_fingerprint == bundle.bundle_fingerprint
    assert repeated.architecture_fingerprint == bundle.architecture_fingerprint
    _assert_safe(bundle.to_payload())
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert model.training
    assert first.grad is gradient and torch.equal(first.grad, gradient_value)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    for key, value in state_before.items():
        assert torch.equal(model.state_dict()[key], value)

    parameter_key = next(iter(dict(model.named_parameters())))
    captured_parameter = bundle.model_state[parameter_key].clone()
    with torch.no_grad():
        first.add_(1.0)
    assert torch.equal(bundle.model_state[parameter_key], captured_parameter)
    bundle.model_state[parameter_key].view(-1)[0] += 0.25
    with pytest.raises(ModelBundleError) as caught:
        bundle.validate()
    assert caught.value.reason_code in {
        "BUNDLE_FINGERPRINT_MISMATCH",
        "DEFAULT_TEMPLATE_STATE_MISMATCH",
        "MODEL_CONVENTION_STATE_MISMATCH",
    }


def test_save_load_instantiate_exact_state_and_direct_runtime(typed_crystal, tmp_path, monkeypatch):
    model, template, _, policy, bundle = _capture(typed_crystal)
    path = tmp_path / "portable.pt"
    save_reference_site_model_bundle(path, bundle)
    safe_globals = list(torch.serialization.get_safe_globals())
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(bundle_module.torch, "load", checked_load)
    loaded = load_reference_site_model_bundle(path)
    assert calls and calls[0]["weights_only"] is True
    assert torch.serialization.get_safe_globals() == safe_globals
    runtime = instantiate_reference_site_model_bundle(loaded)
    assert runtime.default_template_id == template.template_id
    assert runtime.bundle_fingerprint == bundle.bundle_fingerprint
    assert runtime.template_fingerprints[template.template_id] == template.fingerprint
    assert runtime.evaluation_policies[template.template_id].content_fingerprint == policy.content_fingerprint
    assert runtime.metadata["phase_approval_status"][template.template_id] == "provisional"
    assert not runtime.metadata["candidate_neighbor_state_persisted"]
    for key, value in model.state_dict().items():
        assert torch.equal(runtime.model.state_dict()[key], value)

    context = TemplateExecutionContext.from_reference_template(
        template, avg_num_neighbors=6.0
    )
    positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    numbers = torch.tensor([6, 41, 6, 41, 6], dtype=torch.long)
    original = model(
        typed_crystal["positions"][:5].clone().requires_grad_(True),
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
        template_context=context,
    )
    restored = runtime.model(
        positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
        template_context=runtime.template_contexts[template.template_id],
    )
    for left, right in (
        (original.energy, restored.energy),
        (original.site_energy, restored.site_energy),
        (original.auxiliary["ot"].P, restored.auxiliary["ot"].P),
        (original.auxiliary["ot"].q, restored.auxiliary["ot"].q),
        (original.raw_c, restored.raw_c),
    ):
        assert torch.equal(left, right)
    torch.testing.assert_close(original.forces, restored.forces, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(original.stress, restored.stress, atol=2e-14, rtol=2e-14)


def test_policy_optional_and_adaptive_missing_policy_is_not_generated(typed_crystal):
    _, template, _, _, bundle = _capture(typed_crystal, include_policy=False)
    runtime = instantiate_reference_site_model_bundle(bundle)
    assert dict(runtime.evaluation_policies) == {}
    numbers = torch.tensor([6, 41, 6, 41, 6], dtype=torch.long)
    with pytest.raises(ValueError, match="evaluation_policy"):
        runtime.model(
            typed_crystal["positions"][:5],
            numbers,
            typed_crystal["cell"],
            typed_crystal["origin"],
            solver_path=EVAL_ADAPTIVE,
            template_context=runtime.template_contexts[template.template_id],
        )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda payload: payload.update(extra=True), "INVALID_PAYLOAD_KEYS"),
        (
            lambda payload: payload.__setitem__("schema_version", "future_v99"),
            "UNSUPPORTED_SCHEMA",
        ),
        (
            lambda payload: payload["payload"]["model_state"]["atomic_baseline"].fill_(float("nan")),
            "NONFINITE_MODEL_STATE",
        ),
        (
            lambda payload: payload["payload"]["version_metadata"].__setitem__("torch_version", "3.0.0"),
            "UNSUPPORTED_RUNTIME_VERSION",
        ),
        (
            lambda payload: payload["payload"]["template_bindings"][0]["phase_specification"]["mode_weights"].__setitem__(0, 9.0),
            "TEMPLATE_FINGERPRINT_MISMATCH",
        ),
        (
            lambda payload: payload["payload"]["template_bindings"][0]["evaluation_policy"]["maximum_gradient_norm"].__class__,
            None,
        ),
    ],
)
def test_strict_corruption_rejection(typed_crystal, tmp_path, mutation, reason):
    _, _, _, _, bundle = _capture(typed_crystal)
    payload = bundle.to_payload()
    if reason is None:
        payload["payload"]["template_bindings"][0]["evaluation_policy"][
            "maximum_gradient_norm"
        ] *= 2.0
        reason = "INVALID_EVALUATION_POLICY"
    else:
        mutation(payload)
    path = tmp_path / "corrupt.pt"
    torch.save(payload, path)
    with pytest.raises(ModelBundleError) as caught:
        load_reference_site_model_bundle(path)
    assert caught.value.reason_code == reason


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda payload: payload["payload"]["model_config"].__setitem__(
                "num_layers", 0
            ),
            "INVALID_MODEL_CONFIG",
        ),
        (
            lambda payload: payload["payload"]["template_bindings"][0][
                "structural_artifact"
            ]["payload"]["topology"]["reference_cell"].add_(0.01),
            "NESTED_ARTIFACT_INVALID",
        ),
        (
            lambda payload: payload["payload"]["template_bindings"][0].__setitem__(
                "full_template_fingerprint", "0" * 64
            ),
            "TEMPLATE_FINGERPRINT_MISMATCH",
        ),
        (
            lambda payload: payload.__setitem__("bundle_fingerprint", "0" * 64),
            "BUNDLE_FINGERPRINT_MISMATCH",
        ),
        (
            lambda payload: payload["payload"].__setitem__(
                "default_template_id", "missing-template"
            ),
            "MISSING_DEFAULT_TEMPLATE",
        ),
        (
            lambda payload: payload["payload"].__setitem__(
                "species_vocabulary", [41, 6]
            ),
            "SPECIES_ORDER_MISMATCH",
        ),
        (
            lambda payload: payload["payload"]["model_state"].pop(
                "central.embedding.weight"
            ),
            "INVALID_STATE_KEYS",
        ),
        (
            lambda payload: payload["payload"]["model_state"].__setitem__(
                "central.embedding.weight",
                payload["payload"]["model_state"]["central.embedding.weight"][:1],
            ),
            "ARCHITECTURE_FINGERPRINT_MISMATCH",
        ),
        (
            lambda payload: payload["payload"]["model_state"].__setitem__(
                "central.embedding.weight",
                payload["payload"]["model_state"]["central.embedding.weight"].float(),
            ),
            "STATE_DTYPE_MISMATCH",
        ),
        (
            lambda payload: payload["payload"]["version_metadata"].__setitem__(
                "e3nn_version", "1.0.0"
            ),
            "UNSUPPORTED_RUNTIME_VERSION",
        ),
    ],
)
def test_additional_model_bundle_corruption_contracts(
    typed_crystal, tmp_path, mutation, reason
):
    _, _, _, _, bundle = _capture(typed_crystal)
    payload = bundle.to_payload()
    mutation(payload)
    path = tmp_path / "corrupt-extra.pt"
    torch.save(payload, path)
    with pytest.raises(ModelBundleError) as caught:
        load_reference_site_model_bundle(path)
    assert caught.value.reason_code == reason


@pytest.mark.parametrize("contents", [b"not a torch archive", b"PK\x03\x04"])
def test_non_torch_and_truncated_bundle_rejected(tmp_path, contents):
    path = tmp_path / "invalid.pt"
    path.write_bytes(contents)
    with pytest.raises(ModelBundleError) as caught:
        load_reference_site_model_bundle(path)
    assert caught.value.reason_code == "SAFE_LOAD_FAILURE"


def test_atomic_save_failure_overwrite_and_symlink_contract(typed_crystal, tmp_path, monkeypatch):
    _, _, _, _, bundle = _capture(typed_crystal)
    target = tmp_path / "bundle.pt"
    save_reference_site_model_bundle(target, bundle)
    original_fingerprint = load_reference_site_model_bundle(target).bundle_fingerprint
    with pytest.raises(FileExistsError):
        save_reference_site_model_bundle(target, bundle)

    original_save = bundle_module.torch.save

    def failed_save(*args, **kwargs):
        raise OSError("injected save failure")

    monkeypatch.setattr(bundle_module.torch, "save", failed_save)
    with pytest.raises(OSError, match="injected"):
        save_reference_site_model_bundle(target, bundle, overwrite=True)
    monkeypatch.setattr(bundle_module.torch, "save", original_save)
    assert load_reference_site_model_bundle(target).bundle_fingerprint == original_fingerprint

    original_replace = bundle_module.os.replace

    def failed_replace(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(bundle_module.os, "replace", failed_replace)
    with pytest.raises(OSError, match="injected"):
        save_reference_site_model_bundle(target, bundle, overwrite=True)
    monkeypatch.setattr(bundle_module.os, "replace", original_replace)
    assert load_reference_site_model_bundle(target).bundle_fingerprint == original_fingerprint
    assert not list(tmp_path.glob(".bundle.pt.*.tmp"))

    link = tmp_path / "link.pt"
    link.symlink_to(target)
    with pytest.raises(ModelBundleError) as caught:
        load_reference_site_model_bundle(link)
    assert caught.value.reason_code == "SYMLINK_REJECTED"
    with pytest.raises(ModelBundleError):
        save_reference_site_model_bundle(link, bundle, overwrite=True)


def test_relocated_path_semantics_rng_and_no_builder_on_load(typed_crystal, tmp_path, monkeypatch):
    _, _, _, _, bundle = _capture(typed_crystal)
    first = tmp_path / "first.pt"
    second_dir = tmp_path / "relocated"
    second_dir.mkdir()
    second = second_dir / "renamed.pt"
    save_reference_site_model_bundle(first, bundle)
    second.write_bytes(first.read_bytes())
    before = torch.get_rng_state().clone()

    import refsite_mlip.data.reference_builder as builder_module
    import refsite_mlip.graph as graph_module
    import refsite_mlip.phase.stabilizer as stabilizer_module

    def forbidden(*args, **kwargs):
        raise AssertionError("builder must not run during bundle load/instantiate")

    monkeypatch.setattr(builder_module, "build_reference_template_from_atoms", forbidden)
    monkeypatch.setattr(builder_module, "canonicalize_reference_atoms", forbidden)
    monkeypatch.setattr(graph_module, "build_reference_graph_topology", forbidden)
    monkeypatch.setattr(stabilizer_module, "find_typed_stabilizer", forbidden)
    left = load_reference_site_model_bundle(first)
    right = load_reference_site_model_bundle(second)
    runtime = instantiate_reference_site_model_bundle(right)
    assert left.bundle_fingerprint == right.bundle_fingerprint == bundle.bundle_fingerprint
    assert runtime.bundle_fingerprint == bundle.bundle_fingerprint
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_materialization_keeps_stored_snapshot_cpu(typed_crystal, dtype):
    _, template, _, _, bundle = _capture(typed_crystal)
    runtime = instantiate_reference_site_model_bundle(
        bundle, device="cuda", dtype=dtype
    )
    assert all(value.device.type == "cpu" for value in bundle.model_state.values())
    assert all(value.device.type == "cuda" for value in runtime.model.state_dict().values())
    numbers = torch.tensor([6, 41, 6, 41, 6], dtype=torch.long, device="cuda")
    output = runtime.model(
        typed_crystal["positions"][:5]
        .to("cuda", dtype=dtype)
        .requires_grad_(True),
        numbers,
        typed_crystal["cell"].to("cuda", dtype=dtype),
        typed_crystal["origin"].to("cuda", dtype=dtype),
        compute_forces=True,
        compute_stress=True,
        template_context=runtime.template_contexts[template.template_id],
    )
    assert output.energy.dtype == dtype and output.energy.device.type == "cuda"
    assert torch.isfinite(output.energy)
    assert torch.all(torch.isfinite(output.forces))
    assert torch.all(torch.isfinite(output.stress))
