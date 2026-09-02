from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest
import torch

import refsite_mlip.inference.predictor as predictor_module
from refsite_mlip.data import StructureSample, collate_structure_samples
from refsite_mlip.inference import (
    PredictorConfig,
    PredictorError,
    ReferenceSitePredictor,
    load_reference_site_predictor,
)
from refsite_mlip.models import (
    capture_reference_site_model_bundle,
    evaluate_structure_batch,
    save_reference_site_model_bundle,
)
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from test_model_bundle_runtime import _capture_case


def _assert_detached_tree(value):
    if isinstance(value, torch.Tensor):
        assert not value.requires_grad and value.grad_fn is None
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_detached_tree(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_detached_tree(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_detached_tree(getattr(value, field.name))


def _save_case(typed_crystal, tmp_path, *, edge_backend=False, policies=True):
    values = _capture_case(typed_crystal, edge_backend=edge_backend)
    data, model, registry, samples, batch, contexts, evaluation_policies, bundle = values
    if not policies:
        bindings = {value.template_id: value for value in bundle.template_bindings}
        bundle = capture_reference_site_model_bundle(
            model=model,
            structural_artifacts={
                key: value.structural_artifact for key, value in bindings.items()
            },
            phase_specifications={
                key: value.phase_specification for key, value in bindings.items()
            },
            evaluation_policies=None,
            default_template_id=bundle.default_template_id,
            provenance={"purpose": "predictor_missing_policy"},
        )
    path = tmp_path / ("edge.pt" if edge_backend else "dense.pt")
    save_reference_site_model_bundle(path, bundle)
    return (
        data,
        model,
        registry,
        samples,
        batch,
        contexts,
        evaluation_policies,
        bundle,
        path,
    )


def _geometry_only(sample):
    return StructureSample(
        sample_id=sample.sample_id,
        positions=sample.positions.clone(),
        atomic_numbers=sample.atomic_numbers.clone(),
        cell=sample.cell.clone(),
        pbc=sample.pbc.clone(),
        origin=sample.origin.clone(),
        template_id=sample.template_id,
    )


def _direct(predictor, samples, *, solver_path, return_aux=True):
    geometry = tuple(_geometry_only(sample) for sample in samples)
    batch = collate_structure_samples(geometry, predictor.registry)
    batch = replace(batch, positions=batch.positions.requires_grad_(True))
    with torch.enable_grad():
        return evaluate_structure_batch(
            predictor.model,
            batch,
            {
                key: predictor.runtime.template_contexts[key]
                for key in sorted(set(batch.template_ids))
            },
            solver_path=solver_path,
            evaluation_policies=(
                {
                    key: predictor.runtime.evaluation_policies[key]
                    for key in sorted(set(batch.template_ids))
                }
                if solver_path == EVAL_ADAPTIVE
                else None
            ),
            compute_forces=True,
            compute_stress=True,
            return_aux=return_aux,
        )


def _assert_core_equal(left, right):
    for first, second in (
        (left.energy, right.energy),
        (left.baseline_energy, right.baseline_energy),
        (left.residual_energy, right.residual_energy),
        (left.site_energy, right.site_energy),
        (left.site_ptr, right.site_ptr),
    ):
        assert torch.equal(first, second)
    for first, second in (
        (left.forces, right.forces),
        (left.stress, right.stress),
        (left.stress_voigt, right.stress_voigt),
    ):
        torch.testing.assert_close(first, second, atol=3e-14, rtol=3e-14)
    assert left.sample_ids == right.sample_ids
    assert left.template_ids == right.template_ids


def _assert_aux_equal(left, right, *, adaptive):
    for prediction_aux, direct_aux in zip(left.diagnostics, right.auxiliary):
        assert torch.equal(prediction_aux["phase"], direct_aux["phase"])
        assert torch.equal(prediction_aux["ot"].P, direct_aux["ot"].P)
        assert torch.equal(prediction_aux["ot"].q, direct_aux["ot"].q)
        assert torch.equal(
            prediction_aux["multipoles"].equivariant_features,
            direct_aux["multipoles"].equivariant_features,
        )
        assert torch.equal(
            prediction_aux["multipoles"].raw_probability_state,
            direct_aux["multipoles"].raw_probability_state,
        )
        if adaptive:
            assert (
                prediction_aux["evaluation_diagnostics"].selected_grouped_index
                == direct_aux["evaluation_diagnostics"].selected_grouped_index
            )


@pytest.mark.parametrize("solver_path", [TRAIN_FIXED, EVAL_ADAPTIVE])
def test_bundle_runtime_predictor_exact_parity_and_outer_no_grad(
    typed_crystal, tmp_path, solver_path
):
    *_, samples, _, _, _, _, path = _save_case(typed_crystal, tmp_path)
    predictor = load_reference_site_predictor(path)
    direct = _direct(predictor, samples, solver_path=solver_path)
    with torch.no_grad():
        predicted = predictor.predict_samples(
            samples,
            solver_path=solver_path,
            compute_forces=True,
            compute_stress=True,
            return_aux=True,
        )
    _assert_core_equal(predicted, direct)
    _assert_aux_equal(predicted, direct, adaptive=solver_path == EVAL_ADAPTIVE)
    _assert_detached_tree(predicted.diagnostics)
    assert torch.equal(predicted.atom_ptr, collate_structure_samples(samples, predictor.registry).atom_ptr)
    for tensor in (
        predicted.energy,
        predicted.site_energy,
        predicted.forces,
        predicted.stress,
        predicted.stress_voigt,
    ):
        assert not tensor.requires_grad and tensor.grad_fn is None


def test_single_grouped_permutation_split_batch_and_label_independence(
    typed_crystal, tmp_path
):
    *_, samples, batch, _, _, _, path = _save_case(typed_crystal, tmp_path)
    predictor = load_reference_site_predictor(path)
    full = predictor.predict_samples(samples, compute_forces=True)
    individual = tuple(
        predictor.predict_sample(sample, compute_forces=True) for sample in samples
    )
    assert torch.equal(full.energy, torch.stack([value.energy for value in individual]))
    assert torch.equal(full.forces, torch.cat([value.forces for value in individual]))
    assert torch.equal(
        full.site_energy, torch.cat([value.site_energy for value in individual])
    )

    order = (2, 0, 1)
    permuted = predictor.predict_samples(tuple(samples[index] for index in order))
    assert permuted.sample_ids == tuple(samples[index].sample_id for index in order)
    assert torch.equal(permuted.energy, full.energy[list(order)])
    split = predictor.predict_samples(samples[:2])
    tail = predictor.predict_sample(samples[2])
    assert torch.equal(split.energy, full.energy[:2])
    assert torch.equal(tail.energy, full.energy[2])

    from_batch = predictor.predict_batch(batch)
    label_free = predictor.predict_samples(tuple(_geometry_only(value) for value in samples))
    assert torch.equal(from_batch.energy, label_free.energy)
    assert torch.equal(from_batch.site_energy, label_free.site_energy)

    configured = ReferenceSitePredictor(
        predictor.runtime,
        config=PredictorConfig(compute_forces=True, compute_stress=True),
    )
    configured_output = configured.predict_sample(samples[0])
    assert configured_output.forces is not None
    assert configured_output.stress is not None


def test_input_model_gradient_rng_and_bundle_runtime_ownership(
    typed_crystal, tmp_path
):
    *_, samples, _, _, _, bundle, path = _save_case(typed_crystal, tmp_path)
    predictor = load_reference_site_predictor(path)
    model = predictor.model
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, 0.125)
    gradient = parameter.grad
    gradient_value = gradient.clone()
    parameter_ids = tuple(id(value) for value in model.parameters())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    input_state = [
        (value.sample_id, value.template_id, value.positions.clone(), value.cell.clone())
        for value in samples
    ]
    bundle_state = {
        key: value.clone() for key, value in bundle.model_state.items()
    }
    rng = torch.get_rng_state().clone()
    predictor.predict_samples(samples, compute_forces=True, compute_stress=True)
    assert not model.training
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert parameter.grad is gradient and torch.equal(parameter.grad, gradient_value)
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in state.items():
        assert torch.equal(model.state_dict()[key], value)
    for key, value in bundle_state.items():
        assert torch.equal(bundle.model_state[key], value)
    for sample, before in zip(samples, input_state):
        assert (sample.sample_id, sample.template_id) == before[:2]
        assert torch.equal(sample.positions, before[2])
        assert torch.equal(sample.cell, before[3])


def test_candidate_state_fresh_reuse_and_cell_rebuild(typed_crystal, tmp_path):
    *_, samples, _, _, _, _, path = _save_case(
        typed_crystal, tmp_path, edge_backend=True
    )
    predictor = load_reference_site_predictor(path)
    initial = predictor.predict_samples(
        samples, return_candidate_neighbor_states=True, return_aux=True
    )
    assert tuple(initial.candidate_neighbor_states) == initial.sample_ids
    assert all(
        value.reason_code == "INITIAL_BUILD"
        for value in initial.candidate_reuse_decisions.values()
    )
    fingerprints = {
        key: value.integrity_fingerprint
        for key, value in initial.candidate_neighbor_states.items()
    }
    reused = predictor.predict_samples(
        samples,
        candidate_neighbor_states=initial.candidate_neighbor_states,
        return_aux=True,
    )
    fresh = predictor.predict_samples(samples, return_aux=True)
    torch.testing.assert_close(reused.energy, fresh.energy, atol=4e-13, rtol=4e-13)
    assert all(
        value.reason_code == "REUSED"
        for value in reused.candidate_reuse_decisions.values()
    )
    assert all(
        initial.candidate_neighbor_states[key].integrity_fingerprint == value
        for key, value in fingerprints.items()
    )

    changed = list(samples)
    changed[1] = replace(changed[1], cell=changed[1].cell.clone())
    changed[1].cell[0, 0] *= 1.0001
    rebuilt = predictor.predict_samples(
        tuple(changed), candidate_neighbor_states=reused.candidate_neighbor_states
    )
    assert rebuilt.candidate_reuse_decisions[samples[1].sample_id].reason_code == "CELL_CHANGED"
    assert rebuilt.candidate_neighbor_states is not reused.candidate_neighbor_states

    unused = dict(reused.candidate_neighbor_states)
    unused["unused:state"] = object()
    ignored = predictor.predict_samples(samples, candidate_neighbor_states=unused)
    assert "unused:state" not in ignored.candidate_neighbor_states

    sample_id = samples[0].sample_id
    corrupted = replace(
        reused.candidate_neighbor_states[sample_id],
        template_fingerprint="1" * 64,
        integrity_fingerprint=None,
    )
    mapping = dict(reused.candidate_neighbor_states)
    mapping[sample_id] = corrupted
    with pytest.raises(PredictorError) as caught:
        predictor.predict_samples(samples, candidate_neighbor_states=mapping)
    assert caught.value.reason_code == "TEMPLATE_MISMATCH"
    assert caught.value.sample_id == sample_id


def test_structured_preflight_failures_and_inference_mode(
    typed_crystal, tmp_path
):
    *_, samples, batch, _, _, _, path = _save_case(typed_crystal, tmp_path)
    predictor = load_reference_site_predictor(path)
    unknown = replace(samples[0], template_id="unknown-template")
    with pytest.raises(PredictorError) as caught:
        predictor.predict_sample(unknown)
    assert caught.value.reason_code == "UNKNOWN_TEMPLATE"
    assert caught.value.sample_id == unknown.sample_id
    assert caught.value.stage == "template_lookup"

    bad_species = samples[0].atomic_numbers.clone()
    bad_species[0] = 1
    unsupported = replace(samples[0], atomic_numbers=bad_species)
    with pytest.raises(PredictorError) as caught:
        predictor.predict_sample(unsupported)
    assert caught.value.reason_code == "UNSUPPORTED_SPECIES"

    expanded_base = _geometry_only(samples[2])
    too_many = replace(
        expanded_base,
        positions=torch.cat(
            (expanded_base.positions, expanded_base.positions[:1] + 0.1), dim=0
        ),
        atomic_numbers=torch.cat(
            (expanded_base.atomic_numbers, expanded_base.atomic_numbers[:1]), dim=0
        ),
    )
    with pytest.raises(PredictorError) as caught:
        predictor.predict_sample(too_many)
    assert caught.value.reason_code == "INVALID_N_GT_M"

    bad_fingerprints = list(batch.template_fingerprints)
    for index, template_id in enumerate(batch.template_ids):
        if template_id == batch.template_ids[0]:
            bad_fingerprints[index] = "0" * 64
    mismatched = replace(batch, template_fingerprints=tuple(bad_fingerprints))
    with pytest.raises(PredictorError) as caught:
        predictor.predict_batch(mismatched)
    assert caught.value.reason_code == "TEMPLATE_FINGERPRINT_MISMATCH"

    with torch.inference_mode(), pytest.raises(PredictorError) as caught:
        predictor.predict_sample(samples[0], compute_forces=True)
    assert caught.value.reason_code == "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED"
    assert caught.value.solver_path == TRAIN_FIXED

    context = predictor.runtime.template_contexts[samples[0].template_id]
    context.phase_mode_weights[0] += 0.25
    with pytest.raises(PredictorError) as caught:
        predictor.predict_sample(samples[0])
    assert caught.value.reason_code == "TEMPLATE_CONTEXT_FINGERPRINT_MISMATCH"


def test_missing_adaptive_policy_and_nonfinite_output(typed_crystal, tmp_path, monkeypatch):
    *_, samples, _, _, _, _, path = _save_case(
        typed_crystal, tmp_path, policies=False
    )
    predictor = load_reference_site_predictor(path)
    with pytest.raises(PredictorError) as caught:
        predictor.predict_samples(samples, solver_path=EVAL_ADAPTIVE)
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    assert caught.value.sample_id in {sample.sample_id for sample in samples}

    original = predictor_module.evaluate_structure_batch

    def nonfinite(*args, **kwargs):
        output = original(*args, **kwargs)
        return replace(output, energy=torch.full_like(output.energy, float("nan")))

    monkeypatch.setattr(predictor_module, "evaluate_structure_batch", nonfinite)
    with pytest.raises(PredictorError) as caught:
        predictor.predict_samples(samples)
    assert caught.value.reason_code == "NONFINITE_PREDICTION"
    assert caught.value.stage == "output_validation"


def test_policy_content_fingerprint_mutation(typed_crystal, tmp_path):
    *_, samples, _, _, _, _, path = _save_case(typed_crystal, tmp_path)
    predictor = load_reference_site_predictor(path)
    policy = predictor.runtime.evaluation_policies[samples[0].template_id]
    policy.candidate_offsets[0, 0] += 0.125
    with pytest.raises(PredictorError) as caught:
        predictor.predict_samples(samples, solver_path=EVAL_ADAPTIVE)
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    assert caught.value.stage == "policy_preflight"


def test_adaptive_energy_fallback_and_derivative_rejection(
    typed_crystal, tmp_path, monkeypatch
):
    *_, samples, _, _, _, _, path = _save_case(typed_crystal, tmp_path)
    predictor = load_reference_site_predictor(path)
    import refsite_mlip.models.potential as potential_module

    original = potential_module.solve_atom_vacancy_ot

    def fallback(*args, **kwargs):
        return replace(original(*args, **kwargs), fallback_used=True)

    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", fallback)
    energy = predictor.predict_samples(
        samples, solver_path=EVAL_ADAPTIVE, return_aux=True
    )
    assert all(value["ot"].fallback_used for value in energy.diagnostics)
    with pytest.raises(PredictorError) as caught:
        predictor.predict_samples(
            samples, solver_path=EVAL_ADAPTIVE, compute_forces=True
        )
    assert caught.value.reason_code == "DERIVATIVE_FALLBACK_UNSUPPORTED"
    assert caught.value.sample_id == "alpha-pristine"
    assert caught.value.template_id == "alpha"


def test_weights_only_single_load_and_no_builder_path(typed_crystal, tmp_path, monkeypatch):
    *_, path = _save_case(typed_crystal, tmp_path)
    original = predictor_module.load_reference_site_model_bundle
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(predictor_module, "load_reference_site_model_bundle", counted)
    import refsite_mlip.models.bundle as bundle_module

    original_torch_load = bundle_module.torch.load
    load_options = []

    def checked_torch_load(*args, **kwargs):
        load_options.append(dict(kwargs))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(bundle_module.torch, "load", checked_torch_load)
    import refsite_mlip.data.reference_builder as reference_builder
    import refsite_mlip.graph as graph

    monkeypatch.setattr(
        reference_builder,
        "build_reference_template_from_atoms",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("builder called")),
    )
    monkeypatch.setattr(
        reference_builder,
        "build_reference_template_from_poscar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("POSCAR reader called")),
    )
    monkeypatch.setattr(
        graph,
        "build_reference_graph_topology",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("graph builder called")),
    )
    import refsite_mlip.phase as phase

    monkeypatch.setattr(
        phase,
        "find_typed_stabilizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stabilizer builder called")),
    )
    safe_globals = list(torch.serialization.get_safe_globals())
    predictor = load_reference_site_predictor(path)
    assert isinstance(predictor, ReferenceSitePredictor)
    assert len(calls) == 1
    assert len(load_options) == 1 and load_options[0]["weights_only"] is True
    assert torch.serialization.get_safe_globals() == safe_globals


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_predictor_fixed_adaptive_dtype_device(typed_crystal, tmp_path, dtype):
    *_, samples, _, _, _, _, path = _save_case(
        typed_crystal, tmp_path, edge_backend=True
    )
    predictor = load_reference_site_predictor(path, device="cuda", dtype=dtype)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    for solver_path in (TRAIN_FIXED, EVAL_ADAPTIVE):
        output = predictor.predict_samples(
            samples,
            solver_path=solver_path,
            compute_forces=True,
            compute_stress=True,
            return_candidate_neighbor_states=True,
        )
        assert output.device.type == "cuda" and output.dtype == dtype
        assert output.atom_ptr.dtype == torch.long
        assert torch.all(torch.isfinite(output.energy))
        assert torch.all(torch.isfinite(output.forces))
        assert torch.all(torch.isfinite(output.stress))
        assert all(
            state.device.type == "cuda" and state.dtype == dtype
            for state in output.candidate_neighbor_states.values()
        )
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_predictor_explicit_cpu_output(typed_crystal, tmp_path):
    *_, samples, _, _, _, _, path = _save_case(
        typed_crystal, tmp_path, edge_backend=True
    )
    predictor = load_reference_site_predictor(
        path,
        device="cuda",
        dtype=torch.float32,
        config=PredictorConfig(output_device="cpu"),
    )
    output = predictor.predict_samples(
        samples, return_candidate_neighbor_states=True
    )
    assert output.device.type == "cpu" and output.dtype == torch.float32
    assert all(
        state.device.type == "cpu"
        for state in output.candidate_neighbor_states.values()
    )
    reused = predictor.predict_samples(
        samples, candidate_neighbor_states=output.candidate_neighbor_states
    )
    assert all(
        decision.reason_code == "STATE_DEVICE_MATERIALIZATION"
        for decision in reused.candidate_reuse_decisions.values()
    )
