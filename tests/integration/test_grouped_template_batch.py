from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from refsite_mlip.data import (
    ReferenceTemplate,
    StructureSample,
    TemplateRegistry,
    collate_structure_samples,
)
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import (
    PotentialConfig,
    ReferenceSitePotential,
    TemplateExecutionContext,
    evaluate_structure_batch,
)
from refsite_mlip.phase.stabilizer import find_typed_stabilizer
from refsite_mlip.training import (
    LossConfig,
    OptimizerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_optimizer,
    compute_potential_loss,
    run_training_epoch,
    run_validation_epoch,
    train_step,
    validation_step,
)
from refsite_mlip.transport import TRAIN_FIXED


AVG_NUM_NEIGHBORS = 6.0


def _cast_data(data, dtype):
    return {
        key: value.to(dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in data.items()
    }


def _template(data, template_id, site_count, mode_count, mode_scale=1.0):
    sites = data["sites"][:site_count]
    site_types = data["site_types"][:site_count]
    topology = build_reference_graph_topology(
        sites,
        site_types,
        data["cell"],
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    return ReferenceTemplate.snapshot(
        template_id,
        topology,
        data["modes"][:mode_count],
        data["mode_weights"][:mode_count] * mode_scale,
        data["site_weights"][:site_count],
        data["channel_weights"],
        find_typed_stabilizer(sites, site_types),
        (6, 41),
    )


def _configuration(dtype):
    tolerance = 1.0e-6 if dtype == torch.float32 else 1.0e-7
    feature = ProbabilityMultipoleConfig(
        (6, 41),
        2,
        2,
        1.0,
        3.0,
        tolerance,
        site_type_vocabulary=(0, 1),
    )
    irreps = "2x0e+4x0e+4x1o+4x2e"
    higher = HigherBodyConfig(
        irreps,
        2,
        2,
        2,
        1,
        2,
        3,
        (4,),
        AVG_NUM_NEIGHBORS,
        3.0,
        1.0,
    )
    return PotentialConfig((6, 41), 1, feature, higher, 8, 1.0)


def _numbers(data, count):
    return torch.where(
        data["site_types"][:count] == 0,
        torch.tensor(6, dtype=torch.long),
        torch.tensor(41, dtype=torch.long),
    )


def _sample(data, sample_id, template_id, count, *, labeled):
    return StructureSample(
        sample_id=sample_id,
        positions=data["positions"][:count].clone(),
        atomic_numbers=_numbers(data, count),
        cell=data["cell"].clone(),
        pbc=torch.ones(3, dtype=torch.bool),
        origin=data["origin"].clone(),
        template_id=template_id,
        energy=(
            torch.tensor(float(count), dtype=data["positions"].dtype)
            if labeled
            else None
        ),
    )


def _case(typed_crystal, *, dtype=torch.float64, device="cpu"):
    data = _cast_data(typed_crystal, dtype)
    alpha = _template(data, "alpha", site_count=4, mode_count=4)
    zeta = _template(data, "zeta", site_count=6, mode_count=5)
    registry = TemplateRegistry()
    registry.add(alpha)
    registry.add(zeta)
    samples = (
        _sample(data, "zeta-vacancy", "zeta", 5, labeled=True),
        _sample(data, "alpha-pristine", "alpha", 4, labeled=False),
        _sample(data, "zeta-pristine", "zeta", 6, labeled=True),
    )
    if torch.device(device).type != "cpu":
        samples = tuple(sample.to(device=device, dtype=dtype) for sample in samples)
    batch = collate_structure_samples(samples, registry)
    model = ReferenceSitePotential(
        _configuration(dtype),
        zeta.topology,
        zeta.phase_modes,
        zeta.phase_mode_weights,
        torch.eye(2, dtype=dtype),
        zeta.site_alignment_weights,
        zeta.phase_channel_weights,
        (-1.0, 2.0),
    ).to(device=device, dtype=dtype)
    contexts = {
        template_id: TemplateExecutionContext.from_reference_template(
            registry.resolve(template_id),
            avg_num_neighbors=AVG_NUM_NEIGHBORS,
        )
        for template_id in ("alpha", "zeta")
    }
    return data, model, registry, samples, batch, contexts


def _individual(model, batch, contexts, **kwargs):
    outputs = []
    for index in range(batch.num_structures):
        atom_slice = slice(int(batch.atom_ptr[index]), int(batch.atom_ptr[index + 1]))
        outputs.append(
            model(
                batch.positions[atom_slice],
                batch.atomic_numbers[atom_slice],
                batch.cells[index],
                batch.origins[index],
                solver_path=TRAIN_FIXED,
                template_context=contexts[batch.template_ids[index]],
                **kwargs,
            )
        )
    return tuple(outputs)


def _assert_grouped_matches_individual(grouped, individual):
    assert torch.equal(grouped.energy, torch.stack([out.energy for out in individual]))
    assert torch.equal(
        grouped.baseline_energy,
        torch.stack([out.baseline_energy for out in individual]),
    )
    assert torch.equal(
        grouped.residual_energy,
        torch.stack([out.residual_energy for out in individual]),
    )
    assert torch.equal(
        grouped.site_energy,
        torch.cat([out.site_energy for out in individual]),
    )
    if grouped.forces is not None:
        torch.testing.assert_close(
            grouped.forces,
            torch.cat([out.forces for out in individual]),
            atol=2.0e-14,
            rtol=2.0e-14,
        )
    if grouped.stress is not None:
        torch.testing.assert_close(
            grouped.stress,
            torch.stack([out.stress for out in individual]),
            atol=2.0e-14,
            rtol=2.0e-14,
        )
        torch.testing.assert_close(
            grouped.stress_voigt,
            torch.stack([out.stress_voigt for out in individual]),
            atol=2.0e-14,
            rtol=2.0e-14,
        )


def test_grouped_outputs_restore_structure_atom_site_and_aux_order(typed_crystal):
    _, model, _, _, batch, contexts = _case(typed_crystal)
    batch.positions.requires_grad_(True)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    state_keys = tuple(model.state_dict())

    grouped = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    individual = _individual(
        model,
        batch,
        contexts,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    _assert_grouped_matches_individual(grouped, individual)

    assert [group.template_id for group in batch.template_groups] == ["alpha", "zeta"]
    assert grouped.sample_ids == batch.sample_ids
    assert grouped.template_ids == ("zeta", "alpha", "zeta")
    assert grouped.site_ptr.tolist() == [0, 6, 10, 16]
    assert grouped.site_batch.tolist() == [0] * 6 + [1] * 4 + [2] * 6
    assert grouped["energy"] is grouped.energy
    assert grouped.auxiliary is not None
    for index, (grouped_aux, single) in enumerate(
        zip(grouped.auxiliary, individual)
    ):
        assert grouped_aux is not None and single.auxiliary is not None
        assert torch.equal(grouped_aux["phase"], single.auxiliary["phase"])
        assert torch.equal(grouped_aux["ot"].P, single.auxiliary["ot"].P)
        assert torch.equal(grouped_aux["ot"].q, single.auxiliary["ot"].q)
        ot = grouped_aux["ot"]
        atom_count = int(batch.atom_ptr[index + 1] - batch.atom_ptr[index])
        site_count = int(grouped.site_ptr[index + 1] - grouped.site_ptr[index])
        torch.testing.assert_close(
            ot.gamma.sum(dim=1), ot.gamma.new_ones(site_count)
        )
        torch.testing.assert_close(
            ot.P.sum(dim=0), ot.P.new_ones(atom_count)
        )
        torch.testing.assert_close(
            ot.q.sum(), ot.q.new_tensor(float(site_count - atom_count))
        )
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert tuple(model.state_dict()) == state_keys
    assert batch.energy_mask.tolist() == [True, False, True]


def test_structure_permutation_split_and_perturbation_independence(typed_crystal):
    _, model, registry, samples, batch, contexts = _case(typed_crystal)
    batch.positions.requires_grad_(True)
    full = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
    )

    order = (2, 0, 1)
    permuted_batch = collate_structure_samples(
        tuple(samples[index] for index in order), registry
    )
    permuted = evaluate_structure_batch(
        model, permuted_batch, contexts, solver_path=TRAIN_FIXED
    )
    assert torch.equal(permuted.energy, full.energy[list(order)])

    split = [
        evaluate_structure_batch(
            model,
            collate_structure_samples((sample,), registry),
            contexts,
            solver_path=TRAIN_FIXED,
        )
        for sample in samples
    ]
    assert torch.equal(full.energy, torch.cat([output.energy for output in split]))
    assert torch.equal(
        full.site_energy, torch.cat([output.site_energy for output in split])
    )

    moved_first = replace(
        samples[0], positions=samples[0].positions + torch.tensor(
            [0.017, -0.013, 0.009], dtype=samples[0].positions.dtype
        )
    )
    perturbed_batch = collate_structure_samples(
        (moved_first, samples[1], samples[2]), registry
    )
    perturbed_batch.positions.requires_grad_(True)
    perturbed = evaluate_structure_batch(
        model,
        perturbed_batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
    )
    assert torch.equal(perturbed.energy[1:], full.energy[1:])
    first_atom_count = int(batch.atom_ptr[1])
    assert torch.equal(
        perturbed.forces[first_atom_count:], full.forces[first_atom_count:]
    )


def _parameter_gradients(model, batch, contexts):
    output = evaluate_structure_batch(
        model, batch, contexts, solver_path=TRAIN_FIXED
    )
    return torch.autograd.grad(
        output.energy.sum(), tuple(model.parameters()), allow_unused=True
    )


def test_grouped_shared_parameter_gradient_equals_individual_sum(typed_crystal):
    _, model, registry, samples, batch, contexts = _case(typed_crystal)
    grouped = _parameter_gradients(model, batch, contexts)
    singles = [
        _parameter_gradients(
            model,
            collate_structure_samples((sample,), registry),
            contexts,
        )
        for sample in samples
    ]
    for total, *parts in zip(grouped, *singles):
        if total is None:
            assert all(part is None for part in parts)
            continue
        expected = torch.zeros_like(total)
        for part in parts:
            if part is not None:
                expected = expected + part
        torch.testing.assert_close(total, expected, atol=3.0e-12, rtol=3.0e-12)


def test_mixed_template_force_loss_backward_is_finite(typed_crystal):
    _, model, _, _, batch, contexts = _case(typed_crystal)
    batch.positions.requires_grad_(True)
    output = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        create_graph=True,
    )
    selected = (
        model.readout.mlp[-1].weight,
        model.layers[0].edge.radial_head.network[0].weight,
        model.central.embedding.weight,
    )
    gradients = torch.autograd.grad(output.forces.square().sum(), selected)
    assert all(bool(torch.all(torch.isfinite(value))) for value in gradients)
    assert all(bool(torch.any(value != 0)) for value in gradients)


def test_grouped_energy_force_stress_masked_loss_backward(typed_crystal):
    _, model, _, _, batch, contexts = _case(typed_crystal)
    batch.positions.requires_grad_(True)
    prediction = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
    )
    target_batch = replace(
        batch,
        forces=torch.zeros_like(batch.forces),
        force_mask=torch.ones_like(batch.force_mask),
        stress=torch.zeros_like(batch.stress),
        stress_mask=torch.ones_like(batch.stress_mask),
        force_present=torch.ones_like(batch.force_present),
        stress_present=torch.ones_like(batch.stress_present),
        force_mask_provided=torch.zeros_like(batch.force_mask_provided),
        stress_mask_provided=torch.zeros_like(batch.stress_mask_provided),
    )
    loss = compute_potential_loss(
        prediction,
        target_batch,
        LossConfig(energy_weight=1.0, force_weight=1.0, stress_weight=1.0),
    )
    assert int(loss.energy.valid_count) == 2
    assert int(loss.force.valid_count) == 3 * batch.num_atoms
    assert int(loss.stress.valid_count) == 6 * batch.num_structures
    assert bool(torch.isfinite(loss.total))

    selected = (
        model.readout.mlp[-1].weight,
        model.layers[0].edge.radial_head.network[0].weight,
        model.central.embedding.weight,
    )
    gradients = torch.autograd.grad(loss.total, selected)
    assert all(bool(torch.all(torch.isfinite(value))) for value in gradients)
    assert all(bool(torch.any(value != 0)) for value in gradients)


def test_actual_energy_train_step_is_deterministic_and_updates_once(typed_crystal):
    _, model, _, _, batch, contexts = _case(typed_crystal)
    clone = copy.deepcopy(model)
    first_optimizer = build_optimizer(
        model, OptimizerConfig(learning_rate=2.0e-4, weight_decay=0.0)
    )
    second_optimizer = build_optimizer(
        clone, OptimizerConfig(learning_rate=2.0e-4, weight_decay=0.0)
    )
    baseline = model.atomic_baseline.clone()
    original_positions = batch.positions.clone()
    first = train_step(
        model, first_optimizer, batch, contexts, LossConfig(), TrainStepConfig()
    )
    second = train_step(
        clone, second_optimizer, batch, contexts, LossConfig(), TrainStepConfig()
    )
    assert first == second and first.energy.valid_count == 2
    assert not first.need_forces and not first.need_stress
    assert torch.equal(batch.positions, original_positions) and not batch.positions.requires_grad
    assert torch.equal(model.atomic_baseline, baseline)
    for left, right in zip(model.state_dict().values(), clone.state_dict().values()):
        assert torch.equal(left, right)
    assert any(
        float(state["step"]) == 1.0 for state in first_optimizer.state.values()
    )
    assert all(
        float(state["step"]) == 1.0 for state in first_optimizer.state.values()
    )


def test_actual_force_only_step_prepares_leaf_and_mixed_template_gradients(typed_crystal):
    _, model, _, _, batch, contexts = _case(typed_crystal)
    target = replace(
        batch,
        energy=torch.zeros_like(batch.energy),
        energy_mask=torch.zeros_like(batch.energy_mask),
        forces=torch.zeros_like(batch.forces),
        force_mask=torch.ones_like(batch.force_mask),
        force_present=torch.ones_like(batch.force_present),
        force_mask_provided=torch.zeros_like(batch.force_mask_provided),
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    result = train_step(
        model,
        build_optimizer(model, OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0)),
        target,
        contexts,
        LossConfig(energy_weight=0.0, force_weight=1.0),
        TrainStepConfig(),
    )
    assert result.need_forces and not result.need_stress
    assert result.force.valid_count == 3 * batch.num_atoms
    assert result.number_of_parameters_with_grad > 0
    assert result.pre_clip_grad_norm > 0.0
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
    )
    assert not target.positions.requires_grad


def test_actual_energy_force_stress_step_with_clipping(typed_crystal):
    _, model, _, _, batch, contexts = _case(typed_crystal)
    target = replace(
        batch,
        forces=torch.zeros_like(batch.forces),
        force_mask=torch.ones_like(batch.force_mask),
        stress=torch.zeros_like(batch.stress),
        stress_mask=torch.ones_like(batch.stress_mask),
        force_present=torch.ones_like(batch.force_present),
        stress_present=torch.ones_like(batch.stress_present),
        force_mask_provided=torch.zeros_like(batch.force_mask_provided),
        stress_mask_provided=torch.zeros_like(batch.stress_mask_provided),
    )
    result = train_step(
        model,
        build_optimizer(model, OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0)),
        target,
        contexts,
        LossConfig(energy_weight=1.0, force_weight=0.2, stress_weight=0.1),
        TrainStepConfig(gradient_clip_norm=0.5),
    )
    assert result.need_forces and result.need_stress
    assert result.energy.valid_count == 2
    assert result.force.valid_count == 3 * batch.num_atoms
    assert result.stress.valid_count == 6 * batch.num_structures
    assert result.number_of_parameters_with_grad > 0
    assert result.post_clip_grad_norm <= 0.5 + 1.0e-10


def test_actual_mixed_validation_matches_direct_loss_inside_no_grad(typed_crystal):
    _, model, _, _, batch, contexts = _case(typed_crystal)
    target = replace(
        batch,
        forces=torch.zeros_like(batch.forces),
        force_mask=torch.ones_like(batch.force_mask),
        stress=torch.zeros_like(batch.stress),
        stress_mask=torch.ones_like(batch.stress_mask),
        force_present=torch.ones_like(batch.force_present),
        stress_present=torch.ones_like(batch.stress_present),
        force_mask_provided=torch.zeros_like(batch.force_mask_provided),
        stress_mask_provided=torch.zeros_like(batch.stress_mask_provided),
    )
    direct_batch = replace(
        target,
        positions=target.positions.detach().clone().requires_grad_(True),
    )
    model.eval()
    direct_prediction = evaluate_structure_batch(
        model,
        direct_batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        create_graph=False,
    )
    loss_config = LossConfig(energy_weight=1.0, force_weight=0.2, stress_weight=0.1)
    direct_loss = compute_potential_loss(direct_prediction, direct_batch, loss_config)
    model.train()
    with torch.no_grad():
        first = validation_step(
            model, target, contexts, loss_config, ValidationStepConfig()
        )
    second = validation_step(
        model, target, contexts, loss_config, ValidationStepConfig()
    )
    assert first == second and model.training
    assert first.total_loss == float(direct_loss.total.detach())
    assert first.energy.numerator == float(direct_loss.energy.numerator.detach())
    assert first.force.numerator == float(direct_loss.force.numerator.detach())
    assert first.stress.numerator == float(direct_loss.stress.numerator.detach())
    assert first.need_forces and first.need_stress and first.has_supervision


def test_actual_two_batch_training_epoch_determinism_and_split_validation(typed_crystal):
    _, model, registry, samples, batch, contexts = _case(typed_crystal)
    clone = copy.deepcopy(model)
    training_batches = (
        collate_structure_samples((samples[0],), registry),
        collate_structure_samples((samples[2],), registry),
    )
    first_optimizer = build_optimizer(
        model, OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0)
    )
    second_optimizer = build_optimizer(
        clone, OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0)
    )
    first = run_training_epoch(
        model,
        first_optimizer,
        training_batches,
        contexts,
        LossConfig(),
        TrainStepConfig(),
        epoch_index=2,
        global_step_start=5,
    )
    second = run_training_epoch(
        clone,
        second_optimizer,
        training_batches,
        contexts,
        LossConfig(),
        TrainStepConfig(),
        epoch_index=2,
        global_step_start=5,
    )
    assert first == second
    assert first.global_step_end == 7 and first.successful_optimizer_steps == 2
    assert first.ordered_batch_sample_ids == (("zeta-vacancy",), ("zeta-pristine",))
    for left, right in zip(model.state_dict().values(), clone.state_dict().values()):
        assert torch.equal(left, right)
    for left_parameter, right_parameter in zip(model.parameters(), clone.parameters()):
        left_state = first_optimizer.state.get(left_parameter, {})
        right_state = second_optimizer.state.get(right_parameter, {})
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            if isinstance(left_state[key], torch.Tensor):
                assert torch.equal(left_state[key], right_state[key])
            else:
                assert left_state[key] == right_state[key]

    validation_state = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    validation_gradients = tuple(
        (
            parameter.grad,
            None if parameter.grad is None else parameter.grad.clone(),
        )
        for parameter in model.parameters()
    )
    optimizer_steps = tuple(
        float(state["step"])
        for state in first_optimizer.state.values()
        if "step" in state
    )
    validation_rng = torch.random.get_rng_state().clone()
    full = run_validation_epoch(
        model,
        (batch,),
        contexts,
        LossConfig(),
        ValidationStepConfig(),
        epoch_index=2,
        global_step=7,
    )
    full_repeat = run_validation_epoch(
        model,
        (batch,),
        contexts,
        LossConfig(),
        ValidationStepConfig(),
        epoch_index=2,
        global_step=7,
    )
    assert full_repeat == full
    split_batches = tuple(
        collate_structure_samples((sample,), registry) for sample in samples
    )
    split = run_validation_epoch(
        model,
        split_batches,
        contexts,
        LossConfig(),
        ValidationStepConfig(),
        epoch_index=2,
        global_step=7,
    )
    assert full.energy == split.energy and full.total_loss == split.total_loss
    assert full.number_of_structures == split.number_of_structures == 3
    assert split.number_of_supervised_batches == 2
    assert torch.equal(torch.random.get_rng_state(), validation_rng)
    for key, value in model.state_dict().items():
        assert torch.equal(value, validation_state[key])
    for parameter, (identity, value) in zip(model.parameters(), validation_gradients):
        assert parameter.grad is identity
        if value is not None:
            assert torch.equal(parameter.grad, value)
    assert optimizer_steps == tuple(
        float(state["step"])
        for state in first_optimizer.state.values()
        if "step" in state
    )


def test_context_resolution_fail_fast_and_extra_context_is_allowed(typed_crystal):
    data, model, _, _, batch, contexts = _case(typed_crystal)
    with pytest.raises(KeyError, match="missing TemplateExecutionContext"):
        evaluate_structure_batch(
            model, batch, {"zeta": contexts["zeta"]}, solver_path=TRAIN_FIXED
        )

    different_alpha = _template(
        data, "alpha", site_count=4, mode_count=4, mode_scale=1.01
    )
    mismatched = dict(contexts)
    mismatched["alpha"] = TemplateExecutionContext.from_reference_template(
        different_alpha, avg_num_neighbors=AVG_NUM_NEIGHBORS
    )
    with pytest.raises(ValueError, match="fingerprints do not match"):
        evaluate_structure_batch(
            model, batch, mismatched, solver_path=TRAIN_FIXED
        )

    extra = dict(contexts)
    extra["unused"] = contexts["alpha"]
    output = evaluate_structure_batch(
        model, batch, extra, solver_path=TRAIN_FIXED
    )
    assert output.energy.shape == (3,)

    with pytest.raises(ValueError, match="must require gradients"):
        evaluate_structure_batch(
            model,
            batch,
            contexts,
            solver_path=TRAIN_FIXED,
            compute_forces=True,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_cuda_smoke_when_available(typed_crystal, dtype):
    if not torch.cuda.is_available():
        return
    _, model, _, _, batch, contexts = _case(
        typed_crystal, dtype=dtype, device="cuda"
    )
    batch.positions.requires_grad_(True)
    output = evaluate_structure_batch(
        model,
        batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
    )
    assert output.energy.dtype == dtype and output.energy.device.type == "cuda"
    assert output.forces.dtype == dtype and output.forces.device.type == "cuda"
    assert output.stress.dtype == dtype and output.stress.device.type == "cuda"
    assert bool(torch.all(torch.isfinite(output.energy)))
    assert bool(torch.all(torch.isfinite(output.forces)))
    assert bool(torch.all(torch.isfinite(output.stress)))
    target = replace(
        batch,
        positions=batch.positions.detach(),
        forces=torch.zeros_like(batch.forces),
        force_mask=torch.ones_like(batch.force_mask),
        stress=torch.zeros_like(batch.stress),
        stress_mask=torch.ones_like(batch.stress_mask),
        force_present=torch.ones_like(batch.force_present),
        stress_present=torch.ones_like(batch.stress_present),
        force_mask_provided=torch.zeros_like(batch.force_mask_provided),
        stress_mask_provided=torch.zeros_like(batch.stress_mask_provided),
    )
    result = train_step(
        model,
        build_optimizer(model, OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0)),
        target,
        contexts,
        LossConfig(energy_weight=1.0, force_weight=0.1, stress_weight=0.1),
        TrainStepConfig(),
    )
    assert result.need_forces and result.need_stress
    assert result.number_of_parameters_with_grad > 0
    assert result.pre_clip_grad_norm > 0.0
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    with torch.no_grad():
        validation = validation_step(
            model,
            target,
            contexts,
            LossConfig(energy_weight=1.0, force_weight=0.1, stress_weight=0.1),
            ValidationStepConfig(),
        )
    assert validation.need_forces and validation.need_stress
    assert validation.has_supervision
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert all(
        torch.equal(after, before)
        for after, before in zip(torch.cuda.get_rng_state_all(), cuda_rng)
    )
    validation_epoch = run_validation_epoch(
        model,
        (target,),
        contexts,
        LossConfig(energy_weight=1.0, force_weight=0.1, stress_weight=0.1),
        ValidationStepConfig(),
        epoch_index=0,
        global_step=1,
    )
    assert validation_epoch.global_step_start == validation_epoch.global_step_end == 1
    assert validation_epoch.has_supervision
    training_epoch = run_training_epoch(
        model,
        build_optimizer(model, OptimizerConfig(learning_rate=1.0e-4, weight_decay=0.0)),
        (batch,),
        contexts,
        LossConfig(),
        TrainStepConfig(),
        epoch_index=0,
        global_step_start=1,
    )
    assert training_epoch.global_step_end == 2
    assert training_epoch.successful_optimizer_steps == 1
