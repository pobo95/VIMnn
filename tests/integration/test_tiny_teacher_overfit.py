from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import Path
import runpy

import torch

from refsite_mlip.data import collate_structure_samples
from refsite_mlip.models import evaluate_structure_batch
from refsite_mlip.training import (
    LossConfig,
    OptimizerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_optimizer,
    train_step,
    validation_step,
)
from refsite_mlip.transport import TRAIN_FIXED


def _mixed_case(typed_crystal):
    path = Path(__file__).with_name("test_grouped_template_batch.py")
    return runpy.run_path(str(path))["_case"](
        typed_crystal, dtype=torch.float64, device="cpu"
    )


def _tensor_state(value):
    return {
        name: tensor.detach().clone()
        for name, tensor in value.state_dict().items()
    }


def _assert_tensor_state_equal(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def _teacher_labels(teacher, geometry_batch, geometry_samples, registry, contexts):
    positions = geometry_batch.positions.detach().clone().requires_grad_(True)
    differentiable_batch = replace(geometry_batch, positions=positions)
    grouped = evaluate_structure_batch(
        teacher,
        differentiable_batch,
        contexts,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=False,
        create_graph=False,
        return_aux=True,
    )
    assert grouped.forces is not None
    assert bool(torch.all(torch.isfinite(grouped.energy)))
    assert bool(torch.all(torch.isfinite(grouped.forces)))

    for index, sample in enumerate(geometry_samples):
        single = collate_structure_samples((sample,), registry)
        single = replace(
            single,
            positions=single.positions.detach().clone().requires_grad_(True),
        )
        output = evaluate_structure_batch(
            teacher,
            single,
            contexts,
            solver_path=TRAIN_FIXED,
            compute_forces=True,
            compute_stress=False,
            create_graph=False,
            return_aux=True,
        )
        atom_slice = slice(
            int(geometry_batch.atom_ptr[index]),
            int(geometry_batch.atom_ptr[index + 1]),
        )
        torch.testing.assert_close(
            grouped.energy[index], output.energy[0], atol=2.0e-12, rtol=2.0e-12
        )
        torch.testing.assert_close(
            grouped.forces[atom_slice],
            output.forces,
            atol=2.0e-12,
            rtol=2.0e-12,
        )

    assert grouped.auxiliary is not None
    expected_vacancy_mass = (0.0, 1.0)
    for auxiliary, expected in zip(grouped.auxiliary, expected_vacancy_mass):
        assert auxiliary is not None
        q = auxiliary["q"]
        torch.testing.assert_close(
            q.sum(), q.new_tensor(expected), atol=2.0e-10, rtol=0.0
        )

    labeled = []
    for index, sample in enumerate(geometry_samples):
        atom_slice = slice(
            int(geometry_batch.atom_ptr[index]),
            int(geometry_batch.atom_ptr[index + 1]),
        )
        labeled.append(
            replace(
                sample,
                energy=grouped.energy[index].detach().clone(),
                forces=grouped.forces[atom_slice].detach().clone(),
            )
        )
    return collate_structure_samples(tuple(labeled), registry)


def _perturbed_student(teacher, *, seed, scale):
    student = copy.deepcopy(teacher)
    student.load_state_dict(teacher.state_dict(), strict=True)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    targets = {
        "readout": student.readout.mlp[-1].weight,
        "interaction_radial": student.layers[0].edge.radial_head.network[0].weight,
        "central_site_type": student.central.embedding.weight,
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in targets.values():
            noise = torch.randn(
                parameter.shape,
                dtype=parameter.dtype,
                device=parameter.device,
                generator=generator,
            )
            parameter.add_(scale * noise)
    return student, targets


def _overfit_once(teacher, target_batch, contexts, *, steps):
    student, targets = _perturbed_student(teacher, seed=91027, scale=2.0e-3)
    target_before = {
        name: parameter.detach().clone() for name, parameter in targets.items()
    }
    baseline_before = student.atomic_baseline.detach().clone()
    optimizer = build_optimizer(
        student,
        OptimizerConfig(learning_rate=5.0e-4, weight_decay=0.0),
    )
    loss_config = LossConfig(
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.0,
        energy_scale=2.0e-2,
        force_scale=2.0e-3,
    )
    initial = validation_step(
        student,
        target_batch,
        contexts,
        loss_config,
        ValidationStepConfig(),
    )
    curve = [initial.total_loss]
    step_results = []
    for _ in range(steps):
        result = train_step(
            student,
            optimizer,
            target_batch,
            contexts,
            loss_config,
            TrainStepConfig(gradient_clip_norm=None),
        )
        assert math.isfinite(result.total_loss)
        assert math.isfinite(result.energy_loss)
        assert math.isfinite(result.force_loss)
        assert math.isfinite(result.pre_clip_grad_norm)
        assert math.isfinite(result.post_clip_grad_norm)
        curve.append(result.total_loss)
        step_results.append(result)
    final = validation_step(
        student,
        target_batch,
        contexts,
        loss_config,
        ValidationStepConfig(),
    )
    curve.append(final.total_loss)
    changed = {
        name: not torch.equal(parameter.detach(), target_before[name])
        for name, parameter in targets.items()
    }
    assert torch.equal(student.atomic_baseline, baseline_before)
    return student, initial, final, tuple(curve), tuple(step_results), changed


def test_tiny_frozen_teacher_energy_force_overfit(typed_crystal):
    global_rng_before = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(271828)
        _, teacher, registry, samples, _, contexts = _mixed_case(typed_crystal)

    # A: alpha pristine (M=4,N=4,K=0); B: zeta vacancy (M=6,N=5,K=1).
    geometry_samples = tuple(
        replace(sample, energy=None, forces=None, stress=None)
        for sample in (samples[1], samples[0])
    )
    geometry_batch = collate_structure_samples(geometry_samples, registry)
    assert geometry_batch.template_ids == ("alpha", "zeta")
    assert tuple(int(value) for value in geometry_batch.atom_ptr.diff()) == (4, 5)
    assert set(geometry_batch.atomic_numbers.tolist()) == {6, 41}

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_before = _tensor_state(teacher)
    teacher_mode_before = teacher.training
    target_batch = _teacher_labels(
        teacher, geometry_batch, geometry_samples, registry, contexts
    )
    _assert_tensor_state_equal(_tensor_state(teacher), teacher_before)
    assert teacher.training == teacher_mode_before
    assert all(parameter.grad is None for parameter in teacher.parameters())

    target_before = {
        name: tensor.detach().clone()
        for name, tensor in target_batch.__dict__.items()
        if isinstance(tensor, torch.Tensor)
    }
    first = _overfit_once(teacher, target_batch, contexts, steps=20)
    second = _overfit_once(teacher, target_batch, contexts, steps=20)
    student, initial, final, curve, step_results, changed = first
    _, initial_repeat, final_repeat, curve_repeat, _, changed_repeat = second

    assert curve == curve_repeat
    assert initial == initial_repeat and final == final_repeat
    assert changed == changed_repeat
    assert initial.total_loss > 0.0
    assert final.total_loss < initial.total_loss
    assert final.energy_loss < initial.energy_loss
    assert final.force_loss < initial.force_loss
    assert final.total_loss <= 0.5 * initial.total_loss
    assert all(changed.values())
    assert all(result.need_forces and not result.need_stress for result in step_results)
    assert all(result.number_of_parameters_with_grad > 0 for result in step_results)
    assert not any(
        teacher_parameter is student_parameter
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()
        )
    )
    _assert_tensor_state_equal(_tensor_state(teacher), teacher_before)
    assert torch.equal(student.atomic_baseline, teacher.atomic_baseline)
    for name, before in target_before.items():
        assert torch.equal(getattr(target_batch, name), before)
    assert not target_batch.positions.requires_grad
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)

    print(
        "tiny-overfit",
        {
            "steps": 20,
            "learning_rate": 5.0e-4,
            "initial_total": initial.total_loss,
            "final_total": final.total_loss,
            "initial_energy": initial.energy_loss,
            "final_energy": final.energy_loss,
            "initial_force": initial.force_loss,
            "final_force": final.force_loss,
            "ratio": final.total_loss / initial.total_loss,
            "changed": changed,
            "curve": curve,
        },
    )
