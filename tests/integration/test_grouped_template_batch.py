from __future__ import annotations

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
        assert torch.equal(grouped.forces, torch.cat([out.forces for out in individual]))
    if grouped.stress is not None:
        assert torch.equal(grouped.stress, torch.stack([out.stress for out in individual]))
        assert torch.equal(
            grouped.stress_voigt,
            torch.stack([out.stress_voigt for out in individual]),
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
