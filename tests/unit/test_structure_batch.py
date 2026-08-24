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
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.phase.stabilizer import find_typed_stabilizer


def make_template(data, template_id):
    topology = build_reference_graph_topology(
        data["sites"],
        data["site_types"],
        data["cell"],
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    return ReferenceTemplate.snapshot(
        template_id,
        topology,
        data["modes"],
        data["mode_weights"],
        data["site_weights"],
        data["channel_weights"],
        find_typed_stabilizer(data["sites"], data["site_types"]),
        (6, 41),
    )


def make_registry(data):
    registry = TemplateRegistry()
    registry.add(make_template(data, "alpha"))
    registry.add(make_template(data, "zeta"))
    return registry


def atomic_numbers(data, count):
    return torch.where(
        data["site_types"][:count] == 0,
        torch.tensor(6, dtype=torch.long),
        torch.tensor(41, dtype=torch.long),
    )


def make_sample(
    data,
    sample_id,
    count,
    template_id="alpha",
    *,
    energy=None,
    forces=False,
    stress=False,
    partial_masks=False,
):
    dtype = data["positions"].dtype
    force_value = torch.arange(count * 3, dtype=dtype).reshape(count, 3) / 10
    stress_value = torch.arange(9, dtype=dtype).reshape(3, 3) / 20
    force_mask = None
    stress_mask = None
    if partial_masks:
        force_mask = torch.zeros((count, 3), dtype=torch.bool)
        force_mask[::2, 0] = True
        stress_mask = torch.eye(3, dtype=torch.bool)
    return StructureSample(
        sample_id=sample_id,
        positions=data["positions"][:count].clone(),
        atomic_numbers=atomic_numbers(data, count),
        cell=data["cell"].clone(),
        pbc=torch.ones(3, dtype=torch.bool),
        origin=data["origin"].clone(),
        template_id=template_id,
        energy=None if energy is None else torch.tensor(energy, dtype=dtype),
        forces=force_value if forces else None,
        stress=stress_value if stress else None,
        force_mask=force_mask if forces else None,
        stress_mask=stress_mask if stress else None,
    )


def assert_optional_tensor_equal(left, right):
    assert (left is None) == (right is None)
    if left is not None:
        assert torch.equal(left, right)


def assert_sample_equal(left, right):
    assert left.sample_id == right.sample_id
    assert left.template_id == right.template_id
    for name in (
        "positions",
        "atomic_numbers",
        "cell",
        "pbc",
        "origin",
        "energy",
        "forces",
        "stress",
        "force_mask",
        "stress_mask",
    ):
        assert_optional_tensor_equal(getattr(left, name), getattr(right, name))


def test_ragged_collate_atom_ptr_and_atom_batch(typed_crystal):
    registry = make_registry(typed_crystal)
    samples = (
        make_sample(typed_crystal, "six", 6, energy=1.0),
        make_sample(typed_crystal, "three", 3, energy=2.0),
        make_sample(typed_crystal, "five", 5, energy=3.0),
    )
    batch = collate_structure_samples(samples, registry)

    assert batch.positions.shape == (14, 3)
    assert batch.atomic_numbers.shape == (14,)
    assert batch.cells.shape == (3, 3, 3)
    assert batch.origins.shape == (3, 3)
    assert batch.pbc.shape == (3, 3)
    assert batch.atom_ptr.tolist() == [0, 6, 9, 14]
    assert batch.atom_batch.tolist() == [0] * 6 + [1] * 3 + [2] * 5
    assert torch.equal(batch.positions[:6], samples[0].positions)
    assert torch.equal(batch.positions[6:9], samples[1].positions)


def test_mixed_template_grouping_is_sorted_and_preserves_membership(typed_crystal):
    registry = make_registry(typed_crystal)
    samples = (
        make_sample(typed_crystal, "z0", 2, "zeta"),
        make_sample(typed_crystal, "a0", 3, "alpha"),
        make_sample(typed_crystal, "z1", 1, "zeta"),
    )
    batch = collate_structure_samples(samples, registry)
    groups = batch.template_groups

    assert [group.template_id for group in groups] == ["alpha", "zeta"]
    assert groups[0].structure_indices.tolist() == [1]
    assert groups[0].atom_slices == (slice(2, 5),)
    assert groups[0].atom_indices.tolist() == [2, 3, 4]
    assert groups[1].structure_indices.tolist() == [0, 2]
    assert groups[1].atom_slices == (slice(0, 2), slice(5, 6))
    assert groups[1].atom_indices.tolist() == [0, 1, 5]
    for index, sample in enumerate(samples):
        assert batch.template_fingerprints[index] == registry.resolve(
            sample.template_id
        ).fingerprint


def test_mixed_optional_targets_and_zero_label_are_unambiguous(typed_crystal):
    registry = make_registry(typed_crystal)
    fully_labeled_zero = make_sample(
        typed_crystal, "zero", 6, energy=0.0, forces=True, stress=True
    )
    energy_only = make_sample(typed_crystal, "energy", 5, energy=1.5)
    partial = make_sample(
        typed_crystal,
        "partial",
        4,
        forces=True,
        stress=True,
        partial_masks=True,
    )
    inference = make_sample(typed_crystal, "inference", 3)
    batch = collate_structure_samples(
        (fully_labeled_zero, energy_only, partial, inference), registry
    )

    assert batch.energy.tolist() == [0.0, 1.5, 0.0, 0.0]
    assert batch.energy_mask.tolist() == [True, True, False, False]
    assert batch.force_present.tolist() == [True, False, True, False]
    assert batch.stress_present.tolist() == [True, False, True, False]
    assert batch.force_mask_provided.tolist() == [False, False, True, False]
    assert batch.stress_mask_provided.tolist() == [False, False, True, False]
    assert bool(torch.all(batch.force_mask[:6]))
    assert not bool(torch.any(batch.force_mask[6:11]))
    assert torch.equal(batch.force_mask[11:15], partial.force_mask)
    assert not bool(torch.any(batch.force_mask[15:]))
    assert torch.equal(batch.stress_mask[2], partial.stress_mask)
    assert not bool(torch.any(batch.stress_mask[3]))


def test_round_trip_preserves_geometry_targets_and_explicit_masks(typed_crystal):
    registry = make_registry(typed_crystal)
    samples = (
        make_sample(typed_crystal, "full", 6, energy=-2.0, forces=True, stress=True),
        make_sample(
            typed_crystal,
            "partial",
            4,
            "zeta",
            forces=True,
            stress=True,
            partial_masks=True,
        ),
        make_sample(typed_crystal, "none", 2),
    )
    batch = collate_structure_samples(samples, registry)

    for expected, actual in zip(samples, batch.unbind()):
        assert_sample_equal(expected, actual)
    assert_sample_equal(samples[-1], batch.structure_slice(-1))
    with pytest.raises(IndexError):
        batch.structure_slice(3)


def test_batch_to_casts_only_floating_tensors(typed_crystal):
    registry = make_registry(typed_crystal)
    sample = make_sample(
        typed_crystal, "sample", 4, energy=0.0, forces=True, stress=True
    )
    batch = collate_structure_samples((sample,), registry).to(dtype=torch.float32)

    for tensor in (
        batch.positions,
        batch.cells,
        batch.origins,
        batch.energy,
        batch.forces,
        batch.stress,
    ):
        assert tensor.dtype == torch.float32
    for tensor in (batch.atomic_numbers, batch.atom_ptr, batch.atom_batch):
        assert tensor.dtype == torch.long
    for tensor in (
        batch.pbc,
        batch.energy_mask,
        batch.force_mask,
        batch.stress_mask,
        batch.force_present,
        batch.stress_present,
    ):
        assert tensor.dtype == torch.bool


def test_empty_duplicate_and_mixed_dtype_fail_fast(typed_crystal):
    registry = make_registry(typed_crystal)
    with pytest.raises(ValueError, match="empty"):
        collate_structure_samples((), registry)

    sample = make_sample(typed_crystal, "duplicate", 3)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        collate_structure_samples((sample, replace(sample)), registry)

    float32_sample = sample.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="share floating dtype and device"):
        collate_structure_samples((sample, replace(float32_sample, sample_id="f32")), registry)


def test_unknown_template_and_n_greater_than_m_fail_fast(typed_crystal):
    registry = make_registry(typed_crystal)
    unknown = make_sample(typed_crystal, "unknown", 3, "missing")
    with pytest.raises(KeyError, match="unknown template_id"):
        collate_structure_samples((unknown,), registry)

    sample = make_sample(typed_crystal, "too-many", 6)
    too_many = replace(
        sample,
        positions=torch.cat((sample.positions, sample.positions[:1])),
        atomic_numbers=torch.cat(
            (sample.atomic_numbers, torch.tensor([6], dtype=torch.long))
        ),
    )
    with pytest.raises(ValueError, match="N > M"):
        collate_structure_samples((too_many,), registry)


def test_mixed_device_fails_and_cuda_to_smoke_when_available(typed_crystal):
    if not torch.cuda.is_available():
        return
    registry = make_registry(typed_crystal)
    cpu_sample = make_sample(typed_crystal, "cpu", 3)
    cuda_sample = replace(cpu_sample.to(device="cuda"), sample_id="cuda")
    with pytest.raises(ValueError, match="share floating dtype and device"):
        collate_structure_samples((cpu_sample, cuda_sample), registry)

    cuda_batch = collate_structure_samples((cuda_sample,), registry)
    assert cuda_batch.positions.device.type == "cuda"
    assert cuda_batch.atom_ptr.device.type == "cuda"
    round_trip = cuda_batch.structure_slice(0)
    assert round_trip.positions.device.type == "cuda"


def test_collate_does_not_mutate_inputs(typed_crystal):
    registry = make_registry(typed_crystal)
    sample = make_sample(
        typed_crystal,
        "sample",
        5,
        energy=0.0,
        forces=True,
        stress=True,
        partial_masks=True,
    )
    snapshot = replace(
        sample,
        positions=sample.positions.clone(),
        atomic_numbers=sample.atomic_numbers.clone(),
        cell=sample.cell.clone(),
        pbc=sample.pbc.clone(),
        origin=sample.origin.clone(),
        energy=sample.energy.clone(),
        forces=sample.forces.clone(),
        stress=sample.stress.clone(),
        force_mask=sample.force_mask.clone(),
        stress_mask=sample.stress_mask.clone(),
    )
    batch = collate_structure_samples((sample,), registry)
    batch.positions[0, 0] += 10.0

    assert_sample_equal(sample, snapshot)


def test_collate_revalidates_mutated_sample(typed_crystal):
    registry = make_registry(typed_crystal)
    sample = make_sample(typed_crystal, "sample", 3)
    sample.positions[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        collate_structure_samples((sample,), registry)
