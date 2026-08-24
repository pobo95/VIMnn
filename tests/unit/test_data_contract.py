from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from refsite_mlip.data import (
    ENERGY_UNIT,
    FORCE_UNIT,
    LENGTH_UNIT,
    STRESS_SIGN,
    STRESS_UNIT,
    STRESS_VOIGT_ORDER,
    InMemoryStructureDataset,
    ReferenceTemplate,
    StructureSample,
    TemplateRegistry,
)
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.phase.stabilizer import find_typed_stabilizer


def make_template(data, template_id="typed"):
    topology = build_reference_graph_topology(
        data["sites"],
        data["site_types"],
        data["cell"],
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    return ReferenceTemplate.snapshot(
        template_id=template_id,
        topology=topology,
        phase_modes=data["modes"],
        phase_mode_weights=data["mode_weights"],
        site_alignment_weights=data["site_weights"],
        phase_channel_weights=data["channel_weights"],
        stabilizer=find_typed_stabilizer(data["sites"], data["site_types"]),
        supported_species=(6, 41),
    )


def make_sample(data, sample_id="sample", template_id="typed", labeled=True):
    atomic_numbers = torch.where(
        data["site_types"] == 0,
        torch.tensor(6, dtype=torch.long),
        torch.tensor(41, dtype=torch.long),
    )
    kwargs = {}
    if labeled:
        kwargs = {
            "energy": torch.tensor(-1.25, dtype=data["cell"].dtype),
            "forces": torch.zeros_like(data["positions"]),
            "stress": torch.eye(3, dtype=data["cell"].dtype) * 0.02,
            "force_mask": torch.ones_like(data["positions"], dtype=torch.bool),
            "stress_mask": torch.eye(3, dtype=torch.bool),
        }
    return StructureSample(
        sample_id=sample_id,
        positions=data["positions"].clone(),
        atomic_numbers=atomic_numbers,
        cell=data["cell"].clone(),
        pbc=torch.ones(3, dtype=torch.bool),
        origin=data["origin"].clone(),
        template_id=template_id,
        **kwargs,
    )


def test_valid_labeled_and_unlabeled_sample(typed_crystal):
    labeled = make_sample(typed_crystal)
    unlabeled = make_sample(typed_crystal, sample_id="inference", labeled=False)

    assert labeled.num_atoms == 6
    assert labeled.energy is not None
    assert unlabeled.energy is None
    assert unlabeled.forces is None
    assert unlabeled.stress is None
    assert LENGTH_UNIT == "angstrom"
    assert ENERGY_UNIT == "eV"
    assert FORCE_UNIT == "eV/angstrom"
    assert STRESS_UNIT == "eV/angstrom^3"
    assert STRESS_SIGN == "tensile_positive"
    assert STRESS_VOIGT_ORDER == ("xx", "yy", "zz", "yz", "xz", "xy")


def test_sample_to_preserves_tensor_roles_and_device(typed_crystal):
    sample = make_sample(typed_crystal)
    converted = sample.to(dtype=torch.float32)

    for value in (
        converted.positions,
        converted.cell,
        converted.origin,
        converted.energy,
        converted.forces,
        converted.stress,
    ):
        assert value.dtype == torch.float32
        assert value.device.type == "cpu"
    assert converted.atomic_numbers.dtype == torch.long
    assert converted.pbc.dtype == torch.bool
    assert converted.force_mask.dtype == torch.bool

    if torch.cuda.is_available():
        cuda_sample = sample.to(device="cuda", dtype=torch.float64)
        assert cuda_sample.positions.device.type == "cuda"
        assert cuda_sample.atomic_numbers.device.type == "cuda"
        assert cuda_sample.positions.dtype == torch.float64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("positions", torch.zeros(6, dtype=torch.float64)),
        ("atomic_numbers", torch.ones(6, dtype=torch.float64)),
        ("cell", torch.zeros((3, 3), dtype=torch.float64)),
        ("origin", torch.zeros((1, 3), dtype=torch.float64)),
        ("pbc", torch.tensor([True, False, True])),
    ],
)
def test_invalid_shape_dtype_cell_and_pbc_fail_fast(typed_crystal, field, value):
    sample = make_sample(typed_crystal)
    with pytest.raises(ValueError):
        replace(sample, **{field: value})


@pytest.mark.parametrize("field", ["positions", "cell", "origin"])
def test_nonfinite_geometry_fails_fast(typed_crystal, field):
    sample = make_sample(typed_crystal)
    value = getattr(sample, field).clone()
    value.reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        replace(sample, **{field: value})


def test_optional_label_and_mask_validation(typed_crystal):
    sample = make_sample(typed_crystal)
    with pytest.raises(ValueError, match="forces shape"):
        replace(sample, forces=torch.zeros((5, 3), dtype=torch.float64))
    with pytest.raises(ValueError, match="stress shape"):
        replace(sample, stress=torch.zeros(6, dtype=torch.float64))
    with pytest.raises(ValueError, match="force_mask requires"):
        replace(sample, forces=None)
    with pytest.raises(ValueError, match="stress_mask must"):
        replace(sample, stress_mask=torch.ones((3, 3), dtype=torch.float64))
    bad_energy = torch.tensor(float("inf"), dtype=torch.float64)
    with pytest.raises(ValueError, match="energy contains"):
        replace(sample, energy=bad_energy)


def test_registry_add_resolve_and_missing_id(typed_crystal):
    template = make_template(typed_crystal)
    registry = TemplateRegistry()
    registry.add(template)
    registry.add(template)

    assert len(registry) == 1
    assert "typed" in registry
    resolved = registry.resolve("typed")
    assert resolved.fingerprint == template.fingerprint
    with pytest.raises(KeyError, match="unknown template_id"):
        registry.resolve("missing")


def test_registry_rejects_conflicting_duplicate(typed_crystal):
    template = make_template(typed_crystal)
    changed_weights = template.phase_mode_weights.clone()
    changed_weights[0] += 0.25
    conflict = ReferenceTemplate.snapshot(
        template.template_id,
        template.topology,
        template.phase_modes,
        changed_weights,
        template.site_alignment_weights,
        template.phase_channel_weights,
        template.stabilizer,
        template.supported_species,
    )
    registry = TemplateRegistry()
    registry.add(template)
    with pytest.raises(ValueError, match="conflicting template_id"):
        registry.add(conflict)


def test_registry_fingerprint_is_order_independent(typed_crystal):
    first = make_template(typed_crystal, "first")
    second = make_template(typed_crystal, "second")
    registry_ab = TemplateRegistry()
    registry_ab.add(first)
    registry_ab.add(second)
    registry_ba = TemplateRegistry()
    registry_ba.add(second)
    registry_ba.add(first)

    assert registry_ab.fingerprint == registry_ba.fingerprint


def test_physical_field_changes_fingerprint_and_registry_is_immutable(typed_crystal):
    template = make_template(typed_crystal)
    original_fingerprint = template.fingerprint
    changed_weights = template.phase_mode_weights.clone()
    changed_weights[1] += 0.125
    changed = ReferenceTemplate.snapshot(
        template.template_id,
        template.topology,
        template.phase_modes,
        changed_weights,
        template.site_alignment_weights,
        template.phase_channel_weights,
        template.stabilizer,
        template.supported_species,
    )
    assert changed.fingerprint != original_fingerprint

    registry = TemplateRegistry()
    registry.add(template)
    stored_fingerprint = registry.fingerprint
    template.phase_mode_weights[0] += 10.0
    assert registry.fingerprint == stored_fingerprint
    resolved = registry.resolve("typed")
    resolved.topology.reference_fractional[0, 0] += 1.0
    assert registry.fingerprint == stored_fingerprint


def test_dataset_indexing_and_unique_sample_ids(typed_crystal):
    registry = TemplateRegistry()
    registry.add(make_template(typed_crystal))
    first = make_sample(typed_crystal, "first")
    second = make_sample(typed_crystal, "second", labeled=False)
    dataset = InMemoryStructureDataset([first, second], registry)

    assert len(dataset) == 2
    assert dataset[0] is first
    assert dataset[-1] is second
    assert dataset[:] == (first, second)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        InMemoryStructureDataset([first, replace(first)], registry)


def test_dataset_unknown_template_and_n_greater_than_m_fail(typed_crystal):
    registry = TemplateRegistry()
    registry.add(make_template(typed_crystal))
    unknown = make_sample(typed_crystal, template_id="missing")
    with pytest.raises(KeyError, match="unknown template_id"):
        InMemoryStructureDataset([unknown], registry)

    sample = make_sample(typed_crystal, labeled=False)
    too_many = replace(
        sample,
        positions=torch.cat((sample.positions, sample.positions[:1])),
        atomic_numbers=torch.cat(
            (sample.atomic_numbers, torch.tensor([6], dtype=torch.long))
        ),
    )
    with pytest.raises(ValueError, match="N > M"):
        InMemoryStructureDataset([too_many], registry)


def test_dataset_rejects_species_outside_template_contract(typed_crystal):
    registry = TemplateRegistry()
    registry.add(make_template(typed_crystal))
    sample = make_sample(typed_crystal, labeled=False)
    atomic_numbers = sample.atomic_numbers.clone()
    atomic_numbers[0] = 8
    unsupported = replace(sample, atomic_numbers=atomic_numbers)
    with pytest.raises(ValueError, match="unknown species"):
        InMemoryStructureDataset([unsupported], registry)
