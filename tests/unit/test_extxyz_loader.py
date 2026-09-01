from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.stress import voigt_6_to_full_3x3_stress

from refsite_mlip.data import (
    ExtXYZLoadConfig,
    ExtXYZLoadDiagnostics,
    ExtXYZLoadError,
    InMemoryStructureDataset,
    ReferenceTemplate,
    StrictTemplateDomain,
    TemplateRegistry,
    collate_structure_samples,
    load_extxyz_dataset,
    load_extxyz_samples,
)
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.phase import find_typed_stabilizer
from refsite_mlip.training import fingerprint_batch_sequence


TEMPLATE_ID = "tiny_strict_extxyz"


def _registry() -> TemplateRegistry:
    fractional = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    )
    site_types = torch.tensor([0, 1], dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64) * 4.0
    topology = build_reference_graph_topology(
        fractional,
        site_types,
        cell,
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    domain = StrictTemplateDomain(
        reference_site_count=2,
        supercell_shape=(1, 1, 1),
        species_vocabulary=(6, 41),
        reference_composition=(1, 1),
        allowed_compositions=((1, 1), (0, 1)),
        allowed_num_atoms=(2, 1),
        allowed_vacancy_masses=(0, 1),
    )
    template = ReferenceTemplate.snapshot(
        TEMPLATE_ID,
        topology,
        torch.eye(3, dtype=torch.long),
        torch.ones(3, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        find_typed_stabilizer(fractional, site_types),
        (6, 41),
        strict_domain=domain,
    )
    registry = TemplateRegistry()
    registry.add(template)
    return registry


@pytest.fixture
def registry() -> TemplateRegistry:
    return _registry()


def _labeled_atoms(index: int = 0, *, stress=None) -> Atoms:
    scale = 1.0 + 0.005 * index
    cell = np.eye(3) * 4.0 * scale
    positions = np.array(
        [[0.03 * index, 0.0, 0.0], [2.0 * scale, 2.0 * scale, 2.0 * scale]]
    )
    atoms = Atoms(
        numbers=[6, 41],
        positions=positions,
        cell=cell,
        pbc=True,
    )
    energy = -2.5 + 0.25 * index
    forces = np.array(
        [[0.1 + index, -0.2, 0.3], [-0.4, 0.5 + index, -0.6]],
        dtype=float,
    )
    if stress is None:
        stress = np.array(
            [
                0.01 + index,
                0.02 + index,
                0.03 + index,
                0.004,
                -0.005,
                0.006,
            ]
        )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        forces=forces,
        stress=np.asarray(stress),
    )
    return atoms


def _write_extxyz(path, frames) -> None:
    write(path, frames, format="extxyz")


def _config(path, **overrides) -> ExtXYZLoadConfig:
    values = {
        "source_path": str(path),
        "sample_id_prefix": "logical",
        "template_id": TEMPLATE_ID,
    }
    values.update(overrides)
    return ExtXYZLoadConfig(**values)


def _assert_sample_equal(left, right) -> None:
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
    ):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            assert torch.equal(left_value, right_value)


def test_config_and_diagnostics_round_trip(tmp_path, registry):
    path = tmp_path / "data.xyz"
    _write_extxyz(path, [_labeled_atoms(0)])
    config = _config(path)
    assert ExtXYZLoadConfig.from_dict(config.to_dict()) == config

    result = load_extxyz_samples(config, registry)
    restored = ExtXYZLoadDiagnostics.from_dict(result.diagnostics.to_dict())
    assert restored == result.diagnostics
    assert result.diagnostics.frame_count == 1
    assert result.diagnostics.template_fingerprint == registry.resolve(
        TEMPLATE_ID
    ).fingerprint


def test_order_labels_voigt_sign_and_stable_ids(tmp_path, registry):
    path = tmp_path / "ordered.xyz"
    frames = [_labeled_atoms(0), _labeled_atoms(1), _labeled_atoms(2)]
    _write_extxyz(path, frames)
    parsed = read(path, index=":", format="extxyz")

    result = load_extxyz_samples(_config(path), registry)
    assert tuple(sample.sample_id for sample in result.samples) == (
        "logical:000000",
        "logical:000001",
        "logical:000002",
    )
    for sample, atoms in zip(result.samples, parsed):
        assert torch.equal(
            sample.positions,
            torch.tensor(atoms.positions, dtype=torch.float64),
        )
        assert torch.equal(
            sample.atomic_numbers,
            torch.tensor(atoms.numbers, dtype=torch.long),
        )
        assert torch.equal(
            sample.cell, torch.tensor(atoms.cell.array, dtype=torch.float64)
        )
        assert torch.equal(
            sample.energy,
            torch.tensor(atoms.calc.results["energy"], dtype=torch.float64),
        )
        assert torch.equal(
            sample.forces,
            torch.tensor(atoms.calc.results["forces"], dtype=torch.float64),
        )
        expected_stress = torch.tensor(
            voigt_6_to_full_3x3_stress(atoms.calc.results["stress"]),
            dtype=torch.float64,
        )
        assert torch.equal(sample.stress, expected_stress)
        # No stress sign inversion: the tensile-positive xx component survives.
        assert float(sample.stress[0, 0]) == pytest.approx(
            float(atoms.calc.results["stress"][0])
        )


def test_relocated_file_has_tensor_and_semantic_digest_parity(tmp_path, registry):
    left = tmp_path / "left.xyz"
    right = tmp_path / "nested" / "renamed.xyz"
    right.parent.mkdir()
    _write_extxyz(left, [_labeled_atoms(0), _labeled_atoms(1)])
    shutil.copyfile(left, right)

    first = load_extxyz_samples(_config(left), registry)
    second = load_extxyz_samples(_config(right), registry)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics.semantic_sha256 == second.diagnostics.semantic_sha256
    for left_sample, right_sample in zip(first.samples, second.samples):
        _assert_sample_equal(left_sample, right_sample)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_conversion_and_repeat_determinism(tmp_path, registry, dtype):
    path = tmp_path / "dtype.xyz"
    _write_extxyz(path, [_labeled_atoms(0), _labeled_atoms(1)])
    config = _config(path, dtype=dtype)
    first = load_extxyz_samples(config, registry)
    second = load_extxyz_samples(config, registry)
    assert first.diagnostics == second.diagnostics
    for left, right in zip(first.samples, second.samples):
        _assert_sample_equal(left, right)
        assert left.positions.dtype == dtype
        assert left.atomic_numbers.dtype == torch.long
        assert left.pbc.dtype == torch.bool


def test_loader_owns_storage_and_does_not_mutate_ase_atoms(
    tmp_path, registry, monkeypatch
):
    source = tmp_path / "owned.xyz"
    source.touch()
    atoms = _labeled_atoms(0)
    snapshot = (
        atoms.positions.copy(),
        atoms.numbers.copy(),
        atoms.cell.array.copy(),
        atoms.calc.results["forces"].copy(),
    )
    monkeypatch.setattr("ase.io.iread", lambda *args, **kwargs: iter((atoms,)))

    result = load_extxyz_samples(_config(source), registry)
    np.testing.assert_array_equal(atoms.positions, snapshot[0])
    np.testing.assert_array_equal(atoms.numbers, snapshot[1])
    np.testing.assert_array_equal(atoms.cell.array, snapshot[2])
    np.testing.assert_array_equal(atoms.calc.results["forces"], snapshot[3])
    atoms.positions[0, 0] += 9.0
    atoms.calc.results["forces"][0, 0] += 9.0
    assert float(result.samples[0].positions[0, 0]) == snapshot[0][0, 0]
    assert float(result.samples[0].forces[0, 0]) == snapshot[3][0, 0]


def test_missing_policy_and_real_zero_are_distinct_after_collate(
    tmp_path, registry, monkeypatch
):
    source = tmp_path / "missing.xyz"
    source.touch()
    zero = _labeled_atoms(0)
    zero.calc.results["energy"] = 0.0
    missing = _labeled_atoms(0)
    missing.calc.results.pop("energy")
    monkeypatch.setattr(
        "ase.io.iread", lambda *args, **kwargs: iter((zero, missing))
    )
    config = _config(source, require_energy=False)
    result = load_extxyz_samples(config, registry)
    assert result.samples[0].energy is not None
    assert float(result.samples[0].energy) == 0.0
    assert result.samples[1].energy is None
    assert result.diagnostics.missing_energy_count == 1

    batch = collate_structure_samples(result.samples, registry)
    assert torch.equal(batch.energy, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(batch.energy_mask, torch.tensor([True, False]))
    restored = batch.unbind()
    assert restored[0].energy is not None
    assert restored[1].energy is None


def test_inference_style_load_does_not_invent_supervision(
    tmp_path, registry, monkeypatch
):
    source = tmp_path / "inference.xyz"
    source.touch()
    atoms = _labeled_atoms(0)
    atoms.calc = None
    monkeypatch.setattr("ase.io.iread", lambda *args, **kwargs: iter((atoms,)))
    config = _config(
        source,
        require_energy=False,
        require_forces=False,
        require_stress=False,
    )
    result = load_extxyz_samples(config, registry)
    sample = result.samples[0]
    assert sample.energy is None
    assert sample.forces is None
    assert sample.stress is None
    assert result.diagnostics.missing_energy_count == 1
    assert result.diagnostics.missing_forces_count == 1
    assert result.diagnostics.missing_stress_count == 1

    batch = collate_structure_samples(result.samples, registry)
    assert not torch.any(batch.energy_mask)
    assert not torch.any(batch.force_present)
    assert not torch.any(batch.stress_present)


def test_required_missing_label_fails(tmp_path, registry, monkeypatch):
    source = tmp_path / "required.xyz"
    source.touch()
    atoms = _labeled_atoms(0)
    atoms.calc.results.pop("stress")
    monkeypatch.setattr("ase.io.iread", lambda *args, **kwargs: iter((atoms,)))
    with pytest.raises(ExtXYZLoadError, match="MISSING_LABEL") as captured:
        load_extxyz_samples(_config(source), registry)
    assert captured.value.frame_index == 0
    assert captured.value.sample_id == "logical:000000"
    assert captured.value.label == "stress"


def test_conflicting_duplicate_label_rejected(tmp_path, registry, monkeypatch):
    source = tmp_path / "conflict.xyz"
    source.touch()
    atoms = _labeled_atoms(0)
    atoms.info["energy"] = float(atoms.calc.results["energy"]) + 1.0
    monkeypatch.setattr("ase.io.iread", lambda *args, **kwargs: iter((atoms,)))
    with pytest.raises(ExtXYZLoadError, match="CONFLICTING_LABEL"):
        load_extxyz_samples(_config(source), registry)


@pytest.mark.parametrize(
    ("label", "value", "reason"),
    [
        ("energy", np.nan, "NONFINITE_LABEL"),
        ("forces", np.zeros((3, 3)), "MALFORMED_LABEL"),
        ("stress", np.zeros(5), "MALFORMED_LABEL"),
        (
            "stress",
            np.array([[1.0, 0.2, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            "MALFORMED_LABEL",
        ),
    ],
)
def test_nonfinite_and_malformed_labels_rejected(
    tmp_path, registry, monkeypatch, label, value, reason
):
    source = tmp_path / f"bad-{label}.xyz"
    source.touch()
    atoms = _labeled_atoms(0)
    atoms.calc.results[label] = value
    monkeypatch.setattr("ase.io.iread", lambda *args, **kwargs: iter((atoms,)))
    with pytest.raises(ExtXYZLoadError, match=reason):
        load_extxyz_samples(_config(source), registry)


def test_nonperiodic_and_strict_domain_rejections(tmp_path, registry, monkeypatch):
    source = tmp_path / "geometry.xyz"
    source.touch()
    nonperiodic = _labeled_atoms(0)
    nonperiodic.pbc = [True, False, True]
    monkeypatch.setattr(
        "ase.io.iread", lambda *args, **kwargs: iter((nonperiodic,))
    )
    with pytest.raises(ExtXYZLoadError, match="NONPERIODIC_STRUCTURE"):
        load_extxyz_samples(_config(source), registry)

    wrong = _labeled_atoms(0)
    wrong.numbers[0] = 41
    monkeypatch.setattr("ase.io.iread", lambda *args, **kwargs: iter((wrong,)))
    with pytest.raises(ExtXYZLoadError, match="TEMPLATE_DOMAIN_REJECTION"):
        load_extxyz_samples(_config(source), registry)

    strained = _labeled_atoms(0)
    strained.set_cell(strained.cell.array * 1.25, scale_atoms=True)
    monkeypatch.setattr(
        "ase.io.iread", lambda *args, **kwargs: iter((strained,))
    )
    with pytest.raises(ExtXYZLoadError, match="TEMPLATE_DOMAIN_REJECTION"):
        load_extxyz_samples(_config(source), registry)


def test_nonfinite_geometry_rejected(tmp_path, registry, monkeypatch):
    source = tmp_path / "nonfinite-geometry.xyz"
    source.touch()
    atoms = _labeled_atoms(0)
    atoms.positions[0, 0] = np.nan
    monkeypatch.setattr("ase.io.iread", lambda *args, **kwargs: iter((atoms,)))
    with pytest.raises(ExtXYZLoadError, match="NONFINITE_GEOMETRY"):
        load_extxyz_samples(_config(source), registry)


def test_wrong_template_and_cross_source_duplicate_ids_rejected(
    tmp_path, registry
):
    left = tmp_path / "left.xyz"
    right = tmp_path / "right.xyz"
    _write_extxyz(left, [_labeled_atoms(0)])
    _write_extxyz(right, [_labeled_atoms(1)])
    with pytest.raises(ValueError, match="unknown explicit template_id"):
        load_extxyz_samples(
            _config(left, template_id="not-registered"), registry
        )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_extxyz_dataset((_config(left), _config(right)), registry)


def test_collate_unbind_and_checkpoint_fingerprint_deterministic(
    tmp_path, registry
):
    path = tmp_path / "batch.xyz"
    _write_extxyz(path, [_labeled_atoms(0), _labeled_atoms(1)])
    result = load_extxyz_samples(_config(path), registry)
    batch = collate_structure_samples(result.samples, registry)
    assert torch.all(batch.energy_mask)
    assert torch.all(batch.force_present)
    assert torch.all(batch.stress_present)
    assert torch.all(batch.force_mask)
    assert torch.all(batch.stress_mask)
    assert not torch.any(batch.force_mask_provided)
    assert not torch.any(batch.stress_mask_provided)
    for original, restored in zip(result.samples, batch.unbind()):
        _assert_sample_equal(original, restored)
        assert restored.force_mask is None
        assert restored.stress_mask is None

    first = fingerprint_batch_sequence((batch,), split_name="train")
    second = fingerprint_batch_sequence((batch,), split_name="train")
    assert first == second


def test_load_dataset_returns_validated_in_memory_sequence(tmp_path, registry):
    path = tmp_path / "dataset.xyz"
    _write_extxyz(path, [_labeled_atoms(0), _labeled_atoms(1)])
    dataset = load_extxyz_dataset(_config(path), registry)
    assert isinstance(dataset, InMemoryStructureDataset)
    assert len(dataset) == 2
    assert dataset[0].sample_id == "logical:000000"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_load_and_collate_smoke(tmp_path, registry, dtype):
    path = tmp_path / f"cuda-{dtype}.xyz"
    _write_extxyz(path, [_labeled_atoms(0), _labeled_atoms(1)])
    result = load_extxyz_samples(
        _config(path, dtype=dtype, device="cuda"), registry
    )
    batch = collate_structure_samples(result.samples, registry)
    assert batch.positions.device.type == "cuda"
    assert batch.positions.dtype == dtype
    assert batch.atomic_numbers.dtype == torch.long
    assert torch.all(torch.isfinite(batch.energy))
