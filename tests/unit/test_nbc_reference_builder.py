from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase.build import bulk
from ase.io import write

from refsite_mlip.data import (
    InMemoryStructureDataset,
    PhaseSpecification,
    ReferenceTemplate,
    ReferenceTemplateBuilderConfig,
    StrictTemplateDomain,
    StructureSample,
    TemplateRegistry,
    build_reference_template_from_atoms,
    build_reference_template_from_poscar,
    canonicalize_reference_atoms,
    collate_structure_samples,
    nbc_rocksalt_template_builder_config,
)
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.models import TemplateExecutionContext
from refsite_mlip.phase import find_typed_stabilizer
from refsite_mlip.phase.stabilizer import torus_difference


LATTICE = 4.482314244155584


def _atoms(size: int):
    return bulk("NbC", "rocksalt", a=LATTICE, cubic=True).repeat(
        (size, size, size)
    )


def _phase(size: int, *, approval_status="provisional"):
    modes = torch.tensor(
        [
            [-size, size, size],
            [size, -size, size],
            [size, size, -size],
            [2 * size, 0, 0],
            [0, 2 * size, 0],
            [0, 0, 2 * size],
        ],
        dtype=torch.long,
    )
    return PhaseSpecification(
        modes=modes,
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status=approval_status,
    )


def _config_111():
    domain = StrictTemplateDomain(
        reference_site_count=8,
        supercell_shape=(1, 1, 1),
        species_vocabulary=(6, 41),
        reference_composition=(4, 4),
        allowed_compositions=((4, 4), (3, 4)),
        allowed_num_atoms=(8, 7),
        allowed_vacancy_masses=(0, 1),
    )
    return ReferenceTemplateBuilderConfig(
        template_id="nbc_rocksalt_111_test",
        strict_domain=domain,
        site_type_ids=(0, 1),
        expected_stabilizer_size=4,
    )


@pytest.fixture(scope="module")
def built_111():
    return build_reference_template_from_atoms(
        _atoms(1), config=_config_111(), phase_specification=_phase(1)
    )


def test_nbc_222_333_config_and_domain_serialization():
    config_222 = nbc_rocksalt_template_builder_config((2, 2, 2))
    config_333 = nbc_rocksalt_template_builder_config((3, 3, 3))
    assert config_222.template_id == "nbc_rocksalt_222_v1"
    assert config_333.template_id == "nbc_rocksalt_333_v1"
    assert config_222.strict_domain.reference_composition == (32, 32)
    assert config_333.strict_domain.reference_composition == (108, 108)
    assert config_222.strict_domain.allowed_compositions == ((32, 32), (31, 32))
    assert config_333.strict_domain.allowed_compositions == (
        (108, 108),
        (107, 108),
    )
    assert ReferenceTemplateBuilderConfig.from_dict(
        config_222.to_dict()
    ) == config_222
    assert StrictTemplateDomain.from_dict(
        config_333.strict_domain.to_dict()
    ) == config_333.strict_domain
    phase = _phase(2)
    restored_phase = PhaseSpecification.from_dict(phase.to_dict())
    assert torch.equal(restored_phase.modes, phase.modes)
    assert torch.equal(restored_phase.mode_weights, phase.mode_weights)
    assert restored_phase.approval_status == "provisional"


def test_canonical_222_333_count_composition_and_global_site_types():
    for size, expected in ((2, 32), (3, 108)):
        atoms = _atoms(size)
        snapshot = (
            atoms.positions.copy(),
            atoms.numbers.copy(),
            atoms.cell.array.copy(),
        )
        canonical = canonicalize_reference_atoms(
            atoms, nbc_rocksalt_template_builder_config((size, size, size))
        )
        assert canonical.fractional_positions.shape == (2 * expected, 3)
        assert torch.equal(
            torch.bincount(canonical.site_types, minlength=2),
            torch.tensor([expected, expected]),
        )
        assert torch.equal(
            torch.bincount(
                torch.where(
                    canonical.atomic_numbers == 6,
                    torch.tensor(0),
                    torch.tensor(1),
                ),
                minlength=2,
            ),
            torch.tensor([expected, expected]),
        )
        assert torch.equal(
            canonical.site_types[:expected], torch.zeros(expected, dtype=torch.long)
        )
        assert torch.equal(
            canonical.site_types[expected:], torch.ones(expected, dtype=torch.long)
        )
        np.testing.assert_array_equal(atoms.positions, snapshot[0])
        np.testing.assert_array_equal(atoms.numbers, snapshot[1])
        np.testing.assert_array_equal(atoms.cell.array, snapshot[2])


def test_canonicalization_and_full_fingerprint_ignore_order_wrapping_and_path(
    tmp_path,
):
    atoms = _atoms(1)
    order = np.array([5, 0, 7, 2, 1, 6, 3, 4])
    transformed = atoms[order]
    shifted = transformed.get_scaled_positions(wrap=False)
    shifted += np.array(
        [[1, -2, 0], [0, 1, -1], [-1, 0, 2], [2, 0, 0]] * 2,
        dtype=float,
    )
    transformed.set_scaled_positions(shifted)

    config = _config_111()
    first = build_reference_template_from_atoms(
        atoms, config=config, phase_specification=_phase(1)
    )
    second = build_reference_template_from_atoms(
        transformed, config=config, phase_specification=_phase(1)
    )
    assert torch.equal(
        first.template.topology.reference_fractional,
        second.template.topology.reference_fractional,
    )
    assert torch.equal(
        first.template.topology.edge_index, second.template.topology.edge_index
    )
    assert torch.equal(first.template.topology.shifts, second.template.topology.shifts)
    assert first.template.fingerprint == second.template.fingerprint

    left = tmp_path / "left.vasp"
    right = tmp_path / "nested" / "right.vasp"
    right.parent.mkdir()
    write(left, atoms, format="vasp", direct=True)
    write(right, atoms, format="vasp", direct=True)
    from_left = build_reference_template_from_poscar(
        left, config=config, phase_specification=_phase(1)
    )
    from_right = build_reference_template_from_poscar(
        right, config=config, phase_specification=_phase(1)
    )
    assert from_left.template.fingerprint == from_right.template.fingerprint


def test_duplicate_unknown_and_wrong_supercell_fail_before_template_build():
    config = nbc_rocksalt_template_builder_config((2, 2, 2))
    duplicate = _atoms(2)
    scaled = duplicate.get_scaled_positions(wrap=False)
    scaled[1] = scaled[0]
    duplicate.set_scaled_positions(scaled)
    with pytest.raises(ValueError, match="duplicate canonical"):
        canonicalize_reference_atoms(duplicate, config)

    unknown = _atoms(2)
    unknown.numbers[0] = 8
    with pytest.raises(ValueError, match="unknown species"):
        canonicalize_reference_atoms(unknown, config)

    with pytest.raises(ValueError, match="site count"):
        canonicalize_reference_atoms(_atoms(3), config)

    wrong_metric = _atoms(2)
    wrong_cell = wrong_metric.cell.array.copy()
    wrong_cell[0] *= 1.2
    wrong_metric.set_cell(wrong_cell, scale_atoms=True)
    with pytest.raises(ValueError, match="supercell_shape"):
        canonicalize_reference_atoms(wrong_metric, config)


def test_strict_domains_accept_only_pristine_or_one_c_vacancy():
    cases = (
        (nbc_rocksalt_template_builder_config((2, 2, 2)), 32),
        (nbc_rocksalt_template_builder_config((3, 3, 3)), 108),
    )
    for config, count in cases:
        domain = config.strict_domain
        pristine = torch.tensor([6] * count + [41] * count, dtype=torch.long)
        vacancy = pristine[1:]
        assert domain.validate_atomic_numbers(pristine).vacancy_mass == 0
        assert domain.validate_atomic_numbers(vacancy).vacancy_mass == 1
        with pytest.raises(ValueError, match="composition"):
            domain.validate_atomic_numbers(pristine[:-1])  # Nb vacancy
        with pytest.raises(ValueError, match="composition"):
            domain.validate_atomic_numbers(pristine[2:])  # two C vacancies
        antisite = pristine.clone()
        antisite[0] = 41
        with pytest.raises(ValueError, match="composition"):
            domain.validate_atomic_numbers(antisite)
        unknown = pristine.clone()
        unknown[0] = 8
        with pytest.raises(ValueError, match="unknown species"):
            domain.validate_atomic_numbers(unknown)

    n63 = torch.tensor([6] * 31 + [41] * 32, dtype=torch.long)
    with pytest.raises(ValueError, match="composition"):
        nbc_rocksalt_template_builder_config(
            (3, 3, 3)
        ).strict_domain.validate_atomic_numbers(n63)


def test_explicit_phase_required_rank_and_alias_contract():
    with pytest.raises(ValueError, match="explicit PhaseSpecification"):
        build_reference_template_from_atoms(
            _atoms(1), config=_config_111(), phase_specification=None
        )
    modes = _phase(1).modes.clone()
    modes[2] = modes[1]
    with pytest.raises(ValueError, match="rank 3"):
        PhaseSpecification(
            modes,
            torch.ones(6),
            torch.eye(2),
            torch.ones(2),
            "provisional",
        )
    wrong_alias = _phase(1).modes.clone()
    wrong_alias[0, 0] = -2
    specification = PhaseSpecification(
        wrong_alias,
        torch.ones(6),
        torch.eye(2),
        torch.ones(2),
        "provisional",
    )
    with pytest.raises(ValueError, match="alias group"):
        build_reference_template_from_atoms(
            _atoms(1), config=_config_111(), phase_specification=specification
        )


@pytest.mark.parametrize(
    ("modes", "message"),
    (
        (torch.eye(3, dtype=torch.bool), "bool"),
        (
            torch.tensor(
                [[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            "fractional",
        ),
        (
            torch.tensor(
                [[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            "NaN or Inf",
        ),
        (
            torch.tensor(
                [[float("inf"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            "NaN or Inf",
        ),
        (
            torch.tensor(
                [[2**63, 0, 0], [0, 1, 0], [0, 0, 1]],
                dtype=torch.uint64,
            ),
            "represented exactly",
        ),
    ),
)
def test_phase_modes_reject_noninteger_bool_and_nonfinite_before_cast(
    modes, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        PhaseSpecification(
            modes,
            torch.ones(3),
            torch.eye(2),
            torch.ones(2),
            "provisional",
        )


def test_phase_payload_fractional_mode_is_not_silently_truncated():
    payload = _phase(1).to_dict()
    payload["modes"][0][0] = 0.25
    with pytest.raises(ValueError, match="fractional"):
        PhaseSpecification.from_dict(payload)

    integral_float = torch.eye(3, dtype=torch.float64)
    phase = PhaseSpecification(
        integral_float,
        torch.ones(3),
        torch.eye(2),
        torch.ones(2),
        "provisional",
    )
    assert phase.modes.dtype == torch.long
    assert torch.equal(phase.modes, torch.eye(3, dtype=torch.long))


def test_typed_stabilizer_sizes_and_canonical_permutation_consistency():
    for size, expected in ((2, 32), (3, 108)):
        canonical = canonicalize_reference_atoms(
            _atoms(size),
            nbc_rocksalt_template_builder_config((size, size, size)),
        )
        stabilizer = find_typed_stabilizer(
            canonical.fractional_positions,
            canonical.site_types,
            tolerance=1.0e-10,
        )
        assert stabilizer.translations.shape == (expected, 3)
        assert stabilizer.permutations.shape == (
            expected,
            canonical.fractional_positions.shape[0],
        )
        expected_indices = torch.arange(canonical.fractional_positions.shape[0])
        for translation, permutation in zip(
            stabilizer.translations, stabilizer.permutations
        ):
            assert torch.equal(torch.sort(permutation).values, expected_indices)
            translated = canonical.fractional_positions + translation
            mapped = canonical.fractional_positions[permutation]
            assert float(
                torch.linalg.vector_norm(
                    torus_difference(translated, mapped), dim=-1
                ).max()
            ) <= 1.0e-10
            assert torch.equal(
                canonical.site_types,
                canonical.site_types[permutation],
            )


def _legacy_template():
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
    return ReferenceTemplate.snapshot(
        "legacy-regression",
        topology,
        torch.eye(3, dtype=torch.long),
        torch.ones(3, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        find_typed_stabilizer(fractional, site_types),
        (6, 41),
    )


def test_legacy_fingerprint_behavior_and_strict_fingerprint_sensitivity(built_111):
    legacy = _legacy_template()
    assert legacy.strict_domain is None
    assert (
        legacy.fingerprint
        == "1db045471563fe8cfe4dc037c9ef8d9a1c3bf48c61acafb63fcf0447edac2b13"
    )
    legacy.validate_structure(torch.tensor([6], dtype=torch.long))

    template = built_111.template
    pristine_only = replace(
        template.strict_domain,
        allowed_compositions=((4, 4),),
        allowed_num_atoms=(8,),
        allowed_vacancy_masses=(0,),
    )
    changed = ReferenceTemplate.snapshot(
        template.template_id,
        template.topology,
        template.phase_modes,
        template.phase_mode_weights,
        template.site_alignment_weights,
        template.phase_channel_weights,
        template.stabilizer,
        template.supported_species,
        template.convention_version,
        pristine_only,
    )
    assert changed.fingerprint != template.fingerprint


def _sample(template, *, vacancy=False, cell=None, sample_id="sample"):
    numbers = torch.where(
        template.topology.site_types == 0,
        torch.tensor(6, dtype=torch.long),
        torch.tensor(41, dtype=torch.long),
    )
    positions = template.topology.reference_fractional @ template.topology.reference_cell
    if vacancy:
        index = int(torch.nonzero(numbers == 6)[0])
        keep = torch.arange(numbers.numel()) != index
        numbers = numbers[keep]
        positions = positions[keep]
    return StructureSample(
        sample_id=sample_id,
        positions=positions.clone(),
        atomic_numbers=numbers.clone(),
        cell=(template.topology.reference_cell if cell is None else cell).clone(),
        pbc=torch.ones(3, dtype=torch.bool),
        origin=torch.zeros(3, dtype=torch.float64),
        template_id=template.template_id,
    )


def test_registry_dataset_batch_context_and_strain_contract(built_111):
    template = built_111.template.clone()
    registry = TemplateRegistry()
    registry.add(template)
    registry_fingerprint = registry.fingerprint
    template.topology.reference_fractional[0, 0] += 0.125
    assert registry.fingerprint == registry_fingerprint
    resolved = registry.resolve("nbc_rocksalt_111_test")

    pristine = _sample(resolved, sample_id="pristine")
    vacancy = _sample(resolved, vacancy=True, sample_id="vacancy")
    dataset = InMemoryStructureDataset((pristine, vacancy), registry)
    batch = collate_structure_samples(dataset, registry)
    assert batch.atom_ptr.tolist() == [0, 8, 15]
    assert batch.template_fingerprints == (
        resolved.fingerprint,
        resolved.fingerprint,
    )

    bad_numbers = pristine.atomic_numbers.clone()
    bad_numbers[int(torch.nonzero(bad_numbers == 41)[0])] = 6
    with pytest.raises(ValueError, match="composition"):
        InMemoryStructureDataset(
            (replace(pristine, sample_id="nb-vacancy", atomic_numbers=bad_numbers),),
            registry,
        )
    strained_cell = resolved.topology.reference_cell @ torch.diag(
        torch.tensor([1.2, 1.0, 1.0], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="strain domain"):
        InMemoryStructureDataset(
            (_sample(resolved, cell=strained_cell, sample_id="strained"),), registry
        )

    context = TemplateExecutionContext.from_reference_template(
        resolved, avg_num_neighbors=built_111.config.avg_num_neighbors
    )
    assert context.strict_domain == resolved.strict_domain
    context.validate_fingerprint()
    materialized = context.materialize(device="cpu", dtype=torch.float32)
    assert materialized.topology.reference_fractional.dtype == torch.float32
    assert materialized.strict_domain == resolved.strict_domain
    if torch.cuda.is_available():
        cuda = context.materialize(device="cuda", dtype=torch.float64)
        assert cuda.topology.reference_fractional.device.type == "cuda"
        assert cuda.strict_domain == resolved.strict_domain


def test_builder_graph_and_diagnostics_contract(built_111):
    diagnostics = built_111.diagnostics
    assert diagnostics.active_edge_count == 8 * 6
    assert diagnostics.candidate_edge_count == 8 * 18
    assert diagnostics.active_degree_min == diagnostics.active_degree_max == 6
    assert diagnostics.candidate_degree_min == diagnostics.candidate_degree_max == 18
    assert diagnostics.stabilizer_size == 4
    assert diagnostics.phase_rank == 3
    assert diagnostics.phase_approval_status == "provisional"
    assert diagnostics.minimum_edge_length == pytest.approx(LATTICE / 2.0)
    assert diagnostics.fingerprint == built_111.template.fingerprint
    assert diagnostics.graph_build_seconds > 0.0
    assert diagnostics.total_build_seconds >= diagnostics.graph_build_seconds
