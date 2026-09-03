from __future__ import annotations

import copy
from dataclasses import replace
import os
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceStructureArtifactError,
    ReferenceTemplate,
    StrictTemplateDomain,
    TemplateRegistry,
    assemble_reference_template_from_artifact,
    build_reference_template_from_atoms,
    capture_reference_structure_artifact,
    load_reference_structure_artifact,
    nbc_rocksalt_template_builder_config,
    save_reference_structure_artifact,
)
from refsite_mlip.graph import ReferenceGraphTopology, build_reference_graph_topology
from refsite_mlip.models import TemplateExecutionContext
from refsite_mlip.phase import find_typed_stabilizer
from refsite_mlip.phase.types import TypedStabilizer


def _phase(*, scale: float = 1.0) -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.eye(3, dtype=torch.long),
        mode_weights=torch.full((3,), scale, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
        convention_version="synthetic_phase_provenance_v1",
    )


def _tiny_template(*, template_id: str = "artifact-tiny", strict: bool = True):
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
    domain = (
        StrictTemplateDomain(
            reference_site_count=2,
            supercell_shape=(1, 1, 1),
            species_vocabulary=(6, 41),
            reference_composition=(1, 1),
            allowed_compositions=((1, 1), (0, 1)),
            allowed_num_atoms=(2, 1),
            allowed_vacancy_masses=(0, 1),
        )
        if strict
        else None
    )
    phase = _phase()
    return ReferenceTemplate.snapshot(
        template_id,
        topology,
        phase.modes,
        phase.mode_weights,
        phase.site_type_alignment_weights[site_types],
        phase.channel_weights,
        find_typed_stabilizer(fractional, site_types),
        (6, 41),
        strict_domain=domain,
    )


def _style_template(size: int) -> ReferenceTemplate:
    side = 2 * size
    num_sites = side**3
    grid = torch.stack(
        torch.meshgrid(
            *(torch.arange(side, dtype=torch.float64) / side for _ in range(3)),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    half = num_sites // 2
    site_types = torch.cat(
        (torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long))
    )
    keys = set()
    for target in range(num_sites):
        for source in ((target - 1) % num_sites, (target + 1) % num_sites):
            keys.add((target, source, 0, 0, 0))
    ordered = sorted(keys)
    edge_index = torch.tensor(
        [[entry[1] for entry in ordered], [entry[0] for entry in ordered]],
        dtype=torch.long,
    )
    shifts = torch.tensor([entry[2:] for entry in ordered], dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64)
    topology = ReferenceGraphTopology(
        reference_fractional=grid,
        site_types=site_types,
        edge_index=edge_index,
        shifts=shifts,
        reference_cell=cell,
        cutoff=2.0,
        skin=0.5,
        maximum_strain=0.1,
        minimum_edge_length=1.0e-8,
        pbc=(True, True, True),
    )
    domain = StrictTemplateDomain(
        reference_site_count=num_sites,
        supercell_shape=(size, size, size),
        species_vocabulary=(6, 41),
        reference_composition=(half, half),
        allowed_compositions=((half, half), (half - 1, half)),
        allowed_num_atoms=(num_sites, num_sites - 1),
        allowed_vacancy_masses=(0, 1),
    )
    phase = _phase()
    stabilizer = TypedStabilizer(
        torch.zeros((1, 3), dtype=torch.float64),
        torch.arange(num_sites, dtype=torch.long).unsqueeze(0),
    )
    return ReferenceTemplate.snapshot(
        f"synthetic-{size}{size}{size}",
        topology,
        phase.modes,
        phase.mode_weights,
        phase.site_type_alignment_weights[site_types],
        phase.channel_weights,
        stabilizer,
        (6, 41),
        strict_domain=domain,
    )


def _assert_safe_plain(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
        return
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _assert_safe_plain(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_safe_plain(item)
        return
    assert value is None or isinstance(value, (str, int, float, bool))


def _assert_payload_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_payload_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_payload_equal(first, second)
    else:
        assert left == right


@pytest.fixture
def tiny_artifact():
    return capture_reference_structure_artifact(
        _tiny_template(), avg_num_neighbors=6.0
    )


def test_capture_is_phase_independent_owned_and_detects_mutation():
    direct = _tiny_template()
    artifact = capture_reference_structure_artifact(
        direct, avg_num_neighbors=6.0
    )
    changed_phase = _phase(scale=2.0)
    changed_template = ReferenceTemplate.snapshot(
        direct.template_id,
        direct.topology,
        changed_phase.modes,
        changed_phase.mode_weights,
        changed_phase.site_type_alignment_weights[direct.topology.site_types],
        changed_phase.channel_weights,
        direct.stabilizer,
        direct.supported_species,
        direct.convention_version,
        direct.strict_domain,
    )
    changed_artifact = capture_reference_structure_artifact(
        changed_template, avg_num_neighbors=6.0
    )
    assert artifact.structural_fingerprint == changed_artifact.structural_fingerprint
    assert artifact.to_payload()["payload"].keys().isdisjoint(
        {"phase_modes", "phase_mode_weights", "alignment_weights"}
    )

    original = artifact.reference_fractional.clone()
    direct.topology.reference_fractional[0, 0] += 0.125
    assert torch.equal(artifact.reference_fractional, original)
    assert all(
        tensor.device.type == "cpu" and not tensor.requires_grad
        for tensor in (
            artifact.reference_fractional,
            artifact.reference_cell,
            artifact.edge_index,
            artifact.stabilizer_translations,
        )
    )
    artifact.reference_fractional[0, 0] += 0.01
    with pytest.raises(ReferenceStructureArtifactError) as caught:
        artifact.validate()
    assert caught.value.reason_code == "FINGERPRINT_MISMATCH"


def test_bare_template_requires_explicit_average_neighbor_convention():
    with pytest.raises(ValueError, match="avg_num_neighbors"):
        capture_reference_structure_artifact(_tiny_template())


@pytest.mark.parametrize("size,expected", [(2, 64), (3, 216)])
def test_synthetic_222_333_style_schema_counts_and_registry(size, expected):
    template = _style_template(size)
    artifact = capture_reference_structure_artifact(
        template, avg_num_neighbors=6.0
    )
    diagnostics = artifact.diagnostics
    assert diagnostics.num_sites == expected
    assert diagnostics.num_edges == 2 * expected
    assert diagnostics.stabilizer_size == 1
    assert artifact.strict_domain.reference_composition == (
        expected // 2,
        expected // 2,
    )
    assembled = assemble_reference_template_from_artifact(
        artifact, phase_specification=_phase()
    )
    assert assembled.fingerprint == template.fingerprint
    registry = TemplateRegistry()
    registry.add(assembled)
    resolved = registry.resolve(template.template_id)
    assert resolved.fingerprint == template.fingerprint
    context = TemplateExecutionContext.from_reference_template(
        resolved, avg_num_neighbors=artifact.avg_num_neighbors
    )
    assert context.fingerprint == template.fingerprint


def test_safe_payload_save_load_relocation_and_semantic_determinism(
    tmp_path, tiny_artifact
):
    payload = tiny_artifact.to_payload()
    _assert_safe_plain(payload)
    first = tmp_path / "first.pt"
    nested = tmp_path / "moved"
    nested.mkdir()
    second = nested / "renamed.pt"
    save_reference_structure_artifact(first, tiny_artifact)
    shutil.copyfile(first, second)
    loaded_first = load_reference_structure_artifact(first)
    loaded_second = load_reference_structure_artifact(second)
    assert loaded_first.structural_fingerprint == tiny_artifact.structural_fingerprint
    assert loaded_second.structural_fingerprint == tiny_artifact.structural_fingerprint
    _assert_payload_equal(loaded_first.to_payload(), loaded_second.to_payload())
    raw = torch.load(first, map_location="cpu", weights_only=True)
    _assert_payload_equal(raw, payload)

    third = tmp_path / "third.pt"
    save_reference_structure_artifact(third, loaded_first)
    reloaded = load_reference_structure_artifact(third)
    _assert_payload_equal(reloaded.to_payload(), loaded_first.to_payload())


def test_direct_builder_and_assembled_template_fingerprint_exact_parity():
    ase = pytest.importorskip("ase")
    from ase.build import bulk

    atoms = bulk("NbC", "rocksalt", a=4.482314244155584, cubic=True)
    config = replace(
        nbc_rocksalt_template_builder_config((2, 2, 2)),
        template_id="builder-artifact-111",
        strict_domain=StrictTemplateDomain(
            reference_site_count=8,
            supercell_shape=(1, 1, 1),
            species_vocabulary=(6, 41),
            reference_composition=(4, 4),
            allowed_compositions=((4, 4), (3, 4)),
            allowed_num_atoms=(8, 7),
            allowed_vacancy_masses=(0, 1),
        ),
        expected_stabilizer_size=4,
    )
    phase = PhaseSpecification(
        modes=torch.tensor(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ],
            dtype=torch.long,
        ),
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
        convention_version="builder_test_phase_provenance_v1",
    )
    built = build_reference_template_from_atoms(
        atoms, config=config, phase_specification=phase
    )
    artifact = capture_reference_structure_artifact(built)
    assembled = assemble_reference_template_from_artifact(
        artifact, phase_specification=phase
    )
    assert assembled.fingerprint == built.template.fingerprint
    for left, right in (
        (assembled.topology.reference_fractional, built.template.topology.reference_fractional),
        (assembled.topology.site_types, built.template.topology.site_types),
        (assembled.topology.edge_index, built.template.topology.edge_index),
        (assembled.topology.shifts, built.template.topology.shifts),
        (assembled.stabilizer.translations, built.template.stabilizer.translations),
        (assembled.stabilizer.permutations, built.template.stabilizer.permutations),
    ):
        assert torch.equal(left, right)

    phase_b = replace(phase, mode_weights=phase.mode_weights * 1.5)
    assembled_b = assemble_reference_template_from_artifact(
        artifact, phase_specification=phase_b
    )
    assert artifact.structural_fingerprint == capture_reference_structure_artifact(
        built
    ).structural_fingerprint
    assert assembled_b.fingerprint != assembled.fingerprint


def test_legacy_template_fingerprint_is_unchanged_by_artifact_api():
    template = _tiny_template(template_id="legacy-regression", strict=False)
    assert (
        template.fingerprint
        == "1db045471563fe8cfe4dc037c9ef8d9a1c3bf48c61acafb63fcf0447edac2b13"
    )
    artifact = capture_reference_structure_artifact(
        template, avg_num_neighbors=6.0
    )
    assert artifact.strict_domain is None
    restored = assemble_reference_template_from_artifact(
        artifact, phase_specification=_phase()
    )
    assert restored.fingerprint == template.fingerprint


def test_phase_is_required_and_validated_against_site_order_and_stabilizer(
    tiny_artifact,
):
    with pytest.raises(ReferenceStructureArtifactError) as missing:
        assemble_reference_template_from_artifact(
            tiny_artifact, phase_specification=None
        )
    assert missing.value.reason_code == "PHASE_SPECIFICATION_REQUIRED"
    wrong_rows = PhaseSpecification(
        torch.eye(3, dtype=torch.long),
        torch.ones(3),
        torch.ones((1, 1)),
        torch.ones(1),
        "provisional",
    )
    with pytest.raises(ReferenceStructureArtifactError) as rows:
        assemble_reference_template_from_artifact(
            tiny_artifact, phase_specification=wrong_rows
        )
    assert rows.value.reason_code == "PHASE_SITE_TYPE_MISMATCH"

    style = capture_reference_structure_artifact(
        _style_template(2), avg_num_neighbors=6.0
    )
    wrong_alias = PhaseSpecification(
        torch.diag(torch.tensor([2, 1, 1], dtype=torch.long)),
        torch.ones(3),
        torch.eye(2),
        torch.ones(2),
        "provisional",
    )
    with pytest.raises(ReferenceStructureArtifactError) as alias:
        assemble_reference_template_from_artifact(
            style, phase_specification=wrong_alias
        )
    assert alias.value.reason_code == "PHASE_STABILIZER_MISMATCH"


def test_load_and_assembly_do_not_call_any_builder(monkeypatch, tmp_path, tiny_artifact):
    path = tmp_path / "artifact.pt"
    save_reference_structure_artifact(path, tiny_artifact)

    def forbidden(*args, **kwargs):
        raise AssertionError("structural builder called during load/assembly")

    import refsite_mlip.data.reference_builder as builder

    monkeypatch.setattr(builder, "build_reference_template_from_atoms", forbidden)
    monkeypatch.setattr(builder, "build_reference_template_from_poscar", forbidden)
    monkeypatch.setattr(builder, "canonicalize_reference_atoms", forbidden)
    monkeypatch.setattr(builder, "build_reference_graph_topology", forbidden)
    monkeypatch.setattr(builder, "find_typed_stabilizer", forbidden)
    loaded = load_reference_structure_artifact(path)
    assembled = assemble_reference_template_from_artifact(
        loaded, phase_specification=_phase()
    )
    assert assembled.fingerprint == _tiny_template().fingerprint


def _write_payload(path: Path, payload) -> None:
    torch.save(payload, path)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda value: value.update(schema_version="future_v99"), "UNSUPPORTED_SCHEMA"),
        (lambda value: value.update(artifact_scope="model_bundle"), "INVALID_SCOPE"),
        (lambda value: value.pop("structural_fingerprint"), "INVALID_PAYLOAD_KEYS"),
        (lambda value: value.update(extra=1), "INVALID_PAYLOAD_KEYS"),
        (
            lambda value: value["payload"].update(extra=1),
            "INVALID_PAYLOAD_KEYS",
        ),
        (
            lambda value: value["payload"]["topology"]["reference_fractional"].add_(0.01),
            "FINGERPRINT_MISMATCH",
        ),
        (
            lambda value: value["payload"]["topology"].update(
                reference_cell=value["payload"]["topology"]["reference_cell"].float()
            ),
            "DTYPE_MISMATCH",
        ),
        (
            lambda value: value["payload"]["topology"].update(
                site_types=value["payload"]["topology"]["site_types"][:-1]
            ),
            "INVALID_SITE_TYPES",
        ),
        (
            lambda value: value["payload"]["topology"].update(pbc=[1, 1, 1]),
            "PBC_REQUIRED",
        ),
        (
            lambda value: value["payload"]["structural_metadata"].update(
                num_sites=2.0
            ),
            "INVALID_METADATA",
        ),
        (
            lambda value: value["payload"]["topology"]["edge_index"].__setitem__((0, 0), 99),
            "EDGE_INDEX_RANGE",
        ),
        (
            lambda value: value["payload"]["stabilizer"]["permutations"].__setitem__((0, 0), 1),
            "INVALID_STABILIZER",
        ),
        (
            lambda value: value.update(structural_fingerprint="0" * 64),
            "FINGERPRINT_MISMATCH",
        ),
    ],
)
def test_corrupt_payload_is_rejected_before_assembly(
    tmp_path, tiny_artifact, mutation, reason
):
    payload = copy.deepcopy(tiny_artifact.to_payload())
    mutation(payload)
    path = tmp_path / f"corrupt-{reason}.pt"
    _write_payload(path, payload)
    with pytest.raises(ReferenceStructureArtifactError) as caught:
        load_reference_structure_artifact(path)
    assert caught.value.reason_code == reason
    assert caught.value.artifact_path == str(path)
    assert caught.value.validation_stage is not None


def test_invalid_domain_truncated_and_non_torch_files_fail_safely(
    tmp_path, tiny_artifact
):
    payload = copy.deepcopy(tiny_artifact.to_payload())
    payload["payload"]["strict_domain"]["reference_site_count"] = 3
    domain_path = tmp_path / "domain.pt"
    _write_payload(domain_path, payload)
    with pytest.raises(ReferenceStructureArtifactError) as domain:
        load_reference_structure_artifact(domain_path)
    assert domain.value.reason_code == "INVALID_DOMAIN"

    for name, content in (("truncated.pt", b"PK\x03\x04"), ("plain.txt", b"not torch")):
        path = tmp_path / name
        path.write_bytes(content)
        with pytest.raises(ReferenceStructureArtifactError) as caught:
            load_reference_structure_artifact(path)
        assert caught.value.reason_code == "SAFE_LOAD_FAILURE"


def test_loader_calls_torch_load_with_weights_only_true(
    monkeypatch, tmp_path, tiny_artifact
):
    import refsite_mlip.data.reference_artifact as module

    path = tmp_path / "artifact.pt"
    save_reference_structure_artifact(path, tiny_artifact)
    real_load = module.torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(module.torch, "load", recording_load)
    loaded = load_reference_structure_artifact(path, map_location="cpu")
    assert loaded.structural_fingerprint == tiny_artifact.structural_fingerprint
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"


def test_atomic_overwrite_save_and_replace_failure_preserve_target(
    monkeypatch, tmp_path, tiny_artifact
):
    import refsite_mlip.data.reference_artifact as module

    target = tmp_path / "artifact.pt"
    save_reference_structure_artifact(target, tiny_artifact)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        save_reference_structure_artifact(target, tiny_artifact)

    real_save = torch.save

    def fail_save(*args, **kwargs):
        raise OSError("injected torch.save failure")

    monkeypatch.setattr(module.torch, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        save_reference_structure_artifact(target, tiny_artifact, overwrite=True)
    assert target.read_bytes() == original
    assert list(tmp_path.glob(".artifact.pt.*.tmp")) == []
    monkeypatch.setattr(module.torch, "save", real_save)

    def fail_replace(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace"):
        save_reference_structure_artifact(target, tiny_artifact, overwrite=True)
    assert target.read_bytes() == original
    assert list(tmp_path.glob(".artifact.pt.*.tmp")) == []


def test_symlink_is_rejected_for_save_and_load(tmp_path, tiny_artifact):
    target = tmp_path / "target.pt"
    save_reference_structure_artifact(target, tiny_artifact)
    link = tmp_path / "link.pt"
    link.symlink_to(target.name)
    with pytest.raises(ValueError, match="symbolic link"):
        save_reference_structure_artifact(link, tiny_artifact, overwrite=True)
    with pytest.raises(ReferenceStructureArtifactError) as caught:
        load_reference_structure_artifact(link)
    assert caught.value.reason_code == "SYMLINK_REJECTED"


def test_no_overwrite_race_never_clobbers_competing_artifact(
    tmp_path, tiny_artifact, monkeypatch
):
    import refsite_mlip.data.reference_artifact as module

    target = tmp_path / "raced-artifact.pt"

    def competing_link(source, destination, *args, **kwargs):
        del source, args, kwargs
        path = type(target)(destination)
        path.write_bytes(b"competitor")
        raise FileExistsError(f"competing artifact won: {path}")

    monkeypatch.setattr(module.os, "link", competing_link)
    with pytest.raises(FileExistsError, match="competing artifact"):
        save_reference_structure_artifact(
            target, tiny_artifact, overwrite=False
        )
    assert target.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".raced-artifact.pt.*.tmp"))


def test_weights_only_safe_globals_and_rng_are_unchanged(tmp_path, tiny_artifact):
    path = tmp_path / "artifact.pt"
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = (
        tuple(value.clone() for value in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    safe_globals = tuple(torch.serialization.get_safe_globals())
    save_reference_structure_artifact(path, tiny_artifact)
    loaded = load_reference_structure_artifact(path)
    assert loaded.structural_fingerprint == tiny_artifact.structural_fingerprint
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), cpu_state)
    if cuda_state:
        assert all(
            torch.equal(left, right)
            for left, right in zip(torch.cuda.get_rng_state_all(), cuda_state)
        )
    assert tuple(torch.serialization.get_safe_globals()) == safe_globals


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_context_materialization_dtype_and_optional_cuda(tiny_artifact, dtype):
    assembled = assemble_reference_template_from_artifact(
        tiny_artifact, phase_specification=_phase()
    )
    context = TemplateExecutionContext.from_reference_template(
        assembled, avg_num_neighbors=tiny_artifact.avg_num_neighbors
    )
    cpu = context.materialize(device="cpu", dtype=dtype)
    assert cpu.topology.reference_fractional.dtype == dtype
    assert cpu.topology.edge_index.dtype == torch.long
    if torch.cuda.is_available():
        cuda = context.materialize(device="cuda", dtype=dtype)
        assert cuda.topology.reference_fractional.device.type == "cuda"
        assert cuda.topology.reference_fractional.dtype == dtype
        assert cuda.topology.edge_index.dtype == torch.long
