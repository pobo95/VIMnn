from __future__ import annotations

from dataclasses import replace

import torch

from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceTemplate,
    StructureSample,
    TemplateRegistry,
    assemble_reference_template_from_artifact,
    capture_reference_structure_artifact,
    collate_structure_samples,
    load_reference_structure_artifact,
    save_reference_structure_artifact,
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
from refsite_mlip.phase import find_typed_stabilizer


AVG_NUM_NEIGHBORS = 6.0


def _phase(data, mode_count):
    return PhaseSpecification(
        modes=data["modes"][:mode_count],
        mode_weights=data["mode_weights"][:mode_count],
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=data["channel_weights"],
        approval_status="provisional",
        convention_version="artifact_model_test_phase_v1",
    )


def _template(data, template_id, site_count, mode_count):
    topology = build_reference_graph_topology(
        data["sites"][:site_count],
        data["site_types"][:site_count],
        data["cell"],
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    phase = _phase(data, mode_count)
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


def _configuration():
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=2,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        probability_tolerance=1.0e-7,
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
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
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
    torch.manual_seed(20260802)
    return ReferenceSitePotential(
        _configuration(),
        template.topology,
        template.phase_modes,
        template.phase_mode_weights,
        torch.eye(2, dtype=torch.float64),
        template.site_alignment_weights,
        template.phase_channel_weights,
        (-1.0, 2.0),
    ).to(dtype=torch.float64)


def _numbers(data, count):
    return torch.where(
        data["site_types"][:count] == 0,
        torch.tensor(6, dtype=torch.long),
        torch.tensor(41, dtype=torch.long),
    )


def _context(template):
    return TemplateExecutionContext.from_reference_template(
        template, avg_num_neighbors=AVG_NUM_NEIGHBORS
    )


def _assert_direct_equal(left, right):
    for first, second in (
        (left.energy, right.energy),
        (left.site_energy, right.site_energy),
        (left.forces, right.forces),
        (left.stress, right.stress),
        (left.stress_voigt, right.stress_voigt),
        (left.raw_c, right.raw_c),
        (left.site_features, right.site_features),
        (left.auxiliary["phase"], right.auxiliary["phase"]),
        (left.auxiliary["ot"].P, right.auxiliary["ot"].P),
        (left.auxiliary["ot"].q, right.auxiliary["ot"].q),
        (
            left.auxiliary["multipoles"].equivariant_features,
            right.auxiliary["multipoles"].equivariant_features,
        ),
    ):
        assert torch.equal(first, second)


def test_loaded_assembled_template_direct_model_bitwise_parity(
    typed_crystal, tmp_path
):
    direct, phase = _template(typed_crystal, "artifact-zeta", 6, 5)
    artifact = capture_reference_structure_artifact(
        direct, avg_num_neighbors=AVG_NUM_NEIGHBORS
    )
    path = tmp_path / "zeta.pt"
    save_reference_structure_artifact(path, artifact)
    loaded = load_reference_structure_artifact(path)
    assembled = assemble_reference_template_from_artifact(
        loaded, phase_specification=phase
    )
    assert assembled.fingerprint == direct.fingerprint
    direct_context = _context(direct)
    assembled_context = _context(assembled)
    for first, second in (
        (direct_context.reference_fractional, assembled_context.reference_fractional),
        (direct_context.site_types, assembled_context.site_types),
        (direct_context.edge_index, assembled_context.edge_index),
        (direct_context.shifts, assembled_context.shifts),
        (direct_context.phase_modes, assembled_context.phase_modes),
        (direct_context.stabilizer.translations, assembled_context.stabilizer.translations),
        (direct_context.stabilizer.permutations, assembled_context.stabilizer.permutations),
    ):
        assert torch.equal(first, second)

    model = _model(direct)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    numbers = _numbers(typed_crystal, 5)
    first_positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    second_positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    first = model(
        first_positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
        template_context=direct_context,
    )
    second = model(
        second_positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
        template_context=assembled_context,
    )
    _assert_direct_equal(first, second)
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert tuple(model.state_dict()) == tuple(state_before)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key])


def _sample(data, sample_id, template_id, count):
    return StructureSample(
        sample_id=sample_id,
        positions=data["positions"][:count].clone(),
        atomic_numbers=_numbers(data, count),
        cell=data["cell"].clone(),
        pbc=torch.ones(3, dtype=torch.bool),
        origin=data["origin"].clone(),
        template_id=template_id,
    )


def test_loaded_artifacts_mixed_template_grouped_bitwise_parity(
    typed_crystal, tmp_path
):
    direct_templates = {}
    assembled_templates = {}
    for template_id, site_count, mode_count in (
        ("alpha", 4, 4),
        ("zeta", 6, 5),
    ):
        direct, phase = _template(
            typed_crystal, template_id, site_count, mode_count
        )
        direct_templates[template_id] = direct
        artifact = capture_reference_structure_artifact(
            direct, avg_num_neighbors=AVG_NUM_NEIGHBORS
        )
        path = tmp_path / f"{template_id}.pt"
        save_reference_structure_artifact(path, artifact)
        assembled_templates[template_id] = assemble_reference_template_from_artifact(
            load_reference_structure_artifact(path), phase_specification=phase
        )
        assert assembled_templates[template_id].fingerprint == direct.fingerprint

    direct_registry = TemplateRegistry()
    assembled_registry = TemplateRegistry()
    for template_id in ("alpha", "zeta"):
        direct_registry.add(direct_templates[template_id])
        assembled_registry.add(assembled_templates[template_id])
    assert direct_registry.fingerprint == assembled_registry.fingerprint
    samples = (
        _sample(typed_crystal, "zeta-vacancy", "zeta", 5),
        _sample(typed_crystal, "alpha-pristine", "alpha", 4),
        _sample(typed_crystal, "zeta-pristine", "zeta", 6),
    )
    direct_batch = collate_structure_samples(samples, direct_registry)
    assembled_batch = collate_structure_samples(samples, assembled_registry)
    direct_batch = replace(
        direct_batch,
        positions=direct_batch.positions.detach().clone().requires_grad_(True),
    )
    assembled_batch = replace(
        assembled_batch,
        positions=assembled_batch.positions.detach().clone().requires_grad_(True),
    )
    direct_contexts = {
        key: _context(value) for key, value in direct_templates.items()
    }
    assembled_contexts = {
        key: _context(value) for key, value in assembled_templates.items()
    }
    model = _model(direct_templates["zeta"])
    direct = evaluate_structure_batch(
        model,
        direct_batch,
        direct_contexts,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assembled = evaluate_structure_batch(
        model,
        assembled_batch,
        assembled_contexts,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    for first, second in (
        (direct.energy, assembled.energy),
        (direct.site_energy, assembled.site_energy),
        (direct.forces, assembled.forces),
        (direct.stress, assembled.stress),
        (direct.stress_voigt, assembled.stress_voigt),
    ):
        assert torch.equal(first, second)
    assert direct.sample_ids == assembled.sample_ids
    assert direct.site_ptr.tolist() == assembled.site_ptr.tolist()
    for first, second in zip(direct.auxiliary, assembled.auxiliary):
        for left, right in (
            (first["phase"], second["phase"]),
            (first["ot"].P, second["ot"].P),
            (first["ot"].q, second["ot"].q),
            (
                first["multipoles"].equivariant_features,
                second["multipoles"].equivariant_features,
            ),
        ):
            assert torch.equal(left, right)
