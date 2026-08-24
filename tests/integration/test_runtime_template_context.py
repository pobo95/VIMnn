from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from refsite_mlip.data import ReferenceTemplate
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import (
    PotentialConfig,
    ReferenceSitePotential,
    TemplateExecutionContext,
)
from refsite_mlip.phase.stabilizer import find_typed_stabilizer


AVG_NUM_NEIGHBORS = 6.0


def cast_data(data, dtype):
    return {
        key: value.to(dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in data.items()
    }


def make_config(dtype):
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


def make_topology(data, site_count=None, cutoff=3.0, pbc=(True, True, True)):
    count = data["sites"].shape[0] if site_count is None else site_count
    return build_reference_graph_topology(
        data["sites"][:count],
        data["site_types"][:count],
        data["cell"],
        cutoff=cutoff,
        skin=0.5,
        maximum_strain=0.1,
        pbc=pbc,
    )


def make_template(
    data,
    *,
    template_id="runtime",
    site_count=None,
    mode_count=None,
    cutoff=3.0,
    supported_species=(6, 41),
    site_types=None,
    site_alignment_weights=None,
    phase_channel_weights=None,
    convention_version="reference_template_v1",
):
    count = data["sites"].shape[0] if site_count is None else site_count
    topology = make_topology(data, count, cutoff=cutoff)
    if site_types is not None:
        topology = replace(topology, site_types=site_types.clone())
    modes = data["modes"] if mode_count is None else data["modes"][:mode_count]
    mode_weights = (
        data["mode_weights"]
        if mode_count is None
        else data["mode_weights"][:mode_count]
    )
    alignment = (
        data["site_weights"][:count]
        if site_alignment_weights is None
        else site_alignment_weights
    )
    channel_weights = (
        data["channel_weights"]
        if phase_channel_weights is None
        else phase_channel_weights
    )
    return ReferenceTemplate.snapshot(
        template_id,
        topology,
        modes,
        mode_weights,
        alignment,
        channel_weights,
        find_typed_stabilizer(
            topology.reference_fractional, topology.site_types
        ),
        supported_species,
        convention_version,
    )


def make_model_and_template(data):
    config = make_config(data["cell"].dtype)
    template = make_template(data, template_id="default")
    model = ReferenceSitePotential(
        config,
        template.topology,
        template.phase_modes,
        template.phase_mode_weights,
        torch.eye(2, dtype=data["cell"].dtype),
        template.site_alignment_weights,
        template.phase_channel_weights,
        (-1.0, 2.0),
    ).to(data["cell"])
    return model, template


def make_context(template, avg_num_neighbors=AVG_NUM_NEIGHBORS):
    return TemplateExecutionContext.from_reference_template(
        template, avg_num_neighbors=avg_num_neighbors
    )


def numbers(data, count):
    return torch.tensor(
        [6 if int(value) == 0 else 41 for value in data["site_types"][:count]],
        dtype=torch.long,
        device=data["positions"].device,
    )


def assert_output_equal(default, runtime):
    for left, right in (
        (default.auxiliary["phase"], runtime.auxiliary["phase"]),
        (default.auxiliary["ot"].P, runtime.auxiliary["ot"].P),
        (default.auxiliary["ot"].q, runtime.auxiliary["ot"].q),
        (
            default.auxiliary["multipoles"].equivariant_features,
            runtime.auxiliary["multipoles"].equivariant_features,
        ),
        (default.site_energy, runtime.site_energy),
        (default.energy, runtime.energy),
        (default.forces, runtime.forces),
        (default.stress, runtime.stress),
    ):
        assert torch.equal(left, right)


def test_context_owns_an_immutable_cpu_snapshot_and_materializes(typed_crystal):
    template = make_template(typed_crystal)
    context = make_context(template)
    original = context.phase_mode_weights.clone()
    template.phase_mode_weights[0] += 20.0

    assert not isinstance(context, torch.nn.Module)
    assert torch.equal(context.phase_mode_weights, original)
    context.validate_fingerprint()
    materialized = context.materialize(device="cpu", dtype=torch.float32)
    assert materialized.topology.reference_fractional.dtype == torch.float32
    assert materialized.phase_mode_weights.dtype == torch.float32
    assert materialized.phase_modes.dtype == torch.long
    assert materialized.topology.edge_index.dtype == torch.long
    assert materialized.stabilizer.permutations.dtype == torch.long


def test_default_context_phase_ot_feature_energy_force_stress_bitwise_parity(
    typed_crystal,
):
    model, template = make_model_and_template(typed_crystal)
    context = make_context(template)
    atomic_numbers = numbers(typed_crystal, 5)
    default_positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    runtime_positions = typed_crystal["positions"][:5].clone().requires_grad_(True)

    default = model(
        default_positions,
        atomic_numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    runtime = model(
        runtime_positions,
        atomic_numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
        template_context=context,
    )
    assert_output_equal(default, runtime)


def test_one_model_runs_different_m_edges_and_mode_counts(typed_crystal):
    model, default_template = make_model_and_template(typed_crystal)
    smaller_template = make_template(
        typed_crystal,
        template_id="smaller",
        site_count=4,
        mode_count=4,
    )
    default_context = make_context(default_template)
    smaller_context = make_context(smaller_template)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state_keys = tuple(model.state_dict())

    first = model(
        typed_crystal["positions"][:5],
        numbers(typed_crystal, 5),
        typed_crystal["cell"],
        typed_crystal["origin"],
        template_context=default_context,
    )
    second = model(
        typed_crystal["positions"][:3],
        numbers(typed_crystal, 3),
        typed_crystal["cell"],
        typed_crystal["origin"],
        template_context=smaller_context,
    )

    assert first.site_energy.shape == (6,)
    assert second.site_energy.shape == (4,)
    assert default_template.topology.num_edges != smaller_template.topology.num_edges
    assert default_context.phase_modes.shape[0] == 5
    assert smaller_context.phase_modes.shape[0] == 4
    assert bool(torch.isfinite(first.energy)) and bool(torch.isfinite(second.energy))
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    assert tuple(model.state_dict()) == state_keys


def parameter_gradients(model, calls):
    outputs = [
        model(
            positions,
            atomic_numbers,
            cell,
            origin,
            template_context=context,
        ).energy
        for positions, atomic_numbers, cell, origin, context in calls
    ]
    return torch.autograd.grad(
        torch.stack(outputs).sum(), tuple(model.parameters()), allow_unused=True
    )


def test_two_context_gradient_equals_sum_of_individual_gradients(typed_crystal):
    model, default_template = make_model_and_template(typed_crystal)
    smaller_template = make_template(
        typed_crystal, template_id="smaller", site_count=4, mode_count=4
    )
    calls = (
        (
            typed_crystal["positions"][:5],
            numbers(typed_crystal, 5),
            typed_crystal["cell"],
            typed_crystal["origin"],
            make_context(default_template),
        ),
        (
            typed_crystal["positions"][:3],
            numbers(typed_crystal, 3),
            typed_crystal["cell"],
            typed_crystal["origin"],
            make_context(smaller_template),
        ),
    )
    combined = parameter_gradients(model, calls)
    first = parameter_gradients(model, calls[:1])
    second = parameter_gradients(model, calls[1:])

    for total, left, right in zip(combined, first, second):
        if total is None:
            assert left is None and right is None
            continue
        expected = torch.zeros_like(total)
        if left is not None:
            expected = expected + left
        if right is not None:
            expected = expected + right
        torch.testing.assert_close(total, expected, atol=2.0e-12, rtol=2.0e-12)


def test_force_loss_mixed_backward_uses_shared_parameters(typed_crystal):
    model, default_template = make_model_and_template(typed_crystal)
    smaller_template = make_template(
        typed_crystal, template_id="smaller", site_count=4, mode_count=4
    )
    first_positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    second_positions = typed_crystal["positions"][:3].clone().requires_grad_(True)
    first = model(
        first_positions,
        numbers(typed_crystal, 5),
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        create_graph=True,
        template_context=make_context(default_template),
    )
    second = model(
        second_positions,
        numbers(typed_crystal, 3),
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        create_graph=True,
        template_context=make_context(smaller_template),
    )
    selected = (
        model.readout.mlp[-1].weight,
        model.layers[0].edge.radial_head.network[0].weight,
        model.central.embedding.weight,
    )
    gradients = torch.autograd.grad(
        first.forces.square().sum() + second.forces.square().sum(), selected
    )
    assert all(bool(torch.all(torch.isfinite(value))) for value in gradients)
    assert all(bool(torch.any(value != 0)) for value in gradients)


def test_state_dict_keys_strict_and_weights_only_round_trip(typed_crystal, tmp_path):
    model, template = make_model_and_template(typed_crystal)
    keys_before = tuple(model.state_dict())
    model(
        typed_crystal["positions"][:5],
        numbers(typed_crystal, 5),
        typed_crystal["cell"],
        typed_crystal["origin"],
        template_context=make_context(template),
    )
    assert tuple(model.state_dict()) == keys_before

    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    loaded = torch.load(path, weights_only=True)
    clone, _ = make_model_and_template(typed_crystal)
    result = clone.load_state_dict(loaded, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert tuple(clone.state_dict()) == keys_before


def call_with_context(model, data, context, atom_count=3):
    return model(
        data["positions"][:atom_count],
        numbers(data, atom_count),
        data["cell"],
        data["origin"],
        template_context=context,
    )


def test_runtime_compatibility_mismatches_fail_fast(typed_crystal):
    model, template = make_model_and_template(typed_crystal)

    corrupted = make_context(template)
    corrupted.phase_mode_weights[0] += 1.0
    with pytest.raises(ValueError, match="fingerprint"):
        call_with_context(model, typed_crystal, corrupted)

    species = make_context(
        make_template(typed_crystal, template_id="species", supported_species=(6, 8))
    )
    with pytest.raises(ValueError, match="supported species"):
        call_with_context(model, typed_crystal, species)

    atom_species = make_context(
        make_template(typed_crystal, template_id="atom-species", supported_species=(6,))
    )
    with pytest.raises(ValueError, match="unsupported by runtime template"):
        call_with_context(model, typed_crystal, atom_species)

    bad_types = typed_crystal["site_types"].clone()
    bad_types[0] = 2
    site_type = make_context(
        make_template(typed_crystal, template_id="site-type", site_types=bad_types)
    )
    with pytest.raises(ValueError, match="global site type"):
        call_with_context(model, typed_crystal, site_type)

    site_alignment = torch.cat(
        (
            typed_crystal["site_weights"],
            torch.zeros((6, 1), dtype=torch.float64),
        ),
        dim=1,
    )
    channel = make_context(
        make_template(
            typed_crystal,
            template_id="channel",
            site_alignment_weights=site_alignment,
            phase_channel_weights=torch.ones(3, dtype=torch.float64),
        )
    )
    with pytest.raises(ValueError, match="phase channel count"):
        call_with_context(model, typed_crystal, channel)

    cutoff = make_context(
        make_template(typed_crystal, template_id="cutoff", cutoff=2.5)
    )
    with pytest.raises(ValueError, match="topology cutoff"):
        call_with_context(model, typed_crystal, cutoff)

    average = make_context(template, avg_num_neighbors=7.0)
    with pytest.raises(ValueError, match="avg_num_neighbors"):
        call_with_context(model, typed_crystal, average)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_runtime_context_smoke_when_available(typed_crystal, dtype):
    if not torch.cuda.is_available():
        return
    data = cast_data(typed_crystal, dtype)
    model, template = make_model_and_template(data)
    model = copy.deepcopy(model).to(device="cuda", dtype=dtype)
    positions = data["positions"][:5].cuda().requires_grad_(True)
    output = model(
        positions,
        numbers(data, 5).cuda(),
        data["cell"].cuda(),
        data["origin"].cuda(),
        compute_forces=True,
        compute_stress=True,
        template_context=make_context(template),
    )
    assert output.energy.dtype == dtype and output.energy.device.type == "cuda"
    assert bool(torch.isfinite(output.energy))
    assert bool(torch.all(torch.isfinite(output.forces)))
    assert bool(torch.all(torch.isfinite(output.stress)))
