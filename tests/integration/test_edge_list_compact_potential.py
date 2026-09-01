from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from refsite_mlip.models import ReferenceSitePotential, evaluate_structure_batch
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    materialize_dense_plan,
)
from test_compact_support_potential import _compact, _model, _numbers
from test_evaluation_phase_potential import _policy
from test_grouped_template_batch import _case
from test_runtime_template_context import make_context, make_template


def _edge_support():
    return replace(_compact(), backend="edge_list")


def _edge_clone(model):
    configured = ReferenceSitePotential(
        replace(model.config, transport_support=_edge_support()),
        model.topology,
        model.phase_modes,
        model.phase_mode_weights,
        model.species_alignment_weights,
        model.site_alignment_weights,
        model.phase_channel_weights,
        model.atomic_baseline,
    ).to(model.atomic_baseline)
    configured.load_state_dict(model.state_dict(), strict=True)
    return configured


def _strain_energy(model, data, numbers, direction, magnitude):
    deformation = torch.eye(3, dtype=data["cell"].dtype) + magnitude * direction
    return model(
        data["positions"][:5] @ deformation,
        numbers,
        data["cell"] @ deformation,
        data["origin"] @ deformation,
    ).energy


def test_edge_list_potential_oracle_no_implicit_dense_plan_fd_and_double_backward(typed_crystal):
    dense = _model(typed_crystal, _compact())
    sparse = _edge_clone(dense)
    assert type(sparse.config).from_dict(sparse.config.to_dict()) == sparse.config
    assert tuple(sparse.state_dict()) == tuple(dense.state_dict())
    positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    numbers = _numbers(typed_crystal)
    dense_out = dense(
        positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    sparse_out = sparse(
        positions,
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    sparse_ot = sparse_out.auxiliary["ot"]
    assert not hasattr(sparse_ot, "P") and not hasattr(sparse_ot, "gamma")
    assert not sparse_ot.dense_plan_materialized
    plan = materialize_dense_plan(sparse_ot).plan
    dense_ot = dense_out.auxiliary["ot"]
    torch.testing.assert_close(plan, dense_ot.P, atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(sparse_ot.q, dense_ot.q, atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(sparse_out.energy, dense_out.energy, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(sparse_out.baseline_energy, dense_out.baseline_energy, atol=0, rtol=0)
    torch.testing.assert_close(sparse_out.residual_energy, dense_out.residual_energy, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(sparse_out.site_energy, dense_out.site_energy, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(sparse_out.raw_c, dense_out.raw_c, atol=3e-15, rtol=3e-15)
    torch.testing.assert_close(sparse_out.site_features, dense_out.site_features, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(sparse_out.forces, dense_out.forces, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(sparse_out.stress, dense_out.stress, atol=3e-14, rtol=3e-13)
    torch.testing.assert_close(
        sparse_out.auxiliary["multipoles"].equivariant_features,
        dense_out.auxiliary["multipoles"].equivariant_features,
        atol=3e-14,
        rtol=3e-14,
    )

    h = 1e-6
    delta = torch.zeros_like(positions)
    delta[2, 1] = h
    plus = sparse(positions.detach() + delta, numbers, typed_crystal["cell"], typed_crystal["origin"]).energy
    minus = sparse(positions.detach() - delta, numbers, typed_crystal["cell"], typed_crystal["origin"]).energy
    torch.testing.assert_close(sparse_out.forces[2, 1], -(plus - minus) / (2 * h), atol=5e-6, rtol=5e-5)
    directions = []
    for i in range(3):
        direction = torch.zeros((3, 3), dtype=torch.float64)
        direction[i, i] = 1.0
        directions.append(direction)
    for i, j in ((1, 2), (0, 2), (0, 1)):
        direction = torch.zeros((3, 3), dtype=torch.float64)
        direction[i, j] = direction[j, i] = 0.5
        directions.append(direction)
    volume = torch.linalg.det(typed_crystal["cell"]).abs()
    for direction in directions:
        fd = (
            _strain_energy(sparse, typed_crystal, numbers, direction, h)
            - _strain_energy(sparse, typed_crystal, numbers, direction, -h)
        ) / (2 * h)
        torch.testing.assert_close(volume * (sparse_out.stress * direction).sum(), fd, atol=5e-6, rtol=5e-5)

    gradients = torch.autograd.grad(
        sparse_out.forces.square().sum(),
        (
            sparse.readout.mlp[-1].weight,
            sparse.layers[0].corr.C2_product.weight,
            sparse.central.embedding.weight,
        ),
    )
    assert all(torch.isfinite(value).all() and torch.count_nonzero(value) for value in gradients)
    assert torch.linalg.vector_norm(sparse_out.forces.sum(0)) < 2e-11


def test_edge_list_potential_symmetry_state_and_eval_support(typed_crystal):
    sparse = _model(typed_crystal, _edge_support())
    positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    numbers = _numbers(typed_crystal)
    parameter_ids = tuple(id(value) for value in sparse.parameters())
    state = {key: value.clone() for key, value in sparse.state_dict().items()}
    output = sparse(positions, numbers, typed_crystal["cell"], typed_crystal["origin"], compute_forces=True)
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = sparse(positions.detach()[order].requires_grad_(True), numbers[order], typed_crystal["cell"], typed_crystal["origin"], compute_forces=True)
    torch.testing.assert_close(permuted.energy, output.energy, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(permuted.forces, output.forces[order], atol=3e-12, rtol=3e-12)
    shift = torch.tensor([0.7, -0.3, 0.9], dtype=torch.float64)
    translated = sparse((positions.detach() + shift).requires_grad_(True), numbers, typed_crystal["cell"], typed_crystal["origin"] + shift, compute_forces=True)
    torch.testing.assert_close(translated.energy, output.energy, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(translated.forces, output.forces, atol=3e-12, rtol=3e-12)
    wrapped = positions.detach().clone()
    wrapped[1] += typed_crystal["cell"][0]
    wrapped_out = sparse(wrapped.requires_grad_(True), numbers, typed_crystal["cell"], typed_crystal["origin"], compute_forces=True)
    torch.testing.assert_close(wrapped_out.energy, output.energy, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(wrapped_out.forces, output.forces, atol=3e-12, rtol=3e-12)
    assert tuple(id(value) for value in sparse.parameters()) == parameter_ids
    assert all(torch.equal(sparse.state_dict()[key], value) for key, value in state.items())

    template = make_template(typed_crystal, template_id="edge-eval-supported")
    evaluated = sparse(
        positions.detach(),
        numbers,
        typed_crystal["cell"],
        typed_crystal["origin"],
        solver_path=EVAL_ADAPTIVE,
        template_context=make_context(template),
        evaluation_policy=_policy(template),
        return_aux=True,
    )
    assert torch.isfinite(evaluated.energy)
    assert not hasattr(evaluated.auxiliary["ot"], "P")
    assert not evaluated.auxiliary["ot"].dense_plan_materialized
    diagnostics = evaluated.auxiliary["evaluation_diagnostics"]
    assert diagnostics.transport_backend == "edge_list"
    assert diagnostics.transport_solver_name == "edge_list_hybrid"
    assert not diagnostics.transport_dense_plan_materialized


def test_edge_list_normal_path_never_calls_dense_transport_or_feature(monkeypatch, typed_crystal):
    import refsite_mlip.models.potential as potential_module

    model = _model(typed_crystal, _edge_support())

    def forbidden(*args, **kwargs):
        raise AssertionError("dense compact backend was called")

    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", forbidden)
    monkeypatch.setattr(potential_module, "build_probability_multipoles", forbidden)
    output = model(
        typed_crystal["positions"][:5],
        _numbers(typed_crystal),
        typed_crystal["cell"],
        typed_crystal["origin"],
        return_aux=False,
    )
    assert torch.isfinite(output.energy) and output.auxiliary is None


def test_edge_list_grouped_train_fixed_matches_individual(typed_crystal):
    _, base, _, _, batch, contexts = _case(typed_crystal)
    model = _edge_clone(base)
    batch.positions.requires_grad_(True)
    grouped = evaluate_structure_batch(
        model,
        batch,
        contexts,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    individual = []
    for index in range(batch.num_structures):
        atom_slice = slice(int(batch.atom_ptr[index]), int(batch.atom_ptr[index + 1]))
        individual.append(
            model(
                batch.positions[atom_slice],
                batch.atomic_numbers[atom_slice],
                batch.cells[index],
                batch.origins[index],
                template_context=contexts[batch.template_ids[index]],
                compute_forces=True,
                compute_stress=True,
                create_graph=True,
            )
        )
    torch.testing.assert_close(grouped.energy, torch.stack([value.energy for value in individual]), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(grouped.forces, torch.cat([value.forces for value in individual]), atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(grouped.stress, torch.stack([value.stress for value in individual]), atol=3e-14, rtol=3e-13)
    gradient = torch.autograd.grad(
        grouped.forces.square().sum(), model.readout.mlp[-1].weight
    )[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_edge_list_cuda_mixed_template_energy_force_stress(dtype, typed_crystal):
    _, base, _, _, batch, contexts = _case(
        typed_crystal, dtype=dtype, device="cuda"
    )
    model = _edge_clone(base)
    batch.positions.requires_grad_(True)
    output = evaluate_structure_batch(
        model,
        batch,
        contexts,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert output.energy.device.type == "cuda" and output.energy.dtype == dtype
    assert output.forces.device.type == "cuda" and output.forces.dtype == dtype
    assert output.stress.device.type == "cuda" and output.stress.dtype == dtype
    assert torch.isfinite(output.energy).all()
    assert torch.isfinite(output.forces).all()
    assert torch.isfinite(output.stress).all()
    assert all(
        auxiliary["ot"].support_diagnostics.backend == "edge_list"
        for auxiliary in output.auxiliary
    )
