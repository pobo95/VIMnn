from __future__ import annotations

from dataclasses import replace
import copy

import pytest
import torch

import refsite_mlip.models.potential as potential_module
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import EVAL_ADAPTIVE

from test_evaluation_phase_potential import _policy
from test_runtime_template_context import make_context, make_model_and_template, numbers


def _call(
    model,
    data,
    context,
    policy,
    atom_count,
    *,
    positions=None,
    cell=None,
    origin=None,
    forces=False,
    stress=False,
    auxiliary=True,
):
    return model(
        data["positions"][:atom_count] if positions is None else positions,
        numbers(data, atom_count),
        data["cell"] if cell is None else cell,
        data["origin"] if origin is None else origin,
        solver_path=EVAL_ADAPTIVE,
        template_context=context,
        evaluation_policy=policy,
        compute_forces=forces,
        compute_stress=stress,
        create_graph=False,
        return_aux=auxiliary,
    )


def _branch_signature(output):
    diagnostics = output.auxiliary["evaluation_diagnostics"]
    return (
        diagnostics.selected_grouped_index,
        diagnostics.transport_solver_name,
        diagnostics.transport_sinkhorn_iterations,
        diagnostics.transport_newton_iterations,
        diagnostics.transport_cg_iterations,
        diagnostics.transport_fallback_used,
    )


def _same_certified_branch(left, right):
    return (
        left[0] == right[0]
        and left[1] == right[1]
        and left[5] == right[5]
    )


def _accepted_positions(data, atom_count):
    positions = data["positions"][:atom_count].clone()
    if atom_count < data["sites"].shape[0]:
        direction = torch.tensor(
            [
                [0.31, -0.17, 0.23],
                [-0.11, 0.29, 0.07],
                [0.19, 0.13, -0.27],
                [-0.23, 0.05, 0.21],
                [0.09, -0.25, 0.15],
            ],
            dtype=positions.dtype,
            device=positions.device,
        )
        positions = positions + 1.0e-3 * direction
    return positions


@pytest.mark.parametrize("atom_count", [6, 5], ids=["pristine", "vacancy"])
def test_all_force_components_match_central_difference_on_stable_branch(
    typed_crystal, atom_count
):
    model, template = make_model_and_template(typed_crystal)
    context, policy = make_context(template), _policy(template)
    positions = _accepted_positions(typed_crystal, atom_count)
    baseline = _call(
        model, typed_crystal, context, policy, atom_count,
        positions=positions, forces=True,
    )
    assert baseline.forces is not None
    assert not baseline.forces.requires_grad and baseline.forces.grad_fn is None
    expected_branch = _branch_signature(baseline)
    step = 2.0e-6 if atom_count == 6 else 1.0e-6
    maximum_absolute = 0.0
    maximum_relative = 0.0
    adaptive_iterations = {expected_branch[2:6]}
    for atom in range(atom_count):
        for component in range(3):
            displacement = torch.zeros_like(positions)
            displacement[atom, component] = step
            plus = _call(
                model, typed_crystal, context, policy, atom_count,
                positions=positions + displacement,
            )
            minus = _call(
                model, typed_crystal, context, policy, atom_count,
                positions=positions - displacement,
            )
            plus_branch = _branch_signature(plus)
            minus_branch = _branch_signature(minus)
            assert _same_certified_branch(plus_branch, expected_branch)
            assert _same_certified_branch(minus_branch, expected_branch)
            adaptive_iterations.update((plus_branch[2:6], minus_branch[2:6]))
            finite = -(plus.energy - minus.energy) / (2.0 * step)
            automatic = baseline.forces[atom, component]
            absolute = float((automatic - finite).abs())
            relative = absolute / max(
                float(automatic.abs()), float(finite.abs()), 1.0e-12
            )
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
    assert maximum_absolute <= 5.0e-6
    assert maximum_relative <= 5.0e-4
    assert adaptive_iterations
    assert all(not values[-1] for values in adaptive_iterations)
    if atom_count == 5:
        assert adaptive_iterations == {(16, 1, 2, False)}
    diagnostics = baseline.auxiliary["evaluation_diagnostics"]
    assert diagnostics.differentiability_scope == "selected_branch_first_order"
    assert diagnostics.hard_branch_frozen
    assert diagnostics.derivative_order == 1
    assert diagnostics.forces_requested and not diagnostics.stress_requested


def _symmetric_directions(dtype, device):
    directions = []
    for axis in range(3):
        value = torch.zeros((3, 3), dtype=dtype, device=device)
        value[axis, axis] = 1.0
        directions.append(value)
    for left, right in ((1, 2), (0, 2), (0, 1)):
        value = torch.zeros((3, 3), dtype=dtype, device=device)
        value[left, right] = value[right, left] = 0.5
        directions.append(value)
    return directions


@pytest.mark.parametrize("atom_count", [6, 5], ids=["pristine", "vacancy"])
def test_six_stress_directions_match_finite_difference_on_stable_branch(
    typed_crystal, atom_count
):
    model, template = make_model_and_template(typed_crystal)
    context, policy = make_context(template), _policy(template)
    baseline = _call(
        model, typed_crystal, context, policy, atom_count,
        positions=_accepted_positions(typed_crystal, atom_count), stress=True,
    )
    assert baseline.stress is not None and baseline.stress_voigt is not None
    assert baseline.stress.grad_fn is None and not baseline.stress.requires_grad
    torch.testing.assert_close(baseline.stress, baseline.stress.T, atol=0.0, rtol=0.0)
    expected_voigt = baseline.stress[
        (0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)
    ]
    torch.testing.assert_close(baseline.stress_voigt, expected_voigt, atol=0.0, rtol=0.0)
    expected_branch = _branch_signature(baseline)
    identity = torch.eye(3, dtype=torch.float64)
    positions = _accepted_positions(typed_crystal, atom_count)
    volume = torch.linalg.det(typed_crystal["cell"]).abs()
    step = 1.0e-4
    maximum_absolute = 0.0
    maximum_relative = 0.0
    adaptive_iterations = {expected_branch[2:6]}
    for direction in _symmetric_directions(torch.float64, torch.device("cpu")):
        plus_deformation = identity + step * direction
        minus_deformation = identity - step * direction
        plus = _call(
            model, typed_crystal, context, policy, atom_count,
            positions=positions @ plus_deformation,
            cell=typed_crystal["cell"] @ plus_deformation,
            origin=typed_crystal["origin"] @ plus_deformation,
        )
        minus = _call(
            model, typed_crystal, context, policy, atom_count,
            positions=positions @ minus_deformation,
            cell=typed_crystal["cell"] @ minus_deformation,
            origin=typed_crystal["origin"] @ minus_deformation,
        )
        plus_branch = _branch_signature(plus)
        minus_branch = _branch_signature(minus)
        assert _same_certified_branch(plus_branch, expected_branch)
        assert _same_certified_branch(minus_branch, expected_branch)
        adaptive_iterations.update((plus_branch[2:6], minus_branch[2:6]))
        finite = (plus.energy - minus.energy) / (2.0 * step * volume)
        automatic = torch.sum(baseline.stress * direction)
        absolute = float((automatic - finite).abs())
        relative = absolute / max(
            float(automatic.abs()), float(finite.abs()), 1.0e-12
        )
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    assert maximum_absolute <= 5.0e-6
    assert maximum_relative <= 5.0e-4
    assert len(adaptive_iterations) == 1
    assert not next(iter(adaptive_iterations))[-1]
    diagnostics = baseline.auxiliary["evaluation_diagnostics"]
    assert diagnostics.differentiability_scope == "selected_branch_first_order"
    assert not diagnostics.forces_requested and diagnostics.stress_requested


def test_force_stress_combined_api_and_energy_auxiliary_parity(
    typed_crystal, monkeypatch
):
    model, template = make_model_and_template(typed_crystal)
    context, policy = make_context(template), _policy(template)
    force_only = _call(model, typed_crystal, context, policy, 5, forces=True)
    stress_only = _call(model, typed_crystal, context, policy, 5, stress=True)
    original_grad = torch.autograd.grad
    calls = []
    def counted_grad(*args, **kwargs):
        calls.append(1)
        return original_grad(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    combined = _call(
        model, typed_crystal, context, policy, 5, forces=True, stress=True
    )
    assert len(calls) == 1
    monkeypatch.setattr(torch.autograd, "grad", original_grad)
    energy_only = _call(model, typed_crystal, context, policy, 5)
    no_aux = _call(
        model, typed_crystal, context, policy, 5,
        forces=True, stress=True, auxiliary=False,
    )
    for output in (force_only, stress_only, combined):
        torch.testing.assert_close(output.energy, energy_only.energy, atol=0.0, rtol=0.0)
        torch.testing.assert_close(output.auxiliary["phase"], energy_only.auxiliary["phase"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(output.auxiliary["ot"].P, energy_only.auxiliary["ot"].P, atol=0.0, rtol=0.0)
        torch.testing.assert_close(output.auxiliary["ot"].q, energy_only.auxiliary["ot"].q, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            output.auxiliary["multipoles"].equivariant_features,
            energy_only.auxiliary["multipoles"].equivariant_features,
            atol=0.0, rtol=0.0,
        )
    torch.testing.assert_close(force_only.forces, combined.forces, atol=0.0, rtol=0.0)
    torch.testing.assert_close(stress_only.stress, combined.stress, atol=0.0, rtol=0.0)
    assert no_aux.auxiliary is None
    assert combined.forces.grad_fn is None and combined.stress.grad_fn is None
    diagnostics = combined.auxiliary["evaluation_diagnostics"]
    assert diagnostics.forces_requested and diagnostics.stress_requested
    assert diagnostics.effective_transport_tolerance == 1.0e-12


def test_no_grad_symmetry_candidate_order_and_state_preservation(typed_crystal):
    model, template = make_model_and_template(typed_crystal)
    context, policy = make_context(template), _policy(template)
    positions = typed_crystal["positions"][:5].clone()
    positions_before = positions.clone()
    policy_before = policy.to_dict()
    context_fingerprint = context.fingerprint
    parameter_ids = tuple(id(value) for value in model.parameters())
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    model.train()
    gradients = []
    for index, parameter in enumerate(model.parameters()):
        if index < 2:
            parameter.grad = torch.full_like(parameter, 0.125)
        gradients.append((parameter.grad, None if parameter.grad is None else parameter.grad.clone()))
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    with torch.no_grad():
        baseline = _call(
            model, typed_crystal, context, policy, 5,
            positions=positions, forces=True, stress=True,
        )
    assert model.training
    assert torch.equal(positions, positions_before)
    assert policy.to_dict() == policy_before and context.fingerprint == context_fingerprint
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key])
    for parameter, (identity, value) in zip(model.parameters(), gradients):
        assert parameter.grad is identity
        if value is not None:
            assert torch.equal(parameter.grad, value)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng))

    shift = torch.tensor([0.61, -0.47, 0.29], dtype=torch.float64)
    joint = _call(
        model, typed_crystal, context, policy, 5,
        positions=positions + shift,
        origin=typed_crystal["origin"] + shift,
        forces=True, stress=True,
    )
    torch.testing.assert_close(joint.energy, baseline.energy, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(joint.forces, baseline.forces, atol=3e-8, rtol=3e-8)
    torch.testing.assert_close(baseline.forces.sum(0), torch.zeros(3, dtype=torch.float64), atol=3e-8, rtol=0.0)

    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        positions[order], numbers(typed_crystal, 5)[order],
        typed_crystal["cell"], typed_crystal["origin"],
        solver_path=EVAL_ADAPTIVE, template_context=context,
        evaluation_policy=policy, compute_forces=True, compute_stress=True,
        return_aux=True,
    )
    torch.testing.assert_close(permuted.energy, baseline.energy, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(permuted.forces, baseline.forces[order], atol=3e-8, rtol=3e-8)

    reordered_policy = _policy(
        template,
        candidate_offsets=policy.candidate_offsets[
            torch.tensor([2, 1, 0])
        ],
    )
    reordered = _call(
        model, typed_crystal, context, reordered_policy, 5,
        forces=True, stress=True,
    )
    torch.testing.assert_close(reordered.energy, baseline.energy, atol=0.0, rtol=0.0)
    torch.testing.assert_close(reordered.forces, baseline.forces, atol=0.0, rtol=0.0)
    torch.testing.assert_close(reordered.stress, baseline.stress, atol=0.0, rtol=0.0)


def test_derivative_negative_contracts_and_structured_failures(typed_crystal, monkeypatch):
    model, template = make_model_and_template(typed_crystal)
    context, policy = make_context(template), _policy(template)
    arguments = (
        typed_crystal["positions"][:5], numbers(typed_crystal, 5),
        typed_crystal["cell"], typed_crystal["origin"],
    )
    with pytest.raises(EvaluationPhaseError) as caught:
        model(
            *arguments, solver_path=EVAL_ADAPTIVE,
            template_context=context, evaluation_policy=policy,
            compute_forces=True, create_graph=True,
        )
    assert caught.value.reason_code == "CREATE_GRAPH_UNSUPPORTED"
    with torch.inference_mode():
        with pytest.raises(EvaluationPhaseError) as caught:
            model(
                *arguments, solver_path=EVAL_ADAPTIVE,
                template_context=context, evaluation_policy=policy,
                compute_forces=True,
            )
    assert caught.value.reason_code == "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED"

    original_phase = potential_module.solve_evaluation_phase
    def disconnected_phase(*args, **kwargs):
        result = original_phase(*args, **kwargs)
        return replace(
            result,
            refined=replace(result.refined, phase=result.refined.phase.detach()),
        )
    monkeypatch.setattr(potential_module, "solve_evaluation_phase", disconnected_phase)
    with pytest.raises(EvaluationPhaseError) as caught:
        _call(model, typed_crystal, context, policy, 5, forces=True)
    assert caught.value.reason_code == "GRAPH_DISCONNECTED"
    monkeypatch.setattr(potential_module, "solve_evaluation_phase", original_phase)

    original_ot = potential_module.solve_atom_vacancy_ot
    def disconnected_p(*args, **kwargs):
        result = original_ot(*args, **kwargs)
        return replace(result, P=result.P.detach())
    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", disconnected_p)
    with pytest.raises(EvaluationPhaseError) as caught:
        _call(model, typed_crystal, context, policy, 5, forces=True)
    assert caught.value.reason_code == "GRAPH_DISCONNECTED"

    def disconnected_q(*args, **kwargs):
        result = original_ot(*args, **kwargs)
        return replace(result, q=result.q.detach())
    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", disconnected_q)
    with pytest.raises(EvaluationPhaseError) as caught:
        _call(model, typed_crystal, context, policy, 5, stress=True)
    assert caught.value.reason_code == "GRAPH_DISCONNECTED"

    def fallback_ot(*args, **kwargs):
        result = original_ot(*args, **kwargs)
        return replace(result, fallback_used=True)
    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", fallback_ot)
    with pytest.raises(EvaluationPhaseError) as caught:
        _call(model, typed_crystal, context, policy, 5, stress=True)
    assert caught.value.reason_code == "DERIVATIVE_FALLBACK_UNSUPPORTED"
    monkeypatch.setattr(potential_module, "solve_atom_vacancy_ot", original_ot)

    monkeypatch.setattr(
        model.readout,
        "forward",
        lambda hidden, central: torch.full(
            (hidden.shape[0],),
            float("nan"),
            dtype=hidden.dtype,
            device=hidden.device,
        ),
    )
    with pytest.raises(EvaluationPhaseError) as caught:
        _call(model, typed_crystal, context, policy, 5, forces=True)
    assert caught.value.reason_code == "NONFINITE_OUTPUT"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_energy_force_stress_smoke(typed_crystal, dtype):
    if not torch.cuda.is_available():
        return
    data = {
        key: value.to(dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in typed_crystal.items()
    }
    model, template = make_model_and_template(data)
    model = copy.deepcopy(model).cuda()
    output = model(
        data["positions"][:5].cuda(), numbers(data, 5).cuda(),
        data["cell"].cuda(), data["origin"].cuda(),
        solver_path=EVAL_ADAPTIVE,
        template_context=make_context(template),
        evaluation_policy=_policy(template),
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    assert output.energy.dtype == dtype and output.energy.device.type == "cuda"
    assert output.forces.dtype == dtype and output.forces.device.type == "cuda"
    assert output.stress.dtype == dtype and output.stress.device.type == "cuda"
    assert torch.isfinite(output.energy)
    assert torch.all(torch.isfinite(output.forces))
    assert torch.all(torch.isfinite(output.stress))
    expected_tolerance = 1.0e-6 if dtype == torch.float32 else 1.0e-12
    assert output.auxiliary["evaluation_diagnostics"].effective_transport_tolerance == expected_tolerance
