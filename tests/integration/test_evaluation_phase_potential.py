from __future__ import annotations

import pytest
import torch

from refsite_mlip.data import ReferenceTemplate
from refsite_mlip.models import EvaluationPolicy
from refsite_mlip.phase.types import EvaluationPhaseError, TypedStabilizer
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from test_runtime_template_context import make_context, make_model_and_template, numbers


def _policy(template, **changes):
    values = dict(
        template_id=template.template_id,
        template_fingerprint=template.fingerprint,
        candidate_offsets=torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=torch.float64,
        ),
        phase_step_schedule=(0.7, 0.8, 0.9, 1.0),
        phase_damping_schedule=(2.0, 1.0, 0.5, 0.2),
        minimum_objective_gap_absolute=1.0e-2,
        minimum_cross_amplitude_absolute=1.0e-12,
        minimum_atomic_amplitude_absolute=1.0e-12,
        minimum_reference_amplitude_absolute=1.0e-12,
        minimum_curvature=1.0e-2,
        maximum_condition=1.0e8,
        maximum_gradient_norm=2.0e-4,
        equivalence_tolerance=1.0e-8,
    )
    values.update(changes)
    return EvaluationPolicy(**values)


def _evaluate(model, data, context, policy, positions=None, *, return_aux=True):
    positions = data["positions"][:5] if positions is None else positions
    return model(
        positions,
        numbers(data, 5),
        data["cell"],
        data["origin"],
        solver_path=EVAL_ADAPTIVE,
        evaluation_policy=policy,
        template_context=context,
        return_aux=return_aux,
    )


def test_evaluation_policy_snapshot_serialization_and_fingerprint(typed_crystal):
    _, template = make_model_and_template(typed_crystal)
    source = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64)
    policy = _policy(template, candidate_offsets=source)
    snapshot = policy.candidate_offsets.clone()
    source[0, 0] = 3.0
    assert torch.equal(policy.candidate_offsets, snapshot)
    assert policy.candidate_offsets.device.type == "cpu"
    assert not policy.candidate_offsets.requires_grad
    restored = EvaluationPolicy.from_dict(policy.to_dict())
    assert restored.to_dict() == policy.to_dict()
    assert restored.content_fingerprint == policy.content_fingerprint
    policy.candidate_offsets[0, 0] = 0.125
    with pytest.raises(ValueError, match="content fingerprint"):
        policy.validate_fingerprint()


def test_solver_path_migration_and_energy_only_contract(typed_crystal):
    model, template = make_model_and_template(typed_crystal)
    context = make_context(template)
    policy = _policy(template)
    arguments = (
        typed_crystal["positions"][:5], numbers(typed_crystal, 5),
        typed_crystal["cell"], typed_crystal["origin"],
    )
    with pytest.raises(ValueError, match="TRAIN_FIXED"):
        model(*arguments, evaluation_policy=policy)
    with pytest.raises(ValueError, match="evaluation_policy"):
        model(*arguments, solver_path=EVAL_ADAPTIVE, template_context=context)
    with pytest.raises(ValueError, match="TemplateExecutionContext"):
        model(*arguments, solver_path=EVAL_ADAPTIVE, evaluation_policy=policy)
    for keyword in ("compute_forces", "compute_stress", "create_graph"):
        with pytest.raises(ValueError, match="energy-only"):
            model(
                *arguments, solver_path=EVAL_ADAPTIVE,
                evaluation_policy=policy, template_context=context,
                **{keyword: True},
            )


def test_valid_evaluation_energy_diagnostics_graph_and_state_contract(typed_crystal):
    model, template = make_model_and_template(typed_crystal)
    context, policy = make_context(template), _policy(template)
    keys = tuple(model.state_dict())
    parameter_ids = tuple(id(value) for value in model.parameters())
    parameter_count = sum(value.numel() for value in model.parameters())
    positions = typed_crystal["positions"][:5].clone().requires_grad_(True)
    output = _evaluate(model, typed_crystal, context, policy, positions)
    assert output.energy.requires_grad
    assert output.auxiliary["phase"].grad_fn is not None
    diagnostics = output.auxiliary["evaluation_diagnostics"]
    assert diagnostics.input_candidate_count == 3
    assert diagnostics.non_equivalent_group_count == 3
    assert diagnostics.selected_original_candidate_index == 0
    assert diagnostics.selected_grouped_index == 0
    assert diagnostics.absolute_objective_gap > 0.0
    assert diagnostics.transport_path == EVAL_ADAPTIVE
    assert diagnostics.transport_row_residual <= 1.0e-10
    assert diagnostics.transport_column_residual <= 1.0e-10
    assert diagnostics.alias_stabilizer_validated
    assert diagnostics.refined_phase.grad_fn is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert tuple(model.state_dict()) == keys
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert sum(value.numel() for value in model.parameters()) == parameter_count
    assert _evaluate(model, typed_crystal, context, policy, positions, return_aux=False).auxiliary is None


def test_train_fixed_eval_parity_vacancy_mass_and_symmetry(typed_crystal):
    model, template = make_model_and_template(typed_crystal)
    context, policy = make_context(template), _policy(template)
    positions, atomic_numbers = typed_crystal["positions"][:5], numbers(typed_crystal, 5)
    fixed = model(
        positions, atomic_numbers, typed_crystal["cell"], typed_crystal["origin"],
        solver_path=TRAIN_FIXED, return_aux=True, template_context=context,
    )
    adaptive = _evaluate(model, typed_crystal, context, policy)
    torch.testing.assert_close(adaptive.energy, fixed.energy, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(adaptive.auxiliary["ot"].P, fixed.auxiliary["ot"].P, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(adaptive.auxiliary["ot"].q, fixed.auxiliary["ot"].q, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(
        adaptive.auxiliary["multipoles"].equivariant_features,
        fixed.auxiliary["multipoles"].equivariant_features,
        atol=3e-9, rtol=3e-9,
    )
    torch.testing.assert_close(adaptive.auxiliary["ot"].q.sum(), torch.tensor(1.0, dtype=torch.float64), atol=2e-12, rtol=0.0)
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        positions[order], atomic_numbers[order], typed_crystal["cell"], typed_crystal["origin"],
        solver_path=EVAL_ADAPTIVE, evaluation_policy=policy,
        template_context=context, return_aux=True,
    )
    torch.testing.assert_close(permuted.energy, adaptive.energy, atol=3e-10, rtol=3e-10)
    assert permuted.auxiliary["evaluation_diagnostics"].selected_grouped_index == adaptive.auxiliary["evaluation_diagnostics"].selected_grouped_index
    shift = torch.tensor([0.71, -0.39, 0.83], dtype=torch.float64)
    moved = model(
        positions + shift, atomic_numbers, typed_crystal["cell"], typed_crystal["origin"],
        solver_path=EVAL_ADAPTIVE, evaluation_policy=policy, template_context=context,
    )
    joint = model(
        positions + shift, atomic_numbers, typed_crystal["cell"], typed_crystal["origin"] + shift,
        solver_path=EVAL_ADAPTIVE, evaluation_policy=policy, template_context=context,
    )
    lattice = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64) @ typed_crystal["cell"]
    wrapped_positions = positions.clone(); wrapped_positions[0] += lattice
    wrapped = _evaluate(model, typed_crystal, context, policy, wrapped_positions, return_aux=False)
    for result in (moved, joint, wrapped):
        torch.testing.assert_close(result.energy, adaptive.energy, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"minimum_reference_amplitude_absolute": 1.0e6}, "REFERENCE_MODE_EXTINCTION"),
        ({"minimum_atomic_amplitude_absolute": 1.0e6}, "ATOMIC_MODE_EXTINCTION"),
        ({"minimum_cross_amplitude_absolute": 1.0e6}, "CROSS_AMPLITUDE_TOO_SMALL"),
        ({"minimum_objective_gap_absolute": 1.0e6}, "NON_EQUIVALENT_GAP_TOO_SMALL"),
        ({"minimum_curvature": 1.0e6}, "HESSIAN_CURVATURE_FAILURE"),
        ({"maximum_condition": 1.000001}, "HESSIAN_CONDITION_FAILURE"),
        ({"maximum_gradient_norm": 1.0e-12}, "PHASE_RESIDUAL_TOO_LARGE"),
    ],
)
def test_structured_evaluation_domain_failures(typed_crystal, change, reason):
    model, template = make_model_and_template(typed_crystal)
    with pytest.raises(EvaluationPhaseError) as caught:
        _evaluate(model, typed_crystal, make_context(template), _policy(template, **change))
    assert caught.value.reason_code == reason
    assert template.template_id in str(caught.value)


def test_policy_context_and_alias_mismatches_are_structured(typed_crystal):
    model, template = make_model_and_template(typed_crystal)
    context = make_context(template)
    with pytest.raises(EvaluationPhaseError) as caught:
        _evaluate(model, typed_crystal, context, _policy(template, template_id="wrong-template"))
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    with pytest.raises(EvaluationPhaseError) as caught:
        _evaluate(
            model,
            typed_crystal,
            context,
            _policy(template, template_fingerprint="0" * 64),
        )
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    bad_stabilizer = TypedStabilizer(
        torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64),
        torch.arange(template.topology.num_sites, dtype=torch.long).unsqueeze(0),
    )
    invalid_template = ReferenceTemplate.snapshot(
        "invalid-alias", template.topology, template.phase_modes,
        template.phase_mode_weights, template.site_alignment_weights,
        template.phase_channel_weights, bad_stabilizer,
        template.supported_species,
    )
    with pytest.raises(EvaluationPhaseError) as caught:
        _evaluate(model, typed_crystal, make_context(invalid_template), _policy(invalid_template))
    assert caught.value.reason_code == "ALIAS_STABILIZER_MISMATCH"


def test_invalid_candidate_contract_and_cuda_energy_smoke(typed_crystal):
    _, template = make_model_and_template(typed_crystal)
    with pytest.raises(ValueError, match="candidate"):
        _policy(template, candidate_offsets=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64))
    if not torch.cuda.is_available():
        return
    for dtype in (torch.float32, torch.float64):
        data = {
            key: value.to(dtype=dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            for key, value in typed_crystal.items()
        }
        model, template = make_model_and_template(data)
        result = model.cuda()(
            data["positions"][:5].cuda(), numbers(data, 5).cuda(),
            data["cell"].cuda(), data["origin"].cuda(),
            solver_path=EVAL_ADAPTIVE, evaluation_policy=_policy(template),
            template_context=make_context(template),
        )
        assert result.energy.dtype == dtype and result.energy.device.type == "cuda"
        assert torch.isfinite(result.energy)
