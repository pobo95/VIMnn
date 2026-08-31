"""Parameter-free mixed-template execution for ragged structure batches."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from .evaluation_policy import EvaluationPolicy
from .outputs import BatchedPotentialOutput, PotentialOutput
from .potential import ReferenceSitePotential
from .template_context import TemplateExecutionContext


def _validated_context(
    template_id: str,
    expected_fingerprint: str,
    template_contexts: Mapping[str, TemplateExecutionContext],
) -> TemplateExecutionContext:
    if template_id not in template_contexts:
        raise KeyError(f"missing TemplateExecutionContext for {template_id}")
    context = template_contexts[template_id]
    if not isinstance(context, TemplateExecutionContext):
        raise TypeError(
            f"template context for {template_id} must be a TemplateExecutionContext"
        )
    context.validate_fingerprint()
    if context.template_id != template_id:
        raise ValueError("template context ID does not match batch template ID")
    if context.fingerprint != expected_fingerprint:
        raise ValueError("batch and template context fingerprints do not match")
    return context


def _validated_evaluation_policy(
    template_id: str,
    expected_fingerprint: str,
    evaluation_policies: Mapping[str, EvaluationPolicy],
) -> EvaluationPolicy:
    if template_id not in evaluation_policies:
        raise EvaluationPhaseError(
            "POLICY_CONTEXT_MISMATCH",
            f"missing EvaluationPolicy for exact template_id key {template_id!r}",
            template_id=template_id,
        )
    policy = evaluation_policies[template_id]
    if not isinstance(policy, EvaluationPolicy):
        raise TypeError(
            f"evaluation policy for {template_id} must be an EvaluationPolicy"
        )
    policy.validate_fingerprint()
    if policy.template_id != template_id:
        raise EvaluationPhaseError(
            "POLICY_CONTEXT_MISMATCH",
            "evaluation policy mapping key and policy template_id differ",
            template_id=template_id,
        )
    if policy.template_fingerprint != expected_fingerprint:
        raise EvaluationPhaseError(
            "POLICY_CONTEXT_MISMATCH",
            "batch and evaluation policy template fingerprints differ",
            template_id=template_id,
        )
    return policy


def _preflight_runtime_bindings(
    model: ReferenceSitePotential,
    batch: StructureBatch,
    template_contexts: Mapping[str, TemplateExecutionContext],
    evaluation_policies: Mapping[str, EvaluationPolicy] | None,
    *,
    solver_path: str,
    compute_forces: bool,
    compute_stress: bool,
    create_graph: bool,
) -> tuple[
    dict[str, TemplateExecutionContext],
    dict[str, EvaluationPolicy],
]:
    derivative_requested = compute_forces or compute_stress
    if solver_path == TRAIN_FIXED:
        if evaluation_policies:
            raise ValueError(
                "TRAIN_FIXED requires evaluation_policies=None or an empty mapping"
            )
    elif solver_path == EVAL_ADAPTIVE:
        if create_graph:
            raise EvaluationPhaseError(
                "CREATE_GRAPH_UNSUPPORTED",
                "grouped EVAL_ADAPTIVE supports selected-branch first derivatives only",
            )
        if derivative_requested and torch.is_inference_mode_enabled():
            raise EvaluationPhaseError(
                "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED",
                "grouped EVAL_ADAPTIVE derivatives cannot run under torch.inference_mode()",
            )
        if evaluation_policies is None:
            raise EvaluationPhaseError(
                "POLICY_CONTEXT_MISMATCH",
                "EVAL_ADAPTIVE requires an exact template_id-to-EvaluationPolicy mapping",
            )
        if not isinstance(evaluation_policies, Mapping):
            raise TypeError("evaluation_policies must be a mapping")
    else:
        raise ValueError("unsupported solver path")

    contexts: dict[str, TemplateExecutionContext] = {}
    policies: dict[str, EvaluationPolicy] = {}
    for group in batch.template_groups:
        template_id = group.template_id
        expected = group.template_fingerprint
        try:
            context = _validated_context(
                template_id, expected, template_contexts
            )
        except (KeyError, TypeError, ValueError) as error:
            if solver_path != EVAL_ADAPTIVE:
                raise
            raise EvaluationPhaseError(
                "POLICY_CONTEXT_MISMATCH",
                f"template context preflight failed: {error}",
                template_id=template_id,
            ) from error
        contexts[template_id] = context

        if solver_path == EVAL_ADAPTIVE:
            assert evaluation_policies is not None
            policy = _validated_evaluation_policy(
                template_id, expected, evaluation_policies
            )
            if context.template_id != policy.template_id:
                raise EvaluationPhaseError(
                    "POLICY_CONTEXT_MISMATCH",
                    "context and policy template IDs differ",
                    template_id=template_id,
                )
            if context.fingerprint != policy.template_fingerprint:
                raise EvaluationPhaseError(
                    "POLICY_CONTEXT_MISMATCH",
                    "context and policy binding fingerprints differ",
                    template_id=template_id,
                )
            policies[template_id] = policy

        # Validate every structure against the shared model before the first
        # forward call. This checks N <= M, ordered species compatibility,
        # global site-type IDs, phase channels, graph cutoff, and z_avg.
        for structure_index, atom_slice in zip(
            group.structure_indices.tolist(), group.atom_slices
        ):
            try:
                model._resolve_template_context(
                    context,
                    batch.positions[atom_slice],
                    batch.atomic_numbers[atom_slice],
                )
            except (TypeError, ValueError) as error:
                if solver_path != EVAL_ADAPTIVE:
                    raise
                raise EvaluationPhaseError(
                    "POLICY_CONTEXT_MISMATCH",
                    (
                        "runtime template/model compatibility preflight failed "
                        f"for structure_index={structure_index} "
                        f"sample_id={batch.sample_ids[structure_index]!r}: {error}"
                    ),
                    template_id=template_id,
                ) from error
    return contexts, policies


def _raise_grouped_evaluation_error(
    error: Exception,
    *,
    structure_index: int,
    sample_id: str,
    template_id: str,
) -> None:
    context = (
        f"structure_index={structure_index} sample_id={sample_id!r} "
        f"template_id={template_id!r} stage=single_structure_evaluation"
    )
    if isinstance(error, EvaluationPhaseError):
        raise EvaluationPhaseError(
            error.reason_code,
            f"{context}: {error}",
            template_id=template_id,
            observed=error.observed,
            threshold=error.threshold,
        ) from error
    raise EvaluationPhaseError(
        "STRUCTURE_EVALUATION_FAILED",
        f"{context}: {type(error).__name__}: {error}",
        template_id=template_id,
    ) from error


def evaluate_structure_batch(
    model: ReferenceSitePotential,
    batch: StructureBatch,
    template_contexts: Mapping[str, TemplateExecutionContext],
    *,
    solver_path: str = TRAIN_FIXED,
    evaluation_policies: Mapping[str, EvaluationPolicy] | None = None,
    compute_forces: bool = False,
    compute_stress: bool = False,
    create_graph: bool = False,
    return_aux: bool = False,
) -> BatchedPotentialOutput:
    """Evaluate independent structures using one shared model instance.

    Template groups determine deterministic execution order only.  Output slots
    are filled by original structure index, so all public tensors preserve the
    incoming ragged batch order.  For force evaluation, ``batch.positions``
    must already require gradients; this preserves the direct tensor-view path
    and gradients with respect to the caller-owned flattened positions.
    """

    if not isinstance(model, ReferenceSitePotential):
        raise TypeError("model must be a ReferenceSitePotential")
    if not isinstance(batch, StructureBatch):
        raise TypeError("batch must be a StructureBatch")
    if not isinstance(template_contexts, Mapping):
        raise TypeError("template_contexts must be a mapping")
    if evaluation_policies is not None and not isinstance(
        evaluation_policies, Mapping
    ):
        raise TypeError("evaluation_policies must be a mapping")
    batch.validate()
    if compute_forces and not batch.positions.requires_grad:
        raise ValueError(
            "batch.positions must require gradients when compute_forces=True"
        )
    contexts, policies = _preflight_runtime_bindings(
        model,
        batch,
        template_contexts,
        evaluation_policies,
        solver_path=solver_path,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
        create_graph=create_graph,
    )

    outputs: list[PotentialOutput | None] = [None] * batch.num_structures
    for group in batch.template_groups:
        context = contexts[group.template_id]
        policy = policies.get(group.template_id)
        for structure_index, atom_slice in zip(
            group.structure_indices.tolist(), group.atom_slices
        ):
            try:
                outputs[structure_index] = model(
                    batch.positions[atom_slice],
                    batch.atomic_numbers[atom_slice],
                    batch.cells[structure_index],
                    batch.origins[structure_index],
                    solver_path=solver_path,
                    compute_forces=compute_forces,
                    compute_stress=compute_stress,
                    create_graph=create_graph,
                    return_aux=return_aux,
                    template_context=context,
                    evaluation_policy=policy,
                )
            except Exception as error:
                if solver_path != EVAL_ADAPTIVE:
                    raise
                _raise_grouped_evaluation_error(
                    error,
                    structure_index=structure_index,
                    sample_id=batch.sample_ids[structure_index],
                    template_id=group.template_id,
                )

    if any(output is None for output in outputs):
        raise RuntimeError("internal error: not every batch structure was evaluated")
    ordered = tuple(output for output in outputs if output is not None)

    energy = torch.stack([output.energy for output in ordered])
    baseline_energy = torch.stack(
        [output.baseline_energy for output in ordered]
    )
    residual_energy = torch.stack(
        [output.residual_energy for output in ordered]
    )
    site_energy = torch.cat([output.site_energy for output in ordered], dim=0)
    site_counts = torch.tensor(
        [output.site_energy.shape[0] for output in ordered],
        dtype=torch.long,
        device=batch.device,
    )
    site_ptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=batch.device),
            torch.cumsum(site_counts, dim=0),
        )
    )
    site_batch = torch.repeat_interleave(
        torch.arange(
            batch.num_structures, dtype=torch.long, device=batch.device
        ),
        site_counts,
    )

    forces = None
    if compute_forces:
        if any(output.forces is None for output in ordered):
            raise RuntimeError("model omitted requested forces")
        forces = torch.cat(
            [output.forces for output in ordered if output.forces is not None],
            dim=0,
        )

    stress = None
    stress_voigt = None
    if compute_stress:
        if any(
            output.stress is None or output.stress_voigt is None
            for output in ordered
        ):
            raise RuntimeError("model omitted requested stress")
        stress = torch.stack(
            [output.stress for output in ordered if output.stress is not None]
        )
        stress_voigt = torch.stack(
            [
                output.stress_voigt
                for output in ordered
                if output.stress_voigt is not None
            ]
        )

    auxiliary = (
        tuple(output.auxiliary for output in ordered) if return_aux else None
    )
    return BatchedPotentialOutput(
        energy=energy,
        baseline_energy=baseline_energy,
        residual_energy=residual_energy,
        site_energy=site_energy,
        site_ptr=site_ptr,
        site_batch=site_batch,
        forces=forces,
        stress=stress,
        stress_voigt=stress_voigt,
        sample_ids=batch.sample_ids,
        template_ids=batch.template_ids,
        auxiliary=auxiliary,
    )
