"""Parameter-free mixed-template execution for ragged structure batches."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import (
    CandidateReuseDecision,
    CompactCandidateNeighborState,
    EVAL_ADAPTIVE,
    SparseAdaptiveTransportError,
    TRAIN_FIXED,
    TransportSupportError,
)

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
        support = model.config.transport_support
        if support.backend == "edge_list" and support.kind != "compact_c2":
            raise EvaluationPhaseError(
                "INVALID_SUPPORT_CONFIG",
                "edge-list EVAL_ADAPTIVE requires compact_c2 support",
            )
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


def _preflight_candidate_neighbor_states(
    model: ReferenceSitePotential,
    batch: StructureBatch,
    contexts: Mapping[str, TemplateExecutionContext],
    candidate_neighbor_states: Mapping[
        str, CompactCandidateNeighborState
    ] | None,
    *,
    requested: bool,
    solver_path: str,
) -> dict[str, CompactCandidateNeighborState | None]:
    """Resolve only used sample IDs and reject static state faults pre-forward."""

    if not requested:
        return {}
    support = model.config.transport_support
    if not (
        support.kind == "compact_c2"
        and support.backend == "edge_list"
        and support.candidate_backend == "blocked"
        and support.candidate_skin > 0.0
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "grouped candidate neighbor state requires compact_c2 edge_list blocked support with positive skin",
        )

    resolved: dict[str, CompactCandidateNeighborState | None] = {}
    for structure_index, sample_id in enumerate(batch.sample_ids):
        state = (
            None
            if candidate_neighbor_states is None
            else candidate_neighbor_states.get(sample_id)
        )
        if state is not None and not isinstance(
            state, CompactCandidateNeighborState
        ):
            raise TypeError(
                "candidate_neighbor_states values for used sample IDs must be CompactCandidateNeighborState objects"
            )
        if state is not None:
            template_id = batch.template_ids[structure_index]
            atom_start = int(batch.atom_ptr[structure_index])
            atom_stop = int(batch.atom_ptr[structure_index + 1])
            context = contexts[template_id]
            try:
                state.validate_integrity()
                model._validate_candidate_state_binding(
                    state,
                    template_fingerprint=batch.template_fingerprints[
                        structure_index
                    ],
                    support_config=support,
                    atomic_numbers=batch.atomic_numbers[atom_start:atom_stop],
                    num_sites=context.topology.num_sites,
                )
            except TransportSupportError as error:
                raise TransportSupportError(
                    error.reason_code,
                    "grouped candidate-state preflight failed: "
                    f"structure_index={structure_index} sample_id={sample_id!r} "
                    f"template_id={template_id!r} solver_path={solver_path!r} "
                    "phase_branch='unresolved' candidate_backend='blocked' "
                    "rebuild_stage='state_preflight' original_exception="
                    f"{type(error).__name__}: {error}",
                    template_id=template_id,
                    sample_id=sample_id,
                ) from error
        resolved[sample_id] = state
    return resolved


def _raise_grouped_evaluation_error(
    error: Exception,
    *,
    structure_index: int,
    sample_id: str,
    template_id: str,
    backend: str,
    candidate_backend: str,
    site_block_size: int,
    atom_block_size: int,
    candidate_state_requested: bool = False,
    phase_branch: str | None = None,
) -> None:
    support_fingerprint = getattr(error, "support_fingerprint", None)
    solver_stage = getattr(
        error,
        "stage",
        (
            "candidate_neighbor_state_update"
            if candidate_state_requested
            and isinstance(error, TransportSupportError)
            else "candidate_extraction"
            if isinstance(error, TransportSupportError)
            else None
        ),
    )
    context = (
        f"structure_index={structure_index} sample_id={sample_id!r} "
        f"template_id={template_id!r} stage=single_structure_evaluation "
        f"backend={backend!r} candidate_backend={candidate_backend!r} "
        f"site_block_size={site_block_size} atom_block_size={atom_block_size} "
        f"phase_branch={phase_branch!r} "
        f"rebuild_stage={('state_update' if candidate_state_requested else None)!r} "
        f"support_fingerprint={support_fingerprint!r} "
        f"solver_stage={solver_stage!r} original_exception="
        f"{type(error).__name__}: {error}"
    )
    if isinstance(error, EvaluationPhaseError):
        raise EvaluationPhaseError(
            error.reason_code,
            context,
            template_id=template_id,
            observed=error.observed,
            threshold=error.threshold,
        ) from error
    if isinstance(error, TransportSupportError):
        raise EvaluationPhaseError(
            error.reason_code,
            context,
            template_id=template_id,
        ) from error
    if isinstance(error, SparseAdaptiveTransportError):
        raise EvaluationPhaseError(
            error.reason_code,
            context,
            template_id=template_id,
        ) from error
    raise EvaluationPhaseError(
        "STRUCTURE_EVALUATION_FAILED",
        context,
        template_id=template_id,
    ) from error


def _raise_grouped_blocked_transport_error(
    error: TransportSupportError,
    *,
    structure_index: int,
    sample_id: str,
    template_id: str,
    backend: str,
    candidate_backend: str,
    site_block_size: int,
    atom_block_size: int,
    solver_path: str = TRAIN_FIXED,
    candidate_state_requested: bool = False,
    phase_branch: str | None = None,
) -> None:
    """Add ragged-structure context without changing the support reason code."""

    support_fingerprint = getattr(error, "support_fingerprint", None)
    solver_stage = getattr(
        error,
        "stage",
        "candidate_neighbor_state_update"
        if candidate_state_requested
        else "candidate_extraction",
    )
    context = (
        f"structure_index={structure_index} sample_id={sample_id!r} "
        f"template_id={template_id!r} stage={solver_stage!r} "
        f"backend={backend!r} candidate_backend={candidate_backend!r} "
        f"site_block_size={site_block_size} atom_block_size={atom_block_size} "
        f"solver_path={solver_path!r} phase_branch={phase_branch!r} "
        f"rebuild_stage={('state_update' if candidate_state_requested else None)!r} "
        f"support_fingerprint={support_fingerprint!r} original_exception="
        f"{type(error).__name__}: {error}"
    )
    raise TransportSupportError(
        error.reason_code,
        context,
        template_id=template_id,
        sample_id=sample_id,
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
    candidate_neighbor_states: Mapping[
        str, CompactCandidateNeighborState
    ] | None = None,
    return_candidate_neighbor_states: bool = False,
) -> BatchedPotentialOutput:
    """Evaluate independent structures using one shared model instance.

    Template groups determine deterministic execution order only.  Output slots
    are filled by original structure index, so all public tensors preserve the
    incoming ragged batch order.  For force evaluation, ``batch.positions``
    must already require gradients; this preserves the direct tensor-view path
    and gradients with respect to the caller-owned flattened positions.

    Candidate neighbor states are opt-in and keyed only by stable ``sample_id``.
    Used entries are threaded independently through the single-structure model;
    unused mapping entries are never resolved.  Any state input implies that a
    complete next-state mapping is returned in original sample order.
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
    if candidate_neighbor_states is not None and not isinstance(
        candidate_neighbor_states, Mapping
    ):
        raise TypeError("candidate_neighbor_states must be a mapping")
    if not isinstance(return_candidate_neighbor_states, bool):
        raise TypeError("return_candidate_neighbor_states must be bool")
    candidate_state_requested = (
        candidate_neighbor_states is not None
        or return_candidate_neighbor_states
    )
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
    try:
        resolved_candidate_states = _preflight_candidate_neighbor_states(
            model,
            batch,
            contexts,
            candidate_neighbor_states,
            requested=candidate_state_requested,
            solver_path=solver_path,
        )
    except TransportSupportError as error:
        if solver_path != EVAL_ADAPTIVE:
            raise
        raise EvaluationPhaseError(
            error.reason_code,
            str(error),
            template_id=error.template_id,
        ) from error

    outputs: list[PotentialOutput | None] = [None] * batch.num_structures
    for group in batch.template_groups:
        context = contexts[group.template_id]
        policy = policies.get(group.template_id)
        for structure_index, atom_slice in zip(
            group.structure_indices.tolist(), group.atom_slices
        ):
            sample_id = batch.sample_ids[structure_index]
            input_candidate_state = (
                resolved_candidate_states[sample_id]
                if candidate_state_requested
                else None
            )
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
                    candidate_neighbor_state=input_candidate_state,
                    return_candidate_neighbor_state=candidate_state_requested,
                )
            except Exception as error:
                if solver_path != EVAL_ADAPTIVE:
                    support = model.config.transport_support
                    if (
                        support.candidate_backend == "blocked"
                        and isinstance(error, TransportSupportError)
                    ):
                        _raise_grouped_blocked_transport_error(
                            error,
                            structure_index=structure_index,
                            sample_id=batch.sample_ids[structure_index],
                            template_id=group.template_id,
                            backend=support.backend,
                            candidate_backend=support.candidate_backend,
                            site_block_size=support.site_block_size,
                            atom_block_size=support.atom_block_size,
                            solver_path=solver_path,
                            candidate_state_requested=candidate_state_requested,
                            phase_branch=(
                                input_candidate_state.phase_site_branch_fingerprint
                                if input_candidate_state is not None
                                else None
                            ),
                        )
                    raise
                _raise_grouped_evaluation_error(
                    error,
                    structure_index=structure_index,
                    sample_id=batch.sample_ids[structure_index],
                    template_id=group.template_id,
                    backend=model.config.transport_support.backend,
                    candidate_backend=(
                        model.config.transport_support.candidate_backend
                    ),
                    site_block_size=model.config.transport_support.site_block_size,
                    atom_block_size=model.config.transport_support.atom_block_size,
                    candidate_state_requested=candidate_state_requested,
                    phase_branch=(
                        input_candidate_state.phase_site_branch_fingerprint
                        if input_candidate_state is not None
                        else None
                    ),
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
    next_candidate_states = None
    candidate_reuse_decisions = None
    if candidate_state_requested:
        if any(
            output.candidate_neighbor_state is None
            or output.candidate_reuse_decision is None
            for output in ordered
        ):
            raise RuntimeError(
                "model omitted requested candidate neighbor state output"
            )
        next_candidate_states = {
            sample_id: output.candidate_neighbor_state
            for sample_id, output in zip(batch.sample_ids, ordered)
        }
        candidate_reuse_decisions = {
            sample_id: output.candidate_reuse_decision
            for sample_id, output in zip(batch.sample_ids, ordered)
        }
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
        candidate_neighbor_states=next_candidate_states,
        candidate_reuse_decisions=candidate_reuse_decisions,
    )
