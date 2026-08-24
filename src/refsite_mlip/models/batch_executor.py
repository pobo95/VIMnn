"""Parameter-free mixed-template execution for ragged structure batches."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from refsite_mlip.data import StructureBatch

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


def evaluate_structure_batch(
    model: ReferenceSitePotential,
    batch: StructureBatch,
    template_contexts: Mapping[str, TemplateExecutionContext],
    *,
    solver_path: str,
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
    batch.validate()
    if compute_forces and not batch.positions.requires_grad:
        raise ValueError(
            "batch.positions must require gradients when compute_forces=True"
        )

    outputs: list[PotentialOutput | None] = [None] * batch.num_structures
    for group in batch.template_groups:
        context = _validated_context(
            group.template_id,
            group.template_fingerprint,
            template_contexts,
        )
        for structure_index, atom_slice in zip(
            group.structure_indices.tolist(), group.atom_slices
        ):
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
