"""Opt-in batch diagnostics; no optimizer step or parameter .grad writes.

Call separately from training: force/stress gradients require higher derivatives
and can be expensive. Reports contain detached Python values, not autograd graphs.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.models import evaluate_structure_batch
from refsite_mlip.transport import TRAIN_FIXED

from .losses import LossConfig, compute_potential_loss
from .step import _active_terms, _has_force_supervision, _has_stress_supervision, _step_batch


def diagnose_training_batch(
    model: torch.nn.Module,
    batch: StructureBatch,
    template_contexts: Mapping,
    *,
    loss_config: LossConfig,
) -> dict[str, Any]:
    """Measure MP activations and each weighted loss's parameter-gradient norm.

    Uses the same TRAIN_FIXED solver, masking, scales and reduction as train_step.
    Existing parameter gradients, RNG state and module training flags are preserved.
    Activation statistics pool tensor elements across structures; a layer's input
    and output are its hidden state, not its auxiliary correlation dictionary.
    Nonfinite losses/gradients are reported rather than silently treated as zero.
    """
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    modes = [(module, module.training) for module in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    accumulators: dict[str, dict[str, Any]] = {}
    handles = []

    def record(name, value):
        if not isinstance(value, torch.Tensor):
            return
        flat = value.detach().to(dtype=torch.float64).reshape(-1)
        stats = accumulators.setdefault(
            name, dict(calls=0, count=0, nonfinite_count=0, square_sum=0.0, abs_max=0.0)
        )
        finite = torch.isfinite(flat)
        stats['calls'] += 1
        stats['count'] += flat.numel()
        stats['nonfinite_count'] += int((~finite).sum())
        good = flat[finite]
        stats['square_sum'] += float(good.square().sum())
        if good.numel():
            stats['abs_max'] = max(stats['abs_max'], float(good.abs().max()))

    def hook(name):
        def capture(module, args, output):
            if args:
                record(name + '.input', args[0])
            record(name + '.output', output[0] if isinstance(output, tuple) else output)
        return capture

    def gradient_report(value):
        connected = bool(value.requires_grad and parameters)
        gradients = (
            torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            if connected else (None,) * len(parameters)
        )
        present = [g.detach().double() for g in gradients if g is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in present)
        norm = math.sqrt(sum(float(g.square().sum()) for g in present)) if finite else None
        return dict(
            gradient_norm=norm,
            gradients_finite=finite,
            parameters_with_grad=len(present),
            parameters_without_grad=len(parameters) - len(present),
        )

    try:
        for name, module in model.named_modules():
            if (name.startswith('layers.') and name.count('.') == 1) or name in (
                'probability_encoder', 'central_encoder', 'readout', 'readout.raw', 'readout.mlp'
            ):
                handles.append(module.register_forward_hook(hook(name)))
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            model.train()
            need_forces = _has_force_supervision(batch, loss_config)
            need_stress = _has_stress_supervision(batch, loss_config)
            step_batch = _step_batch(batch, need_forces=need_forces)
            prediction = evaluate_structure_batch(
                model, step_batch, template_contexts, solver_path=TRAIN_FIXED,
                compute_forces=need_forces, compute_stress=need_stress,
                create_graph=need_forces or need_stress, return_aux=False,
            )
            loss = compute_potential_loss(prediction, step_batch, loss_config)
            terms = {}
            for name, weight, term in _active_terms(loss, loss_config):
                if not bool(term.valid_count):
                    continue
                weighted = weight * term.mean
                finite = bool(torch.isfinite(weighted))
                terms[name] = dict(
                    mean=float(term.mean) if finite else None,
                    weight=weight, weighted_loss=float(weighted) if finite else None,
                    loss_finite=finite, valid_count=int(term.valid_count),
                    **gradient_report(weighted),
                )
            if not terms:
                raise ValueError('no weighted valid supervision for batch diagnostics')
            total = gradient_report(loss.total)
            total['loss'] = float(loss.total) if bool(torch.isfinite(loss.total)) else None
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training

    activations = {}
    for name, stats in accumulators.items():
        count = stats['count']
        square_sum = stats.pop('square_sum')
        invalid = stats['nonfinite_count'] > 0
        stats['rms'] = math.sqrt(square_sum / count) if count and not invalid else None
        if invalid:
            stats['abs_max'] = None
        activations[name] = stats
    return dict(
        schema_version=1, sample_ids=list(batch.sample_ids), solver_path=TRAIN_FIXED,
        loss_config=loss_config.to_dict(), terms=terms, total=total, activations=activations,
    )
