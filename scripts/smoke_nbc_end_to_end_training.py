#!/usr/bin/env python3
"""Opt-in end-to-end NbC training-pipeline smoke.

This script is a numerical and integration diagnostic, not a production
accuracy benchmark.  Every external path is explicit.  The reference template
is built exactly once, and its provisional six-mode phase specification and
unit weights are reported rather than presented as an approved production
policy.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np
import torch

from refsite_mlip.data import (
    ExtXYZLoadConfig,
    PhaseSpecification,
    TemplateRegistry,
    build_reference_template_from_poscar,
    collate_structure_samples,
    load_extxyz_samples,
    nbc_rocksalt_template_builder_config,
)
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import (
    PotentialConfig,
    ReferenceSitePotential,
    TemplateExecutionContext,
    evaluate_structure_batch,
)
from refsite_mlip.training import (
    FitConfig,
    FitEpochRecord,
    FitProgress,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    ResumePolicy,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_optimizer,
    build_scheduler,
    capture_training_checkpoint,
    compute_potential_loss,
    fingerprint_batch_sequence,
    load_training_checkpoint,
    process_primary_validation,
    restore_training_checkpoint_,
    run_training_epoch,
    run_validation_epoch,
    save_training_checkpoint,
    train_step,
    validation_step,
)
from refsite_mlip.transport import TransportSupportConfig, materialize_dense_plan


TEMPLATE_ID = "nbc_rocksalt_222_v1"
TRAIN_INDICES = (0, 94, 189, 283)
VALIDATION_INDICES = (0, 30)
EXPECTED_TRAIN_DIGEST = (
    "077e4171dbf948c5497bf5f3d7da742ccd10cdeebb613e3ca3e82c38d0317963"
)
EXPECTED_VALIDATION_DIGEST = (
    "12070901e5d7e34cb705e3bd4083bde9afefc5a6fba551d655d198b2f4aa2217"
)
SEED = 20260831
AVG_NUM_NEIGHBORS = 6.0


def _phase_specification() -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.tensor(
            [
                [-2, 2, 2],
                [2, -2, 2],
                [2, 2, -2],
                [4, 0, 0],
                [0, 4, 0],
                [0, 0, 4],
            ],
            dtype=torch.long,
        ),
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )


def _support(backend: str) -> TransportSupportConfig:
    return TransportSupportConfig(
        kind="compact_c2",
        cutoff=4.0,
        switch_width=0.5,
        candidate_skin=0.2,
        backend=backend,
    )


def _model_config(dtype: torch.dtype, backend: str) -> PotentialConfig:
    del dtype  # dtype is a runtime tensor property, not a serialized model knob.
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=1,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        probability_tolerance=None,
        site_type_vocabulary=(0, 1),
    )
    irreps = "2x0e+2x0e+2x1o+2x2e"
    higher = HigherBodyConfig(
        irreps_feature=irreps,
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
        epsilon_ot=0.5,
        ell_ot=1.5,
        train_sinkhorn_iterations=256,
        transport_support=_support(backend),
    )


def _seed(device: torch.device) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def _make_model(
    template,
    *,
    baseline_value: float,
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
) -> ReferenceSitePotential:
    _seed(device)
    model = ReferenceSitePotential(
        _model_config(dtype, backend),
        template.topology,
        template.phase_modes,
        template.phase_mode_weights,
        torch.eye(2, dtype=torch.float64),
        template.site_alignment_weights,
        template.phase_channel_weights,
        (baseline_value, baseline_value),
    )
    return model.to(device=device, dtype=dtype)


def _batch(samples, indices, registry, *, dtype, device):
    selected = tuple(samples[index].to(device=device, dtype=dtype) for index in indices)
    return collate_structure_samples(selected, registry)


def _selected_label_summary(samples, indices):
    selected = [samples[index] for index in indices]
    energies = torch.stack([sample.energy for sample in selected])
    forces = torch.cat([sample.forces for sample in selected])
    stress = torch.stack([sample.stress for sample in selected])
    return {
        "indices": list(indices),
        "sample_ids": [sample.sample_id for sample in selected],
        "energy_min": float(energies.min()),
        "energy_max": float(energies.max()),
        "force_component_min": float(forces.min()),
        "force_component_max": float(forces.max()),
        "force_norm_min": float(torch.linalg.vector_norm(forces, dim=1).min()),
        "force_norm_max": float(torch.linalg.vector_norm(forces, dim=1).max()),
        "stress_component_min": float(stress.min()),
        "stress_component_max": float(stress.max()),
        "stress_frobenius_min": float(torch.linalg.matrix_norm(stress).min()),
        "stress_frobenius_max": float(torch.linalg.matrix_norm(stress).max()),
    }


def _active_gradients(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _gradient_norm(gradients) -> float:
    if not gradients:
        return 0.0
    return float(
        torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(value).to(torch.float64)
                    for value in gradients.values()
                ]
            )
        ).detach().cpu()
    )


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = (left.detach() - right.detach()).abs()
    scale = torch.maximum(
        right.detach().abs(), right.new_tensor(torch.finfo(right.dtype).tiny)
    )
    return {
        "maximum_absolute": float(delta.max().cpu()) if delta.numel() else 0.0,
        "maximum_relative": (
            float((delta / scale).max().cpu()) if delta.numel() else 0.0
        ),
    }


def _tree_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return list(left) == list(right) and all(
            _tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _rng_snapshot():
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda": tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
    }


def _rng_equal(left, right) -> bool:
    if left["python"] != right["python"]:
        return False
    if left["numpy"][0] != right["numpy"][0]:
        return False
    if not np.array_equal(left["numpy"][1], right["numpy"][1]):
        return False
    if left["numpy"][2:] != right["numpy"][2:]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    return len(left["cuda"]) == len(right["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])
    )


def _batch_pass(
    model,
    batch,
    contexts,
    loss_config,
    *,
    return_aux: bool,
    backward: bool,
):
    local = replace(
        batch,
        positions=batch.positions.detach().clone().requires_grad_(True),
    )
    model.zero_grad(set_to_none=True)
    prediction = evaluate_structure_batch(
        model,
        local,
        contexts,
        compute_forces=loss_config.force_weight > 0.0,
        compute_stress=loss_config.stress_weight > 0.0,
        create_graph=(
            loss_config.force_weight > 0.0 or loss_config.stress_weight > 0.0
        ),
        return_aux=return_aux,
    )
    loss = compute_potential_loss(prediction, local, loss_config)
    if backward:
        loss.total.backward()
    return local, prediction, loss


def _probability_errors(auxiliary, atomic_numbers, *, materialized_plan=None):
    ot = auxiliary["ot"]
    multipoles = auxiliary["multipoles"]
    probabilities = multipoles.species_probabilities
    q = ot.q
    expected = torch.stack(
        [
            torch.count_nonzero(atomic_numbers == 6),
            torch.count_nonzero(atomic_numbers == 41),
        ]
    ).to(probabilities)
    result = {
        "site_simplex_error": float(
            (probabilities.sum(1) + q - 1.0).abs().max().detach().cpu()
        ),
        "species_count_error": float(
            (probabilities.sum(0) - expected).abs().max().detach().cpu()
        ),
        "q_mass_error": float((q.sum() - 1.0).abs().detach().cpu()),
        "effective_tolerances": multipoles.config_metadata.get(
            "effective_probability_validation_tolerances"
        ),
    }
    if materialized_plan is not None:
        result["atom_column_error"] = float(
            (materialized_plan.sum(0) - 1.0).abs().max().detach().cpu()
        )
    return result


def _baseline_comparison(template, batch, contexts, centered_value):
    loss_config = LossConfig(
        energy_weight=1.0,
        force_weight=0.0,
        stress_weight=0.0,
        energy_scale=1.0,
        energy_normalization="per_atom",
    )
    result = {}
    residual_features = None
    parameter_state = None
    for name, baseline in (("zero", 0.0), ("centered", centered_value)):
        model = _make_model(
            template,
            baseline_value=baseline,
            backend="edge_list",
            dtype=batch.dtype,
            device=batch.device,
        )
        _, output, loss = _batch_pass(
            model,
            batch,
            contexts,
            loss_config,
            return_aux=False,
            backward=True,
        )
        gradients = _active_gradients(model)
        total_residual = output.energy.detach() - batch.energy
        result[name] = {
            "baseline_value_per_species": baseline,
            "mean_absolute_total_energy_residual": float(
                total_residual.abs().mean().cpu()
            ),
            "maximum_absolute_total_energy_residual": float(
                total_residual.abs().max().cpu()
            ),
            "mean_absolute_per_atom_residual": float(
                (total_residual / 63.0).abs().mean().cpu()
            ),
            "energy_loss": float(loss.energy.mean.detach().cpu()),
            "parameter_gradient_norm": _gradient_norm(gradients),
        }
        if residual_features is None:
            residual_features = output.residual_energy.detach().cpu()
            parameter_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
                if key != "atomic_baseline"
            }
        else:
            result["residual_energy_exact_parity"] = torch.equal(
                residual_features, output.residual_energy.detach().cpu()
            )
            result["nonbaseline_state_exact_parity"] = all(
                torch.equal(parameter_state[key], value.detach().cpu())
                for key, value in model.state_dict().items()
                if key != "atomic_baseline"
            )
    return result


def _dense_edge_parity(template, batch, contexts, centered_value):
    loss_config = LossConfig(
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.01,
        energy_scale=1.0,
        force_scale=1.0,
        stress_scale=0.1,
        energy_normalization="per_atom",
    )
    dense = _make_model(
        template,
        baseline_value=centered_value,
        backend="dense",
        dtype=batch.dtype,
        device=batch.device,
    )
    edge = _make_model(
        template,
        baseline_value=centered_value,
        backend="edge_list",
        dtype=batch.dtype,
        device=batch.device,
    )
    edge.load_state_dict(dense.state_dict(), strict=True)

    started = time.perf_counter()
    _, dense_output, dense_loss = _batch_pass(
        dense,
        batch,
        contexts,
        loss_config,
        return_aux=True,
        backward=True,
    )
    dense_seconds = time.perf_counter() - started
    dense_gradients = _active_gradients(dense)

    started = time.perf_counter()
    _, edge_output, edge_loss = _batch_pass(
        edge,
        batch,
        contexts,
        loss_config,
        return_aux=True,
        backward=True,
    )
    edge_seconds = time.perf_counter() - started
    edge_gradients = _active_gradients(edge)

    dense_aux = dense_output.auxiliary[0]
    edge_aux = edge_output.auxiliary[0]
    edge_ot = edge_aux["ot"]
    was_materialized = edge_ot.dense_plan_materialized
    edge_plan = materialize_dense_plan(edge_ot).plan
    dense_plan = dense_aux["ot"].P
    gradient_error = max(
        (
            float(
                (dense_gradients[name] - edge_gradients[name])
                .abs()
                .max()
                .detach()
                .cpu()
            )
            for name in dense_gradients
        ),
        default=0.0,
    )
    return {
        "energy": _difference(edge_output.energy, dense_output.energy),
        "site_energy": _difference(
            edge_output.site_energy, dense_output.site_energy
        ),
        "forces": _difference(edge_output.forces, dense_output.forces),
        "stress": _difference(edge_output.stress, dense_output.stress),
        "P_explicitly_materialized": _difference(edge_plan, dense_plan),
        "q": _difference(edge_ot.q, dense_aux["ot"].q),
        "multipoles": _difference(
            edge_aux["multipoles"].equivariant_features,
            dense_aux["multipoles"].equivariant_features,
        ),
        "loss_absolute_difference": float(
            (edge_loss.total - dense_loss.total).abs().detach().cpu()
        ),
        "maximum_parameter_gradient_absolute_difference": gradient_error,
        "dense_seconds": dense_seconds,
        "edge_seconds": edge_seconds,
        "edge_dense_plan_materialized_before_opt_in": was_materialized,
        "edge_result_still_records_no_implicit_densification": (
            not edge_ot.dense_plan_materialized
        ),
        "dense_marginal_residual": max(
            float(dense_aux["ot"].row_residual.detach().cpu()),
            float(dense_aux["ot"].column_residual.detach().cpu()),
        ),
        "edge_marginal_residual": max(
            float(edge_ot.row_residual.detach().cpu()),
            float(edge_ot.column_residual.detach().cpu()),
        ),
        "probability_validation": {
            "dense": _probability_errors(
                dense_aux,
                batch.atomic_numbers,
                materialized_plan=dense_plan,
            ),
            "edge": _probability_errors(
                edge_aux,
                batch.atomic_numbers,
                materialized_plan=edge_plan,
            ),
        },
    }


def _group_gradient_report(model):
    groups = {
        "readout": model.readout.mlp[-1].weight,
        "interaction_radial": model.layers[0].edge.radial_head.network[0].weight,
        "central_site_type": model.central.embedding.weight,
    }
    return {
        name: {
            "finite": parameter.grad is not None
            and bool(torch.all(torch.isfinite(parameter.grad))),
            "norm": (
                0.0
                if parameter.grad is None
                else float(torch.linalg.vector_norm(parameter.grad).detach().cpu())
            ),
            "nonzero": parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad)),
        }
        for name, parameter in groups.items()
    }


def _single_step_gate(
    template,
    batch,
    contexts,
    centered_value,
    *,
    dtype,
    device,
):
    model = _make_model(
        template,
        baseline_value=centered_value,
        backend="edge_list",
        dtype=dtype,
        device=device,
    )
    config = LossConfig(
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.01,
        energy_scale=1.0,
        force_scale=1.0,
        stress_scale=0.1,
        energy_normalization="per_atom",
    )
    input_fingerprint = fingerprint_batch_sequence((batch,), split_name="gate")
    baseline_before = model.atomic_baseline.detach().clone()
    local = replace(
        batch,
        positions=batch.positions.detach().clone().requires_grad_(True),
    )
    output = evaluate_structure_batch(
        model,
        local,
        contexts,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    auxiliary = output.auxiliary[0]
    optimizer = build_optimizer(
        model,
        OptimizerConfig(learning_rate=1.0e-3, weight_decay=0.0),
    )
    result = train_step(
        model,
        optimizer,
        batch,
        contexts,
        config,
        TrainStepConfig(gradient_clip_norm=100.0),
    )
    ot = auxiliary["ot"]
    support = ot.support_diagnostics
    return {
        "dtype": str(dtype),
        "device": str(device),
        "loss": {
            "total": result.total_loss,
            "energy": result.energy_loss,
            "force": result.force_loss,
            "stress": result.stress_loss,
        },
        "gradient": {
            "pre_clip_norm": result.pre_clip_grad_norm,
            "post_clip_norm": result.post_clip_grad_norm,
            "parameter_count": result.number_of_parameters_with_grad,
            "groups": _group_gradient_report(model),
        },
        "finite": {
            "energy": bool(torch.all(torch.isfinite(output.energy))),
            "forces": bool(torch.all(torch.isfinite(output.forces))),
            "stress": bool(torch.all(torch.isfinite(output.stress))),
        },
        "zero_net_force_norm": float(
            torch.linalg.vector_norm(output.forces.sum(0)).detach().cpu()
        ),
        "q_sum": float(ot.q.sum().detach().cpu()),
        "marginal_residual": max(
            float(ot.row_residual.detach().cpu()),
            float(ot.column_residual.detach().cpu()),
        ),
        "probability_validation": _probability_errors(
            auxiliary, batch.atomic_numbers
        ),
        "support": {
            "candidate_edges": support.candidate_edge_count,
            "active_edges": support.active_edge_count,
            "matching": support.maximum_atom_matching_size,
            "total_support": support.total_support_feasible,
            "dense_ratio": support.candidate_dense_ratio,
        },
        "input_unchanged": input_fingerprint
        == fingerprint_batch_sequence((batch,), split_name="gate"),
        "baseline_frozen": torch.equal(
            baseline_before, model.atomic_baseline.detach()
        ),
        "baseline_requires_grad": model.atomic_baseline.requires_grad,
        "normal_training_auxiliary_is_none": model(
            batch.positions[:63],
            batch.atomic_numbers[:63],
            batch.cells[0],
            batch.origins[0],
            template_context=contexts[TEMPLATE_ID],
            return_aux=False,
        ).auxiliary
        is None,
    }


def _validation_with_preservation(
    model, batch, contexts, loss_config, config
):
    model_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    gradients = {
        name: (
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for name, parameter in model.named_parameters()
    }
    mode = model.training
    rng = _rng_snapshot()
    result = validation_step(model, batch, contexts, loss_config, config)
    state_preserved = all(
        torch.equal(model.state_dict()[key], value)
        for key, value in model_state.items()
    )
    gradient_preserved = True
    for name, parameter in model.named_parameters():
        identity, value = gradients[name]
        if parameter.grad is not identity:
            gradient_preserved = False
            break
        if value is not None and not torch.equal(parameter.grad, value):
            gradient_preserved = False
            break
    return result, {
        "model_state_preserved": state_preserved,
        "gradient_identity_and_value_preserved": gradient_preserved,
        "mode_preserved": model.training == mode,
        "rng_preserved": _rng_equal(rng, _rng_snapshot()),
    }


def _overfit_once(
    template,
    train_batch,
    validation_batch,
    contexts,
    centered_value,
    *,
    steps,
    learning_rate,
):
    started = time.perf_counter()
    model = _make_model(
        template,
        baseline_value=centered_value,
        backend="edge_list",
        dtype=train_batch.dtype,
        device=train_batch.device,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=learning_rate, weight_decay=0.0
    )
    optimizer = build_optimizer(model, optimizer_config)
    loss_config = LossConfig(
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.0,
        energy_scale=1.0,
        force_scale=1.0,
        stress_scale=0.1,
        energy_normalization="per_atom",
    )
    validation_config = ValidationStepConfig()
    step_config = TrainStepConfig(gradient_clip_norm=10.0)
    input_fingerprint = fingerprint_batch_sequence(
        (train_batch,), split_name="overfit"
    )
    baseline_before = model.atomic_baseline.detach().clone()
    groups = {
        "readout": model.readout.mlp[-1].weight,
        "interaction_radial": model.layers[0].edge.radial_head.network[0].weight,
        "central_site_type": model.central.embedding.weight,
    }
    group_before = {
        name: parameter.detach().clone() for name, parameter in groups.items()
    }
    initial, _ = _validation_with_preservation(
        model, train_batch, contexts, loss_config, validation_config
    )
    validation_before, validation_before_preservation = (
        _validation_with_preservation(
            model,
            validation_batch,
            contexts,
            loss_config,
            validation_config,
        )
    )
    curve = []
    step_results = []
    for _ in range(steps):
        step_result = train_step(
            model,
            optimizer,
            train_batch,
            contexts,
            loss_config,
            step_config,
        )
        curve.append(step_result.total_loss)
        step_results.append(step_result)
    final, final_preservation = _validation_with_preservation(
        model, train_batch, contexts, loss_config, validation_config
    )
    validation_after, validation_after_preservation = (
        _validation_with_preservation(
            model,
            validation_batch,
            contexts,
            loss_config,
            validation_config,
        )
    )
    if not final.total_loss < initial.total_loss:
        raise RuntimeError(
            "real-data overfit did not reduce total loss; inspect reported curve"
        )
    changed = {
        name: float(
            (parameter.detach() - group_before[name]).abs().max().cpu()
        )
        for name, parameter in groups.items()
    }
    return {
        "model": model,
        "optimizer": optimizer,
        "optimizer_config": optimizer_config,
        "loss_config": loss_config,
        "step_config": step_config,
        "validation_config": validation_config,
        "curve": tuple(curve),
        "report": {
            "elapsed_seconds": time.perf_counter() - started,
            "steps": steps,
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "gradient_clip_norm": 10.0,
            "initial": {
                "total": initial.total_loss,
                "energy": initial.energy_loss,
                "force": initial.force_loss,
            },
            "final": {
                "total": final.total_loss,
                "energy": final.energy_loss,
                "force": final.force_loss,
            },
            "final_over_initial": final.total_loss / initial.total_loss,
            "pre_update_loss_curve": list(curve),
            "all_curve_values_finite": all(math.isfinite(value) for value in curve),
            "gradient_norm_range": [
                min(result.pre_clip_grad_norm for result in step_results),
                max(result.pre_clip_grad_norm for result in step_results),
            ],
            "changed_parameter_groups_max_abs": changed,
            "all_required_groups_changed": all(value > 0.0 for value in changed.values()),
            "baseline_frozen": torch.equal(
                baseline_before, model.atomic_baseline.detach()
            ),
            "input_batch_unchanged": input_fingerprint
            == fingerprint_batch_sequence((train_batch,), split_name="overfit"),
            "validation_before": {
                "total": validation_before.total_loss,
                "energy": validation_before.energy_loss,
                "force": validation_before.force_loss,
            },
            "validation_after": {
                "total": validation_after.total_loss,
                "energy": validation_after.energy_loss,
                "force": validation_after.force_loss,
            },
            "validation_state_preservation": {
                "before": validation_before_preservation,
                "train_final": final_preservation,
                "after": validation_after_preservation,
            },
        },
    }


def _overfit_determinism(first, second) -> dict[str, object]:
    curve_equal = first["curve"] == second["curve"]
    model_equal = _tree_equal(
        first["model"].state_dict(), second["model"].state_dict()
    )
    optimizer_equal = _tree_equal(
        first["optimizer"].state_dict(), second["optimizer"].state_dict()
    )
    max_curve_difference = max(
        (abs(a - b) for a, b in zip(first["curve"], second["curve"])),
        default=0.0,
    )
    return {
        "loss_curve_exact": curve_equal,
        "final_model_exact": model_equal,
        "optimizer_state_exact": optimizer_equal,
        "maximum_loss_curve_difference": max_curve_difference,
    }


def _cuda_backend_performance(
    template,
    batch,
    contexts,
    centered_value,
    *,
    dtype,
):
    if not torch.cuda.is_available():
        return {"available": False}
    loss_config = LossConfig(
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.01,
        energy_normalization="per_atom",
        stress_scale=0.1,
    )
    results = {"available": True, "dtype": str(dtype)}
    for backend in ("dense", "edge_list"):
        model = _make_model(
            template,
            baseline_value=centered_value,
            backend=backend,
            dtype=dtype,
            device=torch.device("cuda"),
        )
        model.zero_grad(set_to_none=True)
        _batch_pass(
            model,
            batch,
            contexts,
            loss_config,
            return_aux=False,
            backward=True,
        )
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        started = time.perf_counter()
        _, output, loss = _batch_pass(
            model,
            batch,
            contexts,
            loss_config,
            return_aux=False,
            backward=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        results[backend] = {
            "combined_forward_backward_seconds": elapsed,
            "baseline_allocated_bytes": baseline_memory,
            "peak_allocated_bytes": peak,
            "incremental_peak_bytes": peak - baseline_memory,
            "finite": bool(torch.isfinite(loss.total))
            and bool(torch.all(torch.isfinite(output.forces)))
            and bool(torch.all(torch.isfinite(output.stress))),
            "auxiliary_retained": output.auxiliary is not None,
        }
        del output, loss, model
        torch.cuda.empty_cache()
    return results


def _checkpoint_smoke(
    overfit,
    template,
    train_batch,
    validation_batch,
    contexts,
    centered_value,
    baseline_metadata,
    source_git_commit,
):
    model = overfit["model"]
    optimizer = overfit["optimizer"]
    loss_config = overfit["loss_config"]
    optimizer_config = overfit["optimizer_config"]
    train_step_config = overfit["step_config"]
    validation_step_config = overfit["validation_config"]
    scheduler_config = SchedulerConfig(kind="none")
    selection_config = ModelSelectionConfig(
        monitor="total_loss", mode="min", min_delta=0.0
    )
    scheduler = build_scheduler(optimizer, scheduler_config)
    global_step_start = len(overfit["curve"])
    learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
    training = run_training_epoch(
        model,
        optimizer,
        (train_batch,),
        contexts,
        loss_config,
        train_step_config,
        epoch_index=0,
        global_step_start=global_step_start,
    )
    validation = run_validation_epoch(
        model,
        (validation_batch,),
        contexts,
        loss_config,
        validation_step_config,
        epoch_index=0,
        global_step=training.global_step_end,
    )
    selection, decision = process_primary_validation(
        optimizer,
        scheduler,
        validation,
        scheduler_config,
        selection_config,
        ModelSelectionState(),
    )
    record = FitEpochRecord(
        epoch_index=0,
        training=training,
        validation=validation,
        decision=decision,
        selection_state_after_epoch=selection,
        learning_rates_used_for_training=learning_rates,
        learning_rates_after_validation=tuple(
            float(group["lr"]) for group in optimizer.param_groups
        ),
    )
    progress = FitProgress(
        next_epoch=1,
        global_step=training.global_step_end,
        completed_epochs=1,
        last_completed_epoch=0,
        best_epoch=selection.best_epoch,
        best_global_step=selection.best_global_step,
    )
    saved_fit_config = FitConfig(
        max_epochs=1,
        start_epoch=0,
        global_step_start=global_step_start,
    )
    checkpoint = capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
        selection,
        progress,
        (train_batch,),
        (validation_batch,),
        model_config=model.config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        train_step_config=train_step_config,
        validation_step_config=validation_step_config,
        scheduler_config=scheduler_config,
        model_selection_config=selection_config,
        fit_config=saved_fit_config,
        species_vocabulary=(6, 41),
        fit_history=(record,),
        baseline_fit_metadata=baseline_metadata,
        source_git_commit=source_git_commit,
    )
    with tempfile.TemporaryDirectory(prefix="vimnn-nbc-smoke-") as directory:
        path = Path(directory) / "epoch_000000.pt"
        save_training_checkpoint(checkpoint, path)
        loaded = load_training_checkpoint(path, map_location="cpu")

    fresh = _make_model(
        template,
        baseline_value=centered_value,
        backend="edge_list",
        dtype=model.atomic_baseline.dtype,
        device=model.atomic_baseline.device,
    )
    fresh_optimizer = build_optimizer(fresh, optimizer_config)
    fresh_scheduler = build_scheduler(fresh_optimizer, scheduler_config)
    resumed_fit_config = FitConfig(
        max_epochs=2,
        start_epoch=0,
        global_step_start=global_step_start,
    )
    resolved = {
        "model": fresh.config,
        "loss": loss_config,
        "optimizer": optimizer_config,
        "train_step": train_step_config,
        "validation_step": validation_step_config,
        "scheduler": scheduler_config,
        "model_selection": selection_config,
        "fit": resumed_fit_config,
        "baseline_fit_metadata": baseline_metadata,
    }
    resume = restore_training_checkpoint_(
        loaded,
        fresh,
        fresh_optimizer,
        fresh_scheduler,
        (train_batch,),
        (validation_batch,),
        contexts,
        resolved,
        resumed_max_epochs=2,
        policy=ResumePolicy(),
        current_source_git_commit=source_git_commit,
    )
    return {
        "training_epoch": training.to_dict(),
        "validation_epoch": validation.to_dict(),
        "selection": selection.to_dict(),
        "progress": progress.to_dict(),
        "weights_only_load_schema": loaded.schema_version,
        "checkpoint_baseline_exact": torch.equal(
            loaded.model_state_dict["atomic_baseline"],
            model.atomic_baseline.detach().cpu(),
        ),
        "model_state_exact_after_restore": _tree_equal(
            fresh.state_dict(), loaded.model_state_dict
        ),
        "optimizer_state_exact_after_restore": _tree_equal(
            fresh_optimizer.state_dict(), loaded.optimizer_state_dict
        ),
        "scheduler_state_exact_after_restore": _tree_equal(
            fresh_scheduler.state_dict(), loaded.scheduler_state_dict
        ),
        "template_fingerprint_mapping": loaded.metadata.template_fingerprints,
        "resume": {
            "next_epoch": resume.next_epoch,
            "global_step": resume.global_step,
            "completed_epochs": resume.completed_epochs,
            "exact_resume_ready": resume.exact_resume_ready,
            "restored_rng_domains": list(resume.restored_rng_domains),
            "diagnostics": list(resume.compatibility_diagnostics),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar-222", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--skip-cuda", action="store_true")
    return parser


def main() -> None:
    smoke_started = time.perf_counter()
    args = _parser().parse_args()
    if not 20 <= args.steps <= 50:
        raise ValueError("--steps must remain in the requested 20..50 smoke range")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be finite and positive")

    source_git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    phase = _phase_specification()
    template_build_count = 0
    template_started = time.perf_counter()
    built = build_reference_template_from_poscar(
        args.poscar_222,
        config=nbc_rocksalt_template_builder_config((2, 2, 2)),
        phase_specification=phase,
    )
    template_build_count += 1
    template_seconds = time.perf_counter() - template_started
    registry = TemplateRegistry()
    registry.add(built.template)
    context = TemplateExecutionContext.from_reference_template(
        built.template, avg_num_neighbors=AVG_NUM_NEIGHBORS
    )
    contexts = {TEMPLATE_ID: context}

    train_result = load_extxyz_samples(
        ExtXYZLoadConfig(
            source_path=str(args.train),
            sample_id_prefix="train",
            template_id=TEMPLATE_ID,
            dtype=torch.float64,
            device="cpu",
        ),
        registry,
    )
    validation_result = load_extxyz_samples(
        ExtXYZLoadConfig(
            source_path=str(args.validation),
            sample_id_prefix="validation",
            template_id=TEMPLATE_ID,
            dtype=torch.float64,
            device="cpu",
        ),
        registry,
    )
    if train_result.diagnostics.semantic_sha256 != EXPECTED_TRAIN_DIGEST:
        raise RuntimeError("train semantic digest differs from the 8C-1 contract")
    if validation_result.diagnostics.semantic_sha256 != EXPECTED_VALIDATION_DIGEST:
        raise RuntimeError(
            "validation semantic digest differs from the 8C-1 contract"
        )

    train_mean = math.fsum(
        float(sample.energy) for sample in train_result.samples
    ) / len(train_result.samples)
    centered_value = train_mean / 63.0
    baseline_metadata = {
        "kind": "training_only_fixed_composition_centering_gauge",
        "physical_atomic_e0": False,
        "training_frame_count": 284,
        "training_semantic_sha256": EXPECTED_TRAIN_DIGEST,
        "mean_training_total_energy": train_mean,
        "atoms_per_structure": 63,
        "species_vocabulary": [6, 41],
        "baseline_values": [centered_value, centered_value],
        "validation_labels_used": False,
    }

    cpu = torch.device("cpu")
    train_cpu = _batch(
        train_result.samples,
        TRAIN_INDICES,
        registry,
        dtype=torch.float64,
        device=cpu,
    )
    validation_cpu = _batch(
        validation_result.samples,
        VALIDATION_INDICES,
        registry,
        dtype=torch.float64,
        device=cpu,
    )
    one_cpu = _batch(
        train_result.samples,
        (TRAIN_INDICES[0],),
        registry,
        dtype=torch.float64,
        device=cpu,
    )

    report = {
        "scope": (
            "pipeline smoke only; no production accuracy claim; provisional "
            "phase modes and unit weights"
        ),
        "source_git_commit": source_git_commit,
        "template": {
            "template_id": TEMPLATE_ID,
            "fingerprint": built.template.fingerprint,
            "build_count": template_build_count,
            "build_seconds": template_seconds,
            "phase_approval_status": phase.approval_status,
            "phase_modes": phase.modes.tolist(),
            "phase_weights": phase.mode_weights.tolist(),
            "context_fingerprint": context.fingerprint,
        },
        "data": {
            "train_frames": len(train_result.samples),
            "validation_frames": len(validation_result.samples),
            "train_semantic_digest": train_result.diagnostics.semantic_sha256,
            "validation_semantic_digest": (
                validation_result.diagnostics.semantic_sha256
            ),
            "digests_match_8c1": True,
            "selected_train": _selected_label_summary(
                train_result.samples, TRAIN_INDICES
            ),
            "selected_validation": _selected_label_summary(
                validation_result.samples, VALIDATION_INDICES
            ),
        },
        "baseline": {
            "rank_identifiability": (
                "rank one fixed-composition design; species E0 values are not "
                "separately identifiable"
            ),
            "training_mean_total_energy": train_mean,
            "centering_value_per_active_species": centered_value,
            "validation_labels_used": False,
            "interpretation": "numerical fixed-composition gauge, not physical E0",
        },
    }

    # Keep the baseline comparison on the normative CPU float64 path so its
    # residual-network parity check is bitwise deterministic.
    baseline_batch = train_cpu
    report["baseline"]["initial_comparison"] = _baseline_comparison(
        built.template, baseline_batch, contexts, centered_value
    )

    report["dense_edge_cpu_float64"] = _dense_edge_parity(
        built.template, one_cpu, contexts, centered_value
    )
    report["actual_forward_backward_gates"] = {
        "cpu_float64": _single_step_gate(
            built.template,
            one_cpu,
            contexts,
            centered_value,
            dtype=torch.float64,
            device=cpu,
        )
    }

    if torch.cuda.is_available() and not args.skip_cuda:
        cuda = torch.device("cuda")
        for dtype in (torch.float32, torch.float64):
            one_cuda = _batch(
                train_result.samples,
                (TRAIN_INDICES[0],),
                registry,
                dtype=dtype,
                device=cuda,
            )
            report["actual_forward_backward_gates"][str(dtype)] = (
                _single_step_gate(
                    built.template,
                    one_cuda,
                    contexts,
                    centered_value,
                    dtype=dtype,
                    device=cuda,
                )
            )
        performance_batch = _batch(
            train_result.samples,
            (TRAIN_INDICES[0],),
            registry,
            dtype=torch.float32,
            device=cuda,
        )
        report["cuda_dense_edge_performance"] = _cuda_backend_performance(
            built.template,
            performance_batch,
            contexts,
            centered_value,
            dtype=torch.float32,
        )
    else:
        report["cuda_dense_edge_performance"] = {"available": False}

    # CPU float64 is the normative deterministic overfit/checkpoint path. CUDA
    # is exercised above as a backend smoke without claiming bitwise trajectory
    # determinism for segmented/index-add reductions.
    overfit_train = train_cpu
    overfit_validation = validation_cpu

    first = _overfit_once(
        built.template,
        overfit_train,
        overfit_validation,
        contexts,
        centered_value,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    second = _overfit_once(
        built.template,
        overfit_train,
        overfit_validation,
        contexts,
        centered_value,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    report["overfit"] = first["report"]
    report["overfit"]["device"] = str(overfit_train.device)
    report["overfit"]["dtype"] = str(overfit_train.dtype)
    report["overfit"]["determinism"] = _overfit_determinism(first, second)
    report["checkpoint"] = _checkpoint_smoke(
        first,
        built.template,
        overfit_train,
        overfit_validation,
        contexts,
        centered_value,
        baseline_metadata,
        source_git_commit,
    )
    report["production_limitations"] = [
        "phase modes and unit weights remain provisional",
        "fixed-composition centering is not a physical isolated-atom E0 fit",
        "four-frame overfit and two-frame validation do not measure accuracy",
        "edge-list candidate discovery still uses the existing dense distance audit",
        "no DataLoader, sampler, multi-epoch production fit, or CLI is exercised",
    ]
    report["total_elapsed_seconds"] = time.perf_counter() - smoke_started
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
