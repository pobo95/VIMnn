"""Deep CUDA checks executed by the repository-external installed wheel.

This is support code for ``test_installed_cuda_workflow.py`` rather than a
pytest module.  The test copies it, together with the CPU workflow helpers,
outside the checkout and invokes it with the wheel environment's Python.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

import refsite_mlip
from installed_workflow_probe import (
    _assert_no_presentation_time,
    _bundle_parity,
    _finite_tree,
    _frames,
    _geometry_samples,
    _max_error,
    _nested_max_error,
    _strict_json,
)
from refsite_mlip.cli.evaluate import (
    ExtXYZEvaluationConfig,
    _full_prediction,
    _physical_metrics,
    _prepare_labeled_samples,
)
from refsite_mlip.cli.predict import _prepare_samples
from refsite_mlip.cli.resume import _prepare_resume, _resolved_checkpoint_configs
from refsite_mlip.cli.train import _prepare_training_runtime
from refsite_mlip.data import collate_structure_samples
from refsite_mlip.inference import load_reference_site_predictor
from refsite_mlip.interfaces import ReferenceSiteASECalculator
from refsite_mlip.models import load_reference_site_model_bundle
from refsite_mlip.training import (
    ResumePolicy,
    build_optimizer,
    build_scheduler,
    compute_potential_loss,
    load_training_checkpoint,
    restore_training_checkpoint_,
)
from refsite_mlip.training.resume import (
    _numpy_state_from_safe,
    _python_state_from_safe,
)
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED


@dataclass
class _Difference:
    first_mismatch: str | None = None
    max_absolute: float = 0.0
    max_relative: float = 0.0

    def observe(self, path: str, left: torch.Tensor, right: torch.Tensor) -> None:
        if left.numel() == 0:
            return
        first = left.detach().cpu().to(torch.float64)
        second = right.detach().cpu().to(torch.float64)
        absolute = torch.abs(first - second)
        maximum = float(absolute.max())
        scale = torch.maximum(torch.abs(first), torch.abs(second))
        relative = absolute / torch.clamp(scale, min=torch.finfo(torch.float64).tiny)
        relative_maximum = float(relative.max())
        if maximum != 0.0 and self.first_mismatch is None:
            self.first_mismatch = path
        self.max_absolute = max(self.max_absolute, maximum)
        self.max_relative = max(self.max_relative, relative_maximum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_mismatch": self.first_mismatch,
            "max_absolute": self.max_absolute,
            "max_relative": self.max_relative,
        }


def _compare_numeric_tree(
    left: Any,
    right: Any,
    *,
    name: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> _Difference:
    difference = _Difference()

    def visit(first: Any, second: Any, path: str) -> None:
        if isinstance(first, torch.Tensor):
            if not isinstance(second, torch.Tensor):
                raise AssertionError(f"{path} tensor type differs")
            if first.shape != second.shape or first.dtype != second.dtype:
                raise AssertionError(
                    f"{path} tensor shape/dtype differs: "
                    f"{first.shape}/{first.dtype} != {second.shape}/{second.dtype}"
                )
            if not bool(torch.all(torch.isfinite(first))) or not bool(
                torch.all(torch.isfinite(second))
            ):
                raise AssertionError(f"{path} contains a nonfinite tensor")
            if not first.is_floating_point():
                if not torch.equal(first.detach().cpu(), second.detach().cpu()):
                    raise AssertionError(f"{path} exact tensor differs")
                return
            difference.observe(path, first, second)
            torch.testing.assert_close(
                first.detach().cpu(),
                second.detach().cpu(),
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                msg=lambda message: f"{path} exceeded CUDA comparison policy: {message}",
            )
            return
        if isinstance(first, dict):
            if not isinstance(second, dict) or list(first) != list(second):
                raise AssertionError(f"{path} mapping keys/order differ")
            for key in first:
                visit(first[key], second[key], f"{path}.{key}")
            return
        if isinstance(first, (tuple, list)):
            if type(first) is not type(second) or len(first) != len(second):
                raise AssertionError(f"{path} sequence structure differs")
            for index, (item, other) in enumerate(zip(first, second)):
                visit(item, other, f"{path}[{index}]")
            return
        if type(first) is float:
            if type(second) is not float:
                raise AssertionError(f"{path} scalar type differs")
            if not math.isfinite(first) or not math.isfinite(second):
                # PyTorch scheduler state uses signed infinity as a canonical
                # comparison sentinel (for example ``mode_worse``).  It is not
                # a model/optimizer/output numerical result; preserve it only
                # when both semantic states contain the exact same sentinel.
                if first != second:
                    raise AssertionError(f"{path} nonfinite sentinel differs")
                return
            first_tensor = torch.tensor(first, dtype=torch.float64)
            second_tensor = torch.tensor(second, dtype=torch.float64)
            difference.observe(path, first_tensor, second_tensor)
            if not math.isclose(
                first,
                second,
                abs_tol=absolute_tolerance,
                rel_tol=relative_tolerance,
            ):
                raise AssertionError(
                    f"{path} scalar exceeds CUDA comparison policy: {first} != {second}"
                )
            return
        if type(first) is not type(second) or first != second:
            raise AssertionError(f"{path} exact value differs: {first!r} != {second!r}")

    visit(left, right, name)
    return difference


def _tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float32:
        return {
            "trajectory_absolute": 3.0e-5,
            "trajectory_relative": 3.0e-5,
            "output_absolute": 3.0e-5,
            "output_relative": 3.0e-5,
            "extxyz_absolute": 6.0e-5,
            "metric_absolute": 6.0e-5,
            "q_mass": 2.0e-5,
        }
    return {
        "trajectory_absolute": 3.0e-11,
        "trajectory_relative": 3.0e-11,
        "output_absolute": 3.0e-10,
        "output_relative": 3.0e-10,
        "extxyz_absolute": 6.0e-8,
        "metric_absolute": 3.0e-10,
        "q_mass": 2.0e-11,
    }


def _assert_exact_tree(left: Any, right: Any, *, name: str) -> None:
    if isinstance(left, torch.Tensor):
        if not (
            isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        ):
            raise AssertionError(f"{name} tensor differs")
        return
    if isinstance(left, np.ndarray):
        if not (
            isinstance(right, np.ndarray)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        ):
            raise AssertionError(f"{name} array differs")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or list(left) != list(right):
            raise AssertionError(f"{name} mapping differs")
        for key in left:
            _assert_exact_tree(left[key], right[key], name=f"{name}.{key}")
        return
    if isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"{name} sequence differs")
        for index, (item, other) in enumerate(zip(left, right)):
            _assert_exact_tree(item, other, name=f"{name}[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise AssertionError(f"{name} differs: {left!r} != {right!r}")


def _rng_draws(checkpoint: Any) -> dict[str, Any]:
    before = (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
        tuple(value.clone() for value in torch.cuda.get_rng_state_all()),
    )
    try:
        random.setstate(_python_state_from_safe(checkpoint.python_rng_state))
        python_draw = random.random()
        np.random.set_state(_numpy_state_from_safe(checkpoint.numpy_rng_state))
        numpy_draw = float(np.random.random())
        torch.set_rng_state(checkpoint.torch_cpu_rng_state)
        cpu_draw = torch.rand(4, dtype=torch.float64).tolist()
        torch.cuda.set_rng_state_all(list(checkpoint.cuda_rng_states))
        cuda_draw = torch.rand(4, dtype=torch.float64, device="cuda:0")
        torch.cuda.synchronize()
        cuda_values = cuda_draw.cpu().tolist()
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
        torch.cuda.set_rng_state_all(list(before[3]))
    return {
        "python": python_draw,
        "numpy": numpy_draw,
        "torch_cpu": cpu_draw,
        "torch_cuda": cuda_values,
    }


def _checkpoint_and_restore(
    continuous_run: Path,
    split_run: Path,
    *,
    dtype: torch.dtype,
    tolerance: dict[str, float],
) -> tuple[Any, dict[str, Any]]:
    continuous = load_training_checkpoint(
        continuous_run / "checkpoints" / "latest.pt"
    )
    resumed = load_training_checkpoint(split_run / "checkpoints" / "latest.pt")
    if continuous.progress != resumed.progress:
        raise AssertionError("continuous/resumed progress differs")
    if continuous.progress.last_completed_epoch != 1 or continuous.progress.global_step != 2:
        raise AssertionError("terminal checkpoint is not epoch 1/global step 2")
    if continuous.cuda_device_count != resumed.cuda_device_count or resumed.cuda_device_count != 1:
        raise AssertionError("CUDA checkpoint does not record exactly one visible device")
    if len(continuous.cuda_rng_states) != 1 or len(resumed.cuda_rng_states) != 1:
        raise AssertionError("CUDA RNG payload count is not one")

    model_difference = _compare_numeric_tree(
        continuous.model_state_dict,
        resumed.model_state_dict,
        name="model_state",
        absolute_tolerance=tolerance["trajectory_absolute"],
        relative_tolerance=tolerance["trajectory_relative"],
    )
    optimizer_difference = _compare_numeric_tree(
        continuous.optimizer_state_dict,
        resumed.optimizer_state_dict,
        name="optimizer_state",
        absolute_tolerance=tolerance["trajectory_absolute"],
        relative_tolerance=tolerance["trajectory_relative"],
    )
    scheduler_difference = _compare_numeric_tree(
        continuous.scheduler_state_dict,
        resumed.scheduler_state_dict,
        name="scheduler_state",
        absolute_tolerance=tolerance["trajectory_absolute"],
        relative_tolerance=tolerance["trajectory_relative"],
    )
    selection_difference = _compare_numeric_tree(
        continuous.selection_state.to_dict(),
        resumed.selection_state.to_dict(),
        name="selection_state",
        absolute_tolerance=tolerance["trajectory_absolute"],
        relative_tolerance=tolerance["trajectory_relative"],
    )
    history_difference = _compare_numeric_tree(
        list(continuous.fit_history or ()),
        list(resumed.fit_history or ()),
        name="fit_history",
        absolute_tolerance=tolerance["trajectory_absolute"],
        relative_tolerance=tolerance["trajectory_relative"],
    )
    for name in (
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "cuda_rng_states",
    ):
        _assert_exact_tree(
            getattr(continuous, name), getattr(resumed, name), name=name
        )
    continuous_draws = _rng_draws(continuous)
    resumed_draws = _rng_draws(resumed)
    _assert_exact_tree(continuous_draws, resumed_draws, name="next_rng_draws")

    continuous_journal = [
        json.loads(line)
        for line in (continuous_run / "metrics.jsonl").read_text().splitlines()
    ]
    resumed_journal = [
        json.loads(line)
        for line in (split_run / "metrics.jsonl").read_text().splitlines()
    ]
    if [item["epoch_index"] for item in resumed_journal] != [0, 1]:
        raise AssertionError("journal epoch sequence differs")
    if [item["global_step_end"] for item in resumed_journal] != [1, 2]:
        raise AssertionError("journal global-step sequence differs")
    for left, right in zip(continuous_journal, resumed_journal):
        for field in (
            "learning_rates_before_scheduler",
            "learning_rates_after_scheduler",
        ):
            if left[field] != right[field]:
                raise AssertionError(f"exact journal {field} sequence differs")
    journal_difference = _compare_numeric_tree(
        continuous_journal,
        resumed_journal,
        name="metrics_journal",
        absolute_tolerance=tolerance["metric_absolute"],
        relative_tolerance=tolerance["trajectory_relative"],
    )

    # Exercise the production transactional restore path.  This materializes a
    # fresh CUDA model/optimizer/scheduler and finishes with the checkpoint RNG
    # state as the authoritative trajectory state.
    preflight = _prepare_resume(split_run, max_epochs=3)
    prepared = _prepare_training_runtime(preflight.config, preflight.resolved)
    optimizer = build_optimizer(prepared.loaded.model, preflight.config.optimizer)
    scheduler = build_scheduler(optimizer, preflight.config.scheduler)
    resolved_configs = _resolved_checkpoint_configs(preflight, prepared, 3)
    entry_rng = (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
        tuple(value.clone() for value in torch.cuda.get_rng_state_all()),
    )
    try:
        restored = restore_training_checkpoint_(
            preflight.checkpoint,
            prepared.loaded.model,
            optimizer,
            scheduler,
            prepared.train_batches,
            prepared.validation_batches,
            prepared.loaded.template_contexts,
            resolved_configs,
            resumed_max_epochs=3,
            policy=ResumePolicy(),
        )
        if restored.next_epoch != 2 or restored.global_step != 2:
            raise AssertionError("transactional restore returned wrong progress")
        _assert_exact_tree(
            prepared.loaded.model.state_dict(),
            resumed.model_state_dict,
            name="restored_model_state",
        )
        _assert_exact_tree(
            optimizer.state_dict(),
            resumed.optimizer_state_dict,
            name="restored_optimizer_state",
        )
        _assert_exact_tree(
            scheduler.state_dict(),
            resumed.scheduler_state_dict,
            name="restored_scheduler_state",
        )
        restored_draws = {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch_cpu": torch.rand(4, dtype=torch.float64).tolist(),
            "torch_cuda": torch.rand(
                4, dtype=torch.float64, device="cuda:0"
            ).cpu().tolist(),
        }
        torch.cuda.synchronize()
        _assert_exact_tree(restored_draws, resumed_draws, name="restored_rng_draws")
    finally:
        random.setstate(entry_rng[0])
        np.random.set_state(entry_rng[1])
        torch.set_rng_state(entry_rng[2])
        torch.cuda.set_rng_state_all(list(entry_rng[3]))

    model = prepared.loaded.model
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    bound = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    if tuple(map(id, trainable)) != tuple(map(id, bound)) or len(set(map(id, bound))) != len(bound):
        raise AssertionError("optimizer is not bound exactly once to runtime parameters")
    for value in model.state_dict().values():
        if value.is_floating_point() and (
            value.device != torch.device("cuda:0") or value.dtype != dtype
        ):
            raise AssertionError("runtime floating model state device/dtype mismatch")
    baseline = dict(model.named_buffers())["atomic_baseline"]
    if baseline.requires_grad or any(baseline is parameter for parameter in bound):
        raise AssertionError("atomic baseline is not a frozen optimizer-excluded buffer")
    moment_count = 0
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if not bool(torch.all(torch.isfinite(value))):
                    raise AssertionError(f"optimizer state {key} is nonfinite")
                if key in ("exp_avg", "exp_avg_sq"):
                    moment_count += 1
                    if value.device != torch.device("cuda:0") or value.dtype != dtype:
                        raise AssertionError("optimizer moment device/dtype mismatch")
    if moment_count == 0:
        raise AssertionError("optimizer did not contain updated moment tensors")
    if scheduler.optimizer is not optimizer:
        raise AssertionError("scheduler is not bound to the restored optimizer")
    torch.cuda.synchronize()
    return resumed, {
        "history": history_difference.to_dict(),
        "journal": journal_difference.to_dict(),
        "model": model_difference.to_dict(),
        "optimizer": optimizer_difference.to_dict(),
        "scheduler": scheduler_difference.to_dict(),
        "selection": selection_difference.to_dict(),
        "next_rng_draws": resumed_draws,
        "restore_exact": True,
        "runtime_device": "cuda:0",
        "runtime_dtype": str(dtype).removeprefix("torch."),
        "optimizer_moment_tensors": moment_count,
    }


def _prediction_checks(
    bundle_path: Path,
    input_path: Path,
    predictions_path: Path,
    evaluation_report_path: Path,
    *,
    template_key: str,
    dtype: torch.dtype,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    frames = _frames(input_path)
    written = _frames(predictions_path)
    samples = _geometry_samples(frames, template_key)
    cuda_predictor = load_reference_site_predictor(
        bundle_path, device="cuda:0", dtype=dtype
    )
    cpu_predictor = load_reference_site_predictor(
        bundle_path, device="cpu", dtype=dtype
    )
    for value in cuda_predictor.model.state_dict().values():
        if value.is_floating_point() and (
            value.device != torch.device("cuda:0") or value.dtype != dtype
        ):
            raise AssertionError("predictor model device/dtype mismatch")

    cuda_fixed = cuda_predictor.predict_samples(
        samples,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
        return_candidate_neighbor_states=True,
    )
    torch.cuda.synchronize()
    repeated = cuda_predictor.predict_samples(
        samples,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
        return_candidate_neighbor_states=True,
    )
    torch.cuda.synchronize()
    cpu_fixed = cpu_predictor.predict_samples(
        samples,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    if cuda_fixed.device != torch.device("cuda:0") or cuda_fixed.dtype != dtype:
        raise AssertionError("CUDA Predictor output device/dtype mismatch")
    if cuda_fixed.sample_ids != repeated.sample_ids or cuda_fixed.template_ids != repeated.template_ids:
        raise AssertionError("repeated CUDA prediction changed ordering")
    if cuda_fixed.template_ids != tuple(str(frame.info[template_key]) for frame in frames):
        raise AssertionError("CUDA prediction changed mixed-template ordering")

    output_pairs = (
        ("energy", cuda_fixed.energy, cpu_fixed.energy),
        ("forces", cuda_fixed.forces, cpu_fixed.forces),
        ("stress", cuda_fixed.stress, cpu_fixed.stress),
        ("stress_voigt", cuda_fixed.stress_voigt, cpu_fixed.stress_voigt),
    )
    cpu_cuda = {}
    repeated_errors = {}
    for name, cuda_value, cpu_value in output_pairs:
        assert cuda_value is not None and cpu_value is not None
        error = _max_error(cuda_value.detach().cpu().numpy(), cpu_value.numpy())
        cpu_cuda[name] = error
        torch.testing.assert_close(
            cuda_value.detach().cpu(),
            cpu_value,
            atol=tolerance["output_absolute"],
            rtol=tolerance["output_relative"],
        )
        repeated_value = getattr(repeated, name)
        assert repeated_value is not None
        repeated_error = _max_error(
            cuda_value.detach().cpu().numpy(),
            repeated_value.detach().cpu().numpy(),
        )
        repeated_errors[name] = repeated_error
        torch.testing.assert_close(
            cuda_value,
            repeated_value,
            atol=tolerance["output_absolute"],
            rtol=tolerance["output_relative"],
        )

    # Derive K from the exact assigned template rather than from filenames or
    # insertion order.
    expected_k = tuple(
        cuda_predictor.registry.resolve(template_id).topology.num_sites - len(frame)
        for template_id, frame in zip(cuda_fixed.template_ids, frames)
    )
    q_mass_error = 0.0
    for index, (auxiliary, vacancy_mass) in enumerate(zip(cuda_fixed.diagnostics, expected_k)):
        if auxiliary is None:
            raise AssertionError("fixed prediction omitted diagnostics")
        q = auxiliary["q"]
        ot = auxiliary["ot"]
        if q.device != torch.device("cuda:0") or q.dtype != dtype:
            raise AssertionError("fixed q device/dtype mismatch")
        if ot.edge_plan.device != torch.device("cuda:0") or ot.edge_plan.dtype != dtype:
            raise AssertionError("fixed sparse plan device/dtype mismatch")
        if ot.dense_plan_materialized:
            raise AssertionError("fixed edge-list solve materialized a dense plan")
        error = abs(float(q.sum().detach().cpu()) - float(vacancy_mass))
        q_mass_error = max(q_mass_error, error)
        if error > tolerance["q_mass"]:
            raise AssertionError(f"structure {index} q mass mismatch: {error}")
    states = cuda_fixed.candidate_neighbor_states
    if states is None or tuple(states) != cuda_fixed.sample_ids:
        raise AssertionError("candidate neighbor states are missing or reordered")
    for state in states.values():
        if state.device != torch.device("cuda:0") or state.dtype != dtype:
            raise AssertionError("candidate neighbor state device/dtype mismatch")
        if not state.candidate_pair_set_fingerprint:
            raise AssertionError("candidate state omitted its support fingerprint")

    # Exercise the stable policy branch on pristine and vacancy structures.
    adaptive_samples = samples[:2]
    cuda_adaptive = cuda_predictor.predict_samples(
        adaptive_samples,
        solver_path=EVAL_ADAPTIVE,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    torch.cuda.synchronize()
    cpu_adaptive = cpu_predictor.predict_samples(
        adaptive_samples,
        solver_path=EVAL_ADAPTIVE,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    adaptive_errors = {}
    selected_groups = []
    support_fingerprints = []
    for name in ("energy", "forces", "stress", "stress_voigt"):
        cuda_value = getattr(cuda_adaptive, name)
        cpu_value = getattr(cpu_adaptive, name)
        assert cuda_value is not None and cpu_value is not None
        adaptive_errors[name] = _max_error(
            cuda_value.detach().cpu().numpy(), cpu_value.numpy()
        )
        torch.testing.assert_close(
            cuda_value.detach().cpu(),
            cpu_value,
            atol=tolerance["output_absolute"],
            rtol=tolerance["output_relative"],
        )
    for cuda_aux, cpu_aux in zip(cuda_adaptive.diagnostics, cpu_adaptive.diagnostics):
        if cuda_aux is None or cpu_aux is None:
            raise AssertionError("adaptive prediction omitted diagnostics")
        diagnostics = cuda_aux["evaluation_diagnostics"]
        cpu_diagnostics = cpu_aux["evaluation_diagnostics"]
        if diagnostics.transport_backend != "edge_list":
            raise AssertionError("adaptive solve did not use edge-list backend")
        if diagnostics.transport_dense_plan_materialized:
            raise AssertionError("adaptive solve materialized a dense plan")
        if diagnostics.transport_dense_candidate_allocation_observed:
            raise AssertionError("blocked candidate extraction allocated dense geometry")
        if diagnostics.transport_fallback_used or diagnostics.transport_fallback_reason is not None:
            raise AssertionError("adaptive derivative request used transport fallback")
        if diagnostics.derivative_order != 1 or diagnostics.differentiability_scope != "selected_branch_first_order":
            raise AssertionError("adaptive inference did not use first-order selected branch")
        if diagnostics.transport_support_fingerprint is None:
            raise AssertionError("adaptive diagnostics omitted support fingerprint")
        if diagnostics.selected_grouped_index != cpu_diagnostics.selected_grouped_index:
            raise AssertionError("CPU/CUDA selected different adaptive phase groups")
        if diagnostics.transport_support_fingerprint != cpu_diagnostics.transport_support_fingerprint:
            raise AssertionError("CPU/CUDA adaptive support fingerprints differ")
        selected_groups.append(diagnostics.selected_grouped_index)
        support_fingerprints.append(diagnostics.transport_support_fingerprint)
    for tensor in (
        cuda_adaptive.energy,
        cuda_adaptive.forces,
        cuda_adaptive.stress,
        cuda_adaptive.stress_voigt,
    ):
        assert tensor is not None
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise AssertionError("adaptive Predictor exposed a derivative graph")

    if len(written) != len(frames):
        raise AssertionError("CLI predict changed frame count")
    extxyz_errors = {"energy": 0.0, "forces": 0.0, "stress": 0.0}
    ase_errors = {"energy": 0.0, "forces": 0.0, "stress": 0.0}
    adaptive_ase_errors = {"energy": 0.0, "forces": 0.0, "stress": 0.0}
    for index, (source, output, expected) in enumerate(
        zip(frames, written, cuda_fixed.structures)
    ):
        if not np.array_equal(source.numbers, output.numbers) or not np.array_equal(
            source.arrays["input_order"], output.arrays["input_order"]
        ):
            raise AssertionError("CLI extxyz changed atom ordering")
        extxyz_errors["energy"] = max(
            extxyz_errors["energy"],
            abs(output.get_potential_energy() - float(expected.energy)),
        )
        extxyz_errors["forces"] = max(
            extxyz_errors["forces"],
            _max_error(output.get_forces(), expected.forces.detach().cpu().numpy()),
        )
        extxyz_errors["stress"] = max(
            extxyz_errors["stress"],
            _max_error(output.get_stress(), expected.stress_voigt.detach().cpu().numpy()),
        )
        tensor = output.get_stress(voigt=False)
        voigt = output.get_stress()
        if not np.array_equal(voigt[[3, 4, 5]], tensor[[1, 0, 0], [2, 2, 1]]):
            raise AssertionError("CLI extxyz stress Voigt order changed")
        if index < 2:
            atoms = source.copy()
            atoms.calc = ReferenceSiteASECalculator(
                bundle_path,
                template_id=expected.template_id,
                device="cuda:0",
                dtype=dtype,
                solver_path=TRAIN_FIXED,
            )
            ase_errors["energy"] = max(
                ase_errors["energy"],
                abs(atoms.get_potential_energy() - float(expected.energy)),
            )
            ase_errors["forces"] = max(
                ase_errors["forces"],
                _max_error(atoms.get_forces(), expected.forces.detach().cpu().numpy()),
            )
            ase_errors["stress"] = max(
                ase_errors["stress"],
                _max_error(atoms.get_stress(), expected.stress_voigt.detach().cpu().numpy()),
                _max_error(atoms.get_stress(voigt=False), expected.stress.detach().cpu().numpy()),
            )
            for value in atoms.calc.results.values():
                if isinstance(value, torch.Tensor):
                    raise AssertionError("ASE results retained a CUDA/Torch tensor")
            torch.cuda.synchronize()
    if max(extxyz_errors.values()) > tolerance["extxyz_absolute"]:
        raise AssertionError(f"CLI extxyz parity failed: {extxyz_errors}")
    if max(ase_errors.values()) > tolerance["output_absolute"]:
        raise AssertionError(f"fixed Predictor/ASE parity failed: {ase_errors}")

    adaptive_atoms = frames[0].copy()
    adaptive_atoms.calc = ReferenceSiteASECalculator(
        bundle_path,
        template_id=adaptive_samples[0].template_id,
        device="cuda:0",
        dtype=dtype,
        solver_path=EVAL_ADAPTIVE,
    )
    adaptive_expected = cuda_adaptive.structure(0)
    adaptive_ase_errors["energy"] = abs(
        adaptive_atoms.get_potential_energy() - float(adaptive_expected.energy)
    )
    adaptive_ase_errors["forces"] = _max_error(
        adaptive_atoms.get_forces(), adaptive_expected.forces.detach().cpu().numpy()
    )
    adaptive_ase_errors["stress"] = max(
        _max_error(
            adaptive_atoms.get_stress(),
            adaptive_expected.stress_voigt.detach().cpu().numpy(),
        ),
        _max_error(
            adaptive_atoms.get_stress(voigt=False),
            adaptive_expected.stress.detach().cpu().numpy(),
        ),
    )
    torch.cuda.synchronize()
    if max(adaptive_ase_errors.values()) > tolerance["output_absolute"]:
        raise AssertionError(
            f"adaptive Predictor/ASE parity failed: {adaptive_ase_errors}"
        )

    report = _strict_json(evaluation_report_path)
    evaluation_config = ExtXYZEvaluationConfig(
        bundle_path=str(bundle_path),
        input_path=str(input_path),
        template_key=template_key,
        solver_path=TRAIN_FIXED,
        terms=tuple(report["requested_terms"]),
        device="cuda:0",
        dtype=dtype,
        batch_size=2,
        energy_mode=report["energy_mode"],
        energy_scale=report["scales"]["energy"],
        force_scale=report["scales"]["forces"],
        stress_scale=report["scales"]["stress"],
        energy_weight=report["weights"]["energy"],
        force_weight=report["weights"]["forces"],
        stress_weight=report["weights"]["stress"],
    )
    geometry = _prepare_samples(
        frames, cuda_predictor, evaluation_config, input_path.resolve()
    )
    labeled = _prepare_labeled_samples(
        frames, geometry, evaluation_config, input_path.resolve()
    )
    direct_evaluation = cuda_predictor.predict_samples(
        geometry,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=False,
    )
    prediction = _full_prediction(direct_evaluation.structures, evaluation_config)
    batch = collate_structure_samples(labeled, cuda_predictor.registry).to(
        device="cuda:0", dtype=dtype
    )
    with torch.no_grad():
        loss = compute_potential_loss(
            prediction, batch, evaluation_config.loss_config()
        )
    torch.cuda.synchronize()
    expected_loss = {"terms": {}, "total_normalized": float(loss.total.detach().cpu())}
    for term, attribute in (
        ("energy", "energy"),
        ("forces", "force"),
        ("stress", "stress"),
    ):
        value = getattr(loss, attribute)
        expected_loss["terms"][term] = {
            "denominator": float(value.denominator.detach().cpu()),
            "mean": float(value.mean.detach().cpu()),
            "numerator": float(value.numerator.detach().cpu()),
            "valid_count": int(value.valid_count.detach().cpu()),
            "weight": report["weights"][term],
        }
    expected_metrics = _physical_metrics(prediction, batch, evaluation_config.terms)
    evaluation_error = max(
        _nested_max_error(report["loss"], expected_loss),
        _nested_max_error(report["metrics"], expected_metrics),
    )
    if evaluation_error > tolerance["metric_absolute"]:
        raise AssertionError(f"CUDA evaluate/direct parity failed: {evaluation_error}")
    if report["template_frame_counts"] != {"alpha-111": 2, "zeta-211": 2}:
        raise AssertionError("CUDA evaluate changed template ordering/counts")
    if not _finite_tree(report):
        raise AssertionError("CUDA evaluate report is nonfinite")

    cross_dtype = None
    if dtype == torch.float64:
        cast_predictor = load_reference_site_predictor(
            bundle_path, device="cuda:0", dtype=torch.float32
        )
        cast = cast_predictor.predict_samples(
            samples,
            solver_path=TRAIN_FIXED,
            compute_forces=True,
            compute_stress=True,
        )
        torch.cuda.synchronize()
        cross_dtype = {"first_mismatch": None, "outputs": {}}
        for name in ("energy", "forces", "stress", "stress_voigt"):
            first = getattr(cuda_fixed, name)
            second = getattr(cast, name)
            assert first is not None and second is not None
            first_cpu = first.detach().cpu().to(torch.float64)
            second_cpu = second.detach().cpu().to(torch.float64)
            absolute = torch.abs(first_cpu - second_cpu)
            scale = torch.maximum(torch.abs(first_cpu), torch.abs(second_cpu))
            relative = absolute / torch.clamp(
                scale, min=torch.finfo(torch.float64).tiny
            )
            maximum = float(absolute.max()) if absolute.numel() else 0.0
            if maximum != 0.0 and cross_dtype["first_mismatch"] is None:
                cross_dtype["first_mismatch"] = name
            cross_dtype["outputs"][name] = {
                "max_absolute": maximum,
                "max_relative": float(relative.max()) if relative.numel() else 0.0,
            }
            if not bool(torch.all(torch.isfinite(second_cpu))):
                raise AssertionError(f"float32 cast prediction {name} is nonfinite")

    return {
        "adaptive": {
            "cpu_cuda_max_abs_error": adaptive_errors,
            "dense_plan_materialized": False,
            "fallback_used": False,
            "selected_groups": selected_groups,
            "support_fingerprints": support_fingerprints,
        },
        "ase_max_abs_error": ase_errors,
        "adaptive_ase_max_abs_error": adaptive_ase_errors,
        "cli_extxyz_max_abs_error": extxyz_errors,
        "cpu_cuda_max_abs_error": cpu_cuda,
        "cuda_repeat_max_abs_error": repeated_errors,
        "evaluate_direct_max_abs_error": evaluation_error,
        "q_mass_max_abs_error": q_mass_error,
        # No production contract promises that a float64-trained state cast to
        # float32 follows a particular cross-dtype tolerance.  Record the
        # observed error while same-dtype CPU/CUDA and CLI/direct parity above
        # remain enforced with the existing dtype policy.
        "cross_dtype_same_state_error": cross_dtype,
    }


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AssertionError("installed CUDA probe requires exactly one visible GPU")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise AssertionError("installed CUDA probe requires NVIDIA GeForce RTX 3090")
    dtype = getattr(torch, arguments.dtype)
    tolerance = _tolerances(dtype)
    manifest = _strict_json(arguments.fixture_manifest)
    if manifest["transport_backend"] != "edge_list" or manifest["candidate_backend"] != "blocked":
        raise AssertionError("CUDA fixture did not request sparse blocked transport")
    continuous_run = arguments.continuous_run.resolve()
    split_run = arguments.split_run.resolve()

    # PyTorch 2.6 registers its own DTensor/Dynamo safe globals lazily on the
    # first optimizer construction.  Prime that framework-owned initialization
    # before taking the application invariant snapshot; otherwise this probe
    # incorrectly attributes PyTorch's one-time registry setup to refsite_mlip.
    warmup_parameter = torch.nn.Parameter(torch.zeros((), dtype=dtype))
    torch.optim.AdamW((warmup_parameter,))
    del warmup_parameter
    safe_globals_before = tuple(torch.serialization.get_safe_globals())
    original_torch_load = torch.load
    load_contract: list[bool] = []

    def audited_torch_load(*args: Any, **kwargs: Any) -> Any:
        load_contract.append(kwargs.get("weights_only") is True)
        return original_torch_load(*args, **kwargs)

    torch.load = audited_torch_load
    try:
        checkpoint, trajectory = _checkpoint_and_restore(
            continuous_run,
            split_run,
            dtype=dtype,
            tolerance=tolerance,
        )
        latest_bundle, export = _bundle_parity(
            split_run, arguments.best_bundle, arguments.latest_bundle
        )
        if checkpoint.model_state_dict.keys() != latest_bundle.model_state.keys():
            raise AssertionError("checkpoint/export state keys differ")
        for key in checkpoint.model_state_dict:
            _assert_exact_tree(
                checkpoint.model_state_dict[key],
                latest_bundle.model_state[key],
                name=f"exported_state.{key}",
            )
        prediction = _prediction_checks(
            arguments.latest_bundle,
            Path(manifest["cases"]["split"]["mixed_labeled"]),
            arguments.predictions,
            arguments.evaluation_report,
            template_key=manifest["template_key"],
            dtype=dtype,
            tolerance=tolerance,
        )
        for run in (continuous_run, split_run):
            for basename in (
                "resolved_config.json",
                "preflight.json",
                "data_manifest.json",
                "run_status.json",
            ):
                _assert_no_presentation_time(_strict_json(run / basename))
        _assert_no_presentation_time(checkpoint.to_dict())
        _assert_no_presentation_time(latest_bundle.to_payload())
        torch.cuda.synchronize()
    finally:
        torch.load = original_torch_load
    if not load_contract or not all(load_contract):
        raise AssertionError(f"unsafe torch.load call observed: {load_contract}")
    if tuple(torch.serialization.get_safe_globals()) != safe_globals_before:
        raise AssertionError("safe-global registry changed during CUDA workflow probe")
    installed_module = Path(refsite_mlip.__file__).resolve()
    if "site-packages" not in installed_module.parts:
        raise AssertionError("CUDA probe imported refsite_mlip outside site-packages")
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": "refsite_installed_cuda_workflow_probe_v1",
        "status": "passed",
        "device": {
            "name": torch.cuda.get_device_name(0),
            "capability": [properties.major, properties.minor],
            "count": torch.cuda.device_count(),
            "requested": "cuda:0",
        },
        "dtype": arguments.dtype,
        "installed_module": str(installed_module),
        "trajectory": trajectory,
        "prediction": prediction,
        "export": export,
        "torch_load_calls": len(load_contract),
        "weights_only_calls": all(load_contract),
        "safe_global_unchanged": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--continuous-run", required=True, type=Path)
    parser.add_argument("--split-run", required=True, type=Path)
    parser.add_argument("--best-bundle", required=True, type=Path)
    parser.add_argument("--latest-bundle", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    arguments = parser.parse_args(argv)
    report = _run(arguments)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
