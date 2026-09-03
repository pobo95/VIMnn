"""Fresh deterministic training orchestration for canonical run configs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from numbers import Integral
import os
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import torch

from refsite_mlip.config import (
    ResolvedTrainingRun,
    TrainingRunConfig,
    TrainingRunConfigError,
    load_training_run_config,
    resolve_training_run,
)
from refsite_mlip.config.training_run import _load_split, _split_digest
from refsite_mlip.data import StructureBatch, StructureSample, collate_structure_samples
from refsite_mlip.models import (
    ModelBundleError,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
)
from refsite_mlip.training import (
    CheckpointManager,
    CheckpointManagerConfig,
    CheckpointedFitExecutionError,
    FitExecutionError,
    FitProgress,
    ModelSelectionState,
    apply_atomic_baseline_,
    build_optimizer,
    build_scheduler,
    capture_training_checkpoint,
    fit_atomic_baseline,
    run_checkpointed_fit,
)
from refsite_mlip.training.run_directory import (
    RUN_STATUS_SCHEMA_VERSION,
    RunDirectoryError,
    TrainingRunDirectory,
    canonical_runtime_json,
)
from refsite_mlip.transport import TRAIN_FIXED

from .errors import CLIError, CLIInterruptedError
from .validate_train_config import (
    _cli_error as _preflight_cli_error,
    render_train_config_human,
    render_train_config_json,
)


TRAINING_RESULT_SCHEMA_VERSION = "refsite_training_run_result_v1"


@dataclass(frozen=True)
class _PreparedTrainingRuntime:
    bundle: Any
    loaded: Any
    train_samples: tuple[StructureSample, ...]
    validation_samples: tuple[StructureSample, ...]
    train_batches: tuple[StructureBatch, ...]
    validation_batches: tuple[StructureBatch, ...]


def seed_training_runtime(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU, and every available CUDA RNG."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
    ):
        raise ValueError("seed must be an integer and bool is not accepted")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch_seed = seed % 2**64
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def _raise_cli_preflight(error: TrainingRunConfigError, path: Any) -> None:
    raise _preflight_cli_error(error, requested_path=path) from error


def _load_preflight(
    path: str | os.PathLike[str],
) -> tuple[TrainingRunConfig, ResolvedTrainingRun]:
    try:
        config = load_training_run_config(path)
        resolved = resolve_training_run(config)
    except TrainingRunConfigError as error:
        _raise_cli_preflight(error, path)
    return config, resolved


def _runtime_paths(
    resolved: ResolvedTrainingRun,
) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    paths = resolved.runtime_paths
    return (
        Path(str(paths["initial_bundle"])),
        tuple(Path(str(path)) for path in paths["train_inputs"]),
        tuple(Path(str(path)) for path in paths["validation_inputs"]),
    )


def _batch_samples(
    samples: tuple[StructureSample, ...],
    *,
    batch_size: int,
    registry: Any,
    device: str,
    dtype: torch.dtype,
) -> tuple[StructureBatch, ...]:
    batches = []
    for start in range(0, len(samples), batch_size):
        batch = collate_structure_samples(
            samples[start : start + batch_size], registry
        ).to(device=device, dtype=dtype)
        batches.append(batch)
    return tuple(batches)


def _prepare_training_runtime(
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
) -> _PreparedTrainingRuntime:
    bundle_path, train_paths, validation_paths = _runtime_paths(resolved)
    try:
        bundle = load_reference_site_model_bundle(bundle_path, map_location="cpu")
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "initial bundle changed or failed safe validation after preflight",
            stage=error.validation_stage or "training.bundle_reload",
            bundle_path=bundle_path,
            template_id=error.template_id,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "BUNDLE_RELOAD_FAILED",
            "initial bundle could not be reloaded after preflight",
            stage="training.bundle_reload",
            bundle_path=bundle_path,
            underlying_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error
    if bundle.bundle_fingerprint != resolved.bundle_fingerprint:
        raise CLIError(
            "BUNDLE_TOCTOU_MISMATCH",
            "initial bundle semantic fingerprint changed after preflight",
            stage="training.toctou.bundle",
            bundle_path=bundle_path,
            original_error=None,
        )
    try:
        loaded = instantiate_reference_site_model_bundle(
            bundle,
            device=resolved.resolved_device,
            dtype=config.runtime.torch_dtype,
        )
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "initial bundle runtime could not be instantiated",
            stage=error.validation_stage or "training.runtime_instantiate",
            bundle_path=bundle_path,
            template_id=error.template_id,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "RUNTIME_INSTANTIATION_FAILED",
            "initial bundle runtime could not be instantiated",
            stage="training.runtime_instantiate",
            bundle_path=bundle_path,
            underlying_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error

    config_path = None if config.source_path is None else Path(config.source_path)
    try:
        train_samples = _load_split(
            config.data.train,
            train_paths,
            split="train",
            registry=loaded.registry,
            dtype=torch.float64,
            config_path=config_path,
        )
        validation_samples = _load_split(
            config.data.validation,
            validation_paths,
            split="validation",
            registry=loaded.registry,
            dtype=torch.float64,
            config_path=config_path,
        )
    except TrainingRunConfigError as error:
        _raise_cli_preflight(error, config.source_path or "<in-memory-config>")

    templates = {
        binding.template_id: loaded.registry.resolve(binding.template_id)
        for binding in bundle.template_bindings
    }
    train_digest = _split_digest(train_samples, templates, split="train")
    validation_digest = _split_digest(
        validation_samples, templates, split="validation"
    )
    if train_digest != resolved.train_semantic_digest:
        first = train_samples[0] if train_samples else None
        raise CLIError(
            "TRAIN_DATA_TOCTOU_MISMATCH",
            "training data semantic digest changed after preflight",
            stage="training.toctou.data",
            path=train_paths[0] if train_paths else None,
            frame_index=0 if first is not None else None,
            sample_id=None if first is None else first.sample_id,
            template_id=None if first is None else first.template_id,
            split="train",
        )
    if validation_digest != resolved.validation_semantic_digest:
        first = validation_samples[0] if validation_samples else None
        raise CLIError(
            "VALIDATION_DATA_TOCTOU_MISMATCH",
            "validation data semantic digest changed after preflight",
            stage="training.toctou.data",
            path=validation_paths[0] if validation_paths else None,
            frame_index=0 if first is not None else None,
            sample_id=None if first is None else first.sample_id,
            template_id=None if first is None else first.template_id,
            split="validation",
        )

    train_batches = _batch_samples(
        train_samples,
        batch_size=config.data.batch_size,
        registry=loaded.registry,
        device=resolved.resolved_device,
        dtype=config.runtime.torch_dtype,
    )
    validation_batches = _batch_samples(
        validation_samples,
        batch_size=config.data.batch_size,
        registry=loaded.registry,
        device=resolved.resolved_device,
        dtype=config.runtime.torch_dtype,
    )
    if (
        len(train_batches) != resolved.train_batch_count
        or len(validation_batches) != resolved.validation_batch_count
    ):
        raise CLIError(
            "BATCH_PLAN_MISMATCH",
            "runtime batch count differs from the validated deterministic plan",
            stage="training.batch_plan",
            path=config.source_path,
        )
    return _PreparedTrainingRuntime(
        bundle=bundle,
        loaded=loaded,
        train_samples=train_samples,
        validation_samples=validation_samples,
        train_batches=train_batches,
        validation_batches=validation_batches,
    )


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _baseline_metadata(
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
    prepared: _PreparedTrainingRuntime,
) -> dict[str, Any]:
    enabled = config.baseline is not None
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "parameter_update_applied": False,
        "seed": config.runtime.seed,
        "training_run_config_fingerprint": resolved.config_fingerprint,
        "initial_bundle_fingerprint": resolved.bundle_fingerprint,
    }
    if not enabled:
        metadata["reason"] = "baseline config is null"
        return metadata
    assert config.baseline is not None
    fitted = fit_atomic_baseline(
        prepared.train_samples,
        range(len(prepared.train_samples)),
        resolved.species_vocabulary,
        config.baseline,
    )
    apply_atomic_baseline_(prepared.loaded.model, fitted)
    metadata.update(
        {
            "parameter_update_applied": True,
            "config": fitted.config.to_dict(),
            "baseline_energies": fitted.baseline_energies.tolist(),
            "training_sample_ids": list(fitted.training_sample_ids),
            "num_valid_energy_structures": fitted.num_valid_energy_structures,
            "rank": fitted.rank,
            "rank_deficient": fitted.rank_deficient,
            "singular_values": fitted.singular_values.tolist(),
            "condition_number": _finite_or_none(fitted.condition_number),
            "species_occurrence_counts": fitted.species_occurrence_counts.tolist(),
            "residual_rmse": fitted.residual_rmse,
            "residual_mae": fitted.residual_mae,
            "weighted_objective": fitted.weighted_objective,
        }
    )
    return metadata


def _checkpoint_exists(path: Path) -> str | None:
    return str(path) if path.is_file() and not path.is_symlink() else None


def _recoverable_checkpoint_state(
    manager: CheckpointManager,
) -> tuple[int, int, str | None, str | None, Mapping[str, Any] | None]:
    latest_path = manager.root / "latest.pt"
    best_path = manager.root / "best.pt"
    latest = _checkpoint_exists(latest_path)
    best = _checkpoint_exists(best_path)
    if latest is None:
        return 0, 0, None, best, None
    try:
        checkpoint = manager.load_latest()
    except Exception:
        return 0, 0, latest, best, None
    return (
        checkpoint.progress.completed_epochs,
        checkpoint.progress.global_step,
        latest,
        best,
        checkpoint.selection_state.to_dict(),
    )


def _status_base(
    status: str,
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
    *,
    training_executed: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_STATUS_SCHEMA_VERSION,
        "status": status,
        "training_executed": training_executed,
        "config_fingerprint": resolved.config_fingerprint,
        "bundle_fingerprint": resolved.bundle_fingerprint,
        "train_semantic_digest": resolved.train_semantic_digest,
        "validation_semantic_digest": resolved.validation_semantic_digest,
        "seed": config.runtime.seed,
        "runtime": {
            "device": resolved.resolved_device,
            "dtype": resolved.resolved_dtype,
            "solver_path": TRAIN_FIXED,
        },
        "completed_epochs": 0,
        "global_step": config.fit.global_step_start,
        "latest_checkpoint": None,
        "best_checkpoint": None,
        "recoverable_checkpoint": None,
        "terminal_selection_state": None,
        "fit_result": None,
        "baseline": None,
        "failure_phase": None,
        "error": None,
        "rollback_performed": False,
    }


def _completed_status(
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
    result: Any,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    status = _status_base(
        "completed", config, resolved, training_executed=True
    )
    status["result_schema_version"] = TRAINING_RESULT_SCHEMA_VERSION
    fit = result.fit_result
    status.update(
        {
            "completed_epochs": fit.epochs_completed,
            "global_step": fit.global_step_end,
            "latest_checkpoint": result.latest_path,
            "best_checkpoint": result.best_path,
            "recoverable_checkpoint": result.latest_path,
            "terminal_selection_state": fit.final_selection_state.to_dict(),
            "fit_result": fit.to_dict(),
            "baseline": dict(baseline),
        }
    )
    return json.loads(canonical_runtime_json(status))


def _nested_reason(error: BaseException) -> str | None:
    current: BaseException | None = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        reason = getattr(current, "reason_code", None)
        if isinstance(reason, str) and reason:
            return reason
        current = current.__cause__ or current.__context__
    return None


def _nested_text_attribute(error: BaseException, name: str) -> str | None:
    current: BaseException | None = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, name, None)
        if isinstance(value, str) and value:
            return value
        current = current.__cause__ or current.__context__
    return None


def _batch_context(
    error: BaseException,
    prepared: _PreparedTrainingRuntime | None,
    phase: str,
) -> tuple[int | None, str | None, str | None]:
    match = re.search(r"batch_index=(\d+)", str(error))
    if match is None or prepared is None:
        return None, None, None
    index = int(match.group(1))
    batches = (
        prepared.train_batches
        if phase == "train"
        else prepared.validation_batches
    )
    if index >= len(batches):
        return index, None, None
    batch = batches[index]
    return index, batch.sample_ids[0], batch.template_ids[0]


def _failure_details(
    error: BaseException,
    *,
    phase: str,
    manager: CheckpointManager | None,
    prepared: _PreparedTrainingRuntime | None,
) -> dict[str, Any]:
    completed_epochs = 0
    global_step = 0
    latest = best = None
    selection = None
    if manager is not None:
        completed_epochs, global_step, latest, best, selection = (
            _recoverable_checkpoint_state(manager)
        )
    error_phase = phase
    if isinstance(error, CLIError):
        error_phase = error.failure_phase or error.stage.removeprefix("training.")
    elif isinstance(error, RunDirectoryError):
        error_phase = error.stage.removeprefix("run_directory.")
    epoch_index = getattr(error, "epoch_index", None)
    rollback = bool(getattr(error, "rollback_performed", False))
    if isinstance(error, FitExecutionError):
        error_phase = error.phase
        completed_epochs = error.completed_epochs
        global_step = error.current_global_step
    elif isinstance(error, CheckpointedFitExecutionError):
        error_phase = f"checkpoint.{error.failure_stage}"
        completed_epochs = error.epochs_checkpointed
        global_step = error.global_step
    batch_index, sample_id, template_id = _batch_context(
        error, prepared, error_phase
    )
    sample_id = _nested_text_attribute(error, "sample_id") or sample_id
    template_id = _nested_text_attribute(error, "template_id") or template_id
    original_type = (
        getattr(error, "original_exception_type", None) or type(error).__name__
    )
    original_message = (
        getattr(error, "original_exception_message", None) or str(error)
    )
    return {
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "latest_checkpoint": latest,
        "best_checkpoint": best,
        "recoverable_checkpoint": latest,
        "terminal_selection_state": selection,
        "failure_phase": error_phase,
        "rollback_performed": rollback,
        "epoch_index": epoch_index,
        "batch_index": batch_index,
        "sample_id": sample_id,
        "template_id": template_id,
        "error": {
            "type": original_type,
            "message": original_message,
            "reason_code": _nested_reason(error),
        },
    }


def _failed_status(
    status_name: str,
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
    error: BaseException,
    *,
    phase: str,
    manager: CheckpointManager | None,
    prepared: _PreparedTrainingRuntime | None,
    baseline: Mapping[str, Any] | None,
    training_executed: bool,
) -> dict[str, Any]:
    status = _status_base(
        status_name,
        config,
        resolved,
        training_executed=training_executed,
    )
    status.update(
        _failure_details(
            error, phase=phase, manager=manager, prepared=prepared
        )
    )
    status["baseline"] = None if baseline is None else dict(baseline)
    return json.loads(canonical_runtime_json(status))


def _execution_cli_error(
    error: BaseException,
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
    details: Mapping[str, Any],
) -> CLIError:
    if isinstance(error, CLIError):
        return error
    if isinstance(error, RunDirectoryError):
        return CLIError(
            error.reason_code,
            "training runtime metadata operation failed",
            stage=error.stage,
            path=error.path,
            failure_phase=details.get("failure_phase"),
            rollback_performed=details.get("rollback_performed"),
            underlying_reason_code=error.reason_code,
            original_error=error,
        )
    reason = _nested_reason(error) or "TRAINING_EXECUTION_FAILED"
    prediction_stage = _nested_text_attribute(error, "stage")
    return CLIError(
        reason,
        "training run failed; completed checkpoints and updates were retained",
        stage=f"training.{details.get('failure_phase') or 'runtime'}",
        path=config.source_path,
        sample_id=details.get("sample_id"),
        template_id=details.get("template_id"),
        epoch_index=details.get("epoch_index"),
        batch_index=details.get("batch_index"),
        global_step=details.get("global_step"),
        failure_phase=details.get("failure_phase"),
        rollback_performed=details.get("rollback_performed"),
        solver_path=TRAIN_FIXED,
        prediction_stage=prediction_stage,
        predictor_reason_code=(reason if prediction_stage is not None else None),
        underlying_reason_code=reason,
        original_error=error,
    )


def _write_failure_status(
    directory: TrainingRunDirectory,
    status: Mapping[str, Any],
    original_error: BaseException,
) -> BaseException:
    try:
        directory.write_status(status)
    except Exception as status_error:
        status_error.__context__ = original_error
        return status_error
    return original_error


def run_training(
    path: str | os.PathLike[str],
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ResolvedTrainingRun | dict[str, Any]:
    """Preflight and optionally execute one fresh deterministic training run."""

    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    config, resolved = _load_preflight(path)
    if dry_run:
        return resolved

    seed_training_runtime(config.runtime.seed)
    output = Path(str(resolved.runtime_paths["output_directory"]))
    try:
        directory = TrainingRunDirectory.create(output)
    except RunDirectoryError as error:
        raise _execution_cli_error(
            error,
            config,
            resolved,
            {"failure_phase": "run_directory.create", "rollback_performed": False},
        ) from error

    phase = "metadata.resolved_config"
    manager: CheckpointManager | None = None
    prepared: _PreparedTrainingRuntime | None = None
    baseline: dict[str, Any] | None = None
    training_executed = False
    try:
        directory.write_resolved_config(config.to_dict())
        phase = "metadata.preflight"
        directory.write_preflight(resolved.to_dict())
        phase = "metadata.running_status"
        directory.write_status(_status_base("running", config, resolved))

        phase = "runtime.instantiate"
        prepared = _prepare_training_runtime(config, resolved)
        phase = "baseline.fit_apply"
        baseline = _baseline_metadata(config, resolved, prepared)
        phase = "optimizer.create"
        optimizer = build_optimizer(prepared.loaded.model, config.optimizer)
        phase = "scheduler.create"
        scheduler = build_scheduler(optimizer, config.scheduler)
        selection = ModelSelectionState()
        phase = "checkpoint_manager.create"
        manager = CheckpointManager(
            CheckpointManagerConfig(directory=str(directory.checkpoints))
        )
        phase = "checkpoint_metadata.capture"
        checkpoint_metadata = capture_training_checkpoint(
            prepared.loaded.model,
            optimizer,
            scheduler,
            selection,
            FitProgress(next_epoch=0, global_step=0, completed_epochs=0),
            prepared.train_batches,
            prepared.validation_batches,
            model_config=prepared.loaded.model.config,
            loss_config=config.loss,
            optimizer_config=config.optimizer,
            train_step_config=config.train_step,
            validation_step_config=config.validation_step,
            scheduler_config=config.scheduler,
            model_selection_config=config.selection,
            fit_config=config.fit,
            species_vocabulary=resolved.species_vocabulary,
            fit_history=(),
            baseline_fit_metadata=baseline,
        ).metadata
        if progress is not None:
            progress(
                "training started: "
                f"epochs={config.fit.max_epochs}, "
                f"train_batches={len(prepared.train_batches)}, "
                f"validation_batches={len(prepared.validation_batches)}"
            )
        phase = "fit"
        training_executed = True
        result = run_checkpointed_fit(
            prepared.loaded.model,
            optimizer,
            scheduler,
            prepared.train_batches,
            prepared.validation_batches,
            prepared.loaded.template_contexts,
            config.loss,
            config.train_step,
            config.validation_step,
            config.scheduler,
            config.selection,
            selection,
            config.fit,
            manager,
            checkpoint_metadata,
            config.checkpointed_fit,
        )
        status = _completed_status(config, resolved, result, baseline)
        phase = "metadata.completed_status"
        directory.write_status(status)
        if progress is not None:
            progress(
                "training completed: "
                f"epochs={result.fit_result.epochs_completed}, "
                f"global_step={result.fit_result.global_step_end}"
            )
        return status
    except KeyboardInterrupt as error:
        status = _failed_status(
            "interrupted",
            config,
            resolved,
            error,
            phase=phase,
            manager=manager,
            prepared=prepared,
            baseline=baseline,
            training_executed=training_executed,
        )
        stored_error = _write_failure_status(directory, status, error)
        details = status
        raise CLIInterruptedError(
            "TRAINING_INTERRUPTED",
            "training was interrupted; recoverable completed checkpoints were retained",
            stage=f"training.{details['failure_phase']}",
            path=config.source_path,
            sample_id=details.get("sample_id"),
            template_id=details.get("template_id"),
            epoch_index=details.get("epoch_index"),
            batch_index=details.get("batch_index"),
            global_step=details.get("global_step"),
            failure_phase=details.get("failure_phase"),
            rollback_performed=False,
            solver_path=TRAIN_FIXED,
            underlying_reason_code="KEYBOARD_INTERRUPT",
            original_error=stored_error,
        ) from error
    except Exception as error:
        status = _failed_status(
            "failed",
            config,
            resolved,
            error,
            phase=phase,
            manager=manager,
            prepared=prepared,
            baseline=baseline,
            training_executed=training_executed,
        )
        stored_error = _write_failure_status(directory, status, error)
        raise _execution_cli_error(
            stored_error, config, resolved, status
        ) from error


def render_training_json(report: Mapping[str, Any]) -> str:
    """Render a completed/failed run result as deterministic strict JSON."""

    if not isinstance(report, Mapping):
        raise TypeError("training report must be a mapping")
    return canonical_runtime_json(report)


def render_training_human(report: Mapping[str, Any]) -> str:
    """Render a concise deterministic terminal training summary."""

    if not isinstance(report, Mapping):
        raise TypeError("training report must be a mapping")
    if report.get("status") != "completed":
        raise ValueError("human terminal summary requires a completed report")
    fit = report["fit_result"]
    baseline = report["baseline"]
    return "\n".join(
        (
            "Reference-site MLIP training run",
            "Status: completed",
            f"Config SHA-256: {report['config_fingerprint']}",
            f"Bundle SHA-256: {report['bundle_fingerprint']}",
            f"Train semantic SHA-256: {report['train_semantic_digest']}",
            f"Validation semantic SHA-256: {report['validation_semantic_digest']}",
            f"Seed: {report['seed']}",
            f"Runtime: {report['runtime']['device']} / {report['runtime']['dtype']}",
            f"Solver: {report['runtime']['solver_path']}",
            f"Epochs completed: {report['completed_epochs']}",
            f"Global step: {report['global_step']}",
            f"Stopped early: {'yes' if fit['stopped_early'] else 'no'}",
            f"Best epoch: {fit['best_epoch']}",
            f"Best metric: {fit['best_metric']}",
            "Atomic baseline fit applied: "
            f"{'yes' if baseline['parameter_update_applied'] else 'no'}",
            f"Latest checkpoint: {report['latest_checkpoint']}",
            f"Best checkpoint: {report['best_checkpoint']}",
            "No portable prediction bundle was exported.",
        )
    )


def render_train_result_json(
    result: ResolvedTrainingRun | Mapping[str, Any],
) -> str:
    if isinstance(result, ResolvedTrainingRun):
        return render_train_config_json(result)
    return render_training_json(result)


def render_train_result_human(
    result: ResolvedTrainingRun | Mapping[str, Any],
) -> str:
    if isinstance(result, ResolvedTrainingRun):
        return render_train_config_human(result)
    return render_training_human(result)


# Descriptive alias for callers that prefer the command-oriented name.
train_from_config = run_training


__all__ = [
    "TRAINING_RESULT_SCHEMA_VERSION",
    "render_train_result_human",
    "render_train_result_json",
    "render_training_human",
    "render_training_json",
    "run_training",
    "seed_training_runtime",
    "train_from_config",
]
