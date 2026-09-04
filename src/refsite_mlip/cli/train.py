"""Fresh deterministic training orchestration for canonical run configs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
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
    ResolvedScratchTrainingRun,
    ResolvedTrainingRun,
    ScratchModelSourceConfig,
    TrainingRunConfig,
    TrainingRunConfigOverrides,
    TrainingRunConfigError,
    load_effective_training_run_config,
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
    MetricsJournal,
    MetricsJournalError,
    ModelSelectionState,
    ScratchTrainingPreparation,
    apply_atomic_baseline_,
    build_optimizer,
    build_scheduler,
    capture_training_checkpoint,
    committed_epoch_provenance_from_checkpoint_metadata,
    fit_atomic_baseline,
    prepare_scratch_training_run,
    run_checkpointed_fit,
)
from refsite_mlip.training.run_directory import (
    RUN_STATUS_SCHEMA_VERSION,
    ResumeRunLock,
    RunDirectoryError,
    TrainingRunDirectory,
    canonical_runtime_json,
)
from refsite_mlip.transport import TRAIN_FIXED

from .errors import CLIConfigPreflightError, CLIError, CLIInterruptedError
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


def _raise_cli_preflight(
    error: TrainingRunConfigError,
    path: Any,
    *,
    error_type: type[CLIError] = CLIError,
) -> None:
    raise _preflight_cli_error(
        error,
        requested_path=path,
        error_type=error_type,
    ) from error


def _load_preflight(
    path: str | os.PathLike[str],
    *,
    overrides: TrainingRunConfigOverrides | None = None,
    cli_cwd: str | os.PathLike[str] | None = None,
    stage: Callable[[str], None] | None = None,
) -> tuple[
    TrainingRunConfig,
    ResolvedTrainingRun | ResolvedScratchTrainingRun | ScratchTrainingPreparation,
]:
    try:
        config = load_effective_training_run_config(
            path, overrides, cli_cwd=cli_cwd
        )
    except TrainingRunConfigError as error:
        _raise_cli_preflight(error, path, error_type=CLIConfigPreflightError)
    if isinstance(config.model_source, ScratchModelSourceConfig):
        if stage is not None:
            stage("preparing data and reference templates")
        try:
            resolved = prepare_scratch_training_run(config)
        except TrainingRunConfigError as error:
            converted = _preflight_cli_error(
                error,
                requested_path=path,
                error_type=CLIConfigPreflightError,
            )
            raise converted from error
    else:
        if stage is not None:
            stage("preparing data and model bundle")
        try:
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
        batch_size=config.data.effective_validation_batch_size,
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
    journal: MetricsJournal,
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
    status.update(journal.summary().to_dict())
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


def _nested_exception(
    error: BaseException, expected_type: type[BaseException]
) -> BaseException | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, expected_type):
            return current
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
        if error.failure_stage == "epoch_observer":
            error_phase = (
                "metrics_journal"
                if _nested_exception(error, MetricsJournalError) is not None
                else "epoch_observer"
            )
        else:
            error_phase = f"checkpoint.{error.failure_stage}"
        completed_epochs = error.epochs_checkpointed
        global_step = error.global_step
    batch_index, sample_id, template_id = _batch_context(
        error, prepared, error_phase
    )
    sample_id = _nested_text_attribute(error, "sample_id") or sample_id
    template_id = _nested_text_attribute(error, "template_id") or template_id
    journal_error = _nested_exception(error, MetricsJournalError)
    diagnostic_error = error if journal_error is None else journal_error
    original_type = (
        getattr(diagnostic_error, "original_exception_type", None)
        or type(diagnostic_error).__name__
    )
    original_message = (
        getattr(diagnostic_error, "original_exception_message", None)
        or str(diagnostic_error)
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
    journal: MetricsJournal | None = None,
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
    if journal is not None:
        try:
            status.update(journal.summary().to_dict())
        except MetricsJournalError as summary_error:
            journal_error = _nested_exception(error, MetricsJournalError)
            diagnostic = (
                journal_error
                if journal_error is not None
                and getattr(
                    journal_error, "last_valid_semantic_sha256", None
                )
                is not None
                else summary_error
            )
            last_valid = getattr(diagnostic, "last_valid_epoch", None)
            count = getattr(diagnostic, "last_valid_event_count", None)
            if count is None:
                count = 0 if last_valid is None else last_valid + 1
            semantic_sha = getattr(
                diagnostic, "last_valid_semantic_sha256", None
            ) or hashlib.sha256(b"").hexdigest()
            status.update(
                {
                    "metrics_journal": journal.config.filename,
                    "metrics_event_count": count,
                    "metrics_last_epoch": last_valid,
                    "metrics_semantic_sha256": semantic_sha,
                }
            )
    return json.loads(canonical_runtime_json(status))


def _attach_progress_context(
    error: CLIError,
    details: Mapping[str, Any],
) -> CLIError:
    """Retain non-semantic recovery facts for terminal presentation only."""

    for name in (
        "completed_epochs",
        "global_step",
        "recoverable_checkpoint",
        "recoverable_initial_bundle",
    ):
        value = details.get(name)
        if value is not None and getattr(error, name, None) is None:
            setattr(error, name, value)
    return error


def _execution_cli_error(
    error: BaseException,
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
    details: Mapping[str, Any],
) -> CLIError:
    if isinstance(error, CLIError):
        return _attach_progress_context(error, details)
    if isinstance(error, RunDirectoryError):
        return _attach_progress_context(CLIError(
            error.reason_code,
            "training runtime metadata operation failed",
            stage=error.stage,
            path=error.path,
            failure_phase=details.get("failure_phase"),
            rollback_performed=details.get("rollback_performed"),
            underlying_reason_code=error.reason_code,
            original_error=error,
        ), details)
    reason = _nested_reason(error) or "TRAINING_EXECUTION_FAILED"
    prediction_stage = _nested_text_attribute(error, "stage")
    return _attach_progress_context(CLIError(
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
    ), details)


def _nested_exception_is_interrupt(error: BaseException) -> bool:
    """Recognize an interrupt retained behind a structured training error."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, KeyboardInterrupt):
            return True
        if getattr(current, "interrupted", False) is True or getattr(
            current, "status", None
        ) == "interrupted":
            return True
        for nested in (
            getattr(current, "original_error", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _scratch_error_attribute(
    error: BaseException,
    *names: str,
) -> Any:
    """Return the first retained scratch-error attribute with a value."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for name in names:
            value = getattr(current, name, None)
            if value is not None:
                return value
        for nested in (
            getattr(current, "original_error", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return None


def _scratch_execution_cli_error(
    error: BaseException,
    config: TrainingRunConfig,
) -> CLIError:
    """Preserve scratch orchestration context at the public CLI boundary."""

    interrupted = _nested_exception_is_interrupt(error)
    error_type = CLIInterruptedError if interrupted else CLIError
    reason = _scratch_error_attribute(error, "reason_code")
    if not isinstance(reason, str) or not reason:
        reason = (
            "SCRATCH_TRAINING_INTERRUPTED"
            if interrupted
            else "SCRATCH_TRAINING_FAILED"
        )
    stage = _scratch_error_attribute(error, "stage")
    if not isinstance(stage, str) or not stage:
        stage = "scratch_training"
    failure_phase = _scratch_error_attribute(error, "failure_phase")
    if not isinstance(failure_phase, str) or not failure_phase:
        failure_phase = stage
    message = _scratch_error_attribute(error, "message")
    if not isinstance(message, str) or not message:
        message = (
            "scratch training was interrupted; durable completed state was retained"
            if interrupted
            else "scratch checkpointed training failed; durable state was retained"
        )
    underlying = _scratch_error_attribute(error, "original_reason_code")
    if not isinstance(underlying, str) or not underlying:
        underlying = _nested_reason(error) or reason
    path = _scratch_error_attribute(error, "output_path", "path")
    if path is None:
        path = config.source_path
    epoch_index = _scratch_error_attribute(
        error, "epoch_index", "completed_epochs"
    )
    converted = error_type(
        reason,
        message,
        stage=stage,
        path=path,
        source_path=config.source_path,
        sample_id=_scratch_error_attribute(error, "sample_id"),
        template_id=_scratch_error_attribute(error, "template_id"),
        epoch_index=epoch_index,
        batch_index=_scratch_error_attribute(error, "batch_index"),
        global_step=_scratch_error_attribute(error, "global_step"),
        failure_phase=failure_phase,
        rollback_performed=getattr(error, "rollback_performed", False),
        source_kind="scratch",
        run_directory=_scratch_error_attribute(error, "output_path"),
        bundle_fingerprint=_scratch_error_attribute(
            error, "bundle_fingerprint", "initial_bundle_fingerprint"
        ),
        config_fingerprint=_scratch_error_attribute(
            error, "config_fingerprint"
        ),
        solver_path=TRAIN_FIXED,
        underlying_reason_code=underlying,
        original_error=error,
    )
    return _attach_progress_context(
        converted,
        {
            "completed_epochs": _scratch_error_attribute(
                error, "completed_epochs"
            ),
            "global_step": _scratch_error_attribute(error, "global_step"),
            "recoverable_checkpoint": _scratch_error_attribute(
                error, "recoverable_checkpoint"
            ),
            "recoverable_initial_bundle": _scratch_error_attribute(
                error, "recoverable_initial_bundle"
            ),
        },
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


def _reload_effective_config_for_toctou(
    config: TrainingRunConfig,
    *,
    overrides: TrainingRunConfigOverrides | None,
    cli_cwd: str | os.PathLike[str],
) -> TrainingRunConfig:
    """Rebuild exactly the effective config validated before lock acquisition."""

    if config.source_path is None:
        raise CLIError(
            "CONFIG_SOURCE_MISSING",
            "fresh training requires the original resolved config path",
            stage="training.config.toctou",
        )
    try:
        current = load_effective_training_run_config(
            config.source_path,
            overrides,
            cli_cwd=cli_cwd,
        )
    except TrainingRunConfigError as error:
        _raise_cli_preflight(error, config.source_path)
    if current.config_fingerprint != config.config_fingerprint:
        raise CLIError(
            "TRAIN_CONFIG_TOCTOU_MISMATCH",
            "training-run config changed between preflight and lock acquisition",
            stage="training.config.toctou",
            path=config.source_path,
        )
    return current


def _parameter_summary(model: torch.nn.Module) -> tuple[int, int]:
    """Snapshot parameter cardinality without retaining any live tensor."""

    parameters = tuple(model.parameters())
    return len(parameters), sum(int(parameter.numel()) for parameter in parameters)


def _composition_key(entry: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    species = entry.get("species")
    if not isinstance(species, (tuple, list)):
        raise TypeError("composition species must be a sequence")
    return tuple(
        sorted(
            (
                int(item["atomic_number"]),
                int(item["count"]),
            )
            for item in species
        )
    )


def _composition_name(composition: tuple[tuple[int, int], ...]) -> str:
    # ASE is already a direct runtime dependency.  Use its canonical element
    # symbols only for presentation; no chemistry or training state is derived
    # here.
    from ase.data import chemical_symbols

    def symbol(atomic_number: int) -> str:
        if (
            0 < atomic_number < len(chemical_symbols)
            and chemical_symbols[atomic_number]
        ):
            return chemical_symbols[atomic_number]
        return f"Z{atomic_number}"

    return " ".join(
        f"{symbol(atomic_number)}{count}"
        for atomic_number, count in composition
    )


def _composition_summary(
    train: Any,
    validation: Any,
) -> tuple[tuple[str, int, int], ...]:
    def counts(values: Any) -> dict[tuple[tuple[int, int], ...], int]:
        result: dict[tuple[tuple[int, int], ...], int] = {}
        for item in values:
            if not isinstance(item, Mapping):
                raise TypeError("composition statistics entries must be mappings")
            key = _composition_key(item)
            result[key] = result.get(key, 0) + int(item["frame_count"])
        return result

    train_counts = counts(train)
    validation_counts = counts(validation)
    return tuple(
        (
            _composition_name(key),
            train_counts.get(key, 0),
            validation_counts.get(key, 0),
        )
        for key in sorted(set(train_counts) | set(validation_counts))
    )


def _label_presence(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> tuple[tuple[str, int, int, int, int], ...]:
    return tuple(
        (
            display_term,
            int(train[source_term]["present_frames"]),
            int(train[source_term]["missing_frames"]),
            int(validation[source_term]["present_frames"]),
            int(validation[source_term]["missing_frames"]),
        )
        for display_term, source_term in (
            ("energy", "energy"),
            ("force", "forces"),
            ("stress", "stress"),
        )
    )


def _baseline_presentation(
    config: TrainingRunConfig,
    metadata: Mapping[str, Any],
) -> tuple[bool, tuple[float, ...], str, str]:
    enabled = bool(metadata.get("enabled", config.baseline is not None))
    if not enabled:
        return False, (), "n/a", "disabled"
    if config.baseline is None:
        raise ValueError("enabled baseline metadata requires baseline config")
    values = tuple(float(value) for value in metadata.get("baseline_energies", ()))
    rank_deficient = bool(metadata.get("rank_deficient", False))
    status = "rank_deficient" if rank_deficient else "full_rank"
    return True, values, config.baseline.rank_policy, status


def _start_summary(
    *,
    config: TrainingRunConfig,
    source_kind: str,
    model: torch.nn.Module,
    device: str,
    dtype: str,
    initialization_seed: int | None,
    species_vocabulary: tuple[int, ...],
    templates: tuple[tuple[str, int], ...],
    default_template_id: str,
    train_frame_count: int,
    validation_frame_count: int,
    train_batch_count: int,
    validation_batch_count: int,
    template_frame_counts: tuple[tuple[str, int, int], ...],
    composition_summary: tuple[tuple[str, int, int], ...],
    label_presence: tuple[tuple[str, int, int, int, int], ...],
    baseline_metadata: Mapping[str, Any],
    initial_bundle_fingerprint: str,
    train_semantic_digest: str,
    validation_semantic_digest: str,
    output_directory: str,
) -> Any:
    from .training_progress import TrainingStartSummary

    parameter_tensors, parameter_elements = _parameter_summary(model)
    baseline_enabled, baseline_values, rank_policy, baseline_status = (
        _baseline_presentation(config, baseline_metadata)
    )
    radii = config.radii.derived
    return TrainingStartSummary(
        run_name=Path(output_directory).name,
        source_kind=source_kind,
        device=device,
        dtype=dtype,
        training_seed=config.runtime.seed,
        initialization_seed=initialization_seed,
        parameter_tensor_count=parameter_tensors,
        parameter_element_count=parameter_elements,
        species_vocabulary=tuple(species_vocabulary),
        templates=tuple(sorted(templates)),
        default_template_id=default_template_id,
        train_frame_count=train_frame_count,
        validation_frame_count=validation_frame_count,
        train_batch_count=train_batch_count,
        validation_batch_count=validation_batch_count,
        train_batch_size=config.data.batch_size,
        validation_batch_size=config.data.effective_validation_batch_size,
        template_frame_counts=tuple(sorted(template_frame_counts)),
        composition_summary=composition_summary,
        label_presence=label_presence,
        r_ot=config.radii.r_ot,
        r_mp=config.radii.r_mp,
        r_candidate_ot=radii.r_candidate_ot,
        r_candidate_mp=radii.r_candidate_mp,
        ot_backend=model.config.transport_support.backend,
        solver_path="TRAIN_FIXED",
        baseline_enabled=baseline_enabled,
        baseline_values=baseline_values,
        baseline_rank_policy=rank_policy,
        baseline_status=baseline_status,
        loss_terms=(
            ("energy", config.loss.energy_weight, config.loss.energy_scale),
            ("force", config.loss.force_weight, config.loss.force_scale),
            ("stress", config.loss.stress_weight, config.loss.stress_scale),
        ),
        optimizer_kind=config.optimizer.optimizer,
        initial_learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        scheduler_kind=config.scheduler.kind,
        scheduler_monitor=config.scheduler.monitor,
        scheduler_mode=config.scheduler.mode,
        max_epochs=config.fit.max_epochs,
        early_stop_patience=config.selection.early_stopping_patience,
        output_directory=output_directory,
        initial_bundle_fingerprint=initial_bundle_fingerprint,
        train_semantic_digest=train_semantic_digest,
        validation_semantic_digest=validation_semantic_digest,
    )


def _bundle_training_start_summary(
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
    prepared: _PreparedTrainingRuntime,
    baseline: Mapping[str, Any],
) -> Any:
    templates = tuple(
        (
            binding.template_id,
            int(binding.structural_artifact.diagnostics.num_sites),
        )
        for binding in prepared.bundle.template_bindings
    )
    template_ids = tuple(template_id for template_id, _ in templates)
    return _start_summary(
        config=config,
        source_kind="bundle",
        model=prepared.loaded.model,
        device=resolved.resolved_device,
        dtype=resolved.resolved_dtype,
        initialization_seed=None,
        species_vocabulary=resolved.species_vocabulary,
        templates=templates,
        default_template_id=prepared.bundle.default_template_id,
        train_frame_count=resolved.train_frame_count,
        validation_frame_count=resolved.validation_frame_count,
        train_batch_count=resolved.train_batch_count,
        validation_batch_count=resolved.validation_batch_count,
        template_frame_counts=tuple(
            (
                template_id,
                int(resolved.train_template_frame_counts.get(template_id, 0)),
                int(resolved.validation_template_frame_counts.get(template_id, 0)),
            )
            for template_id in sorted(template_ids)
        ),
        composition_summary=_composition_summary(
            resolved.train_composition_statistics,
            resolved.validation_composition_statistics,
        ),
        label_presence=_label_presence(
            resolved.train_label_statistics,
            resolved.validation_label_statistics,
        ),
        baseline_metadata=baseline,
        initial_bundle_fingerprint=resolved.bundle_fingerprint,
        train_semantic_digest=resolved.train_semantic_digest,
        validation_semantic_digest=resolved.validation_semantic_digest,
        output_directory=str(resolved.runtime_paths["output_directory"]),
    )


def _scratch_manifest_template_counts(
    preparation: ScratchTrainingPreparation,
) -> tuple[tuple[str, int, int], ...]:
    def split_counts(name: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in preparation.data_manifest[name]["samples"]:
            template_id = str(item["template_id"])
            result[template_id] = result.get(template_id, 0) + 1
        return result

    train = split_counts("train")
    validation = split_counts("validation")
    return tuple(
        (template_id, train.get(template_id, 0), validation.get(template_id, 0))
        for template_id in sorted(set(train) | set(validation))
    )


def _scratch_training_start_summary(
    config: TrainingRunConfig,
    preparation: ScratchTrainingPreparation,
    startup: Any,
) -> Any:
    templates = tuple(
        (
            template_id,
            int(preparation.template_fingerprints[template_id]["num_sites"]),
        )
        for template_id in sorted(preparation.template_fingerprints)
    )
    return _start_summary(
        config=config,
        source_kind="scratch",
        model=startup.model,
        device=preparation.resolved_device,
        dtype=preparation.resolved_dtype,
        initialization_seed=startup.initialization_seed,
        species_vocabulary=preparation.species_vocabulary,
        templates=templates,
        default_template_id=startup.initial_bundle.default_template_id,
        train_frame_count=int(preparation.data_manifest["train"]["frame_count"]),
        validation_frame_count=int(
            preparation.data_manifest["validation"]["frame_count"]
        ),
        train_batch_count=len(startup.train_batches),
        validation_batch_count=len(startup.validation_batches),
        template_frame_counts=_scratch_manifest_template_counts(preparation),
        composition_summary=_composition_summary(
            preparation.train_composition_statistics,
            preparation.validation_composition_statistics,
        ),
        label_presence=_label_presence(
            preparation.train_label_statistics,
            preparation.validation_label_statistics,
        ),
        baseline_metadata=startup.baseline_metadata,
        initial_bundle_fingerprint=startup.initial_bundle_fingerprint,
        train_semantic_digest=preparation.train_semantic_digest,
        validation_semantic_digest=preparation.validation_semantic_digest,
        output_directory=str(startup.run_directory.root),
    )


def _execute_training(
    config: TrainingRunConfig,
    resolved: ResolvedTrainingRun,
    directory: TrainingRunDirectory,
    lock: ResumeRunLock,
    *,
    progress: Callable[[str], None] | None,
    progress_renderer: Any | None,
    overrides: TrainingRunConfigOverrides | None,
    cli_cwd: str | os.PathLike[str],
) -> dict[str, Any]:
    """Execute a fresh run while the caller owns the run-directory lock."""

    phase = "config.toctou"
    manager: CheckpointManager | None = None
    prepared: _PreparedTrainingRuntime | None = None
    baseline: dict[str, Any] | None = None
    journal: MetricsJournal | None = None
    training_executed = False
    try:
        _reload_effective_config_for_toctou(
            config,
            overrides=overrides,
            cli_cwd=cli_cwd,
        )
        phase = "metadata.resolved_config"
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
        phase = "metrics_journal.initialize"
        provenance = committed_epoch_provenance_from_checkpoint_metadata(
            checkpoint_metadata,
            initial_bundle_fingerprint=resolved.bundle_fingerprint,
        )
        journal = MetricsJournal(directory, lock, provenance)
        if progress_renderer is not None:
            progress_renderer.render_start_from(
                lambda: _bundle_training_start_summary(
                    config, resolved, prepared, baseline
                )
            )
            progress_renderer.render_stage("training started")
        if progress is not None:
            progress(
                "training started: "
                f"epochs={config.fit.max_epochs}, "
                f"train_batches={len(prepared.train_batches)}, "
                f"validation_batches={len(prepared.validation_batches)}"
            )
        phase = "fit"
        training_executed = True
        epoch_observer = journal
        if progress_renderer is not None:
            from .training_progress import journal_then_progress_observer

            epoch_observer = journal_then_progress_observer(
                journal, progress_renderer
            )
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
            epoch_metrics_provenance=provenance,
            epoch_metrics_observer=epoch_observer,
        )
        status = _completed_status(
            config, resolved, result, baseline, journal
        )
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
            journal=journal,
        )
        stored_error = _write_failure_status(directory, status, error)
        details = status
        interrupted_error = CLIInterruptedError(
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
        )
        raise _attach_progress_context(interrupted_error, details) from error
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
            journal=journal,
        )
        stored_error = _write_failure_status(directory, status, error)
        raise _execution_cli_error(
            stored_error, config, resolved, status
        ) from error


def _run_training_impl(
    path: str | os.PathLike[str],
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
    progress_renderer: Any | None = None,
    overrides: TrainingRunConfigOverrides | None = None,
    cli_cwd: str | os.PathLike[str] | None = None,
) -> (
    ResolvedTrainingRun
    | ResolvedScratchTrainingRun
    | ScratchTrainingPreparation
    | dict[str, Any]
):
    """Preflight and optionally execute one fresh deterministic training run."""

    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    if progress_renderer is not None:
        progress_renderer.render_stage("loading training configuration")
    effective_cli_cwd = Path.cwd() if cli_cwd is None else Path(cli_cwd)
    config, resolved = _load_preflight(
        path,
        overrides=overrides,
        cli_cwd=effective_cli_cwd,
        stage=(
            None
            if progress_renderer is None
            else progress_renderer.render_stage
        ),
    )
    if dry_run:
        return resolved
    if isinstance(resolved, ScratchTrainingPreparation):
        # Imported only on the execution branch so validate/dry-run retain their
        # strictly read-only dependency boundary.
        from refsite_mlip.training import (
            ScratchCheckpointedTrainingError,
            run_scratch_checkpointed_training,
        )

        try:
            if progress_renderer is not None:
                progress_renderer.render_stage("initializing training run")

            def observe_startup(startup: Any) -> None:
                if progress_renderer is None:
                    return
                progress_renderer.render_start_from(
                    lambda: _scratch_training_start_summary(
                        config, resolved, startup
                    )
                )
                progress_renderer.render_stage("training started")

            scratch_presentation: dict[str, Any] = {}
            if progress_renderer is not None:
                scratch_presentation.update(
                    startup_observer=observe_startup,
                    committed_epoch_observer=progress_renderer,
                )
            result = run_scratch_checkpointed_training(
                config,
                resolved,
                progress=progress,
                **scratch_presentation,
            )
        except KeyboardInterrupt as error:
            retained = getattr(error, "scratch_training_error", error)
            raise _scratch_execution_cli_error(retained, config) from error
        except ScratchCheckpointedTrainingError as error:
            raise _scratch_execution_cli_error(error, config) from error
        # Keep the command-level JSON contract source-independent: both
        # bundle and scratch fresh runs return the flat terminal status.  The
        # richer composed result remains available from the public training
        # orchestration API.
        payload = result.to_dict()["terminal_status"]
        if not isinstance(payload, Mapping):
            raise CLIError(
                "INVALID_SCRATCH_TRAINING_RESULT",
                "scratch training result did not provide a mapping payload",
                stage="scratch_training.result",
                path=config.source_path,
                source_kind="scratch",
            )
        return json.loads(canonical_runtime_json(payload))
    if isinstance(resolved, ResolvedScratchTrainingRun):
        raise CLIError(
            "SCRATCH_FULL_PREFLIGHT_REQUIRED",
            "scratch execution requires full POSCAR/data preparation",
            stage="model_source.preflight",
            path=config.source_path,
            config_field="model_source.kind",
            source_kind="scratch",
            underlying_reason_code="SCRATCH_FULL_PREFLIGHT_REQUIRED",
        )

    if progress_renderer is not None:
        progress_renderer.render_stage("initializing training run")
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
    try:
        lock = directory.acquire_resume_lock()
    except RunDirectoryError as error:
        raise _execution_cli_error(
            error,
            config,
            resolved,
            {"failure_phase": "run_directory.lock", "rollback_performed": False},
        ) from error
    try:
        with lock:
            return _execute_training(
                config,
                resolved,
                directory,
                lock,
                progress=progress,
                progress_renderer=progress_renderer,
                overrides=overrides,
                cli_cwd=effective_cli_cwd,
            )
    except (CLIError, CLIInterruptedError):
        raise
    except RunDirectoryError as error:
        raise _execution_cli_error(
            error,
            config,
            resolved,
            {"failure_phase": "run_directory.lock", "rollback_performed": False},
        ) from error


def _terminal_basename(value: Any) -> str | None:
    return None if value is None else Path(str(value)).name


def _nested_value(error: BaseException, name: str) -> Any:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, name, None)
        if value is not None:
            return value
        current = current.__cause__ or current.__context__
    return None


def _render_terminal_result(progress_renderer: Any, report: Mapping[str, Any]) -> None:
    fit = report.get("fit_result")
    fit = fit if isinstance(fit, Mapping) else {}
    stopped_early = bool(fit.get("stopped_early", False))
    status = "early_stopped" if stopped_early else "completed"
    progress_renderer.render_terminal(
        status,
        epochs=int(report.get("completed_epochs", 0)),
        global_step=int(report.get("global_step", 0)),
        best_epoch=(
            None if fit.get("best_epoch") is None else int(fit["best_epoch"])
        ),
        best_value=(
            None if fit.get("best_metric") is None else float(fit["best_metric"])
        ),
        latest_checkpoint=_terminal_basename(report.get("latest_checkpoint")),
        reason=(None if fit.get("stop_reason") is None else str(fit["stop_reason"])),
        recoverable=_terminal_basename(report.get("recoverable_checkpoint")),
    )


def _render_terminal_error(progress_renderer: Any, error: BaseException) -> None:
    interrupted = isinstance(error, (KeyboardInterrupt, CLIInterruptedError))
    status = "interrupted" if interrupted else "failed"
    completed = _nested_value(error, "completed_epochs")
    epoch_index = _nested_value(error, "epoch_index")
    global_step = _nested_value(error, "global_step")
    phase = _nested_value(error, "failure_phase") or _nested_value(
        error, "stage"
    )
    recoverable = _nested_value(error, "recoverable_checkpoint")
    if recoverable is None:
        recoverable = _nested_value(error, "recoverable_initial_bundle")
    # Epoch indices in structured runtime failures are persisted 0-based,
    # while console epoch numbers are consistently human-facing and 1-based.
    # Preflight/startup failures have no epoch index and retain the completed
    # epoch count (normally zero).
    displayed_epoch = (
        max(0, int(epoch_index)) + 1
        if epoch_index is not None
        else (0 if completed is None else max(0, int(completed)))
    )
    progress_renderer.render_terminal(
        status,
        epochs=displayed_epoch,
        global_step=0 if global_step is None else max(0, int(global_step)),
        phase=None if phase is None else str(phase),
        recoverable=_terminal_basename(recoverable),
    )


def run_training(
    path: str | os.PathLike[str],
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
    progress_renderer: Any | None = None,
    overrides: TrainingRunConfigOverrides | None = None,
    cli_cwd: str | os.PathLike[str] | None = None,
) -> (
    ResolvedTrainingRun
    | ResolvedScratchTrainingRun
    | ScratchTrainingPreparation
    | dict[str, Any]
):
    """Run training while keeping progress presentation non-semantic."""

    try:
        result = _run_training_impl(
            path,
            dry_run=dry_run,
            progress=progress,
            progress_renderer=progress_renderer,
            overrides=overrides,
            cli_cwd=cli_cwd,
        )
    except BaseException as error:
        if progress_renderer is not None:
            try:
                _render_terminal_error(progress_renderer, error)
            except Exception as presentation_error:
                # Presentation is never allowed to replace the primary
                # training/configuration/interrupt failure.
                setattr(error, "training_progress_error", presentation_error)
        raise
    if (
        progress_renderer is not None
        and not dry_run
        and isinstance(result, Mapping)
    ):
        try:
            _render_terminal_result(progress_renderer, result)
        except Exception:
            # A malformed or unavailable console must not retroactively turn a
            # successfully checkpointed run into a failure.
            pass
    return result


def render_training_json(report: Mapping[str, Any]) -> str:
    """Render a completed/failed run result as deterministic strict JSON."""

    if not isinstance(report, Mapping):
        raise TypeError("training report must be a mapping")
    return canonical_runtime_json(report)


def render_training_human(report: Mapping[str, Any]) -> str:
    """Render a concise deterministic terminal training summary."""

    if not isinstance(report, Mapping):
        raise TypeError("training report must be a mapping")
    terminal_status = report.get("terminal_status")
    if terminal_status is not None:
        if not isinstance(terminal_status, Mapping):
            raise TypeError("terminal_status must be a mapping")
        report = terminal_status
    if report.get("status") not in ("completed", "early_stopped"):
        raise ValueError("human terminal summary requires a completed report")
    status = str(report["status"])
    fit = report["fit_result"]
    baseline = report["baseline"]
    return "\n".join(
        (
            "Reference-site MLIP training run",
            f"Status: {status}",
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
    result: (
        ResolvedTrainingRun
        | ResolvedScratchTrainingRun
        | ScratchTrainingPreparation
        | Mapping[str, Any]
    ),
) -> str:
    if isinstance(
        result,
        (ResolvedTrainingRun, ResolvedScratchTrainingRun, ScratchTrainingPreparation),
    ):
        return render_train_config_json(result)
    return render_training_json(result)


def render_train_result_human(
    result: (
        ResolvedTrainingRun
        | ResolvedScratchTrainingRun
        | ScratchTrainingPreparation
        | Mapping[str, Any]
    ),
) -> str:
    if isinstance(
        result,
        (ResolvedTrainingRun, ResolvedScratchTrainingRun, ScratchTrainingPreparation),
    ):
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
