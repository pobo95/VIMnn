"""Export a portable prediction bundle from one managed training checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterator

import numpy as np
import torch

from refsite_mlip.config import (
    ScratchModelSourceConfig,
    TrainingRunConfig,
    TrainingRunConfigError,
)
from refsite_mlip.config.radii import (
    RadiusConfigError,
    validate_radius_artifact_compatibility,
    validate_radius_model_compatibility,
)
from refsite_mlip.config.training_run import (
    TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2,
    _phase_specification_fingerprint,
)
from refsite_mlip.models import (
    ModelBundleError,
    PotentialConfig,
    ReferenceSiteModelBundle,
    capture_reference_site_model_bundle,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
    reference_site_model_architecture_fingerprint,
    save_reference_site_model_bundle,
)
from refsite_mlip.training import (
    CheckpointManager,
    CheckpointManagerConfig,
    FitConfig,
    RunDirectoryError,
    TrainingCheckpoint,
    TrainingRunDirectory,
    canonical_runtime_json,
    load_runtime_json,
    validate_checkpoint_history,
)
from refsite_mlip.training.checkpoint import (
    _plain as _checkpoint_plain,
    _unit_conventions,
)
from refsite_mlip.training.scratch_preparation import (
    SCRATCH_DATA_MANIFEST_CONVENTION_VERSION,
    _fingerprint as _scratch_metadata_fingerprint,
)
from refsite_mlip.transport import TRAIN_FIXED

from .errors import CLIError


EXPORT_BUNDLE_RESULT_SCHEMA_VERSION = "refsite_export_bundle_result_v1"
EXPORT_BUNDLE_PROVENANCE_SCHEMA_VERSION = "refsite_checkpoint_export_v2"
_EXPORT_BUNDLE_PROVENANCE_SOURCE = "managed_epoch_checkpoint"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCES = frozenset({"best", "latest"})
_METRICS_STATUS_FIELDS = {
    "metrics_journal",
    "metrics_event_count",
    "metrics_last_epoch",
    "metrics_semantic_sha256",
}


@dataclass(frozen=True)
class ExportBundleConfig:
    """Filesystem and selection controls for one immutable bundle export."""

    run_directory: str | os.PathLike[str]
    source: str
    output_path: str | os.PathLike[str]
    initial_bundle_path: str | os.PathLike[str] | None = None
    dry_run: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError("source must be 'best' or 'latest'")
        for name in ("dry_run", "overwrite"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in ("run_directory", "output_path"):
            value = os.fspath(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must be a nonempty path")
        if self.initial_bundle_path is not None and not os.fspath(
            self.initial_bundle_path
        ):
            raise ValueError("initial_bundle_path must be a nonempty path or None")


@dataclass(frozen=True)
class _StoredRun:
    directory: TrainingRunDirectory
    config: TrainingRunConfig
    preflight: Mapping[str, Any]
    status: Mapping[str, Any]
    config_fingerprint: str
    bundle_fingerprint: str
    train_semantic_digest: str
    validation_semantic_digest: str
    stored_initial_bundle_path: Path
    train_frame_count: int
    validation_frame_count: int
    train_batch_count: int
    validation_batch_count: int


@dataclass(frozen=True)
class _PreparedExport:
    request: ExportBundleConfig
    stored: _StoredRun
    manager: CheckpointManager
    checkpoint: TrainingCheckpoint
    checkpoint_path: Path
    records: tuple[Any, ...]
    initial_bundle_path: Path
    initial_bundle: ReferenceSiteModelBundle
    exported_bundle: ReferenceSiteModelBundle
    output_path: Path
    source_metric: float
    report: Mapping[str, Any]


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_runtime_json({"value": left}) == canonical_runtime_json(
            {"value": right}
        )
    except (TypeError, ValueError):
        return False


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        return (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes, bytearray))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _require_mapping(value: Any, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CLIError(
            "INVALID_RUN_METADATA",
            f"{field} must be a JSON object",
            stage="export.metadata.validate",
            path=path,
            config_field=field,
        )
    return value


def _require_equal(
    field: str,
    actual: Any,
    expected: Any,
    *,
    path: Path,
    reason: str = "RUN_METADATA_MISMATCH",
) -> None:
    if not _canonical_equal(actual, expected):
        raise CLIError(
            reason,
            f"stored {field} does not match the immutable training-run contract",
            stage="export.metadata.validate",
            path=path,
            config_field=field,
        )


def _sha256(value: Any, *, field: str, path: Path) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CLIError(
            "INVALID_STORED_FINGERPRINT",
            f"{field} must be a lowercase SHA-256 string",
            stage="export.metadata.validate",
            path=path,
            config_field=field,
        )
    return value


def _absolute_stored_path(value: Any, *, field: str, path: Path) -> Path:
    if type(value) is not str or not value:
        raise CLIError(
            "INVALID_RESOLVED_PATH",
            "stored runtime path must be a nonempty absolute path",
            stage="export.metadata.paths",
            path=path,
            config_field=field,
        )
    result = Path(value)
    if not result.is_absolute():
        raise CLIError(
            "INVALID_RESOLVED_PATH",
            "stored runtime paths are absolute and are never reinterpreted",
            stage="export.metadata.paths",
            path=result,
            config_field=field,
        )
    return result


def _load_stored_config(directory: TrainingRunDirectory) -> TrainingRunConfig:
    payload = load_runtime_json(
        directory.resolved_config_path,
        stage="export.metadata.resolved_config",
    )
    try:
        return TrainingRunConfig.from_dict(payload)
    except TrainingRunConfigError as error:
        raise CLIError(
            error.reason_code,
            "stored resolved_config.json is invalid",
            stage=error.stage or "export.metadata.resolved_config",
            path=directory.resolved_config_path,
            config_field=error.field,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "INVALID_RESOLVED_CONFIG",
            "stored resolved_config.json is invalid",
            stage="export.metadata.resolved_config",
            path=directory.resolved_config_path,
            original_error=error,
        ) from error


def _split_metadata(
    preflight: Mapping[str, Any],
    *,
    split: str,
    path: Path,
) -> tuple[str, int, int]:
    data = _require_mapping(preflight.get("data"), field="data", path=path)
    value = _require_mapping(data.get(split), field=f"data.{split}", path=path)
    required = {
        "semantic_digest",
        "frame_count",
        "batch_count",
        "template_frame_counts",
        "composition_statistics",
        "label_statistics",
    }
    if set(value) != required:
        raise CLIError(
            "INVALID_PREFLIGHT_METADATA",
            f"stored data.{split} fields are invalid",
            stage="export.metadata.preflight",
            path=path,
            config_field=f"data.{split}",
            split=split,
        )
    digest = _sha256(
        value["semantic_digest"], field=f"data.{split}.semantic_digest", path=path
    )
    for name in ("frame_count", "batch_count"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise CLIError(
                "INVALID_PREFLIGHT_METADATA",
                f"data.{split}.{name} must be a positive integer",
                stage="export.metadata.preflight",
                path=path,
                config_field=f"data.{split}.{name}",
                split=split,
            )
    _require_mapping(
        value["template_frame_counts"],
        field=f"data.{split}.template_frame_counts",
        path=path,
    )
    return digest, int(value["frame_count"]), int(value["batch_count"])


def _validate_stored_run(directory: TrainingRunDirectory) -> _StoredRun:
    config = _load_stored_config(directory)
    preflight = load_runtime_json(
        directory.preflight_path,
        stage="export.metadata.preflight",
    )
    status = load_runtime_json(
        directory.status_path,
        stage="export.metadata.status",
    )
    required = {
        "status",
        "training_executed",
        "schema_version",
        "config_fingerprint",
        "bundle_fingerprint",
        "data",
        "runtime",
        "radii",
        "species_vocabulary",
        "template_fingerprints",
        "baseline_preflight",
        "expected_paths",
        "training_configuration",
    }
    if set(preflight) != required:
        raise CLIError(
            "INVALID_PREFLIGHT_METADATA",
            "preflight.json has missing or unknown top-level fields",
            stage="export.metadata.preflight",
            path=directory.preflight_path,
        )
    _require_equal(
        "preflight.status",
        preflight["status"],
        "preflight_ready",
        path=directory.preflight_path,
    )
    _require_equal(
        "preflight.training_executed",
        preflight["training_executed"],
        False,
        path=directory.preflight_path,
    )
    _require_equal(
        "preflight.schema_version",
        preflight["schema_version"],
        config.schema_version,
        path=directory.preflight_path,
    )
    config_fingerprint = _sha256(
        preflight["config_fingerprint"],
        field="config_fingerprint",
        path=directory.preflight_path,
    )
    _require_equal(
        "preflight.config_fingerprint",
        config_fingerprint,
        config.config_fingerprint,
        path=directory.preflight_path,
        reason="CONFIG_FINGERPRINT_MISMATCH",
    )
    bundle_fingerprint = _sha256(
        preflight["bundle_fingerprint"],
        field="bundle_fingerprint",
        path=directory.preflight_path,
    )

    runtime = _require_mapping(
        preflight["runtime"], field="runtime", path=directory.preflight_path
    )
    if set(runtime) != {"device", "dtype", "seed", "configured_paths", "paths"}:
        raise CLIError(
            "INVALID_PREFLIGHT_METADATA",
            "preflight runtime metadata fields are invalid",
            stage="export.metadata.preflight",
            path=directory.preflight_path,
            config_field="runtime",
        )
    for field, expected in (
        ("runtime.device", config.runtime.device),
        ("runtime.dtype", config.runtime.dtype),
        ("runtime.seed", config.runtime.seed),
    ):
        _require_equal(
            field,
            runtime[field.removeprefix("runtime.")],
            expected,
            path=directory.preflight_path,
        )
    configured = {
        "initial_bundle": config.initial_bundle,
        "output_directory": config.output_directory,
        "train_inputs": [source.path for source in config.data.train],
        "validation_inputs": [source.path for source in config.data.validation],
        "path_kind": "original_config_expression_in_semantic_fingerprint",
    }
    _require_equal(
        "runtime.configured_paths",
        runtime["configured_paths"],
        configured,
        path=directory.preflight_path,
    )
    paths = _require_mapping(
        runtime["paths"], field="runtime.paths", path=directory.preflight_path
    )
    path_keys = {
        "config",
        "initial_bundle",
        "output_directory",
        "train_inputs",
        "validation_inputs",
        "path_kind",
    }
    if set(paths) != path_keys:
        raise CLIError(
            "INVALID_PREFLIGHT_METADATA",
            "stored runtime path manifest fields are invalid",
            stage="export.metadata.preflight",
            path=directory.preflight_path,
            config_field="runtime.paths",
        )
    _require_equal(
        "runtime.paths.path_kind",
        paths["path_kind"],
        "runtime_location_not_semantic_fingerprint",
        path=directory.preflight_path,
    )
    output = _absolute_stored_path(
        paths["output_directory"],
        field="runtime.paths.output_directory",
        path=directory.preflight_path,
    )
    try:
        output = output.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CLIError(
            "RUN_DIRECTORY_PATH_MISMATCH",
            "stored training output directory is unavailable",
            stage="export.metadata.paths",
            path=output,
            original_error=error,
        ) from error
    if output != directory.root:
        raise CLIError(
            "RUN_DIRECTORY_PATH_MISMATCH",
            "requested run directory differs from its stored runtime location",
            stage="export.metadata.paths",
            path=directory.root,
        )
    for split, entries, expected_count in (
        ("train", paths["train_inputs"], len(config.data.train)),
        ("validation", paths["validation_inputs"], len(config.data.validation)),
    ):
        if not isinstance(entries, list) or len(entries) != expected_count:
            raise CLIError(
                "DATA_PATH_MANIFEST_MISMATCH",
                f"stored {split} path count differs from the config",
                stage="export.metadata.paths",
                path=directory.preflight_path,
                split=split,
            )
        for index, value in enumerate(entries):
            _absolute_stored_path(
                value,
                field=f"runtime.paths.{split}_inputs[{index}]",
                path=directory.preflight_path,
            )
    stored_initial = _absolute_stored_path(
        paths["initial_bundle"],
        field="runtime.paths.initial_bundle",
        path=directory.preflight_path,
    )
    _absolute_stored_path(
        paths["config"],
        field="runtime.paths.config",
        path=directory.preflight_path,
    )
    expected_paths = {
        "output_directory": str(directory.root),
        "resolved_config": str(directory.resolved_config_path),
        "preflight": str(directory.preflight_path),
        "run_status": str(directory.status_path),
        "latest_checkpoint": str(directory.checkpoints / "latest.pt"),
        "best_checkpoint": str(directory.checkpoints / "best.pt"),
        "epoch_checkpoint_pattern": str(directory.checkpoints / "epoch_XXXXXX.pt"),
    }
    _require_equal(
        "expected_paths",
        preflight["expected_paths"],
        expected_paths,
        path=directory.preflight_path,
    )
    expected_training = {
        "loss": config.loss.to_dict(),
        "baseline": None if config.baseline is None else config.baseline.to_dict(),
        "optimizer": config.optimizer.to_dict(),
        "train_step": config.train_step.to_dict(),
        "validation_step": config.validation_step.to_dict(),
        "scheduler": config.scheduler.to_dict(),
        "selection": config.selection.to_dict(),
        "fit": config.fit.to_dict(),
        "checkpointed_fit": config.checkpointed_fit.to_dict(),
        "batch_size": config.data.batch_size,
        "shuffle": False,
    }
    if config.schema_version == TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2:
        expected_training["validation_batch_size"] = (
            config.data.effective_validation_batch_size
        )
    _require_equal(
        "training_configuration",
        preflight["training_configuration"],
        expected_training,
        path=directory.preflight_path,
    )
    expected_radii = {
        "user": {"r_ot": config.radii.r_ot, "r_mp": config.radii.r_mp},
        "advanced": {
            "ot_switch_width": config.radii.ot_switch_width,
            "ot_skin": config.radii.ot_skin,
            "mp_skin": config.radii.mp_skin,
        },
        "derived": config.radii.derived.to_dict(),
        "diagnostics": config.radii.derived.to_diagnostics_dict(),
    }
    _require_equal(
        "radii", preflight["radii"], expected_radii, path=directory.preflight_path
    )
    train_digest, train_frames, train_batches = _split_metadata(
        preflight, split="train", path=directory.preflight_path
    )
    validation_digest, validation_frames, validation_batches = _split_metadata(
        preflight, split="validation", path=directory.preflight_path
    )

    required_status = {
        "schema_version",
        "status",
        "training_executed",
        "config_fingerprint",
        "bundle_fingerprint",
        "train_semantic_digest",
        "validation_semantic_digest",
        "seed",
        "runtime",
        "completed_epochs",
        "global_step",
        "latest_checkpoint",
        "best_checkpoint",
        "recoverable_checkpoint",
        "terminal_selection_state",
        "fit_result",
        "baseline",
        "failure_phase",
        "error",
        "rollback_performed",
    }
    scratch_status_fields = {
        "data_manifest_fingerprint",
        "first_optimizer_update_executed",
        "initial_bundle_fingerprint",
        "initialization_seed",
        "preparation_fingerprint",
        "recoverable_initial_bundle",
        "recovery",
        "source_kind",
        "template_fingerprints",
    }
    allowed_status = required_status | {
        "result_schema_version",
        "epoch_index",
        "batch_index",
        "sample_id",
        "template_id",
        "operation",
        "operation_phase",
        "path_kind",
        "resume_source",
        "resume_from_epoch",
        "requested_max_epochs",
        "resumed_epochs_completed",
        "recoverable_global_step",
        "new_epoch_checkpoints",
        "restored_rng_domains",
        "exact_resume",
        "rollback_succeeded",
        "partial_update_retained",
    } | scratch_status_fields | _METRICS_STATUS_FIELDS
    if not required_status.issubset(status) or not set(status).issubset(allowed_status):
        raise CLIError(
            "INVALID_RUN_STATUS",
            "run_status.json has missing or unknown fields",
            stage="export.metadata.status",
            path=directory.status_path,
        )
    if status.get("schema_version") != "refsite_training_run_status_v1":
        raise CLIError(
            "INVALID_RUN_STATUS",
            "run_status.json schema is unsupported",
            stage="export.metadata.status",
            path=directory.status_path,
        )
    status_name = status.get("status")
    if status_name == "running":
        raise CLIError(
            "ACTIVE_RUN_REJECTED",
            "a running training or resume operation cannot be exported",
            stage="export.active_run",
            path=directory.status_path,
        )
    if status_name not in {
        "completed",
        "early_stopped",
        "failed",
        "interrupted",
    }:
        raise CLIError(
            "INVALID_RUN_STATUS",
            "run status must be completed, early_stopped, failed, or "
            "interrupted for export",
            stage="export.metadata.status",
            path=directory.status_path,
        )
    if type(status.get("training_executed")) is not bool:
        raise CLIError(
            "INVALID_RUN_STATUS",
            "run_status.training_executed must be a bool",
            stage="export.metadata.status",
            path=directory.status_path,
            config_field="training_executed",
        )
    present_metrics_fields = set(status).intersection(_METRICS_STATUS_FIELDS)
    if present_metrics_fields and present_metrics_fields != _METRICS_STATUS_FIELDS:
        raise CLIError(
            "INVALID_RUN_STATUS",
            "run_status metrics journal metadata must be present as one complete group",
            stage="export.metadata.status",
            path=directory.status_path,
        )
    if present_metrics_fields:
        if status["metrics_journal"] != "metrics.jsonl":
            raise CLIError(
                "INVALID_RUN_STATUS",
                "run_status.metrics_journal must be the managed relative path 'metrics.jsonl'",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field="metrics_journal",
            )
        event_count = status["metrics_event_count"]
        if type(event_count) is not int or event_count < 0:
            raise CLIError(
                "INVALID_RUN_STATUS",
                "run_status.metrics_event_count must be a nonnegative integer",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field="metrics_event_count",
            )
        last_epoch = status["metrics_last_epoch"]
        if event_count == 0:
            if last_epoch is not None:
                raise CLIError(
                    "INVALID_RUN_STATUS",
                    "an empty metrics journal requires metrics_last_epoch=null",
                    stage="export.metadata.status",
                    path=directory.status_path,
                    config_field="metrics_last_epoch",
                )
        elif type(last_epoch) is not int or last_epoch != event_count - 1:
            raise CLIError(
                "INVALID_RUN_STATUS",
                "metrics_last_epoch must identify the contiguous terminal journal epoch",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field="metrics_last_epoch",
            )
        if (
            type(status["metrics_semantic_sha256"]) is not str
            or _SHA256.fullmatch(status["metrics_semantic_sha256"]) is None
        ):
            raise CLIError(
                "INVALID_RUN_STATUS",
                "run_status.metrics_semantic_sha256 must be a lowercase SHA-256",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field="metrics_semantic_sha256",
            )
    for field in ("completed_epochs", "global_step"):
        value = status.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CLIError(
                "INVALID_RUN_STATUS",
                f"run_status.{field} must be a nonnegative integer",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field=field,
            )
    for field, expected in (
        ("config_fingerprint", config_fingerprint),
        ("bundle_fingerprint", bundle_fingerprint),
        ("train_semantic_digest", train_digest),
        ("validation_semantic_digest", validation_digest),
        ("seed", config.runtime.seed),
    ):
        _require_equal(
            f"run_status.{field}",
            status.get(field),
            expected,
            path=directory.status_path,
            reason="RUN_STATUS_IDENTITY_MISMATCH",
        )
    scratch_config = isinstance(config.model_source, ScratchModelSourceConfig)
    if scratch_config and status.get("source_kind") is None:
        raise CLIError(
            "INVALID_RUN_STATUS",
            "scratch run_status.json is missing source_kind provenance",
            stage="export.metadata.status",
            path=directory.status_path,
            config_field="source_kind",
        )
    if not scratch_config and scratch_status_fields.intersection(status):
        raise CLIError(
            "INVALID_RUN_STATUS",
            "bundle-source run_status.json contains scratch-only provenance fields",
            stage="export.metadata.status",
            path=directory.status_path,
        )
    if status.get("source_kind") is not None:
        if status.get("source_kind") != "scratch" or not scratch_config:
            raise CLIError(
                "INVALID_RUN_STATUS",
                "run_status source_kind is inconsistent with the training config",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field="source_kind",
            )
        if not scratch_status_fields.issubset(status):
            raise CLIError(
                "INVALID_RUN_STATUS",
                "scratch run_status.json is missing provenance fields",
                stage="export.metadata.status",
                path=directory.status_path,
            )
        for field in ("data_manifest_fingerprint", "preparation_fingerprint"):
            _sha256(
                status[field],
                field=f"run_status.{field}",
                path=directory.status_path,
            )
        data_manifest = load_runtime_json(
            directory.data_manifest_path,
            stage="export.metadata.data_manifest",
        )
        manifest_fields = {
            "convention_version",
            "fingerprint",
            "observed_species",
            "train",
            "train_semantic_digest",
            "validation",
            "validation_semantic_digest",
        }
        if set(data_manifest) != manifest_fields:
            raise CLIError(
                "INVALID_DATA_MANIFEST",
                "data_manifest.json has missing or unknown top-level fields",
                stage="export.metadata.data_manifest",
                path=directory.data_manifest_path,
            )
        manifest_payload = dict(data_manifest)
        manifest_fingerprint = _sha256(
            manifest_payload.pop("fingerprint", None),
            field="data_manifest.fingerprint",
            path=directory.data_manifest_path,
        )
        _require_equal(
            "data_manifest.fingerprint",
            manifest_fingerprint,
            _scratch_metadata_fingerprint(
                SCRATCH_DATA_MANIFEST_CONVENTION_VERSION,
                manifest_payload,
            ),
            path=directory.data_manifest_path,
            reason="DATA_MANIFEST_FINGERPRINT_MISMATCH",
        )
        for field, expected in (
            (
                "convention_version",
                SCRATCH_DATA_MANIFEST_CONVENTION_VERSION,
            ),
            ("train_semantic_digest", train_digest),
            ("validation_semantic_digest", validation_digest),
        ):
            _require_equal(
                f"data_manifest.{field}",
                data_manifest[field],
                expected,
                path=directory.data_manifest_path,
                reason="DATA_MANIFEST_IDENTITY_MISMATCH",
            )
        _require_equal(
            "run_status.data_manifest_fingerprint",
            status["data_manifest_fingerprint"],
            manifest_fingerprint,
            path=directory.status_path,
            reason="RUN_STATUS_IDENTITY_MISMATCH",
        )
        for split, expected_frames, expected_batches in (
            ("train", train_frames, train_batches),
            (
                "validation",
                validation_frames,
                validation_batches,
            ),
        ):
            manifest_split = _require_mapping(
                data_manifest[split],
                field=f"data_manifest.{split}",
                path=directory.data_manifest_path,
            )
            _require_equal(
                f"data_manifest.{split}.frame_count",
                manifest_split.get("frame_count"),
                expected_frames,
                path=directory.data_manifest_path,
                reason="DATA_MANIFEST_IDENTITY_MISMATCH",
            )
            _require_equal(
                f"data_manifest.{split}.batch_count",
                manifest_split.get("batch_count"),
                expected_batches,
                path=directory.data_manifest_path,
                reason="DATA_MANIFEST_IDENTITY_MISMATCH",
            )
        _require_equal(
            "run_status.initial_bundle_fingerprint",
            status["initial_bundle_fingerprint"],
            bundle_fingerprint,
            path=directory.status_path,
            reason="RUN_STATUS_IDENTITY_MISMATCH",
        )
        _require_equal(
            "run_status.initialization_seed",
            status["initialization_seed"],
            config.model_source.initialization_seed,
            path=directory.status_path,
            reason="RUN_STATUS_IDENTITY_MISMATCH",
        )
        _require_equal(
            "run_status.template_fingerprints",
            status["template_fingerprints"],
            preflight["template_fingerprints"],
            path=directory.status_path,
            reason="RUN_STATUS_IDENTITY_MISMATCH",
        )
        if type(status["first_optimizer_update_executed"]) is not bool:
            raise CLIError(
                "INVALID_RUN_STATUS",
                "run_status.first_optimizer_update_executed must be a bool",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field="first_optimizer_update_executed",
            )
        _require_equal(
            "run_status.first_optimizer_update_executed",
            status["first_optimizer_update_executed"],
            status["global_step"] > 0,
            path=directory.status_path,
            reason="RUN_STATUS_IDENTITY_MISMATCH",
        )
        expected_initial = str(stored_initial)
        initial_recovery = status["recoverable_initial_bundle"]
        if initial_recovery != expected_initial:
            raise CLIError(
                "RUN_STATUS_CHECKPOINT_PATH_MISMATCH",
                "run_status.recoverable_initial_bundle is not the managed initial bundle",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field="recoverable_initial_bundle",
            )
        recovery = _require_mapping(
            status["recovery"],
            field="run_status.recovery",
            path=directory.status_path,
        )
        if set(recovery) != {"kind", "path"}:
            raise CLIError(
                "INVALID_RUN_STATUS",
                "run_status.recovery must contain exactly kind and path",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field="recovery",
            )
        recovery_path = status.get("recoverable_checkpoint")
        recovery_kind = "latest_checkpoint"
        if recovery_path is None:
            recovery_path = initial_recovery
            recovery_kind = "initial_bundle" if recovery_path is not None else None
        _require_equal(
            "run_status.recovery",
            recovery,
            {"kind": recovery_kind, "path": recovery_path},
            path=directory.status_path,
            reason="RUN_STATUS_IDENTITY_MISMATCH",
        )
    status_runtime = _require_mapping(
        status.get("runtime"), field="run_status.runtime", path=directory.status_path
    )
    _require_equal(
        "run_status.runtime",
        status_runtime,
        {
            "device": config.runtime.device,
            "dtype": config.runtime.dtype,
            "solver_path": TRAIN_FIXED,
        },
        path=directory.status_path,
        reason="RUN_STATUS_IDENTITY_MISMATCH",
    )
    for field in ("completed_epochs", "global_step"):
        value = status.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CLIError(
                "INVALID_RUN_STATUS",
                f"run_status.{field} must be a nonnegative integer",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field=field,
            )
    expected_checkpoint_paths = {
        "latest_checkpoint": str(directory.checkpoints / "latest.pt"),
        "best_checkpoint": str(directory.checkpoints / "best.pt"),
        "recoverable_checkpoint": str(directory.checkpoints / "latest.pt"),
    }
    for field, expected in expected_checkpoint_paths.items():
        value = status.get(field)
        if value is not None and value != expected:
            raise CLIError(
                "RUN_STATUS_CHECKPOINT_PATH_MISMATCH",
                f"run_status.{field} is not the managed checkpoint path",
                stage="export.metadata.status",
                path=directory.status_path,
                config_field=field,
            )
    return _StoredRun(
        directory=directory,
        config=config,
        preflight=copy.deepcopy(dict(preflight)),
        status=copy.deepcopy(dict(status)),
        config_fingerprint=config_fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        train_semantic_digest=train_digest,
        validation_semantic_digest=validation_digest,
        stored_initial_bundle_path=stored_initial,
        train_frame_count=train_frames,
        validation_frame_count=validation_frames,
        train_batch_count=train_batches,
        validation_batch_count=validation_batches,
    )


def _bundle_template_metadata(bundle: ReferenceSiteModelBundle) -> dict[str, Any]:
    return {
        binding.template_id: {
            "structural_artifact_fingerprint": (
                binding.structural_artifact.structural_fingerprint
            ),
            "full_template_fingerprint": binding.full_template_fingerprint,
            "phase_specification_fingerprint": _phase_specification_fingerprint(
                binding.phase_specification
            ),
            "binding_fingerprint": binding.binding_fingerprint,
            "evaluation_policy_fingerprint": (
                None
                if binding.evaluation_policy is None
                else binding.evaluation_policy.content_fingerprint
            ),
        }
        for binding in sorted(bundle.template_bindings, key=lambda item: item.template_id)
    }


def _load_initial_bundle(
    stored: _StoredRun,
    override: str | os.PathLike[str] | None,
    *,
    source: str,
) -> tuple[Path, ReferenceSiteModelBundle]:
    path = stored.stored_initial_bundle_path if override is None else Path(override)
    if path.is_symlink():
        raise CLIError(
            "INITIAL_BUNDLE_SYMLINK_REJECTED",
            "initial bundle source must not be a symbolic link",
            stage="export.initial_bundle.path",
            bundle_path=path,
            source_kind=source,
        )
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        message = (
            "stored initial bundle moved or is missing; provide --initial-bundle "
            "with an exact-fingerprint replacement"
            if override is None
            else "replacement initial bundle does not exist"
        )
        raise CLIError(
            "INITIAL_BUNDLE_NOT_FOUND",
            message,
            stage="export.initial_bundle.path",
            bundle_path=path,
            source_kind=source,
            bundle_fingerprint=stored.bundle_fingerprint,
            original_error=error,
        ) from error
    if not resolved.is_file():
        raise CLIError(
            "INVALID_INITIAL_BUNDLE_SOURCE",
            "initial bundle source must be a regular file",
            stage="export.initial_bundle.path",
            bundle_path=resolved,
            source_kind=source,
        )
    try:
        bundle = load_reference_site_model_bundle(resolved, map_location="cpu")
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "initial bundle failed weights-only validation",
            stage=error.validation_stage or "export.initial_bundle.load",
            bundle_path=resolved,
            template_id=error.template_id,
            source_kind=source,
            bundle_fingerprint=stored.bundle_fingerprint,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "INITIAL_BUNDLE_LOAD_FAILED",
            "initial bundle failed weights-only loading",
            stage="export.initial_bundle.load",
            bundle_path=resolved,
            source_kind=source,
            bundle_fingerprint=stored.bundle_fingerprint,
            original_error=error,
        ) from error
    if bundle.bundle_fingerprint != stored.bundle_fingerprint:
        raise CLIError(
            "INITIAL_BUNDLE_FINGERPRINT_MISMATCH",
            "replacement initial bundle does not exactly match the stored SHA-256",
            stage="export.initial_bundle.compatibility",
            bundle_path=resolved,
            source_kind=source,
            bundle_fingerprint=stored.bundle_fingerprint,
            underlying_reason_code="BUNDLE_FINGERPRINT_MISMATCH",
        )
    return resolved, bundle


def _checkpoint_directory(stored: _StoredRun) -> CheckpointManager:
    path = stored.directory.checkpoints
    if path.is_symlink():
        raise CLIError(
            "CHECKPOINT_DIRECTORY_SYMLINK_REJECTED",
            "managed checkpoints directory must not be a symbolic link",
            stage="export.checkpoint.path",
            path=path,
        )
    if not path.exists() or not path.is_dir():
        raise CLIError(
            "CHECKPOINT_DIRECTORY_INVALID",
            "managed checkpoints directory is missing or not a directory",
            stage="export.checkpoint.path",
            path=path,
        )
    if path.resolve(strict=True).parent != stored.directory.root:
        raise CLIError(
            "CHECKPOINT_DIRECTORY_ESCAPE",
            "managed checkpoints directory resolves outside the run directory",
            stage="export.checkpoint.path",
            path=path,
        )
    return CheckpointManager(CheckpointManagerConfig(directory=str(path)))


def _load_checkpoint(
    manager: CheckpointManager,
    source: str,
) -> tuple[Path, TrainingCheckpoint, tuple[Any, ...]]:
    path = manager.root / f"{source}.pt"
    try:
        checkpoint = manager.load_best() if source == "best" else manager.load_latest()
    except Exception as error:
        raise CLIError(
            "SOURCE_CHECKPOINT_LOAD_FAILED",
            f"managed checkpoints/{source}.pt failed weights-only loading",
            stage="export.checkpoint.load",
            path=path,
            source_kind=source,
            checkpoint_stage="weights_only_load",
            underlying_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error
    try:
        records = validate_checkpoint_history(
            checkpoint, allow_stopped_early=True
        )
    except Exception as error:
        raise CLIError(
            "CHECKPOINT_HISTORY_INVALID",
            "selected checkpoint history/progress is not a valid epoch-boundary history",
            stage="export.checkpoint.history",
            path=path,
            source_kind=source,
            checkpoint_stage="history_progress",
            epoch_index=checkpoint.progress.last_completed_epoch,
            global_step=checkpoint.progress.global_step,
            original_error=error,
        ) from error
    if source == "best":
        last = records[-1]
        if (
            not last.decision.is_best
            or checkpoint.selection_state.best_epoch
            != checkpoint.progress.last_completed_epoch
            or checkpoint.selection_state.best_global_step
            != checkpoint.progress.global_step
        ):
            raise CLIError(
                "BEST_CHECKPOINT_IDENTITY_MISMATCH",
                "best.pt does not represent the best event at its terminal epoch",
                stage="export.checkpoint.history",
                path=path,
                source_kind=source,
                checkpoint_stage="best_identity",
                epoch_index=checkpoint.progress.last_completed_epoch,
                global_step=checkpoint.progress.global_step,
            )
    return path, checkpoint, records


def _manifest_sample_ids(manifest: Any, *, split: str) -> tuple[str, ...]:
    values = tuple(
        sample_id
        for batch_ids in manifest.ordered_batch_sample_ids
        for sample_id in batch_ids
    )
    if len(values) != manifest.number_of_structures or len(set(values)) != len(values):
        raise CLIError(
            "CHECKPOINT_DATA_MANIFEST_INVALID",
            "checkpoint data manifest has duplicate or inconsistent sample IDs",
            stage="export.checkpoint.compatibility",
            split=split,
        )
    return values


def _validate_checkpoint_compatibility(
    stored: _StoredRun,
    checkpoint: TrainingCheckpoint,
    bundle: ReferenceSiteModelBundle,
    records: tuple[Any, ...],
    *,
    source: str,
    checkpoint_path: Path,
) -> float:
    config = stored.config
    status_completed = int(stored.status["completed_epochs"])
    if (
        source == "latest"
        and status_completed != checkpoint.progress.completed_epochs
    ) or (
        source == "best"
        and status_completed < checkpoint.progress.completed_epochs
    ):
        raise CLIError(
            "RUN_STATUS_CHECKPOINT_PROGRESS_MISMATCH",
            "selected checkpoint progress is inconsistent with run_status.json",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="status_progress",
            epoch_index=checkpoint.progress.last_completed_epoch,
            global_step=checkpoint.progress.global_step,
        )
    terminal_selection = stored.status.get("terminal_selection_state")
    if stored.status.get("status") == "completed" and terminal_selection is None:
        raise CLIError(
            "INVALID_RUN_STATUS",
            "completed run status requires terminal model-selection state",
            stage="export.metadata.status",
            path=stored.directory.status_path,
            source_kind=source,
            config_field="terminal_selection_state",
        )
    if terminal_selection is not None and not isinstance(terminal_selection, Mapping):
        raise CLIError(
            "INVALID_RUN_STATUS",
            "terminal selection state must be a JSON object or null",
            stage="export.metadata.status",
            path=stored.directory.status_path,
            source_kind=source,
            config_field="terminal_selection_state",
        )
    if terminal_selection is not None:
        if source == "latest":
            selection_matches = _canonical_equal(
                terminal_selection, checkpoint.selection_state.to_dict()
            )
        else:
            selection_matches = all(
                _canonical_equal(terminal_selection.get(field), expected)
                for field, expected in (
                    ("best_epoch", checkpoint.progress.last_completed_epoch),
                    ("best_global_step", checkpoint.progress.global_step),
                    ("best_metric", checkpoint.selection_state.best_metric),
                )
            )
        if not selection_matches:
            raise CLIError(
                "RUN_STATUS_CHECKPOINT_SELECTION_MISMATCH",
                "selected checkpoint is inconsistent with terminal model selection",
                stage="export.checkpoint.compatibility",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="status_selection",
                epoch_index=checkpoint.progress.last_completed_epoch,
                global_step=checkpoint.progress.global_step,
            )
    try:
        potential_config = PotentialConfig.from_dict(bundle.model_config)
    except Exception as error:
        raise CLIError(
            "MODEL_CONFIG_INVALID",
            "validated initial bundle has an invalid model config",
            stage="export.bundle.compatibility",
            bundle_fingerprint=stored.bundle_fingerprint,
            source_kind=source,
            original_error=error,
        ) from error
    template_metadata = _bundle_template_metadata(bundle)
    stored_templates = stored.preflight["template_fingerprints"]
    if not _canonical_equal(stored_templates, template_metadata):
        candidate_ids = sorted(
            set(stored_templates if isinstance(stored_templates, Mapping) else ())
            | set(template_metadata)
        )
        template_id = next(
            (
                value
                for value in candidate_ids
                if not isinstance(stored_templates, Mapping)
                or not _canonical_equal(
                    stored_templates.get(value), template_metadata.get(value)
                )
            ),
            None,
        )
        expected_fingerprint = (
            None
            if template_id is None or template_id not in template_metadata
            else template_metadata[template_id]["full_template_fingerprint"]
        )
        raise CLIError(
            "TEMPLATE_FINGERPRINT_MISMATCH",
            "stored template fingerprints differ from the initial bundle",
            stage="export.bundle.compatibility",
            path=stored.directory.preflight_path,
            source_kind=source,
            template_id=template_id,
            template_fingerprint=expected_fingerprint,
            bundle_fingerprint=stored.bundle_fingerprint,
            config_fingerprint=stored.config_fingerprint,
        )
    _require_equal(
        "species_vocabulary",
        stored.preflight["species_vocabulary"],
        list(bundle.species_vocabulary),
        path=stored.directory.preflight_path,
        reason="SPECIES_VOCABULARY_MISMATCH",
    )
    try:
        validate_radius_model_compatibility(config.radii, potential_config)
        for binding in bundle.template_bindings:
            validate_radius_artifact_compatibility(
                config.radii, binding.structural_artifact
            )
    except RadiusConfigError as error:
        mismatch = error.mismatches[0] if error.mismatches else (None, None, None)
        raise CLIError(
            error.reason_code,
            "stored radius configuration is incompatible with the initial bundle",
            stage="export.radii.compatibility",
            source_kind=source,
            template_id=getattr(error, "template_id", None),
            config_field=mismatch[0] or error.field,
            bundle_fingerprint=stored.bundle_fingerprint,
            config_fingerprint=stored.config_fingerprint,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error

    saved = checkpoint.metadata.resolved_configuration
    expected_configs = {
        "model": _checkpoint_plain(
            potential_config, path="export.bundle.model_config"
        ),
        "loss": config.loss.to_dict(),
        "optimizer": config.optimizer.to_dict(),
        "train_step": config.train_step.to_dict(),
        "validation_step": config.validation_step.to_dict(),
        "scheduler": config.scheduler.to_dict(),
        "model_selection": config.selection.to_dict(),
    }
    for key, expected in expected_configs.items():
        if not _canonical_equal(saved.get(key), expected):
            raise CLIError(
                "CHECKPOINT_CONFIG_MISMATCH",
                f"checkpoint {key} config differs from the immutable run config",
                stage="export.checkpoint.compatibility",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="resolved_configuration",
                config_field=key,
                epoch_index=checkpoint.progress.last_completed_epoch,
                global_step=checkpoint.progress.global_step,
                config_fingerprint=stored.config_fingerprint,
            )
    try:
        saved_fit = FitConfig.from_dict(saved["fit"])
    except Exception as error:
        raise CLIError(
            "CHECKPOINT_CONFIG_MISMATCH",
            "checkpoint fit config is invalid",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="resolved_configuration",
            config_field="fit",
            original_error=error,
        ) from error
    expected_fit = config.fit.to_dict()
    actual_fit = saved_fit.to_dict()
    expected_fit.pop("max_epochs")
    actual_fit.pop("max_epochs")
    if not _canonical_equal(expected_fit, actual_fit) or (
        saved_fit.max_epochs < config.fit.max_epochs
    ):
        raise CLIError(
            "CHECKPOINT_CONFIG_MISMATCH",
            "checkpoint fit config differs from the immutable run contract",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="resolved_configuration",
            config_field="fit",
        )
    if checkpoint.metadata.unit_conventions != _unit_conventions():
        raise CLIError(
            "CHECKPOINT_UNIT_MISMATCH",
            "checkpoint unit/stress/Voigt conventions are invalid",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="unit_conventions",
        )
    if tuple(checkpoint.metadata.species_vocabulary) != tuple(
        bundle.species_vocabulary
    ):
        raise CLIError(
            "CHECKPOINT_SPECIES_MISMATCH",
            "checkpoint species vocabulary differs from the initial bundle",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="species_vocabulary",
        )
    all_templates = {
        binding.template_id: binding.full_template_fingerprint
        for binding in bundle.template_bindings
    }
    count_ids: set[str] = set()
    for split in ("train", "validation"):
        values = stored.preflight["data"][split]["template_frame_counts"]
        for template_id, count in values.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise CLIError(
                    "INVALID_PREFLIGHT_METADATA",
                    "template frame counts must be positive integers",
                    stage="export.metadata.preflight",
                    path=stored.directory.preflight_path,
                    split=split,
                    template_id=template_id,
                )
            count_ids.add(template_id)
    expected_used = {
        template_id: all_templates[template_id]
        for template_id in sorted(count_ids)
        if template_id in all_templates
    }
    if (
        set(expected_used) != count_ids
        or checkpoint.metadata.template_fingerprints != expected_used
    ):
        differing_ids = sorted(
            set(expected_used) | set(checkpoint.metadata.template_fingerprints) | count_ids
        )
        template_id = next(
            (
                value
                for value in differing_ids
                if checkpoint.metadata.template_fingerprints.get(value)
                != expected_used.get(value)
            ),
            differing_ids[0] if differing_ids else None,
        )
        raise CLIError(
            "CHECKPOINT_TEMPLATE_FINGERPRINT_MISMATCH",
            "checkpoint template fingerprints differ from the run and initial bundle",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="template_fingerprints",
            template_id=template_id,
            template_fingerprint=expected_used.get(template_id),
            bundle_fingerprint=stored.bundle_fingerprint,
            config_fingerprint=stored.config_fingerprint,
        )
    for split, manifest, frame_count, batch_count in (
        (
            "train",
            checkpoint.metadata.training_data,
            stored.train_frame_count,
            stored.train_batch_count,
        ),
        (
            "validation",
            checkpoint.metadata.validation_data,
            stored.validation_frame_count,
            stored.validation_batch_count,
        ),
    ):
        if (
            manifest.number_of_structures != frame_count
            or manifest.number_of_batches != batch_count
        ):
            raise CLIError(
                "CHECKPOINT_DATA_MANIFEST_MISMATCH",
                "checkpoint frame/batch manifest differs from preflight metadata",
                stage="export.checkpoint.compatibility",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="data_manifest",
                split=split,
            )
        _manifest_sample_ids(manifest, split=split)
    baseline = checkpoint.metadata.baseline_fit_metadata
    if not isinstance(baseline, Mapping):
        raise CLIError(
            "CHECKPOINT_RUN_IDENTITY_MISSING",
            "checkpoint lacks training-run identity metadata",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="baseline_metadata",
        )
    for key, expected in (
        ("seed", config.runtime.seed),
        ("training_run_config_fingerprint", stored.config_fingerprint),
        ("initial_bundle_fingerprint", stored.bundle_fingerprint),
    ):
        if not _canonical_equal(baseline.get(key), expected):
            raise CLIError(
                "CHECKPOINT_RUN_IDENTITY_MISMATCH",
                f"checkpoint {key} differs from the training run",
                stage="export.checkpoint.compatibility",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="baseline_metadata",
                config_field=key,
                bundle_fingerprint=stored.bundle_fingerprint,
                config_fingerprint=stored.config_fingerprint,
            )
    last = records[-1]
    if last.decision.metric_name != config.selection.monitor:
        raise CLIError(
            "CHECKPOINT_SELECTION_MISMATCH",
            "checkpoint monitored metric differs from the run selection config",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="selection",
            config_field="selection.monitor",
        )
    metric = float(last.decision.metric_value)
    if not math.isfinite(metric):
        raise CLIError(
            "NONFINITE_CHECKPOINT_METRIC",
            "checkpoint monitored metric is not finite",
            stage="export.checkpoint.compatibility",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="selection",
            epoch_index=checkpoint.progress.last_completed_epoch,
            global_step=checkpoint.progress.global_step,
        )
    return metric


def _strict_load_checkpoint_state(
    checkpoint: TrainingCheckpoint,
    bundle: ReferenceSiteModelBundle,
    stored: _StoredRun,
    *,
    source: str,
    checkpoint_path: Path,
) -> Any:
    try:
        loaded = instantiate_reference_site_model_bundle(
            bundle,
            device="cpu",
            dtype=stored.config.runtime.torch_dtype,
        )
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "fresh CPU model instantiation from the initial bundle failed",
            stage=error.validation_stage or "export.model.instantiate",
            source_kind=source,
            checkpoint_stage="model_instantiate",
            bundle_fingerprint=bundle.bundle_fingerprint,
            template_id=error.template_id,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "MODEL_INSTANTIATION_FAILED",
            "fresh CPU model instantiation from the initial bundle failed",
            stage="export.model.instantiate",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="model_instantiate",
            original_error=error,
        ) from error
    model = loaded.model
    expected = model.state_dict()
    trained = checkpoint.model_state_dict
    if list(trained) != list(expected):
        raise CLIError(
            "CHECKPOINT_MODEL_STATE_KEY_MISMATCH",
            "checkpoint model state keys/order differ from the fresh model",
            stage="export.model_state.validate",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="model_state_keys",
            epoch_index=checkpoint.progress.last_completed_epoch,
            global_step=checkpoint.progress.global_step,
        )
    for key, target in expected.items():
        value = trained[key]
        if not isinstance(value, torch.Tensor):
            raise CLIError(
                "CHECKPOINT_MODEL_STATE_VALUE_INVALID",
                "checkpoint model state values must be tensors",
                stage="export.model_state.validate",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="model_state_value",
                config_field=key,
            )
        if (
            value.device.type != "cpu"
            or value.requires_grad
            or value.grad_fn is not None
        ):
            raise CLIError(
                "CHECKPOINT_MODEL_STATE_OWNERSHIP_INVALID",
                "checkpoint model state must contain detached CPU tensors",
                stage="export.model_state.validate",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="model_state_ownership",
                config_field=key,
            )
        if value.shape != target.shape:
            raise CLIError(
                "CHECKPOINT_MODEL_STATE_SHAPE_MISMATCH",
                "checkpoint model state tensor shape differs from the fresh model",
                stage="export.model_state.validate",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="model_state_shape",
                config_field=key,
            )
        if value.dtype != target.dtype:
            raise CLIError(
                "CHECKPOINT_MODEL_STATE_DTYPE_MISMATCH",
                "checkpoint model state tensor dtype differs from the fresh model",
                stage="export.model_state.validate",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="model_state_dtype",
                config_field=key,
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.all(torch.isfinite(value))
        ):
            raise CLIError(
                "NONFINITE_CHECKPOINT_MODEL_STATE",
                "checkpoint model state contains NaN or Infinity",
                stage="export.model_state.validate",
                path=checkpoint_path,
                source_kind=source,
                checkpoint_stage="model_state_finite",
                config_field=key,
            )
    try:
        checkpoint_architecture = reference_site_model_architecture_fingerprint(
            bundle.model_config,
            trained,
            tuple(trained),
            bundle.species_vocabulary,
            bundle.conventions,
        )
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "checkpoint model state violates the initial bundle architecture",
            stage="export.model_state.architecture",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="architecture_semantics",
            config_field=error.state_key,
            bundle_fingerprint=bundle.bundle_fingerprint,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    if checkpoint_architecture != bundle.architecture_fingerprint:
        raise CLIError(
            "CHECKPOINT_ARCHITECTURE_FINGERPRINT_MISMATCH",
            "checkpoint architecture differs from the initial portable bundle",
            stage="export.model_state.architecture",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="architecture_fingerprint",
            bundle_fingerprint=bundle.bundle_fingerprint,
        )
    try:
        with torch.no_grad():
            model.load_state_dict(trained, strict=True)
        model.eval()
    except Exception as error:
        raise CLIError(
            "CHECKPOINT_MODEL_STATE_LOAD_FAILED",
            "strict checkpoint model-state loading failed",
            stage="export.model_state.load",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="strict_load",
            epoch_index=checkpoint.progress.last_completed_epoch,
            global_step=checkpoint.progress.global_step,
            original_error=error,
        ) from error
    actual = model.state_dict()
    if any(not torch.equal(actual[key].detach().cpu(), trained[key]) for key in trained):
        raise CLIError(
            "CHECKPOINT_MODEL_STATE_LOAD_MISMATCH",
            "fresh model state is not exactly equal after strict checkpoint restore",
            stage="export.model_state.load",
            path=checkpoint_path,
            source_kind=source,
            checkpoint_stage="post_load_equality",
        )
    return loaded


def _capture_export_bundle(
    stored: _StoredRun,
    checkpoint: TrainingCheckpoint,
    initial: ReferenceSiteModelBundle,
    loaded: Any,
    *,
    source: str,
    metric: float,
) -> ReferenceSiteModelBundle:
    bindings = {
        binding.template_id: binding for binding in initial.template_bindings
    }
    template_fingerprints = {
        template_id: bindings[template_id].full_template_fingerprint
        for template_id in sorted(bindings)
    }
    provenance = {
        "schema_version": EXPORT_BUNDLE_PROVENANCE_SCHEMA_VERSION,
        "parent_initial_bundle_sha256": stored.bundle_fingerprint,
        "training_config_sha256": stored.config_fingerprint,
        "train_semantic_digest": stored.train_semantic_digest,
        "validation_semantic_digest": stored.validation_semantic_digest,
        # ``best`` and ``latest`` are filesystem selection aliases, not
        # properties of the selected managed epoch.  Keeping the alias out of
        # semantic provenance makes two exports of the same checkpoint state
        # produce the same portable-bundle fingerprint while the CLI report
        # still records the alias requested by the caller.
        "source": _EXPORT_BUNDLE_PROVENANCE_SOURCE,
        "checkpoint_epoch": checkpoint.progress.last_completed_epoch,
        "global_step": checkpoint.progress.global_step,
        "selection_monitor": stored.config.selection.monitor,
        "selection_mode": stored.config.selection.mode,
        "checkpoint_metric": metric,
        "template_fingerprints": template_fingerprints,
        "radius_config_fingerprint": stored.config.radii.content_fingerprint,
    }
    try:
        exported = capture_reference_site_model_bundle(
            model=loaded.model,
            structural_artifacts={
                template_id: bindings[template_id].structural_artifact
                for template_id in sorted(bindings)
            },
            phase_specifications={
                template_id: bindings[template_id].phase_specification
                for template_id in sorted(bindings)
            },
            evaluation_policies={
                template_id: bindings[template_id].evaluation_policy
                for template_id in sorted(bindings)
                if bindings[template_id].evaluation_policy is not None
            },
            default_template_id=initial.default_template_id,
            provenance=provenance,
        )
        exported.validate()
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "portable bundle capture failed validation",
            stage=error.validation_stage or "export.bundle.capture",
            source_kind=source,
            template_id=error.template_id,
            bundle_fingerprint=stored.bundle_fingerprint,
            config_fingerprint=stored.config_fingerprint,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "BUNDLE_CAPTURE_FAILED",
            "portable bundle capture failed validation",
            stage="export.bundle.capture",
            source_kind=source,
            bundle_fingerprint=stored.bundle_fingerprint,
            config_fingerprint=stored.config_fingerprint,
            original_error=error,
        ) from error
    if exported.default_template_id != initial.default_template_id:
        raise CLIError(
            "EXPORTED_TEMPLATE_DEFAULT_MISMATCH",
            "captured bundle changed the default template ID",
            stage="export.bundle.verify",
            source_kind=source,
        )
    original_bindings = {
        binding.template_id: binding for binding in initial.template_bindings
    }
    exported_bindings = {
        binding.template_id: binding for binding in exported.template_bindings
    }
    if set(original_bindings) != set(exported_bindings):
        raise CLIError(
            "EXPORTED_TEMPLATE_SET_MISMATCH",
            "captured bundle changed the template binding set",
            stage="export.bundle.verify",
            source_kind=source,
        )
    for template_id in sorted(original_bindings):
        before = original_bindings[template_id]
        after = exported_bindings[template_id]
        policy_before = (
            None
            if before.evaluation_policy is None
            else before.evaluation_policy.to_dict()
        )
        policy_after = (
            None if after.evaluation_policy is None else after.evaluation_policy.to_dict()
        )
        if not (
            _tree_equal(
                before.structural_artifact.to_payload(),
                after.structural_artifact.to_payload(),
            )
            and _canonical_equal(
                before.phase_specification.to_dict(),
                after.phase_specification.to_dict(),
            )
            and _canonical_equal(policy_before, policy_after)
            and before.full_template_fingerprint == after.full_template_fingerprint
        ):
            raise CLIError(
                "EXPORTED_TEMPLATE_CONTENT_MISMATCH",
                "captured bundle did not preserve template/phase/policy content",
                stage="export.bundle.verify",
                source_kind=source,
                template_id=template_id,
                template_fingerprint=before.full_template_fingerprint,
            )
    exported_public_conventions = {
        key: value
        for key, value in exported.conventions.items()
        if key != "species_alignment_weights"
    }
    initial_public_conventions = {
        key: value
        for key, value in initial.conventions.items()
        if key != "species_alignment_weights"
    }
    exported_alignment = exported.conventions["species_alignment_weights"]
    initial_alignment = initial.conventions["species_alignment_weights"]
    alignment_preserved = (
        isinstance(exported_alignment, torch.Tensor)
        and isinstance(initial_alignment, torch.Tensor)
        and torch.equal(
            exported_alignment,
            initial_alignment.to(dtype=exported_alignment.dtype),
        )
    )
    if (
        not _tree_equal(exported_public_conventions, initial_public_conventions)
        or not alignment_preserved
    ):
        raise CLIError(
            "EXPORTED_CONVENTION_MISMATCH",
            "captured bundle did not preserve model/unit/stress conventions",
            stage="export.bundle.verify",
            source_kind=source,
        )
    if tuple(exported.model_state_keys) != tuple(checkpoint.model_state_dict):
        raise CLIError(
            "EXPORTED_MODEL_STATE_KEY_MISMATCH",
            "captured bundle state keys differ from the selected checkpoint",
            stage="export.bundle.verify",
            source_kind=source,
        )
    for key in exported.model_state_keys:
        if not torch.equal(exported.model_state[key], checkpoint.model_state_dict[key]):
            raise CLIError(
                "EXPORTED_MODEL_STATE_MISMATCH",
                "captured bundle model state differs from the selected checkpoint",
                stage="export.bundle.verify",
                source_kind=source,
                config_field=key,
            )
    return exported


def _normalized_target(path: Path) -> Path:
    if path.exists():
        return path.resolve(strict=True)
    return path.parent.resolve(strict=True) / path.name


def _validate_output_path(
    request: ExportBundleConfig,
    stored: _StoredRun,
    checkpoint_path: Path,
    initial_bundle_path: Path,
) -> Path:
    target = Path(request.output_path)
    if target.is_symlink():
        raise CLIError(
            "OUTPUT_SYMLINK_REJECTED",
            "bundle output target must not be a symbolic link",
            stage="export.output.path",
            path=target,
            source_kind=request.source,
        )
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise CLIError(
            "OUTPUT_PARENT_INVALID",
            "bundle output parent directory does not exist or is not a directory",
            stage="export.output.path",
            path=parent,
            source_kind=request.source,
        )
    try:
        normalized = _normalized_target(target)
    except (OSError, RuntimeError) as error:
        raise CLIError(
            "OUTPUT_PATH_INVALID",
            "bundle output path could not be resolved safely",
            stage="export.output.path",
            path=target,
            source_kind=request.source,
            original_error=error,
        ) from error
    try:
        normalized.relative_to(stored.directory.root)
    except ValueError:
        pass
    else:
        raise CLIError(
            "OUTPUT_INSIDE_RUN_DIRECTORY",
            "export output must not add or modify a file in the training run directory",
            stage="export.output.collision",
            path=target,
            source_kind=request.source,
        )
    protected = (
        stored.directory.resolved_config_path,
        stored.directory.preflight_path,
        stored.directory.status_path,
        stored.directory.checkpoints / "best.pt",
        stored.directory.checkpoints / "latest.pt",
        checkpoint_path,
        initial_bundle_path,
        stored.stored_initial_bundle_path,
        Path(str(stored.preflight["runtime"]["paths"]["config"])),
        *(
            Path(str(value))
            for value in stored.preflight["runtime"]["paths"]["train_inputs"]
        ),
        *(
            Path(str(value))
            for value in stored.preflight["runtime"]["paths"]["validation_inputs"]
        ),
    )
    for source_path in protected:
        try:
            collision = (
                target.exists()
                and source_path.exists()
                and os.path.samefile(target, source_path)
            ) or normalized == source_path.resolve(strict=False)
        except OSError:
            collision = normalized == source_path.absolute()
        if collision:
            raise CLIError(
                "OUTPUT_INPUT_COLLISION",
                "bundle output must differ from run metadata, checkpoints, "
                "initial bundle, and stored inputs",
                stage="export.output.collision",
                path=target,
                source_kind=request.source,
            )
    if target.exists():
        if not target.is_file():
            raise CLIError(
                "OUTPUT_TARGET_INVALID",
                "existing bundle output target must be a regular file",
                stage="export.output.path",
                path=target,
                source_kind=request.source,
            )
        if not request.overwrite:
            raise CLIError(
                "OUTPUT_EXISTS",
                "bundle output already exists; use --overwrite to replace it atomically",
                stage="export.output.path",
                path=target,
                source_kind=request.source,
            )
    return normalized


def _state_summary(model: torch.nn.Module, bundle: ReferenceSiteModelBundle) -> dict[str, int]:
    parameters = tuple(model.named_parameters())
    buffers = tuple(model.named_buffers())

    def elements(values: Sequence[tuple[str, torch.Tensor]]) -> int:
        return sum(int(value.numel()) for _, value in values)

    def byte_count(values: Sequence[tuple[str, torch.Tensor]]) -> int:
        return sum(int(value.numel()) * int(value.element_size()) for _, value in values)

    state_values = tuple(bundle.model_state.values())
    return {
        "parameter_tensor_count": len(parameters),
        "parameter_count": elements(parameters),
        "parameter_bytes": byte_count(parameters),
        "buffer_tensor_count": len(buffers),
        "buffer_count": elements(buffers),
        "buffer_bytes": byte_count(buffers),
        "state_tensor_count": len(state_values),
        "total_bytes": sum(
            int(value.numel()) * int(value.element_size()) for value in state_values
        ),
    }


def _build_report(
    request: ExportBundleConfig,
    stored: _StoredRun,
    checkpoint: TrainingCheckpoint,
    bundle: ReferenceSiteModelBundle,
    model: torch.nn.Module,
    output_path: Path,
    metric: float,
) -> dict[str, Any]:
    report = {
        "schema_version": EXPORT_BUNDLE_RESULT_SCHEMA_VERSION,
        "status": "dry_run_ready" if request.dry_run else "completed",
        "dry_run": request.dry_run,
        "output_written": not request.dry_run,
        "run_directory": str(stored.directory.root),
        "path_kind": "runtime_location_not_semantic_fingerprint",
        "source": {
            "kind": request.source,
            "epoch": checkpoint.progress.last_completed_epoch,
            "global_step": checkpoint.progress.global_step,
            "selection_monitor": stored.config.selection.monitor,
            "selection_mode": stored.config.selection.mode,
            "monitored_metric": metric,
        },
        "bundle_sha256": bundle.bundle_fingerprint,
        "architecture_fingerprint": bundle.architecture_fingerprint,
        "parent_initial_bundle_sha256": stored.bundle_fingerprint,
        "training_config_sha256": stored.config_fingerprint,
        "train_semantic_digest": stored.train_semantic_digest,
        "validation_semantic_digest": stored.validation_semantic_digest,
        "template_ids": sorted(bundle.binding_ids),
        "template_fingerprints": {
            binding.template_id: binding.full_template_fingerprint
            for binding in bundle.template_bindings
        },
        "species_vocabulary": list(bundle.species_vocabulary),
        "state": _state_summary(model, bundle),
        "radii": {
            "config_fingerprint": stored.config.radii.content_fingerprint,
            "user": {
                "r_ot": stored.config.radii.r_ot,
                "r_mp": stored.config.radii.r_mp,
            },
            "advanced": {
                "ot_switch_width": stored.config.radii.ot_switch_width,
                "ot_skin": stored.config.radii.ot_skin,
                "mp_skin": stored.config.radii.mp_skin,
            },
            "derived": stored.config.radii.derived.to_dict(),
        },
        "output_path": str(output_path),
        "excluded_state": [
            "candidate_neighbor_state",
            "dataset",
            "optimizer",
            "rng",
            "scheduler",
            "selection",
            "training_history",
        ],
        "message": "optimizer/training state excluded",
    }
    # Canonicalization doubles as a strict plain-JSON/nonfinite assertion.
    return json.loads(canonical_runtime_json(report))


def _check_run_unchanged(prepared: _PreparedExport) -> None:
    directory = prepared.stored.directory
    try:
        directory.validate_resume_lock_available()
    except RunDirectoryError as error:
        raise CLIError(
            error.reason_code,
            "an active or foreign run lock prevents bundle export",
            stage="export.active_run",
            path=error.path,
            source_kind=prepared.request.source,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    current_config = _load_stored_config(directory)
    current_preflight = load_runtime_json(
        directory.preflight_path, stage="export.toctou.preflight"
    )
    current_status = load_runtime_json(
        directory.status_path, stage="export.toctou.status"
    )
    if (
        current_config.to_dict() != prepared.stored.config.to_dict()
        or not _canonical_equal(current_preflight, prepared.stored.preflight)
        or not _canonical_equal(current_status, prepared.stored.status)
    ):
        raise CLIError(
            "RUN_METADATA_TOCTOU_MISMATCH",
            "training run metadata changed during export preflight",
            stage="export.toctou.metadata",
            path=directory.root,
            source_kind=prepared.request.source,
        )
    try:
        current_checkpoint = (
            prepared.manager.load_best()
            if prepared.request.source == "best"
            else prepared.manager.load_latest()
        )
    except Exception as error:
        raise CLIError(
            "CHECKPOINT_TOCTOU_LOAD_FAILED",
            "selected checkpoint could not be reloaded before export commit",
            stage="export.toctou.checkpoint",
            path=prepared.checkpoint_path,
            source_kind=prepared.request.source,
            checkpoint_stage="reload",
            original_error=error,
        ) from error
    if not _tree_equal(current_checkpoint.to_dict(), prepared.checkpoint.to_dict()):
        raise CLIError(
            "CHECKPOINT_TOCTOU_MISMATCH",
            "selected checkpoint changed during export preflight",
            stage="export.toctou.checkpoint",
            path=prepared.checkpoint_path,
            source_kind=prepared.request.source,
            checkpoint_stage="content_compare",
            epoch_index=prepared.checkpoint.progress.last_completed_epoch,
            global_step=prepared.checkpoint.progress.global_step,
        )
    try:
        current_initial = load_reference_site_model_bundle(
            prepared.initial_bundle_path, map_location="cpu"
        )
    except Exception as error:
        raise CLIError(
            getattr(error, "reason_code", None) or "INITIAL_BUNDLE_TOCTOU_LOAD_FAILED",
            "initial bundle could not be reloaded before export commit",
            stage="export.toctou.initial_bundle",
            bundle_path=prepared.initial_bundle_path,
            source_kind=prepared.request.source,
            underlying_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error
    if (
        current_initial.bundle_fingerprint
        != prepared.initial_bundle.bundle_fingerprint
        or not _tree_equal(
            current_initial.to_payload(), prepared.initial_bundle.to_payload()
        )
    ):
        raise CLIError(
            "INITIAL_BUNDLE_TOCTOU_MISMATCH",
            "initial bundle changed during export preflight",
            stage="export.toctou.initial_bundle",
            bundle_path=prepared.initial_bundle_path,
            source_kind=prepared.request.source,
            bundle_fingerprint=prepared.stored.bundle_fingerprint,
        )


def _prepare_export(request: ExportBundleConfig) -> _PreparedExport:
    try:
        directory = TrainingRunDirectory.open_existing(request.run_directory)
    except RunDirectoryError as error:
        raise CLIError(
            error.reason_code,
            "training run directory or active lock validation failed",
            stage=error.stage,
            path=error.path,
            source_kind=request.source,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    # Check the common run lock before reading metadata.  A fresh scratch run
    # acquires it immediately after creating the directory, so export reports
    # active ownership even while the first metadata files are still being
    # committed.
    try:
        directory.validate_resume_lock_available()
    except RunDirectoryError as error:
        raise CLIError(
            error.reason_code,
            "an active or foreign run lock prevents bundle export",
            stage="export.active_run",
            path=error.path,
            source_kind=request.source,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    stored = _validate_stored_run(directory)
    try:
        directory.validate_resume_lock_available()
    except RunDirectoryError as error:
        raise CLIError(
            error.reason_code,
            "an active or foreign run lock prevents bundle export",
            stage="export.active_run",
            path=error.path,
            source_kind=request.source,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    manager = _checkpoint_directory(stored)
    checkpoint_path, checkpoint, records = _load_checkpoint(manager, request.source)
    initial_path, initial = _load_initial_bundle(
        stored, request.initial_bundle_path, source=request.source
    )
    metric = _validate_checkpoint_compatibility(
        stored,
        checkpoint,
        initial,
        records,
        source=request.source,
        checkpoint_path=checkpoint_path,
    )
    loaded = _strict_load_checkpoint_state(
        checkpoint,
        initial,
        stored,
        source=request.source,
        checkpoint_path=checkpoint_path,
    )
    exported = _capture_export_bundle(
        stored,
        checkpoint,
        initial,
        loaded,
        source=request.source,
        metric=metric,
    )
    output = _validate_output_path(request, stored, checkpoint_path, initial_path)
    report = _build_report(
        request,
        stored,
        checkpoint,
        exported,
        loaded.model,
        output,
        metric,
    )
    return _PreparedExport(
        request=request,
        stored=stored,
        manager=manager,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        records=records,
        initial_bundle_path=initial_path,
        initial_bundle=initial,
        exported_bundle=exported,
        output_path=output,
        source_metric=metric,
        report=report,
    )


@contextmanager
def _preserve_rng_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_states = None
    if torch.cuda.is_initialized():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(list(cuda_states))


def _export_bundle_impl(request: ExportBundleConfig) -> dict[str, Any]:
    prepared = _prepare_export(request)
    _check_run_unchanged(prepared)
    if request.dry_run:
        return dict(prepared.report)
    current_output = _validate_output_path(
        request,
        prepared.stored,
        prepared.checkpoint_path,
        prepared.initial_bundle_path,
    )
    if current_output != prepared.output_path:
        raise CLIError(
            "OUTPUT_PATH_TOCTOU_MISMATCH",
            "bundle output resolved to a different location before commit",
            stage="export.toctou.output",
            path=current_output,
            source_kind=request.source,
        )
    try:
        save_reference_site_model_bundle(
            prepared.output_path,
            prepared.exported_bundle,
            overwrite=request.overwrite,
        )
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "atomic portable bundle save failed",
            stage=error.validation_stage or "export.output.save",
            path=prepared.output_path,
            source_kind=request.source,
            checkpoint_stage="bundle_save",
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "BUNDLE_SAVE_FAILED",
            "atomic portable bundle save failed; an existing target was preserved",
            stage="export.output.save",
            path=prepared.output_path,
            source_kind=request.source,
            checkpoint_stage="bundle_save",
            underlying_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error
    try:
        saved = load_reference_site_model_bundle(
            prepared.output_path, map_location="cpu"
        )
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "saved bundle failed final weights-only validation",
            stage=error.validation_stage or "export.output.verify",
            path=prepared.output_path,
            source_kind=request.source,
            checkpoint_stage="saved_bundle_reload",
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "SAVED_BUNDLE_RELOAD_FAILED",
            "saved bundle failed final weights-only validation",
            stage="export.output.verify",
            path=prepared.output_path,
            source_kind=request.source,
            checkpoint_stage="saved_bundle_reload",
            original_error=error,
        ) from error
    if (
        saved.bundle_fingerprint != prepared.exported_bundle.bundle_fingerprint
        or not _tree_equal(saved.to_payload(), prepared.exported_bundle.to_payload())
    ):
        raise CLIError(
            "SAVED_BUNDLE_CONTENT_MISMATCH",
            "saved bundle differs from the validated in-memory capture",
            stage="export.output.verify",
            path=prepared.output_path,
            source_kind=request.source,
            checkpoint_stage="saved_bundle_compare",
        )
    return dict(prepared.report)


def _with_export_context(error: CLIError, request: ExportBundleConfig) -> CLIError:
    if error.source_kind is not None and error.run_directory is not None:
        return error
    return CLIError(
        error.reason_code,
        error.message,
        stage=error.stage,
        bundle_path=error.bundle_path,
        path=error.path,
        frame_index=error.frame_index,
        sample_id=error.sample_id,
        template_id=error.template_id,
        term=error.term,
        config_field=error.config_field,
        split=error.split,
        epoch_index=error.epoch_index,
        batch_index=error.batch_index,
        global_step=error.global_step,
        failure_phase=error.failure_phase,
        rollback_performed=error.rollback_performed,
        source_kind=error.source_kind or request.source,
        run_directory=error.run_directory or request.run_directory,
        checkpoint_stage=error.checkpoint_stage,
        bundle_fingerprint=error.bundle_fingerprint,
        config_fingerprint=error.config_fingerprint,
        template_fingerprint=error.template_fingerprint,
        solver_path=error.solver_path,
        prediction_stage=error.prediction_stage,
        predictor_reason_code=error.predictor_reason_code,
        underlying_reason_code=error.underlying_reason_code,
        original_error=error.original_error,
    )


def export_bundle(
    run_directory: str | os.PathLike[str] | ExportBundleConfig,
    *,
    source: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    initial_bundle_path: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and export ``best.pt`` or ``latest.pt`` without model execution."""

    if isinstance(run_directory, ExportBundleConfig):
        if source is not None or output_path is not None or initial_bundle_path is not None:
            raise TypeError("keyword paths/source must be omitted with ExportBundleConfig")
        if dry_run is not False or overwrite is not False:
            raise TypeError("dry_run/overwrite must be carried by ExportBundleConfig")
        request = run_directory
    else:
        if source is None:
            raise TypeError("source is required")
        if output_path is None:
            raise TypeError("output_path is required")
        request = ExportBundleConfig(
            run_directory=run_directory,
            source=source,
            output_path=output_path,
            initial_bundle_path=initial_bundle_path,
            dry_run=dry_run,
            overwrite=overwrite,
        )
    try:
        with _preserve_rng_state():
            return _export_bundle_impl(request)
    except CLIError as error:
        contextual = _with_export_context(error, request)
        if contextual is error:
            raise
        raise contextual from error
    except RunDirectoryError as error:
        raise CLIError(
            error.reason_code,
            "training run directory validation failed",
            stage=error.stage,
            path=error.path,
            source_kind=request.source,
            run_directory=request.run_directory,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        reason = getattr(error, "reason_code", None) or "EXPORT_BUNDLE_FAILED"
        raise CLIError(
            reason,
            "portable checkpoint bundle export failed",
            stage="export.runtime",
            path=request.run_directory,
            source_kind=request.source,
            run_directory=request.run_directory,
            underlying_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error


def render_export_bundle_json(report: Mapping[str, Any]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("export report must be a mapping")
    return canonical_runtime_json(report)


def render_export_bundle_human(report: Mapping[str, Any]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("export report must be a mapping")
    if report.get("status") not in {"dry_run_ready", "completed"}:
        raise ValueError("export report status must be dry_run_ready or completed")
    source = report["source"]
    state = report["state"]
    radii = report["radii"]
    lines = [
        "Reference-site MLIP portable bundle export",
        f"Status: {'ready (dry run)' if report['dry_run'] else 'completed'}",
        f"Source: {source['kind']} (epoch {source['epoch']}, global step {source['global_step']})",
        "Monitored metric: "
        f"{source['selection_monitor']}={source['monitored_metric']} "
        f"({source['selection_mode']})",
        f"Bundle SHA-256: {report['bundle_sha256']}",
        f"Architecture fingerprint: {report['architecture_fingerprint']}",
        "Template IDs: " + ", ".join(report["template_ids"]),
        "Species vocabulary: " + ", ".join(str(v) for v in report["species_vocabulary"]),
        (
            "Model state: "
            f"{state['parameter_count']} parameter elements "
            f"({state['parameter_tensor_count']} tensors) / "
            f"{state['buffer_count']} buffer elements "
            f"({state['buffer_tensor_count']} tensors), "
            f"{state['total_bytes']} bytes"
        ),
        (
            "Interaction radii: "
            f"r_ot={radii['user']['r_ot']} A, r_mp={radii['user']['r_mp']} A, "
            f"r_on_ot={radii['derived']['r_on_ot']} A, "
            f"r_candidate_ot={radii['derived']['r_candidate_ot']} A, "
            f"r_candidate_mp={radii['derived']['r_candidate_mp']} A"
        ),
        f"Output: {report['output_path']}",
        "Optimizer/training state excluded.",
    ]
    if report["dry_run"]:
        lines.append("No output file or run-directory file was changed.")
    return "\n".join(lines)


# Descriptive aliases for API callers.
export_checkpoint_bundle = export_bundle
render_export_json = render_export_bundle_json
render_export_human = render_export_bundle_human


__all__ = [
    "EXPORT_BUNDLE_PROVENANCE_SCHEMA_VERSION",
    "EXPORT_BUNDLE_RESULT_SCHEMA_VERSION",
    "ExportBundleConfig",
    "export_bundle",
    "export_checkpoint_bundle",
    "render_export_bundle_human",
    "render_export_bundle_json",
    "render_export_human",
    "render_export_json",
]
