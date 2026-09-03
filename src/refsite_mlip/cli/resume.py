"""Safe exact continuation of a checkpointed training run directory."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import copy
import json
from numbers import Integral
import os
from pathlib import Path
import re
from typing import Any

import torch

from refsite_mlip.config import (
    TrainingRunConfig,
    TrainingRunConfigError,
)
from refsite_mlip.config.radii import (
    RadiusConfigError,
    validate_radius_artifact_compatibility,
    validate_radius_model_compatibility,
)
from refsite_mlip.config.training_run import (
    TRAINING_RUN_CONFIG_SCHEMA_VERSION,
    _baseline_preflight,
    _composition_statistics,
    _label_statistics,
    _load_split,
    _phase_specification_fingerprint,
    _preflight_device,
    _split_digest,
)
from refsite_mlip.data import TemplateRegistry
from refsite_mlip.models import (
    ModelBundleError,
    PotentialConfig,
    load_reference_site_model_bundle,
)
from refsite_mlip.training import (
    CheckpointManager,
    CheckpointManagerConfig,
    CheckpointRestoreError,
    CheckpointedFitExecutionError,
    FitConfig,
    FitExecutionError,
    ResumePolicy,
    RunDirectoryError,
    TrainingCheckpoint,
    TrainingRunDirectory,
    build_optimizer,
    build_scheduler,
    canonical_runtime_json,
    load_runtime_json,
    run_checkpointed_resumed_fit,
    validate_checkpoint_history,
    validate_managed_checkpoint_history,
)
from refsite_mlip.training.checkpoint import (
    _data_manifest,
    _package_versions,
    _plain as _checkpoint_plain,
    _template_fingerprint_mapping,
    _unit_conventions,
)
from refsite_mlip.transport import TRAIN_FIXED

from .errors import CLIError, CLIInterruptedError
from .train import (
    _PreparedTrainingRuntime,
    _batch_context,
    _batch_samples,
    _nested_reason,
    _nested_text_attribute,
    _prepare_training_runtime,
    seed_training_runtime,
)
from .validate_train_config import _cli_error as _training_config_cli_error


RESUME_PREFLIGHT_SCHEMA_VERSION = "refsite_training_resume_preflight_v1"
RESUME_RESULT_SCHEMA_VERSION = "refsite_training_resume_result_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _StoredResolvedRun:
    config_fingerprint: str
    bundle_fingerprint: str
    train_semantic_digest: str
    validation_semantic_digest: str
    train_batch_count: int
    validation_batch_count: int
    resolved_device: str
    resolved_dtype: str
    species_vocabulary: tuple[int, ...]
    runtime_paths: Mapping[str, Any]


@dataclass(frozen=True)
class _ResumePreflight:
    directory: TrainingRunDirectory
    config: TrainingRunConfig
    resolved: _StoredResolvedRun
    checkpoint: TrainingCheckpoint
    manager: CheckpointManager
    previous_epochs: tuple[int, ...]
    report: Mapping[str, Any]
    stored_preflight: Mapping[str, Any]
    stored_status: Mapping[str, Any]


def _canonical_equal(first: Any, second: Any) -> bool:
    try:
        left = canonical_runtime_json({"value": first})
        right = canonical_runtime_json({"value": second})
    except (TypeError, ValueError):
        return False
    return left == right


def _tree_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.shape == second.shape
            and first.dtype == second.dtype
            and torch.equal(first.detach().cpu(), second.detach().cpu())
        )
    if isinstance(first, Mapping):
        return (
            isinstance(second, Mapping)
            and first.keys() == second.keys()
            and all(_tree_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (tuple, list)):
        return (
            isinstance(second, (tuple, list))
            and len(first) == len(second)
            and all(_tree_equal(left, right) for left, right in zip(first, second))
        )
    return first == second


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CLIError(
            "INVALID_RUN_METADATA",
            f"{field} must be a JSON object",
            stage="resume.metadata.validate",
            config_field=field,
        )
    return value


def _require_equal(
    field: str,
    actual: Any,
    expected: Any,
    *,
    reason: str = "RUN_METADATA_MISMATCH",
) -> None:
    if not _canonical_equal(actual, expected):
        raise CLIError(
            reason,
            f"stored {field} does not match the immutable run contract",
            stage="resume.metadata.validate",
            config_field=field,
        )


def _safe_runtime_file(value: Any, *, field: str) -> Path:
    if type(value) is not str or not value:
        raise CLIError(
            "INVALID_RESOLVED_PATH",
            "stored runtime input path must be a nonempty absolute path",
            stage="resume.paths",
            config_field=field,
        )
    path = Path(value)
    if not path.is_absolute():
        raise CLIError(
            "INVALID_RESOLVED_PATH",
            "stored runtime input path must be absolute and is never reinterpreted",
            stage="resume.paths",
            path=path,
            config_field=field,
        )
    if path.is_symlink():
        raise CLIError(
            "RESOLVED_INPUT_SYMLINK_REJECTED",
            "stored runtime input path must not become a symbolic link",
            stage="resume.paths",
            path=path,
            config_field=field,
        )
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CLIError(
            "RESOLVED_INPUT_MISSING",
            "stored runtime input moved or no longer exists",
            stage="resume.paths",
            path=path,
            config_field=field,
            original_error=error,
        ) from error
    if not resolved.is_file():
        raise CLIError(
            "INVALID_RESOLVED_INPUT",
            "stored runtime input must remain a regular file",
            stage="resume.paths",
            path=resolved,
            config_field=field,
        )
    return resolved


def _load_config(directory: TrainingRunDirectory) -> TrainingRunConfig:
    payload = load_runtime_json(
        directory.resolved_config_path,
        stage="resume.metadata.resolved_config",
    )
    try:
        return TrainingRunConfig.from_dict(payload)
    except TrainingRunConfigError as error:
        raise _training_config_cli_error(
            error, requested_path=directory.resolved_config_path
        ) from error
    except Exception as error:
        raise CLIError(
            "INVALID_RESOLVED_CONFIG",
            "stored resolved_config.json is invalid",
            stage="resume.metadata.resolved_config",
            path=directory.resolved_config_path,
            original_error=error,
        ) from error


def _validate_preflight_metadata(
    directory: TrainingRunDirectory,
    config: TrainingRunConfig,
    preflight: Mapping[str, Any],
) -> tuple[_StoredResolvedRun, tuple[Path, ...], tuple[Path, ...], Path]:
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
            stage="resume.metadata.preflight",
            path=directory.preflight_path,
        )
    _require_equal("preflight.status", preflight["status"], "preflight_ready")
    _require_equal("preflight.training_executed", preflight["training_executed"], False)
    _require_equal(
        "preflight.schema_version",
        preflight["schema_version"],
        TRAINING_RUN_CONFIG_SCHEMA_VERSION,
    )
    _require_equal(
        "preflight.config_fingerprint",
        preflight["config_fingerprint"],
        config.config_fingerprint,
        reason="CONFIG_FINGERPRINT_MISMATCH",
    )
    bundle_fingerprint = preflight["bundle_fingerprint"]
    if type(bundle_fingerprint) is not str or _SHA256.fullmatch(bundle_fingerprint) is None:
        raise CLIError(
            "INVALID_BUNDLE_FINGERPRINT",
            "stored bundle fingerprint is not a lowercase SHA-256",
            stage="resume.metadata.preflight",
            config_field="bundle_fingerprint",
        )

    runtime = _require_mapping(preflight["runtime"], field="runtime")
    if set(runtime) != {"device", "dtype", "seed", "configured_paths", "paths"}:
        raise CLIError(
            "INVALID_PREFLIGHT_METADATA",
            "preflight runtime metadata fields are invalid",
            stage="resume.metadata.preflight",
            config_field="runtime",
        )
    _require_equal("runtime.device", runtime["device"], config.runtime.device)
    _require_equal("runtime.dtype", runtime["dtype"], config.runtime.dtype)
    _require_equal("runtime.seed", runtime["seed"], config.runtime.seed)
    resolved_device = _preflight_device(config.runtime)
    _require_equal("runtime.device", runtime["device"], resolved_device)

    paths = _require_mapping(runtime["paths"], field="runtime.paths")
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
            stage="resume.metadata.preflight",
            config_field="runtime.paths",
        )
    _require_equal(
        "runtime.paths.path_kind",
        paths["path_kind"],
        "runtime_location_not_semantic_fingerprint",
    )
    output = Path(str(paths["output_directory"]))
    try:
        output_resolved = output.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CLIError(
            "RUN_DIRECTORY_PATH_MISMATCH",
            "stored output directory is unavailable",
            stage="resume.paths",
            path=output,
            original_error=error,
        ) from error
    if output_resolved != directory.root:
        raise CLIError(
            "RUN_DIRECTORY_PATH_MISMATCH",
            "requested run directory differs from its stored runtime location",
            stage="resume.paths",
            path=directory.root,
        )
    expected_paths = {
        "output_directory": str(directory.root),
        "resolved_config": str(directory.resolved_config_path),
        "preflight": str(directory.preflight_path),
        "run_status": str(directory.status_path),
        "latest_checkpoint": str(directory.checkpoints / "latest.pt"),
        "best_checkpoint": str(directory.checkpoints / "best.pt"),
        "epoch_checkpoint_pattern": str(
            directory.checkpoints / "epoch_XXXXXX.pt"
        ),
    }
    _require_equal("expected_paths", preflight["expected_paths"], expected_paths)

    train_values = paths["train_inputs"]
    validation_values = paths["validation_inputs"]
    if not isinstance(train_values, list) or len(train_values) != len(config.data.train):
        raise CLIError(
            "DATA_PATH_MANIFEST_MISMATCH",
            "stored train input path count differs from the config",
            stage="resume.paths",
            split="train",
        )
    if not isinstance(validation_values, list) or len(validation_values) != len(
        config.data.validation
    ):
        raise CLIError(
            "DATA_PATH_MANIFEST_MISMATCH",
            "stored validation input path count differs from the config",
            stage="resume.paths",
            split="validation",
        )
    bundle_path = _safe_runtime_file(
        paths["initial_bundle"], field="runtime.paths.initial_bundle"
    )
    train_paths = tuple(
        _safe_runtime_file(value, field=f"runtime.paths.train_inputs[{index}]")
        for index, value in enumerate(train_values)
    )
    validation_paths = tuple(
        _safe_runtime_file(
            value, field=f"runtime.paths.validation_inputs[{index}]"
        )
        for index, value in enumerate(validation_values)
    )

    configured = {
        "initial_bundle": config.initial_bundle,
        "output_directory": config.output_directory,
        "train_inputs": [source.path for source in config.data.train],
        "validation_inputs": [source.path for source in config.data.validation],
        "path_kind": "original_config_expression_in_semantic_fingerprint",
    }
    _require_equal("runtime.configured_paths", runtime["configured_paths"], configured)
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
    _require_equal(
        "training_configuration",
        preflight["training_configuration"],
        expected_training,
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
    _require_equal("radii", preflight["radii"], expected_radii)

    data = _require_mapping(preflight["data"], field="data")
    train = _require_mapping(data.get("train"), field="data.train")
    validation = _require_mapping(
        data.get("validation"), field="data.validation"
    )
    try:
        train_digest = str(train["semantic_digest"])
        validation_digest = str(validation["semantic_digest"])
        train_batches = int(train["batch_count"])
        validation_batches = int(validation["batch_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise CLIError(
            "INVALID_PREFLIGHT_METADATA",
            "stored data digest or batch metadata is invalid",
            stage="resume.metadata.preflight",
            original_error=error,
        ) from error
    for field, digest in (
        ("data.train.semantic_digest", train_digest),
        ("data.validation.semantic_digest", validation_digest),
    ):
        if _SHA256.fullmatch(digest) is None:
            raise CLIError(
                "INVALID_DATA_DIGEST",
                "stored semantic digest is invalid",
                stage="resume.metadata.preflight",
                config_field=field,
            )
    if train_batches <= 0 or validation_batches <= 0:
        raise CLIError(
            "INVALID_BATCH_MANIFEST",
            "stored batch counts must be positive",
            stage="resume.metadata.preflight",
        )
    species = preflight["species_vocabulary"]
    if not isinstance(species, list) or not species:
        raise CLIError(
            "INVALID_SPECIES_VOCABULARY",
            "stored species vocabulary is invalid",
            stage="resume.metadata.preflight",
        )

    stored = _StoredResolvedRun(
        config_fingerprint=config.config_fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        train_semantic_digest=train_digest,
        validation_semantic_digest=validation_digest,
        train_batch_count=train_batches,
        validation_batch_count=validation_batches,
        resolved_device=resolved_device,
        resolved_dtype=config.runtime.dtype,
        species_vocabulary=tuple(int(value) for value in species),
        runtime_paths=copy.deepcopy(dict(paths)),
    )
    return stored, train_paths, validation_paths, bundle_path


def _bundle_template_metadata(bundle: Any) -> dict[str, Any]:
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


def _validate_bundle_and_checkpoint(
    config: TrainingRunConfig,
    stored: _StoredResolvedRun,
    preflight: Mapping[str, Any],
    checkpoint: TrainingCheckpoint,
    bundle_path: Path,
) -> tuple[Any, TemplateRegistry, Mapping[str, Any]]:
    try:
        bundle = load_reference_site_model_bundle(bundle_path, map_location="cpu")
        templates = bundle.validate(bundle_path=str(bundle_path))
        potential_config = PotentialConfig.from_dict(bundle.model_config)
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "stored initial bundle failed safe validation during resume",
            stage=error.validation_stage or "resume.bundle",
            bundle_path=bundle_path,
            template_id=error.template_id,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            "BUNDLE_RELOAD_FAILED",
            "stored initial bundle could not be safely loaded",
            stage="resume.bundle",
            bundle_path=bundle_path,
            original_error=error,
        ) from error
    if bundle.bundle_fingerprint != stored.bundle_fingerprint:
        raise CLIError(
            "BUNDLE_FINGERPRINT_MISMATCH",
            "initial bundle moved or changed since the fresh run",
            stage="resume.bundle",
            bundle_path=bundle_path,
        )
    _require_equal(
        "template_fingerprints",
        preflight["template_fingerprints"],
        _bundle_template_metadata(bundle),
        reason="TEMPLATE_FINGERPRINT_MISMATCH",
    )
    _require_equal(
        "species_vocabulary",
        list(stored.species_vocabulary),
        list(bundle.species_vocabulary),
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
            "stored radius configuration is incompatible with the bundle",
            stage="resume.radii",
            template_id=getattr(error, "template_id", None),
            config_field=mismatch[0],
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error

    saved = checkpoint.metadata.resolved_configuration
    expected_configs = {
        "model": _checkpoint_plain(
            potential_config, path="resume.bundle.model_config"
        ),
        "loss": config.loss.to_dict(),
        "optimizer": config.optimizer.to_dict(),
        "train_step": config.train_step.to_dict(),
        "validation_step": config.validation_step.to_dict(),
        "scheduler": config.scheduler.to_dict(),
        "model_selection": config.selection.to_dict(),
    }
    for key, expected in expected_configs.items():
        _require_equal(
            f"checkpoint.{key}",
            saved.get(key),
            expected,
            reason="CHECKPOINT_CONFIG_MISMATCH",
        )
    saved_fit = FitConfig.from_dict(saved["fit"])
    original_nonmax = config.fit.to_dict()
    saved_nonmax = saved_fit.to_dict()
    original_nonmax.pop("max_epochs")
    saved_nonmax.pop("max_epochs")
    _require_equal(
        "checkpoint.fit except max_epochs",
        saved_nonmax,
        original_nonmax,
        reason="CHECKPOINT_CONFIG_MISMATCH",
    )
    if saved_fit.max_epochs < config.fit.max_epochs:
        raise CLIError(
            "CHECKPOINT_CONFIG_MISMATCH",
            "latest checkpoint max_epochs predates the immutable run config",
            stage="resume.checkpoint.compatibility",
            config_field="fit.max_epochs",
        )
    if checkpoint.metadata.package_versions != _package_versions():
        raise CLIError(
            "CHECKPOINT_VERSION_MISMATCH",
            "checkpoint package versions differ from the current runtime",
            stage="resume.checkpoint.compatibility",
        )
    if checkpoint.metadata.unit_conventions != _unit_conventions():
        raise CLIError(
            "CHECKPOINT_UNIT_MISMATCH",
            "checkpoint unit/stress/Voigt conventions differ from the runtime",
            stage="resume.checkpoint.compatibility",
        )
    if tuple(checkpoint.metadata.species_vocabulary) != tuple(
        bundle.species_vocabulary
    ):
        raise CLIError(
            "CHECKPOINT_SPECIES_MISMATCH",
            "checkpoint species vocabulary differs from the bundle",
            stage="resume.checkpoint.compatibility",
        )
    if list(checkpoint.model_state_dict) != list(bundle.model_state_keys):
        raise CLIError(
            "CHECKPOINT_MODEL_STATE_MISMATCH",
            "checkpoint and bundle model state keys/order differ",
            stage="resume.checkpoint.compatibility",
        )
    expected_dtype = config.runtime.torch_dtype
    for key, initial in bundle.model_state.items():
        trained = checkpoint.model_state_dict[key]
        if trained.shape != initial.shape:
            raise CLIError(
                "CHECKPOINT_MODEL_STATE_MISMATCH",
                "checkpoint and bundle model state shapes differ",
                stage="resume.checkpoint.compatibility",
                config_field=key,
            )
        if trained.is_floating_point() and trained.dtype != expected_dtype:
            raise CLIError(
                "CHECKPOINT_DTYPE_MISMATCH",
                "checkpoint floating model state differs from runtime dtype",
                stage="resume.checkpoint.compatibility",
                config_field=key,
            )
    baseline = checkpoint.metadata.baseline_fit_metadata
    if not isinstance(baseline, Mapping):
        raise CLIError(
            "CHECKPOINT_BASELINE_METADATA_MISSING",
            "checkpoint lacks the fresh-run baseline/config identity metadata",
            stage="resume.checkpoint.compatibility",
        )
    for key, expected in (
        ("seed", config.runtime.seed),
        ("training_run_config_fingerprint", config.config_fingerprint),
        ("initial_bundle_fingerprint", bundle.bundle_fingerprint),
    ):
        _require_equal(
            f"checkpoint.baseline.{key}",
            baseline.get(key),
            expected,
            reason="CHECKPOINT_RUN_IDENTITY_MISMATCH",
        )
    if checkpoint.cuda_device_count:
        if not torch.cuda.is_available():
            raise CLIError(
                "CHECKPOINT_CUDA_UNAVAILABLE",
                "checkpoint contains CUDA RNG state but CUDA is unavailable",
                stage="resume.checkpoint.compatibility",
            )
        if checkpoint.cuda_device_count != torch.cuda.device_count():
            raise CLIError(
                "CHECKPOINT_CUDA_DEVICE_COUNT_MISMATCH",
                "checkpoint CUDA RNG state count differs from this runtime",
                stage="resume.checkpoint.compatibility",
            )

    registry = TemplateRegistry()
    for template_id in sorted(templates):
        registry.add(templates[template_id])
    return bundle, registry, templates


def _validate_data(
    config: TrainingRunConfig,
    stored: _StoredResolvedRun,
    preflight: Mapping[str, Any],
    checkpoint: TrainingCheckpoint,
    registry: TemplateRegistry,
    templates: Mapping[str, Any],
    train_paths: tuple[Path, ...],
    validation_paths: tuple[Path, ...],
) -> None:
    try:
        train_samples = _load_split(
            config.data.train,
            train_paths,
            split="train",
            registry=registry,
            dtype=torch.float64,
            config_path=None,
        )
        validation_samples = _load_split(
            config.data.validation,
            validation_paths,
            split="validation",
            registry=registry,
            dtype=torch.float64,
            config_path=None,
        )
    except TrainingRunConfigError as error:
        raise _training_config_cli_error(
            error, requested_path="<stored-runtime-paths>"
        ) from error
    train_digest = _split_digest(train_samples, templates, split="train")
    validation_digest = _split_digest(
        validation_samples, templates, split="validation"
    )
    if train_digest != stored.train_semantic_digest:
        first = train_samples[0] if train_samples else None
        raise CLIError(
            "TRAIN_DATA_DIGEST_MISMATCH",
            "training data moved or changed since preflight",
            stage="resume.data.digest",
            path=train_paths[0] if train_paths else None,
            split="train",
            frame_index=0 if first is not None else None,
            sample_id=None if first is None else first.sample_id,
            template_id=None if first is None else first.template_id,
        )
    if validation_digest != stored.validation_semantic_digest:
        first = validation_samples[0] if validation_samples else None
        raise CLIError(
            "VALIDATION_DATA_DIGEST_MISMATCH",
            "validation data moved or changed since preflight",
            stage="resume.data.digest",
            path=validation_paths[0] if validation_paths else None,
            split="validation",
            frame_index=0 if first is not None else None,
            sample_id=None if first is None else first.sample_id,
            template_id=None if first is None else first.template_id,
        )
    train_batches = _batch_samples(
        train_samples,
        batch_size=config.data.batch_size,
        registry=registry,
        device="cpu",
        dtype=config.runtime.torch_dtype,
    )
    validation_batches = _batch_samples(
        validation_samples,
        batch_size=config.data.batch_size,
        registry=registry,
        device="cpu",
        dtype=config.runtime.torch_dtype,
    )
    if len(train_batches) != stored.train_batch_count or len(
        validation_batches
    ) != stored.validation_batch_count:
        raise CLIError(
            "BATCH_MANIFEST_MISMATCH",
            "deterministic batch count differs from stored preflight",
            stage="resume.data.manifest",
        )
    if _data_manifest(train_batches, split_name="train") != (
        checkpoint.metadata.training_data
    ):
        raise CLIError(
            "TRAIN_BATCH_MANIFEST_MISMATCH",
            "training batch boundaries/content differ from the checkpoint",
            stage="resume.data.manifest",
            split="train",
        )
    if _data_manifest(validation_batches, split_name="validation") != (
        checkpoint.metadata.validation_data
    ):
        raise CLIError(
            "VALIDATION_BATCH_MANIFEST_MISMATCH",
            "validation batch boundaries/content differ from the checkpoint",
            stage="resume.data.manifest",
            split="validation",
        )
    if _template_fingerprint_mapping(train_batches, validation_batches) != (
        checkpoint.metadata.template_fingerprints
    ):
        raise CLIError(
            "CHECKPOINT_TEMPLATE_MANIFEST_MISMATCH",
            "data template IDs/fingerprints differ from the checkpoint",
            stage="resume.data.manifest",
        )
    data = _require_mapping(preflight["data"], field="data")
    if set(data) != {"train", "validation"}:
        raise CLIError(
            "INVALID_PREFLIGHT_METADATA",
            "stored data metadata must contain only train and validation",
            stage="resume.metadata.preflight",
            config_field="data",
        )
    split_fields = {
        "semantic_digest",
        "frame_count",
        "batch_count",
        "template_frame_counts",
        "composition_statistics",
        "label_statistics",
    }
    actual_splits = {
        "train": {
            "semantic_digest": train_digest,
            "frame_count": len(train_samples),
            "batch_count": len(train_batches),
            "template_frame_counts": dict(
                sorted(Counter(sample.template_id for sample in train_samples).items())
            ),
            "composition_statistics": _composition_statistics(train_samples),
            "label_statistics": _label_statistics(train_samples),
        },
        "validation": {
            "semantic_digest": validation_digest,
            "frame_count": len(validation_samples),
            "batch_count": len(validation_batches),
            "template_frame_counts": dict(
                sorted(
                    Counter(
                        sample.template_id for sample in validation_samples
                    ).items()
                )
            ),
            "composition_statistics": _composition_statistics(validation_samples),
            "label_statistics": _label_statistics(validation_samples),
        },
    }
    for split, expected in actual_splits.items():
        stored_split = _require_mapping(data[split], field=f"data.{split}")
        if set(stored_split) != split_fields:
            raise CLIError(
                "INVALID_PREFLIGHT_METADATA",
                f"stored {split} metadata fields are invalid",
                stage="resume.metadata.preflight",
                config_field=f"data.{split}",
                split=split,
            )
        _require_equal(
            f"data.{split}",
            stored_split,
            expected,
            reason="DATA_PREFLIGHT_METADATA_MISMATCH",
        )
    baseline = _baseline_preflight(
        train_samples,
        stored.species_vocabulary,
        config.baseline,
    )
    _require_equal(
        "baseline_preflight",
        preflight["baseline_preflight"],
        baseline,
        reason="BASELINE_PREFLIGHT_METADATA_MISMATCH",
    )


def _validate_status(
    directory: TrainingRunDirectory,
    status: Mapping[str, Any],
    stored: _StoredResolvedRun,
    config: TrainingRunConfig,
) -> None:
    if status.get("schema_version") != "refsite_training_run_status_v1":
        raise CLIError(
            "INVALID_RUN_STATUS",
            "run_status.json schema is unsupported",
            stage="resume.metadata.status",
            path=directory.status_path,
        )
    for field, expected in (
        ("config_fingerprint", stored.config_fingerprint),
        ("bundle_fingerprint", stored.bundle_fingerprint),
        ("train_semantic_digest", stored.train_semantic_digest),
        ("validation_semantic_digest", stored.validation_semantic_digest),
        ("seed", config.runtime.seed),
    ):
        _require_equal(
            f"run_status.{field}",
            status.get(field),
            expected,
            reason="RUN_STATUS_IDENTITY_MISMATCH",
        )


def _resume_preflight_report(
    directory: TrainingRunDirectory,
    config: TrainingRunConfig,
    stored: _StoredResolvedRun,
    checkpoint: TrainingCheckpoint,
    previous_epochs: tuple[int, ...],
    requested_max_epochs: int,
) -> dict[str, Any]:
    saved_fit = FitConfig.from_dict(
        checkpoint.metadata.resolved_configuration["fit"]
    )
    return json.loads(
        canonical_runtime_json(
            {
                "schema_version": RESUME_PREFLIGHT_SCHEMA_VERSION,
                "status": "resume_preflight_ready",
                "training_executed": False,
                "mutation_performed": False,
                "run_directory": str(directory.root),
                "path_kind": "runtime_location_not_semantic_fingerprint",
                "config_fingerprint": stored.config_fingerprint,
                "bundle_fingerprint": stored.bundle_fingerprint,
                "train_semantic_digest": stored.train_semantic_digest,
                "validation_semantic_digest": stored.validation_semantic_digest,
                "seed": config.runtime.seed,
                "runtime": {
                    "device": stored.resolved_device,
                    "dtype": stored.resolved_dtype,
                    "solver_path": TRAIN_FIXED,
                },
                "checkpoint": {
                    "source": str(directory.checkpoints / "latest.pt"),
                    "scope": checkpoint.checkpoint_scope,
                    "completed_epochs": checkpoint.progress.completed_epochs,
                    "next_epoch": checkpoint.progress.next_epoch,
                    "global_step": checkpoint.progress.global_step,
                    "max_epochs": saved_fit.max_epochs,
                    "best_epoch": checkpoint.selection_state.best_epoch,
                    "best_global_step": checkpoint.selection_state.best_global_step,
                    "managed_epochs": list(previous_epochs),
                },
                "requested_max_epochs": requested_max_epochs,
                "continuation_epoch_count": (
                    requested_max_epochs - checkpoint.progress.next_epoch
                ),
                "train_batch_count": stored.train_batch_count,
                "validation_batch_count": stored.validation_batch_count,
                "template_ids": sorted(checkpoint.metadata.template_fingerprints),
                "resume_policy": ResumePolicy().to_dict(),
                "exact_rng_restore_required": True,
                "lock_state": "available_not_acquired",
                "message": "no training was executed",
            }
        )
    )


def _prepare_resume(
    run_directory: str | os.PathLike[str],
    *,
    max_epochs: int,
) -> _ResumePreflight:
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, Integral):
        raise CLIError(
            "INVALID_MAX_EPOCHS",
            "max_epochs must be a positive integer",
            stage="resume.arguments",
            path=run_directory,
            config_field="max_epochs",
        )
    max_epochs = int(max_epochs)
    if max_epochs <= 0:
        raise CLIError(
            "INVALID_MAX_EPOCHS",
            "max_epochs must be a positive integer",
            stage="resume.arguments",
            path=run_directory,
            config_field="max_epochs",
        )
    try:
        directory = TrainingRunDirectory.open_existing(run_directory)
        directory.validate_resume_lock_available()
        config = _load_config(directory)
        preflight = load_runtime_json(
            directory.preflight_path, stage="resume.metadata.preflight"
        )
        status = load_runtime_json(
            directory.status_path, stage="resume.metadata.status"
        )
        stored, train_paths, validation_paths, bundle_path = (
            _validate_preflight_metadata(directory, config, preflight)
        )
        checkpoints_path = directory.checkpoints
        if checkpoints_path.is_symlink():
            raise CLIError(
                "CHECKPOINT_DIRECTORY_SYMLINK_REJECTED",
                "managed checkpoints directory must not be a symbolic link",
                stage="resume.checkpoint.path",
                path=checkpoints_path,
            )
        if not checkpoints_path.exists() or not checkpoints_path.is_dir():
            raise CLIError(
                "CHECKPOINT_DIRECTORY_INVALID",
                "managed checkpoints directory is missing or not a directory",
                stage="resume.checkpoint.path",
                path=checkpoints_path,
            )
        if checkpoints_path.resolve(strict=True).parent != directory.root:
            raise CLIError(
                "CHECKPOINT_DIRECTORY_ESCAPE",
                "managed checkpoints directory resolves outside the run directory",
                stage="resume.checkpoint.path",
                path=checkpoints_path,
            )
        manager = CheckpointManager(
            CheckpointManagerConfig(directory=str(directory.checkpoints))
        )
        if max_epochs - 1 >= 10**manager.config.epoch_filename_width:
            raise CLIError(
                "RESUME_MAX_EPOCHS_OUT_OF_RANGE",
                "requested max_epochs exceeds the managed epoch filename range",
                stage="resume.checkpoint.compatibility",
                path=directory.root,
                config_field="max_epochs",
            )
        try:
            checkpoint = manager.load_latest()
        except Exception as error:
            raise CLIError(
                "LATEST_CHECKPOINT_LOAD_FAILED",
                "only checkpoints/latest.pt is accepted and it failed safe loading",
                stage="resume.checkpoint.load",
                path=directory.checkpoints / "latest.pt",
                original_error=error,
            ) from error
        try:
            validate_checkpoint_history(checkpoint)
            previous_epochs = validate_managed_checkpoint_history(
                manager, checkpoint
            )
        except Exception as error:
            raise CLIError(
                "CHECKPOINT_HISTORY_INVALID",
                "latest/epoch/best checkpoint history is not a contiguous exact history",
                stage="resume.checkpoint.history",
                path=directory.checkpoints / "latest.pt",
                epoch_index=checkpoint.progress.last_completed_epoch,
                global_step=checkpoint.progress.global_step,
                original_error=error,
            ) from error
        saved_fit = FitConfig.from_dict(
            checkpoint.metadata.resolved_configuration["fit"]
        )
        if max_epochs <= saved_fit.max_epochs:
            raise CLIError(
                "RESUME_MAX_EPOCHS_NOT_INCREASED",
                "--max-epochs must strictly increase the latest checkpoint value",
                stage="resume.checkpoint.compatibility",
                path=directory.root,
                config_field="max_epochs",
                epoch_index=checkpoint.progress.last_completed_epoch,
                global_step=checkpoint.progress.global_step,
            )
        _validate_status(directory, status, stored, config)
        _, registry, templates = _validate_bundle_and_checkpoint(
            config, stored, preflight, checkpoint, bundle_path
        )
        _validate_data(
            config,
            stored,
            preflight,
            checkpoint,
            registry,
            templates,
            train_paths,
            validation_paths,
        )
        report = _resume_preflight_report(
            directory,
            config,
            stored,
            checkpoint,
            previous_epochs,
            max_epochs,
        )
        return _ResumePreflight(
            directory=directory,
            config=config,
            resolved=stored,
            checkpoint=checkpoint,
            manager=manager,
            previous_epochs=previous_epochs,
            report=report,
            stored_preflight=preflight,
            stored_status=status,
        )
    except CLIError:
        raise
    except RunDirectoryError as error:
        raise CLIError(
            error.reason_code,
            "training run directory validation failed",
            stage=error.stage,
            path=error.path,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except Exception as error:
        raise CLIError(
            _nested_reason(error) or "RESUME_PREFLIGHT_FAILED",
            "resume preflight failed without changing the run directory",
            stage="resume.preflight",
            path=run_directory,
            underlying_reason_code=_nested_reason(error),
            original_error=error,
        ) from error


def _resume_status_base(
    preflight: _ResumePreflight,
    status: str,
    *,
    training_executed: bool,
) -> dict[str, Any]:
    checkpoint = preflight.checkpoint
    return {
        "schema_version": "refsite_training_run_status_v1",
        "result_schema_version": RESUME_RESULT_SCHEMA_VERSION,
        "status": status,
        "operation": "resume",
        "operation_phase": "resuming" if status == "running" else status,
        "path_kind": "runtime_location_not_semantic_fingerprint",
        "training_executed": training_executed,
        "config_fingerprint": preflight.resolved.config_fingerprint,
        "bundle_fingerprint": preflight.resolved.bundle_fingerprint,
        "train_semantic_digest": preflight.resolved.train_semantic_digest,
        "validation_semantic_digest": preflight.resolved.validation_semantic_digest,
        "seed": preflight.config.runtime.seed,
        "runtime": {
            "device": preflight.resolved.resolved_device,
            "dtype": preflight.resolved.resolved_dtype,
            "solver_path": TRAIN_FIXED,
        },
        "resume_source": str(preflight.directory.checkpoints / "latest.pt"),
        "resume_from_epoch": checkpoint.progress.next_epoch,
        "requested_max_epochs": preflight.report["requested_max_epochs"],
        "completed_epochs": checkpoint.progress.completed_epochs,
        "resumed_epochs_completed": 0,
        "global_step": checkpoint.progress.global_step,
        "recoverable_global_step": checkpoint.progress.global_step,
        "latest_checkpoint": str(preflight.directory.checkpoints / "latest.pt"),
        "best_checkpoint": str(preflight.directory.checkpoints / "best.pt"),
        "recoverable_checkpoint": str(
            preflight.directory.checkpoints / "latest.pt"
        ),
        "new_epoch_checkpoints": [],
        "terminal_selection_state": checkpoint.selection_state.to_dict(),
        "fit_result": None,
        "baseline": checkpoint.metadata.baseline_fit_metadata,
        "restored_rng_domains": [],
        "exact_resume": False,
        "failure_phase": None,
        "error": None,
        "rollback_performed": False,
        "rollback_succeeded": None,
        "partial_update_retained": False,
    }


def _completed_status(preflight: _ResumePreflight, result: Any) -> dict[str, Any]:
    status = _resume_status_base(
        preflight, "completed", training_executed=True
    )
    fit = result.fit_result
    checkpointed = result.checkpointed_fit_result
    status.update(
        {
            "completed_epochs": fit.epochs_completed,
            "resumed_epochs_completed": result.continuation_fit_result.epochs_completed,
            "global_step": fit.global_step_end,
            "recoverable_global_step": fit.global_step_end,
            "latest_checkpoint": checkpointed.latest_path,
            "best_checkpoint": checkpointed.best_path,
            "recoverable_checkpoint": checkpointed.latest_path,
            "new_epoch_checkpoints": list(checkpointed.epoch_paths),
            "terminal_selection_state": fit.final_selection_state.to_dict(),
            "fit_result": fit.to_dict(),
            "restored_rng_domains": list(result.resume_state.restored_rng_domains),
            "exact_resume": result.resume_state.exact_resume_ready,
        }
    )
    return json.loads(canonical_runtime_json(status))


def _recoverable(preflight: _ResumePreflight) -> TrainingCheckpoint:
    try:
        return preflight.manager.load_latest()
    except Exception:
        return preflight.checkpoint


def _failure_status(
    preflight: _ResumePreflight,
    error: BaseException,
    *,
    phase: str,
    prepared: _PreparedTrainingRuntime | None,
    interrupted: bool,
    training_executed: bool,
) -> dict[str, Any]:
    recoverable = _recoverable(preflight)
    status = _resume_status_base(
        preflight,
        "interrupted" if interrupted else "failed",
        training_executed=training_executed,
    )
    failure_phase = phase
    if isinstance(error, FitExecutionError):
        failure_phase = error.phase
    elif isinstance(error, CheckpointedFitExecutionError):
        failure_phase = f"checkpoint.{error.failure_stage}"
    elif isinstance(error, CheckpointRestoreError):
        failure_phase = f"restore.{error.stage}"
    elif isinstance(error, CLIError):
        failure_phase = error.failure_phase or error.stage.removeprefix("resume.")
    epoch = getattr(error, "epoch_index", None)
    if epoch is None:
        epoch = getattr(error, "failure_epoch", None)
    current_step = getattr(error, "current_global_step", None)
    if current_step is None:
        current_step = getattr(error, "global_step", recoverable.progress.global_step)
    batch_index, sample_id, template_id = _batch_context(
        error, prepared, failure_phase
    )
    sample_id = _nested_text_attribute(error, "sample_id") or sample_id
    template_id = _nested_text_attribute(error, "template_id") or template_id
    try:
        current_epochs = preflight.manager.list_epochs()
    except Exception:
        current_epochs = preflight.previous_epochs
    new_epochs = tuple(
        epoch_value
        for epoch_value in current_epochs
        if epoch_value not in set(preflight.previous_epochs)
    )
    reason = _nested_reason(error)
    original_type = getattr(error, "original_exception_type", None) or type(error).__name__
    original_message = getattr(error, "original_exception_message", None) or str(error)
    restore_rollback = isinstance(error, CheckpointRestoreError)
    status.update(
        {
            "completed_epochs": recoverable.progress.completed_epochs,
            "resumed_epochs_completed": max(
                0,
                recoverable.progress.completed_epochs
                - preflight.checkpoint.progress.completed_epochs,
            ),
            "global_step": int(current_step),
            "recoverable_global_step": recoverable.progress.global_step,
            "latest_checkpoint": str(
                preflight.directory.checkpoints / "latest.pt"
            ),
            "best_checkpoint": str(preflight.directory.checkpoints / "best.pt"),
            "recoverable_checkpoint": str(
                preflight.directory.checkpoints / "latest.pt"
            ),
            "new_epoch_checkpoints": [
                str(
                    preflight.directory.checkpoints
                    / f"epoch_{value:06d}.pt"
                )
                for value in new_epochs
            ],
            "terminal_selection_state": recoverable.selection_state.to_dict(),
            "failure_phase": failure_phase,
            "rollback_performed": (
                True
                if restore_rollback
                else bool(getattr(error, "rollback_performed", False))
            ),
            "rollback_succeeded": (
                error.rollback_succeeded if restore_rollback else None
            ),
            "partial_update_retained": int(current_step)
            != recoverable.progress.global_step,
            "epoch_index": epoch,
            "batch_index": batch_index,
            "sample_id": sample_id,
            "template_id": template_id,
            "error": {
                "type": original_type,
                "message": original_message,
                "reason_code": reason,
            },
        }
    )
    return json.loads(canonical_runtime_json(status))


def _write_failure_status(
    preflight: _ResumePreflight,
    status: Mapping[str, Any],
    original_error: BaseException,
) -> BaseException:
    try:
        preflight.directory.write_status(status)
    except Exception as status_error:
        status_error.__context__ = original_error
        return status_error
    return original_error


def _execution_error(
    preflight: _ResumePreflight,
    error: BaseException,
    details: Mapping[str, Any],
) -> CLIError:
    if isinstance(error, CLIError):
        if error.stage.startswith("resume."):
            return error
        return CLIError(
            error.reason_code,
            error.message,
            stage=f"resume.{details.get('failure_phase') or 'runtime'}",
            path=preflight.directory.root,
            frame_index=error.frame_index,
            sample_id=error.sample_id or details.get("sample_id"),
            template_id=error.template_id or details.get("template_id"),
            term=error.term,
            config_field=error.config_field,
            split=error.split,
            epoch_index=(
                error.epoch_index
                if error.epoch_index is not None
                else details.get("epoch_index")
            ),
            batch_index=(
                error.batch_index
                if error.batch_index is not None
                else details.get("batch_index")
            ),
            global_step=(
                error.global_step
                if error.global_step is not None
                else details.get("global_step")
            ),
            failure_phase=details.get("failure_phase"),
            rollback_performed=details.get("rollback_performed"),
            solver_path=TRAIN_FIXED,
            prediction_stage=error.prediction_stage,
            predictor_reason_code=error.predictor_reason_code,
            underlying_reason_code=(
                error.underlying_reason_code or error.reason_code
            ),
            original_error=error,
        )
    reason = _nested_reason(error) or "RESUME_EXECUTION_FAILED"
    prediction_stage = _nested_text_attribute(error, "stage")
    return CLIError(
        reason,
        "resume failed; completed checkpoints and partial updates were not deleted",
        stage=f"resume.{details.get('failure_phase') or 'runtime'}",
        path=preflight.directory.root,
        sample_id=details.get("sample_id"),
        template_id=details.get("template_id"),
        epoch_index=details.get("epoch_index"),
        batch_index=details.get("batch_index"),
        global_step=details.get("global_step"),
        failure_phase=details.get("failure_phase"),
        rollback_performed=details.get("rollback_performed"),
        solver_path=TRAIN_FIXED,
        prediction_stage=prediction_stage,
        predictor_reason_code=reason if prediction_stage is not None else None,
        underlying_reason_code=reason,
        original_error=error,
    )


def _resolved_checkpoint_configs(
    preflight: _ResumePreflight,
    prepared: _PreparedTrainingRuntime,
    max_epochs: int,
) -> dict[str, Any]:
    config = preflight.config
    saved_fit = FitConfig.from_dict(
        preflight.checkpoint.metadata.resolved_configuration["fit"]
    )
    return {
        "model": prepared.loaded.model.config,
        "loss": config.loss,
        "optimizer": config.optimizer,
        "train_step": config.train_step,
        "validation_step": config.validation_step,
        "scheduler": config.scheduler,
        "model_selection": config.selection,
        "fit": FitConfig(
            max_epochs=max_epochs,
            start_epoch=saved_fit.start_epoch,
            global_step_start=saved_fit.global_step_start,
        ),
        "species_vocabulary": preflight.resolved.species_vocabulary,
        "unit_conventions": _unit_conventions(),
        "baseline_fit_metadata": (
            preflight.checkpoint.metadata.baseline_fit_metadata
        ),
    }


def _execute_resume(
    preflight: _ResumePreflight,
    *,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    max_epochs = int(preflight.report["requested_max_epochs"])
    phase = "checkpoint.toctou"
    prepared: _PreparedTrainingRuntime | None = None
    training_executed = False
    try:
        current_config = _load_config(preflight.directory)
        if not _canonical_equal(
            current_config.to_dict(), preflight.config.to_dict()
        ):
            raise CLIError(
                "RESOLVED_CONFIG_TOCTOU_MISMATCH",
                "resolved_config.json changed between preflight and lock acquisition",
                stage="resume.config.toctou",
                path=preflight.directory.resolved_config_path,
            )
        current_preflight = load_runtime_json(
            preflight.directory.preflight_path,
            stage="resume.metadata.preflight_toctou",
        )
        if not _canonical_equal(current_preflight, preflight.stored_preflight):
            raise CLIError(
                "PREFLIGHT_TOCTOU_MISMATCH",
                "preflight.json changed between preflight and lock acquisition",
                stage="resume.preflight.toctou",
                path=preflight.directory.preflight_path,
            )
        current = preflight.manager.load_latest()
        if not _tree_equal(current.to_dict(), preflight.checkpoint.to_dict()):
            raise CLIError(
                "LATEST_CHECKPOINT_TOCTOU_MISMATCH",
                "latest checkpoint changed between preflight and lock acquisition",
                stage="resume.checkpoint.toctou",
                path=preflight.directory.checkpoints / "latest.pt",
            )
        phase = "runtime.seed"
        seed_training_runtime(preflight.config.runtime.seed)
        phase = "metadata.running_status"
        preflight.directory.write_status(
            _resume_status_base(
                preflight, "running", training_executed=False
            )
        )
        phase = "runtime.instantiate"
        prepared = _prepare_training_runtime(
            preflight.config, preflight.resolved
        )
        phase = "optimizer.create"
        optimizer = build_optimizer(
            prepared.loaded.model, preflight.config.optimizer
        )
        phase = "scheduler.create"
        scheduler = build_scheduler(optimizer, preflight.config.scheduler)
        resolved_configs = _resolved_checkpoint_configs(
            preflight, prepared, max_epochs
        )
        if progress is not None:
            progress(
                "resume started: "
                f"next_epoch={preflight.checkpoint.progress.next_epoch}, "
                f"max_epochs={max_epochs}, "
                f"global_step={preflight.checkpoint.progress.global_step}"
            )
        phase = "fit"
        training_executed = True
        checkpoint_contexts = {
            template_id: prepared.loaded.template_contexts[template_id]
            for template_id in preflight.checkpoint.metadata.template_fingerprints
        }
        result = run_checkpointed_resumed_fit(
            preflight.checkpoint,
            prepared.loaded.model,
            optimizer,
            scheduler,
            prepared.train_batches,
            prepared.validation_batches,
            checkpoint_contexts,
            preflight.config.loss,
            preflight.config.train_step,
            preflight.config.validation_step,
            preflight.config.scheduler,
            preflight.config.selection,
            resolved_configs,
            preflight.manager,
            preflight.config.checkpointed_fit,
            resumed_max_epochs=max_epochs,
            policy=ResumePolicy(),
        )
        status = _completed_status(preflight, result)
        phase = "metadata.completed_status"
        preflight.directory.write_status(status)
        if progress is not None:
            progress(
                "resume completed: "
                f"epochs={result.fit_result.epochs_completed}, "
                f"global_step={result.fit_result.global_step_end}"
            )
        return status
    except KeyboardInterrupt as error:
        status = _failure_status(
            preflight,
            error,
            phase=phase,
            prepared=prepared,
            interrupted=True,
            training_executed=training_executed,
        )
        stored_error = _write_failure_status(preflight, status, error)
        raise CLIInterruptedError(
            "RESUME_INTERRUPTED",
            "resume was interrupted; the recoverable latest checkpoint was retained",
            stage=f"resume.{status['failure_phase']}",
            path=preflight.directory.root,
            sample_id=status.get("sample_id"),
            template_id=status.get("template_id"),
            epoch_index=status.get("epoch_index"),
            batch_index=status.get("batch_index"),
            global_step=status.get("global_step"),
            failure_phase=status.get("failure_phase"),
            rollback_performed=False,
            solver_path=TRAIN_FIXED,
            underlying_reason_code="KEYBOARD_INTERRUPT",
            original_error=stored_error,
        ) from error
    except Exception as error:
        status = _failure_status(
            preflight,
            error,
            phase=phase,
            prepared=prepared,
            interrupted=False,
            training_executed=training_executed,
        )
        stored_error = _write_failure_status(preflight, status, error)
        raise _execution_error(preflight, stored_error, status) from error


def resume_training(
    run_directory: str | os.PathLike[str],
    *,
    max_epochs: int,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Preflight and optionally resume exactly from checkpoints/latest.pt."""

    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    preflight = _prepare_resume(run_directory, max_epochs=max_epochs)
    if dry_run:
        return dict(preflight.report)
    try:
        lock = preflight.directory.acquire_resume_lock()
    except RunDirectoryError as error:
        raise CLIError(
            error.reason_code,
            "resume lock could not be acquired",
            stage=error.stage,
            path=error.path,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error
    try:
        with lock:
            return _execute_resume(preflight, progress=progress)
    except (CLIError, CLIInterruptedError):
        raise
    except RunDirectoryError as error:
        raise CLIError(
            error.reason_code,
            "resume lock operation failed; a foreign lock was not removed",
            stage=error.stage,
            path=error.path,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error


def render_resume_json(report: Mapping[str, Any]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("resume report must be a mapping")
    return canonical_runtime_json(report)


def render_resume_human(report: Mapping[str, Any]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("resume report must be a mapping")
    status = report.get("status")
    if status == "resume_preflight_ready":
        checkpoint = report["checkpoint"]
        return "\n".join(
            (
                "Reference-site MLIP training resume preflight",
                "Status: ready",
                f"Run directory: {report['run_directory']}",
                f"Config SHA-256: {report['config_fingerprint']}",
                f"Bundle SHA-256: {report['bundle_fingerprint']}",
                f"Checkpoint next epoch: {checkpoint['next_epoch']}",
                f"Checkpoint global step: {checkpoint['global_step']}",
                f"Requested max epochs: {report['requested_max_epochs']}",
                f"Runtime: {report['runtime']['device']} / {report['runtime']['dtype']}",
                f"Seed: {report['seed']}",
                "No training was executed and no run-directory file was changed.",
            )
        )
    if status != "completed":
        raise ValueError("human resume summary requires ready or completed status")
    fit = report["fit_result"]
    return "\n".join(
        (
            "Reference-site MLIP resumed training run",
            "Status: completed",
            f"Config SHA-256: {report['config_fingerprint']}",
            f"Bundle SHA-256: {report['bundle_fingerprint']}",
            f"Resume from epoch: {report['resume_from_epoch']}",
            f"Requested max epochs: {report['requested_max_epochs']}",
            f"Resumed epochs completed: {report['resumed_epochs_completed']}",
            f"Total epochs completed: {report['completed_epochs']}",
            f"Global step: {report['global_step']}",
            f"Best epoch: {fit['best_epoch']}",
            f"Exact RNG restore: {'yes' if report['exact_resume'] else 'no'}",
            f"Latest checkpoint: {report['latest_checkpoint']}",
            f"Best checkpoint: {report['best_checkpoint']}",
            "No portable prediction bundle was exported.",
        )
    )


# Command-oriented alias.
resume_from_run_directory = resume_training


__all__ = [
    "RESUME_PREFLIGHT_SCHEMA_VERSION",
    "RESUME_RESULT_SCHEMA_VERSION",
    "render_resume_human",
    "render_resume_json",
    "resume_from_run_directory",
    "resume_training",
]
