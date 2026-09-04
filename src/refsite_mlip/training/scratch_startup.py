"""Scratch training startup stopped immediately before the first update.

The orchestration in this module deliberately has a narrow boundary.  It
creates and verifies the durable initial bundle, materializes the runtime only
from that saved bundle, prepares the optional atomic baseline and deterministic
batches, and constructs fresh optimizer/controller state.  It never evaluates
the model and never writes an epoch checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import random
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from refsite_mlip.data import (
    StructureBatch,
    StructureSample,
    TemplateRegistry,
    collate_structure_samples,
)
from refsite_mlip.models import (
    EvaluationPolicy,
    PotentialConfig,
    ReferenceSiteModelBundle,
    ReferenceSitePotential,
    TemplateExecutionContext,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
    save_reference_site_model_bundle,
)

from .baseline import AtomicBaselineFit, apply_atomic_baseline_, fit_atomic_baseline
from ._scratch_run_metadata import scratch_runtime_preflight_metadata
from .checkpoint import FitProgress
from .optimizer import build_optimizer, optimizer_parameters, validate_optimizer_binding
from .run_directory import (
    ResumeRunLock,
    TrainingRunDirectory,
)
from .scheduler import build_scheduler
from .scratch_initialization import (
    ScratchModelInitialization,
    _validate_preparation,
    initialize_scratch_model,
)
from .scratch_preparation import (
    ScratchTrainingPreparation,
    prepare_scratch_training_run,
    verify_scratch_preparation_input_digests,
)
from .selection import ModelSelectionState

if TYPE_CHECKING:
    from refsite_mlip.config import TrainingRunConfig


SCRATCH_TRAINING_STARTUP_CONVENTION_VERSION = "scratch_training_startup_v1"
SCRATCH_TRAINING_STARTUP_STATUS_SCHEMA_VERSION = (
    "refsite_scratch_training_startup_status_v1"
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("startup metadata mapping keys must be strings")
        return {key: _plain(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("startup metadata must not contain NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"startup metadata contains non-plain {type(value).__name__}")


def _freeze_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_plain(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_plain(item) for item in value)
    if type(value) is float and not math.isfinite(value):
        raise ValueError("startup metadata must not contain NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"startup metadata contains non-plain {type(value).__name__}")


def _sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _nested_attribute(error: BaseException, name: str) -> Any:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, name, None)
        if value is not None:
            return value
        current = current.__cause__ or current.__context__
    return None


class ScratchTrainingStartupError(RuntimeError):
    """Structured failure from one pre-update scratch startup transaction."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        output_path: str | None = None,
        initialization_seed: int | None = None,
        training_seed: int | None = None,
        template_id: str | None = None,
        sample_id: str | None = None,
        config_fingerprint: str | None = None,
        bundle_fingerprint: str | None = None,
        original_reason_code: str | None = None,
        original_error: BaseException | None = None,
        recoverable_initial_bundle: str | None = None,
        status_write_error: BaseException | None = None,
    ) -> None:
        if type(reason_code) is not str or not reason_code:
            raise ValueError("reason_code must be a nonempty string")
        if type(message) is not str or not message:
            raise ValueError("message must be a nonempty string")
        if type(stage) is not str or not stage:
            raise ValueError("stage must be a nonempty string")
        self.reason_code = reason_code
        self.message = message
        self.stage = stage
        self.output_path = output_path
        self.initialization_seed = initialization_seed
        self.training_seed = training_seed
        self.template_id = template_id
        self.sample_id = sample_id
        self.config_fingerprint = config_fingerprint
        self.bundle_fingerprint = bundle_fingerprint
        self.original_reason_code = original_reason_code
        self.original_error = original_error
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        self.recoverable_initial_bundle = recoverable_initial_bundle
        self.first_optimizer_update_executed = False
        self.status_write_exception_type = (
            None if status_write_error is None else type(status_write_error).__name__
        )
        self.status_write_exception_message = (
            None if status_write_error is None else str(status_write_error)
        )
        context = []
        for name in (
            "output_path",
            "template_id",
            "sample_id",
            "config_fingerprint",
            "bundle_fingerprint",
            "original_reason_code",
            "original_exception_type",
            "original_exception_message",
        ):
            value = getattr(self, name)
            if value is not None:
                context.append(f"{name}={value!r}")
        suffix = "" if not context else " " + " ".join(context)
        super().__init__(f"[{reason_code}] stage={stage!r}{suffix} {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_fingerprint": self.bundle_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "first_optimizer_update_executed": False,
            "initialization_seed": self.initialization_seed,
            "message": self.message,
            "original_exception_message": self.original_exception_message,
            "original_exception_type": self.original_exception_type,
            "original_reason_code": self.original_reason_code,
            "output_path": self.output_path,
            "reason_code": self.reason_code,
            "recoverable_initial_bundle": self.recoverable_initial_bundle,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "status_write_exception_message": self.status_write_exception_message,
            "status_write_exception_type": self.status_write_exception_type,
            "template_id": self.template_id,
            "training_seed": self.training_seed,
        }


@dataclass(frozen=True)
class _ProcessState:
    python_rng: object
    numpy_rng: tuple[Any, ...]
    torch_cpu_rng: torch.Tensor
    torch_cuda_rng: tuple[torch.Tensor, ...] | None
    default_dtype: torch.dtype
    grad_enabled: bool
    deterministic_algorithms: bool


def _capture_process_state() -> _ProcessState:
    numpy_state = np.random.get_state()
    numpy_snapshot = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = tuple(value.clone() for value in torch.cuda.get_rng_state_all())
    return _ProcessState(
        python_rng=random.getstate(),
        numpy_rng=numpy_snapshot,
        torch_cpu_rng=torch.get_rng_state().clone(),
        torch_cuda_rng=cuda_state,
        default_dtype=torch.get_default_dtype(),
        grad_enabled=torch.is_grad_enabled(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def _restore_process_state(state: _ProcessState) -> None:
    random.setstate(state.python_rng)
    np.random.set_state(state.numpy_rng)
    torch.set_rng_state(state.torch_cpu_rng)
    if state.torch_cuda_rng is not None:
        torch.cuda.set_rng_state_all(list(state.torch_cuda_rng))
    if torch.get_default_dtype() != state.default_dtype:
        torch.set_default_dtype(state.default_dtype)
    if torch.is_grad_enabled() != state.grad_enabled:
        torch.set_grad_enabled(state.grad_enabled)
    if torch.are_deterministic_algorithms_enabled() != state.deterministic_algorithms:
        torch.use_deterministic_algorithms(state.deterministic_algorithms)


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _process_state_equal(left: _ProcessState, right: _ProcessState) -> bool:
    cuda_equal = (left.torch_cuda_rng is None) == (right.torch_cuda_rng is None)
    if cuda_equal and left.torch_cuda_rng is not None:
        assert right.torch_cuda_rng is not None
        cuda_equal = len(left.torch_cuda_rng) == len(right.torch_cuda_rng) and all(
            torch.equal(a, b)
            for a, b in zip(left.torch_cuda_rng, right.torch_cuda_rng)
        )
    return (
        left.python_rng == right.python_rng
        and _numpy_rng_equal(left.numpy_rng, right.numpy_rng)
        and torch.equal(left.torch_cpu_rng, right.torch_cpu_rng)
        and cuda_equal
        and left.default_dtype == right.default_dtype
        and left.grad_enabled == right.grad_enabled
        and left.deterministic_algorithms == right.deterministic_algorithms
    )


def _seed_training_runtime(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("training seed must be an integer")
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value % (2**64))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value % (2**64))


def _state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _state_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return tuple(left) == tuple(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


def _live_split_digest(
    preparation: ScratchTrainingPreparation,
    samples: tuple[StructureSample, ...],
    *,
    split: str,
) -> str:
    # Reuse the established semantic digest rather than defining another data
    # identity contract here.
    from refsite_mlip.config.training_run import _split_digest

    templates = {
        template_id: preparation.registry.resolve(template_id)
        for template_id in sorted(preparation.template_contexts)
    }
    return _split_digest(samples, templates, split=split)


def _revalidate_preparation(
    preparation: ScratchTrainingPreparation,
    *,
    refresh_from_files: bool = True,
) -> ScratchTrainingPreparation:
    from refsite_mlip.config import (
        ScratchModelSourceConfig,
        TrainingRunConfig,
        validate_training_run_config,
    )

    if not isinstance(preparation, ScratchTrainingPreparation):
        raise TypeError("preparation must be a ScratchTrainingPreparation")
    config = preparation.config
    if not isinstance(config, TrainingRunConfig):
        raise TypeError("preparation.config must be a TrainingRunConfig")
    validate_training_run_config(config)
    if not isinstance(config.model_source, ScratchModelSourceConfig):
        raise ValueError("scratch startup requires a scratch model source")
    if config.config_fingerprint != preparation.config_fingerprint:
        raise ValueError("preparation config fingerprint is stale")
    # Validate the caller-owned artifact/context/policy snapshot itself before
    # replacing it with a fresh file-backed preparation.  Otherwise a mutated
    # runtime tensor could be silently ignored instead of being rejected.
    _validate_preparation(preparation)
    if _live_split_digest(
        preparation, preparation.train_samples, split="train"
    ) != preparation.train_semantic_digest:
        raise ValueError("prepared train sample content changed after preflight")
    if _live_split_digest(
        preparation, preparation.validation_samples, split="validation"
    ) != preparation.validation_semantic_digest:
        raise ValueError("prepared validation sample content changed after preflight")
    verify_scratch_preparation_input_digests(preparation)

    # An outer fresh-run orchestration may already own the configured output
    # directory and its lock.  Re-running the full preparation in that case
    # would reject the owned directory as an output collision.  The checks
    # above still validate the complete caller-owned preparation, semantic
    # sample digests, and every raw config/POSCAR/extxyz byte digest.
    if not refresh_from_files:
        return preparation

    base_directory = preparation.runtime_paths.get("base_directory")
    if type(base_directory) is not str or not base_directory:
        raise ValueError("prepared runtime base_directory is missing or invalid")
    refreshed = prepare_scratch_training_run(
        config, base_directory=base_directory
    )
    comparisons = (
        (
            "config fingerprint",
            refreshed.config_fingerprint,
            preparation.config_fingerprint,
        ),
        (
            "preparation fingerprint",
            refreshed.preparation_fingerprint,
            preparation.preparation_fingerprint,
        ),
        (
            "train semantic digest",
            refreshed.train_semantic_digest,
            preparation.train_semantic_digest,
        ),
        (
            "validation semantic digest",
            refreshed.validation_semantic_digest,
            preparation.validation_semantic_digest,
        ),
        (
            "data manifest",
            _plain(refreshed.data_manifest),
            _plain(preparation.data_manifest),
        ),
        (
            "template fingerprints",
            _plain(refreshed.template_fingerprints),
            _plain(preparation.template_fingerprints),
        ),
        (
            "runtime paths",
            _plain(refreshed.runtime_paths),
            _plain(preparation.runtime_paths),
        ),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(f"scratch {name} changed after preflight")
    return refreshed


def _validate_supplied_run_ownership(
    preparation: ScratchTrainingPreparation,
    directory: TrainingRunDirectory,
    run_lock: ResumeRunLock,
) -> None:
    if not isinstance(directory, TrainingRunDirectory):
        raise TypeError("run_directory must be a TrainingRunDirectory")
    if not isinstance(run_lock, ResumeRunLock):
        raise TypeError("run_lock must be a ResumeRunLock")
    output_value = preparation.runtime_paths.get("output_directory")
    if type(output_value) is not str or not output_value:
        raise ValueError("prepared output_directory runtime path is invalid")
    configured_output = Path(output_value)
    if not configured_output.is_absolute():
        raise ValueError("prepared output_directory runtime path must be absolute")
    try:
        resolved_output = configured_output.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            "configured scratch output directory is unavailable"
        ) from error
    if resolved_output != directory.root:
        raise ValueError(
            "supplied run directory differs from the configured output directory"
        )
    run_lock.validate_owned(directory.resume_lock_path)


def _validate_runtime_bundle(
    preparation: ScratchTrainingPreparation,
    initialization: ScratchModelInitialization,
    bundle: ReferenceSiteModelBundle,
) -> None:
    from refsite_mlip.config.radii import (
        validate_radius_artifact_compatibility,
        validate_radius_model_compatibility,
    )

    if bundle.bundle_fingerprint != initialization.bundle_fingerprint:
        raise ValueError("saved bundle fingerprint differs from initialization")
    if bundle.architecture_fingerprint != initialization.architecture_fingerprint:
        raise ValueError("saved bundle architecture fingerprint differs")
    if tuple(bundle.species_vocabulary) != preparation.species_vocabulary:
        raise ValueError("saved bundle species vocabulary differs from preparation")
    if bundle.default_template_id != preparation.model_source.default_template_id:
        raise ValueError("saved bundle default template differs from preparation")
    if bundle.binding_ids != tuple(sorted(preparation.template_contexts)):
        raise ValueError("saved bundle template IDs differ from preparation")
    potential = PotentialConfig.from_dict(bundle.model_config)
    if potential.to_dict() != preparation.model_source.potential.to_dict():
        raise ValueError("saved bundle PotentialConfig differs from preparation")
    validate_radius_model_compatibility(preparation.radius_config, potential)
    for binding in bundle.template_bindings:
        validate_radius_artifact_compatibility(
            preparation.radius_config, binding.structural_artifact
        )
        expected = preparation.template_fingerprints[binding.template_id]
        if (
            binding.full_template_fingerprint
            != expected["full_template_fingerprint"]
            or binding.structural_artifact.structural_fingerprint
            != expected["structural_artifact_fingerprint"]
            or (
                None
                if binding.evaluation_policy is None
                else binding.evaluation_policy.content_fingerprint
            )
            != expected["evaluation_policy_fingerprint"]
        ):
            raise ValueError(
                f"saved bundle template binding differs for {binding.template_id!r}"
            )
    baseline = bundle.model_state.get("atomic_baseline")
    if baseline is None or int(torch.count_nonzero(baseline)) != 0:
        raise ValueError("saved initial bundle baseline is not exact zero")


def _validate_loaded_runtime(
    preparation: ScratchTrainingPreparation,
    bundle: ReferenceSiteModelBundle,
    loaded: Any,
) -> None:
    model = loaded.model
    if not isinstance(model, ReferenceSitePotential):
        raise TypeError("bundle runtime model has an unexpected type")
    if loaded.bundle_fingerprint != bundle.bundle_fingerprint:
        raise ValueError("runtime bundle fingerprint differs")
    if loaded.default_template_id != bundle.default_template_id:
        raise ValueError("runtime default template differs")
    if tuple(sorted(loaded.template_contexts)) != bundle.binding_ids:
        raise ValueError("runtime template contexts differ")
    if tuple(sorted(loaded.evaluation_policies)) != tuple(
        sorted(
            binding.template_id
            for binding in bundle.template_bindings
            if binding.evaluation_policy is not None
        )
    ):
        raise ValueError("runtime evaluation policies differ")
    if model.config.to_dict() != preparation.model_source.potential.to_dict():
        raise ValueError("runtime PotentialConfig differs")
    target_device = torch.device(preparation.resolved_device)
    target_dtype = preparation.runtime.torch_dtype
    for name, value in model.state_dict().items():
        device_matches = (
            value.device.type == "cuda"
            if target_device.type == "cuda" and target_device.index is None
            else value.device == target_device
        )
        if not device_matches:
            raise ValueError(f"runtime state {name!r} is on the wrong device")
        if value.is_floating_point() and value.dtype != target_dtype:
            raise ValueError(f"runtime state {name!r} has the wrong dtype")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.all(torch.isfinite(value))
        ):
            raise ValueError(f"runtime state {name!r} is nonfinite")
        stored = bundle.model_state[name].to(device=target_device, dtype=value.dtype)
        if not torch.equal(value, stored):
            raise ValueError(f"runtime pre-baseline state differs at {name!r}")
    if model.training:
        raise ValueError("runtime model must remain in eval mode before training")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("runtime model parameters must have no gradients")


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _fit_baseline(
    preparation: ScratchTrainingPreparation,
    model: ReferenceSitePotential,
    bundle_fingerprint: str,
) -> tuple[AtomicBaselineFit | None, dict[str, Any]]:
    config = preparation.config
    baseline = model.atomic_baseline
    baseline_identity = id(baseline)
    state_before = _state_snapshot(model)
    metadata: dict[str, Any] = {
        "enabled": config.baseline is not None,
        "parameter_update_applied": False,
        "seed": config.runtime.seed,
        "training_run_config_fingerprint": preparation.config_fingerprint,
        "initial_bundle_fingerprint": bundle_fingerprint,
    }
    if config.baseline is None:
        metadata["reason"] = "baseline config is null"
        if int(torch.count_nonzero(baseline)) != 0:
            raise ValueError("disabled baseline must leave the exact-zero buffer")
        return None, metadata

    fitted = fit_atomic_baseline(
        preparation.train_samples,
        range(len(preparation.train_samples)),
        preparation.species_vocabulary,
        config.baseline,
    )
    preflight = preparation.baseline_preflight
    expected = {
        "num_valid_energy_structures": fitted.num_valid_energy_structures,
        "rank": fitted.rank,
        "rank_deficient": fitted.rank_deficient,
        "rank_policy": fitted.config.rank_policy,
        "species_occurrence_counts": fitted.species_occurrence_counts.tolist(),
        "residual_rmse": fitted.residual_rmse,
        "residual_mae": fitted.residual_mae,
        "weighted_objective": fitted.weighted_objective,
    }
    for key, value in expected.items():
        if _plain(preflight.get(key)) != _plain(value):
            raise ValueError(f"runtime baseline fit differs from preflight at {key!r}")
    cast = fitted.baseline_energies.to(
        device=baseline.device, dtype=baseline.dtype
    )
    if not bool(torch.all(torch.isfinite(cast))):
        raise ValueError("atomic baseline is nonfinite in runtime dtype")
    apply_atomic_baseline_(model, fitted)
    if id(model.atomic_baseline) != baseline_identity:
        raise ValueError("atomic baseline application replaced the registered buffer")
    if model.atomic_baseline.requires_grad:
        raise ValueError("atomic baseline buffer must remain frozen")
    state_after = model.state_dict()
    if tuple(state_before) != tuple(state_after):
        raise ValueError("baseline application changed model state keys")
    for key in state_before:
        if key != "atomic_baseline" and not torch.equal(state_before[key], state_after[key]):
            raise ValueError(f"baseline application changed non-baseline state {key!r}")
    if not torch.equal(model.atomic_baseline, cast):
        raise ValueError("runtime atomic baseline differs from the fitted values")
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
    return fitted, metadata


def _build_batches(
    samples: tuple[StructureSample, ...],
    *,
    batch_size: int,
    registry: TemplateRegistry,
    device: str,
    dtype: torch.dtype,
    expected_manifest: Mapping[str, Any],
) -> tuple[StructureBatch, ...]:
    if not samples:
        raise ValueError("scratch startup refuses an empty split")
    batches = tuple(
        collate_structure_samples(samples[start : start + batch_size], registry).to(
            device=device, dtype=dtype
        )
        for start in range(0, len(samples), batch_size)
    )
    plans = expected_manifest.get("batches")
    if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes, bytearray)):
        raise TypeError("prepared batch manifest is invalid")
    if len(batches) != expected_manifest.get("batch_count") or len(plans) != len(batches):
        raise ValueError("runtime batch count differs from the preflight manifest")
    offset = 0
    flattened: list[str] = []
    for index, (batch, plan) in enumerate(zip(batches, plans)):
        stop = offset + batch.num_structures
        expected_ids = tuple(plan.get("sample_ids", ()))
        expected_templates = tuple(plan.get("template_ids", ()))
        if (
            plan.get("batch_index") != index
            or plan.get("start") != offset
            or plan.get("stop") != stop
            or batch.sample_ids != expected_ids
            or batch.template_ids != expected_templates
            or batch.sample_ids
            != tuple(sample.sample_id for sample in samples[offset:stop])
            or batch.template_ids
            != tuple(sample.template_id for sample in samples[offset:stop])
        ):
            raise ValueError(f"runtime batch {index} differs from preflight plan")
        for template_id, fingerprint in zip(
            batch.template_ids, batch.template_fingerprints
        ):
            if fingerprint != registry.resolve(template_id).fingerprint:
                raise ValueError("runtime batch template fingerprint differs")
        flattened.extend(batch.sample_ids)
        offset = stop
    expected_all = tuple(sample.sample_id for sample in samples)
    if offset != len(samples) or tuple(flattened) != expected_all:
        raise ValueError("runtime batching dropped, duplicated, or reordered samples")
    if len(set(flattened)) != len(flattened):
        raise ValueError("runtime batching contains duplicate sample IDs")
    return batches


def _paths(directory: TrainingRunDirectory) -> dict[str, str]:
    return {
        "checkpoints": str(directory.checkpoints),
        "data_manifest": str(directory.data_manifest_path),
        "initial_bundle": str(directory.initial_bundle_path),
        "preflight": str(directory.preflight_path),
        "resolved_config": str(directory.resolved_config_path),
        "root": str(directory.root),
        "run_status": str(directory.status_path),
    }


def _status(
    status: str,
    preparation: ScratchTrainingPreparation,
    *,
    bundle_fingerprint: str | None,
    baseline: Mapping[str, Any] | None,
    directory: TrainingRunDirectory,
    failure_phase: str | None = None,
    error: ScratchTrainingStartupError | None = None,
    rng_restored: bool | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCRATCH_TRAINING_STARTUP_STATUS_SCHEMA_VERSION,
        "status": status,
        "startup_convention_version": SCRATCH_TRAINING_STARTUP_CONVENTION_VERSION,
        "training_executed": False,
        "first_optimizer_update_executed": False,
        "config_fingerprint": preparation.config_fingerprint,
        "preparation_fingerprint": preparation.preparation_fingerprint,
        "bundle_fingerprint": bundle_fingerprint,
        "train_semantic_digest": preparation.train_semantic_digest,
        "validation_semantic_digest": preparation.validation_semantic_digest,
        "data_manifest_fingerprint": preparation.data_manifest["fingerprint"],
        "initialization_seed": preparation.model_source.initialization_seed,
        "seed": preparation.runtime.seed,
        "runtime": {
            "device": preparation.resolved_device,
            "dtype": preparation.resolved_dtype,
            "solver_path": preparation.config.train_step.solver_path,
        },
        "completed_epochs": 0,
        "global_step": 0,
        "latest_checkpoint": None,
        "best_checkpoint": None,
        "recoverable_checkpoint": None,
        "recoverable_initial_bundle": (
            str(directory.initial_bundle_path)
            if bundle_fingerprint is not None
            else None
        ),
        "terminal_selection_state": ModelSelectionState().to_dict(),
        "fit_progress": FitProgress(0, 0, 0).to_dict(),
        "fit_result": None,
        "baseline": None if baseline is None else _plain(baseline),
        "paths": _paths(directory),
        "failure_phase": failure_phase,
        "error": None if error is None else error.to_dict(),
        "rng_restored_to_entry": rng_restored,
        "rollback_performed": False,
    }


@dataclass(frozen=True)
class ScratchTrainingStartup:
    """Owned runtime state ready for, but preceding, the first update."""

    model: ReferenceSitePotential
    registry: TemplateRegistry
    template_contexts: Mapping[str, TemplateExecutionContext]
    evaluation_policies: Mapping[str, EvaluationPolicy]
    train_batches: tuple[StructureBatch, ...]
    validation_batches: tuple[StructureBatch, ...]
    fitted_atomic_baseline: AtomicBaselineFit | None
    baseline_metadata: Mapping[str, Any]
    optimizer: torch.optim.Optimizer
    scheduler: Any
    initial_selection_state: ModelSelectionState
    initial_fit_progress: FitProgress
    config: TrainingRunConfig
    effective_potential_config: PotentialConfig
    run_directory: TrainingRunDirectory
    run_directory_paths: Mapping[str, str]
    initial_bundle: ReferenceSiteModelBundle
    initial_bundle_fingerprint: str
    train_semantic_digest: str
    validation_semantic_digest: str
    data_manifest: Mapping[str, Any]
    initialization_seed: int
    training_seed: int
    fit_history: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model, ReferenceSitePotential):
            raise TypeError("model must be a ReferenceSitePotential")
        if not isinstance(self.registry, TemplateRegistry):
            raise TypeError("registry must be a TemplateRegistry")
        object.__setattr__(self, "train_batches", tuple(self.train_batches))
        object.__setattr__(
            self, "validation_batches", tuple(self.validation_batches)
        )
        if not self.train_batches or not self.validation_batches:
            raise ValueError("startup requires nonempty train and validation batches")
        if not isinstance(self.initial_selection_state, ModelSelectionState):
            raise TypeError("initial_selection_state must be ModelSelectionState")
        if self.initial_selection_state != ModelSelectionState():
            raise ValueError("selection state is not fresh")
        if self.initial_fit_progress != FitProgress(0, 0, 0):
            raise ValueError("fit progress is not fresh")
        validate_optimizer_binding(self.model, self.optimizer)
        if getattr(self.scheduler, "optimizer", None) is not self.optimizer:
            raise ValueError("scheduler is not bound to the startup optimizer")
        actual_parameter_ids = tuple(
            id(parameter) for parameter in optimizer_parameters(self.optimizer)
        )
        expected_parameter_ids = tuple(
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        if actual_parameter_ids != expected_parameter_ids:
            raise ValueError("optimizer parameter order differs from runtime model")
        if self.optimizer.state:
            raise ValueError("startup optimizer state must be empty")
        if self.model.atomic_baseline.requires_grad or isinstance(
            self.model.atomic_baseline, torch.nn.Parameter
        ):
            raise ValueError("atomic baseline must be a frozen buffer")
        if id(self.model.atomic_baseline) in {
            id(parameter) for parameter in optimizer_parameters(self.optimizer)
        }:
            raise ValueError("atomic baseline must be excluded from optimizer")
        if not bool(torch.all(torch.isfinite(self.model.atomic_baseline))):
            raise ValueError("runtime atomic baseline must be finite")
        _sha256(
            self.initial_bundle_fingerprint,
            name="initial_bundle_fingerprint",
        )
        if self.initial_bundle.bundle_fingerprint != self.initial_bundle_fingerprint:
            raise ValueError("initial bundle fingerprint differs")
        for name in ("train_semantic_digest", "validation_semantic_digest"):
            _sha256(getattr(self, name), name=name)
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, Integral
        ):
            raise TypeError("initialization_seed must be an integer")
        if isinstance(self.training_seed, bool) or not isinstance(
            self.training_seed, Integral
        ):
            raise TypeError("training_seed must be an integer")
        object.__setattr__(self, "initialization_seed", int(self.initialization_seed))
        object.__setattr__(self, "training_seed", int(self.training_seed))
        object.__setattr__(
            self, "template_contexts", MappingProxyType(dict(self.template_contexts))
        )
        object.__setattr__(
            self,
            "evaluation_policies",
            MappingProxyType(dict(self.evaluation_policies)),
        )
        object.__setattr__(
            self, "baseline_metadata", _freeze_plain(self.baseline_metadata)
        )
        object.__setattr__(
            self, "run_directory_paths", _freeze_plain(self.run_directory_paths)
        )
        object.__setattr__(self, "data_manifest", _freeze_plain(self.data_manifest))
        object.__setattr__(self, "fit_history", tuple(self.fit_history))
        if self.fit_history:
            raise ValueError("scratch startup history must be empty")
        if not self.run_directory.checkpoints.is_dir() or any(
            self.run_directory.checkpoints.iterdir()
        ):
            raise ValueError("startup checkpoint directory must exist and be empty")

    @property
    def baseline_fit(self) -> AtomicBaselineFit | None:
        return self.fitted_atomic_baseline

    @property
    def selection_state(self) -> ModelSelectionState:
        return self.initial_selection_state

    @property
    def fit_progress(self) -> FitProgress:
        return self.initial_fit_progress

    @property
    def history(self) -> tuple[Any, ...]:
        return self.fit_history


def _startup_error(
    phase: str,
    error: BaseException,
    preparation: ScratchTrainingPreparation | None,
    *,
    directory: TrainingRunDirectory | None,
    bundle_fingerprint: str | None,
    recoverable: bool,
    status_write_error: BaseException | None = None,
) -> ScratchTrainingStartupError:
    reason = _nested_attribute(error, "reason_code")
    if type(reason) is not str or not reason:
        reason = f"SCRATCH_STARTUP_{phase.upper()}_FAILED"
    output_path = None
    config_fingerprint = None
    initialization_seed = None
    training_seed = None
    if preparation is not None:
        output_path = str(preparation.runtime_paths.get("output_directory"))
        config_fingerprint = preparation.config_fingerprint
        initialization_seed = preparation.model_source.initialization_seed
        training_seed = preparation.runtime.seed
    if directory is not None:
        output_path = str(directory.root)
    return ScratchTrainingStartupError(
        reason,
        f"scratch training startup failed: {type(error).__name__}: {error}",
        stage=phase,
        output_path=output_path,
        initialization_seed=initialization_seed,
        training_seed=training_seed,
        template_id=_nested_attribute(error, "template_id"),
        sample_id=_nested_attribute(error, "sample_id"),
        config_fingerprint=config_fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        original_reason_code=_nested_attribute(error, "reason_code"),
        original_error=error,
        recoverable_initial_bundle=(
            str(directory.initial_bundle_path)
            if recoverable and directory is not None
            else None
        ),
        status_write_error=status_write_error,
    )


def _initialize_scratch_training_startup(
    preparation: ScratchTrainingPreparation,
    *,
    run_directory: TrainingRunDirectory | None = None,
    run_lock: ResumeRunLock | None = None,
) -> ScratchTrainingStartup:
    """Create durable scratch startup state without evaluating or updating it."""

    entry_state = _capture_process_state()
    verified: ScratchTrainingPreparation | None = None
    directory: TrainingRunDirectory | None = None
    initialization: ScratchModelInitialization | None = None
    verified_bundle: ReferenceSiteModelBundle | None = None
    initial_bundle_verified = False
    baseline_metadata: Mapping[str, Any] | None = None
    phase = "preflight"
    try:
        supplied_run = run_directory is not None or run_lock is not None
        if (run_directory is None) != (run_lock is None):
            raise ValueError(
                "run_directory and run_lock must be supplied together"
            )
        if supplied_run:
            if not isinstance(preparation, ScratchTrainingPreparation):
                raise TypeError(
                    "preparation must be a ScratchTrainingPreparation"
                )
            assert run_directory is not None and run_lock is not None
            _validate_supplied_run_ownership(
                preparation, run_directory, run_lock
            )
            # Only after exact path and active-inode ownership have been
            # established may failure reporting write into the supplied root.
            directory = run_directory
            verified = _revalidate_preparation(
                preparation, refresh_from_files=False
            )
        else:
            verified = _revalidate_preparation(preparation)
        if not _process_state_equal(entry_state, _capture_process_state()):
            raise RuntimeError("scratch preflight changed process RNG or execution state")

        phase = "initialization"
        initialization = initialize_scratch_model(verified)
        if not _process_state_equal(entry_state, _capture_process_state()):
            raise RuntimeError("scratch model initialization leaked process state")

        if supplied_run:
            phase = "run_directory_validate"
            assert run_directory is not None and run_lock is not None
            _validate_supplied_run_ownership(
                verified, run_directory, run_lock
            )
            # Close the initialization-time TOCTOU window without invoking
            # the output-collision gate against our already-owned root.
            verify_scratch_preparation_input_digests(verified)
        else:
            phase = "run_directory_create"
            output = Path(str(verified.runtime_paths["output_directory"]))
            directory = TrainingRunDirectory.create(output)
        assert directory is not None
        directory.create_checkpoints_directory()

        phase = "metadata_save"
        directory.write_resolved_config(verified.config.to_dict())
        directory.write_preflight(
            scratch_runtime_preflight_metadata(
                verified,
                initialization,
                directory,
            )
        )
        directory.write_data_manifest(_plain(verified.data_manifest))

        phase = "initial_bundle_save"
        save_reference_site_model_bundle(
            directory.initial_bundle_path,
            initialization.bundle,
            overwrite=False,
        )

        phase = "initial_bundle_reload"
        verified_bundle = load_reference_site_model_bundle(
            directory.initial_bundle_path, map_location="cpu"
        )
        _validate_runtime_bundle(verified, initialization, verified_bundle)
        initial_bundle_verified = True

        phase = "runtime_materialization"
        loaded = instantiate_reference_site_model_bundle(
            verified_bundle,
            device=verified.resolved_device,
            dtype=verified.runtime.torch_dtype,
        )
        _validate_loaded_runtime(verified, verified_bundle, loaded)

        phase = "baseline_fit"
        fitted, baseline_metadata = _fit_baseline(
            verified, loaded.model, initialization.bundle_fingerprint
        )
        state_after_baseline = _state_snapshot(loaded.model)
        parameter_ids = tuple(id(value) for value in loaded.model.parameters())
        parameter_grads = tuple(value.grad for value in loaded.model.parameters())
        model_mode = loaded.model.training

        phase = "batching"
        train_batches = _build_batches(
            verified.train_samples,
            batch_size=verified.data.batch_size,
            registry=loaded.registry,
            device=verified.resolved_device,
            dtype=verified.runtime.torch_dtype,
            expected_manifest=verified.data_manifest["train"],
        )
        validation_batches = _build_batches(
            verified.validation_samples,
            batch_size=verified.data.effective_validation_batch_size,
            registry=loaded.registry,
            device=verified.resolved_device,
            dtype=verified.runtime.torch_dtype,
            expected_manifest=verified.data_manifest["validation"],
        )

        phase = "optimizer"
        before_factory_state = _capture_process_state()
        optimizer = build_optimizer(loaded.model, verified.config.optimizer)
        validate_optimizer_binding(loaded.model, optimizer)
        if not _process_state_equal(before_factory_state, _capture_process_state()):
            raise RuntimeError("optimizer construction changed process RNG/state")
        if not _state_equal(state_after_baseline, loaded.model.state_dict()):
            raise RuntimeError("optimizer construction changed model state")
        if tuple(id(value) for value in loaded.model.parameters()) != parameter_ids:
            raise RuntimeError("optimizer construction replaced model parameters")
        if tuple(value.grad for value in loaded.model.parameters()) != parameter_grads:
            raise RuntimeError("optimizer construction changed parameter gradients")
        if loaded.model.training != model_mode:
            raise RuntimeError("optimizer construction changed model mode")

        phase = "scheduler"
        before_factory_state = _capture_process_state()
        scheduler = build_scheduler(optimizer, verified.config.scheduler)
        if getattr(scheduler, "optimizer", None) is not optimizer:
            raise RuntimeError("scheduler is bound to a different optimizer")
        if not _process_state_equal(before_factory_state, _capture_process_state()):
            raise RuntimeError("scheduler construction changed process RNG/state")
        if not _state_equal(state_after_baseline, loaded.model.state_dict()):
            raise RuntimeError("scheduler construction changed model state")
        selection = ModelSelectionState()
        progress = FitProgress(next_epoch=0, global_step=0, completed_epochs=0)

        result = ScratchTrainingStartup(
            model=loaded.model,
            registry=loaded.registry,
            template_contexts=loaded.template_contexts,
            evaluation_policies=loaded.evaluation_policies,
            train_batches=train_batches,
            validation_batches=validation_batches,
            fitted_atomic_baseline=fitted,
            baseline_metadata=baseline_metadata,
            optimizer=optimizer,
            scheduler=scheduler,
            initial_selection_state=selection,
            initial_fit_progress=progress,
            config=verified.config,
            effective_potential_config=PotentialConfig.from_dict(
                verified_bundle.model_config
            ),
            run_directory=directory,
            run_directory_paths=_paths(directory),
            initial_bundle=verified_bundle,
            initial_bundle_fingerprint=initialization.bundle_fingerprint,
            train_semantic_digest=verified.train_semantic_digest,
            validation_semantic_digest=verified.validation_semantic_digest,
            data_manifest=verified.data_manifest,
            initialization_seed=initialization.initialization_seed,
            training_seed=verified.runtime.seed,
        )

        phase = "status_save"
        if supplied_run:
            assert run_lock is not None
            run_lock.validate_owned(directory.resume_lock_path)
        directory.write_status(
            _status(
                "startup_ready",
                verified,
                bundle_fingerprint=initialization.bundle_fingerprint,
                baseline=baseline_metadata,
                directory=directory,
            )
        )
        if supplied_run:
            run_lock.validate_owned(directory.resume_lock_path)
        if not _process_state_equal(entry_state, _capture_process_state()):
            raise RuntimeError("startup changed process state before training seeding")
        _seed_training_runtime(verified.runtime.seed)
        return result
    except BaseException as original:
        _restore_process_state(entry_state)
        recoverable = initial_bundle_verified
        structured = _startup_error(
            phase,
            original,
            verified or (
                preparation
                if isinstance(preparation, ScratchTrainingPreparation)
                else None
            ),
            directory=directory,
            bundle_fingerprint=(
                None
                if initialization is None
                else initialization.bundle_fingerprint
            ),
            recoverable=recoverable,
        )
        if directory is not None:
            try:
                if supplied_run:
                    assert run_lock is not None
                    run_lock.validate_owned(directory.resume_lock_path)
                status_preparation = verified or preparation
                directory.write_status(
                    _status(
                        "failed",
                        status_preparation,
                        bundle_fingerprint=(
                            initialization.bundle_fingerprint
                            if recoverable and initialization is not None
                            else None
                        ),
                        baseline=baseline_metadata,
                        directory=directory,
                        failure_phase=phase,
                        error=structured,
                        rng_restored=True,
                    )
                )
            except BaseException as status_error:
                structured = _startup_error(
                    phase,
                    original,
                    verified or preparation,
                    directory=directory,
                    bundle_fingerprint=(
                        None
                        if initialization is None
                        else initialization.bundle_fingerprint
                    ),
                    recoverable=recoverable,
                    status_write_error=status_error,
                )
            finally:
                # A test hook or failing filesystem wrapper must not be able to
                # leak RNG/process-state changes from failure reporting itself.
                _restore_process_state(entry_state)
        raise structured from original


def initialize_scratch_training_startup(
    preparation: ScratchTrainingPreparation,
    *,
    run_directory: TrainingRunDirectory | None = None,
    run_lock: ResumeRunLock | None = None,
) -> ScratchTrainingStartup:
    """Create a normal trainable runtime even under caller no-grad state.

    ``run_directory`` and ``run_lock`` are an all-or-nothing injection point
    for a surrounding fresh-run transaction.  The caller retains ownership of
    a supplied lock on both success and failure; startup never releases it.
    """

    # Construction inside inference mode would permanently mark Parameters and
    # buffers as inference tensors.  A nested disabled context creates ordinary
    # tensors while restoring the caller's grad/inference state on exit.
    with torch.inference_mode(False), torch.enable_grad():
        return _initialize_scratch_training_startup(
            preparation,
            run_directory=run_directory,
            run_lock=run_lock,
        )


__all__ = [
    "SCRATCH_TRAINING_STARTUP_CONVENTION_VERSION",
    "SCRATCH_TRAINING_STARTUP_STATUS_SCHEMA_VERSION",
    "ScratchTrainingStartup",
    "ScratchTrainingStartupError",
    "initialize_scratch_training_startup",
]
