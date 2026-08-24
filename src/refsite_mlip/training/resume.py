"""Strict, transactional restoration of epoch-boundary training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import copy
import math
from numbers import Integral
import random
from typing import Any

import numpy as np
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.models.batch_executor import _validated_context
from refsite_mlip.models.template_context import TemplateExecutionContext

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_SCOPE,
    CheckpointMetadata,
    TrainingCheckpoint,
    _CONFIG_KEYS,
    _data_manifest,
    _package_versions,
    _plain,
    _snapshot,
    _template_fingerprint_mapping,
    _unit_conventions,
    _validated_batch_sequence,
)
from .fit import FitConfig
from .scheduler import SchedulerConfig, _validate_scheduler_binding
from .selection import ModelSelectionState


class CheckpointCompatibilityError(ValueError):
    """The checkpoint and current execution contract are not identical."""


class CheckpointRestoreError(RuntimeError):
    """A restore stage failed after preflight and was rolled back."""

    def __init__(
        self,
        *,
        stage: str,
        original_exception: BaseException,
        rollback_succeeded: bool,
        rollback_error: BaseException | None = None,
    ) -> None:
        self.stage = stage
        self.original_exception_type = type(original_exception).__name__
        self.original_exception_message = str(original_exception)
        self.rollback_succeeded = bool(rollback_succeeded)
        self.rollback_error_type = (
            None if rollback_error is None else type(rollback_error).__name__
        )
        self.rollback_error_message = (
            None if rollback_error is None else str(rollback_error)
        )
        rollback_text = "succeeded" if rollback_succeeded else "failed"
        suffix = ""
        if rollback_error is not None:
            suffix = (
                f"; rollback error={self.rollback_error_type}: "
                f"{self.rollback_error_message}"
            )
        super().__init__(
            f"checkpoint restore failed at stage={stage!r}: "
            f"{self.original_exception_type}: {self.original_exception_message}; "
            f"transactional rollback {rollback_text}{suffix}"
        )


@dataclass(frozen=True)
class ResumePolicy:
    require_version_match: bool = True
    require_git_commit_match: bool = False
    restore_python_rng: bool = True
    restore_numpy_rng: bool = True
    restore_torch_cpu_rng: bool = True
    restore_cuda_rng: bool = True
    allow_max_epochs_extension: bool = True

    def __post_init__(self) -> None:
        for name in (
            "require_version_match",
            "require_git_commit_match",
            "restore_python_rng",
            "restore_numpy_rng",
            "restore_torch_cpu_rng",
            "restore_cuda_rng",
            "allow_max_epochs_extension",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ResumePolicy":
        if not isinstance(values, Mapping):
            raise TypeError("resume policy must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class ResumeState:
    next_epoch: int
    global_step: int
    completed_epochs: int
    selection_state: ModelSelectionState
    fit_history: tuple[dict[str, Any], ...] | None
    resumed_fit_config: FitConfig
    checkpoint_metadata: CheckpointMetadata
    exact_resume_ready: bool
    restored_rng_domains: tuple[str, ...]
    compatibility_diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("next_epoch", "global_step", "completed_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.selection_state, ModelSelectionState):
            raise TypeError("selection_state must be a ModelSelectionState")
        if not isinstance(self.resumed_fit_config, FitConfig):
            raise TypeError("resumed_fit_config must be a FitConfig")
        if not isinstance(self.checkpoint_metadata, CheckpointMetadata):
            raise TypeError("checkpoint_metadata must be CheckpointMetadata")
        if not isinstance(self.exact_resume_ready, bool):
            raise TypeError("exact_resume_ready must be a bool")
        if self.resumed_fit_config.start_epoch != self.next_epoch:
            raise ValueError("resumed fit start epoch must equal next_epoch")
        if self.resumed_fit_config.global_step_start != self.global_step:
            raise ValueError("resumed fit global step must equal checkpoint global step")

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_epoch": self.next_epoch,
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "selection_state": self.selection_state.to_dict(),
            "fit_history": (
                None
                if self.fit_history is None
                else copy.deepcopy(list(self.fit_history))
            ),
            "resumed_fit_config": self.resumed_fit_config.to_dict(),
            "checkpoint_metadata": self.checkpoint_metadata.to_dict(),
            "exact_resume_ready": self.exact_resume_ready,
            "restored_rng_domains": list(self.restored_rng_domains),
            "compatibility_diagnostics": list(self.compatibility_diagnostics),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ResumeState":
        if not isinstance(values, Mapping):
            raise TypeError("resume state must be reconstructed from a mapping")
        data = dict(values)
        data["selection_state"] = ModelSelectionState.from_dict(
            data["selection_state"]
        )
        data["checkpoint_metadata"] = CheckpointMetadata.from_dict(
            data["checkpoint_metadata"]
        )
        data["resumed_fit_config"] = FitConfig.from_dict(
            data["resumed_fit_config"]
        )
        if data["fit_history"] is not None:
            data["fit_history"] = tuple(copy.deepcopy(data["fit_history"]))
        data["restored_rng_domains"] = tuple(data["restored_rng_domains"])
        data["compatibility_diagnostics"] = tuple(
            data["compatibility_diagnostics"]
        )
        return cls(**data)


def _compatibility_error(message: str) -> None:
    raise CheckpointCompatibilityError(message)


def _canonical(value, *, path: str):
    return _plain(value, path=path)


def _require_equal(name: str, current, saved) -> None:
    if current != saved:
        _compatibility_error(f"{name} mismatch between checkpoint and current run")


def _model_species(model: torch.nn.Module, resolved_configs: Mapping[str, Any]):
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "species_vocabulary"):
        return tuple(int(value) for value in config.species_vocabulary)
    if "species_vocabulary" not in resolved_configs:
        _compatibility_error(
            "current ordered species vocabulary is unavailable; provide "
            "resolved_configs['species_vocabulary']"
        )
    return tuple(int(value) for value in resolved_configs["species_vocabulary"])


def _validate_configs(
    checkpoint: TrainingCheckpoint,
    resolved_configs: Mapping[str, Any],
    resumed_max_epochs: int,
    policy: ResumePolicy,
) -> FitConfig:
    if not isinstance(resolved_configs, Mapping):
        raise TypeError("resolved_configs must be a mapping")
    missing = set(_CONFIG_KEYS) - set(resolved_configs)
    if missing:
        _compatibility_error(
            f"resolved configuration is missing keys: {sorted(missing)}"
        )
    current = {
        key: _canonical(resolved_configs[key], path=f"resolved_configs.{key}")
        for key in _CONFIG_KEYS
    }
    saved = checkpoint.metadata.resolved_configuration
    for key in _CONFIG_KEYS:
        if key == "fit":
            continue
        _require_equal(f"resolved {key} configuration", current[key], saved[key])

    if isinstance(resumed_max_epochs, bool) or not isinstance(
        resumed_max_epochs, Integral
    ):
        raise TypeError("resumed_max_epochs must be a positive integer")
    resumed_max_epochs = int(resumed_max_epochs)
    if resumed_max_epochs <= checkpoint.progress.next_epoch:
        _compatibility_error(
            "resumed max_epochs must be greater than checkpoint next_epoch"
        )
    saved_fit = FitConfig.from_dict(saved["fit"])
    current_fit = FitConfig.from_dict(current["fit"])
    if current_fit.max_epochs != resumed_max_epochs:
        _compatibility_error(
            "resolved fit max_epochs must equal resumed_max_epochs"
        )
    saved_nonmax = saved_fit.to_dict()
    current_nonmax = current_fit.to_dict()
    saved_nonmax.pop("max_epochs")
    current_nonmax.pop("max_epochs")
    _require_equal("fit configuration except max_epochs", current_nonmax, saved_nonmax)
    if policy.allow_max_epochs_extension:
        if resumed_max_epochs < saved_fit.max_epochs:
            _compatibility_error("max_epochs decrease is not allowed")
    elif resumed_max_epochs != saved_fit.max_epochs:
        _compatibility_error("max_epochs change is disallowed by resume policy")
    return FitConfig(
        max_epochs=resumed_max_epochs,
        start_epoch=checkpoint.progress.next_epoch,
        global_step_start=checkpoint.progress.global_step,
    )


def _validate_data_and_templates(
    checkpoint: TrainingCheckpoint,
    train_batches: Sequence[StructureBatch],
    validation_batches: Sequence[StructureBatch],
    template_contexts: Mapping[str, TemplateExecutionContext],
) -> None:
    train_batches = _validated_batch_sequence(train_batches, name="train_batches")
    validation_batches = _validated_batch_sequence(
        validation_batches, name="validation_batches"
    )
    train_manifest = _data_manifest(train_batches, split_name="train")
    validation_manifest = _data_manifest(
        validation_batches, split_name="validation"
    )
    _require_equal(
        "training data manifest",
        train_manifest.to_dict(),
        checkpoint.metadata.training_data.to_dict(),
    )
    _require_equal(
        "validation data manifest",
        validation_manifest.to_dict(),
        checkpoint.metadata.validation_data.to_dict(),
    )
    mapping = _template_fingerprint_mapping(train_batches, validation_batches)
    _require_equal(
        "template ID/fingerprint mapping",
        mapping,
        checkpoint.metadata.template_fingerprints,
    )
    if not isinstance(template_contexts, Mapping):
        raise TypeError("template_contexts must be a mapping")
    _require_equal(
        "template context ID set",
        set(template_contexts),
        set(mapping),
    )
    for template_id, fingerprint in mapping.items():
        try:
            _validated_context(template_id, fingerprint, template_contexts)
        except Exception as error:
            raise CheckpointCompatibilityError(
                f"template context {template_id!r} is incompatible: {error}"
            ) from error


def _validate_model_state(checkpoint: TrainingCheckpoint, model: torch.nn.Module) -> None:
    current = model.state_dict()
    saved = checkpoint.model_state_dict
    if list(current) != list(saved):
        _compatibility_error("model state_dict keys or ordering do not match")
    for key in current:
        left, right = current[key], saved[key]
        if not isinstance(right, torch.Tensor):
            _compatibility_error(f"checkpoint model state {key!r} is not a tensor")
        if left.shape != right.shape:
            _compatibility_error(
                f"model state shape mismatch for {key!r}: "
                f"current={tuple(left.shape)}, checkpoint={tuple(right.shape)}"
            )
        if left.dtype != right.dtype:
            _compatibility_error(
                f"model state dtype mismatch for {key!r}: "
                f"current={left.dtype}, checkpoint={right.dtype}"
            )


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _validate_optimizer_binding(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    optimizer_parameters = _optimizer_parameters(optimizer)
    if len({id(parameter) for parameter in optimizer_parameters}) != len(
        optimizer_parameters
    ):
        _compatibility_error("optimizer contains a parameter more than once")
    model_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if tuple(map(id, optimizer_parameters)) != tuple(map(id, model_parameters)):
        _compatibility_error(
            "optimizer parameters are not the current model trainable parameters "
            "in exact order"
        )


def _validate_optimizer_structure(
    checkpoint: TrainingCheckpoint,
    optimizer: torch.optim.Optimizer,
    optimizer_config: Mapping[str, Any],
) -> None:
    saved_groups = checkpoint.optimizer_state_dict.get("param_groups")
    current_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        _compatibility_error("optimizer parameter groups have invalid structure")
    if len(saved_groups) != len(current_groups):
        _compatibility_error("optimizer parameter-group count mismatch")
    for index, (saved, current) in enumerate(zip(saved_groups, current_groups)):
        if len(saved.get("params", ())) != len(current.get("params", ())):
            _compatibility_error(
                f"optimizer parameter count mismatch in group {index}"
            )
    kind = optimizer_config.get("optimizer")
    if kind != "adamw" or not isinstance(optimizer, torch.optim.AdamW):
        _compatibility_error("optimizer kind must match configured AdamW")


def _validate_scheduler_structure(
    checkpoint: TrainingCheckpoint,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scheduler_config: SchedulerConfig,
) -> None:
    try:
        _validate_scheduler_binding(optimizer, scheduler, scheduler_config)
    except Exception as error:
        raise CheckpointCompatibilityError(
            f"scheduler kind or optimizer binding mismatch: {error}"
        ) from error
    current_state = scheduler.state_dict()
    saved_state = checkpoint.scheduler_state_dict
    if set(current_state) != set(saved_state):
        _compatibility_error("scheduler state structure mismatch")


def _validate_versions_and_git(
    checkpoint: TrainingCheckpoint,
    policy: ResumePolicy,
    current_source_git_commit: str | None,
) -> list[str]:
    diagnostics: list[str] = []
    current_versions = _package_versions()
    if policy.require_version_match:
        _require_equal(
            "package versions", current_versions, checkpoint.metadata.package_versions
        )
    elif current_versions != checkpoint.metadata.package_versions:
        diagnostics.append("package version mismatch allowed by policy")
    saved_git = checkpoint.metadata.source_git_commit
    if policy.require_git_commit_match:
        if not saved_git or not current_source_git_commit:
            _compatibility_error(
                "git commit matching was required but one commit is unavailable"
            )
        _require_equal("source git commit", current_source_git_commit, saved_git)
    elif saved_git != current_source_git_commit:
        diagnostics.append(
            "source git commit mismatch is diagnostic-only under current policy"
        )
    return diagnostics


def _validate_cuda_restore_preconditions(
    checkpoint: TrainingCheckpoint, policy: ResumePolicy
) -> None:
    if not policy.restore_cuda_rng or checkpoint.cuda_device_count == 0:
        return
    if not torch.cuda.is_available():
        _compatibility_error(
            "checkpoint contains CUDA RNG state but CUDA is unavailable"
        )
    if torch.cuda.device_count() != checkpoint.cuda_device_count:
        _compatibility_error(
            "CUDA device count does not match checkpoint RNG metadata"
        )


def validate_checkpoint_compatibility(
    checkpoint: TrainingCheckpoint,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_batches: Sequence[StructureBatch],
    validation_batches: Sequence[StructureBatch],
    template_contexts: Mapping[str, TemplateExecutionContext],
    resolved_configs: Mapping[str, Any],
    *,
    resumed_max_epochs: int,
    policy: ResumePolicy | None = None,
    current_source_git_commit: str | None = None,
) -> tuple[str, ...]:
    """Validate an exact epoch-boundary resume without mutating live state."""

    policy = ResumePolicy() if policy is None else policy
    if not isinstance(policy, ResumePolicy):
        raise TypeError("policy must be a ResumePolicy")
    if not isinstance(checkpoint, TrainingCheckpoint):
        raise TypeError("checkpoint must be a TrainingCheckpoint")
    if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
        _compatibility_error("checkpoint schema version mismatch")
    if checkpoint.checkpoint_scope != CHECKPOINT_SCOPE:
        _compatibility_error("only epoch-boundary checkpoints can be resumed")
    if checkpoint.progress.next_batch_index != 0:
        _compatibility_error("mid-epoch checkpoint resume is unsupported")
    if checkpoint.progress.stopped_early or checkpoint.selection_state.stopped_early:
        _compatibility_error("a checkpoint whose selection state already stopped cannot resume")
    if checkpoint.progress.stopped_early != checkpoint.selection_state.stopped_early:
        _compatibility_error("checkpoint progress and selection stop states disagree")
    if checkpoint.progress.best_epoch != checkpoint.selection_state.best_epoch:
        _compatibility_error("checkpoint progress and selection best epochs disagree")
    if checkpoint.progress.best_global_step != checkpoint.selection_state.best_global_step:
        _compatibility_error(
            "checkpoint progress and selection best global steps disagree"
        )
    if checkpoint.selection_state.validation_events:
        if (
            checkpoint.progress.global_step
            != checkpoint.selection_state.last_validation_global_step
        ):
            _compatibility_error(
                "checkpoint progress does not match last validation global step"
            )
        if (
            checkpoint.progress.next_epoch
            <= checkpoint.selection_state.last_validation_epoch
        ):
            _compatibility_error(
                "checkpoint next epoch does not follow last validation event"
            )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(scheduler, "state_dict") or not hasattr(
        scheduler, "load_state_dict"
    ):
        raise TypeError("scheduler must provide state_dict/load_state_dict")

    resumed_fit = _validate_configs(
        checkpoint, resolved_configs, resumed_max_epochs, policy
    )
    del resumed_fit
    species = _model_species(model, resolved_configs)
    _require_equal(
        "ordered species vocabulary", species, checkpoint.metadata.species_vocabulary
    )
    current_units = _canonical(
        resolved_configs.get("unit_conventions", _unit_conventions()),
        path="resolved_configs.unit_conventions",
    )
    _require_equal(
        "unit/stress/Voigt conventions",
        current_units,
        checkpoint.metadata.unit_conventions,
    )
    baseline = _canonical(
        resolved_configs.get("baseline_fit_metadata"),
        path="resolved_configs.baseline_fit_metadata",
    )
    _require_equal(
        "atomic baseline fit metadata",
        baseline,
        checkpoint.metadata.baseline_fit_metadata,
    )
    _validate_data_and_templates(
        checkpoint, train_batches, validation_batches, template_contexts
    )
    _validate_model_state(checkpoint, model)
    _validate_optimizer_binding(model, optimizer)
    _validate_optimizer_structure(
        checkpoint, optimizer, checkpoint.metadata.resolved_configuration["optimizer"]
    )
    scheduler_config = SchedulerConfig.from_dict(
        checkpoint.metadata.resolved_configuration["scheduler"]
    )
    _validate_scheduler_structure(
        checkpoint, optimizer, scheduler, scheduler_config
    )
    _validate_cuda_restore_preconditions(checkpoint, policy)
    diagnostics = _validate_versions_and_git(
        checkpoint, policy, current_source_git_commit
    )
    diagnostics.extend(
        (
            "checkpoint schema and epoch-boundary scope match",
            "ordered train/validation manifests and template fingerprints match",
            "physics/configuration and state structures match",
            "parameter and scheduler optimizer bindings are valid",
        )
    )
    return tuple(diagnostics)


def _tree_equal(first, second) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return first.shape == second.shape and first.dtype == second.dtype and torch.equal(
            first.detach().cpu(), second.detach().cpu()
        )
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return list(first) == list(second) and all(
            _tree_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(
            _tree_equal(left, right) for left, right in zip(first, second)
        )
    return first == second


def _python_state_from_safe(state):
    if isinstance(state, list):
        return tuple(_python_state_from_safe(value) for value in state)
    return state


def _numpy_state_from_safe(state: Mapping[str, Any]):
    required = {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("checkpoint NumPy RNG state has invalid structure")
    return (
        str(state["bit_generator"]),
        np.asarray(state["state"], dtype=np.uint32),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def _rng_snapshot_raw() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda": tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
    }


def _restore_raw_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if snapshot["cuda"]:
        torch.cuda.set_rng_state_all(list(snapshot["cuda"]))


def _restore_checkpoint_rng(
    checkpoint: TrainingCheckpoint, policy: ResumePolicy
) -> tuple[str, ...]:
    domains: list[str] = []
    if policy.restore_python_rng:
        random.setstate(_python_state_from_safe(checkpoint.python_rng_state))
        domains.append("python")
    if policy.restore_numpy_rng:
        np.random.set_state(_numpy_state_from_safe(checkpoint.numpy_rng_state))
        domains.append("numpy")
    if policy.restore_torch_cpu_rng:
        torch.set_rng_state(checkpoint.torch_cpu_rng_state)
        domains.append("torch_cpu")
    if policy.restore_cuda_rng and checkpoint.cuda_rng_states:
        torch.cuda.set_rng_state_all(list(checkpoint.cuda_rng_states))
        domains.append("cuda")
    return tuple(domains)


def _gradient_snapshot(model: torch.nn.Module):
    return tuple(
        (parameter, parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
        for parameter in model.parameters()
    )


def _restore_gradients(snapshot) -> None:
    for parameter, original_gradient, value in snapshot:
        parameter.grad = original_gradient
        if original_gradient is not None:
            with torch.no_grad():
                original_gradient.copy_(value)


def _rollback(
    model,
    optimizer,
    scheduler,
    model_state,
    optimizer_state,
    scheduler_state,
    gradients,
    training_mode,
    rng_state,
) -> None:
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    _restore_gradients(gradients)
    model.train(training_mode)
    _restore_raw_rng(rng_state)


def restore_training_checkpoint_(
    checkpoint: TrainingCheckpoint,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_batches: Sequence[StructureBatch],
    validation_batches: Sequence[StructureBatch],
    template_contexts: Mapping[str, TemplateExecutionContext],
    resolved_configs: Mapping[str, Any],
    *,
    resumed_max_epochs: int,
    policy: ResumePolicy | None = None,
    current_source_git_commit: str | None = None,
) -> ResumeState:
    """Apply a compatible checkpoint atomically to live training state.

    Compatibility errors occur before any mutation.  Every exception after
    preflight triggers restoration of model, optimizer, scheduler, gradients,
    train/eval mode, and all process RNG domains.
    """

    policy = ResumePolicy() if policy is None else policy
    diagnostics = validate_checkpoint_compatibility(
        checkpoint,
        model,
        optimizer,
        scheduler,
        train_batches,
        validation_batches,
        template_contexts,
        resolved_configs,
        resumed_max_epochs=resumed_max_epochs,
        policy=policy,
        current_source_git_commit=current_source_git_commit,
    )
    resumed_fit_config = _validate_configs(
        checkpoint, resolved_configs, resumed_max_epochs, policy
    )

    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    model_state = _snapshot(model.state_dict(), path="rollback.model")
    optimizer_state = _snapshot(optimizer.state_dict(), path="rollback.optimizer")
    scheduler_state = _snapshot(scheduler.state_dict(), path="rollback.scheduler")
    gradients = _gradient_snapshot(model)
    training_mode = model.training
    rng_state = _rng_snapshot_raw()
    stage = "model_state"
    try:
        model.load_state_dict(checkpoint.model_state_dict, strict=True)
        stage = "optimizer_state"
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        stage = "scheduler_state"
        scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        stage = "state_bindings"
        _validate_optimizer_binding(model, optimizer)
        scheduler_config = SchedulerConfig.from_dict(
            checkpoint.metadata.resolved_configuration["scheduler"]
        )
        _validate_scheduler_binding(optimizer, scheduler, scheduler_config)
        if tuple(id(parameter) for parameter in model.parameters()) != parameter_ids:
            raise RuntimeError("model Parameter identity changed during restore")
        stage = "selection_progress_history"
        selection_state = ModelSelectionState.from_dict(
            checkpoint.selection_state.to_dict()
        )
        fit_history = (
            None
            if checkpoint.fit_history is None
            else tuple(copy.deepcopy(checkpoint.fit_history))
        )
        stage = "clear_gradients"
        for parameter in model.parameters():
            parameter.grad = None
        stage = "rng_state"
        restored_domains = _restore_checkpoint_rng(checkpoint, policy)
        stage = "final_invariants"
        if tuple(id(parameter) for parameter in model.parameters()) != parameter_ids:
            raise RuntimeError("model Parameter identity changed during restore")
        _validate_optimizer_binding(model, optimizer)
        _validate_scheduler_binding(optimizer, scheduler, scheduler_config)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("restored model gradients were not cleared")
        if not _tree_equal(model.state_dict(), checkpoint.model_state_dict):
            raise RuntimeError("restored model state does not equal checkpoint")
        if not _tree_equal(optimizer.state_dict(), checkpoint.optimizer_state_dict):
            raise RuntimeError("restored optimizer state does not equal checkpoint")
        if not _tree_equal(scheduler.state_dict(), checkpoint.scheduler_state_dict):
            raise RuntimeError("restored scheduler state does not equal checkpoint")
        model.train(training_mode)
    except Exception as error:
        rollback_error = None
        rollback_succeeded = False
        try:
            _rollback(
                model,
                optimizer,
                scheduler,
                model_state,
                optimizer_state,
                scheduler_state,
                gradients,
                training_mode,
                rng_state,
            )
            rollback_succeeded = True
        except Exception as candidate:
            rollback_error = candidate
        raise CheckpointRestoreError(
            stage=stage,
            original_exception=error,
            rollback_succeeded=rollback_succeeded,
            rollback_error=rollback_error,
        ) from error

    return ResumeState(
        next_epoch=checkpoint.progress.next_epoch,
        global_step=checkpoint.progress.global_step,
        completed_epochs=checkpoint.progress.completed_epochs,
        selection_state=selection_state,
        fit_history=fit_history,
        resumed_fit_config=resumed_fit_config,
        checkpoint_metadata=CheckpointMetadata.from_dict(
            checkpoint.metadata.to_dict()
        ),
        exact_resume_ready=(
            policy.restore_python_rng
            and policy.restore_numpy_rng
            and policy.restore_torch_cpu_rng
            and (checkpoint.cuda_device_count == 0 or policy.restore_cuda_rng)
        ),
        restored_rng_domains=restored_domains,
        compatibility_diagnostics=diagnostics,
    )


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointRestoreError",
    "ResumePolicy",
    "ResumeState",
    "restore_training_checkpoint_",
    "validate_checkpoint_compatibility",
]
