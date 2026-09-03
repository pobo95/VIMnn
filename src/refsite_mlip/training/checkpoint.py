"""Weights-only-safe epoch-boundary training checkpoint snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import importlib.metadata
import math
from numbers import Integral
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from refsite_mlip._atomic import commit_temporary_file

from refsite_mlip.data import StructureBatch

from .fit import FitConfig, FitEpochRecord
from .optimizer import validate_optimizer_binding
from .selection import ModelSelectionState


CHECKPOINT_SCHEMA_VERSION = "refsite_training_checkpoint_v1"
CHECKPOINT_SCOPE = "epoch_boundary"
DATA_MANIFEST_VERSION = "prebatched_structure_sequence_v1"
UNIT_CONVENTION_VERSION = "angstrom_ev_tensile_voigt_v1"

_CONFIG_KEYS = (
    "model",
    "loss",
    "optimizer",
    "train_step",
    "validation_step",
    "scheduler",
    "model_selection",
    "fit",
)
_TENSOR_BATCH_FIELDS = (
    "positions",
    "atomic_numbers",
    "cells",
    "origins",
    "pbc",
    "energy",
    "energy_mask",
    "forces",
    "force_mask",
    "stress",
    "stress_mask",
    "force_present",
    "stress_present",
    "force_mask_provided",
    "stress_mask_provided",
    "atom_ptr",
    "atom_batch",
)


def _nonnegative_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _optional_nonnegative_integer(name: str, value) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(name, value)


def _sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal string")
    return value.lower()


def _plain(value, *, path: str = "value"):
    """Convert configuration metadata to weights-only-safe built-in types."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Enum):
        return _plain(value.value, path=path)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name), path=f"{path}.{field.name}")
            for field in fields(value)
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict(), path=path)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError(f"{path} contains an unsupported mapping key")
            result[key] = _plain(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_plain(item, path=f"{path}[]") for item in value]
    module_name = type(value).__module__
    if module_name.startswith("e3nn"):
        return str(value)
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _snapshot(value, *, path: str):
    """Own an immutable CPU clone of tensor/container state."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError(f"{path} contains an unsupported mapping key")
            result[key] = _snapshot(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_snapshot(item, path=f"{path}[]") for item in value]
    raise TypeError(f"{path} contains unsupported state type {type(value).__name__}")


def _validate_safe_tree(value, *, path: str, tensors_must_be_cpu: bool) -> None:
    if isinstance(value, torch.Tensor):
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError(f"{path} tensor must not retain an autograd graph")
        if tensors_must_be_cpu and value.device.type != "cpu":
            raise ValueError(f"{path} snapshot tensor must be on CPU")
        return
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError(f"{path} contains an unsupported mapping key")
            _validate_safe_tree(
                item,
                path=f"{path}.{key}",
                tensors_must_be_cpu=tensors_must_be_cpu,
            )
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_safe_tree(
                item,
                path=f"{path}[{index}]",
                tensors_must_be_cpu=tensors_must_be_cpu,
            )
        return
    raise TypeError(f"{path} contains weights-only-unsafe type {type(value).__name__}")


@dataclass(frozen=True)
class FitProgress:
    next_epoch: int
    global_step: int
    completed_epochs: int
    next_batch_index: int = 0
    last_completed_epoch: int | None = None
    stopped_early: bool = False
    best_epoch: int | None = None
    best_global_step: int | None = None

    def __post_init__(self) -> None:
        for name in ("next_epoch", "global_step", "completed_epochs"):
            object.__setattr__(self, name, _nonnegative_integer(name, getattr(self, name)))
        next_batch = _nonnegative_integer("next_batch_index", self.next_batch_index)
        if next_batch != 0:
            raise ValueError("epoch-boundary checkpoints require next_batch_index=0")
        object.__setattr__(self, "next_batch_index", next_batch)
        for name in ("last_completed_epoch", "best_epoch", "best_global_step"):
            object.__setattr__(
                self, name, _optional_nonnegative_integer(name, getattr(self, name))
            )
        if not isinstance(self.stopped_early, bool):
            raise TypeError("stopped_early must be a bool")
        if self.completed_epochs == 0:
            if self.last_completed_epoch is not None:
                raise ValueError("zero completed epochs require no last completed epoch")
        elif self.last_completed_epoch is None:
            raise ValueError("completed epochs require last_completed_epoch")
        if self.last_completed_epoch is not None and self.next_epoch != self.last_completed_epoch + 1:
            raise ValueError("next_epoch must follow last_completed_epoch")
        if (self.best_epoch is None) != (self.best_global_step is None):
            raise ValueError("best epoch and global step must be set together")

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_epoch": self.next_epoch,
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "next_batch_index": self.next_batch_index,
            "last_completed_epoch": self.last_completed_epoch,
            "stopped_early": self.stopped_early,
            "best_epoch": self.best_epoch,
            "best_global_step": self.best_global_step,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FitProgress":
        if not isinstance(values, Mapping):
            raise TypeError("fit progress must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class TrainingDataManifest:
    split_name: str
    fingerprint: str
    manifest_version: str
    unit_convention_version: str
    number_of_batches: int
    number_of_structures: int
    number_of_atoms: int
    ordered_batch_sample_ids: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.split_name, str) or not self.split_name:
            raise ValueError("split_name must be a nonempty string")
        object.__setattr__(
            self, "fingerprint", _sha256(self.fingerprint, name="data fingerprint")
        )
        if self.manifest_version != DATA_MANIFEST_VERSION:
            raise ValueError("unsupported data manifest version")
        if not isinstance(self.unit_convention_version, str) or not self.unit_convention_version:
            raise ValueError("unit convention version must be a nonempty string")
        for name in ("number_of_batches", "number_of_structures", "number_of_atoms"):
            object.__setattr__(self, name, _nonnegative_integer(name, getattr(self, name)))
        if len(self.ordered_batch_sample_ids) != self.number_of_batches:
            raise ValueError("batch sample-ID boundaries do not match batch count")
        if any(
            not isinstance(sample_id, str) or not sample_id
            for batch_ids in self.ordered_batch_sample_ids
            for sample_id in batch_ids
        ):
            raise ValueError("manifest sample IDs must be nonempty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "fingerprint": self.fingerprint,
            "manifest_version": self.manifest_version,
            "unit_convention_version": self.unit_convention_version,
            "number_of_batches": self.number_of_batches,
            "number_of_structures": self.number_of_structures,
            "number_of_atoms": self.number_of_atoms,
            "ordered_batch_sample_ids": [
                list(sample_ids) for sample_ids in self.ordered_batch_sample_ids
            ],
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingDataManifest":
        if not isinstance(values, Mapping):
            raise TypeError("data manifest must be reconstructed from a mapping")
        data = dict(values)
        data["ordered_batch_sample_ids"] = tuple(
            tuple(sample_ids) for sample_ids in data["ordered_batch_sample_ids"]
        )
        return cls(**data)


@dataclass(frozen=True)
class CheckpointMetadata:
    resolved_configuration: dict[str, Any]
    species_vocabulary: tuple[int, ...]
    unit_conventions: dict[str, Any]
    template_fingerprints: dict[str, str]
    training_data: TrainingDataManifest
    validation_data: TrainingDataManifest
    package_versions: dict[str, str]
    baseline_fit_metadata: dict[str, Any] | None = None
    source_git_commit: str | None = None

    def __post_init__(self) -> None:
        if set(self.resolved_configuration) != set(_CONFIG_KEYS):
            raise ValueError(
                "resolved configuration must contain exactly: " + ", ".join(_CONFIG_KEYS)
            )
        if (
            not self.species_vocabulary
            or any(isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0 for value in self.species_vocabulary)
            or len(set(self.species_vocabulary)) != len(self.species_vocabulary)
        ):
            raise ValueError("species vocabulary must contain unique positive integers")
        if self.unit_conventions.get("version") != UNIT_CONVENTION_VERSION:
            raise ValueError("unsupported unit convention metadata")
        if not self.template_fingerprints:
            raise ValueError("template fingerprint mapping must not be empty")
        for template_id, fingerprint in self.template_fingerprints.items():
            if not isinstance(template_id, str) or not template_id:
                raise ValueError("template IDs must be nonempty strings")
            _sha256(fingerprint, name=f"template fingerprint for {template_id}")
        if not isinstance(self.training_data, TrainingDataManifest) or not isinstance(
            self.validation_data, TrainingDataManifest
        ):
            raise TypeError("training and validation manifests are required")
        if self.training_data.split_name != "train" or self.validation_data.split_name != "validation":
            raise ValueError("data manifest split names must be train and validation")
        if not self.package_versions or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.package_versions.items()
        ):
            raise ValueError("package versions must be a nonempty string mapping")
        if self.source_git_commit is not None and (
            not isinstance(self.source_git_commit, str) or not self.source_git_commit
        ):
            raise ValueError("source_git_commit must be a nonempty string or None")
        _validate_safe_tree(
            self.resolved_configuration,
            path="resolved_configuration",
            tensors_must_be_cpu=True,
        )
        _validate_safe_tree(
            self.unit_conventions,
            path="unit_conventions",
            tensors_must_be_cpu=True,
        )
        if self.baseline_fit_metadata is not None:
            _validate_safe_tree(
                self.baseline_fit_metadata,
                path="baseline_fit_metadata",
                tensors_must_be_cpu=True,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_configuration": _snapshot(
                self.resolved_configuration, path="resolved_configuration"
            ),
            "species_vocabulary": list(self.species_vocabulary),
            "unit_conventions": _snapshot(self.unit_conventions, path="unit_conventions"),
            "template_fingerprints": dict(self.template_fingerprints),
            "training_data": self.training_data.to_dict(),
            "validation_data": self.validation_data.to_dict(),
            "package_versions": dict(self.package_versions),
            "baseline_fit_metadata": (
                None
                if self.baseline_fit_metadata is None
                else _snapshot(self.baseline_fit_metadata, path="baseline_fit_metadata")
            ),
            "source_git_commit": self.source_git_commit,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CheckpointMetadata":
        if not isinstance(values, Mapping):
            raise TypeError("checkpoint metadata must be reconstructed from a mapping")
        data = dict(values)
        data["species_vocabulary"] = tuple(data["species_vocabulary"])
        data["training_data"] = TrainingDataManifest.from_dict(data["training_data"])
        data["validation_data"] = TrainingDataManifest.from_dict(data["validation_data"])
        return cls(**data)


def _update_hash_text(digest, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_hash_tensor(digest, name: str, tensor: torch.Tensor) -> None:
    _update_hash_text(digest, name)
    _update_hash_text(digest, str(tensor.dtype))
    _update_hash_text(digest, ",".join(str(size) for size in tensor.shape))
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.numpy().tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _validated_batch_sequence(batches, *, name: str) -> Sequence[StructureBatch]:
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError(f"{name} must be a deterministic Sequence[StructureBatch]")
    if len(batches) == 0:
        raise ValueError(f"{name} must not be empty")
    for index, batch in enumerate(batches):
        if not isinstance(batch, StructureBatch):
            raise TypeError(f"{name}[{index}] must be a StructureBatch")
        batch.validate()
    return batches


def fingerprint_batch_sequence(
    batches: Sequence[StructureBatch],
    *,
    split_name: str,
    unit_convention_version: str = UNIT_CONVENTION_VERSION,
) -> str:
    """SHA-256 over ordered prebatched tensors, labels, masks, and metadata."""

    batches = _validated_batch_sequence(batches, name=f"{split_name}_batches")
    if not isinstance(split_name, str) or not split_name:
        raise ValueError("split_name must be a nonempty string")
    if not isinstance(unit_convention_version, str) or not unit_convention_version:
        raise ValueError("unit_convention_version must be a nonempty string")
    digest = hashlib.sha256()
    _update_hash_text(digest, DATA_MANIFEST_VERSION)
    _update_hash_text(digest, unit_convention_version)
    _update_hash_text(digest, split_name)
    _update_hash_text(digest, str(len(batches)))
    for batch_index, batch in enumerate(batches):
        _update_hash_text(digest, f"batch:{batch_index}")
        _update_hash_text(digest, str(batch.num_structures))
        for sample_id, template_id, fingerprint in zip(
            batch.sample_ids, batch.template_ids, batch.template_fingerprints
        ):
            _update_hash_text(digest, sample_id)
            _update_hash_text(digest, template_id)
            _update_hash_text(digest, fingerprint)
        for field_name in _TENSOR_BATCH_FIELDS:
            _update_hash_tensor(digest, field_name, getattr(batch, field_name))
    return digest.hexdigest()


def _data_manifest(batches, *, split_name: str) -> TrainingDataManifest:
    batches = _validated_batch_sequence(batches, name=f"{split_name}_batches")
    return TrainingDataManifest(
        split_name=split_name,
        fingerprint=fingerprint_batch_sequence(batches, split_name=split_name),
        manifest_version=DATA_MANIFEST_VERSION,
        unit_convention_version=UNIT_CONVENTION_VERSION,
        number_of_batches=len(batches),
        number_of_structures=sum(batch.num_structures for batch in batches),
        number_of_atoms=sum(batch.num_atoms for batch in batches),
        ordered_batch_sample_ids=tuple(batch.sample_ids for batch in batches),
    )


def _template_fingerprint_mapping(train_batches, validation_batches) -> dict[str, str]:
    result = {}
    for batch in tuple(train_batches) + tuple(validation_batches):
        for template_id, fingerprint in zip(
            batch.template_ids, batch.template_fingerprints
        ):
            previous = result.setdefault(template_id, fingerprint)
            if previous != fingerprint:
                raise ValueError("one template ID has inconsistent fingerprints")
    return dict(sorted(result.items()))


def _python_rng_state() -> list[Any]:
    return _plain(random.getstate(), path="python_rng_state")


def _numpy_rng_state() -> dict[str, Any]:
    bit_generator, state, position, has_gauss, cached = np.random.get_state()
    return {
        "bit_generator": str(bit_generator),
        "state": [int(value) for value in state.tolist()],
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _package_versions() -> dict[str, str]:
    result = {
        "python": ".".join(str(value) for value in os.sys.version_info[:3]),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }
    try:
        result["e3nn"] = importlib.metadata.version("e3nn")
    except importlib.metadata.PackageNotFoundError:
        result["e3nn"] = "unavailable"
    try:
        result["refsite_mlip"] = importlib.metadata.version("refsite-mlip")
    except importlib.metadata.PackageNotFoundError:
        result["refsite_mlip"] = "0.1.0"
    return result


def _unit_conventions() -> dict[str, Any]:
    return {
        "version": UNIT_CONVENTION_VERSION,
        "length": "angstrom",
        "energy": "eV",
        "force": "eV/angstrom",
        "stress": "eV/angstrom^3",
        "stress_sign": "tensile_positive",
        "stress_tensor": "symmetric_3x3",
        "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
    }


@dataclass(frozen=True)
class TrainingCheckpoint:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    selection_state: ModelSelectionState
    progress: FitProgress
    metadata: CheckpointMetadata
    python_rng_state: list[Any]
    numpy_rng_state: dict[str, Any]
    torch_cpu_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]
    cuda_device_count: int
    fit_history: tuple[dict[str, Any], ...] | None = None
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    checkpoint_scope: str = CHECKPOINT_SCOPE

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema {self.schema_version!r}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
            )
        if self.checkpoint_scope != CHECKPOINT_SCOPE:
            raise ValueError("only epoch_boundary checkpoints are supported")
        if not isinstance(self.selection_state, ModelSelectionState):
            raise TypeError("selection_state must be a ModelSelectionState")
        if not isinstance(self.progress, FitProgress):
            raise TypeError("progress must be a FitProgress")
        if not isinstance(self.metadata, CheckpointMetadata):
            raise TypeError("metadata must be CheckpointMetadata")
        for name in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a plain dict")
            _validate_safe_tree(value, path=name, tensors_must_be_cpu=False)
        if not self.model_state_dict:
            raise ValueError("model_state_dict must not be empty")
        if not {"state", "param_groups"}.issubset(self.optimizer_state_dict):
            raise ValueError("optimizer_state_dict has an invalid basic structure")
        _validate_safe_tree(
            self.python_rng_state, path="python_rng_state", tensors_must_be_cpu=False
        )
        _validate_safe_tree(
            self.numpy_rng_state, path="numpy_rng_state", tensors_must_be_cpu=False
        )
        if (
            not isinstance(self.torch_cpu_rng_state, torch.Tensor)
            or self.torch_cpu_rng_state.dtype != torch.uint8
            or self.torch_cpu_rng_state.device.type != "cpu"
            or self.torch_cpu_rng_state.requires_grad
        ):
            raise ValueError("torch CPU RNG state must be a detached CPU uint8 tensor")
        count = _nonnegative_integer("cuda_device_count", self.cuda_device_count)
        object.__setattr__(self, "cuda_device_count", count)
        if len(self.cuda_rng_states) != count:
            raise ValueError("CUDA RNG state count does not match CUDA device count")
        for state in self.cuda_rng_states:
            if (
                not isinstance(state, torch.Tensor)
                or state.dtype != torch.uint8
                or state.device.type != "cpu"
                or state.requires_grad
            ):
                raise ValueError("CUDA RNG states must be detached CPU uint8 tensors")
        if self.fit_history is not None:
            if not isinstance(self.fit_history, tuple):
                raise TypeError("fit_history must be a tuple or None")
            _validate_safe_tree(
                self.fit_history, path="fit_history", tensors_must_be_cpu=False
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_scope": self.checkpoint_scope,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "selection_state": self.selection_state.to_dict(),
            "progress": self.progress.to_dict(),
            "fit_history": None if self.fit_history is None else list(self.fit_history),
            "metadata": self.metadata.to_dict(),
            "python_rng_state": self.python_rng_state,
            "numpy_rng_state": self.numpy_rng_state,
            "torch_cpu_rng_state": self.torch_cpu_rng_state,
            "cuda_rng_states": list(self.cuda_rng_states),
            "cuda_device_count": self.cuda_device_count,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingCheckpoint":
        if not isinstance(values, Mapping):
            raise TypeError("checkpoint payload must be a mapping")
        required = {
            "schema_version",
            "checkpoint_scope",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "selection_state",
            "progress",
            "fit_history",
            "metadata",
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "cuda_rng_states",
            "cuda_device_count",
        }
        if set(values) != required:
            missing = sorted(required - set(values))
            unknown = sorted(set(values) - required)
            raise ValueError(
                f"checkpoint payload keys are invalid; missing={missing}, unknown={unknown}"
            )
        data = dict(values)
        data["selection_state"] = ModelSelectionState.from_dict(data["selection_state"])
        data["progress"] = FitProgress.from_dict(data["progress"])
        data["metadata"] = CheckpointMetadata.from_dict(data["metadata"])
        if data["fit_history"] is not None:
            data["fit_history"] = tuple(data["fit_history"])
        data["cuda_rng_states"] = tuple(
            state.detach().clone().cpu() for state in data["cuda_rng_states"]
        )
        data["torch_cpu_rng_state"] = data["torch_cpu_rng_state"].detach().clone().cpu()
        return cls(**data)


def capture_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    selection_state: ModelSelectionState,
    progress: FitProgress,
    train_batches: Sequence[StructureBatch],
    validation_batches: Sequence[StructureBatch],
    *,
    model_config,
    loss_config,
    optimizer_config,
    train_step_config,
    validation_step_config,
    scheduler_config,
    model_selection_config,
    fit_config: FitConfig,
    species_vocabulary,
    fit_history: Sequence[FitEpochRecord] | None = None,
    baseline_fit_metadata=None,
    source_git_commit: str | None = None,
) -> TrainingCheckpoint:
    """Capture an owned CPU snapshot without mutating live state or RNG."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    validate_optimizer_binding(model, optimizer)
    if not hasattr(scheduler, "state_dict"):
        raise TypeError("scheduler must provide state_dict")
    if not isinstance(selection_state, ModelSelectionState):
        raise TypeError("selection_state must be a ModelSelectionState")
    if not isinstance(progress, FitProgress):
        raise TypeError("progress must be a FitProgress")
    if not isinstance(fit_config, FitConfig):
        raise TypeError("fit_config must be a FitConfig")
    if progress.stopped_early != selection_state.stopped_early:
        raise ValueError("progress and selection stopped_early states do not match")
    if progress.best_epoch != selection_state.best_epoch:
        raise ValueError("progress and selection best epochs do not match")
    if progress.best_global_step != selection_state.best_global_step:
        raise ValueError("progress and selection best global steps do not match")
    if selection_state.validation_events > 0:
        if progress.global_step != selection_state.last_validation_global_step:
            raise ValueError(
                "progress global step must match the last validation global step"
            )
        if progress.next_epoch <= selection_state.last_validation_epoch:
            raise ValueError(
                "progress next epoch must follow the last validation epoch"
            )
    train_batches = _validated_batch_sequence(train_batches, name="train_batches")
    validation_batches = _validated_batch_sequence(
        validation_batches, name="validation_batches"
    )
    configuration = {
        "model": _plain(model_config, path="model_config"),
        "loss": _plain(loss_config, path="loss_config"),
        "optimizer": _plain(optimizer_config, path="optimizer_config"),
        "train_step": _plain(train_step_config, path="train_step_config"),
        "validation_step": _plain(
            validation_step_config, path="validation_step_config"
        ),
        "scheduler": _plain(scheduler_config, path="scheduler_config"),
        "model_selection": _plain(
            model_selection_config, path="model_selection_config"
        ),
        "fit": _plain(fit_config, path="fit_config"),
    }
    history = None
    if fit_history is not None:
        if isinstance(fit_history, (str, bytes)) or not isinstance(fit_history, Sequence):
            raise TypeError("fit_history must be a Sequence[FitEpochRecord] or None")
        history = tuple(
            _plain(record.to_dict(), path="fit_history[]")
            if isinstance(record, FitEpochRecord)
            else (_ for _ in ()).throw(
                TypeError("fit_history entries must be FitEpochRecord objects")
            )
            for record in fit_history
        )
    baseline = (
        None
        if baseline_fit_metadata is None
        else _plain(baseline_fit_metadata, path="baseline_fit_metadata")
    )
    metadata = CheckpointMetadata(
        resolved_configuration=configuration,
        species_vocabulary=tuple(species_vocabulary),
        unit_conventions=_unit_conventions(),
        template_fingerprints=_template_fingerprint_mapping(
            train_batches, validation_batches
        ),
        training_data=_data_manifest(train_batches, split_name="train"),
        validation_data=_data_manifest(
            validation_batches, split_name="validation"
        ),
        package_versions=_package_versions(),
        baseline_fit_metadata=baseline,
        source_git_commit=source_git_commit,
    )
    torch_cpu_rng = torch.random.get_rng_state().detach().clone().cpu()
    if torch.cuda.is_available():
        cuda_states = tuple(
            state.detach().clone().cpu() for state in torch.cuda.get_rng_state_all()
        )
        cuda_count = torch.cuda.device_count()
    else:
        cuda_states = ()
        cuda_count = 0
    return TrainingCheckpoint(
        model_state_dict=_snapshot(model.state_dict(), path="model_state_dict"),
        optimizer_state_dict=_snapshot(
            optimizer.state_dict(), path="optimizer_state_dict"
        ),
        scheduler_state_dict=_snapshot(
            scheduler.state_dict(), path="scheduler_state_dict"
        ),
        selection_state=selection_state,
        progress=progress,
        metadata=metadata,
        python_rng_state=_python_rng_state(),
        numpy_rng_state=_numpy_rng_state(),
        torch_cpu_rng_state=torch_cpu_rng,
        cuda_rng_states=cuda_states,
        cuda_device_count=cuda_count,
        fit_history=history,
    )


def save_training_checkpoint(
    checkpoint: TrainingCheckpoint,
    path: str | os.PathLike,
    *,
    overwrite: bool = False,
) -> None:
    """Write completely, then atomically create or replace the target path."""

    if not isinstance(checkpoint, TrainingCheckpoint):
        raise TypeError("checkpoint must be a TrainingCheckpoint")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    target = Path(path)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError(f"checkpoint parent directory does not exist: {parent}")
    if target.exists():
        if not target.is_file():
            raise ValueError("checkpoint target exists and is not a regular file")
        if not overwrite:
            raise FileExistsError(f"checkpoint already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(checkpoint.to_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        commit_temporary_file(temporary, target, overwrite=overwrite)
        try:
            directory_descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def load_training_checkpoint(
    path: str | os.PathLike,
    *,
    map_location: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    """Load a validated payload using PyTorch's restricted weights-only loader."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {target}")
    if not target.is_file():
        raise ValueError("checkpoint path must be a regular file")
    try:
        payload = torch.load(
            target,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as error:
        raise ValueError(
            f"failed to safely load training checkpoint {target}: "
            f"{type(error).__name__}: {error}"
        ) from error
    try:
        return TrainingCheckpoint.from_dict(payload)
    except Exception as error:
        raise ValueError(
            f"invalid training checkpoint {target}: {type(error).__name__}: {error}"
        ) from error
