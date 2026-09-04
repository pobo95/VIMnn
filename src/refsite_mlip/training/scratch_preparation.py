"""Read-only scratch reference and training-data preparation.

This module deliberately owns no model-construction or optimization logic.  It
materializes only immutable reference metadata and validated CPU data needed by
the later scratch execution milestone.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from refsite_mlip.data import (
    InMemoryStructureDataset,
    ReferenceStructureArtifact,
    StructureSample,
    TemplateRegistry,
    assemble_reference_template_from_artifact,
    build_reference_template_from_poscar,
    capture_reference_structure_artifact,
    collate_structure_samples,
)
from refsite_mlip.models import (
    EvaluationPolicy,
    ModelBundleTemplateBinding,
    TemplateExecutionContext,
)

if TYPE_CHECKING:
    from refsite_mlip.config import (
        InteractionRadiusConfig,
        ScratchModelSourceConfig,
        TrainingDataConfig,
        TrainingRunConfig,
        TrainingRuntimeConfig,
    )


SCRATCH_PREPARATION_CONVENTION_VERSION = "scratch_training_preparation_v1"
SCRATCH_DATA_MANIFEST_CONVENTION_VERSION = "scratch_training_data_manifest_v1"
SCRATCH_INPUT_FILE_DIGEST_CONVENTION_VERSION = "scratch_input_file_digests_v1"


def _freeze_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_plain(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_plain(item) for item in value)
    if type(value) is float and not math.isfinite(value):
        raise ValueError("scratch preparation metadata contains NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(
        "scratch preparation metadata contains non-plain "
        f"{type(value).__name__}"
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("scratch preparation metadata contains NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(
        "scratch preparation metadata contains non-plain "
        f"{type(value).__name__}"
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(scope: str, value: Mapping[str, Any]) -> str:
    payload = {"scope": scope, "value": _plain(value)}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(sorted(value.items())))


def _input_digest_error(
    reason_code: str,
    message: str,
    *,
    entry: Mapping[str, Any],
    config_path: Path | str | None,
    expected: Any = None,
    actual: Any = None,
    original_error: BaseException | None = None,
):
    # Delayed import preserves the existing config/training package dependency
    # direction while retaining TrainingRunConfigError's structured context.
    from refsite_mlip.config import training_run as run_config

    return run_config._error(
        reason_code,
        message,
        stage="scratch.input_digest",
        config_path=None if config_path is None else str(config_path),
        source_path=str(entry.get("runtime_path", "")) or None,
        field=entry.get("field"),
        split=entry.get("split"),
        template_id=entry.get("template_id"),
        expected=expected,
        actual=actual,
        original_reason_code=getattr(original_error, "reason_code", None),
        original_error=original_error,
    )


def _regular_file_sha256(
    entry: Mapping[str, Any], *, config_path: Path | str | None
) -> str:
    """Hash one pinned regular file without following a replacement symlink."""

    path = Path(str(entry["runtime_path"]))
    try:
        before = os.lstat(path)
    except FileNotFoundError as error:
        raise _input_digest_error(
            "INPUT_DIGEST_FILE_NOT_FOUND",
            "scratch input disappeared while verifying its raw digest",
            entry=entry,
            config_path=config_path,
            original_error=error,
        ) from error
    except OSError as error:
        raise _input_digest_error(
            "INPUT_DIGEST_READ_FAILED",
            "scratch input metadata could not be read for raw digest verification",
            entry=entry,
            config_path=config_path,
            original_error=error,
        ) from error
    if stat.S_ISLNK(before.st_mode):
        raise _input_digest_error(
            "INPUT_DIGEST_SYMLINK_REJECTED",
            "scratch input digest verification refuses symbolic links",
            entry=entry,
            config_path=config_path,
            actual=str(path),
        )
    if not stat.S_ISREG(before.st_mode):
        raise _input_digest_error(
            "INPUT_DIGEST_NOT_REGULAR_FILE",
            "scratch input digest verification requires a regular file",
            entry=entry,
            config_path=config_path,
            actual=str(path),
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise _input_digest_error(
            "INPUT_DIGEST_READ_FAILED",
            "scratch input could not be opened without following symlinks",
            entry=entry,
            config_path=config_path,
            original_error=error,
        ) from error

    digest = hashlib.sha256()
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise _input_digest_error(
                "INPUT_DIGEST_NOT_REGULAR_FILE",
                "opened scratch input is not a regular file",
                entry=entry,
                config_path=config_path,
                actual=str(path),
            )
        if (opened_before.st_dev, opened_before.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise _input_digest_error(
                "INPUT_DIGEST_FILE_CHANGED",
                "scratch input path changed while opening it for digest verification",
                entry=entry,
                config_path=config_path,
                actual=str(path),
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
    except OSError as error:
        raise _input_digest_error(
            "INPUT_DIGEST_READ_FAILED",
            "scratch input bytes could not be read for raw digest verification",
            entry=entry,
            config_path=config_path,
            original_error=error,
        ) from error
    finally:
        os.close(descriptor)

    try:
        after = os.lstat(path)
    except OSError as error:
        raise _input_digest_error(
            "INPUT_DIGEST_FILE_CHANGED",
            "scratch input path changed during raw digest verification",
            entry=entry,
            config_path=config_path,
            original_error=error,
        ) from error
    if stat.S_ISLNK(after.st_mode):
        raise _input_digest_error(
            "INPUT_DIGEST_SYMLINK_REJECTED",
            "scratch input became a symbolic link during digest verification",
            entry=entry,
            config_path=config_path,
            actual=str(path),
        )
    identity = (opened_before.st_dev, opened_before.st_ino)
    if (
        not stat.S_ISREG(after.st_mode)
        or identity != (opened_after.st_dev, opened_after.st_ino)
        or identity != (after.st_dev, after.st_ino)
        or opened_before.st_size != opened_after.st_size
        or opened_before.st_mtime_ns != opened_after.st_mtime_ns
    ):
        raise _input_digest_error(
            "INPUT_DIGEST_FILE_CHANGED",
            "scratch input changed during raw digest verification",
            entry=entry,
            config_path=config_path,
            actual=str(path),
        )
    return digest.hexdigest()


def _input_file_specs(
    config: Any,
    *,
    config_path: Path | None,
    reference_paths: tuple[Path, ...],
    train_paths: tuple[Path, ...],
    validation_paths: tuple[Path, ...],
) -> dict[str, dict[str, Any]]:
    source = config.model_source
    specs: dict[str, dict[str, Any]] = {}

    def add(
        label: str,
        *,
        role: str,
        configured_path: str,
        runtime_path: Path,
        field: str,
        source_index: int | None = None,
        split: str | None = None,
        template_id: str | None = None,
    ) -> None:
        specs[label] = {
            "label": label,
            "role": role,
            "source_index": source_index,
            "split": split,
            "template_id": template_id,
            "field": field,
            "configured_path": configured_path,
            "runtime_path": str(runtime_path),
        }

    if config_path is not None:
        add(
            "config",
            role="config",
            configured_path=str(config_path),
            runtime_path=config_path,
            field="config",
        )
    for index, (template, path) in enumerate(
        zip(source.reference_templates, reference_paths)
    ):
        add(
            f"reference_poscar[{index:06d}]",
            role="reference_poscar",
            configured_path=template.poscar_path,
            runtime_path=path,
            field=f"model_source.reference_templates[{index}].poscar_path",
            source_index=index,
            template_id=template.template_id,
        )
    for split, sources, paths in (
        ("train", config.data.train, train_paths),
        ("validation", config.data.validation, validation_paths),
    ):
        for index, (source_config, path) in enumerate(zip(sources, paths)):
            add(
                f"{split}[{index:06d}]",
                role="extxyz",
                configured_path=source_config.path,
                runtime_path=path,
                field=f"data.{split}[{index}].path",
                source_index=index,
                split=split,
            )
    return dict(sorted(specs.items()))


def _capture_input_file_digests(
    specs: Mapping[str, Mapping[str, Any]], *, config_path: Path | None
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for label, spec in sorted(specs.items()):
        entry = dict(spec)
        entry["sha256"] = _regular_file_sha256(
            entry, config_path=config_path
        )
        files[label] = entry
    return {
        "convention_version": SCRATCH_INPUT_FILE_DIGEST_CONVENTION_VERSION,
        "path_kind": "runtime_location_not_semantic_fingerprint",
        "files": files,
    }


class _ReadOnlyTemplateRegistry(TemplateRegistry):
    """A TemplateRegistry snapshot whose public mutation hook is sealed."""

    def __init__(self, templates: Mapping[str, Any]) -> None:
        super().__init__()
        self._sealed = False
        for template_id in sorted(templates):
            super().add(templates[template_id])
        self._sealed = True

    def add(self, template: Any) -> None:
        if self._sealed:
            raise TypeError("prepared TemplateRegistry is read-only")
        super().add(template)


def _batch_plan(
    samples: tuple[StructureSample, ...], batch_size: int
) -> list[dict[str, Any]]:
    return [
        {
            "batch_index": batch_index,
            "start": start,
            "stop": min(start + batch_size, len(samples)),
            "sample_ids": [sample.sample_id for sample in samples[start : start + batch_size]],
            "template_ids": [
                sample.template_id for sample in samples[start : start + batch_size]
            ],
        }
        for batch_index, start in enumerate(range(0, len(samples), batch_size))
    ]


def _validate_batch_plan(
    samples: tuple[StructureSample, ...],
    *,
    batch_size: int,
    registry: TemplateRegistry,
) -> None:
    """Exercise the existing deterministic collation contract without a model."""

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        batch = collate_structure_samples(chunk, registry)
        if batch.sample_ids != tuple(sample.sample_id for sample in chunk):
            raise ValueError("scratch batch collation changed sample ordering")


def _assignment_kind(selection_rule: str) -> str:
    return {
        "configured_exact_template_id": "exact_template_id",
        "frame_exact_template_key": "exact_template_key",
        "unique_full_domain_match": "unique_automatic",
    }[selection_rule]


def _split_manifest(
    samples: tuple[StructureSample, ...],
    assignments: tuple[Any, ...],
    templates: Mapping[str, Any],
    *,
    split: str,
    batch_size: int,
) -> dict[str, Any]:
    by_sample = {assignment.sample_id: assignment for assignment in assignments}
    entries: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        template = templates[sample.template_id]
        validation = template.validate_structure(
            sample.atomic_numbers,
            cell=sample.cell,
            pbc=sample.pbc,
            sample_id=sample.sample_id,
        )
        assignment = by_sample[sample.sample_id]
        entries.append(
            {
                "frame_index": index,
                "source_index": assignment.source_index,
                "source_frame_index": assignment.frame_index,
                "sample_id": sample.sample_id,
                "template_id": sample.template_id,
                "template_fingerprint": template.fingerprint,
                "num_atoms": validation.num_atoms,
                "num_sites": template.topology.num_sites,
                "vacancy_mass": validation.vacancy_mass,
                "composition": list(validation.composition),
                "maximum_strain_seen": validation.maximum_strain_seen,
                "template_assignment": {
                    "kind": _assignment_kind(assignment.selection_rule),
                    "selection_rule": assignment.selection_rule,
                    "compatible_template_ids": list(
                        assignment.compatible_template_ids
                    ),
                    "rejected_templates": [
                        {"template_id": template_id, "reason": reason}
                        for template_id, reason in assignment.rejected_templates
                    ],
                },
            }
        )
    return {
        "split": split,
        "frame_count": len(samples),
        "batch_count": math.ceil(len(samples) / batch_size),
        "samples": entries,
        "batches": _batch_plan(samples, batch_size),
    }


def _observed_species(samples: tuple[StructureSample, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(value)
                for sample in samples
                for value in sample.atomic_numbers.detach().cpu().tolist()
            }
        )
    )


@dataclass(frozen=True)
class ScratchTrainingPreparation:
    """Immutable ownership of read-only scratch references and input data."""

    config_fingerprint: str
    preparation_fingerprint: str
    config: TrainingRunConfig
    model_source: ScratchModelSourceConfig
    runtime: TrainingRuntimeConfig
    data: TrainingDataConfig
    radius_config: InteractionRadiusConfig
    resolved_device: str
    resolved_dtype: str
    train_samples: tuple[StructureSample, ...]
    validation_samples: tuple[StructureSample, ...]
    registry: TemplateRegistry
    structural_artifacts: Mapping[str, ReferenceStructureArtifact]
    template_contexts: Mapping[str, TemplateExecutionContext]
    evaluation_policies: Mapping[str, EvaluationPolicy | None]
    template_fingerprints: Mapping[str, Any]
    train_semantic_digest: str
    validation_semantic_digest: str
    data_manifest: Mapping[str, Any]
    species_vocabulary: tuple[int, ...]
    observed_species_vocabulary: tuple[int, ...]
    train_label_statistics: Mapping[str, Any]
    validation_label_statistics: Mapping[str, Any]
    train_composition_statistics: Any
    validation_composition_statistics: Any
    baseline_preflight: Mapping[str, Any]
    configured_paths: Mapping[str, Any]
    runtime_paths: Mapping[str, Any]
    input_file_digests: Mapping[str, Any]
    training_configuration: Mapping[str, Any]
    training_executed: bool = False
    scratch_execution_implemented: bool = True

    def __post_init__(self) -> None:
        if type(self.training_executed) is not bool or self.training_executed:
            raise ValueError("scratch preparation cannot execute training")
        if (
            type(self.scratch_execution_implemented) is not bool
            or not self.scratch_execution_implemented
        ):
            raise ValueError("prepared scratch execution must be enabled")
        for name in ("config_fingerprint", "preparation_fingerprint"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 string")
        if getattr(self.config, "config_fingerprint", None) != self.config_fingerprint:
            raise ValueError("prepared config fingerprint differs from its metadata")
        if not isinstance(self.registry, TemplateRegistry):
            raise TypeError("registry must be a TemplateRegistry")
        object.__setattr__(self, "train_samples", tuple(self.train_samples))
        object.__setattr__(self, "validation_samples", tuple(self.validation_samples))
        object.__setattr__(
            self, "structural_artifacts", _immutable_mapping(self.structural_artifacts)
        )
        object.__setattr__(
            self, "template_contexts", _immutable_mapping(self.template_contexts)
        )
        object.__setattr__(
            self, "evaluation_policies", _immutable_mapping(self.evaluation_policies)
        )
        for name in (
            "template_fingerprints",
            "data_manifest",
            "train_label_statistics",
            "validation_label_statistics",
            "baseline_preflight",
            "configured_paths",
            "runtime_paths",
            "input_file_digests",
            "training_configuration",
        ):
            object.__setattr__(self, name, _freeze_plain(getattr(self, name)))
        object.__setattr__(
            self,
            "train_composition_statistics",
            _freeze_plain(self.train_composition_statistics),
        )
        object.__setattr__(
            self,
            "validation_composition_statistics",
            _freeze_plain(self.validation_composition_statistics),
        )
        object.__setattr__(
            self, "species_vocabulary", tuple(int(value) for value in self.species_vocabulary)
        )
        object.__setattr__(
            self,
            "observed_species_vocabulary",
            tuple(int(value) for value in self.observed_species_vocabulary),
        )

    def to_dict(self) -> dict[str, Any]:
        train_manifest = self.data_manifest["train"]
        validation_manifest = self.data_manifest["validation"]
        train_counts = dict(
            sorted(Counter(sample.template_id for sample in self.train_samples).items())
        )
        validation_counts = dict(
            sorted(
                Counter(sample.template_id for sample in self.validation_samples).items()
            )
        )
        return {
            "status": "scratch_preflight_ready",
            "training_executed": False,
            "schema_version": "refsite_training_run_config_v2",
            "config_fingerprint": self.config_fingerprint,
            "preparation_fingerprint": self.preparation_fingerprint,
            "execution": {
                "implemented": True,
                "reason_code": None,
            },
            "model_source": self.model_source.to_dict(),
            "registry_fingerprint": self.registry.fingerprint,
            "data": {
                "train": {
                    "semantic_digest": self.train_semantic_digest,
                    "frame_count": len(self.train_samples),
                    "batch_count": train_manifest["batch_count"],
                    "template_frame_counts": train_counts,
                    "composition_statistics": _plain(
                        self.train_composition_statistics
                    ),
                    "label_statistics": _plain(self.train_label_statistics),
                },
                "validation": {
                    "semantic_digest": self.validation_semantic_digest,
                    "frame_count": len(self.validation_samples),
                    "batch_count": validation_manifest["batch_count"],
                    "template_frame_counts": validation_counts,
                    "composition_statistics": _plain(
                        self.validation_composition_statistics
                    ),
                    "label_statistics": _plain(
                        self.validation_label_statistics
                    ),
                },
            },
            "data_manifest": _plain(self.data_manifest),
            "runtime": {
                "device": self.resolved_device,
                "dtype": self.resolved_dtype,
                "seed": self.runtime.seed,
                "configured_paths": _plain(self.configured_paths),
                "paths": _plain(self.runtime_paths),
                "input_file_digests": _plain(self.input_file_digests),
            },
            "radii": {
                "config": self.radius_config.to_dict(),
                "derived": self.radius_config.derived.to_dict(),
                "diagnostics": self.radius_config.derived.to_diagnostics_dict(),
            },
            "species_vocabulary": list(self.species_vocabulary),
            "observed_species_vocabulary": list(
                self.observed_species_vocabulary
            ),
            "template_fingerprints": _plain(self.template_fingerprints),
            "baseline_preflight": _plain(self.baseline_preflight),
            "training_configuration": _plain(self.training_configuration),
            "side_effects": {
                "initial_bundle_created": False,
                "model_parameters_created": False,
                "optimizer_created": False,
                "output_directory_created": False,
            },
        }


def _expected_input_file_specs(
    preparation: ScratchTrainingPreparation,
) -> tuple[dict[str, dict[str, Any]], Path | None]:
    context = {
        "runtime_path": str(
            preparation.runtime_paths.get("config") or "<in-memory-config>"
        ),
        "field": "input_file_digests",
        "split": None,
        "template_id": None,
    }
    try:
        config_text = preparation.runtime_paths["config"]
        config_path = None if config_text is None else Path(str(config_text))
        reference_values = tuple(preparation.runtime_paths["reference_poscars"])
        train_values = tuple(preparation.runtime_paths["train_inputs"])
        validation_values = tuple(preparation.runtime_paths["validation_inputs"])
        reference_paths = tuple(
            Path(str(value["path"])) for value in reference_values
        )
        train_paths = tuple(Path(str(value)) for value in train_values)
        validation_paths = tuple(Path(str(value)) for value in validation_values)
        if (
            len(reference_paths)
            != len(preparation.model_source.reference_templates)
            or len(train_paths) != len(preparation.data.train)
            or len(validation_paths) != len(preparation.data.validation)
        ):
            raise ValueError("runtime input path counts differ from the config")
        for index, (runtime, source) in enumerate(
            zip(reference_values, preparation.model_source.reference_templates)
        ):
            if runtime["template_id"] != source.template_id:
                raise ValueError(
                    "runtime reference template ordering differs at "
                    f"index {index}"
                )
        specs = _input_file_specs(
            preparation.config,
            config_path=config_path,
            reference_paths=reference_paths,
            train_paths=train_paths,
            validation_paths=validation_paths,
        )
    except Exception as error:
        raise _input_digest_error(
            "INVALID_INPUT_DIGEST_METADATA",
            "scratch input digest paths are inconsistent with the preparation",
            entry=context,
            config_path=preparation.runtime_paths.get("config"),
            original_error=error,
        ) from error
    return specs, config_path


def verify_scratch_preparation_input_digests(
    preparation: ScratchTrainingPreparation,
) -> None:
    """Verify every prepared config/POSCAR/extxyz byte snapshot in place.

    Resolved paths are runtime identity only.  They are deliberately excluded
    from the semantic preparation fingerprint so relocation does not change
    model/data semantics, while a fresh-run TOCTOU gate can still pin the exact
    files inspected by preflight.
    """

    if not isinstance(preparation, ScratchTrainingPreparation):
        raise TypeError("preparation must be a ScratchTrainingPreparation")
    expected_specs, config_path = _expected_input_file_specs(preparation)
    metadata = preparation.input_file_digests
    generic = {
        "runtime_path": "<input-file-digest-metadata>",
        "field": "input_file_digests",
        "split": None,
        "template_id": None,
    }
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "convention_version",
        "path_kind",
        "files",
    }:
        raise _input_digest_error(
            "INVALID_INPUT_DIGEST_METADATA",
            "scratch input digest metadata has invalid top-level keys",
            entry=generic,
            config_path=config_path,
        )
    if (
        metadata["convention_version"]
        != SCRATCH_INPUT_FILE_DIGEST_CONVENTION_VERSION
        or metadata["path_kind"]
        != "runtime_location_not_semantic_fingerprint"
        or not isinstance(metadata["files"], Mapping)
    ):
        raise _input_digest_error(
            "INVALID_INPUT_DIGEST_METADATA",
            "scratch input digest metadata has an invalid convention or file mapping",
            entry=generic,
            config_path=config_path,
        )
    files = metadata["files"]
    if set(files) != set(expected_specs):
        raise _input_digest_error(
            "INVALID_INPUT_DIGEST_METADATA",
            "scratch input digest labels do not cover the configured input files",
            entry=generic,
            config_path=config_path,
            expected=tuple(sorted(expected_specs)),
            actual=tuple(sorted(files)),
        )

    expected_entry_keys = {
        "label",
        "role",
        "source_index",
        "split",
        "template_id",
        "field",
        "configured_path",
        "runtime_path",
        "sha256",
    }
    for label, expected_spec in sorted(expected_specs.items()):
        entry = files[label]
        if not isinstance(entry, Mapping) or set(entry) != expected_entry_keys:
            raise _input_digest_error(
                "INVALID_INPUT_DIGEST_METADATA",
                "scratch input digest entry has invalid keys",
                entry=(entry if isinstance(entry, Mapping) else generic),
                config_path=config_path,
                expected=tuple(sorted(expected_entry_keys)),
                actual=(
                    type(entry).__name__
                    if not isinstance(entry, Mapping)
                    else tuple(sorted(entry))
                ),
            )
        actual_spec = {key: entry[key] for key in expected_spec}
        if actual_spec != expected_spec:
            raise _input_digest_error(
                "INVALID_INPUT_DIGEST_METADATA",
                "scratch input digest entry differs from resolved config/runtime paths",
                entry=entry,
                config_path=config_path,
                expected=expected_spec,
                actual=actual_spec,
            )
        expected_digest = entry["sha256"]
        if (
            type(expected_digest) is not str
            or len(expected_digest) != 64
            or any(value not in "0123456789abcdef" for value in expected_digest)
        ):
            raise _input_digest_error(
                "INVALID_INPUT_DIGEST_METADATA",
                "scratch input digest must be a lowercase SHA-256 string",
                entry=entry,
                config_path=config_path,
                actual=expected_digest,
            )
        actual_digest = _regular_file_sha256(entry, config_path=config_path)
        if actual_digest != expected_digest:
            raise _input_digest_error(
                "INPUT_DIGEST_MISMATCH",
                "scratch input bytes changed after full preflight",
                entry=entry,
                config_path=config_path,
                expected=expected_digest,
                actual=actual_digest,
            )


def prepare_scratch_training_run(
    config: Any,
    *,
    base_directory: str | os.PathLike[str] | None = None,
) -> ScratchTrainingPreparation:
    """Perform complete scratch reference/data preflight without side effects."""

    # Delayed imports avoid a package cycle: training_run imports the public
    # training config classes before this module is exported by training.
    from refsite_mlip.config.model_source import ScratchModelSourceConfig
    from refsite_mlip.config.radii import (
        RadiusConfigError,
        validate_radius_artifact_compatibility,
        validate_radius_model_compatibility,
    )
    from refsite_mlip.config import training_run as run_config

    if not isinstance(config, run_config.TrainingRunConfig):
        raise TypeError("config must be a TrainingRunConfig")
    run_config.validate_training_run_config(config)
    # Own a canonical snapshot instead of retaining the caller's config.
    # Several established config objects contain tensors; their constructors
    # snapshot those tensors, but merely storing ``config`` here would still
    # let a later caller-side tensor mutation invalidate prepared metadata.
    # Runtime path anchors are intentionally outside canonical serialization,
    # so restore them only after the semantic round trip.
    config = replace(
        run_config.TrainingRunConfig.from_dict(config.to_dict()),
        source_path=config.source_path,
        output_directory_base=config.output_directory_base,
    )
    source = config.model_source
    if not isinstance(source, ScratchModelSourceConfig):
        raise run_config._error(
            "INVALID_MODEL_SOURCE_KIND",
            "scratch preparation requires a schema-v2 scratch model source",
            stage="scratch.config",
            field="model_source.kind",
            actual=getattr(source, "kind", None),
        )

    base, config_path = run_config._base_directory(config, base_directory)
    resolved_device = run_config._preflight_device(config.runtime)
    reference_paths = tuple(
        run_config._resolve_existing_file(
            item.poscar_path,
            base=base,
            field_name=(
                f"model_source.reference_templates[{index}].poscar_path"
            ),
            config_path=config_path,
        )
        for index, item in enumerate(source.reference_templates)
    )
    train_paths = tuple(
        run_config._resolve_existing_file(
            item.path,
            base=base,
            field_name=f"data.train[{index}].path",
            config_path=config_path,
        )
        for index, item in enumerate(config.data.train)
    )
    validation_paths = tuple(
        run_config._resolve_existing_file(
            item.path,
            base=base,
            field_name=f"data.validation[{index}].path",
            config_path=config_path,
        )
        for index, item in enumerate(config.data.validation)
    )
    protected: list[tuple[str, Path]] = []
    protected.extend(
        (f"reference[{index}]", path) for index, path in enumerate(reference_paths)
    )
    protected.extend(
        (f"data.train[{index}]", path) for index, path in enumerate(train_paths)
    )
    protected.extend(
        (f"data.validation[{index}]", path)
        for index, path in enumerate(validation_paths)
    )
    if config_path is not None:
        protected.append(("config", config_path))
    output_base = (
        base
        if config.output_directory_base is None
        else Path(config.output_directory_base)
    )
    output_path = run_config._resolve_output_directory(
        config.output_directory,
        base=output_base,
        config_path=config_path,
        protected=tuple(protected),
    )
    input_file_digests = _capture_input_file_digests(
        _input_file_specs(
            config,
            config_path=config_path,
            reference_paths=reference_paths,
            train_paths=train_paths,
            validation_paths=validation_paths,
        ),
        config_path=config_path,
    )

    try:
        validate_radius_model_compatibility(config.radii, source.potential)
    except RadiusConfigError as error:
        mismatch = error.mismatches[0] if error.mismatches else (None, None, None)
        raise run_config._error(
            error.reason_code,
            "scratch PotentialConfig is incompatible with interaction radii",
            stage="scratch.radii.model",
            config_path=None if config_path is None else str(config_path),
            field=mismatch[0],
            expected=mismatch[1],
            actual=mismatch[2],
            original_reason_code=error.reason_code,
            original_error=error,
        ) from error

    registry = TemplateRegistry()
    templates: dict[str, Any] = {}
    artifacts: dict[str, ReferenceStructureArtifact] = {}
    contexts: dict[str, TemplateExecutionContext] = {}
    policies: dict[str, EvaluationPolicy | None] = {}
    template_fingerprints: dict[str, Any] = {}
    for index, (template_source, poscar_path) in enumerate(
        zip(source.reference_templates, reference_paths)
    ):
        template_id = template_source.template_id
        try:
            built = build_reference_template_from_poscar(
                poscar_path,
                config=template_source.builder,
                phase_specification=template_source.phase_specification,
            )
        except Exception as error:
            raise run_config._error(
                getattr(error, "reason_code", "REFERENCE_BUILD_FAILED"),
                "scratch POSCAR reference construction failed: "
                f"{type(error).__name__}: {error}",
                stage="scratch.reference.build",
                config_path=None if config_path is None else str(config_path),
                source_path=str(poscar_path),
                field=f"model_source.reference_templates[{index}]",
                template_id=template_id,
                original_reason_code=getattr(error, "reason_code", None),
                original_error=error,
            ) from error
        try:
            artifact = capture_reference_structure_artifact(built)
            validate_radius_artifact_compatibility(config.radii, artifact)
        except Exception as error:
            mismatch = getattr(error, "mismatches", ())
            first = mismatch[0] if mismatch else (None, None, None)
            raise run_config._error(
                getattr(error, "reason_code", "REFERENCE_ARTIFACT_FAILED"),
                "scratch structural artifact validation failed: "
                f"{type(error).__name__}: {error}",
                stage="scratch.reference.artifact",
                config_path=None if config_path is None else str(config_path),
                source_path=str(poscar_path),
                field=first[0] or f"model_source.reference_templates[{index}]",
                template_id=template_id,
                expected=first[1],
                actual=first[2],
                original_reason_code=getattr(error, "reason_code", None),
                original_error=error,
            ) from error
        try:
            assembled = assemble_reference_template_from_artifact(
                artifact,
                phase_specification=template_source.phase_specification,
            )
        except Exception as error:
            raise run_config._error(
                getattr(error, "reason_code", "PHASE_ASSEMBLY_FAILED"),
                "scratch phase specification could not be combined with the "
                f"structural artifact: {type(error).__name__}: {error}",
                stage="scratch.reference.phase",
                config_path=None if config_path is None else str(config_path),
                source_path=str(poscar_path),
                field=(
                    f"model_source.reference_templates[{index}].phase_specification"
                ),
                template_id=template_id,
                original_reason_code=getattr(error, "reason_code", None),
                original_error=error,
            ) from error
        if assembled.fingerprint != built.template.fingerprint:
            raise run_config._error(
                "TEMPLATE_REASSEMBLY_MISMATCH",
                "artifact plus phase did not reproduce the direct builder template",
                stage="scratch.reference.phase",
                config_path=None if config_path is None else str(config_path),
                source_path=str(poscar_path),
                template_id=template_id,
                expected=built.template.fingerprint,
                actual=assembled.fingerprint,
            )

        policy = template_source.evaluation_policy
        if policy is not None:
            try:
                policy = EvaluationPolicy.from_dict(policy.to_dict())
                policy.validate_fingerprint()
            except Exception as error:
                raise run_config._error(
                    getattr(error, "reason_code", "POLICY_CONTENT_MISMATCH"),
                    "scratch evaluation policy content is invalid",
                    stage="scratch.reference.policy",
                    config_path=None if config_path is None else str(config_path),
                    source_path=str(poscar_path),
                    field=(
                        f"model_source.reference_templates[{index}].evaluation_policy"
                    ),
                    template_id=template_id,
                    original_reason_code=getattr(error, "reason_code", None),
                    original_error=error,
                ) from error
            if policy.template_id != template_id:
                raise run_config._error(
                    "POLICY_TEMPLATE_ID_MISMATCH",
                    "evaluation policy template_id differs from the built template",
                    stage="scratch.reference.policy",
                    config_path=None if config_path is None else str(config_path),
                    source_path=str(poscar_path),
                    template_id=template_id,
                    expected=template_id,
                    actual=policy.template_id,
                )
            if policy.template_fingerprint != assembled.fingerprint:
                raise run_config._error(
                    "POLICY_TEMPLATE_FINGERPRINT_MISMATCH",
                    "evaluation policy fingerprint does not bind the built template",
                    stage="scratch.reference.policy",
                    config_path=None if config_path is None else str(config_path),
                    source_path=str(poscar_path),
                    template_id=template_id,
                    expected=assembled.fingerprint,
                    actual=policy.template_fingerprint,
                )

        try:
            binding = ModelBundleTemplateBinding(
                template_id=template_id,
                structural_artifact=artifact,
                phase_specification=template_source.phase_specification,
                full_template_fingerprint=assembled.fingerprint,
                evaluation_policy=policy,
                approval_status=template_source.phase_specification.approval_status,
                provenance={"source_kind": "scratch_preflight"},
            )
            binding.validate()
            registry.add(assembled)
            context = TemplateExecutionContext.from_reference_template(
                assembled, avg_num_neighbors=artifact.avg_num_neighbors
            )
        except Exception as error:
            raise run_config._error(
                getattr(error, "reason_code", "TEMPLATE_CONTEXT_FAILED"),
                "scratch template execution context validation failed: "
                f"{type(error).__name__}: {error}",
                stage="scratch.reference.context",
                config_path=None if config_path is None else str(config_path),
                source_path=str(poscar_path),
                template_id=template_id,
                original_reason_code=getattr(error, "reason_code", None),
                original_error=error,
            ) from error

        templates[template_id] = assembled
        artifacts[template_id] = artifact
        contexts[template_id] = context
        policies[template_id] = policy
        artifact_diagnostics = artifact.diagnostics.to_dict()
        template_fingerprints[template_id] = {
            "structural_artifact_fingerprint": artifact.structural_fingerprint,
            "full_template_fingerprint": assembled.fingerprint,
            "phase_specification_fingerprint": (
                run_config._phase_specification_fingerprint(
                    template_source.phase_specification
                )
            ),
            "binding_fingerprint": binding.binding_fingerprint,
            "evaluation_policy_fingerprint": (
                None if policy is None else policy.content_fingerprint
            ),
            "evaluation_policy_present": policy is not None,
            "phase_approval_status": template_source.phase_specification.approval_status,
            "phase_rank": int(
                torch.linalg.matrix_rank(
                    template_source.phase_specification.modes[:3].to(torch.float64)
                )
            ),
            "num_sites": artifact.diagnostics.num_sites,
            "artifact_diagnostics": artifact_diagnostics,
        }

    candidate_ids = tuple(sorted(templates))
    train_samples, train_assignments = run_config._load_split_with_assignments(
        config.data.train,
        train_paths,
        split="train",
        registry=registry,
        dtype=torch.float64,
        config_path=config_path,
        automatic_template_ids=candidate_ids,
    )
    validation_samples, validation_assignments = (
        run_config._load_split_with_assignments(
            config.data.validation,
            validation_paths,
            split="validation",
            registry=registry,
            dtype=torch.float64,
            config_path=config_path,
            automatic_template_ids=candidate_ids,
        )
    )
    if set(sample.sample_id for sample in train_samples) & set(
        sample.sample_id for sample in validation_samples
    ):
        raise run_config._error(
            "CROSS_SPLIT_SAMPLE_ID_COLLISION",
            "train and validation sample ID namespaces overlap",
            stage="data.identity",
            config_path=None if config_path is None else str(config_path),
        )
    # The dataset constructor is a final parameter-free registry/domain check.
    try:
        InMemoryStructureDataset(train_samples, registry)
        InMemoryStructureDataset(validation_samples, registry)
    except Exception as error:
        raise run_config._error(
            getattr(error, "reason_code", "DATASET_PREFLIGHT_FAILED"),
            "scratch in-memory dataset validation failed: "
            f"{type(error).__name__}: {error}",
            stage="data.dataset",
            config_path=None if config_path is None else str(config_path),
            original_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error

    train_species = _observed_species(train_samples)
    validation_species = _observed_species(validation_samples)
    validation_only_species = tuple(
        sorted(set(validation_species) - set(train_species))
    )
    if validation_only_species:
        raise run_config._error(
            "VALIDATION_SPECIES_NOT_IN_TRAIN",
            "validation contains species that are absent from training data",
            stage="data.species",
            config_path=None if config_path is None else str(config_path),
            expected=train_species,
            actual=validation_only_species,
        )
    observed_species = tuple(sorted(set(train_species) | set(validation_species)))
    configured_species = tuple(source.potential.species_vocabulary)
    if not set(observed_species).issubset(set(configured_species)):
        raise run_config._error(
            "DATA_SPECIES_VOCABULARY_MISMATCH",
            "dataset species are not included in the scratch PotentialConfig",
            stage="data.species",
            config_path=None if config_path is None else str(config_path),
            expected=configured_species,
            actual=observed_species,
        )

    run_config._validate_supervision(train_samples, validation_samples, config)
    try:
        _validate_batch_plan(
            train_samples, batch_size=config.data.batch_size, registry=registry
        )
        _validate_batch_plan(
            validation_samples,
            batch_size=config.data.effective_validation_batch_size,
            registry=registry,
        )
    except Exception as error:
        raise run_config._error(
            getattr(error, "reason_code", "BATCH_PREFLIGHT_FAILED"),
            "deterministic scratch batch collation failed: "
            f"{type(error).__name__}: {error}",
            stage="data.batch_plan",
            config_path=None if config_path is None else str(config_path),
            original_reason_code=getattr(error, "reason_code", None),
            original_error=error,
        ) from error
    baseline = run_config._baseline_preflight(
        train_samples, configured_species, config.baseline
    )
    train_digest = run_config._split_digest(train_samples, templates, split="train")
    validation_digest = run_config._split_digest(
        validation_samples, templates, split="validation"
    )
    train_manifest = _split_manifest(
        train_samples,
        train_assignments,
        templates,
        split="train",
        batch_size=config.data.batch_size,
    )
    validation_manifest = _split_manifest(
        validation_samples,
        validation_assignments,
        templates,
        split="validation",
        batch_size=config.data.effective_validation_batch_size,
    )
    manifest_payload = {
        "convention_version": SCRATCH_DATA_MANIFEST_CONVENTION_VERSION,
        "train_semantic_digest": train_digest,
        "validation_semantic_digest": validation_digest,
        "observed_species": {
            "train": list(train_species),
            "validation": list(validation_species),
            "union": list(observed_species),
            "configured": list(configured_species),
        },
        "train": train_manifest,
        "validation": validation_manifest,
    }
    manifest_payload["fingerprint"] = _fingerprint(
        SCRATCH_DATA_MANIFEST_CONVENTION_VERSION, manifest_payload
    )

    configured_paths = {
        "output_directory": config.output_directory,
        "train_inputs": [item.path for item in config.data.train],
        "validation_inputs": [item.path for item in config.data.validation],
        "reference_poscars": [
            {"template_id": item.template_id, "path": item.poscar_path}
            for item in source.reference_templates
        ],
        "path_kind": "original_config_expression_in_semantic_fingerprint",
    }
    runtime_paths = {
        "base_directory": str(base),
        "config": None if config_path is None else str(config_path),
        "output_directory": str(output_path),
        "train_inputs": [str(path) for path in train_paths],
        "validation_inputs": [str(path) for path in validation_paths],
        "reference_poscars": [
            {"template_id": item.template_id, "path": str(path)}
            for item, path in zip(source.reference_templates, reference_paths)
        ],
        "path_kind": "runtime_location_not_semantic_fingerprint",
    }
    preparation_semantics = {
        "convention_version": SCRATCH_PREPARATION_CONVENTION_VERSION,
        "config_fingerprint": config.config_fingerprint,
        "registry_fingerprint": registry.fingerprint,
        "template_fingerprints": template_fingerprints,
        "train_semantic_digest": train_digest,
        "validation_semantic_digest": validation_digest,
        "data_manifest_fingerprint": manifest_payload["fingerprint"],
    }
    preparation_fingerprint = _fingerprint(
        SCRATCH_PREPARATION_CONVENTION_VERSION, preparation_semantics
    )
    prepared_registry = _ReadOnlyTemplateRegistry(templates)
    preparation = ScratchTrainingPreparation(
        config_fingerprint=config.config_fingerprint,
        preparation_fingerprint=preparation_fingerprint,
        config=config,
        model_source=source,
        runtime=config.runtime,
        data=config.data,
        radius_config=config.radii,
        resolved_device=resolved_device,
        resolved_dtype=config.runtime.dtype,
        train_samples=train_samples,
        validation_samples=validation_samples,
        registry=prepared_registry,
        structural_artifacts=artifacts,
        template_contexts=contexts,
        evaluation_policies=policies,
        template_fingerprints=template_fingerprints,
        train_semantic_digest=train_digest,
        validation_semantic_digest=validation_digest,
        data_manifest=manifest_payload,
        species_vocabulary=configured_species,
        observed_species_vocabulary=observed_species,
        train_label_statistics=run_config._label_statistics(train_samples),
        validation_label_statistics=run_config._label_statistics(validation_samples),
        train_composition_statistics=run_config._composition_statistics(train_samples),
        validation_composition_statistics=run_config._composition_statistics(
            validation_samples
        ),
        baseline_preflight=baseline,
        configured_paths=configured_paths,
        runtime_paths=runtime_paths,
        input_file_digests=input_file_digests,
        training_configuration=run_config._training_configuration_metadata(config),
        training_executed=False,
        scratch_execution_implemented=True,
    )
    # Detect files changed while the relatively expensive builder/data
    # preflight was running, not only changes observed later by training.
    verify_scratch_preparation_input_digests(preparation)
    return preparation


__all__ = [
    "SCRATCH_DATA_MANIFEST_CONVENTION_VERSION",
    "SCRATCH_INPUT_FILE_DIGEST_CONVENTION_VERSION",
    "SCRATCH_PREPARATION_CONVENTION_VERSION",
    "ScratchTrainingPreparation",
    "prepare_scratch_training_run",
    "verify_scratch_preparation_input_digests",
]
