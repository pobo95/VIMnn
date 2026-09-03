"""Portable-bundle prediction from ordered extxyz frames."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch

from refsite_mlip.data import STRESS_SIGN, STRESS_VOIGT_ORDER, StructureSample
from refsite_mlip.inference import (
    PredictorConfig,
    PredictorError,
    load_reference_site_predictor,
)
from refsite_mlip.models import ModelBundleError
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from .errors import CLIError
from .inspect_bundle import render_json as _render_json


PREDICTION_PROPERTIES = ("energy", "forces", "stress")
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)\Z")
_RESULT_INFO_KEYS = frozenset({"energy", "free_energy", "forces", "stress"})
_RESULT_ARRAY_KEYS = frozenset({"energy", "free_energy", "forces", "stress"})


def normalize_properties(values: str | Sequence[str]) -> tuple[str, ...]:
    """Validate properties and return their canonical output order.

    Potential energy is intrinsic to every predictor call and every output
    frame, so it is included even if a caller lists derivative properties only.
    """

    if isinstance(values, str):
        items = tuple(item.strip() for item in values.split(","))
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = tuple(values)
    else:
        raise TypeError("properties must be a comma-separated string or sequence")
    if not items or any(type(item) is not str or not item for item in items):
        raise ValueError("properties must be a nonempty comma-separated list")
    if len(set(items)) != len(items):
        raise ValueError("properties must not contain duplicates")
    unknown = sorted(set(items) - set(PREDICTION_PROPERTIES))
    if unknown:
        raise ValueError(f"unknown prediction properties: {unknown}")
    selected = set(items)
    selected.add("energy")
    return tuple(name for name in PREDICTION_PROPERTIES if name in selected)


def solver_path_from_name(value: str) -> str:
    paths = {
        "train-fixed": TRAIN_FIXED,
        "eval-adaptive": EVAL_ADAPTIVE,
        TRAIN_FIXED: TRAIN_FIXED,
        EVAL_ADAPTIVE: EVAL_ADAPTIVE,
    }
    if value not in paths:
        raise ValueError("solver must be train-fixed or eval-adaptive")
    return paths[value]


def solver_name(path: str) -> str:
    if path == TRAIN_FIXED:
        return "train-fixed"
    if path == EVAL_ADAPTIVE:
        return "eval-adaptive"
    raise ValueError("unsupported solver path")


@dataclass(frozen=True)
class ExtXYZPredictionConfig:
    """Validated controls for one stateless extxyz prediction command."""

    bundle_path: str
    input_path: str
    output_path: str
    index: str = ":"
    template_id: str | None = None
    template_key: str | None = None
    solver_path: str = TRAIN_FIXED
    properties: tuple[str, ...] = ("energy", "forces")
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    batch_size: int = 8
    overwrite: bool = False

    def __post_init__(self) -> None:
        for name in ("bundle_path", "input_path", "output_path"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError(f"{name} must be a nonempty path")
            object.__setattr__(self, name, str(value))
        if not isinstance(self.index, str) or not self.index:
            raise ValueError("index must be a nonempty ASE index expression")
        if self.template_id is not None and (
            not isinstance(self.template_id, str) or not self.template_id
        ):
            raise ValueError("template_id must be a nonempty string or None")
        if self.template_key is not None and (
            not isinstance(self.template_key, str) or not self.template_key
        ):
            raise ValueError("template_key must be a nonempty string or None")
        if self.template_id is not None and self.template_key is not None:
            raise ValueError("template_id and template_key are mutually exclusive")
        object.__setattr__(self, "solver_path", solver_path_from_name(self.solver_path))
        object.__setattr__(self, "properties", normalize_properties(self.properties))
        if not isinstance(self.device, str) or _DEVICE.fullmatch(self.device) is None:
            raise ValueError("device must be cpu, cuda, or cuda:N")
        target_dtype = self.dtype
        if isinstance(target_dtype, str):
            target_dtype = {
                "float32": torch.float32,
                "float64": torch.float64,
            }.get(target_dtype)
        if target_dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")
        object.__setattr__(self, "dtype", target_dtype)
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if type(self.overwrite) is not bool:
            raise TypeError("overwrite must be bool")

    @property
    def dtype_name(self) -> str:
        return "float32" if self.dtype == torch.float32 else "float64"

    @property
    def solver_name(self) -> str:
        return solver_name(self.solver_path)

    @property
    def compute_forces(self) -> bool:
        return "forces" in self.properties

    @property
    def compute_stress(self) -> bool:
        return "stress" in self.properties

    @property
    def operation_name(self) -> str:
        return "prediction"

    @property
    def sample_id_prefix(self) -> str:
        return "predict"


def _operation_name(config: Any) -> str:
    return getattr(config, "operation_name", "prediction")


def _operation_stage(config: Any, stage: str) -> str:
    return f"{_operation_name(config)}.{stage}"


def _sample_id(config: Any, frame_index: int) -> str:
    prefix = getattr(config, "sample_id_prefix", "predict")
    return f"{prefix}:{frame_index:06d}"


def _term_context(config: Any) -> str | None:
    terms = getattr(config, "terms", None)
    return None if terms is None else ",".join(terms)


def _file_error(
    reason_code: str,
    message: str,
    *,
    stage: str,
    path: Path | str,
    original_error: BaseException | None = None,
) -> CLIError:
    return CLIError(
        reason_code,
        message,
        stage=stage,
        path=path,
        original_error=original_error,
    )


def _prediction_error(
    reason_code: str,
    message: str,
    *,
    config: ExtXYZPredictionConfig,
    input_path: Path,
    frame_index: int | None,
    sample_id: str | None,
    template_id: str | None,
    prediction_stage: str,
    original_error: BaseException | None = None,
) -> CLIError:
    return CLIError(
        reason_code,
        message,
        stage=_operation_stage(config, prediction_stage),
        path=input_path,
        frame_index=frame_index,
        sample_id=sample_id,
        template_id=template_id,
        term=_term_context(config),
        solver_path=config.solver_path,
        prediction_stage=prediction_stage,
        predictor_reason_code=reason_code,
        original_error=original_error,
    )


def _preflight_device(config: ExtXYZPredictionConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type != "cuda":
        return device
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
    except Exception as error:
        raise CLIError(
            "UNAVAILABLE_CUDA_DEVICE",
            "CUDA availability could not be established",
            stage=_operation_stage(config, "device_preflight"),
            term=_term_context(config),
            solver_path=config.solver_path,
            prediction_stage="device_preflight",
            predictor_reason_code="UNAVAILABLE_CUDA_DEVICE",
            original_error=error,
        ) from error
    index = 0 if device.index is None else device.index
    if not available or index < 0 or index >= count:
        raise CLIError(
            "UNAVAILABLE_CUDA_DEVICE",
            f"requested CUDA device {config.device!r} is unavailable",
            stage=_operation_stage(config, "device_preflight"),
            term=_term_context(config),
            solver_path=config.solver_path,
            prediction_stage="device_preflight",
            predictor_reason_code="UNAVAILABLE_CUDA_DEVICE",
        )
    return device


def _resolve_input_path(config: Any) -> Path:
    source = Path(config.input_path).expanduser()
    try:
        resolved_source = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise _file_error(
            "INPUT_NOT_FOUND",
            "input extxyz does not exist",
            stage=_operation_stage(config, "input_path"),
            path=source,
            original_error=error,
        ) from error
    except OSError as error:
        raise _file_error(
            "INPUT_PATH_ERROR",
            "input extxyz path could not be resolved",
            stage=_operation_stage(config, "input_path"),
            path=source,
            original_error=error,
        ) from error
    if not resolved_source.is_file():
        raise _file_error(
            "INVALID_INPUT_PATH",
            "input extxyz path must be a regular file",
            stage=_operation_stage(config, "input_path"),
            path=source,
        )
    return resolved_source


def _preflight_paths(config: ExtXYZPredictionConfig) -> tuple[Path, Path]:
    resolved_source = _resolve_input_path(config)
    bundle_source = Path(config.bundle_path).expanduser()
    try:
        resolved_bundle = bundle_source.resolve(strict=False)
    except OSError as error:
        raise _file_error(
            "BUNDLE_PATH_ERROR",
            "portable bundle path identity could not be resolved",
            stage=_operation_stage(config, "output_path"),
            path=bundle_source,
            original_error=error,
        ) from error

    target = Path(config.output_path).expanduser()
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise _file_error(
            "INVALID_OUTPUT_DIRECTORY",
            "output parent must be an existing directory",
            stage=_operation_stage(config, "output_path"),
            path=target,
        )
    _validate_output_target(
        target,
        resolved_source=resolved_source,
        resolved_bundle=resolved_bundle,
        overwrite=config.overwrite,
    )
    return resolved_source, target


def _validate_output_target(
    target: Path,
    *,
    resolved_source: Path,
    resolved_bundle: Path,
    overwrite: bool,
) -> None:
    if target.is_symlink():
        raise _file_error(
            "OUTPUT_SYMLINK_REJECTED",
            "output target must not be a symbolic link",
            stage="prediction.output_path",
            path=target,
        )
    try:
        same_path = target.resolve(strict=False) == resolved_source
        if target.exists():
            same_path = same_path or os.path.samefile(target, resolved_source)
    except OSError as error:
        raise _file_error(
            "OUTPUT_PATH_ERROR",
            "output target identity could not be checked",
            stage="prediction.output_path",
            path=target,
            original_error=error,
        ) from error
    if same_path:
        raise _file_error(
            "INPUT_OUTPUT_COLLISION",
            "output must not replace the input extxyz",
            stage="prediction.output_path",
            path=target,
        )
    try:
        same_bundle = target.resolve(strict=False) == resolved_bundle
        if target.exists() and resolved_bundle.exists():
            same_bundle = same_bundle or os.path.samefile(target, resolved_bundle)
    except OSError as error:
        raise _file_error(
            "OUTPUT_PATH_ERROR",
            "output and portable bundle identity could not be checked",
            stage="prediction.output_path",
            path=target,
            original_error=error,
        ) from error
    if same_bundle:
        raise _file_error(
            "BUNDLE_OUTPUT_COLLISION",
            "prediction output must not replace the portable input bundle",
            stage="prediction.output_path",
            path=target,
        )
    if target.exists():
        if not target.is_file():
            raise _file_error(
                "INVALID_OUTPUT_PATH",
                "output target must be a regular file",
                stage="prediction.output_path",
                path=target,
            )
        if not overwrite:
            raise _file_error(
                "OUTPUT_EXISTS",
                "output already exists; pass --overwrite to replace it",
                stage="prediction.output_path",
                path=target,
            )


def _load_predictor(config: ExtXYZPredictionConfig):
    try:
        return load_reference_site_predictor(
            config.bundle_path,
            device=config.device,
            dtype=config.dtype,
            config=PredictorConfig(output_device="cpu"),
        )
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "portable bundle load or runtime instantiation failed",
            stage=error.validation_stage or _operation_stage(config, "predictor_load"),
            bundle_path=error.bundle_path or config.bundle_path,
            term=_term_context(config),
            solver_path=config.solver_path,
            prediction_stage="predictor_load",
            predictor_reason_code=error.reason_code,
            original_error=error,
        ) from error
    except FileNotFoundError as error:
        raise CLIError(
            "BUNDLE_NOT_FOUND",
            "portable bundle does not exist",
            stage=_operation_stage(config, "predictor_load"),
            bundle_path=config.bundle_path,
            term=_term_context(config),
            solver_path=config.solver_path,
            prediction_stage="predictor_load",
            predictor_reason_code="BUNDLE_NOT_FOUND",
            original_error=error,
        ) from error
    except (OSError, ValueError, TypeError) as error:
        reason = getattr(error, "reason_code", "PREDICTOR_LOAD_FAILED")
        raise CLIError(
            reason,
            "portable bundle predictor could not be loaded",
            stage=(
                getattr(error, "validation_stage", None)
                or _operation_stage(config, "predictor_load")
            ),
            bundle_path=config.bundle_path,
            term=_term_context(config),
            solver_path=config.solver_path,
            prediction_stage="predictor_load",
            predictor_reason_code=reason,
            original_error=error,
        ) from error


def _read_frames(
    source: Path,
    config: ExtXYZPredictionConfig,
) -> tuple[Any, ...]:
    try:
        from ase.io import iread
    except ImportError as error:  # pragma: no cover - optional dependency
        raise _file_error(
            "ASE_UNAVAILABLE",
            f"ASE is required for extxyz {_operation_name(config)}",
            stage=_operation_stage(config, "input_parse"),
            path=source,
            original_error=error,
        ) from error

    frames = []
    try:
        iterator = iter(
            iread(str(source), index=config.index, format="extxyz")
        )
        while True:
            try:
                frames.append(next(iterator))
            except StopIteration:
                break
            except Exception as error:
                frame_index = len(frames)
                raise CLIError(
                    "MALFORMED_EXTXYZ",
                    "ASE could not parse an extxyz frame",
                    stage=_operation_stage(config, "input_parse"),
                    path=source,
                    frame_index=frame_index,
                    sample_id=_sample_id(config, frame_index),
                    term=_term_context(config),
                    solver_path=config.solver_path,
                    prediction_stage="input_parse",
                    original_error=error,
                ) from error
    except CLIError:
        raise
    except Exception as error:
        raise _file_error(
            "MALFORMED_EXTXYZ",
            "ASE could not open or parse the extxyz input",
            stage=_operation_stage(config, "input_parse"),
            path=source,
            original_error=error,
        ) from error
    if not frames:
        raise _file_error(
            "EMPTY_INPUT",
            "selected extxyz input contains no frames",
            stage=_operation_stage(config, "input_parse"),
            path=source,
        )
    return tuple(frames)


def _selected_template_id(
    atoms: Any,
    *,
    frame_index: int,
    sample_id: str,
    predictor: Any,
    config: ExtXYZPredictionConfig,
    source: Path,
) -> str:
    if config.template_id is not None:
        return config.template_id
    if config.template_key is None:
        return predictor.runtime.default_template_id
    if config.template_key not in atoms.info:
        raise _prediction_error(
            "MISSING_TEMPLATE_KEY",
            f"Atoms.info lacks template key {config.template_key!r}",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=None,
            prediction_stage="template_selection",
        )
    value = atoms.info[config.template_key]
    if not isinstance(value, str) or not value:
        raise _prediction_error(
            "INVALID_TEMPLATE_ID",
            f"Atoms.info[{config.template_key!r}] must be a nonempty exact ID",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=None,
            prediction_stage="template_selection",
        )
    return value


def _geometry_sample(
    atoms: Any,
    *,
    frame_index: int,
    sample_id: str,
    template_id: str,
    predictor: Any,
    config: ExtXYZPredictionConfig,
    source: Path,
) -> StructureSample:
    try:
        positions = torch.tensor(
            atoms.get_positions().copy(), dtype=torch.float64
        )
        numbers = torch.tensor(
            atoms.get_atomic_numbers().copy(), dtype=torch.long
        )
        cell = torch.tensor(atoms.cell.array.copy(), dtype=torch.float64)
        pbc = torch.tensor(atoms.get_pbc().copy(), dtype=torch.bool)
    except Exception as error:
        raise _prediction_error(
            "MALFORMED_GEOMETRY",
            "frame geometry could not be converted",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="geometry_conversion",
            original_error=error,
        ) from error
    if pbc.shape != (3,) or not bool(torch.all(pbc)):
        raise _prediction_error(
            "NONPERIODIC_STRUCTURE",
            "only full three-dimensional PBC is supported",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="geometry_preflight",
        )
    if positions.shape != (len(atoms), 3) or cell.shape != (3, 3):
        raise _prediction_error(
            "MALFORMED_GEOMETRY",
            "positions/cell must have shapes [N,3] and [3,3]",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="geometry_preflight",
        )
    if (
        not bool(torch.all(torch.isfinite(positions)))
        or not bool(torch.all(torch.isfinite(cell)))
    ):
        raise _prediction_error(
            "NONFINITE_GEOMETRY",
            "positions or cell contain NaN or Infinity",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="geometry_preflight",
        )
    try:
        singular = bool(
            torch.linalg.svdvals(cell)[-1] <= torch.finfo(torch.float64).eps
        )
    except Exception as error:
        raise _prediction_error(
            "MALFORMED_GEOMETRY",
            "cell singular values could not be evaluated",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="geometry_preflight",
            original_error=error,
        ) from error
    if singular:
        raise _prediction_error(
            "SINGULAR_CELL",
            "cell must be nonsingular",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="geometry_preflight",
        )
    if template_id not in predictor.runtime.template_contexts:
        raise _prediction_error(
            "UNKNOWN_TEMPLATE",
            "exact template ID is absent from the bundle",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="template_lookup",
        )
    context = predictor.runtime.template_contexts[template_id]
    if len(atoms) > context.topology.num_sites:
        raise _prediction_error(
            "INVALID_N_GT_M",
            "atom count exceeds the exact template reference-site count",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="structure_domain_preflight",
        )
    try:
        sample = StructureSample(
            sample_id=sample_id,
            positions=positions,
            atomic_numbers=numbers,
            cell=cell,
            pbc=pbc,
            origin=torch.zeros(3, dtype=torch.float64),
            template_id=template_id,
        )
        template = predictor.registry.resolve(template_id)
        template.validate_structure(
            numbers,
            cell=cell if template.strict_domain is not None else None,
            pbc=pbc if template.strict_domain is not None else None,
            sample_id=sample_id,
        )
        context.validate_fingerprint()
    except Exception as error:
        message = str(error).lower()
        reason = (
            "INVALID_N_GT_M"
            if "n > m" in message or "exceeds reference-site" in message
            else "UNSUPPORTED_SPECIES"
            if "species" in message and ("unknown" in message or "unsupported" in message)
            else "UNSUPPORTED_COMPOSITION"
            if "composition" in message or "vacancy" in message
            else "TEMPLATE_DOMAIN_MISMATCH"
        )
        raise _prediction_error(
            reason,
            "frame is incompatible with its exact template",
            config=config,
            input_path=source,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            prediction_stage="structure_domain_preflight",
            original_error=error,
        ) from error
    return sample


def _prepare_samples(
    frames: tuple[Any, ...],
    predictor: Any,
    config: ExtXYZPredictionConfig,
    source: Path,
) -> tuple[StructureSample, ...]:
    samples = []
    first_by_template: dict[str, tuple[int, str]] = {}
    for frame_index, atoms in enumerate(frames):
        sample_id = _sample_id(config, frame_index)
        template_id = _selected_template_id(
            atoms,
            frame_index=frame_index,
            sample_id=sample_id,
            predictor=predictor,
            config=config,
            source=source,
        )
        sample = _geometry_sample(
            atoms,
            frame_index=frame_index,
            sample_id=sample_id,
            template_id=template_id,
            predictor=predictor,
            config=config,
            source=source,
        )
        samples.append(sample)
        first_by_template.setdefault(template_id, (frame_index, sample_id))

    if config.solver_path == EVAL_ADAPTIVE:
        for template_id in sorted(first_by_template):
            frame_index, sample_id = first_by_template[template_id]
            if template_id not in predictor.runtime.evaluation_policies:
                raise _prediction_error(
                    "POLICY_CONTEXT_MISMATCH",
                    "eval-adaptive requires a policy for every used template",
                    config=config,
                    input_path=source,
                    frame_index=frame_index,
                    sample_id=sample_id,
                    template_id=template_id,
                    prediction_stage="policy_lookup",
                )
            policy = predictor.runtime.evaluation_policies[template_id]
            try:
                policy.validate_fingerprint()
                if (
                    policy.template_id != template_id
                    or policy.template_fingerprint
                    != predictor.runtime.template_fingerprints[template_id]
                ):
                    raise ValueError("policy/template fingerprint binding differs")
            except Exception as error:
                raise _prediction_error(
                    "POLICY_CONTEXT_MISMATCH",
                    "evaluation policy preflight failed",
                    config=config,
                    input_path=source,
                    frame_index=frame_index,
                    sample_id=sample_id,
                    template_id=template_id,
                    prediction_stage="policy_preflight",
                    original_error=error,
                ) from error
    return tuple(samples)


def _predict_batches(
    samples: tuple[StructureSample, ...],
    predictor: Any,
    config: ExtXYZPredictionConfig,
    source: Path,
) -> tuple[Any, ...]:
    frame_by_sample = {
        sample.sample_id: index for index, sample in enumerate(samples)
    }
    predictions = []
    for start in range(0, len(samples), config.batch_size):
        chunk = samples[start : start + config.batch_size]
        try:
            batch = predictor.predict_samples(
                chunk,
                solver_path=config.solver_path,
                compute_forces=config.compute_forces,
                compute_stress=config.compute_stress,
                return_aux=False,
                candidate_neighbor_states=None,
                return_candidate_neighbor_states=False,
            )
        except PredictorError as error:
            sample_id = error.sample_id or chunk[0].sample_id
            matching = next(
                (sample for sample in chunk if sample.sample_id == sample_id),
                chunk[0],
            )
            reason = error.reason_code
            raise _prediction_error(
                reason,
                "Predictor batch execution failed",
                config=config,
                input_path=source,
                frame_index=frame_by_sample.get(sample_id, start),
                sample_id=sample_id,
                template_id=error.template_id or matching.template_id,
                prediction_stage=error.stage,
                original_error=error,
            ) from error
        except Exception as error:
            first = chunk[0]
            reason = getattr(error, "reason_code", "PREDICTION_EXECUTION_FAILED")
            raise _prediction_error(
                reason,
                "prediction batch failed unexpectedly",
                config=config,
                input_path=source,
                frame_index=start,
                sample_id=first.sample_id,
                template_id=first.template_id,
                prediction_stage=getattr(error, "stage", "model_evaluation"),
                original_error=error,
            ) from error
        if batch.sample_ids != tuple(sample.sample_id for sample in chunk):
            first = chunk[0]
            raise _prediction_error(
                "OUTPUT_ORDER_MISMATCH",
                "Predictor did not preserve sample ordering",
                config=config,
                input_path=source,
                frame_index=start,
                sample_id=first.sample_id,
                template_id=first.template_id,
                prediction_stage="output_validation",
            )
        predictions.extend(batch.structures)
    return tuple(predictions)


def _validate_predictions(
    frames: tuple[Any, ...],
    samples: tuple[StructureSample, ...],
    predictions: tuple[Any, ...],
    config: ExtXYZPredictionConfig,
    source: Path,
) -> None:
    if len(predictions) != len(frames):
        raise _prediction_error(
            "OUTPUT_COUNT_MISMATCH",
            "prediction count differs from selected frame count",
            config=config,
            input_path=source,
            frame_index=None,
            sample_id=None,
            template_id=None,
            prediction_stage="output_validation",
        )
    for frame_index, (atoms, sample, prediction) in enumerate(
        zip(frames, samples, predictions)
    ):
        values = [prediction.energy]
        if prediction.sample_id != sample.sample_id or prediction.template_id != sample.template_id:
            raise _prediction_error(
                "OUTPUT_ORDER_MISMATCH",
                "prediction identity differs from its input frame",
                config=config,
                input_path=source,
                frame_index=frame_index,
                sample_id=sample.sample_id,
                template_id=sample.template_id,
                prediction_stage="output_validation",
            )
        if config.compute_forces:
            if prediction.forces is None or prediction.forces.shape != (len(atoms), 3):
                raise _prediction_error(
                    "MISSING_FORCE_OUTPUT",
                    "requested force output is absent or malformed",
                    config=config,
                    input_path=source,
                    frame_index=frame_index,
                    sample_id=sample.sample_id,
                    template_id=sample.template_id,
                    prediction_stage="output_validation",
                )
            values.append(prediction.forces)
        elif prediction.forces is not None:
            raise _prediction_error(
                "UNREQUESTED_DERIVATIVE_OUTPUT",
                "Predictor returned forces that were not requested",
                config=config,
                input_path=source,
                frame_index=frame_index,
                sample_id=sample.sample_id,
                template_id=sample.template_id,
                prediction_stage="output_validation",
            )
        if config.compute_stress:
            if (
                prediction.stress is None
                or prediction.stress_voigt is None
                or prediction.stress.shape != (3, 3)
                or prediction.stress_voigt.shape != (6,)
            ):
                raise _prediction_error(
                    "MISSING_STRESS_OUTPUT",
                    "requested stress output is absent or malformed",
                    config=config,
                    input_path=source,
                    frame_index=frame_index,
                    sample_id=sample.sample_id,
                    template_id=sample.template_id,
                    prediction_stage="output_validation",
                )
            values.extend((prediction.stress, prediction.stress_voigt))
        elif prediction.stress is not None or prediction.stress_voigt is not None:
            raise _prediction_error(
                "UNREQUESTED_DERIVATIVE_OUTPUT",
                "Predictor returned stress that was not requested",
                config=config,
                input_path=source,
                frame_index=frame_index,
                sample_id=sample.sample_id,
                template_id=sample.template_id,
                prediction_stage="output_validation",
            )
        if any(not bool(torch.all(torch.isfinite(value))) for value in values):
            raise _prediction_error(
                "NONFINITE_PREDICTION",
                "prediction contains NaN or Infinity",
                config=config,
                input_path=source,
                frame_index=frame_index,
                sample_id=sample.sample_id,
                template_id=sample.template_id,
                prediction_stage="output_validation",
            )


def _prediction_frames(
    frames: tuple[Any, ...],
    predictions: tuple[Any, ...],
    *,
    config: ExtXYZPredictionConfig,
    bundle_fingerprint: str,
) -> tuple[Any, ...]:
    try:
        from ase.calculators.singlepoint import SinglePointCalculator
    except ImportError as error:  # pragma: no cover - optional dependency
        raise _file_error(
            "ASE_UNAVAILABLE",
            "ASE SinglePointCalculator is required for output",
            stage="prediction.output_assembly",
            path=config.output_path,
            original_error=error,
        ) from error
    result = []
    for atoms, prediction in zip(frames, predictions):
        output = atoms.copy()
        output.info = copy.deepcopy(dict(atoms.info))
        for key in _RESULT_INFO_KEYS:
            output.info.pop(key, None)
        for key in _RESULT_ARRAY_KEYS:
            if key in output.arrays:
                output.arrays.pop(key)
        output.info.update(
            {
                "refsite_template_id": prediction.template_id,
                "refsite_solver_path": config.solver_path,
                "refsite_bundle_sha256": bundle_fingerprint,
            }
        )
        energy = float(prediction.energy.detach().cpu())
        calculator_results: dict[str, Any] = {
            "energy": energy,
            "free_energy": energy,
        }
        if config.compute_forces:
            calculator_results["forces"] = (
                prediction.forces.detach().cpu().contiguous().numpy().copy()
            )
        if config.compute_stress:
            calculator_results["stress"] = (
                prediction.stress_voigt.detach().cpu().contiguous().numpy().copy()
            )
        output.calc = SinglePointCalculator(output, **calculator_results)
        result.append(output)
    return tuple(result)


def _summary(
    predictions: tuple[Any, ...],
    config: ExtXYZPredictionConfig,
    *,
    bundle_fingerprint: str,
) -> dict[str, Any]:
    energies = [float(value.energy.detach().cpu()) for value in predictions]
    templates = Counter(value.template_id for value in predictions)
    report: dict[str, Any] = {
        "bundle_sha256": bundle_fingerprint,
        "device": config.device,
        "dtype": config.dtype_name,
        "energy": {
            "max": max(energies),
            "mean": math.fsum(energies) / len(energies),
            "min": min(energies),
        },
        "forces": None,
        "frame_count": len(predictions),
        "output_path": config.output_path,
        "requested_properties": list(config.properties),
        "solver": config.solver_name,
        "stress": None,
        "template_frame_counts": {
            key: templates[key] for key in sorted(templates)
        },
    }
    if config.compute_forces:
        components = []
        norms = []
        for prediction in predictions:
            force = prediction.forces.detach().cpu().to(dtype=torch.float64)
            components.extend(float(value) for value in force.reshape(-1))
            norms.extend(
                float(value) for value in torch.linalg.vector_norm(force, dim=-1)
            )
        report["forces"] = {
            "component_rms": math.sqrt(
                math.fsum(value * value for value in components) / len(components)
            ),
            "max_force_norm": max(norms),
        }
    if config.compute_stress:
        components = [
            float(value)
            for prediction in predictions
            for value in prediction.stress_voigt.detach().cpu().to(torch.float64)
        ]
        report["stress"] = {
            "component_max": max(components),
            "component_min": min(components),
            "sign": STRESS_SIGN,
            "voigt_order": list(STRESS_VOIGT_ORDER),
        }
    return report


def _write_atomic_extxyz(
    target: Path,
    frames: tuple[Any, ...],
    *,
    source: Path,
    config: ExtXYZPredictionConfig,
) -> None:
    try:
        from ase.io import write
    except ImportError as error:  # pragma: no cover - optional dependency
        raise _file_error(
            "ASE_UNAVAILABLE",
            "ASE is required to write extxyz output",
            stage="prediction.output_write",
            path=target,
            original_error=error,
        ) from error
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
    except OSError as error:
        raise _file_error(
            "OUTPUT_TEMPFILE_FAILED",
            "same-directory output temporary file could not be created",
            stage="prediction.output_write",
            path=target,
            original_error=error,
        ) from error
    temporary = Path(temporary_name)
    try:
        try:
            write(str(temporary), frames, format="extxyz")
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise OSError("ASE produced an empty temporary output")
        except Exception as error:
            raise _file_error(
                "OUTPUT_WRITE_FAILED",
                "prediction extxyz temporary output could not be written",
                stage="prediction.output_write",
                path=target,
                original_error=error,
            ) from error
        _validate_output_target(
            target,
            resolved_source=source,
            resolved_bundle=Path(config.bundle_path).expanduser().resolve(
                strict=False
            ),
            overwrite=config.overwrite,
        )
        try:
            os.replace(temporary, target)
        except OSError as error:
            raise _file_error(
                "OUTPUT_COMMIT_FAILED",
                "atomic output replacement failed",
                stage="prediction.output_commit",
                path=target,
                original_error=error,
            ) from error
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                # Preserve the primary failure. A successful command cannot
                # reach this branch because os.replace consumes the temp path.
                pass


def predict_extxyz(config: ExtXYZPredictionConfig) -> dict[str, Any]:
    """Run one transactional extxyz prediction command."""

    if not isinstance(config, ExtXYZPredictionConfig):
        raise TypeError("config must be an ExtXYZPredictionConfig")
    _preflight_device(config)
    source, target = _preflight_paths(config)
    predictor = _load_predictor(config)
    frames = _read_frames(source, config)
    samples = _prepare_samples(frames, predictor, config, source)
    predictions = _predict_batches(samples, predictor, config, source)
    _validate_predictions(frames, samples, predictions, config, source)
    output_frames = _prediction_frames(
        frames,
        predictions,
        config=config,
        bundle_fingerprint=predictor.bundle_fingerprint,
    )
    report = _summary(
        predictions,
        config,
        bundle_fingerprint=predictor.bundle_fingerprint,
    )
    # Strict serialization catches NaN/Infinity in summary scalars before any
    # filesystem commit.
    render_prediction_json(report)
    _write_atomic_extxyz(target, output_frames, source=source, config=config)
    return report


def render_prediction_json(report: Mapping[str, Any]) -> str:
    return _render_json(report)


def _number(value: float) -> str:
    return format(value, ".12g")


def render_prediction_human(report: Mapping[str, Any]) -> str:
    templates = ", ".join(
        f"{key}={report['template_frame_counts'][key]}"
        for key in sorted(report["template_frame_counts"])
    )
    energy = report["energy"]
    lines = [
        "Reference-site MLIP extxyz prediction",
        f"Frames: {report['frame_count']}",
        f"Templates: {templates}",
        f"Properties: {','.join(report['requested_properties'])}",
        (
            f"Runtime: solver={report['solver']} device={report['device']} "
            f"dtype={report['dtype']}"
        ),
        f"Output: {report['output_path']}",
        f"Bundle SHA-256: {report['bundle_sha256']}",
        (
            "Energy (eV): "
            f"min={_number(energy['min'])} mean={_number(energy['mean'])} "
            f"max={_number(energy['max'])}"
        ),
    ]
    if report["forces"] is not None:
        forces = report["forces"]
        lines.append(
            "Forces (eV/angstrom): "
            f"component_rms={_number(forces['component_rms'])} "
            f"max_norm={_number(forces['max_force_norm'])}"
        )
    if report["stress"] is not None:
        stress = report["stress"]
        lines.append(
            "Stress (eV/angstrom^3, tensile-positive, xx yy zz yz xz xy): "
            f"component_min={_number(stress['component_min'])} "
            f"component_max={_number(stress['component_max'])}"
        )
    return "\n".join(lines)


__all__ = [
    "ExtXYZPredictionConfig",
    "PREDICTION_PROPERTIES",
    "normalize_properties",
    "predict_extxyz",
    "render_prediction_human",
    "render_prediction_json",
    "solver_name",
    "solver_path_from_name",
]
