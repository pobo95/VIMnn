"""Read-only extxyz evaluation for portable reference-site bundles."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any

import torch

from refsite_mlip.data import (
    ENERGY_UNIT,
    FORCE_UNIT,
    STRESS_SIGN,
    STRESS_UNIT,
    STRESS_VOIGT_ORDER,
    StructureSample,
    collate_structure_samples,
)
from refsite_mlip.data.extxyz import ExtXYZLoadError, _extract_label
from refsite_mlip.training import LossConfig, compute_potential_loss
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from .errors import CLIError
from .inspect_bundle import render_json as _render_json
from .predict import (
    _load_predictor,
    _predict_batches,
    _preflight_device,
    _prepare_samples,
    _read_frames,
    _resolve_input_path,
    _validate_predictions,
    solver_name,
    solver_path_from_name,
)


EVALUATION_TERMS = ("energy", "forces", "stress")
_LABEL_FOR_TERM = {"energy": "energy", "forces": "forces", "stress": "stress"}


def normalize_terms(values: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        items = tuple(item.strip() for item in values.split(","))
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = tuple(values)
    else:
        raise TypeError("terms must be a comma-separated string or sequence")
    if not items or any(type(item) is not str or not item for item in items):
        raise ValueError("terms must be a nonempty comma-separated list")
    if len(set(items)) != len(items):
        raise ValueError("terms must not contain duplicates")
    unknown = sorted(set(items) - set(EVALUATION_TERMS))
    if unknown:
        raise ValueError(f"unknown evaluation terms: {unknown}")
    selected = set(items)
    return tuple(name for name in EVALUATION_TERMS if name in selected)


def _finite_real(name: str, value: Real, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    invalid = result <= 0.0 if positive else result < 0.0
    if not math.isfinite(result) or invalid:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True)
class ExtXYZEvaluationConfig:
    """Validated controls for one read-only extxyz evaluation command."""

    bundle_path: str
    input_path: str
    index: str = ":"
    template_id: str | None = None
    template_key: str | None = None
    solver_path: str = TRAIN_FIXED
    terms: tuple[str, ...] = ("energy", "forces")
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    batch_size: int = 8
    energy_mode: str = "per-structure"
    energy_scale: float = 1.0
    force_scale: float = 1.0
    stress_scale: float = 1.0
    energy_weight: float = 1.0
    force_weight: float = 1.0
    stress_weight: float = 1.0
    output_path: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        for name in ("bundle_path", "input_path"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError(f"{name} must be a nonempty path")
            object.__setattr__(self, name, str(value))
        if self.output_path is not None:
            if not isinstance(self.output_path, (str, Path)) or not str(self.output_path):
                raise ValueError("output_path must be a nonempty path or None")
            object.__setattr__(self, "output_path", str(self.output_path))
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
        object.__setattr__(self, "terms", normalize_terms(self.terms))
        if (
            not isinstance(self.device, str)
            or re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", self.device) is None
        ):
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
        modes = {
            "per-structure": "per_structure",
            "per-atom": "per_atom",
            "per_structure": "per_structure",
            "per_atom": "per_atom",
        }
        if self.energy_mode not in modes:
            raise ValueError("energy_mode must be per-structure or per-atom")
        object.__setattr__(self, "energy_mode", modes[self.energy_mode])
        for name in ("energy_scale", "force_scale", "stress_scale"):
            object.__setattr__(
                self,
                name,
                _finite_real(name, getattr(self, name), positive=True),
            )
        for name in ("energy_weight", "force_weight", "stress_weight"):
            object.__setattr__(
                self,
                name,
                _finite_real(name, getattr(self, name), positive=False),
            )
        if type(self.overwrite) is not bool:
            raise TypeError("overwrite must be bool")
        if self.overwrite and self.output_path is None:
            raise ValueError("overwrite requires output_path")

    @property
    def operation_name(self) -> str:
        return "evaluation"

    @property
    def sample_id_prefix(self) -> str:
        return "evaluate"

    @property
    def dtype_name(self) -> str:
        return "float32" if self.dtype == torch.float32 else "float64"

    @property
    def solver_name(self) -> str:
        return solver_name(self.solver_path)

    @property
    def energy_mode_name(self) -> str:
        return self.energy_mode.replace("_", "-")

    @property
    def compute_forces(self) -> bool:
        return "forces" in self.terms

    @property
    def compute_stress(self) -> bool:
        return "stress" in self.terms

    def loss_config(self) -> LossConfig:
        return LossConfig(
            energy_weight=(self.energy_weight if "energy" in self.terms else 0.0),
            force_weight=(self.force_weight if "forces" in self.terms else 0.0),
            stress_weight=(self.stress_weight if "stress" in self.terms else 0.0),
            energy_scale=self.energy_scale,
            force_scale=self.force_scale,
            stress_scale=self.stress_scale,
            energy_normalization=self.energy_mode,
        )


def _evaluation_error(
    reason_code: str,
    message: str,
    *,
    config: ExtXYZEvaluationConfig,
    source: Path | str,
    stage: str,
    frame_index: int | None = None,
    sample_id: str | None = None,
    template_id: str | None = None,
    term: str | None = None,
    original_error: BaseException | None = None,
) -> CLIError:
    return CLIError(
        reason_code,
        message,
        stage=f"evaluation.{stage}",
        path=source,
        frame_index=frame_index,
        sample_id=sample_id,
        template_id=template_id,
        term=term,
        solver_path=config.solver_path,
        prediction_stage=stage,
        predictor_reason_code=reason_code,
        original_error=original_error,
    )


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve(strict=False) == right.resolve(strict=False):
        return True
    if left.exists() and right.exists():
        return os.path.samefile(left, right)
    return False


def _validate_report_target(
    target: Path,
    *,
    source: Path,
    config: ExtXYZEvaluationConfig,
) -> None:
    if target.is_symlink():
        raise _evaluation_error(
            "OUTPUT_SYMLINK_REJECTED",
            "evaluation report target must not be a symbolic link",
            config=config,
            source=target,
            stage="output_path",
        )
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise _evaluation_error(
            "INVALID_OUTPUT_DIRECTORY",
            "report parent must be an existing directory",
            config=config,
            source=target,
            stage="output_path",
        )
    try:
        if _same_file(target, source):
            raise _evaluation_error(
                "INPUT_OUTPUT_COLLISION",
                "report output must not replace the input extxyz",
                config=config,
                source=target,
                stage="output_path",
            )
        bundle = Path(config.bundle_path).expanduser()
        if _same_file(target, bundle):
            raise _evaluation_error(
                "BUNDLE_OUTPUT_COLLISION",
                "report output must not replace the portable bundle",
                config=config,
                source=target,
                stage="output_path",
            )
    except CLIError:
        raise
    except OSError as error:
        raise _evaluation_error(
            "OUTPUT_PATH_ERROR",
            "report target identity could not be checked",
            config=config,
            source=target,
            stage="output_path",
            original_error=error,
        ) from error
    if target.exists():
        if not target.is_file():
            raise _evaluation_error(
                "INVALID_OUTPUT_PATH",
                "report target must be a regular file",
                config=config,
                source=target,
                stage="output_path",
            )
        if not config.overwrite:
            raise _evaluation_error(
                "OUTPUT_EXISTS",
                "report already exists; pass --overwrite to replace it",
                config=config,
                source=target,
                stage="output_path",
            )


def _preflight_report_output(
    source: Path,
    config: ExtXYZEvaluationConfig,
) -> Path | None:
    if config.output_path is None:
        return None
    target = Path(config.output_path).expanduser()
    _validate_report_target(target, source=source, config=config)
    return target


def _canonical_mask(
    value: Any,
    *,
    term: str,
    num_atoms: int,
    frame_index: int,
    sample_id: str,
) -> torch.Tensor:
    import numpy as np

    try:
        array = np.asarray(value)
    except Exception as error:
        raise ExtXYZLoadError(
            "MALFORMED_MASK",
            f"{term} component mask could not be converted",
            frame_index=frame_index,
            sample_id=sample_id,
            label=term,
        ) from error
    if np.issubdtype(array.dtype, np.bool_):
        boolean = np.array(array, dtype=np.bool_, copy=True)
    elif np.issubdtype(array.dtype, np.number):
        numeric = np.array(array, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(numeric)) or not np.all(
            np.logical_or(numeric == 0.0, numeric == 1.0)
        ):
            raise ExtXYZLoadError(
                "MALFORMED_MASK",
                f"{term} component mask must contain only 0/1 values",
                frame_index=frame_index,
                sample_id=sample_id,
                label=term,
            )
        boolean = numeric.astype(np.bool_)
    else:
        raise ExtXYZLoadError(
            "MALFORMED_MASK",
            f"{term} component mask must be boolean or numeric 0/1",
            frame_index=frame_index,
            sample_id=sample_id,
            label=term,
        )
    if term == "forces":
        expected = (num_atoms, 3)
        if boolean.shape != expected:
            raise ExtXYZLoadError(
                "MALFORMED_MASK",
                f"force_mask must have shape [{num_atoms},3], got {boolean.shape}",
                frame_index=frame_index,
                sample_id=sample_id,
                label=term,
            )
    else:
        if boolean.shape == (6,):
            xx, yy, zz, yz, xz, xy = boolean.tolist()
            boolean = np.array(
                [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.bool_
            )
        elif boolean.shape == (9,):
            boolean = boolean.reshape(3, 3)
        elif boolean.shape != (3, 3):
            raise ExtXYZLoadError(
                "MALFORMED_MASK",
                f"stress_mask must have shape [6] or [3,3], got {boolean.shape}",
                frame_index=frame_index,
                sample_id=sample_id,
                label=term,
            )
        if not np.array_equal(boolean, boolean.T):
            raise ExtXYZLoadError(
                "MALFORMED_MASK",
                "stress_mask must be symmetric",
                frame_index=frame_index,
                sample_id=sample_id,
                label=term,
            )
    return torch.tensor(boolean, dtype=torch.bool)


def _extract_component_mask(
    atoms: Any,
    *,
    term: str,
    label_present: bool,
    frame_index: int,
    sample_id: str,
) -> torch.Tensor | None:
    key = f"{term[:-1] if term == 'forces' else term}_mask"
    results = (
        atoms.calc.results
        if atoms.calc is not None and isinstance(atoms.calc.results, Mapping)
        else {}
    )
    sources = []
    if key in results:
        sources.append(("atoms.calc.results", results[key]))
    if key in atoms.info:
        sources.append(("Atoms.info", atoms.info[key]))
    if key in atoms.arrays:
        sources.append(("Atoms.arrays", atoms.arrays[key]))
    if not sources:
        return None
    if not label_present:
        raise ExtXYZLoadError(
            "MASK_WITHOUT_LABEL",
            f"{key} is present without its {term} label",
            frame_index=frame_index,
            sample_id=sample_id,
            label=term,
        )
    canonical = _canonical_mask(
        sources[0][1],
        term=term,
        num_atoms=len(atoms),
        frame_index=frame_index,
        sample_id=sample_id,
    )
    for source_name, value in sources[1:]:
        duplicate = _canonical_mask(
            value,
            term=term,
            num_atoms=len(atoms),
            frame_index=frame_index,
            sample_id=sample_id,
        )
        if not torch.equal(canonical, duplicate):
            raise ExtXYZLoadError(
                "CONFLICTING_MASK",
                f"component mask conflicts with {source_name}",
                frame_index=frame_index,
                sample_id=sample_id,
                label=term,
            )
    return canonical


def _independent_stress_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (mask[0, 0], mask[1, 1], mask[2, 2], mask[1, 2], mask[0, 2], mask[0, 1])
    )


def _sample_term_valid_count(sample: StructureSample, term: str) -> int:
    if term == "energy":
        return int(sample.energy is not None)
    if term == "forces":
        if sample.forces is None:
            return 0
        mask = sample.force_mask
        return sample.num_atoms * 3 if mask is None else int(torch.count_nonzero(mask))
    if sample.stress is None:
        return 0
    mask = sample.stress_mask
    if mask is None:
        return 6
    return int(torch.count_nonzero(_independent_stress_mask(mask)))


def _prepare_labeled_samples(
    frames: tuple[Any, ...],
    geometry_samples: tuple[StructureSample, ...],
    config: ExtXYZEvaluationConfig,
    source: Path,
) -> tuple[StructureSample, ...]:
    labeled = []
    for frame_index, (atoms, geometry) in enumerate(zip(frames, geometry_samples)):
        labels: dict[str, torch.Tensor | None] = {
            "energy": None,
            "forces": None,
            "stress": None,
        }
        masks: dict[str, torch.Tensor | None] = {
            "forces": None,
            "stress": None,
        }
        try:
            for term in config.terms:
                label = _extract_label(
                    atoms,
                    _LABEL_FOR_TERM[term],
                    required=False,
                    frame_index=frame_index,
                    sample_id=geometry.sample_id,
                )
                labels[term] = label
                if term in ("forces", "stress"):
                    masks[term] = _extract_component_mask(
                        atoms,
                        term=term,
                        label_present=label is not None,
                        frame_index=frame_index,
                        sample_id=geometry.sample_id,
                    )
        except ExtXYZLoadError as error:
            term = error.label
            raise _evaluation_error(
                error.reason_code,
                "extxyz label or component mask validation failed",
                config=config,
                source=source,
                stage="label_loading",
                frame_index=error.frame_index,
                sample_id=error.sample_id,
                template_id=geometry.template_id,
                term=term,
                original_error=error,
            ) from error
        base = geometry.to(device="cpu", dtype=config.dtype)

        def floating(value):
            return (
                None
                if value is None
                else value.detach().clone().to(dtype=config.dtype, device="cpu")
            )

        def boolean(value):
            return None if value is None else value.detach().clone().to(device="cpu")

        labeled.append(
            replace(
                base,
                energy=floating(labels["energy"]),
                forces=floating(labels["forces"]),
                stress=floating(labels["stress"]),
                force_mask=boolean(masks["forces"]),
                stress_mask=boolean(masks["stress"]),
            )
        )
    result = tuple(labeled)
    for term in config.terms:
        if sum(_sample_term_valid_count(sample, term) for sample in result) == 0:
            first = result[0]
            raise _evaluation_error(
                "NO_VALID_LABELS",
                "requested evaluation term has no valid labels",
                config=config,
                source=source,
                stage="label_preflight",
                frame_index=0,
                sample_id=first.sample_id,
                template_id=first.template_id,
                term=term,
            )
    return result


def _hash_text(digest: Any, value: str) -> None:
    raw = value.encode("utf-8")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _hash_tensor(digest: Any, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    _hash_text(digest, name)
    _hash_text(digest, str(tensor.dtype))
    _hash_text(digest, ",".join(str(size) for size in tensor.shape))
    raw = tensor.contiguous().numpy().tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _input_semantic_digest(
    samples: tuple[StructureSample, ...],
    terms: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "refsite_extxyz_evaluation_input_v1")
    _hash_text(digest, ",".join(terms))
    _hash_text(digest, str(len(samples)))
    for frame_index, sample in enumerate(samples):
        _hash_text(digest, f"frame:{frame_index}")
        _hash_text(digest, sample.template_id)
        for name in ("positions", "atomic_numbers", "cell", "pbc", "origin"):
            _hash_tensor(digest, name, getattr(sample, name))
        for term in terms:
            label = getattr(sample, term)
            _hash_text(digest, f"{term}_present:{label is not None}")
            if label is not None:
                _hash_tensor(digest, term, label)
            if term in ("forces", "stress"):
                mask = getattr(sample, f"{term[:-1] if term == 'forces' else term}_mask")
                _hash_text(digest, f"{term}_mask_explicit:{mask is not None}")
                if mask is not None:
                    _hash_tensor(digest, f"{term}_mask", mask)
    return digest.hexdigest()


def _full_prediction(predictions: tuple[Any, ...], config: ExtXYZEvaluationConfig):
    return SimpleNamespace(
        energy=torch.stack(tuple(value.energy for value in predictions)),
        forces=(
            torch.cat(tuple(value.forces for value in predictions), dim=0)
            if config.compute_forces
            else None
        ),
        stress=(
            torch.stack(tuple(value.stress for value in predictions))
            if config.compute_stress
            else None
        ),
    )


def _error_statistics(values: torch.Tensor) -> dict[str, float]:
    errors = [float(value) for value in values.detach().cpu().to(torch.float64).reshape(-1)]
    count = len(errors)
    return {
        "mae": math.fsum(abs(value) for value in errors) / count,
        "rmse": math.sqrt(math.fsum(value * value for value in errors) / count),
    }


def _physical_metrics(prediction: Any, batch: Any, terms: tuple[str, ...]):
    metrics: dict[str, Any] = {}
    if "energy" in terms:
        valid = batch.energy_mask
        residual = prediction.energy[valid] - batch.energy[valid]
        counts = (batch.atom_ptr[1:] - batch.atom_ptr[:-1]).to(
            dtype=residual.dtype
        )[valid]
        metrics["energy"] = {
            "per_atom": {
                **_error_statistics(residual / counts),
                "unit": f"{ENERGY_UNIT}/atom",
            },
            "total": {**_error_statistics(residual), "unit": ENERGY_UNIT},
            "valid_structures": int(torch.count_nonzero(valid)),
        }
    if "forces" in terms:
        valid = batch.force_mask & batch.force_present[batch.atom_batch, None]
        residual = prediction.forces - batch.forces
        component = residual[valid]
        complete_atoms = torch.all(valid, dim=1)
        vector_errors = torch.linalg.vector_norm(residual[complete_atoms], dim=1)
        vector_values = [
            float(value)
            for value in vector_errors.detach().cpu().to(torch.float64)
        ]
        metrics["forces"] = {
            "components": {
                **_error_statistics(component),
                "unit": FORCE_UNIT,
            },
            "valid_atoms": int(torch.count_nonzero(complete_atoms)),
            "valid_components": int(torch.count_nonzero(valid)),
            "vector_error": {
                "max": max(vector_values) if vector_values else None,
                "mean": (
                    math.fsum(vector_values) / len(vector_values)
                    if vector_values
                    else None
                ),
                "unit": FORCE_UNIT,
            },
        }
    if "stress" in terms:
        valid = batch.stress_mask & batch.stress_present[:, None, None]
        residual = prediction.stress - batch.stress
        voigt_residual = torch.stack(
            (
                residual[:, 0, 0],
                residual[:, 1, 1],
                residual[:, 2, 2],
                residual[:, 1, 2],
                residual[:, 0, 2],
                residual[:, 0, 1],
            ),
            dim=1,
        )
        voigt_valid = torch.stack(
            (
                valid[:, 0, 0],
                valid[:, 1, 1],
                valid[:, 2, 2],
                valid[:, 1, 2],
                valid[:, 0, 2],
                valid[:, 0, 1],
            ),
            dim=1,
        )
        independent = voigt_residual[voigt_valid]
        diagonal = voigt_residual[:, :3][voigt_valid[:, :3]]
        off_diagonal = voigt_residual[:, 3:][voigt_valid[:, 3:]]
        numerator = math.fsum(
            float(value) ** 2
            for value in diagonal.detach().cpu().to(torch.float64)
        ) + 2.0 * math.fsum(
            float(value) ** 2
            for value in off_diagonal.detach().cpu().to(torch.float64)
        )
        count = int(torch.count_nonzero(voigt_valid))
        metrics["stress"] = {
            "components": {
                **_error_statistics(independent),
                "unit": STRESS_UNIT,
            },
            "frobenius_mean": numerator / count,
            "frobenius_numerator": numerator,
            "frobenius_squared_unit": f"({STRESS_UNIT})^2",
            "unit": STRESS_UNIT,
            "valid_independent_components": count,
        }
    return metrics


def _loss_report(loss: Any, config: ExtXYZEvaluationConfig) -> dict[str, Any]:
    terms = {}
    names = {"energy": "energy", "forces": "force", "stress": "stress"}
    for term in config.terms:
        value = getattr(loss, names[term])
        terms[term] = {
            "denominator": float(value.denominator.detach().cpu()),
            "mean": float(value.mean.detach().cpu()),
            "numerator": float(value.numerator.detach().cpu()),
            "valid_count": int(value.valid_count.detach().cpu()),
            "weight": getattr(config, f"{'force' if term == 'forces' else term}_weight"),
        }
    return {
        "terms": terms,
        "total_normalized": float(loss.total.detach().cpu()),
    }


def _label_report(samples: tuple[StructureSample, ...], terms: tuple[str, ...]):
    report = {}
    for term in terms:
        present = sum(getattr(sample, term) is not None for sample in samples)
        entry = {
            "missing_frames": len(samples) - present,
            "present_frames": present,
            "valid_count": sum(_sample_term_valid_count(sample, term) for sample in samples),
        }
        if term == "forces":
            entry["valid_count_kind"] = "cartesian_components"
        elif term == "stress":
            entry["valid_count_kind"] = "independent_voigt_components"
        else:
            entry["valid_count_kind"] = "structures"
        report[term] = entry
    return report


def _composition_report(samples: tuple[StructureSample, ...]):
    histogram: Counter[tuple[tuple[int, int], ...]] = Counter()
    for sample in samples:
        counts = Counter(int(value) for value in sample.atomic_numbers.tolist())
        histogram[tuple(sorted(counts.items()))] += 1
    return [
        {
            "frame_count": histogram[composition],
            "species": [
                {"atomic_number": atomic_number, "count": count}
                for atomic_number, count in composition
            ],
        }
        for composition in sorted(histogram)
    ]


def _write_atomic_json(
    target: Path,
    encoded: str,
    *,
    source: Path,
    config: ExtXYZEvaluationConfig,
) -> None:
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        os.close(descriptor)
    except OSError as error:
        raise _evaluation_error(
            "OUTPUT_TEMPFILE_FAILED",
            "same-directory report temporary file could not be created",
            config=config,
            source=target,
            stage="output_write",
            original_error=error,
        ) from error
    temporary = Path(temporary_name)
    try:
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise _evaluation_error(
                "OUTPUT_WRITE_FAILED",
                "evaluation JSON temporary report could not be written",
                config=config,
                source=target,
                stage="output_write",
                original_error=error,
            ) from error
        _validate_report_target(target, source=source, config=config)
        try:
            os.replace(temporary, target)
        except OSError as error:
            raise _evaluation_error(
                "OUTPUT_COMMIT_FAILED",
                "atomic evaluation report replacement failed",
                config=config,
                source=target,
                stage="output_commit",
                original_error=error,
            ) from error
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def evaluate_extxyz(config: ExtXYZEvaluationConfig) -> dict[str, Any]:
    """Evaluate selected labels without training, mutation, or backward."""

    if not isinstance(config, ExtXYZEvaluationConfig):
        raise TypeError("config must be an ExtXYZEvaluationConfig")
    _preflight_device(config)
    source = _resolve_input_path(config)
    target = _preflight_report_output(source, config)
    predictor = _load_predictor(config)
    frames = _read_frames(source, config)
    geometry_samples = _prepare_samples(frames, predictor, config, source)
    labeled_samples = _prepare_labeled_samples(
        frames, geometry_samples, config, source
    )
    predictions = _predict_batches(geometry_samples, predictor, config, source)
    _validate_predictions(frames, geometry_samples, predictions, config, source)
    prediction = _full_prediction(predictions, config)
    batch = collate_structure_samples(labeled_samples, predictor.registry)
    try:
        with torch.no_grad():
            loss = compute_potential_loss(prediction, batch, config.loss_config())
    except Exception as error:
        message = str(error).lower()
        term = next(
            (name for name in config.terms if name.removesuffix("s") in message),
            None,
        )
        first = labeled_samples[0]
        raise _evaluation_error(
            "LOSS_COMPUTATION_FAILED",
            "masked normalized loss computation failed",
            config=config,
            source=source,
            stage="loss_computation",
            frame_index=0,
            sample_id=first.sample_id,
            template_id=first.template_id,
            term=term,
            original_error=error,
        ) from error
    report = {
        "bundle_sha256": predictor.bundle_fingerprint,
        "composition": _composition_report(labeled_samples),
        "conventions": {
            "energy_unit": ENERGY_UNIT,
            "force_unit": FORCE_UNIT,
            "stress_sign": STRESS_SIGN,
            "stress_unit": STRESS_UNIT,
            "stress_voigt_order": list(STRESS_VOIGT_ORDER),
        },
        "device": config.device,
        "dtype": config.dtype_name,
        "energy_mode": config.energy_mode_name,
        "frame_count": len(labeled_samples),
        "input_semantic_sha256": _input_semantic_digest(
            labeled_samples, config.terms
        ),
        "labels": _label_report(labeled_samples, config.terms),
        "loss": _loss_report(loss, config),
        "metrics": _physical_metrics(prediction, batch, config.terms),
        "requested_terms": list(config.terms),
        "scales": {
            term: getattr(
                config, f"{'force' if term == 'forces' else term}_scale"
            )
            for term in config.terms
        },
        "solver": config.solver_name,
        "template_frame_counts": {
            key: count
            for key, count in sorted(
                Counter(sample.template_id for sample in labeled_samples).items()
            )
        },
        "weights": {
            term: getattr(
                config, f"{'force' if term == 'forces' else term}_weight"
            )
            for term in config.terms
        },
    }
    encoded = render_evaluation_json(report)
    if target is not None:
        _write_atomic_json(
            target,
            encoded,
            source=source,
            config=config,
        )
    return report


def render_evaluation_json(report: Mapping[str, Any]) -> str:
    return _render_json(report)


def _number(value: float) -> str:
    return format(value, ".12g")


def render_evaluation_human(report: Mapping[str, Any]) -> str:
    templates = ", ".join(
        f"{key}={report['template_frame_counts'][key]}"
        for key in sorted(report["template_frame_counts"])
    )
    compositions = ", ".join(
        (
            "+".join(
                f"Z{entry['atomic_number']}x{entry['count']}"
                for entry in composition["species"]
            )
            + f"={composition['frame_count']}"
        )
        for composition in report["composition"]
    )
    lines = [
        "Reference-site MLIP extxyz evaluation",
        f"Frames: {report['frame_count']}",
        f"Templates: {templates}",
        f"Compositions: {compositions}",
        f"Terms: {','.join(report['requested_terms'])}",
        (
            f"Runtime: solver={report['solver']} device={report['device']} "
            f"dtype={report['dtype']} energy_mode={report['energy_mode']}"
        ),
        f"Bundle SHA-256: {report['bundle_sha256']}",
        f"Input semantic SHA-256: {report['input_semantic_sha256']}",
    ]
    for term in report["requested_terms"]:
        labels = report["labels"][term]
        lines.append(
            f"Labels {term}: present={labels['present_frames']} "
            f"missing={labels['missing_frames']} valid={labels['valid_count']}"
        )
        loss = report["loss"]["terms"][term]
        lines.append(
            f"Normalized loss {term}: numerator={_number(loss['numerator'])} "
            f"denominator={_number(loss['denominator'])} mean={_number(loss['mean'])} "
            f"valid_count={loss['valid_count']} scale={_number(report['scales'][term])} "
            f"weight={_number(loss['weight'])}"
        )
    metrics = report["metrics"]
    if "energy" in metrics:
        total = metrics["energy"]["total"]
        per_atom = metrics["energy"]["per_atom"]
        lines.append(
            "Energy: "
            f"total_MAE={_number(total['mae'])} total_RMSE={_number(total['rmse'])} eV; "
            f"per_atom_MAE={_number(per_atom['mae'])} "
            f"per_atom_RMSE={_number(per_atom['rmse'])} eV/atom"
        )
    if "forces" in metrics:
        force = metrics["forces"]
        vector = force["vector_error"]
        vector_mean = "n/a" if vector["mean"] is None else _number(vector["mean"])
        vector_max = "n/a" if vector["max"] is None else _number(vector["max"])
        lines.append(
            "Forces: "
            f"component_MAE={_number(force['components']['mae'])} "
            f"component_RMSE={_number(force['components']['rmse'])} "
            f"vector_mean={vector_mean} vector_max={vector_max} eV/angstrom"
        )
    if "stress" in metrics:
        stress = metrics["stress"]
        lines.append(
            "Stress (tensile-positive, xx yy zz yz xz xy): "
            f"component_MAE={_number(stress['components']['mae'])} "
            f"component_RMSE={_number(stress['components']['rmse'])} "
            f"frobenius_mean={_number(stress['frobenius_mean'])}"
        )
    lines.append(f"Total normalized loss: {_number(report['loss']['total_normalized'])}")
    return "\n".join(lines)


__all__ = [
    "EVALUATION_TERMS",
    "ExtXYZEvaluationConfig",
    "evaluate_extxyz",
    "normalize_terms",
    "render_evaluation_human",
    "render_evaluation_json",
]
