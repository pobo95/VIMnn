"""Strict ASE extxyz to :class:`StructureSample` loading."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

import torch

from .dataset import InMemoryStructureDataset
from .schema import StructureSample
from .templates import TemplateRegistry


EXTXYZ_LOADER_CONVENTION_VERSION = "extxyz_structure_sample_loader_v1"
EXTXYZ_UNIT_CONVENTION_VERSION = "angstrom_ev_tensile_voigt_v1"
_PREFIX = re.compile(r"[A-Za-z0-9_.-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExtXYZLoadError(ValueError):
    """Actionable frame-local load failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        frame_index: int | None = None,
        sample_id: str | None = None,
        label: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.frame_index = frame_index
        self.sample_id = sample_id
        self.label = label
        context = []
        if frame_index is not None:
            context.append(f"frame_index={frame_index}")
        if sample_id is not None:
            context.append(f"sample_id={sample_id}")
        if label is not None:
            context.append(f"label={label}")
        suffix = " " + " ".join(context) if context else ""
        super().__init__(f"[{reason_code}]{suffix} {message}")


@dataclass(frozen=True)
class ExtXYZLoadConfig:
    source_path: str
    sample_id_prefix: str
    template_id: str
    require_energy: bool = True
    require_forces: bool = True
    require_stress: bool = True
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    origin_convention: str = "zero"
    unit_convention_version: str = EXTXYZ_UNIT_CONVENTION_VERSION
    convention_version: str = EXTXYZ_LOADER_CONVENTION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, (str, Path)) or not str(self.source_path):
            raise ValueError("source_path must be nonempty")
        object.__setattr__(self, "source_path", str(self.source_path))
        if (
            not isinstance(self.sample_id_prefix, str)
            or not _PREFIX.fullmatch(self.sample_id_prefix)
        ):
            raise ValueError(
                "sample_id_prefix must use only letters, digits, '.', '_', or '-'"
            )
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be nonempty")
        for name in ("require_energy", "require_forces", "require_stress"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("loader dtype must be torch.float32 or torch.float64")
        try:
            target = torch.device(self.device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("invalid loader target device") from error
        object.__setattr__(self, "device", str(target))
        if self.origin_convention != "zero":
            raise ValueError("only origin_convention='zero' is supported")
        if self.unit_convention_version != EXTXYZ_UNIT_CONVENTION_VERSION:
            raise ValueError("unsupported extxyz unit/convention version")
        if self.convention_version != EXTXYZ_LOADER_CONVENTION_VERSION:
            raise ValueError("unsupported extxyz loader convention version")

    @property
    def dtype_name(self) -> str:
        return "float64" if self.dtype == torch.float64 else "float32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "sample_id_prefix": self.sample_id_prefix,
            "template_id": self.template_id,
            "require_energy": self.require_energy,
            "require_forces": self.require_forces,
            "require_stress": self.require_stress,
            "dtype": self.dtype_name,
            "device": self.device,
            "origin_convention": self.origin_convention,
            "unit_convention_version": self.unit_convention_version,
            "convention_version": self.convention_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExtXYZLoadConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("extxyz load config payload must be a mapping")
        dtype_name = payload.get("dtype", "float64")
        dtypes = {"float32": torch.float32, "float64": torch.float64}
        if dtype_name not in dtypes:
            raise ValueError("serialized loader dtype must be float32 or float64")
        return cls(
            source_path=payload["source_path"],
            sample_id_prefix=payload["sample_id_prefix"],
            template_id=payload["template_id"],
            require_energy=payload.get("require_energy", True),
            require_forces=payload.get("require_forces", True),
            require_stress=payload.get("require_stress", True),
            dtype=dtypes[dtype_name],
            device=payload.get("device", "cpu"),
            origin_convention=payload.get("origin_convention", "zero"),
            unit_convention_version=payload.get(
                "unit_convention_version", EXTXYZ_UNIT_CONVENTION_VERSION
            ),
            convention_version=payload.get(
                "convention_version", EXTXYZ_LOADER_CONVENTION_VERSION
            ),
        )


@dataclass(frozen=True)
class NumericStatistics:
    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    rms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "rms": self.rms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NumericStatistics":
        return cls(
            count=int(payload["count"]),
            minimum=payload["minimum"],
            maximum=payload["maximum"],
            mean=payload["mean"],
            rms=payload["rms"],
        )


@dataclass(frozen=True)
class ExtXYZLoadDiagnostics:
    source_logical_name: str
    frame_count: int
    first_sample_id: str
    last_sample_id: str
    template_id: str
    template_fingerprint: str
    composition_histogram: tuple[tuple[tuple[int, ...], int], ...]
    atom_count_histogram: tuple[tuple[int, int], ...]
    cell_volume: NumericStatistics
    strain: NumericStatistics
    energy: NumericStatistics
    force_components: NumericStatistics
    force_norms: NumericStatistics
    stress_components: NumericStatistics
    stress_frobenius_norms: NumericStatistics
    missing_energy_count: int
    missing_forces_count: int
    missing_stress_count: int
    nonfinite_label_count: int
    semantic_sha256: str
    output_dtype: str
    unit_convention_version: str = EXTXYZ_UNIT_CONVENTION_VERSION
    convention_version: str = EXTXYZ_LOADER_CONVENTION_VERSION

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("extxyz diagnostics require at least one frame")
        if not _SHA256.fullmatch(self.template_fingerprint):
            raise ValueError("template_fingerprint must be lowercase SHA-256")
        if not _SHA256.fullmatch(self.semantic_sha256):
            raise ValueError("semantic_sha256 must be lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_logical_name": self.source_logical_name,
            "frame_count": self.frame_count,
            "first_sample_id": self.first_sample_id,
            "last_sample_id": self.last_sample_id,
            "template_id": self.template_id,
            "template_fingerprint": self.template_fingerprint,
            "composition_histogram": [
                {"composition": list(composition), "count": count}
                for composition, count in self.composition_histogram
            ],
            "atom_count_histogram": [
                {"num_atoms": num_atoms, "count": count}
                for num_atoms, count in self.atom_count_histogram
            ],
            "cell_volume": self.cell_volume.to_dict(),
            "strain": self.strain.to_dict(),
            "energy": self.energy.to_dict(),
            "force_components": self.force_components.to_dict(),
            "force_norms": self.force_norms.to_dict(),
            "stress_components": self.stress_components.to_dict(),
            "stress_frobenius_norms": self.stress_frobenius_norms.to_dict(),
            "missing_energy_count": self.missing_energy_count,
            "missing_forces_count": self.missing_forces_count,
            "missing_stress_count": self.missing_stress_count,
            "nonfinite_label_count": self.nonfinite_label_count,
            "semantic_sha256": self.semantic_sha256,
            "output_dtype": self.output_dtype,
            "unit_convention_version": self.unit_convention_version,
            "convention_version": self.convention_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExtXYZLoadDiagnostics":
        if not isinstance(payload, Mapping):
            raise TypeError("extxyz diagnostics payload must be a mapping")
        return cls(
            source_logical_name=payload["source_logical_name"],
            frame_count=int(payload["frame_count"]),
            first_sample_id=payload["first_sample_id"],
            last_sample_id=payload["last_sample_id"],
            template_id=payload["template_id"],
            template_fingerprint=payload["template_fingerprint"],
            composition_histogram=tuple(
                (tuple(entry["composition"]), int(entry["count"]))
                for entry in payload["composition_histogram"]
            ),
            atom_count_histogram=tuple(
                (int(entry["num_atoms"]), int(entry["count"]))
                for entry in payload["atom_count_histogram"]
            ),
            cell_volume=NumericStatistics.from_dict(payload["cell_volume"]),
            strain=NumericStatistics.from_dict(payload["strain"]),
            energy=NumericStatistics.from_dict(payload["energy"]),
            force_components=NumericStatistics.from_dict(
                payload["force_components"]
            ),
            force_norms=NumericStatistics.from_dict(payload["force_norms"]),
            stress_components=NumericStatistics.from_dict(
                payload["stress_components"]
            ),
            stress_frobenius_norms=NumericStatistics.from_dict(
                payload["stress_frobenius_norms"]
            ),
            missing_energy_count=int(payload["missing_energy_count"]),
            missing_forces_count=int(payload["missing_forces_count"]),
            missing_stress_count=int(payload["missing_stress_count"]),
            nonfinite_label_count=int(payload["nonfinite_label_count"]),
            semantic_sha256=payload["semantic_sha256"],
            output_dtype=payload["output_dtype"],
            unit_convention_version=payload.get(
                "unit_convention_version", EXTXYZ_UNIT_CONVENTION_VERSION
            ),
            convention_version=payload.get(
                "convention_version", EXTXYZ_LOADER_CONVENTION_VERSION
            ),
        )


@dataclass(frozen=True)
class ExtXYZLoadResult:
    samples: tuple[StructureSample, ...]
    diagnostics: ExtXYZLoadDiagnostics

    def __post_init__(self) -> None:
        if len(self.samples) != self.diagnostics.frame_count:
            raise ValueError("sample count and extxyz diagnostics disagree")
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("duplicate sample_id in extxyz load result")
        if (
            sample_ids[0] != self.diagnostics.first_sample_id
            or sample_ids[-1] != self.diagnostics.last_sample_id
        ):
            raise ValueError("sample ID range and extxyz diagnostics disagree")


def _statistics(values: list[float]) -> NumericStatistics:
    if not values:
        return NumericStatistics(0, None, None, None, None)
    count = len(values)
    return NumericStatistics(
        count=count,
        minimum=min(values),
        maximum=max(values),
        mean=math.fsum(values) / count,
        rms=math.sqrt(math.fsum(value * value for value in values) / count),
    )


def _canonical_label(
    name: str,
    value: Any,
    *,
    num_atoms: int,
    frame_index: int,
    sample_id: str,
) -> torch.Tensor:
    import numpy as np

    array = np.array(value, dtype=np.float64, copy=True)
    if name == "energy":
        if array.shape != ():
            raise ExtXYZLoadError(
                "MALFORMED_LABEL",
                f"energy must be scalar, got shape {array.shape}",
                frame_index=frame_index,
                sample_id=sample_id,
                label=name,
            )
        tensor = torch.tensor(float(array), dtype=torch.float64)
    elif name == "forces":
        if array.shape != (num_atoms, 3):
            raise ExtXYZLoadError(
                "MALFORMED_LABEL",
                f"forces must have shape [{num_atoms},3], got {array.shape}",
                frame_index=frame_index,
                sample_id=sample_id,
                label=name,
            )
        tensor = torch.tensor(array, dtype=torch.float64)
    elif name == "stress":
        if array.shape == (6,):
            from ase.stress import voigt_6_to_full_3x3_stress

            array = np.array(
                voigt_6_to_full_3x3_stress(array),
                dtype=np.float64,
                copy=True,
            )
        elif array.shape != (3, 3):
            raise ExtXYZLoadError(
                "MALFORMED_LABEL",
                f"stress must have shape [6] or [3,3], got {array.shape}",
                frame_index=frame_index,
                sample_id=sample_id,
                label=name,
            )
        scale = max(1.0, float(np.max(np.abs(array))))
        symmetry_error = float(np.max(np.abs(array - array.T)))
        if symmetry_error > 64.0 * np.finfo(np.float64).eps * scale:
            raise ExtXYZLoadError(
                "MALFORMED_LABEL",
                f"stress is not symmetric; max error={symmetry_error}",
                frame_index=frame_index,
                sample_id=sample_id,
                label=name,
            )
        tensor = torch.tensor(0.5 * (array + array.T), dtype=torch.float64)
    else:  # pragma: no cover - internal invariant
        raise RuntimeError(f"unsupported canonical label {name}")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ExtXYZLoadError(
            "NONFINITE_LABEL",
            "label contains NaN or Inf",
            frame_index=frame_index,
            sample_id=sample_id,
            label=name,
        )
    return tensor


def _extract_label(
    atoms: Any,
    name: str,
    *,
    required: bool,
    frame_index: int,
    sample_id: str,
) -> torch.Tensor | None:
    results = (
        atoms.calc.results
        if atoms.calc is not None and isinstance(atoms.calc.results, Mapping)
        else {}
    )
    authoritative = results.get(name)
    if authoritative is None:
        if required:
            raise ExtXYZLoadError(
                "MISSING_LABEL",
                "required label is absent from atoms.calc.results",
                frame_index=frame_index,
                sample_id=sample_id,
                label=name,
            )
        return None
    canonical = _canonical_label(
        name,
        authoritative,
        num_atoms=len(atoms),
        frame_index=frame_index,
        sample_id=sample_id,
    )
    duplicate_sources = []
    if name in atoms.info:
        duplicate_sources.append(("Atoms.info", atoms.info[name]))
    if name in atoms.arrays:
        duplicate_sources.append(("Atoms.arrays", atoms.arrays[name]))
    for source, value in duplicate_sources:
        duplicate = _canonical_label(
            name,
            value,
            num_atoms=len(atoms),
            frame_index=frame_index,
            sample_id=sample_id,
        )
        if not torch.equal(canonical, duplicate):
            raise ExtXYZLoadError(
                "CONFLICTING_LABEL",
                f"atoms.calc.results conflicts with {source}",
                frame_index=frame_index,
                sample_id=sample_id,
                label=name,
            )
    return canonical


def _hash_text(digest: Any, value: str) -> None:
    raw = value.encode("utf-8")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _hash_tensor(digest: Any, name: str, tensor: torch.Tensor) -> None:
    _hash_text(digest, name)
    _hash_text(digest, str(tensor.dtype))
    _hash_text(digest, ",".join(str(size) for size in tensor.shape))
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.numpy().tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _semantic_digest(
    samples: tuple[StructureSample, ...],
    config: ExtXYZLoadConfig,
    template_fingerprint: str,
) -> str:
    digest = hashlib.sha256()
    for value in (
        config.convention_version,
        config.unit_convention_version,
        config.sample_id_prefix,
        config.template_id,
        template_fingerprint,
        config.dtype_name,
        config.origin_convention,
        str(config.require_energy),
        str(config.require_forces),
        str(config.require_stress),
        str(len(samples)),
    ):
        _hash_text(digest, value)
    for index, sample in enumerate(samples):
        _hash_text(digest, f"frame:{index}")
        _hash_text(digest, sample.sample_id)
        for name in (
            "positions",
            "atomic_numbers",
            "cell",
            "pbc",
            "origin",
        ):
            _hash_tensor(digest, name, getattr(sample, name))
        for name in ("energy", "forces", "stress"):
            value = getattr(sample, name)
            _hash_text(digest, f"{name}_present:{value is not None}")
            if value is not None:
                _hash_tensor(digest, name, value)
        _hash_text(
            digest,
            "force_mask:implicit_all_true"
            if sample.forces is not None
            else "force_mask:missing",
        )
        _hash_text(
            digest,
            "stress_mask:implicit_all_true"
            if sample.stress is not None
            else "stress_mask:missing",
        )
    return digest.hexdigest()


def _stress_voigt_components(stress: torch.Tensor) -> list[float]:
    return [
        float(stress[0, 0]),
        float(stress[1, 1]),
        float(stress[2, 2]),
        float(stress[1, 2]),
        float(stress[0, 2]),
        float(stress[0, 1]),
    ]


def load_extxyz_samples(
    config: ExtXYZLoadConfig,
    template_registry: TemplateRegistry,
) -> ExtXYZLoadResult:
    """Load ordered frames without building or inferring a reference template."""

    if not isinstance(config, ExtXYZLoadConfig):
        raise TypeError("config must be an ExtXYZLoadConfig")
    if not isinstance(template_registry, TemplateRegistry):
        raise TypeError("template_registry must be a TemplateRegistry")
    source = Path(config.source_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("extxyz source path must be a regular file")
    try:
        template = template_registry.resolve(config.template_id)
    except KeyError as error:
        raise ValueError(
            f"unknown explicit template_id: {config.template_id}"
        ) from error
    if template.strict_domain is None:
        raise ValueError("production extxyz loading requires a strict template domain")
    registry_fingerprint = template_registry.fingerprint

    try:
        from ase.io import iread
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("ASE is required to load extxyz data") from error

    samples = []
    compositions: Counter[tuple[int, ...]] = Counter()
    atom_counts: Counter[int] = Counter()
    cell_volumes: list[float] = []
    strains: list[float] = []
    energies: list[float] = []
    force_components: list[float] = []
    force_norms: list[float] = []
    stress_components: list[float] = []
    stress_norms: list[float] = []
    missing = {"energy": 0, "forces": 0, "stress": 0}
    requirements = {
        "energy": config.require_energy,
        "forces": config.require_forces,
        "stress": config.require_stress,
    }
    try:
        iterator = iread(str(source), index=":", format="extxyz")
        for frame_index, atoms in enumerate(iterator):
            sample_id = f"{config.sample_id_prefix}:{frame_index:06d}"
            positions64 = torch.tensor(
                atoms.get_positions().copy(), dtype=torch.float64
            )
            atomic_numbers64 = torch.tensor(
                atoms.get_atomic_numbers().copy(), dtype=torch.long
            )
            cell64 = torch.tensor(atoms.cell.array.copy(), dtype=torch.float64)
            pbc_cpu = torch.tensor(atoms.pbc.copy(), dtype=torch.bool)
            if positions64.shape != (len(atoms), 3):
                raise ExtXYZLoadError(
                    "MALFORMED_GEOMETRY",
                    "positions must have shape [N,3]",
                    frame_index=frame_index,
                    sample_id=sample_id,
                )
            if atomic_numbers64.numel() and bool(
                torch.any(atomic_numbers64 <= 0)
            ):
                raise ExtXYZLoadError(
                    "MALFORMED_GEOMETRY",
                    "atomic numbers must be positive",
                    frame_index=frame_index,
                    sample_id=sample_id,
                )
            if not bool(torch.all(torch.isfinite(positions64))) or not bool(
                torch.all(torch.isfinite(cell64))
            ):
                raise ExtXYZLoadError(
                    "NONFINITE_GEOMETRY",
                    "positions or cell contain NaN or Inf",
                    frame_index=frame_index,
                    sample_id=sample_id,
                )
            if pbc_cpu.shape != (3,) or not bool(torch.all(pbc_cpu)):
                raise ExtXYZLoadError(
                    "NONPERIODIC_STRUCTURE",
                    "only full PBC extxyz frames are supported",
                    frame_index=frame_index,
                    sample_id=sample_id,
                )
            if cell64.shape != (3, 3) or bool(
                torch.linalg.svdvals(cell64)[-1]
                <= torch.finfo(torch.float64).eps
            ):
                raise ExtXYZLoadError(
                    "MALFORMED_GEOMETRY",
                    "cell must be nonsingular [3,3]",
                    frame_index=frame_index,
                    sample_id=sample_id,
                )

            labels = {
                name: _extract_label(
                    atoms,
                    name,
                    required=requirements[name],
                    frame_index=frame_index,
                    sample_id=sample_id,
                )
                for name in ("energy", "forces", "stress")
            }
            for name, value in labels.items():
                if value is None:
                    missing[name] += 1
            try:
                domain = template.validate_structure(
                    atomic_numbers64,
                    cell=cell64,
                    pbc=pbc_cpu,
                    sample_id=sample_id,
                )
            except ValueError as error:
                raise ExtXYZLoadError(
                    "TEMPLATE_DOMAIN_REJECTION",
                    str(error),
                    frame_index=frame_index,
                    sample_id=sample_id,
                ) from error

            device = torch.device(config.device)
            floating = lambda value: value.detach().clone().to(
                device=device, dtype=config.dtype
            )
            fixed = lambda value: value.detach().clone().to(device=device)
            sample = StructureSample(
                sample_id=sample_id,
                positions=floating(positions64),
                atomic_numbers=fixed(atomic_numbers64),
                cell=floating(cell64),
                pbc=fixed(pbc_cpu),
                origin=torch.zeros(3, dtype=config.dtype, device=device),
                template_id=config.template_id,
                energy=(
                    None if labels["energy"] is None else floating(labels["energy"])
                ),
                forces=(
                    None if labels["forces"] is None else floating(labels["forces"])
                ),
                stress=(
                    None if labels["stress"] is None else floating(labels["stress"])
                ),
            )
            samples.append(sample)
            compositions[domain.composition] += 1
            atom_counts[domain.num_atoms] += 1
            cell_volumes.append(float(torch.abs(torch.linalg.det(cell64))))
            strains.append(float(domain.maximum_strain_seen))
            if labels["energy"] is not None:
                energies.append(float(labels["energy"]))
            if labels["forces"] is not None:
                force_components.extend(
                    float(value) for value in labels["forces"].reshape(-1)
                )
                force_norms.extend(
                    float(value)
                    for value in torch.linalg.vector_norm(
                        labels["forces"], dim=-1
                    )
                )
            if labels["stress"] is not None:
                stress_components.extend(
                    _stress_voigt_components(labels["stress"])
                )
                stress_norms.append(float(torch.linalg.matrix_norm(labels["stress"])))
    except ExtXYZLoadError:
        raise
    except Exception as error:
        raise ExtXYZLoadError(
            "ASE_PARSE_FAILURE", f"ASE extxyz parse failed: {error}"
        ) from error

    samples_tuple = tuple(samples)
    if not samples_tuple:
        raise ExtXYZLoadError("EMPTY_SOURCE", "extxyz source contains no frames")
    sample_ids = tuple(sample.sample_id for sample in samples_tuple)
    if len(set(sample_ids)) != len(sample_ids):
        raise ExtXYZLoadError("DUPLICATE_SAMPLE_ID", "generated sample IDs collide")
    if template_registry.fingerprint != registry_fingerprint:
        raise RuntimeError("template registry changed during extxyz loading")
    semantic_sha256 = _semantic_digest(samples_tuple, config, template.fingerprint)
    diagnostics = ExtXYZLoadDiagnostics(
        source_logical_name=config.sample_id_prefix,
        frame_count=len(samples_tuple),
        first_sample_id=sample_ids[0],
        last_sample_id=sample_ids[-1],
        template_id=config.template_id,
        template_fingerprint=template.fingerprint,
        composition_histogram=tuple(sorted(compositions.items())),
        atom_count_histogram=tuple(sorted(atom_counts.items())),
        cell_volume=_statistics(cell_volumes),
        strain=_statistics(strains),
        energy=_statistics(energies),
        force_components=_statistics(force_components),
        force_norms=_statistics(force_norms),
        stress_components=_statistics(stress_components),
        stress_frobenius_norms=_statistics(stress_norms),
        missing_energy_count=missing["energy"],
        missing_forces_count=missing["forces"],
        missing_stress_count=missing["stress"],
        nonfinite_label_count=0,
        semantic_sha256=semantic_sha256,
        output_dtype=config.dtype_name,
    )
    return ExtXYZLoadResult(samples_tuple, diagnostics)


def load_extxyz_dataset(
    configs: ExtXYZLoadConfig | Sequence[ExtXYZLoadConfig],
    template_registry: TemplateRegistry,
) -> InMemoryStructureDataset:
    """Load one or more ordered sources and reject cross-source ID collisions."""

    if isinstance(configs, ExtXYZLoadConfig):
        resolved = (configs,)
    elif isinstance(configs, Sequence) and not isinstance(configs, (str, bytes)):
        resolved = tuple(configs)
    else:
        raise TypeError("configs must be ExtXYZLoadConfig or a deterministic Sequence")
    if not resolved or any(
        not isinstance(value, ExtXYZLoadConfig) for value in resolved
    ):
        raise ValueError("configs must contain at least one ExtXYZLoadConfig")
    samples = tuple(
        sample
        for config in resolved
        for sample in load_extxyz_samples(config, template_registry).samples
    )
    return InMemoryStructureDataset(samples, template_registry)
