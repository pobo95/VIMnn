"""Deterministic POSCAR-first reference preparation for beginner recipes.

This module performs geometry-only preparation.  It deliberately does not
construct a model, solve transport, inspect labels, or consume random state.
The generated objects use the existing strict reference builder and phase
validation contracts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceTemplateBuilderConfig,
    StrictTemplateDomain,
    build_reference_template_from_atoms,
    capture_reference_structure_artifact,
)
from refsite_mlip.graph import (
    build_reference_graph_topology,
    update_reference_edge_geometry,
)
from refsite_mlip.phase import phase_gradient_hessian, phase_objective
from refsite_mlip.phase.modes import validate_static_mode_amplitudes
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.phase.stabilizer import (
    find_typed_stabilizer,
    stabilizer_equivalent,
    validate_alias_matches_stabilizer,
)


AUTO_REFERENCE_CONVENTION_VERSION = "poscar_first_reference_preparation_v1"
AUTO_STRAIN_POLICY_VERSION = "dataset_bounded_strain_margin_v1"
AUTO_PHASE_SEARCH_VERSION = "bounded_typed_reciprocal_search_v1"
AUTO_STRAIN_MARGIN = 0.005
AUTO_STRAIN_GRID = 0.005
AUTO_STRAIN_MINIMUM = 0.010
AUTO_STRAIN_CEILING = 0.050
_MODE_SEARCH_BOUND = 12
_PHASE_MINIMUM_AMPLITUDE = 1.0e-12
_PHASE_MINIMUM_CURVATURE = 1.0e-10
_PHASE_MAXIMUM_CONDITION = 1.0e12
_PHASE_MAXIMUM_RESIDUAL = 1.0e-10
_PHASE_MINIMUM_GRID_GAP = 1.0e-10


class AutomaticReferenceError(ValueError):
    """Structured failure raised by POSCAR-first preparation."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        source_path: str | None = None,
        sample_id: str | None = None,
        template_id: str | None = None,
        expected: Any = None,
        actual: Any = None,
        diagnostics: Any = None,
        original_error: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.message = message
        self.stage = stage
        self.source_path = source_path
        self.sample_id = sample_id
        self.template_id = template_id
        self.expected = expected
        self.actual = actual
        self.diagnostics = diagnostics
        self.original_error = original_error
        context = " ".join(
            f"{name}={value!r}"
            for name, value in (
                ("source_path", source_path),
                ("sample_id", sample_id),
                ("template_id", template_id),
                ("expected", expected),
                ("actual", actual),
            )
            if value is not None
        )
        suffix = "" if not context else " " + context
        super().__init__(f"[{reason_code}] stage={stage!r}{suffix} {message}")


def _error(reason: str, message: str, *, stage: str, **context: Any) -> AutomaticReferenceError:
    return AutomaticReferenceError(reason, message, stage=stage, **context)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("automatic reference metadata is nonfinite")
        return value
    raise TypeError(f"automatic reference metadata contains {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return _plain(value)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Geometry:
    sample_id: str
    split: str
    source_index: int
    frame_index: int
    atomic_numbers: tuple[int, ...]
    cell: tuple[tuple[float, float, float], ...]
    pbc: tuple[bool, bool, bool]
    exact_template_id: str | None

    @property
    def num_atoms(self) -> int:
        return len(self.atomic_numbers)

    def cell_tensor(self) -> torch.Tensor:
        return torch.tensor(self.cell, dtype=torch.float64)


@dataclass(frozen=True)
class AutomaticReferenceResult:
    """One generated specification plus immutable dataset-bounded certificate."""

    source_index: int
    original_poscar: str
    resolved_poscar: str
    specification: Any
    certificate: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "certificate", _freeze(self.certificate))

    @property
    def template_id(self) -> str:
        return self.specification.builder.template_id

    def to_dict(self) -> dict[str, Any]:
        result = _plain(self.certificate)
        result["poscar"] = self.original_poscar
        return result

    def semantic_dict(self) -> dict[str, Any]:
        """Return path-free content used by reference fingerprints."""

        return _plain(self.certificate)


@dataclass(frozen=True)
class AutomaticReferencePreparation:
    results: tuple[AutomaticReferenceResult, ...]
    train_assignments: tuple[tuple[str, str], ...]
    validation_assignments: tuple[tuple[str, str], ...]
    species_vocabulary: tuple[int, ...]
    content_fingerprint: str
    convention_version: str = AUTO_REFERENCE_CONVENTION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "train_assignments", tuple(self.train_assignments))
        object.__setattr__(self, "validation_assignments", tuple(self.validation_assignments))
        object.__setattr__(self, "species_vocabulary", tuple(self.species_vocabulary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "convention_version": self.convention_version,
            "scope": "dataset_bounded",
            "species_vocabulary": list(self.species_vocabulary),
            "references": [result.to_dict() for result in self.results],
            "assignments": {
                "train": [
                    {"sample_id": sample_id, "template_id": template_id}
                    for sample_id, template_id in self.train_assignments
                ],
                "validation": [
                    {"sample_id": sample_id, "template_id": template_id}
                    for sample_id, template_id in self.validation_assignments
                ],
            },
            "content_fingerprint": self.content_fingerprint,
        }


def resolve_auto_maximum_strain(observed: float) -> float:
    """Apply the versioned integer-tick auto-strain policy exactly."""

    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise TypeError("observed strain must be a real number")
    value = float(observed)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("observed strain must be finite and nonnegative")
    # Work in 0.005 ticks.  A tiny binary guard prevents an exactly integral
    # mathematical quotient from being rounded to the next tick.
    target = max(AUTO_STRAIN_MINIMUM, value + AUTO_STRAIN_MARGIN)
    ticks = math.ceil(target / AUTO_STRAIN_GRID - 8.0 * math.ulp(target / AUTO_STRAIN_GRID))
    return float(ticks) * AUTO_STRAIN_GRID


def _read_atoms(payload: bytes, *, path: Path, format_name: str, multiple: bool) -> tuple[Any, ...]:
    try:
        from ase.io import iread, read
    except ImportError as error:  # pragma: no cover - dependency contract
        raise _error(
            "ASE_UNAVAILABLE", "ASE is required for automatic reference preparation",
            stage="automatic_reference.parse", source_path=str(path), original_error=error,
        ) from error
    try:
        text = payload.decode("utf-8")
        if multiple:
            values = tuple(iread(io.StringIO(text), index=":", format=format_name))
        else:
            value = read(io.StringIO(text), index=0, format=format_name)
            values = (value,)
    except Exception as error:
        raise _error(
            "STRUCTURE_PARSE_FAILED", f"ASE could not parse {format_name}: {error}",
            stage="automatic_reference.parse", source_path=str(path), original_error=error,
        ) from error
    if not values:
        raise _error(
            "EMPTY_STRUCTURE_SOURCE", "structure input contains no frames",
            stage="automatic_reference.parse", source_path=str(path),
        )
    return values


def _validate_atoms(atoms: Any, *, source_path: Path, sample_id: str) -> None:
    pbc = tuple(bool(value) for value in atoms.pbc.tolist())
    if pbc != (True, True, True):
        raise _error(
            "FULL_PBC_REQUIRED", "automatic reference preparation requires full PBC",
            stage="automatic_reference.geometry", source_path=str(source_path), sample_id=sample_id,
        )
    numbers = tuple(int(value) for value in atoms.get_atomic_numbers().tolist())
    if not numbers or any(value <= 0 for value in numbers):
        raise _error(
            "INVALID_ATOMIC_NUMBERS", "structure must contain positive atomic numbers",
            stage="automatic_reference.geometry", source_path=str(source_path), sample_id=sample_id,
        )
    cell = torch.tensor(atoms.cell.array.copy(), dtype=torch.float64)
    positions = torch.tensor(atoms.positions.copy(), dtype=torch.float64)
    if cell.shape != (3, 3) or positions.shape != (len(numbers), 3):
        raise _error(
            "INVALID_STRUCTURE_SHAPE", "structure geometry has an invalid shape",
            stage="automatic_reference.geometry", source_path=str(source_path), sample_id=sample_id,
        )
    if not bool(torch.all(torch.isfinite(cell))) or not bool(torch.all(torch.isfinite(positions))):
        raise _error(
            "NONFINITE_STRUCTURE", "structure geometry contains NaN or Infinity",
            stage="automatic_reference.geometry", source_path=str(source_path), sample_id=sample_id,
        )
    if bool(torch.linalg.svdvals(cell)[-1] <= torch.finfo(torch.float64).eps):
        raise _error(
            "SINGULAR_CELL", "structure cell is singular",
            stage="automatic_reference.geometry", source_path=str(source_path), sample_id=sample_id,
        )


def _geometry(atoms: Any, *, path: Path, split: str, source_index: int, frame_index: int, exact: str | None) -> _Geometry:
    sample_id = f"{split}.{source_index:04d}:{frame_index:06d}"
    _validate_atoms(atoms, source_path=path, sample_id=sample_id)
    return _Geometry(
        sample_id=sample_id,
        split=split,
        source_index=source_index,
        frame_index=frame_index,
        atomic_numbers=tuple(int(value) for value in atoms.get_atomic_numbers().tolist()),
        cell=tuple(tuple(float(value) for value in row) for row in atoms.cell.array.tolist()),
        pbc=(True, True, True),
        exact_template_id=exact,
    )


def _canonical_reference_content(atoms: Any) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    numbers = torch.tensor(atoms.get_atomic_numbers().copy(), dtype=torch.long)
    cell = torch.tensor(atoms.cell.array.copy(), dtype=torch.float64)
    fractional = torch.tensor(atoms.get_scaled_positions(wrap=False).copy(), dtype=torch.float64)
    fractional = fractional - torch.floor(fractional)
    tolerance = 64.0 * torch.finfo(torch.float64).eps
    fractional = torch.round(fractional / tolerance) * tolerance
    fractional = torch.where(
        (torch.abs(fractional) <= 1.0e-10) | (torch.abs(fractional - 1.0) <= 1.0e-10),
        torch.zeros_like(fractional),
        fractional,
    )
    keys = [
        (int(numbers[index]),) + tuple(float(value) for value in fractional[index].tolist())
        for index in range(numbers.numel())
    ]
    order = torch.tensor(sorted(range(numbers.numel()), key=keys.__getitem__), dtype=torch.long)
    numbers = numbers[order].contiguous()
    fractional = fractional[order].contiguous()
    payload = {
        "atomic_numbers": numbers.tolist(),
        "fractional_positions": fractional.tolist(),
        "cell": cell.tolist(),
        "pbc": [True, True, True],
    }
    return _fingerprint(payload), numbers, fractional, cell


def _composition(numbers: Sequence[int], species: tuple[int, ...]) -> tuple[int, ...]:
    counts = Counter(int(value) for value in numbers)
    return tuple(counts.get(value, 0) for value in species)


def _strain(reference_cell: torch.Tensor, current_cell: torch.Tensor) -> float:
    deformation = torch.linalg.solve(reference_cell, current_cell)
    value = torch.linalg.matrix_norm(
        deformation - torch.eye(3, dtype=torch.float64), ord=2
    )
    return float(value)


def _infer_supercell_shape(stabilizer: Any) -> tuple[int, int, int]:
    translations = stabilizer.translations.detach().cpu().to(torch.float64)
    result = []
    tolerance = 1.0e-9
    for axis in range(3):
        others = [index for index in range(3) if index != axis]
        candidates = []
        for row in translations:
            wrapped = row - torch.floor(row)
            if all(min(abs(float(wrapped[item])), abs(float(wrapped[item]) - 1.0)) <= tolerance for item in others):
                value = float(wrapped[axis])
                value = min(value, 1.0 - value) if value > tolerance else 0.0
                if value > tolerance:
                    candidates.append(value)
        if not candidates:
            result.append(1)
            continue
        denominator = int(round(1.0 / min(candidates)))
        result.append(max(1, denominator))
    return tuple(result)  # type: ignore[return-value]


def _typed_orbit_count(
    atomic_numbers: torch.Tensor, stabilizer: Any, species: int
) -> int:
    remaining = set(
        int(value)
        for value in torch.nonzero(atomic_numbers == species).reshape(-1).tolist()
    )
    count = 0
    while remaining:
        seed = min(remaining)
        orbit = {
            int(row[seed])
            for row in stabilizer.permutations.detach().cpu()
        }
        remaining.difference_update(orbit)
        count += 1
    return count


def _canonical_mode(vector: tuple[int, int, int]) -> tuple[int, int, int]:
    for value in vector:
        if value:
            return vector if value > 0 else tuple(-item for item in vector)
    return vector


def _determinant3(rows: Sequence[tuple[int, int, int]]) -> int:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _mode_amplitude(
    mode: tuple[int, int, int],
    fractional: torch.Tensor,
    site_types: torch.Tensor,
    num_types: int,
) -> float:
    vector = torch.tensor(mode, dtype=torch.float64)
    angles = 2.0 * math.pi * (fractional @ vector)
    phases = torch.polar(torch.ones_like(angles), angles)
    total = 0.0
    for site_type in range(num_types):
        field = phases[site_types == site_type].sum()
        total += float(field.abs().square())
    return total


def _automatic_phase(
    fractional: torch.Tensor,
    site_types: torch.Tensor,
    cell: torch.Tensor,
    stabilizer: Any,
    num_types: int,
) -> tuple[PhaseSpecification, dict[str, Any]]:
    size = int(stabilizer.translations.shape[0])
    candidates: list[tuple[int, int, int]] = []
    seen = set()
    for vector in itertools.product(range(-_MODE_SEARCH_BOUND, _MODE_SEARCH_BOUND + 1), repeat=3):
        if vector == (0, 0, 0):
            continue
        canonical = _canonical_mode(vector)
        if canonical in seen:
            continue
        seen.add(canonical)
        mode = torch.tensor(canonical, dtype=torch.float64)
        phases = stabilizer.translations.to(torch.float64) @ mode
        if bool(torch.any(torch.abs(phases - torch.round(phases)) > 1.0e-9)):
            continue
        if _mode_amplitude(canonical, fractional, site_types, num_types) <= _PHASE_MINIMUM_AMPLITUDE:
            continue
        candidates.append(canonical)
    candidates.sort(key=lambda value: (sum(item * item for item in value), sum(abs(item) for item in value), value))
    limited = candidates[:256]
    primary = None
    for indices in itertools.combinations(range(len(limited)), 3):
        rows = [limited[index] for index in indices]
        determinant = abs(_determinant3(rows))
        if determinant != size:
            continue
        modes = torch.tensor(rows, dtype=torch.long)
        try:
            validate_alias_matches_stabilizer(modes, stabilizer, tolerance=1.0e-9)
        except ValueError:
            continue
        primary = [limited[index] for index in indices]
        break
    if primary is None:
        raise _error(
            "PHASE_PRIMARY_MODE_SEARCH_FAILED",
            "bounded reciprocal search found no rank-3 mode set with the exact typed-stabilizer alias group",
            stage="automatic_reference.phase", actual={"stabilizer_size": size, "candidate_count": len(candidates)},
        )
    selected = list(primary)
    for candidate in candidates:
        if candidate in selected:
            continue
        trial = torch.tensor(selected + [candidate], dtype=torch.long)
        try:
            validate_alias_matches_stabilizer(trial, stabilizer, tolerance=1.0e-9)
        except ValueError:
            continue
        selected.append(candidate)
        if len(selected) == 6:
            break
    modes = torch.tensor(selected, dtype=torch.long)
    identity = torch.eye(num_types, dtype=torch.float64)
    specification = PhaseSpecification(
        modes=modes,
        mode_weights=torch.ones(len(selected), dtype=torch.float64),
        site_type_alignment_weights=identity,
        channel_weights=torch.ones(num_types, dtype=torch.float64),
        approval_status="provisional",
        convention_version="automatic_phase_specification_v1",
    )
    positions = fractional @ cell
    atomic, reference, cross = typed_reciprocal_fields(
        positions,
        torch.zeros(3, dtype=torch.float64),
        cell,
        fractional,
        identity[site_types],
        identity[site_types],
        modes,
        specification.channel_weights,
    )
    amplitudes = validate_static_mode_amplitudes(
        reference, specification.channel_weights, _PHASE_MINIMUM_AMPLITUDE
    )
    phase = torch.zeros(3, dtype=torch.float64)
    gradient, hessian = phase_gradient_hessian(
        phase, cross, modes, specification.mode_weights
    )
    curvature = torch.linalg.eigvalsh(-hessian)
    minimum_curvature = float(curvature.min())
    maximum_curvature = float(curvature.max())
    if minimum_curvature <= _PHASE_MINIMUM_CURVATURE or not math.isfinite(minimum_curvature):
        raise _error(
            "PHASE_HESSIAN_CERTIFICATE_FAILED",
            "automatic phase Hessian is not strictly negative definite at the reference origin",
            stage="automatic_reference.phase", actual=minimum_curvature,
        )
    condition = maximum_curvature / minimum_curvature
    residual = float(torch.linalg.vector_norm(gradient))
    if not math.isfinite(condition) or condition > _PHASE_MAXIMUM_CONDITION:
        raise _error(
            "PHASE_HESSIAN_CERTIFICATE_FAILED",
            "automatic phase Hessian condition exceeds the fixed search certificate",
            stage="automatic_reference.phase", expected=f"<= {_PHASE_MAXIMUM_CONDITION}", actual=condition,
        )
    if not math.isfinite(residual) or residual > _PHASE_MAXIMUM_RESIDUAL:
        raise _error(
            "PHASE_RESIDUAL_CERTIFICATE_FAILED",
            "automatic phase residual exceeds the fixed search certificate",
            stage="automatic_reference.phase", expected=f"<= {_PHASE_MAXIMUM_RESIDUAL}", actual=residual,
        )
    baseline = float(phase_objective(phase, cross, modes, specification.mode_weights))
    competitor_scores = []
    for coordinates in itertools.product((0.0, 0.25, 0.5, 0.75), repeat=3):
        candidate = torch.tensor(coordinates, dtype=torch.float64)
        if stabilizer_equivalent(candidate, phase, stabilizer, tolerance=1.0e-10):
            continue
        competitor_scores.append(
            float(phase_objective(candidate, cross, modes, specification.mode_weights))
        )
    if not competitor_scores:
        raise _error(
            "PHASE_GAP_CERTIFICATE_FAILED", "phase gap audit found no non-equivalent candidate",
            stage="automatic_reference.phase",
        )
    gap = baseline - max(competitor_scores)
    if not math.isfinite(gap) or gap <= _PHASE_MINIMUM_GRID_GAP:
        raise _error(
            "PHASE_GAP_CERTIFICATE_FAILED",
            "automatic phase candidates are tied at certificate resolution",
            stage="automatic_reference.phase", actual=gap,
        )
    return specification, {
        "search_version": AUTO_PHASE_SEARCH_VERSION,
        "candidate_order": "squared_norm,l1_norm,lexicographic; sign fixed by first nonzero component",
        "candidate_bound": _MODE_SEARCH_BOUND,
        "candidate_count": len(candidates),
        "primary_modes": [list(value) for value in primary],
        "secondary_modes": [list(value) for value in selected[3:]],
        "mode_amplitudes": amplitudes.tolist(),
        "final_residual": residual,
        "hessian_minimum_curvature": minimum_curvature,
        "hessian_condition": condition,
        "non_equivalent_candidate_objective_gap": gap,
        "certificate_thresholds": {
            "minimum_mode_amplitude": _PHASE_MINIMUM_AMPLITUDE,
            "minimum_curvature": _PHASE_MINIMUM_CURVATURE,
            "maximum_condition": _PHASE_MAXIMUM_CONDITION,
            "maximum_residual": _PHASE_MAXIMUM_RESIDUAL,
            "minimum_grid_objective_gap": _PHASE_MINIMUM_GRID_GAP,
        },
        "approval_status": "provisional",
    }


def _load_geometries(
    data: Any,
    *,
    base: Path,
    read_file: Callable[..., tuple[Path, bytes]],
) -> tuple[tuple[_Geometry, ...], tuple[_Geometry, ...]]:
    output = []
    for split, sources in (("train", data.train), ("validation", data.validation)):
        frames: list[_Geometry] = []
        for source_index, source in enumerate(sources):
            path = Path(source.path)
            if not path.is_absolute():
                path = base / path
            resolved, payload = read_file(path, text=False)
            atoms_values = _read_atoms(payload, path=resolved, format_name="extxyz", multiple=True)
            frames.extend(
                _geometry(
                    atoms,
                    path=resolved,
                    split=split,
                    source_index=source_index,
                    frame_index=frame_index,
                    exact=source.template_id,
                )
                for frame_index, atoms in enumerate(atoms_values)
            )
        output.append(tuple(frames))
    return output[0], output[1]


def prepare_automatic_references(
    recipe: Any,
    *,
    base: Path,
    read_file: Callable[..., tuple[Path, bytes]],
    specification_factory: Callable[..., Any],
) -> AutomaticReferencePreparation:
    """Generate canonical references from automatic recipe sources."""

    reference_inputs = []
    raw_content = set()
    for index, source in enumerate(recipe.reference.sources):
        path = Path(source.poscar)
        if not path.is_absolute():
            path = base / path
        resolved, raw = read_file(path, text=False)
        atoms = _read_atoms(raw, path=resolved, format_name="vasp", multiple=False)[0]
        _validate_atoms(atoms, source_path=resolved, sample_id=f"reference[{index}]")
        semantic_sha, numbers, fractional, cell = _canonical_reference_content(atoms)
        if semantic_sha in raw_content:
            raise _error(
                "DUPLICATE_REFERENCE_CONTENT",
                "the same canonical POSCAR content was supplied more than once",
                stage="automatic_reference.identity", source_path=str(resolved), actual=semantic_sha,
            )
        raw_content.add(semantic_sha)
        template_id = source.template_id or f"ref_m{len(numbers)}_{semantic_sha[:12]}"
        reference_inputs.append(
            {
                "source_index": index,
                "source": source,
                "path": resolved,
                "raw_sha": hashlib.sha256(raw).hexdigest(),
                "semantic_sha": semantic_sha,
                "atoms": atoms,
                "numbers": numbers,
                "fractional": fractional,
                "cell": cell,
                "template_id": template_id,
            }
        )
    ids = [item["template_id"] for item in reference_inputs]
    if len(ids) != len(set(ids)):
        raise _error(
            "DUPLICATE_TEMPLATE_ID", "automatic reference template IDs collide",
            stage="automatic_reference.identity", actual=ids,
        )
    reference_inputs.sort(key=lambda item: (item["template_id"], item["semantic_sha"]))

    train, validation = _load_geometries(recipe.data, base=base, read_file=read_file)
    reference_species = {
        int(value) for item in reference_inputs for value in item["numbers"].tolist()
    }
    train_species = {value for frame in train for value in frame.atomic_numbers}
    validation_species = {value for frame in validation for value in frame.atomic_numbers}
    validation_only = validation_species - train_species
    if validation_only:
        raise _error(
            "VALIDATION_ONLY_SPECIES",
            "validation contains species absent from every POSCAR and train split",
            stage="automatic_reference.species", actual=tuple(sorted(validation_only)),
        )
    species = tuple(sorted(reference_species | train_species))
    if any(value not in reference_species for value in train_species):
        raise _error(
            "UNSUPPORTED_SPECIES", "train data contains species absent from every reference",
            stage="automatic_reference.species", actual=tuple(sorted(train_species - reference_species)),
        )

    for item in reference_inputs:
        item["reference_composition"] = _composition(item["numbers"].tolist(), species)
        item["site_types"] = torch.tensor(
            [species.index(int(value)) for value in item["numbers"].tolist()], dtype=torch.long
        )
        item["stabilizer"] = find_typed_stabilizer(
            item["fractional"], item["site_types"], tolerance=1.0e-10
        )

    def assign(frame: _Geometry) -> tuple[str, tuple[dict[str, Any], ...]]:
        candidates = []
        diagnostics = []
        for item in reference_inputs:
            template_id = item["template_id"]
            reasons = []
            if frame.exact_template_id is not None and frame.exact_template_id != template_id:
                reasons.append("not the configured exact template_id")
            composition = _composition(frame.atomic_numbers, species)
            reference_composition = item["reference_composition"]
            if frame.num_atoms > len(item["numbers"]):
                reasons.append(f"N={frame.num_atoms} exceeds M={len(item['numbers'])}")
            excess = tuple(
                max(0, value - allowed)
                for value, allowed in zip(composition, reference_composition)
            )
            if any(excess):
                reasons.append(f"species counts exceed reference composition by {excess}")
            observed = _strain(item["cell"], frame.cell_tensor())
            limit = (
                AUTO_STRAIN_CEILING
                if item["source"].maximum_strain == "auto"
                else float(item["source"].maximum_strain)
            )
            if observed > limit + 32.0 * torch.finfo(torch.float64).eps:
                if (
                    item["source"].maximum_strain != "auto"
                    and (
                        frame.exact_template_id == template_id
                        or len(reference_inputs) == 1
                    )
                ):
                    raise _error(
                        "EXPLICIT_MAXIMUM_STRAIN_TOO_SMALL",
                        "supplied maximum_strain does not cover the dataset frame selected for this reference",
                        stage="automatic_reference.strain",
                        template_id=template_id,
                        sample_id=frame.sample_id,
                        expected=observed,
                        actual=limit,
                    )
                reasons.append(f"row-vector cell strain {observed:.17g} exceeds {limit:.17g}")
            diagnostics.append(
                {"template_id": template_id, "approved": not reasons, "reasons": reasons, "observed_strain": observed}
            )
            if not reasons:
                candidates.append(template_id)
        if not candidates:
            raise _error(
                "NO_COMPATIBLE_TEMPLATE",
                "no automatic reference passed complete composition/cell domain screening",
                stage="automatic_reference.assignment", sample_id=frame.sample_id,
                diagnostics=diagnostics,
            )
        if len(candidates) != 1:
            raise _error(
                "AMBIGUOUS_TEMPLATE_ASSIGNMENT",
                "multiple automatic references passed complete composition/cell domain screening",
                stage="automatic_reference.assignment", sample_id=frame.sample_id,
                actual=tuple(candidates), diagnostics=diagnostics,
            )
        return candidates[0], tuple(diagnostics)

    train_selected = tuple((frame, *assign(frame)) for frame in train)
    validation_selected = tuple((frame, *assign(frame)) for frame in validation)
    derived = recipe.radii.derived
    results = []
    for item in reference_inputs:
        template_id = item["template_id"]
        assigned_train = [entry for entry in train_selected if entry[1] == template_id]
        assigned_validation = [entry for entry in validation_selected if entry[1] == template_id]
        if not assigned_train:
            raise _error(
                "UNUSED_AUTOMATIC_REFERENCE",
                "every automatic reference must own at least one train frame",
                stage="automatic_reference.assignment", template_id=template_id,
            )
        observed_train = max(_strain(item["cell"], entry[0].cell_tensor()) for entry in assigned_train)
        observed_validation = max(
            (_strain(item["cell"], entry[0].cell_tensor()) for entry in assigned_validation),
            default=0.0,
        )
        observed_all = max(observed_train, observed_validation)
        if item["source"].maximum_strain == "auto":
            maximum_strain = resolve_auto_maximum_strain(observed_all)
            if maximum_strain > AUTO_STRAIN_CEILING:
                raise _error(
                    "AUTO_MAXIMUM_STRAIN_LIMIT_EXCEEDED",
                    "dataset-bounded automatic strain exceeds the 0.05 safety ceiling; provide an explicit value",
                    stage="automatic_reference.strain", template_id=template_id,
                    expected=f"<= {AUTO_STRAIN_CEILING}", actual=maximum_strain,
                )
        else:
            maximum_strain = float(item["source"].maximum_strain)
            if maximum_strain + 32.0 * torch.finfo(torch.float64).eps < observed_all:
                limiting = max(
                    assigned_train + assigned_validation,
                    key=lambda entry: _strain(item["cell"], entry[0].cell_tensor()),
                )[0]
                raise _error(
                    "EXPLICIT_MAXIMUM_STRAIN_TOO_SMALL",
                    "supplied maximum_strain does not cover an assigned dataset frame",
                    stage="automatic_reference.strain", template_id=template_id,
                    sample_id=limiting.sample_id, expected=observed_all, actual=maximum_strain,
                )
        limiting_margin_mp = (1.0 - maximum_strain) * derived.r_candidate_mp - derived.r_mp
        limiting_margin_ot = (1.0 - maximum_strain) * derived.r_candidate_ot - derived.r_off_ot
        if limiting_margin_mp < 0.0 or limiting_margin_ot < 0.0:
            raise _error(
                "REFERENCE_RADIUS_CERTIFICATE_FAILED",
                "configured radii/skins cannot certify the requested maximum_strain; radii were not adjusted",
                stage="automatic_reference.certificate", template_id=template_id,
                diagnostics={
                    "maximum_strain": maximum_strain,
                    "r_ot": recipe.radii.r_ot,
                    "r_mp": recipe.radii.r_mp,
                    "mp_limiting_margin": limiting_margin_mp,
                    "ot_limiting_margin": limiting_margin_ot,
                },
            )
        compositions = {
            item["reference_composition"],
            *(_composition(entry[0].atomic_numbers, species) for entry in assigned_train),
            *(_composition(entry[0].atomic_numbers, species) for entry in assigned_validation),
        }
        ordered_compositions = tuple(sorted(compositions, key=lambda value: (-sum(value), value)))
        domain = StrictTemplateDomain(
            reference_site_count=len(item["numbers"]),
            supercell_shape=_infer_supercell_shape(item["stabilizer"]),
            species_vocabulary=species,
            reference_composition=item["reference_composition"],
            allowed_compositions=ordered_compositions,
            allowed_num_atoms=tuple(sum(value) for value in ordered_compositions),
            allowed_vacancy_masses=tuple(len(item["numbers"]) - sum(value) for value in ordered_compositions),
        )
        preliminary = build_reference_graph_topology(
            item["fractional"], item["site_types"], item["cell"],
            cutoff=recipe.radii.r_mp,
            skin=derived.r_candidate_mp - derived.r_mp,
            maximum_strain=maximum_strain,
            minimum_edge_length=1.0e-8,
        )
        geometry = update_reference_edge_geometry(preliminary, item["cell"], edge_length_scale=1.0)
        active_degrees = torch.bincount(
            preliminary.edge_index[1][geometry.active_mask], minlength=preliminary.num_sites
        )
        candidate_degrees = torch.bincount(preliminary.edge_index[1], minlength=preliminary.num_sites)
        if active_degrees.numel() == 0 or candidate_degrees.numel() == 0 or (
            int(active_degrees.min()) != int(active_degrees.max())
            or int(candidate_degrees.min()) != int(candidate_degrees.max())
            or int(active_degrees.min()) <= 0
        ):
            raise _error(
                "NONUNIFORM_REFERENCE_GRAPH",
                "existing reference builder requires uniform nonzero active/candidate degree",
                stage="automatic_reference.graph", template_id=template_id,
                diagnostics={
                    "active_degree_min": int(active_degrees.min()) if active_degrees.numel() else 0,
                    "active_degree_max": int(active_degrees.max()) if active_degrees.numel() else 0,
                    "candidate_degree_min": int(candidate_degrees.min()) if candidate_degrees.numel() else 0,
                    "candidate_degree_max": int(candidate_degrees.max()) if candidate_degrees.numel() else 0,
                },
            )
        builder = ReferenceTemplateBuilderConfig(
            template_id=template_id,
            strict_domain=domain,
            site_type_ids=tuple(range(len(species))),
            graph_cutoff=recipe.radii.r_mp,
            graph_skin=derived.r_candidate_mp - derived.r_mp,
            maximum_strain=maximum_strain,
            avg_num_neighbors=float(active_degrees.to(torch.float64).mean()),
            expected_active_degree=int(active_degrees[0]),
            expected_candidate_degree=int(candidate_degrees[0]),
            expected_stabilizer_size=int(item["stabilizer"].translations.shape[0]),
        )
        phase, phase_certificate = _automatic_phase(
            item["fractional"],
            item["site_types"],
            item["cell"],
            item["stabilizer"],
            len(species),
        )
        try:
            built = build_reference_template_from_atoms(
                item["atoms"], config=builder, phase_specification=phase
            )
            artifact = capture_reference_structure_artifact(built)
            for frame, _, _ in assigned_train + assigned_validation:
                built.template.validate_structure(
                    torch.tensor(frame.atomic_numbers, dtype=torch.long),
                    cell=frame.cell_tensor(),
                    pbc=torch.tensor(frame.pbc, dtype=torch.bool),
                    sample_id=frame.sample_id,
                )
        except Exception as error:
            raise _error(
                getattr(error, "reason_code", "REFERENCE_BUILD_FAILED"),
                f"existing reference builder rejected automatic configuration: {error}",
                stage="automatic_reference.build", source_path=str(item["path"]),
                template_id=template_id, original_error=error,
            ) from error
        specification = specification_factory(
            builder=builder,
            phase_specification=phase,
            evaluation_policy=None,
            species_alignment_weights=tuple(
                tuple(float(value) for value in row)
                for row in torch.eye(len(species), dtype=torch.float64).tolist()
            ),
            poscar_sha256=item["raw_sha"],
        )

        def vacancy_payload(entries: Sequence[Any]) -> dict[str, Any]:
            records = []
            for frame, _, _ in entries:
                composition = _composition(frame.atomic_numbers, species)
                deficit = tuple(
                    reference - actual
                    for reference, actual in zip(item["reference_composition"], composition)
                )
                records.append((frame.sample_id, len(item["numbers"]) - frame.num_atoms, composition, deficit))
            ks = [record[1] for record in records]
            return {
                "sample_ids": [record[0] for record in records],
                "observed_K_values": sorted(set(ks)),
                "K_min": min(ks) if ks else None,
                "K_max": max(ks) if ks else None,
                "frame_compositions": [
                    {"sample_id": record[0], "composition": list(record[2]), "species_deficit": list(record[3])}
                    for record in records
                ],
            }

        train_vacancy = vacancy_payload(assigned_train)
        validation_vacancy = vacancy_payload(assigned_validation)
        certificate = {
            "schema_version": "refsite_automatic_reference_certificate_v1",
            "scope": "dataset_bounded",
            "approval_status": "provisional",
            "template_id": template_id,
            "poscar_content_sha256": item["raw_sha"],
            "poscar_semantic_sha256": item["semantic_sha"],
            "specification_sha256": specification.content_fingerprint,
            "artifact_sha256": artifact.structural_fingerprint,
            "template_sha256": built.template.fingerprint,
            "phase_specification_sha256": _fingerprint(phase.to_dict()),
            "num_reference_sites": len(item["numbers"]),
            "reference_composition": list(item["reference_composition"]),
            "global_species_ordering": list(species),
            "global_site_type_ordering": list(species),
            "site_typing": "by_species",
            "same_species_orbit_count": {
                str(species_value): _typed_orbit_count(
                    item["numbers"], item["stabilizer"], species_value
                )
                for species_value in species
            },
            "stabilizer_size": int(item["stabilizer"].translations.shape[0]),
            "stabilizer_permutation_sha256": hashlib.sha256(
                item["stabilizer"].permutations.contiguous().numpy().tobytes()
            ).hexdigest(),
            "graph": {
                "active_edge_count": int(geometry.active_mask.sum()),
                "candidate_edge_count": preliminary.num_edges,
                "average_neighbors": float(active_degrees.to(torch.float64).mean()),
            },
            "radii": {
                "r_ot": recipe.radii.r_ot,
                "r_mp": recipe.radii.r_mp,
                "derived": derived.to_dict(),
                "mp_limiting_margin": limiting_margin_mp,
                "ot_limiting_margin": limiting_margin_ot,
            },
            "strain": {
                "observed_train": observed_train,
                "observed_validation": observed_validation,
                "observed_all": observed_all,
                "resolved_maximum_strain": maximum_strain,
                "source": "auto" if item["source"].maximum_strain == "auto" else "explicit",
                "policy_version": AUTO_STRAIN_POLICY_VERSION,
                "margin": AUTO_STRAIN_MARGIN,
                "rounding_grid": AUTO_STRAIN_GRID,
                "minimum_value": AUTO_STRAIN_MINIMUM,
                "auto_safety_ceiling": AUTO_STRAIN_CEILING,
            },
            "vacancies": {"train": train_vacancy, "validation": validation_vacancy},
            "phase": phase_certificate,
            "evaluation_policy": None,
            "future_configuration_guarantee": False,
        }
        certificate["certificate_sha256"] = _fingerprint(certificate)
        results.append(
            AutomaticReferenceResult(
                source_index=item["source_index"],
                original_poscar=item["source"].poscar,
                resolved_poscar=str(item["path"]),
                specification=specification,
                certificate=certificate,
            )
        )
    results.sort(key=lambda result: result.template_id)
    semantic = {
        "convention_version": AUTO_REFERENCE_CONVENTION_VERSION,
        "scope": "dataset_bounded",
        "species_vocabulary": list(species),
        "references": [result.semantic_dict() for result in results],
        "train_assignments": [(entry[0].sample_id, entry[1]) for entry in train_selected],
        "validation_assignments": [(entry[0].sample_id, entry[1]) for entry in validation_selected],
    }
    return AutomaticReferencePreparation(
        results=tuple(results),
        train_assignments=tuple((entry[0].sample_id, entry[1]) for entry in train_selected),
        validation_assignments=tuple((entry[0].sample_id, entry[1]) for entry in validation_selected),
        species_vocabulary=species,
        content_fingerprint=_fingerprint(semantic),
    )


__all__ = [
    "AUTO_PHASE_SEARCH_VERSION",
    "AUTO_REFERENCE_CONVENTION_VERSION",
    "AUTO_STRAIN_CEILING",
    "AUTO_STRAIN_GRID",
    "AUTO_STRAIN_MARGIN",
    "AUTO_STRAIN_MINIMUM",
    "AUTO_STRAIN_POLICY_VERSION",
    "AutomaticReferenceError",
    "AutomaticReferencePreparation",
    "AutomaticReferenceResult",
    "prepare_automatic_references",
    "resolve_auto_maximum_strain",
]
