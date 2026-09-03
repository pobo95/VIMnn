"""Deterministic ASE/POSCAR reference-template construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from refsite_mlip.graph import (
    build_reference_graph_topology,
    update_reference_edge_geometry,
)
from refsite_mlip.phase.stabilizer import (
    find_typed_stabilizer,
    torus_difference,
    validate_alias_matches_stabilizer,
)
from refsite_mlip.phase.types import TypedStabilizer

from .template_domain import StrictTemplateDomain
from .templates import ReferenceTemplate


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _cpu_clone(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous().clone()


def _integer_modes(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("phase modes must be supplied as a torch.Tensor")
    if value.dtype == torch.bool:
        raise TypeError("phase modes must be integers; bool is not accepted")
    if value.is_complex():
        raise TypeError("phase modes must be real integers")
    if value.is_floating_point():
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError("phase modes contain NaN or Inf")
        if not bool(torch.all(value == torch.round(value))):
            raise ValueError("phase modes contain fractional reciprocal indices")
    else:
        try:
            torch.iinfo(value.dtype)
        except TypeError as error:
            raise TypeError("phase modes must use an integer tensor dtype") from error
    long_limits = torch.iinfo(torch.long)
    if any(
        item < long_limits.min or item > long_limits.max
        for item in value.detach().cpu().reshape(-1).tolist()
    ):
        raise ValueError("phase modes cannot be represented exactly as torch.long")
    converted = _cpu_clone(value, dtype=torch.long)
    if value.is_floating_point() and not bool(
        torch.all(converted.to(dtype=value.dtype) == value.detach().cpu())
    ):
        raise ValueError("phase modes cannot be represented exactly as torch.long")
    return converted


@dataclass(frozen=True)
class PhaseSpecification:
    """Explicit reciprocal modes and every associated alignment weight."""

    modes: torch.Tensor
    mode_weights: torch.Tensor
    site_type_alignment_weights: torch.Tensor
    channel_weights: torch.Tensor
    approval_status: str
    convention_version: str = "explicit_phase_specification_v1"

    def __post_init__(self) -> None:
        modes = _integer_modes(self.modes)
        mode_weights = _cpu_clone(self.mode_weights, dtype=torch.float64)
        site_weights = _cpu_clone(
            self.site_type_alignment_weights, dtype=torch.float64
        )
        channel_weights = _cpu_clone(self.channel_weights, dtype=torch.float64)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "mode_weights", mode_weights)
        object.__setattr__(self, "site_type_alignment_weights", site_weights)
        object.__setattr__(self, "channel_weights", channel_weights)

        if modes.ndim != 2 or modes.shape[1] != 3 or modes.shape[0] < 3:
            raise ValueError("phase modes must be long [G,3] with G>=3")
        if mode_weights.shape != (modes.shape[0],):
            raise ValueError("phase mode_weights must have shape [G]")
        if site_weights.ndim != 2 or site_weights.shape[0] == 0:
            raise ValueError(
                "site_type_alignment_weights must have shape [A,C]"
            )
        channel_count = site_weights.shape[1]
        if channel_count == 0 or channel_weights.shape != (channel_count,):
            raise ValueError("phase channel alignment shape mismatch")
        floating = (mode_weights, site_weights, channel_weights)
        if any(not bool(torch.all(torch.isfinite(value))) for value in floating):
            raise ValueError("phase specification contains NaN or Inf")
        if bool(torch.any(mode_weights <= 0)) or bool(
            torch.any(channel_weights <= 0)
        ):
            raise ValueError("phase mode/channel weights must be positive")
        if bool(torch.any(torch.sum(torch.abs(site_weights), dim=1) == 0)):
            raise ValueError("each global site type needs a phase alignment channel")
        if int(torch.linalg.matrix_rank(modes[:3].to(torch.float64))) != 3:
            raise ValueError("primary phase modes must have rank 3")
        if self.approval_status not in {"provisional", "production_approved"}:
            raise ValueError(
                "approval_status must be provisional or production_approved"
            )
        if not isinstance(self.convention_version, str) or not self.convention_version:
            raise ValueError("phase specification convention_version must be nonempty")

    @property
    def num_channels(self) -> int:
        return int(self.site_type_alignment_weights.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "modes": self.modes.tolist(),
            "mode_weights": self.mode_weights.tolist(),
            "site_type_alignment_weights": self.site_type_alignment_weights.tolist(),
            "channel_weights": self.channel_weights.tolist(),
            "approval_status": self.approval_status,
            "convention_version": self.convention_version,
            "floating_dtype": "float64",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhaseSpecification":
        if not isinstance(payload, Mapping):
            raise TypeError("phase specification payload must be a mapping")
        if payload.get("floating_dtype", "float64") != "float64":
            raise ValueError("canonical phase specification dtype must be float64")
        return cls(
            modes=torch.as_tensor(payload["modes"]),
            mode_weights=torch.tensor(
                payload["mode_weights"], dtype=torch.float64
            ),
            site_type_alignment_weights=torch.tensor(
                payload["site_type_alignment_weights"], dtype=torch.float64
            ),
            channel_weights=torch.tensor(
                payload["channel_weights"], dtype=torch.float64
            ),
            approval_status=payload["approval_status"],
            convention_version=payload.get(
                "convention_version", "explicit_phase_specification_v1"
            ),
        )


@dataclass(frozen=True)
class ReferenceTemplateBuilderConfig:
    """Fully resolved non-phase contract for a POSCAR reference template."""

    template_id: str
    strict_domain: StrictTemplateDomain
    site_type_ids: tuple[int, ...]
    graph_cutoff: float = 3.0
    graph_skin: float = 0.5
    maximum_strain: float = 0.10
    minimum_edge_length: float = 1.0e-8
    avg_num_neighbors: float = 6.0
    expected_active_degree: int = 6
    expected_candidate_degree: int = 18
    expected_stabilizer_size: int = 1
    canonical_tolerance: float = 1.0e-10
    metric_tolerance: float = 1.0e-8
    template_convention_version: str = "reference_template_v1"
    builder_convention_version: str = "poscar_reference_builder_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be nonempty")
        if not isinstance(self.strict_domain, StrictTemplateDomain):
            raise TypeError("strict_domain must be a StrictTemplateDomain")
        ids = tuple(int(value) for value in self.site_type_ids)
        if (
            len(ids) != len(self.strict_domain.species_vocabulary)
            or len(set(ids)) != len(ids)
            or ids != tuple(range(len(ids)))
        ):
            raise ValueError(
                "site_type_ids must be the ordered global IDs 0..A-1"
            )
        object.__setattr__(self, "site_type_ids", ids)
        object.__setattr__(
            self, "graph_cutoff", _finite_positive(self.graph_cutoff, name="graph_cutoff")
        )
        object.__setattr__(
            self, "graph_skin", _finite_positive(self.graph_skin, name="graph_skin")
        )
        maximum_strain = float(self.maximum_strain)
        if not math.isfinite(maximum_strain) or not 0.0 <= maximum_strain < 1.0:
            raise ValueError("maximum_strain must be finite and in [0,1)")
        object.__setattr__(self, "maximum_strain", maximum_strain)
        object.__setattr__(
            self,
            "minimum_edge_length",
            _finite_positive(self.minimum_edge_length, name="minimum_edge_length"),
        )
        object.__setattr__(
            self,
            "avg_num_neighbors",
            _finite_positive(self.avg_num_neighbors, name="avg_num_neighbors"),
        )
        for name in (
            "expected_active_degree",
            "expected_candidate_degree",
            "expected_stabilizer_size",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        if self.expected_candidate_degree < self.expected_active_degree:
            raise ValueError("candidate degree cannot be below active degree")
        object.__setattr__(
            self,
            "canonical_tolerance",
            _finite_positive(self.canonical_tolerance, name="canonical_tolerance"),
        )
        object.__setattr__(
            self,
            "metric_tolerance",
            _finite_positive(self.metric_tolerance, name="metric_tolerance"),
        )
        if (1.0 - maximum_strain) * (
            self.graph_cutoff + self.graph_skin
        ) < self.graph_cutoff:
            raise ValueError("graph skin cannot certify maximum_strain")
        for name in ("template_convention_version", "builder_convention_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "strict_domain": self.strict_domain.to_dict(),
            "site_type_ids": list(self.site_type_ids),
            "graph_cutoff": self.graph_cutoff,
            "graph_skin": self.graph_skin,
            "maximum_strain": self.maximum_strain,
            "minimum_edge_length": self.minimum_edge_length,
            "avg_num_neighbors": self.avg_num_neighbors,
            "expected_active_degree": self.expected_active_degree,
            "expected_candidate_degree": self.expected_candidate_degree,
            "expected_stabilizer_size": self.expected_stabilizer_size,
            "canonical_tolerance": self.canonical_tolerance,
            "metric_tolerance": self.metric_tolerance,
            "template_convention_version": self.template_convention_version,
            "builder_convention_version": self.builder_convention_version,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ReferenceTemplateBuilderConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("template builder config payload must be a mapping")
        return cls(
            template_id=payload["template_id"],
            strict_domain=StrictTemplateDomain.from_dict(payload["strict_domain"]),
            site_type_ids=tuple(payload["site_type_ids"]),
            graph_cutoff=payload.get("graph_cutoff", 3.0),
            graph_skin=payload.get("graph_skin", 0.5),
            maximum_strain=payload.get("maximum_strain", 0.10),
            minimum_edge_length=payload.get("minimum_edge_length", 1.0e-8),
            avg_num_neighbors=payload.get("avg_num_neighbors", 6.0),
            expected_active_degree=payload.get("expected_active_degree", 6),
            expected_candidate_degree=payload.get("expected_candidate_degree", 18),
            expected_stabilizer_size=payload["expected_stabilizer_size"],
            canonical_tolerance=payload.get("canonical_tolerance", 1.0e-10),
            metric_tolerance=payload.get("metric_tolerance", 1.0e-8),
            template_convention_version=payload.get(
                "template_convention_version", "reference_template_v1"
            ),
            builder_convention_version=payload.get(
                "builder_convention_version", "poscar_reference_builder_v1"
            ),
        )


def nbc_rocksalt_template_builder_config(
    supercell_shape: tuple[int, int, int],
) -> ReferenceTemplateBuilderConfig:
    """Return the approved strict domain for the 222 or 333 NbC family."""

    shape = tuple(int(value) for value in supercell_shape)
    contracts = {
        (2, 2, 2): ("nbc_rocksalt_222_v1", 32, 64, 32),
        (3, 3, 3): ("nbc_rocksalt_333_v1", 108, 216, 108),
    }
    if shape not in contracts:
        raise ValueError("NbC builder supports only (2,2,2) and (3,3,3)")
    template_id, per_species, site_count, stabilizer_size = contracts[shape]
    domain = StrictTemplateDomain(
        reference_site_count=site_count,
        supercell_shape=shape,
        species_vocabulary=(6, 41),
        reference_composition=(per_species, per_species),
        allowed_compositions=(
            (per_species, per_species),
            (per_species - 1, per_species),
        ),
        allowed_num_atoms=(site_count, site_count - 1),
        allowed_vacancy_masses=(0, 1),
    )
    return ReferenceTemplateBuilderConfig(
        template_id=template_id,
        strict_domain=domain,
        site_type_ids=(0, 1),
        expected_stabilizer_size=stabilizer_size,
    )


@dataclass(frozen=True)
class CanonicalReferenceStructure:
    fractional_positions: torch.Tensor
    atomic_numbers: torch.Tensor
    site_types: torch.Tensor
    cell: torch.Tensor
    original_to_canonical: torch.Tensor
    canonical_to_original: torch.Tensor
    pbc: tuple[bool, bool, bool]


@dataclass(frozen=True)
class ReferenceTemplateBuildDiagnostics:
    template_id: str
    num_sites: int
    composition: tuple[int, ...]
    original_to_canonical: tuple[int, ...]
    reference_cell: tuple[tuple[float, float, float], ...]
    active_edge_count: int
    candidate_edge_count: int
    active_degree_min: int
    active_degree_max: int
    candidate_degree_min: int
    candidate_degree_max: int
    minimum_edge_length: float
    stabilizer_size: int
    phase_rank: int
    phase_approval_status: str
    strict_domain: StrictTemplateDomain
    fingerprint: str
    canonicalization_seconds: float
    graph_build_seconds: float
    stabilizer_build_seconds: float
    total_build_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "num_sites": self.num_sites,
            "composition": list(self.composition),
            "original_to_canonical": list(self.original_to_canonical),
            "reference_cell": [list(row) for row in self.reference_cell],
            "active_edge_count": self.active_edge_count,
            "candidate_edge_count": self.candidate_edge_count,
            "active_degree_min": self.active_degree_min,
            "active_degree_max": self.active_degree_max,
            "candidate_degree_min": self.candidate_degree_min,
            "candidate_degree_max": self.candidate_degree_max,
            "minimum_edge_length": self.minimum_edge_length,
            "stabilizer_size": self.stabilizer_size,
            "phase_rank": self.phase_rank,
            "phase_approval_status": self.phase_approval_status,
            "strict_domain": self.strict_domain.to_dict(),
            "fingerprint": self.fingerprint,
            "canonicalization_seconds": self.canonicalization_seconds,
            "graph_build_seconds": self.graph_build_seconds,
            "stabilizer_build_seconds": self.stabilizer_build_seconds,
            "total_build_seconds": self.total_build_seconds,
        }


@dataclass(frozen=True)
class ReferenceTemplateBuildResult:
    template: ReferenceTemplate
    config: ReferenceTemplateBuilderConfig
    phase_specification: PhaseSpecification
    diagnostics: ReferenceTemplateBuildDiagnostics


def _validate_supercell_metric(
    cell: torch.Tensor,
    shape: tuple[int, int, int],
    tolerance: float,
) -> None:
    lengths = torch.linalg.vector_norm(cell, dim=1)
    primitive = lengths / cell.new_tensor(shape)
    scale = float(torch.mean(primitive))
    if scale <= 0.0 or bool(
        torch.any(torch.abs(primitive - scale) > tolerance * max(scale, 1.0))
    ):
        raise ValueError(
            "reference cell metric is incompatible with declared supercell_shape"
        )
    directions = cell / lengths[:, None]
    gram = directions @ directions.T
    if bool(
        torch.any(
            torch.abs(gram - torch.eye(3, dtype=cell.dtype)) > tolerance
        )
    ):
        raise ValueError("NbC reference cell must be an orthogonal supercell")


def canonicalize_reference_atoms(
    atoms: Any,
    config: ReferenceTemplateBuilderConfig,
) -> CanonicalReferenceStructure:
    """Canonicalize an ASE Atoms object without mutating caller-owned state."""

    try:
        from ase import Atoms
    except ImportError as error:  # pragma: no cover - environment contract
        raise ImportError("ASE is required to build a POSCAR template") from error
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ase.Atoms instance")
    if not isinstance(config, ReferenceTemplateBuilderConfig):
        raise TypeError("config must be a ReferenceTemplateBuilderConfig")

    pbc = tuple(bool(value) for value in atoms.pbc.tolist())
    if pbc != (True, True, True):
        raise ValueError("reference POSCAR must use full PBC")
    numbers = torch.tensor(atoms.get_atomic_numbers().copy(), dtype=torch.long)
    if int(numbers.numel()) != config.strict_domain.reference_site_count:
        raise ValueError(
            "POSCAR site count is incompatible with strict template domain"
        )
    composition = config.strict_domain.composition_for(numbers)
    if composition != config.strict_domain.reference_composition:
        raise ValueError(
            "POSCAR composition differs from strict reference composition"
        )
    species_to_type = dict(
        zip(config.strict_domain.species_vocabulary, config.site_type_ids)
    )
    site_types = torch.tensor(
        [species_to_type[int(value)] for value in numbers.tolist()],
        dtype=torch.long,
    )

    cell = torch.tensor(atoms.cell.array.copy(), dtype=torch.float64)
    if cell.shape != (3, 3) or not bool(torch.all(torch.isfinite(cell))):
        raise ValueError("reference cell must be a finite [3,3] tensor")
    if bool(torch.linalg.svdvals(cell)[-1] <= torch.finfo(cell.dtype).eps):
        raise ValueError("reference cell is singular")
    _validate_supercell_metric(
        cell,
        config.strict_domain.supercell_shape,
        config.metric_tolerance,
    )

    fractional = torch.tensor(
        atoms.get_scaled_positions(wrap=False).copy(), dtype=torch.float64
    )
    if fractional.shape != (numbers.numel(), 3) or not bool(
        torch.all(torch.isfinite(fractional))
    ):
        raise ValueError("fractional coordinates must be finite [M,3]")
    fractional = fractional - torch.floor(fractional)
    tolerance = config.canonical_tolerance
    boundary = (torch.abs(fractional) <= tolerance) | (
        torch.abs(fractional - 1.0) <= tolerance
    )
    fractional = torch.where(boundary, torch.zeros_like(fractional), fractional)
    # Integer wrapping may change the last binary ulp (for example x versus
    # (x + 1) - 1).  Normalize only that machine-scale representation; the
    # physical 1e-10 sorting tolerance is deliberately not used to round the
    # stored coordinates.
    storage_quantum = 64.0 * torch.finfo(fractional.dtype).eps
    fractional = torch.round(fractional / storage_quantum) * storage_quantum
    fractional = torch.where(
        (torch.abs(fractional) <= tolerance)
        | (torch.abs(fractional - 1.0) <= tolerance),
        torch.zeros_like(fractional),
        fractional,
    )

    if fractional.shape[0] > 1:
        difference = torus_difference(
            fractional[:, None, :], fractional[None, :, :]
        )
        distance = torch.linalg.vector_norm(difference, dim=-1)
        duplicate = torch.triu(distance <= tolerance, diagonal=1)
        if bool(torch.any(duplicate)):
            pair = torch.nonzero(duplicate, as_tuple=False)[0].tolist()
            raise ValueError(f"duplicate canonical reference site at indices {pair}")

    quantized = torch.round(fractional / tolerance).to(torch.int64)
    keys = [
        (
            int(site_types[index]),
            int(quantized[index, 0]),
            int(quantized[index, 1]),
            int(quantized[index, 2]),
        )
        for index in range(numbers.numel())
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "canonical sorting key is ambiguous at configured tolerance"
        )
    canonical_to_original = torch.tensor(
        sorted(range(numbers.numel()), key=keys.__getitem__), dtype=torch.long
    )
    original_to_canonical = torch.empty_like(canonical_to_original)
    original_to_canonical[canonical_to_original] = torch.arange(
        numbers.numel(), dtype=torch.long
    )
    return CanonicalReferenceStructure(
        fractional_positions=fractional[canonical_to_original].contiguous(),
        atomic_numbers=numbers[canonical_to_original].contiguous(),
        site_types=site_types[canonical_to_original].contiguous(),
        cell=cell.contiguous(),
        original_to_canonical=original_to_canonical,
        canonical_to_original=canonical_to_original,
        pbc=pbc,
    )


def _translation_key(translation: torch.Tensor, tolerance: float) -> tuple[int, ...]:
    canonical = translation - torch.floor(translation)
    canonical = torch.where(
        (torch.abs(canonical) <= tolerance)
        | (torch.abs(canonical - 1.0) <= tolerance),
        torch.zeros_like(canonical),
        canonical,
    )
    return tuple(int(value) for value in torch.round(canonical / tolerance).tolist())


def _validate_stabilizer_group(
    stabilizer: TypedStabilizer,
    num_sites: int,
    tolerance: float,
) -> None:
    translations = stabilizer.translations
    permutations = stabilizer.permutations
    size = int(translations.shape[0])
    if translations.shape != (size, 3) or permutations.shape != (size, num_sites):
        raise ValueError("typed stabilizer shape mismatch")
    expected = torch.arange(num_sites, dtype=torch.long)
    if any(not torch.equal(torch.sort(row).values, expected) for row in permutations):
        raise ValueError("typed stabilizer contains a non-bijective permutation")
    keys = [_translation_key(row, tolerance) for row in translations]
    if len(set(keys)) != size:
        raise ValueError("typed stabilizer contains duplicate translations")
    if len({tuple(int(value) for value in row.tolist()) for row in permutations}) != size:
        raise ValueError("typed stabilizer contains duplicate permutations")
    by_key = {key: index for index, key in enumerate(keys)}
    zero = _translation_key(torch.zeros(3, dtype=translations.dtype), tolerance)
    if zero not in by_key or not torch.equal(permutations[by_key[zero]], expected):
        raise ValueError("typed stabilizer lacks a valid identity")
    for left in range(size):
        inverse_key = _translation_key(-translations[left], tolerance)
        if inverse_key not in by_key:
            raise ValueError("typed stabilizer lacks an inverse")
        for right in range(size):
            product_key = _translation_key(
                translations[left] + translations[right], tolerance
            )
            if product_key not in by_key:
                raise ValueError("typed stabilizer is not closed")
            product = permutations[right][permutations[left]]
            if not torch.equal(product, permutations[by_key[product_key]]):
                raise ValueError(
                    "typed stabilizer translation/permutation composition disagrees"
                )


def build_reference_template_from_atoms(
    atoms: Any,
    *,
    config: ReferenceTemplateBuilderConfig,
    phase_specification: PhaseSpecification | None,
) -> ReferenceTemplateBuildResult:
    """Build one immutable CPU template from an ASE Atoms reference."""

    if phase_specification is None:
        raise ValueError(
            "an explicit PhaseSpecification is required; no production phase weights are inferred"
        )
    if not isinstance(phase_specification, PhaseSpecification):
        raise TypeError("phase_specification must be a PhaseSpecification")
    if phase_specification.site_type_alignment_weights.shape[0] != len(
        config.site_type_ids
    ):
        raise ValueError(
            "phase site-type alignment rows differ from global site-type ordering"
        )

    started = time.perf_counter()
    canonical_started = time.perf_counter()
    canonical = canonicalize_reference_atoms(atoms, config)
    canonical_seconds = time.perf_counter() - canonical_started

    graph_started = time.perf_counter()
    topology = build_reference_graph_topology(
        canonical.fractional_positions,
        canonical.site_types,
        canonical.cell,
        cutoff=config.graph_cutoff,
        skin=config.graph_skin,
        maximum_strain=config.maximum_strain,
        minimum_edge_length=config.minimum_edge_length,
        pbc=canonical.pbc,
    )
    graph_seconds = time.perf_counter() - graph_started
    geometry = update_reference_edge_geometry(
        topology, topology.reference_cell, edge_length_scale=1.0
    )
    target = topology.edge_index[1]
    active_degree = torch.bincount(
        target[geometry.active_mask], minlength=topology.num_sites
    )
    candidate_degree = torch.bincount(target, minlength=topology.num_sites)
    if bool(
        torch.any(active_degree != config.expected_active_degree)
    ):
        raise ValueError(
            "reference graph active degree differs from the declared rocksalt domain"
        )
    if bool(
        torch.any(candidate_degree != config.expected_candidate_degree)
    ):
        raise ValueError(
            "reference graph candidate degree differs from the declared rocksalt domain"
        )

    stabilizer_started = time.perf_counter()
    stabilizer = find_typed_stabilizer(
        canonical.fractional_positions,
        canonical.site_types,
        tolerance=config.canonical_tolerance,
    )
    _validate_stabilizer_group(
        stabilizer, topology.num_sites, config.canonical_tolerance
    )
    stabilizer_seconds = time.perf_counter() - stabilizer_started
    if int(stabilizer.translations.shape[0]) != config.expected_stabilizer_size:
        raise ValueError(
            "typed stabilizer size differs from the declared supercell domain"
        )

    validate_alias_matches_stabilizer(
        phase_specification.modes[:3],
        stabilizer,
        tolerance=config.canonical_tolerance,
    )
    validate_alias_matches_stabilizer(
        phase_specification.modes,
        stabilizer,
        tolerance=config.canonical_tolerance,
    )
    site_alignment = phase_specification.site_type_alignment_weights[
        canonical.site_types
    ]
    template = ReferenceTemplate.snapshot(
        config.template_id,
        topology,
        phase_specification.modes,
        phase_specification.mode_weights,
        site_alignment,
        phase_specification.channel_weights,
        stabilizer,
        config.strict_domain.species_vocabulary,
        config.template_convention_version,
        config.strict_domain,
    )
    total_seconds = time.perf_counter() - started
    composition = tuple(
        int(torch.sum(canonical.atomic_numbers == species))
        for species in config.strict_domain.species_vocabulary
    )
    lengths = torch.sqrt(geometry.squared_lengths)
    diagnostics = ReferenceTemplateBuildDiagnostics(
        template_id=config.template_id,
        num_sites=topology.num_sites,
        composition=composition,
        original_to_canonical=tuple(
            int(value) for value in canonical.original_to_canonical.tolist()
        ),
        reference_cell=tuple(
            tuple(float(value) for value in row)
            for row in canonical.cell.tolist()
        ),
        active_edge_count=int(geometry.active_mask.sum()),
        candidate_edge_count=topology.num_edges,
        active_degree_min=int(active_degree.min()),
        active_degree_max=int(active_degree.max()),
        candidate_degree_min=int(candidate_degree.min()),
        candidate_degree_max=int(candidate_degree.max()),
        minimum_edge_length=float(lengths.min()),
        stabilizer_size=int(stabilizer.translations.shape[0]),
        phase_rank=int(
            torch.linalg.matrix_rank(
                phase_specification.modes[:3].to(torch.float64)
            )
        ),
        phase_approval_status=phase_specification.approval_status,
        strict_domain=config.strict_domain,
        fingerprint=template.fingerprint,
        canonicalization_seconds=canonical_seconds,
        graph_build_seconds=graph_seconds,
        stabilizer_build_seconds=stabilizer_seconds,
        total_build_seconds=total_seconds,
    )
    return ReferenceTemplateBuildResult(
        template=template,
        config=ReferenceTemplateBuilderConfig.from_dict(config.to_dict()),
        phase_specification=PhaseSpecification.from_dict(
            phase_specification.to_dict()
        ),
        diagnostics=diagnostics,
    )


def build_reference_template_from_poscar(
    path: str | Path,
    *,
    config: ReferenceTemplateBuilderConfig,
    phase_specification: PhaseSpecification | None,
) -> ReferenceTemplateBuildResult:
    """Read one POSCAR and delegate to the path-independent Atoms builder."""

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("POSCAR path must be a regular file")
    try:
        from ase.io import read
    except ImportError as error:  # pragma: no cover - environment contract
        raise ImportError("ASE is required to read POSCAR templates") from error
    atoms = read(str(source), index=0)
    return build_reference_template_from_atoms(
        atoms,
        config=config,
        phase_specification=phase_specification,
    )
