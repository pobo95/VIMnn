"""Safe, phase-independent serialization of structural reference metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping

import torch

from refsite_mlip._atomic import commit_temporary_file

from refsite_mlip.graph import ReferenceGraphTopology
from refsite_mlip.phase.stabilizer import (
    torus_difference,
    validate_alias_matches_stabilizer,
)
from refsite_mlip.phase.types import TypedStabilizer

from .reference_builder import (
    PhaseSpecification,
    ReferenceTemplateBuildResult,
    _validate_stabilizer_group,
)
from .template_domain import StrictTemplateDomain
from .templates import ReferenceTemplate


REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION = "reference_structure_artifact_v1"
REFERENCE_STRUCTURE_ARTIFACT_SCOPE = "structural_reference"

_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "artifact_scope", "payload", "structural_fingerprint"}
)
_PAYLOAD_KEYS = frozenset(
    {
        "template_id",
        "convention_version",
        "floating_dtype",
        "species_vocabulary",
        "avg_num_neighbors",
        "strict_domain",
        "topology",
        "stabilizer",
        "structural_metadata",
    }
)
_TOPOLOGY_KEYS = frozenset(
    {
        "reference_fractional",
        "site_types",
        "edge_index",
        "periodic_shifts",
        "reference_cell",
        "pbc",
        "mp_cutoff",
        "mp_skin",
        "maximum_strain",
        "minimum_edge_length",
    }
)
_STABILIZER_KEYS = frozenset({"translations", "permutations"})
_DOMAIN_KEYS = frozenset(
    {
        "reference_site_count",
        "supercell_shape",
        "species_vocabulary",
        "reference_composition",
        "allowed_compositions",
        "allowed_num_atoms",
        "allowed_vacancy_masses",
        "convention_version",
    }
)
_METADATA_KEYS = frozenset(
    {
        "num_sites",
        "num_edges",
        "site_type_count",
        "site_type_composition",
        "active_edge_count",
        "candidate_edge_count",
        "active_degree_min",
        "active_degree_max",
        "candidate_degree_min",
        "candidate_degree_max",
        "stabilizer_size",
        "strict_domain_present",
    }
)
_FLOAT_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class ReferenceStructureArtifactError(ValueError):
    """Structured validation failure for a structural reference artifact."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        artifact_path: str | None = None,
        schema: str | None = None,
        template_id: str | None = None,
        validation_stage: str | None = None,
        expected_fingerprint: str | None = None,
        actual_fingerprint: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.artifact_path = artifact_path
        self.schema = schema
        self.template_id = template_id
        self.validation_stage = validation_stage
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint
        context = (
            f" path={artifact_path!r} schema={schema!r} template_id={template_id!r}"
            f" stage={validation_stage!r} expected_fingerprint={expected_fingerprint!r}"
            f" actual_fingerprint={actual_fingerprint!r}"
        )
        super().__init__(f"[{reason_code}]{context} {message}")


def _error(
    reason_code: str,
    message: str,
    *,
    artifact_path: str | None = None,
    schema: str | None = None,
    template_id: str | None = None,
    validation_stage: str | None = None,
    expected_fingerprint: str | None = None,
    actual_fingerprint: str | None = None,
) -> ReferenceStructureArtifactError:
    return ReferenceStructureArtifactError(
        reason_code,
        message,
        artifact_path=artifact_path,
        schema=schema,
        template_id=template_id,
        validation_stage=validation_stage,
        expected_fingerprint=expected_fingerprint,
        actual_fingerprint=actual_fingerprint,
    )


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("artifact tensor field must be a torch.Tensor")
    return tensor.detach().to(device="cpu").contiguous().clone()


def _dtype_name(dtype: torch.dtype) -> str:
    for name, value in _FLOAT_DTYPES.items():
        if dtype == value:
            return name
    raise ValueError("artifact floating dtype must be float32 or float64")


def _finite_real(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _hash_text(digest: Any, field: str, value: Any) -> None:
    field_bytes = field.encode("utf-8")
    value_bytes = str(value).encode("utf-8")
    digest.update(struct.pack("<Q", len(field_bytes)))
    digest.update(field_bytes)
    digest.update(struct.pack("<Q", len(value_bytes)))
    digest.update(value_bytes)


def _hash_tensor(digest: Any, field: str, tensor: torch.Tensor) -> None:
    value = _cpu_clone(tensor)
    _hash_text(digest, f"{field}.dtype", value.dtype)
    _hash_text(digest, f"{field}.shape", tuple(value.shape))
    field_bytes = field.encode("utf-8")
    digest.update(struct.pack("<Q", len(field_bytes)))
    digest.update(field_bytes)
    digest.update(value.numpy().tobytes())


def _hash_plain(digest: Any, field: str, value: Any) -> None:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    _hash_text(digest, field, encoded)


def _require_exact_keys(
    value: Any,
    expected: frozenset[str],
    *,
    name: str,
    path: str | None,
    schema: str | None,
    template_id: str | None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(
            "INVALID_PAYLOAD",
            f"{name} must be a mapping",
            artifact_path=path,
            schema=schema,
            template_id=template_id,
            validation_stage="payload_schema",
        )
    keys = set(value)
    if keys != set(expected):
        missing = sorted(repr(value) for value in set(expected) - keys)
        unknown = sorted(repr(value) for value in keys - set(expected))
        raise _error(
            "INVALID_PAYLOAD_KEYS",
            f"{name} key mismatch missing={missing} unknown={unknown}",
            artifact_path=path,
            schema=schema,
            template_id=template_id,
            validation_stage="payload_schema",
        )
    return value


def _edge_geometry(
    reference_fractional: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    reference_cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source, target = edge_index
    displacement = (
        reference_fractional[source]
        - reference_fractional[target]
        + shifts.to(dtype=reference_fractional.dtype)
    ) @ reference_cell
    return displacement, torch.linalg.vector_norm(displacement, dim=-1)


def _derive_metadata(
    reference_fractional: torch.Tensor,
    site_types: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    reference_cell: torch.Tensor,
    cutoff: float,
    stabilizer_size: int,
    strict_domain: StrictTemplateDomain | None,
    species_count: int,
) -> dict[str, Any]:
    num_sites = int(reference_fractional.shape[0])
    num_edges = int(edge_index.shape[1])
    _, lengths = _edge_geometry(
        reference_fractional, edge_index, shifts, reference_cell
    )
    active = lengths <= cutoff + 1.0e-12
    target = edge_index[1]
    active_degree = torch.bincount(target[active], minlength=num_sites)
    candidate_degree = torch.bincount(target, minlength=num_sites)
    composition = [
        int(torch.sum(site_types == site_type))
        for site_type in range(species_count)
    ]
    return {
        "num_sites": num_sites,
        "num_edges": num_edges,
        "site_type_count": species_count,
        "site_type_composition": composition,
        "active_edge_count": int(active.sum()),
        "candidate_edge_count": num_edges,
        "active_degree_min": int(active_degree.min()) if num_sites else 0,
        "active_degree_max": int(active_degree.max()) if num_sites else 0,
        "candidate_degree_min": int(candidate_degree.min()) if num_sites else 0,
        "candidate_degree_max": int(candidate_degree.max()) if num_sites else 0,
        "stabilizer_size": int(stabilizer_size),
        "strict_domain_present": strict_domain is not None,
    }


def _freeze_metadata(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    frozen: list[tuple[str, Any]] = []
    for key in sorted(value):
        item = value[key]
        if isinstance(item, list):
            item = tuple(item)
        frozen.append((str(key), item))
    return tuple(frozen)


def _metadata_dict(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value:
        result[key] = list(item) if isinstance(item, tuple) else item
    return result


@dataclass(frozen=True)
class ReferenceStructureArtifactDiagnostics:
    schema_version: str
    artifact_scope: str
    template_id: str
    structural_fingerprint: str
    floating_dtype: str
    num_sites: int
    num_edges: int
    active_edge_count: int
    candidate_edge_count: int
    active_degree_min: int
    active_degree_max: int
    candidate_degree_min: int
    candidate_degree_max: int
    stabilizer_size: int
    strict_domain_present: bool
    avg_num_neighbors: float
    mp_cutoff: float
    mp_skin: float
    maximum_strain: float
    minimum_edge_length: float

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ReferenceStructureArtifact:
    """Immutable CPU ownership of phase-independent reference structure data."""

    schema_version: str
    artifact_scope: str
    template_id: str
    convention_version: str
    floating_dtype: str
    reference_fractional: torch.Tensor
    site_types: torch.Tensor
    species_vocabulary: tuple[int, ...]
    reference_cell: torch.Tensor
    pbc: tuple[bool, bool, bool]
    edge_index: torch.Tensor
    periodic_shifts: torch.Tensor
    mp_cutoff: float
    mp_skin: float
    maximum_strain: float
    minimum_edge_length: float
    avg_num_neighbors: float
    strict_domain: StrictTemplateDomain | None
    stabilizer_translations: torch.Tensor
    stabilizer_permutations: torch.Tensor
    structural_metadata: tuple[tuple[str, Any], ...]
    structural_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "reference_fractional",
            "site_types",
            "reference_cell",
            "edge_index",
            "periodic_shifts",
            "stabilizer_translations",
            "stabilizer_permutations",
        ):
            object.__setattr__(self, name, _cpu_clone(getattr(self, name)))
        object.__setattr__(
            self, "species_vocabulary", tuple(int(v) for v in self.species_vocabulary)
        )
        object.__setattr__(self, "pbc", tuple(bool(v) for v in self.pbc))
        if self.strict_domain is not None:
            object.__setattr__(
                self,
                "strict_domain",
                StrictTemplateDomain.from_dict(self.strict_domain.to_dict()),
            )
        object.__setattr__(
            self,
            "structural_metadata",
            _freeze_metadata(_metadata_dict(tuple(self.structural_metadata))),
        )
        self.validate()

    @property
    def diagnostics(self) -> ReferenceStructureArtifactDiagnostics:
        metadata = _metadata_dict(self.structural_metadata)
        return ReferenceStructureArtifactDiagnostics(
            schema_version=self.schema_version,
            artifact_scope=self.artifact_scope,
            template_id=self.template_id,
            structural_fingerprint=self.structural_fingerprint,
            floating_dtype=self.floating_dtype,
            num_sites=metadata["num_sites"],
            num_edges=metadata["num_edges"],
            active_edge_count=metadata["active_edge_count"],
            candidate_edge_count=metadata["candidate_edge_count"],
            active_degree_min=metadata["active_degree_min"],
            active_degree_max=metadata["active_degree_max"],
            candidate_degree_min=metadata["candidate_degree_min"],
            candidate_degree_max=metadata["candidate_degree_max"],
            stabilizer_size=metadata["stabilizer_size"],
            strict_domain_present=metadata["strict_domain_present"],
            avg_num_neighbors=self.avg_num_neighbors,
            mp_cutoff=self.mp_cutoff,
            mp_skin=self.mp_skin,
            maximum_strain=self.maximum_strain,
            minimum_edge_length=self.minimum_edge_length,
        )

    def _computed_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for field, value in (
            ("schema_version", self.schema_version),
            ("artifact_scope", self.artifact_scope),
            ("template_id", self.template_id),
            ("convention_version", self.convention_version),
            ("floating_dtype", self.floating_dtype),
        ):
            _hash_text(digest, field, value)
        for field, tensor in (
            ("reference_fractional", self.reference_fractional),
            ("site_types", self.site_types),
            ("reference_cell", self.reference_cell),
            ("edge_index", self.edge_index),
            ("periodic_shifts", self.periodic_shifts),
            ("stabilizer_translations", self.stabilizer_translations),
            ("stabilizer_permutations", self.stabilizer_permutations),
        ):
            _hash_tensor(digest, field, tensor)
        _hash_plain(digest, "species_vocabulary", list(self.species_vocabulary))
        _hash_plain(digest, "pbc", list(self.pbc))
        for field, value in (
            ("mp_cutoff", self.mp_cutoff),
            ("mp_skin", self.mp_skin),
            ("maximum_strain", self.maximum_strain),
            ("minimum_edge_length", self.minimum_edge_length),
            ("avg_num_neighbors", self.avg_num_neighbors),
        ):
            _hash_plain(digest, field, value)
        _hash_plain(
            digest,
            "strict_domain",
            None if self.strict_domain is None else self.strict_domain.to_dict(),
        )
        _hash_plain(
            digest, "structural_metadata", _metadata_dict(self.structural_metadata)
        )
        return digest.hexdigest()

    def validate(self, *, artifact_path: str | None = None) -> None:
        actual_fingerprint = self._computed_fingerprint()

        def fail(reason: str, message: str, stage: str) -> None:
            raise _error(
                reason,
                message,
                artifact_path=artifact_path,
                schema=self.schema_version,
                template_id=self.template_id,
                validation_stage=stage,
                expected_fingerprint=self.structural_fingerprint,
                actual_fingerprint=actual_fingerprint,
            )

        if self.schema_version != REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION:
            fail("UNSUPPORTED_SCHEMA", "unsupported artifact schema", "schema")
        if self.artifact_scope != REFERENCE_STRUCTURE_ARTIFACT_SCOPE:
            fail("INVALID_SCOPE", "artifact scope must be structural_reference", "schema")
        if not isinstance(self.template_id, str) or not self.template_id:
            fail("INVALID_TEMPLATE_ID", "template_id must be nonempty", "identity")
        if not isinstance(self.convention_version, str) or not self.convention_version:
            fail("INVALID_CONVENTION", "convention_version must be nonempty", "identity")
        if self.floating_dtype not in _FLOAT_DTYPES:
            fail("INVALID_FLOAT_DTYPE", "unsupported floating dtype", "tensor_contract")
        floating_dtype = _FLOAT_DTYPES[self.floating_dtype]
        floating = (
            self.reference_fractional,
            self.reference_cell,
            self.stabilizer_translations,
        )
        if any(value.dtype != floating_dtype for value in floating):
            fail("DTYPE_MISMATCH", "floating tensor dtype disagrees with payload", "tensor_contract")
        if any(value.device.type != "cpu" for value in (*floating, self.site_types, self.edge_index, self.periodic_shifts, self.stabilizer_permutations)):
            fail("NON_CPU_ARTIFACT", "artifact snapshots must reside on CPU", "ownership")
        if any(value.requires_grad for value in (*floating, self.site_types, self.edge_index, self.periodic_shifts, self.stabilizer_permutations)):
            fail("GRADIENT_OWNERSHIP", "artifact tensors cannot require gradients", "ownership")
        if any(not bool(torch.all(torch.isfinite(value))) for value in floating):
            fail("NONFINITE_STRUCTURE", "artifact contains NaN or Inf", "tensor_contract")

        if self.reference_fractional.ndim != 2 or self.reference_fractional.shape[1] != 3:
            fail("INVALID_SITE_SHAPE", "reference_fractional must be [M,3]", "sites")
        num_sites = int(self.reference_fractional.shape[0])
        if num_sites <= 0 or self.site_types.shape != (num_sites,) or self.site_types.dtype != torch.long:
            fail("INVALID_SITE_TYPES", "site_types must be torch.long [M]", "sites")
        tolerance = max(1.0e-10, 64.0 * torch.finfo(floating_dtype).eps)
        if bool(torch.any(self.reference_fractional < 0.0)) or bool(
            torch.any(self.reference_fractional >= 1.0)
        ):
            fail("NONCANONICAL_SITES", "reference fractional sites must lie in [0,1)", "sites")
        difference = torus_difference(
            self.reference_fractional[:, None, :],
            self.reference_fractional[None, :, :],
        )
        duplicate = torch.triu(
            torch.linalg.vector_norm(difference, dim=-1) <= tolerance, diagonal=1
        )
        if bool(torch.any(duplicate)):
            fail("DUPLICATE_SITE", "reference sites are not unique on the torus", "sites")
        if (
            not self.species_vocabulary
            or len(set(self.species_vocabulary)) != len(self.species_vocabulary)
            or any(value <= 0 for value in self.species_vocabulary)
        ):
            fail("INVALID_SPECIES", "species vocabulary must be unique positive integers", "sites")
        if bool(torch.any(self.site_types < 0)) or bool(
            torch.any(self.site_types >= len(self.species_vocabulary))
        ):
            fail("INVALID_SITE_TYPES", "site type outside global species ordering", "sites")

        if self.reference_cell.shape != (3, 3):
            fail("INVALID_CELL_SHAPE", "reference_cell must be [3,3]", "cell")
        if bool(torch.linalg.svdvals(self.reference_cell)[-1] <= torch.finfo(floating_dtype).eps):
            fail("SINGULAR_CELL", "reference cell is singular", "cell")
        if self.pbc != (True, True, True):
            fail("PBC_REQUIRED", "structural artifacts require full PBC", "cell")

        try:
            cutoff = _finite_real(self.mp_cutoff, name="mp_cutoff", positive=True)
            skin = _finite_real(self.mp_skin, name="mp_skin")
            strain = _finite_real(self.maximum_strain, name="maximum_strain")
            minimum = _finite_real(
                self.minimum_edge_length,
                name="minimum_edge_length",
                positive=True,
            )
            _finite_real(self.avg_num_neighbors, name="avg_num_neighbors", positive=True)
        except (TypeError, ValueError) as error:
            fail("INVALID_GRAPH_CONFIG", str(error), "graph")
        if skin < 0.0 or not 0.0 <= strain < 1.0:
            fail("INVALID_GRAPH_CONFIG", "skin/strain outside valid range", "graph")
        if (1.0 - strain) * (cutoff + skin) < cutoff:
            fail("INVALID_STRAIN_CERTIFICATE", "cutoff and skin do not certify maximum strain", "graph")

        num_edges = int(self.edge_index.shape[1]) if self.edge_index.ndim == 2 else -1
        if self.edge_index.shape != (2, num_edges) or self.edge_index.dtype != torch.long:
            fail("INVALID_EDGE_SHAPE", "edge_index must be torch.long [2,E]", "graph")
        if self.periodic_shifts.shape != (num_edges, 3) or self.periodic_shifts.dtype != torch.long:
            fail("INVALID_SHIFT_SHAPE", "periodic shifts must be torch.long [E,3]", "graph")
        if num_edges <= 0:
            fail("EMPTY_GRAPH", "reference graph must contain candidate edges", "graph")
        if bool(torch.any(self.edge_index < 0)) or bool(torch.any(self.edge_index >= num_sites)):
            fail("EDGE_INDEX_RANGE", "edge index is outside [0,M)", "graph")
        keys = [
            (
                int(self.edge_index[1, index]),
                int(self.edge_index[0, index]),
                *(int(value) for value in self.periodic_shifts[index].tolist()),
            )
            for index in range(num_edges)
        ]
        if len(set(keys)) != num_edges:
            fail("DUPLICATE_EDGE", "graph contains duplicate periodic edges", "graph")
        if keys != sorted(keys):
            fail("NONCANONICAL_EDGE_ORDER", "graph edges are not target/source/shift ordered", "graph")
        edge_set = set(keys)
        for target, source, sx, sy, sz in keys:
            if source == target and sx == sy == sz == 0:
                fail("TRIVIAL_SELF_EDGE", "graph contains a zero-image self edge", "graph")
            if (source, target, -sx, -sy, -sz) not in edge_set:
                fail("MISSING_REVERSE_EDGE", "graph edge lacks reverse/opposite-shift edge", "graph")
        _, lengths = _edge_geometry(
            self.reference_fractional,
            self.edge_index,
            self.periodic_shifts,
            self.reference_cell,
        )
        if not bool(torch.all(torch.isfinite(lengths))):
            fail("NONFINITE_EDGE", "reference edge geometry is nonfinite", "graph")
        if bool(torch.any(lengths <= minimum)):
            fail("EDGE_TOO_SHORT", "reference graph violates minimum edge length", "graph")
        if bool(torch.any(lengths > cutoff + skin + 1.0e-12)):
            fail("EDGE_OUTSIDE_CANDIDATE", "reference edge exceeds cutoff+skin", "graph")
        adjacency = [set() for _ in range(num_sites)]
        for target, source, *_ in keys:
            adjacency[target].add(source)
            adjacency[source].add(target)
        reached = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] - reached:
                reached.add(neighbor)
                frontier.append(neighbor)
        if len(reached) != num_sites:
            fail("DISCONNECTED_GRAPH", "reference graph is disconnected", "graph")

        if self.strict_domain is not None:
            try:
                if self.strict_domain.reference_site_count != num_sites:
                    raise ValueError("strict domain site count differs from graph")
                if self.strict_domain.species_vocabulary != self.species_vocabulary:
                    raise ValueError("strict domain species order differs from artifact")
                self.strict_domain.validate_reference_site_types(self.site_types)
            except (TypeError, ValueError) as error:
                fail("INVALID_DOMAIN", str(error), "domain")

        size = int(self.stabilizer_translations.shape[0]) if self.stabilizer_translations.ndim == 2 else -1
        if self.stabilizer_translations.shape != (size, 3):
            fail("INVALID_STABILIZER_SHAPE", "stabilizer translations must be [S,3]", "stabilizer")
        if self.stabilizer_permutations.shape != (size, num_sites) or self.stabilizer_permutations.dtype != torch.long:
            fail("INVALID_STABILIZER_SHAPE", "stabilizer permutations must be torch.long [S,M]", "stabilizer")
        try:
            stabilizer = TypedStabilizer(
                self.stabilizer_translations, self.stabilizer_permutations
            )
            _validate_stabilizer_group(stabilizer, num_sites, tolerance)
            for translation, permutation in zip(
                stabilizer.translations, stabilizer.permutations
            ):
                mapped = self.reference_fractional[permutation]
                translated = self.reference_fractional + translation
                if bool(
                    torch.any(
                        torch.linalg.vector_norm(
                            torus_difference(translated, mapped), dim=-1
                        )
                        > tolerance
                    )
                ):
                    raise ValueError("stabilizer translation does not map reference sites")
                if not torch.equal(self.site_types, self.site_types[permutation]):
                    raise ValueError("stabilizer permutation changes a global site type")
        except (TypeError, ValueError, IndexError) as error:
            fail("INVALID_STABILIZER", str(error), "stabilizer")

        metadata = _metadata_dict(self.structural_metadata)
        if set(metadata) != set(_METADATA_KEYS):
            fail("INVALID_METADATA_KEYS", "structural metadata key mismatch", "metadata")
        derived = _derive_metadata(
            self.reference_fractional,
            self.site_types,
            self.edge_index,
            self.periodic_shifts,
            self.reference_cell,
            cutoff,
            size,
            self.strict_domain,
            len(self.species_vocabulary),
        )
        if metadata != derived:
            fail("METADATA_MISMATCH", f"stored={metadata} derived={derived}", "metadata")
        if (
            not isinstance(self.structural_fingerprint, str)
            or len(self.structural_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.structural_fingerprint
            )
        ):
            fail("INVALID_FINGERPRINT", "structural fingerprint must be lowercase SHA-256", "fingerprint")
        if actual_fingerprint != self.structural_fingerprint:
            fail("FINGERPRINT_MISMATCH", "structural content fingerprint mismatch", "fingerprint")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "artifact_scope": self.artifact_scope,
            "payload": {
                "template_id": self.template_id,
                "convention_version": self.convention_version,
                "floating_dtype": self.floating_dtype,
                "species_vocabulary": list(self.species_vocabulary),
                "avg_num_neighbors": self.avg_num_neighbors,
                "strict_domain": (
                    None if self.strict_domain is None else self.strict_domain.to_dict()
                ),
                "topology": {
                    "reference_fractional": _cpu_clone(self.reference_fractional),
                    "site_types": _cpu_clone(self.site_types),
                    "edge_index": _cpu_clone(self.edge_index),
                    "periodic_shifts": _cpu_clone(self.periodic_shifts),
                    "reference_cell": _cpu_clone(self.reference_cell),
                    "pbc": list(self.pbc),
                    "mp_cutoff": self.mp_cutoff,
                    "mp_skin": self.mp_skin,
                    "maximum_strain": self.maximum_strain,
                    "minimum_edge_length": self.minimum_edge_length,
                },
                "stabilizer": {
                    "translations": _cpu_clone(self.stabilizer_translations),
                    "permutations": _cpu_clone(self.stabilizer_permutations),
                },
                "structural_metadata": _metadata_dict(self.structural_metadata),
            },
            "structural_fingerprint": self.structural_fingerprint,
        }


def _new_artifact(
    *,
    template: ReferenceTemplate,
    avg_num_neighbors: float,
) -> ReferenceStructureArtifact:
    template.validate()
    topology = template.topology
    floating_dtype = _dtype_name(topology.reference_fractional.dtype)
    if topology.reference_cell.dtype != topology.reference_fractional.dtype:
        raise _error(
            "DTYPE_MISMATCH",
            "reference topology floating tensors must share dtype",
            schema=REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION,
            template_id=template.template_id,
            validation_stage="capture",
        )
    if template.stabilizer.translations.dtype != topology.reference_fractional.dtype:
        raise _error(
            "DTYPE_MISMATCH",
            "stabilizer translations must share structural floating dtype",
            schema=REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION,
            template_id=template.template_id,
            validation_stage="capture",
        )
    metadata = _derive_metadata(
        topology.reference_fractional,
        topology.site_types,
        topology.edge_index,
        topology.shifts,
        topology.reference_cell,
        topology.cutoff,
        int(template.stabilizer.translations.shape[0]),
        template.strict_domain,
        len(template.supported_species),
    )
    components = dict(
        schema_version=REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION,
        artifact_scope=REFERENCE_STRUCTURE_ARTIFACT_SCOPE,
        template_id=template.template_id,
        convention_version=template.convention_version,
        floating_dtype=floating_dtype,
        reference_fractional=_cpu_clone(topology.reference_fractional),
        site_types=_cpu_clone(topology.site_types),
        species_vocabulary=tuple(template.supported_species),
        reference_cell=_cpu_clone(topology.reference_cell),
        pbc=tuple(topology.pbc),
        edge_index=_cpu_clone(topology.edge_index),
        periodic_shifts=_cpu_clone(topology.shifts),
        mp_cutoff=float(topology.cutoff),
        mp_skin=float(topology.skin),
        maximum_strain=float(topology.maximum_strain),
        minimum_edge_length=float(topology.minimum_edge_length),
        avg_num_neighbors=_finite_real(
            avg_num_neighbors, name="avg_num_neighbors", positive=True
        ),
        strict_domain=template.strict_domain,
        stabilizer_translations=_cpu_clone(template.stabilizer.translations),
        stabilizer_permutations=_cpu_clone(template.stabilizer.permutations),
        structural_metadata=_freeze_metadata(metadata),
    )
    provisional = object.__new__(ReferenceStructureArtifact)
    for name, value in components.items():
        object.__setattr__(provisional, name, value)
    fingerprint = ReferenceStructureArtifact._computed_fingerprint(provisional)
    return ReferenceStructureArtifact(
        **components, structural_fingerprint=fingerprint
    )


def capture_reference_structure_artifact(
    source: ReferenceTemplateBuildResult | ReferenceTemplate,
    *,
    avg_num_neighbors: Real | None = None,
) -> ReferenceStructureArtifact:
    """Capture structural data without retaining phase fields or source storage."""

    if isinstance(source, ReferenceTemplateBuildResult):
        if avg_num_neighbors is not None and float(avg_num_neighbors) != source.config.avg_num_neighbors:
            raise ValueError(
                "explicit avg_num_neighbors differs from builder result configuration"
            )
        if source.config.template_id != source.template.template_id:
            raise ValueError("builder result template/config ID mismatch")
        if source.config.strict_domain != source.template.strict_domain:
            raise ValueError("builder result template/config domain mismatch")
        topology = source.template.topology
        for actual, expected, name in (
            (topology.cutoff, source.config.graph_cutoff, "cutoff"),
            (topology.skin, source.config.graph_skin, "skin"),
            (topology.maximum_strain, source.config.maximum_strain, "maximum_strain"),
            (topology.minimum_edge_length, source.config.minimum_edge_length, "minimum_edge_length"),
        ):
            if actual != expected:
                raise ValueError(f"builder result topology/config {name} mismatch")
        template = source.template
        average = source.config.avg_num_neighbors
    elif isinstance(source, ReferenceTemplate):
        if avg_num_neighbors is None:
            raise ValueError(
                "avg_num_neighbors is required when capturing a bare ReferenceTemplate"
            )
        template = source
        average = avg_num_neighbors
    else:
        raise TypeError(
            "source must be a ReferenceTemplateBuildResult or ReferenceTemplate"
        )
    return _new_artifact(template=template, avg_num_neighbors=float(average))


def save_reference_structure_artifact(
    path: str | os.PathLike[str],
    artifact: ReferenceStructureArtifact,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically save a weights-only-safe structural payload."""

    if not isinstance(artifact, ReferenceStructureArtifact):
        raise TypeError("artifact must be a ReferenceStructureArtifact")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    artifact.validate()
    target = Path(path)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError(f"artifact parent directory does not exist: {parent}")
    if target.is_symlink():
        raise ValueError("artifact target must not be a symbolic link")
    if target.exists():
        if not target.is_file():
            raise ValueError("artifact target exists and is not a regular file")
        if not overwrite:
            raise FileExistsError(f"artifact already exists: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(artifact.to_payload(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if target.is_symlink():
            raise ValueError("artifact target became a symbolic link")
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


def _artifact_from_safe_payload(
    raw: Any,
    *,
    artifact_path: str | None,
) -> ReferenceStructureArtifact:
    top = _require_exact_keys(
        raw,
        _TOP_LEVEL_KEYS,
        name="artifact",
        path=artifact_path,
        schema=raw.get("schema_version") if isinstance(raw, Mapping) else None,
        template_id=None,
    )
    schema = top["schema_version"]
    scope = top["artifact_scope"]
    if schema != REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION:
        raise _error(
            "UNSUPPORTED_SCHEMA",
            "artifact schema is not supported",
            artifact_path=artifact_path,
            schema=str(schema),
            validation_stage="payload_schema",
        )
    if scope != REFERENCE_STRUCTURE_ARTIFACT_SCOPE:
        raise _error(
            "INVALID_SCOPE",
            "artifact scope is not structural_reference",
            artifact_path=artifact_path,
            schema=schema,
            validation_stage="payload_schema",
        )
    payload = _require_exact_keys(
        top["payload"],
        _PAYLOAD_KEYS,
        name="payload",
        path=artifact_path,
        schema=schema,
        template_id=None,
    )
    template_id = payload["template_id"]
    if not isinstance(template_id, str) or not template_id:
        raise _error(
            "INVALID_TEMPLATE_ID",
            "payload template_id must be a nonempty string",
            artifact_path=artifact_path,
            schema=schema,
            validation_stage="payload_schema",
        )
    if not isinstance(payload["convention_version"], str) or not payload[
        "convention_version"
    ]:
        raise _error(
            "INVALID_CONVENTION",
            "payload convention_version must be a nonempty string",
            artifact_path=artifact_path,
            schema=schema,
            template_id=template_id,
            validation_stage="payload_schema",
        )
    if payload["floating_dtype"] not in _FLOAT_DTYPES:
        raise _error(
            "INVALID_FLOAT_DTYPE",
            "payload floating_dtype must be float32 or float64",
            artifact_path=artifact_path,
            schema=schema,
            template_id=template_id,
            validation_stage="payload_schema",
        )
    vocabulary = payload["species_vocabulary"]
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or any(type(value) is not int for value in vocabulary)
    ):
        raise _error(
            "INVALID_SPECIES",
            "species_vocabulary must be a nonempty canonical list of integers",
            artifact_path=artifact_path,
            schema=schema,
            template_id=template_id,
            validation_stage="payload_schema",
        )
    fingerprint = top["structural_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise _error(
            "INVALID_FINGERPRINT",
            "structural_fingerprint must be lowercase SHA-256",
            artifact_path=artifact_path,
            schema=schema,
            template_id=template_id,
            validation_stage="payload_schema",
            expected_fingerprint=(fingerprint if isinstance(fingerprint, str) else None),
        )
    topology = _require_exact_keys(
        payload["topology"],
        _TOPOLOGY_KEYS,
        name="topology",
        path=artifact_path,
        schema=schema,
        template_id=template_id,
    )
    stabilizer = _require_exact_keys(
        payload["stabilizer"],
        _STABILIZER_KEYS,
        name="stabilizer",
        path=artifact_path,
        schema=schema,
        template_id=template_id,
    )
    metadata = _require_exact_keys(
        payload["structural_metadata"],
        _METADATA_KEYS,
        name="structural_metadata",
        path=artifact_path,
        schema=schema,
        template_id=template_id,
    )
    integer_metadata = _METADATA_KEYS - {
        "site_type_composition",
        "strict_domain_present",
    }
    if any(type(metadata[key]) is not int for key in integer_metadata) or (
        not isinstance(metadata["site_type_composition"], list)
        or any(
            type(value) is not int
            for value in metadata["site_type_composition"]
        )
    ) or type(metadata["strict_domain_present"]) is not bool:
        raise _error(
            "INVALID_METADATA",
            "structural metadata must use canonical integer/list/bool primitives",
            artifact_path=artifact_path,
            schema=schema,
            template_id=template_id,
            validation_stage="payload_schema",
        )
    if (
        not isinstance(topology["pbc"], list)
        or len(topology["pbc"]) != 3
        or any(type(value) is not bool for value in topology["pbc"])
    ):
        raise _error(
            "PBC_REQUIRED",
            "topology pbc must be a canonical list of three booleans",
            artifact_path=artifact_path,
            schema=schema,
            template_id=template_id,
            validation_stage="payload_schema",
        )
    for name in (
        "reference_fractional",
        "site_types",
        "edge_index",
        "periodic_shifts",
        "reference_cell",
    ):
        if not isinstance(topology[name], torch.Tensor):
            raise _error(
                "INVALID_PAYLOAD",
                f"topology {name} must be a tensor",
                artifact_path=artifact_path,
                schema=schema,
                template_id=template_id,
                validation_stage="payload_schema",
            )
    for name in ("translations", "permutations"):
        if not isinstance(stabilizer[name], torch.Tensor):
            raise _error(
                "INVALID_PAYLOAD",
                f"stabilizer {name} must be a tensor",
                artifact_path=artifact_path,
                schema=schema,
                template_id=template_id,
                validation_stage="payload_schema",
            )
    for name in (
        "avg_num_neighbors",
        "mp_cutoff",
        "mp_skin",
        "maximum_strain",
        "minimum_edge_length",
    ):
        value = (
            payload[name]
            if name == "avg_num_neighbors"
            else topology[name]
        )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise _error(
                "INVALID_GRAPH_CONFIG",
                f"{name} must be a real primitive",
                artifact_path=artifact_path,
                schema=schema,
                template_id=template_id,
                validation_stage="payload_schema",
            )
    domain_payload = payload["strict_domain"]
    domain = None
    if domain_payload is not None:
        domain_mapping = _require_exact_keys(
            domain_payload,
            _DOMAIN_KEYS,
            name="strict_domain",
            path=artifact_path,
            schema=schema,
            template_id=template_id,
        )
        try:
            domain = StrictTemplateDomain.from_dict(domain_mapping)
        except Exception as error:
            raise _error(
                "INVALID_DOMAIN",
                f"{type(error).__name__}: {error}",
                artifact_path=artifact_path,
                schema=schema,
                template_id=template_id,
                validation_stage="payload_domain",
            ) from error
    try:
        artifact = ReferenceStructureArtifact(
            schema_version=schema,
            artifact_scope=scope,
            template_id=template_id,
            convention_version=payload["convention_version"],
            floating_dtype=payload["floating_dtype"],
            reference_fractional=topology["reference_fractional"],
            site_types=topology["site_types"],
            species_vocabulary=tuple(vocabulary),
            reference_cell=topology["reference_cell"],
            pbc=tuple(topology["pbc"]),
            edge_index=topology["edge_index"],
            periodic_shifts=topology["periodic_shifts"],
            mp_cutoff=topology["mp_cutoff"],
            mp_skin=topology["mp_skin"],
            maximum_strain=topology["maximum_strain"],
            minimum_edge_length=topology["minimum_edge_length"],
            avg_num_neighbors=payload["avg_num_neighbors"],
            strict_domain=domain,
            stabilizer_translations=stabilizer["translations"],
            stabilizer_permutations=stabilizer["permutations"],
            structural_metadata=_freeze_metadata(metadata),
            structural_fingerprint=fingerprint,
        )
        artifact.validate(artifact_path=artifact_path)
        return artifact
    except ReferenceStructureArtifactError as error:
        if error.artifact_path == artifact_path:
            raise
        raise _error(
            error.reason_code,
            str(error),
            artifact_path=artifact_path,
            schema=schema,
            template_id=str(template_id),
            validation_stage=error.validation_stage,
            expected_fingerprint=error.expected_fingerprint,
            actual_fingerprint=error.actual_fingerprint,
        ) from error
    except Exception as error:
        raise _error(
            "INVALID_PAYLOAD",
            f"{type(error).__name__}: {error}",
            artifact_path=artifact_path,
            schema=schema,
            template_id=str(template_id),
            validation_stage="payload_construction",
        ) from error


def load_reference_structure_artifact(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> ReferenceStructureArtifact:
    """Load through ``torch.load(..., weights_only=True)`` and fully validate."""

    target = Path(path)
    display_path = str(target)
    if target.is_symlink():
        raise _error(
            "SYMLINK_REJECTED",
            "artifact path must not be a symbolic link",
            artifact_path=display_path,
            validation_stage="path",
        )
    if not target.exists():
        raise FileNotFoundError(f"artifact does not exist: {target}")
    if not target.is_file():
        raise ValueError("artifact path must be a regular file")
    try:
        raw = torch.load(
            target,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as error:
        raise _error(
            "SAFE_LOAD_FAILURE",
            f"{type(error).__name__}: {error}",
            artifact_path=display_path,
            validation_stage="weights_only_load",
        ) from error
    return _artifact_from_safe_payload(raw, artifact_path=display_path)


def assemble_reference_template_from_artifact(
    artifact: ReferenceStructureArtifact,
    *,
    phase_specification: PhaseSpecification | None,
) -> ReferenceTemplate:
    """Combine verified structure with an explicit phase specification."""

    if not isinstance(artifact, ReferenceStructureArtifact):
        raise TypeError("artifact must be a ReferenceStructureArtifact")
    artifact.validate()
    if phase_specification is None:
        raise _error(
            "PHASE_SPECIFICATION_REQUIRED",
            "an explicit PhaseSpecification is required for assembly",
            schema=artifact.schema_version,
            template_id=artifact.template_id,
            validation_stage="phase_assembly",
            expected_fingerprint=artifact.structural_fingerprint,
            actual_fingerprint=artifact.structural_fingerprint,
        )
    if not isinstance(phase_specification, PhaseSpecification):
        raise TypeError("phase_specification must be a PhaseSpecification")
    phase = PhaseSpecification.from_dict(phase_specification.to_dict())
    if phase.site_type_alignment_weights.shape[0] != len(
        artifact.species_vocabulary
    ):
        raise _error(
            "PHASE_SITE_TYPE_MISMATCH",
            "phase site-type alignment rows differ from global species/site-type ordering",
            schema=artifact.schema_version,
            template_id=artifact.template_id,
            validation_stage="phase_assembly",
        )
    stabilizer = TypedStabilizer(
        _cpu_clone(artifact.stabilizer_translations),
        _cpu_clone(artifact.stabilizer_permutations),
    )
    try:
        tolerance = max(
            1.0e-10,
            64.0 * torch.finfo(artifact.reference_fractional.dtype).eps,
        )
        validate_alias_matches_stabilizer(
            phase.modes[:3], stabilizer, tolerance=tolerance
        )
        validate_alias_matches_stabilizer(
            phase.modes, stabilizer, tolerance=tolerance
        )
    except ValueError as error:
        raise _error(
            "PHASE_STABILIZER_MISMATCH",
            str(error),
            schema=artifact.schema_version,
            template_id=artifact.template_id,
            validation_stage="phase_assembly",
        ) from error
    topology = ReferenceGraphTopology(
        reference_fractional=_cpu_clone(artifact.reference_fractional),
        site_types=_cpu_clone(artifact.site_types),
        edge_index=_cpu_clone(artifact.edge_index),
        shifts=_cpu_clone(artifact.periodic_shifts),
        reference_cell=_cpu_clone(artifact.reference_cell),
        cutoff=artifact.mp_cutoff,
        skin=artifact.mp_skin,
        maximum_strain=artifact.maximum_strain,
        minimum_edge_length=artifact.minimum_edge_length,
        pbc=tuple(artifact.pbc),
    )
    site_alignment = phase.site_type_alignment_weights[topology.site_types]
    return ReferenceTemplate.snapshot(
        artifact.template_id,
        topology,
        phase.modes,
        phase.mode_weights,
        site_alignment,
        phase.channel_weights,
        stabilizer,
        artifact.species_vocabulary,
        artifact.convention_version,
        artifact.strict_domain,
    )


__all__ = [
    "REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION",
    "REFERENCE_STRUCTURE_ARTIFACT_SCOPE",
    "ReferenceStructureArtifact",
    "ReferenceStructureArtifactDiagnostics",
    "ReferenceStructureArtifactError",
    "assemble_reference_template_from_artifact",
    "capture_reference_structure_artifact",
    "load_reference_structure_artifact",
    "save_reference_structure_artifact",
]
