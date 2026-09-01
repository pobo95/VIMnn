"""Dense-shaped compact transport support and deterministic diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Any, Mapping

import torch


TRANSPORT_SUPPORT_CONVENTION_VERSION = "dense_shaped_compact_c2_v1"


class TransportSupportError(ValueError):
    """Structured failure raised before a masked transport solve."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        template_id: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        self.reason_code = str(reason_code)
        self.template_id = template_id
        self.sample_id = sample_id
        context = []
        if sample_id is not None:
            context.append(f"sample_id={sample_id}")
        if template_id is not None:
            context.append(f"template_id={template_id}")
        suffix = "" if not context else " (" + ", ".join(context) + ")"
        super().__init__(f"{self.reason_code}: {message}{suffix}")


def _finite_real(value: Real, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG", f"{name} must be a real number"
        )
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG", f"{name} must be {qualifier}"
        )
    return result


@dataclass(frozen=True)
class TransportSupportConfig:
    """Transport support policy; ``dense`` exactly preserves the legacy path."""

    kind: str = "dense"
    cutoff: float = 4.0
    switch_width: float = 0.5
    candidate_skin: float = 0.2
    convention_version: str = TRANSPORT_SUPPORT_CONVENTION_VERSION
    backend: str = "dense"
    candidate_backend: str = "dense"
    site_block_size: int = 32
    atom_block_size: int = 32

    def __post_init__(self) -> None:
        if self.kind not in ("dense", "compact_c2"):
            raise TransportSupportError(
                "INVALID_SUPPORT_CONFIG", "kind must be dense or compact_c2"
            )
        if self.backend not in ("dense", "edge_list"):
            raise TransportSupportError(
                "INVALID_SUPPORT_CONFIG", "backend must be dense or edge_list"
            )
        if self.backend == "edge_list" and self.kind != "compact_c2":
            raise TransportSupportError(
                "INVALID_SUPPORT_CONFIG",
                "edge_list backend is supported only for compact_c2 transport",
            )
        if self.candidate_backend not in ("dense", "blocked"):
            raise TransportSupportError(
                "INVALID_SUPPORT_CONFIG",
                "candidate_backend must be dense or blocked",
            )
        for name, value in (
            ("site_block_size", self.site_block_size),
            ("atom_block_size", self.atom_block_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
            ):
                raise TransportSupportError(
                    "INVALID_SUPPORT_CONFIG",
                    f"{name} must be a positive integer",
                )
        if self.candidate_backend == "blocked" and self.backend != "edge_list":
            raise TransportSupportError(
                "INVALID_SUPPORT_CONFIG",
                "blocked candidate extraction requires backend=edge_list",
            )
        cutoff = _finite_real(self.cutoff, "cutoff", positive=True)
        width = _finite_real(self.switch_width, "switch_width", positive=True)
        skin = _finite_real(self.candidate_skin, "candidate_skin", positive=False)
        if width >= cutoff:
            raise TransportSupportError(
                "INVALID_SUPPORT_CONFIG", "switch_width must be smaller than cutoff"
            )
        if skin < 0.0:
            raise TransportSupportError(
                "INVALID_SUPPORT_CONFIG", "candidate_skin must be nonnegative"
            )
        if self.convention_version != TRANSPORT_SUPPORT_CONVENTION_VERSION:
            raise TransportSupportError(
                "INVALID_SUPPORT_CONFIG", "unsupported support convention version"
            )
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "switch_width", width)
        object.__setattr__(self, "candidate_skin", skin)
        object.__setattr__(self, "site_block_size", int(self.site_block_size))
        object.__setattr__(self, "atom_block_size", int(self.atom_block_size))

    @property
    def r_on(self) -> float:
        return self.cutoff - self.switch_width

    @property
    def r_candidate(self) -> float:
        return self.cutoff + self.candidate_skin

    def to_dict(self) -> dict[str, Any]:
        values = {
            "kind": self.kind,
            "cutoff": self.cutoff,
            "switch_width": self.switch_width,
            "candidate_skin": self.candidate_skin,
            "backend": self.backend,
            "convention_version": self.convention_version,
        }
        # Candidate blocking is an opt-in execution policy.  Omitting these
        # keys for the legacy dense-candidate path preserves canonical payloads
        # and checkpoint-resolved configs produced before S3C-1.
        if self.candidate_backend == "blocked":
            values.update(
                {
                    "candidate_backend": self.candidate_backend,
                    "site_block_size": self.site_block_size,
                    "atom_block_size": self.atom_block_size,
                }
            )
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "TransportSupportConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise TypeError("transport support config must be reconstructed from a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class TransportSupportDiagnostics:
    kind: str
    backend: str
    template_id: str | None
    sample_id: str | None
    candidate_edge_count: int
    active_edge_count: int
    core_edge_count: int
    atom_candidate_degrees: tuple[int, ...]
    atom_active_degrees: tuple[int, ...]
    atom_core_degrees: tuple[int, ...]
    site_candidate_degrees: tuple[int, ...]
    site_active_degrees: tuple[int, ...]
    site_core_degrees: tuple[int, ...]
    duplicate_atom_site_edge_count: int
    maximum_atom_matching_size: int
    total_matching_size: int
    total_support_feasible: bool
    cutoff_boundary_gap: float
    switch_on_boundary_gap: float
    candidate_boundary_gap: float
    active_dense_ratio: float
    candidate_dense_ratio: float
    minimum_switch_value: float
    maximum_switch_value: float
    effective_diagnostic_tolerance: float | None = None
    convention_version: str = TRANSPORT_SUPPORT_CONVENTION_VERSION
    candidate_backend: str = "dense"
    num_sites: int | None = None
    num_atoms: int | None = None
    site_block_size: int | None = None
    atom_block_size: int | None = None
    processed_block_count: int = 0
    maximum_pair_block_elements: int = 0
    theoretical_full_pair_elements: int = 0
    peak_temporary_geometry_elements: int = 0
    dense_candidate_allocation_observed: bool = True
    mic_image_gap: float = math.inf
    maximum_mic_image_count: int = 1
    candidate_fingerprint: str | None = None

    def with_effective_tolerance(self, value: float) -> "TransportSupportDiagnostics":
        return replace(self, effective_diagnostic_tolerance=float(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "backend": self.backend,
            "template_id": self.template_id,
            "sample_id": self.sample_id,
            "candidate_edge_count": self.candidate_edge_count,
            "active_edge_count": self.active_edge_count,
            "core_edge_count": self.core_edge_count,
            "atom_candidate_degrees": list(self.atom_candidate_degrees),
            "atom_active_degrees": list(self.atom_active_degrees),
            "atom_core_degrees": list(self.atom_core_degrees),
            "site_candidate_degrees": list(self.site_candidate_degrees),
            "site_active_degrees": list(self.site_active_degrees),
            "site_core_degrees": list(self.site_core_degrees),
            "duplicate_atom_site_edge_count": self.duplicate_atom_site_edge_count,
            "maximum_atom_matching_size": self.maximum_atom_matching_size,
            "total_matching_size": self.total_matching_size,
            "total_support_feasible": self.total_support_feasible,
            "cutoff_boundary_gap": self.cutoff_boundary_gap,
            "switch_on_boundary_gap": self.switch_on_boundary_gap,
            "candidate_boundary_gap": self.candidate_boundary_gap,
            "active_dense_ratio": self.active_dense_ratio,
            "candidate_dense_ratio": self.candidate_dense_ratio,
            "minimum_switch_value": self.minimum_switch_value,
            "maximum_switch_value": self.maximum_switch_value,
            "effective_diagnostic_tolerance": self.effective_diagnostic_tolerance,
            "convention_version": self.convention_version,
            "candidate_backend": self.candidate_backend,
            "num_sites": self.num_sites,
            "num_atoms": self.num_atoms,
            "site_block_size": self.site_block_size,
            "atom_block_size": self.atom_block_size,
            "processed_block_count": self.processed_block_count,
            "maximum_pair_block_elements": self.maximum_pair_block_elements,
            "theoretical_full_pair_elements": self.theoretical_full_pair_elements,
            "peak_temporary_geometry_elements": self.peak_temporary_geometry_elements,
            "dense_candidate_allocation_observed": self.dense_candidate_allocation_observed,
            "mic_image_gap": self.mic_image_gap,
            "maximum_mic_image_count": self.maximum_mic_image_count,
            "candidate_fingerprint": self.candidate_fingerprint,
        }


def compact_c2_switch(
    distances: torch.Tensor, config: TransportSupportConfig
) -> torch.Tensor:
    """Plateaued quintic switch, exactly zero at and beyond ``cutoff``."""

    if config.kind != "compact_c2":
        return torch.ones_like(distances)
    if not distances.is_floating_point() or distances.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY", "distances must use float32 or float64"
        )
    if not bool(torch.all(torch.isfinite(distances)).detach()):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY", "distances contain NaN or Inf"
        )
    r_on = distances.new_tensor(config.r_on)
    r_off = distances.new_tensor(config.cutoff)
    u = (distances - r_on) / (r_off - r_on)
    # Algebraically identical to 1 - 10u^3 + 15u^4 - 6u^5, while avoiding
    # cancellation to a negative float32 value immediately below r_off.
    polynomial = (1.0 - u).pow(3) * (1.0 + 3.0 * u + 6.0 * u.square())
    return torch.where(
        distances <= r_on,
        torch.ones_like(distances),
        torch.where(distances < r_off, polynomial, torch.zeros_like(distances)),
    )


def _maximum_matching(mask: list[list[bool]]) -> tuple[int, list[int], list[int]]:
    rows = len(mask)
    columns = len(mask[0]) if rows else 0
    column_to_row = [-1] * columns

    def augment(row: int, seen: list[bool]) -> bool:
        for column in range(columns):
            if not mask[row][column] or seen[column]:
                continue
            seen[column] = True
            previous = column_to_row[column]
            if previous < 0 or augment(previous, seen):
                column_to_row[column] = row
                return True
        return False

    count = 0
    for row in range(rows):
        count += int(augment(row, [False] * columns))
    row_to_column = [-1] * rows
    for column, row in enumerate(column_to_row):
        if row >= 0:
            row_to_column[row] = column
    return count, row_to_column, column_to_row


def _maximum_matching_adjacency(
    row_adjacency: list[list[int]], columns: int
) -> tuple[int, list[int], list[int]]:
    """Deterministic bipartite matching without a dense boolean matrix."""

    column_to_row = [-1] * columns

    def augment(row: int, seen: list[bool]) -> bool:
        for column in row_adjacency[row]:
            if column < 0 or column >= columns:
                raise ValueError("sparse matching adjacency index is out of range")
            if seen[column]:
                continue
            seen[column] = True
            previous = column_to_row[column]
            if previous < 0 or augment(previous, seen):
                column_to_row[column] = row
                return True
        return False

    count = 0
    for row in range(len(row_adjacency)):
        count += int(augment(row, [False] * columns))
    row_to_column = [-1] * len(row_adjacency)
    for column, row in enumerate(column_to_row):
        if row >= 0:
            row_to_column[row] = column
    return count, row_to_column, column_to_row


def _strong_components(adjacency: list[list[int]]) -> list[int]:
    index = 0
    stack: list[int] = []
    on_stack = [False] * len(adjacency)
    indices = [-1] * len(adjacency)
    low = [0] * len(adjacency)
    components = [-1] * len(adjacency)
    component_index = 0

    def visit(vertex: int) -> None:
        nonlocal index, component_index
        indices[vertex] = low[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack[vertex] = True
        for target in adjacency[vertex]:
            if indices[target] < 0:
                visit(target)
                low[vertex] = min(low[vertex], low[target])
            elif on_stack[target]:
                low[vertex] = min(low[vertex], indices[target])
        if low[vertex] == indices[vertex]:
            while True:
                member = stack.pop()
                on_stack[member] = False
                components[member] = component_index
                if member == vertex:
                    break
            component_index += 1

    for vertex in range(len(adjacency)):
        if indices[vertex] < 0:
            visit(vertex)
    return components


def _has_total_support(mask: list[list[bool]], matching_size: int) -> bool:
    size = len(mask)
    if matching_size != size or any(len(row) != size for row in mask):
        return False
    _, row_to_column, column_to_row = _maximum_matching(mask)
    adjacency: list[list[int]] = [[] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            if mask[row][column]:
                adjacency[row].append(column_to_row[column])
    components = _strong_components(adjacency)
    for row in range(size):
        for column in range(size):
            if not mask[row][column] or row_to_column[row] == column:
                continue
            if components[row] != components[column_to_row[column]]:
                return False
    return True


def _has_total_support_adjacency(
    row_adjacency: list[list[int]], size: int, matching_size: int
) -> bool:
    """Sparse equivalent of :func:`_has_total_support`."""

    if matching_size != size or len(row_adjacency) != size:
        return False
    _, row_to_column, column_to_row = _maximum_matching_adjacency(
        row_adjacency, size
    )
    adjacency: list[list[int]] = [[] for _ in range(size)]
    for row, columns in enumerate(row_adjacency):
        for column in columns:
            matched_row = column_to_row[column]
            if matched_row < 0:
                return False
            adjacency[row].append(matched_row)
    components = _strong_components(adjacency)
    for row, columns in enumerate(row_adjacency):
        for column in columns:
            if row_to_column[row] == column:
                continue
            if components[row] != components[column_to_row[column]]:
                return False
    return True


def _has_total_support_with_dense_vacancy(
    site_atomic_adjacency: list[list[int]],
    atom_to_site: list[int],
    num_sites: int,
    num_atoms: int,
) -> bool:
    """Certify total support with implicit fully connected vacancy clones.

    A complete atom matching leaves exactly ``K=M-N`` site rows for the
    aggregate vacancy mass.  Dense vacancy-clone columns match those rows.  In
    the alternating directed graph, every row connects to every vacancy row;
    representing that complete relation by one root plus a vacancy-row star is
    reachability-equivalent and costs ``O(E+M)`` rather than ``O(M*K)``.
    """

    vacancies = num_sites - num_atoms
    if vacancies <= 0:
        matching, _, _ = _maximum_matching_adjacency(
            site_atomic_adjacency, num_atoms
        )
        return _has_total_support_adjacency(
            site_atomic_adjacency, num_sites, matching
        )
    matched_sites = {site for site in atom_to_site if site >= 0}
    vacancy_sites = [site for site in range(num_sites) if site not in matched_sites]
    if len(vacancy_sites) != vacancies:
        return False
    root = vacancy_sites[0]
    alternating: list[list[int]] = [[] for _ in range(num_sites)]
    for site, atoms in enumerate(site_atomic_adjacency):
        alternating[site].extend(atom_to_site[atom] for atom in atoms)
        alternating[site].append(root)
    # The complete vacancy relation makes all vacancy-matched rows mutually
    # reachable.  This star has exactly the same transitive closure.
    alternating[root].extend(vacancy_sites)
    for site in vacancy_sites:
        alternating[site].append(root)
    components = _strong_components(alternating)
    return len(set(components)) == 1


def validate_compact_support_edges(
    site_index: torch.Tensor,
    atom_index: torch.Tensor,
    distances: torch.Tensor,
    switch: torch.Tensor,
    num_sites: int,
    num_atoms: int,
    config: TransportSupportConfig,
    *,
    template_id: str | None = None,
    sample_id: str | None = None,
    cutoff_boundary_gap: float | None = None,
    switch_on_boundary_gap: float | None = None,
    candidate_boundary_gap: float | None = None,
    mic_image_gap: float = math.inf,
    maximum_mic_image_count: int = 1,
    candidate_fingerprint: str | None = None,
    processed_block_count: int = 0,
    maximum_pair_block_elements: int = 0,
    peak_temporary_geometry_elements: int = 0,
) -> tuple[torch.Tensor, TransportSupportDiagnostics]:
    """Validate compact candidate edges using sparse adjacency only.

    ``site_index``/``atom_index`` enumerate every ``d < r_candidate`` pair in
    canonical site-major order.  Matching and total-support certification use
    only the strictly positive ``d < r_off`` graph; the dense aggregate
    vacancy reservoir is represented by implicit vacancy-clone columns.
    """

    if not isinstance(config, TransportSupportConfig) or (
        config.kind != "compact_c2" or config.backend != "edge_list"
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "sparse support validation requires compact_c2 edge_list config",
            template_id=template_id,
            sample_id=sample_id,
        )
    if (
        isinstance(num_sites, bool)
        or isinstance(num_atoms, bool)
        or not isinstance(num_sites, Integral)
        or not isinstance(num_atoms, Integral)
        or num_sites <= 0
        or num_atoms < 0
        or num_atoms > num_sites
    ):
        raise TransportSupportError(
            "NO_TOTAL_SUPPORT",
            f"invalid site/atom counts M={num_sites}, N={num_atoms}",
            template_id=template_id,
            sample_id=sample_id,
        )
    edges = int(site_index.numel())
    if (
        site_index.shape != (edges,)
        or atom_index.shape != (edges,)
        or site_index.dtype != torch.long
        or atom_index.dtype != torch.long
        or distances.shape != (edges,)
        or switch.shape != (edges,)
    ):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "candidate indices/distances/switch have inconsistent edge shapes",
            template_id=template_id,
            sample_id=sample_id,
        )
    if (
        site_index.device != distances.device
        or atom_index.device != distances.device
        or switch.device != distances.device
        or switch.dtype != distances.dtype
        or distances.dtype not in (torch.float32, torch.float64)
    ):
        raise TransportSupportError(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "candidate edge tensors must share float/index device and dtype",
            template_id=template_id,
            sample_id=sample_id,
        )
    if not bool(torch.all(torch.isfinite(distances)).detach()) or bool(
        torch.any(distances < 0.0).detach()
    ) or not bool(torch.all(torch.isfinite(switch)).detach()):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "candidate edge geometry/switch is nonfinite or negative",
            template_id=template_id,
            sample_id=sample_id,
        )
    if edges and (
        bool(torch.any((site_index < 0) | (site_index >= num_sites)).detach())
        or bool(torch.any((atom_index < 0) | (atom_index >= num_atoms)).detach())
    ):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "candidate edge index lies outside the site/atom domain",
            template_id=template_id,
            sample_id=sample_id,
        )
    pair_key = site_index * max(num_atoms, 1) + atom_index
    duplicate_count = edges - int(torch.unique(pair_key).numel())
    if duplicate_count:
        raise TransportSupportError(
            "DUPLICATE_EDGE",
            f"candidate list contains {duplicate_count} duplicate atom-site pairs",
            template_id=template_id,
            sample_id=sample_id,
        )
    if edges and bool(torch.any(pair_key[1:] <= pair_key[:-1]).detach()):
        raise TransportSupportError(
            "DUPLICATE_EDGE",
            "candidate list is not in unique site-major canonical order",
            template_id=template_id,
            sample_id=sample_id,
        )
    if edges and bool(
        torch.any(distances >= distances.new_tensor(config.r_candidate)).detach()
    ):
        raise TransportSupportError(
            "CANDIDATE_BOUNDARY_INSTABILITY",
            "candidate list contains an edge outside the strict candidate radius",
            template_id=template_id,
            sample_id=sample_id,
        )

    active = distances < distances.new_tensor(config.cutoff)
    core = distances <= distances.new_tensor(config.r_on)
    if bool(torch.any(active & (switch <= 0.0)).detach()) or bool(
        torch.any((~active) & (switch != 0.0)).detach()
    ):
        raise TransportSupportError(
            "NO_TOTAL_SUPPORT",
            "compact switch does not match the positive-weight active graph",
            template_id=template_id,
            sample_id=sample_id,
        )

    candidate_site_cpu = site_index.detach().cpu().tolist()
    candidate_atom_cpu = atom_index.detach().cpu().tolist()
    active_cpu = active.detach().cpu().tolist()
    core_cpu = core.detach().cpu().tolist()
    site_candidate = [0] * num_sites
    site_active = [0] * num_sites
    site_core = [0] * num_sites
    atom_candidate = [0] * num_atoms
    atom_active = [0] * num_atoms
    atom_core = [0] * num_atoms
    atom_adjacency: list[list[int]] = [[] for _ in range(num_atoms)]
    site_active_adjacency: list[list[int]] = [[] for _ in range(num_sites)]
    for site, atom, is_active, is_core in zip(
        candidate_site_cpu, candidate_atom_cpu, active_cpu, core_cpu
    ):
        site_candidate[site] += 1
        atom_candidate[atom] += 1
        if is_active:
            site_active[site] += 1
            atom_active[atom] += 1
            atom_adjacency[atom].append(site)
            site_active_adjacency[site].append(atom)
        if is_core:
            site_core[site] += 1
            atom_core[atom] += 1
    if num_atoms and min(atom_active) == 0:
        atom = atom_active.index(0)
        raise TransportSupportError(
            "ATOM_WITHOUT_SUPPORT",
            f"atom column {atom} has no active site edge",
            template_id=template_id,
            sample_id=sample_id,
        )
    atom_matching, atom_to_site, _ = _maximum_matching_adjacency(
        atom_adjacency, num_sites
    )
    if atom_matching != num_atoms:
        raise TransportSupportError(
            "INCOMPLETE_ATOM_MATCHING",
            f"maximum matching size {atom_matching} is smaller than N={num_atoms}",
            template_id=template_id,
            sample_id=sample_id,
        )
    total_matching = num_sites
    total_support = _has_total_support_with_dense_vacancy(
        site_active_adjacency,
        atom_to_site,
        num_sites,
        num_atoms,
    )
    if total_matching != num_sites or not total_support:
        raise TransportSupportError(
            "NO_TOTAL_SUPPORT",
            "atom plus dense vacancy support lacks total support for positive scaling",
            template_id=template_id,
            sample_id=sample_id,
        )

    def resolved_gap(value: float | None, boundary: float) -> float:
        if value is not None:
            return float(value)
        if edges:
            return float(torch.min(torch.abs(distances.detach() - boundary)).cpu())
        return math.inf

    dense_edges = num_sites * num_atoms
    minimum_switch = (
        0.0
        if edges < dense_edges or not edges
        else float(switch.detach().min().cpu())
    )
    maximum_switch = float(switch.detach().max().cpu()) if edges else 0.0
    diagnostics = TransportSupportDiagnostics(
        kind=config.kind,
        backend=config.backend,
        template_id=template_id,
        sample_id=sample_id,
        candidate_edge_count=edges,
        active_edge_count=sum(site_active),
        core_edge_count=sum(site_core),
        atom_candidate_degrees=tuple(atom_candidate),
        atom_active_degrees=tuple(atom_active),
        atom_core_degrees=tuple(atom_core),
        site_candidate_degrees=tuple(site_candidate),
        site_active_degrees=tuple(site_active),
        site_core_degrees=tuple(site_core),
        duplicate_atom_site_edge_count=0,
        maximum_atom_matching_size=atom_matching,
        total_matching_size=total_matching,
        total_support_feasible=True,
        cutoff_boundary_gap=resolved_gap(cutoff_boundary_gap, config.cutoff),
        switch_on_boundary_gap=resolved_gap(
            switch_on_boundary_gap, config.r_on
        ),
        candidate_boundary_gap=resolved_gap(
            candidate_boundary_gap, config.r_candidate
        ),
        active_dense_ratio=(sum(site_active) / dense_edges if dense_edges else 0.0),
        candidate_dense_ratio=(edges / dense_edges if dense_edges else 0.0),
        minimum_switch_value=minimum_switch,
        maximum_switch_value=maximum_switch,
        candidate_backend=config.candidate_backend,
        num_sites=num_sites,
        num_atoms=num_atoms,
        site_block_size=(
            config.site_block_size if config.candidate_backend == "blocked" else None
        ),
        atom_block_size=(
            config.atom_block_size if config.candidate_backend == "blocked" else None
        ),
        processed_block_count=int(processed_block_count),
        maximum_pair_block_elements=int(maximum_pair_block_elements),
        theoretical_full_pair_elements=dense_edges,
        peak_temporary_geometry_elements=int(peak_temporary_geometry_elements),
        dense_candidate_allocation_observed=config.candidate_backend == "dense",
        mic_image_gap=float(mic_image_gap),
        maximum_mic_image_count=int(maximum_mic_image_count),
        candidate_fingerprint=candidate_fingerprint,
    )
    return active, diagnostics


def validate_compact_support(
    distances: torch.Tensor,
    switch: torch.Tensor,
    config: TransportSupportConfig,
    *,
    template_id: str | None = None,
    sample_id: str | None = None,
) -> tuple[torch.Tensor, TransportSupportDiagnostics]:
    """Validate a fixed support branch before masked Sinkhorn arithmetic."""

    if distances.ndim != 2 or switch.shape != distances.shape:
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "distance and switch tensors must have the same [M,N] shape",
            template_id=template_id,
            sample_id=sample_id,
        )
    if not bool(torch.all(torch.isfinite(distances)).detach()) or bool(
        torch.any(distances < 0.0).detach()
    ):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "support distances contain NaN, Inf, or negative values",
            template_id=template_id,
            sample_id=sample_id,
        )
    if not bool(torch.all(torch.isfinite(switch)).detach()) or bool(
        torch.any((switch < 0.0) | (switch > 1.0)).detach()
    ):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "compact switch contains nonfinite or out-of-range values",
            template_id=template_id,
            sample_id=sample_id,
        )
    sites, atoms = (int(value) for value in distances.shape)
    if atoms > sites:
        raise TransportSupportError(
            "NO_TOTAL_SUPPORT",
            f"atom count N={atoms} exceeds site count M={sites}",
            template_id=template_id,
            sample_id=sample_id,
        )
    active = distances < distances.new_tensor(config.cutoff)
    candidate = distances < distances.new_tensor(config.r_candidate)
    core = distances <= distances.new_tensor(config.r_on)
    if bool(torch.any(active & (switch <= 0.0)).detach()):
        raise TransportSupportError(
            "NO_TOTAL_SUPPORT",
            "an active compact edge has a nonpositive switch value",
            template_id=template_id,
            sample_id=sample_id,
        )
    active_cpu = active.detach().cpu()
    candidate_cpu = candidate.detach().cpu()
    core_cpu = core.detach().cpu()
    atom_degrees = active_cpu.sum(dim=0).tolist()
    site_degrees = active_cpu.sum(dim=1).tolist()
    if atoms and min(atom_degrees) == 0:
        atom = atom_degrees.index(0)
        raise TransportSupportError(
            "ATOM_WITHOUT_SUPPORT",
            f"atom column {atom} has no active site edge",
            template_id=template_id,
            sample_id=sample_id,
        )
    atom_mask = active_cpu.tolist()
    atom_rows = [
        [atom_mask[site][atom] for site in range(sites)] for atom in range(atoms)
    ]
    atom_matching, _, _ = _maximum_matching(atom_rows)
    if atom_matching != atoms:
        raise TransportSupportError(
            "INCOMPLETE_ATOM_MATCHING",
            f"maximum matching size {atom_matching} is smaller than N={atoms}",
            template_id=template_id,
            sample_id=sample_id,
        )
    vacancies = sites - atoms
    augmented = [row[:] + [True] * vacancies for row in atom_mask]
    total_matching, _, _ = _maximum_matching(augmented)
    total_support = _has_total_support(augmented, total_matching)
    if total_matching != sites or not total_support:
        raise TransportSupportError(
            "NO_TOTAL_SUPPORT",
            "atom plus dense vacancy support lacks total support for positive scaling",
            template_id=template_id,
            sample_id=sample_id,
        )
    dense_edges = sites * atoms
    if dense_edges:
        cutoff_gap = float(
            torch.min(torch.abs(distances.detach() - config.cutoff)).cpu()
        )
        switch_on_gap = float(
            torch.min(torch.abs(distances.detach() - config.r_on)).cpu()
        )
        candidate_gap = float(
            torch.min(torch.abs(distances.detach() - config.r_candidate)).cpu()
        )
        minimum_switch = float(switch.detach().min().cpu())
        maximum_switch = float(switch.detach().max().cpu())
    else:
        cutoff_gap = math.inf
        switch_on_gap = math.inf
        candidate_gap = math.inf
        minimum_switch = 0.0
        maximum_switch = 0.0
    diagnostics = TransportSupportDiagnostics(
        kind=config.kind,
        backend=config.backend,
        template_id=template_id,
        sample_id=sample_id,
        candidate_edge_count=int(candidate.detach().sum().cpu()),
        active_edge_count=int(active_cpu.sum()),
        core_edge_count=int(core.detach().sum().cpu()),
        atom_candidate_degrees=tuple(
            int(value) for value in candidate_cpu.sum(dim=0).tolist()
        ),
        atom_active_degrees=tuple(int(value) for value in atom_degrees),
        atom_core_degrees=tuple(
            int(value) for value in core_cpu.sum(dim=0).tolist()
        ),
        site_candidate_degrees=tuple(
            int(value) for value in candidate_cpu.sum(dim=1).tolist()
        ),
        site_active_degrees=tuple(int(value) for value in site_degrees),
        site_core_degrees=tuple(
            int(value) for value in core_cpu.sum(dim=1).tolist()
        ),
        duplicate_atom_site_edge_count=0,
        maximum_atom_matching_size=atom_matching,
        total_matching_size=total_matching,
        total_support_feasible=True,
        cutoff_boundary_gap=cutoff_gap,
        switch_on_boundary_gap=switch_on_gap,
        candidate_boundary_gap=candidate_gap,
        active_dense_ratio=(float(active_cpu.sum()) / dense_edges if dense_edges else 0.0),
        candidate_dense_ratio=(
            float(candidate.detach().sum().cpu()) / dense_edges if dense_edges else 0.0
        ),
        minimum_switch_value=minimum_switch,
        maximum_switch_value=maximum_switch,
    )
    return active, diagnostics
