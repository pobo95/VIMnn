"""Dense-shaped compact transport support and deterministic diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real
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

    @property
    def r_on(self) -> float:
        return self.cutoff - self.switch_width

    @property
    def r_candidate(self) -> float:
        return self.cutoff + self.candidate_skin

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "cutoff": self.cutoff,
            "switch_width": self.switch_width,
            "candidate_skin": self.candidate_skin,
            "backend": self.backend,
            "convention_version": self.convention_version,
        }

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
