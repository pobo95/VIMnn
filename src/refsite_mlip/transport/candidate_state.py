"""Functional compact-candidate neighbor state with certified skin reuse."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Integral, Real
import struct
from typing import Any, Sequence

import torch

from .candidate import (
    _candidate_fingerprint,
    build_periodic_compact_transport_edges,
)
from .cost import minimum_image_diagnostics
from .edge_list import (
    CompactTransportEdges,
    build_compact_transport_edges_from_candidates,
)
from .support import (
    TransportSupportConfig,
    TransportSupportDiagnostics,
    TransportSupportError,
    compact_c2_switch,
    validate_compact_support_edges,
)


CANDIDATE_NEIGHBOR_STATE_SCHEMA_VERSION = "compact_candidate_neighbor_state_v1"
_GUARD_FORMULA = (
    "gamma_n * max(||cell||_2, r_candidate, 1), "
    "gamma_n=(n*eps)/(1-n*eps), "
    "n=48+12*ceil(log2(max(M,N,2)))"
)


def _error(
    reason_code: str,
    message: str,
    *,
    template_fingerprint: str | None = None,
    sample_id: str | None = None,
) -> TransportSupportError:
    return TransportSupportError(
        reason_code,
        message,
        template_id=template_fingerprint,
        sample_id=sample_id,
    )


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().clone()


def _hash_text(digest: Any, value: Any) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    _hash_text(digest, value.dtype)
    digest.update(struct.pack("<I", value.ndim))
    if value.ndim:
        digest.update(struct.pack("<" + "q" * value.ndim, *value.shape))
    digest.update(value.numpy().tobytes())


def _pair_set_fingerprint(
    num_sites: int,
    num_atoms: int,
    site_index: torch.Tensor,
    atom_index: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "compact_candidate_pair_set_v1")
    digest.update(struct.pack("<qq", int(num_sites), int(num_atoms)))
    _hash_tensor(digest, site_index)
    _hash_tensor(digest, atom_index)
    return digest.hexdigest()


def _live_support_fingerprint(
    num_sites: int,
    num_atoms: int,
    site_index: torch.Tensor,
    atom_index: torch.Tensor,
    periodic_shift: torch.Tensor,
    active: torch.Tensor,
) -> str:
    # Reuse the S3C-1 content hash so fresh and reused support fingerprints are
    # directly comparable across candidate backends and block sizes.
    return _candidate_fingerprint(
        num_sites,
        num_atoms,
        site_index,
        atom_index,
        periodic_shift,
        active,
    )


def _active_support_fingerprint(edges: CompactTransportEdges) -> str:
    """Hash the physical positive-weight pair set, independent of MIC images."""

    active = edges.active
    return _pair_set_fingerprint(
        edges.num_sites,
        edges.num_atoms,
        edges.site_index[active],
        edges.atom_index[active],
    )


def _support_content_fingerprint(config: TransportSupportConfig) -> str:
    """Hash physical support content while excluding execution block sizes."""

    payload = {
        "kind": config.kind,
        "cutoff": config.cutoff,
        "switch_width": config.switch_width,
        "candidate_skin": config.candidate_skin,
        "backend": config.backend,
        "convention_version": config.convention_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atom_order_fingerprint(
    atomic_numbers: torch.Tensor, atom_order_identity: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "compact_candidate_atom_order_v1")
    _hash_tensor(digest, atomic_numbers)
    _hash_tensor(digest, atom_order_identity)
    return digest.hexdigest()


def _full_pbc(
    pbc: Sequence[bool] | torch.Tensor,
    *,
    template_fingerprint: str,
    sample_id: str | None,
) -> tuple[bool, bool, bool]:
    if isinstance(pbc, torch.Tensor):
        if pbc.shape != (3,) or pbc.dtype != torch.bool:
            raise _error(
                "UNSUPPORTED_PBC",
                "pbc tensor must be bool [3]",
                template_fingerprint=template_fingerprint,
                sample_id=sample_id,
            )
        values = pbc.detach().cpu().tolist()
    else:
        values = list(pbc)
    if len(values) != 3 or any(not isinstance(value, bool) for value in values):
        raise _error(
            "UNSUPPORTED_PBC",
            "pbc must contain exactly three booleans",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    result = tuple(values)
    if result != (True, True, True):
        raise _error(
            "UNSUPPORTED_PBC",
            "candidate neighbor reuse requires full PBC",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    return result


def _validate_fingerprint(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _validate_support_config(config: TransportSupportConfig) -> None:
    if not isinstance(config, TransportSupportConfig) or (
        config.kind != "compact_c2"
        or config.backend != "edge_list"
        or config.candidate_backend != "blocked"
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "candidate neighbor state requires compact_c2 edge_list blocked config",
        )
    if not config.candidate_skin > 0.0:
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "candidate neighbor state requires positive candidate_skin",
        )


def _validate_image_range(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG", "image_range must be a positive integer"
        )
    return int(value)


def _validate_threshold(value: Real | None, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            f"{name} must be finite and nonnegative",
        )
    return float(value)


def _validate_current_geometry(
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    origin: torch.Tensor | None,
    atomic_numbers: torch.Tensor,
    atom_order_identity: torch.Tensor | None,
    *,
    template_fingerprint: str,
    sample_id: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise _error(
            "NONFINITE_SUPPORT_GEOMETRY",
            "positions must have shape [N,3]",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    if reference_sites.ndim != 2 or reference_sites.shape[1:] != (3,):
        raise _error(
            "NONFINITE_SUPPORT_GEOMETRY",
            "aligned reference_sites must have shape [M,3]",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    if cell.shape != (3, 3):
        raise _error(
            "SINGULAR_CELL",
            "cell must have shape [3,3]",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    if positions.dtype not in (torch.float32, torch.float64):
        raise _error(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "candidate state supports float32 and float64 geometry",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    if (
        reference_sites.dtype != positions.dtype
        or cell.dtype != positions.dtype
        or reference_sites.device != positions.device
        or cell.device != positions.device
        or positions.device.type not in ("cpu", "cuda")
    ):
        raise _error(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "positions, references, and cell must share a CPU/CUDA dtype/device",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    if not all(
        bool(torch.all(torch.isfinite(value)).detach())
        for value in (positions, reference_sites, cell)
    ):
        raise _error(
            "NONFINITE_SUPPORT_GEOMETRY",
            "candidate state geometry contains NaN or Inf",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    determinant = torch.linalg.det(cell)
    if not bool(torch.isfinite(determinant).detach()) or float(
        determinant.abs().detach().cpu()
    ) <= torch.finfo(cell.dtype).eps:
        raise _error(
            "SINGULAR_CELL",
            "physical cell is singular",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    if origin is None:
        origin = positions.new_zeros(3)
    if (
        origin.shape != (3,)
        or origin.dtype != positions.dtype
        or origin.device != positions.device
        or not bool(torch.all(torch.isfinite(origin)).detach())
    ):
        raise _error(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "origin must be finite [3] and share geometry dtype/device",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    atoms = int(positions.shape[0])
    if (
        atomic_numbers.shape != (atoms,)
        or atomic_numbers.dtype != torch.long
        or atomic_numbers.device != positions.device
        or bool(torch.any(atomic_numbers <= 0).detach())
    ):
        raise _error(
            "ATOM_ORDER_CHANGED",
            "atomic_numbers must be positive device-local long [N]",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    if atom_order_identity is None:
        atom_order_identity = torch.arange(
            atoms, dtype=torch.long, device=positions.device
        )
    if (
        atom_order_identity.shape != (atoms,)
        or atom_order_identity.dtype != torch.long
        or atom_order_identity.device != positions.device
        or int(torch.unique(atom_order_identity).numel()) != atoms
    ):
        raise _error(
            "ATOM_ORDER_CHANGED",
            "atom_order_identity must be unique device-local long [N]",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    return origin, atom_order_identity


def _effective_numerical_guard(
    dtype: torch.dtype,
    cell: torch.Tensor,
    r_candidate: float,
    num_sites: int,
    num_atoms: int,
    configured_guard: Real | None,
) -> float:
    """Bound comparison roundoff for the two build/current MIC reductions.

    The fixed 48-operation term conservatively covers each 3x3 solve,
    reconstruction matmul, norm, and comparison.  Twelve operations per
    pairwise reduction level cover the atom/site maxima and the final bound
    additions.  Multiplication by the larger cell/cutoff scale converts this
    standard floating-point ``gamma_n``-style count to Cartesian length.
    """

    depth = math.ceil(math.log2(max(num_sites, num_atoms, 2)))
    operation_count = 48 + 12 * depth
    cell_scale = float(
        torch.linalg.matrix_norm(cell.detach(), ord=2).cpu()
    )
    scale = max(cell_scale, float(r_candidate), 1.0)
    accumulated_epsilon = torch.finfo(dtype).eps * operation_count
    if accumulated_epsilon >= 1.0:
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "dtype/geometry comparison roundoff bound is undefined",
        )
    automatic = (
        accumulated_epsilon / (1.0 - accumulated_epsilon)
    ) * scale
    if configured_guard is None:
        return automatic
    if (
        isinstance(configured_guard, bool)
        or not isinstance(configured_guard, Real)
        or not math.isfinite(float(configured_guard))
        or float(configured_guard) < 0.0
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "numerical_guard must be finite and nonnegative",
        )
    # A user override may make reuse more conservative, never less safe than
    # the dtype/geometry-derived floor.
    return max(automatic, float(configured_guard))


@dataclass(frozen=True)
class CandidateReuseDecision:
    reused: bool
    rebuilt: bool
    reason_code: str
    build_generation: int
    build_count: int
    reuse_count: int
    delta_atom: float | None
    delta_site: float | None
    delta_pair_bound: float | None
    numerical_guard: float
    numerical_guard_formula: str
    skin: float
    remaining_skin: float | None
    cached_candidate_count: int
    current_active_count: int
    candidate_boundary_lower_bound: float
    cached_pair_set_fingerprint: str
    current_live_support_fingerprint: str
    fresh_candidate_fingerprint: str | None
    atom_mic_image_gap: float
    site_mic_image_gap: float
    fresh_retry_performed: bool
    fresh_retry_reason: str | None
    state_materialized: bool
    processed_block_count: int
    maximum_pair_block_elements: int
    dense_allocation_observed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class CompactCandidateNeighborState:
    """Immutable build snapshot controlling compact candidate reuse.

    Tensor mutation remains possible in PyTorch despite a frozen dataclass, so
    every consuming API recomputes ``integrity_fingerprint`` before reading any
    geometry or compatibility metadata.
    """

    template_fingerprint: str
    support_content_fingerprint: str
    phase_site_branch_fingerprint: str
    num_sites: int
    num_atoms: int
    atom_order_fingerprint: str
    site_index: torch.Tensor
    atom_index: torch.Tensor
    build_periodic_shift: torch.Tensor
    build_positions: torch.Tensor
    build_reference_sites: torch.Tensor
    build_cell: torch.Tensor
    build_origin: torch.Tensor
    build_atomic_numbers: torch.Tensor
    build_atom_order_identity: torch.Tensor
    r_off: float
    r_candidate: float
    skin: float
    candidate_pair_set_fingerprint: str
    build_candidate_fingerprint: str
    build_generation: int
    reuse_count: int
    image_range: int
    build_diagnostics: TransportSupportDiagnostics | None = None
    schema_version: str = CANDIDATE_NEIGHBOR_STATE_SCHEMA_VERSION
    integrity_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_NEIGHBOR_STATE_SCHEMA_VERSION:
            raise TransportSupportError(
                "STATE_SCHEMA_MISMATCH",
                f"unsupported candidate state schema {self.schema_version!r}",
            )
        for name in (
            "template_fingerprint",
            "support_content_fingerprint",
            "phase_site_branch_fingerprint",
            "atom_order_fingerprint",
            "candidate_pair_set_fingerprint",
            "build_candidate_fingerprint",
        ):
            _validate_fingerprint(getattr(self, name), name)
        for name in (
            "site_index",
            "atom_index",
            "build_periodic_shift",
            "build_positions",
            "build_reference_sites",
            "build_cell",
            "build_origin",
            "build_atomic_numbers",
            "build_atom_order_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            object.__setattr__(self, name, _clone(value))
        if self.num_sites <= 0 or self.num_atoms < 0 or self.num_atoms > self.num_sites:
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH", "invalid candidate state M/N"
            )
        edges = int(self.site_index.numel())
        floating = self.build_positions
        if (
            floating.dtype not in (torch.float32, torch.float64)
            or self.build_positions.shape != (self.num_atoms, 3)
            or self.build_reference_sites.shape != (self.num_sites, 3)
            or self.build_cell.shape != (3, 3)
            or self.build_origin.shape != (3,)
        ):
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH",
                "candidate state build geometry has invalid shape/dtype",
            )
        if any(
            value.dtype != floating.dtype or value.device != floating.device
            for value in (
                self.build_reference_sites,
                self.build_cell,
                self.build_origin,
            )
        ):
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH",
                "candidate state floating snapshots must share dtype/device",
            )
        for value in (
            self.build_positions,
            self.build_reference_sites,
            self.build_cell,
            self.build_origin,
        ):
            if value.requires_grad or not bool(torch.all(torch.isfinite(value))):
                raise TransportSupportError(
                    "STATE_INTEGRITY_MISMATCH",
                    "candidate state snapshots must be finite and detached",
                )
        if (
            self.site_index.shape != (edges,)
            or self.atom_index.shape != (edges,)
            or self.build_periodic_shift.shape != (edges, 3)
            or self.site_index.dtype != torch.long
            or self.atom_index.dtype != torch.long
            or self.build_periodic_shift.dtype != torch.long
            or self.build_atomic_numbers.shape != (self.num_atoms,)
            or self.build_atom_order_identity.shape != (self.num_atoms,)
            or self.build_atomic_numbers.dtype != torch.long
            or self.build_atom_order_identity.dtype != torch.long
        ):
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH",
                "candidate state index/identity tensors have invalid shape/dtype",
            )
        for value in (
            self.site_index,
            self.atom_index,
            self.build_periodic_shift,
            self.build_atomic_numbers,
            self.build_atom_order_identity,
        ):
            if value.device != floating.device or value.requires_grad:
                raise TransportSupportError(
                    "STATE_INTEGRITY_MISMATCH",
                    "candidate state tensors must share device and be detached",
                )
        if edges and (
            bool(torch.any((self.site_index < 0) | (self.site_index >= self.num_sites)))
            or bool(torch.any((self.atom_index < 0) | (self.atom_index >= self.num_atoms)))
        ):
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH", "candidate state edge index out of range"
            )
        pair_key = self.site_index * max(self.num_atoms, 1) + self.atom_index
        if edges and bool(torch.any(pair_key[1:] <= pair_key[:-1])):
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH",
                "candidate state pairs are not unique site-major",
            )
        if (
            not math.isfinite(self.r_off)
            or not math.isfinite(self.r_candidate)
            or not math.isfinite(self.skin)
            or self.r_off <= 0.0
            or self.r_candidate <= self.r_off
            or not math.isclose(
                self.skin,
                self.r_candidate - self.r_off,
                rel_tol=0.0,
                abs_tol=8.0 * math.ulp(max(self.r_candidate, self.r_off)),
            )
        ):
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH", "candidate state radii are inconsistent"
            )
        if self.build_generation < 1 or self.reuse_count < 0:
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH", "candidate state counters are invalid"
            )
        _validate_image_range(self.image_range)
        expected_pairs = _pair_set_fingerprint(
            self.num_sites,
            self.num_atoms,
            self.site_index,
            self.atom_index,
        )
        if expected_pairs != self.candidate_pair_set_fingerprint:
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH",
                "candidate pair-set fingerprint mismatch",
            )
        if self.build_diagnostics is not None and (
            self.build_diagnostics.candidate_edge_count != edges
            or self.build_diagnostics.candidate_fingerprint
            != self.build_candidate_fingerprint
        ):
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH",
                "candidate state build diagnostics mismatch",
            )
        actual = self._compute_integrity_fingerprint()
        if self.integrity_fingerprint is None:
            object.__setattr__(self, "integrity_fingerprint", actual)
        elif self.integrity_fingerprint != actual:
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH",
                "candidate state integrity fingerprint mismatch",
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.build_positions.dtype

    @property
    def device(self) -> torch.device:
        return self.build_positions.device

    @property
    def candidate_count(self) -> int:
        return int(self.site_index.numel())

    def _compute_integrity_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.schema_version,
            self.template_fingerprint,
            self.support_content_fingerprint,
            self.phase_site_branch_fingerprint,
            self.atom_order_fingerprint,
            self.candidate_pair_set_fingerprint,
            self.build_candidate_fingerprint,
            self.num_sites,
            self.num_atoms,
            self.r_off,
            self.r_candidate,
            self.skin,
            self.build_generation,
            self.reuse_count,
            self.image_range,
        ):
            _hash_text(digest, value)
        for tensor in (
            self.site_index,
            self.atom_index,
            self.build_periodic_shift,
            self.build_positions,
            self.build_reference_sites,
            self.build_cell,
            self.build_origin,
            self.build_atomic_numbers,
            self.build_atom_order_identity,
        ):
            _hash_tensor(digest, tensor)
        return digest.hexdigest()

    def validate_integrity(self) -> None:
        if self.schema_version != CANDIDATE_NEIGHBOR_STATE_SCHEMA_VERSION:
            raise TransportSupportError(
                "STATE_SCHEMA_MISMATCH",
                f"unsupported candidate state schema {self.schema_version!r}",
            )
        if self.integrity_fingerprint != self._compute_integrity_fingerprint():
            raise TransportSupportError(
                "STATE_INTEGRITY_MISMATCH",
                "candidate state tensor or metadata was mutated",
            )

    def matches_support_config(self, config: TransportSupportConfig) -> bool:
        """Return whether ``config`` has the same physical support content.

        Candidate block sizes are execution details and intentionally do not
        participate.  Consumers can therefore reject a stale physical binding
        before doing a fresh traversal while still permitting block-size
        changes during reuse.
        """

        self.validate_integrity()
        _validate_support_config(config)
        return self.support_content_fingerprint == _support_content_fingerprint(
            config
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "CompactCandidateNeighborState":
        self.validate_integrity()
        target_dtype = self.dtype if dtype is None else dtype
        if target_dtype not in (torch.float32, torch.float64):
            raise TransportSupportError(
                "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
                "candidate state materialization supports float32 and float64",
            )
        target_device = self.device if device is None else torch.device(device)
        if target_device.type not in ("cpu", "cuda"):
            raise TransportSupportError(
                "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
                "candidate state materialization supports CPU and CUDA",
            )
        floating_names = (
            "build_positions",
            "build_reference_sites",
            "build_cell",
            "build_origin",
        )
        integer_names = (
            "site_index",
            "atom_index",
            "build_periodic_shift",
            "build_atomic_numbers",
            "build_atom_order_identity",
        )
        values = {
            name: getattr(self, name).to(device=target_device, dtype=target_dtype)
            for name in floating_names
        }
        values.update(
            {
                name: getattr(self, name).to(device=target_device)
                for name in integer_names
            }
        )
        values["integrity_fingerprint"] = None
        return replace(self, **values)


@dataclass(frozen=True)
class CandidateNeighborUpdate:
    state: CompactCandidateNeighborState
    decision: CandidateReuseDecision
    edges: CompactTransportEdges


def _state_from_edges(
    edges: CompactTransportEdges,
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    origin: torch.Tensor,
    atomic_numbers: torch.Tensor,
    atom_order_identity: torch.Tensor,
    *,
    template_fingerprint: str,
    phase_site_branch_fingerprint: str,
    config: TransportSupportConfig,
    build_generation: int,
    image_range: int,
) -> CompactCandidateNeighborState:
    diagnostics = edges.support_diagnostics
    candidate_fingerprint = diagnostics.candidate_fingerprint
    if candidate_fingerprint is None:
        raise TransportSupportError(
            "STATE_INTEGRITY_MISMATCH",
            "fresh candidate build did not provide a candidate fingerprint",
        )
    return CompactCandidateNeighborState(
        template_fingerprint=template_fingerprint,
        support_content_fingerprint=_support_content_fingerprint(config),
        phase_site_branch_fingerprint=phase_site_branch_fingerprint,
        num_sites=edges.num_sites,
        num_atoms=edges.num_atoms,
        atom_order_fingerprint=_atom_order_fingerprint(
            atomic_numbers, atom_order_identity
        ),
        site_index=edges.site_index,
        atom_index=edges.atom_index,
        build_periodic_shift=edges.periodic_shift,
        build_positions=positions,
        build_reference_sites=reference_sites,
        build_cell=cell,
        build_origin=origin,
        build_atomic_numbers=atomic_numbers,
        build_atom_order_identity=atom_order_identity,
        r_off=config.cutoff,
        r_candidate=config.r_candidate,
        skin=config.candidate_skin,
        candidate_pair_set_fingerprint=_pair_set_fingerprint(
            edges.num_sites, edges.num_atoms, edges.site_index, edges.atom_index
        ),
        build_candidate_fingerprint=candidate_fingerprint,
        build_generation=int(build_generation),
        reuse_count=0,
        image_range=image_range,
        build_diagnostics=diagnostics,
    )


def _decision_for_build(
    state: CompactCandidateNeighborState,
    edges: CompactTransportEdges,
    *,
    reason_code: str,
    numerical_guard: float,
    delta_atom: float | None,
    delta_site: float | None,
    delta_pair_bound: float | None,
    fresh_retry_performed: bool,
    fresh_retry_reason: str | None,
    state_materialized: bool,
    previous_state: CompactCandidateNeighborState | None = None,
    atom_mic_image_gap: float = math.inf,
    site_mic_image_gap: float = math.inf,
) -> CandidateReuseDecision:
    diagnostics = edges.support_diagnostics
    remaining = (
        None
        if delta_pair_bound is None
        else state.skin - delta_pair_bound - numerical_guard
    )
    live_fingerprint = _active_support_fingerprint(edges)
    return CandidateReuseDecision(
        reused=False,
        rebuilt=True,
        reason_code=reason_code,
        build_generation=state.build_generation,
        build_count=state.build_generation,
        reuse_count=0,
        delta_atom=delta_atom,
        delta_site=delta_site,
        delta_pair_bound=delta_pair_bound,
        numerical_guard=numerical_guard,
        numerical_guard_formula=_GUARD_FORMULA,
        skin=state.skin,
        remaining_skin=remaining,
        cached_candidate_count=(
            state.candidate_count
            if previous_state is None
            else previous_state.candidate_count
        ),
        current_active_count=edges.num_active_edges,
        candidate_boundary_lower_bound=diagnostics.candidate_boundary_gap,
        cached_pair_set_fingerprint=(
            state.candidate_pair_set_fingerprint
            if previous_state is None
            else previous_state.candidate_pair_set_fingerprint
        ),
        current_live_support_fingerprint=live_fingerprint,
        fresh_candidate_fingerprint=diagnostics.candidate_fingerprint,
        atom_mic_image_gap=atom_mic_image_gap,
        site_mic_image_gap=site_mic_image_gap,
        fresh_retry_performed=fresh_retry_performed,
        fresh_retry_reason=fresh_retry_reason,
        state_materialized=state_materialized,
        processed_block_count=diagnostics.processed_block_count,
        maximum_pair_block_elements=diagnostics.maximum_pair_block_elements,
        dense_allocation_observed=diagnostics.dense_candidate_allocation_observed,
    )


def _fresh_build(
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    pbc: tuple[bool, bool, bool],
    origin: torch.Tensor,
    atomic_numbers: torch.Tensor,
    atom_order_identity: torch.Tensor,
    *,
    template_fingerprint: str,
    phase_site_branch_fingerprint: str,
    epsilon_ot: float,
    ell_ot: float,
    config: TransportSupportConfig,
    image_range: int,
    minimum_mic_image_gap: float | None,
    minimum_candidate_boundary_gap: Real | None,
    sample_id: str | None,
    build_generation: int,
    reason_code: str,
    numerical_guard: float,
    delta_atom: float | None = None,
    delta_site: float | None = None,
    delta_pair_bound: float | None = None,
    fresh_retry_performed: bool = False,
    fresh_retry_reason: str | None = None,
    state_materialized: bool = False,
    previous_state: CompactCandidateNeighborState | None = None,
    atom_mic_image_gap: float = math.inf,
    site_mic_image_gap: float = math.inf,
) -> CandidateNeighborUpdate:
    edges = build_periodic_compact_transport_edges(
        positions,
        reference_sites,
        cell,
        pbc,
        origin=origin,
        epsilon_ot=epsilon_ot,
        ell_ot=ell_ot,
        config=config,
        image_range=image_range,
        minimum_mic_image_gap=minimum_mic_image_gap,
        minimum_candidate_boundary_gap=minimum_candidate_boundary_gap,
        template_id=template_fingerprint,
        sample_id=sample_id,
    )
    state = _state_from_edges(
        edges,
        positions,
        reference_sites,
        cell,
        origin,
        atomic_numbers,
        atom_order_identity,
        template_fingerprint=template_fingerprint,
        phase_site_branch_fingerprint=phase_site_branch_fingerprint,
        config=config,
        build_generation=build_generation,
        image_range=image_range,
    )
    decision = _decision_for_build(
        state,
        edges,
        reason_code=reason_code,
        numerical_guard=numerical_guard,
        delta_atom=delta_atom,
        delta_site=delta_site,
        delta_pair_bound=delta_pair_bound,
        fresh_retry_performed=fresh_retry_performed,
        fresh_retry_reason=fresh_retry_reason,
        state_materialized=state_materialized,
        previous_state=previous_state,
        atom_mic_image_gap=atom_mic_image_gap,
        site_mic_image_gap=site_mic_image_gap,
    )
    return CandidateNeighborUpdate(state=state, decision=decision, edges=edges)


def build_candidate_neighbor_state(
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    pbc: Sequence[bool] | torch.Tensor,
    *,
    origin: torch.Tensor | None = None,
    atomic_numbers: torch.Tensor,
    atom_order_identity: torch.Tensor | None = None,
    template_fingerprint: str,
    phase_site_branch_fingerprint: str | None = None,
    epsilon_ot: float,
    ell_ot: float,
    config: TransportSupportConfig,
    numerical_guard: Real | None = None,
    image_range: int = 2,
    minimum_mic_image_gap: Real | None = None,
    minimum_candidate_boundary_gap: Real | None = None,
    sample_id: str | None = None,
) -> CandidateNeighborUpdate:
    """Build generation one of an explicit, caller-owned neighbor state."""

    _validate_support_config(config)
    template_fingerprint = _validate_fingerprint(
        template_fingerprint, "template_fingerprint"
    )
    phase_fingerprint = _validate_fingerprint(
        template_fingerprint
        if phase_site_branch_fingerprint is None
        else phase_site_branch_fingerprint,
        "phase_site_branch_fingerprint",
    )
    pbc_tuple = _full_pbc(
        pbc,
        template_fingerprint=template_fingerprint,
        sample_id=sample_id,
    )
    image_range = _validate_image_range(image_range)
    mic_threshold = _validate_threshold(
        minimum_mic_image_gap, "minimum_mic_image_gap"
    )
    origin, atom_identity = _validate_current_geometry(
        positions,
        reference_sites,
        cell,
        origin,
        atomic_numbers,
        atom_order_identity,
        template_fingerprint=template_fingerprint,
        sample_id=sample_id,
    )
    guard = _effective_numerical_guard(
        positions.dtype,
        cell,
        config.r_candidate,
        int(reference_sites.shape[0]),
        int(positions.shape[0]),
        numerical_guard,
    )
    if guard >= config.candidate_skin:
        raise _error(
            "INVALID_SUPPORT_CONFIG",
            f"effective numerical guard {guard:.17g} exhausts skin "
            f"{config.candidate_skin:.17g}",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    return _fresh_build(
        positions,
        reference_sites,
        cell,
        pbc_tuple,
        origin,
        atomic_numbers,
        atom_identity,
        template_fingerprint=template_fingerprint,
        phase_site_branch_fingerprint=phase_fingerprint,
        epsilon_ot=epsilon_ot,
        ell_ot=ell_ot,
        config=config,
        image_range=image_range,
        minimum_mic_image_gap=mic_threshold,
        minimum_candidate_boundary_gap=minimum_candidate_boundary_gap,
        sample_id=sample_id,
        build_generation=1,
        reason_code="INITIAL_BUILD",
        numerical_guard=guard,
    )


def _movement_bound(
    state: CompactCandidateNeighborState,
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    origin: torch.Tensor,
    pbc: tuple[bool, bool, bool],
    *,
    image_range: int,
    minimum_mic_image_gap: float | None,
    sample_id: str | None,
) -> tuple[float, float, float, float]:
    with torch.no_grad():
        atom_delta = (positions.detach() - origin.detach()) - (
            state.build_positions - state.build_origin
        )
        site_delta = (reference_sites.detach() - origin.detach()) - (
            state.build_reference_sites - state.build_origin
        )
        try:
            atom_mic = minimum_image_diagnostics(
                atom_delta, cell.detach(), pbc, image_range=image_range
            )
            site_mic = minimum_image_diagnostics(
                site_delta, cell.detach(), pbc, image_range=image_range
            )
        except ValueError as error:
            raise _error(
                "MIC_AMBIGUITY",
                f"candidate-state movement MIC failed: {error}",
                template_fingerprint=state.template_fingerprint,
                sample_id=sample_id,
            ) from error

        def maximum_and_gap(diagnostics: Any) -> tuple[float, float]:
            if diagnostics.nearest_distance.numel() == 0:
                return 0.0, math.inf
            return (
                float(diagnostics.nearest_distance.max().cpu()),
                float(diagnostics.unique_image_gap.min().cpu()),
            )

        delta_atom, atom_gap = maximum_and_gap(atom_mic)
        delta_site, site_gap = maximum_and_gap(site_mic)
    if minimum_mic_image_gap is not None and (
        atom_gap <= minimum_mic_image_gap or site_gap <= minimum_mic_image_gap
    ):
        raise _error(
            "MIC_AMBIGUITY",
            "build/current identity MIC is not uniquely certified",
            template_fingerprint=state.template_fingerprint,
            sample_id=sample_id,
        )
    return delta_atom, delta_site, atom_gap, site_gap


def _materialize_cached_edges(
    state: CompactCandidateNeighborState,
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    pbc: tuple[bool, bool, bool],
    *,
    epsilon_ot: float,
    ell_ot: float,
    config: TransportSupportConfig,
    image_range: int,
    minimum_mic_image_gap: float | None,
    template_fingerprint: str,
    sample_id: str | None,
) -> tuple[CompactTransportEdges, float]:
    site_index = state.site_index
    atom_index = state.atom_index
    raw = positions[atom_index] - reference_sites[site_index]
    with torch.no_grad():
        try:
            image = minimum_image_diagnostics(
                raw.detach(), cell.detach(), pbc, image_range=image_range
            )
        except ValueError as error:
            raise _error(
                "MIC_AMBIGUITY",
                f"cached-pair MIC selection failed: {error}",
                template_fingerprint=template_fingerprint,
                sample_id=sample_id,
            ) from error
        assert image.periodic_shift is not None
        current_shift = image.periodic_shift.to(device=positions.device)
        mic_gap = (
            math.inf
            if image.unique_image_gap.numel() == 0
            else float(image.unique_image_gap.min().cpu())
        )
    if minimum_mic_image_gap is not None and mic_gap <= minimum_mic_image_gap:
        raise _error(
            "MIC_AMBIGUITY",
            "cached live edge MIC is not uniquely certified",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    if raw.numel():
        fractional = torch.linalg.solve(cell.T, raw.T).T
        displacements = (
            fractional - current_shift.to(dtype=raw.dtype)
        ) @ cell
    else:
        displacements = raw
    distances = torch.linalg.vector_norm(displacements, dim=-1)
    switch = compact_c2_switch(distances, config)
    active = distances < distances.new_tensor(config.cutoff)
    live_fingerprint = _live_support_fingerprint(
        state.num_sites,
        state.num_atoms,
        site_index,
        atom_index,
        current_shift,
        active,
    )
    _, diagnostics = validate_compact_support_edges(
        site_index,
        atom_index,
        distances,
        switch,
        state.num_sites,
        state.num_atoms,
        config,
        template_id=template_fingerprint,
        sample_id=sample_id,
        mic_image_gap=mic_gap,
        maximum_mic_image_count=image.maximum_image_count,
        candidate_fingerprint=live_fingerprint,
        processed_block_count=0,
        maximum_pair_block_elements=0,
        peak_temporary_geometry_elements=(
            state.candidate_count * (4 + 10 * image.maximum_image_count)
        ),
        enforce_candidate_radius=False,
    )
    edges = build_compact_transport_edges_from_candidates(
        site_index,
        atom_index,
        current_shift,
        displacements,
        num_sites=state.num_sites,
        num_atoms=state.num_atoms,
        epsilon_ot=epsilon_ot,
        ell_ot=ell_ot,
        config=config,
        support_diagnostics=diagnostics,
        template_id=template_fingerprint,
        sample_id=sample_id,
    )
    return edges, mic_gap


def update_candidate_neighbor_state(
    previous_state: CompactCandidateNeighborState | None,
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    pbc: Sequence[bool] | torch.Tensor,
    *,
    origin: torch.Tensor | None = None,
    atomic_numbers: torch.Tensor,
    atom_order_identity: torch.Tensor | None = None,
    template_fingerprint: str,
    phase_site_branch_fingerprint: str | None = None,
    epsilon_ot: float,
    ell_ot: float,
    config: TransportSupportConfig,
    numerical_guard: Real | None = None,
    image_range: int = 2,
    minimum_mic_image_gap: Real | None = None,
    minimum_candidate_boundary_gap: Real | None = None,
    explicit_rebuild: bool = False,
    sample_id: str | None = None,
) -> CandidateNeighborUpdate:
    """Reuse a certified pair set or return a fresh immutable generation.

    Reuse is permitted only when

    ``delta_atom + delta_site + numerical_guard < r_candidate - r_off``.

    Movement is always measured from the original build snapshot, never from
    the previous call, so incremental updates cannot hide cumulative motion.
    """

    if previous_state is None:
        return build_candidate_neighbor_state(
            positions,
            reference_sites,
            cell,
            pbc,
            origin=origin,
            atomic_numbers=atomic_numbers,
            atom_order_identity=atom_order_identity,
            template_fingerprint=template_fingerprint,
            phase_site_branch_fingerprint=phase_site_branch_fingerprint,
            epsilon_ot=epsilon_ot,
            ell_ot=ell_ot,
            config=config,
            numerical_guard=numerical_guard,
            image_range=image_range,
            minimum_mic_image_gap=minimum_mic_image_gap,
            minimum_candidate_boundary_gap=minimum_candidate_boundary_gap,
            sample_id=sample_id,
        )
    if not isinstance(previous_state, CompactCandidateNeighborState):
        raise TypeError("previous_state must be CompactCandidateNeighborState or None")
    previous_state.validate_integrity()
    if not isinstance(explicit_rebuild, bool):
        raise TypeError("explicit_rebuild must be bool")
    _validate_support_config(config)
    template_fingerprint = _validate_fingerprint(
        template_fingerprint, "template_fingerprint"
    )
    phase_fingerprint = _validate_fingerprint(
        template_fingerprint
        if phase_site_branch_fingerprint is None
        else phase_site_branch_fingerprint,
        "phase_site_branch_fingerprint",
    )
    pbc_tuple = _full_pbc(
        pbc,
        template_fingerprint=template_fingerprint,
        sample_id=sample_id,
    )
    image_range = _validate_image_range(image_range)
    mic_threshold = _validate_threshold(
        minimum_mic_image_gap, "minimum_mic_image_gap"
    )
    candidate_threshold = _validate_threshold(
        minimum_candidate_boundary_gap,
        "minimum_candidate_boundary_gap",
    )
    origin, atom_identity = _validate_current_geometry(
        positions,
        reference_sites,
        cell,
        origin,
        atomic_numbers,
        atom_order_identity,
        template_fingerprint=template_fingerprint,
        sample_id=sample_id,
    )
    guard = _effective_numerical_guard(
        positions.dtype,
        cell,
        config.r_candidate,
        int(reference_sites.shape[0]),
        int(positions.shape[0]),
        numerical_guard,
    )
    if guard >= config.candidate_skin:
        raise _error(
            "INVALID_SUPPORT_CONFIG",
            f"effective numerical guard {guard:.17g} exhausts skin "
            f"{config.candidate_skin:.17g}",
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )

    state_materialized = (
        previous_state.device != positions.device
        or previous_state.dtype != positions.dtype
    )
    state = (
        previous_state.to(device=positions.device, dtype=positions.dtype)
        if state_materialized
        else previous_state
    )

    rebuild_reason: str | None = None
    if state.template_fingerprint != template_fingerprint:
        rebuild_reason = "TEMPLATE_MISMATCH"
    elif state.support_content_fingerprint != _support_content_fingerprint(config):
        rebuild_reason = "SUPPORT_CONFIG_MISMATCH"
    elif state.num_atoms != int(positions.shape[0]):
        rebuild_reason = "ATOM_COUNT_CHANGED"
    elif state.num_sites != int(reference_sites.shape[0]):
        rebuild_reason = "PHASE_SITE_BRANCH_CHANGED"
    elif state.atom_order_fingerprint != _atom_order_fingerprint(
        atomic_numbers, atom_identity
    ):
        rebuild_reason = "ATOM_ORDER_CHANGED"
    elif state.phase_site_branch_fingerprint != phase_fingerprint:
        rebuild_reason = "PHASE_SITE_BRANCH_CHANGED"
    elif state.image_range != image_range:
        rebuild_reason = "SUPPORT_CONFIG_MISMATCH"
    elif not torch.equal(state.build_cell, cell):
        rebuild_reason = "CELL_CHANGED"
    elif explicit_rebuild:
        rebuild_reason = "EXPLICIT_REBUILD"

    if rebuild_reason is not None:
        return _fresh_build(
            positions,
            reference_sites,
            cell,
            pbc_tuple,
            origin,
            atomic_numbers,
            atom_identity,
            template_fingerprint=template_fingerprint,
            phase_site_branch_fingerprint=phase_fingerprint,
            epsilon_ot=epsilon_ot,
            ell_ot=ell_ot,
            config=config,
            image_range=image_range,
            minimum_mic_image_gap=mic_threshold,
            minimum_candidate_boundary_gap=candidate_threshold,
            sample_id=sample_id,
            build_generation=state.build_generation + 1,
            reason_code=rebuild_reason,
            numerical_guard=guard,
            state_materialized=state_materialized,
            previous_state=state,
        )

    delta_atom, delta_site, atom_gap, site_gap = _movement_bound(
        state,
        positions,
        reference_sites,
        cell,
        origin,
        pbc_tuple,
        image_range=image_range,
        minimum_mic_image_gap=mic_threshold,
        sample_id=sample_id,
    )
    delta_pair = delta_atom + delta_site
    remaining_skin = state.skin - delta_pair - guard
    if not remaining_skin > 0.0:
        return _fresh_build(
            positions,
            reference_sites,
            cell,
            pbc_tuple,
            origin,
            atomic_numbers,
            atom_identity,
            template_fingerprint=template_fingerprint,
            phase_site_branch_fingerprint=phase_fingerprint,
            epsilon_ot=epsilon_ot,
            ell_ot=ell_ot,
            config=config,
            image_range=image_range,
            minimum_mic_image_gap=mic_threshold,
            minimum_candidate_boundary_gap=candidate_threshold,
            sample_id=sample_id,
            build_generation=state.build_generation + 1,
            reason_code="SKIN_EXHAUSTED",
            numerical_guard=guard,
            delta_atom=delta_atom,
            delta_site=delta_site,
            delta_pair_bound=delta_pair,
            state_materialized=state_materialized,
            previous_state=state,
            atom_mic_image_gap=atom_gap,
            site_mic_image_gap=site_gap,
        )

    candidate_boundary_lower_bound = max(
        0.0,
        (
            state.build_diagnostics.candidate_boundary_gap
            if state.build_diagnostics is not None
            else 0.0
        )
        - delta_pair
        - guard,
    )
    if (
        candidate_threshold is not None
        and candidate_boundary_lower_bound <= candidate_threshold
    ):
        return _fresh_build(
            positions,
            reference_sites,
            cell,
            pbc_tuple,
            origin,
            atomic_numbers,
            atom_identity,
            template_fingerprint=template_fingerprint,
            phase_site_branch_fingerprint=phase_fingerprint,
            epsilon_ot=epsilon_ot,
            ell_ot=ell_ot,
            config=config,
            image_range=image_range,
            minimum_mic_image_gap=mic_threshold,
            minimum_candidate_boundary_gap=candidate_threshold,
            sample_id=sample_id,
            build_generation=state.build_generation + 1,
            reason_code="CANDIDATE_CERTIFICATE_EXHAUSTED",
            numerical_guard=guard,
            delta_atom=delta_atom,
            delta_site=delta_site,
            delta_pair_bound=delta_pair,
            state_materialized=state_materialized,
            previous_state=state,
            atom_mic_image_gap=atom_gap,
            site_mic_image_gap=site_gap,
        )

    try:
        edges, live_mic_gap = _materialize_cached_edges(
            state,
            positions,
            reference_sites,
            cell,
            pbc_tuple,
            epsilon_ot=epsilon_ot,
            ell_ot=ell_ot,
            config=config,
            image_range=image_range,
            minimum_mic_image_gap=mic_threshold,
            template_fingerprint=template_fingerprint,
            sample_id=sample_id,
        )
    except TransportSupportError as cached_error:
        try:
            return _fresh_build(
                positions,
                reference_sites,
                cell,
                pbc_tuple,
                origin,
                atomic_numbers,
                atom_identity,
                template_fingerprint=template_fingerprint,
                phase_site_branch_fingerprint=phase_fingerprint,
                epsilon_ot=epsilon_ot,
                ell_ot=ell_ot,
                config=config,
                image_range=image_range,
                minimum_mic_image_gap=mic_threshold,
                minimum_candidate_boundary_gap=candidate_threshold,
                sample_id=sample_id,
                build_generation=state.build_generation + 1,
                reason_code="REUSE_FEASIBILITY_RETRY",
                numerical_guard=guard,
                delta_atom=delta_atom,
                delta_site=delta_site,
                delta_pair_bound=delta_pair,
                fresh_retry_performed=True,
                fresh_retry_reason=cached_error.reason_code,
                state_materialized=state_materialized,
                previous_state=state,
                atom_mic_image_gap=atom_gap,
                site_mic_image_gap=site_gap,
            )
        except TransportSupportError as fresh_error:
            raise _error(
                fresh_error.reason_code,
                "cached support failed with "
                f"{cached_error.reason_code}; fresh rebuild also failed: {fresh_error}",
                template_fingerprint=template_fingerprint,
                sample_id=sample_id,
            ) from fresh_error

    next_state = replace(
        state,
        reuse_count=state.reuse_count + 1,
        integrity_fingerprint=None,
    )
    live_fingerprint = _active_support_fingerprint(edges)
    decision = CandidateReuseDecision(
        reused=True,
        rebuilt=False,
        reason_code=(
            "STATE_DEVICE_MATERIALIZATION" if state_materialized else "REUSED"
        ),
        build_generation=next_state.build_generation,
        build_count=next_state.build_generation,
        reuse_count=next_state.reuse_count,
        delta_atom=delta_atom,
        delta_site=delta_site,
        delta_pair_bound=delta_pair,
        numerical_guard=guard,
        numerical_guard_formula=_GUARD_FORMULA,
        skin=next_state.skin,
        remaining_skin=remaining_skin,
        cached_candidate_count=next_state.candidate_count,
        current_active_count=edges.num_active_edges,
        candidate_boundary_lower_bound=candidate_boundary_lower_bound,
        cached_pair_set_fingerprint=next_state.candidate_pair_set_fingerprint,
        current_live_support_fingerprint=live_fingerprint,
        fresh_candidate_fingerprint=None,
        atom_mic_image_gap=atom_gap,
        site_mic_image_gap=site_gap,
        fresh_retry_performed=False,
        fresh_retry_reason=None,
        state_materialized=state_materialized,
        processed_block_count=0,
        maximum_pair_block_elements=0,
        dense_allocation_observed=False,
    )
    # live_mic_gap is kept in the standard support diagnostics; movement MIC
    # gaps above separately certify build/current identity.
    assert math.isfinite(live_mic_gap) or math.isinf(live_mic_gap)
    return CandidateNeighborUpdate(state=next_state, decision=decision, edges=edges)


__all__ = [
    "CANDIDATE_NEIGHBOR_STATE_SCHEMA_VERSION",
    "CandidateNeighborUpdate",
    "CandidateReuseDecision",
    "CompactCandidateNeighborState",
    "build_candidate_neighbor_state",
    "update_candidate_neighbor_state",
]
