"""Stateless bounded-memory periodic compact-transport candidate extraction."""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from typing import Sequence

import torch

from .cost import minimum_image_diagnostics
from .edge_list import (
    CompactTransportEdges,
    build_compact_transport_edges_from_candidates,
)
from .support import (
    TransportSupportConfig,
    TransportSupportError,
    compact_c2_switch,
    validate_compact_support_edges,
)


def _context_error(
    reason_code: str,
    message: str,
    *,
    template_id: str | None,
    sample_id: str | None,
) -> TransportSupportError:
    return TransportSupportError(
        reason_code,
        message,
        template_id=template_id,
        sample_id=sample_id,
    )


def _validate_threshold(
    value: Real | None,
    name: str,
    *,
    template_id: str | None,
    sample_id: str | None,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise _context_error(
            "INVALID_SUPPORT_CONFIG",
            f"{name} must be finite and nonnegative",
            template_id=template_id,
            sample_id=sample_id,
        )
    return float(value)


def _full_pbc_tuple(
    pbc: Sequence[bool] | torch.Tensor,
    *,
    template_id: str | None,
    sample_id: str | None,
) -> tuple[bool, bool, bool]:
    if isinstance(pbc, torch.Tensor):
        if pbc.shape != (3,) or pbc.dtype != torch.bool:
            raise _context_error(
                "UNSUPPORTED_PBC",
                "pbc tensor must be bool [3]",
                template_id=template_id,
                sample_id=sample_id,
            )
        values = pbc.detach().cpu().tolist()
    else:
        values = list(pbc)
    if len(values) != 3 or any(not isinstance(value, bool) for value in values):
        raise _context_error(
            "UNSUPPORTED_PBC",
            "pbc must contain exactly three booleans",
            template_id=template_id,
            sample_id=sample_id,
        )
    result = tuple(values)
    if result != (True, True, True):
        raise _context_error(
            "UNSUPPORTED_PBC",
            "blocked periodic candidate extraction requires full PBC",
            template_id=template_id,
            sample_id=sample_id,
        )
    return result


def _candidate_fingerprint(
    num_sites: int,
    num_atoms: int,
    site_index: torch.Tensor,
    atom_index: torch.Tensor,
    periodic_shift: torch.Tensor,
    active: torch.Tensor,
) -> str:
    """Hash support content, deliberately excluding execution block sizes."""

    payload = (
        num_sites,
        num_atoms,
        tuple(int(value) for value in site_index.detach().cpu().tolist()),
        tuple(int(value) for value in atom_index.detach().cpu().tolist()),
        tuple(
            tuple(int(component) for component in row)
            for row in periodic_shift.detach().cpu().tolist()
        ),
        tuple(bool(value) for value in active.detach().cpu().tolist()),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _live_selected_displacements(
    positions: torch.Tensor,
    references: torch.Tensor,
    cell: torch.Tensor,
    site_index: torch.Tensor,
    atom_index: torch.Tensor,
    periodic_shift: torch.Tensor,
) -> torch.Tensor:
    """Re-evaluate selected MIC branches on the live geometry graph."""

    if site_index.numel() == 0:
        return positions.new_empty((0, 3))
    raw = positions[atom_index] - references[site_index]
    fractional = torch.linalg.solve(cell.T, raw.T).T
    return (fractional - periodic_shift.to(dtype=raw.dtype)) @ cell


def build_periodic_compact_transport_edges(
    positions: torch.Tensor,
    reference_sites: torch.Tensor,
    cell: torch.Tensor,
    pbc: Sequence[bool] | torch.Tensor,
    *,
    origin: torch.Tensor | None = None,
    reference_coordinates: str = "cartesian",
    epsilon_ot: float,
    ell_ot: float,
    config: TransportSupportConfig,
    image_range: int = 2,
    minimum_mic_image_gap: Real | None = None,
    minimum_candidate_boundary_gap: Real | None = None,
    template_id: str | None = None,
    sample_id: str | None = None,
) -> CompactTransportEdges:
    """Build canonical compact edges with dense or bounded blockwise search.

    Candidate search and MIC image selection are discrete control flow.  Only
    candidate indices and integer shifts survive that search; displacements,
    distances, switches, and kernels are then recomputed from the live input
    tensors.  With ``candidate_backend='blocked'`` no full ``[M,N,3]`` or
    ``[M,N]`` pair tensor is constructed.
    """

    if not isinstance(config, TransportSupportConfig) or (
        config.kind != "compact_c2" or config.backend != "edge_list"
    ):
        raise _context_error(
            "INVALID_SUPPORT_CONFIG",
            "periodic edge extraction requires compact_c2 edge_list config",
            template_id=template_id,
            sample_id=sample_id,
        )
    _full_pbc_tuple(
        pbc, template_id=template_id, sample_id=sample_id
    )
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise _context_error(
            "NONFINITE_SUPPORT_GEOMETRY",
            "positions must have shape [N,3]",
            template_id=template_id,
            sample_id=sample_id,
        )
    if reference_sites.ndim != 2 or reference_sites.shape[1:] != (3,):
        raise _context_error(
            "NONFINITE_SUPPORT_GEOMETRY",
            "reference_sites must have shape [M,3]",
            template_id=template_id,
            sample_id=sample_id,
        )
    if cell.shape != (3, 3):
        raise _context_error(
            "SINGULAR_CELL",
            "cell must have shape [3,3]",
            template_id=template_id,
            sample_id=sample_id,
        )
    if positions.dtype not in (torch.float32, torch.float64):
        raise _context_error(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "candidate geometry supports float32 and float64",
            template_id=template_id,
            sample_id=sample_id,
        )
    if (
        reference_sites.dtype != positions.dtype
        or cell.dtype != positions.dtype
        or reference_sites.device != positions.device
        or cell.device != positions.device
    ):
        raise _context_error(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "positions, references, and cell must share dtype/device",
            template_id=template_id,
            sample_id=sample_id,
        )
    if positions.device.type not in ("cpu", "cuda"):
        raise _context_error(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            f"unsupported candidate device {positions.device.type!r}",
            template_id=template_id,
            sample_id=sample_id,
        )
    if not bool(torch.all(torch.isfinite(positions)).detach()) or not bool(
        torch.all(torch.isfinite(reference_sites)).detach()
    ) or not bool(torch.all(torch.isfinite(cell)).detach()):
        raise _context_error(
            "NONFINITE_SUPPORT_GEOMETRY",
            "candidate geometry contains NaN or Inf",
            template_id=template_id,
            sample_id=sample_id,
        )
    determinant = torch.linalg.det(cell)
    if not bool(torch.isfinite(determinant).detach()) or float(
        determinant.abs().detach().cpu()
    ) <= torch.finfo(cell.dtype).eps:
        raise _context_error(
            "SINGULAR_CELL",
            "physical cell is singular",
            template_id=template_id,
            sample_id=sample_id,
        )
    if (
        isinstance(image_range, bool)
        or not isinstance(image_range, Integral)
        or int(image_range) < 1
    ):
        raise _context_error(
            "INVALID_SUPPORT_CONFIG",
            "image_range must be a positive integer",
            template_id=template_id,
            sample_id=sample_id,
        )
    mic_threshold = _validate_threshold(
        minimum_mic_image_gap,
        "minimum_mic_image_gap",
        template_id=template_id,
        sample_id=sample_id,
    )
    candidate_threshold = _validate_threshold(
        minimum_candidate_boundary_gap,
        "minimum_candidate_boundary_gap",
        template_id=template_id,
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
        raise _context_error(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "origin must be finite [3] and share geometry dtype/device",
            template_id=template_id,
            sample_id=sample_id,
        )
    if reference_coordinates == "fractional":
        references = origin + reference_sites @ cell
    elif reference_coordinates == "cartesian":
        references = reference_sites
    else:
        raise _context_error(
            "INVALID_SUPPORT_CONFIG",
            "reference_coordinates must be cartesian or fractional",
            template_id=template_id,
            sample_id=sample_id,
        )

    num_sites = int(references.shape[0])
    num_atoms = int(positions.shape[0])
    if num_sites <= 0 or num_atoms > num_sites:
        raise _context_error(
            "NO_TOTAL_SUPPORT",
            f"invalid transport dimensions M={num_sites}, N={num_atoms}",
            template_id=template_id,
            sample_id=sample_id,
        )
    if config.candidate_backend == "dense":
        site_block_size = num_sites
        atom_block_size = max(num_atoms, 1)
    else:
        site_block_size = config.site_block_size
        atom_block_size = config.atom_block_size

    site_parts: list[torch.Tensor] = []
    atom_parts: list[torch.Tensor] = []
    shift_parts: list[torch.Tensor] = []
    switch_on_gap = math.inf
    switch_off_gap = math.inf
    candidate_gap = math.inf
    mic_gap = math.inf
    processed_blocks = 0
    maximum_pair_block_elements = 0
    maximum_image_count = 1

    with torch.no_grad():
        control_positions = positions.detach()
        control_references = references.detach()
        control_cell = cell.detach()
        for site_start in range(0, num_sites, site_block_size):
            site_stop = min(site_start + site_block_size, num_sites)
            for atom_start in range(0, num_atoms, atom_block_size):
                atom_stop = min(atom_start + atom_block_size, num_atoms)
                raw = (
                    control_positions[atom_start:atom_stop].unsqueeze(0)
                    - control_references[site_start:site_stop].unsqueeze(1)
                )
                pair_elements = int(raw.shape[0] * raw.shape[1])
                processed_blocks += 1
                maximum_pair_block_elements = max(
                    maximum_pair_block_elements, pair_elements
                )
                try:
                    image = minimum_image_diagnostics(
                        raw,
                        control_cell,
                        (True, True, True),
                        image_range=int(image_range),
                    )
                except ValueError as error:
                    reason = (
                        "SINGULAR_CELL"
                        if "singular" in str(error).lower()
                        else "MIC_AMBIGUITY"
                        if "unique" in str(error).lower()
                        else "NONFINITE_SUPPORT_GEOMETRY"
                    )
                    raise _context_error(
                        reason,
                        f"MIC candidate search failed: {error}",
                        template_id=template_id,
                        sample_id=sample_id,
                    ) from error
                maximum_image_count = max(
                    maximum_image_count, int(image.maximum_image_count)
                )
                distances = torch.linalg.vector_norm(image.displacement, dim=-1)
                if pair_elements:
                    switch_on_gap = min(
                        switch_on_gap,
                        float(torch.min(torch.abs(distances - config.r_on)).cpu()),
                    )
                    switch_off_gap = min(
                        switch_off_gap,
                        float(torch.min(torch.abs(distances - config.cutoff)).cpu()),
                    )
                    candidate_gap = min(
                        candidate_gap,
                        float(
                            torch.min(
                                torch.abs(distances - config.r_candidate)
                            ).cpu()
                        ),
                    )
                    mic_gap = min(
                        mic_gap,
                        float(torch.min(image.unique_image_gap).cpu()),
                    )
                candidate = distances < distances.new_tensor(config.r_candidate)
                local = torch.nonzero(candidate, as_tuple=False)
                if local.numel():
                    site_parts.append(local[:, 0].to(torch.long) + site_start)
                    atom_parts.append(local[:, 1].to(torch.long) + atom_start)
                    assert image.periodic_shift is not None
                    shift_parts.append(
                        image.periodic_shift[local[:, 0], local[:, 1]].to(
                            torch.long
                        )
                    )

    if mic_threshold is not None and mic_gap <= mic_threshold:
        raise _context_error(
            "MIC_AMBIGUITY",
            f"minimum MIC image gap {mic_gap:.17g} <= {mic_threshold:.17g}",
            template_id=template_id,
            sample_id=sample_id,
        )
    if candidate_threshold is not None and candidate_gap <= candidate_threshold:
        raise _context_error(
            "CANDIDATE_BOUNDARY_INSTABILITY",
            f"candidate boundary gap {candidate_gap:.17g} <= {candidate_threshold:.17g}",
            template_id=template_id,
            sample_id=sample_id,
        )

    if site_parts:
        site_index = torch.cat(site_parts)
        atom_index = torch.cat(atom_parts)
        periodic_shift = torch.cat(shift_parts)
        pair_key = site_index * max(num_atoms, 1) + atom_index
        order = torch.argsort(pair_key, stable=True)
        site_index = site_index[order]
        atom_index = atom_index[order]
        periodic_shift = periodic_shift[order]
    else:
        site_index = torch.empty(0, dtype=torch.long, device=positions.device)
        atom_index = torch.empty(0, dtype=torch.long, device=positions.device)
        periodic_shift = torch.empty(
            (0, 3), dtype=torch.long, device=positions.device
        )

    live_displacements = _live_selected_displacements(
        positions,
        references,
        cell,
        site_index,
        atom_index,
        periodic_shift,
    )
    live_distances = torch.linalg.vector_norm(live_displacements, dim=-1)
    live_switch = compact_c2_switch(live_distances, config)
    live_active = live_distances < live_distances.new_tensor(config.cutoff)
    fingerprint = _candidate_fingerprint(
        num_sites,
        num_atoms,
        site_index,
        atom_index,
        periodic_shift,
        live_active,
    )
    # Conservative element upper bound for the current certified-MIC helper:
    # integer-shift candidates, fractional candidates, Cartesian candidates,
    # squared norms, and the base pair geometry coexist at its peak.
    peak_geometry = maximum_pair_block_elements * (
        4 + 10 * maximum_image_count
    )
    _, diagnostics = validate_compact_support_edges(
        site_index,
        atom_index,
        live_distances,
        live_switch,
        num_sites,
        num_atoms,
        config,
        template_id=template_id,
        sample_id=sample_id,
        cutoff_boundary_gap=switch_off_gap,
        switch_on_boundary_gap=switch_on_gap,
        candidate_boundary_gap=candidate_gap,
        mic_image_gap=mic_gap,
        maximum_mic_image_count=maximum_image_count,
        candidate_fingerprint=fingerprint,
        processed_block_count=processed_blocks,
        maximum_pair_block_elements=maximum_pair_block_elements,
        peak_temporary_geometry_elements=peak_geometry,
    )
    return build_compact_transport_edges_from_candidates(
        site_index,
        atom_index,
        periodic_shift,
        live_displacements,
        num_sites=num_sites,
        num_atoms=num_atoms,
        epsilon_ot=epsilon_ot,
        ell_ot=ell_ot,
        config=config,
        support_diagnostics=diagnostics,
        template_id=template_id,
        sample_id=sample_id,
    )
