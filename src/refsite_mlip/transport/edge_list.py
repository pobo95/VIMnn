"""Canonical candidate-edge representation for compact atom-site transport."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .result import DensePlanMaterialization, SparseOTResult
from .support import (
    TransportSupportConfig,
    TransportSupportDiagnostics,
    TransportSupportError,
    compact_c2_switch,
    validate_compact_support,
)


@dataclass(frozen=True)
class CompactTransportEdges:
    """Site-major candidate edges and atom-major reduction metadata.

    Candidate edges extend through ``r_candidate``.  Entries between ``r_off``
    and ``r_candidate`` are retained with an exactly zero switch and ``-inf``
    log kernel, while all transport arithmetic uses the positive-weight branch.
    """

    site_index: torch.Tensor
    atom_index: torch.Tensor
    displacements: torch.Tensor
    distances: torch.Tensor
    switch: torch.Tensor
    log_kernel: torch.Tensor
    active: torch.Tensor
    atom_major_permutation: torch.Tensor
    site_ptr: torch.Tensor
    atom_ptr: torch.Tensor
    num_sites: int
    num_atoms: int
    num_vacancies: int
    epsilon: torch.Tensor
    support_diagnostics: TransportSupportDiagnostics
    periodic_shift: torch.Tensor | None = None

    def __post_init__(self) -> None:
        edges = int(self.site_index.numel())
        if self.num_sites <= 0 or self.num_atoms < 0 or self.num_atoms > self.num_sites:
            raise ValueError("invalid compact edge-list site/atom counts")
        if self.num_vacancies != self.num_sites - self.num_atoms:
            raise ValueError("compact edge-list vacancy count is inconsistent")
        if self.site_index.shape != (edges,) or self.atom_index.shape != (edges,):
            raise ValueError("edge indices must have shape [E]")
        if self.site_index.dtype != torch.long or self.atom_index.dtype != torch.long:
            raise ValueError("edge indices must use torch.long")
        if self.periodic_shift is None:
            object.__setattr__(
                self,
                "periodic_shift",
                torch.zeros(
                    (edges, 3), dtype=torch.long, device=self.site_index.device
                ),
            )
        assert self.periodic_shift is not None
        if (
            self.periodic_shift.shape != (edges, 3)
            or self.periodic_shift.dtype != torch.long
            or self.periodic_shift.device != self.site_index.device
        ):
            raise ValueError("periodic_shift must be device-local long [E,3]")
        if self.active.shape != (edges,) or self.active.dtype != torch.bool:
            raise ValueError("active edge mask must be bool [E]")
        for name, value, shape in (
            ("displacements", self.displacements, (edges, 3)),
            ("distances", self.distances, (edges,)),
            ("switch", self.switch, (edges,)),
            ("log_kernel", self.log_kernel, (edges,)),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} has incorrect edge-list shape")
            if value.dtype != self.distances.dtype or value.device != self.distances.device:
                raise ValueError("edge floating tensors must share dtype/device")
        if self.distances.dtype not in (torch.float32, torch.float64):
            raise ValueError("edge floating tensors must use float32 or float64")
        if self.site_index.device != self.distances.device or self.atom_index.device != self.distances.device:
            raise ValueError("edge indices and values must share device")
        if edges and (
            bool(torch.any((self.site_index < 0) | (self.site_index >= self.num_sites)).detach())
            or bool(torch.any((self.atom_index < 0) | (self.atom_index >= self.num_atoms)).detach())
        ):
            raise ValueError("edge index lies outside site/atom range")
        if not bool(torch.all(torch.isfinite(self.displacements)).detach()) or not bool(
            torch.all(torch.isfinite(self.distances)).detach()
        ) or not bool(torch.all(torch.isfinite(self.switch)).detach()):
            raise ValueError("edge geometry/switch contains NaN or Inf")
        if edges and (
            not bool(torch.all(torch.isfinite(self.log_kernel[self.active])).detach())
            or not bool(torch.all(torch.isneginf(self.log_kernel[~self.active])).detach())
        ):
            raise ValueError("edge log kernel does not match the exact active mask")
        if self.epsilon.shape != () or self.epsilon.dtype != self.distances.dtype or self.epsilon.device != self.distances.device:
            raise ValueError("epsilon must be a scalar sharing edge dtype/device")
        if not bool(torch.isfinite(self.epsilon).detach()) or bool((self.epsilon <= 0).detach()):
            raise ValueError("epsilon must be finite and positive")
        for pointer, size, name in (
            (self.site_ptr, self.num_sites, "site_ptr"),
            (self.atom_ptr, self.num_atoms, "atom_ptr"),
        ):
            if pointer.shape != (size + 1,) or pointer.dtype != torch.long or pointer.device != self.distances.device:
                raise ValueError(f"{name} must be device-local long [{size + 1}]")
            if int(pointer[0].detach().cpu()) != 0 or int(pointer[-1].detach().cpu()) != edges:
                raise ValueError(f"{name} endpoints do not span all candidate edges")
            if bool(torch.any(pointer[1:] < pointer[:-1]).detach()):
                raise ValueError(f"{name} must be monotone")
        if self.atom_major_permutation.shape != (edges,) or self.atom_major_permutation.dtype != torch.long:
            raise ValueError("atom-major permutation must be long [E]")
        if self.atom_major_permutation.device != self.distances.device:
            raise ValueError("atom-major permutation must share edge device")
        expected = torch.arange(edges, device=self.distances.device)
        if not torch.equal(torch.sort(self.atom_major_permutation).values, expected):
            raise ValueError("atom-major permutation is not bijective")
        if edges:
            key = self.site_index * max(self.num_atoms, 1) + self.atom_index
            if bool(torch.any(key[1:] <= key[:-1]).detach()):
                raise ValueError("candidate edges must be unique site-major canonical order")

    @property
    def num_candidate_edges(self) -> int:
        return int(self.site_index.numel())

    @property
    def num_active_edges(self) -> int:
        return int(self.active.detach().sum().cpu())


def _pointers(indices: torch.Tensor, size: int) -> torch.Tensor:
    counts = torch.bincount(indices, minlength=size)
    return torch.cat((counts.new_zeros(1), torch.cumsum(counts, dim=0)))


def build_compact_transport_edges_from_candidates(
    site_index: torch.Tensor,
    atom_index: torch.Tensor,
    periodic_shift: torch.Tensor,
    displacements: torch.Tensor,
    *,
    num_sites: int,
    num_atoms: int,
    epsilon_ot: float,
    ell_ot: float,
    config: TransportSupportConfig,
    support_diagnostics: TransportSupportDiagnostics,
    template_id: str | None = None,
    sample_id: str | None = None,
) -> CompactTransportEdges:
    """Assemble live candidate arithmetic without any dense pair tensor."""

    if not isinstance(config, TransportSupportConfig) or (
        config.kind != "compact_c2" or config.backend != "edge_list"
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "edge construction requires compact_c2 with backend=edge_list",
            template_id=template_id,
            sample_id=sample_id,
        )
    edges = int(site_index.numel())
    if (
        site_index.shape != (edges,)
        or atom_index.shape != (edges,)
        or periodic_shift.shape != (edges, 3)
        or displacements.shape != (edges, 3)
        or site_index.dtype != torch.long
        or atom_index.dtype != torch.long
        or periodic_shift.dtype != torch.long
    ):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "candidate indices, shifts, and displacements have invalid shapes",
            template_id=template_id,
            sample_id=sample_id,
        )
    if (
        displacements.dtype not in (torch.float32, torch.float64)
        or site_index.device != displacements.device
        or atom_index.device != displacements.device
        or periodic_shift.device != displacements.device
        or not bool(torch.all(torch.isfinite(displacements)).detach())
    ):
        raise TransportSupportError(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "candidate edge geometry must be finite and share dtype/device",
            template_id=template_id,
            sample_id=sample_id,
        )
    if (
        not math.isfinite(float(epsilon_ot))
        or float(epsilon_ot) <= 0.0
        or not math.isfinite(float(ell_ot))
        or float(ell_ot) <= 0.0
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "epsilon_ot and ell_ot must be finite and positive",
            template_id=template_id,
            sample_id=sample_id,
        )

    edge_distances = torch.linalg.vector_norm(displacements, dim=-1)
    edge_switch = compact_c2_switch(edge_distances, config)
    active = edge_distances < edge_distances.new_tensor(config.cutoff)
    safe_switch = torch.where(active, edge_switch, torch.ones_like(edge_switch))
    edge_cost = edge_distances.square() / edge_distances.new_tensor(
        2.0 * float(ell_ot) ** 2
    )
    epsilon = edge_distances.new_tensor(float(epsilon_ot))
    live_log_kernel = -edge_cost / epsilon + torch.log(safe_switch)
    log_kernel = torch.where(
        active, live_log_kernel, torch.full_like(live_log_kernel, -torch.inf)
    )
    if (
        support_diagnostics.candidate_edge_count != edges
        or support_diagnostics.active_edge_count
        != int(active.detach().sum().cpu())
    ):
        raise TransportSupportError(
            "NO_TOTAL_SUPPORT",
            "support diagnostics do not match live candidate count/activity",
            template_id=template_id,
            sample_id=sample_id,
        )
    if edges:
        pair_key = site_index * max(num_atoms, 1) + atom_index
        if bool(torch.any(pair_key[1:] <= pair_key[:-1]).detach()):
            raise TransportSupportError(
                "DUPLICATE_EDGE",
                "candidate edge ordering is not unique site-major",
                template_id=template_id,
                sample_id=sample_id,
            )
    site_ptr = _pointers(site_index, num_sites)
    atom_key = atom_index * max(num_sites, 1) + site_index
    atom_major = torch.argsort(atom_key, stable=True)
    atom_ptr = _pointers(atom_index[atom_major], num_atoms)

    return CompactTransportEdges(
        site_index=site_index,
        atom_index=atom_index,
        displacements=displacements,
        distances=edge_distances,
        switch=edge_switch,
        log_kernel=log_kernel,
        active=active,
        atom_major_permutation=atom_major,
        site_ptr=site_ptr,
        atom_ptr=atom_ptr,
        num_sites=num_sites,
        num_atoms=num_atoms,
        num_vacancies=num_sites - num_atoms,
        epsilon=epsilon,
        support_diagnostics=support_diagnostics,
        periodic_shift=periodic_shift,
    )


def build_compact_transport_edges(
    displacements: torch.Tensor,
    *,
    epsilon_ot: float,
    ell_ot: float,
    config: TransportSupportConfig,
    template_id: str | None = None,
    sample_id: str | None = None,
) -> CompactTransportEdges:
    """Build a live-valued edge list after deterministic support preflight."""

    if not isinstance(config, TransportSupportConfig) or (
        config.kind != "compact_c2" or config.backend != "edge_list"
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "edge construction requires compact_c2 with backend=edge_list",
            template_id=template_id,
            sample_id=sample_id,
        )
    if config.candidate_backend != "dense":
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "blocked candidates require build_periodic_compact_transport_edges",
            template_id=template_id,
            sample_id=sample_id,
        )
    if displacements.ndim != 3 or displacements.shape[-1] != 3:
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "displacements must have shape [M,N,3]",
            template_id=template_id,
            sample_id=sample_id,
        )
    if displacements.dtype not in (torch.float32, torch.float64) or not bool(
        torch.all(torch.isfinite(displacements)).detach()
    ):
        raise TransportSupportError(
            "NONFINITE_SUPPORT_GEOMETRY",
            "displacements must be finite float32 or float64",
            template_id=template_id,
            sample_id=sample_id,
        )
    if (
        not math.isfinite(float(epsilon_ot))
        or float(epsilon_ot) <= 0.0
        or not math.isfinite(float(ell_ot))
        or float(ell_ot) <= 0.0
    ):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG",
            "epsilon_ot and ell_ot must be finite and positive",
            template_id=template_id,
            sample_id=sample_id,
        )

    sites, atoms = int(displacements.shape[0]), int(displacements.shape[1])
    distances = torch.linalg.vector_norm(displacements, dim=-1)
    # This dense switch is used only for discrete support certification.  The
    # live kernel below is rebuilt solely on gathered candidate-edge values.
    audit_switch = compact_c2_switch(distances, config)
    _, diagnostics = validate_compact_support(
        distances,
        audit_switch,
        config,
        template_id=template_id,
        sample_id=sample_id,
    )
    candidate = distances < distances.new_tensor(config.r_candidate)
    pairs = torch.nonzero(candidate, as_tuple=False)
    if pairs.numel() == 0 and atoms:
        raise TransportSupportError(
            "ATOM_WITHOUT_SUPPORT",
            "candidate edge list is empty",
            template_id=template_id,
            sample_id=sample_id,
        )
    site_index = pairs[:, 0].to(dtype=torch.long)
    atom_index = pairs[:, 1].to(dtype=torch.long)
    edge_displacements = displacements[site_index, atom_index]
    return build_compact_transport_edges_from_candidates(
        site_index=site_index,
        atom_index=atom_index,
        periodic_shift=torch.zeros(
            (int(site_index.numel()), 3),
            dtype=torch.long,
            device=site_index.device,
        ),
        displacements=edge_displacements,
        num_sites=sites,
        num_atoms=atoms,
        epsilon_ot=epsilon_ot,
        ell_ot=ell_ot,
        config=config,
        support_diagnostics=diagnostics,
        template_id=template_id,
        sample_id=sample_id,
    )


def materialize_dense_plan(result: SparseOTResult) -> DensePlanMaterialization:
    """Explicitly materialize ``[M,N]`` for oracle comparison or diagnostics."""

    if not isinstance(result, SparseOTResult):
        raise TypeError("materialize_dense_plan requires SparseOTResult")
    edges = result.edges
    plan = result.edge_plan.new_zeros((edges.num_sites, edges.num_atoms))
    if result.edge_plan.numel():
        plan = plan.index_put(
            (edges.site_index, edges.atom_index), result.edge_plan, accumulate=False
        )
    return DensePlanMaterialization(plan=plan)
