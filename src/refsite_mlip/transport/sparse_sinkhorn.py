"""Fixed-count log-Sinkhorn on canonical compact candidate edges."""

from __future__ import annotations

import math
from numbers import Integral

import torch

from .edge_list import CompactTransportEdges
from .gauge import project_duals
from .result import SparseOTResult, TrainSinkhornConfig


def _segmented_logsumexp(
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    *,
    extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stable differentiable reduction using only PyTorch 2.6 primitives.

    The segment maximum is retained in the graph.  It cancels algebraically
    from logsumexp on a stable branch, and PyTorch's scatter-reduce/index-add
    combination supports both first and second derivatives.  Only support
    indices are discrete control data.
    """

    if num_segments == 0:
        return values.new_empty((0,))
    maxima = values.new_full((num_segments,), -torch.inf)
    if values.numel():
        maxima = maxima.scatter_reduce(
            0, index, values, reduce="amax", include_self=True
        )
    if extra is not None:
        maxima = torch.maximum(maxima, extra)
    if not bool(torch.all(torch.isfinite(maxima)).detach()):
        raise ValueError("segmented logsumexp encountered an all-masked segment")
    sums = values.new_zeros((num_segments,))
    if values.numel():
        sums = sums.index_add(0, index, torch.exp(values - maxima[index]))
    if extra is not None:
        sums = sums + torch.exp(extra - maxima)
    return maxima + torch.log(sums)


def _segmented_sum(
    values: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    return values.new_zeros((num_segments,)).index_add(0, index, values)


def sparse_sinkhorn_full_update(
    edges: CompactTransportEdges,
    f: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One exact-zero-aware update without constructing a dense kernel."""

    epsilon = edges.epsilon
    atomic_values = g[edges.atom_index] / epsilon + edges.log_kernel
    vacancy_values = (
        g[edges.num_atoms].expand(edges.num_sites) / epsilon
        if edges.num_vacancies > 0
        else None
    )
    row_lse = _segmented_logsumexp(
        atomic_values,
        edges.site_index,
        edges.num_sites,
        extra=vacancy_values,
    )
    updated_f = -epsilon * row_lse

    column_values = updated_f[edges.site_index] / epsilon + edges.log_kernel
    atom_lse = _segmented_logsumexp(
        column_values, edges.atom_index, edges.num_atoms
    )
    updated_atomic_g = -epsilon * atom_lse
    if edges.num_vacancies > 0:
        vacancy_g = epsilon * (
            torch.log(updated_f.new_tensor(float(edges.num_vacancies)))
            - torch.logsumexp(updated_f / epsilon, dim=0)
        )
        updated_g = torch.cat((updated_atomic_g, vacancy_g.reshape(1)))
    else:
        updated_g = updated_atomic_g
    return project_duals(updated_f, updated_g)


def sparse_transport_plan(
    edges: CompactTransportEdges, f: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    live = f[edges.site_index] / edges.epsilon + g[edges.atom_index] / edges.epsilon
    active_log_plan = live + edges.log_kernel
    safe = torch.where(edges.active, active_log_plan, torch.zeros_like(active_log_plan))
    edge_plan = torch.where(edges.active, torch.exp(safe), torch.zeros_like(safe))
    if edges.num_vacancies > 0:
        q = torch.exp(f / edges.epsilon + g[edges.num_atoms] / edges.epsilon)
    else:
        q = torch.zeros_like(f)
    return edge_plan, q


def sparse_marginal_residuals(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    site_mass = _segmented_sum(
        edge_plan, edges.site_index, edges.num_sites
    ) + q
    atom_mass = _segmented_sum(
        edge_plan, edges.atom_index, edges.num_atoms
    )
    if edges.num_vacancies > 0:
        column = torch.cat(
            (atom_mass - 1.0, (q.sum() - float(edges.num_vacancies)).reshape(1))
        )
    else:
        column = atom_mass - 1.0
    return site_mass - 1.0, column


def solve_sparse_sinkhorn_train_fixed(
    edges: CompactTransportEdges, config: TrainSinkhornConfig
) -> SparseOTResult:
    if (
        isinstance(config.iterations, bool)
        or not isinstance(config.iterations, Integral)
        or int(config.iterations) <= 0
    ):
        raise ValueError("Sinkhorn iterations must be a positive integer")
    tolerance = (
        1.0e-6
        if config.diagnostic_tolerance is None and edges.distances.dtype == torch.float32
        else 1.0e-7
        if config.diagnostic_tolerance is None
        else float(config.diagnostic_tolerance)
    )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("diagnostic_tolerance must be finite and positive")
    f = edges.distances.new_zeros(edges.num_sites)
    g = edges.distances.new_zeros(
        edges.num_atoms + int(edges.num_vacancies > 0)
    )
    with torch.autocast(device_type=edges.distances.device.type, enabled=False):
        for _ in range(int(config.iterations)):
            f, g = sparse_sinkhorn_full_update(
                edges,
                f,
                g,
            )
    edge_plan, q = sparse_transport_plan(edges, f, g)
    row, column = sparse_marginal_residuals(
        edges,
        edge_plan,
        q,
    )
    row_max, column_max = row.abs().max(), column.abs().max()
    diagnostics = edges.support_diagnostics.with_effective_tolerance(tolerance)
    return SparseOTResult(
        edges=edges,
        edge_plan=edge_plan,
        q=q,
        f=f,
        g=g,
        row_residual=row_max,
        column_residual=column_max,
        converged=torch.maximum(row_max, column_max) <= tolerance,
        sinkhorn_iterations=int(config.iterations),
        support_diagnostics=diagnostics,
        effective_diagnostic_tolerance=tolerance,
    )
