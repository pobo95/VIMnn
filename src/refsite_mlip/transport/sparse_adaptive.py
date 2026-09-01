"""Matrix-free adaptive hybrid OT on canonical compact transport edges.

The support and its ordering are discrete control data.  All selected
arithmetic from the live edge log-kernel through dual updates, ``edge_plan``,
and ``q`` remains connected to the input graph.  This module never constructs
an atom-site dense kernel/plan or a dense dual Hessian.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real

import torch

from .edge_list import CompactTransportEdges
from .gauge import project_duals, project_gauge
from .newton_krylov import validate_eval_config
from .result import (
    DualVariables,
    EvalOTConfig,
    SparseAdaptiveDiagnostics,
    SparseLineSearchDiagnostics,
    SparseNewtonOutcome,
    SparseOTResult,
    SparsePCGDiagnostics,
    SparsePCGOutcome,
)
from .sparse_sinkhorn import (
    _atom_major_segmented_sum,
    _segmented_sum,
    sparse_fixed_sinkhorn_updates,
    sparse_marginal_residual_components,
    sparse_sinkhorn_full_update,
    sparse_transport_plan,
    validate_sparse_duals,
)


class SparseAdaptiveTransportError(ValueError):
    """Actionable failure from the edge-list adaptive solver."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
    ) -> None:
        self.reason_code = str(reason_code)
        self.stage = str(stage)
        super().__init__(f"{self.reason_code}: {message} (stage={self.stage})")


def sparse_support_fingerprint(edges: CompactTransportEdges) -> str:
    """Hash only canonical discrete support metadata, never live geometry."""

    payload = (
        edges.num_sites,
        edges.num_atoms,
        edges.num_vacancies,
        tuple(int(value) for value in edges.site_index.detach().cpu().tolist()),
        tuple(int(value) for value in edges.atom_index.detach().cpu().tolist()),
        tuple(bool(value) for value in edges.active.detach().cpu().tolist()),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _num_columns(edges: CompactTransportEdges) -> int:
    return edges.num_atoms + int(edges.num_vacancies > 0)


def _validate_sparse_solver_support(edges: CompactTransportEdges) -> None:
    """Recheck the immutable support certificate before solver arithmetic."""

    if not isinstance(edges, CompactTransportEdges):
        raise SparseAdaptiveTransportError(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "edge-list adaptive solve requires CompactTransportEdges",
            stage="support_preflight",
        )
    if edges.distances.device.type not in ("cpu", "cuda"):
        raise SparseAdaptiveTransportError(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            f"unsupported edge tensor device {edges.distances.device.type!r}",
            stage="support_preflight",
        )
    diagnostics = edges.support_diagnostics
    if (
        len(diagnostics.atom_active_degrees) != edges.num_atoms
        or any(degree <= 0 for degree in diagnostics.atom_active_degrees)
    ):
        raise SparseAdaptiveTransportError(
            "ATOM_WITHOUT_SUPPORT",
            "support certificate contains an atom with zero active degree",
            stage="support_preflight",
        )
    if diagnostics.maximum_atom_matching_size != edges.num_atoms:
        raise SparseAdaptiveTransportError(
            "INCOMPLETE_ATOM_MATCHING",
            "support certificate does not contain a complete atom matching",
            stage="support_preflight",
        )
    if (
        not diagnostics.total_support_feasible
        or diagnostics.total_matching_size != edges.num_sites
    ):
        raise SparseAdaptiveTransportError(
            "NO_TOTAL_SUPPORT",
            "support certificate does not admit positive balanced scaling",
            stage="support_preflight",
        )
    if diagnostics.active_edge_count != edges.num_active_edges:
        raise SparseAdaptiveTransportError(
            "NO_TOTAL_SUPPORT",
            "support certificate active-edge count is inconsistent",
            stage="support_preflight",
        )


def _validate_dual_vector(
    edges: CompactTransportEdges, vector: torch.Tensor
) -> None:
    if vector.shape != (edges.num_sites + _num_columns(edges),):
        raise ValueError("sparse dual vector has incorrect shape")
    if vector.dtype != edges.distances.dtype or vector.device != edges.distances.device:
        raise ValueError("sparse dual vector dtype/device must match edges")
    if not bool(torch.all(torch.isfinite(vector)).detach()):
        raise SparseAdaptiveTransportError(
            "NONFINITE_DUAL",
            "dual vector contains NaN or Inf",
            stage="dual_validation",
        )


def _validate_plan(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
) -> None:
    if edge_plan.shape != edges.distances.shape or q.shape != (edges.num_sites,):
        raise ValueError("sparse plan has incorrect shape")
    if not bool(torch.all(torch.isfinite(edge_plan)).detach()) or not bool(
        torch.all(torch.isfinite(q)).detach()
    ):
        raise SparseAdaptiveTransportError(
            "NONFINITE_PLAN",
            "edge plan or vacancy plan contains NaN or Inf",
            stage="plan",
        )
    if bool(torch.any(edge_plan < 0.0).detach()) or bool(torch.any(q < 0.0).detach()):
        raise SparseAdaptiveTransportError(
            "NONFINITE_PLAN",
            "edge plan or vacancy plan contains negative mass",
            stage="plan",
        )
    if edge_plan.numel() and not torch.equal(
        edge_plan[~edges.active], torch.zeros_like(edge_plan[~edges.active])
    ):
        raise SparseAdaptiveTransportError(
            "NONFINITE_PLAN",
            "inactive candidate edge acquired nonzero mass",
            stage="plan",
        )


def sparse_dual_objective(
    edges: CompactTransportEdges,
    f: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Dense compact dual objective written as edge and vacancy reductions."""

    duals = validate_sparse_duals(edges, DualVariables(f=f, g=g))
    edge_plan, q = sparse_transport_plan(edges, duals.f, duals.g)
    objective = edges.epsilon * (edge_plan.sum() + q.sum()) - f.sum()
    if edges.num_atoms:
        objective = objective - g[: edges.num_atoms].sum()
    if edges.num_vacancies > 0:
        objective = objective - g[edges.num_atoms] * float(edges.num_vacancies)
    return objective


def sparse_residual_components(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_plan(edges, edge_plan, q)
    return sparse_marginal_residual_components(edges, edge_plan, q)


def sparse_residual_vector(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    site, atom, vacancy = sparse_residual_components(edges, edge_plan, q)
    if edges.num_vacancies > 0:
        return torch.cat((site, atom, vacancy.reshape(1)))
    return torch.cat((site, atom))


def sparse_jacobian_vector_product(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Apply the dual Hessian using only active edge and vacancy mass."""

    _validate_plan(edges, edge_plan, q)
    _validate_dual_vector(edges, vector)
    rows = edges.num_sites
    u = vector[:rows]
    v = vector[rows:]
    live_edge_sum = u[edges.site_index] + v[edges.atom_index]
    safe_edge_sum = torch.where(
        edges.active, live_edge_sum, torch.zeros_like(live_edge_sum)
    )
    weighted_edges = torch.where(
        edges.active,
        edge_plan * safe_edge_sum / edges.epsilon,
        torch.zeros_like(edge_plan),
    )
    site = _segmented_sum(weighted_edges, edges.site_index, rows)
    atom = _atom_major_segmented_sum(edges, weighted_edges)
    if edges.num_vacancies > 0:
        vacancy_live = q * (u + v[edges.num_atoms]) / edges.epsilon
        site = site + vacancy_live
        return torch.cat((site, atom, vacancy_live.sum().reshape(1)))
    return torch.cat((site, atom))


def sparse_gauge_fixed_operator(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
    vector: torch.Tensor,
    rho_gauge: float,
) -> torch.Tensor:
    columns = _num_columns(edges)
    projected = project_gauge(vector, edges.num_sites, columns)
    jacobian = project_gauge(
        sparse_jacobian_vector_product(edges, edge_plan, q, projected),
        edges.num_sites,
        columns,
    )
    return jacobian + vector.new_tensor(float(rho_gauge)) * (vector - projected)


def sparse_jacobi_inverse(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Exact Jacobi inverse; feasibility supplies strictly positive diagonals."""

    _validate_plan(edges, edge_plan, q)
    site = _segmented_sum(edge_plan, edges.site_index, edges.num_sites) + q
    atom = _atom_major_segmented_sum(edges, edge_plan)
    pieces = (site, atom, q.sum().reshape(1)) if edges.num_vacancies > 0 else (site, atom)
    diagonal = torch.cat(pieces) / edges.epsilon
    if not bool(torch.all(torch.isfinite(diagonal)).detach()) or bool(
        torch.any(diagonal <= 0.0).detach()
    ):
        raise SparseAdaptiveTransportError(
            "PCG_BREAKDOWN",
            "Jacobi diagonal is nonfinite or nonpositive",
            stage="preconditioner",
        )
    return torch.reciprocal(diagonal)


def _validate_pcg_controls(
    maximum_iterations: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, Integral)
        or int(maximum_iterations) <= 0
    ):
        raise SparseAdaptiveTransportError(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
            "PCG maximum iterations must be a positive integer",
            stage="config",
        )
    for name, value in (
        ("absolute tolerance", absolute_tolerance),
        ("relative tolerance", relative_tolerance),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise SparseAdaptiveTransportError(
                "UNSUPPORTED_DTYPE_DEVICE_CONFIG",
                f"PCG {name} must be finite and positive",
                stage="config",
            )


def sparse_projected_pcg(
    edges: CompactTransportEdges,
    edge_plan: torch.Tensor,
    q: torch.Tensor,
    rhs: torch.Tensor,
    inverse_diagonal: torch.Tensor,
    *,
    gauge_rho: float,
    maximum_iterations: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> SparsePCGOutcome:
    """Projected PCG whose operator is the sparse matrix-free HVP."""

    _validate_pcg_controls(
        maximum_iterations, absolute_tolerance, relative_tolerance
    )
    _validate_dual_vector(edges, rhs)
    _validate_dual_vector(edges, inverse_diagonal)
    if bool(torch.any(inverse_diagonal <= 0.0).detach()):
        raise SparseAdaptiveTransportError(
            "PCG_BREAKDOWN",
            "Jacobi inverse contains a nonpositive entry",
            stage="preconditioner",
        )
    columns = _num_columns(edges)
    projector = lambda value: project_gauge(value, edges.num_sites, columns)
    operator = lambda value: sparse_gauge_fixed_operator(
        edges, edge_plan, q, value, gauge_rho
    )
    solution = torch.zeros_like(rhs)
    residual = projector(rhs - operator(solution))
    initial_norm = torch.linalg.vector_norm(residual)
    threshold = max(
        float(absolute_tolerance),
        float(relative_tolerance) * float(initial_norm.detach().cpu()),
    )
    preconditioner_min = inverse_diagonal.min()
    preconditioner_max = inverse_diagonal.max()

    def outcome(
        *,
        converged: bool,
        iterations: int,
        final_norm: torch.Tensor,
        breakdown: str | None,
        curvature: torch.Tensor | None,
    ) -> SparsePCGOutcome:
        return SparsePCGOutcome(
            solution=solution,
            diagnostics=SparsePCGDiagnostics(
                converged=converged,
                iterations=iterations,
                initial_projected_residual=initial_norm,
                final_projected_residual=final_norm,
                breakdown_reason=breakdown,
                last_curvature=curvature,
                preconditioner_min=preconditioner_min,
                preconditioner_max=preconditioner_max,
            ),
        )

    if not bool(torch.isfinite(initial_norm).detach()):
        return outcome(
            converged=False,
            iterations=0,
            final_norm=initial_norm,
            breakdown="non-finite residual",
            curvature=None,
        )
    if float(initial_norm.detach().cpu()) <= threshold:
        return outcome(
            converged=True,
            iterations=0,
            final_norm=initial_norm,
            breakdown=None,
            curvature=None,
        )

    preconditioned = projector(inverse_diagonal * residual)
    direction = preconditioned
    residual_preconditioned = torch.dot(residual, preconditioned)
    final_norm = initial_norm
    last_curvature = None
    for iteration in range(int(maximum_iterations)):
        applied = operator(direction)
        curvature = torch.dot(direction, applied)
        last_curvature = curvature
        if not bool(torch.isfinite(curvature).detach()) or float(
            curvature.detach().cpu()
        ) <= 0.0:
            return outcome(
                converged=False,
                iterations=iteration,
                final_norm=final_norm,
                breakdown="non-finite or non-positive curvature",
                curvature=curvature,
            )
        alpha = residual_preconditioned / curvature
        solution = projector(solution + alpha * direction)
        residual = projector(residual - alpha * applied)
        final_norm = torch.linalg.vector_norm(residual)
        used = iteration + 1
        if not bool(torch.isfinite(final_norm).detach()):
            return outcome(
                converged=False,
                iterations=used,
                final_norm=final_norm,
                breakdown="non-finite residual",
                curvature=curvature,
            )
        if float(final_norm.detach().cpu()) <= threshold:
            return outcome(
                converged=True,
                iterations=used,
                final_norm=final_norm,
                breakdown=None,
                curvature=curvature,
            )
        next_preconditioned = projector(inverse_diagonal * residual)
        next_scalar = torch.dot(residual, next_preconditioned)
        if not bool(torch.isfinite(next_scalar).detach()) or float(
            next_scalar.detach().cpu()
        ) <= 0.0:
            return outcome(
                converged=False,
                iterations=used,
                final_norm=final_norm,
                breakdown="preconditioner breakdown",
                curvature=curvature,
            )
        beta = next_scalar / residual_preconditioned
        direction = projector(next_preconditioned + beta * direction)
        residual_preconditioned = next_scalar
    return outcome(
        converged=False,
        iterations=int(maximum_iterations),
        final_norm=final_norm,
        breakdown="maximum iterations reached",
        curvature=last_curvature,
    )


def _projected_residual(
    edges: CompactTransportEdges,
    f: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_plan, q = sparse_transport_plan(edges, f, g)
    _validate_plan(edges, edge_plan, q)
    residual = sparse_residual_vector(edges, edge_plan, q)
    projected = project_gauge(residual, edges.num_sites, _num_columns(edges))
    return edge_plan, q, projected


def solve_sparse_newton_krylov(
    edges: CompactTransportEdges,
    config: EvalOTConfig,
    initial: DualVariables | None = None,
) -> SparseNewtonOutcome:
    """Adaptive Newton-PCG on the existing compact edge support."""

    _validate_sparse_solver_support(edges)
    try:
        validate_eval_config(config)
        duals = validate_sparse_duals(edges, initial)
    except SparseAdaptiveTransportError:
        raise
    except (TypeError, ValueError) as error:
        raise SparseAdaptiveTransportError(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG", str(error), stage="config"
        ) from error
    f, g = project_duals(duals.f, duals.g)
    _, _, initial_projected = _projected_residual(edges, f, g)
    initial_residual = initial_projected.abs().max()
    total_cg = 0
    total_reductions = 0
    final_linear_residual = None
    last_accepted_damping = None
    pcg_history: list[SparsePCGDiagnostics] = []
    line_history: list[SparseLineSearchDiagnostics] = []

    def finish(
        *,
        converged: bool,
        iterations: int,
        failure_reason: str | None,
        final_projected_residual: torch.Tensor,
    ) -> SparseNewtonOutcome:
        return SparseNewtonOutcome(
            f=f,
            g=g,
            converged=converged,
            iterations=iterations,
            cg_iterations=total_cg,
            line_search_reductions=total_reductions,
            final_linear_residual=final_linear_residual,
            failure_reason=failure_reason,
            accepted_damping=last_accepted_damping,
            initial_projected_residual=initial_residual,
            final_projected_residual=final_projected_residual,
            pcg_steps=tuple(pcg_history),
            line_search_steps=tuple(line_history),
        )

    for newton_index in range(config.max_newton_iterations):
        edge_plan, q, projected_residual = _projected_residual(edges, f, g)
        residual_max = projected_residual.abs().max()
        if float(residual_max.detach().cpu()) <= config.convergence_tolerance:
            return finish(
                converged=True,
                iterations=newton_index,
                failure_reason=None,
                final_projected_residual=residual_max,
            )
        residual = sparse_residual_vector(edges, edge_plan, q)
        try:
            inverse_diagonal = sparse_jacobi_inverse(edges, edge_plan, q)
        except SparseAdaptiveTransportError as error:
            return finish(
                converged=False,
                iterations=newton_index,
                failure_reason=f"{error.reason_code}: {error}",
                final_projected_residual=residual_max,
            )
        pcg = sparse_projected_pcg(
            edges,
            edge_plan,
            q,
            -projected_residual,
            inverse_diagonal,
            gauge_rho=config.gauge_rho,
            maximum_iterations=config.pcg_max_iterations,
            absolute_tolerance=config.pcg_absolute_tolerance,
            relative_tolerance=config.pcg_relative_tolerance,
        )
        pcg_history.append(pcg.diagnostics)
        total_cg += pcg.diagnostics.iterations
        final_linear_residual = pcg.diagnostics.final_projected_residual
        if not pcg.diagnostics.converged:
            return finish(
                converged=False,
                iterations=newton_index,
                failure_reason=(
                    "PCG_BREAKDOWN: "
                    + str(pcg.diagnostics.breakdown_reason)
                ),
                final_projected_residual=residual_max,
            )

        correction = pcg.solution
        directional_derivative = torch.dot(residual, correction)
        if not bool(torch.isfinite(directional_derivative).detach()) or float(
            directional_derivative.detach().cpu()
        ) >= 0.0:
            return finish(
                converged=False,
                iterations=newton_index,
                failure_reason="PCG_BREAKDOWN: Newton direction is not descent",
                final_projected_residual=residual_max,
            )
        objective = sparse_dual_objective(edges, f, g)
        if not bool(torch.isfinite(objective).detach()):
            return finish(
                converged=False,
                iterations=newton_index,
                failure_reason="NONFINITE_OBJECTIVE",
                final_projected_residual=residual_max,
            )
        accepted = False
        accepted_f, accepted_g = f, g
        accepted_objective = None
        step = 1.0
        attempted: list[float] = []
        reductions_this_step = 0
        for reduction in range(config.max_line_search_reductions + 1):
            attempted.append(step)
            candidate_f = f + f.new_tensor(step) * correction[: edges.num_sites]
            candidate_g = g + g.new_tensor(step) * correction[edges.num_sites :]
            candidate_f, candidate_g = project_duals(candidate_f, candidate_g)
            candidate_objective = sparse_dual_objective(
                edges, candidate_f, candidate_g
            )
            bound = objective + objective.new_tensor(
                config.armijo_coefficient * step
            ) * directional_derivative
            if bool(torch.isfinite(candidate_objective).detach()) and float(
                (candidate_objective - bound).detach().cpu()
            ) <= 0.0:
                accepted = True
                accepted_f, accepted_g = candidate_f, candidate_g
                accepted_objective = candidate_objective
                reductions_this_step = reduction
                break
            step *= config.line_search_reduction
        line_history.append(
            SparseLineSearchDiagnostics(
                attempted_dampings=tuple(attempted),
                accepted_damping=(step if accepted else None),
                reductions=reductions_this_step,
                objective_before=objective,
                objective_after=accepted_objective,
                directional_derivative=directional_derivative,
                failure_reason=None if accepted else "ARMIJO_FAILURE",
            )
        )
        if not accepted:
            return finish(
                converged=False,
                iterations=newton_index,
                failure_reason="ARMIJO_FAILURE",
                final_projected_residual=residual_max,
            )
        f, g = accepted_f, accepted_g
        last_accepted_damping = step
        total_reductions += reductions_this_step

    _, _, final_projected = _projected_residual(edges, f, g)
    final_residual = final_projected.abs().max()
    converged = bool(torch.isfinite(final_residual).detach()) and float(
        final_residual.detach().cpu()
    ) <= config.convergence_tolerance
    return finish(
        converged=converged,
        iterations=config.max_newton_iterations,
        failure_reason=None if converged else "MAXIMUM_NEWTON_ITERATIONS",
        final_projected_residual=final_residual,
    )


def _build_sparse_adaptive_result(
    edges: CompactTransportEdges,
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    config: EvalOTConfig,
    outcome: SparseNewtonOutcome,
    sinkhorn_iterations: int,
    fallback_used: bool,
    fallback_reason: str | None,
    fallback_iterations: int,
    fallback_residual: torch.Tensor | None,
) -> SparseOTResult:
    edge_plan, q = sparse_transport_plan(edges, f, g)
    site, atom, vacancy = sparse_residual_components(edges, edge_plan, q)
    row_max = site.abs().max()
    atom_max = atom.abs().max() if atom.numel() else row_max.new_zeros(())
    vacancy_max = vacancy.abs()
    column_max = torch.maximum(atom_max, vacancy_max)
    residual = torch.maximum(row_max, column_max)
    q_mass_error = vacancy_max
    if not bool(torch.isfinite(residual).detach()):
        raise SparseAdaptiveTransportError(
            "NONFINITE_PLAN", "final residual is nonfinite", stage="result"
        )
    if float(q_mass_error.detach().cpu()) > config.convergence_tolerance:
        raise SparseAdaptiveTransportError(
            "Q_MASS_FAILURE",
            f"q mass error {float(q_mass_error.detach().cpu()):.9e} exceeds "
            f"{config.convergence_tolerance:.9e}",
            stage="result",
        )
    if float(residual.detach().cpu()) > config.convergence_tolerance:
        raise SparseAdaptiveTransportError(
            "RESIDUAL_TOLERANCE_FAILURE",
            f"marginal residual {float(residual.detach().cpu()):.9e} exceeds "
            f"{config.convergence_tolerance:.9e}",
            stage="result",
        )
    diagnostics = SparseAdaptiveDiagnostics(
        support_fingerprint=sparse_support_fingerprint(edges),
        initial_projected_residual=outcome.initial_projected_residual,
        final_projected_residual=outcome.final_projected_residual,
        q_mass_error=q_mass_error,
        pcg_steps=outcome.pcg_steps,
        line_search_steps=outcome.line_search_steps,
        fallback_reason=fallback_reason,
        fallback_residual=fallback_residual,
        dense_plan_materialized=False,
    )
    return SparseOTResult(
        edges=edges,
        edge_plan=edge_plan,
        q=q,
        f=f,
        g=g,
        row_residual=row_max,
        column_residual=column_max,
        converged=True,
        sinkhorn_iterations=sinkhorn_iterations,
        solver_name="edge_list_hybrid",
        path_name="eval_adaptive",
        support_diagnostics=edges.support_diagnostics.with_effective_tolerance(
            config.convergence_tolerance
        ),
        effective_diagnostic_tolerance=config.convergence_tolerance,
        dense_plan_materialized=False,
        newton_iterations=outcome.iterations,
        cg_iterations=outcome.cg_iterations,
        line_search_reductions=outcome.line_search_reductions,
        fallback_used=fallback_used,
        failure_reason=fallback_reason,
        accepted_damping=outcome.accepted_damping,
        warmup_sinkhorn_iterations=config.sinkhorn_iterations,
        fallback_sinkhorn_iterations=fallback_iterations,
        vacancy_residual=vacancy,
        adaptive_diagnostics=diagnostics,
    )


def _sparse_adaptive_sinkhorn_fallback(
    edges: CompactTransportEdges,
    config: EvalOTConfig,
    initial: DualVariables,
    trigger_reason: str | None,
) -> tuple[DualVariables, int, torch.Tensor]:
    duals = validate_sparse_duals(edges, initial)
    f, g = duals.f, duals.g
    final_residual = f.new_tensor(torch.inf)
    with torch.autocast(device_type=edges.distances.device.type, enabled=False):
        for index in range(config.fallback_sinkhorn_iterations):
            f, g = sparse_sinkhorn_full_update(edges, f, g)
            edge_plan, q = sparse_transport_plan(edges, f, g)
            residual = sparse_residual_vector(edges, edge_plan, q)
            final_residual = residual.abs().max()
            if float(final_residual.detach().cpu()) <= config.convergence_tolerance:
                return DualVariables(f=f, g=g), index + 1, final_residual
    raise SparseAdaptiveTransportError(
        "FALLBACK_CONVERGENCE_FAILURE",
        "sparse fallback Sinkhorn did not converge: "
        f"trigger={trigger_reason!r}, "
        f"iterations={config.fallback_sinkhorn_iterations}, "
        f"residual={float(final_residual.detach().cpu()):.9e}, "
        f"tolerance={config.convergence_tolerance:.9e}, "
        f"support_fingerprint={sparse_support_fingerprint(edges)}",
        stage="fallback",
    )


def solve_sparse_hybrid_eval(
    edges: CompactTransportEdges,
    config: EvalOTConfig,
    initial: DualVariables | None = None,
) -> SparseOTResult:
    """Warm-started sparse Newton-PCG with same-support sparse fallback."""

    _validate_sparse_solver_support(edges)
    try:
        validate_eval_config(config)
        warm = sparse_fixed_sinkhorn_updates(
            edges, config.sinkhorn_iterations, initial=initial
        )
    except SparseAdaptiveTransportError:
        raise
    except (TypeError, ValueError) as error:
        raise SparseAdaptiveTransportError(
            "UNSUPPORTED_DTYPE_DEVICE_CONFIG", str(error), stage="config"
        ) from error
    outcome = solve_sparse_newton_krylov(edges, config, warm)
    if outcome.converged:
        try:
            return _build_sparse_adaptive_result(
                edges,
                outcome.f,
                outcome.g,
                config=config,
                outcome=outcome,
                sinkhorn_iterations=config.sinkhorn_iterations,
                fallback_used=False,
                fallback_reason=None,
                fallback_iterations=0,
                fallback_residual=None,
            )
        except SparseAdaptiveTransportError as error:
            if error.reason_code not in (
                "Q_MASS_FAILURE",
                "RESIDUAL_TOLERANCE_FAILURE",
            ):
                raise
            fallback_reason = error.reason_code
    else:
        fallback_reason = outcome.failure_reason
    fallback, used, residual = _sparse_adaptive_sinkhorn_fallback(
        edges,
        config,
        DualVariables(outcome.f, outcome.g),
        fallback_reason,
    )
    return _build_sparse_adaptive_result(
        edges,
        fallback.f,
        fallback.g,
        config=config,
        outcome=outcome,
        sinkhorn_iterations=config.sinkhorn_iterations + used,
        fallback_used=True,
        fallback_reason=fallback_reason,
        fallback_iterations=used,
        fallback_residual=residual,
    )
