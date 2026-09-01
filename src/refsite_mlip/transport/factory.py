"""Public solver factory with strict training/evaluation path separation."""

from __future__ import annotations

from typing import Optional, Union

import torch

from .diagnostics import build_result
from .hybrid import solve_hybrid_eval
from .newton_krylov import solve_newton_krylov, validate_eval_config
from .problem import build_ot_problem
from .result import (
    DualVariables,
    EvalOTConfig,
    OTResult,
    TrainSinkhornConfig,
)
from .sinkhorn import (
    solve_sinkhorn_eval_adaptive,
    solve_sinkhorn_train_fixed,
)
from .support import TransportSupportConfig, TransportSupportError


TRAIN_FIXED = "train_fixed"
EVAL_ADAPTIVE = "eval_adaptive"


def _analytic_empty_atom_result(problem, path: str, config) -> OTResult:
    f = torch.zeros_like(problem.row_marginal)
    g = torch.zeros_like(problem.column_marginal)
    effective_tolerance = None
    if path == TRAIN_FIXED and isinstance(config, TrainSinkhornConfig):
        effective_tolerance = (
            1.0e-6
            if config.diagnostic_tolerance is None
            and problem.cost.dtype == torch.float32
            else 1.0e-7
            if config.diagnostic_tolerance is None
            else float(config.diagnostic_tolerance)
        )
    return build_result(
        problem,
        f,
        g,
        converged=True,
        sinkhorn_iterations=0,
        newton_iterations=0,
        cg_iterations=0,
        line_search_reductions=0,
        fallback_used=False,
        solver_name="analytic_empty_atoms",
        path_name=path,
        effective_diagnostic_tolerance=effective_tolerance,
    )


def solve_atom_vacancy_ot(
    atom_cost: torch.Tensor,
    epsilon_ot: float,
    path: str,
    solver: str,
    config: Union[TrainSinkhornConfig, EvalOTConfig],
    init_duals: Optional[DualVariables] = None,
    *,
    support_config: TransportSupportConfig | None = None,
    atom_distances: torch.Tensor | None = None,
    template_id: str | None = None,
    sample_id: str | None = None,
) -> OTResult:
    if path not in (TRAIN_FIXED, EVAL_ADAPTIVE):
        raise ValueError("path must be train_fixed or eval_adaptive")
    support = TransportSupportConfig() if support_config is None else support_config
    if not isinstance(support, TransportSupportConfig):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG", "support_config must be TransportSupportConfig"
        )
    if support.backend == "edge_list":
        raise TransportSupportError(
            "EDGE_LIST_REQUIRES_DISPLACEMENTS",
            "solve_atom_vacancy_ot is the dense backend; edge-list transport requires live displacement vectors via build_compact_transport_edges and the sparse fixed/adaptive solver",
            template_id=template_id,
            sample_id=sample_id,
        )
    problem = build_ot_problem(
        atom_cost,
        epsilon_ot,
        support_config=support,
        atom_distances=atom_distances,
        template_id=template_id,
        sample_id=sample_id,
    )
    if problem.num_atoms == 0:
        return _analytic_empty_atom_result(problem, path, config)

    if path == TRAIN_FIXED:
        if solver != "sinkhorn":
            raise ValueError(
                "TRAIN_FIXED supports only fixed-unrolled log-Sinkhorn; "
                "fixed Newton/PCG training is not a production option"
            )
        if init_duals is not None:
            raise ValueError("TRAIN_FIXED uses deterministic zero dual initialization")
        if not isinstance(config, TrainSinkhornConfig):
            raise ValueError("TRAIN_FIXED requires TrainSinkhornConfig")
        return solve_sinkhorn_train_fixed(problem, config)

    if not isinstance(config, EvalOTConfig):
        raise ValueError("EVAL_ADAPTIVE requires EvalOTConfig")
    validate_eval_config(config)
    if solver == "sinkhorn":
        return solve_sinkhorn_eval_adaptive(
            problem,
            maximum_iterations=config.sinkhorn_iterations,
            tolerance=config.convergence_tolerance,
            initial=init_duals,
        )
    if solver == "newton_krylov":
        outcome = solve_newton_krylov(problem, config, init_duals)
        if not outcome.converged:
            raise ValueError(
                "Newton-Krylov evaluation did not converge without fallback: "
                f"{outcome.failure_reason}"
            )
        return build_result(
            problem,
            outcome.f,
            outcome.g,
            converged=True,
            sinkhorn_iterations=0,
            newton_iterations=outcome.iterations,
            cg_iterations=outcome.cg_iterations,
            line_search_reductions=outcome.line_search_reductions,
            fallback_used=False,
            solver_name="newton_krylov",
            path_name=EVAL_ADAPTIVE,
            final_linear_residual=outcome.final_linear_residual,
            accepted_damping=outcome.accepted_damping,
            effective_diagnostic_tolerance=(
                config.convergence_tolerance
                if problem.support_diagnostics is not None
                else None
            ),
        )
    if solver == "hybrid":
        return solve_hybrid_eval(problem, config, init_duals)
    raise ValueError("solver must be sinkhorn, newton_krylov, or hybrid")
