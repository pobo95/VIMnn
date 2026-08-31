"""Result and configuration records for dense atom-vacancy transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .support import TransportSupportDiagnostics


@dataclass(frozen=True)
class DualVariables:
    f: torch.Tensor
    g: torch.Tensor


@dataclass(frozen=True)
class TrainSinkhornConfig:
    iterations: int = 256
    diagnostic_tolerance: float | None = None


@dataclass(frozen=True)
class EvalOTConfig:
    sinkhorn_iterations: int = 128
    max_newton_iterations: int = 20
    convergence_tolerance: float = 1.0e-10
    pcg_max_iterations: int = 256
    pcg_absolute_tolerance: float = 1.0e-12
    pcg_relative_tolerance: float = 1.0e-10
    gauge_rho: float = 1.0
    armijo_coefficient: float = 1.0e-4
    line_search_reduction: float = 0.5
    max_line_search_reductions: int = 12
    fallback_sinkhorn_iterations: int = 1024


@dataclass(frozen=True)
class OTResult:
    gamma: torch.Tensor
    P: torch.Tensor
    q: torch.Tensor
    f: torch.Tensor
    g: torch.Tensor
    row_residual: torch.Tensor
    column_residual: torch.Tensor
    converged: Union[bool, torch.Tensor]
    sinkhorn_iterations: int
    newton_iterations: int
    cg_iterations: int
    line_search_reductions: int
    fallback_used: bool
    solver_name: str
    path_name: str
    final_linear_residual: Optional[torch.Tensor] = None
    failure_reason: Optional[str] = None
    support_diagnostics: TransportSupportDiagnostics | None = None
    effective_diagnostic_tolerance: float | None = None
    accepted_damping: Optional[float] = None
    warmup_sinkhorn_iterations: int = 0
    fallback_sinkhorn_iterations: int = 0


@dataclass(frozen=True)
class PCGResult:
    solution: torch.Tensor
    converged: bool
    iterations: int
    initial_residual: torch.Tensor
    final_residual: torch.Tensor
    breakdown: Optional[str]


@dataclass(frozen=True)
class NewtonOutcome:
    f: torch.Tensor
    g: torch.Tensor
    converged: bool
    iterations: int
    cg_iterations: int
    line_search_reductions: int
    final_linear_residual: Optional[torch.Tensor]
    failure_reason: Optional[str]
    accepted_damping: Optional[float] = None
