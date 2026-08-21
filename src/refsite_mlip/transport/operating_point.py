"""Reusable TRAIN_FIXED operating-domain metadata and audits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import torch

from .result import OTResult


OT_SOLVER_CONTRACT_VERSION = "dense_aggregate_vacancy_ot_v1"
OT_COST_CONVENTION = "mic_squared_distance_over_2_ell_ot_squared"


@dataclass(frozen=True)
class OTOperatingDomain:
    epsilon_ot: float
    ell_ot: float
    dtype: str
    fixed_sinkhorn_iterations: int
    marginal_tolerance: float = 1.0e-7
    cost_convention: str = OT_COST_CONVENTION
    solver_contract_version: str = OT_SOLVER_CONTRACT_VERSION

    def validate(self) -> None:
        for name, value in (
            ("epsilon_ot", self.epsilon_ot),
            ("ell_ot", self.ell_ot),
            ("marginal_tolerance", self.marginal_tolerance),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.fixed_sinkhorn_iterations, bool)
            or not isinstance(self.fixed_sinkhorn_iterations, Integral)
            or self.fixed_sinkhorn_iterations <= 0
        ):
            raise ValueError("fixed_sinkhorn_iterations must be positive")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.cost_convention != OT_COST_CONVENTION:
            raise ValueError("unsupported OT cost convention")
        if self.solver_contract_version != OT_SOLVER_CONTRACT_VERSION:
            raise ValueError("unsupported OT solver contract version")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class OperatingPointAudit:
    structure_id: str
    residual: float
    q_min: float
    q_max: float
    vacancy_mass_error: float
    cost_span_over_epsilon: float


def audit_train_fixed_operating_point(
    result: OTResult,
    atom_cost: torch.Tensor,
    domain: OTOperatingDomain,
    *,
    structure_id: str,
) -> OperatingPointAudit:
    domain.validate()
    if result.path_name != "train_fixed" or result.solver_name != "sinkhorn":
        raise ValueError("operating-point audit requires TRAIN_FIXED Sinkhorn")
    if result.sinkhorn_iterations != domain.fixed_sinkhorn_iterations:
        raise ValueError("TRAIN_FIXED iteration metadata mismatch")
    if str(atom_cost.dtype).removeprefix("torch.") != domain.dtype:
        raise ValueError("TRAIN_FIXED dtype metadata mismatch")
    tensors = (result.gamma, result.P, result.q, result.row_residual, result.column_residual)
    if not all(bool(torch.all(torch.isfinite(value))) for value in tensors):
        raise ValueError("TRAIN_FIXED operating point contains non-finite values")
    residual_tensor = torch.maximum(result.row_residual, result.column_residual)
    residual = float(residual_tensor.detach().cpu())
    vacancies = atom_cost.shape[0] - atom_cost.shape[1]
    vacancy_error_tensor = (result.q.sum() - float(vacancies)).abs()
    vacancy_error = float(vacancy_error_tensor.detach().cpu())
    q_min = float(result.q.min().detach().cpu())
    q_max = float(result.q.max().detach().cpu())
    if residual > domain.marginal_tolerance:
        raise ValueError(
            f"TRAIN_FIXED residual {residual:.9e} exceeds operating-domain "
            f"tolerance {domain.marginal_tolerance:.9e} for {structure_id}"
        )
    if vacancy_error > domain.marginal_tolerance or q_min < 0.0 or q_max > 1.0:
        raise ValueError("TRAIN_FIXED vacancy field violates operating domain")
    cost_span = float(
        ((atom_cost.max() - atom_cost.min()) / domain.epsilon_ot)
        .detach()
        .cpu()
    )
    return OperatingPointAudit(
        structure_id=structure_id,
        residual=residual,
        q_min=q_min,
        q_max=q_max,
        vacancy_mass_error=vacancy_error,
        cost_span_over_epsilon=cost_span,
    )
