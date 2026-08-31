"""Validated dense balanced atom-vacancy transport problems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

from .support import (
    TransportSupportConfig,
    TransportSupportDiagnostics,
    TransportSupportError,
    compact_c2_switch,
    validate_compact_support,
)


@dataclass(frozen=True)
class OTProblem:
    atom_cost: torch.Tensor
    cost: torch.Tensor
    row_marginal: torch.Tensor
    column_marginal: torch.Tensor
    epsilon: torch.Tensor
    num_sites: int
    num_atoms: int
    num_vacancies: int
    log_kernel: torch.Tensor | None = None
    support_diagnostics: TransportSupportDiagnostics | None = None

    @property
    def num_columns(self) -> int:
        return int(self.cost.shape[1])


def _positive_fixed_scalar(value: Real, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive fixed real")
    return float(value)


def build_ot_problem(
    atom_cost: torch.Tensor,
    epsilon_ot: Real,
    *,
    support_config: TransportSupportConfig | None = None,
    atom_distances: torch.Tensor | None = None,
    template_id: str | None = None,
    sample_id: str | None = None,
) -> OTProblem:
    support = TransportSupportConfig() if support_config is None else support_config
    if not isinstance(support, TransportSupportConfig):
        raise TransportSupportError(
            "INVALID_SUPPORT_CONFIG", "support_config must be TransportSupportConfig"
        )
    if support.kind == "compact_c2" and atom_distances is not None:
        if not bool(torch.all(torch.isfinite(atom_distances)).detach()):
            raise TransportSupportError(
                "NONFINITE_SUPPORT_GEOMETRY",
                "support distances contain NaN or Inf",
                template_id=template_id,
                sample_id=sample_id,
            )
    if atom_cost.ndim != 2:
        raise ValueError("atom_cost must have shape [M, N]")
    if atom_cost.dtype not in (torch.float32, torch.float64):
        raise ValueError("atom_cost must use float32 or float64")
    if atom_cost.shape[0] <= 0:
        raise ValueError("an OT problem requires at least one reference site")
    if not bool(torch.all(torch.isfinite(atom_cost))):
        raise ValueError("atom_cost contains NaN or Inf")
    if bool(torch.any(atom_cost < 0.0)):
        raise ValueError("geometry atom_cost must be nonnegative")

    epsilon_value = _positive_fixed_scalar(epsilon_ot, "epsilon_ot")
    sites, atoms = (int(value) for value in atom_cost.shape)
    if atoms > sites:
        raise ValueError(
            f"atom count N={atoms} exceeds reference-site count M={sites}"
        )
    vacancies = sites - atoms
    epsilon = atom_cost.new_tensor(epsilon_value)
    rows = torch.ones(sites, dtype=atom_cost.dtype, device=atom_cost.device)
    atomic_columns = torch.ones(
        atoms, dtype=atom_cost.dtype, device=atom_cost.device
    )
    if vacancies > 0:
        vacancy_cost = torch.zeros(
            (sites, 1), dtype=atom_cost.dtype, device=atom_cost.device
        )
        cost = torch.cat((atom_cost, vacancy_cost), dim=1)
        vacancy_mass = atom_cost.new_tensor([float(vacancies)])
        columns = torch.cat((atomic_columns, vacancy_mass), dim=0)
    else:
        cost = atom_cost
        columns = atomic_columns
    if not bool(torch.isfinite(cost.abs().max() / epsilon)):
        raise ValueError("atom_cost/epsilon_ot scale is non-finite")
    log_kernel = None
    support_diagnostics = None
    if support.kind == "compact_c2":
        if atom_distances is None:
            raise TransportSupportError(
                "NONFINITE_SUPPORT_GEOMETRY",
                "compact_c2 requires atom_distances with shape [M,N]",
                template_id=template_id,
                sample_id=sample_id,
            )
        if (
            atom_distances.shape != atom_cost.shape
            or atom_distances.dtype != atom_cost.dtype
            or atom_distances.device != atom_cost.device
        ):
            raise TransportSupportError(
                "NONFINITE_SUPPORT_GEOMETRY",
                "atom_distances shape/dtype/device must match atom_cost",
                template_id=template_id,
                sample_id=sample_id,
            )
        switch = compact_c2_switch(atom_distances, support)
        active, support_diagnostics = validate_compact_support(
            atom_distances,
            switch,
            support,
            template_id=template_id,
            sample_id=sample_id,
        )
        safe_switch = torch.where(active, switch, torch.ones_like(switch))
        atom_log_kernel = -atom_cost / epsilon + torch.log(safe_switch)
        atom_log_kernel = torch.where(
            active, atom_log_kernel, torch.full_like(atom_log_kernel, -torch.inf)
        )
        if vacancies > 0:
            vacancy_log_kernel = torch.zeros(
                (sites, 1), dtype=atom_cost.dtype, device=atom_cost.device
            )
            log_kernel = torch.cat((atom_log_kernel, vacancy_log_kernel), dim=1)
        else:
            log_kernel = atom_log_kernel
    return OTProblem(
        atom_cost=atom_cost,
        cost=cost,
        row_marginal=rows,
        column_marginal=columns,
        epsilon=epsilon,
        num_sites=sites,
        num_atoms=atoms,
        num_vacancies=vacancies,
        log_kernel=log_kernel,
        support_diagnostics=support_diagnostics,
    )
