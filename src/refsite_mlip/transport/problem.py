"""Validated dense balanced atom-vacancy transport problems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch


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


def build_ot_problem(atom_cost: torch.Tensor, epsilon_ot: Real) -> OTProblem:
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
    return OTProblem(
        atom_cost=atom_cost,
        cost=cost,
        row_marginal=rows,
        column_marginal=columns,
        epsilon=epsilon,
        num_sites=sites,
        num_atoms=atoms,
        num_vacancies=vacancies,
    )
