"""Atom-vacancy marginals and chemical probability fields."""

from __future__ import annotations

import torch

from .problem import OTProblem


def split_atom_vacancy_plan(
    problem: OTProblem, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if gamma.shape != problem.cost.shape:
        raise ValueError("gamma shape does not match the OT problem")
    P = gamma[:, : problem.num_atoms]
    if problem.num_vacancies > 0:
        q = gamma[:, problem.num_atoms]
    else:
        q = torch.zeros(
            problem.num_sites, dtype=gamma.dtype, device=gamma.device
        )
    return P, q


def species_probability_field(
    P: torch.Tensor,
    atomic_numbers: torch.Tensor,
    species: torch.Tensor,
) -> torch.Tensor:
    if atomic_numbers.shape != (P.shape[1],) or atomic_numbers.dtype != torch.long:
        raise ValueError("atomic_numbers must be long with shape [N]")
    if species.ndim != 1 or species.dtype != torch.long:
        raise ValueError("species must be a one-dimensional long tensor")
    if atomic_numbers.device != P.device or species.device != P.device:
        raise ValueError("species inputs must share P device")
    matches = atomic_numbers.unsqueeze(-1) == species.unsqueeze(0)
    if P.shape[1] > 0 and bool(torch.any(matches.sum(dim=-1) != 1)):
        raise ValueError("every atom must match exactly one modeled species")
    return P @ matches.to(dtype=P.dtype)
