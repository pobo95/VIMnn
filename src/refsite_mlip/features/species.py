"""Fixed-vocabulary chemical probability fields."""

from __future__ import annotations

import torch


def species_indicator(
    atomic_numbers: torch.Tensor,
    species_vocabulary: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if atomic_numbers.ndim != 1 or atomic_numbers.dtype != torch.long:
        raise ValueError("atomic_numbers must be one-dimensional torch.long")
    vocabulary = torch.tensor(
        species_vocabulary,
        dtype=torch.long,
        device=atomic_numbers.device,
    )
    matches = atomic_numbers.unsqueeze(-1) == vocabulary.unsqueeze(0)
    if atomic_numbers.numel() > 0 and bool(torch.any(matches.sum(dim=-1) != 1)):
        unknown = atomic_numbers[matches.sum(dim=-1) != 1]
        raise ValueError(
            f"unknown atomic species in probability multipoles: {unknown.tolist()}"
        )
    return matches.to(dtype=dtype)


def species_probabilities(
    P: torch.Tensor,
    atomic_numbers: torch.Tensor,
    species_vocabulary: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    indicator = species_indicator(
        atomic_numbers, species_vocabulary, dtype=P.dtype
    )
    return P @ indicator, indicator
