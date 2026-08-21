"""Typed translation stabilizers and exact integer alias checks."""

from __future__ import annotations

import itertools
import math
from typing import List, Tuple

import torch

from .types import TypedStabilizer


def torus_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    difference = left - right
    return difference - torch.round(difference)


def canonical_phase(phase: torch.Tensor) -> torch.Tensor:
    return phase - torch.floor(phase)


def _match_translation(
    sites: torch.Tensor, site_types: torch.Tensor, translation: torch.Tensor, tolerance: float
) -> torch.Tensor | None:
    targets = canonical_phase(sites + translation)
    permutation = torch.full(
        (sites.shape[0],), -1, dtype=torch.long, device=sites.device
    )
    used = set()
    for source in range(sites.shape[0]):
        distances = torch.linalg.vector_norm(
            torus_difference(sites, targets[source]), dim=-1
        )
        valid = torch.nonzero(
            (site_types == site_types[source]) & (distances <= tolerance)
        ).reshape(-1)
        candidates = [int(index) for index in valid.cpu().tolist() if int(index) not in used]
        if len(candidates) != 1:
            return None
        target = candidates[0]
        permutation[source] = target
        used.add(target)
    return permutation


def find_typed_stabilizer(
    sites: torch.Tensor, site_types: torch.Tensor, tolerance: float = 1.0e-10
) -> TypedStabilizer:
    """Enumerate finite typed translations induced by site-to-site differences."""

    if sites.ndim != 2 or sites.shape[1] != 3 or site_types.shape != (sites.shape[0],):
        raise ValueError("expected sites [M,3] and site_types [M]")
    if sites.shape[0] == 0:
        raise ValueError("typed template must contain at least one site")
    anchor_type = site_types[0]
    candidate_indices = torch.nonzero(site_types == anchor_type).reshape(-1)
    translations: List[torch.Tensor] = []
    permutations: List[torch.Tensor] = []
    for index in candidate_indices.cpu().tolist():
        translation = canonical_phase(sites[int(index)] - sites[0])
        if any(
            bool(torch.linalg.vector_norm(torus_difference(translation, known)) <= tolerance)
            for known in translations
        ):
            continue
        permutation = _match_translation(sites, site_types, translation, tolerance)
        if permutation is not None:
            translations.append(translation)
            permutations.append(permutation)
    return TypedStabilizer(torch.stack(translations), torch.stack(permutations))


def _determinant3(rows: torch.Tensor) -> int:
    a = [[int(value) for value in row] for row in rows.cpu().tolist()]
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def integer_alias_order(modes: torch.Tensor) -> int:
    """Return the finite torus-kernel order, the gcd of full-rank minors."""

    if modes.ndim != 2 or modes.shape[1] != 3 or modes.dtype != torch.long:
        raise ValueError("integer modes must have shape [G,3] and dtype long")
    determinants = [
        abs(_determinant3(modes[list(indices)]))
        for indices in itertools.combinations(range(modes.shape[0]), 3)
    ]
    nonzero = [value for value in determinants if value != 0]
    if not nonzero:
        raise ValueError("primary reciprocal modes do not have integer rank 3")
    order = nonzero[0]
    for value in nonzero[1:]:
        order = math.gcd(order, value)
    return order


def validate_alias_matches_stabilizer(
    modes: torch.Tensor, stabilizer: TypedStabilizer, tolerance: float = 1.0e-10
) -> None:
    """Require the integer-mode kernel to equal the typed stabilizer."""

    order = integer_alias_order(modes)
    if order != stabilizer.translations.shape[0]:
        raise ValueError(
            "primary integer alias group does not equal typed stabilizer"
        )
    phases = stabilizer.translations @ modes.to(
        dtype=stabilizer.translations.dtype,
        device=stabilizer.translations.device,
    ).T
    if bool(torch.any(torch.abs(phases - torch.round(phases)) > tolerance)):
        raise ValueError(
            "primary integer alias group does not contain typed stabilizer"
        )


def stabilizer_equivalent(
    left: torch.Tensor,
    right: torch.Tensor,
    stabilizer: TypedStabilizer,
    tolerance: float = 1.0e-8,
) -> bool:
    difference = left - right
    distances = torch.linalg.vector_norm(
        torus_difference(difference.unsqueeze(-2), stabilizer.translations), dim=-1
    )
    return bool(torch.any(distances <= tolerance))


def permutation_for_translation(
    translation: torch.Tensor,
    stabilizer: TypedStabilizer,
    tolerance: float = 1.0e-8,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(
        torus_difference(translation.unsqueeze(0), stabilizer.translations), dim=-1
    )
    valid = torch.nonzero(distances <= tolerance).reshape(-1)
    if valid.numel() != 1:
        raise ValueError("translation is not a unique typed stabilizer element")
    return stabilizer.permutations[int(valid[0])]
