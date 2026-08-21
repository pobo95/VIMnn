"""Structure-isolated ragged transport batches."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from .factory import solve_atom_vacancy_ot
from .result import DualVariables, EvalOTConfig, OTResult, TrainSinkhornConfig


def solve_ragged_atom_vacancy_ot(
    atom_costs: Sequence[torch.Tensor],
    epsilon_ot: float,
    path: str,
    solver: str,
    config: Union[TrainSinkhornConfig, EvalOTConfig],
    init_duals: Optional[Sequence[Optional[DualVariables]]] = None,
) -> tuple[OTResult, ...]:
    if len(atom_costs) == 0:
        return ()
    if init_duals is None:
        initial = [None] * len(atom_costs)
    else:
        if len(init_duals) != len(atom_costs):
            raise ValueError("ragged init_duals length must match atom_costs")
        initial = list(init_duals)
    return tuple(
        solve_atom_vacancy_ot(
            cost,
            epsilon_ot,
            path,
            solver,
            config,
            init_duals=duals,
        )
        for cost, duals in zip(atom_costs, initial)
    )
