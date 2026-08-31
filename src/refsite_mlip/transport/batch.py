"""Structure-isolated ragged transport batches."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from .factory import solve_atom_vacancy_ot
from .result import DualVariables, EvalOTConfig, OTResult, TrainSinkhornConfig
from .support import TransportSupportConfig


def solve_ragged_atom_vacancy_ot(
    atom_costs: Sequence[torch.Tensor],
    epsilon_ot: float,
    path: str,
    solver: str,
    config: Union[TrainSinkhornConfig, EvalOTConfig],
    init_duals: Optional[Sequence[Optional[DualVariables]]] = None,
    *,
    support_config: TransportSupportConfig | None = None,
    atom_distances: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[OTResult, ...]:
    if len(atom_costs) == 0:
        return ()
    if init_duals is None:
        initial = [None] * len(atom_costs)
    else:
        if len(init_duals) != len(atom_costs):
            raise ValueError("ragged init_duals length must match atom_costs")
        initial = list(init_duals)
    if atom_distances is None:
        distances = [None] * len(atom_costs)
    else:
        if len(atom_distances) != len(atom_costs):
            raise ValueError("ragged atom_distances length must match atom_costs")
        distances = list(atom_distances)
    return tuple(
        solve_atom_vacancy_ot(
            cost,
            epsilon_ot,
            path,
            solver,
            config,
            init_duals=duals,
            support_config=support_config,
            atom_distances=distance,
        )
        for cost, duals, distance in zip(atom_costs, initial, distances)
    )
