"""Audit a small synthetic TRAIN_FIXED operating-domain sweep."""

from __future__ import annotations

import json

import torch

from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    EvalOTConfig,
    TrainSinkhornConfig,
    atom_site_cost,
    solve_atom_vacancy_ot,
)
from refsite_mlip.transport.operating_point import (
    OTOperatingDomain,
    audit_train_fixed_operating_point,
)


def _synthetic_structure(M: int, N: int):
    dtype = torch.float64
    cell = torch.tensor(
        [[6.0, 0.2, 0.1], [0.3, 5.8, 0.2], [0.1, 0.4, 6.2]],
        dtype=dtype,
    )
    index = torch.arange(M, dtype=dtype)
    fractional = torch.stack(
        (
            (0.13 + 0.31 * index).remainder(1.0),
            (0.07 + 0.23 * index).remainder(1.0),
            (0.19 + 0.17 * index).remainder(1.0),
        ),
        dim=1,
    )
    references = fractional @ cell
    displacement = 0.03 * torch.stack(
        (torch.sin(index[:N]), torch.cos(index[:N]), torch.sin(0.7 * index[:N])),
        dim=1,
    )
    positions = (references[:N] + displacement).requires_grad_(True)
    return positions, references, cell


def main() -> None:
    domain = OTOperatingDomain(
        epsilon_ot=0.5,
        ell_ot=1.5,
        dtype="float64",
        fixed_sinkhorn_iterations=256,
        marginal_tolerance=1.0e-7,
    )
    audits = []
    p_errors = []
    q_errors = []
    energy_errors = []
    force_errors = []
    for structure_id, M, N in (
        ("synthetic_pristine", 3, 3),
        ("synthetic_vacancy_1", 4, 3),
        ("synthetic_vacancy_2", 5, 3),
        ("synthetic_vacancy_1_large", 6, 5),
    ):
        positions, references, cell = _synthetic_structure(M, N)
        cost = atom_site_cost(
            positions, references, cell, (True, True, True), domain.ell_ot
        )
        fixed = solve_atom_vacancy_ot(
            cost,
            domain.epsilon_ot,
            TRAIN_FIXED,
            "sinkhorn",
            TrainSinkhornConfig(domain.fixed_sinkhorn_iterations),
        )
        adaptive = solve_atom_vacancy_ot(
            cost,
            domain.epsilon_ot,
            EVAL_ADAPTIVE,
            "hybrid",
            EvalOTConfig(
                sinkhorn_iterations=16,
                convergence_tolerance=1.0e-12,
            ),
        )
        audits.append(
            audit_train_fixed_operating_point(
                fixed, cost, domain, structure_id=structure_id
            )
        )
        fixed_energy = torch.sum(fixed.P * cost.square()) + 0.2 * fixed.q.square().sum()
        adaptive_energy = torch.sum(adaptive.P * cost.square()) + 0.2 * adaptive.q.square().sum()
        fixed_force = -torch.autograd.grad(
            fixed_energy, positions, create_graph=True, retain_graph=True
        )[0]
        adaptive_force = -torch.autograd.grad(
            adaptive_energy, positions, create_graph=True
        )[0]
        p_errors.append((fixed.P - adaptive.P).abs().max())
        q_errors.append((fixed.q - adaptive.q).abs().max())
        energy_errors.append((fixed_energy - adaptive_energy).abs())
        force_errors.append((fixed_force - adaptive_force).abs().max())

    residuals = torch.tensor([entry.residual for entry in audits], dtype=torch.float64)
    worst = audits[int(torch.argmax(residuals))]
    report = {
        "scope": "synthetic_only",
        "domain": domain.to_dict(),
        "structures": len(audits),
        "residual_max": float(residuals.max()),
        "residual_p99": float(torch.quantile(residuals, 0.99)),
        "residual_median": float(torch.median(residuals)),
        "worst_structure_id": worst.structure_id,
        "cost_span_over_epsilon_max": max(
            entry.cost_span_over_epsilon for entry in audits
        ),
        "q_min": min(entry.q_min for entry in audits),
        "q_max": max(entry.q_max for entry in audits),
        "P_parity_max_abs": float(torch.stack(p_errors).max()),
        "q_parity_max_abs": float(torch.stack(q_errors).max()),
        "toy_energy_parity_max_abs": float(torch.stack(energy_errors).max()),
        "toy_force_parity_max_abs": float(torch.stack(force_errors).max()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
