#!/usr/bin/env python3
"""Opt-in NbC compact-transport numerical gate.

This diagnostic requires explicit external extxyz/POSCAR paths.  The six phase
modes below are the provisional 8A radius-audit fixture, not a production
EvaluationPolicy.  The script never writes or rewrites input structures.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from refsite_mlip.features import (
    ProbabilityMultipoleConfig,
    build_probability_multipoles,
    effective_probability_validation_tolerances,
)
from refsite_mlip.features.species import species_probabilities
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    EvalOTConfig,
    TrainSinkhornConfig,
    TransportSupportConfig,
    atom_site_displacements,
    minimum_image_diagnostics,
    solve_atom_vacancy_ot,
)


MODES = torch.tensor(
    [[-2, 2, 2], [2, -2, 2], [2, 2, -2], [4, 0, 0], [0, 4, 0], [0, 0, 4]],
    dtype=torch.long,
)
PHASE_STEPS = (0.7, 0.8, 0.9, 1.0)
PHASE_DAMPING = (2.0, 1.0, 0.5, 0.2)
SPECIES = (6, 41)


def _statistics(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(values.min()),
    }


def _sample_id(split: str, index: int, atoms) -> str:
    metadata = (
        atoms.info.get("sample_id")
        or atoms.info.get("config_type")
        or atoms.info.get("name")
    )
    return f"{split}:{index}" + ("" if metadata is None else f":{metadata}")


def _phase(atoms, reference_fractional, site_weights) -> torch.Tensor:
    positions = torch.as_tensor(atoms.positions, dtype=torch.float64)
    cell = torch.as_tensor(atoms.cell.array, dtype=torch.float64)
    numbers = torch.as_tensor(atoms.numbers, dtype=torch.long)
    atom_weights = torch.stack(
        ((numbers == 6).to(torch.float64), (numbers == 41).to(torch.float64)), 1
    )
    _, _, cross = typed_reciprocal_fields(
        positions,
        torch.zeros(3, dtype=torch.float64),
        cell,
        reference_fractional,
        atom_weights,
        site_weights,
        MODES,
        torch.ones(2, dtype=torch.float64),
    )
    initial = primary_phase_initialization(cross[:3], MODES[:3])
    return solve_training_phase(
        cross,
        MODES,
        torch.ones(6, dtype=torch.float64),
        initial,
        PHASE_STEPS,
        PHASE_DAMPING,
    ).phase.detach()


def _solve(distance64, displacement64, numbers, path, dtype, warmup=16):
    distance = distance64.to(dtype)
    result = solve_atom_vacancy_ot(
        distance.square() / (2.0 * 1.5**2),
        0.5,
        TRAIN_FIXED if path == "fixed" else EVAL_ADAPTIVE,
        "sinkhorn" if path == "fixed" else "hybrid",
        TrainSinkhornConfig(256)
        if path == "fixed"
        else EvalOTConfig(
            sinkhorn_iterations=warmup,
            convergence_tolerance=1.0e-6 if dtype == torch.float32 else 1.0e-12,
        ),
        support_config=TransportSupportConfig(
            kind="compact_c2", cutoff=4.0, switch_width=0.5, candidate_skin=0.2
        ),
        atom_distances=distance,
    )
    probabilities, indicator = species_probabilities(result.P, numbers, SPECIES)
    row_error = (result.P.sum(1) + result.q - 1).abs()
    column_error = (result.P.sum(0) - 1).abs()
    simplex_error = (probabilities.sum(1) + result.q - 1).abs()
    count_error = (probabilities.sum(0) - indicator.sum(0)).abs()
    errors = {
        "row": float(row_error.max()),
        "row_site": int(row_error.argmax()),
        "column": float(column_error.max()),
        "column_atom": int(column_error.argmax()),
        "simplex": float(simplex_error.max()),
        "simplex_site": int(simplex_error.argmax()),
        "species_count": float(count_error.max()),
        "species_channel": int(count_error.argmax()),
        "species_atomic_number": SPECIES[int(count_error.argmax())],
        "q_mass": float((result.q.sum() - 1).abs()),
    }
    tolerances = effective_probability_validation_tolerances(
        result.P, None
    )
    features = build_probability_multipoles(
        result.P,
        result.q,
        numbers,
        displacement64.to(dtype),
        ProbabilityMultipoleConfig(species_vocabulary=SPECIES),
    ).equivariant_features
    return result, errors, tolerances, features


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--poscar", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, nargs="+", default=(16, 24, 32, 64))
    return parser


def main() -> None:
    args = _parser().parse_args()
    from ase.io import read

    for path in (args.train, args.validation, args.poscar):
        if not path.is_file():
            raise FileNotFoundError(path)
    train = read(args.train, ":")
    validation = read(args.validation, ":")
    records = [("train", i, a) for i, a in enumerate(train)] + [
        ("validation", i, a) for i, a in enumerate(validation)
    ]
    parent = read(args.poscar)
    if len(parent) != 64 or set(parent.numbers) != {6, 41}:
        raise ValueError("this opt-in audit requires the verified 64-site NbC POSCAR")
    reference_fractional = torch.as_tensor(
        parent.get_scaled_positions(wrap=True), dtype=torch.float64
    )
    reference_numbers = torch.as_tensor(parent.numbers, dtype=torch.long)
    site_weights = torch.stack(
        (
            (reference_numbers == 6).to(torch.float64),
            (reference_numbers == 41).to(torch.float64),
        ),
        1,
    )
    geometry = []
    family_keys = []
    for split, index, atoms in records:
        if len(atoms) != 63 or tuple(sorted(set(atoms.numbers))) != SPECIES:
            raise ValueError("all audited frames must be 63-atom C/Nb structures")
        phase = _phase(atoms, reference_fractional, site_weights)
        cell = torch.as_tensor(atoms.cell.array, dtype=torch.float64)
        positions = torch.as_tensor(atoms.positions, dtype=torch.float64)
        references = (reference_fractional + phase) @ cell
        raw = positions.unsqueeze(0) - references.unsqueeze(1)
        mic = minimum_image_diagnostics(raw, cell, (True, True, True))
        displacement = mic.displacement
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        geometry.append(
            (
                _sample_id(split, index, atoms),
                cell,
                torch.as_tensor(atoms.numbers, dtype=torch.long),
                distance,
                displacement,
                mic.unique_image_gap,
            )
        )
        family_keys.append(tuple(np.round(atoms.cell.array.reshape(-1), 7)))
    families = {key: i for i, key in enumerate(sorted(set(family_keys)))}

    started = time.time()
    outputs = {}
    path_report = {}
    for path in ("fixed", "adaptive"):
        for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
            key = f"{path}_{name}"
            records_out = []
            P, q, feature = [], [], []
            fallback = Counter()
            for b, (sample, _, numbers, distance, displacement, _) in enumerate(geometry):
                result, errors, tolerances, features = _solve(
                    distance, displacement, numbers, path, dtype
                )
                if result.fallback_used:
                    fallback[result.failure_reason or "UNKNOWN"] += 1
                records_out.append(
                    {
                        "frame": b,
                        "sample_id": sample,
                        "cell_family": families[family_keys[b]],
                        **errors,
                        "fallback": result.fallback_used,
                        "fallback_reason": result.failure_reason,
                        "warmup": result.warmup_sinkhorn_iterations,
                        "newton": result.newton_iterations,
                        "cg": result.cg_iterations,
                        "effective_validation_tolerances": tolerances,
                    }
                )
                P.append(result.P.detach().double().numpy())
                q.append(result.q.detach().double().numpy())
                feature.append(features.detach().double().numpy())
            outputs[key] = tuple(np.stack(x) for x in (P, q, feature))
            path_report[key] = {
                "fallback_reasons": dict(fallback),
                "invariants": {
                    metric: _statistics([row[metric] for row in records_out])
                    for metric in ("row", "column", "simplex", "species_count", "q_mass")
                },
                "frames": records_out,
            }

    oracle_errors = {}
    for path in ("fixed", "adaptive"):
        f32 = outputs[f"{path}_float32"]
        f64 = outputs[f"{path}_float64"]
        oracle_errors[path] = {}
        for label, a, b in zip(("P", "q", "multipole"), f32, f64):
            per_frame = np.max(np.abs(a - b).reshape(len(records), -1), 1)
            worst = int(np.argmax(per_frame))
            oracle_errors[path][label] = {
                **_statistics(per_frame),
                "worst_frame": worst,
                "worst_sample_id": geometry[worst][0],
                "worst_cell_family": families[family_keys[worst]],
            }

    support_report = {name: [] for name in ("r_on", "r_off", "r_candidate")}
    mic_active = []
    atom_degree = []
    for _, _, _, distance, _, image_gap in geometry:
        for name, radius in (("r_on", 3.5), ("r_off", 4.0), ("r_candidate", 4.2)):
            support_report[name].append(float(torch.min(torch.abs(distance - radius))))
        active = distance < 4.0
        atom_degree.extend(int(x) for x in active.sum(0).tolist())
        mic_active.append(float(image_gap[active].min()))

    sweep = {}
    for warmup in args.warmups:
        begin = time.time()
        fallback, P_error, q_error = Counter(), 0.0, 0.0
        fixed_P, fixed_q, _ = outputs["fixed_float64"]
        for b, (sample, _, numbers, distance, displacement, _) in enumerate(geometry):
            result, _, _, _ = _solve(
                distance, displacement, numbers, "adaptive", torch.float64, warmup
            )
            if result.fallback_used:
                fallback[result.failure_reason or "UNKNOWN"] += 1
            P_error = max(P_error, float(np.max(np.abs(result.P.detach().numpy() - fixed_P[b]))))
            q_error = max(q_error, float(np.max(np.abs(result.q.detach().numpy() - fixed_q[b]))))
        sweep[str(warmup)] = {
            "fallback_reasons": dict(fallback),
            "max_P_error_vs_fixed": P_error,
            "max_q_error_vs_fixed": q_error,
            "runtime_seconds": time.time() - begin,
        }

    report = {
        "contract": {
            "frames": len(records),
            "M": 64,
            "N": 63,
            "K": 1,
            "r_on": 3.5,
            "r_off": 4.0,
            "r_candidate": 4.2,
            "phase_status": "provisional six-mode radius diagnostic; not a production EvaluationPolicy",
        },
        "paths": path_report,
        "float32_vs_float64": oracle_errors,
        "support": {
            **{name: _statistics(values) for name, values in support_report.items()},
            "active_atom_degree": _statistics(atom_degree),
            "maximum_matching_size": {"min": 63, "max": 63},
            "total_support_feasible_frames": len(records),
            "active_MIC_image_gap": _statistics(mic_active),
            "matching_note": "the solver preflight verified exact matching and total support for every solve",
        },
        "adaptive_warmup_sweep_float64": sweep,
        "roundoff": {
            "float32_epsilon": torch.finfo(torch.float32).eps,
            "float64_epsilon": torch.finfo(torch.float64).eps,
            "automatic_rule": "float64 validation accumulation plus dtype/size pairwise-reduction-depth bound",
        },
        "runtime_seconds": time.time() - started,
    }
    payload = json.dumps(report, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
