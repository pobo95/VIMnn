#!/usr/bin/env python3
"""Opt-in dense-masked versus edge-list TRAIN_FIXED NbC audit.

All external paths are explicit arguments.  The six reciprocal modes are the
provisional radius-audit fixture and are not a production EvaluationPolicy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from refsite_mlip.features import (
    ProbabilityMultipoleConfig,
    build_probability_multipoles,
    build_sparse_probability_multipoles,
)
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.transport import (
    TRAIN_FIXED,
    TrainSinkhornConfig,
    TransportSupportConfig,
    atom_site_displacements,
    build_compact_transport_edges,
    materialize_dense_plan,
    solve_atom_vacancy_ot,
    solve_sparse_sinkhorn_train_fixed,
)


MODES = torch.tensor(
    [[-2, 2, 2], [2, -2, 2], [2, 2, -2], [4, 0, 0], [0, 4, 0], [0, 0, 4]],
    dtype=torch.long,
)
SPECIES = (6, 41)


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _phase(atoms, reference_fractional, site_weights):
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
        (0.7, 0.8, 0.9, 1.0),
        (2.0, 1.0, 0.5, 0.2),
    ).phase.detach()


def _configs():
    dense = TransportSupportConfig(
        kind="compact_c2", cutoff=4.0, switch_width=0.5, candidate_skin=0.2
    )
    edge = TransportSupportConfig(
        kind="compact_c2",
        cutoff=4.0,
        switch_width=0.5,
        candidate_skin=0.2,
        backend="edge_list",
    )
    return dense, edge


def _solve(displacements64, numbers, dtype):
    dense_config, edge_config = _configs()
    displacement = displacements64.to(dtype)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    cost = distance.square() / (2.0 * 1.5**2)
    started = time.perf_counter()
    dense = solve_atom_vacancy_ot(
        cost,
        0.5,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(256),
        support_config=dense_config,
        atom_distances=distance,
    )
    dense_seconds = time.perf_counter() - started
    started = time.perf_counter()
    edges = build_compact_transport_edges(
        displacement,
        epsilon_ot=0.5,
        ell_ot=1.5,
        config=edge_config,
    )
    sparse = solve_sparse_sinkhorn_train_fixed(edges, TrainSinkhornConfig(256))
    sparse_seconds = time.perf_counter() - started
    plan = materialize_dense_plan(sparse).plan
    feature_config = ProbabilityMultipoleConfig(SPECIES)
    dense_feature = build_probability_multipoles(
        dense.P, dense.q, numbers, displacement, feature_config
    ).equivariant_features
    sparse_feature = build_sparse_probability_multipoles(
        sparse.edge_plan, sparse.q, sparse.edges, numbers, feature_config
    ).equivariant_features
    edge_species = torch.stack(
        (
            (numbers[edges.atom_index] == 6).to(dtype),
            (numbers[edges.atom_index] == 41).to(dtype),
        ),
        1,
    )
    sparse_species = torch.zeros(
        (edges.num_sites, 2), dtype=dtype
    ).index_add(0, edges.site_index, sparse.edge_plan[:, None] * edge_species)
    expected_species = torch.stack(((numbers == 6).sum(), (numbers == 41).sum())).to(dtype)
    return {
        "P_error": float((plan - dense.P).abs().max()),
        "q_error": float((sparse.q - dense.q).abs().max()),
        "multipole_error": float((sparse_feature - dense_feature).abs().max()),
        "dense_residual": max(float(dense.row_residual), float(dense.column_residual)),
        "sparse_residual": max(float(sparse.row_residual), float(sparse.column_residual)),
        "q_mass_error": float((sparse.q.sum() - sparse.q.new_tensor(float(edges.num_vacancies))).abs()),
        "species_count_error": float((sparse_species.sum(0) - expected_species).abs().max()),
        "site_simplex_error": float((sparse_species.sum(1) + sparse.q - 1.0).abs().max()),
        "candidate_edges": edges.num_candidate_edges,
        "active_edges": edges.num_active_edges,
        "dense_pairs": edges.num_sites * edges.num_atoms,
        "num_sites": edges.num_sites,
        "num_atoms": edges.num_atoms,
        "dense_seconds": dense_seconds,
        "sparse_seconds": sparse_seconds,
        "matching": edges.support_diagnostics.maximum_atom_matching_size,
        "total_support": edges.support_diagnostics.total_support_feasible,
        "cutoff_gap": edges.support_diagnostics.cutoff_boundary_gap,
        "candidate_gap": edges.support_diagnostics.candidate_boundary_gap,
        "dense_result_elements": sum(
            value.numel()
            for value in (dense.gamma, dense.P, dense.q, dense.f, dense.g)
        ),
        "sparse_result_elements": sum(
            value.numel()
            for value in (sparse.edge_plan, sparse.q, sparse.f, sparse.g)
        ),
        "sparse_index_elements": sum(
            value.numel()
            for value in (
                edges.site_index,
                edges.atom_index,
                edges.atom_major_permutation,
                edges.site_ptr,
                edges.atom_ptr,
            )
        ),
        "sparse_edge_float_elements": sum(
            value.numel()
            for value in (
                edges.displacements,
                edges.distances,
                edges.switch,
                edges.log_kernel,
            )
        ),
        "dense_result_bytes": sum(
            value.numel() * value.element_size()
            for value in (dense.gamma, dense.P, dense.q, dense.f, dense.g)
        ),
        "sparse_retained_bytes": sum(
            value.numel() * value.element_size()
            for value in (
                sparse.edge_plan,
                sparse.q,
                sparse.f,
                sparse.g,
                edges.site_index,
                edges.atom_index,
                edges.atom_major_permutation,
                edges.site_ptr,
                edges.atom_ptr,
                edges.displacements,
                edges.distances,
                edges.switch,
                edges.log_kernel,
                edges.active,
            )
        ),
    }


def _aggregate(rows):
    metrics = (
        "P_error",
        "q_error",
        "multipole_error",
        "dense_residual",
        "sparse_residual",
        "q_mass_error",
        "species_count_error",
        "site_simplex_error",
        "candidate_edges",
        "active_edges",
        "dense_seconds",
        "sparse_seconds",
        "cutoff_gap",
        "candidate_gap",
    )
    return {
        key: _stats([row[key] for row in rows]) for key in metrics
    } | {
        "failures": sum(
            not row["total_support"] or row["matching"] != row["num_atoms"] for row in rows
        ),
        "element_accounting_first_frame": {
            key: rows[0][key]
            for key in (
                "dense_pairs",
                "dense_result_elements",
                "sparse_result_elements",
                "sparse_index_elements",
                "sparse_edge_float_elements",
                "dense_result_bytes",
                "sparse_retained_bytes",
            )
        },
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--poscar-222", required=True, type=Path)
    parser.add_argument("--poscar-333", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    args = _parser().parse_args()
    from ase.io import read

    for path in (args.train, args.validation, args.poscar_222):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = read(args.poscar_222)
    if len(parent) != 64 or set(parent.numbers) != set(SPECIES):
        raise ValueError("POSCAR_222 must be the verified 64-site C/Nb parent")
    reference_fractional = torch.as_tensor(
        parent.get_scaled_positions(wrap=True), dtype=torch.float64
    )
    reference_numbers = torch.as_tensor(parent.numbers, dtype=torch.long)
    site_weights = torch.stack(
        ((reference_numbers == 6).double(), (reference_numbers == 41).double()), 1
    )
    frames = read(args.train, ":") + read(args.validation, ":")
    if len(frames) != 315:
        raise ValueError(f"expected 315 frames, found {len(frames)}")
    geometries = []
    for atoms in frames:
        if len(atoms) != 63 or set(atoms.numbers) != set(SPECIES):
            raise ValueError("all frames must be verified 63-atom C/Nb structures")
        phase = _phase(atoms, reference_fractional, site_weights)
        cell = torch.as_tensor(atoms.cell.array, dtype=torch.float64)
        positions = torch.as_tensor(atoms.positions, dtype=torch.float64)
        references = (reference_fractional + phase) @ cell
        geometries.append(
            (
                atom_site_displacements(positions, references, cell, (True, True, True)),
                torch.as_tensor(atoms.numbers, dtype=torch.long),
            )
        )
    report = {
        "contract": {
            "frames": len(frames),
            "M": 64,
            "N": 63,
            "K": 1,
            "r_on": 3.5,
            "r_off": 4.0,
            "r_candidate": 4.2,
            "phase_status": "provisional six-mode transport audit; not a production policy",
        },
        "frames": {},
    }
    for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        rows = [_solve(displacement, numbers, dtype) for displacement, numbers in geometries]
        report["frames"][name] = _aggregate(rows)

    if args.poscar_333 is not None:
        if not args.poscar_333.is_file():
            raise FileNotFoundError(args.poscar_333)
        parent333 = read(args.poscar_333)
        if len(parent333) != 216 or set(parent333.numbers) != set(SPECIES):
            raise ValueError("POSCAR_333 must be the verified 216-site C/Nb parent")
        removed = next(i for i, number in enumerate(parent333.numbers) if number == 6)
        atom_indices = [i for i in range(len(parent333)) if i != removed]
        cell = torch.as_tensor(parent333.cell.array, dtype=torch.float64)
        references = torch.as_tensor(parent333.positions, dtype=torch.float64)
        positions = references[atom_indices]
        numbers = torch.as_tensor(parent333.numbers[atom_indices], dtype=torch.long)
        displacement = atom_site_displacements(
            positions, references, cell, (True, True, True)
        )
        report["synthetic_333"] = {
            name: _solve(displacement, numbers, dtype)
            for name, dtype in (("float64", torch.float64), ("float32", torch.float32))
        }
        report["synthetic_333"]["scope"] = (
            "transport-only synthetic C vacancy; current N=63 data are not assigned"
        )

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
