#!/usr/bin/env python3
"""Opt-in, read-only NbC POSCAR/domain audit.

The six reciprocal modes and unit weights used here are explicitly
``provisional``.  This script does not create a production EvaluationPolicy or
an extxyz loader and never writes the input files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from refsite_mlip.data import (
    PhaseSpecification,
    TemplateRegistry,
    build_reference_template_from_poscar,
    nbc_rocksalt_template_builder_config,
)
from refsite_mlip.models import TemplateExecutionContext


def _phase(size: int) -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.tensor(
            [
                [-size, size, size],
                [size, -size, size],
                [size, size, -size],
                [2 * size, 0, 0],
                [0, 2 * size, 0],
                [0, 0, 2 * size],
            ],
            dtype=torch.long,
        ),
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )


def _audit_split(path: Path, template_222, template_333):
    from ase.io import iread

    accepted_222 = 0
    rejected_333 = 0
    unexpected_222 = []
    unexpected_333 = []
    composition = Counter()
    atom_counts = Counter()
    vacancy_masses = Counter()
    strains = []
    first = None
    for index, atoms in enumerate(iread(str(path), index=":")):
        if first is None:
            first = atoms.copy()
        numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float64)
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool)
        key = tuple(
            int(torch.sum(numbers == species)) for species in (6, 41)
        )
        composition[key] += 1
        atom_counts[len(atoms)] += 1
        try:
            validation = template_222.validate_structure(
                numbers,
                cell=cell,
                pbc=pbc,
                sample_id=f"{path.name}:{index}",
            )
            accepted_222 += 1
            vacancy_masses[validation.vacancy_mass] += 1
            strains.append(validation.maximum_strain_seen)
        except Exception as error:  # report every observed domain failure
            unexpected_222.append(
                {"frame": index, "type": type(error).__name__, "message": str(error)}
            )
        try:
            template_333.validate_structure(
                numbers,
                cell=cell,
                pbc=pbc,
                sample_id=f"{path.name}:{index}",
            )
        except ValueError:
            rejected_333 += 1
        else:
            unexpected_333.append(index)

    permutation_stable = False
    wrapping_stable = False
    if first is not None:
        numbers = torch.tensor(first.get_atomic_numbers(), dtype=torch.long)
        direct = template_222.validate_structure(numbers)
        permuted = template_222.validate_structure(torch.flip(numbers, dims=(0,)))
        permutation_stable = direct.to_dict() == permuted.to_dict()
        scaled = first.get_scaled_positions(wrap=False)
        shifted = scaled.copy()
        shifted[:, 0] += 1.0
        first.set_scaled_positions(shifted)
        wrapped_numbers = torch.tensor(first.get_atomic_numbers(), dtype=torch.long)
        wrapping_stable = (
            direct.to_dict()
            == template_222.validate_structure(wrapped_numbers).to_dict()
        )

    return {
        "path": str(path.resolve()),
        "frames": sum(atom_counts.values()),
        "atom_counts": {str(key): value for key, value in sorted(atom_counts.items())},
        "composition_C_Nb": {
            f"{key[0]},{key[1]}": value for key, value in sorted(composition.items())
        },
        "vacancy_masses": {
            str(key): value for key, value in sorted(vacancy_masses.items())
        },
        "accepted_by_222": accepted_222,
        "rejected_by_333": rejected_333,
        "unexpected_222": unexpected_222,
        "unexpected_333": unexpected_333,
        "maximum_strain": max(strains) if strains else None,
        "permutation_decision_stable": permutation_stable,
        "wrapping_decision_stable": wrapping_stable,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar-222", type=Path, required=True)
    parser.add_argument("--poscar-333", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config_222 = nbc_rocksalt_template_builder_config((2, 2, 2))
    config_333 = nbc_rocksalt_template_builder_config((3, 3, 3))
    result_222 = build_reference_template_from_poscar(
        args.poscar_222,
        config=config_222,
        phase_specification=_phase(2),
    )
    result_333 = build_reference_template_from_poscar(
        args.poscar_333,
        config=config_333,
        phase_specification=_phase(3),
    )
    registry = TemplateRegistry()
    registry.add(result_222.template)
    registry.add(result_333.template)
    contexts = {
        result.template.template_id: TemplateExecutionContext.from_reference_template(
            result.template, avg_num_neighbors=result.config.avg_num_neighbors
        )
        for result in (result_222, result_333)
    }
    for context in contexts.values():
        context.validate_fingerprint()

    report = {
        "scope": (
            "read-only domain/template compatibility audit; phase modes and unit "
            "weights are provisional and not a production EvaluationPolicy"
        ),
        "templates": {
            "222": result_222.diagnostics.to_dict(),
            "333": result_333.diagnostics.to_dict(),
        },
        "registry_fingerprint": registry.fingerprint,
        "context_fingerprints": {
            key: value.fingerprint for key, value in sorted(contexts.items())
        },
        "splits": {
            "train": _audit_split(
                args.train, result_222.template, result_333.template
            ),
            "validation": _audit_split(
                args.validation, result_222.template, result_333.template
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

