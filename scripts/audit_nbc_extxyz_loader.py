#!/usr/bin/env python3
"""Opt-in, read-only audit of the production extxyz loader on NbC data.

Every external path is an explicit command-line argument.  The caller builds
and registers the reference template; :func:`load_extxyz_samples` never reads
the POSCAR or constructs graph/stabilizer state.  The six reciprocal modes and
unit weights below remain provisional template-audit inputs, not an approved
production phase or evaluation policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from refsite_mlip.data import (
    ExtXYZLoadConfig,
    PhaseSpecification,
    TemplateRegistry,
    build_reference_template_from_poscar,
    collate_structure_samples,
    load_extxyz_samples,
    nbc_rocksalt_template_builder_config,
)
from refsite_mlip.training import fingerprint_batch_sequence


def _phase() -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.tensor(
            [
                [-2, 2, 2],
                [2, -2, 2],
                [2, 2, -2],
                [4, 0, 0],
                [0, 4, 0],
                [0, 0, 4],
            ],
            dtype=torch.long,
        ),
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )


def _maximum_difference(sample, atoms) -> dict[str, float]:
    from ase.stress import voigt_6_to_full_3x3_stress

    expected = {
        "positions": np.asarray(atoms.positions),
        "atomic_numbers": np.asarray(atoms.numbers),
        "cell": np.asarray(atoms.cell.array),
        "energy": np.asarray(atoms.calc.results["energy"]),
        "forces": np.asarray(atoms.calc.results["forces"]),
        "stress": np.asarray(
            voigt_6_to_full_3x3_stress(atoms.calc.results["stress"])
        ),
    }
    result = {}
    for name, reference in expected.items():
        loaded = getattr(sample, name).detach().cpu().numpy()
        result[name] = float(np.max(np.abs(loaded - reference)))
    return result


def _audit_split(path: Path, prefix: str, registry, domain_333):
    from ase.io import read

    config = ExtXYZLoadConfig(
        source_path=str(path),
        sample_id_prefix=prefix,
        template_id="nbc_rocksalt_222_v1",
        require_energy=True,
        require_forces=True,
        require_stress=True,
        dtype=torch.float64,
        device="cpu",
    )
    result = load_extxyz_samples(config, registry)
    repeated = load_extxyz_samples(config, registry)
    if result.diagnostics != repeated.diagnostics:
        raise RuntimeError("repeated loader diagnostics are not deterministic")
    for first, second in zip(result.samples, repeated.samples):
        for name in (
            "positions",
            "atomic_numbers",
            "cell",
            "pbc",
            "origin",
            "energy",
            "forces",
            "stress",
        ):
            if not torch.equal(getattr(first, name), getattr(second, name)):
                raise RuntimeError(f"repeated loader tensor differs: {name}")

    atoms_frames = read(path, index=":", format="extxyz")
    indices = sorted({0, len(result.samples) // 2, len(result.samples) - 1})
    comparisons = {
        str(index): _maximum_difference(result.samples[index], atoms_frames[index])
        for index in indices
    }
    rejected_333 = 0
    for sample in result.samples:
        try:
            domain_333.validate_atomic_numbers(
                sample.atomic_numbers,
                template_id="nbc_rocksalt_333_v1",
                sample_id=sample.sample_id,
            )
        except ValueError:
            rejected_333 += 1

    batch = collate_structure_samples(result.samples, registry)
    return {
        "diagnostics": result.diagnostics.to_dict(),
        "ase_comparison_indices": indices,
        "ase_max_absolute_difference": comparisons,
        "accepted_by_222": len(result.samples),
        "rejected_by_333": rejected_333,
        "repeated_load_exact": True,
        "batch": {
            "num_structures": batch.num_structures,
            "num_atoms": batch.num_atoms,
            "all_energy_mask": bool(torch.all(batch.energy_mask)),
            "all_force_present": bool(torch.all(batch.force_present)),
            "all_stress_present": bool(torch.all(batch.stress_present)),
            "implicit_force_masks": not bool(
                torch.any(batch.force_mask_provided)
            ),
            "implicit_stress_masks": not bool(
                torch.any(batch.stress_mask_provided)
            ),
            "checkpoint_data_fingerprint": fingerprint_batch_sequence(
                (batch,), split_name=prefix
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar-222", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config_222 = nbc_rocksalt_template_builder_config((2, 2, 2))
    built = build_reference_template_from_poscar(
        args.poscar_222,
        config=config_222,
        phase_specification=_phase(),
    )
    registry = TemplateRegistry()
    registry.add(built.template)
    domain_333 = nbc_rocksalt_template_builder_config(
        (3, 3, 3)
    ).strict_domain

    report = {
        "scope": (
            "read-only loader audit; the caller supplies a prebuilt registry; "
            "phase modes and unit weights are provisional and are not a "
            "production EvaluationPolicy"
        ),
        "template": built.diagnostics.to_dict(),
        "registry_fingerprint": registry.fingerprint,
        "splits": {
            "train": _audit_split(args.train, "train", registry, domain_333),
            "validation": _audit_split(
                args.validation, "validation", registry, domain_333
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
