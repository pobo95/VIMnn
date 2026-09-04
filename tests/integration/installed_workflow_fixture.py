"""External fixture generator for the installed-wheel workflow gate.

The integration test copies this file outside the source checkout and executes
it with the wheel environment's Python.  Consequently every ``refsite_mlip``
import below resolves from the installed wheel rather than ``src/``.

This is test support, not a pytest test module.  Run it as::

    python installed_workflow_fixture.py /path/to/external/root

It writes a deterministic family of semantically equivalent scratch configs
and prints one canonical JSON manifest to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceTemplateBuilderConfig,
    StrictTemplateDomain,
    build_reference_template_from_poscar,
)
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import EvaluationPolicy, PotentialConfig
from refsite_mlip.training import (
    AtomicBaselineConfig,
    CheckpointedFitConfig,
    FitConfig,
    LossConfig,
    ModelSelectionConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
)
from refsite_mlip.transport import TransportSupportConfig


LATTICE = 4.482314244155584
ALPHA_TEMPLATE_ID = "alpha-111"
ZETA_TEMPLATE_ID = "zeta-211"
INITIALIZATION_SEED = 20260904
TRAINING_SEED = 1701


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _reference(shape: tuple[int, int, int]):
    return bulk("NbC", "rocksalt", a=LATTICE, cubic=True).repeat(shape)


def _remove_first_carbon(atoms):
    result = atoms.copy()
    carbon_index = next(
        index
        for index, atomic_number in enumerate(result.numbers)
        if int(atomic_number) == 6
    )
    del result[carbon_index]
    return result


def _labeled(atoms, *, energy: float, offset: float, template_id: str):
    result = atoms.copy()
    components = np.arange(len(result) * 3, dtype=np.float64).reshape(-1, 3)
    forces = (components + float(offset)) / 100.0
    stress = np.asarray(
        [
            0.10 + offset / 100.0,
            0.20 + offset / 100.0,
            0.30 + offset / 100.0,
            0.04 + offset / 1000.0,
            0.05 + offset / 1000.0,
            0.06 + offset / 1000.0,
        ],
        dtype=np.float64,
    )
    result.calc = SinglePointCalculator(
        result,
        energy=float(energy),
        forces=forces,
        stress=stress,
    )
    result.info["template"] = template_id
    result.info["fixture_role"] = (
        f"{template_id}:{'pristine' if len(result) in (8, 16) else 'vacancy'}"
    )
    result.arrays["input_order"] = np.arange(len(result), dtype=np.int64)
    return result


def _phase_111() -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.tensor(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ],
            dtype=torch.long,
        ),
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )


def _phase_211() -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.tensor(
            [
                [-2, 1, 1],
                [2, -1, 1],
                [2, 1, -1],
                [4, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ],
            dtype=torch.long,
        ),
        mode_weights=torch.ones(6, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )


def _builder(
    *,
    template_id: str,
    shape: tuple[int, int, int],
    sites: int,
    per_species: int,
    stabilizer_size: int,
) -> ReferenceTemplateBuilderConfig:
    return ReferenceTemplateBuilderConfig(
        template_id=template_id,
        strict_domain=StrictTemplateDomain(
            reference_site_count=sites,
            supercell_shape=shape,
            species_vocabulary=(6, 41),
            reference_composition=(per_species, per_species),
            allowed_compositions=(
                (per_species, per_species),
                (per_species - 1, per_species),
            ),
            allowed_num_atoms=(sites, sites - 1),
            allowed_vacancy_masses=(0, 1),
        ),
        site_type_ids=(0, 1),
        expected_stabilizer_size=stabilizer_size,
    )


def _potential(
    *,
    transport_backend: str = "dense",
    candidate_backend: str = "dense",
) -> PotentialConfig:
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=2,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        probability_tolerance=1.0e-8,
        site_type_vocabulary=(0, 1),
    )
    higher_body = HigherBodyConfig(
        irreps_feature="2x0e+4x0e+4x1o+4x2e",
        species_count=2,
        site_type_count=2,
        site_type_embedding_dim=2,
        n_correlation_channels=1,
        lmax=2,
        radial_feature_dim=3,
        radial_hidden_dims=(4,),
        avg_num_neighbors=6.0,
        cutoff=3.0,
        edge_length_scale=1.0,
    )
    return PotentialConfig(
        species_vocabulary=(6, 41),
        num_layers=1,
        feature=feature,
        higher_body=higher_body,
        transport_support=TransportSupportConfig(
            kind="compact_c2",
            cutoff=4.0,
            switch_width=0.5,
            candidate_skin=0.2,
            backend=transport_backend,
            candidate_backend=candidate_backend,
        ),
    )


def _policy(template) -> EvaluationPolicy:
    # These four representatives are non-equivalent for both typed rocksalt
    # stabilizers.  The exact pristine/vacancy structures below were exercised
    # with adaptive energy, forces and stress before this gate was introduced.
    return EvaluationPolicy(
        template_id=template.template_id,
        template_fingerprint=template.fingerprint,
        candidate_offsets=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=torch.float64,
        ),
        phase_step_schedule=(0.7, 0.8, 0.9, 1.0),
        phase_damping_schedule=(2.0, 1.0, 0.5, 0.2),
        minimum_objective_gap_absolute=1.0e-12,
        minimum_cross_amplitude_absolute=1.0e-12,
        minimum_atomic_amplitude_absolute=1.0e-12,
        minimum_reference_amplitude_absolute=1.0e-12,
        minimum_curvature=1.0e-12,
        maximum_condition=1.0e12,
        maximum_gradient_norm=1.0e-5,
        equivalence_tolerance=1.0e-10,
    )


def _write_xyz(path: Path, frames) -> None:
    write(path, list(frames), format="extxyz")


def _write_case(
    directory: Path,
    *,
    max_epochs: int,
    transport_backend: str,
    candidate_backend: str,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    alpha_reference = _reference((1, 1, 1))
    zeta_reference = _reference((2, 1, 1))
    alpha_poscar = directory / "alpha.POSCAR"
    zeta_poscar = directory / "zeta.POSCAR"
    write(alpha_poscar, alpha_reference, format="vasp", direct=True)
    write(zeta_poscar, zeta_reference, format="vasp", direct=True)

    alpha_builder = _builder(
        template_id=ALPHA_TEMPLATE_ID,
        shape=(1, 1, 1),
        sites=8,
        per_species=4,
        stabilizer_size=4,
    )
    zeta_builder = _builder(
        template_id=ZETA_TEMPLATE_ID,
        shape=(2, 1, 1),
        sites=16,
        per_species=8,
        stabilizer_size=8,
    )
    alpha_phase = _phase_111()
    zeta_phase = _phase_211()
    alpha_template = build_reference_template_from_poscar(
        alpha_poscar,
        config=alpha_builder,
        phase_specification=alpha_phase,
    ).template
    zeta_template = build_reference_template_from_poscar(
        zeta_poscar,
        config=zeta_builder,
        phase_specification=zeta_phase,
    ).template

    # Train includes K=0 for M=8 and K=1 for M=16.  Validation reverses the
    # vacancy assignment, exercising both accepted domains for both templates.
    train_alpha = _labeled(
        alpha_reference,
        energy=-8.0,
        offset=1.0,
        template_id=ALPHA_TEMPLATE_ID,
    )
    train_zeta = _labeled(
        _remove_first_carbon(zeta_reference),
        energy=-15.0,
        offset=2.0,
        template_id=ZETA_TEMPLATE_ID,
    )
    validation_alpha = _labeled(
        _remove_first_carbon(alpha_reference),
        energy=-6.75,
        offset=3.0,
        template_id=ALPHA_TEMPLATE_ID,
    )
    validation_zeta = _labeled(
        zeta_reference,
        energy=-15.75,
        offset=4.0,
        template_id=ZETA_TEMPLATE_ID,
    )

    paths = {
        "train_alpha": directory / "train-alpha.xyz",
        "train_zeta": directory / "train-zeta.xyz",
        "validation_alpha": directory / "validation-alpha.xyz",
        "validation_zeta": directory / "validation-zeta.xyz",
        "mixed_labeled": directory / "mixed-labeled.xyz",
    }
    _write_xyz(paths["train_alpha"], (train_alpha,))
    _write_xyz(paths["train_zeta"], (train_zeta,))
    _write_xyz(paths["validation_alpha"], (validation_alpha,))
    _write_xyz(paths["validation_zeta"], (validation_zeta,))
    _write_xyz(
        paths["mixed_labeled"],
        (train_alpha, train_zeta, validation_alpha, validation_zeta),
    )

    templates = [
        {
            "poscar_path": zeta_poscar.name,
            "builder": zeta_builder.to_dict(),
            "phase_specification": zeta_phase.to_dict(),
            "evaluation_policy": _policy(zeta_template).to_dict(),
        },
        {
            "poscar_path": alpha_poscar.name,
            "builder": alpha_builder.to_dict(),
            "phase_specification": alpha_phase.to_dict(),
            "evaluation_policy": _policy(alpha_template).to_dict(),
        },
    ]
    scheduler = SchedulerConfig(
        kind="reduce_on_plateau",
        monitor="total_loss",
        mode="min",
        factor=0.5,
        patience=0,
        threshold=1.0e6,
        threshold_mode="abs",
        cooldown=0,
        min_lr=0.0,
        eps=1.0e-8,
    )
    selection = ModelSelectionConfig(
        monitor="total_loss",
        mode="min",
        min_delta=1.0e6,
        early_stopping_patience=10,
    )
    payload = {
        "schema_version": "refsite_training_run_config_v2",
        "model_source": {
            "kind": "scratch",
            "initialization_seed": INITIALIZATION_SEED,
            "potential": _potential(
                transport_backend=transport_backend,
                candidate_backend=candidate_backend,
            ).to_dict(),
            "species_alignment_weights": [[1.0, -0.5], [-1.0, 2.0]],
            "reference_templates": templates,
            "default_template_id": ALPHA_TEMPLATE_ID,
        },
        "radii": {"r_ot": 4.0, "r_mp": 3.0},
        "data": {
            "train": [
                {
                    "path": paths["train_alpha"].name,
                    "template_id": ALPHA_TEMPLATE_ID,
                },
                {
                    "path": paths["train_zeta"].name,
                    "template_id": ZETA_TEMPLATE_ID,
                },
            ],
            "validation": [
                {
                    "path": paths["validation_alpha"].name,
                    "template_id": ALPHA_TEMPLATE_ID,
                },
                {
                    "path": paths["validation_zeta"].name,
                    "template_id": ZETA_TEMPLATE_ID,
                },
            ],
            "batch_size": 2,
            "validation_batch_size": 2,
            "shuffle": False,
        },
        "runtime": {
            "device": "cpu",
            "dtype": "float64",
            "seed": TRAINING_SEED,
        },
        "loss": LossConfig(
            energy_weight=1.0,
            force_weight=0.05,
            stress_weight=0.02,
        ).to_dict(),
        "baseline": AtomicBaselineConfig(rank_policy="error").to_dict(),
        "optimizer": OptimizerConfig(
            learning_rate=5.0e-4,
            weight_decay=0.0,
        ).to_dict(),
        "train_step": TrainStepConfig().to_dict(),
        "validation_step": ValidationStepConfig().to_dict(),
        "scheduler": scheduler.to_dict(),
        "selection": selection.to_dict(),
        "fit": FitConfig(max_epochs=max_epochs).to_dict(),
        "checkpointed_fit": CheckpointedFitConfig().to_dict(),
        "output_directory": "run-output",
    }
    config_path = directory / "run.json"
    _canonical_write(config_path, payload)
    all_inputs = (
        alpha_poscar,
        zeta_poscar,
        *paths.values(),
        config_path,
    )
    return {
        "directory": str(directory.resolve()),
        "config": str(config_path.resolve()),
        "output_directory": str((directory / "run-output").resolve()),
        "mixed_labeled": str(paths["mixed_labeled"].resolve()),
        "max_epochs": max_epochs,
        "input_sha256": {
            path.name: _sha256(path) for path in sorted(all_inputs)
        },
    }


def generate(
    root: Path,
    *,
    transport_backend: str = "dense",
    candidate_backend: str = "dense",
) -> dict[str, Any]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = {
        "continuous": _write_case(
            root / "continuous",
            max_epochs=2,
            transport_backend=transport_backend,
            candidate_backend=candidate_backend,
        ),
        "split": _write_case(
            root / "split",
            max_epochs=1,
            transport_backend=transport_backend,
            candidate_backend=candidate_backend,
        ),
    }
    manifest = {
        "schema_version": "refsite_installed_workflow_fixture_v1",
        "species_vocabulary": [6, 41],
        "template_ids": [ALPHA_TEMPLATE_ID, ZETA_TEMPLATE_ID],
        "default_template_id": ALPHA_TEMPLATE_ID,
        "template_site_counts": {
            ALPHA_TEMPLATE_ID: 8,
            ZETA_TEMPLATE_ID: 16,
        },
        "train_structures": [
            {"template_id": ALPHA_TEMPLATE_ID, "M": 8, "N": 8, "K": 0},
            {"template_id": ZETA_TEMPLATE_ID, "M": 16, "N": 15, "K": 1},
        ],
        "validation_structures": [
            {"template_id": ALPHA_TEMPLATE_ID, "M": 8, "N": 7, "K": 1},
            {"template_id": ZETA_TEMPLATE_ID, "M": 16, "N": 16, "K": 0},
        ],
        "labels": ["energy", "forces", "stress"],
        "initialization_seed": INITIALIZATION_SEED,
        "training_seed": TRAINING_SEED,
        "device": "cpu",
        "dtype": "float64",
        "template_key": "template",
        "transport_backend": transport_backend,
        "candidate_backend": candidate_backend,
        "cases": cases,
    }
    manifest_path = root / "fixture-manifest.json"
    _canonical_write(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--transport-backend",
        choices=("dense", "edge_list"),
        default="dense",
    )
    parser.add_argument(
        "--candidate-backend",
        choices=("dense", "blocked"),
        default="dense",
    )
    arguments = parser.parse_args(argv)
    manifest = generate(
        arguments.root,
        transport_backend=arguments.transport_backend,
        candidate_backend=arguments.candidate_backend,
    )
    print(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
