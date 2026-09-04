from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("ase")
from ase.build import bulk
from ase.io import write

from refsite_mlip.config import load_training_run_config
from refsite_mlip.models import load_reference_site_model_bundle
from refsite_mlip.training import (
    load_training_checkpoint,
    prepare_scratch_training_run,
    run_scratch_checkpointed_training,
)

from test_scratch_training_preparation import (
    LATTICE,
    _atoms,
    _case,
    _config_211,
    _labeled,
    _phase_211,
    _template_payload,
    _vacancy,
    _write_extxyz,
)


def _mixed_pristine_vacancy_preparation(directory: Path):
    """Build one deterministic mixed-M batch containing K=0 and K=1."""

    directory.mkdir(parents=True, exist_ok=True)
    reference_111 = _atoms(1)
    reference_211 = bulk(
        "NbC", "rocksalt", a=LATTICE, cubic=True
    ).repeat((2, 1, 1))

    poscar_211 = directory / "reference-211.POSCAR"
    train_211 = directory / "train-211.xyz"
    validation_211 = directory / "validation-211.xyz"
    write(poscar_211, reference_211, format="vasp", direct=True)
    _write_extxyz(
        train_211,
        (
            _labeled(reference_211, -16.0),
            _labeled(_vacancy(reference_211), -14.5),
        ),
    )
    _write_extxyz(
        validation_211, (_labeled(reference_211, -15.5),)
    )

    templates = (
        _template_payload(
            template_id="zeta-211",
            poscar_path=poscar_211.name,
            builder=_config_211(),
            phase=_phase_211(),
        ),
        _template_payload(
            template_id="alpha-111",
            poscar_path="reference.POSCAR",
        ),
    )
    config_path, _, _, payload = _case(
        directory,
        train_frames=(
            _labeled(reference_111, -8.0),
            _labeled(_vacancy(reference_111), -6.5),
        ),
        validation_frames=(_labeled(reference_111, -7.75),),
        selector={"template_id": "alpha-111"},
        templates=templates,
        baseline=True,
    )
    payload["data"] = {
        "train": [
            {"path": train_211.name, "template_id": "zeta-211"},
            {"path": "train.xyz", "template_id": "alpha-111"},
        ],
        "validation": [
            {"path": "validation.xyz", "template_id": "alpha-111"},
            {
                "path": validation_211.name,
                "template_id": "zeta-211",
            },
        ],
        "batch_size": 4,
        "validation_batch_size": 2,
        "shuffle": False,
    }
    payload["fit"]["max_epochs"] = 1
    config_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    config = load_training_run_config(config_path)
    return config, prepare_scratch_training_run(config)


def test_mixed_template_pristine_vacancy_one_epoch_checkpoint_contract(
    tmp_path,
):
    config, preparation = _mixed_pristine_vacancy_preparation(tmp_path)
    expected_train_ids = tuple(
        sample.sample_id for sample in preparation.train_samples
    )
    expected_validation_ids = tuple(
        sample.sample_id for sample in preparation.validation_samples
    )

    assert tuple(sample.template_id for sample in preparation.train_samples) == (
        "zeta-211",
        "zeta-211",
        "alpha-111",
        "alpha-111",
    )
    assert tuple(sample.num_atoms for sample in preparation.train_samples) == (
        16,
        15,
        8,
        7,
    )
    assert {
        template_id: artifact.diagnostics.num_sites
        for template_id, artifact in preparation.structural_artifacts.items()
    } == {"alpha-111": 8, "zeta-211": 16}

    result = run_scratch_checkpointed_training(config, preparation)
    startup = result.startup
    assert result.status == "completed"
    assert result.completed_epochs == 1
    assert result.global_step == 1
    assert len(startup.train_batches) == len(startup.validation_batches) == 1

    train_batch = startup.train_batches[0]
    validation_batch = startup.validation_batches[0]
    assert train_batch.sample_ids == expected_train_ids
    assert validation_batch.sample_ids == expected_validation_ids
    assert train_batch.template_ids == (
        "zeta-211",
        "zeta-211",
        "alpha-111",
        "alpha-111",
    )
    assert tuple(group.template_id for group in train_batch.template_groups) == (
        "alpha-111",
        "zeta-211",
    )
    assert tuple(train_batch.template_groups[0].structure_indices.tolist()) == (
        2,
        3,
    )
    assert tuple(train_batch.template_groups[1].structure_indices.tolist()) == (
        0,
        1,
    )

    for template_id, fingerprint in zip(
        train_batch.template_ids, train_batch.template_fingerprints
    ):
        assert fingerprint == startup.registry.resolve(template_id).fingerprint
        assert fingerprint == startup.template_contexts[template_id].fingerprint
    assert {
        template_id: context.topology.num_sites
        for template_id, context in startup.template_contexts.items()
    } == {"alpha-111": 8, "zeta-211": 16}
    assert startup.data_manifest == preparation.data_manifest

    checkpoint_directory = Path(result.run_directory) / "checkpoints"
    assert sorted(path.name for path in checkpoint_directory.iterdir()) == [
        "best.pt",
        "epoch_000000.pt",
        "latest.pt",
    ]
    initial = load_reference_site_model_bundle(
        Path(result.run_directory) / "initial_bundle.pt"
    )
    latest = load_training_checkpoint(checkpoint_directory / "latest.pt")
    assert torch.equal(
        initial.model_state["atomic_baseline"],
        torch.zeros_like(initial.model_state["atomic_baseline"]),
    )
    assert startup.baseline_fit is not None
    assert startup.baseline_fit.rank == 2
    assert startup.baseline_fit.training_sample_ids == expected_train_ids
    assert torch.equal(
        latest.model_state_dict["atomic_baseline"],
        startup.model.atomic_baseline.detach().cpu(),
    )
    assert bool(torch.any(latest.model_state_dict["atomic_baseline"] != 0.0))
    parameter_names = tuple(dict(startup.model.named_parameters()))
    assert any(
        not torch.equal(
            initial.model_state[name], latest.model_state_dict[name]
        )
        for name in parameter_names
    )
    assert latest.progress.completed_epochs == 1
    assert latest.progress.global_step == 1
    assert latest.fit_history is not None and len(latest.fit_history) == 1
    assert latest.metadata.template_fingerprints == {
        template_id: startup.registry.resolve(template_id).fingerprint
        for template_id in ("alpha-111", "zeta-211")
    }
    assert result.fit_result.records[0].training.ordered_batch_sample_ids == (
        expected_train_ids,
    )
    assert result.fit_result.records[0].validation.ordered_batch_sample_ids == (
        expected_validation_ids,
    )
    assert not Path(result.run_directory, ".resume.lock").exists()
