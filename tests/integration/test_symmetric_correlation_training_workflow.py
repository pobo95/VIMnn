from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from refsite_mlip.cli.export_bundle import export_bundle
from refsite_mlip.cli.resume import resume_training
from refsite_mlip.cli.train import run_training
from refsite_mlip.inference import ReferenceSitePredictor, load_reference_site_predictor
from refsite_mlip.interfaces import ReferenceSiteASECalculator
from refsite_mlip.models import (
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
)
from refsite_mlip.training import (
    load_training_checkpoint,
    run_scratch_checkpointed_training,
)
from refsite_mlip.transport import TRAIN_FIXED

from test_scratch_checkpointed_training_mixed import (
    _mixed_pristine_vacancy_preparation,
)


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def test_v2_scratch_checkpoint_resume_and_best_latest_export(tmp_path):
    config, preparation = _mixed_pristine_vacancy_preparation(
        tmp_path, symmetric_v2=True
    )
    first = run_scratch_checkpointed_training(config, preparation)
    run_directory = Path(first.run_directory)
    checkpoint_directory = run_directory / "checkpoints"
    epoch_zero = checkpoint_directory.joinpath("epoch_000000.pt").read_bytes()
    journal_prefix = run_directory.joinpath("metrics.jsonl").read_bytes()

    initial = load_reference_site_model_bundle(run_directory / "initial_bundle.pt")
    assert initial.model_config["higher_body"]["contract_version"] == (
        "central_conditioned_symmetric_power_v2"
    )
    assert torch.count_nonzero(initial.model_state["atomic_baseline"]) == 0

    # Feeding the exact scratch initial bundle through the bundle-source path
    # must produce the same one-epoch numerical trajectory.
    bundle_payload = config.to_dict()
    bundle_payload["model_source"] = {
        "kind": "bundle",
        "path": "scratch-output/initial_bundle.pt",
    }
    bundle_payload["output_directory"] = "bundle-output"
    bundle_config = tmp_path / "bundle-run.json"
    bundle_config.write_text(
        json.dumps(bundle_payload, sort_keys=True), encoding="utf-8"
    )
    bundle_result = run_training(bundle_config)
    bundle_run = Path(bundle_result["latest_checkpoint"]).parent.parent
    scratch_epoch_zero = load_training_checkpoint(
        checkpoint_directory / "epoch_000000.pt"
    )
    bundle_epoch_zero = load_training_checkpoint(
        bundle_run / "checkpoints" / "epoch_000000.pt"
    )
    assert all(
        torch.equal(
            scratch_epoch_zero.model_state_dict[key],
            bundle_epoch_zero.model_state_dict[key],
        )
        for key in scratch_epoch_zero.model_state_dict
    )
    assert _tree_equal(
        scratch_epoch_zero.optimizer_state_dict,
        bundle_epoch_zero.optimizer_state_dict,
    )
    assert _tree_equal(
        scratch_epoch_zero.scheduler_state_dict,
        bundle_epoch_zero.scheduler_state_dict,
    )
    assert scratch_epoch_zero.selection_state == bundle_epoch_zero.selection_state
    assert scratch_epoch_zero.fit_history == bundle_epoch_zero.fit_history

    resumed = resume_training(run_directory, max_epochs=2)
    assert resumed["status"] == "completed"
    assert checkpoint_directory.joinpath("epoch_000001.pt").is_file()
    assert checkpoint_directory.joinpath("epoch_000000.pt").read_bytes() == epoch_zero
    journal = run_directory.joinpath("metrics.jsonl").read_bytes()
    assert journal.startswith(journal_prefix)
    events = [json.loads(line) for line in journal.decode("utf-8").splitlines()]
    assert [event["epoch_index"] for event in events] == [0, 1]

    checkpoint = load_training_checkpoint(checkpoint_directory / "latest.pt")
    assert checkpoint.schema_version == "refsite_training_checkpoint_v1"
    assert checkpoint.progress.completed_epochs == 2
    assert checkpoint.progress.global_step == 2
    assert torch.count_nonzero(checkpoint.model_state_dict["atomic_baseline"]) > 0
    u_keys = [
        key
        for key in checkpoint.model_state_dict
        if key.startswith("symmetric_cg_basis.")
    ]
    w_keys = [
        key
        for key in checkpoint.model_state_dict
        if ".symmetric_contraction.weight_" in key
    ]
    assert len(u_keys) == 9
    assert len(w_keys) == 9

    for source in ("best", "latest"):
        output = tmp_path / f"{source}-v2.pt"
        report = export_bundle(
            run_directory,
            source=source,
            output_path=output,
        )
        exported = load_reference_site_model_bundle(output)
        source_checkpoint = load_training_checkpoint(
            checkpoint_directory / f"{source}.pt"
        )
        assert report["bundle_sha256"] == exported.bundle_fingerprint
        assert exported.architecture_fingerprint == initial.architecture_fingerprint
        assert tuple(exported.model_state_keys) == tuple(
            source_checkpoint.model_state_dict
        )
        assert all(
            torch.equal(
                exported.model_state[key], source_checkpoint.model_state_dict[key]
            )
            for key in exported.model_state_keys
        )
        payload = exported.to_payload()
        assert "optimizer_state_dict" not in payload
        assert "scheduler_state_dict" not in payload
        assert "fit_history" not in payload
        assert "torch_cpu_rng_state" not in payload

        source_runtime = instantiate_reference_site_model_bundle(
            initial, device="cpu", dtype=torch.float64
        )
        source_runtime.model.load_state_dict(
            source_checkpoint.model_state_dict, strict=True
        )
        source_predictor = ReferenceSitePredictor(source_runtime)
        exported_predictor = load_reference_site_predictor(
            output, device="cpu", dtype=torch.float64
        )
        sample = preparation.train_samples[0]
        source_prediction = source_predictor.predict_sample(
            sample,
            solver_path=TRAIN_FIXED,
            compute_forces=True,
            compute_stress=True,
        )
        exported_prediction = exported_predictor.predict_sample(
            sample,
            solver_path=TRAIN_FIXED,
            compute_forces=True,
            compute_stress=True,
        )
        for name in ("energy", "forces", "stress", "stress_voigt"):
            assert torch.equal(
                getattr(source_prediction, name),
                getattr(exported_prediction, name),
            )

        atoms = Atoms(
            numbers=sample.atomic_numbers.numpy(),
            positions=sample.positions.numpy(),
            cell=sample.cell.numpy(),
            pbc=sample.pbc.numpy(),
        )
        atoms.calc = ReferenceSiteASECalculator(
            output,
            template_id=sample.template_id,
            device="cpu",
            dtype=torch.float64,
            solver_path=TRAIN_FIXED,
        )
        assert atoms.get_potential_energy() == float(exported_prediction.energy)
        np.testing.assert_array_equal(
            atoms.get_forces(), exported_prediction.forces.numpy()
        )
        np.testing.assert_array_equal(
            atoms.get_stress(), exported_prediction.stress_voigt.numpy()
        )


def test_v2_metrics_journal_continuous_and_resumed_are_byte_exact(tmp_path):
    continuous_config, continuous_preparation = (
        _mixed_pristine_vacancy_preparation(
            tmp_path / "continuous", symmetric_v2=True, max_epochs=3
        )
    )
    continuous = run_scratch_checkpointed_training(
        continuous_config, continuous_preparation
    )

    split_config, split_preparation = _mixed_pristine_vacancy_preparation(
        tmp_path / "resumed", symmetric_v2=True, max_epochs=1
    )
    first = run_scratch_checkpointed_training(split_config, split_preparation)
    resumed = resume_training(first.run_directory, max_epochs=3)

    continuous_run = Path(continuous.run_directory)
    resumed_run = Path(first.run_directory)
    continuous_bytes = continuous_run.joinpath("metrics.jsonl").read_bytes()
    resumed_bytes = resumed_run.joinpath("metrics.jsonl").read_bytes()
    assert continuous_bytes == resumed_bytes
    semantic_sha = hashlib.sha256(continuous_bytes).hexdigest()
    events = [
        json.loads(line) for line in continuous_bytes.decode("utf-8").splitlines()
    ]
    assert [event["epoch_index"] for event in events] == [0, 1, 2]
    assert [event["global_step_end"] for event in events] == [1, 2, 3]
    for previous, current in zip(events, events[1:]):
        assert current["global_step_start"] == previous["global_step_end"]
        assert current["learning_rates_before_scheduler"] == (
            previous["learning_rates_after_scheduler"]
        )
    assert len(
        {
            event["provenance"]["training_configuration_fingerprint"]
            for event in events
        }
    ) == 1
    assert len(
        {
            event["provenance"]["initial_bundle_fingerprint"]
            for event in events
        }
    ) == 1
    assert resumed["metrics_semantic_sha256"] == semantic_sha
    for run in (continuous_run, resumed_run):
        status = json.loads(run.joinpath("run_status.json").read_text("utf-8"))
        assert status["metrics_event_count"] == 3
        assert status["metrics_last_epoch"] == 2
        assert status["metrics_semantic_sha256"] == semantic_sha
