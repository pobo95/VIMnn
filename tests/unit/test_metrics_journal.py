from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib
import json
import os
import random

import numpy as np
import pytest
import torch

from refsite_mlip.training import (
    CheckpointMetadata,
    CommittedEpochMetrics,
    EpochResult,
    EpochTermMetrics,
    FitEpochRecord,
    FitProgress,
    MetricsJournal,
    MetricsJournalConfig,
    MetricsJournalError,
    ModelSelectionState,
    TrainingCheckpoint,
    TrainingDataManifest,
    TrainingRunDirectory,
    ValidationDecision,
    canonical_runtime_json,
    committed_epoch_metrics_from_record,
    committed_epoch_provenance_from_checkpoint_metadata,
)
from refsite_mlip.training.checkpoint_manager import ManagedCheckpointResult


module = importlib.import_module("refsite_mlip.training.metrics_journal")


def _term(value: float, *, count: int = 2) -> EpochTermMetrics:
    if count == 0:
        return EpochTermMetrics(0.0, 0.0, 0.0, 0)
    return EpochTermMetrics(value * count, float(count), value, count)


def _epoch_result(
    phase: str,
    epoch: int,
    start: int,
    metric: float,
) -> EpochResult:
    training = phase == "train"
    end = start + 2 if training else start
    return EpochResult(
        energy=_term(metric),
        force=_term(metric + 1.0),
        stress=_term(metric + 2.0),
        total_loss=metric + 3.0,
        has_supervision=True,
        phase=phase,
        epoch_index=epoch,
        global_step_start=start,
        global_step_end=end,
        number_of_batches=2,
        number_of_supervised_batches=2,
        number_of_structures=2,
        number_of_atoms=4,
        successful_optimizer_steps=2 if training else 0,
        ordered_batch_sample_ids=((f"sample-{epoch}-0",), (f"sample-{epoch}-1",)),
        metric_semantics=(
            "pre_update_batch_observations"
            if training
            else "fixed_model_validation"
        ),
    )


def _record(epoch: int, *, metric: float | None = None) -> FitEpochRecord:
    value = float(epoch + 1 if metric is None else metric)
    start = 2 * epoch
    is_best = epoch == 0
    best_value = 4.0
    best_epoch = 0
    state = ModelSelectionState(
        best_metric=best_value,
        best_epoch=best_epoch,
        best_global_step=2,
        epochs_since_improvement=epoch,
        validation_events=epoch + 1,
        last_validation_epoch=epoch,
        last_validation_global_step=start + 2,
    )
    decision = ValidationDecision(
        metric_name="total_loss",
        metric_value=value + 3.0,
        is_best=is_best,
        should_stop=False,
        best_metric=best_value,
        best_epoch=best_epoch,
        best_global_step=2,
        epochs_since_improvement=epoch,
        validation_events=epoch + 1,
        learning_rates_before=(0.1,),
        learning_rates_after=(0.1,),
        scheduler_stepped=True,
        learning_rate_changed=False,
    )
    return FitEpochRecord(
        epoch_index=epoch,
        training=_epoch_result("train", epoch, start, value),
        validation=_epoch_result("validation", epoch, start + 2, value),
        decision=decision,
        selection_state_after_epoch=state,
        learning_rates_used_for_training=(0.1,),
        learning_rates_after_validation=(0.1,),
    )


def _metadata(max_epochs: int = 2) -> CheckpointMetadata:
    train = TrainingDataManifest(
        split_name="train",
        fingerprint="2" * 64,
        manifest_version="prebatched_structure_sequence_v1",
        unit_convention_version="angstrom_ev_tensile_voigt_v1",
        number_of_batches=1,
        number_of_structures=1,
        number_of_atoms=1,
        ordered_batch_sample_ids=(("train:0",),),
    )
    validation = replace(
        train,
        split_name="validation",
        fingerprint="3" * 64,
        ordered_batch_sample_ids=(("validation:0",),),
    )
    return CheckpointMetadata(
        resolved_configuration={
            "model": {"kind": "tiny"},
            "loss": {"energy_weight": 1.0},
            "optimizer": {"learning_rate": 0.1},
            "train_step": {"solver_path": "train-fixed"},
            "validation_step": {"solver_path": "train-fixed"},
            "scheduler": {"kind": "none"},
            "model_selection": {"monitor": "total_loss", "mode": "min"},
            "fit": {
                "max_epochs": max_epochs,
                "start_epoch": 0,
                "global_step_start": 0,
            },
        },
        species_vocabulary=(6,),
        unit_conventions={
            "version": "angstrom_ev_tensile_voigt_v1",
        },
        template_fingerprints={"template-b": "5" * 64, "template-a": "4" * 64},
        training_data=train,
        validation_data=validation,
        package_versions={"python": "3.10"},
    )


def _provenance(metadata: CheckpointMetadata | None = None):
    return committed_epoch_provenance_from_checkpoint_metadata(
        _metadata() if metadata is None else metadata,
        initial_bundle_fingerprint="1" * 64,
    )


def _managed(record: FitEpochRecord, root) -> ManagedCheckpointResult:
    is_best = record.decision.is_best
    return ManagedCheckpointResult(
        epoch_index=record.epoch_index,
        global_step=record.training.global_step_end,
        is_best=is_best,
        epoch_path=str(root / f"epoch_{record.epoch_index:06d}.pt"),
        latest_path=str(root / "latest.pt"),
        best_path=str(root / "best.pt") if is_best else None,
        epoch_written=True,
        latest_written=True,
        best_written=is_best,
        completed_stages=("epoch", "latest") + (("best",) if is_best else ()),
    )


def _event(epoch: int, root, *, metric: float | None = None):
    record = _record(epoch, metric=metric)
    return committed_epoch_metrics_from_record(
        record,
        _managed(record, root),
        selection_mode="min",
        provenance=_provenance(),
    )


def _checkpoint(records: tuple[FitEpochRecord, ...]) -> TrainingCheckpoint:
    latest = records[-1]
    return TrainingCheckpoint(
        model_state_dict={"weight": torch.tensor([1.0])},
        optimizer_state_dict={"state": {}, "param_groups": [{"lr": 0.1}]},
        scheduler_state_dict={},
        selection_state=latest.selection_state_after_epoch,
        progress=FitProgress(
            next_epoch=latest.epoch_index + 1,
            global_step=latest.training.global_step_end,
            completed_epochs=len(records),
            last_completed_epoch=latest.epoch_index,
            best_epoch=latest.selection_state_after_epoch.best_epoch,
            best_global_step=latest.selection_state_after_epoch.best_global_step,
        ),
        metadata=_metadata(max_epochs=len(records)),
        python_rng_state=[],
        numpy_rng_state={},
        torch_cpu_rng_state=torch.get_rng_state().clone(),
        cuda_rng_states=(),
        cuda_device_count=0,
        fit_history=tuple(record.to_dict() for record in records),
    )


@pytest.fixture
def owned_journal(tmp_path):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    lock = directory.acquire_resume_lock()
    journal = MetricsJournal(directory, lock, _provenance())
    try:
        yield directory, lock, journal
    finally:
        if lock.owned:
            lock.release()


def test_event_projection_is_immutable_plain_and_exact(tmp_path):
    event = _event(0, tmp_path)
    payload = event.to_dict()
    assert payload["schema_version"] == "refsite_training_metrics_v1"
    assert payload["event"] == "epoch_committed"
    assert payload["training"]["force"] == {
        "numerator": 4.0,
        "denominator": 2.0,
        "mean": 2.0,
        "valid_count": 2,
    }
    assert payload["validation"]["stress"]["mean"] == 3.0
    assert payload["epoch_checkpoint_basename"] == "epoch_000000.pt"
    assert payload["latest_checkpoint_basename"] == "latest.pt"
    assert payload["best_checkpoint_basename"] == "best.pt"
    assert CommittedEpochMetrics.from_dict(payload) == event

    def assert_tensor_free(value):
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for item in value.values():
                assert_tensor_free(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                assert_tensor_free(item)

    assert_tensor_free(payload)
    with pytest.raises(Exception):
        event.epoch_index = 9


def test_provenance_is_order_independent_and_stable_across_max_epoch_extension():
    first = _metadata(max_epochs=1)
    reversed_templates = dict(reversed(tuple(first.template_fingerprints.items())))
    second = replace(
        _metadata(max_epochs=20), template_fingerprints=reversed_templates
    )
    left = _provenance(first)
    right = _provenance(second)
    assert left == right
    assert left.template_fingerprints == (
        ("template-a", "4" * 64),
        ("template-b", "5" * 64),
    )
    assert type(left.training_configuration_fingerprint) is str
    assert len(left.training_configuration_fingerprint) == 64
    assert type(left.from_dict(left.to_dict()).template_fingerprints) is tuple


def test_provenance_tracks_baseline_semantics_but_not_run_source_identity():
    metadata = _metadata(max_epochs=1)
    first = replace(
        metadata,
        baseline_fit_metadata={
            "enabled": False,
            "seed": 17,
            "training_run_config_fingerprint": "8" * 64,
            "initial_bundle_fingerprint": "9" * 64,
        },
    )
    equivalent_source = replace(
        metadata,
        baseline_fit_metadata={
            "enabled": False,
            "seed": 17,
            "training_run_config_fingerprint": "a" * 64,
            "initial_bundle_fingerprint": "b" * 64,
        },
    )
    changed_seed = replace(
        equivalent_source,
        baseline_fit_metadata={
            **equivalent_source.baseline_fit_metadata,
            "seed": 18,
        },
    )

    first_provenance = _provenance(first)
    assert first_provenance == _provenance(equivalent_source)
    assert (
        first_provenance.training_configuration_fingerprint
        != _provenance(changed_seed).training_configuration_fingerprint
    )


def test_config_round_trip_and_strict_validation():
    config = MetricsJournalConfig()
    assert MetricsJournalConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="filename"):
        MetricsJournalConfig(filename="other.jsonl")
    with pytest.raises(TypeError):
        MetricsJournalConfig(epoch_filename_width=True)
    with pytest.raises(ValueError, match="unknown"):
        MetricsJournalConfig.from_dict({**config.to_dict(), "unknown": 1})


def test_atomic_one_and_two_epoch_append_preserves_prefix(owned_journal):
    directory, _, journal = owned_journal
    first = _event(0, directory.checkpoints)
    summary = journal.append(first)
    first_bytes = directory.metrics_path.read_bytes()
    assert first_bytes == (canonical_runtime_json(first.to_dict()) + "\n").encode()
    assert summary.to_dict() == {
        "metrics_journal": "metrics.jsonl",
        "metrics_event_count": 1,
        "metrics_last_epoch": 0,
        "metrics_semantic_sha256": summary.metrics_semantic_sha256,
    }
    second = _event(1, directory.checkpoints)
    journal.append(second)
    final = directory.metrics_path.read_bytes()
    assert final.startswith(first_bytes)
    assert final == first_bytes + (
        canonical_runtime_json(second.to_dict()) + "\n"
    ).encode()
    assert journal.summary().metrics_event_count == 2


def test_journal_append_does_not_consume_process_rng(owned_journal):
    directory, _, journal = owned_journal
    random.seed(9041)
    np.random.seed(9041)
    torch.manual_seed(9041)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    journal.append(_event(0, directory.checkpoints))

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_journal_is_an_epoch_observer_callable(owned_journal):
    directory, _, journal = owned_journal
    event = _event(0, directory.checkpoints)
    assert journal(event) is None
    assert journal.summary().metrics_event_count == 1


def test_mutation_requires_owned_lock_but_read_only_access_does_not(tmp_path):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    journal = MetricsJournal(directory, None, _provenance())
    assert journal.summary().metrics_event_count == 0
    with pytest.raises(MetricsJournalError) as caught:
        journal.append(_event(0, directory.checkpoints))
    assert caught.value.reason_code == "METRICS_JOURNAL_LOCK_REQUIRED"

    lock = directory.acquire_resume_lock()
    locked = MetricsJournal(directory, lock, _provenance())
    lock.release()
    with pytest.raises(MetricsJournalError) as caught:
        locked.append(_event(0, directory.checkpoints))
    assert caught.value.reason_code == "METRICS_JOURNAL_LOCK_NOT_OWNED"


def test_duplicate_gap_and_global_step_mismatch_are_rejected(owned_journal):
    directory, _, journal = owned_journal
    journal.append(_event(0, directory.checkpoints))
    before = directory.metrics_path.read_bytes()
    with pytest.raises(MetricsJournalError, match="exact next"):
        journal.append(_event(0, directory.checkpoints))
    assert directory.metrics_path.read_bytes() == before

    gap_record = _record(2)
    gap = committed_epoch_metrics_from_record(
        gap_record,
        _managed(gap_record, directory.checkpoints),
        selection_mode="min",
        provenance=_provenance(),
    )
    with pytest.raises(MetricsJournalError, match="exact next"):
        journal.append(gap)
    assert directory.metrics_path.read_bytes() == before

    second = _event(1, directory.checkpoints)
    shifted = replace(
        second,
        global_step_start=5,
        global_step_end=7,
    )
    with pytest.raises(MetricsJournalError, match="global-step"):
        journal.append(shifted)
    assert directory.metrics_path.read_bytes() == before


def test_sequence_rejects_events_after_early_stop(owned_journal):
    directory, _, journal = owned_journal
    journal.append(_event(0, directory.checkpoints))
    stopped = replace(_event(1, directory.checkpoints), should_stop=True)
    journal.append(stopped)
    before = directory.metrics_path.read_bytes()

    with pytest.raises(MetricsJournalError, match="terminate after should_stop"):
        journal.append(_event(2, directory.checkpoints))

    assert directory.metrics_path.read_bytes() == before


def test_event_rejects_inconsistent_monitored_and_best_values(tmp_path):
    event = _event(0, tmp_path)
    with pytest.raises(ValueError, match="differs from validation"):
        replace(event, monitored_metric_value=event.monitored_metric_value + 1.0)
    with pytest.raises(ValueError, match="store its monitored value"):
        replace(event, best_value=event.best_value + 1.0)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"event":"epoch_committed"',
        b'{"schema_version":"x","schema_version":"y"}\n',
        b'{"value":NaN}\n',
        b"\xff\n",
        b"{}\r\n",
    ],
)
def test_malformed_nonfinite_duplicate_and_truncated_lines_fail(tmp_path, payload):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    directory.metrics_path.write_bytes(payload)
    journal = MetricsJournal(directory, None, _provenance())
    with pytest.raises(MetricsJournalError) as caught:
        journal.summary()
    assert caught.value.reason_code == "METRICS_JOURNAL_LOAD_FAILED"


def test_malformed_suffix_reports_last_complete_canonical_prefix(tmp_path):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    first = (
        canonical_runtime_json(_event(0, directory.checkpoints).to_dict()) + "\n"
    ).encode("utf-8")
    directory.metrics_path.write_bytes(first + b'{"truncated":')
    journal = MetricsJournal(directory, None, _provenance())

    with pytest.raises(MetricsJournalError) as caught:
        journal.summary()

    error = caught.value
    assert error.reason_code == "METRICS_JOURNAL_LOAD_FAILED"
    assert error.last_valid_epoch == 0
    assert error.last_valid_event_count == 1
    assert error.last_valid_semantic_sha256 == hashlib.sha256(first).hexdigest()


def test_symlink_is_rejected_without_changing_target(tmp_path):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    target = tmp_path / "sentinel"
    target.write_bytes(b"sentinel")
    directory.metrics_path.symlink_to(target)
    journal = MetricsJournal(directory, None, _provenance())
    with pytest.raises(MetricsJournalError) as caught:
        journal.summary()
    assert caught.value.reason_code == "METRICS_JOURNAL_SYMLINK_REJECTED"
    assert target.read_bytes() == b"sentinel"


def test_atomic_precommit_failure_preserves_previous_bytes_and_cleans_temp(
    owned_journal, monkeypatch
):
    directory, _, journal = owned_journal
    journal.append(_event(0, directory.checkpoints))
    before = directory.metrics_path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected commit failure")

    monkeypatch.setattr(module, "commit_temporary_file", fail)
    with pytest.raises(MetricsJournalError) as caught:
        journal.append(_event(1, directory.checkpoints))
    assert not caught.value.commit_completed
    assert directory.metrics_path.read_bytes() == before
    assert not tuple(directory.root.glob(".metrics.jsonl.*.tmp"))


def test_initial_create_race_never_clobbers_competing_target(
    tmp_path, monkeypatch
):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    lock = directory.acquire_resume_lock()
    journal = MetricsJournal(directory, lock, _provenance())
    real_commit = module.commit_temporary_file
    competitor = b"competitor-created-after-validation\n"

    def race(temporary, target, *, overwrite):
        assert overwrite is False
        target.write_bytes(competitor)
        return real_commit(temporary, target, overwrite=overwrite)

    monkeypatch.setattr(module, "commit_temporary_file", race)
    try:
        with pytest.raises(MetricsJournalError) as caught:
            journal.append(_event(0, directory.checkpoints))
        assert caught.value.commit_completed is False
        assert directory.metrics_path.read_bytes() == competitor
        assert not tuple(directory.root.glob(".metrics.jsonl.*.tmp"))
    finally:
        lock.release()


def test_existing_journal_toctou_change_is_detected_and_preserved(
    owned_journal, monkeypatch
):
    directory, _, journal = owned_journal
    journal.append(_event(0, directory.checkpoints))
    competitor_event = replace(
        _event(0, directory.checkpoints), training_total_loss=8.0
    )
    competitor = (
        canonical_runtime_json(competitor_event.to_dict()) + "\n"
    ).encode()
    real_read = journal._read
    calls = 0

    def replace_before_commit_validation():
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement = directory.root / ".competitor-metrics"
            replacement.write_bytes(competitor)
            os.replace(replacement, directory.metrics_path)
        return real_read()

    monkeypatch.setattr(journal, "_read", replace_before_commit_validation)
    with pytest.raises(MetricsJournalError) as caught:
        journal.append(_event(1, directory.checkpoints))
    assert caught.value.reason_code == "METRICS_JOURNAL_ATOMIC_WRITE_FAILED"
    assert caught.value.commit_completed is False
    assert directory.metrics_path.read_bytes() == competitor
    assert not tuple(directory.root.glob(".metrics.jsonl.*.tmp"))


def test_existing_journal_disappearance_is_detected_without_recreation(
    owned_journal, monkeypatch
):
    directory, _, journal = owned_journal
    journal.append(_event(0, directory.checkpoints))
    real_read = journal._read
    calls = 0

    def remove_before_commit_validation():
        nonlocal calls
        calls += 1
        if calls == 2:
            directory.metrics_path.unlink()
        return real_read()

    monkeypatch.setattr(journal, "_read", remove_before_commit_validation)
    with pytest.raises(MetricsJournalError) as caught:
        journal.append(_event(1, directory.checkpoints))
    assert caught.value.reason_code == "METRICS_JOURNAL_ATOMIC_WRITE_FAILED"
    assert caught.value.commit_completed is False
    assert not directory.metrics_path.exists()
    assert not tuple(directory.root.glob(".metrics.jsonl.*.tmp"))


def test_lock_inode_replacement_is_rejected_without_removing_foreign_lock(
    tmp_path,
):
    directory = TrainingRunDirectory.create(tmp_path / "run")
    lock = directory.acquire_resume_lock()
    journal = MetricsJournal(directory, lock, _provenance())
    directory.resume_lock_path.unlink()
    foreign = b"foreign-lock\n"
    directory.resume_lock_path.write_bytes(foreign)

    with pytest.raises(MetricsJournalError) as caught:
        journal.append(_event(0, directory.checkpoints))
    assert caught.value.reason_code == "METRICS_JOURNAL_LOCK_NOT_OWNED"
    assert directory.resume_lock_path.read_bytes() == foreign
    with pytest.raises(Exception, match="replaced"):
        lock.release()
    assert directory.resume_lock_path.read_bytes() == foreign


def test_postcommit_directory_fsync_failure_reports_complete_new_target(
    owned_journal, monkeypatch
):
    directory, _, journal = owned_journal
    first = _event(0, directory.checkpoints)
    real_fsync = module.os.fsync
    calls = 0

    def fail_directory(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_directory)
    with pytest.raises(MetricsJournalError) as caught:
        journal.append(first)
    assert caught.value.commit_completed
    assert directory.metrics_path.read_bytes() == (
        canonical_runtime_json(first.to_dict()) + "\n"
    ).encode()


def test_resume_prefix_recovery_exact_noop_and_extra_rejection(owned_journal):
    directory, _, journal = owned_journal
    records = (_record(0), _record(1))
    checkpoint = _checkpoint(records)
    journal.append(_event(0, directory.checkpoints))
    prefix = directory.metrics_path.read_bytes()
    missing = journal.inspect_checkpoint(checkpoint)
    assert tuple(event.epoch_index for event in missing) == (1,)
    assert directory.metrics_path.read_bytes() == prefix

    summary = journal.reconcile_checkpoint(checkpoint)
    complete = directory.metrics_path.read_bytes()
    assert summary.metrics_event_count == 2
    assert journal.inspect_checkpoint(checkpoint) == ()
    journal.reconcile_checkpoint(checkpoint)
    assert directory.metrics_path.read_bytes() == complete

    shorter = _checkpoint((_record(0),))
    with pytest.raises(MetricsJournalError) as caught:
        journal.inspect_checkpoint(shorter)
    assert caught.value.reason_code == "METRICS_JOURNAL_HISTORY_DIVERGENCE"
    assert caught.value.last_valid_epoch == 0
    assert directory.metrics_path.read_bytes() == complete


def test_divergent_checkpoint_history_is_not_rewritten(owned_journal):
    directory, _, journal = owned_journal
    journal.append(_event(0, directory.checkpoints))
    journal.append(_event(1, directory.checkpoints, metric=8.0))
    before = directory.metrics_path.read_bytes()
    with pytest.raises(MetricsJournalError, match="exact prefix"):
        journal.reconcile_checkpoint(_checkpoint((_record(0), _record(1))))
    assert directory.metrics_path.read_bytes() == before


def test_nonfinite_event_is_rejected_before_journal_write(tmp_path):
    event = _event(0, tmp_path)
    payload = copy.deepcopy(event.to_dict())
    payload["training"]["energy"]["numerator"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        CommittedEpochMetrics.from_dict(payload)
