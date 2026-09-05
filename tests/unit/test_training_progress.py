from __future__ import annotations

from dataclasses import FrozenInstanceError
import io
import math
from pathlib import Path

import pytest

from refsite_mlip.cli.training_progress import (
    TrainingProgressConfig,
    TrainingProgressError,
    TrainingProgressRenderer,
    TrainingStartSummary,
    journal_then_progress_observer,
)
from refsite_mlip.training.metrics_journal import (
    CommittedEpochMetrics,
    CommittedEpochProvenance,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _summary(**overrides: object) -> TrainingStartSummary:
    values: dict[str, object] = {
        "run_name": "synthetic-run",
        "source_kind": "scratch",
        "device": "cpu",
        "dtype": "float64",
        "training_seed": -17,
        "initialization_seed": -3,
        "parameter_tensor_count": 4,
        "parameter_element_count": 120,
        "species_vocabulary": (41, 6),
        "templates": (("z-template", 8), ("a-template", 4)),
        "default_template_id": "a-template",
        "train_frame_count": 2,
        "validation_frame_count": 1,
        "train_batch_count": 1,
        "validation_batch_count": 1,
        "train_batch_size": 2,
        "validation_batch_size": 1,
        "template_frame_counts": (
            ("z-template", 1, 0),
            ("a-template", 1, 1),
        ),
        "composition_summary": (("C2Nb2", 1, 1), ("CNb", 1, 0)),
        "label_presence": (
            ("stress", 1, 1, 0, 1),
            ("energy", 2, 0, 1, 0),
            ("force", 2, 0, 1, 0),
        ),
        "r_ot": 4.0,
        "r_mp": 3.0,
        "r_candidate_ot": 4.2,
        "r_candidate_mp": 3.5,
        "ot_backend": "edge-list",
        "solver_path": "TRAIN_FIXED",
        "baseline_enabled": True,
        "baseline_values": (-1.25, 2.5),
        "baseline_rank_policy": "minimum_norm",
        "baseline_status": "fitted",
        "loss_terms": (
            ("stress", 3.0, 0.5),
            ("energy", 1.0, 2.0),
            ("force", 2.0, 1.0),
        ),
        "optimizer_kind": "AdamW",
        "initial_learning_rate": 5.0e-4,
        "weight_decay": 1.0e-6,
        "scheduler_kind": "reduce_on_plateau",
        "scheduler_monitor": "total_loss",
        "scheduler_mode": "min",
        "max_epochs": 4,
        "early_stop_patience": 3,
        "output_directory": "/display/run-output",
        "initial_bundle_fingerprint": _HASH_A,
        "train_semantic_digest": _HASH_B,
        "validation_semantic_digest": _HASH_C,
    }
    values.update(overrides)
    return TrainingStartSummary(**values)  # type: ignore[arg-type]


def _event(
    epoch: int = 0,
    *,
    start: int = 0,
    end: int = 2,
    is_best: bool = True,
    should_stop: bool = False,
) -> CommittedEpochMetrics:
    validation_total = 2.0 + epoch
    best_epoch = epoch if is_best else 0
    best_value = validation_total if is_best else 2.0
    return CommittedEpochMetrics(
        epoch_index=epoch,
        global_step_start=start,
        global_step_end=end,
        successful_optimizer_steps=end - start,
        training_metric_semantics="pre_update_batch_observations",
        validation_metric_semantics="fixed_model_validation",
        training_total_loss=1.25 + epoch,
        validation_total_loss=validation_total,
        training_energy=(2.0, 2.0, 1.0, 2),
        training_force=(0.0, 0.0, 0.0, 0),
        training_stress=(6.0, 3.0, 2.0, 3),
        validation_energy=(3.0, 3.0, 1.0, 3),
        validation_force=(8.0, 4.0, 2.0, 4),
        validation_stress=(0.0, 0.0, 0.0, 0),
        learning_rates_before_scheduler=(5.0e-4,),
        learning_rates_after_scheduler=(2.5e-4,),
        monitored_metric_name="total_loss",
        monitored_metric_value=validation_total,
        monitored_metric_mode="min",
        is_best=is_best,
        should_stop=should_stop,
        best_epoch=best_epoch,
        best_value=best_value,
        bad_validation_count=0 if is_best else epoch,
        epoch_checkpoint_basename=f"epoch_{epoch:06d}.pt",
        latest_checkpoint_basename="latest.pt",
        best_checkpoint_basename="best.pt" if is_best else None,
        provenance=CommittedEpochProvenance(
            initial_bundle_fingerprint=_HASH_A,
            training_configuration_fingerprint=_HASH_B,
            train_data_fingerprint=_HASH_B,
            validation_data_fingerprint=_HASH_C,
            template_fingerprints=(("a-template", _HASH_A),),
        ),
    )


class _Clock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class _FailingStream:
    def __init__(self, error: OSError) -> None:
        self.error = error
        self.writes = 0

    def write(self, value: str) -> int:
        del value
        self.writes += 1
        raise self.error

    def flush(self) -> None:
        raise AssertionError("flush must not follow a failed write")


class _FlushFailingStream:
    def __init__(self, error: OSError) -> None:
        self.error = error
        self.flushes = 0
        self.values: list[str] = []

    def write(self, value: str) -> int:
        self.values.append(value)
        return len(value)

    def flush(self) -> None:
        self.flushes += 1
        raise self.error

    def getvalue(self) -> str:
        return "".join(self.values)


def test_progress_config_is_frozen_and_strict() -> None:
    config = TrainingProgressConfig(enabled=False, float_precision=8)
    assert config.enabled is False
    assert config.float_precision == 8
    with pytest.raises(FrozenInstanceError):
        config.enabled = True  # type: ignore[misc]
    with pytest.raises(TypeError, match="bool"):
        TrainingProgressConfig(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exceed"):
        TrainingProgressConfig(float_precision=18)


def test_start_summary_is_frozen_canonical_and_rejects_nonfinite() -> None:
    source_species = [41, 6]
    source_templates = [["z-template", 8], ["a-template", 4]]
    summary = _summary(
        species_vocabulary=source_species,
        templates=source_templates,
    )
    source_species.append(8)
    source_templates[0][1] = 999

    assert summary.species_vocabulary == (6, 41)
    assert summary.templates == (("a-template", 4), ("z-template", 8))
    assert tuple(item[0] for item in summary.label_presence) == (
        "energy",
        "force",
        "stress",
    )
    assert tuple(item[0] for item in summary.loss_terms) == (
        "energy",
        "force",
        "stress",
    )
    with pytest.raises(FrozenInstanceError):
        summary.run_name = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        _summary(r_ot=math.nan)
    with pytest.raises(ValueError, match="finite"):
        _summary(initial_learning_rate=math.inf)

    without_default = _summary(default_template_id=None)
    assert without_default.default_template_id is None


def test_start_block_contains_runtime_snapshot_and_resume_context() -> None:
    stream = io.StringIO()
    summary = _summary(
        resumed=True,
        resume_checkpoint_epoch=1,
        resume_global_step=7,
        existing_best_epoch=0,
        existing_best_value=1.25,
        recovered_journal_event_count=1,
    )
    renderer = TrainingProgressRenderer(stream=stream, monotonic=_Clock(10.0))
    renderer.render_stage("loading training configuration")
    renderer.render_start(summary)
    text = stream.getvalue()

    assert text.startswith("refsite-mlip: loading training configuration\n")
    assert "Reference-site MLIP training\n" in text
    assert "  Source: scratch (resumed)\n" in text
    assert "  Device: cpu, dtype=float64\n" in text
    assert "  Species: C(6), Nb(41)\n" in text
    assert "a-template (M=4, default), z-template (M=8)" in text
    assert "E=train:yes/validation:yes" in text
    assert "S=train:partial(1/2)/validation:no" in text
    assert "r_ot=4 A, r_mp=3 A, candidates=(4.2, 3.5) A" in text
    assert "Baseline: minimum_norm/fitted [-1.25, 2.5]" in text
    assert "Resume: checkpoint_epoch=2, step=7" in text
    assert "journal_recovered=1" in text


def test_epoch_block_is_one_based_fixed_lines_and_formats_na() -> None:
    stream = io.StringIO()
    renderer = TrainingProgressRenderer(
        TrainingProgressConfig(float_precision=6, duration_precision=1),
        stream=stream,
        monotonic=_Clock(100.0, 104.0),
    )
    renderer.render_start(_summary())
    renderer.render_epoch(_event())
    lines = stream.getvalue().splitlines()
    epoch_lines = lines[-4:]

    assert epoch_lines[0] == "Epoch 001/4 | step=2"
    assert epoch_lines[1] == (
        "  train [pre_update_batch_observations] total=1.25 "
        "E=1 F=n/a S=2"
    )
    assert epoch_lines[2] == (
        "  valid [fixed_model_validation] total=2 E=1 F=2 S=n/a"
    )
    assert "lr(before)=0.0005 lr(next)=0.00025" in epoch_lines[3]
    assert "best=yes stop=no checkpoint=epoch_000000.pt" in epoch_lines[3]
    assert epoch_lines[3].endswith("elapsed=4.0s eta=12.0s")
    assert renderer.session_event_count == 1


def test_resume_elapsed_eta_uses_only_new_session_events() -> None:
    stream = io.StringIO()
    summary = _summary(
        resumed=True,
        resume_checkpoint_epoch=1,
        resume_global_step=4,
        existing_best_epoch=0,
        existing_best_value=2.0,
        recovered_journal_event_count=1,
    )
    renderer = TrainingProgressRenderer(
        stream=stream, monotonic=_Clock(10.0, 14.0, 20.0)
    )
    renderer.render_start(summary)
    renderer.render_epoch(
        _event(2, start=4, end=6, is_best=False, should_stop=False)
    )
    renderer.render_epoch(
        _event(3, start=6, end=8, is_best=False, should_stop=True)
    )
    epoch_headers = [
        line for line in stream.getvalue().splitlines() if line.startswith("Epoch ")
    ]
    metric_lines = [
        line for line in stream.getvalue().splitlines() if "elapsed=" in line
    ]
    assert epoch_headers == ["Epoch 003/4 | step=6", "Epoch 004/4 | step=8"]
    assert metric_lines[0].endswith("elapsed=4.0s eta=4.0s")
    assert metric_lines[1].endswith("elapsed=10.0s eta=0.0s")


def test_early_stop_event_reports_zero_eta_before_configured_maximum() -> None:
    stream = io.StringIO()
    renderer = TrainingProgressRenderer(
        stream=stream,
        monotonic=_Clock(10.0, 14.0),
    )
    renderer.render_start(_summary(max_epochs=10))
    renderer.render_epoch(
        _event(1, start=2, end=4, is_best=False, should_stop=True)
    )

    metric_line = stream.getvalue().splitlines()[-1]
    assert "stop=yes" in metric_line
    assert metric_line.endswith("elapsed=4.0s eta=0.0s")


@pytest.mark.parametrize("error", [BrokenPipeError("closed"), OSError("failed")])
def test_stream_failure_disables_progress_without_raising(error: OSError) -> None:
    stream = _FailingStream(error)
    renderer = TrainingProgressRenderer(stream=stream)

    renderer.render_stage("loading training configuration")
    renderer.render_epoch(_event())
    renderer.render_terminal("completed", epochs=1, global_step=2)

    assert stream.writes == 1
    assert not renderer.enabled
    assert isinstance(renderer.presentation_error, TrainingProgressError)
    assert renderer.presentation_error.original_error is error


@pytest.mark.parametrize("error", [BrokenPipeError("closed"), OSError("failed")])
def test_flush_failure_disables_progress_without_second_write(error: OSError) -> None:
    stream = _FlushFailingStream(error)
    renderer = TrainingProgressRenderer(stream=stream)

    renderer.render_stage("loading training configuration")
    first_bytes = stream.getvalue()
    renderer.render_stage("must not be written")
    renderer.render_terminal("completed", epochs=0, global_step=0)

    assert first_bytes == "refsite-mlip: loading training configuration\n"
    assert stream.getvalue() == first_bytes
    assert stream.flushes == 1
    assert renderer.presentation_error is not None
    assert renderer.presentation_error.original_error is error


def test_run_log_replays_early_output_and_matches_console(tmp_path: Path) -> None:
    stream = io.StringIO()
    renderer = TrainingProgressRenderer(
        stream=stream,
        monotonic=_Clock(10.0, 12.0),
    )
    renderer.render_stage("loading training configuration")
    log_path = tmp_path / "training.log"
    renderer.attach_log(log_path, append=False)
    renderer.render_start(_summary(max_epochs=1))
    renderer.render_epoch(_event())
    renderer.render_terminal(
        "completed",
        epochs=1,
        global_step=2,
        latest_checkpoint="latest.pt",
    )
    renderer.close_log()

    assert renderer.log_path == log_path.absolute()
    assert renderer.log_error is None
    assert log_path.read_text(encoding="utf-8") == stream.getvalue()
    assert log_path.read_text(encoding="utf-8").startswith(
        "refsite-mlip: loading training configuration\n"
    )


def test_resume_log_appends_and_preserves_existing_prefix(tmp_path: Path) -> None:
    log_path = tmp_path / "training.log"
    first_stream = io.StringIO()
    first = TrainingProgressRenderer(stream=first_stream)
    first.render_stage("first session")
    first.attach_log(log_path, append=False)
    first.render_terminal("completed", epochs=1, global_step=2)
    first.close_log()
    prefix = log_path.read_bytes()

    resumed_stream = io.StringIO()
    resumed = TrainingProgressRenderer(stream=resumed_stream)
    resumed.render_stage("loading resumed run")
    resumed.attach_log(log_path, append=True)
    resumed.render_terminal("completed", epochs=2, global_step=4)
    resumed.close_log()

    assert log_path.read_bytes().startswith(prefix)
    assert log_path.read_bytes()[len(prefix) :] == resumed_stream.getvalue().encode()


def test_console_failure_does_not_disable_attached_run_log(tmp_path: Path) -> None:
    log_path = tmp_path / "training.log"
    renderer = TrainingProgressRenderer(
        stream=_FailingStream(BrokenPipeError("closed"))
    )
    renderer.attach_log(log_path, append=False)
    renderer.render_stage("first")
    renderer.render_stage("second")
    renderer.render_terminal("completed", epochs=1, global_step=2)
    renderer.close_log()

    assert renderer.presentation_error is not None
    assert renderer.log_error is None
    assert log_path.read_text(encoding="utf-8") == (
        "refsite-mlip: first\n"
        "refsite-mlip: second\n"
        "Training completed | epochs=1 step=2 best_epoch=n/a "
        "best=n/a latest=n/a\n"
    )


def test_run_log_rejects_symlink_without_interrupting_console(tmp_path: Path) -> None:
    target = tmp_path / "target.log"
    target.write_text("owned by another target\n", encoding="utf-8")
    link = tmp_path / "training.log"
    link.symlink_to(target)
    stream = io.StringIO()
    renderer = TrainingProgressRenderer(stream=stream)

    renderer.render_stage("before attach")
    renderer.attach_log(link, append=True)
    renderer.render_stage("after attach")

    assert renderer.log_error is not None
    assert renderer.enabled
    assert target.read_text(encoding="utf-8") == "owned by another target\n"
    assert "before attach" in stream.getvalue()
    assert "after attach" in stream.getvalue()


def test_quiet_suppresses_progress_but_not_terminal_result() -> None:
    stream = io.StringIO()
    renderer = TrainingProgressRenderer(
        TrainingProgressConfig(enabled=False), stream=stream
    )
    renderer.render_stage("loading training configuration")
    renderer.render_start(_summary())
    renderer.render_epoch(_event())
    renderer.render_terminal(
        "completed",
        epochs=1,
        global_step=2,
        best_epoch=0,
        best_value=2.0,
        latest_checkpoint="latest.pt",
    )
    assert stream.getvalue() == (
        "Training completed | epochs=1 step=2 best_epoch=1 "
        "best=2 latest=latest.pt\n"
    )
    assert renderer.session_event_count == 0


def test_lazy_start_summary_is_skipped_when_quiet_and_failure_is_nonfatal() -> None:
    quiet = TrainingProgressRenderer(
        TrainingProgressConfig(enabled=False), stream=io.StringIO()
    )

    def must_not_build() -> TrainingStartSummary:
        raise AssertionError("quiet presentation evaluated summary factory")

    quiet.render_start_from(must_not_build)
    assert quiet.presentation_error is None

    error = ValueError("synthetic summary projection failure")
    visible = TrainingProgressRenderer(stream=io.StringIO())

    def fail() -> TrainingStartSummary:
        raise error

    visible.render_start_from(fail)
    assert visible.presentation_error is not None
    assert visible.presentation_error.original_error is error
    visible.render_stage("must remain disabled")


def test_terminal_variants_are_single_line_and_sanitize_context() -> None:
    stream = io.StringIO()
    renderer = TrainingProgressRenderer(stream=stream)
    renderer.render_terminal(
        "early_stopped",
        epochs=3,
        global_step=6,
        best_epoch=1,
        reason="patience\nexhausted",
    )
    renderer.render_terminal(
        "failed",
        epochs=2,
        global_step=4,
        phase="metrics_journal",
        recoverable="latest.pt",
    )
    renderer.render_terminal(
        "interrupted", epochs=2, global_step=4, recoverable="latest.pt"
    )
    assert stream.getvalue().splitlines() == [
        "Training early-stopped | epoch=3 step=6 best_epoch=2 "
        "reason=patience exhausted",
        "Training failed | phase=metrics_journal epoch=2 step=4 "
        "recoverable=latest.pt",
        "Training interrupted | epoch=2 step=4 recoverable=latest.pt",
    ]


def test_nonfinite_terminal_value_is_not_written_or_raised() -> None:
    stream = io.StringIO()
    renderer = TrainingProgressRenderer(stream=stream)
    renderer.render_terminal(
        "completed", epochs=1, global_step=2, best_value=float("nan")
    )
    assert stream.getvalue() == ""
    assert isinstance(renderer.presentation_error, TrainingProgressError)


def test_journal_is_called_before_console_and_failure_suppresses_line() -> None:
    order: list[str] = []

    class Journal:
        def __call__(self, event: CommittedEpochMetrics) -> None:
            assert event.epoch_index == 0
            order.append("journal")

    class Stream(io.StringIO):
        def write(self, value: str) -> int:
            order.append("console")
            return super().write(value)

    renderer = TrainingProgressRenderer(stream=Stream(), monotonic=_Clock(0.0))
    observer = journal_then_progress_observer(Journal(), renderer)
    observer(_event())
    assert order == ["journal", "console"]

    class FailedJournal:
        def __call__(self, event: CommittedEpochMetrics) -> None:
            del event
            raise RuntimeError("journal failed")

    untouched = io.StringIO()
    failed_observer = journal_then_progress_observer(
        FailedJournal(), TrainingProgressRenderer(stream=untouched)
    )
    with pytest.raises(RuntimeError, match="journal failed"):
        failed_observer(_event())
    assert untouched.getvalue() == ""


def test_renderer_failure_never_escapes_composed_observer() -> None:
    committed: list[int] = []

    def journal(event: CommittedEpochMetrics) -> None:
        committed.append(event.epoch_index)

    renderer = TrainingProgressRenderer(
        stream=_FailingStream(BrokenPipeError("closed")),
        monotonic=_Clock(0.0),
    )
    observer = journal_then_progress_observer(journal, renderer)
    observer(_event())
    observer(_event(1, start=2, end=4, is_best=False))
    assert committed == [0, 1]
    assert renderer.presentation_error is not None
