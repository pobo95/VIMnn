"""Fresh scratch training through the established checkpointed-fit engine.

This module owns orchestration only.  Scratch startup constructs the durable
initial bundle and all live runtime objects; the existing checkpointed-fit
engine remains the sole owner of epoch execution and checkpoint capture.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from refsite_mlip.transport import TRAIN_FIXED

from .checkpoint import FitProgress, capture_training_checkpoint
from ._scratch_run_metadata import scratch_runtime_template_fingerprints
from .checkpoint_manager import CheckpointManager, CheckpointManagerConfig
from .checkpointed_fit import CheckpointedFitResult, run_checkpointed_fit
from .run_directory import (
    RUN_STATUS_SCHEMA_VERSION,
    ResumeRunLock,
    TrainingRunDirectory,
    canonical_runtime_json,
)
from .scratch_preparation import (
    ScratchTrainingPreparation,
    verify_scratch_preparation_input_digests,
)
from .scratch_startup import (
    ScratchTrainingStartup,
    initialize_scratch_training_startup,
)

if TYPE_CHECKING:
    from refsite_mlip.config import TrainingRunConfig


SCRATCH_CHECKPOINTED_TRAINING_RESULT_SCHEMA_VERSION = (
    "refsite_scratch_checkpointed_training_result_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _plain(value: Any, *, path: str = "value") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} mapping keys must be strings")
            result[key] = _plain(item, path=f"{path}.{key}")
        return dict(sorted(result.items()))
    if isinstance(value, (tuple, list)):
        return [_plain(item, path=f"{path}[]") for item in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or Infinity")
        return value
    if value is None or type(value) in (str, bool, int):
        return value
    raise TypeError(f"{path} contains non-plain {type(value).__name__}")


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Own a deterministic strict-JSON round trip."""

    return json.loads(canonical_runtime_json(_plain(value)))


def _freeze_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_plain(item)
                for key, item in sorted(value.items())
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_plain(item) for item in value)
    if type(value) is float and not math.isfinite(value):
        raise ValueError("metadata contains NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"metadata contains non-plain {type(value).__name__}")


def _sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _nested_attribute(error: BaseException, name: str) -> Any:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, name, None)
        if value is not None:
            return value
        current = current.__cause__ or current.__context__
    return None


def _nested_exception(
    error: BaseException, expected_type: type[BaseException]
) -> BaseException | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, expected_type):
            return current
        current = current.__cause__ or current.__context__
    return None


def _default_reason(stage: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", stage).strip("_").upper()
    return f"SCRATCH_CHECKPOINTED_TRAINING_{normalized or 'RUNTIME'}_FAILED"


class ScratchCheckpointedTrainingError(RuntimeError):
    """Structured scratch-run failure with explicit recovery state."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        output_path: str | None = None,
        config_fingerprint: str | None = None,
        preparation_fingerprint: str | None = None,
        initial_bundle_fingerprint: str | None = None,
        template_fingerprints: Mapping[str, Any] | None = None,
        train_semantic_digest: str | None = None,
        validation_semantic_digest: str | None = None,
        completed_epochs: int = 0,
        global_step: int = 0,
        latest_checkpoint: str | None = None,
        best_checkpoint: str | None = None,
        recoverable_checkpoint: str | None = None,
        recoverable_initial_bundle: str | None = None,
        interrupted: bool = False,
        rollback_performed: bool = False,
        original_reason_code: str | None = None,
        original_error: BaseException | None = None,
        status_write_error: BaseException | None = None,
        lock_release_error: BaseException | None = None,
    ) -> None:
        if type(reason_code) is not str or not reason_code:
            raise ValueError("reason_code must be a nonempty string")
        if type(message) is not str or not message:
            raise ValueError("message must be a nonempty string")
        if type(stage) is not str or not stage:
            raise ValueError("stage must be a nonempty string")
        if type(completed_epochs) is not int or completed_epochs < 0:
            raise ValueError("completed_epochs must be a nonnegative integer")
        if type(global_step) is not int or global_step < 0:
            raise ValueError("global_step must be a nonnegative integer")
        if type(interrupted) is not bool or type(rollback_performed) is not bool:
            raise TypeError("interrupted and rollback_performed must be bools")

        self.reason_code = reason_code
        self.message = message
        self.stage = stage
        self.output_path = output_path
        self.config_fingerprint = config_fingerprint
        self.preparation_fingerprint = preparation_fingerprint
        self.initial_bundle_fingerprint = initial_bundle_fingerprint
        self.bundle_fingerprint = initial_bundle_fingerprint
        self.template_fingerprints = _freeze_plain(
            {} if template_fingerprints is None else template_fingerprints
        )
        self.train_semantic_digest = train_semantic_digest
        self.validation_semantic_digest = validation_semantic_digest
        self.completed_epochs = completed_epochs
        self.global_step = global_step
        self.latest_checkpoint = latest_checkpoint
        self.best_checkpoint = best_checkpoint
        self.recoverable_checkpoint = recoverable_checkpoint
        self.recoverable_initial_bundle = recoverable_initial_bundle
        self.interrupted = interrupted
        self.rollback_performed = rollback_performed
        self.original_reason_code = original_reason_code
        self.original_error = original_error
        self.original_exception_type = (
            None
            if original_error is None
            else (
                getattr(original_error, "original_exception_type", None)
                or type(original_error).__name__
            )
        )
        self.original_exception_message = (
            None
            if original_error is None
            else (
                getattr(original_error, "original_exception_message", None)
                or str(original_error)
            )
        )
        self.status_write_error = status_write_error
        self.status_write_exception_type = (
            None if status_write_error is None else type(status_write_error).__name__
        )
        self.status_write_exception_message = (
            None if status_write_error is None else str(status_write_error)
        )
        self.lock_release_error = lock_release_error
        self.lock_release_exception_type = (
            None if lock_release_error is None else type(lock_release_error).__name__
        )
        self.lock_release_exception_message = (
            None if lock_release_error is None else str(lock_release_error)
        )
        super().__init__(
            f"[{reason_code}] stage={stage!r} output_path={output_path!r} "
            f"completed_epochs={completed_epochs} global_step={global_step} "
            f"{message}"
        )

    def attach_lock_release_error(self, error: BaseException) -> None:
        """Preserve a secondary release failure without replacing this error."""

        self.lock_release_error = error
        self.lock_release_exception_type = type(error).__name__
        self.lock_release_exception_message = str(error)

    def to_dict(self) -> dict[str, Any]:
        return _canonical_mapping(
            {
                "best_checkpoint": self.best_checkpoint,
                "bundle_fingerprint": self.bundle_fingerprint,
                "completed_epochs": self.completed_epochs,
                "config_fingerprint": self.config_fingerprint,
                "global_step": self.global_step,
                "initial_bundle_fingerprint": self.initial_bundle_fingerprint,
                "interrupted": self.interrupted,
                "latest_checkpoint": self.latest_checkpoint,
                "lock_release_exception_message": (
                    self.lock_release_exception_message
                ),
                "lock_release_exception_type": self.lock_release_exception_type,
                "message": self.message,
                "original_exception_message": self.original_exception_message,
                "original_exception_type": self.original_exception_type,
                "original_reason_code": self.original_reason_code,
                "output_path": self.output_path,
                "preparation_fingerprint": self.preparation_fingerprint,
                "reason_code": self.reason_code,
                "recoverable_checkpoint": self.recoverable_checkpoint,
                "recoverable_initial_bundle": self.recoverable_initial_bundle,
                "rollback_performed": self.rollback_performed,
                "stage": self.stage,
                "status_write_exception_message": (
                    self.status_write_exception_message
                ),
                "status_write_exception_type": self.status_write_exception_type,
                "template_fingerprints": _plain(self.template_fingerprints),
                "train_semantic_digest": self.train_semantic_digest,
                "validation_semantic_digest": self.validation_semantic_digest,
            }
        )


@dataclass(frozen=True)
class ScratchCheckpointedTrainingResult:
    """Scratch startup state composed with the established fit result."""

    startup: ScratchTrainingStartup
    checkpointed_fit_result: CheckpointedFitResult
    terminal_status: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.startup, ScratchTrainingStartup):
            raise TypeError("startup must be a ScratchTrainingStartup")
        if not isinstance(self.checkpointed_fit_result, CheckpointedFitResult):
            raise TypeError(
                "checkpointed_fit_result must be a CheckpointedFitResult"
            )
        status = _canonical_mapping(self.terminal_status)
        expected_status = (
            "early_stopped"
            if self.checkpointed_fit_result.fit_result.stopped_early
            else "completed"
        )
        if status.get("status") != expected_status:
            raise ValueError("terminal status differs from fit stop state")
        if status.get("latest_checkpoint") != (
            self.checkpointed_fit_result.latest_path
        ):
            raise ValueError("terminal status latest checkpoint differs")
        if status.get("best_checkpoint") != self.checkpointed_fit_result.best_path:
            raise ValueError("terminal status best checkpoint differs")
        if status.get("bundle_fingerprint") != (
            self.startup.initial_bundle_fingerprint
        ):
            raise ValueError("terminal status initial bundle differs")
        object.__setattr__(self, "terminal_status", _freeze_plain(status))

    @property
    def fit_result(self):
        return self.checkpointed_fit_result.fit_result

    @property
    def latest_path(self) -> str:
        return self.checkpointed_fit_result.latest_path

    @property
    def best_path(self) -> str:
        return self.checkpointed_fit_result.best_path

    @property
    def initial_bundle_fingerprint(self) -> str:
        return self.startup.initial_bundle_fingerprint

    @property
    def train_semantic_digest(self) -> str:
        return self.startup.train_semantic_digest

    @property
    def validation_semantic_digest(self) -> str:
        return self.startup.validation_semantic_digest

    @property
    def baseline_fit_metadata(self) -> Mapping[str, Any]:
        return self.startup.baseline_metadata

    @property
    def status(self) -> str:
        return str(self.terminal_status["status"])

    @property
    def final_selection(self):
        return self.fit_result.final_selection_state

    @property
    def final_progress(self) -> FitProgress:
        fit = self.fit_result
        return FitProgress(
            next_epoch=fit.next_epoch,
            global_step=fit.global_step_end,
            completed_epochs=fit.epochs_completed,
            next_batch_index=0,
            last_completed_epoch=fit.next_epoch - 1,
            stopped_early=fit.stopped_early,
            best_epoch=fit.best_epoch,
            best_global_step=fit.best_global_step,
        )

    @property
    def completed_epochs(self) -> int:
        return self.fit_result.epochs_completed

    @property
    def global_step(self) -> int:
        return self.fit_result.global_step_end

    @property
    def stopped_early(self) -> bool:
        return self.fit_result.stopped_early

    @property
    def stop_reason(self) -> str | None:
        return self.fit_result.stop_reason

    @property
    def terminal_model_is_best(self) -> bool:
        return self.fit_result.terminal_model_is_best

    @property
    def run_directory(self) -> str:
        return str(self.startup.run_directory.root)

    @property
    def recoverability(self) -> Mapping[str, Any]:
        return self.terminal_status["recovery"]

    def to_dict(self) -> dict[str, Any]:
        terminal = _plain(self.terminal_status)
        startup = {
            "baseline": _plain(self.startup.baseline_metadata),
            "bundle_fingerprint": self.startup.initial_bundle_fingerprint,
            "config_fingerprint": terminal["config_fingerprint"],
            "data_manifest_fingerprint": terminal["data_manifest_fingerprint"],
            "initial_bundle_fingerprint": (
                self.startup.initial_bundle_fingerprint
            ),
            "initialization_seed": self.startup.initialization_seed,
            "paths": _plain(self.startup.run_directory_paths),
            "preparation_fingerprint": terminal["preparation_fingerprint"],
            "runtime": terminal["runtime"],
            "seed": self.startup.training_seed,
            "template_fingerprints": terminal["template_fingerprints"],
            "train_semantic_digest": self.startup.train_semantic_digest,
            "validation_semantic_digest": (
                self.startup.validation_semantic_digest
            ),
        }
        return _canonical_mapping(
            {
                "checkpointed_fit_result": (
                    self.checkpointed_fit_result.to_dict()
                ),
                "startup": startup,
                "terminal_status": terminal,
            }
        )


@dataclass(frozen=True)
class _RecoveryState:
    completed_epochs: int
    global_step: int
    latest_checkpoint: str | None
    best_checkpoint: str | None
    recoverable_checkpoint: str | None
    recoverable_initial_bundle: str | None
    recovery_kind: str | None
    recovery_path: str | None
    selection_state: Mapping[str, Any] | None


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _recovery_state(
    directory: TrainingRunDirectory,
    manager: CheckpointManager | None,
    startup: ScratchTrainingStartup | None,
) -> _RecoveryState:
    latest_path = directory.checkpoints / "latest.pt"
    best_path = directory.checkpoints / "best.pt"
    initial_path = directory.initial_bundle_path
    latest_text = str(latest_path) if _regular_file(latest_path) else None
    best_text = str(best_path) if _regular_file(best_path) else None
    initial_text = str(initial_path) if _regular_file(initial_path) else None
    checkpoint = None
    if latest_text is not None and manager is not None:
        try:
            checkpoint = manager.load_latest()
        except Exception:
            checkpoint = None
    if checkpoint is not None:
        return _RecoveryState(
            completed_epochs=checkpoint.progress.completed_epochs,
            global_step=checkpoint.progress.global_step,
            latest_checkpoint=latest_text,
            best_checkpoint=best_text,
            recoverable_checkpoint=latest_text,
            recoverable_initial_bundle=initial_text,
            recovery_kind="latest_checkpoint",
            recovery_path=latest_text,
            selection_state=checkpoint.selection_state.to_dict(),
        )
    selection = (
        None
        if startup is None
        else startup.initial_selection_state.to_dict()
    )
    return _RecoveryState(
        completed_epochs=0,
        global_step=0,
        latest_checkpoint=latest_text,
        best_checkpoint=best_text,
        recoverable_checkpoint=None,
        recoverable_initial_bundle=initial_text,
        recovery_kind=("initial_bundle" if initial_text is not None else None),
        recovery_path=initial_text,
        selection_state=selection,
    )


def _error_from(
    stage: str,
    error: BaseException,
    *,
    config: TrainingRunConfig,
    preparation: ScratchTrainingPreparation,
    directory: TrainingRunDirectory | None,
    manager: CheckpointManager | None,
    startup: ScratchTrainingStartup | None,
    interrupted: bool,
    status_write_error: BaseException | None = None,
    lock_release_error: BaseException | None = None,
) -> ScratchCheckpointedTrainingError:
    config_fingerprint = getattr(config, "config_fingerprint", None)
    preparation_fingerprint = getattr(
        preparation, "preparation_fingerprint", None
    )
    template_fingerprints = getattr(preparation, "template_fingerprints", {})
    train_digest = getattr(preparation, "train_semantic_digest", None)
    validation_digest = getattr(preparation, "validation_semantic_digest", None)
    recovery = (
        _RecoveryState(0, 0, None, None, None, None, None, None, None)
        if directory is None
        else _recovery_state(directory, manager, startup)
    )
    if startup is None:
        declared_initial = _nested_attribute(
            error, "recoverable_initial_bundle"
        )
        if type(declared_initial) is str and _regular_file(Path(declared_initial)):
            recovery = _RecoveryState(
                completed_epochs=recovery.completed_epochs,
                global_step=recovery.global_step,
                latest_checkpoint=recovery.latest_checkpoint,
                best_checkpoint=recovery.best_checkpoint,
                recoverable_checkpoint=recovery.recoverable_checkpoint,
                recoverable_initial_bundle=declared_initial,
                recovery_kind=(
                    recovery.recovery_kind
                    if recovery.recoverable_checkpoint is not None
                    else "initial_bundle"
                ),
                recovery_path=(
                    recovery.recovery_path
                    if recovery.recoverable_checkpoint is not None
                    else declared_initial
                ),
                selection_state=recovery.selection_state,
            )
        elif recovery.recoverable_checkpoint is None:
            recovery = _RecoveryState(
                completed_epochs=recovery.completed_epochs,
                global_step=recovery.global_step,
                latest_checkpoint=recovery.latest_checkpoint,
                best_checkpoint=recovery.best_checkpoint,
                recoverable_checkpoint=None,
                recoverable_initial_bundle=None,
                recovery_kind=None,
                recovery_path=None,
                selection_state=recovery.selection_state,
            )
    reason = _nested_attribute(error, "reason_code")
    if type(reason) is not str or not reason:
        reason = _default_reason(stage)
    bundle_fingerprint = (
        startup.initial_bundle_fingerprint
        if startup is not None
        else _nested_attribute(error, "bundle_fingerprint")
    )
    completed = recovery.completed_epochs
    global_step = recovery.global_step
    if completed == 0:
        candidate = getattr(error, "completed_epochs", None)
        if type(candidate) is int and candidate >= 0:
            completed = candidate
    if global_step == 0:
        candidate = getattr(error, "current_global_step", None)
        if type(candidate) is not int:
            candidate = getattr(error, "global_step", None)
        if type(candidate) is int and candidate >= 0:
            global_step = candidate
    return ScratchCheckpointedTrainingError(
        reason,
        "scratch checkpointed training failed: "
        f"{type(error).__name__}: {error}",
        stage=stage,
        output_path=(None if directory is None else str(directory.root)),
        config_fingerprint=config_fingerprint,
        preparation_fingerprint=preparation_fingerprint,
        initial_bundle_fingerprint=bundle_fingerprint,
        template_fingerprints=template_fingerprints,
        train_semantic_digest=train_digest,
        validation_semantic_digest=validation_digest,
        completed_epochs=completed,
        global_step=global_step,
        latest_checkpoint=recovery.latest_checkpoint,
        best_checkpoint=recovery.best_checkpoint,
        recoverable_checkpoint=recovery.recoverable_checkpoint,
        recoverable_initial_bundle=recovery.recoverable_initial_bundle,
        interrupted=interrupted,
        rollback_performed=bool(getattr(error, "rollback_performed", False)),
        original_reason_code=_nested_attribute(error, "reason_code"),
        original_error=error,
        status_write_error=status_write_error,
        lock_release_error=lock_release_error,
    )


def _status(
    name: str,
    *,
    config: TrainingRunConfig,
    preparation: ScratchTrainingPreparation,
    directory: TrainingRunDirectory,
    manager: CheckpointManager | None,
    startup: ScratchTrainingStartup | None,
    fit_result: CheckpointedFitResult | None,
    training_executed: bool,
    failure_phase: str | None = None,
    error: ScratchCheckpointedTrainingError | None = None,
) -> dict[str, Any]:
    recovery = _recovery_state(directory, manager, startup)
    if fit_result is not None:
        fit = fit_result.fit_result
        completed_epochs = fit.epochs_completed
        global_step = fit.global_step_end
        latest = fit_result.latest_path
        best = fit_result.best_path
        recoverable = fit_result.latest_path
        selection = fit.final_selection_state.to_dict()
        fit_payload = fit.to_dict()
    else:
        completed_epochs = recovery.completed_epochs
        global_step = recovery.global_step
        latest = recovery.latest_checkpoint
        best = recovery.best_checkpoint
        recoverable = recovery.recoverable_checkpoint
        selection = recovery.selection_state
        fit_payload = None
        # A train update may have completed before validation or checkpoint
        # persistence failed.  Durable recovery remains anchored to the last
        # checkpoint (or the zero-baseline initial bundle), while status must
        # still report the retained in-memory update/progress truthfully.
        if error is not None:
            completed_epochs = max(completed_epochs, error.completed_epochs)
            global_step = max(global_step, error.global_step)
    bundle_fingerprint = (
        (
            None
            if error is None
            else error.initial_bundle_fingerprint
        )
        if startup is None
        else startup.initial_bundle_fingerprint
    )
    baseline = None if startup is None else _plain(startup.baseline_metadata)
    recoverable_initial = (
        recovery.recoverable_initial_bundle
        if startup is not None or error is None
        else error.recoverable_initial_bundle
    )
    recovery_path = recoverable or recoverable_initial
    recovery_kind = (
        "latest_checkpoint"
        if recoverable is not None
        else ("initial_bundle" if recovery_path is not None else None)
    )
    return _canonical_mapping(
        {
            "baseline": baseline,
            "best_checkpoint": best,
            "bundle_fingerprint": bundle_fingerprint,
            "completed_epochs": completed_epochs,
            "config_fingerprint": preparation.config_fingerprint,
            "data_manifest_fingerprint": preparation.data_manifest["fingerprint"],
            "error": None if error is None else error.to_dict(),
            "failure_phase": failure_phase,
            "first_optimizer_update_executed": global_step > 0,
            "fit_result": fit_payload,
            "global_step": global_step,
            "initial_bundle_fingerprint": bundle_fingerprint,
            "initialization_seed": preparation.model_source.initialization_seed,
            "latest_checkpoint": latest,
            "preparation_fingerprint": preparation.preparation_fingerprint,
            "recoverable_checkpoint": recoverable,
            "recoverable_initial_bundle": recoverable_initial,
            "recovery": {
                "kind": recovery_kind,
                "path": recovery_path,
            },
            "result_schema_version": (
                SCRATCH_CHECKPOINTED_TRAINING_RESULT_SCHEMA_VERSION
            ),
            "rollback_performed": (
                False if error is None else error.rollback_performed
            ),
            "runtime": {
                "device": preparation.resolved_device,
                "dtype": preparation.resolved_dtype,
                "solver_path": TRAIN_FIXED,
            },
            "schema_version": RUN_STATUS_SCHEMA_VERSION,
            "seed": config.runtime.seed,
            "source_kind": "scratch",
            "status": name,
            "template_fingerprints": _plain(
                scratch_runtime_template_fingerprints(
                    preparation,
                    None if startup is None else startup.initial_bundle,
                )
            ),
            "terminal_selection_state": selection,
            "train_semantic_digest": preparation.train_semantic_digest,
            "training_executed": training_executed,
            "validation_semantic_digest": (
                preparation.validation_semantic_digest
            ),
        }
    )


def _validate_inputs(
    config: TrainingRunConfig,
    preparation: ScratchTrainingPreparation,
    progress: Callable[[str], None] | None,
    event_callback: Callable[[str], None] | None,
) -> Path:
    # Import lazily: training-run configuration reuses training dataclasses,
    # so importing it while ``refsite_mlip.training`` is initializing would
    # form a package-level cycle.
    from refsite_mlip.config import (
        ScratchModelSourceConfig,
        TrainingRunConfig,
        validate_training_run_config,
    )
    from refsite_mlip.config.training_run import (
        TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2,
    )

    if not isinstance(config, TrainingRunConfig):
        raise TypeError("config must be a TrainingRunConfig")
    validate_training_run_config(config)
    if config.schema_version != TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2:
        raise ValueError("scratch training requires training-run config schema v2")
    if not isinstance(config.model_source, ScratchModelSourceConfig):
        raise ValueError("scratch training requires model_source.kind='scratch'")
    if not isinstance(preparation, ScratchTrainingPreparation):
        raise TypeError("preparation must be a ScratchTrainingPreparation")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if event_callback is not None and not callable(event_callback):
        raise TypeError("event_callback must be callable or None")
    if config.config_fingerprint != preparation.config_fingerprint:
        raise ValueError("config fingerprint differs from scratch preparation")
    if config.to_dict() != preparation.config.to_dict():
        raise ValueError("config semantics differ from scratch preparation")
    if config.source_path != preparation.config.source_path:
        raise ValueError("config source path differs from scratch preparation")
    if config.output_directory_base != preparation.config.output_directory_base:
        raise ValueError(
            "config output path anchor differs from scratch preparation"
        )
    if config.model_source.to_dict() != preparation.model_source.to_dict():
        raise ValueError("scratch model source differs from preparation")
    if preparation.training_executed:
        raise ValueError("scratch preparation already records training execution")
    if config.fit.start_epoch != 0 or config.fit.global_step_start != 0:
        raise ValueError("fresh scratch training requires zero fit progress")
    _sha256(preparation.preparation_fingerprint, name="preparation_fingerprint")
    _sha256(preparation.train_semantic_digest, name="train_semantic_digest")
    _sha256(
        preparation.validation_semantic_digest,
        name="validation_semantic_digest",
    )
    verify_scratch_preparation_input_digests(preparation)
    output = preparation.runtime_paths.get("output_directory")
    if type(output) is not str or not output:
        raise ValueError("preparation output directory path is missing")
    path = Path(output)
    if not path.is_absolute():
        raise ValueError("prepared output directory must be an absolute path")
    return path


def _emit(callback: Callable[[str], None] | None, event: str) -> None:
    if callback is not None:
        callback(event)


def _write_failure_status(
    directory: TrainingRunDirectory,
    *,
    lock: ResumeRunLock,
    name: str,
    config: TrainingRunConfig,
    preparation: ScratchTrainingPreparation,
    manager: CheckpointManager | None,
    startup: ScratchTrainingStartup | None,
    training_executed: bool,
    stage: str,
    structured: ScratchCheckpointedTrainingError,
) -> BaseException | None:
    try:
        lock.validate_owned(directory.resume_lock_path)
        directory.write_status(
            _status(
                name,
                config=config,
                preparation=preparation,
                directory=directory,
                manager=manager,
                startup=startup,
                fit_result=None,
                training_executed=training_executed,
                failure_phase=stage,
                error=structured,
            )
        )
    except BaseException as error:
        return error
    return None


def run_scratch_checkpointed_training(
    config: TrainingRunConfig,
    preparation: ScratchTrainingPreparation,
    *,
    progress: Callable[[str], None] | None = None,
    event_callback: Callable[[str], None] | None = None,
) -> ScratchCheckpointedTrainingResult:
    """Run one fresh scratch fit without duplicating the epoch loop."""

    try:
        output = _validate_inputs(config, preparation, progress, event_callback)
    except Exception as error:
        raise _error_from(
            "validation",
            error,
            config=config,
            preparation=preparation,
            directory=None,
            manager=None,
            startup=None,
            interrupted=False,
        ) from error

    try:
        directory = TrainingRunDirectory.create(output)
    except Exception as error:
        raise _error_from(
            "run_directory.create",
            error,
            config=config,
            preparation=preparation,
            directory=None,
            manager=None,
            startup=None,
            interrupted=False,
        ) from error

    try:
        lock = directory.acquire_resume_lock()
    except Exception as error:
        raise _error_from(
            "lock.acquire",
            error,
            config=config,
            preparation=preparation,
            directory=directory,
            manager=None,
            startup=None,
            interrupted=False,
        ) from error

    startup: ScratchTrainingStartup | None = None
    manager: CheckpointManager | None = None
    fit_result: CheckpointedFitResult | None = None
    training_executed = False
    phase = "event.lock_acquired"
    outcome: ScratchCheckpointedTrainingResult | None = None
    primary: BaseException | None = None
    primary_traceback = None
    try:
        phase = "status.initializing"
        lock.validate_owned(directory.resume_lock_path)
        directory.write_status(
            _status(
                "initializing",
                config=config,
                preparation=preparation,
                directory=directory,
                manager=None,
                startup=None,
                fit_result=None,
                training_executed=False,
            )
        )
        lock.validate_owned(directory.resume_lock_path)
        phase = "event.lock_acquired"
        _emit(event_callback, "lock_acquired")
        lock.validate_owned(directory.resume_lock_path)

        phase = "startup"
        startup = initialize_scratch_training_startup(
            preparation,
            run_directory=directory,
            run_lock=lock,
        )
        if startup.run_directory.root != directory.root:
            raise RuntimeError("scratch startup returned a different run directory")
        if startup.config.config_fingerprint != config.config_fingerprint:
            raise RuntimeError("scratch startup returned a different config")
        lock.validate_owned(directory.resume_lock_path)

        phase = "event.startup_ready"
        _emit(event_callback, "startup_ready")
        lock.validate_owned(directory.resume_lock_path)
        phase = "status.running"
        directory.write_status(
            _status(
                "running",
                config=config,
                preparation=preparation,
                directory=directory,
                manager=None,
                startup=startup,
                fit_result=None,
                training_executed=False,
            )
        )
        manager = CheckpointManager(
            CheckpointManagerConfig(directory=str(directory.checkpoints))
        )

        phase = "checkpoint.metadata"
        checkpoint_metadata = capture_training_checkpoint(
            startup.model,
            startup.optimizer,
            startup.scheduler,
            startup.initial_selection_state,
            startup.initial_fit_progress,
            startup.train_batches,
            startup.validation_batches,
            model_config=startup.model.config,
            loss_config=config.loss,
            optimizer_config=config.optimizer,
            train_step_config=config.train_step,
            validation_step_config=config.validation_step,
            scheduler_config=config.scheduler,
            model_selection_config=config.selection,
            fit_config=config.fit,
            species_vocabulary=startup.initial_bundle.species_vocabulary,
            fit_history=startup.fit_history,
            baseline_fit_metadata=startup.baseline_metadata,
        ).metadata

        if progress is not None:
            progress(
                "scratch training started: "
                f"epochs={config.fit.max_epochs}, "
                f"train_batches={len(startup.train_batches)}, "
                f"validation_batches={len(startup.validation_batches)}"
            )
        phase = "event.before_fit"
        _emit(event_callback, "before_fit")
        lock.validate_owned(directory.resume_lock_path)
        phase = "fit"
        training_executed = True
        fit_result = run_checkpointed_fit(
            startup.model,
            startup.optimizer,
            startup.scheduler,
            startup.train_batches,
            startup.validation_batches,
            startup.template_contexts,
            config.loss,
            config.train_step,
            config.validation_step,
            config.scheduler,
            config.selection,
            startup.initial_selection_state,
            config.fit,
            manager,
            checkpoint_metadata,
            config.checkpointed_fit,
        )
        phase = "event.after_fit"
        _emit(event_callback, "after_fit")
        lock.validate_owned(directory.resume_lock_path)

        terminal_name = (
            "early_stopped"
            if fit_result.fit_result.stopped_early
            else "completed"
        )
        terminal_status = _status(
            terminal_name,
            config=config,
            preparation=preparation,
            directory=directory,
            manager=manager,
            startup=startup,
            fit_result=fit_result,
            training_executed=True,
        )
        phase = "status.terminal"
        directory.write_status(terminal_status)
        if progress is not None:
            progress(
                "scratch training finished: "
                f"status={terminal_name}, "
                f"epochs={fit_result.fit_result.epochs_completed}, "
                f"global_step={fit_result.fit_result.global_step_end}"
            )
        phase = "event.terminal_status"
        _emit(event_callback, "terminal_status_written")
        outcome = ScratchCheckpointedTrainingResult(
            startup=startup,
            checkpointed_fit_result=fit_result,
            terminal_status=terminal_status,
        )
    except KeyboardInterrupt as error:
        interrupted = _error_from(
            phase,
            error,
            config=config,
            preparation=preparation,
            directory=directory,
            manager=manager,
            startup=startup,
            interrupted=True,
        )
        status_error = _write_failure_status(
            directory,
            lock=lock,
            name="interrupted",
            config=config,
            preparation=preparation,
            manager=manager,
            startup=startup,
            training_executed=training_executed,
            stage=phase,
            structured=interrupted,
        )
        if status_error is not None:
            interrupted.status_write_error = status_error
            interrupted.status_write_exception_type = type(status_error).__name__
            interrupted.status_write_exception_message = str(status_error)
            setattr(error, "status_write_error", status_error)
        # Preserve the complete durable-recovery context for the CLI while
        # keeping the public interruption semantics as KeyboardInterrupt.
        setattr(error, "scratch_training_error", interrupted)
        primary = error
        primary_traceback = error.__traceback__
    except BaseException as error:
        nested_interrupt = _nested_exception(error, KeyboardInterrupt)
        if nested_interrupt is not None:
            assert isinstance(nested_interrupt, KeyboardInterrupt)
            interrupted = _error_from(
                phase,
                error,
                config=config,
                preparation=preparation,
                directory=directory,
                manager=manager,
                startup=startup,
                interrupted=True,
            )
            status_error = _write_failure_status(
                directory,
                lock=lock,
                name="interrupted",
                config=config,
                preparation=preparation,
                manager=manager,
                startup=startup,
                training_executed=training_executed,
                stage=phase,
                structured=interrupted,
            )
            if status_error is not None:
                interrupted.status_write_error = status_error
                interrupted.status_write_exception_type = type(status_error).__name__
                interrupted.status_write_exception_message = str(status_error)
                setattr(nested_interrupt, "status_write_error", status_error)
            setattr(nested_interrupt, "scratch_training_error", interrupted)
            primary = nested_interrupt
            primary_traceback = nested_interrupt.__traceback__
        else:
            structured = _error_from(
                phase,
                error,
                config=config,
                preparation=preparation,
                directory=directory,
                manager=manager,
                startup=startup,
                interrupted=False,
            )
            status_error = _write_failure_status(
                directory,
                lock=lock,
                name="failed",
                config=config,
                preparation=preparation,
                manager=manager,
                startup=startup,
                training_executed=training_executed,
                stage=phase,
                structured=structured,
            )
            if status_error is not None:
                structured.status_write_error = status_error
                structured.status_write_exception_type = type(status_error).__name__
                structured.status_write_exception_message = str(status_error)
            primary = structured
            primary_traceback = structured.__traceback__

    release_error: BaseException | None = None
    try:
        lock.release()
    except BaseException as error:
        release_error = error

    if primary is not None:
        if release_error is not None:
            if isinstance(primary, ScratchCheckpointedTrainingError):
                primary.attach_lock_release_error(release_error)
            else:
                interrupted = getattr(primary, "scratch_training_error", None)
                if isinstance(interrupted, ScratchCheckpointedTrainingError):
                    interrupted.attach_lock_release_error(release_error)
            raise primary.with_traceback(primary_traceback) from release_error
        raise primary.with_traceback(primary_traceback)

    if release_error is not None:
        structured = _error_from(
            "lock.release",
            release_error,
            config=config,
            preparation=preparation,
            directory=directory,
            manager=manager,
            startup=startup,
            interrupted=False,
            lock_release_error=release_error,
        )
        if lock.owned:
            status_error = _write_failure_status(
                directory,
                lock=lock,
                name="failed",
                config=config,
                preparation=preparation,
                manager=manager,
                startup=startup,
                training_executed=training_executed,
                stage="lock.release",
                structured=structured,
            )
            if status_error is not None:
                structured.status_write_error = status_error
                structured.status_write_exception_type = type(status_error).__name__
                structured.status_write_exception_message = str(status_error)
        raise structured from release_error

    if outcome is None:
        raise RuntimeError("scratch checkpointed training produced no outcome")
    return outcome


__all__ = [
    "SCRATCH_CHECKPOINTED_TRAINING_RESULT_SCHEMA_VERSION",
    "ScratchCheckpointedTrainingError",
    "ScratchCheckpointedTrainingResult",
    "run_scratch_checkpointed_training",
]
