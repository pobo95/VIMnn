"""Atomic latest/best/immutable-epoch checkpoint file management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Mapping

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_SCOPE,
    TrainingCheckpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .fit import FitEpochRecord


def _positive_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class CheckpointManagerConfig:
    directory: str
    save_epoch_snapshots: bool = True
    epoch_filename_width: int = 6

    def __post_init__(self) -> None:
        if not isinstance(self.directory, (str, Path)):
            raise TypeError("directory must be a path-like string")
        text = str(self.directory)
        if not text:
            raise ValueError("directory must be nonempty")
        root = Path(text).expanduser().resolve(strict=False)
        object.__setattr__(self, "directory", str(root))
        if not isinstance(self.save_epoch_snapshots, bool):
            raise TypeError("save_epoch_snapshots must be a bool")
        object.__setattr__(
            self,
            "epoch_filename_width",
            _positive_integer("epoch_filename_width", self.epoch_filename_width),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CheckpointManagerConfig":
        if not isinstance(values, Mapping):
            raise TypeError("checkpoint manager config requires a mapping")
        return cls(**dict(values))


@dataclass(frozen=True)
class ManagedCheckpointResult:
    epoch_index: int
    global_step: int
    is_best: bool
    epoch_path: str | None
    latest_path: str
    best_path: str | None
    epoch_written: bool
    latest_written: bool
    best_written: bool
    completed_stages: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epoch_index", _nonnegative_integer("epoch_index", self.epoch_index)
        )
        object.__setattr__(
            self, "global_step", _nonnegative_integer("global_step", self.global_step)
        )
        for name in ("is_best", "epoch_written", "latest_written", "best_written"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.latest_path, str) or not self.latest_path:
            raise ValueError("latest_path must be a nonempty string")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["completed_stages"] = list(self.completed_stages)
        return data

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ManagedCheckpointResult":
        if not isinstance(values, Mapping):
            raise TypeError("managed checkpoint result requires a mapping")
        data = dict(values)
        data["completed_stages"] = tuple(data["completed_stages"])
        return cls(**data)


class CheckpointManagerError(RuntimeError):
    """One file stage failed; previously completed files remain recoverable."""

    def __init__(
        self,
        *,
        stage: str,
        epoch_index: int,
        completed_stages: tuple[str, ...],
        epoch_path: Path | None,
        cause: BaseException,
    ) -> None:
        self.stage = stage
        self.epoch_index = epoch_index
        self.completed_stages = completed_stages
        self.epoch_path = None if epoch_path is None else str(epoch_path)
        self.orphan_epoch_snapshot = "epoch" in completed_stages and stage != "epoch"
        self.original_exception_type = type(cause).__name__
        self.original_exception_message = str(cause)
        super().__init__(
            f"checkpoint manager save failed at stage={stage!r}, "
            f"epoch_index={epoch_index}, completed_stages={completed_stages}, "
            f"orphan_epoch_snapshot={self.orphan_epoch_snapshot}; "
            f"cause={self.original_exception_type}: {self.original_exception_message}; "
            "completed files are retained and existing latest/best files are not deleted"
        )


def _validate_epoch_record(record: FitEpochRecord) -> None:
    if not isinstance(record, FitEpochRecord):
        raise TypeError("epoch_record must be a FitEpochRecord")
    epoch = _nonnegative_integer("epoch_record.epoch_index", record.epoch_index)
    if record.training.phase != "train" or record.validation.phase != "validation":
        raise ValueError("epoch record requires train and validation phase results")
    if record.training.epoch_index != epoch or record.validation.epoch_index != epoch:
        raise ValueError("epoch record result epoch indices are inconsistent")
    if record.validation.global_step_start != record.training.global_step_end:
        raise ValueError("validation global step must follow training")
    if record.validation.global_step_end != record.training.global_step_end:
        raise ValueError("validation must not change the global optimizer step")
    if record.decision.learning_rates_after != record.learning_rates_after_validation:
        raise ValueError("decision and record post-validation learning rates differ")


def _validate_checkpoint_record(
    checkpoint: TrainingCheckpoint, record: FitEpochRecord
) -> None:
    if not isinstance(checkpoint, TrainingCheckpoint):
        raise TypeError("checkpoint must be a TrainingCheckpoint")
    if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is unsupported")
    if checkpoint.checkpoint_scope != CHECKPOINT_SCOPE:
        raise ValueError("checkpoint manager only accepts epoch-boundary snapshots")
    if checkpoint.progress.next_batch_index != 0:
        raise ValueError("checkpoint manager rejects mid-epoch progress")
    _validate_epoch_record(record)
    epoch = record.epoch_index
    progress = checkpoint.progress
    if progress.last_completed_epoch != epoch:
        raise ValueError("checkpoint last_completed_epoch does not match record")
    if progress.next_epoch != epoch + 1:
        raise ValueError("checkpoint next_epoch must be record epoch + 1")
    if progress.global_step != record.training.global_step_end:
        raise ValueError("checkpoint global step does not match training result")
    if progress.global_step != record.validation.global_step_end:
        raise ValueError("checkpoint global step does not match validation result")
    state = checkpoint.selection_state
    if state != record.selection_state_after_epoch:
        raise ValueError("checkpoint and epoch-record selection states differ")
    decision = record.decision
    if decision.metric_name == "total_loss":
        metric_value = record.validation.total_loss
    else:
        metric_value = getattr(record.validation, decision.metric_name).mean
    if decision.metric_value != metric_value:
        raise ValueError("decision metric does not match validation epoch result")
    comparisons = (
        ("best_metric", decision.best_metric, state.best_metric),
        ("best_epoch", decision.best_epoch, state.best_epoch),
        ("best_global_step", decision.best_global_step, state.best_global_step),
        (
            "epochs_since_improvement",
            decision.epochs_since_improvement,
            state.epochs_since_improvement,
        ),
        ("validation_events", decision.validation_events, state.validation_events),
    )
    for name, left, right in comparisons:
        if left != right:
            raise ValueError(f"decision and selection state {name} differ")
    if decision.is_best and state.best_epoch != epoch:
        raise ValueError("is_best=True requires current epoch to be selected best")
    if not decision.is_best and state.best_epoch == epoch:
        raise ValueError("is_best=False cannot record current epoch as a new best")
    if decision.should_stop != state.stopped_early:
        raise ValueError("decision stop flag and selection state differ")
    if progress.stopped_early != state.stopped_early:
        raise ValueError("checkpoint progress and selection stop states differ")
    if progress.best_epoch != state.best_epoch:
        raise ValueError("checkpoint progress and selection best epochs differ")
    if progress.best_global_step != state.best_global_step:
        raise ValueError("checkpoint progress and selection best global steps differ")
    if checkpoint.fit_history is None or not checkpoint.fit_history:
        raise ValueError("managed checkpoint requires full nonempty fit history")
    last = checkpoint.fit_history[-1]
    if not isinstance(last, Mapping):
        raise TypeError("checkpoint history entries must be plain mappings")
    if FitEpochRecord.from_dict(last) != record:
        raise ValueError("full checkpoint history does not end with epoch_record")


class CheckpointManager:
    """Manage independent atomic checkpoint files under one resolved root."""

    def __init__(self, config: CheckpointManagerConfig) -> None:
        if not isinstance(config, CheckpointManagerConfig):
            raise TypeError("config must be a CheckpointManagerConfig")
        self.config = config
        self.root = Path(config.directory)
        if self.root.exists() and not self.root.is_dir():
            raise NotADirectoryError(
                f"checkpoint manager directory is an existing file: {self.root}"
            )
        self._epoch_pattern = re.compile(
            rf"epoch_([0-9]{{{config.epoch_filename_width}}})\.pt"
        )

    def _epoch_name(self, epoch_index: int) -> str:
        epoch = _nonnegative_integer("epoch_index", epoch_index)
        if epoch >= 10**self.config.epoch_filename_width:
            raise ValueError("epoch_index does not fit configured filename width")
        return f"epoch_{epoch:0{self.config.epoch_filename_width}d}.pt"

    def _path(self, filename: str) -> Path:
        if filename not in ("latest.pt", "best.pt") and self._epoch_pattern.fullmatch(
            filename
        ) is None:
            raise ValueError("unsafe managed checkpoint filename")
        candidate = self.root / filename
        if candidate.parent != self.root:
            raise ValueError("managed checkpoint path escaped manager root")
        return candidate

    def _safe_existing(self, path: Path, *, label: str) -> Path:
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(f"managed {label} checkpoint does not exist: {path}")
        if path.is_symlink():
            raise ValueError(f"managed {label} checkpoint must not be a symlink: {path}")
        if not path.is_file():
            raise ValueError(f"managed {label} checkpoint is not a regular file: {path}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                f"managed {label} checkpoint resolves outside manager root"
            ) from error
        return resolved

    def save_epoch(
        self,
        checkpoint: TrainingCheckpoint,
        epoch_record: FitEpochRecord,
    ) -> ManagedCheckpointResult:
        """Validate fully, then save epoch, latest, and optionally best in order."""

        _validate_checkpoint_record(checkpoint, epoch_record)
        epoch = epoch_record.epoch_index
        epoch_path = (
            self._path(self._epoch_name(epoch))
            if self.config.save_epoch_snapshots
            else None
        )
        latest_path = self._path("latest.pt")
        best_path = self._path("best.pt") if epoch_record.decision.is_best else None
        for target in (epoch_path, latest_path, best_path):
            if target is not None and target.is_symlink():
                raise ValueError(
                    f"managed checkpoint save target must not be a symlink: {target}"
                )
        completed: list[str] = []
        stage = "directory"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if not self.root.is_dir():
                raise NotADirectoryError(f"checkpoint manager root is not a directory")
            if epoch_path is not None:
                stage = "epoch"
                save_training_checkpoint(checkpoint, epoch_path, overwrite=False)
                completed.append("epoch")
            stage = "latest"
            save_training_checkpoint(checkpoint, latest_path, overwrite=True)
            completed.append("latest")
            if best_path is not None:
                stage = "best"
                save_training_checkpoint(checkpoint, best_path, overwrite=True)
                completed.append("best")
        except Exception as error:
            raise CheckpointManagerError(
                stage=stage,
                epoch_index=epoch,
                completed_stages=tuple(completed),
                epoch_path=epoch_path,
                cause=error,
            ) from error
        return ManagedCheckpointResult(
            epoch_index=epoch,
            global_step=checkpoint.progress.global_step,
            is_best=epoch_record.decision.is_best,
            epoch_path=None if epoch_path is None else str(epoch_path),
            latest_path=str(latest_path),
            best_path=None if best_path is None else str(best_path),
            epoch_written=epoch_path is not None,
            latest_written=True,
            best_written=best_path is not None,
            completed_stages=tuple(completed),
        )

    def load_latest(self) -> TrainingCheckpoint:
        path = self._safe_existing(self._path("latest.pt"), label="latest")
        return load_training_checkpoint(path)

    def load_best(self) -> TrainingCheckpoint:
        path = self._safe_existing(self._path("best.pt"), label="best")
        return load_training_checkpoint(path)

    def load_epoch(self, epoch_index: int) -> TrainingCheckpoint:
        epoch = _nonnegative_integer("epoch_index", epoch_index)
        path = self._safe_existing(
            self._path(self._epoch_name(epoch)), label=f"epoch {epoch}"
        )
        return load_training_checkpoint(path)

    def list_epochs(self) -> tuple[int, ...]:
        if not self.root.exists():
            return ()
        if not self.root.is_dir():
            raise NotADirectoryError(f"checkpoint manager root is not a directory")
        epochs = []
        for entry in self.root.iterdir():
            match = self._epoch_pattern.fullmatch(entry.name)
            if match is None:
                continue
            self._safe_existing(entry, label=f"epoch {match.group(1)}")
            epochs.append(int(match.group(1)))
        return tuple(sorted(epochs))


__all__ = [
    "CheckpointManager",
    "CheckpointManagerConfig",
    "CheckpointManagerError",
    "ManagedCheckpointResult",
]
