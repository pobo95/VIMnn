"""Exclusive training-run directories and atomic machine-readable JSON state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


RUN_STATUS_SCHEMA_VERSION = "refsite_training_run_status_v1"


class RunDirectoryError(RuntimeError):
    """Structured filesystem failure for one fresh training run."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        path: str | os.PathLike[str],
        original_error: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.stage = stage
        self.path = str(path)
        self.original_error = original_error
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        super().__init__(
            f"[{reason_code}] stage={stage!r} path={self.path!r} {message}"
        )


def _plain_json(value: Any, *, path: str = "value") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} JSON object keys must be strings")
            result[key] = _plain_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain_json(item, path=f"{path}[]") for item in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or Infinity")
        return value
    if value is None or type(value) in (str, bool, int):
        return value
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def canonical_runtime_json(value: Mapping[str, Any]) -> str:
    """Return deterministic strict JSON suitable for recovery metadata."""

    plain = _plain_json(value)
    return json.dumps(
        plain,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_write_text(
    target: Path,
    encoded: str,
    *,
    overwrite: bool,
    stage: str,
) -> None:
    if target.is_symlink():
        raise RunDirectoryError(
            "RUNTIME_FILE_SYMLINK_REJECTED",
            "runtime metadata target must not be a symbolic link",
            stage=stage,
            path=target,
        )
    if target.exists():
        if not target.is_file():
            raise RunDirectoryError(
                "INVALID_RUNTIME_FILE_TARGET",
                "runtime metadata target must be a regular file",
                stage=stage,
                path=target,
            )
        if not overwrite:
            raise RunDirectoryError(
                "RUNTIME_FILE_ALREADY_EXISTS",
                "immutable runtime metadata target already exists",
                stage=stage,
                path=target,
            )
    descriptor = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if target.is_symlink():
            raise RunDirectoryError(
                "RUNTIME_FILE_SYMLINK_REJECTED",
                "runtime metadata target became a symbolic link before commit",
                stage=stage,
                path=target,
            )
        if target.exists() and not overwrite:
            raise RunDirectoryError(
                "RUNTIME_FILE_ALREADY_EXISTS",
                "immutable runtime metadata target appeared before commit",
                stage=stage,
                path=target,
            )
        os.replace(temporary, target)
        temporary = None
    except RunDirectoryError:
        raise
    except OSError as error:
        raise RunDirectoryError(
            "ATOMIC_RUNTIME_WRITE_FAILED",
            "same-directory atomic runtime metadata write failed",
            stage=stage,
            path=target,
            original_error=error,
        ) from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


@dataclass(frozen=True)
class TrainingRunDirectory:
    """One exclusively owned fresh-run directory."""

    root: Path

    def __post_init__(self) -> None:
        unresolved = Path(self.root)
        if unresolved.is_symlink() or not unresolved.is_dir():
            raise RunDirectoryError(
                "INVALID_OUTPUT_DIRECTORY",
                "training run directory must be an owned regular directory",
                stage="run_directory.validate",
                path=unresolved,
            )
        root = unresolved.resolve(strict=True)
        object.__setattr__(self, "root", root)

    @classmethod
    def create(cls, path: str | os.PathLike[str]) -> "TrainingRunDirectory":
        target = Path(path)
        if target.is_symlink():
            raise RunDirectoryError(
                "OUTPUT_SYMLINK_REJECTED",
                "output directory must not be a symbolic link",
                stage="run_directory.create",
                path=target,
            )
        try:
            target.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise RunDirectoryError(
                "OUTPUT_ALREADY_EXISTS",
                "output directory already exists; fresh training refuses it",
                stage="run_directory.create",
                path=target,
                original_error=error,
            ) from error
        except OSError as error:
            raise RunDirectoryError(
                "OUTPUT_DIRECTORY_CREATE_FAILED",
                "output directory could not be created exclusively",
                stage="run_directory.create",
                path=target,
                original_error=error,
            ) from error
        return cls(target)

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def resolved_config_path(self) -> Path:
        return self.root / "resolved_config.json"

    @property
    def preflight_path(self) -> Path:
        return self.root / "preflight.json"

    @property
    def status_path(self) -> Path:
        return self.root / "run_status.json"

    def write_resolved_config(self, value: Mapping[str, Any]) -> None:
        _atomic_write_text(
            self.resolved_config_path,
            canonical_runtime_json(value),
            overwrite=False,
            stage="run_directory.resolved_config",
        )

    def write_preflight(self, value: Mapping[str, Any]) -> None:
        _atomic_write_text(
            self.preflight_path,
            canonical_runtime_json(value),
            overwrite=False,
            stage="run_directory.preflight",
        )

    def write_status(self, value: Mapping[str, Any]) -> None:
        _atomic_write_text(
            self.status_path,
            canonical_runtime_json(value),
            overwrite=True,
            stage="run_directory.status",
        )


__all__ = [
    "RUN_STATUS_SCHEMA_VERSION",
    "RunDirectoryError",
    "TrainingRunDirectory",
    "canonical_runtime_json",
]
