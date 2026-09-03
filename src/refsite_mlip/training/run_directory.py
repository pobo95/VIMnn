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
RESUME_LOCK_FILENAME = ".resume.lock"


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


def load_runtime_json(
    target: str | os.PathLike[str],
    *,
    stage: str,
) -> dict[str, Any]:
    """Safely read one strict, plain JSON object without following a symlink."""

    path = Path(target)
    if path.is_symlink():
        raise RunDirectoryError(
            "RUNTIME_FILE_SYMLINK_REJECTED",
            "runtime metadata source must not be a symbolic link",
            stage=stage,
            path=path,
        )
    if not path.exists():
        raise RunDirectoryError(
            "RUNTIME_FILE_NOT_FOUND",
            "required runtime metadata file does not exist",
            stage=stage,
            path=path,
        )
    if not path.is_file():
        raise RunDirectoryError(
            "INVALID_RUNTIME_FILE_SOURCE",
            "runtime metadata source must be a regular file",
            stage=stage,
            path=path,
        )

    def reject_constant(value: str):
        raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")

    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        encoded = path.read_text(encoding="utf-8")
        value = json.loads(
            encoded,
            parse_constant=reject_constant,
            object_pairs_hook=strict_object,
        )
        if not isinstance(value, Mapping):
            raise TypeError("runtime metadata root must be a JSON object")
        return _plain_json(value, path="runtime_metadata")
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunDirectoryError(
            "RUNTIME_JSON_LOAD_FAILED",
            "runtime metadata could not be loaded as strict JSON",
            stage=stage,
            path=path,
            original_error=error,
        ) from error


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

    @classmethod
    def open_existing(
        cls, path: str | os.PathLike[str]
    ) -> "TrainingRunDirectory":
        target = Path(path)
        if target.is_symlink():
            raise RunDirectoryError(
                "RUN_DIRECTORY_SYMLINK_REJECTED",
                "resume run directory must not be a symbolic link",
                stage="run_directory.open",
                path=target,
            )
        if not target.exists():
            raise RunDirectoryError(
                "RUN_DIRECTORY_NOT_FOUND",
                "resume run directory does not exist",
                stage="run_directory.open",
                path=target,
            )
        if not target.is_dir():
            raise RunDirectoryError(
                "INVALID_RUN_DIRECTORY",
                "resume source must be a training run directory",
                stage="run_directory.open",
                path=target,
            )
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

    @property
    def resume_lock_path(self) -> Path:
        return self.root / RESUME_LOCK_FILENAME

    def validate_resume_lock_available(self) -> None:
        path = self.resume_lock_path
        if path.is_symlink():
            raise RunDirectoryError(
                "RESUME_LOCK_SYMLINK_REJECTED",
                "resume lock path must not be a symbolic link",
                stage="run_directory.resume_lock",
                path=path,
            )
        if path.exists():
            raise RunDirectoryError(
                "RESUME_LOCK_EXISTS",
                "another or stale resume lock already exists; it is not removed automatically",
                stage="run_directory.resume_lock",
                path=path,
            )

    def acquire_resume_lock(self) -> "ResumeRunLock":
        return ResumeRunLock.acquire(self)


class ResumeRunLock:
    """Exclusively-created lock removed only while inode ownership is retained."""

    def __init__(
        self,
        path: Path,
        *,
        device: int,
        inode: int,
    ) -> None:
        self.path = path
        self._device = int(device)
        self._inode = int(inode)
        self._owned = True

    @classmethod
    def acquire(cls, directory: TrainingRunDirectory) -> "ResumeRunLock":
        if not isinstance(directory, TrainingRunDirectory):
            raise TypeError("directory must be a TrainingRunDirectory")
        directory.validate_resume_lock_available()
        path = directory.resume_lock_path
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = None
        identity: tuple[int, int] | None = None
        try:
            descriptor = os.open(path, flags, 0o600)
            stat = os.fstat(descriptor)
            identity = (stat.st_dev, stat.st_ino)
            encoded = canonical_runtime_json(
                {
                    "schema_version": "refsite_resume_lock_v1",
                    "pid": os.getpid(),
                }
            ).encode("utf-8") + b"\n"
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("resume lock write made no progress")
                offset += written
            os.fsync(descriptor)
        except FileExistsError as error:
            reason = (
                "RESUME_LOCK_SYMLINK_REJECTED"
                if path.is_symlink()
                else "RESUME_LOCK_EXISTS"
            )
            raise RunDirectoryError(
                reason,
                "resume lock could not be acquired exclusively",
                stage="run_directory.resume_lock.acquire",
                path=path,
                original_error=error,
            ) from error
        except OSError as error:
            if identity is not None:
                try:
                    current = path.lstat()
                    if (current.st_dev, current.st_ino) == identity:
                        path.unlink()
                except OSError:
                    pass
            raise RunDirectoryError(
                "RESUME_LOCK_ACQUIRE_FAILED",
                "resume lock could not be created atomically",
                stage="run_directory.resume_lock.acquire",
                path=path,
                original_error=error,
            ) from error
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        assert identity is not None
        return cls(path, device=identity[0], inode=identity[1])

    @property
    def owned(self) -> bool:
        return self._owned

    def release(self) -> None:
        if not self._owned:
            return
        try:
            stat = self.path.lstat()
        except FileNotFoundError as error:
            self._owned = False
            raise RunDirectoryError(
                "RESUME_LOCK_OWNERSHIP_LOST",
                "owned resume lock disappeared before release",
                stage="run_directory.resume_lock.release",
                path=self.path,
                original_error=error,
            ) from error
        if self.path.is_symlink() or (stat.st_dev, stat.st_ino) != (
            self._device,
            self._inode,
        ):
            self._owned = False
            raise RunDirectoryError(
                "RESUME_LOCK_OWNERSHIP_LOST",
                "resume lock was replaced; the foreign lock was not removed",
                stage="run_directory.resume_lock.release",
                path=self.path,
            )
        try:
            self.path.unlink()
        except OSError as error:
            raise RunDirectoryError(
                "RESUME_LOCK_RELEASE_FAILED",
                "owned resume lock could not be removed",
                stage="run_directory.resume_lock.release",
                path=self.path,
                original_error=error,
            ) from error
        self._owned = False

    def __enter__(self) -> "ResumeRunLock":
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        try:
            self.release()
        except Exception:
            if exception is None:
                raise
        return False


__all__ = [
    "RESUME_LOCK_FILENAME",
    "RUN_STATUS_SCHEMA_VERSION",
    "ResumeRunLock",
    "RunDirectoryError",
    "TrainingRunDirectory",
    "canonical_runtime_json",
    "load_runtime_json",
]
