"""Structured errors shared by the command-line interface."""

from __future__ import annotations

from os import PathLike
from typing import Any


class CLIError(RuntimeError):
    """A concise user-facing error with stable machine-readable context."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        bundle_path: str | PathLike[str] | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("reason_code must be a nonempty string")
        if not isinstance(message, str) or not message:
            raise ValueError("message must be a nonempty string")
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a nonempty string")
        self.reason_code = reason_code
        self.stage = stage
        self.bundle_path = None if bundle_path is None else str(bundle_path)
        self.message = message
        self.original_error = original_error
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        path_context = (
            "" if self.bundle_path is None else f" path={self.bundle_path!r}"
        )
        super().__init__(
            f"[{self.reason_code}]{path_context} stage={self.stage!r} {message}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return plain structured context without serializing a traceback."""

        return {
            "bundle_path": self.bundle_path,
            "message": self.message,
            "original_exception_type": self.original_exception_type,
            "reason_code": self.reason_code,
            "stage": self.stage,
        }


def format_cli_error(error: CLIError) -> str:
    """Format one escaped, single-line diagnostic for stderr."""

    path = "" if error.bundle_path is None else f" path={error.bundle_path!r}"
    return (
        "refsite-mlip: error:"
        f"{path} stage={error.stage!r} reason={error.reason_code!r}: "
        f"{error.message}"
    )


__all__ = ["CLIError", "format_cli_error"]
