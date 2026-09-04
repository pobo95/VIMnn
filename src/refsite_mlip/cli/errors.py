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
        path: str | PathLike[str] | None = None,
        source_path: str | PathLike[str] | None = None,
        frame_index: int | None = None,
        sample_id: str | None = None,
        template_id: str | None = None,
        term: str | None = None,
        config_field: str | None = None,
        split: str | None = None,
        epoch_index: int | None = None,
        batch_index: int | None = None,
        global_step: int | None = None,
        failure_phase: str | None = None,
        rollback_performed: bool | None = None,
        source_kind: str | None = None,
        run_directory: str | PathLike[str] | None = None,
        checkpoint_stage: str | None = None,
        bundle_fingerprint: str | None = None,
        config_fingerprint: str | None = None,
        template_fingerprint: str | None = None,
        solver_path: str | None = None,
        prediction_stage: str | None = None,
        predictor_reason_code: str | None = None,
        underlying_reason_code: str | None = None,
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
        self.path = (
            str(path)
            if path is not None
            else self.bundle_path
        )
        self.source_path = None if source_path is None else str(source_path)
        self.frame_index = frame_index
        self.sample_id = sample_id
        self.template_id = template_id
        self.term = term
        self.config_field = config_field
        self.split = split
        self.epoch_index = epoch_index
        self.batch_index = batch_index
        self.global_step = global_step
        self.failure_phase = failure_phase
        self.rollback_performed = rollback_performed
        self.source_kind = source_kind
        self.run_directory = (
            None if run_directory is None else str(run_directory)
        )
        self.checkpoint_stage = checkpoint_stage
        self.bundle_fingerprint = bundle_fingerprint
        self.config_fingerprint = config_fingerprint
        self.template_fingerprint = template_fingerprint
        self.solver_path = solver_path
        self.prediction_stage = prediction_stage
        self.predictor_reason_code = predictor_reason_code
        self.underlying_reason_code = underlying_reason_code
        self.message = message
        self.original_error = original_error
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        context = []
        for name in (
            "path",
            "source_path",
            "frame_index",
            "sample_id",
            "template_id",
            "term",
            "config_field",
            "split",
            "epoch_index",
            "batch_index",
            "global_step",
            "failure_phase",
            "rollback_performed",
            "source_kind",
            "run_directory",
            "checkpoint_stage",
            "bundle_fingerprint",
            "config_fingerprint",
            "template_fingerprint",
            "solver_path",
            "prediction_stage",
            "predictor_reason_code",
            "underlying_reason_code",
        ):
            value = getattr(self, name)
            if value is not None:
                context.append(f"{name}={value!r}")
        context_text = "" if not context else " " + " ".join(context)
        super().__init__(
            f"[{self.reason_code}]{context_text} stage={self.stage!r} {message}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return plain structured context without serializing a traceback."""

        return {
            "bundle_path": self.bundle_path,
            "config_field": self.config_field,
            "frame_index": self.frame_index,
            "epoch_index": self.epoch_index,
            "batch_index": self.batch_index,
            "global_step": self.global_step,
            "failure_phase": self.failure_phase,
            "message": self.message,
            "original_exception_type": self.original_exception_type,
            "path": self.path,
            "source_path": self.source_path,
            "prediction_stage": self.prediction_stage,
            "predictor_reason_code": self.predictor_reason_code,
            "reason_code": self.reason_code,
            "rollback_performed": self.rollback_performed,
            "source_kind": self.source_kind,
            "run_directory": self.run_directory,
            "checkpoint_stage": self.checkpoint_stage,
            "bundle_fingerprint": self.bundle_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "template_fingerprint": self.template_fingerprint,
            "sample_id": self.sample_id,
            "solver_path": self.solver_path,
            "split": self.split,
            "stage": self.stage,
            "template_id": self.template_id,
            "term": self.term,
            "underlying_reason_code": self.underlying_reason_code,
        }


class CLIInterruptedError(CLIError):
    """A handled KeyboardInterrupt whose process contract is exit status 130."""


def format_cli_error(error: CLIError) -> str:
    """Format one escaped, single-line diagnostic for stderr."""

    context = []
    for name in (
        "path",
        "source_path",
        "frame_index",
        "sample_id",
        "template_id",
        "term",
        "config_field",
        "split",
        "epoch_index",
        "batch_index",
        "global_step",
        "failure_phase",
        "rollback_performed",
        "source_kind",
        "run_directory",
        "checkpoint_stage",
        "bundle_fingerprint",
        "config_fingerprint",
        "template_fingerprint",
        "solver_path",
        "prediction_stage",
        "predictor_reason_code",
        "underlying_reason_code",
    ):
        value = getattr(error, name)
        if value is not None:
            context.append(f"{name}={value!r}")
    context_text = "" if not context else " " + " ".join(context)
    return (
        "refsite-mlip: error:"
        f"{context_text} stage={error.stage!r} reason={error.reason_code!r}: "
        f"{error.message}"
    )


__all__ = ["CLIError", "CLIInterruptedError", "format_cli_error"]
