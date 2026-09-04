"""Validate CUDA-checkpoint restore rejection with accelerators hidden."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import refsite_mlip
from refsite_mlip.training import (
    CheckpointCompatibilityError,
    ResumePolicy,
    load_training_checkpoint,
)
from refsite_mlip.training.resume import _validate_cuda_restore_preconditions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _critical_files(run: Path) -> dict[str, str]:
    names = (
        "run_status.json",
        "metrics.jsonl",
        "checkpoints/epoch_000000.pt",
        "checkpoints/epoch_000001.pt",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
    )
    return {name: _sha256(run / name) for name in names}


def run(run_directory: Path) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise AssertionError("hidden-device probe unexpectedly sees CUDA")
    before = _critical_files(run_directory)
    safe_globals = tuple(torch.serialization.get_safe_globals())
    original_load = torch.load
    load_contract: list[bool] = []

    def audited_load(*args: Any, **kwargs: Any) -> Any:
        load_contract.append(kwargs.get("weights_only") is True)
        return original_load(*args, **kwargs)

    torch.load = audited_load
    try:
        checkpoint = load_training_checkpoint(
            run_directory / "checkpoints" / "latest.pt"
        )
        if checkpoint.cuda_device_count != 1 or len(checkpoint.cuda_rng_states) != 1:
            raise AssertionError("source checkpoint lacks one-device CUDA RNG state")
        try:
            _validate_cuda_restore_preconditions(checkpoint, ResumePolicy())
        except CheckpointCompatibilityError as error:
            message = str(error)
        else:
            raise AssertionError("CUDA-hidden restore compatibility unexpectedly passed")
    finally:
        torch.load = original_load
    if "checkpoint contains CUDA RNG state but CUDA is unavailable" not in message:
        raise AssertionError(f"unexpected compatibility message: {message}")
    if before != _critical_files(run_directory):
        raise AssertionError("hidden-device restore validation changed run artifacts")
    if tuple(torch.serialization.get_safe_globals()) != safe_globals:
        raise AssertionError("hidden-device probe changed safe globals")
    if not load_contract or not all(load_contract):
        raise AssertionError(f"unsafe torch.load call observed: {load_contract}")
    installed_module = Path(refsite_mlip.__file__).resolve()
    if "site-packages" not in installed_module.parts:
        raise AssertionError("hidden-device probe imported outside site-packages")
    return {
        "schema_version": "refsite_cuda_hidden_restore_probe_v1",
        "status": "rejected",
        "reason": message,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "artifacts_unchanged": True,
        "safe_global_unchanged": True,
        "weights_only_calls": all(load_contract),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            run(arguments.run_directory),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
