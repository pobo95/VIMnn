"""Canonical on-disk run metadata shared with resume and bundle export."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from refsite_mlip.models import ReferenceSiteModelBundle

    from .run_directory import TrainingRunDirectory
    from .scratch_initialization import ScratchModelInitialization
    from .scratch_preparation import ScratchTrainingPreparation


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("scratch run metadata keys must be strings")
        return {key: _plain(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("scratch run metadata must not contain NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(
        "scratch run metadata contains non-plain "
        f"{type(value).__name__}"
    )


def scratch_runtime_template_fingerprints(
    preparation: "ScratchTrainingPreparation",
    bundle: "ReferenceSiteModelBundle | None" = None,
) -> dict[str, dict[str, Any]]:
    """Return resume/export fingerprints for the persisted runtime binding.

    Scratch preparation owns the pre-capture template context.  Capturing the
    portable initial bundle adds canonical binding provenance, so its binding
    fingerprint is intentionally different.  Once a bundle exists, on-disk
    run metadata must describe that portable binding because it is the source
    later consumed by resume and export.
    """

    required = (
        "structural_artifact_fingerprint",
        "full_template_fingerprint",
        "phase_specification_fingerprint",
        "binding_fingerprint",
        "evaluation_policy_fingerprint",
    )
    if bundle is None:
        sources = preparation.template_fingerprints
    else:
        bundle.validate()
        sources = {}
        for binding in bundle.template_bindings:
            phase_payload = {
                "scope": "reference_site_phase_specification_inspection_v1",
                "value": binding.phase_specification.to_dict(),
            }
            phase_json = json.dumps(
                phase_payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            sources[binding.template_id] = {
                "binding_fingerprint": binding.binding_fingerprint,
                "evaluation_policy_fingerprint": (
                    None
                    if binding.evaluation_policy is None
                    else binding.evaluation_policy.content_fingerprint
                ),
                "full_template_fingerprint": (
                    binding.full_template_fingerprint
                ),
                "phase_specification_fingerprint": hashlib.sha256(
                    phase_json.encode("utf-8")
                ).hexdigest(),
                "structural_artifact_fingerprint": (
                    binding.structural_artifact.structural_fingerprint
                ),
            }
        prepared_ids = set(preparation.template_fingerprints)
        bundle_ids = set(sources)
        if bundle_ids != prepared_ids:
            raise ValueError(
                "portable bundle template IDs differ from scratch preparation: "
                f"bundle={sorted(bundle_ids)!r}, "
                f"preparation={sorted(prepared_ids)!r}"
            )

    result: dict[str, dict[str, Any]] = {}
    for template_id in sorted(sources):
        source = sources[template_id]
        missing = tuple(key for key in required if key not in source)
        if missing:
            raise ValueError(
                f"template {template_id!r} metadata is missing {missing!r}"
            )
        result[template_id] = {
            key: _plain(source[key]) for key in required
        }
    return result


def scratch_runtime_preflight_metadata(
    preparation: "ScratchTrainingPreparation",
    initialization: "ScratchModelInitialization",
    directory: "TrainingRunDirectory",
) -> dict[str, Any]:
    """Adapt full scratch preparation to the stable run-directory contract.

    POSCAR paths and builder diagnostics remain in ``data_manifest.json`` and
    the canonical v2 config.  Runtime resume/export consumes only the generated
    portable initial bundle and never rebuilds a scratch reference artifact.
    """

    config = preparation.config
    prepared = preparation.to_dict()
    config_runtime_path = preparation.runtime_paths.get("config")
    if config_runtime_path is None:
        config_runtime_path = str(directory.resolved_config_path)
    runtime_paths = {
        "config": str(config_runtime_path),
        "initial_bundle": str(directory.initial_bundle_path),
        "output_directory": str(directory.root),
        "train_inputs": list(preparation.runtime_paths["train_inputs"]),
        "validation_inputs": list(
            preparation.runtime_paths["validation_inputs"]
        ),
        "path_kind": "runtime_location_not_semantic_fingerprint",
    }
    configured_paths = {
        "initial_bundle": config.initial_bundle,
        "output_directory": config.output_directory,
        "train_inputs": [source.path for source in config.data.train],
        "validation_inputs": [
            source.path for source in config.data.validation
        ],
        "path_kind": "original_config_expression_in_semantic_fingerprint",
    }
    expected_paths = {
        "output_directory": str(directory.root),
        "resolved_config": str(directory.resolved_config_path),
        "preflight": str(directory.preflight_path),
        "run_status": str(directory.status_path),
        "latest_checkpoint": str(directory.checkpoints / "latest.pt"),
        "best_checkpoint": str(directory.checkpoints / "best.pt"),
        "epoch_checkpoint_pattern": str(
            directory.checkpoints / "epoch_XXXXXX.pt"
        ),
    }
    radii = preparation.radius_config
    return _plain(
        {
            "status": "preflight_ready",
            "training_executed": False,
            "schema_version": config.schema_version,
            "config_fingerprint": preparation.config_fingerprint,
            "bundle_fingerprint": initialization.bundle_fingerprint,
            "data": prepared["data"],
            "runtime": {
                "device": preparation.resolved_device,
                "dtype": preparation.resolved_dtype,
                "seed": preparation.runtime.seed,
                "configured_paths": configured_paths,
                "paths": runtime_paths,
            },
            "radii": {
                "user": {"r_ot": radii.r_ot, "r_mp": radii.r_mp},
                "advanced": {
                    "ot_switch_width": radii.ot_switch_width,
                    "ot_skin": radii.ot_skin,
                    "mp_skin": radii.mp_skin,
                },
                "derived": radii.derived.to_dict(),
                "diagnostics": radii.derived.to_diagnostics_dict(),
            },
            "species_vocabulary": list(preparation.species_vocabulary),
            "template_fingerprints": scratch_runtime_template_fingerprints(
                preparation,
                initialization.bundle,
            ),
            "baseline_preflight": _plain(preparation.baseline_preflight),
            "expected_paths": expected_paths,
            "training_configuration": _plain(
                preparation.training_configuration
            ),
        }
    )


__all__ = [
    "scratch_runtime_preflight_metadata",
    "scratch_runtime_template_fingerprints",
]
