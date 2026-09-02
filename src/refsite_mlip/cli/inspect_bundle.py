"""Safe, inference-free inspection of portable model bundles."""

from __future__ import annotations

import hashlib
import json
import math
from os import PathLike
from typing import Any, Mapping

import torch

from refsite_mlip.models import ModelBundleError, load_reference_site_model_bundle

from .errors import CLIError


def _plain(value: Any, *, field: str) -> Any:
    """Copy supported metadata into JSON-native values and reject surprises."""

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} contains NaN or Infinity")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{field} contains a non-string mapping key")
        return {
            key: _plain(value[key], field=f"{field}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _plain(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field} contains non-JSON metadata {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _phase_specification_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash the canonical public phase specification for inspection output."""

    payload = {
        "scope": "reference_site_phase_specification_inspection_v1",
        "value": _plain(value, field="phase_specification"),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _state_summary(bundle: Any) -> dict[str, Any]:
    tensors = tuple(bundle.model_state.values())
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        # The safe loader already enforces this. Keep the inspection boundary
        # explicit in case a caller supplies a future incompatible loader.
        raise TypeError("validated model state contains a non-tensor value")
    return {
        "element_count": sum(int(value.numel()) for value in tensors),
        "floating_dtype": bundle.model_floating_dtype,
        "includes": ["parameters", "buffers"],
        "tensor_count": len(tensors),
        "total_bytes": sum(
            int(value.numel()) * int(value.element_size()) for value in tensors
        ),
    }


def _convention_summary(bundle: Any) -> dict[str, Any]:
    conventions = bundle.conventions
    alignment = conventions["species_alignment_weights"]
    if not isinstance(alignment, torch.Tensor):
        raise TypeError("validated species alignment convention is not a tensor")
    public = {
        key: value
        for key, value in conventions.items()
        if key != "species_alignment_weights"
    }
    public["species_alignment_summary"] = {
        "dtype": str(alignment.dtype).removeprefix("torch."),
        "element_count": int(alignment.numel()),
        "shape": list(alignment.shape),
    }
    return _plain(public, field="conventions")


def _template_summary(binding: Any) -> dict[str, Any]:
    phase = binding.phase_specification.to_dict()
    policy = binding.evaluation_policy
    return {
        "approval_status": binding.approval_status,
        "evaluation_policy_fingerprint": (
            None if policy is None else policy.content_fingerprint
        ),
        "evaluation_policy_present": policy is not None,
        "full_template_fingerprint": binding.full_template_fingerprint,
        "phase_convention_version": binding.phase_specification.convention_version,
        "phase_specification_fingerprint": _phase_specification_fingerprint(phase),
        "provenance": _plain(
            binding.provenance or {}, field=f"templates.{binding.template_id}.provenance"
        ),
        "structural_artifact_fingerprint": (
            binding.structural_artifact.structural_fingerprint
        ),
    }


def summarize_bundle(bundle: Any) -> dict[str, Any]:
    """Build the stable public metadata view of one validated bundle."""

    template_ids = sorted(binding.template_id for binding in bundle.template_bindings)
    bindings = {binding.template_id: binding for binding in bundle.template_bindings}
    report = {
        "architecture_fingerprint": bundle.architecture_fingerprint,
        # bundle_fingerprint is the validated, mapping-order-independent SHA-256
        # of semantic bundle content. Archive bytes and paths are intentionally
        # not part of this public report.
        "bundle_sha256": bundle.bundle_fingerprint,
        "bundle_scope": bundle.bundle_scope,
        "conventions": _convention_summary(bundle),
        "default_template_id": bundle.default_template_id,
        "model": {
            "config": _plain(bundle.model_config, field="model.config"),
            "state": _state_summary(bundle),
        },
        "provenance": _plain(bundle.provenance, field="provenance"),
        "schema_version": bundle.schema_version,
        "species_vocabulary": list(bundle.species_vocabulary),
        "template_ids": template_ids,
        "templates": {
            template_id: _template_summary(bindings[template_id])
            for template_id in template_ids
        },
        "version_metadata": _plain(
            bundle.version_metadata, field="version_metadata"
        ),
    }
    return _plain(report, field="inspection")


def inspect_bundle(path: str | PathLike[str]) -> dict[str, Any]:
    """Safely load, validate, and summarize a portable model bundle."""

    display_path = str(path)
    try:
        bundle = load_reference_site_model_bundle(path, map_location="cpu")
    except ModelBundleError as error:
        raise CLIError(
            error.reason_code,
            "safe bundle load or validation failed",
            stage=error.validation_stage or "load.validation",
            bundle_path=error.bundle_path or display_path,
            original_error=error,
        ) from error
    except FileNotFoundError as error:
        raise CLIError(
            "BUNDLE_NOT_FOUND",
            "bundle does not exist",
            stage="load.path",
            bundle_path=display_path,
            original_error=error,
        ) from error
    except (IsADirectoryError, NotADirectoryError, ValueError) as error:
        raise CLIError(
            "INVALID_BUNDLE_PATH",
            "bundle path is not a regular readable file",
            stage="load.path",
            bundle_path=display_path,
            original_error=error,
        ) from error
    except OSError as error:
        raise CLIError(
            "BUNDLE_IO_ERROR",
            "bundle could not be read",
            stage="load.io",
            bundle_path=display_path,
            original_error=error,
        ) from error

    try:
        return summarize_bundle(bundle)
    except Exception as error:
        raise CLIError(
            "INSPECTION_METADATA_ERROR",
            "validated bundle metadata could not be summarized",
            stage="inspect.metadata",
            bundle_path=display_path,
            original_error=error,
        ) from error


def render_json(report: Mapping[str, Any]) -> str:
    """Render deterministic compact JSON with strict finite-number handling."""

    return _canonical_json(_plain(report, field="report"))


def _display(value: Any) -> str:
    if type(value) is bool:
        return "yes" if value else "no"
    if value is None:
        return "none"
    if type(value) is str:
        return value
    return _canonical_json(value)


def render_human(report: Mapping[str, Any]) -> str:
    """Render a deterministic human-readable metadata report."""

    data = _plain(report, field="report")
    model = data["model"]
    state = model["state"]
    conventions = data["conventions"]
    template_ids = sorted(data["template_ids"])
    lines = [
        "Reference-site MLIP portable bundle",
        f"Schema: {data['schema_version']}",
        f"Scope: {data['bundle_scope']}",
        f"Bundle SHA-256: {data['bundle_sha256']}",
        f"Architecture fingerprint: {data['architecture_fingerprint']}",
        f"Default template ID: {data['default_template_id']}",
        f"Included template IDs: {', '.join(template_ids)}",
        "",
        "Model",
        f"  Floating dtype: {state['floating_dtype']}",
        f"  Species vocabulary: {_display(data['species_vocabulary'])}",
        "  Parameter/buffer state: "
        f"{state['tensor_count']} tensors, {state['element_count']} elements, "
        f"{state['total_bytes']} bytes",
        f"  Public config: {_display(model['config'])}",
        "",
        "Conventions",
        f"  Convention version: {conventions['convention_version']}",
        "  Site-type vocabulary: "
        f"{_display(conventions['ordered_site_type_vocabulary'])}",
        f"  Phase channels: {conventions['phase_channel_count']}",
        "  Species alignment: "
        f"{_display(conventions['species_alignment_summary'])}",
        f"  Length unit: {conventions['length_unit']}",
        f"  Energy unit: {conventions['energy_unit']}",
        f"  Force unit: {conventions['force_unit']}",
        f"  Stress unit: {conventions['stress_unit']}",
        f"  Stress sign: {conventions['stress_sign']} (no sign reversal)",
        f"  Stress Voigt order: {_display(conventions['stress_voigt_order'])}",
        f"  Cell convention: {conventions['cell_convention']}",
        f"  PBC convention: {conventions['pbc_convention']}",
        "",
        f"Templates ({len(template_ids)})",
    ]
    for template_id in template_ids:
        template = data["templates"][template_id]
        lines.extend(
            [
                f"  {template_id}",
                "    Structural artifact fingerprint: "
                f"{template['structural_artifact_fingerprint']}",
                "    Full template fingerprint: "
                f"{template['full_template_fingerprint']}",
                "    Phase specification fingerprint: "
                f"{template['phase_specification_fingerprint']}",
                "    Evaluation policy present: "
                f"{_display(template['evaluation_policy_present'])}",
                "    Evaluation policy fingerprint: "
                f"{_display(template['evaluation_policy_fingerprint'])}",
                f"    Approval status: {template['approval_status']}",
                f"    Provenance: {_display(template['provenance'])}",
            ]
        )
    lines.extend(["", "Version metadata"])
    for key in sorted(data["version_metadata"]):
        lines.append(f"  {key}: {_display(data['version_metadata'][key])}")
    lines.extend(["", f"Provenance: {_display(data['provenance'])}"])
    return "\n".join(lines)


__all__ = [
    "inspect_bundle",
    "render_human",
    "render_json",
    "summarize_bundle",
]
