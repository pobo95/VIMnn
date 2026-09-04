"""Portable, weights-only-safe reference-site potential bundles.

The bundle is deliberately separate from training checkpoints.  It owns a
CPU model snapshot and embeds verified structural artifacts, while phase
specifications remain explicit bindings assembled without invoking builders.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import torch

from refsite_mlip._atomic import commit_temporary_file

from refsite_mlip import __version__ as _PACKAGE_VERSION
from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.data.reference_artifact import (
    REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION,
    ReferenceStructureArtifact,
    ReferenceStructureArtifactError,
    _artifact_from_safe_payload,
    assemble_reference_template_from_artifact,
)
from refsite_mlip.data.reference_builder import PhaseSpecification
from refsite_mlip.data.schema import (
    ENERGY_UNIT,
    FORCE_UNIT,
    LENGTH_UNIT,
    STRESS_SIGN,
    STRESS_UNIT,
    STRESS_VOIGT_ORDER,
)
from refsite_mlip.data.templates import ReferenceTemplate, TemplateRegistry

from .config import PotentialConfig
from .evaluation_policy import EvaluationPolicy
from .potential import ReferenceSitePotential
from .template_context import TemplateExecutionContext


REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION = "reference_site_model_bundle_v1"
REFERENCE_SITE_MODEL_BUNDLE_SCOPE = "portable_reference_site_potential"
MODEL_BUNDLE_CONVENTION_VERSION = "reference_site_model_bundle_conventions_v1"
UNIT_CONVENTION_VERSION = "angstrom_ev_tensile_voigt_v1"
RUNTIME_COMPATIBILITY_POLICY = "major_minor_exact_patch_compatible_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_DTYPES = {"float32": torch.float32, "float64": torch.float64}
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "bundle_scope", "payload", "bundle_fingerprint"}
)
_PAYLOAD_KEYS = frozenset(
    {
        "model_config",
        "model_state",
        "model_state_keys",
        "model_floating_dtype",
        "species_vocabulary",
        "conventions",
        "default_template_id",
        "template_bindings",
        "version_metadata",
        "provenance",
        "architecture_fingerprint",
    }
)
_BINDING_KEYS = frozenset(
    {
        "template_id",
        "structural_artifact",
        "structural_fingerprint",
        "phase_specification",
        "full_template_fingerprint",
        "evaluation_policy",
        "approval_status",
        "provenance",
        "binding_fingerprint",
    }
)
_PHASE_KEYS = frozenset(
    {
        "modes",
        "mode_weights",
        "site_type_alignment_weights",
        "channel_weights",
        "approval_status",
        "convention_version",
        "floating_dtype",
    }
)
_POLICY_KEYS = frozenset(
    {
        "template_id",
        "template_fingerprint",
        "candidate_offsets",
        "candidate_dtype",
        "phase_step_schedule",
        "phase_damping_schedule",
        "minimum_objective_gap_absolute",
        "minimum_cross_amplitude_absolute",
        "minimum_atomic_amplitude_absolute",
        "minimum_reference_amplitude_absolute",
        "minimum_curvature",
        "maximum_condition",
        "maximum_gradient_norm",
        "equivalence_tolerance",
        "transport_path",
        "convention_version",
        "content_fingerprint",
    }
)
_CONVENTION_KEYS = frozenset(
    {
        "convention_version",
        "ordered_species_vocabulary",
        "ordered_site_type_vocabulary",
        "phase_channel_count",
        "species_alignment_weights",
        "length_unit",
        "energy_unit",
        "force_unit",
        "stress_unit",
        "stress_sign",
        "stress_voigt_order",
        "cell_convention",
        "pbc_convention",
        "atomic_baseline_convention",
        "unit_convention_version",
    }
)
_VERSION_KEYS = frozenset(
    {
        "compatibility_policy",
        "refsite_mlip_version",
        "torch_version",
        "e3nn_version",
        "python_version",
        "bundle_schema_version",
        "artifact_schema_versions",
        "template_convention_versions",
        "domain_convention_versions",
        "phase_convention_versions",
        "unit_convention_version",
    }
)


class ModelBundleError(ValueError):
    """Structured portable-bundle validation or reconstruction failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        bundle_path: str | None = None,
        schema: str | None = None,
        validation_stage: str | None = None,
        template_id: str | None = None,
        state_key: str | None = None,
        expected_fingerprint: str | None = None,
        actual_fingerprint: str | None = None,
        original_exception_type: str | None = None,
        original_exception_message: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.bundle_path = bundle_path
        self.schema = schema
        self.validation_stage = validation_stage
        self.template_id = template_id
        self.state_key = state_key
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint
        self.original_exception_type = original_exception_type
        self.original_exception_message = original_exception_message
        context = (
            f" path={bundle_path!r} schema={schema!r} stage={validation_stage!r}"
            f" template_id={template_id!r} state_key={state_key!r}"
            f" expected_fingerprint={expected_fingerprint!r}"
            f" actual_fingerprint={actual_fingerprint!r}"
            f" original_exception_type={original_exception_type!r}"
            f" original_exception_message={original_exception_message!r}"
        )
        super().__init__(f"[{reason_code}]{context} {message}")


def _error(reason_code: str, message: str, **context: Any) -> ModelBundleError:
    return ModelBundleError(reason_code, message, **context)


def _wrap_error(
    reason_code: str,
    message: str,
    error: BaseException,
    **context: Any,
) -> ModelBundleError:
    return _error(
        reason_code,
        message,
        original_exception_type=type(error).__name__,
        original_exception_message=str(error),
        **context,
    )


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("bundle tensor field must be a torch.Tensor")
    return value.detach().to(device="cpu").contiguous().clone()


def _canonical_plain(
    value: Any,
    *,
    path: str,
    allow_tensors: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        if not allow_tensors:
            raise TypeError(f"{path} must not contain tensors")
        return _cpu_clone(value)
    if value is None or type(value) in (str, bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite float")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path} mapping keys must be strings")
        return {
            key: _canonical_plain(
                value[key], path=f"{path}.{key}", allow_tensors=allow_tensors
            )
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical_plain(
                item, path=f"{path}[{index}]", allow_tensors=allow_tensors
            )
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains weights-only-unsafe type {type(value).__name__}")


def _deep_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_deep_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_deep_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _hash_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)


def _hash_safe(digest: Any, field: str, value: Any) -> None:
    _hash_text(digest, field)
    if isinstance(value, torch.Tensor):
        tensor = _cpu_clone(value)
        _hash_text(digest, "tensor")
        _hash_text(digest, str(tensor.dtype))
        _hash_text(digest, str(tuple(tensor.shape)))
        # ``view(dtype)`` rejects zero-dimensional tensors when element sizes
        # differ; flattening preserves the exact contiguous byte stream.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        _hash_text(digest, "mapping")
        for key in sorted(value):
            _hash_safe(digest, f"{field}.{key}", value[key])
        return
    if isinstance(value, (tuple, list)):
        _hash_text(digest, "sequence")
        _hash_text(digest, str(len(value)))
        for index, item in enumerate(value):
            _hash_safe(digest, f"{field}[{index}]", item)
        return
    if value is None:
        _hash_text(digest, "none")
        return
    if type(value) in (str, bool, int, float):
        _hash_text(digest, type(value).__name__)
        _hash_text(digest, repr(value))
        return
    raise TypeError(f"cannot fingerprint unsafe value at {field}")


def _fingerprint(scope: str, value: Any) -> str:
    digest = hashlib.sha256()
    _hash_safe(digest, scope, value)
    return digest.hexdigest()


def _require_exact_keys(
    value: Any,
    expected: frozenset[str],
    *,
    name: str,
    bundle_path: str | None,
    schema: str | None,
    stage: str,
    template_id: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(
            "INVALID_PAYLOAD",
            f"{name} must be a mapping",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage=stage,
            template_id=template_id,
        )
    actual = set(value)
    if actual != set(expected):
        raise _error(
            "INVALID_PAYLOAD_KEYS",
            f"{name} key mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage=stage,
            template_id=template_id,
        )
    if any(type(key) is not str for key in value):
        raise _error(
            "INVALID_PAYLOAD_KEYS",
            f"{name} keys must be strings",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage=stage,
            template_id=template_id,
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    for name, value in _FLOAT_DTYPES.items():
        if dtype == value:
            return name
    raise ValueError("model floating dtype must be float32 or float64")


def _version_triplet(value: str, *, name: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        raise ValueError(f"{name} version is not semantic: {value!r}")
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def _current_versions() -> tuple[str, str]:
    e3nn, _ = import_e3nn_0_4_4()
    return str(torch.__version__), str(e3nn.__version__)


def _validate_version_metadata(
    metadata: Mapping[str, Any],
    *,
    bundle_path: str | None,
    schema: str,
) -> None:
    _require_exact_keys(
        metadata,
        _VERSION_KEYS,
        name="version_metadata",
        bundle_path=bundle_path,
        schema=schema,
        stage="version",
    )
    if metadata["compatibility_policy"] != RUNTIME_COMPATIBILITY_POLICY:
        raise _error(
            "UNSUPPORTED_VERSION_POLICY",
            "unsupported runtime compatibility policy",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version",
        )
    if metadata["bundle_schema_version"] != REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION:
        raise _error(
            "UNSUPPORTED_SCHEMA",
            "version metadata names a different bundle schema",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version",
        )
    try:
        recorded_torch = _version_triplet(metadata["torch_version"], name="torch")
        recorded_e3nn = _version_triplet(metadata["e3nn_version"], name="e3nn")
        recorded_package = _version_triplet(
            metadata["refsite_mlip_version"], name="refsite_mlip"
        )
        current_torch_text, current_e3nn_text = _current_versions()
        current_torch = _version_triplet(current_torch_text, name="torch")
        current_e3nn = _version_triplet(current_e3nn_text, name="e3nn")
        current_package = _version_triplet(_PACKAGE_VERSION, name="refsite_mlip")
    except (TypeError, ValueError) as error:
        raise _wrap_error(
            "INVALID_VERSION_METADATA",
            "version metadata could not be parsed",
            error,
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version",
        ) from error
    if recorded_torch[:2] != (2, 6) or current_torch[:2] != (2, 6):
        raise _error(
            "UNSUPPORTED_RUNTIME_VERSION",
            "portable bundles require PyTorch 2.6.x",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version.torch",
        )
    if recorded_e3nn[:2] != (0, 4) or current_e3nn[:2] != (0, 4):
        raise _error(
            "UNSUPPORTED_RUNTIME_VERSION",
            "portable bundles require e3nn 0.4.x",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version.e3nn",
        )
    if recorded_torch[:2] != current_torch[:2] or recorded_e3nn[:2] != current_e3nn[:2]:
        raise _error(
            "RUNTIME_VERSION_MISMATCH",
            "bundle/runtime major-minor versions differ",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version",
        )
    if recorded_package[0] != current_package[0]:
        raise _error(
            "PACKAGE_VERSION_MISMATCH",
            "bundle/runtime package major versions differ",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version.refsite_mlip",
        )
    python_version = metadata["python_version"]
    if (
        not isinstance(python_version, list)
        or len(python_version) != 3
        or any(type(value) is not int for value in python_version)
    ):
        raise _error(
            "INVALID_VERSION_METADATA",
            "python_version must be a canonical [major,minor,micro] list",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version.python",
        )
    if python_version[0] != sys.version_info.major:
        raise _error(
            "UNSUPPORTED_RUNTIME_VERSION",
            "Python major version differs from the bundle",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version.python",
        )
    for key in (
        "artifact_schema_versions",
        "template_convention_versions",
        "domain_convention_versions",
        "phase_convention_versions",
    ):
        values = metadata[key]
        if not isinstance(values, list) or any(type(value) is not str for value in values):
            raise _error(
                "INVALID_VERSION_METADATA",
                f"{key} must be a canonical string list",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="version.conventions",
            )
    if metadata["artifact_schema_versions"] != [REFERENCE_STRUCTURE_ARTIFACT_SCHEMA_VERSION]:
        raise _error(
            "UNSUPPORTED_ARTIFACT_SCHEMA",
            "bundle references an unsupported structural artifact schema",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version.artifact",
        )
    if metadata["unit_convention_version"] != UNIT_CONVENTION_VERSION:
        raise _error(
            "UNIT_CONVENTION_MISMATCH",
            "unsupported bundle unit convention",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="version.units",
        )


def _binding_payload_without_fingerprint(
    *,
    template_id: str,
    artifact: ReferenceStructureArtifact,
    phase: PhaseSpecification,
    full_template_fingerprint: str,
    policy: EvaluationPolicy | None,
    approval_status: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "template_id": template_id,
        "structural_artifact": artifact.to_payload(),
        "structural_fingerprint": artifact.structural_fingerprint,
        "phase_specification": _canonical_plain(
            phase.to_dict(), path="phase_specification", allow_tensors=False
        ),
        "full_template_fingerprint": full_template_fingerprint,
        "evaluation_policy": (
            None
            if policy is None
            else _canonical_plain(
                policy.to_dict(), path="evaluation_policy", allow_tensors=False
            )
        ),
        "approval_status": approval_status,
        "provenance": _canonical_plain(
            provenance, path="binding.provenance", allow_tensors=False
        ),
    }


@dataclass(frozen=True)
class ModelBundleTemplateBinding:
    """One exact structure + explicit phase + optional evaluation-policy binding."""

    template_id: str
    structural_artifact: ReferenceStructureArtifact
    phase_specification: PhaseSpecification
    full_template_fingerprint: str
    evaluation_policy: EvaluationPolicy | None = None
    approval_status: str = "provisional"
    provenance: Mapping[str, Any] | None = None
    binding_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.structural_artifact, ReferenceStructureArtifact):
            raise TypeError("structural_artifact must be a ReferenceStructureArtifact")
        artifact = _artifact_from_safe_payload(
            self.structural_artifact.to_payload(), artifact_path=None
        )
        if not isinstance(self.phase_specification, PhaseSpecification):
            raise TypeError("phase_specification must be a PhaseSpecification")
        phase = PhaseSpecification.from_dict(self.phase_specification.to_dict())
        policy = self.evaluation_policy
        if policy is not None:
            if not isinstance(policy, EvaluationPolicy):
                raise TypeError("evaluation_policy must be an EvaluationPolicy or None")
            policy = EvaluationPolicy.from_dict(policy.to_dict())
        provenance = _canonical_plain(
            {} if self.provenance is None else self.provenance,
            path="binding.provenance",
            allow_tensors=False,
        )
        object.__setattr__(self, "structural_artifact", artifact)
        object.__setattr__(self, "phase_specification", phase)
        object.__setattr__(self, "evaluation_policy", policy)
        object.__setattr__(self, "provenance", provenance)
        payload = _binding_payload_without_fingerprint(
            template_id=self.template_id,
            artifact=artifact,
            phase=phase,
            full_template_fingerprint=self.full_template_fingerprint,
            policy=policy,
            approval_status=self.approval_status,
            provenance=provenance,
        )
        actual = _fingerprint("model_bundle_template_binding_v1", payload)
        if self.binding_fingerprint is None:
            object.__setattr__(self, "binding_fingerprint", actual)
        self.validate()

    def validate(self, *, bundle_path: str | None = None) -> ReferenceTemplate:
        schema = REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION
        if not isinstance(self.template_id, str) or not self.template_id:
            raise _error(
                "INVALID_TEMPLATE_ID",
                "binding template_id must be nonempty",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.identity",
            )
        if self.structural_artifact.template_id != self.template_id:
            raise _error(
                "ARTIFACT_TEMPLATE_MISMATCH",
                "structural artifact is bound to another template ID",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.artifact",
                template_id=self.template_id,
            )
        try:
            self.structural_artifact.validate(artifact_path=bundle_path)
            template = assemble_reference_template_from_artifact(
                self.structural_artifact,
                phase_specification=self.phase_specification,
            )
        except ReferenceStructureArtifactError as error:
            raise _wrap_error(
                "NESTED_ARTIFACT_INVALID",
                "nested structural artifact validation failed",
                error,
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.artifact",
                template_id=self.template_id,
            ) from error
        except Exception as error:
            raise _wrap_error(
                "PHASE_ASSEMBLY_FAILED",
                "explicit phase could not be assembled with the structural artifact",
                error,
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.phase",
                template_id=self.template_id,
            ) from error
        try:
            expected = _sha256(
                self.full_template_fingerprint, name="full_template_fingerprint"
            )
        except ValueError as error:
            raise _wrap_error(
                "INVALID_TEMPLATE_FINGERPRINT",
                str(error),
                error,
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.fingerprint",
                template_id=self.template_id,
            ) from error
        if template.fingerprint != expected:
            raise _error(
                "TEMPLATE_FINGERPRINT_MISMATCH",
                "assembled template fingerprint differs from its binding",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.fingerprint",
                template_id=self.template_id,
                expected_fingerprint=expected,
                actual_fingerprint=template.fingerprint,
            )
        if self.approval_status != self.phase_specification.approval_status:
            raise _error(
                "PHASE_APPROVAL_MISMATCH",
                "binding approval status differs from explicit phase status",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.phase",
                template_id=self.template_id,
            )
        policy = self.evaluation_policy
        if policy is not None:
            try:
                policy.validate_fingerprint()
            except Exception as error:
                raise _wrap_error(
                    "POLICY_CONTENT_MISMATCH",
                    "evaluation policy content fingerprint is invalid",
                    error,
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="binding.policy",
                    template_id=self.template_id,
                ) from error
            if (
                policy.template_id != self.template_id
                or policy.template_fingerprint != template.fingerprint
            ):
                raise _error(
                    "POLICY_TEMPLATE_MISMATCH",
                    "evaluation policy does not bind the assembled template",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="binding.policy",
                    template_id=self.template_id,
                    expected_fingerprint=template.fingerprint,
                    actual_fingerprint=policy.template_fingerprint,
                )
        payload = _binding_payload_without_fingerprint(
            template_id=self.template_id,
            artifact=self.structural_artifact,
            phase=self.phase_specification,
            full_template_fingerprint=self.full_template_fingerprint,
            policy=policy,
            approval_status=self.approval_status,
            provenance=self.provenance or {},
        )
        actual_binding = _fingerprint("model_bundle_template_binding_v1", payload)
        if self.binding_fingerprint != actual_binding:
            raise _error(
                "BINDING_FINGERPRINT_MISMATCH",
                "template binding content fingerprint mismatch",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.fingerprint",
                template_id=self.template_id,
                expected_fingerprint=self.binding_fingerprint,
                actual_fingerprint=actual_binding,
            )
        return template

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        result = _binding_payload_without_fingerprint(
            template_id=self.template_id,
            artifact=self.structural_artifact,
            phase=self.phase_specification,
            full_template_fingerprint=self.full_template_fingerprint,
            policy=self.evaluation_policy,
            approval_status=self.approval_status,
            provenance=self.provenance or {},
        )
        result["binding_fingerprint"] = self.binding_fingerprint
        return result


def _state_descriptors(
    state: Mapping[str, torch.Tensor], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "dtype": str(state[key].dtype),
            "shape": list(state[key].shape),
        }
        for key in keys
    ]


def _architecture_payload(
    model_config: Mapping[str, Any],
    model_state: Mapping[str, torch.Tensor],
    model_state_keys: tuple[str, ...],
    species_vocabulary: tuple[int, ...],
    conventions: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_config": model_config,
        "state_contract": _state_descriptors(model_state, model_state_keys),
        "species_vocabulary": list(species_vocabulary),
        "conventions": conventions,
    }


def _legacy_radius_contract_mismatch(
    model_config: Any,
) -> tuple[float, float] | None:
    """Recognize the pre-hardening v1 radius ambiguity without migrating it.

    Both cutoffs participate in trained arithmetic, so choosing either value
    as the other would silently change model meaning.  Only an otherwise
    numeric mismatch is classified here; malformed payloads remain ordinary
    model-config corruption.
    """

    if not isinstance(model_config, Mapping):
        return None
    feature = model_config.get("feature")
    higher_body = model_config.get("higher_body")
    if not isinstance(feature, Mapping) or not isinstance(higher_body, Mapping):
        return None
    feature_cutoff = feature.get("r_cut")
    higher_cutoff = higher_body.get("cutoff")
    if type(feature_cutoff) not in (int, float) or type(higher_cutoff) not in (
        int,
        float,
    ):
        return None
    left = float(feature_cutoff)
    right = float(higher_cutoff)
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0 or right <= 0:
        return None
    if left == right:
        return None
    return left, right


@dataclass(frozen=True)
class ReferenceSiteModelBundle:
    """Owned CPU snapshot of one portable reference-site potential runtime."""

    schema_version: str
    bundle_scope: str
    model_config: Mapping[str, Any]
    model_state: Mapping[str, torch.Tensor]
    model_state_keys: tuple[str, ...]
    model_floating_dtype: str
    species_vocabulary: tuple[int, ...]
    conventions: Mapping[str, Any]
    default_template_id: str
    template_bindings: tuple[ModelBundleTemplateBinding, ...]
    version_metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    architecture_fingerprint: str | None = None
    bundle_fingerprint: str | None = None

    def __post_init__(self) -> None:
        model_config = _canonical_plain(
            self.model_config, path="model_config", allow_tensors=False
        )
        state = {}
        if not isinstance(self.model_state, Mapping):
            raise TypeError("model_state must be a mapping")
        for key in sorted(self.model_state):
            if type(key) is not str:
                raise TypeError("model_state keys must be strings")
            state[key] = _cpu_clone(self.model_state[key])
        conventions = _canonical_plain(
            self.conventions, path="conventions", allow_tensors=True
        )
        versions = _canonical_plain(
            self.version_metadata, path="version_metadata", allow_tensors=False
        )
        provenance = _canonical_plain(
            self.provenance, path="provenance", allow_tensors=False
        )
        bindings = tuple(sorted(self.template_bindings, key=lambda item: item.template_id))
        object.__setattr__(self, "model_config", model_config)
        object.__setattr__(self, "model_state", state)
        object.__setattr__(self, "model_state_keys", tuple(self.model_state_keys))
        object.__setattr__(self, "species_vocabulary", tuple(int(v) for v in self.species_vocabulary))
        object.__setattr__(self, "conventions", conventions)
        object.__setattr__(self, "template_bindings", bindings)
        object.__setattr__(self, "version_metadata", versions)
        object.__setattr__(self, "provenance", provenance)
        if (
            not self.model_state_keys
            or len(self.model_state_keys) != len(set(self.model_state_keys))
            or set(self.model_state_keys) != set(state)
        ):
            raise _error(
                "INVALID_STATE_KEYS",
                "model state key contract is missing, duplicated, or inconsistent",
                schema=self.schema_version,
                validation_stage="model_state",
            )
        architecture = _fingerprint(
            "reference_site_model_architecture_v1",
            _architecture_payload(
                model_config,
                state,
                tuple(self.model_state_keys),
                tuple(self.species_vocabulary),
                conventions,
            ),
        )
        if self.architecture_fingerprint is None:
            object.__setattr__(self, "architecture_fingerprint", architecture)
        bundle_fingerprint = _fingerprint(
            "reference_site_model_bundle_v1", self._payload_without_fingerprint()
        )
        if self.bundle_fingerprint is None:
            object.__setattr__(self, "bundle_fingerprint", bundle_fingerprint)
        self.validate()

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return tuple(binding.template_id for binding in self.template_bindings)

    def _payload_without_fingerprint(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "model_state": self.model_state,
            "model_state_keys": list(self.model_state_keys),
            "model_floating_dtype": self.model_floating_dtype,
            "species_vocabulary": list(self.species_vocabulary),
            "conventions": self.conventions,
            "default_template_id": self.default_template_id,
            "template_bindings": [
                binding.to_payload() for binding in self.template_bindings
            ],
            "version_metadata": self.version_metadata,
            "provenance": self.provenance,
            "architecture_fingerprint": self.architecture_fingerprint,
        }

    def validate(self, *, bundle_path: str | None = None) -> dict[str, ReferenceTemplate]:
        schema = self.schema_version
        if schema != REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION:
            raise _error(
                "UNSUPPORTED_SCHEMA",
                "unsupported model bundle schema",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="schema",
            )
        if self.bundle_scope != REFERENCE_SITE_MODEL_BUNDLE_SCOPE:
            raise _error(
                "INVALID_SCOPE",
                "bundle scope must be portable_reference_site_potential",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="schema",
            )
        legacy_radius_mismatch = _legacy_radius_contract_mismatch(
            self.model_config
        )
        if legacy_radius_mismatch is not None:
            feature_cutoff, higher_cutoff = legacy_radius_mismatch
            raise _error(
                "INCOMPATIBLE_LEGACY_RADIUS_CONTRACT",
                "reference_site_model_bundle_v1 contains incompatible legacy "
                "cutoffs: feature.r_cut="
                f"{feature_cutoff!r}, higher_body.cutoff={higher_cutoff!r}; "
                "these trained operators cannot be migrated deterministically "
                "without changing model meaning, so the model must be rebuilt "
                "or re-exported with equal cutoffs",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="model_config.radius_contract",
            )
        try:
            config = PotentialConfig.from_dict(self.model_config)
            canonical_config = config.to_dict()
        except Exception as error:
            raise _wrap_error(
                "INVALID_MODEL_CONFIG",
                "PotentialConfig reconstruction failed",
                error,
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="model_config",
            ) from error
        if not _deep_equal(canonical_config, self.model_config):
            raise _error(
                "NONCANONICAL_MODEL_CONFIG",
                "model_config is not the canonical PotentialConfig representation",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="model_config",
            )
        if self.species_vocabulary != config.species_vocabulary:
            raise _error(
                "SPECIES_ORDER_MISMATCH",
                "bundle and PotentialConfig species ordering differ",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="model_compatibility",
            )
        keys = self.model_state_keys
        if (
            not keys
            or len(keys) != len(set(keys))
            or any(type(key) is not str or not key for key in keys)
            or set(keys) != set(self.model_state)
        ):
            raise _error(
                "INVALID_STATE_KEYS",
                "model state key contract is missing, duplicated, or inconsistent",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="model_state",
            )
        if self.model_floating_dtype not in _FLOAT_DTYPES:
            raise _error(
                "INVALID_STATE_DTYPE",
                "unsupported model floating dtype",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="model_state",
            )
        expected_dtype = _FLOAT_DTYPES[self.model_floating_dtype]
        for key in keys:
            value = self.model_state[key]
            if not isinstance(value, torch.Tensor):
                raise _error(
                    "INVALID_STATE_VALUE",
                    "model state values must be tensors",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="model_state",
                    state_key=key,
                )
            if value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None:
                raise _error(
                    "INVALID_STATE_OWNERSHIP",
                    "model state tensors must be detached CPU snapshots",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="model_state",
                    state_key=key,
                )
            if value.is_floating_point():
                if value.dtype != expected_dtype:
                    raise _error(
                        "STATE_DTYPE_MISMATCH",
                        "floating model state tensors must share model_floating_dtype",
                        bundle_path=bundle_path,
                        schema=schema,
                        validation_stage="model_state",
                        state_key=key,
                    )
                if not bool(torch.all(torch.isfinite(value))):
                    raise _error(
                        "NONFINITE_MODEL_STATE",
                        "model state contains NaN or Inf",
                        bundle_path=bundle_path,
                        schema=schema,
                        validation_stage="model_state",
                        state_key=key,
                    )
        _require_exact_keys(
            self.conventions,
            _CONVENTION_KEYS,
            name="conventions",
            bundle_path=bundle_path,
            schema=schema,
            stage="conventions",
        )
        expected_units = {
            "convention_version": MODEL_BUNDLE_CONVENTION_VERSION,
            "ordered_species_vocabulary": list(self.species_vocabulary),
            "length_unit": LENGTH_UNIT,
            "energy_unit": ENERGY_UNIT,
            "force_unit": FORCE_UNIT,
            "stress_unit": STRESS_UNIT,
            "stress_sign": STRESS_SIGN,
            "stress_voigt_order": list(STRESS_VOIGT_ORDER),
            "cell_convention": "row_vector",
            "pbc_convention": "full_3d",
            "atomic_baseline_convention": "frozen_model_buffer_v1",
            "unit_convention_version": UNIT_CONVENTION_VERSION,
        }
        for key, expected in expected_units.items():
            if not _deep_equal(self.conventions[key], expected):
                raise _error(
                    "CONVENTION_MISMATCH",
                    f"bundle convention {key!r} is incompatible",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="conventions",
                )
        site_vocabulary = self.conventions["ordered_site_type_vocabulary"]
        expected_site_vocabulary = (
            list(config.feature.site_type_vocabulary)
            if config.feature.site_type_vocabulary is not None
            else list(range(config.higher_body.site_type_count))
        )
        if site_vocabulary != expected_site_vocabulary:
            raise _error(
                "SITE_TYPE_ORDER_MISMATCH",
                "site-type convention differs from PotentialConfig",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="conventions",
            )
        alignment = self.conventions["species_alignment_weights"]
        channel_count = self.conventions["phase_channel_count"]
        if (
            not isinstance(alignment, torch.Tensor)
            or alignment.device.type != "cpu"
            or alignment.requires_grad
            or alignment.dtype != expected_dtype
            or type(channel_count) is not int
            or channel_count <= 0
            or alignment.shape != (len(self.species_vocabulary), channel_count)
            or not bool(torch.all(torch.isfinite(alignment)))
        ):
            raise _error(
                "INVALID_PHASE_CHANNEL_CONVENTION",
                "species alignment/channel convention is invalid",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="conventions",
            )
        if "species_alignment_weights" not in self.model_state or not torch.equal(
            alignment, self.model_state["species_alignment_weights"]
        ):
            raise _error(
                "MODEL_CONVENTION_STATE_MISMATCH",
                "species alignment convention differs from model state",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="model_compatibility",
                state_key="species_alignment_weights",
            )
        if "atomic_baseline" not in self.model_state or self.model_state[
            "atomic_baseline"
        ].shape != (len(self.species_vocabulary),):
            raise _error(
                "ATOMIC_BASELINE_MISMATCH",
                "atomic baseline buffer shape differs from species ordering",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="model_compatibility",
                state_key="atomic_baseline",
            )
        if not self.template_bindings:
            raise _error(
                "MISSING_TEMPLATE_BINDING",
                "bundle must contain at least one template binding",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="bindings",
            )
        ids = self.binding_ids
        if len(ids) != len(set(ids)) or ids != tuple(sorted(ids)):
            raise _error(
                "DUPLICATE_TEMPLATE_BINDING",
                "template bindings must be unique and lexically ordered",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="bindings",
            )
        if self.default_template_id not in set(ids):
            raise _error(
                "MISSING_DEFAULT_TEMPLATE",
                "default template ID is absent from bundle bindings",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="bindings.default",
                template_id=self.default_template_id,
            )
        _validate_version_metadata(
            self.version_metadata, bundle_path=bundle_path, schema=schema
        )
        templates: dict[str, ReferenceTemplate] = {}
        for binding in self.template_bindings:
            template = binding.validate(bundle_path=bundle_path)
            artifact = binding.structural_artifact
            if artifact.species_vocabulary != self.species_vocabulary:
                raise _error(
                    "SPECIES_ORDER_MISMATCH",
                    "template species ordering differs from the model",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="model_template_compatibility",
                    template_id=binding.template_id,
                )
            if not math.isclose(
                artifact.mp_cutoff,
                config.higher_body.cutoff,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                artifact.avg_num_neighbors,
                config.higher_body.avg_num_neighbors,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise _error(
                    "GRAPH_CONVENTION_MISMATCH",
                    "template cutoff/average-neighbor convention differs from the model",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="model_template_compatibility",
                    template_id=binding.template_id,
                )
            if (
                template.site_alignment_weights.shape[1] != channel_count
                or template.phase_channel_weights.shape != (channel_count,)
                or (
                    template.topology.site_types.numel() > 0
                    and bool(
                        torch.any(
                            template.topology.site_types
                            >= config.higher_body.site_type_count
                        )
                    )
                )
                or (
                    config.feature.site_type_vocabulary is not None
                    and not set(template.topology.site_types.tolist()).issubset(
                        set(config.feature.site_type_vocabulary)
                    )
                )
            ):
                raise _error(
                    "PHASE_CHANNEL_MISMATCH",
                    "template site/channel ordering is incompatible with the model",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="model_template_compatibility",
                    template_id=binding.template_id,
                )
            templates[binding.template_id] = template
        expected_version_bindings = {
            "artifact_schema_versions": sorted(
                {
                    binding.structural_artifact.schema_version
                    for binding in self.template_bindings
                }
            ),
            "template_convention_versions": sorted(
                {
                    binding.structural_artifact.convention_version
                    for binding in self.template_bindings
                }
            ),
            "domain_convention_versions": sorted(
                {
                    binding.structural_artifact.strict_domain.convention_version
                    for binding in self.template_bindings
                    if binding.structural_artifact.strict_domain is not None
                }
            ),
            "phase_convention_versions": sorted(
                {
                    binding.phase_specification.convention_version
                    for binding in self.template_bindings
                }
            ),
        }
        for key, expected in expected_version_bindings.items():
            if self.version_metadata[key] != expected:
                raise _error(
                    "VERSION_BINDING_MISMATCH",
                    f"version metadata {key!r} differs from embedded bindings",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="version.bindings",
                )
        default = templates[self.default_template_id]
        expected_default_buffers = {
            "phase_modes": default.phase_modes,
            "phase_mode_weights": default.phase_mode_weights,
            "site_alignment_weights": default.site_alignment_weights,
            "phase_channel_weights": default.phase_channel_weights,
        }
        for key, expected in expected_default_buffers.items():
            actual = self.model_state.get(key)
            if actual is None or actual.shape != expected.shape or not torch.equal(
                actual, expected.to(dtype=actual.dtype if actual is not None else expected.dtype)
            ):
                raise _error(
                    "DEFAULT_TEMPLATE_STATE_MISMATCH",
                    f"model buffer {key!r} differs from the default template binding",
                    bundle_path=bundle_path,
                    schema=schema,
                    validation_stage="model_template_compatibility",
                    template_id=self.default_template_id,
                    state_key=key,
                )
        architecture = _fingerprint(
            "reference_site_model_architecture_v1",
            _architecture_payload(
                self.model_config,
                self.model_state,
                self.model_state_keys,
                self.species_vocabulary,
                self.conventions,
            ),
        )
        if self.architecture_fingerprint != architecture:
            raise _error(
                "ARCHITECTURE_FINGERPRINT_MISMATCH",
                "model architecture fingerprint mismatch",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="architecture_fingerprint",
                expected_fingerprint=self.architecture_fingerprint,
                actual_fingerprint=architecture,
            )
        actual_bundle = _fingerprint(
            "reference_site_model_bundle_v1", self._payload_without_fingerprint()
        )
        if self.bundle_fingerprint != actual_bundle:
            raise _error(
                "BUNDLE_FINGERPRINT_MISMATCH",
                "bundle semantic content fingerprint mismatch",
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="bundle_fingerprint",
                expected_fingerprint=self.bundle_fingerprint,
                actual_fingerprint=actual_bundle,
            )
        return templates

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "bundle_scope": self.bundle_scope,
            "payload": _canonical_plain(
                self._payload_without_fingerprint(),
                path="payload",
                allow_tensors=True,
            ),
            "bundle_fingerprint": self.bundle_fingerprint,
        }


@dataclass(frozen=True)
class LoadedReferenceSiteModel:
    """Reconstructed runtime; caller owns the model and immutable bindings."""

    model: ReferenceSitePotential
    registry: TemplateRegistry
    template_contexts: Mapping[str, TemplateExecutionContext]
    evaluation_policies: Mapping[str, EvaluationPolicy]
    default_template_id: str
    bundle_fingerprint: str
    architecture_fingerprint: str
    template_fingerprints: Mapping[str, str]
    structural_fingerprints: Mapping[str, str]
    metadata: Mapping[str, Any]


def _model_floating_dtype(model: ReferenceSitePotential) -> torch.dtype:
    values = [
        tensor.dtype
        for tensor in model.state_dict().values()
        if tensor.is_floating_point()
    ]
    if not values or any(value != values[0] for value in values):
        raise ValueError("model state must use one float32/float64 floating dtype")
    _dtype_name(values[0])
    return values[0]


def _topology_matches_model(model: ReferenceSitePotential, template: ReferenceTemplate) -> bool:
    left = model.topology
    right = template.topology
    return (
        all(
            torch.equal(a.detach().cpu(), b.detach().cpu())
            for a, b in (
                (left.reference_fractional, right.reference_fractional),
                (left.site_types, right.site_types),
                (left.edge_index, right.edge_index),
                (left.shifts, right.shifts),
                (left.reference_cell, right.reference_cell),
            )
        )
        and left.cutoff == right.cutoff
        and left.skin == right.skin
        and left.maximum_strain == right.maximum_strain
        and left.minimum_edge_length == right.minimum_edge_length
        and tuple(left.pbc) == tuple(right.pbc)
    )


def _version_metadata(bindings: tuple[ModelBundleTemplateBinding, ...]) -> dict[str, Any]:
    torch_version, e3nn_version = _current_versions()
    return {
        "compatibility_policy": RUNTIME_COMPATIBILITY_POLICY,
        "refsite_mlip_version": _PACKAGE_VERSION,
        "torch_version": torch_version,
        "e3nn_version": e3nn_version,
        "python_version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ],
        "bundle_schema_version": REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
        "artifact_schema_versions": sorted(
            {binding.structural_artifact.schema_version for binding in bindings}
        ),
        "template_convention_versions": sorted(
            {binding.structural_artifact.convention_version for binding in bindings}
        ),
        "domain_convention_versions": sorted(
            {
                binding.structural_artifact.strict_domain.convention_version
                for binding in bindings
                if binding.structural_artifact.strict_domain is not None
            }
        ),
        "phase_convention_versions": sorted(
            {binding.phase_specification.convention_version for binding in bindings}
        ),
        "unit_convention_version": UNIT_CONVENTION_VERSION,
    }


def capture_reference_site_model_bundle(
    *,
    model: ReferenceSitePotential,
    structural_artifacts: Mapping[str, ReferenceStructureArtifact],
    phase_specifications: Mapping[str, PhaseSpecification],
    evaluation_policies: Mapping[str, EvaluationPolicy] | None = None,
    default_template_id: str,
    provenance: Mapping[str, Any] | None = None,
) -> ReferenceSiteModelBundle:
    """Capture model and exact template bindings without mutating caller state."""

    if not isinstance(model, ReferenceSitePotential):
        raise TypeError("model must be a ReferenceSitePotential")
    if not isinstance(structural_artifacts, Mapping) or not structural_artifacts:
        raise ValueError("structural_artifacts must be a nonempty mapping")
    if not isinstance(phase_specifications, Mapping):
        raise TypeError("phase_specifications must be a mapping")
    artifact_ids = set(structural_artifacts)
    if artifact_ids != set(phase_specifications):
        raise _error(
            "TEMPLATE_BINDING_SET_MISMATCH",
            "structural artifact and phase specification IDs must match exactly",
            schema=REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
            validation_stage="capture.bindings",
        )
    policies = {} if evaluation_policies is None else evaluation_policies
    if not isinstance(policies, Mapping) or not set(policies).issubset(artifact_ids):
        raise _error(
            "POLICY_BINDING_SET_MISMATCH",
            "evaluation policy IDs must be a subset of template bindings",
            schema=REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
            validation_stage="capture.bindings",
        )
    if default_template_id not in artifact_ids:
        raise _error(
            "MISSING_DEFAULT_TEMPLATE",
            "default template ID is absent from structural artifacts",
            schema=REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
            validation_stage="capture.default",
            template_id=default_template_id,
        )
    for mapping_name, mapping in (
        ("structural_artifacts", structural_artifacts),
        ("phase_specifications", phase_specifications),
        ("evaluation_policies", policies),
    ):
        if any(type(key) is not str or not key for key in mapping):
            raise TypeError(f"{mapping_name} keys must be nonempty strings")
    bindings = []
    templates = {}
    for template_id in sorted(artifact_ids):
        artifact = structural_artifacts[template_id]
        phase = phase_specifications[template_id]
        if not isinstance(artifact, ReferenceStructureArtifact):
            raise TypeError("structural_artifact values must be ReferenceStructureArtifact")
        if artifact.template_id != template_id:
            raise _error(
                "ARTIFACT_TEMPLATE_MISMATCH",
                "structural mapping key differs from artifact template ID",
                schema=REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
                validation_stage="capture.bindings",
                template_id=template_id,
            )
        template = assemble_reference_template_from_artifact(
            artifact, phase_specification=phase
        )
        policy = policies.get(template_id)
        binding = ModelBundleTemplateBinding(
            template_id=template_id,
            structural_artifact=artifact,
            phase_specification=phase,
            full_template_fingerprint=template.fingerprint,
            evaluation_policy=policy,
            approval_status=phase.approval_status,
            provenance={
                "phase_convention_version": phase.convention_version,
                "policy_convention_version": (
                    None if policy is None else policy.convention_version
                ),
            },
        )
        bindings.append(binding)
        templates[template_id] = template
    bindings_tuple = tuple(bindings)
    default_template = templates[default_template_id]
    if model.config.species_vocabulary != default_template.supported_species:
        raise _error(
            "SPECIES_ORDER_MISMATCH",
            "model species ordering differs from the default template",
            schema=REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
            validation_stage="capture.model_compatibility",
            template_id=default_template_id,
        )
    if not _topology_matches_model(model, default_template):
        raise _error(
            "DEFAULT_TEMPLATE_TOPOLOGY_MISMATCH",
            "model topology differs from the default structural artifact",
            schema=REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
            validation_stage="capture.model_compatibility",
            template_id=default_template_id,
        )
    floating_dtype = _model_floating_dtype(model)
    state = {
        key: _cpu_clone(value) for key, value in model.state_dict().items()
    }
    state_keys = tuple(model.state_dict().keys())
    site_vocabulary = (
        list(model.config.feature.site_type_vocabulary)
        if model.config.feature.site_type_vocabulary is not None
        else list(range(model.config.higher_body.site_type_count))
    )
    conventions = {
        "convention_version": MODEL_BUNDLE_CONVENTION_VERSION,
        "ordered_species_vocabulary": list(model.config.species_vocabulary),
        "ordered_site_type_vocabulary": site_vocabulary,
        "phase_channel_count": int(model.species_alignment_weights.shape[1]),
        "species_alignment_weights": _cpu_clone(model.species_alignment_weights),
        "length_unit": LENGTH_UNIT,
        "energy_unit": ENERGY_UNIT,
        "force_unit": FORCE_UNIT,
        "stress_unit": STRESS_UNIT,
        "stress_sign": STRESS_SIGN,
        "stress_voigt_order": list(STRESS_VOIGT_ORDER),
        "cell_convention": "row_vector",
        "pbc_convention": "full_3d",
        "atomic_baseline_convention": "frozen_model_buffer_v1",
        "unit_convention_version": UNIT_CONVENTION_VERSION,
    }
    return ReferenceSiteModelBundle(
        schema_version=REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION,
        bundle_scope=REFERENCE_SITE_MODEL_BUNDLE_SCOPE,
        model_config=model.config.to_dict(),
        model_state=state,
        model_state_keys=state_keys,
        model_floating_dtype=_dtype_name(floating_dtype),
        species_vocabulary=tuple(model.config.species_vocabulary),
        conventions=conventions,
        default_template_id=default_template_id,
        template_bindings=bindings_tuple,
        version_metadata=_version_metadata(bindings_tuple),
        provenance=_canonical_plain(
            {} if provenance is None else provenance,
            path="provenance",
            allow_tensors=False,
        ),
    )


def save_reference_site_model_bundle(
    path: str | os.PathLike[str],
    bundle: ReferenceSiteModelBundle,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically save one plain weights-only-safe bundle payload."""

    if not isinstance(bundle, ReferenceSiteModelBundle):
        raise TypeError("bundle must be a ReferenceSiteModelBundle")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    bundle.validate()
    target = Path(path)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise FileNotFoundError(f"bundle parent directory does not exist: {parent}")
    if target.is_symlink():
        raise _error(
            "SYMLINK_REJECTED",
            "bundle target must not be a symbolic link",
            bundle_path=str(target),
            validation_stage="save.path",
        )
    if target.exists():
        if not target.is_file():
            raise ValueError("bundle target exists and is not a regular file")
        if not overwrite:
            raise FileExistsError(f"bundle already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(bundle.to_payload(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if target.is_symlink():
            raise _error(
                "SYMLINK_REJECTED",
                "bundle target became a symbolic link",
                bundle_path=str(target),
                validation_stage="save.replace",
            )
        commit_temporary_file(temporary, target, overwrite=overwrite)
        try:
            directory_descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _binding_from_payload(
    value: Any,
    *,
    bundle_path: str | None,
    schema: str,
) -> ModelBundleTemplateBinding:
    provisional_template_id = (
        value.get("template_id") if isinstance(value, Mapping) else None
    )
    payload = _require_exact_keys(
        value,
        _BINDING_KEYS,
        name="template_binding",
        bundle_path=bundle_path,
        schema=schema,
        stage="binding.payload",
        template_id=provisional_template_id,
    )
    template_id = payload["template_id"]
    if not isinstance(template_id, str) or not template_id:
        raise _error(
            "INVALID_TEMPLATE_ID",
            "binding template_id must be nonempty",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="binding.payload",
        )
    try:
        artifact = _artifact_from_safe_payload(
            payload["structural_artifact"],
            artifact_path=(
                None if bundle_path is None else f"{bundle_path}#{template_id}"
            ),
        )
    except ReferenceStructureArtifactError as error:
        raise _wrap_error(
            "NESTED_ARTIFACT_INVALID",
            "nested structural artifact payload is invalid",
            error,
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="binding.artifact",
            template_id=template_id,
        ) from error
    if payload["structural_fingerprint"] != artifact.structural_fingerprint:
        raise _error(
            "STRUCTURAL_FINGERPRINT_MISMATCH",
            "binding structural fingerprint differs from nested artifact",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="binding.artifact",
            template_id=template_id,
            expected_fingerprint=payload["structural_fingerprint"],
            actual_fingerprint=artifact.structural_fingerprint,
        )
    phase_payload = _require_exact_keys(
        payload["phase_specification"],
        _PHASE_KEYS,
        name="phase_specification",
        bundle_path=bundle_path,
        schema=schema,
        stage="binding.phase",
        template_id=template_id,
    )
    try:
        phase = PhaseSpecification.from_dict(phase_payload)
    except Exception as error:
        raise _wrap_error(
            "INVALID_PHASE_SPECIFICATION",
            "phase specification payload is invalid",
            error,
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="binding.phase",
            template_id=template_id,
        ) from error
    policy_payload = payload["evaluation_policy"]
    policy = None
    if policy_payload is not None:
        policy_mapping = _require_exact_keys(
            policy_payload,
            _POLICY_KEYS,
            name="evaluation_policy",
            bundle_path=bundle_path,
            schema=schema,
            stage="binding.policy",
            template_id=template_id,
        )
        try:
            policy = EvaluationPolicy.from_dict(dict(policy_mapping))
        except Exception as error:
            raise _wrap_error(
                "INVALID_EVALUATION_POLICY",
                "evaluation policy payload is invalid",
                error,
                bundle_path=bundle_path,
                schema=schema,
                validation_stage="binding.policy",
                template_id=template_id,
            ) from error
    try:
        return ModelBundleTemplateBinding(
            template_id=template_id,
            structural_artifact=artifact,
            phase_specification=phase,
            full_template_fingerprint=payload["full_template_fingerprint"],
            evaluation_policy=policy,
            approval_status=payload["approval_status"],
            provenance=payload["provenance"],
            binding_fingerprint=payload["binding_fingerprint"],
        )
    except ModelBundleError:
        raise
    except Exception as error:
        raise _wrap_error(
            "INVALID_TEMPLATE_BINDING",
            "template binding construction failed",
            error,
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="binding.construction",
            template_id=template_id,
        ) from error


def _bundle_from_safe_payload(
    raw: Any, *, bundle_path: str | None
) -> ReferenceSiteModelBundle:
    top = _require_exact_keys(
        raw,
        _TOP_LEVEL_KEYS,
        name="bundle",
        bundle_path=bundle_path,
        schema=(raw.get("schema_version") if isinstance(raw, Mapping) else None),
        stage="top_level",
    )
    schema = top["schema_version"]
    if schema != REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION:
        raise _error(
            "UNSUPPORTED_SCHEMA",
            "unsupported model bundle schema",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="schema",
        )
    if top["bundle_scope"] != REFERENCE_SITE_MODEL_BUNDLE_SCOPE:
        raise _error(
            "INVALID_SCOPE",
            "invalid model bundle scope",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="schema",
        )
    payload = _require_exact_keys(
        top["payload"],
        _PAYLOAD_KEYS,
        name="payload",
        bundle_path=bundle_path,
        schema=schema,
        stage="payload",
    )
    bindings_payload = payload["template_bindings"]
    if not isinstance(bindings_payload, list):
        raise _error(
            "INVALID_TEMPLATE_BINDINGS",
            "template_bindings must be a canonical list",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="bindings",
        )
    bindings = tuple(
        _binding_from_payload(item, bundle_path=bundle_path, schema=schema)
        for item in bindings_payload
    )
    state_payload = payload["model_state"]
    if not isinstance(state_payload, Mapping) or any(
        type(key) is not str or not isinstance(value, torch.Tensor)
        for key, value in state_payload.items()
    ):
        raise _error(
            "INVALID_MODEL_STATE",
            "model_state must be a string-to-tensor mapping",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="model_state",
        )
    keys = payload["model_state_keys"]
    species = payload["species_vocabulary"]
    if not isinstance(keys, list) or any(type(key) is not str for key in keys):
        raise _error(
            "INVALID_STATE_KEYS",
            "model_state_keys must be a canonical string list",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="model_state",
        )
    if not isinstance(species, list) or any(type(value) is not int for value in species):
        raise _error(
            "INVALID_SPECIES_ORDER",
            "species_vocabulary must be a canonical integer list",
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="model_config",
        )
    try:
        bundle = ReferenceSiteModelBundle(
            schema_version=schema,
            bundle_scope=top["bundle_scope"],
            model_config=payload["model_config"],
            model_state=state_payload,
            model_state_keys=tuple(keys),
            model_floating_dtype=payload["model_floating_dtype"],
            species_vocabulary=tuple(species),
            conventions=payload["conventions"],
            default_template_id=payload["default_template_id"],
            template_bindings=bindings,
            version_metadata=payload["version_metadata"],
            provenance=payload["provenance"],
            architecture_fingerprint=payload["architecture_fingerprint"],
            bundle_fingerprint=top["bundle_fingerprint"],
        )
        bundle.validate(bundle_path=bundle_path)
        return bundle
    except ModelBundleError:
        raise
    except Exception as error:
        raise _wrap_error(
            "INVALID_PAYLOAD",
            "bundle construction failed",
            error,
            bundle_path=bundle_path,
            schema=schema,
            validation_stage="construction",
        ) from error


def load_reference_site_model_bundle(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> ReferenceSiteModelBundle:
    """Weights-only load followed by complete nested/content validation."""

    target = Path(path)
    display_path = str(target)
    if target.is_symlink():
        raise _error(
            "SYMLINK_REJECTED",
            "bundle path must not be a symbolic link",
            bundle_path=display_path,
            validation_stage="load.path",
        )
    if not target.exists():
        raise FileNotFoundError(f"bundle does not exist: {target}")
    if not target.is_file():
        raise ValueError("bundle path must be a regular file")
    try:
        raw = torch.load(target, map_location=map_location, weights_only=True)
    except Exception as error:
        raise _wrap_error(
            "SAFE_LOAD_FAILURE",
            "weights-only bundle load failed",
            error,
            bundle_path=display_path,
            validation_stage="weights_only_load",
        ) from error
    return _bundle_from_safe_payload(raw, bundle_path=display_path)


def _freeze_runtime_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_runtime_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_runtime_metadata(item) for item in value)
    return value


def instantiate_reference_site_model_bundle(
    bundle: ReferenceSiteModelBundle,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> LoadedReferenceSiteModel:
    """Reconstruct architecture, registry, contexts, and policies without builders."""

    if not isinstance(bundle, ReferenceSiteModelBundle):
        raise TypeError("bundle must be a ReferenceSiteModelBundle")
    if dtype not in (torch.float32, torch.float64):
        raise _error(
            "UNSUPPORTED_DTYPE",
            "runtime dtype must be float32 or float64",
            schema=bundle.schema_version,
            validation_stage="instantiate.device",
        )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise _error(
            "UNSUPPORTED_DEVICE",
            "CUDA runtime was requested but is unavailable",
            schema=bundle.schema_version,
            validation_stage="instantiate.device",
        )
    templates = bundle.validate()
    bindings = {binding.template_id: binding for binding in bundle.template_bindings}
    default = templates[bundle.default_template_id]
    config = PotentialConfig.from_dict(bundle.model_config)
    stored_dtype = _FLOAT_DTYPES[bundle.model_floating_dtype]
    alignment = bundle.conventions["species_alignment_weights"]
    try:
        with torch.random.fork_rng(devices=[]):
            model = ReferenceSitePotential(
                config,
                default.topology,
                default.phase_modes,
                default.phase_mode_weights,
                alignment,
                default.site_alignment_weights,
                default.phase_channel_weights,
                atomic_baseline=None,
            ).to(device="cpu", dtype=stored_dtype)
            actual_state = model.state_dict()
            if tuple(actual_state) != bundle.model_state_keys:
                raise _error(
                    "STATE_KEY_MISMATCH",
                    "reconstructed architecture state keys differ from the bundle",
                    schema=bundle.schema_version,
                    validation_stage="instantiate.state_contract",
                )
            for key in bundle.model_state_keys:
                expected = bundle.model_state[key]
                actual = actual_state[key]
                if actual.shape != expected.shape:
                    raise _error(
                        "STATE_SHAPE_MISMATCH",
                        "reconstructed state tensor shape differs from the bundle",
                        schema=bundle.schema_version,
                        validation_stage="instantiate.state_contract",
                        state_key=key,
                    )
                if actual.dtype != expected.dtype:
                    raise _error(
                        "STATE_DTYPE_MISMATCH",
                        "reconstructed state tensor dtype differs from the bundle",
                        schema=bundle.schema_version,
                        validation_stage="instantiate.state_contract",
                        state_key=key,
                    )
            model.load_state_dict(dict(bundle.model_state), strict=True)
            model.to(device=target_device, dtype=dtype)
            model.eval()
    except ModelBundleError:
        raise
    except Exception as error:
        raise _wrap_error(
            "MODEL_INSTANTIATION_FAILED",
            "ReferenceSitePotential reconstruction or strict state load failed",
            error,
            schema=bundle.schema_version,
            validation_stage="instantiate.model",
        ) from error

    registry = TemplateRegistry()
    contexts: dict[str, TemplateExecutionContext] = {}
    policies: dict[str, EvaluationPolicy] = {}
    template_fingerprints = {}
    structural_fingerprints = {}
    try:
        for template_id in sorted(templates):
            template = templates[template_id]
            registry.add(template)
            binding = bindings[template_id]
            context = TemplateExecutionContext.from_reference_template(
                template,
                avg_num_neighbors=binding.structural_artifact.avg_num_neighbors,
            )
            context.materialize(device=target_device, dtype=dtype)
            contexts[template_id] = context
            if binding.evaluation_policy is not None:
                policy = EvaluationPolicy.from_dict(
                    binding.evaluation_policy.to_dict()
                )
                policy.materialize_candidate_offsets(
                    device=target_device, dtype=dtype
                )
                policies[template_id] = policy
            template_fingerprints[template_id] = template.fingerprint
            structural_fingerprints[template_id] = (
                binding.structural_artifact.structural_fingerprint
            )
    except Exception as error:
        raise _wrap_error(
            "RUNTIME_BINDING_FAILED",
            "registry/context/policy materialization failed",
            error,
            schema=bundle.schema_version,
            validation_stage="instantiate.runtime_bindings",
        ) from error
    metadata = _freeze_runtime_metadata(
        {
            "schema_version": bundle.schema_version,
            "bundle_scope": bundle.bundle_scope,
            "version_metadata": bundle.version_metadata,
            "conventions": {
                key: value
                for key, value in bundle.conventions.items()
                if key != "species_alignment_weights"
            },
            "provenance": bundle.provenance,
            "phase_approval_status": {
                binding.template_id: binding.approval_status
                for binding in bundle.template_bindings
            },
            "candidate_neighbor_state_persisted": False,
            "runtime_device": str(target_device),
            "runtime_dtype": str(dtype),
        }
    )
    return LoadedReferenceSiteModel(
        model=model,
        registry=registry,
        template_contexts=MappingProxyType(contexts),
        evaluation_policies=MappingProxyType(policies),
        default_template_id=bundle.default_template_id,
        bundle_fingerprint=bundle.bundle_fingerprint or "",
        architecture_fingerprint=bundle.architecture_fingerprint or "",
        template_fingerprints=MappingProxyType(template_fingerprints),
        structural_fingerprints=MappingProxyType(structural_fingerprints),
        metadata=metadata,
    )


__all__ = [
    "MODEL_BUNDLE_CONVENTION_VERSION",
    "REFERENCE_SITE_MODEL_BUNDLE_SCHEMA_VERSION",
    "REFERENCE_SITE_MODEL_BUNDLE_SCOPE",
    "RUNTIME_COMPATIBILITY_POLICY",
    "UNIT_CONVENTION_VERSION",
    "LoadedReferenceSiteModel",
    "ModelBundleError",
    "ModelBundleTemplateBinding",
    "ReferenceSiteModelBundle",
    "capture_reference_site_model_bundle",
    "instantiate_reference_site_model_bundle",
    "load_reference_site_model_bundle",
    "save_reference_site_model_bundle",
]
