"""Persist POSCAR-first reference preparation without rebuilding it.

The automatic builder is deliberately upstream of this module.  This adapter
only commits its canonical reference specification and audit certificate,
strictly reloads both, and binds the reloaded specification objects back to
the already validated immutable structural-artifact snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from refsite_mlip.models import ReferenceSiteModelBundle

from .run_directory import (
    ResumeRunLock,
    TrainingRunDirectory,
    _atomic_write_text,
    canonical_runtime_json,
    load_runtime_json,
)
from .scratch_preparation import ScratchTrainingPreparation

if TYPE_CHECKING:
    from refsite_mlip.config import ReferenceSpecificationConfig


AUTOMATIC_REFERENCE_MATERIALIZATION_VERSION = (
    "automatic_reference_materialization_v1"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("materialization metadata keys must be strings")
        return {key: _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("materialization metadata contains a nonfinite value")
        return value
    if value is None or type(value) in (str, bool, int):
        return value
    raise TypeError(
        "materialization metadata contains non-plain "
        f"{type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return _plain(value)


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = canonical_runtime_json(_plain(value)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_fingerprint(value: Any) -> str:
    return _fingerprint(value.to_dict())


def _safe_template_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    folded: set[str] = set()
    for value in result:
        if (
            type(value) is not str
            or _SAFE_ID.fullmatch(value) is None
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
        ):
            raise AutomaticReferenceMaterializationError(
                "UNSAFE_TEMPLATE_ID",
                "automatic template ID is not a canonical filesystem-safe ID",
                stage="reference_materialization.identity",
                template_id=value if type(value) is str else None,
            )
        normalized = value.casefold()
        if normalized in folded:
            raise AutomaticReferenceMaterializationError(
                "CASE_NORMALIZED_TEMPLATE_ID_COLLISION",
                "automatic template IDs collide after case normalization",
                stage="reference_materialization.identity",
                template_id=value,
            )
        folded.add(normalized)
    if len(set(result)) != len(result):
        raise AutomaticReferenceMaterializationError(
            "DUPLICATE_TEMPLATE_ID",
            "automatic template IDs must be unique",
            stage="reference_materialization.identity",
        )
    return result


class AutomaticReferenceMaterializationError(RuntimeError):
    """Structured persistence/reload failure with committed-file accounting."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        template_id: str | None = None,
        path: str | Path | None = None,
        completed_files: Sequence[str] = (),
        original_error: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.message = message
        self.stage = stage
        self.template_id = template_id
        self.path = None if path is None else str(path)
        self.completed_persistence_stages = tuple(completed_files)
        self.recoverable_artifacts = tuple(completed_files)
        self.rollback_performed = False
        self.original_error = original_error
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        super().__init__(
            f"[{reason_code}] stage={stage!r} template_id={template_id!r} "
            f"path={self.path!r} {message}"
        )


@dataclass(frozen=True)
class MaterializedAutomaticReferences:
    """Reload-bound preparation and path-independent persistence manifest."""

    preparation: ScratchTrainingPreparation
    manifest: Mapping[str, Any]
    completed_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.preparation, ScratchTrainingPreparation):
            raise TypeError("preparation must be ScratchTrainingPreparation")
        object.__setattr__(self, "manifest", _freeze(self.manifest))
        object.__setattr__(self, "completed_files", tuple(self.completed_files))


def _automatic_payload(preparation: ScratchTrainingPreparation) -> dict[str, Any]:
    value = preparation.automatic_reference_preparation
    if not isinstance(value, Mapping):
        raise AutomaticReferenceMaterializationError(
            "AUTOMATIC_REFERENCE_PREPARATION_MISSING",
            "scratch preparation has no automatic-reference snapshot",
            stage="reference_materialization.preflight",
        )
    payload = _plain(value)
    required = {
        "assignments",
        "content_fingerprint",
        "convention_version",
        "references",
        "scope",
        "species_vocabulary",
    }
    if set(payload) != required or payload.get("scope") != "dataset_bounded":
        raise AutomaticReferenceMaterializationError(
            "INVALID_AUTOMATIC_REFERENCE_PREPARATION",
            "automatic-reference snapshot has missing or unknown fields",
            stage="reference_materialization.preflight",
        )
    if _SHA256.fullmatch(str(payload["content_fingerprint"])) is None:
        raise AutomaticReferenceMaterializationError(
            "INVALID_AUTOMATIC_REFERENCE_FINGERPRINT",
            "automatic-reference fingerprint is invalid",
            stage="reference_materialization.preflight",
        )
    return payload


def _certificates(
    preparation: ScratchTrainingPreparation,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    payload = _automatic_payload(preparation)
    references = payload["references"]
    if not isinstance(references, list) or not references:
        raise AutomaticReferenceMaterializationError(
            "INVALID_AUTOMATIC_REFERENCE_PREPARATION",
            "automatic-reference snapshot must contain references",
            stage="reference_materialization.preflight",
        )
    certificates: dict[str, dict[str, Any]] = {}
    evaluation_certificates: dict[str, dict[str, Any]] = {}
    for item in references:
        if not isinstance(item, Mapping):
            raise AutomaticReferenceMaterializationError(
                "INVALID_AUTOMATIC_REFERENCE_CERTIFICATE",
                "automatic reference certificate must be a mapping",
                stage="reference_materialization.certificate",
            )
        certificate = _plain(item)
        evaluation_certificate = certificate.pop("evaluation_certificate", None)
        template_id = certificate.get("template_id")
        if type(template_id) is not str:
            raise AutomaticReferenceMaterializationError(
                "INVALID_AUTOMATIC_REFERENCE_CERTIFICATE",
                "certificate template ID is missing",
                stage="reference_materialization.certificate",
            )
        certificate.pop("poscar", None)
        declared = certificate.get("certificate_sha256")
        content = dict(certificate)
        content.pop("certificate_sha256", None)
        if declared != _fingerprint(content):
            raise AutomaticReferenceMaterializationError(
                "REFERENCE_CERTIFICATE_FINGERPRINT_MISMATCH",
                "automatic reference certificate fingerprint differs from content",
                stage="reference_materialization.certificate",
                template_id=template_id,
            )
        if template_id in certificates:
            raise AutomaticReferenceMaterializationError(
                "DUPLICATE_TEMPLATE_ID",
                "automatic certificates contain a duplicate template ID",
                stage="reference_materialization.certificate",
                template_id=template_id,
            )
        certificates[template_id] = certificate
        if evaluation_certificate is not None:
            if not isinstance(evaluation_certificate, Mapping):
                raise AutomaticReferenceMaterializationError(
                    "INVALID_AUTOMATIC_EVALUATION_CERTIFICATE",
                    "automatic evaluation certificate must be a mapping",
                    stage="reference_materialization.evaluation_certificate",
                    template_id=template_id,
                )
            from refsite_mlip.config import (
                AutomaticEvaluationPolicyAuditError,
                validate_automatic_evaluation_certificate,
            )

            try:
                evaluation = validate_automatic_evaluation_certificate(
                    evaluation_certificate
                )
            except AutomaticEvaluationPolicyAuditError as error:
                reason = (
                    "EVALUATION_CERTIFICATE_FINGERPRINT_MISMATCH"
                    if error.reason_code
                    == "EVALUATION_CERTIFICATE_INTEGRITY_MISMATCH"
                    else "INVALID_AUTOMATIC_EVALUATION_CERTIFICATE"
                )
                raise AutomaticReferenceMaterializationError(
                    reason,
                    "automatic evaluation certificate validation failed",
                    stage="reference_materialization.evaluation_certificate",
                    template_id=template_id,
                    original_error=error,
                ) from error
            evaluation_certificates[template_id] = evaluation
    _safe_template_ids(tuple(certificates))
    return payload, certificates, evaluation_certificates


def _expected_specification(
    preparation: ScratchTrainingPreparation,
    source: Any,
    certificate: Mapping[str, Any],
) -> ReferenceSpecificationConfig:
    from refsite_mlip.config import ReferenceSpecificationConfig

    specification = ReferenceSpecificationConfig(
        builder=source.builder,
        phase_specification=source.phase_specification,
        evaluation_policy=source.evaluation_policy,
        species_alignment_weights=preparation.model_source.species_alignment_weights,
        poscar_sha256=certificate["poscar_content_sha256"],
    )
    if specification.content_fingerprint != certificate.get(
        "specification_sha256"
    ):
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_SPECIFICATION_FINGERPRINT_MISMATCH",
            "compiled reference specification differs from automatic preparation",
            stage="reference_materialization.specification",
            template_id=source.template_id,
        )
    return specification


def _validate_preparation_binding(
    preparation: ScratchTrainingPreparation,
    template_id: str,
    specification: ReferenceSpecificationConfig,
    certificate: Mapping[str, Any],
) -> None:
    metadata = preparation.template_fingerprints.get(template_id)
    artifact = preparation.structural_artifacts.get(template_id)
    context = preparation.template_contexts.get(template_id)
    policy = preparation.evaluation_policies.get(template_id)
    evaluation_metadata = certificate.get("evaluation_policy")
    if not isinstance(metadata, Mapping) or artifact is None or context is None:
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_BINDING_MISSING",
            "prepared reference binding is incomplete",
            stage="reference_materialization.binding",
            template_id=template_id,
        )
    comparisons = (
        (certificate.get("artifact_sha256"), artifact.structural_fingerprint),
        (certificate.get("template_sha256"), context.fingerprint),
        (
            certificate.get("phase_specification_sha256"),
            _phase_fingerprint(specification.phase_specification),
        ),
        (
            metadata.get("structural_artifact_fingerprint"),
            artifact.structural_fingerprint,
        ),
        (metadata.get("full_template_fingerprint"), context.fingerprint),
    )
    if any(left != right for left, right in comparisons):
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_BINDING_FINGERPRINT_MISMATCH",
            "reference specification, artifact, phase, or template fingerprint differs",
            stage="reference_materialization.binding",
            template_id=template_id,
        )
    if (
        specification.phase_specification.approval_status != "provisional"
        or certificate.get("approval_status") != "provisional"
    ):
        raise AutomaticReferenceMaterializationError(
            "AUTOMATIC_PHASE_NOT_PROVISIONAL",
            "automatic phase specification must remain provisional",
            stage="reference_materialization.binding",
            template_id=template_id,
        )
    if evaluation_metadata is None:
        if specification.evaluation_policy is not None or policy is not None:
            raise AutomaticReferenceMaterializationError(
                "AUTOMATIC_EVALUATION_POLICY_PRESENT",
                "sinkhorn-only automatic references must not create an EvaluationPolicy",
                stage="reference_materialization.binding",
                template_id=template_id,
            )
    else:
        if (
            not isinstance(evaluation_metadata, Mapping)
            or evaluation_metadata.get("status") != "qualified"
            or specification.evaluation_policy is None
            or policy is None
            or specification.evaluation_policy.content_fingerprint
            != evaluation_metadata.get("content_fingerprint")
            or policy.content_fingerprint
            != specification.evaluation_policy.content_fingerprint
            or certificate.get("radius_fingerprint")
            != preparation.radius_config.content_fingerprint
        ):
            raise AutomaticReferenceMaterializationError(
                "AUTOMATIC_EVALUATION_POLICY_MISMATCH",
                "qualified automatic EvaluationPolicy binding differs from preparation",
                stage="reference_materialization.binding",
                template_id=template_id,
            )
    if certificate.get("num_reference_sites") != artifact.diagnostics.num_sites:
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_SITE_COUNT_MISMATCH",
            "automatic certificate site count differs from structural artifact",
            stage="reference_materialization.binding",
            template_id=template_id,
        )
    domain = artifact.strict_domain
    reference_composition = (
        None if domain is None else list(domain.reference_composition)
    )
    stabilizer_fingerprint = hashlib.sha256(
        artifact.stabilizer_permutations.contiguous().numpy().tobytes()
    ).hexdigest()
    if (
        certificate.get("global_species_ordering")
        != list(preparation.species_vocabulary)
        or certificate.get("global_site_type_ordering")
        != list(preparation.species_vocabulary)
        or certificate.get("reference_composition") != reference_composition
        or certificate.get("stabilizer_size")
        != int(artifact.stabilizer_permutations.shape[0])
        or certificate.get("stabilizer_permutation_sha256")
        != stabilizer_fingerprint
    ):
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_DOMAIN_METADATA_MISMATCH",
            "automatic species/site-type/composition/stabilizer metadata differs",
            stage="reference_materialization.binding",
            template_id=template_id,
        )
    vacancies = certificate.get("vacancies")
    if not isinstance(vacancies, Mapping):
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_ASSIGNMENT_MANIFEST_MISMATCH",
            "automatic vacancy/assignment manifest is missing",
            stage="reference_materialization.binding",
            template_id=template_id,
        )
    for split, samples in (
        ("train", preparation.train_samples),
        ("validation", preparation.validation_samples),
    ):
        split_payload = vacancies.get(split)
        selected = tuple(
            sample for sample in samples if sample.template_id == template_id
        )
        expected_ids = [sample.sample_id for sample in selected]
        expected_k = sorted(
            {
                artifact.diagnostics.num_sites - sample.num_atoms
                for sample in selected
            }
        )
        if (
            not isinstance(split_payload, Mapping)
            or split_payload.get("sample_ids") != expected_ids
            or split_payload.get("observed_K_values") != expected_k
        ):
            raise AutomaticReferenceMaterializationError(
                "REFERENCE_ASSIGNMENT_MANIFEST_MISMATCH",
                "automatic sample assignment or vacancy K manifest differs",
                stage="reference_materialization.binding",
                template_id=template_id,
            )


def _write_json(path: Path, value: Mapping[str, Any], *, stage: str) -> None:
    _atomic_write_text(
        path,
        canonical_runtime_json(_plain(value)),
        overwrite=False,
        stage=stage,
    )


def _load_reference(path: Path) -> "ReferenceSpecificationConfig":
    from refsite_mlip.config import load_reference_specification

    return load_reference_specification(path)


def _load_certificate(path: Path, *, stage: str) -> dict[str, Any]:
    return load_runtime_json(path, stage=stage)


def materialize_automatic_references(
    preparation: ScratchTrainingPreparation,
    directory: TrainingRunDirectory,
    lock: ResumeRunLock,
) -> MaterializedAutomaticReferences:
    """Commit/reload automatic references and return a reload-bound snapshot."""

    completed: list[str] = []
    current_template: str | None = None
    current_path: Path | None = None
    try:
        from refsite_mlip.config import ScratchModelSourceConfig

        if not isinstance(preparation.model_source, ScratchModelSourceConfig):
            raise TypeError("automatic materialization requires scratch model source")
        lock.validate_owned(directory.resume_lock_path)
        payload, certificates, evaluation_certificates = _certificates(preparation)
        sources = {
            source.template_id: source
            for source in preparation.model_source.reference_templates
        }
        ids = _safe_template_ids(tuple(sorted(sources)))
        if set(ids) != set(certificates):
            raise AutomaticReferenceMaterializationError(
                "REFERENCE_TEMPLATE_SET_MISMATCH",
                "compiled sources and automatic certificates name different templates",
                stage="reference_materialization.preflight",
            )
        if not set(evaluation_certificates).issubset(set(ids)):
            raise AutomaticReferenceMaterializationError(
                "REFERENCE_TEMPLATE_SET_MISMATCH",
                "evaluation certificates name an unknown template",
                stage="reference_materialization.preflight",
            )
        if preparation.model_source.default_template_id not in set(ids):
            raise AutomaticReferenceMaterializationError(
                "DEFAULT_TEMPLATE_MISSING",
                "automatic default template is absent before persistence",
                stage="reference_materialization.preflight",
                template_id=preparation.model_source.default_template_id,
            )
        specifications: dict[str, ReferenceSpecificationConfig] = {}
        for template_id in ids:
            specifications[template_id] = _expected_specification(
                preparation, sources[template_id], certificates[template_id]
            )
            _validate_preparation_binding(
                preparation,
                template_id,
                specifications[template_id],
                certificates[template_id],
            )

        directory.create_references_directory()
        completed.append("references/")
        reloaded: dict[str, ReferenceSpecificationConfig] = {}
        entries: dict[str, Any] = {}
        for template_id in ids:
            lock.validate_owned(directory.resume_lock_path)
            reference_relative = f"references/{template_id}.reference.json"
            certificate_relative = f"references/{template_id}.certificate.json"
            evaluation_relative = (
                f"references/{template_id}.evaluation-certificate.json"
                if template_id in evaluation_certificates
                else None
            )
            reference_path = directory.root / reference_relative
            certificate_path = directory.root / certificate_relative
            evaluation_path = (
                None
                if evaluation_relative is None
                else directory.root / evaluation_relative
            )
            current_template = template_id
            current_path = reference_path
            _write_json(
                reference_path,
                specifications[template_id].to_dict(),
                stage="reference_materialization.reference_save",
            )
            completed.append(reference_relative)
            current_path = certificate_path
            _write_json(
                certificate_path,
                certificates[template_id],
                stage="reference_materialization.certificate_save",
            )
            completed.append(certificate_relative)
            if evaluation_path is not None:
                current_path = evaluation_path
                _write_json(
                    evaluation_path,
                    evaluation_certificates[template_id],
                    stage="reference_materialization.evaluation_certificate_save",
                )
                completed.append(str(evaluation_relative))

            current_path = reference_path
            loaded = _load_reference(reference_path)
            current_path = certificate_path
            loaded_certificate = _load_certificate(
                certificate_path,
                stage="reference_materialization.certificate_reload",
            )
            loaded_evaluation = None
            if evaluation_path is not None:
                current_path = evaluation_path
                loaded_evaluation = _load_certificate(
                    evaluation_path,
                    stage="reference_materialization.evaluation_certificate_reload",
                )
                from refsite_mlip.config import (
                    AutomaticEvaluationPolicyAuditError,
                    validate_automatic_evaluation_certificate,
                )

                try:
                    loaded_evaluation = validate_automatic_evaluation_certificate(
                        loaded_evaluation
                    )
                except AutomaticEvaluationPolicyAuditError as error:
                    raise AutomaticReferenceMaterializationError(
                        "EVALUATION_CERTIFICATE_RELOAD_CONTENT_MISMATCH",
                        "reloaded evaluation certificate failed strict validation",
                        stage="reference_materialization.evaluation_certificate_reload",
                        template_id=template_id,
                        path=evaluation_path,
                        completed_files=completed,
                        original_error=error,
                    ) from error
            if loaded.to_dict() != specifications[template_id].to_dict():
                raise AutomaticReferenceMaterializationError(
                    "REFERENCE_RELOAD_CONTENT_MISMATCH",
                    "reloaded reference specification differs from committed content",
                    stage="reference_materialization.reference_reload",
                    template_id=template_id,
                    path=reference_path,
                    completed_files=completed,
                )
            if loaded_certificate != certificates[template_id]:
                raise AutomaticReferenceMaterializationError(
                    "CERTIFICATE_RELOAD_CONTENT_MISMATCH",
                    "reloaded certificate differs from committed content",
                    stage="reference_materialization.certificate_reload",
                    template_id=template_id,
                    path=certificate_path,
                    completed_files=completed,
                )
            if loaded_evaluation != evaluation_certificates.get(template_id):
                raise AutomaticReferenceMaterializationError(
                    "EVALUATION_CERTIFICATE_RELOAD_CONTENT_MISMATCH",
                    "reloaded evaluation certificate differs from committed content",
                    stage="reference_materialization.evaluation_certificate_reload",
                    template_id=template_id,
                    path=evaluation_path,
                    completed_files=completed,
                )
            if loaded_evaluation is not None:
                policy = loaded.evaluation_policy
                if (
                    policy is None
                    or loaded_evaluation.get("status") != "qualified"
                    or loaded_evaluation.get("scope")
                    != "assigned_dataset_local_neighborhood"
                    or loaded_evaluation.get("structural_certificate_sha256")
                    != loaded_certificate["certificate_sha256"]
                    or loaded_evaluation.get("specification_sha256")
                    != loaded.content_fingerprint
                    or loaded_evaluation.get("policy_fingerprint")
                    != policy.content_fingerprint
                    or loaded_evaluation.get("radius_fingerprint")
                    != preparation.radius_config.content_fingerprint
                ):
                    raise AutomaticReferenceMaterializationError(
                        "EVALUATION_CERTIFICATE_BINDING_MISMATCH",
                        "evaluation certificate hash chain differs from persisted reference",
                        stage="reference_materialization.evaluation_certificate_reload",
                        template_id=template_id,
                        path=evaluation_path,
                        completed_files=completed,
                    )
            _validate_preparation_binding(
                preparation, template_id, loaded, loaded_certificate
            )
            reloaded[template_id] = loaded
            entries[template_id] = {
                "reference_path": reference_relative,
                "certificate_path": certificate_relative,
                "specification_fingerprint": loaded.content_fingerprint,
                "certificate_fingerprint": loaded_certificate[
                    "certificate_sha256"
                ],
                "full_template_fingerprint": loaded_certificate["template_sha256"],
                "artifact_fingerprint": loaded_certificate["artifact_sha256"],
                "phase_fingerprint": loaded_certificate[
                    "phase_specification_sha256"
                ],
                "radius_fingerprint": preparation.radius_config.content_fingerprint,
                "approval_status": "provisional",
                "evaluation_policy": None,
            }
            if loaded.evaluation_policy is not None:
                entries[template_id]["evaluation_policy"] = {
                    "content_fingerprint": loaded.evaluation_policy.content_fingerprint,
                    "status": "qualified",
                    "scope": "assigned_dataset_local_neighborhood",
                }
                entries[template_id]["evaluation_certificate_path"] = (
                    evaluation_relative
                )
                entries[template_id]["evaluation_certificate_fingerprint"] = (
                    loaded_evaluation["evaluation_certificate_sha256"]
                )
                semantic_fingerprint = loaded_evaluation.get(
                    "evaluation_semantic_fingerprint_sha256"
                )
                if semantic_fingerprint is not None:
                    entries[template_id][
                        "evaluation_certificate_semantic_fingerprint"
                    ] = semantic_fingerprint

        rebound_sources = tuple(
            replace(
                source,
                builder=reloaded[source.template_id].builder,
                phase_specification=reloaded[source.template_id].phase_specification,
                evaluation_policy=reloaded[source.template_id].evaluation_policy,
            )
            for source in preparation.model_source.reference_templates
        )
        rebound_model_source = replace(
            preparation.model_source, reference_templates=rebound_sources
        )
        rebound_config = replace(
            preparation.config, model_source=rebound_model_source
        )
        if (
            rebound_config.to_dict() != preparation.config.to_dict()
            or rebound_config.config_fingerprint != preparation.config_fingerprint
        ):
            raise AutomaticReferenceMaterializationError(
                "MATERIALIZED_CONFIG_MISMATCH",
                "strictly reloaded references changed canonical training config",
                stage="reference_materialization.rebind",
                completed_files=completed,
            )

        config_entry = next(
            (
                item
                for item in preparation.input_file_digests["files"].values()
                if item["role"] == "config"
            ),
            None,
        )
        manifest = {
            "schema_version": AUTOMATIC_REFERENCE_MATERIALIZATION_VERSION,
            "reference_resolution_mode": "materialized",
            "automatic_reference_convention_version": payload[
                "convention_version"
            ],
            "automatic_reference_fingerprint": payload["content_fingerprint"],
            "authored_recipe_sha256": (
                None if config_entry is None else config_entry["sha256"]
            ),
            "preparation_fingerprint": preparation.preparation_fingerprint,
            "default_template_id": rebound_model_source.default_template_id,
            "templates": entries,
        }
        manifest["content_fingerprint"] = _fingerprint(manifest)
        augmented = dict(payload)
        augmented["materialization"] = manifest
        rebound = replace(
            preparation,
            config=rebound_config,
            model_source=rebound_model_source,
            automatic_reference_preparation=augmented,
        )
        lock.validate_owned(directory.resume_lock_path)
        return MaterializedAutomaticReferences(
            preparation=rebound,
            manifest=manifest,
            completed_files=tuple(completed),
        )
    except AutomaticReferenceMaterializationError as error:
        if not error.completed_persistence_stages and completed:
            error.completed_persistence_stages = tuple(completed)
            error.recoverable_artifacts = tuple(completed)
        raise
    except BaseException as error:
        raise AutomaticReferenceMaterializationError(
            getattr(error, "reason_code", None)
            or "AUTOMATIC_REFERENCE_MATERIALIZATION_FAILED",
            "automatic reference persistence or strict reload failed",
            stage=getattr(error, "stage", None)
            or "reference_materialization",
            template_id=current_template,
            path=current_path,
            completed_files=completed,
            original_error=error,
        ) from error


def validate_materialized_reference_files(
    directory: TrainingRunDirectory,
    *,
    config: Any,
    status: Mapping[str, Any],
    bundle: ReferenceSiteModelBundle,
) -> None:
    """Validate persisted references against config and bundle without POSCAR."""

    from refsite_mlip.config import (
        ReferenceSpecificationConfig,
        ScratchModelSourceConfig,
    )

    materialization = status.get("reference_materialization")
    if materialization is None:
        return
    if not isinstance(materialization, Mapping):
        raise AutomaticReferenceMaterializationError(
            "INVALID_REFERENCE_MATERIALIZATION_METADATA",
            "run status materialization metadata must be a mapping",
            stage="reference_materialization.validate",
        )
    manifest = _plain(materialization)
    declared = manifest.pop("content_fingerprint", None)
    if declared != _fingerprint(manifest):
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_MATERIALIZATION_FINGERPRINT_MISMATCH",
            "run status materialization fingerprint differs from content",
            stage="reference_materialization.validate",
        )
    manifest["content_fingerprint"] = declared
    if (
        manifest.get("schema_version")
        != AUTOMATIC_REFERENCE_MATERIALIZATION_VERSION
        or manifest.get("reference_resolution_mode") != "materialized"
    ):
        raise AutomaticReferenceMaterializationError(
            "UNSUPPORTED_REFERENCE_MATERIALIZATION",
            "run status reference materialization convention is unsupported",
            stage="reference_materialization.validate",
        )
    source = getattr(config, "model_source", None)
    if not isinstance(source, ScratchModelSourceConfig):
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_MATERIALIZATION_SOURCE_MISMATCH",
            "materialized references require a scratch model source",
            stage="reference_materialization.validate",
        )
    entries = manifest.get("templates")
    if not isinstance(entries, Mapping):
        raise AutomaticReferenceMaterializationError(
            "INVALID_REFERENCE_MATERIALIZATION_METADATA",
            "materialized reference template mapping is invalid",
            stage="reference_materialization.validate",
        )
    ids = _safe_template_ids(tuple(sorted(entries)))
    source_by_id = {item.template_id: item for item in source.reference_templates}
    binding_by_id = {
        item.template_id: item for item in bundle.template_bindings
    }
    if set(ids) != set(source_by_id) or set(ids) != set(binding_by_id):
        raise AutomaticReferenceMaterializationError(
            "REFERENCE_TEMPLATE_SET_MISMATCH",
            "materialized references differ from config or initial bundle",
            stage="reference_materialization.validate",
        )
    if manifest.get("default_template_id") != bundle.default_template_id:
        raise AutomaticReferenceMaterializationError(
            "DEFAULT_TEMPLATE_MISMATCH",
            "materialized default template differs from initial bundle",
            stage="reference_materialization.validate",
        )
    for template_id in ids:
        entry = entries[template_id]
        if not isinstance(entry, Mapping):
            raise AutomaticReferenceMaterializationError(
                "INVALID_REFERENCE_MATERIALIZATION_METADATA",
                "materialized template entry must be a mapping",
                stage="reference_materialization.validate",
                template_id=template_id,
            )
        reference_path = directory.root / str(entry.get("reference_path"))
        certificate_path = directory.root / str(entry.get("certificate_path"))
        evaluation_relative = entry.get("evaluation_certificate_path")
        evaluation_path = (
            None
            if evaluation_relative is None
            else directory.root / str(evaluation_relative)
        )
        expected_reference = directory.references / f"{template_id}.reference.json"
        expected_certificate = directory.references / f"{template_id}.certificate.json"
        expected_evaluation = (
            directory.references / f"{template_id}.evaluation-certificate.json"
        )
        if (
            reference_path != expected_reference
            or certificate_path != expected_certificate
            or (evaluation_path is not None and evaluation_path != expected_evaluation)
        ):
            raise AutomaticReferenceMaterializationError(
                "REFERENCE_PATH_MISMATCH",
                "materialized reference path is not the canonical run-relative path",
                stage="reference_materialization.validate",
                template_id=template_id,
            )
        specification = _load_reference(reference_path)
        certificate = _load_certificate(
            certificate_path,
            stage="reference_materialization.certificate_validate",
        )
        evaluation_certificate = (
            None
            if evaluation_path is None
            else _load_certificate(
                evaluation_path,
                stage="reference_materialization.evaluation_certificate_validate",
            )
        )
        declared_certificate = certificate.get("certificate_sha256")
        certificate_content = dict(certificate)
        certificate_content.pop("certificate_sha256", None)
        if declared_certificate != _fingerprint(certificate_content):
            raise AutomaticReferenceMaterializationError(
                "REFERENCE_CERTIFICATE_FINGERPRINT_MISMATCH",
                "persisted certificate differs from its declared fingerprint",
                stage="reference_materialization.validate",
                template_id=template_id,
                path=certificate_path,
            )
        source_item = source_by_id[template_id]
        binding = binding_by_id[template_id]
        expected_specification = ReferenceSpecificationConfig(
            builder=source_item.builder,
            phase_specification=source_item.phase_specification,
            evaluation_policy=source_item.evaluation_policy,
            species_alignment_weights=source.species_alignment_weights,
            poscar_sha256=certificate["poscar_content_sha256"],
        )
        comparisons = (
            (specification.to_dict(), expected_specification.to_dict()),
            (entry.get("specification_fingerprint"), specification.content_fingerprint),
            (entry.get("certificate_fingerprint"), declared_certificate),
            (entry.get("full_template_fingerprint"), binding.full_template_fingerprint),
            (
                entry.get("artifact_fingerprint"),
                binding.structural_artifact.structural_fingerprint,
            ),
            (
                entry.get("phase_fingerprint"),
                _phase_fingerprint(binding.phase_specification),
            ),
            (entry.get("radius_fingerprint"), config.radii.content_fingerprint),
            (certificate.get("specification_sha256"), specification.content_fingerprint),
            (certificate.get("template_sha256"), binding.full_template_fingerprint),
            (
                certificate.get("artifact_sha256"),
                binding.structural_artifact.structural_fingerprint,
            ),
        )
        if any(left != right for left, right in comparisons):
            raise AutomaticReferenceMaterializationError(
                "MATERIALIZED_REFERENCE_FINGERPRINT_MISMATCH",
                "persisted reference differs from config or initial bundle",
                stage="reference_materialization.validate",
                template_id=template_id,
            )
        if specification.phase_specification.approval_status != "provisional":
            raise AutomaticReferenceMaterializationError(
                "AUTOMATIC_REFERENCE_POLICY_MISMATCH",
                "automatic reference phase/policy contract changed",
                stage="reference_materialization.validate",
                template_id=template_id,
            )
        policy = specification.evaluation_policy
        if evaluation_certificate is None:
            if policy is not None or binding.evaluation_policy is not None:
                raise AutomaticReferenceMaterializationError(
                    "AUTOMATIC_REFERENCE_POLICY_MISMATCH",
                    "sinkhorn-only materialized reference acquired an EvaluationPolicy",
                    stage="reference_materialization.validate",
                    template_id=template_id,
                )
        else:
            from refsite_mlip.config import (
                AutomaticEvaluationPolicyAuditError,
                validate_automatic_evaluation_certificate,
            )

            try:
                evaluation_certificate = validate_automatic_evaluation_certificate(
                    evaluation_certificate
                )
            except AutomaticEvaluationPolicyAuditError as error:
                raise AutomaticReferenceMaterializationError(
                    "EVALUATION_CERTIFICATE_BINDING_MISMATCH",
                    "persisted evaluation certificate failed strict validation",
                    stage="reference_materialization.validate",
                    template_id=template_id,
                    path=evaluation_path,
                    original_error=error,
                ) from error
            declared_evaluation = evaluation_certificate[
                "evaluation_certificate_sha256"
            ]
            semantic_evaluation = evaluation_certificate.get(
                "evaluation_semantic_fingerprint_sha256"
            )
            if (
                policy is None
                or binding.evaluation_policy is None
                or policy.content_fingerprint
                != binding.evaluation_policy.content_fingerprint
                or entry.get("evaluation_certificate_fingerprint")
                != declared_evaluation
                or entry.get(
                    "evaluation_certificate_semantic_fingerprint"
                )
                != semantic_evaluation
                or evaluation_certificate.get("status") != "qualified"
                or evaluation_certificate.get("scope")
                != "assigned_dataset_local_neighborhood"
                or evaluation_certificate.get("structural_certificate_sha256")
                != declared_certificate
                or evaluation_certificate.get("specification_sha256")
                != specification.content_fingerprint
                or evaluation_certificate.get("policy_fingerprint")
                != policy.content_fingerprint
                or evaluation_certificate.get("radius_fingerprint")
                != config.radii.content_fingerprint
            ):
                raise AutomaticReferenceMaterializationError(
                    "EVALUATION_CERTIFICATE_BINDING_MISMATCH",
                    "persisted evaluation certificate differs from reference/bundle binding",
                    stage="reference_materialization.validate",
                    template_id=template_id,
                    path=evaluation_path,
                )


__all__ = [
    "AUTOMATIC_REFERENCE_MATERIALIZATION_VERSION",
    "AutomaticReferenceMaterializationError",
    "MaterializedAutomaticReferences",
    "materialize_automatic_references",
    "validate_materialized_reference_files",
]
