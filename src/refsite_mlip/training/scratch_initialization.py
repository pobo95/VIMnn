"""Deterministic, side-effect-isolated scratch model initialization.

This module connects a fully validated :class:`ScratchTrainingPreparation` to
the existing potential and portable-bundle APIs.  It intentionally owns no
optimizer, baseline fitting, forward, loss, checkpoint, or run-directory
logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
import random
import threading
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np
import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceStructureArtifact,
    assemble_reference_template_from_artifact,
)
from refsite_mlip.models import (
    EvaluationPolicy,
    ModelBundleTemplateBinding,
    PotentialConfig,
    ReferenceSiteModelBundle,
    ReferenceSitePotential,
    TemplateExecutionContext,
    capture_reference_site_model_bundle,
    instantiate_reference_site_model_bundle,
)

from .scratch_preparation import ScratchTrainingPreparation


SCRATCH_MODEL_INITIALIZATION_CONVENTION_VERSION = (
    "scratch_model_initialization_v1"
)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}
_INITIALIZATION_LOCK = threading.RLock()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("initialization metadata mapping keys must be strings")
        return {
            key: _plain(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("initialization metadata contains NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(
        "initialization metadata contains non-plain "
        f"{type(value).__name__}"
    )


def _freeze_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("initialization metadata mapping keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_plain(value[key])
                for key in sorted(value)
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_plain(item) for item in value)
    if type(value) is float and not math.isfinite(value):
        raise ValueError("initialization metadata contains NaN or Infinity")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(
        "initialization metadata contains non-plain "
        f"{type(value).__name__}"
    )


def _sha256(value: str, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _hash_text(digest: Any, value: Any) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _model_state_fingerprint(
    state: Mapping[str, torch.Tensor], keys: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "scratch_initial_model_state_v1")
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        _hash_text(digest, key)
        _hash_text(digest, tensor.dtype)
        _hash_text(digest, tuple(tensor.shape))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _phase_specification_fingerprint(value: PhaseSpecification) -> str:
    payload = {
        "scope": "reference_site_phase_specification_inspection_v1",
        "value": value.to_dict(),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.numel() == 0 or right.numel() == 0:
        return False
    return (
        left.untyped_storage().data_ptr()
        == right.untyped_storage().data_ptr()
    )


class ScratchModelInitializationError(RuntimeError):
    """Structured scratch initialization/capture/reconstruction failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        initialization_seed: int | None = None,
        template_id: str | None = None,
        config_fingerprint: str | None = None,
        template_fingerprint: str | None = None,
        bundle_fingerprint: str | None = None,
        original_reason_code: str | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        if type(reason_code) is not str or not reason_code:
            raise ValueError("reason_code must be a nonempty string")
        if type(message) is not str or not message:
            raise ValueError("message must be a nonempty string")
        if type(stage) is not str or not stage:
            raise ValueError("stage must be a nonempty string")
        self.reason_code = reason_code
        self.message = message
        self.stage = stage
        self.initialization_seed = initialization_seed
        self.template_id = template_id
        self.config_fingerprint = config_fingerprint
        self.template_fingerprint = template_fingerprint
        self.bundle_fingerprint = bundle_fingerprint
        self.original_reason_code = original_reason_code
        self.original_error = original_error
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        context = []
        for name in (
            "initialization_seed",
            "template_id",
            "config_fingerprint",
            "template_fingerprint",
            "bundle_fingerprint",
            "original_reason_code",
            "original_exception_type",
            "original_exception_message",
        ):
            value = getattr(self, name)
            if value is not None:
                context.append(f"{name}={value!r}")
        suffix = "" if not context else " " + " ".join(context)
        super().__init__(
            f"[{reason_code}] stage={stage!r}{suffix} {message}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_fingerprint": self.bundle_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "initialization_seed": self.initialization_seed,
            "message": self.message,
            "original_exception_message": self.original_exception_message,
            "original_exception_type": self.original_exception_type,
            "original_reason_code": self.original_reason_code,
            "reason_code": self.reason_code,
            "stage": self.stage,
            "template_fingerprint": self.template_fingerprint,
            "template_id": self.template_id,
        }


def _error(
    reason_code: str,
    message: str,
    *,
    stage: str,
    preparation: ScratchTrainingPreparation | None = None,
    initialization_seed: int | None = None,
    template_id: str | None = None,
    template_fingerprint: str | None = None,
    bundle_fingerprint: str | None = None,
    original_error: BaseException | None = None,
) -> ScratchModelInitializationError:
    return ScratchModelInitializationError(
        reason_code,
        message,
        stage=stage,
        initialization_seed=(
            initialization_seed
            if initialization_seed is not None
            else (
                None
                if preparation is None
                else getattr(
                    getattr(preparation, "model_source", None),
                    "initialization_seed",
                    None,
                )
            )
        ),
        template_id=template_id,
        config_fingerprint=(
            None
            if preparation is None
            else getattr(preparation, "config_fingerprint", None)
        ),
        template_fingerprint=template_fingerprint,
        bundle_fingerprint=bundle_fingerprint,
        original_reason_code=getattr(original_error, "reason_code", None),
        original_error=original_error,
    )


@contextmanager
def _isolated_cpu_initialization(
    seed: int, dtype: torch.dtype
) -> Iterator[None]:
    """Temporarily own only the process state needed by implicit initializers."""

    with _INITIALIZATION_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        numpy_state = (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
        default_dtype = torch.get_default_dtype()
        try:
            # The bundle reconstruction path already uses a CPU-only fork.
            # Seed the CPU generator directly: torch.manual_seed would also
            # alter CUDA (including lazy CUDA seed callbacks).
            with torch.random.fork_rng(devices=[]):
                import_e3nn_0_4_4()
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.random.default_generator.manual_seed(seed % (2**64))
                torch.set_default_dtype(dtype)
                try:
                    with (
                        torch.device("cpu"),
                        torch.inference_mode(False),
                        torch.enable_grad(),
                    ):
                        yield
                finally:
                    torch.set_default_dtype(default_dtype)
        finally:
            # Keep restoration unconditional on constructor/capture failures.
            if torch.get_default_dtype() != default_dtype:
                torch.set_default_dtype(default_dtype)
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def _template_fingerprint(
    preparation: ScratchTrainingPreparation, template_id: str
) -> str | None:
    values = preparation.template_fingerprints.get(template_id)
    if isinstance(values, Mapping):
        value = values.get("full_template_fingerprint")
        if type(value) is str:
            return value
    return None


def _validate_preparation(
    preparation: ScratchTrainingPreparation,
) -> tuple[
    PotentialConfig,
    torch.dtype,
    tuple[str, ...],
    dict[str, PhaseSpecification],
    dict[str, EvaluationPolicy],
]:
    source = getattr(preparation, "model_source", None)
    seed = getattr(source, "initialization_seed", None)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or int(seed) != seed
    ):
        raise _error(
            "INVALID_PREPARATION",
            "scratch initialization seed is not a canonical integer",
            stage="initialization.preparation",
            preparation=preparation,
        )
    if preparation.training_executed or preparation.scratch_execution_implemented:
        raise _error(
            "INVALID_PREPARATION",
            "scratch preparation execution flags are invalid",
            stage="initialization.preparation",
            preparation=preparation,
        )
    try:
        if preparation.config.config_fingerprint != preparation.config_fingerprint:
            raise ValueError("prepared config fingerprint no longer matches content")
        config_source = getattr(preparation.config, "model_source", None)
        if (
            config_source is None
            or config_source.to_dict() != source.to_dict()
        ):
            raise ValueError(
                "prepared model source differs from the fingerprinted config"
            )
        if preparation.config.runtime.to_dict() != preparation.runtime.to_dict():
            raise ValueError(
                "prepared runtime differs from the fingerprinted config"
            )
        if preparation.config.data.to_dict() != preparation.data.to_dict():
            raise ValueError(
                "prepared data config differs from the fingerprinted config"
            )
        if preparation.config.radii.to_dict() != preparation.radius_config.to_dict():
            raise ValueError(
                "prepared radius config differs from the fingerprinted config"
            )
        config = PotentialConfig.from_dict(source.potential.to_dict())
        # Import lazily to avoid a config/training package initialization cycle.
        from refsite_mlip.config.radii import validate_radius_model_compatibility

        validate_radius_model_compatibility(preparation.radius_config, config)
    except Exception as error:
        raise _error(
            "INVALID_PREPARATION",
            "scratch preparation/config content is invalid",
            stage="initialization.preparation",
            preparation=preparation,
            original_error=error,
        ) from error
    dtype = _DTYPES.get(preparation.resolved_dtype)
    if dtype is None or preparation.runtime.dtype != preparation.resolved_dtype:
        raise _error(
            "INVALID_PREPARATION",
            "prepared runtime dtype is inconsistent or unsupported",
            stage="initialization.preparation",
            preparation=preparation,
        )

    sources = {item.template_id: item for item in source.reference_templates}
    template_ids = tuple(sorted(sources))
    default_id = source.default_template_id
    required_mappings = (
        preparation.structural_artifacts,
        preparation.template_contexts,
        preparation.evaluation_policies,
        preparation.template_fingerprints,
    )
    if (
        default_id not in sources
        or any(default_id not in values for values in required_mappings)
        or default_id not in preparation.registry
    ):
        raise _error(
            "DEFAULT_TEMPLATE_MISSING",
            "default template is absent from prepared scratch bindings",
            stage="initialization.default_template",
            preparation=preparation,
            template_id=default_id,
        )
    expected_ids = set(template_ids)
    if (
        any(set(values) != expected_ids for values in required_mappings)
        or len(preparation.registry) != len(expected_ids)
    ):
        raise _error(
            "INVALID_PREPARATION",
            "prepared template mapping key sets differ",
            stage="initialization.preparation",
            preparation=preparation,
            template_id=default_id,
        )

    configured_species = tuple(config.species_vocabulary)
    if tuple(preparation.species_vocabulary) != configured_species:
        raise _error(
            "SPECIES_MISMATCH",
            "prepared and PotentialConfig species vocabularies differ",
            stage="initialization.species",
            preparation=preparation,
            template_id=default_id,
        )
    site_vocabulary = tuple(config.feature.site_type_vocabulary or ())
    phases: dict[str, PhaseSpecification] = {}
    policies: dict[str, EvaluationPolicy] = {}
    for template_id in template_ids:
        template_source = sources[template_id]
        fingerprint = _template_fingerprint(preparation, template_id)
        artifact = preparation.structural_artifacts[template_id]
        context = preparation.template_contexts[template_id]
        if not isinstance(artifact, ReferenceStructureArtifact) or not isinstance(
            context, TemplateExecutionContext
        ):
            raise _error(
                "INVALID_PREPARATION",
                "prepared artifact/context has an invalid runtime type",
                stage="initialization.preparation",
                preparation=preparation,
                template_id=template_id,
                template_fingerprint=fingerprint,
            )
        try:
            from refsite_mlip.config.radii import (
                validate_radius_artifact_compatibility,
            )

            validate_radius_artifact_compatibility(
                preparation.radius_config, artifact
            )
        except Exception as error:
            raise _error(
                "INVALID_PREPARATION",
                "prepared artifact is incompatible with the radius config",
                stage="initialization.radius",
                preparation=preparation,
                template_id=template_id,
                template_fingerprint=fingerprint,
                original_error=error,
            ) from error
        if tuple(artifact.species_vocabulary) != configured_species or tuple(
            context.supported_species
        ) != configured_species:
            raise _error(
                "SPECIES_MISMATCH",
                "template species vocabulary differs from the scratch model",
                stage="initialization.species",
                preparation=preparation,
                template_id=template_id,
                template_fingerprint=fingerprint,
            )
        if tuple(template_source.builder.site_type_ids) != site_vocabulary:
            raise _error(
                "SITE_TYPE_MISMATCH",
                "builder site-type vocabulary differs from PotentialConfig",
                stage="initialization.site_type",
                preparation=preparation,
                template_id=template_id,
                template_fingerprint=fingerprint,
            )
        site_types = artifact.site_types
        if site_types.numel() and (
            int(site_types.min()) < 0
            or int(site_types.max()) >= len(site_vocabulary)
        ):
            raise _error(
                "SITE_TYPE_MISMATCH",
                "structural artifact does not realize the global site-type vocabulary",
                stage="initialization.site_type",
                preparation=preparation,
                template_id=template_id,
                template_fingerprint=fingerprint,
            )
        try:
            artifact.validate()
            phase = PhaseSpecification.from_dict(
                template_source.phase_specification.to_dict()
            )
            template = assemble_reference_template_from_artifact(
                artifact, phase_specification=phase
            )
            registered = preparation.registry.resolve(template_id)
            context.validate_fingerprint()
            metadata = preparation.template_fingerprints[template_id]
            if not isinstance(metadata, Mapping):
                raise TypeError("prepared template fingerprints must be a mapping")
            if (
                fingerprint is None
                or template.fingerprint != fingerprint
                or registered.fingerprint != fingerprint
                or context.fingerprint != fingerprint
                or metadata.get("structural_artifact_fingerprint")
                != artifact.structural_fingerprint
                or metadata.get("phase_specification_fingerprint")
                != _phase_specification_fingerprint(phase)
                or metadata.get("num_sites") != artifact.diagnostics.num_sites
                or not math.isclose(
                    context.avg_num_neighbors,
                    artifact.avg_num_neighbors,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError("prepared template content/fingerprints differ")
        except Exception as error:
            raise _error(
                "PHASE_MISMATCH",
                "prepared artifact, phase, registry, or context is inconsistent",
                stage="initialization.phase",
                preparation=preparation,
                template_id=template_id,
                template_fingerprint=fingerprint,
                original_error=error,
            ) from error
        phases[template_id] = phase

        source_policy = template_source.evaluation_policy
        prepared_policy = preparation.evaluation_policies[template_id]
        try:
            if (source_policy is None) != (prepared_policy is None):
                raise ValueError("policy presence differs between source and preparation")
            if prepared_policy is not None:
                prepared_policy.validate_fingerprint()
                if (
                    prepared_policy.template_id != template_id
                    or prepared_policy.template_fingerprint != fingerprint
                    or source_policy is None
                    or source_policy.content_fingerprint
                    != prepared_policy.content_fingerprint
                ):
                    raise ValueError("evaluation policy binding differs")
                policies[template_id] = EvaluationPolicy.from_dict(
                    prepared_policy.to_dict()
                )
            canonical_policy = policies.get(template_id)
            expected_policy_fingerprint = (
                None
                if canonical_policy is None
                else canonical_policy.content_fingerprint
            )
            metadata = preparation.template_fingerprints[template_id]
            binding = ModelBundleTemplateBinding(
                template_id=template_id,
                structural_artifact=artifact,
                phase_specification=phase,
                full_template_fingerprint=template.fingerprint,
                evaluation_policy=canonical_policy,
                approval_status=phase.approval_status,
                provenance={"source_kind": "scratch_preflight"},
            )
            if (
                metadata.get("evaluation_policy_present")
                is not (canonical_policy is not None)
                or metadata.get("evaluation_policy_fingerprint")
                != expected_policy_fingerprint
                or metadata.get("binding_fingerprint")
                != binding.binding_fingerprint
            ):
                raise ValueError("prepared policy/binding fingerprints differ")
        except Exception as error:
            raise _error(
                "POLICY_MISMATCH",
                "prepared evaluation policy is inconsistent with its template",
                stage="initialization.policy",
                preparation=preparation,
                template_id=template_id,
                template_fingerprint=fingerprint,
                original_error=error,
            ) from error
    return config, dtype, template_ids, phases, policies


def _validate_initial_state(
    model: ReferenceSitePotential,
    *,
    dtype: torch.dtype,
    preparation: ScratchTrainingPreparation,
    default_template_id: str,
    default_template_fingerprint: str | None,
) -> tuple[
    tuple[tuple[str, torch.nn.Parameter], ...],
    tuple[tuple[str, torch.Tensor], ...],
]:
    state = model.state_dict()
    for key, value in state.items():
        if value.device.type != "cpu" or (
            value.is_floating_point() and value.dtype != dtype
        ):
            raise _error(
                "MODEL_INITIALIZATION_FAILED",
                "initial model state is not canonical CPU/configured dtype",
                stage="initialization.model_state",
                preparation=preparation,
                template_id=default_template_id,
                template_fingerprint=default_template_fingerprint,
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.all(torch.isfinite(value))
        ):
            raise _error(
                "NONFINITE_INITIAL_STATE",
                f"initial model state tensor {key!r} is nonfinite",
                stage="initialization.model_state",
                preparation=preparation,
                template_id=default_template_id,
                template_fingerprint=default_template_fingerprint,
            )
    parameters = tuple(model.named_parameters())
    unfiltered_parameters = tuple(model.named_parameters(remove_duplicate=False))
    if (
        len(parameters) != len(unfiltered_parameters)
        or len({id(value) for _, value in parameters}) != len(parameters)
        or any(
            not value.requires_grad
            or value.grad is not None
            or value.is_inference()
            for _, value in parameters
        )
    ):
        raise _error(
            "MODEL_INITIALIZATION_FAILED",
            "initial model parameter identity/trainability contract is invalid",
            stage="initialization.model_state",
            preparation=preparation,
            template_id=default_template_id,
            template_fingerprint=default_template_fingerprint,
        )
    buffers = tuple(model.named_buffers())
    buffer_map = dict(buffers)
    parameter_map = dict(parameters)
    baseline = buffer_map.get("atomic_baseline")
    expected_shape = (len(preparation.species_vocabulary),)
    if (
        baseline is None
        or "atomic_baseline" in parameter_map
        or baseline.shape != expected_shape
        or baseline.device.type != "cpu"
        or baseline.dtype != dtype
        or baseline.requires_grad
        or not bool(torch.all(torch.isfinite(baseline)))
        or int(torch.count_nonzero(baseline)) != 0
    ):
        raise _error(
            "NONZERO_INITIAL_BASELINE",
            "scratch atomic baseline must be an exact-zero frozen model buffer",
            stage="initialization.baseline",
            preparation=preparation,
            template_id=default_template_id,
            template_fingerprint=default_template_fingerprint,
        )
    return parameters, buffers


def _elements(values: Sequence[tuple[str, torch.Tensor]]) -> int:
    return sum(int(value.numel()) for _, value in values)


def _bytes(values: Sequence[tuple[str, torch.Tensor]]) -> int:
    return sum(
        int(value.numel()) * int(value.element_size()) for _, value in values
    )


def _bundle_template_fingerprints(
    bundle: ReferenceSiteModelBundle,
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for binding in bundle.template_bindings:
        values[binding.template_id] = {
            "binding_fingerprint": binding.binding_fingerprint,
            "evaluation_policy_fingerprint": (
                None
                if binding.evaluation_policy is None
                else binding.evaluation_policy.content_fingerprint
            ),
            "full_template_fingerprint": binding.full_template_fingerprint,
            "phase_specification_fingerprint": (
                _phase_specification_fingerprint(binding.phase_specification)
            ),
            "structural_artifact_fingerprint": (
                binding.structural_artifact.structural_fingerprint
            ),
        }
    return values


@dataclass(frozen=True)
class ScratchModelInitialization:
    """Owned in-memory initial bundle plus plain deterministic diagnostics."""

    bundle: ReferenceSiteModelBundle
    initialization_seed: int
    effective_potential_config: PotentialConfig
    architecture_fingerprint: str
    model_state_fingerprint: str
    bundle_fingerprint: str
    parameter_tensor_count: int
    parameter_element_count: int
    parameter_byte_count: int
    buffer_tensor_count: int
    buffer_element_count: int
    buffer_byte_count: int
    template_ids: tuple[str, ...]
    template_fingerprints: Mapping[str, Mapping[str, Any]]
    default_template_id: str
    species_vocabulary: tuple[int, ...]
    baseline_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, ReferenceSiteModelBundle):
            raise TypeError("bundle must be a ReferenceSiteModelBundle")
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, Integral
        ):
            raise TypeError("initialization_seed must be an integer")
        object.__setattr__(self, "initialization_seed", int(self.initialization_seed))
        if not isinstance(self.effective_potential_config, PotentialConfig):
            raise TypeError("effective_potential_config must be PotentialConfig")
        for name in (
            "architecture_fingerprint",
            "model_state_fingerprint",
            "bundle_fingerprint",
        ):
            _sha256(getattr(self, name), name=name)
        if self.bundle.architecture_fingerprint != self.architecture_fingerprint:
            raise ValueError("bundle architecture fingerprint differs from result")
        if self.bundle.bundle_fingerprint != self.bundle_fingerprint:
            raise ValueError("bundle fingerprint differs from result")
        if (
            self.effective_potential_config.to_dict()
            != PotentialConfig.from_dict(self.bundle.model_config).to_dict()
        ):
            raise ValueError("effective PotentialConfig differs from bundle")
        if self.model_state_fingerprint != _model_state_fingerprint(
            self.bundle.model_state, self.bundle.model_state_keys
        ):
            raise ValueError("model state fingerprint differs from bundle state")
        for name in (
            "parameter_tensor_count",
            "parameter_element_count",
            "parameter_byte_count",
            "buffer_tensor_count",
            "buffer_element_count",
            "buffer_byte_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if (
            self.parameter_tensor_count + self.buffer_tensor_count
            != len(self.bundle.model_state)
            or self.parameter_element_count + self.buffer_element_count
            != sum(int(value.numel()) for value in self.bundle.model_state.values())
            or self.parameter_byte_count + self.buffer_byte_count
            != sum(
                int(value.numel()) * int(value.element_size())
                for value in self.bundle.model_state.values()
            )
        ):
            raise ValueError("parameter/buffer totals differ from bundle state")
        template_ids = tuple(self.template_ids)
        if (
            not template_ids
            or template_ids != tuple(sorted(template_ids))
            or len(set(template_ids)) != len(template_ids)
            or set(template_ids) != set(self.template_fingerprints)
            or template_ids != self.bundle.binding_ids
            or self.default_template_id != self.bundle.default_template_id
        ):
            raise ValueError("template ID/fingerprint metadata is inconsistent")
        object.__setattr__(self, "template_ids", template_ids)
        object.__setattr__(
            self,
            "template_fingerprints",
            _freeze_plain(self.template_fingerprints),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in self.species_vocabulary
        ):
            raise TypeError("species_vocabulary must contain only integers")
        species = tuple(int(value) for value in self.species_vocabulary)
        if species != tuple(self.bundle.species_vocabulary):
            raise ValueError("result and bundle species vocabularies differ")
        object.__setattr__(self, "species_vocabulary", species)
        object.__setattr__(
            self, "baseline_metadata", _freeze_plain(self.baseline_metadata)
        )
        self.bundle.validate()
        if _plain(self.template_fingerprints) != _bundle_template_fingerprints(
            self.bundle
        ):
            raise ValueError("template fingerprint metadata differs from bundle")
        baseline = self.bundle.model_state.get("atomic_baseline")
        expected_baseline = {
            "buffer_name": "atomic_baseline",
            "device": "cpu",
            "dtype": self.bundle.model_floating_dtype,
            "exact_zero": (
                baseline is not None
                and int(torch.count_nonzero(baseline)) == 0
            ),
            "is_parameter": False,
            "requires_grad": False,
            "shape": [] if baseline is None else list(baseline.shape),
        }
        if _plain(self.baseline_metadata) != expected_baseline:
            raise ValueError("baseline metadata differs from bundle state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_fingerprint": self.architecture_fingerprint,
            "baseline": _plain(self.baseline_metadata),
            "bundle_fingerprint": self.bundle_fingerprint,
            "bundle_in_memory": True,
            "convention_version": SCRATCH_MODEL_INITIALIZATION_CONVENTION_VERSION,
            "default_template_id": self.default_template_id,
            "device": "cpu",
            "dtype": self.bundle.model_floating_dtype,
            "effective_potential_config": self.effective_potential_config.to_dict(),
            "initialization_seed": self.initialization_seed,
            "model_state_fingerprint": self.model_state_fingerprint,
            "state": {
                "buffer_byte_count": self.buffer_byte_count,
                "buffer_element_count": self.buffer_element_count,
                "buffer_tensor_count": self.buffer_tensor_count,
                "parameter_byte_count": self.parameter_byte_count,
                "parameter_element_count": self.parameter_element_count,
                "parameter_tensor_count": self.parameter_tensor_count,
            },
            "species_vocabulary": list(self.species_vocabulary),
            "template_fingerprints": _plain(self.template_fingerprints),
            "template_ids": list(self.template_ids),
            "training_state_included": False,
        }


def _initialize_scratch_model(
    preparation: ScratchTrainingPreparation,
) -> ScratchModelInitialization:
    """Create and verify one deterministic CPU-owned portable initial bundle."""

    if not isinstance(preparation, ScratchTrainingPreparation):
        raise _error(
            "INVALID_PREPARATION",
            "preparation must be a ScratchTrainingPreparation",
            stage="initialization.preparation",
        )
    source = getattr(preparation, "model_source", None)
    seed = getattr(source, "initialization_seed", None)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise _error(
            "INVALID_PREPARATION",
            "preparation initialization seed must be an integer",
            stage="initialization.preparation",
            preparation=preparation,
        )
    dtype = _DTYPES.get(preparation.resolved_dtype)
    if dtype is None:
        raise _error(
            "INVALID_PREPARATION",
            "preparation dtype must be float32 or float64",
            stage="initialization.preparation",
            preparation=preparation,
        )

    with _isolated_cpu_initialization(int(seed), dtype):
        before = preparation.to_dict()
        config, dtype, template_ids, phases, policies = _validate_preparation(
            preparation
        )
        default_id = preparation.model_source.default_template_id
        default_fingerprint = _template_fingerprint(preparation, default_id)
        try:
            # Preparation validation is intentionally RNG-free, but reset at
            # the exact construction boundary so the seed denotes parameter
            # initialization rather than any preceding validation work.
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            torch.random.default_generator.manual_seed(int(seed) % (2**64))
            default_template = assemble_reference_template_from_artifact(
                preparation.structural_artifacts[default_id],
                phase_specification=phases[default_id],
            )
            species_alignment = torch.tensor(
                preparation.model_source.species_alignment_weights,
                dtype=dtype,
                device="cpu",
            )
            model = ReferenceSitePotential(
                config,
                default_template.topology,
                default_template.phase_modes,
                default_template.phase_mode_weights,
                species_alignment,
                default_template.site_alignment_weights,
                default_template.phase_channel_weights,
                atomic_baseline=torch.zeros(
                    len(config.species_vocabulary), dtype=dtype, device="cpu"
                ),
            ).to(device="cpu", dtype=dtype)
            model.eval()
        except ScratchModelInitializationError:
            raise
        except Exception as error:
            raise _error(
                "MODEL_INITIALIZATION_FAILED",
                "ReferenceSitePotential scratch initialization failed",
                stage="initialization.model",
                preparation=preparation,
                template_id=default_id,
                template_fingerprint=default_fingerprint,
                original_error=error,
            ) from error

        parameters, buffers = _validate_initial_state(
            model,
            dtype=dtype,
            preparation=preparation,
            default_template_id=default_id,
            default_template_fingerprint=default_fingerprint,
        )
        model_state = model.state_dict()
        state_keys = tuple(model_state)
        state_fingerprint = _model_state_fingerprint(model_state, state_keys)
        provenance = {
            "atomic_baseline_initialization": "exact_zero",
            "canonical_device": "cpu",
            "initialization_convention_version": (
                SCRATCH_MODEL_INITIALIZATION_CONVENTION_VERSION
            ),
            "initialization_seed": int(seed),
            "model_floating_dtype": preparation.resolved_dtype,
            "source_kind": "scratch",
        }
        try:
            bundle = capture_reference_site_model_bundle(
                model=model,
                structural_artifacts=preparation.structural_artifacts,
                phase_specifications=phases,
                evaluation_policies=policies or None,
                default_template_id=default_id,
                provenance=provenance,
            )
            if tuple(bundle.model_state_keys) != state_keys:
                raise ValueError("captured model state keys differ")
            if _model_state_fingerprint(
                bundle.model_state, bundle.model_state_keys
            ) != state_fingerprint:
                raise ValueError("captured model state values differ")
            for key in state_keys:
                if _shares_storage(model_state[key], bundle.model_state[key]):
                    raise ValueError(
                        f"captured model state aliases runtime tensor {key!r}"
                    )
        except Exception as error:
            raise _error(
                "BUNDLE_CAPTURE_FAILED",
                "portable initial model bundle capture failed",
                stage="initialization.bundle_capture",
                preparation=preparation,
                template_id=default_id,
                template_fingerprint=default_fingerprint,
                original_error=error,
            ) from error
        try:
            reconstructed = instantiate_reference_site_model_bundle(
                bundle, device="cpu", dtype=dtype
            )
            reconstructed_state = reconstructed.model.state_dict()
            if tuple(reconstructed_state) != tuple(bundle.model_state_keys):
                raise ValueError("reconstructed model state keys differ")
            for key in bundle.model_state_keys:
                if not torch.equal(
                    reconstructed_state[key], bundle.model_state[key]
                ):
                    raise ValueError(
                        f"reconstructed model state differs at {key!r}"
                    )
                if _shares_storage(
                    reconstructed_state[key], bundle.model_state[key]
                ):
                    raise ValueError(
                        f"reconstructed model state aliases bundle tensor {key!r}"
                    )
            if reconstructed.model.config.to_dict() != config.to_dict():
                raise ValueError("reconstructed PotentialConfig differs")
            reconstructed_parameters, reconstructed_buffers = (
                _validate_initial_state(
                    reconstructed.model,
                    dtype=dtype,
                    preparation=preparation,
                    default_template_id=default_id,
                    default_template_fingerprint=default_fingerprint,
                )
            )
            if (
                tuple(name for name, _ in reconstructed_parameters)
                != tuple(name for name, _ in parameters)
                or tuple(name for name, _ in reconstructed_buffers)
                != tuple(name for name, _ in buffers)
            ):
                raise ValueError("reconstructed parameter/buffer contract differs")
            if reconstructed.default_template_id != default_id or tuple(
                sorted(reconstructed.template_contexts)
            ) != template_ids:
                raise ValueError("reconstructed template bindings differ")
            for template_id in template_ids:
                expected = _template_fingerprint(preparation, template_id)
                context = reconstructed.template_contexts[template_id]
                context.validate_fingerprint()
                if (
                    reconstructed.template_fingerprints[template_id] != expected
                    or context.fingerprint != expected
                    or reconstructed.structural_fingerprints[template_id]
                    != preparation.structural_artifacts[
                        template_id
                    ].structural_fingerprint
                ):
                    raise ValueError(
                        "reconstructed template/artifact fingerprints differ"
                    )
            if tuple(sorted(reconstructed.evaluation_policies)) != tuple(
                sorted(policies)
            ):
                raise ValueError("reconstructed evaluation policies differ")
            for template_id, policy in policies.items():
                rebuilt = reconstructed.evaluation_policies[template_id]
                rebuilt.validate_fingerprint()
                if rebuilt.to_dict() != policy.to_dict():
                    raise ValueError(
                        "reconstructed evaluation policy content differs"
                    )
        except Exception as error:
            raise _error(
                "BUNDLE_RECONSTRUCTION_FAILED",
                "portable initial bundle reconstruction/state parity failed",
                stage="initialization.bundle_reconstruction",
                preparation=preparation,
                template_id=default_id,
                template_fingerprint=default_fingerprint,
                bundle_fingerprint=bundle.bundle_fingerprint,
                original_error=error,
            ) from error

        if preparation.to_dict() != before:
            raise _error(
                "INVALID_PREPARATION",
                "scratch preparation changed during initialization",
                stage="initialization.ownership",
                preparation=preparation,
                template_id=default_id,
                template_fingerprint=default_fingerprint,
                bundle_fingerprint=bundle.bundle_fingerprint,
            )
        bundle_fingerprint = bundle.bundle_fingerprint or ""
        architecture_fingerprint = bundle.architecture_fingerprint or ""
        return ScratchModelInitialization(
            bundle=bundle,
            initialization_seed=int(seed),
            effective_potential_config=PotentialConfig.from_dict(config.to_dict()),
            architecture_fingerprint=architecture_fingerprint,
            model_state_fingerprint=state_fingerprint,
            bundle_fingerprint=bundle_fingerprint,
            parameter_tensor_count=len(parameters),
            parameter_element_count=_elements(parameters),
            parameter_byte_count=_bytes(parameters),
            buffer_tensor_count=len(buffers),
            buffer_element_count=_elements(buffers),
            buffer_byte_count=_bytes(buffers),
            template_ids=template_ids,
            template_fingerprints=_bundle_template_fingerprints(
                bundle
            ),
            default_template_id=default_id,
            species_vocabulary=tuple(config.species_vocabulary),
            baseline_metadata={
                "buffer_name": "atomic_baseline",
                "device": "cpu",
                "dtype": preparation.resolved_dtype,
                "exact_zero": True,
                "is_parameter": False,
                "requires_grad": False,
                "shape": [len(config.species_vocabulary)],
            },
        )


def initialize_scratch_model(
    preparation: ScratchTrainingPreparation,
) -> ScratchModelInitialization:
    """Create a deterministic scratch bundle behind a structured boundary."""

    try:
        return _initialize_scratch_model(preparation)
    except ScratchModelInitializationError:
        raise
    except Exception as error:
        raise _error(
            "MODEL_INITIALIZATION_FAILED",
            "unexpected scratch initialization boundary failure",
            stage="initialization.boundary",
            preparation=(
                preparation
                if isinstance(preparation, ScratchTrainingPreparation)
                else None
            ),
            original_error=error,
        ) from error


__all__ = [
    "SCRATCH_MODEL_INITIALIZATION_CONVENTION_VERSION",
    "ScratchModelInitialization",
    "ScratchModelInitializationError",
    "initialize_scratch_model",
]
