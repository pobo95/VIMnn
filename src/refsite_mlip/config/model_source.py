"""Canonical tagged model sources for training-run schema v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

from refsite_mlip.data import PhaseSpecification, ReferenceTemplateBuilderConfig
from refsite_mlip.models import EvaluationPolicy, PotentialConfig


class ModelSourceConfigError(ValueError):
    """Structured model-source schema or cross-contract failure."""

    def __init__(self, reason_code: str, message: str, *, field: str) -> None:
        self.reason_code = reason_code
        self.field = field
        self.message = message
        super().__init__(f"[{reason_code}] field={field!r} {message}")


def _error(reason: str, message: str, field: str) -> ModelSourceConfigError:
    return ModelSourceConfigError(reason, message, field=field)


def _strict_mapping(
    value: Any,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(
            "INVALID_MODEL_SOURCE_SECTION",
            "value must be a mapping",
            field,
        )
    keys = frozenset(value)
    if any(type(key) is not str for key in keys):
        raise _error(
            "UNKNOWN_CONFIG_KEY",
            "mapping keys must be strings",
            field,
        )
    unknown = keys - allowed
    missing = required - keys
    if unknown:
        raise _error(
            "UNKNOWN_CONFIG_KEY",
            f"unknown keys: {sorted(unknown)!r}",
            field,
        )
    if missing:
        raise _error(
            "MISSING_CONFIG_KEY",
            f"missing keys: {sorted(missing)!r}",
            field,
        )
    return value


def _path_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise _error(
            "INVALID_CONFIG_PATH",
            "path must be a nonempty string without NUL",
            field,
        )
    return value


def _parse_canonical(
    value: Any,
    *,
    cls: type,
    allowed: frozenset[str],
    required: frozenset[str],
    field: str,
):
    payload = _strict_mapping(
        value,
        allowed=allowed,
        required=required,
        field=field,
    )
    try:
        return cls.from_dict(payload)
    except Exception as error:
        raise _error(
            "INVALID_MODEL_SOURCE_CONFIG",
            f"{cls.__name__} validation failed: {error}",
            field,
        ) from error


_POTENTIAL_KEYS = frozenset(
    {
        "species_vocabulary",
        "num_layers",
        "feature",
        "higher_body",
        "readout_hidden",
        "energy_scale",
        "epsilon_ot",
        "ell_ot",
        "train_sinkhorn_iterations",
        "phase_steps",
        "phase_damping",
        "transport_support",
        "eval_sinkhorn_warmup_iterations",
    }
)
_FEATURE_KEYS = frozenset(
    {
        "species_vocabulary",
        "n_radial",
        "lmax",
        "ell_feature",
        "r_cut",
        "probability_tolerance",
        "site_type_vocabulary",
        "feature_layout_version",
        "radial_basis_version",
        "e3nn_normalization",
        "e3nn_normalize",
        "displacement_orientation",
    }
)
_HIGHER_BODY_KEYS = frozenset(
    {
        "irreps_feature",
        "species_count",
        "site_type_count",
        "site_type_embedding_dim",
        "n_correlation_channels",
        "lmax",
        "radial_feature_dim",
        "radial_hidden_dims",
        "avg_num_neighbors",
        "cutoff",
        "edge_length_scale",
        "correlation_mode",
        "contract_version",
    }
)
_SUPPORT_BASE_KEYS = frozenset(
    {
        "kind",
        "cutoff",
        "switch_width",
        "candidate_skin",
        "backend",
        "convention_version",
    }
)
_SUPPORT_BLOCK_KEYS = frozenset(
    {"candidate_backend", "site_block_size", "atom_block_size"}
)
_BUILDER_KEYS = frozenset(
    {
        "template_id",
        "strict_domain",
        "site_type_ids",
        "graph_cutoff",
        "graph_skin",
        "maximum_strain",
        "minimum_edge_length",
        "avg_num_neighbors",
        "expected_active_degree",
        "expected_candidate_degree",
        "expected_stabilizer_size",
        "canonical_tolerance",
        "metric_tolerance",
        "template_convention_version",
        "builder_convention_version",
    }
)
_DOMAIN_KEYS = frozenset(
    {
        "reference_site_count",
        "supercell_shape",
        "species_vocabulary",
        "reference_composition",
        "allowed_compositions",
        "allowed_num_atoms",
        "allowed_vacancy_masses",
        "convention_version",
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


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise _error(
            "INVALID_MODEL_SOURCE_CONFIG",
            "value must be an integer and bool is forbidden",
            field,
        )
    return int(value)


def _real(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _error(
            "INVALID_MODEL_SOURCE_CONFIG",
            "value must be a real number and bool is forbidden",
            field,
        )
    result = float(value)
    if not math.isfinite(result):
        raise _error(
            "NONFINITE_CONFIG_VALUE",
            "value must be finite",
            field,
        )
    return result


def _string(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise _error(
            "INVALID_MODEL_SOURCE_CONFIG",
            "value must be a string",
            field,
        )
    return value


def _sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise _error(
            "INVALID_MODEL_SOURCE_CONFIG",
            "value must be a deterministic sequence",
            field,
        )
    return tuple(value)


def _integer_vector(value: Any, *, field: str) -> list[int]:
    return [
        _integer(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    ]


def _real_vector(value: Any, *, field: str) -> list[float]:
    return [
        _real(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    ]


def _integer_matrix(value: Any, *, field: str) -> list[list[int]]:
    return [
        _integer_vector(row, field=f"{field}[{index}]")
        for index, row in enumerate(_sequence(value, field=field))
    ]


def _real_matrix(value: Any, *, field: str) -> list[list[float]]:
    return [
        _real_vector(row, field=f"{field}[{index}]")
        for index, row in enumerate(_sequence(value, field=field))
    ]


def _canonical_feature_payload(value: Any) -> dict[str, Any]:
    field = "model_source.potential.feature"
    payload = dict(
        _strict_mapping(
            value,
            allowed=_FEATURE_KEYS,
            required=_FEATURE_KEYS,
            field=field,
        )
    )
    payload["species_vocabulary"] = _integer_vector(
        payload["species_vocabulary"], field=f"{field}.species_vocabulary"
    )
    for name in ("n_radial", "lmax"):
        payload[name] = _integer(payload[name], field=f"{field}.{name}")
    for name in ("ell_feature", "r_cut"):
        payload[name] = _real(payload[name], field=f"{field}.{name}")
    tolerance = payload["probability_tolerance"]
    if tolerance is not None:
        payload["probability_tolerance"] = _real(
            tolerance, field=f"{field}.probability_tolerance"
        )
    site_types = payload["site_type_vocabulary"]
    if site_types is not None:
        payload["site_type_vocabulary"] = _integer_vector(
            site_types, field=f"{field}.site_type_vocabulary"
        )
    for name in (
        "feature_layout_version",
        "radial_basis_version",
        "e3nn_normalization",
        "displacement_orientation",
    ):
        payload[name] = _string(payload[name], field=f"{field}.{name}")
    if type(payload["e3nn_normalize"]) is not bool:
        raise _error(
            "INVALID_MODEL_SOURCE_CONFIG",
            "e3nn_normalize must be a bool",
            f"{field}.e3nn_normalize",
        )
    return payload


def _canonical_higher_body_payload(value: Any) -> dict[str, Any]:
    field = "model_source.potential.higher_body"
    payload = dict(
        _strict_mapping(
            value,
            allowed=_HIGHER_BODY_KEYS,
            required=_HIGHER_BODY_KEYS,
            field=field,
        )
    )
    for name in (
        "species_count",
        "site_type_count",
        "site_type_embedding_dim",
        "n_correlation_channels",
        "lmax",
        "radial_feature_dim",
    ):
        payload[name] = _integer(payload[name], field=f"{field}.{name}")
    payload["radial_hidden_dims"] = _integer_vector(
        payload["radial_hidden_dims"], field=f"{field}.radial_hidden_dims"
    )
    for name in ("avg_num_neighbors", "cutoff", "edge_length_scale"):
        payload[name] = _real(payload[name], field=f"{field}.{name}")
    for name in ("irreps_feature", "correlation_mode", "contract_version"):
        payload[name] = _string(payload[name], field=f"{field}.{name}")
    return payload


def _canonical_support_payload(value: Any) -> dict[str, Any]:
    field = "model_source.potential.transport_support"
    payload = dict(
        _strict_mapping(
            value,
            allowed=_SUPPORT_BASE_KEYS | _SUPPORT_BLOCK_KEYS,
            required=_SUPPORT_BASE_KEYS,
            field=field,
        )
    )
    present = frozenset(payload) & _SUPPORT_BLOCK_KEYS
    if present and present != _SUPPORT_BLOCK_KEYS:
        raise _error(
            "MISSING_CONFIG_KEY",
            "blocked candidate support keys must be supplied together",
            field,
        )
    if present and payload["candidate_backend"] != "blocked":
        raise _error(
            "INVALID_MODEL_SOURCE_CONFIG",
            "explicit candidate block sizes require candidate_backend='blocked'",
            f"{field}.candidate_backend",
        )
    for name in ("cutoff", "switch_width", "candidate_skin"):
        payload[name] = _real(payload[name], field=f"{field}.{name}")
    for name in ("kind", "backend", "convention_version"):
        payload[name] = _string(payload[name], field=f"{field}.{name}")
    if present:
        payload["candidate_backend"] = _string(
            payload["candidate_backend"], field=f"{field}.candidate_backend"
        )
        for name in ("site_block_size", "atom_block_size"):
            payload[name] = _integer(payload[name], field=f"{field}.{name}")
    return payload


def _canonical_potential_payload(value: Any) -> dict[str, Any]:
    field = "model_source.potential"
    payload = dict(
        _strict_mapping(
            value,
            allowed=_POTENTIAL_KEYS,
            required=_POTENTIAL_KEYS,
            field=field,
        )
    )
    payload["species_vocabulary"] = _integer_vector(
        payload["species_vocabulary"], field=f"{field}.species_vocabulary"
    )
    for name in (
        "num_layers",
        "readout_hidden",
        "train_sinkhorn_iterations",
        "eval_sinkhorn_warmup_iterations",
    ):
        payload[name] = _integer(payload[name], field=f"{field}.{name}")
    for name in ("energy_scale", "epsilon_ot", "ell_ot"):
        payload[name] = _real(payload[name], field=f"{field}.{name}")
    for name in ("phase_steps", "phase_damping"):
        payload[name] = _real_vector(payload[name], field=f"{field}.{name}")
    payload["feature"] = _canonical_feature_payload(payload["feature"])
    payload["higher_body"] = _canonical_higher_body_payload(
        payload["higher_body"]
    )
    payload["transport_support"] = _canonical_support_payload(
        payload["transport_support"]
    )
    return payload


def _canonical_domain_payload(value: Any) -> dict[str, Any]:
    field = "model_source.reference_templates[].builder.strict_domain"
    payload = dict(
        _strict_mapping(
            value,
            allowed=_DOMAIN_KEYS,
            required=_DOMAIN_KEYS,
            field=field,
        )
    )
    payload["reference_site_count"] = _integer(
        payload["reference_site_count"], field=f"{field}.reference_site_count"
    )
    for name in (
        "supercell_shape",
        "species_vocabulary",
        "reference_composition",
        "allowed_num_atoms",
        "allowed_vacancy_masses",
    ):
        payload[name] = _integer_vector(payload[name], field=f"{field}.{name}")
    payload["allowed_compositions"] = _integer_matrix(
        payload["allowed_compositions"], field=f"{field}.allowed_compositions"
    )
    payload["convention_version"] = _string(
        payload["convention_version"], field=f"{field}.convention_version"
    )
    return payload


def _canonical_builder_payload(value: Any) -> dict[str, Any]:
    field = "model_source.reference_templates[].builder"
    payload = dict(
        _strict_mapping(
            value,
            allowed=_BUILDER_KEYS,
            required=_BUILDER_KEYS,
            field=field,
        )
    )
    payload["template_id"] = _string(
        payload["template_id"], field=f"{field}.template_id"
    )
    payload["strict_domain"] = _canonical_domain_payload(
        payload["strict_domain"]
    )
    payload["site_type_ids"] = _integer_vector(
        payload["site_type_ids"], field=f"{field}.site_type_ids"
    )
    for name in (
        "graph_cutoff",
        "graph_skin",
        "maximum_strain",
        "minimum_edge_length",
        "avg_num_neighbors",
        "canonical_tolerance",
        "metric_tolerance",
    ):
        payload[name] = _real(payload[name], field=f"{field}.{name}")
    for name in (
        "expected_active_degree",
        "expected_candidate_degree",
        "expected_stabilizer_size",
    ):
        payload[name] = _integer(payload[name], field=f"{field}.{name}")
    for name in ("template_convention_version", "builder_convention_version"):
        payload[name] = _string(payload[name], field=f"{field}.{name}")
    return payload


def _canonical_phase_payload(value: Any) -> dict[str, Any]:
    field = "model_source.reference_templates[].phase_specification"
    payload = dict(
        _strict_mapping(
            value,
            allowed=_PHASE_KEYS,
            required=_PHASE_KEYS,
            field=field,
        )
    )
    payload["modes"] = _integer_matrix(
        payload["modes"], field=f"{field}.modes"
    )
    payload["mode_weights"] = _real_vector(
        payload["mode_weights"], field=f"{field}.mode_weights"
    )
    payload["site_type_alignment_weights"] = _real_matrix(
        payload["site_type_alignment_weights"],
        field=f"{field}.site_type_alignment_weights",
    )
    payload["channel_weights"] = _real_vector(
        payload["channel_weights"], field=f"{field}.channel_weights"
    )
    for name in ("approval_status", "convention_version", "floating_dtype"):
        payload[name] = _string(payload[name], field=f"{field}.{name}")
    return payload


def _canonical_policy_payload(value: Any) -> dict[str, Any]:
    field = "model_source.reference_templates[].evaluation_policy"
    payload = dict(
        _strict_mapping(
            value,
            allowed=_POLICY_KEYS,
            required=_POLICY_KEYS,
            field=field,
        )
    )
    for name in (
        "template_id",
        "template_fingerprint",
        "candidate_dtype",
        "transport_path",
        "convention_version",
        "content_fingerprint",
    ):
        payload[name] = _string(payload[name], field=f"{field}.{name}")
    payload["candidate_offsets"] = _real_matrix(
        payload["candidate_offsets"], field=f"{field}.candidate_offsets"
    )
    for name in ("phase_step_schedule", "phase_damping_schedule"):
        payload[name] = _real_vector(payload[name], field=f"{field}.{name}")
    for name in (
        "minimum_objective_gap_absolute",
        "minimum_cross_amplitude_absolute",
        "minimum_atomic_amplitude_absolute",
        "minimum_reference_amplitude_absolute",
        "minimum_curvature",
        "maximum_condition",
        "maximum_gradient_norm",
        "equivalence_tolerance",
    ):
        payload[name] = _real(payload[name], field=f"{field}.{name}")
    return payload


@dataclass(frozen=True)
class BundleModelSourceConfig:
    """A portable bundle used as model/registry initialization state."""

    path: str
    kind: str = "bundle"

    def __post_init__(self) -> None:
        if self.kind != "bundle":
            raise _error(
                "INVALID_MODEL_SOURCE_KIND",
                "bundle source kind must be 'bundle'",
                "model_source.kind",
            )
        object.__setattr__(
            self,
            "path",
            _path_text(self.path, field="model_source.path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "bundle", "path": self.path}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BundleModelSourceConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"kind", "path"}),
            required=frozenset({"kind", "path"}),
            field="model_source",
        )
        if payload["kind"] != "bundle":
            raise _error(
                "INVALID_MODEL_SOURCE_KIND",
                "kind must be 'bundle'",
                "model_source.kind",
            )
        return cls(path=payload["path"])


@dataclass(frozen=True, eq=False)
class ScratchReferenceTemplateSourceConfig:
    """One explicit POSCAR-backed reference source, not yet materialized."""

    poscar_path: str
    builder: ReferenceTemplateBuilderConfig
    phase_specification: PhaseSpecification
    evaluation_policy: EvaluationPolicy | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "poscar_path",
            _path_text(
                self.poscar_path,
                field="model_source.reference_templates[].poscar_path",
            ),
        )
        if not isinstance(self.builder, ReferenceTemplateBuilderConfig):
            raise TypeError("builder must be ReferenceTemplateBuilderConfig")
        if not isinstance(self.phase_specification, PhaseSpecification):
            raise TypeError("phase_specification must be PhaseSpecification")
        if self.evaluation_policy is not None and not isinstance(
            self.evaluation_policy, EvaluationPolicy
        ):
            raise TypeError("evaluation_policy must be EvaluationPolicy or None")
        builder = ReferenceTemplateBuilderConfig.from_dict(
            _canonical_builder_payload(self.builder.to_dict())
        )
        phase = PhaseSpecification.from_dict(
            _canonical_phase_payload(self.phase_specification.to_dict())
        )
        policy = (
            None
            if self.evaluation_policy is None
            else EvaluationPolicy.from_dict(
                _canonical_policy_payload(self.evaluation_policy.to_dict())
            )
        )
        object.__setattr__(self, "builder", builder)
        object.__setattr__(self, "phase_specification", phase)
        object.__setattr__(self, "evaluation_policy", policy)
        if (
            policy is not None
            and policy.template_id != builder.template_id
        ):
            raise _error(
                "MODEL_SOURCE_TEMPLATE_MISMATCH",
                "evaluation policy template_id differs from builder template_id",
                "model_source.reference_templates[].evaluation_policy.template_id",
            )

    @property
    def template_id(self) -> str:
        return self.builder.template_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "poscar_path": self.poscar_path,
            "builder": self.builder.to_dict(),
            "phase_specification": self.phase_specification.to_dict(),
            "evaluation_policy": (
                None
                if self.evaluation_policy is None
                else self.evaluation_policy.to_dict()
            ),
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ScratchReferenceTemplateSourceConfig)
            and self.to_dict() == other.to_dict()
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ScratchReferenceTemplateSourceConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset(
                {
                    "poscar_path",
                    "builder",
                    "phase_specification",
                    "evaluation_policy",
                }
            ),
            required=frozenset(
                {"poscar_path", "builder", "phase_specification"}
            ),
            field="model_source.reference_templates[]",
        )
        builder_payload = _canonical_builder_payload(payload["builder"])
        builder = _parse_canonical(
            builder_payload,
            cls=ReferenceTemplateBuilderConfig,
            allowed=_BUILDER_KEYS,
            required=_BUILDER_KEYS,
            field="model_source.reference_templates[].builder",
        )
        phase = _parse_canonical(
            _canonical_phase_payload(payload["phase_specification"]),
            cls=PhaseSpecification,
            allowed=_PHASE_KEYS,
            required=_PHASE_KEYS,
            field="model_source.reference_templates[].phase_specification",
        )
        policy_payload = payload.get("evaluation_policy")
        policy = None
        if policy_payload is not None:
            policy = _parse_canonical(
                _canonical_policy_payload(policy_payload),
                cls=EvaluationPolicy,
                allowed=_POLICY_KEYS,
                required=_POLICY_KEYS,
                field="model_source.reference_templates[].evaluation_policy",
            )
        return cls(
            poscar_path=payload["poscar_path"],
            builder=builder,
            phase_specification=phase,
            evaluation_policy=policy,
        )


def _alignment_matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _error(
            "INVALID_SPECIES_ALIGNMENT",
            "species_alignment_weights must be a nonempty matrix",
            "model_source.species_alignment_weights",
        )
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise _error(
                "INVALID_SPECIES_ALIGNMENT",
                "each species alignment row must be a sequence",
                f"model_source.species_alignment_weights[{row_index}]",
            )
        converted = []
        for column_index, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, Real):
                raise _error(
                    "INVALID_SPECIES_ALIGNMENT",
                    "alignment values must be real numbers; bool is forbidden",
                    "model_source.species_alignment_weights"
                    f"[{row_index}][{column_index}]",
                )
            result = float(item)
            if not math.isfinite(result):
                raise _error(
                    "NONFINITE_CONFIG_VALUE",
                    "alignment values must be finite",
                    "model_source.species_alignment_weights"
                    f"[{row_index}][{column_index}]",
                )
            converted.append(result)
        if not converted:
            raise _error(
                "INVALID_SPECIES_ALIGNMENT",
                "alignment rows must not be empty",
                f"model_source.species_alignment_weights[{row_index}]",
            )
        rows.append(tuple(converted))
    if not rows or len({len(row) for row in rows}) != 1:
        raise _error(
            "INVALID_SPECIES_ALIGNMENT",
            "species_alignment_weights must be a nonempty rectangular matrix",
            "model_source.species_alignment_weights",
        )
    return tuple(rows)


@dataclass(frozen=True, eq=False)
class ScratchModelSourceConfig:
    """Complete, deterministic inputs for future scratch construction."""

    initialization_seed: int
    potential: PotentialConfig
    species_alignment_weights: tuple[tuple[float, ...], ...]
    reference_templates: tuple[ScratchReferenceTemplateSourceConfig, ...]
    default_template_id: str
    kind: str = "scratch"
    __hash__ = None

    def __post_init__(self) -> None:
        if self.kind != "scratch":
            raise _error(
                "INVALID_MODEL_SOURCE_KIND",
                "scratch source kind must be 'scratch'",
                "model_source.kind",
            )
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, Integral
        ):
            raise _error(
                "INVALID_INITIALIZATION_SEED",
                "initialization_seed must be an integer; bool is forbidden",
                "model_source.initialization_seed",
            )
        object.__setattr__(self, "initialization_seed", int(self.initialization_seed))
        if not isinstance(self.potential, PotentialConfig):
            raise TypeError("potential must be PotentialConfig")
        potential = PotentialConfig.from_dict(
            _canonical_potential_payload(self.potential.to_dict())
        )
        object.__setattr__(self, "potential", potential)
        matrix = _alignment_matrix(self.species_alignment_weights)
        object.__setattr__(self, "species_alignment_weights", matrix)
        if not isinstance(self.reference_templates, Sequence) or isinstance(
            self.reference_templates, (str, bytes, bytearray)
        ):
            raise _error(
                "INVALID_REFERENCE_TEMPLATE_SEQUENCE",
                "reference_templates must be an ordered sequence",
                "model_source.reference_templates",
            )
        templates = tuple(self.reference_templates)
        if not templates or any(
            not isinstance(item, ScratchReferenceTemplateSourceConfig)
            for item in templates
        ):
            raise _error(
                "INVALID_REFERENCE_TEMPLATE_SEQUENCE",
                "reference_templates must be nonempty canonical template sources",
                "model_source.reference_templates",
            )
        object.__setattr__(self, "reference_templates", templates)
        ids = tuple(item.template_id for item in templates)
        if len(set(ids)) != len(ids):
            raise _error(
                "DUPLICATE_TEMPLATE_ID",
                "reference template IDs must be unique",
                "model_source.reference_templates",
            )
        if type(self.default_template_id) is not str or self.default_template_id not in ids:
            raise _error(
                "MISSING_DEFAULT_TEMPLATE",
                "default_template_id must exactly name one reference template",
                "model_source.default_template_id",
            )

        vocabulary = potential.species_vocabulary
        site_vocabulary = potential.feature.site_type_vocabulary
        channel_count = templates[0].phase_specification.num_channels
        if len(matrix) != len(vocabulary) or any(
            len(row) != channel_count for row in matrix
        ):
            raise _error(
                "SPECIES_ALIGNMENT_SHAPE_MISMATCH",
                "species alignment shape must be [species, phase_channels]",
                "model_source.species_alignment_weights",
            )
        for index, template in enumerate(templates):
            builder = template.builder
            phase = template.phase_specification
            prefix = f"model_source.reference_templates[{index}]"
            if builder.strict_domain.species_vocabulary != vocabulary:
                raise _error(
                    "MODEL_SOURCE_SPECIES_MISMATCH",
                    "template domain species order differs from PotentialConfig",
                    f"{prefix}.builder.strict_domain.species_vocabulary",
                )
            if tuple(builder.site_type_ids) != site_vocabulary:
                raise _error(
                    "MODEL_SOURCE_SITE_TYPE_MISMATCH",
                    "builder global site types differ from PotentialConfig",
                    f"{prefix}.builder.site_type_ids",
                )
            if phase.site_type_alignment_weights.shape[0] != len(site_vocabulary):
                raise _error(
                    "MODEL_SOURCE_SITE_TYPE_MISMATCH",
                    "phase site-type alignment row count is incompatible",
                    f"{prefix}.phase_specification.site_type_alignment_weights",
                )
            if phase.num_channels != channel_count:
                raise _error(
                    "MODEL_SOURCE_PHASE_CHANNEL_MISMATCH",
                    "all templates must use one global phase channel count",
                    f"{prefix}.phase_specification",
                )
            if not math.isclose(
                builder.avg_num_neighbors,
                potential.higher_body.avg_num_neighbors,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise _error(
                    "MODEL_SOURCE_GRAPH_MISMATCH",
                    "builder and PotentialConfig avg_num_neighbors differ",
                    f"{prefix}.builder.avg_num_neighbors",
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "scratch",
            "initialization_seed": self.initialization_seed,
            "potential": self.potential.to_dict(),
            "species_alignment_weights": [
                list(row) for row in self.species_alignment_weights
            ],
            "reference_templates": [
                item.to_dict() for item in self.reference_templates
            ],
            "default_template_id": self.default_template_id,
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ScratchModelSourceConfig)
            and self.to_dict() == other.to_dict()
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScratchModelSourceConfig":
        keys = frozenset(
            {
                "kind",
                "initialization_seed",
                "potential",
                "species_alignment_weights",
                "reference_templates",
                "default_template_id",
            }
        )
        payload = _strict_mapping(
            value,
            allowed=keys,
            required=keys,
            field="model_source",
        )
        if payload["kind"] != "scratch":
            raise _error(
                "INVALID_MODEL_SOURCE_KIND",
                "kind must be 'scratch'",
                "model_source.kind",
            )
        potential = _parse_canonical(
            _canonical_potential_payload(payload["potential"]),
            cls=PotentialConfig,
            allowed=_POTENTIAL_KEYS,
            required=_POTENTIAL_KEYS,
            field="model_source.potential",
        )
        values = payload["reference_templates"]
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray)
        ):
            raise _error(
                "INVALID_REFERENCE_TEMPLATE_SEQUENCE",
                "reference_templates must be an ordered sequence",
                "model_source.reference_templates",
            )
        return cls(
            initialization_seed=payload["initialization_seed"],
            potential=potential,
            species_alignment_weights=_alignment_matrix(
                payload["species_alignment_weights"]
            ),
            reference_templates=tuple(
                ScratchReferenceTemplateSourceConfig.from_dict(item)
                for item in values
            ),
            default_template_id=payload["default_template_id"],
        )


ModelSourceConfig = BundleModelSourceConfig | ScratchModelSourceConfig


def model_source_from_dict(value: Mapping[str, Any]) -> ModelSourceConfig:
    if not isinstance(value, Mapping):
        raise _error(
            "INVALID_MODEL_SOURCE_SECTION",
            "model_source must be a mapping",
            "model_source",
        )
    kind = value.get("kind")
    if type(kind) is not str or kind not in ("bundle", "scratch"):
        raise _error(
            "INVALID_MODEL_SOURCE_KIND",
            "kind must be exactly 'bundle' or 'scratch'",
            "model_source.kind",
        )
    if kind == "bundle":
        return BundleModelSourceConfig.from_dict(value)
    return ScratchModelSourceConfig.from_dict(value)


__all__ = [
    "BundleModelSourceConfig",
    "ModelSourceConfig",
    "ModelSourceConfigError",
    "ScratchModelSourceConfig",
    "ScratchReferenceTemplateSourceConfig",
    "model_source_from_dict",
]
