"""Beginner training recipes compiled into canonical training-run schema v2.

Recipes are authoring inputs only.  They never form part of the execution,
checkpoint, resume, or export contracts; those consumers receive the complete
``TrainingRunConfig`` produced here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import stat
from typing import Any

import yaml

from refsite_mlip.data import PhaseSpecification, ReferenceTemplateBuilderConfig
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig, SymmetricCorrelationConfig
from refsite_mlip.interactions.higher_body import (
    LEGACY_HIGHER_BODY_CONTRACT_VERSION,
    SYMMETRIC_POWER_CONTRACT_VERSION,
)
from refsite_mlip.models import EvaluationPolicy, PotentialConfig
from refsite_mlip.training import (
    AtomicBaselineConfig,
    CheckpointedFitConfig,
    FitConfig,
    LossConfig,
    ModelSelectionConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
)
from refsite_mlip.transport import TRAIN_FIXED

from .model_source import (
    ScratchModelSourceConfig,
    ScratchReferenceTemplateSourceConfig,
)
from .radii import InteractionRadiusConfig, transport_support_config_from_radii
from .training_run import (
    TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2,
    TrainingDataConfig,
    TrainingDataSourceConfig,
    TrainingRunConfig,
    TrainingRunConfigOverrides,
    TrainingRuntimeConfig,
    apply_training_run_overrides,
)


TRAINING_RECIPE_SCHEMA_VERSION = "refsite_training_recipe_v1"
REFERENCE_SPECIFICATION_SCHEMA_VERSION = "refsite_reference_specification_v1"
RECIPE_RESOLVER_VERSION = "refsite_training_recipe_resolver_v1"
SYMMETRIC_MODEL_DEFAULTS_VERSION = "symmetric_model_defaults_v2"
SEQUENTIAL_MODEL_DEFAULTS_VERSION = "sequential_model_defaults_v2"
TRAINING_DEFAULTS_VERSION = "training_defaults_v2"
RADIUS_DERIVATION_VERSION = "radius_derivation_v1"
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_EARLY_STOPPING_RELATIVE_DELTA = 0.0

SYMMETRIC_CORRELATION_METHOD = "symmetric"
SEQUENTIAL_CORRELATION_METHOD = "sequential"
SINKHORN_OT_SOLVER = "sinkhorn"
SINKHORN_NEWTON_KRYLOV_OT_SOLVER = "sinkhorn_newton_krylov"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_IRREP_TERM = re.compile(r"^(?P<mul>[1-9][0-9]*)x(?P<l>[0-9]+)(?P<p>[eo])$")


class TrainingRecipeError(ValueError):
    """Structured recipe parse, resolution, or compatibility error."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        field: str | None = None,
        path: str | os.PathLike[str] | None = None,
        expected: Any = None,
        actual: Any = None,
        original_error: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.message = message
        self.stage = stage
        self.field = field
        self.path = None if path is None else str(path)
        self.expected = expected
        self.actual = actual
        self.original_error = original_error
        context = " ".join(
            f"{name}={value!r}"
            for name, value in (
                ("path", self.path),
                ("field", field),
                ("expected", expected),
                ("actual", actual),
            )
            if value is not None
        )
        suffix = "" if not context else " " + context
        super().__init__(f"[{reason_code}] stage={stage!r}{suffix} {message}")


def _error(reason: str, message: str, *, stage: str, **context: Any) -> TrainingRecipeError:
    return TrainingRecipeError(reason, message, stage=stage, **context)


def _strict_mapping(
    value: Any,
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(
            "INVALID_RECIPE_SECTION",
            "value must be a mapping",
            stage="recipe.schema",
            field=field_name,
            actual=type(value).__name__,
        )
    if any(type(key) is not str for key in value):
        raise _error(
            "UNKNOWN_RECIPE_KEY",
            "mapping keys must be strings",
            stage="recipe.schema",
            field=field_name,
        )
    keys = frozenset(value)
    unknown = keys - allowed
    missing = required - keys
    if unknown:
        raise _error(
            "UNKNOWN_RECIPE_KEY",
            f"unknown keys: {sorted(unknown)!r}",
            stage="recipe.schema",
            field=field_name,
        )
    if missing:
        raise _error(
            "MISSING_RECIPE_KEY",
            f"missing keys: {sorted(missing)!r}",
            stage="recipe.schema",
            field=field_name,
        )
    return value


def _path(value: Any, *, field_name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise _error(
            "INVALID_RECIPE_PATH",
            "path must be a nonempty string without NUL",
            stage="recipe.validation",
            field=field_name,
            actual=value,
        )
    return value


def _reference_alias(value: Any, *, field_name: str) -> str:
    if (
        type(value) is not str
        or _NAME.fullmatch(value) is None
        or value in (".", "..")
        or ".." in value
    ):
        raise _error(
            "INVALID_REFERENCE_ALIAS",
            "reference alias must be a nonempty filesystem-safe name, not a path",
            stage="recipe.reference",
            field=field_name,
            actual=value,
        )
    return value


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise _error(
            "INVALID_RECIPE_INTEGER",
            "value must be an integer and bool is forbidden",
            stage="recipe.validation",
            field=field_name,
            actual=value,
        )
    result = int(value)
    if result <= 0:
        raise _error(
            "INVALID_RECIPE_INTEGER",
            "value must be positive",
            stage="recipe.validation",
            field=field_name,
            actual=value,
        )
    return result


def _finite_nonnegative(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _error(
            "INVALID_RECIPE_REAL",
            "value must be a real number and bool is forbidden",
            stage="recipe.validation",
            field=field_name,
            actual=value,
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise _error(
            "INVALID_RECIPE_REAL",
            "value must be finite and nonnegative",
            stage="recipe.validation",
            field=field_name,
            actual=value,
        )
    return result


def _finite_real(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _error(
            "INVALID_RECIPE_REAL",
            "value must be a real number and bool is forbidden",
            stage="recipe.validation",
            field=field_name,
            actual=value,
        )
    result = float(value)
    if not math.isfinite(result):
        raise _error(
            "INVALID_RECIPE_REAL",
            "value must be finite",
            stage="recipe.validation",
            field=field_name,
            actual=value,
        )
    return result


def _positive_real(value: Any, *, field_name: str) -> float:
    result = _finite_nonnegative(value, field_name=field_name)
    if result <= 0.0:
        raise _error(
            "INVALID_RECIPE_REAL",
            "value must be positive",
            stage="recipe.validation",
            field=field_name,
            actual=value,
        )
    return result


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _natural_hidden_irreps(channels: int, maximum_l: int) -> str:
    return " + ".join(
        f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}"
        for l in range(maximum_l + 1)
    )


def _parse_hidden_irreps(value: Any) -> tuple[int, int, str]:
    if type(value) is not str or not value.strip():
        raise _error(
            "INVALID_HIDDEN_IRREPS",
            "hidden_irreps must be a nonempty string",
            stage="recipe.model",
            field="model.hidden_irreps",
            actual=value,
        )
    terms = [item.strip().replace(" ", "") for item in value.split("+")]
    parsed: dict[int, tuple[int, str]] = {}
    for term in terms:
        match = _IRREP_TERM.fullmatch(term)
        if match is None:
            raise _error(
                "INVALID_HIDDEN_IRREPS",
                "only KxLe/KxLo terms are supported",
                stage="recipe.model",
                field="model.hidden_irreps",
                actual=value,
            )
        multiplicity = int(match.group("mul"))
        angular = int(match.group("l"))
        parity = match.group("p")
        if angular in parsed:
            raise _error(
                "DUPLICATE_HIDDEN_IRREP",
                "each angular momentum must appear exactly once",
                stage="recipe.model",
                field="model.hidden_irreps",
                actual=value,
            )
        parsed[angular] = (multiplicity, parity)
    maximum_l = max(parsed)
    if maximum_l > 2 or tuple(sorted(parsed)) != tuple(range(maximum_l + 1)):
        raise _error(
            "UNSUPPORTED_HIDDEN_IRREPS",
            "hidden irreps require every l=0..L exactly once with L<=2",
            stage="recipe.model",
            field="model.hidden_irreps",
            actual=value,
        )
    multiplicities = {item[0] for item in parsed.values()}
    if len(multiplicities) != 1:
        raise _error(
            "NONUNIFORM_HIDDEN_MULTIPLICITY",
            "all hidden irreps must have one uniform multiplicity",
            stage="recipe.model",
            field="model.hidden_irreps",
            actual=value,
        )
    for angular, (_, parity) in parsed.items():
        expected = "e" if angular % 2 == 0 else "o"
        if parity != expected:
            raise _error(
                "INVALID_HIDDEN_PARITY",
                "hidden irreps must use natural O(3) parity",
                stage="recipe.model",
                field="model.hidden_irreps",
                expected=f"{angular}{expected}",
                actual=f"{angular}{parity}",
            )
    channels = next(iter(multiplicities))
    return channels, maximum_l, _natural_hidden_irreps(channels, maximum_l)


@dataclass(frozen=True)
class RecipeModelConfig:
    correlation_method: str = SYMMETRIC_CORRELATION_METHOD
    num_interactions: int = 2
    correlation: int | None = None
    correlation_mode: str | None = None
    hidden_channels: int | None = None
    max_L: int | None = None
    hidden_irreps: str | None = None
    initialization_seed: int = 0
    _provided_fields: tuple[str, ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.correlation_method not in (
            SYMMETRIC_CORRELATION_METHOD,
            SEQUENTIAL_CORRELATION_METHOD,
        ):
            raise _error(
                "UNSUPPORTED_CORRELATION_METHOD",
                "correlation_method must be symmetric or sequential",
                stage="recipe.model",
                field="model.correlation_method",
                actual=self.correlation_method,
            )
        object.__setattr__(
            self,
            "num_interactions",
            _positive_int(self.num_interactions, field_name="model.num_interactions"),
        )
        if self.correlation_method == SYMMETRIC_CORRELATION_METHOD:
            if self.correlation_mode is not None:
                raise _error(
                    "CONFLICTING_CORRELATION_SETTING",
                    "correlation_mode is available only for sequential correlation",
                    stage="recipe.model",
                    field="model.correlation_mode",
                )
            order = 3 if self.correlation is None else _positive_int(
                self.correlation, field_name="model.correlation"
            )
            if order not in (1, 2, 3):
                raise _error(
                    "UNSUPPORTED_CORRELATION_ORDER",
                    "correlation must be 1, 2, or 3",
                    stage="recipe.model",
                    field="model.correlation",
                    actual=order,
                )
            object.__setattr__(self, "correlation", order)
        else:
            if self.correlation is not None:
                raise _error(
                    "CONFLICTING_CORRELATION_SETTING",
                    "sequential correlation uses correlation_mode and forbids correlation",
                    stage="recipe.model",
                    field="model.correlation",
                )
            if self.correlation_mode not in ("uuu", "uvw"):
                raise _error(
                    "UNSUPPORTED_CORRELATION_MODE",
                    "sequential correlation_mode must be uuu or uvw",
                    stage="recipe.model",
                    field="model.correlation_mode",
                    actual=self.correlation_mode,
                )
        if self.hidden_irreps is not None:
            if self.correlation_method == SEQUENTIAL_CORRELATION_METHOD:
                raise _error(
                    "UNSUPPORTED_SEQUENTIAL_HIDDEN_IRREPS",
                    "hidden_irreps is available only for symmetric correlation",
                    stage="recipe.model",
                    field="model.hidden_irreps",
                )
            if self.hidden_channels is not None or self.max_L is not None:
                raise _error(
                    "CONFLICTING_HIDDEN_LAYOUT",
                    "hidden_irreps conflicts with hidden_channels/max_L",
                    stage="recipe.model",
                    field="model.hidden_irreps,model.hidden_channels,model.max_L",
                )
            channels, maximum_l, canonical = _parse_hidden_irreps(self.hidden_irreps)
            object.__setattr__(self, "hidden_channels", channels)
            object.__setattr__(self, "max_L", maximum_l)
            object.__setattr__(self, "hidden_irreps", canonical)
        else:
            channels = 8 if self.hidden_channels is None else _positive_int(
                self.hidden_channels, field_name="model.hidden_channels"
            )
            maximum_l = 2 if self.max_L is None else self.max_L
            if (
                isinstance(maximum_l, bool)
                or not isinstance(maximum_l, Integral)
                or int(maximum_l) not in (0, 1, 2)
            ):
                raise _error(
                    "UNSUPPORTED_MAX_L",
                    "max_L must be 0, 1, or 2",
                    stage="recipe.model",
                    field="model.max_L",
                    actual=maximum_l,
                )
            object.__setattr__(self, "hidden_channels", int(channels))
            object.__setattr__(self, "max_L", int(maximum_l))
        if (
            self.correlation_method == SEQUENTIAL_CORRELATION_METHOD
            and self.max_L != 2
        ):
            raise _error(
                "UNSUPPORTED_SEQUENTIAL_MAX_L",
                "sequential correlation requires max_L=2",
                stage="recipe.model",
                field="model.max_L",
                expected=2,
                actual=self.max_L,
            )
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, Integral
        ):
            raise _error(
                "INVALID_INITIALIZATION_SEED",
                "initialization_seed must be an integer and bool is forbidden",
                stage="recipe.model",
                field="model.initialization_seed",
                actual=self.initialization_seed,
            )
        object.__setattr__(self, "initialization_seed", int(self.initialization_seed))
        object.__setattr__(self, "_provided_fields", tuple(sorted(self._provided_fields)))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "correlation_method": self.correlation_method,
            "num_interactions": self.num_interactions,
            "initialization_seed": self.initialization_seed,
        }
        if self.correlation_method == SYMMETRIC_CORRELATION_METHOD:
            result["correlation"] = self.correlation
        else:
            result["correlation_mode"] = self.correlation_mode
        if self.hidden_irreps is None:
            result.update(
                hidden_channels=self.hidden_channels,
                max_L=self.max_L,
            )
        else:
            result["hidden_irreps"] = self.hidden_irreps
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeModelConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset(
                {
                    "correlation_method",
                    "num_interactions",
                    "correlation",
                    "correlation_mode",
                    "hidden_channels",
                    "max_L",
                    "hidden_irreps",
                    "initialization_seed",
                }
            ),
            field_name="model",
        )
        if "hidden_irreps" in payload and (
            "hidden_channels" in payload or "max_L" in payload
        ):
            raise _error(
                "CONFLICTING_HIDDEN_LAYOUT",
                "hidden_irreps conflicts with hidden_channels/max_L",
                stage="recipe.model",
                field="model",
            )
        return cls(**dict(payload), _provided_fields=tuple(payload))


@dataclass(frozen=True)
class RecipeOTSolverConfig:
    """Beginner-facing OT algorithm names independent of runtime enum names."""

    training: str
    inference: str

    def __post_init__(self) -> None:
        if self.training != SINKHORN_OT_SOLVER:
            if self.training == SINKHORN_NEWTON_KRYLOV_OT_SOLVER:
                raise _error(
                    "UNSUPPORTED_TRAINING_OT_SOLVER",
                    "sinkhorn_newton_krylov is inference-only because training requires differentiable fixed Sinkhorn",
                    stage="recipe.ot_solver",
                    field="ot_solver.training",
                    actual=self.training,
                )
            raise _error(
                "UNSUPPORTED_TRAINING_OT_SOLVER",
                "training OT solver must be sinkhorn",
                stage="recipe.ot_solver",
                field="ot_solver.training",
                actual=self.training,
            )
        if self.inference not in (
            SINKHORN_OT_SOLVER,
            SINKHORN_NEWTON_KRYLOV_OT_SOLVER,
        ):
            raise _error(
                "UNSUPPORTED_INFERENCE_OT_SOLVER",
                "inference OT solver must be sinkhorn or sinkhorn_newton_krylov",
                stage="recipe.ot_solver",
                field="ot_solver.inference",
                actual=self.inference,
            )

    def to_dict(self) -> dict[str, str]:
        return {"training": self.training, "inference": self.inference}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeOTSolverConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"training", "inference"}),
            required=frozenset({"training"}),
            field_name="ot_solver",
        )
        return cls(
            training=payload["training"],
            inference=payload.get("inference", SINKHORN_OT_SOLVER),
        )


@dataclass(frozen=True)
class RecipeReferenceSourceConfig:
    specification: str | None
    poscar: str
    allow_provisional_phase: bool = False
    template_id: str | None = None
    maximum_strain: str | float | None = None
    authoring_alias: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.specification is not None:
            object.__setattr__(
                self,
                "specification",
                _path(self.specification, field_name="reference.specification"),
            )
        object.__setattr__(self, "poscar", _path(self.poscar, field_name="reference.poscar"))
        if self.authoring_alias is not None:
            object.__setattr__(
                self,
                "authoring_alias",
                _reference_alias(
                    self.authoring_alias, field_name="reference.sources"
                ),
            )
        if type(self.allow_provisional_phase) is not bool:
            raise _error(
                "INVALID_PROVISIONAL_PHASE_FLAG",
                "allow_provisional_phase must be a bool",
                stage="recipe.reference",
                field="reference.allow_provisional_phase",
            )
        if self.template_id is not None and (
            type(self.template_id) is not str or not self.template_id
        ):
            raise _error(
                "INVALID_TEMPLATE_ID",
                "template_id must be a nonempty string",
                stage="recipe.reference",
                field="reference.template_id",
            )
        if self.specification is not None:
            if self.template_id is not None or self.maximum_strain is not None:
                raise _error(
                    "CONFLICTING_REFERENCE_MODE",
                    "explicit specification fields cannot be mixed with automatic reference fields",
                    stage="recipe.reference",
                    field="reference",
                )
        else:
            if self.allow_provisional_phase:
                raise _error(
                    "CONFLICTING_REFERENCE_MODE",
                    "allow_provisional_phase belongs only to explicit specifications; automatic references are explicitly provisional",
                    stage="recipe.reference",
                    field="reference.allow_provisional_phase",
                )
            value = "auto" if self.maximum_strain is None else self.maximum_strain
            if type(value) is str:
                if value != "auto":
                    raise _error(
                        "INVALID_MAXIMUM_STRAIN",
                        "automatic maximum_strain must be the exact token 'auto' or a positive finite number",
                        stage="recipe.reference",
                        field="reference.maximum_strain",
                        actual=value,
                    )
            elif isinstance(value, bool) or not isinstance(value, Real):
                raise _error(
                    "INVALID_MAXIMUM_STRAIN",
                    "maximum_strain must be 'auto' or a positive finite number; bool is forbidden",
                    stage="recipe.reference",
                    field="reference.maximum_strain",
                    actual=value,
                )
            else:
                value = float(value)
                if not math.isfinite(value) or value <= 0.0:
                    raise _error(
                        "INVALID_MAXIMUM_STRAIN",
                        "explicit maximum_strain must be finite and positive",
                        stage="recipe.reference",
                        field="reference.maximum_strain",
                        actual=value,
                    )
            object.__setattr__(self, "maximum_strain", value)

    @property
    def is_automatic(self) -> bool:
        return self.specification is None

    def to_dict(self) -> dict[str, Any]:
        if self.is_automatic:
            result: dict[str, Any] = {"poscar": self.poscar}
            if self.template_id is not None:
                result["template_id"] = self.template_id
            if self.maximum_strain != "auto":
                result["maximum_strain"] = self.maximum_strain
            return result
        return {
            "specification": self.specification,
            "poscar": self.poscar,
            "allow_provisional_phase": self.allow_provisional_phase,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, field_name: str) -> "RecipeReferenceSourceConfig":
        if not isinstance(value, Mapping):
            raise _error(
                "INVALID_REFERENCE_SOURCE",
                "reference source must be a POSCAR path or strict mapping",
                stage="recipe.reference",
                field=field_name,
            )
        automatic = "specification" not in value
        allowed = (
            frozenset({"poscar", "template_id", "maximum_strain"})
            if automatic
            else frozenset({"specification", "poscar", "allow_provisional_phase"})
        )
        payload = _strict_mapping(
            value,
            allowed=allowed,
            required=(frozenset({"poscar"}) if automatic else frozenset({"specification", "poscar"})),
            field_name=field_name,
        )
        return cls(specification=None, **dict(payload)) if automatic else cls(**dict(payload))

    @classmethod
    def from_value(cls, value: Any, *, field_name: str) -> "RecipeReferenceSourceConfig":
        if type(value) is str:
            return cls(specification=None, poscar=value)
        return cls.from_dict(value, field_name=field_name)


@dataclass(frozen=True)
class RecipeReferenceConfig:
    sources: tuple[RecipeReferenceSourceConfig, ...]
    default_template_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sources, Sequence) or isinstance(self.sources, (str, bytes)):
            raise _error(
                "INVALID_REFERENCE_SEQUENCE",
                "reference sources must be a deterministic sequence",
                stage="recipe.reference",
                field="reference",
            )
        sources = tuple(self.sources)
        if not sources or any(not isinstance(item, RecipeReferenceSourceConfig) for item in sources):
            raise _error(
                "INVALID_REFERENCE_SEQUENCE",
                "at least one canonical reference source is required",
                stage="recipe.reference",
                field="reference",
            )
        object.__setattr__(self, "sources", sources)
        aliases = tuple(source.authoring_alias for source in sources)
        if any(alias is not None for alias in aliases) and not all(
            alias is not None for alias in aliases
        ):
            raise _error(
                "CONFLICTING_REFERENCE_ALIAS_MODE",
                "aliased and unaliased reference sources cannot be mixed",
                stage="recipe.reference",
                field="reference.sources",
            )
        present_aliases = tuple(alias for alias in aliases if alias is not None)
        if len(present_aliases) != len(set(present_aliases)):
            raise _error(
                "DUPLICATE_REFERENCE_ALIAS",
                "reference aliases must be unique",
                stage="recipe.reference",
                field="reference.sources",
                actual=present_aliases,
            )
        automatic_count = sum(source.is_automatic for source in sources)
        if automatic_count not in (0, len(sources)):
            raise _error(
                "CONFLICTING_REFERENCE_MODE",
                "explicit and automatic reference sources cannot be mixed",
                stage="recipe.reference",
                field="reference",
            )
        if self.default_template_id is not None and (
            type(self.default_template_id) is not str or not self.default_template_id
        ):
            raise _error(
                "INVALID_DEFAULT_TEMPLATE",
                "default_template_id must be a nonempty string",
                stage="recipe.reference",
                field="reference.default_template_id",
            )

    def to_dict(self) -> dict[str, Any]:
        if all(source.authoring_alias is not None for source in self.sources):
            result: dict[str, Any] = {
                "sources": {
                    source.authoring_alias: source.to_dict()
                    for source in self.sources
                }
            }
            if self.default_template_id is not None:
                result["default_template_id"] = self.default_template_id
            return result
        if all(source.is_automatic for source in self.sources):
            simple = all(
                source.template_id is None and source.maximum_strain == "auto"
                for source in self.sources
            )
            if simple:
                values = [source.poscar for source in self.sources]
                return values[0] if len(values) == 1 else values
            result: dict[str, Any] = {
                "sources": [source.to_dict() for source in self.sources]
            }
            if self.default_template_id is not None:
                result["default_template_id"] = self.default_template_id
            return result
        if len(self.sources) == 1 and self.default_template_id is None:
            return self.sources[0].to_dict()
        return {
            "templates": [item.to_dict() for item in self.sources],
            "default_template_id": self.default_template_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RecipeReferenceConfig":
        if type(value) is str:
            return cls((RecipeReferenceSourceConfig.from_value(value, field_name="reference"),))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                raise _error(
                    "EMPTY_REFERENCE_SEQUENCE", "reference list must not be empty",
                    stage="recipe.reference", field="reference"
                )
            return cls(
                tuple(
                    RecipeReferenceSourceConfig.from_value(
                        item, field_name=f"reference[{index}]"
                    )
                    for index, item in enumerate(value)
                )
            )
        if not isinstance(value, Mapping):
            raise _error(
                "INVALID_RECIPE_SECTION", "reference must be a POSCAR path, list, or mapping",
                stage="recipe.schema", field="reference"
            )
        if "sources" in value:
            payload = _strict_mapping(
                value,
                allowed=frozenset({"sources", "default_template_id"}),
                required=frozenset({"sources"}),
                field_name="reference",
            )
            raw = payload["sources"]
            if isinstance(raw, Mapping):
                if not raw:
                    raise _error(
                        "EMPTY_REFERENCE_SEQUENCE",
                        "reference.sources must be a nonempty alias mapping",
                        stage="recipe.reference",
                        field="reference.sources",
                    )
                if payload.get("default_template_id") is not None:
                    raise _error(
                        "CONFLICTING_REFERENCE_ALIAS_MODE",
                        "aliased references use deterministic content-derived defaults",
                        stage="recipe.reference",
                        field="reference.default_template_id",
                    )
                sources = []
                for alias, item in raw.items():
                    canonical_alias = _reference_alias(
                        alias, field_name="reference.sources"
                    )
                    source = RecipeReferenceSourceConfig.from_dict(
                        item,
                        field_name=f"reference.sources[{canonical_alias!r}]",
                    )
                    sources.append(replace(source, authoring_alias=canonical_alias))
                return cls(tuple(sources))
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
                raise _error(
                    "EMPTY_REFERENCE_SEQUENCE", "reference.sources must be a nonempty sequence",
                    stage="recipe.reference", field="reference.sources"
                )
            sources = tuple(
                RecipeReferenceSourceConfig.from_value(
                    item, field_name=f"reference.sources[{index}]"
                )
                for index, item in enumerate(raw)
            )
            return cls(sources, default_template_id=payload.get("default_template_id"))
        if "templates" in value:
            payload = _strict_mapping(
                value,
                allowed=frozenset({"templates", "default_template_id"}),
                required=frozenset({"templates", "default_template_id"}),
                field_name="reference",
            )
            raw = payload["templates"]
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise _error(
                    "INVALID_REFERENCE_SEQUENCE", "templates must be a sequence",
                    stage="recipe.reference", field="reference.templates"
                )
            sources = tuple(
                RecipeReferenceSourceConfig.from_dict(
                    item, field_name=f"reference.templates[{index}]"
                )
                for index, item in enumerate(raw)
            )
            return cls(sources=sources, default_template_id=payload["default_template_id"])
        source = RecipeReferenceSourceConfig.from_dict(value, field_name="reference")
        return cls((source,))


@dataclass(frozen=True)
class RecipeDataSourceConfig:
    path: str
    template_id: str | None = None
    automatic_template_assignment: bool = False
    reference_alias: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, field_name="data.path"))
        if self.template_id is not None and (type(self.template_id) is not str or not self.template_id):
            raise _error(
                "INVALID_TEMPLATE_SELECTOR", "template_id must be a nonempty string",
                stage="recipe.data", field="data.template_id"
            )
        if self.reference_alias is not None:
            object.__setattr__(
                self,
                "reference_alias",
                _reference_alias(
                    self.reference_alias, field_name="data.reference"
                ),
            )
        if type(self.automatic_template_assignment) is not bool:
            raise _error(
                "INVALID_TEMPLATE_SELECTOR", "automatic_template_assignment must be a bool",
                stage="recipe.data", field="data.automatic_template_assignment"
            )
        active = sum(
            (
                self.template_id is not None,
                self.automatic_template_assignment,
                self.reference_alias is not None,
            )
        )
        if active > 1:
            raise _error(
                "CONFLICTING_TEMPLATE_SELECTOR",
                "reference alias, exact template, and automatic selection conflict",
                stage="recipe.data",
                field="data.reference,template_id,automatic_template_assignment",
            )

    def to_dict(self) -> dict[str, Any]:
        if self.reference_alias is not None:
            return {"file": self.path, "reference": self.reference_alias}
        result: dict[str, Any] = {"path": self.path}
        if self.template_id is not None:
            result["template_id"] = self.template_id
        if self.automatic_template_assignment:
            result["automatic_template_assignment"] = True
        return result

    @classmethod
    def from_value(cls, value: Any, *, field_name: str) -> "RecipeDataSourceConfig":
        if type(value) is str:
            return cls(value)
        if isinstance(value, Mapping) and (
            "file" in value or "reference" in value
        ):
            payload = _strict_mapping(
                value,
                allowed=frozenset({"file", "reference"}),
                required=frozenset({"file", "reference"}),
                field_name=field_name,
            )
            return cls(
                path=payload["file"], reference_alias=payload["reference"]
            )
        payload = _strict_mapping(
            value,
            allowed=frozenset({"path", "template_id", "automatic_template_assignment"}),
            required=frozenset({"path"}),
            field_name=field_name,
        )
        return cls(**dict(payload))


def _recipe_data_sequence(value: Any, *, field_name: str) -> tuple[RecipeDataSourceConfig, ...]:
    if type(value) is str or isinstance(value, Mapping):
        return (RecipeDataSourceConfig.from_value(value, field_name=field_name),)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise _error(
            "INVALID_DATA_SEQUENCE", "data split must be a path or deterministic sequence",
            stage="recipe.data", field=field_name
        )
    result = tuple(
        RecipeDataSourceConfig.from_value(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise _error(
            "EMPTY_DATA_SPLIT", "data split must not be empty",
            stage="recipe.data", field=field_name
        )
    return result


@dataclass(frozen=True)
class RecipeDataConfig:
    train: tuple[RecipeDataSourceConfig, ...]
    validation: tuple[RecipeDataSourceConfig, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", tuple(self.train))
        object.__setattr__(self, "validation", tuple(self.validation))
        if not self.train or not self.validation:
            raise _error(
                "EMPTY_DATA_SPLIT", "train and validation must be nonempty",
                stage="recipe.data", field="data"
            )
        for split, sources in (("train", self.train), ("validation", self.validation)):
            aliased = tuple(source.reference_alias is not None for source in sources)
            if any(aliased) and not all(aliased):
                raise _error(
                    "MIXED_DATA_BINDING_MODE",
                    "one split cannot mix explicit alias bindings with legacy or automatic entries",
                    stage="recipe.data",
                    field=f"data.{split}",
                )

    def to_dict(self) -> dict[str, Any]:
        def encode(items: tuple[RecipeDataSourceConfig, ...]) -> Any:
            if (
                len(items) == 1
                and items[0].template_id is None
                and items[0].reference_alias is None
                and not items[0].automatic_template_assignment
            ):
                return items[0].path
            return [item.to_dict() for item in items]
        return {"train": encode(self.train), "validation": encode(self.validation)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeDataConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"train", "validation"}),
            required=frozenset({"train", "validation"}),
            field_name="data",
        )
        return cls(
            _recipe_data_sequence(payload["train"], field_name="data.train"),
            _recipe_data_sequence(payload["validation"], field_name="data.validation"),
        )


@dataclass(frozen=True)
class RecipeLossConfig:
    energy_weight: float = 1.0
    forces_weight: float = 100.0
    stress_weight: float = 0.0
    energy_scale: float = field(default=1.0, kw_only=True)
    force_scale: float = field(default=1.0, kw_only=True)
    stress_scale: float = field(default=1.0, kw_only=True)
    energy_normalization: str = field(default="per_structure", kw_only=True)
    _provided_fields: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("energy_scale", "force_scale", "stress_scale"):
            object.__setattr__(self, name, _positive_real(getattr(self, name), field_name=f"loss.{name}"))
        if self.energy_normalization not in ("per_structure", "per_atom"):
            raise _error("INVALID_ENERGY_NORMALIZATION", "use per_structure or per_atom",
                         stage="recipe.loss", field="loss.energy_normalization")
        for name in ("energy_weight", "forces_weight", "stress_weight"):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), field_name=f"loss.{name}")
            )
        if not any(getattr(self, name) > 0.0 for name in ("energy_weight", "forces_weight", "stress_weight")):
            raise _error(
                "NO_ACTIVE_LOSS_TERM", "at least one loss weight must be positive",
                stage="recipe.loss", field="loss"
            )
        object.__setattr__(self, "_provided_fields", tuple(sorted(self._provided_fields)))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "energy_weight": self.energy_weight,
            "forces_weight": self.forces_weight,
            "stress_weight": self.stress_weight,
        }
        # Preserve legacy recipe serialization when the new options are omitted.
        for name, default in (("energy_scale", 1.0), ("force_scale", 1.0),
                              ("stress_scale", 1.0), ("energy_normalization", "per_structure")):
            if getattr(self, name) != default or name in self._provided_fields:
                result[name] = getattr(self, name)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeLossConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"energy_weight", "forces_weight", "stress_weight",
                               "energy_scale", "force_scale", "stress_scale", "energy_normalization"}),
            field_name="loss",
        )
        return cls(**dict(payload), _provided_fields=tuple(payload))


@dataclass(frozen=True)
class RecipeTrainingConfig:
    max_epochs: int
    batch_size: int = 4
    validation_batch_size: int | None = None
    learning_rate: float = 1.0e-3
    gradient_clip_norm: float | None = field(default=None, kw_only=True)
    scheduler: SchedulerConfig | None = field(default=None, kw_only=True)
    early_stopping_patience: int | None = field(
        default=DEFAULT_EARLY_STOPPING_PATIENCE, kw_only=True
    )
    early_stopping_relative_delta: float = field(
        default=DEFAULT_EARLY_STOPPING_RELATIVE_DELTA, kw_only=True
    )
    _provided_fields: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.gradient_clip_norm is not None:
            object.__setattr__(self, "gradient_clip_norm", _positive_real(
                self.gradient_clip_norm, field_name="training.gradient_clip_norm"))
        if self.scheduler is not None and not isinstance(self.scheduler, SchedulerConfig):
            raise _error("INVALID_RECIPE_SCHEDULER", "scheduler must be a SchedulerConfig",
                         stage="recipe.training", field="training.scheduler")
        if self.scheduler is not None and (self.scheduler.monitor != "total_loss" or self.scheduler.mode != "min"):
            raise _error("INVALID_RECIPE_SCHEDULER", "recipe scheduler must monitor total_loss in min mode, matching model selection",
                         stage="recipe.training", field="training.scheduler")
        for name in ("max_epochs", "batch_size"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), field_name=f"training.{name}"))
        validation = self.batch_size if self.validation_batch_size is None else _positive_int(
            self.validation_batch_size, field_name="training.validation_batch_size"
        )
        object.__setattr__(self, "validation_batch_size", validation)
        object.__setattr__(self, "learning_rate", _positive_real(self.learning_rate, field_name="training.learning_rate"))
        patience = self.early_stopping_patience
        if patience is not None:
            if isinstance(patience, bool) or not isinstance(patience, Integral):
                raise _error(
                    "INVALID_RECIPE_INTEGER",
                    "value must be an integer and bool is forbidden",
                    stage="recipe.validation",
                    field="training.early_stopping_patience",
                    actual=patience,
                )
            patience = int(patience)
            if patience < 0:
                raise _error(
                    "INVALID_RECIPE_INTEGER",
                    "value must be nonnegative",
                    stage="recipe.validation",
                    field="training.early_stopping_patience",
                    actual=patience,
                )
        object.__setattr__(self, "early_stopping_patience", patience)
        relative_delta = _finite_nonnegative(
            self.early_stopping_relative_delta,
            field_name="training.early_stopping_relative_delta",
        )
        if relative_delta >= 1.0:
            raise _error(
                "INVALID_EARLY_STOPPING_RELATIVE_DELTA",
                "relative improvement threshold must be smaller than 1",
                stage="recipe.validation",
                field="training.early_stopping_relative_delta",
                actual=relative_delta,
            )
        object.__setattr__(
            self, "early_stopping_relative_delta", relative_delta
        )
        object.__setattr__(self, "_provided_fields", tuple(sorted(self._provided_fields)))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "batch_size": self.batch_size,
            "validation_batch_size": self.validation_batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
        }
        if self.gradient_clip_norm is not None or "gradient_clip_norm" in self._provided_fields:
            result["gradient_clip_norm"] = self.gradient_clip_norm
        if self.scheduler is not None:
            result["scheduler"] = self.scheduler.to_dict()
        if (
            self.early_stopping_patience is not None
            or "early_stopping_patience" in self._provided_fields
        ):
            result["early_stopping_patience"] = self.early_stopping_patience
        if (
            self.early_stopping_relative_delta != 0.0
            or "early_stopping_relative_delta" in self._provided_fields
        ):
            result["early_stopping_relative_delta"] = (
                self.early_stopping_relative_delta
            )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeTrainingConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({
                "batch_size",
                "validation_batch_size",
                "max_epochs",
                "learning_rate",
                "gradient_clip_norm",
                "scheduler",
                "early_stopping_patience",
                "early_stopping_relative_delta",
            }),
            required=frozenset({"max_epochs"}),
            field_name="training",
        )
        values = dict(payload)
        if "scheduler" in values:
            try:
                values["scheduler"] = SchedulerConfig.from_dict(values["scheduler"])
            except (TypeError, ValueError) as error:
                raise _error("INVALID_RECIPE_SCHEDULER", str(error),
                             stage="recipe.training", field="training.scheduler") from error
        return cls(**values, _provided_fields=tuple(payload))


@dataclass(frozen=True)
class RecipeRuntimeConfig:
    seed: int
    device: str = "cpu"
    dtype: str = "float64"
    _provided_fields: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise _error(
                "INVALID_RUNTIME_SEED", "seed must be an integer and bool is forbidden",
                stage="recipe.runtime", field="runtime.seed", actual=self.seed
            )
        object.__setattr__(self, "seed", int(self.seed))
        if type(self.device) is not str or re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", self.device) is None:
            raise _error(
                "INVALID_RUNTIME_DEVICE", "device must be cpu, cuda, or cuda:N",
                stage="recipe.runtime", field="runtime.device", actual=self.device
            )
        if self.dtype not in ("float32", "float64"):
            raise _error(
                "INVALID_RUNTIME_DTYPE", "dtype must be float32 or float64",
                stage="recipe.runtime", field="runtime.dtype", actual=self.dtype
            )
        object.__setattr__(self, "_provided_fields", tuple(sorted(self._provided_fields)))

    def to_dict(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype, "seed": self.seed}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeRuntimeConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"device", "dtype", "seed"}),
            required=frozenset({"seed"}),
            field_name="runtime",
        )
        return cls(**dict(payload), _provided_fields=tuple(payload))


@dataclass(frozen=True)
class TrainingRecipeConfig:
    schema_version: str
    model: RecipeModelConfig
    reference: RecipeReferenceConfig
    data: RecipeDataConfig
    ot_solver: RecipeOTSolverConfig
    loss: RecipeLossConfig
    baseline: str
    training: RecipeTrainingConfig
    runtime: RecipeRuntimeConfig
    radii: InteractionRadiusConfig = field(default_factory=InteractionRadiusConfig)
    name: str | None = None
    output_directory: str | None = None
    source_path: str | None = field(default=None, repr=False, compare=False)
    _provided_fields: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_RECIPE_SCHEMA_VERSION:
            raise _error(
                "UNSUPPORTED_RECIPE_SCHEMA", "unsupported training recipe schema",
                stage="recipe.schema", field="schema_version",
                expected=TRAINING_RECIPE_SCHEMA_VERSION, actual=self.schema_version
            )
        for name, cls in (
            ("model", RecipeModelConfig), ("reference", RecipeReferenceConfig),
            ("data", RecipeDataConfig), ("ot_solver", RecipeOTSolverConfig),
            ("loss", RecipeLossConfig),
            ("training", RecipeTrainingConfig), ("runtime", RecipeRuntimeConfig),
            ("radii", InteractionRadiusConfig),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be {cls.__name__}")
        if (self.name is None) == (self.output_directory is None):
            raise _error(
                "CONFLICTING_OUTPUT_IDENTITY",
                "provide exactly one of name or output_directory",
                stage="recipe.validation", field="name,output_directory"
            )
        if self.name is not None and (_NAME.fullmatch(self.name) is None or self.name in (".", "..")):
            raise _error(
                "INVALID_RECIPE_NAME", "name must be a simple filesystem-safe identifier",
                stage="recipe.validation", field="name", actual=self.name
            )
        if self.output_directory is not None:
            object.__setattr__(self, "output_directory", _path(self.output_directory, field_name="output_directory"))
        if self.baseline not in ("zero", "fit_full_rank", "minimum_norm"):
            raise _error(
                "UNSUPPORTED_BASELINE_POLICY",
                "baseline must be zero, fit_full_rank, or minimum_norm",
                stage="recipe.baseline", field="baseline", actual=self.baseline
            )
        reference_aliases = tuple(
            source.authoring_alias
            for source in self.reference.sources
            if source.authoring_alias is not None
        )
        data_sources = self.data.train + self.data.validation
        data_aliases = tuple(
            source.reference_alias
            for source in data_sources
            if source.reference_alias is not None
        )
        if reference_aliases:
            if len(data_aliases) != len(data_sources):
                raise _error(
                    "MISSING_EXPLICIT_REFERENCE_BINDING",
                    "aliased references require every train and validation source to name a reference alias",
                    stage="recipe.data",
                    field="data",
                )
            unknown = tuple(sorted(set(data_aliases) - set(reference_aliases)))
            if unknown:
                raise _error(
                    "UNKNOWN_REFERENCE_ALIAS",
                    "data source names an unknown reference alias",
                    stage="recipe.data",
                    field="data.reference",
                    actual=unknown,
                )
            unused = tuple(sorted(set(reference_aliases) - set(data_aliases)))
            if unused:
                raise _error(
                    "UNUSED_REFERENCE_ALIAS",
                    "every declared reference alias must be used by data",
                    stage="recipe.data",
                    field="reference.sources",
                    actual=unused,
                )
        elif data_aliases:
            raise _error(
                "UNKNOWN_REFERENCE_ALIAS",
                "data reference aliases require reference.sources to be an alias mapping",
                stage="recipe.data",
                field="data.reference",
                actual=tuple(sorted(set(data_aliases))),
            )
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))
        object.__setattr__(self, "_provided_fields", tuple(sorted(self._provided_fields)))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "reference": self.reference.to_dict(),
            "data": self.data.to_dict(),
            "ot_solver": self.ot_solver.to_dict(),
            "loss": self.loss.to_dict(),
            "baseline": self.baseline,
            "radii": {"r_ot": self.radii.r_ot, "r_mp": self.radii.r_mp},
            "training": self.training.to_dict(),
            "runtime": self.runtime.to_dict(),
        }
        if self.name is not None:
            result["name"] = self.name
        else:
            result["output_directory"] = self.output_directory
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRecipeConfig":
        allowed = frozenset(
            {"schema_version", "name", "output_directory", "model", "reference", "data", "ot_solver", "loss", "baseline", "radii", "training", "runtime"}
        )
        required = frozenset({"schema_version", "model", "reference", "data", "loss", "baseline", "training", "runtime"})
        payload = _strict_mapping(value, allowed=allowed, required=required, field_name="recipe")
        reference = RecipeReferenceConfig.from_dict(payload["reference"])
        if "ot_solver" not in payload and not all(
            source.is_automatic for source in reference.sources
        ):
            raise _error(
                "MISSING_RECIPE_KEY",
                "explicit reference recipes require ot_solver",
                stage="recipe.schema",
                field="ot_solver",
            )
        radii_value = payload.get("radii", {"r_ot": 4.0, "r_mp": 3.0})
        radii_payload = _strict_mapping(
            radii_value,
            allowed=frozenset({"r_ot", "r_mp"}),
            required=frozenset({"r_ot", "r_mp"}),
            field_name="radii",
        )
        try:
            radii = InteractionRadiusConfig(r_ot=radii_payload["r_ot"], r_mp=radii_payload["r_mp"])
        except Exception as error:
            raise _error(
                getattr(error, "reason_code", "INVALID_RADIUS_CONFIG"),
                "recipe radii are invalid", stage="recipe.radii", field="radii",
                original_error=error
            ) from error
        return cls(
            schema_version=payload["schema_version"],
            name=payload.get("name"), output_directory=payload.get("output_directory"),
            model=RecipeModelConfig.from_dict(payload["model"]),
            reference=reference,
            data=RecipeDataConfig.from_dict(payload["data"]),
            ot_solver=(
                RecipeOTSolverConfig(
                    training=SINKHORN_OT_SOLVER,
                    inference=SINKHORN_OT_SOLVER,
                )
                if "ot_solver" not in payload
                else RecipeOTSolverConfig.from_dict(payload["ot_solver"])
            ),
            loss=RecipeLossConfig.from_dict(payload["loss"]),
            baseline=payload["baseline"], radii=radii,
            training=RecipeTrainingConfig.from_dict(payload["training"]),
            runtime=RecipeRuntimeConfig.from_dict(payload["runtime"]),
            _provided_fields=tuple(payload),
        )

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def content_fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


_BUILDER_KEYS = frozenset(
    {"template_id", "strict_domain", "site_type_ids", "graph_cutoff", "graph_skin", "maximum_strain", "minimum_edge_length", "avg_num_neighbors", "expected_active_degree", "expected_candidate_degree", "expected_stabilizer_size", "canonical_tolerance", "metric_tolerance", "template_convention_version", "builder_convention_version"}
)
_PHASE_KEYS = frozenset(
    {"modes", "mode_weights", "site_type_alignment_weights", "channel_weights", "approval_status", "convention_version", "floating_dtype"}
)
_POLICY_KEYS = frozenset(
    {"template_id", "template_fingerprint", "candidate_offsets", "candidate_dtype", "phase_step_schedule", "phase_damping_schedule", "minimum_objective_gap_absolute", "minimum_cross_amplitude_absolute", "minimum_atomic_amplitude_absolute", "minimum_reference_amplitude_absolute", "minimum_curvature", "maximum_condition", "maximum_gradient_norm", "equivalence_tolerance", "transport_path", "convention_version", "content_fingerprint"}
)


@dataclass(frozen=True, eq=False)
class ReferenceSpecificationConfig:
    """Versioned wrapper around existing canonical builder/phase/policy APIs."""

    builder: ReferenceTemplateBuilderConfig
    phase_specification: PhaseSpecification
    species_alignment_weights: tuple[tuple[float, ...], ...]
    poscar_sha256: str
    evaluation_policy: EvaluationPolicy | None = None
    schema_version: str = REFERENCE_SPECIFICATION_SCHEMA_VERSION
    declared_content_fingerprint: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_SPECIFICATION_SCHEMA_VERSION:
            raise _error(
                "UNSUPPORTED_REFERENCE_SPECIFICATION_SCHEMA",
                "unsupported reference specification schema",
                stage="recipe.reference_specification", field="schema_version",
                actual=self.schema_version
            )
        if not isinstance(self.builder, ReferenceTemplateBuilderConfig):
            raise TypeError("builder must be ReferenceTemplateBuilderConfig")
        if not isinstance(self.phase_specification, PhaseSpecification):
            raise TypeError("phase_specification must be PhaseSpecification")
        if self.evaluation_policy is not None and not isinstance(self.evaluation_policy, EvaluationPolicy):
            raise TypeError("evaluation_policy must be EvaluationPolicy or None")
        rows = tuple(tuple(_finite_real(value, field_name="species_alignment_weights") for value in row) for row in self.species_alignment_weights)
        if not rows or len({len(row) for row in rows}) != 1 or not rows[0]:
            raise _error(
                "INVALID_SPECIES_ALIGNMENT", "species_alignment_weights must be a nonempty rectangular matrix",
                stage="recipe.reference_specification", field="species_alignment_weights"
            )
        object.__setattr__(self, "species_alignment_weights", rows)
        if _SHA256.fullmatch(self.poscar_sha256) is None:
            raise _error(
                "INVALID_POSCAR_FINGERPRINT", "poscar_sha256 must be lowercase SHA-256",
                stage="recipe.reference_specification", field="poscar_sha256", actual=self.poscar_sha256
            )
        if len(rows) != len(self.builder.strict_domain.species_vocabulary) or any(
            len(row) != self.phase_specification.num_channels for row in rows
        ):
            raise _error(
                "SPECIES_ALIGNMENT_SHAPE_MISMATCH", "alignment shape must be [species, phase_channels]",
                stage="recipe.reference_specification", field="species_alignment_weights"
            )
        if self.evaluation_policy is not None and self.evaluation_policy.template_id != self.builder.template_id:
            raise _error(
                "POLICY_TEMPLATE_MISMATCH", "evaluation policy template differs from builder",
                stage="recipe.reference_specification", field="evaluation_policy.template_id"
            )
        declared = self.declared_content_fingerprint
        if declared is not None and declared != self.content_fingerprint:
            raise _error(
                "REFERENCE_SPECIFICATION_FINGERPRINT_MISMATCH",
                "declared reference specification fingerprint does not match content",
                stage="recipe.reference_specification", field="content_fingerprint",
                expected=self.content_fingerprint, actual=declared
            )

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "builder": self.builder.to_dict(),
            "phase_specification": self.phase_specification.to_dict(),
            "evaluation_policy": None if self.evaluation_policy is None else self.evaluation_policy.to_dict(),
            "species_alignment_weights": [list(row) for row in self.species_alignment_weights],
            "poscar_sha256": self.poscar_sha256,
        }

    @property
    def content_fingerprint(self) -> str:
        return _fingerprint(self.semantic_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self.semantic_dict()
        result["content_fingerprint"] = self.content_fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReferenceSpecificationConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"schema_version", "builder", "phase_specification", "evaluation_policy", "species_alignment_weights", "poscar_sha256", "content_fingerprint"}),
            required=frozenset({"schema_version", "builder", "phase_specification", "evaluation_policy", "species_alignment_weights", "poscar_sha256", "content_fingerprint"}),
            field_name="reference_specification",
        )
        _strict_mapping(payload["builder"], allowed=_BUILDER_KEYS, required=_BUILDER_KEYS, field_name="reference_specification.builder")
        _strict_mapping(payload["phase_specification"], allowed=_PHASE_KEYS, required=_PHASE_KEYS, field_name="reference_specification.phase_specification")
        try:
            policy = None
            if payload["evaluation_policy"] is not None:
                _strict_mapping(payload["evaluation_policy"], allowed=_POLICY_KEYS, required=_POLICY_KEYS, field_name="reference_specification.evaluation_policy")
                policy = EvaluationPolicy.from_dict(payload["evaluation_policy"])
            builder = ReferenceTemplateBuilderConfig.from_dict(payload["builder"])
            phase = PhaseSpecification.from_dict(payload["phase_specification"])
        except TrainingRecipeError:
            raise
        except Exception as error:
            raise _error(
                "INVALID_REFERENCE_SPECIFICATION", "canonical reference configuration is invalid",
                stage="recipe.reference_specification", original_error=error
            ) from error
        matrix_value = payload["species_alignment_weights"]
        if not isinstance(matrix_value, Sequence) or isinstance(matrix_value, (str, bytes)):
            raise _error(
                "INVALID_SPECIES_ALIGNMENT", "species alignment must be a sequence",
                stage="recipe.reference_specification", field="species_alignment_weights"
            )
        return cls(
            builder=builder, phase_specification=phase,
            evaluation_policy=policy,
            species_alignment_weights=tuple(tuple(row) for row in matrix_value),
            poscar_sha256=payload["poscar_sha256"],
            schema_version=payload["schema_version"],
            declared_content_fingerprint=payload["content_fingerprint"],
        )


@dataclass(frozen=True)
class RecipeResolutionManifest:
    recipe_fingerprint: str
    compiled_config_fingerprint: str
    reference_specification_fingerprints: tuple[tuple[str, str], ...]
    field_origins: tuple[tuple[str, str], ...]
    paths: tuple[tuple[str, str, str], ...]
    training_ot_solver: str
    inference_ot_solver: str
    sinkhorn_iterations: int
    sinkhorn_residual_tolerance: float
    nonconvergence_policy: str = "fail-fast"
    resolver_version: str = RECIPE_RESOLVER_VERSION
    recipe_schema_version: str = TRAINING_RECIPE_SCHEMA_VERSION
    preset_versions: tuple[tuple[str, str], ...] = (
        ("model", SYMMETRIC_MODEL_DEFAULTS_VERSION),
        ("radii", RADIUS_DERIVATION_VERSION),
        ("training", TRAINING_DEFAULTS_VERSION),
    )
    preset_fingerprints: tuple[tuple[str, str], ...] = ()
    automatic_reference_fingerprint: str | None = None
    automatic_reference_certificates: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for value in (self.recipe_fingerprint, self.compiled_config_fingerprint):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("manifest fingerprints must be lowercase SHA-256")
        object.__setattr__(self, "reference_specification_fingerprints", tuple(sorted(self.reference_specification_fingerprints)))
        object.__setattr__(self, "field_origins", tuple(sorted(self.field_origins)))
        object.__setattr__(self, "paths", tuple(sorted(self.paths)))
        object.__setattr__(self, "preset_versions", tuple(sorted(self.preset_versions)))
        object.__setattr__(self, "preset_fingerprints", tuple(sorted(self.preset_fingerprints)))
        object.__setattr__(
            self,
            "automatic_reference_certificates",
            tuple(sorted(self.automatic_reference_certificates)),
        )
        if self.automatic_reference_fingerprint is not None and _SHA256.fullmatch(
            self.automatic_reference_fingerprint
        ) is None:
            raise ValueError("automatic reference fingerprint must be lowercase SHA-256")
        RecipeOTSolverConfig(
            training=self.training_ot_solver,
            inference=self.inference_ot_solver,
        )
        object.__setattr__(
            self,
            "sinkhorn_iterations",
            _positive_int(self.sinkhorn_iterations, field_name="sinkhorn_iterations"),
        )
        tolerance = _positive_real(
            self.sinkhorn_residual_tolerance,
            field_name="sinkhorn_residual_tolerance",
        )
        object.__setattr__(self, "sinkhorn_residual_tolerance", tolerance)
        if self.nonconvergence_policy != "fail-fast":
            raise ValueError("nonconvergence_policy must be fail-fast")

    def semantic_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": "refsite_training_recipe_resolution_manifest_v1",
            "recipe_schema_version": self.recipe_schema_version,
            "recipe_fingerprint": self.recipe_fingerprint,
            "resolver_version": self.resolver_version,
            "preset_versions": dict(self.preset_versions),
            "preset_fingerprints": dict(self.preset_fingerprints),
            "reference_specification_fingerprints": dict(self.reference_specification_fingerprints),
            "field_origins": dict(self.field_origins),
            "ot_solver": {
                "training": self.training_ot_solver,
                "inference": self.inference_ot_solver,
                "inference_application": "prediction/evaluation call-time preference",
                "sinkhorn_iterations": self.sinkhorn_iterations,
                "sinkhorn_residual_tolerance": self.sinkhorn_residual_tolerance,
                "nonconvergence_policy": self.nonconvergence_policy,
            },
            "paths": [
                {"field": name, "original": original, "resolved": resolved}
                for name, original, resolved in self.paths
            ],
            "compiled_config_fingerprint": self.compiled_config_fingerprint,
        }
        # Do not perturb the byte-level manifest contract of explicit recipes.
        if self.automatic_reference_fingerprint is not None:
            result["automatic_reference_preparation"] = {
                "scope": "dataset_bounded",
                "approval_status": "provisional",
                "content_fingerprint": self.automatic_reference_fingerprint,
                "certificates": {
                    template_id: certificate
                    for template_id, certificate in self.automatic_reference_certificates
                },
            }
        return result

    @property
    def content_fingerprint(self) -> str:
        return _fingerprint(self.semantic_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self.semantic_dict()
        result["content_fingerprint"] = self.content_fingerprint
        return result


@dataclass(frozen=True)
class ResolvedTrainingRecipe:
    config: TrainingRunConfig
    manifest: RecipeResolutionManifest
    recipe: TrainingRecipeConfig
    automatic_reference_preparation: Any = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.config.schema_version != TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2:
            raise ValueError("recipe resolution must produce canonical schema v2")
        if self.config.config_fingerprint != self.manifest.compiled_config_fingerprint:
            raise ValueError("manifest and compiled canonical config disagree")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "recipe_summary": {
                "correlation_method": self.recipe.model.correlation_method,
                "maximum_correlation_order": (
                    self.recipe.model.correlation
                    if self.recipe.model.correlation_method
                    == SYMMETRIC_CORRELATION_METHOD
                    else 3
                ),
                "training_ot_solver": self.manifest.training_ot_solver,
                "inference_ot_solver": self.manifest.inference_ot_solver,
                "inference_application": "prediction/evaluation call-time preference",
                "training_batch_size": self.config.data.batch_size,
                "validation_batch_size": self.config.data.effective_validation_batch_size,
            },
            "compiled_config": self.config.to_dict(),
            "resolution_manifest": self.manifest.to_dict(),
        }
        if self.automatic_reference_preparation is not None:
            result["reference_preparation"] = (
                self.automatic_reference_preparation.to_dict()
            )
        return result


_MODEL_DEFAULTS = {
    "initialization_seed": 0,
    "num_interactions": 2,
    "correlation": 3,
    "hidden_channels": 8,
    "max_L": 2,
    "n_radial": 3,
    "ell_feature": 1.0,
    "site_type_embedding_dim": 2,
    "radial_feature_dim": 3,
    "radial_hidden_dims": [8],
    "edge_length_scale": "radii.r_mp",
    "readout_hidden": 16,
    "energy_scale": 1.0,
    "epsilon_ot": 0.5,
    "ell_ot": 1.5,
    "train_sinkhorn_iterations": 256,
    "phase_steps": [0.7, 0.8, 0.9, 1.0],
    "phase_damping": [2.0, 1.0, 0.5, 0.2],
    "eval_sinkhorn_warmup_iterations": 16,
    "transport_backend": "dense",
    "candidate_backend": "dense",
}
_TRAINING_DEFAULTS = {
    "batch_size": 4,
    "validation_batch_size": "batch_size",
    "learning_rate": 1.0e-3,
    "shuffle": False,
    "optimizer": "adamw",
    "weight_decay": 0.0,
    "train_solver": TRAIN_FIXED,
    "validation_solver": TRAIN_FIXED,
    "scheduler": "none",
    "monitor": "total_loss",
    "mode": "min",
    "early_stopping_patience": DEFAULT_EARLY_STOPPING_PATIENCE,
    "save_every_epoch": True,
    "energy_scale": 1.0,
    "force_scale": 1.0,
    "stress_scale": 1.0,
    "energy_normalization": "per_structure",
}
_RADIUS_DEFAULTS = {"r_ot": 4.0, "r_mp": 3.0, "derivation": RADIUS_DERIVATION_VERSION}


def _feature_irreps(species_count: int, radial_count: int, maximum_l: int) -> str:
    terms = [f"{species_count}x0e"]
    for angular in range(maximum_l + 1):
        terms.append(
            f"{species_count * radial_count}x{angular}{'e' if angular % 2 == 0 else 'o'}"
        )
    return "+".join(terms)


def _compile_data_source(
    source: RecipeDataSourceConfig,
    *,
    template_ids: tuple[str, ...],
    reference_aliases: Mapping[str, str] | None = None,
    automatic_reference_mode: bool = False,
) -> TrainingDataSourceConfig:
    if source.reference_alias is not None:
        if reference_aliases is None or source.reference_alias not in reference_aliases:
            raise _error(
                "UNKNOWN_REFERENCE_ALIAS",
                "data source names an unknown reference alias",
                stage="recipe.compile",
                field="data.reference",
                actual=source.reference_alias,
            )
        return TrainingDataSourceConfig(
            path=source.path,
            template_id=reference_aliases[source.reference_alias],
            reference_alias=source.reference_alias,
        )
    if source.template_id is not None:
        if source.template_id not in template_ids:
            raise _error(
                "UNKNOWN_TEMPLATE_ID", "data source names an unknown template",
                stage="recipe.compile", field="data.template_id", actual=source.template_id
            )
        return TrainingDataSourceConfig(path=source.path, template_id=source.template_id)
    if source.automatic_template_assignment or automatic_reference_mode:
        return TrainingDataSourceConfig(path=source.path, automatic_template_assignment=True)
    if len(template_ids) != 1:
        raise _error(
            "MISSING_TEMPLATE_ASSIGNMENT",
            "multi-template recipes require template_id or explicit automatic assignment for every data source",
            stage="recipe.compile", field="data"
        )
    return TrainingDataSourceConfig(path=source.path, template_id=template_ids[0])


def _baseline_config(policy: str) -> AtomicBaselineConfig | None:
    if policy == "zero":
        return None
    return AtomicBaselineConfig(
        weighting="per_structure",
        rank_policy="error" if policy == "fit_full_rank" else "minimum_norm",
    )


def _origin(provided: Sequence[str], name: str) -> str:
    return "user" if name in provided else "preset"


def _resolved_text(original: str, base: Path) -> str:
    candidate = Path(original)
    return str((candidate if candidate.is_absolute() else base / candidate).resolve(strict=False))


def _compile_training_recipe_impl(
    recipe: TrainingRecipeConfig,
    reference_specifications: Sequence[ReferenceSpecificationConfig],
    *,
    overrides: TrainingRunConfigOverrides | None = None,
    cli_cwd: str | os.PathLike[str] | None = None,
) -> ResolvedTrainingRecipe:
    """Purely compile canonical objects; no source file is read or model built."""

    if not isinstance(recipe, TrainingRecipeConfig):
        raise TypeError("recipe must be TrainingRecipeConfig")
    specifications = tuple(reference_specifications)
    if len(specifications) != len(recipe.reference.sources) or any(
        not isinstance(item, ReferenceSpecificationConfig) for item in specifications
    ):
        raise _error(
            "REFERENCE_SPECIFICATION_COUNT_MISMATCH",
            "one loaded reference specification is required per recipe source",
            stage="recipe.compile", field="reference"
        )
    template_ids = tuple(item.builder.template_id for item in specifications)
    if len(set(template_ids)) != len(template_ids):
        raise _error(
            "DUPLICATE_TEMPLATE_ID", "reference specifications contain duplicate template IDs",
            stage="recipe.compile", field="reference"
        )
    reference_aliases = {
        source.authoring_alias: specification.builder.template_id
        for source, specification in zip(recipe.reference.sources, specifications)
        if source.authoring_alias is not None
    }
    default_template = recipe.reference.default_template_id
    if default_template is None:
        if len(template_ids) != 1 and not all(
            source.is_automatic for source in recipe.reference.sources
        ):
            raise _error(
                "MISSING_DEFAULT_TEMPLATE", "multi-template recipe requires default_template_id",
                stage="recipe.compile", field="reference.default_template_id"
            )
        default_template = sorted(template_ids)[0]
    if default_template not in template_ids:
        raise _error(
            "DEFAULT_TEMPLATE_MISSING", "default_template_id does not name a loaded specification",
            stage="recipe.compile", field="reference.default_template_id", actual=default_template
        )

    if recipe.ot_solver.inference == SINKHORN_NEWTON_KRYLOV_OT_SOLVER:
        for specification in specifications:
            policy = specification.evaluation_policy
            if policy is None:
                raise _error(
                    "INFERENCE_EVALUATION_POLICY_REQUIRED",
                    "sinkhorn_newton_krylov inference requires a compatible evaluation policy for every reference template",
                    stage="recipe.ot_solver",
                    field="ot_solver.inference",
                    actual=specification.builder.template_id,
                )
            try:
                policy.validate_fingerprint()
            except Exception as error:
                raise _error(
                    "INFERENCE_EVALUATION_POLICY_MISMATCH",
                    "the reference evaluation policy failed its content fingerprint check",
                    stage="recipe.ot_solver",
                    field="ot_solver.inference",
                    actual=specification.builder.template_id,
                    original_error=error,
                ) from error

    first = specifications[0]
    species = first.builder.strict_domain.species_vocabulary
    site_types = first.builder.site_type_ids
    channel_count = first.phase_specification.num_channels
    alignment = first.species_alignment_weights
    for source, specification in zip(recipe.reference.sources, specifications):
        phase = specification.phase_specification
        if (
            phase.approval_status == "provisional"
            and not source.is_automatic
            and not source.allow_provisional_phase
        ):
            raise _error(
                "PROVISIONAL_PHASE_NOT_APPROVED",
                "provisional phase requires allow_provisional_phase: true",
                stage="recipe.reference", field="reference.allow_provisional_phase",
                actual=specification.builder.template_id
            )
        if specification.builder.strict_domain.species_vocabulary != species:
            raise _error(
                "REFERENCE_SPECIES_MISMATCH", "all specifications must use one ordered species vocabulary",
                stage="recipe.compile", field="reference.specification"
            )
        if specification.builder.site_type_ids != site_types:
            raise _error(
                "REFERENCE_SITE_TYPE_MISMATCH", "all specifications must use one global site-type vocabulary",
                stage="recipe.compile", field="reference.specification"
            )
        if phase.num_channels != channel_count or specification.species_alignment_weights != alignment:
            raise _error(
                "REFERENCE_CHANNEL_MISMATCH", "all specifications must use one phase/species channel ordering",
                stage="recipe.compile", field="reference.specification"
            )
        if not math.isclose(
            specification.builder.avg_num_neighbors,
            first.builder.avg_num_neighbors,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise _error(
                "REFERENCE_GRAPH_CONTRACT_MISMATCH",
                "all specifications must use one avg_num_neighbors contract",
                stage="recipe.compile",
                field="reference.specification.builder.avg_num_neighbors",
                expected=first.builder.avg_num_neighbors,
                actual=specification.builder.avg_num_neighbors,
            )
        for field_name, expected, actual in (
            ("graph_cutoff", recipe.radii.r_mp, specification.builder.graph_cutoff),
            ("graph_skin", recipe.radii.mp_skin, specification.builder.graph_skin),
        ):
            if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=1.0e-12):
                raise _error(
                    "REFERENCE_RADIUS_MISMATCH",
                    "reference specification radii are incompatible with recipe radii",
                    stage="recipe.compile", field=f"reference.builder.{field_name}",
                    expected=expected, actual=actual
                )

    maximum_l = int(recipe.model.max_L)
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=species,
        n_radial=int(_MODEL_DEFAULTS["n_radial"]),
        lmax=maximum_l,
        ell_feature=float(_MODEL_DEFAULTS["ell_feature"]),
        r_cut=recipe.radii.r_mp,
        probability_tolerance=None,
        site_type_vocabulary=site_types,
    )
    higher_common = dict(
        irreps_feature=_feature_irreps(len(species), feature.n_radial, maximum_l),
        species_count=len(species),
        site_type_count=len(site_types),
        site_type_embedding_dim=int(_MODEL_DEFAULTS["site_type_embedding_dim"]),
        n_correlation_channels=int(recipe.model.hidden_channels),
        lmax=maximum_l,
        radial_feature_dim=int(_MODEL_DEFAULTS["radial_feature_dim"]),
        radial_hidden_dims=tuple(_MODEL_DEFAULTS["radial_hidden_dims"]),
        avg_num_neighbors=first.builder.avg_num_neighbors,
        cutoff=recipe.radii.r_mp,
        # New recipes normalize MP radial powers by their physical cutoff.
        # Persist the resolved scale so legacy bundles/resumes keep their
        # original arithmetic (commonly edge_length_scale=1.0).
        edge_length_scale=recipe.radii.r_mp,
    )
    if recipe.model.correlation_method == SYMMETRIC_CORRELATION_METHOD:
        higher = HigherBodyConfig(
            **higher_common,
            correlation_mode=None,
            contract_version=SYMMETRIC_POWER_CONTRACT_VERSION,
            symmetric_correlation=SymmetricCorrelationConfig(recipe.model.correlation),
        )
    else:
        higher = HigherBodyConfig(
            **higher_common,
            correlation_mode=recipe.model.correlation_mode,
            contract_version=LEGACY_HIGHER_BODY_CONTRACT_VERSION,
            symmetric_correlation=None,
        )
    potential = PotentialConfig(
        species_vocabulary=species,
        num_layers=recipe.model.num_interactions,
        feature=feature,
        higher_body=higher,
        readout_hidden=int(_MODEL_DEFAULTS["readout_hidden"]),
        energy_scale=float(_MODEL_DEFAULTS["energy_scale"]),
        epsilon_ot=float(_MODEL_DEFAULTS["epsilon_ot"]),
        ell_ot=float(_MODEL_DEFAULTS["ell_ot"]),
        train_sinkhorn_iterations=int(_MODEL_DEFAULTS["train_sinkhorn_iterations"]),
        phase_steps=tuple(_MODEL_DEFAULTS["phase_steps"]),
        phase_damping=tuple(_MODEL_DEFAULTS["phase_damping"]),
        transport_support=transport_support_config_from_radii(
            recipe.radii,
            backend=str(_MODEL_DEFAULTS["transport_backend"]),
            candidate_backend=str(_MODEL_DEFAULTS["candidate_backend"]),
        ),
        eval_sinkhorn_warmup_iterations=int(_MODEL_DEFAULTS["eval_sinkhorn_warmup_iterations"]),
    )
    reference_templates = tuple(
        ScratchReferenceTemplateSourceConfig(
            poscar_path=source.poscar,
            builder=specification.builder,
            phase_specification=specification.phase_specification,
            evaluation_policy=specification.evaluation_policy,
        )
        for source, specification in zip(recipe.reference.sources, specifications)
    )
    model_source = ScratchModelSourceConfig(
        initialization_seed=recipe.model.initialization_seed,
        potential=potential,
        species_alignment_weights=alignment,
        reference_templates=reference_templates,
        default_template_id=default_template,
    )
    data = TrainingDataConfig(
        train=tuple(
            _compile_data_source(
                item,
                template_ids=template_ids,
                reference_aliases=reference_aliases,
                automatic_reference_mode=all(
                    source.is_automatic for source in recipe.reference.sources
                ),
            )
            for item in recipe.data.train
        ),
        validation=tuple(
            _compile_data_source(
                item,
                template_ids=template_ids,
                reference_aliases=reference_aliases,
                automatic_reference_mode=all(
                    source.is_automatic for source in recipe.reference.sources
                ),
            )
            for item in recipe.data.validation
        ),
        batch_size=recipe.training.batch_size,
        validation_batch_size=recipe.training.validation_batch_size,
        shuffle=False,
    )
    config = TrainingRunConfig(
        schema_version=TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2,
        initial_bundle=None,
        model_source=model_source,
        radii=recipe.radii,
        data=data,
        runtime=TrainingRuntimeConfig(
            seed=recipe.runtime.seed, device=recipe.runtime.device, dtype=recipe.runtime.dtype
        ),
        loss=LossConfig(
            energy_weight=recipe.loss.energy_weight,
            force_weight=recipe.loss.forces_weight,
            stress_weight=recipe.loss.stress_weight,
            energy_scale=recipe.loss.energy_scale,
            force_scale=recipe.loss.force_scale,
            stress_scale=recipe.loss.stress_scale,
            energy_normalization=recipe.loss.energy_normalization,
        ),
        baseline=_baseline_config(recipe.baseline),
        optimizer=OptimizerConfig(learning_rate=recipe.training.learning_rate, weight_decay=0.0),
        train_step=TrainStepConfig(solver_path=TRAIN_FIXED, gradient_clip_norm=recipe.training.gradient_clip_norm),
        validation_step=ValidationStepConfig(solver_path=TRAIN_FIXED),
        scheduler=recipe.training.scheduler or SchedulerConfig(kind="none", monitor="total_loss", mode="min"),
        selection=ModelSelectionConfig(
            monitor="total_loss",
            mode="min",
            early_stopping_patience=recipe.training.early_stopping_patience,
            relative_min_delta=recipe.training.early_stopping_relative_delta,
        ),
        fit=FitConfig(max_epochs=recipe.training.max_epochs),
        checkpointed_fit=CheckpointedFitConfig(save_every_epoch=True, require_empty_manager=True),
        output_directory=(f"runs/{recipe.name}" if recipe.name is not None else str(recipe.output_directory)),
        source_path=recipe.source_path,
    )
    effective = apply_training_run_overrides(config, overrides, cli_cwd=cli_cwd)

    origins: dict[str, str] = {
        "model.contract_version": "fixed",
        "model.edge_length_scale": "derived",
        "data.shuffle": "fixed", "optimizer.kind": "fixed",
        "optimizer.weight_decay": "fixed", "train_step.solver_path": "fixed",
        "validation_step.solver_path": "fixed", "scheduler.monitor": "fixed",
        "scheduler.mode": "fixed", "selection.monitor": "fixed",
        "selection.mode": "fixed", "checkpointed_fit.save_every_epoch": "fixed",
        "model.correlation_method": _origin(recipe.model._provided_fields, "correlation_method"),
        "model.num_layers": _origin(recipe.model._provided_fields, "num_interactions"),
        "model.hidden_channels": ("user" if ("hidden_channels" in recipe.model._provided_fields or "hidden_irreps" in recipe.model._provided_fields) else "preset"),
        "model.max_L": ("user" if ("max_L" in recipe.model._provided_fields or "hidden_irreps" in recipe.model._provided_fields) else "preset"),
        "model.initialization_seed": _origin(recipe.model._provided_fields, "initialization_seed"),
        "loss.energy_weight": _origin(recipe.loss._provided_fields, "energy_weight"),
        "loss.force_weight": _origin(recipe.loss._provided_fields, "forces_weight"),
        "loss.stress_weight": _origin(recipe.loss._provided_fields, "stress_weight"),
        "baseline": "user", "runtime.seed": "user",
        "runtime.device": _origin(recipe.runtime._provided_fields, "device"),
        "runtime.dtype": _origin(recipe.runtime._provided_fields, "dtype"),
        "data.batch_size": _origin(recipe.training._provided_fields, "batch_size"),
        "data.validation_batch_size": _origin(recipe.training._provided_fields, "validation_batch_size"),
        "fit.max_epochs": "user", "optimizer.learning_rate": _origin(recipe.training._provided_fields, "learning_rate"),
        "radii.r_ot": ("user" if "radii" in recipe._provided_fields else "preset"),
        "radii.r_mp": ("user" if "radii" in recipe._provided_fields else "preset"),
        "output_directory": "user" if recipe.output_directory is not None else "preset",
        "ot_solver.training": "user",
        "ot_solver.inference": "user",
    }
    origins["selection.early_stopping_patience"] = _origin(
        recipe.training._provided_fields, "early_stopping_patience"
    )
    for name in ("energy_scale", "force_scale", "stress_scale", "energy_normalization"):
        origins[f"loss.{name}"] = _origin(recipe.loss._provided_fields, name)
    origins["train_step.gradient_clip_norm"] = _origin(recipe.training._provided_fields, "gradient_clip_norm")
    for name in config.scheduler.to_dict():
        if name not in ("monitor", "mode"):
            origins[f"scheduler.{name}"] = _origin(recipe.training._provided_fields, "scheduler")
    origins["selection.relative_min_delta"] = _origin(
        recipe.training._provided_fields,
        "early_stopping_relative_delta",
    )
    if recipe.model.correlation_method == SYMMETRIC_CORRELATION_METHOD:
        origins.update(
            {
                "model.basis_kind": "fixed",
                "model.normalization": "fixed",
                "model.basis_version": "fixed",
                "model.correlation_order": _origin(
                    recipe.model._provided_fields, "correlation"
                ),
            }
        )
    else:
        origins["model.correlation_mode"] = "user"
    if overrides is not None:
        for override_name, field_name in (
            ("device", "runtime.device"), ("dtype", "runtime.dtype"),
            ("max_epochs", "fit.max_epochs"), ("batch_size", "data.batch_size"),
            ("validation_batch_size", "data.validation_batch_size"),
            ("learning_rate", "optimizer.learning_rate"), ("r_ot", "radii.r_ot"),
            ("r_mp", "radii.r_mp"), ("output_directory", "output_directory"),
        ):
            if getattr(overrides, override_name) is not None:
                origins[field_name] = "CLI"

    base = Path(recipe.source_path).parent if recipe.source_path is not None else Path(".")
    paths: list[tuple[str, str, str]] = []
    if recipe.source_path is not None:
        paths.append(("recipe", recipe.source_path, str(Path(recipe.source_path).resolve(strict=False))))
    for index, source in enumerate(recipe.reference.sources):
        if source.specification is not None:
            paths.append(
                (
                    f"reference[{index}].specification",
                    source.specification,
                    _resolved_text(source.specification, base),
                )
            )
        paths.append(
            (
                f"reference[{index}].poscar",
                source.poscar,
                _resolved_text(source.poscar, base),
            )
        )
    for split, sources in (("train", recipe.data.train), ("validation", recipe.data.validation)):
        for index, source in enumerate(sources):
            paths.append((f"data.{split}[{index}]", source.path, _resolved_text(source.path, base)))
    output_base = Path(cli_cwd) if overrides is not None and overrides.output_directory is not None and cli_cwd is not None else base
    paths.append(("output_directory", effective.output_directory, _resolved_text(effective.output_directory, output_base)))
    model_defaults_version = (
        SYMMETRIC_MODEL_DEFAULTS_VERSION
        if recipe.model.correlation_method == SYMMETRIC_CORRELATION_METHOD
        else SEQUENTIAL_MODEL_DEFAULTS_VERSION
    )
    preset_fingerprints = tuple(
        (name, _fingerprint(payload))
        for name, payload in (
            (model_defaults_version, _MODEL_DEFAULTS),
            (TRAINING_DEFAULTS_VERSION, _TRAINING_DEFAULTS),
            (RADIUS_DERIVATION_VERSION, _RADIUS_DEFAULTS),
        )
    )
    manifest = RecipeResolutionManifest(
        recipe_fingerprint=recipe.content_fingerprint,
        compiled_config_fingerprint=effective.config_fingerprint,
        reference_specification_fingerprints=tuple(
            (spec.builder.template_id, spec.content_fingerprint) for spec in specifications
        ),
        field_origins=tuple(origins.items()), paths=tuple(paths),
        training_ot_solver=recipe.ot_solver.training,
        inference_ot_solver=recipe.ot_solver.inference,
        sinkhorn_iterations=potential.train_sinkhorn_iterations,
        sinkhorn_residual_tolerance=(
            1.0e-6 if recipe.runtime.dtype == "float32" else 1.0e-7
        ),
        preset_versions=(
            ("model", model_defaults_version),
            ("radii", RADIUS_DERIVATION_VERSION),
            ("training", TRAINING_DEFAULTS_VERSION),
        ),
        preset_fingerprints=preset_fingerprints,
    )
    return ResolvedTrainingRecipe(effective, manifest, recipe)


def compile_training_recipe(
    recipe: TrainingRecipeConfig,
    reference_specifications: Sequence[ReferenceSpecificationConfig],
    *,
    overrides: TrainingRunConfigOverrides | None = None,
    cli_cwd: str | os.PathLike[str] | None = None,
) -> ResolvedTrainingRecipe:
    """Purely compile canonical objects; no source file is read or model built."""

    try:
        return _compile_training_recipe_impl(
            recipe,
            reference_specifications,
            overrides=overrides,
            cli_cwd=cli_cwd,
        )
    except TrainingRecipeError:
        raise
    except Exception as error:
        raise _error(
            "CANONICAL_RECIPE_COMPILATION_FAILED",
            "recipe values could not form a valid canonical training-run v2 object",
            stage="recipe.compile",
            original_error=error,
        ) from error


class _RecipeSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            raise _error(
                "INVALID_RECIPE_YAML_TYPE", "YAML keys must be strings",
                stage="recipe.parse", field=repr(key)
            )
        if key == "<<":
            raise _error(
                "UNSUPPORTED_YAML_MERGE", "YAML merge keys are forbidden",
                stage="recipe.parse", field=key
            )
        if key in result:
            raise _error(
                "DUPLICATE_RECIPE_KEY", "duplicate YAML mapping key is forbidden",
                stage="recipe.parse", field=key
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_RecipeSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _validate_plain_tree(value: Any, *, field_name: str = "recipe") -> None:
    if type(value) is float and not math.isfinite(value):
        raise _error(
            "NONFINITE_RECIPE_VALUE", "NaN and Infinity are forbidden",
            stage="recipe.parse", field=field_name
        )
    if value is None or type(value) in (str, bool, int, float):
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if type(key) is not str:
                raise _error(
                    "INVALID_RECIPE_YAML_TYPE", "mapping keys must be strings",
                    stage="recipe.parse", field=field_name
                )
            _validate_plain_tree(nested, field_name=f"{field_name}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_plain_tree(nested, field_name=f"{field_name}[{index}]")
        return
    raise _error(
        "INVALID_RECIPE_YAML_TYPE", "only plain JSON-compatible values are allowed",
        stage="recipe.parse", field=field_name, actual=type(value).__name__
    )


def _decode_plain(encoded: str, *, suffix: str) -> Mapping[str, Any]:
    try:
        if suffix in (".yaml", ".yml"):
            tokens = tuple(yaml.scan(encoded))
            if any(isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)) for token in tokens):
                raise _error(
                    "UNSUPPORTED_YAML_ALIAS", "YAML anchors and aliases are forbidden",
                    stage="recipe.parse"
                )
            value = yaml.load(encoded, Loader=_RecipeSafeLoader)
        else:
            def reject_constant(constant: str):
                raise _error(
                    "NONFINITE_RECIPE_VALUE", "NaN and Infinity are forbidden",
                    stage="recipe.parse", actual=constant
                )
            def strict_object(pairs):
                result = {}
                for key, item in pairs:
                    if key in result:
                        raise _error(
                            "DUPLICATE_RECIPE_KEY", "duplicate JSON key is forbidden",
                            stage="recipe.parse", field=key
                        )
                    result[key] = item
                return result
            value = json.loads(encoded, object_pairs_hook=strict_object, parse_constant=reject_constant)
    except TrainingRecipeError:
        raise
    except (yaml.YAMLError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise _error(
            "INVALID_RECIPE_DOCUMENT", "configuration document could not be decoded safely",
            stage="recipe.parse", original_error=error
        ) from error
    _validate_plain_tree(value)
    if not isinstance(value, Mapping):
        raise _error(
            "INVALID_RECIPE_SECTION", "document root must be a mapping",
            stage="recipe.parse", field="recipe"
        )
    return value


def _read_regular_file(path: str | os.PathLike[str], *, text: bool) -> tuple[Path, bytes]:
    requested = Path(path)
    if requested.is_symlink():
        raise _error(
            "INPUT_SYMLINK_REJECTED", "recipe inputs must not be symbolic links",
            stage="recipe.path", path=requested
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(requested, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _error(
                "INVALID_RECIPE_INPUT", "input must be a regular file",
                stage="recipe.path", path=requested
            )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise _error(
                "INPUT_TOCTOU_MISMATCH", "input changed while it was being read",
                stage="recipe.path", path=requested
            )
        resolved = Path(os.path.realpath(requested))
        data = b"".join(chunks)
        if text:
            data.decode("utf-8")
        return resolved, data
    except TrainingRecipeError:
        raise
    except FileNotFoundError as error:
        raise _error(
            "RECIPE_INPUT_NOT_FOUND", "required recipe input does not exist",
            stage="recipe.path", path=requested, original_error=error
        ) from error
    except (OSError, UnicodeError) as error:
        raise _error(
            "RECIPE_INPUT_READ_FAILED", "recipe input could not be read safely",
            stage="recipe.path", path=requested, original_error=error
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def load_training_recipe(path: str | os.PathLike[str]) -> TrainingRecipeConfig:
    resolved, raw = _read_regular_file(path, text=True)
    payload = _decode_plain(raw.decode("utf-8"), suffix=resolved.suffix.lower())
    recipe = TrainingRecipeConfig.from_dict(payload)
    return replace(recipe, source_path=str(resolved))


def load_reference_specification(path: str | os.PathLike[str]) -> ReferenceSpecificationConfig:
    resolved, raw = _read_regular_file(path, text=True)
    payload = _decode_plain(raw.decode("utf-8"), suffix=resolved.suffix.lower())
    try:
        return ReferenceSpecificationConfig.from_dict(payload)
    except TrainingRecipeError as error:
        if error.path is None:
            error.path = str(resolved)
        raise


def _validate_aliased_data_paths(
    recipe: TrainingRecipeConfig, *, base: Path
) -> None:
    if not any(
        source.reference_alias is not None
        for source in recipe.data.train + recipe.data.validation
    ):
        return
    identities: dict[str, tuple[str, int, str]] = {}
    for split, sources in (
        ("train", recipe.data.train),
        ("validation", recipe.data.validation),
    ):
        split_identities: dict[str, tuple[int, str]] = {}
        for index, source in enumerate(sources):
            original = source.path
            candidate = Path(original)
            unresolved = candidate if candidate.is_absolute() else base / candidate
            try:
                resolved = str(unresolved.resolve(strict=False))
            except (OSError, RuntimeError) as error:
                raise _error(
                    "RECIPE_DATA_PATH_ERROR",
                    "bound extxyz path could not be resolved",
                    stage="recipe.data",
                    field=f"data.{split}[{index}].file",
                    path=original,
                    original_error=error,
                ) from error
            if resolved in split_identities:
                first_index, first_original = split_identities[resolved]
                raise _error(
                    "DUPLICATE_DATA_SOURCE",
                    "the same resolved extxyz file is registered twice in one split",
                    stage="recipe.data",
                    field=f"data.{split}[{index}].file",
                    path=original,
                    expected={
                        "source_index": first_index,
                        "original_path": first_original,
                    },
                    actual={"source_index": index, "resolved_path": resolved},
                )
            split_identities[resolved] = (index, original)
            if resolved in identities and identities[resolved][0] != split:
                other_split, other_index, other_original = identities[resolved]
                raise _error(
                    "TRAIN_VALIDATION_DATA_LEAKAGE",
                    "the same resolved extxyz file cannot be used for train and validation",
                    stage="recipe.data",
                    field=f"data.{split}[{index}].file",
                    path=original,
                    expected={
                        "split": other_split,
                        "source_index": other_index,
                        "original_path": other_original,
                    },
                    actual={
                        "split": split,
                        "source_index": index,
                        "resolved_path": resolved,
                    },
                )
            identities[resolved] = (split, index, original)


def resolve_training_recipe(
    path: str | os.PathLike[str],
    *,
    overrides: TrainingRunConfigOverrides | None = None,
    cli_cwd: str | os.PathLike[str] | None = None,
) -> ResolvedTrainingRecipe:
    """Resolve file bindings then invoke the deterministic pure compiler."""

    recipe = load_training_recipe(path)
    assert recipe.source_path is not None
    base = Path(recipe.source_path).parent
    _validate_aliased_data_paths(recipe, base=base)
    automatic_mode = all(
        source.is_automatic for source in recipe.reference.sources
    )
    if automatic_mode:
        from .automatic_reference import (
            AutomaticReferenceError,
            prepare_automatic_references,
        )

        try:
            automatic = prepare_automatic_references(
                recipe,
                base=base,
                read_file=_read_regular_file,
                specification_factory=ReferenceSpecificationConfig,
                evaluation_policy_requested=(
                    recipe.ot_solver.inference
                    == SINKHORN_NEWTON_KRYLOV_OT_SOLVER
                ),
            )
        except AutomaticReferenceError as error:
            context = " ".join(
                f"{name}={value!r}"
                for name, value in (
                    ("sample_id", error.sample_id),
                    ("template_id", error.template_id),
                )
                if value is not None
            )
            raise _error(
                error.reason_code,
                error.message if not context else f"{error.message}; {context}",
                stage=error.stage,
                path=error.source_path,
                expected=error.expected,
                actual=(error.diagnostics if error.diagnostics is not None else error.actual),
                original_error=error,
            ) from error
        ordered_sources = tuple(
            recipe.reference.sources[result.source_index]
            for result in automatic.results
        )
        ordered_reference = RecipeReferenceConfig(
            sources=ordered_sources,
            default_template_id=recipe.reference.default_template_id,
        )
        recipe = replace(recipe, reference=ordered_reference)
        compiled = compile_training_recipe(
            recipe,
            tuple(result.specification for result in automatic.results),
            overrides=overrides,
            cli_cwd=cli_cwd,
        )
        if recipe.ot_solver.inference == SINKHORN_NEWTON_KRYLOV_OT_SOLVER:
            from .automatic_evaluation import (
                AutomaticEvaluationPolicyAuditError,
                automatic_evaluation_certificate_semantic_identity,
                qualify_automatic_evaluation_policies,
            )

            try:
                automatic = qualify_automatic_evaluation_policies(
                    automatic, compiled.config.model_source.potential
                )
            except AutomaticEvaluationPolicyAuditError as error:
                raise _error(
                    "AUTOMATIC_EVALUATION_POLICY_AUDIT_FAILED",
                    error.message,
                    stage=error.stage,
                    field="ot_solver.inference",
                    expected=error.required,
                    actual={
                        "reason_code": error.reason_code,
                        "template_id": error.template_id,
                        "sample_id": error.sample_id,
                        "geometry_digest": error.geometry_digest,
                        "probe": error.probe,
                        "dtype": error.dtype,
                        "backend": error.backend,
                        "observed": error.observed,
                        "diagnostics": error.diagnostics,
                    },
                    original_error=error,
                ) from error
        manifest = replace(
            compiled.manifest,
            automatic_reference_fingerprint=automatic.content_fingerprint,
            automatic_reference_certificates=tuple(
                (
                    result.template_id,
                    (
                        result.to_dict()
                        if result.evaluation_certificate is None
                        else {
                            **result.to_dict(),
                            "evaluation_certificate": (
                                automatic_evaluation_certificate_semantic_identity(
                                    result.evaluation_certificate
                                )
                            ),
                        }
                    ),
                )
                for result in automatic.results
            ),
        )
        return ResolvedTrainingRecipe(
            compiled.config,
            manifest,
            recipe,
            automatic,
        )
    specifications = []
    for index, source in enumerate(recipe.reference.sources):
        assert source.specification is not None
        specification_path = Path(source.specification)
        if not specification_path.is_absolute():
            specification_path = base / specification_path
        specification = load_reference_specification(specification_path)
        poscar_path = Path(source.poscar)
        if not poscar_path.is_absolute():
            poscar_path = base / poscar_path
        _, poscar_bytes = _read_regular_file(poscar_path, text=False)
        actual = hashlib.sha256(poscar_bytes).hexdigest()
        if actual != specification.poscar_sha256:
            raise _error(
                "POSCAR_FINGERPRINT_MISMATCH",
                "POSCAR content does not match its reference specification",
                stage="recipe.reference", field=f"reference.sources[{index}].poscar",
                path=poscar_path, expected=specification.poscar_sha256, actual=actual
            )
        specifications.append(specification)
    return compile_training_recipe(
        recipe,
        specifications,
        overrides=overrides,
        cli_cwd=cli_cwd,
    )


def is_training_recipe(path: str | os.PathLike[str]) -> bool:
    """Safely identify the top-level schema without accepting loose parsing."""

    resolved, raw = _read_regular_file(path, text=True)
    payload = _decode_plain(raw.decode("utf-8"), suffix=resolved.suffix.lower())
    return payload.get("schema_version") == TRAINING_RECIPE_SCHEMA_VERSION


__all__ = [
    "RADIUS_DERIVATION_VERSION",
    "RECIPE_RESOLVER_VERSION",
    "REFERENCE_SPECIFICATION_SCHEMA_VERSION",
    "RecipeOTSolverConfig",
    "RecipeDataConfig",
    "RecipeLossConfig",
    "RecipeModelConfig",
    "RecipeReferenceConfig",
    "RecipeReferenceSourceConfig",
    "RecipeResolutionManifest",
    "RecipeRuntimeConfig",
    "RecipeTrainingConfig",
    "ReferenceSpecificationConfig",
    "ResolvedTrainingRecipe",
    "SEQUENTIAL_MODEL_DEFAULTS_VERSION",
    "SYMMETRIC_MODEL_DEFAULTS_VERSION",
    "TRAINING_DEFAULTS_VERSION",
    "TRAINING_RECIPE_SCHEMA_VERSION",
    "TrainingRecipeConfig",
    "TrainingRecipeError",
    "compile_training_recipe",
    "is_training_recipe",
    "load_reference_specification",
    "load_training_recipe",
    "resolve_training_recipe",
]
