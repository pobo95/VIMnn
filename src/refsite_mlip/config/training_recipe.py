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
SYMMETRIC_MODEL_DEFAULTS_VERSION = "symmetric_model_defaults_v1"
SEQUENTIAL_MODEL_DEFAULTS_VERSION = "sequential_model_defaults_v1"
TRAINING_DEFAULTS_VERSION = "training_defaults_v1"
RADIUS_DERIVATION_VERSION = "radius_derivation_v1"

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
            required=frozenset({"training", "inference"}),
            field_name="ot_solver",
        )
        return cls(training=payload["training"], inference=payload["inference"])


@dataclass(frozen=True)
class RecipeReferenceSourceConfig:
    specification: str
    poscar: str
    allow_provisional_phase: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "specification", _path(self.specification, field_name="reference.specification"))
        object.__setattr__(self, "poscar", _path(self.poscar, field_name="reference.poscar"))
        if type(self.allow_provisional_phase) is not bool:
            raise _error(
                "INVALID_PROVISIONAL_PHASE_FLAG",
                "allow_provisional_phase must be a bool",
                stage="recipe.reference",
                field="reference.allow_provisional_phase",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification": self.specification,
            "poscar": self.poscar,
            "allow_provisional_phase": self.allow_provisional_phase,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, field_name: str) -> "RecipeReferenceSourceConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"specification", "poscar", "allow_provisional_phase"}),
            required=frozenset({"specification", "poscar"}),
            field_name=field_name,
        )
        return cls(**dict(payload))


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
        if len(self.sources) == 1 and self.default_template_id is None:
            return self.sources[0].to_dict()
        return {
            "templates": [item.to_dict() for item in self.sources],
            "default_template_id": self.default_template_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeReferenceConfig":
        if not isinstance(value, Mapping):
            raise _error(
                "INVALID_RECIPE_SECTION", "reference must be a mapping",
                stage="recipe.schema", field="reference"
            )
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, field_name="data.path"))
        if self.template_id is not None and (type(self.template_id) is not str or not self.template_id):
            raise _error(
                "INVALID_TEMPLATE_SELECTOR", "template_id must be a nonempty string",
                stage="recipe.data", field="data.template_id"
            )
        if type(self.automatic_template_assignment) is not bool:
            raise _error(
                "INVALID_TEMPLATE_SELECTOR", "automatic_template_assignment must be a bool",
                stage="recipe.data", field="data.automatic_template_assignment"
            )
        if self.template_id is not None and self.automatic_template_assignment:
            raise _error(
                "CONFLICTING_TEMPLATE_SELECTOR", "exact and automatic template selection conflict",
                stage="recipe.data", field="data.template_id,automatic_template_assignment"
            )

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, Any]:
        def encode(items: tuple[RecipeDataSourceConfig, ...]) -> Any:
            if len(items) == 1 and items[0].template_id is None and not items[0].automatic_template_assignment:
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
    _provided_fields: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
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
        return {
            "energy_weight": self.energy_weight,
            "forces_weight": self.forces_weight,
            "stress_weight": self.stress_weight,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeLossConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"energy_weight", "forces_weight", "stress_weight"}),
            field_name="loss",
        )
        return cls(**dict(payload), _provided_fields=tuple(payload))


@dataclass(frozen=True)
class RecipeTrainingConfig:
    max_epochs: int
    batch_size: int = 4
    validation_batch_size: int | None = None
    learning_rate: float = 1.0e-3
    _provided_fields: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("max_epochs", "batch_size"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), field_name=f"training.{name}"))
        validation = self.batch_size if self.validation_batch_size is None else _positive_int(
            self.validation_batch_size, field_name="training.validation_batch_size"
        )
        object.__setattr__(self, "validation_batch_size", validation)
        object.__setattr__(self, "learning_rate", _positive_real(self.learning_rate, field_name="training.learning_rate"))
        object.__setattr__(self, "_provided_fields", tuple(sorted(self._provided_fields)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "validation_batch_size": self.validation_batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeTrainingConfig":
        payload = _strict_mapping(
            value,
            allowed=frozenset({"batch_size", "validation_batch_size", "max_epochs", "learning_rate"}),
            required=frozenset({"max_epochs"}),
            field_name="training",
        )
        return cls(**dict(payload), _provided_fields=tuple(payload))


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
        required = frozenset({"schema_version", "model", "reference", "data", "ot_solver", "loss", "baseline", "training", "runtime"})
        payload = _strict_mapping(value, allowed=allowed, required=required, field_name="recipe")
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
            reference=RecipeReferenceConfig.from_dict(payload["reference"]),
            data=RecipeDataConfig.from_dict(payload["data"]),
            ot_solver=RecipeOTSolverConfig.from_dict(payload["ot_solver"]),
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

    def __post_init__(self) -> None:
        for value in (self.recipe_fingerprint, self.compiled_config_fingerprint):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("manifest fingerprints must be lowercase SHA-256")
        object.__setattr__(self, "reference_specification_fingerprints", tuple(sorted(self.reference_specification_fingerprints)))
        object.__setattr__(self, "field_origins", tuple(sorted(self.field_origins)))
        object.__setattr__(self, "paths", tuple(sorted(self.paths)))
        object.__setattr__(self, "preset_versions", tuple(sorted(self.preset_versions)))
        object.__setattr__(self, "preset_fingerprints", tuple(sorted(self.preset_fingerprints)))
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
        return {
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

    def __post_init__(self) -> None:
        if self.config.schema_version != TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2:
            raise ValueError("recipe resolution must produce canonical schema v2")
        if self.config.config_fingerprint != self.manifest.compiled_config_fingerprint:
            raise ValueError("manifest and compiled canonical config disagree")

    def to_dict(self) -> dict[str, Any]:
        return {
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
    "edge_length_scale": 1.0,
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
) -> TrainingDataSourceConfig:
    if source.template_id is not None:
        if source.template_id not in template_ids:
            raise _error(
                "UNKNOWN_TEMPLATE_ID", "data source names an unknown template",
                stage="recipe.compile", field="data.template_id", actual=source.template_id
            )
        return TrainingDataSourceConfig(path=source.path, template_id=source.template_id)
    if source.automatic_template_assignment:
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
    default_template = recipe.reference.default_template_id
    if default_template is None:
        if len(template_ids) != 1:
            raise _error(
                "MISSING_DEFAULT_TEMPLATE", "multi-template recipe requires default_template_id",
                stage="recipe.compile", field="reference.default_template_id"
            )
        default_template = template_ids[0]
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
        if phase.approval_status == "provisional" and not source.allow_provisional_phase:
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
        edge_length_scale=float(_MODEL_DEFAULTS["edge_length_scale"]),
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
        train=tuple(_compile_data_source(item, template_ids=template_ids) for item in recipe.data.train),
        validation=tuple(_compile_data_source(item, template_ids=template_ids) for item in recipe.data.validation),
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
            energy_scale=float(_TRAINING_DEFAULTS["energy_scale"]),
            force_scale=float(_TRAINING_DEFAULTS["force_scale"]),
            stress_scale=float(_TRAINING_DEFAULTS["stress_scale"]),
            energy_normalization=str(_TRAINING_DEFAULTS["energy_normalization"]),
        ),
        baseline=_baseline_config(recipe.baseline),
        optimizer=OptimizerConfig(learning_rate=recipe.training.learning_rate, weight_decay=0.0),
        train_step=TrainStepConfig(solver_path=TRAIN_FIXED),
        validation_step=ValidationStepConfig(solver_path=TRAIN_FIXED),
        scheduler=SchedulerConfig(kind="none", monitor="total_loss", mode="min"),
        selection=ModelSelectionConfig(monitor="total_loss", mode="min"),
        fit=FitConfig(max_epochs=recipe.training.max_epochs),
        checkpointed_fit=CheckpointedFitConfig(save_every_epoch=True, require_empty_manager=True),
        output_directory=(f"runs/{recipe.name}" if recipe.name is not None else str(recipe.output_directory)),
        source_path=recipe.source_path,
    )
    effective = apply_training_run_overrides(config, overrides, cli_cwd=cli_cwd)

    origins: dict[str, str] = {
        "model.contract_version": "fixed",
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
        paths.extend(
            (
                (f"reference[{index}].specification", source.specification, _resolved_text(source.specification, base)),
                (f"reference[{index}].poscar", source.poscar, _resolved_text(source.poscar, base)),
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
    specifications = []
    for index, source in enumerate(recipe.reference.sources):
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
