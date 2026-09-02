"""User-facing interaction radii and legacy-runtime compatibility checks.

Only the two physical interaction radii are basic user controls.  Switch and
candidate margins remain explicit advanced controls, while solver/graph
execution policies stay outside this content-addressed configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

from refsite_mlip.transport import TransportSupportConfig, TransportSupportError


INTERACTION_RADIUS_CONFIG_SCHEMA_VERSION = "interaction_radius_config_v1"
INTERACTION_RADIUS_CONVENTION_VERSION = "interaction_radii_angstrom_v1"
INTERACTION_RADIUS_LENGTH_UNIT = "angstrom"

_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "convention_version",
        "length_unit",
        "r_ot",
        "r_mp",
        "ot_switch_width",
        "ot_skin",
        "mp_skin",
    }
)
_DERIVED_OR_INTERNAL_KEYS = frozenset(
    {
        "r_on_ot",
        "r_off_ot",
        "r_candidate_ot",
        "r_candidate_mp",
        "cutoff",
        "switch_width",
        "candidate_skin",
        "graph_cutoff",
        "graph_skin",
        "mp_cutoff",
        "content_fingerprint",
        "fingerprint",
    }
)


class RadiusConfigError(ValueError):
    """Structured validation or compatibility failure for interaction radii."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        stage: str,
        field: str | None = None,
        expected: Any = None,
        actual: Any = None,
        action: str | None = None,
        mismatches: tuple[tuple[str, float, float], ...] = (),
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
        self.field = field
        self.expected = expected
        self.actual = actual
        self.action = action
        self.mismatches = tuple(mismatches)
        self.original_error = original_error
        context = [f"stage={stage!r}"]
        if field is not None:
            context.append(f"field={field!r}")
        if expected is not None:
            context.append(f"expected={expected!r}")
        if actual is not None:
            context.append(f"actual={actual!r}")
        if self.mismatches:
            rendered = ", ".join(
                f"{name}: expected {wanted!r}, actual {found!r}"
                for name, wanted, found in self.mismatches
            )
            context.append(f"mismatches=[{rendered}]")
        suffix = "" if action is None else f" Action: {action}"
        super().__init__(
            f"[{reason_code}] {' '.join(context)} {message}{suffix}"
        )


def _radius_error(
    reason_code: str,
    message: str,
    *,
    stage: str,
    field: str | None = None,
    expected: Any = None,
    actual: Any = None,
    action: str | None = None,
    mismatches: tuple[tuple[str, float, float], ...] = (),
    original_error: BaseException | None = None,
) -> RadiusConfigError:
    return RadiusConfigError(
        reason_code,
        message,
        stage=stage,
        field=field,
        expected=expected,
        actual=actual,
        action=action,
        mismatches=mismatches,
        original_error=original_error,
    )


def _finite_real(value: Any, *, field: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _radius_error(
            "INVALID_RADIUS_VALUE",
            f"{field} must be a real number and bool is not accepted",
            stage="config.validation",
            field=field,
            actual=value,
        )
    result = float(value)
    invalid = result <= 0.0 if positive else result < 0.0
    if not math.isfinite(result) or invalid:
        qualifier = "finite and positive" if positive else "finite and nonnegative"
        raise _radius_error(
            "INVALID_RADIUS_VALUE",
            f"{field} must be {qualifier}",
            stage="config.validation",
            field=field,
            actual=result,
        )
    # Canonicalize negative zero so semantically identical configurations have
    # byte-identical JSON and fingerprints.
    return 0.0 if result == 0.0 else result


def _canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:  # pragma: no cover - internal guard
        raise _radius_error(
            "NON_CANONICAL_RADIUS_CONFIG",
            "radius config could not be represented as strict plain JSON",
            stage="config.serialization",
            original_error=error,
        ) from error


@dataclass(frozen=True)
class DerivedInteractionRadii:
    """Fully resolved OT and message-passing interaction radii in angstrom."""

    r_on_ot: float
    r_off_ot: float
    r_candidate_ot: float
    r_mp: float
    r_candidate_mp: float
    length_unit: str = INTERACTION_RADIUS_LENGTH_UNIT

    def __post_init__(self) -> None:
        values = {}
        for field in (
            "r_on_ot",
            "r_off_ot",
            "r_candidate_ot",
            "r_mp",
            "r_candidate_mp",
        ):
            values[field] = _finite_real(
                getattr(self, field),
                field=field,
                positive=field in ("r_off_ot", "r_mp"),
            )
            object.__setattr__(self, field, values[field])
        if self.length_unit != INTERACTION_RADIUS_LENGTH_UNIT:
            raise _radius_error(
                "UNSUPPORTED_RADIUS_UNIT",
                "interaction radii must use angstrom",
                stage="derived.validation",
                field="length_unit",
                expected=INTERACTION_RADIUS_LENGTH_UNIT,
                actual=self.length_unit,
            )
        if values["r_on_ot"] > values["r_off_ot"]:
            raise _radius_error(
                "INVALID_DERIVED_RADIUS",
                "r_on_ot must not exceed r_off_ot",
                stage="derived.validation",
                field="r_on_ot",
                expected=f"<= {values['r_off_ot']}",
                actual=values["r_on_ot"],
            )
        if values["r_candidate_ot"] < values["r_off_ot"]:
            raise _radius_error(
                "INVALID_DERIVED_RADIUS",
                "r_candidate_ot must be at least r_off_ot",
                stage="derived.validation",
                field="r_candidate_ot",
                expected=f">= {values['r_off_ot']}",
                actual=values["r_candidate_ot"],
            )
        if values["r_candidate_mp"] < values["r_mp"]:
            raise _radius_error(
                "INVALID_DERIVED_RADIUS",
                "r_candidate_mp must be at least r_mp",
                stage="derived.validation",
                field="r_candidate_mp",
                expected=f">= {values['r_mp']}",
                actual=values["r_candidate_mp"],
            )

    @property
    def ot_candidate_reuse_margin(self) -> float:
        return self.r_candidate_ot - self.r_off_ot

    @property
    def mp_graph_strain_margin(self) -> float:
        return self.r_candidate_mp - self.r_mp

    @property
    def ot_candidate_reuse_margin_available(self) -> bool:
        return self.ot_candidate_reuse_margin > 0.0

    @property
    def mp_graph_strain_margin_available(self) -> bool:
        return self.mp_graph_strain_margin > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_unit": self.length_unit,
            "r_on_ot": self.r_on_ot,
            "r_off_ot": self.r_off_ot,
            "r_candidate_ot": self.r_candidate_ot,
            "r_mp": self.r_mp,
            "r_candidate_mp": self.r_candidate_mp,
        }

    def to_diagnostics_dict(self) -> dict[str, Any]:
        ot_available = self.ot_candidate_reuse_margin_available
        mp_available = self.mp_graph_strain_margin_available
        return {
            "length_unit": self.length_unit,
            "mp_graph_strain_margin": self.mp_graph_strain_margin,
            "mp_graph_strain_margin_available": mp_available,
            "mp_graph_strain_note": (
                "graph strain margin is available"
                if mp_available
                else "mp_skin=0: no graph strain margin is available"
            ),
            "ot_candidate_reuse_margin": self.ot_candidate_reuse_margin,
            "ot_candidate_reuse_margin_available": ot_available,
            "ot_candidate_reuse_note": (
                "candidate reuse margin is available"
                if ot_available
                else "ot_skin=0: no candidate reuse margin is available"
            ),
        }

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        """Return a read-only snapshot describing zero-skin semantics."""

        return MappingProxyType(self.to_diagnostics_dict())


@dataclass(frozen=True)
class InteractionRadiusConfig:
    """Content-addressed user controls for OT and MP interaction radii."""

    r_ot: float = 4.0
    r_mp: float = 3.0
    ot_switch_width: float = 0.5
    ot_skin: float = 0.2
    mp_skin: float = 0.5
    schema_version: str = INTERACTION_RADIUS_CONFIG_SCHEMA_VERSION
    convention_version: str = INTERACTION_RADIUS_CONVENTION_VERSION
    length_unit: str = INTERACTION_RADIUS_LENGTH_UNIT

    def __post_init__(self) -> None:
        for field, expected in (
            ("schema_version", INTERACTION_RADIUS_CONFIG_SCHEMA_VERSION),
            ("convention_version", INTERACTION_RADIUS_CONVENTION_VERSION),
            ("length_unit", INTERACTION_RADIUS_LENGTH_UNIT),
        ):
            actual = getattr(self, field)
            if actual != expected:
                reason = (
                    "UNSUPPORTED_RADIUS_UNIT"
                    if field == "length_unit"
                    else "UNSUPPORTED_RADIUS_SCHEMA"
                )
                raise _radius_error(
                    reason,
                    f"unsupported interaction radius {field}",
                    stage="config.schema",
                    field=field,
                    expected=expected,
                    actual=actual,
                )
        for field in ("r_ot", "r_mp"):
            object.__setattr__(
                self,
                field,
                _finite_real(getattr(self, field), field=field, positive=True),
            )
        object.__setattr__(
            self,
            "ot_switch_width",
            _finite_real(
                self.ot_switch_width,
                field="ot_switch_width",
                positive=True,
            ),
        )
        for field in ("ot_skin", "mp_skin"):
            object.__setattr__(
                self,
                field,
                _finite_real(getattr(self, field), field=field, positive=False),
            )
        if self.ot_switch_width >= self.r_ot:
            raise _radius_error(
                "INVALID_RADIUS_RELATION",
                "ot_switch_width must be smaller than r_ot",
                stage="config.validation",
                field="ot_switch_width",
                expected=f"< {self.r_ot}",
                actual=self.ot_switch_width,
            )
        # Constructing the derived record is the single validation path for
        # every internal ordering constraint.
        derive_interaction_radii(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete, strict-schema plain JSON payload."""

        return {
            "schema_version": self.schema_version,
            "convention_version": self.convention_version,
            "length_unit": self.length_unit,
            "r_ot": self.r_ot,
            "r_mp": self.r_mp,
            "ot_switch_width": self.ot_switch_width,
            "ot_skin": self.ot_skin,
            "mp_skin": self.mp_skin,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "InteractionRadiusConfig":
        if not isinstance(values, Mapping):
            raise _radius_error(
                "INVALID_RADIUS_PAYLOAD",
                "interaction radius config must be reconstructed from a mapping",
                stage="config.deserialization",
                actual=type(values).__name__,
            )
        keys = frozenset(values)
        conflicting = keys & _DERIVED_OR_INTERNAL_KEYS
        if conflicting:
            raise _radius_error(
                "CONFLICTING_RADIUS_KEY",
                "derived/internal radius keys must not be supplied in user config",
                stage="config.deserialization",
                field=", ".join(sorted(str(key) for key in conflicting)),
            )
        unknown = keys - _CONFIG_KEYS
        if unknown:
            raise _radius_error(
                "UNKNOWN_RADIUS_KEY",
                "unknown interaction radius config key",
                stage="config.deserialization",
                field=", ".join(sorted((repr(key) for key in unknown))),
            )
        missing = _CONFIG_KEYS - keys
        if missing:
            raise _radius_error(
                "MISSING_RADIUS_KEY",
                "complete serialized radius config is missing required keys",
                stage="config.deserialization",
                field=", ".join(sorted(missing)),
            )
        return cls(**{key: values[key] for key in _CONFIG_KEYS})

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_json(self) -> str:
        return self.canonical_json()

    @classmethod
    def from_json(cls, value: str) -> "InteractionRadiusConfig":
        if not isinstance(value, str):
            raise _radius_error(
                "INVALID_RADIUS_JSON",
                "interaction radius JSON must be a string",
                stage="config.deserialization",
                actual=type(value).__name__,
            )

        def reject_constant(constant: str):
            raise _radius_error(
                "NONFINITE_RADIUS_JSON",
                "NaN and Infinity are forbidden in radius config JSON",
                stage="config.deserialization",
                actual=constant,
            )

        def strict_object(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise _radius_error(
                        "CONFLICTING_RADIUS_KEY",
                        "duplicate JSON object key is forbidden",
                        stage="config.deserialization",
                        field=key,
                    )
                result[key] = item
            return result

        try:
            payload = json.loads(
                value,
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
        except RadiusConfigError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise _radius_error(
                "INVALID_RADIUS_JSON",
                "interaction radius JSON could not be decoded",
                stage="config.deserialization",
                original_error=error,
            ) from error
        return cls.from_dict(payload)

    @property
    def content_fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Compatibility alias for the SHA-256 content fingerprint."""

        return self.content_fingerprint

    @property
    def derived(self) -> DerivedInteractionRadii:
        return derive_interaction_radii(self)

    def to_transport_support_config(
        self,
        *,
        backend: str = "dense",
        candidate_backend: str = "dense",
        site_block_size: int = 32,
        atom_block_size: int = 32,
    ) -> TransportSupportConfig:
        return transport_support_config_from_radii(
            self,
            backend=backend,
            candidate_backend=candidate_backend,
            site_block_size=site_block_size,
            atom_block_size=atom_block_size,
        )


def derive_interaction_radii(
    config: InteractionRadiusConfig,
) -> DerivedInteractionRadii:
    """Resolve internal OT/MP cutoffs without imposing an OT-versus-MP order."""

    if not isinstance(config, InteractionRadiusConfig):
        raise _radius_error(
            "INVALID_RADIUS_CONFIG",
            "config must be an InteractionRadiusConfig",
            stage="derived.input",
            actual=type(config).__name__,
        )
    return DerivedInteractionRadii(
        r_on_ot=config.r_ot - config.ot_switch_width,
        r_off_ot=config.r_ot,
        r_candidate_ot=config.r_ot + config.ot_skin,
        r_mp=config.r_mp,
        r_candidate_mp=config.r_mp + config.mp_skin,
    )


def transport_support_config_from_radii(
    config: InteractionRadiusConfig,
    *,
    backend: str = "dense",
    candidate_backend: str = "dense",
    site_block_size: int = 32,
    atom_block_size: int = 32,
) -> TransportSupportConfig:
    """Create existing compact-C2 support while keeping backend policy separate."""

    if not isinstance(config, InteractionRadiusConfig):
        raise _radius_error(
            "INVALID_RADIUS_CONFIG",
            "config must be an InteractionRadiusConfig",
            stage="transport_conversion.input",
            actual=type(config).__name__,
        )
    normalized_backend = {
        "dense": "dense",
        "dense-masked": "dense",
        "dense_masked": "dense",
        "edge-list": "edge_list",
        "edge_list": "edge_list",
    }.get(backend, backend)
    try:
        return TransportSupportConfig(
            kind="compact_c2",
            cutoff=config.r_ot,
            switch_width=config.ot_switch_width,
            candidate_skin=config.ot_skin,
            backend=normalized_backend,
            candidate_backend=candidate_backend,
            site_block_size=site_block_size,
            atom_block_size=atom_block_size,
        )
    except TransportSupportError as error:
        raise _radius_error(
            "TRANSPORT_RADIUS_CONVERSION_FAILED",
            "derived radii could not be represented by TransportSupportConfig",
            stage="transport_conversion",
            action="Choose a supported backend policy independently of the radii.",
            original_error=error,
        ) from error


def _target_value(target: Any, field: str, *, stage: str) -> Any:
    if isinstance(target, Mapping):
        if field not in target:
            raise _radius_error(
                "MISSING_COMPATIBILITY_FIELD",
                "compatibility target is missing a required radius field",
                stage=stage,
                field=field,
            )
        return target[field]
    if not hasattr(target, field):
        raise _radius_error(
            "MISSING_COMPATIBILITY_FIELD",
            "compatibility target is missing a required radius field",
            stage=stage,
            field=field,
            actual=type(target).__name__,
        )
    return getattr(target, field)


def _compatibility_real(value: Any, *, field: str, stage: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _radius_error(
            "INVALID_COMPATIBILITY_FIELD",
            "stored compatibility radius must be a real number",
            stage=stage,
            field=field,
            actual=value,
        )
    result = float(value)
    if not math.isfinite(result):
        raise _radius_error(
            "INVALID_COMPATIBILITY_FIELD",
            "stored compatibility radius must be finite",
            stage=stage,
            field=field,
            actual=result,
        )
    return 0.0 if result == 0.0 else result


def validate_radius_artifact_compatibility(
    config: InteractionRadiusConfig,
    artifact: Any,
) -> DerivedInteractionRadii:
    """Require an existing structural artifact to match MP radius controls."""

    if not isinstance(config, InteractionRadiusConfig):
        raise _radius_error(
            "INVALID_RADIUS_CONFIG",
            "config must be an InteractionRadiusConfig",
            stage="artifact_compatibility.input",
            actual=type(config).__name__,
        )
    actual_cutoff = _compatibility_real(
        _target_value(artifact, "mp_cutoff", stage="artifact_compatibility"),
        field="mp_cutoff",
        stage="artifact_compatibility",
    )
    actual_skin = _compatibility_real(
        _target_value(artifact, "mp_skin", stage="artifact_compatibility"),
        field="mp_skin",
        stage="artifact_compatibility",
    )
    mismatches = tuple(
        (field, expected, actual)
        for field, expected, actual in (
            ("mp_cutoff", config.r_mp, actual_cutoff),
            ("mp_skin", config.mp_skin, actual_skin),
        )
        if expected != actual
    )
    if mismatches:
        raise _radius_error(
            "RADIUS_ARTIFACT_MISMATCH",
            "structural artifact MP radii do not match the requested config",
            stage="artifact_compatibility",
            mismatches=mismatches,
            action=(
                "Rebuild the structural artifact after changing r_mp or mp_skin; "
                "the existing artifact was not modified."
            ),
        )
    return derive_interaction_radii(config)


def _model_support(model_or_config: Any) -> Any:
    target = model_or_config
    if not isinstance(target, Mapping) and not hasattr(target, "transport_support"):
        target = getattr(target, "config", target)
    return _target_value(target, "transport_support", stage="model_compatibility")


def validate_radius_model_compatibility(
    config: InteractionRadiusConfig,
    model_or_config: Any,
) -> DerivedInteractionRadii:
    """Require existing model OT support radii to equal the derived controls.

    The support kind/backend are deliberately ignored: they are execution
    policy, whereas the three compared values are the physical support radii.
    """

    if not isinstance(config, InteractionRadiusConfig):
        raise _radius_error(
            "INVALID_RADIUS_CONFIG",
            "config must be an InteractionRadiusConfig",
            stage="model_compatibility.input",
            actual=type(config).__name__,
        )
    support = _model_support(model_or_config)
    cutoff = _compatibility_real(
        _target_value(support, "cutoff", stage="model_compatibility"),
        field="transport_support.cutoff",
        stage="model_compatibility",
    )
    switch_width = _compatibility_real(
        _target_value(support, "switch_width", stage="model_compatibility"),
        field="transport_support.switch_width",
        stage="model_compatibility",
    )
    candidate_skin = _compatibility_real(
        _target_value(support, "candidate_skin", stage="model_compatibility"),
        field="transport_support.candidate_skin",
        stage="model_compatibility",
    )
    actual = {
        "r_on": cutoff - switch_width,
        "r_off": cutoff,
        "r_candidate": cutoff + candidate_skin,
    }
    expected_radii = derive_interaction_radii(config)
    expected = {
        "r_on": expected_radii.r_on_ot,
        "r_off": expected_radii.r_off_ot,
        "r_candidate": expected_radii.r_candidate_ot,
    }
    mismatches = tuple(
        (field, expected[field], actual[field])
        for field in ("r_on", "r_off", "r_candidate")
        if expected[field] != actual[field]
    )
    if mismatches:
        raise _radius_error(
            "RADIUS_MODEL_MISMATCH",
            "model transport support radii do not match the requested config",
            stage="model_compatibility",
            mismatches=mismatches,
            action=(
                "Treat changes to r_ot, ot_switch_width, or ot_skin as a new "
                "model run; the existing model/checkpoint was not modified."
            ),
        )
    return expected_radii


__all__ = [
    "INTERACTION_RADIUS_CONFIG_SCHEMA_VERSION",
    "INTERACTION_RADIUS_CONVENTION_VERSION",
    "INTERACTION_RADIUS_LENGTH_UNIT",
    "DerivedInteractionRadii",
    "InteractionRadiusConfig",
    "RadiusConfigError",
    "derive_interaction_radii",
    "transport_support_config_from_radii",
    "validate_radius_artifact_compatibility",
    "validate_radius_model_compatibility",
]
