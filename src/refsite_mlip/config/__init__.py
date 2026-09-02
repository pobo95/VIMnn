"""Stable user-facing configuration contracts."""

from .radii import (
    INTERACTION_RADIUS_CONFIG_SCHEMA_VERSION,
    INTERACTION_RADIUS_CONVENTION_VERSION,
    INTERACTION_RADIUS_LENGTH_UNIT,
    DerivedInteractionRadii,
    InteractionRadiusConfig,
    RadiusConfigError,
    derive_interaction_radii,
    transport_support_config_from_radii,
    validate_radius_artifact_compatibility,
    validate_radius_model_compatibility,
)

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
