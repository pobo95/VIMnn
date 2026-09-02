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
from .training_run import (
    TRAINING_RUN_CONFIG_SCHEMA_VERSION,
    ResolvedTrainingRun,
    TrainingDataConfig,
    TrainingRuntimeConfig,
    TrainingRunConfig,
    TrainingRunConfigError,
    load_training_run_config,
    resolve_training_run,
    validate_training_run_config,
)

__all__ = [
    "INTERACTION_RADIUS_CONFIG_SCHEMA_VERSION",
    "INTERACTION_RADIUS_CONVENTION_VERSION",
    "INTERACTION_RADIUS_LENGTH_UNIT",
    "TRAINING_RUN_CONFIG_SCHEMA_VERSION",
    "DerivedInteractionRadii",
    "InteractionRadiusConfig",
    "RadiusConfigError",
    "ResolvedTrainingRun",
    "TrainingDataConfig",
    "TrainingRuntimeConfig",
    "TrainingRunConfig",
    "TrainingRunConfigError",
    "derive_interaction_radii",
    "load_training_run_config",
    "resolve_training_run",
    "transport_support_config_from_radii",
    "validate_radius_artifact_compatibility",
    "validate_radius_model_compatibility",
    "validate_training_run_config",
]
