"""Generic in-memory structure and reference-template contracts."""

from .batch import StructureBatch, TemplateGroup, collate_structure_samples
from .dataset import InMemoryStructureDataset
from .schema import (
    ENERGY_UNIT,
    FORCE_UNIT,
    LENGTH_UNIT,
    STRESS_SIGN,
    STRESS_UNIT,
    STRESS_VOIGT_ORDER,
    StructureSample,
)
from .templates import ReferenceTemplate, TemplateRegistry
from .template_domain import StrictTemplateDomain, TemplateDomainValidation
from .reference_builder import (
    CanonicalReferenceStructure,
    PhaseSpecification,
    ReferenceTemplateBuildDiagnostics,
    ReferenceTemplateBuilderConfig,
    ReferenceTemplateBuildResult,
    build_reference_template_from_atoms,
    build_reference_template_from_poscar,
    canonicalize_reference_atoms,
    nbc_rocksalt_template_builder_config,
)
from .extxyz import (
    EXTXYZ_LOADER_CONVENTION_VERSION,
    EXTXYZ_UNIT_CONVENTION_VERSION,
    ExtXYZLoadConfig,
    ExtXYZLoadDiagnostics,
    ExtXYZLoadError,
    ExtXYZLoadResult,
    NumericStatistics,
    load_extxyz_dataset,
    load_extxyz_samples,
)

__all__ = [
    "ENERGY_UNIT",
    "EXTXYZ_LOADER_CONVENTION_VERSION",
    "EXTXYZ_UNIT_CONVENTION_VERSION",
    "ExtXYZLoadConfig",
    "ExtXYZLoadDiagnostics",
    "ExtXYZLoadError",
    "ExtXYZLoadResult",
    "FORCE_UNIT",
    "CanonicalReferenceStructure",
    "InMemoryStructureDataset",
    "LENGTH_UNIT",
    "NumericStatistics",
    "ReferenceTemplate",
    "ReferenceTemplateBuildDiagnostics",
    "ReferenceTemplateBuilderConfig",
    "ReferenceTemplateBuildResult",
    "PhaseSpecification",
    "STRESS_SIGN",
    "STRESS_UNIT",
    "STRESS_VOIGT_ORDER",
    "StrictTemplateDomain",
    "StructureBatch",
    "StructureSample",
    "TemplateGroup",
    "TemplateDomainValidation",
    "TemplateRegistry",
    "build_reference_template_from_atoms",
    "build_reference_template_from_poscar",
    "canonicalize_reference_atoms",
    "collate_structure_samples",
    "load_extxyz_dataset",
    "load_extxyz_samples",
    "nbc_rocksalt_template_builder_config",
]
