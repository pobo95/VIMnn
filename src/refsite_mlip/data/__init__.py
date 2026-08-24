"""Generic in-memory structure and reference-template contracts."""

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

__all__ = [
    "ENERGY_UNIT",
    "FORCE_UNIT",
    "InMemoryStructureDataset",
    "LENGTH_UNIT",
    "ReferenceTemplate",
    "STRESS_SIGN",
    "STRESS_UNIT",
    "STRESS_VOIGT_ORDER",
    "StructureSample",
    "TemplateRegistry",
]
