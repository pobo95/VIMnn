"""Configuration, layout metadata, and results for probability multipoles."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Optional

import torch

from .radial import RADIAL_BASIS_VERSION


FEATURE_LAYOUT_VERSION = "ot_probability_density_multipoles_v1"


@dataclass(frozen=True)
class ProbabilityMultipoleConfig:
    species_vocabulary: tuple[int, ...]
    n_radial: int = 3
    lmax: int = 2
    ell_feature: float = 1.0
    r_cut: float = 3.0
    probability_tolerance: Optional[float] = None
    site_type_vocabulary: Optional[tuple[int, ...]] = None
    feature_layout_version: str = FEATURE_LAYOUT_VERSION
    radial_basis_version: str = RADIAL_BASIS_VERSION
    e3nn_normalization: str = "component"
    e3nn_normalize: bool = False
    displacement_orientation: str = "reference_site_to_atom"

    def validate(self) -> None:
        if (
            len(self.species_vocabulary) == 0
            or len(set(self.species_vocabulary)) != len(self.species_vocabulary)
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
                for value in self.species_vocabulary
            )
        ):
            raise ValueError("species_vocabulary must contain unique positive integers")
        if (
            isinstance(self.n_radial, bool)
            or not isinstance(self.n_radial, Integral)
            or self.n_radial <= 0
        ):
            raise ValueError("n_radial must be a positive integer")
        if self.lmax != 2:
            raise ValueError("feature layout v1 requires lmax=2")
        for name, value in (
            ("ell_feature", self.ell_feature),
            ("r_cut", self.r_cut),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.probability_tolerance is not None:
            value = self.probability_tolerance
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    "probability_tolerance must be None or finite and positive"
                )
        if self.site_type_vocabulary is not None and (
            len(set(self.site_type_vocabulary))
            != len(self.site_type_vocabulary)
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
                for value in self.site_type_vocabulary
            )
        ):
            raise ValueError(
                "site_type_vocabulary must contain unique nonnegative integers"
            )
        if self.feature_layout_version != FEATURE_LAYOUT_VERSION:
            raise ValueError("unsupported feature layout version")
        if self.radial_basis_version != RADIAL_BASIS_VERSION:
            raise ValueError("unsupported radial basis version")
        if self.e3nn_normalization != "component" or self.e3nn_normalize:
            raise ValueError("v1 requires component normalization and normalize=False")
        if self.displacement_orientation != "reference_site_to_atom":
            raise ValueError("unsupported displacement orientation")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["species_vocabulary"] = list(self.species_vocabulary)
        if self.site_type_vocabulary is not None:
            result["site_type_vocabulary"] = list(self.site_type_vocabulary)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ProbabilityMultipoleConfig":
        copied = dict(values)
        copied["species_vocabulary"] = tuple(copied["species_vocabulary"])
        if copied.get("site_type_vocabulary") is not None:
            copied["site_type_vocabulary"] = tuple(
                copied["site_type_vocabulary"]
            )
        config = cls(**copied)
        config.validate()
        return config


@dataclass(frozen=True)
class ChannelMetadata:
    block_order: int
    block_name: str
    species: int
    l: int
    parity: str
    radial_index: Optional[int]
    component_slice: tuple[int, int]
    exact_occupancy: bool


@dataclass(frozen=True)
class ProbabilityMultipoleResult:
    species_probabilities: torch.Tensor
    vacancy_probabilities: torch.Tensor
    raw_probability_state: torch.Tensor
    equivariant_features: torch.Tensor
    irreps_out: object
    channel_metadata: tuple[ChannelMetadata, ...]
    species_vocabulary: tuple[int, ...]
    site_types: Optional[torch.Tensor]
    site_type_vocabulary: Optional[tuple[int, ...]]
    config_metadata: dict[str, Any]
