"""Validated immutable configuration for the reference-site potential."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Mapping

from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.transport import TransportSupportConfig


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _positive_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real")
    return result


def _positive_schedule(name: str, values: Any) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a nonempty sequence")
    if not values:
        raise ValueError(f"{name} must be a nonempty sequence")
    return tuple(
        _positive_real(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )


@dataclass(frozen=True)
class PotentialConfig:
    species_vocabulary: tuple[int, ...]
    num_layers: int
    feature: ProbabilityMultipoleConfig
    higher_body: HigherBodyConfig
    readout_hidden: int = 16
    energy_scale: float = 1.0
    epsilon_ot: float = 0.5
    ell_ot: float = 1.5
    train_sinkhorn_iterations: int = 256
    phase_steps: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)
    phase_damping: tuple[float, ...] = (2.0, 1.0, 0.5, 0.2)
    transport_support: TransportSupportConfig = field(
        default_factory=TransportSupportConfig
    )
    eval_sinkhorn_warmup_iterations: int = 16

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            not isinstance(self.species_vocabulary, tuple)
            or not self.species_vocabulary
            or len(set(self.species_vocabulary)) != len(self.species_vocabulary)
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
                for value in self.species_vocabulary
            )
        ):
            raise ValueError(
                "species_vocabulary must be a nonempty tuple of unique positive integers"
            )
        _positive_integer("num_layers", self.num_layers)
        _positive_integer("readout_hidden", self.readout_hidden)
        _positive_integer(
            "train_sinkhorn_iterations", self.train_sinkhorn_iterations
        )
        _nonnegative_integer(
            "eval_sinkhorn_warmup_iterations",
            self.eval_sinkhorn_warmup_iterations,
        )
        _positive_real("energy_scale", self.energy_scale)
        _positive_real("epsilon_ot", self.epsilon_ot)
        _positive_real("ell_ot", self.ell_ot)
        steps = _positive_schedule("phase_steps", self.phase_steps)
        damping = _positive_schedule("phase_damping", self.phase_damping)
        if len(steps) != len(damping):
            raise ValueError(
                "phase_steps and phase_damping must have equal positive length"
            )
        if not isinstance(self.feature, ProbabilityMultipoleConfig):
            raise TypeError("feature must be a ProbabilityMultipoleConfig")
        if not isinstance(self.higher_body, HigherBodyConfig):
            raise TypeError("higher_body must be a HigherBodyConfig")
        if not isinstance(self.transport_support, TransportSupportConfig):
            raise TypeError("transport_support must be TransportSupportConfig")
        self.feature.validate()
        self.higher_body.validate()
        if self.feature.species_vocabulary != self.species_vocabulary:
            raise ValueError("feature species mismatch")
        if self.higher_body.species_count != len(self.species_vocabulary):
            raise ValueError("higher-body species mismatch")
        if self.higher_body.irreps_feature != str(self.feature_irreps):
            raise ValueError("higher-body feature irreps mismatch")
        if self.feature.r_cut != self.higher_body.cutoff:
            raise ValueError(
                "feature.r_cut and higher_body.cutoff must match the v1 MP radius"
            )
        if self.feature.site_type_vocabulary is not None and (
            self.higher_body.site_type_count
            != len(self.feature.site_type_vocabulary)
        ):
            raise ValueError(
                "higher-body site_type_count must match feature site-type channels"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "species_vocabulary": list(self.species_vocabulary),
            "num_layers": self.num_layers,
            "feature": self.feature.to_dict(),
            "higher_body": self.higher_body.to_dict(),
            "readout_hidden": self.readout_hidden,
            "energy_scale": self.energy_scale,
            "epsilon_ot": self.epsilon_ot,
            "ell_ot": self.ell_ot,
            "train_sinkhorn_iterations": self.train_sinkhorn_iterations,
            "phase_steps": list(self.phase_steps),
            "phase_damping": list(self.phase_damping),
            "transport_support": self.transport_support.to_dict(),
            "eval_sinkhorn_warmup_iterations": self.eval_sinkhorn_warmup_iterations,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PotentialConfig":
        if not isinstance(values, Mapping):
            raise TypeError("potential config must be reconstructed from a mapping")
        data = dict(values)
        data["species_vocabulary"] = tuple(data["species_vocabulary"])
        data["feature"] = ProbabilityMultipoleConfig.from_dict(data["feature"])
        data["higher_body"] = HigherBodyConfig.from_dict(data["higher_body"])
        data["phase_steps"] = tuple(data.get("phase_steps", (0.7, 0.8, 0.9, 1.0)))
        data["phase_damping"] = tuple(
            data.get("phase_damping", (2.0, 1.0, 0.5, 0.2))
        )
        data["transport_support"] = TransportSupportConfig.from_dict(
            data.get("transport_support")
        )
        return cls(**data)

    @property
    def feature_irreps(self):
        from refsite_mlip.compatibility import import_e3nn_0_4_4

        _, o3 = import_e3nn_0_4_4()
        species_count = len(self.species_vocabulary)
        radial_count = self.feature.n_radial
        return o3.Irreps(
            f"{species_count}x0e + {species_count * radial_count}x0e + "
            f"{species_count * radial_count}x1o + "
            f"{species_count * radial_count}x2e"
        )
