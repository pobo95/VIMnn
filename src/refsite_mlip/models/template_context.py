"""Immutable runtime ownership for nontrainable reference-template state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch

from refsite_mlip.data.templates import ReferenceTemplate
from refsite_mlip.data.template_domain import StrictTemplateDomain
from refsite_mlip.graph import ReferenceGraphTopology
from refsite_mlip.phase.types import TypedStabilizer


def _validate_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("template floating dtype must be float32 or float64")


def _validate_avg_num_neighbors(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("avg_num_neighbors must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("avg_num_neighbors must be a positive finite real number")
    return result


@dataclass(frozen=True)
class _MaterializedTemplateExecutionContext:
    template_id: str
    fingerprint: str
    topology: ReferenceGraphTopology
    phase_modes: torch.Tensor
    phase_mode_weights: torch.Tensor
    site_alignment_weights: torch.Tensor
    phase_channel_weights: torch.Tensor
    stabilizer: TypedStabilizer
    supported_species: tuple[int, ...]
    convention_version: str
    avg_num_neighbors: float
    strict_domain: StrictTemplateDomain | None = None


@dataclass(frozen=True)
class TemplateExecutionContext:
    """Canonical CPU snapshot of one template's nontrainable execution state.

    The context is intentionally not an ``nn.Module`` and owns no trainable
    tensors.  Geometry depending on the current positions or cell is never
    stored here.
    """

    template_id: str
    fingerprint: str
    topology: ReferenceGraphTopology
    phase_modes: torch.Tensor
    phase_mode_weights: torch.Tensor
    site_alignment_weights: torch.Tensor
    phase_channel_weights: torch.Tensor
    stabilizer: TypedStabilizer
    supported_species: tuple[int, ...]
    convention_version: str
    avg_num_neighbors: float
    strict_domain: StrictTemplateDomain | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "avg_num_neighbors",
            _validate_avg_num_neighbors(self.avg_num_neighbors),
        )
        tensors = (
            self.topology.reference_fractional,
            self.topology.site_types,
            self.topology.edge_index,
            self.topology.shifts,
            self.topology.reference_cell,
            self.phase_modes,
            self.phase_mode_weights,
            self.site_alignment_weights,
            self.phase_channel_weights,
            self.stabilizer.translations,
            self.stabilizer.permutations,
        )
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("TemplateExecutionContext must own CPU tensor snapshots")
        if any(tensor.requires_grad for tensor in tensors):
            raise ValueError("TemplateExecutionContext tensors must not require gradients")
        self.validate_fingerprint()

    @classmethod
    def from_reference_template(
        cls,
        template: ReferenceTemplate,
        *,
        avg_num_neighbors: Real,
    ) -> "TemplateExecutionContext":
        """Snapshot a data-layer template without creating a dependency cycle."""

        if not isinstance(template, ReferenceTemplate):
            raise TypeError("template must be a ReferenceTemplate")
        avg_num_neighbors = _validate_avg_num_neighbors(avg_num_neighbors)
        snapshot = template.clone()
        context = cls(
            template_id=snapshot.template_id,
            fingerprint=snapshot.fingerprint,
            topology=snapshot.topology,
            phase_modes=snapshot.phase_modes,
            phase_mode_weights=snapshot.phase_mode_weights,
            site_alignment_weights=snapshot.site_alignment_weights,
            phase_channel_weights=snapshot.phase_channel_weights,
            stabilizer=snapshot.stabilizer,
            supported_species=snapshot.supported_species,
            convention_version=snapshot.convention_version,
            avg_num_neighbors=avg_num_neighbors,
            strict_domain=snapshot.strict_domain,
        )
        return context

    @property
    def reference_fractional(self) -> torch.Tensor:
        return self.topology.reference_fractional

    @property
    def site_types(self) -> torch.Tensor:
        return self.topology.site_types

    @property
    def reference_cell(self) -> torch.Tensor:
        return self.topology.reference_cell

    @property
    def edge_index(self) -> torch.Tensor:
        return self.topology.edge_index

    @property
    def shifts(self) -> torch.Tensor:
        return self.topology.shifts

    def _reference_template_snapshot(self) -> ReferenceTemplate:
        return ReferenceTemplate.snapshot(
            self.template_id,
            self.topology,
            self.phase_modes,
            self.phase_mode_weights,
            self.site_alignment_weights,
            self.phase_channel_weights,
            self.stabilizer,
            self.supported_species,
            self.convention_version,
            self.strict_domain,
        )

    def validate_fingerprint(self) -> None:
        actual = self._reference_template_snapshot().fingerprint
        if actual != self.fingerprint:
            raise ValueError(
                "template context fingerprint does not match its content"
            )

    def materialize(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> _MaterializedTemplateExecutionContext:
        """Return a fresh runtime copy; no persistent device cache is retained."""

        _validate_dtype(dtype)
        self.validate_fingerprint()
        target_device = torch.device(device)
        topology = self.topology.to(device=target_device, dtype=dtype)
        return _MaterializedTemplateExecutionContext(
            template_id=self.template_id,
            fingerprint=self.fingerprint,
            topology=topology,
            phase_modes=self.phase_modes.to(device=target_device),
            phase_mode_weights=self.phase_mode_weights.to(
                device=target_device, dtype=dtype
            ),
            site_alignment_weights=self.site_alignment_weights.to(
                device=target_device, dtype=dtype
            ),
            phase_channel_weights=self.phase_channel_weights.to(
                device=target_device, dtype=dtype
            ),
            stabilizer=TypedStabilizer(
                self.stabilizer.translations.to(
                    device=target_device, dtype=dtype
                ),
                self.stabilizer.permutations.to(device=target_device),
            ),
            supported_species=self.supported_species,
            convention_version=self.convention_version,
            avg_num_neighbors=self.avg_num_neighbors,
            strict_domain=self.strict_domain,
        )
