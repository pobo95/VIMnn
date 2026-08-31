"""Immutable snapshots of existing phase and reference-graph metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import torch

from refsite_mlip.graph import (
    ReferenceGraphTopology,
    update_reference_edge_geometry,
)
from refsite_mlip.phase.types import TypedStabilizer

from .template_domain import (
    StrictTemplateDomain,
    TemplateDomainValidation,
)


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().clone()


def _tensor_hash(digest, tensor: torch.Tensor) -> None:
    tensor = _clone(tensor)
    digest.update(str(tensor.dtype).encode())
    digest.update(struct.pack("<I", tensor.ndim))
    digest.update(struct.pack("<" + "q" * tensor.ndim, *tensor.shape))
    digest.update(tensor.numpy().tobytes())


def _text(digest, value) -> None:
    data = str(value).encode()
    digest.update(struct.pack("<Q", len(data)))
    digest.update(data)


@dataclass(frozen=True)
class ReferenceTemplate:
    """CPU snapshot of all physical metadata defining one template."""

    template_id: str
    topology: ReferenceGraphTopology
    phase_modes: torch.Tensor
    phase_mode_weights: torch.Tensor
    site_alignment_weights: torch.Tensor
    phase_channel_weights: torch.Tensor
    stabilizer: TypedStabilizer
    supported_species: tuple[int, ...]
    convention_version: str = "reference_template_v1"
    strict_domain: StrictTemplateDomain | None = None

    @classmethod
    def snapshot(
        cls,
        template_id,
        topology,
        phase_modes,
        phase_mode_weights,
        site_alignment_weights,
        phase_channel_weights,
        stabilizer,
        supported_species,
        convention_version="reference_template_v1",
        strict_domain=None,
    ) -> "ReferenceTemplate":
        if not isinstance(template_id, str) or not template_id:
            raise ValueError("template_id must be nonempty")
        topology_snapshot = ReferenceGraphTopology(
            reference_fractional=_clone(topology.reference_fractional),
            site_types=_clone(topology.site_types),
            edge_index=_clone(topology.edge_index),
            shifts=_clone(topology.shifts),
            reference_cell=_clone(topology.reference_cell),
            cutoff=topology.cutoff,
            skin=topology.skin,
            maximum_strain=topology.maximum_strain,
            minimum_edge_length=topology.minimum_edge_length,
            pbc=tuple(topology.pbc),
        )
        result = cls(
            template_id=template_id,
            topology=topology_snapshot,
            phase_modes=_clone(phase_modes),
            phase_mode_weights=_clone(phase_mode_weights),
            site_alignment_weights=_clone(site_alignment_weights),
            phase_channel_weights=_clone(phase_channel_weights),
            stabilizer=TypedStabilizer(
                _clone(stabilizer.translations),
                _clone(stabilizer.permutations),
            ),
            supported_species=tuple(int(value) for value in supported_species),
            convention_version=convention_version,
            strict_domain=(
                None
                if strict_domain is None
                else StrictTemplateDomain.from_dict(strict_domain.to_dict())
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not isinstance(self.convention_version, str) or not self.convention_version:
            raise ValueError("convention_version must be nonempty")
        num_sites = self.topology.num_sites
        if (
            self.phase_modes.ndim != 2
            or self.phase_modes.shape[1] != 3
            or self.phase_modes.dtype != torch.long
        ):
            raise ValueError("phase_modes must be long [G,3]")
        num_modes = self.phase_modes.shape[0]
        if (
            self.phase_mode_weights.shape != (num_modes,)
            or self.site_alignment_weights.ndim != 2
            or self.site_alignment_weights.shape[0] != num_sites
        ):
            raise ValueError("phase/template alignment shape mismatch")
        num_channels = self.site_alignment_weights.shape[1]
        if self.phase_channel_weights.shape != (num_channels,):
            raise ValueError("phase channel shape mismatch")
        if (
            self.stabilizer.translations.ndim != 2
            or self.stabilizer.translations.shape[1] != 3
            or self.stabilizer.permutations.ndim != 2
            or self.stabilizer.permutations.shape[1] != num_sites
            or self.stabilizer.translations.shape[0]
            != self.stabilizer.permutations.shape[0]
        ):
            raise ValueError("stabilizer shape mismatch")
        if self.stabilizer.permutations.dtype != torch.long:
            raise ValueError("stabilizer permutations must use torch.long")
        if (
            not self.supported_species
            or len(set(self.supported_species)) != len(self.supported_species)
            or any(value <= 0 for value in self.supported_species)
        ):
            raise ValueError(
                "supported_species must be unique positive integers"
            )
        if self.strict_domain is not None:
            if not isinstance(self.strict_domain, StrictTemplateDomain):
                raise TypeError("strict_domain must be a StrictTemplateDomain")
            if self.strict_domain.reference_site_count != num_sites:
                raise ValueError("strict domain M differs from template topology")
            if self.strict_domain.species_vocabulary != self.supported_species:
                raise ValueError(
                    "strict domain species order differs from supported_species"
                )
            self.strict_domain.validate_reference_site_types(
                self.topology.site_types
            )
        floating_metadata = (
            self.phase_mode_weights,
            self.site_alignment_weights,
            self.phase_channel_weights,
            self.stabilizer.translations,
        )
        for tensor in floating_metadata:
            if not tensor.is_floating_point() or not bool(
                torch.all(torch.isfinite(tensor))
            ):
                raise ValueError("template contains invalid floating metadata")

    @property
    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 over all physical conventions."""

        digest = hashlib.sha256()
        _text(digest, self.convention_version)
        _text(digest, self.template_id)
        topology = self.topology
        tensors = (
            topology.reference_fractional,
            topology.site_types,
            topology.edge_index,
            topology.shifts,
            topology.reference_cell,
            self.phase_modes,
            self.phase_mode_weights,
            self.site_alignment_weights,
            self.phase_channel_weights,
            self.stabilizer.translations,
            self.stabilizer.permutations,
        )
        for tensor in tensors:
            _tensor_hash(digest, tensor)
        scalars = (
            topology.cutoff,
            topology.skin,
            topology.maximum_strain,
            topology.minimum_edge_length,
            topology.pbc,
            self.supported_species,
        )
        for value in scalars:
            _text(digest, value)
        # Preserve the exact legacy hash byte stream when no strict domain is
        # present.  New strict templates bind every domain field explicitly.
        if self.strict_domain is not None:
            _text(digest, "strict_domain")
            for value in self.strict_domain.fingerprint_values():
                _text(digest, value)
        return digest.hexdigest()

    def validate_structure(
        self,
        atomic_numbers: torch.Tensor,
        *,
        cell: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
        sample_id: str | None = None,
    ) -> TemplateDomainValidation:
        """Validate assignment without mutating structure or template tensors."""

        if not isinstance(atomic_numbers, torch.Tensor):
            raise TypeError("atomic_numbers must be a torch.Tensor")
        if atomic_numbers.ndim != 1 or atomic_numbers.dtype != torch.long:
            raise ValueError("atomic_numbers must be torch.long [N]")
        num_atoms = int(atomic_numbers.numel())
        if num_atoms > self.topology.num_sites:
            raise ValueError(
                f"N > M for sample {sample_id}" if sample_id else "N > M"
            )
        actual_species = set(
            int(value) for value in atomic_numbers.detach().cpu().tolist()
        )
        if not actual_species.issubset(set(self.supported_species)):
            raise ValueError(f"unknown species for template {self.template_id}")

        if self.strict_domain is None:
            return TemplateDomainValidation(
                num_atoms,
                self.topology.num_sites - num_atoms,
                tuple(
                    int(torch.sum(atomic_numbers.detach().cpu() == species))
                    for species in self.supported_species
                ),
            )

        result = self.strict_domain.validate_atomic_numbers(
            atomic_numbers,
            template_id=self.template_id,
            sample_id=sample_id,
        )
        if pbc is not None:
            if (
                pbc.shape != (3,)
                or pbc.dtype != torch.bool
                or not bool(torch.all(pbc))
            ):
                raise ValueError("strict template domain requires full PBC")
        if cell is None:
            return result
        if cell.shape != (3, 3) or cell.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("strict template cell must be float32/float64 [3,3]")
        if not bool(torch.all(torch.isfinite(cell))):
            raise ValueError("strict template cell contains NaN or Inf")
        singular_values = torch.linalg.svdvals(cell)
        if bool(singular_values[-1] <= torch.finfo(cell.dtype).eps):
            raise ValueError("strict template cell must be nonsingular")
        topology = self.topology.to(device=cell.device, dtype=cell.dtype)
        geometry = update_reference_edge_geometry(
            topology,
            cell,
            edge_length_scale=1.0,
        )
        return TemplateDomainValidation(
            result.num_atoms,
            result.vacancy_mass,
            result.composition,
            float(geometry.maximum_strain_seen.detach().cpu()),
        )

    def clone(self) -> "ReferenceTemplate":
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


class TemplateRegistry:
    """In-memory template lookup with immutable stored snapshots."""

    def __init__(self) -> None:
        self._entries: dict[str, ReferenceTemplate] = {}

    def add(self, template: ReferenceTemplate) -> None:
        if not isinstance(template, ReferenceTemplate):
            raise TypeError("template must be a ReferenceTemplate")
        snapshot = template.clone()
        current = self._entries.get(snapshot.template_id)
        if current is not None and current.fingerprint != snapshot.fingerprint:
            raise ValueError(f"conflicting template_id: {snapshot.template_id}")
        self._entries[snapshot.template_id] = snapshot

    def resolve(self, template_id: str) -> ReferenceTemplate:
        if template_id not in self._entries:
            raise KeyError(f"unknown template_id: {template_id}")
        return self._entries[template_id].clone()

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for key in sorted(self._entries):
            _text(digest, key)
            _text(digest, self._entries[key].fingerprint)
        return digest.hexdigest()
