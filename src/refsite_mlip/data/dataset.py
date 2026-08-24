"""Small generic in-memory structure dataset."""

from __future__ import annotations

from collections.abc import Sequence

from .schema import StructureSample
from .templates import TemplateRegistry


class InMemoryStructureDataset(Sequence):
    """Validated sequence of structures backed by an explicit registry."""

    def __init__(self, samples, registry: TemplateRegistry) -> None:
        if not isinstance(registry, TemplateRegistry):
            raise TypeError("registry must be a TemplateRegistry")
        self.registry = registry
        self._samples = tuple(samples)
        sample_ids: set[str] = set()
        for sample in self._samples:
            if not isinstance(sample, StructureSample):
                raise TypeError("dataset entries must be StructureSample")
            if sample.sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id: {sample.sample_id}")
            sample_ids.add(sample.sample_id)
            template = registry.resolve(sample.template_id)
            if sample.num_atoms > template.topology.num_sites:
                raise ValueError(f"N > M for sample {sample.sample_id}")
            species = set(sample.atomic_numbers.detach().cpu().tolist())
            if not species.issubset(set(template.supported_species)):
                raise ValueError(
                    f"unknown species for template {sample.template_id}"
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]
