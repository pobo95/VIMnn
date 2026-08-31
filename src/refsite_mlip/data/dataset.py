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
            template.validate_structure(
                sample.atomic_numbers,
                cell=sample.cell if template.strict_domain is not None else None,
                pbc=sample.pbc if template.strict_domain is not None else None,
                sample_id=sample.sample_id,
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]
