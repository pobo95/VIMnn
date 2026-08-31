"""Strict, serializable structure domains for reference templates."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import torch


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class TemplateDomainValidation:
    """Detached result of validating one composition against a template."""

    num_atoms: int
    vacancy_mass: int
    composition: tuple[int, ...]
    maximum_strain_seen: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_atoms": self.num_atoms,
            "vacancy_mass": self.vacancy_mass,
            "composition": list(self.composition),
            "maximum_strain_seen": self.maximum_strain_seen,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemplateDomainValidation":
        if not isinstance(payload, Mapping):
            raise TypeError("template domain validation payload must be a mapping")
        strain = payload.get("maximum_strain_seen")
        return cls(
            num_atoms=_nonnegative_int(payload["num_atoms"], name="num_atoms"),
            vacancy_mass=_nonnegative_int(
                payload["vacancy_mass"], name="vacancy_mass"
            ),
            composition=tuple(
                _nonnegative_int(value, name="composition count")
                for value in payload["composition"]
            ),
            maximum_strain_seen=None if strain is None else float(strain),
        )


@dataclass(frozen=True)
class StrictTemplateDomain:
    """Exact composition and vacancy contract for one reference-site family.

    Entries in ``reference_composition`` and every allowed composition use the
    authoritative ``species_vocabulary`` order.  The three allowed tuples are
    parallel: entry ``j`` describes one complete permitted structure state.
    """

    reference_site_count: int
    supercell_shape: tuple[int, int, int]
    species_vocabulary: tuple[int, ...]
    reference_composition: tuple[int, ...]
    allowed_compositions: tuple[tuple[int, ...], ...]
    allowed_num_atoms: tuple[int, ...]
    allowed_vacancy_masses: tuple[int, ...]
    convention_version: str = "strict_template_domain_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_site_count",
            _positive_int(self.reference_site_count, name="reference_site_count"),
        )
        shape = tuple(
            _positive_int(value, name="supercell dimension")
            for value in self.supercell_shape
        )
        if len(shape) != 3:
            raise ValueError("supercell_shape must contain three dimensions")
        object.__setattr__(self, "supercell_shape", shape)

        vocabulary = tuple(
            _positive_int(value, name="atomic number")
            for value in self.species_vocabulary
        )
        if not vocabulary or len(set(vocabulary)) != len(vocabulary):
            raise ValueError(
                "species_vocabulary must contain unique positive atomic numbers"
            )
        object.__setattr__(self, "species_vocabulary", vocabulary)

        reference = tuple(
            _nonnegative_int(value, name="reference composition count")
            for value in self.reference_composition
        )
        if len(reference) != len(vocabulary):
            raise ValueError(
                "reference_composition must follow species_vocabulary order"
            )
        if sum(reference) != self.reference_site_count:
            raise ValueError(
                "reference composition must sum to reference_site_count"
            )
        object.__setattr__(self, "reference_composition", reference)

        compositions = tuple(
            tuple(
                _nonnegative_int(value, name="allowed composition count")
                for value in composition
            )
            for composition in self.allowed_compositions
        )
        numbers = tuple(
            _positive_int(value, name="allowed_num_atoms")
            for value in self.allowed_num_atoms
        )
        vacancies = tuple(
            _nonnegative_int(value, name="allowed_vacancy_masses")
            for value in self.allowed_vacancy_masses
        )
        if not compositions or not (
            len(compositions) == len(numbers) == len(vacancies)
        ):
            raise ValueError(
                "allowed compositions, atom counts, and vacancy masses must be nonempty parallel tuples"
            )
        if len(set(compositions)) != len(compositions):
            raise ValueError("allowed compositions must be unique")
        for composition, num_atoms, vacancy_mass in zip(
            compositions, numbers, vacancies
        ):
            if len(composition) != len(vocabulary):
                raise ValueError(
                    "allowed composition must follow species_vocabulary order"
                )
            if any(
                count > reference_count
                for count, reference_count in zip(composition, reference)
            ):
                raise ValueError(
                    "allowed composition cannot exceed reference composition"
                )
            if sum(composition) != num_atoms:
                raise ValueError("allowed composition does not sum to N")
            if self.reference_site_count - num_atoms != vacancy_mass:
                raise ValueError("allowed N and vacancy mass K=M-N disagree")
        object.__setattr__(self, "allowed_compositions", compositions)
        object.__setattr__(self, "allowed_num_atoms", numbers)
        object.__setattr__(self, "allowed_vacancy_masses", vacancies)

        if not isinstance(self.convention_version, str) or not self.convention_version:
            raise ValueError("domain convention_version must be nonempty")

    def composition_for(self, atomic_numbers: torch.Tensor) -> tuple[int, ...]:
        if not isinstance(atomic_numbers, torch.Tensor):
            raise TypeError("atomic_numbers must be a torch.Tensor")
        if atomic_numbers.ndim != 1 or atomic_numbers.dtype != torch.long:
            raise ValueError("atomic_numbers must be torch.long [N]")
        values = atomic_numbers.detach().cpu()
        known = torch.tensor(self.species_vocabulary, dtype=torch.long)
        if values.numel():
            matches = values[:, None] == known[None, :]
            if bool(torch.any(matches.sum(dim=1) != 1)):
                unknown = sorted(
                    set(int(value) for value in values.tolist())
                    - set(self.species_vocabulary)
                )
                raise ValueError(
                    f"strict template domain rejects unknown species {unknown}"
                )
        return tuple(int(torch.sum(values == value)) for value in known)

    def validate_atomic_numbers(
        self,
        atomic_numbers: torch.Tensor,
        *,
        template_id: str | None = None,
        sample_id: str | None = None,
    ) -> TemplateDomainValidation:
        composition = self.composition_for(atomic_numbers)
        num_atoms = int(atomic_numbers.numel())
        vacancy_mass = self.reference_site_count - num_atoms
        context = ""
        if sample_id is not None:
            context += f" sample_id={sample_id}"
        if template_id is not None:
            context += f" template_id={template_id}"
        if composition not in self.allowed_compositions:
            observed = {
                str(species): count
                for species, count in zip(self.species_vocabulary, composition)
            }
            allowed = [
                {
                    str(species): count
                    for species, count in zip(self.species_vocabulary, entry)
                }
                for entry in self.allowed_compositions
            ]
            raise ValueError(
                "strict template domain rejects composition"
                f"{context} observed={observed} allowed={allowed}"
            )
        index = self.allowed_compositions.index(composition)
        if (
            num_atoms != self.allowed_num_atoms[index]
            or vacancy_mass != self.allowed_vacancy_masses[index]
        ):
            raise ValueError(
                "strict template domain N/K invariant failed"
                f"{context} N={num_atoms} K={vacancy_mass}"
            )
        return TemplateDomainValidation(num_atoms, vacancy_mass, composition)

    def validate_reference_site_types(self, site_types: torch.Tensor) -> None:
        if site_types.shape != (self.reference_site_count,):
            raise ValueError("strict domain reference site count mismatch")
        if site_types.dtype != torch.long:
            raise ValueError("strict domain site_types must use torch.long")
        expected_types = set(range(len(self.species_vocabulary)))
        actual_types = set(int(value) for value in site_types.detach().cpu().tolist())
        if not actual_types.issubset(expected_types):
            raise ValueError("strict domain contains an unknown global site type")
        composition = tuple(
            int(torch.sum(site_types.detach().cpu() == site_type))
            for site_type in range(len(self.species_vocabulary))
        )
        if composition != self.reference_composition:
            raise ValueError(
                "strict domain reference site-type composition mismatch"
            )

    def fingerprint_values(self) -> tuple[Any, ...]:
        """Canonical primitives appended only for strict-template hashes."""

        return (
            self.convention_version,
            self.reference_site_count,
            self.supercell_shape,
            self.species_vocabulary,
            self.reference_composition,
            self.allowed_compositions,
            self.allowed_num_atoms,
            self.allowed_vacancy_masses,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_site_count": self.reference_site_count,
            "supercell_shape": list(self.supercell_shape),
            "species_vocabulary": list(self.species_vocabulary),
            "reference_composition": list(self.reference_composition),
            "allowed_compositions": [
                list(composition) for composition in self.allowed_compositions
            ],
            "allowed_num_atoms": list(self.allowed_num_atoms),
            "allowed_vacancy_masses": list(self.allowed_vacancy_masses),
            "convention_version": self.convention_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrictTemplateDomain":
        if not isinstance(payload, Mapping):
            raise TypeError("strict template domain payload must be a mapping")
        return cls(
            reference_site_count=payload["reference_site_count"],
            supercell_shape=tuple(payload["supercell_shape"]),
            species_vocabulary=tuple(payload["species_vocabulary"]),
            reference_composition=tuple(payload["reference_composition"]),
            allowed_compositions=tuple(
                tuple(composition) for composition in payload["allowed_compositions"]
            ),
            allowed_num_atoms=tuple(payload["allowed_num_atoms"]),
            allowed_vacancy_masses=tuple(payload["allowed_vacancy_masses"]),
            convention_version=payload.get(
                "convention_version", "strict_template_domain_v1"
            ),
        )

