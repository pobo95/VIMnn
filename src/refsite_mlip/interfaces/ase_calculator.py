"""ASE Calculator adapter backed exclusively by :mod:`refsite_mlip.inference`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorError, all_changes

from refsite_mlip.data import StructureSample
from refsite_mlip.inference import (
    PredictorConfig,
    PredictorError,
    ReferenceSitePredictor,
    load_reference_site_predictor,
)
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    CompactCandidateNeighborState,
)


@dataclass(frozen=True)
class ASECalculatorConfig:
    """Execution controls owned by the ASE adapter.

    Candidate-state reuse is enabled by default only when the loaded model
    uses the ``compact_c2``/``edge_list``/``blocked`` transport combination.
    Other models remain stateless without changing their prediction path.
    """

    solver_path: str = TRAIN_FIXED
    reuse_candidate_state: bool = True
    collect_diagnostics: bool = True
    sample_id: str = "ase:structure"
    origin_convention: str = "zero"

    def __post_init__(self) -> None:
        if self.solver_path not in (TRAIN_FIXED, EVAL_ADAPTIVE):
            raise ValueError("solver_path must be TRAIN_FIXED or EVAL_ADAPTIVE")
        for name in ("reuse_candidate_state", "collect_diagnostics"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a nonempty string")
        if self.origin_convention != "zero":
            raise ValueError("only origin_convention='zero' is supported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_path": self.solver_path,
            "reuse_candidate_state": self.reuse_candidate_state,
            "collect_diagnostics": self.collect_diagnostics,
            "sample_id": self.sample_id,
            "origin_convention": self.origin_convention,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ASECalculatorConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("ASECalculatorConfig payload must be a mapping")
        expected = {
            "solver_path",
            "reuse_candidate_state",
            "collect_diagnostics",
            "sample_id",
            "origin_convention",
        }
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"unknown ASECalculatorConfig keys: {sorted(unknown)}")
        return cls(**dict(payload))


class ReferenceSiteASECalculatorError(CalculatorError):
    """Actionable ASE failure retaining Predictor execution context."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        requested_properties: Sequence[str] = (),
        template_id: str | None,
        solver_path: str,
        atom_count: int | None = None,
        species: Sequence[int] = (),
        composition: Sequence[tuple[int, int]] = (),
        predictor_reason_code: str | None = None,
        predictor_stage: str | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.requested_properties = tuple(requested_properties)
        self.template_id = template_id
        self.solver_path = solver_path
        self.atom_count = atom_count
        self.species = tuple(int(value) for value in species)
        self.composition = tuple(
            (int(number), int(count)) for number, count in composition
        )
        self.predictor_reason_code = predictor_reason_code
        self.predictor_stage = predictor_stage
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        context = (
            f"requested_properties={self.requested_properties!r} "
            f"template_id={template_id!r} solver_path={solver_path!r} "
            f"atom_count={atom_count!r} species={self.species!r} "
            f"composition={self.composition!r} "
            f"predictor_reason_code={predictor_reason_code!r} "
            f"predictor_stage={predictor_stage!r}"
        )
        if original_error is not None:
            context += (
                f" original_exception={type(original_error).__name__}: "
                f"{original_error}"
            )
        super().__init__(f"[{reason_code}] {message}; {context}")


@dataclass(frozen=True)
class _CandidateGeometryIdentity:
    template_id: str
    atomic_numbers: tuple[int, ...]
    pbc: tuple[bool, bool, bool]
    cell_values: tuple[str, ...]


def _composition(numbers: np.ndarray) -> tuple[tuple[int, int], ...]:
    counts = Counter(int(value) for value in numbers.tolist())
    return tuple(sorted(counts.items()))


def _readonly_snapshot(value: Any) -> Any:
    """Return a recursively read-only CPU diagnostic snapshot."""

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy().copy()
        array.setflags(write=False)
        return array
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value).copy()
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _readonly_snapshot(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_readonly_snapshot(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return MappingProxyType(
            {
                field.name: _readonly_snapshot(getattr(value, field.name))
                for field in fields(value)
            }
        )
    if isinstance(value, np.generic):
        return value.item()
    return value


def _state_storage_bytes(state: CompactCandidateNeighborState | None) -> int:
    if state is None:
        return 0
    total = 0
    for field in fields(state):
        value = getattr(state, field.name)
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


class ReferenceSiteASECalculator(Calculator):
    """Portable-bundle ASE adapter with transactional functional state reuse."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        template_id: str | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        solver_path: str | None = None,
        config: ASECalculatorConfig | None = None,
        **calculator_kwargs: Any,
    ) -> None:
        self._candidate_neighbor_state: CompactCandidateNeighborState | None = None
        self._candidate_geometry_identity: _CandidateGeometryIdentity | None = None
        self._last_diagnostics: Mapping[str, Any] | None = None
        self._predictor: ReferenceSitePredictor | None = None
        self._template_id: str | None = template_id
        super().__init__(**calculator_kwargs)

        if config is not None and not isinstance(config, ASECalculatorConfig):
            raise TypeError("config must be ASECalculatorConfig or None")
        if solver_path is not None:
            if config is not None and solver_path != config.solver_path:
                raise ValueError("solver_path conflicts with ASECalculatorConfig")
            config = ASECalculatorConfig(
                solver_path=solver_path,
                reuse_candidate_state=(
                    True if config is None else config.reuse_candidate_state
                ),
                collect_diagnostics=(
                    True if config is None else config.collect_diagnostics
                ),
                sample_id=("ase:structure" if config is None else config.sample_id),
                origin_convention=(
                    "zero" if config is None else config.origin_convention
                ),
            )
        self._config = ASECalculatorConfig() if config is None else config

        try:
            predictor = load_reference_site_predictor(
                bundle_path,
                device=device,
                dtype=dtype,
                config=PredictorConfig(output_device="runtime"),
            )
        except Exception as error:
            raise ReferenceSiteASECalculatorError(
                getattr(error, "reason_code", "BUNDLE_RUNTIME_LOAD_FAILED"),
                "portable model bundle could not be loaded and instantiated",
                template_id=template_id,
                solver_path=self._config.solver_path,
                predictor_reason_code=getattr(error, "reason_code", None),
                predictor_stage=getattr(error, "validation_stage", "bundle_load"),
                original_error=error,
            ) from error
        self._predictor = predictor
        self._template_id = self._resolve_template_id(template_id)
        self._candidate_state_supported = self._detect_candidate_state_support()
        self._predictor.model.eval()

    @property
    def config(self) -> ASECalculatorConfig:
        return self._config

    @property
    def predictor(self) -> ReferenceSitePredictor:
        assert self._predictor is not None
        return self._predictor

    @property
    def template_id(self) -> str:
        assert self._template_id is not None
        return self._template_id

    @property
    def candidate_state_enabled(self) -> bool:
        return self._config.reuse_candidate_state and self._candidate_state_supported

    @property
    def candidate_neighbor_state(self) -> CompactCandidateNeighborState | None:
        """Caller-safe clone of the adapter-owned runtime state."""

        state = self._candidate_neighbor_state
        if state is None:
            return None
        return state.to(device=state.device, dtype=state.dtype)

    @property
    def last_diagnostics(self) -> Mapping[str, Any] | None:
        """Read-only diagnostics from the last successful calculation."""

        return self._last_diagnostics

    def _resolve_template_id(self, requested: str | None) -> str:
        contexts = self.predictor.runtime.template_contexts
        selected = requested
        if selected is None:
            selected = self.predictor.runtime.default_template_id
            if not isinstance(selected, str) or not selected:
                reason = "AMBIGUOUS_TEMPLATE" if len(contexts) > 1 else "MISSING_DEFAULT_TEMPLATE"
                raise ReferenceSiteASECalculatorError(
                    reason,
                    "template_id was omitted and the bundle has no unambiguous default",
                    template_id=None,
                    solver_path=self._config.solver_path,
                )
        if not isinstance(selected, str) or not selected:
            raise ReferenceSiteASECalculatorError(
                "UNKNOWN_TEMPLATE",
                "template_id must be a nonempty exact bundle template ID",
                template_id=selected,
                solver_path=self._config.solver_path,
            )
        if selected not in contexts:
            raise ReferenceSiteASECalculatorError(
                "UNKNOWN_TEMPLATE",
                "exact template_id is absent from the portable bundle",
                template_id=selected,
                solver_path=self._config.solver_path,
            )
        context = contexts[selected]
        expected = self.predictor.runtime.template_fingerprints[selected]
        try:
            context.validate_fingerprint()
        except Exception as error:
            raise ReferenceSiteASECalculatorError(
                "TEMPLATE_CONTEXT_FINGERPRINT_MISMATCH",
                "bundle template context failed fingerprint validation",
                template_id=selected,
                solver_path=self._config.solver_path,
                predictor_stage="template_selection",
                original_error=error,
            ) from error
        if context.template_id != selected or context.fingerprint != expected:
            raise ReferenceSiteASECalculatorError(
                "TEMPLATE_CONTEXT_FINGERPRINT_MISMATCH",
                "bundle template/context/full fingerprint binding differs",
                template_id=selected,
                solver_path=self._config.solver_path,
                predictor_stage="template_selection",
            )
        return selected

    def _detect_candidate_state_support(self) -> bool:
        support = self.predictor.model.config.transport_support
        return bool(
            support.kind == "compact_c2"
            and support.backend == "edge_list"
            and support.candidate_backend == "blocked"
            and support.candidate_skin > 0.0
        )

    def _error(
        self,
        reason_code: str,
        message: str,
        *,
        properties: Sequence[str],
        atoms: Atoms | None,
        predictor_reason_code: str | None = None,
        predictor_stage: str | None = None,
        original_error: BaseException | None = None,
    ) -> ReferenceSiteASECalculatorError:
        atom_count = None
        numbers = np.empty(0, dtype=np.int64)
        if atoms is not None:
            try:
                numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
                atom_count = len(atoms)
            except Exception:
                pass
        return ReferenceSiteASECalculatorError(
            reason_code,
            message,
            requested_properties=properties,
            template_id=self._template_id,
            solver_path=self._config.solver_path,
            atom_count=atom_count,
            species=tuple(sorted(set(int(value) for value in numbers.tolist()))),
            composition=_composition(numbers),
            predictor_reason_code=predictor_reason_code,
            predictor_stage=predictor_stage,
            original_error=original_error,
        )

    def _normalize_properties(self, properties: Sequence[str]) -> tuple[str, ...]:
        if isinstance(properties, (str, bytes)):
            properties = (str(properties),)
        result = tuple(dict.fromkeys(properties))
        if not result:
            raise self._error(
                "INVALID_PROPERTY_REQUEST",
                "at least one ASE property must be requested",
                properties=(),
                atoms=self.atoms,
            )
        unsupported = sorted(set(result) - set(self.implemented_properties))
        if unsupported:
            raise self._error(
                "UNSUPPORTED_PROPERTY",
                f"unsupported ASE properties: {unsupported}",
                properties=result,
                atoms=self.atoms,
            )
        return result

    def _structure_sample(
        self, atoms: Atoms, properties: Sequence[str]
    ) -> tuple[StructureSample, _CandidateGeometryIdentity]:
        if not isinstance(atoms, Atoms):
            raise self._error(
                "INVALID_ASE_ATOMS",
                "calculate requires an ase.Atoms instance",
                properties=properties,
                atoms=None,
            )
        try:
            positions = np.asarray(atoms.get_positions(wrap=False), dtype=np.float64)
            numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
            pbc = np.asarray(atoms.get_pbc(), dtype=np.bool_)
        except Exception as error:
            raise self._error(
                "ASE_STRUCTURE_EXTRACTION_FAILED",
                "ASE geometry extraction failed",
                properties=properties,
                atoms=atoms,
                predictor_stage="structure_conversion",
                original_error=error,
            ) from error
        if positions.shape != (len(atoms), 3) or numbers.shape != (len(atoms),):
            raise self._error(
                "INVALID_ASE_GEOMETRY",
                "ASE positions/numbers have inconsistent shape",
                properties=properties,
                atoms=atoms,
                predictor_stage="structure_conversion",
            )
        if cell.shape != (3, 3) or pbc.shape != (3,):
            raise self._error(
                "INVALID_ASE_GEOMETRY",
                "ASE cell/PBC have inconsistent shape",
                properties=properties,
                atoms=atoms,
                predictor_stage="structure_conversion",
            )
        if not bool(np.all(pbc)):
            raise self._error(
                "UNSUPPORTED_PBC",
                "ReferenceSiteASECalculator requires full three-dimensional PBC",
                properties=properties,
                atoms=atoms,
                predictor_stage="structure_conversion",
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cell)):
            raise self._error(
                "NONFINITE_GEOMETRY",
                "ASE positions/cell contain NaN or Inf",
                properties=properties,
                atoms=atoms,
                predictor_stage="structure_conversion",
            )
        singular_values = np.linalg.svd(cell, compute_uv=False)
        threshold = 8.0 * np.finfo(np.float64).eps * max(
            float(singular_values.max(initial=0.0)), 1.0
        )
        if singular_values.size != 3 or float(singular_values.min()) <= threshold:
            raise self._error(
                "SINGULAR_CELL",
                "ASE cell is singular or numerically degenerate",
                properties=properties,
                atoms=atoms,
                predictor_stage="structure_conversion",
            )
        if numbers.size and bool(np.any(numbers <= 0)):
            raise self._error(
                "UNSUPPORTED_SPECIES",
                "ASE atomic numbers must be positive",
                properties=properties,
                atoms=atoms,
                predictor_stage="structure_conversion",
            )

        sample = StructureSample(
            sample_id=self._config.sample_id,
            positions=torch.tensor(positions, dtype=torch.float64),
            atomic_numbers=torch.tensor(numbers, dtype=torch.long),
            cell=torch.tensor(cell, dtype=torch.float64),
            pbc=torch.tensor(pbc, dtype=torch.bool),
            origin=torch.zeros(3, dtype=torch.float64),
            template_id=self.template_id,
        )
        identity = _CandidateGeometryIdentity(
            template_id=self.template_id,
            atomic_numbers=tuple(int(value) for value in numbers.tolist()),
            pbc=tuple(bool(value) for value in pbc.tolist()),
            cell_values=tuple(float(value).hex() for value in cell.reshape(-1)),
        )
        return sample, identity

    def _candidate_input(
        self, identity: _CandidateGeometryIdentity
    ) -> tuple[CompactCandidateNeighborState | None, str]:
        if not self.candidate_state_enabled:
            return None, "REUSE_DISABLED_OR_UNSUPPORTED"
        state = self._candidate_neighbor_state
        previous = self._candidate_geometry_identity
        if state is None or previous is None:
            return None, "INITIAL_BUILD"
        if previous.template_id != identity.template_id:
            return None, "TEMPLATE_CHANGED"
        if previous.atomic_numbers != identity.atomic_numbers:
            return None, "ATOM_ORDER_CHANGED"
        if previous.pbc != identity.pbc:
            return None, "PBC_CHANGED"
        if previous.cell_values != identity.cell_values:
            return None, "CELL_CHANGED"
        return state, "COMPATIBLE"

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        requested = self._normalize_properties(properties)
        target_atoms = self.atoms if atoms is None else atoms
        if target_atoms is None:
            raise self._error(
                "MISSING_ASE_ATOMS",
                "no ase.Atoms object was supplied or cached",
                properties=requested,
                atoms=None,
            )
        changes = tuple(system_changes or ())
        old_results = dict(self.results)
        baseline_results = {} if changes else old_results

        try:
            sample, identity = self._structure_sample(target_atoms, requested)
            previous_state, adapter_state_reason = self._candidate_input(identity)
            super().calculate(atoms, list(requested), list(changes))

            compute_forces = "forces" in requested
            compute_stress = "stress" in requested
            state_mapping = (
                None
                if previous_state is None
                else {self._config.sample_id: previous_state}
            )
            prediction = self.predictor.predict_sample(
                sample,
                solver_path=self._config.solver_path,
                compute_forces=compute_forces,
                compute_stress=compute_stress,
                return_aux=self._config.collect_diagnostics,
                candidate_neighbor_states=state_mapping,
                return_candidate_neighbor_states=self.candidate_state_enabled,
            )

            energy = float(prediction.energy.detach().cpu().item())
            pending_results: dict[str, Any] = dict(baseline_results)
            pending_results["energy"] = energy
            pending_results["free_energy"] = energy
            if compute_forces:
                if prediction.forces is None:
                    raise ValueError("Predictor omitted requested forces")
                pending_results["forces"] = (
                    prediction.forces.detach().cpu().contiguous().numpy().copy()
                )
            if compute_stress:
                if prediction.stress_voigt is None or prediction.stress is None:
                    raise ValueError("Predictor omitted requested stress")
                pending_results["stress"] = (
                    prediction.stress_voigt.detach().cpu().contiguous().numpy().copy()
                )

            for key, value in pending_results.items():
                if isinstance(value, np.ndarray) and not bool(np.all(np.isfinite(value))):
                    raise FloatingPointError(f"ASE result {key!r} is nonfinite")
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise FloatingPointError(f"ASE result {key!r} is nonfinite")

            pending_state = self._candidate_neighbor_state
            pending_identity = self._candidate_geometry_identity
            decision = prediction.candidate_reuse_decision
            if self.candidate_state_enabled:
                pending_state = prediction.candidate_neighbor_state
                if pending_state is None:
                    raise ValueError("Predictor omitted requested candidate neighbor state")
                pending_state.validate_integrity()
                pending_identity = identity
            else:
                pending_state = None
                pending_identity = None

            pending_diagnostics = _readonly_snapshot(
                {
                    "sample_id": sample.sample_id,
                    "template_id": self.template_id,
                    "template_fingerprint": self.predictor.runtime.template_fingerprints[
                        self.template_id
                    ],
                    "solver_path": self._config.solver_path,
                    "requested_properties": requested,
                    "predictor": prediction.diagnostics,
                    "candidate_state": {
                        "configured": self._config.reuse_candidate_state,
                        "backend_supported": self._candidate_state_supported,
                        "enabled": self.candidate_state_enabled,
                        "adapter_input_reason": adapter_state_reason,
                        "decision": decision,
                        "integrity_fingerprint": (
                            None
                            if pending_state is None
                            else pending_state.integrity_fingerprint
                        ),
                        "retained_bytes": _state_storage_bytes(pending_state),
                    },
                }
            )
        except ReferenceSiteASECalculatorError:
            self.results = {} if changes else old_results
            raise
        except PredictorError as error:
            self.results = {} if changes else old_results
            raise self._error(
                error.reason_code,
                "ReferenceSitePredictor failed while evaluating ASE properties",
                properties=requested,
                atoms=target_atoms,
                predictor_reason_code=error.reason_code,
                predictor_stage=error.stage,
                original_error=error,
            ) from error
        except Exception as error:
            self.results = {} if changes else old_results
            reason = (
                "NONFINITE_OUTPUT"
                if isinstance(error, FloatingPointError)
                else getattr(error, "reason_code", "ASE_PREDICTION_FAILED")
            )
            raise self._error(
                reason,
                "ASE prediction adapter failed before transactional commit",
                properties=requested,
                atoms=target_atoms,
                predictor_reason_code=getattr(error, "reason_code", None),
                predictor_stage=getattr(error, "stage", "calculator_adapter"),
                original_error=error,
            ) from error

        # Commit only after every result/state/diagnostic validation succeeds.
        self.results = pending_results
        self._candidate_neighbor_state = pending_state
        self._candidate_geometry_identity = pending_identity
        self._last_diagnostics = pending_diagnostics

    def reset(self) -> None:
        """Clear ASE cache and all adapter-owned runtime state."""

        super().reset()
        self._candidate_neighbor_state = None
        self._candidate_geometry_identity = None
        self._last_diagnostics = None


__all__ = [
    "ASECalculatorConfig",
    "ReferenceSiteASECalculator",
    "ReferenceSiteASECalculatorError",
]
