"""Bundle-backed production inference without hidden mutable state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from refsite_mlip.data import (
    StructureBatch,
    StructureSample,
    collate_structure_samples,
)
from refsite_mlip.models import (
    LoadedReferenceSiteModel,
    evaluate_structure_batch,
    instantiate_reference_site_model_bundle,
    load_reference_site_model_bundle,
)
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import (
    EVAL_ADAPTIVE,
    TRAIN_FIXED,
    CandidateReuseDecision,
    CompactCandidateNeighborState,
)

from .outputs import BatchPrediction, StructurePrediction


class PredictorError(EvaluationPhaseError):
    """Prediction failure with stable structure and execution context."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        sample_id: str | None,
        template_id: str | None,
        solver_path: str,
        stage: str,
        original_error: Exception | None = None,
    ) -> None:
        self.sample_id = sample_id
        self.solver_path = solver_path
        self.stage = stage
        self.original_exception_type = (
            None if original_error is None else type(original_error).__name__
        )
        self.original_exception_message = (
            None if original_error is None else str(original_error)
        )
        context = (
            f"sample_id={sample_id!r} template_id={template_id!r} "
            f"solver_path={solver_path!r} stage={stage!r}"
        )
        if original_error is not None:
            context += (
                f" original_exception={type(original_error).__name__}: "
                f"{original_error}"
            )
        super().__init__(
            reason_code,
            f"{message}; {context}",
            template_id=template_id,
        )


@dataclass(frozen=True)
class PredictorConfig:
    """Default prediction controls.

    ``output_device='runtime'`` keeps detached results beside the model.
    ``output_device='cpu'`` explicitly transfers all public tensors and
    candidate-state snapshots to CPU after detaching.
    """

    solver_path: str = TRAIN_FIXED
    compute_forces: bool = False
    compute_stress: bool = False
    return_aux: bool = False
    return_candidate_neighbor_states: bool = False
    output_device: str = "runtime"

    def __post_init__(self) -> None:
        if self.solver_path not in (TRAIN_FIXED, EVAL_ADAPTIVE):
            raise ValueError("solver_path must be TRAIN_FIXED or EVAL_ADAPTIVE")
        for name in (
            "compute_forces",
            "compute_stress",
            "return_aux",
            "return_candidate_neighbor_states",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.output_device not in ("runtime", "cpu"):
            raise ValueError("output_device must be 'runtime' or 'cpu'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_path": self.solver_path,
            "compute_forces": self.compute_forces,
            "compute_stress": self.compute_stress,
            "return_aux": self.return_aux,
            "return_candidate_neighbor_states": (
                self.return_candidate_neighbor_states
            ),
            "output_device": self.output_device,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictorConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("PredictorConfig payload must be a mapping")
        expected = {
            "solver_path",
            "compute_forces",
            "compute_stress",
            "return_aux",
            "return_candidate_neighbor_states",
            "output_device",
        }
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"unknown PredictorConfig keys: {sorted(unknown)}")
        return cls(**dict(payload))


def _snapshot(value: Any, *, device: torch.device) -> Any:
    """Recursively detach auxiliary arithmetic while preserving record types."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).contiguous().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item, device=device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item, device=device) for item in value)
    if isinstance(value, list):
        return [_snapshot(item, device=device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _snapshot(getattr(value, field.name), device=device)
            for field in fields(value)
        }
        return replace(value, **updates)
    return value


def _copy_tensor(
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target = value.detach().to(device=device, dtype=dtype)
    return target.contiguous().clone()


def _option(value: bool | None, default: bool, name: str) -> bool:
    if value is None:
        return default
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool or None")
    return value


class ReferenceSitePredictor:
    """Reusable bundle runtime for explicit, stateless-by-default prediction."""

    def __init__(
        self,
        runtime: LoadedReferenceSiteModel,
        *,
        config: PredictorConfig | None = None,
    ) -> None:
        if not isinstance(runtime, LoadedReferenceSiteModel):
            raise TypeError("runtime must be a LoadedReferenceSiteModel")
        if config is not None and not isinstance(config, PredictorConfig):
            raise TypeError("config must be PredictorConfig or None")
        self._runtime = runtime
        self._config = PredictorConfig() if config is None else config
        floating = [
            value
            for value in runtime.model.state_dict().values()
            if value.is_floating_point()
        ]
        if not floating:
            raise ValueError("predictor model has no floating runtime state")
        self._device = floating[0].device
        self._dtype = floating[0].dtype
        if self._dtype not in (torch.float32, torch.float64) or any(
            value.device != self._device or value.dtype != self._dtype
            for value in floating
        ):
            raise ValueError("predictor model state must share float32/float64 dtype/device")
        runtime.model.eval()

    @property
    def config(self) -> PredictorConfig:
        return self._config

    @property
    def runtime(self) -> LoadedReferenceSiteModel:
        return self._runtime

    @property
    def model(self):
        return self._runtime.model

    @property
    def registry(self):
        return self._runtime.registry

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def bundle_fingerprint(self) -> str:
        return self._runtime.bundle_fingerprint

    def _raise(
        self,
        reason_code: str,
        message: str,
        *,
        sample_id: str | None,
        template_id: str | None,
        solver_path: str,
        stage: str,
        original_error: Exception | None = None,
    ) -> None:
        raise PredictorError(
            reason_code,
            message,
            sample_id=sample_id,
            template_id=template_id,
            solver_path=solver_path,
            stage=stage,
            original_error=original_error,
        ) from original_error

    def _input_records(
        self,
        values: StructureSample | Sequence[StructureSample] | StructureBatch,
        *,
        solver_path: str,
    ) -> tuple[tuple[StructureSample, ...], tuple[str, ...] | None]:
        if isinstance(values, StructureBatch):
            try:
                values.validate()
            except Exception as error:
                self._raise(
                    "INVALID_STRUCTURE_BATCH",
                    "StructureBatch validation failed",
                    sample_id=(values.sample_ids[0] if values.sample_ids else None),
                    template_id=(
                        values.template_ids[0] if values.template_ids else None
                    ),
                    solver_path=solver_path,
                    stage="input_validation",
                    original_error=error,
                )
            samples = []
            for index in range(values.num_structures):
                start = int(values.atom_ptr[index].detach().cpu())
                stop = int(values.atom_ptr[index + 1].detach().cpu())
                samples.append(
                    StructureSample(
                        sample_id=values.sample_ids[index],
                        positions=values.positions[start:stop].detach(),
                        atomic_numbers=values.atomic_numbers[start:stop].detach(),
                        cell=values.cells[index].detach(),
                        pbc=values.pbc[index].detach(),
                        origin=values.origins[index].detach(),
                        template_id=values.template_ids[index],
                    )
                )
            return tuple(samples), values.template_fingerprints
        if isinstance(values, StructureSample):
            return (values,), None
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(
                "prediction input must be StructureSample, deterministic Sequence[StructureSample], or StructureBatch"
            )
        samples = tuple(values)
        if not samples:
            raise ValueError("prediction input sequence must not be empty")
        if any(not isinstance(sample, StructureSample) for sample in samples):
            raise TypeError("every prediction input must be a StructureSample")
        return samples, None

    def _preflight_and_copy(
        self,
        samples: tuple[StructureSample, ...],
        batch_fingerprints: tuple[str, ...] | None,
        *,
        solver_path: str,
    ) -> tuple[StructureSample, ...]:
        sample_ids = tuple(sample.sample_id for sample in samples)
        if len(set(sample_ids)) != len(sample_ids):
            first = sample_ids[0] if sample_ids else None
            self._raise(
                "DUPLICATE_SAMPLE_ID",
                "prediction sample IDs must be unique",
                sample_id=first,
                template_id=samples[0].template_id if samples else None,
                solver_path=solver_path,
                stage="input_preflight",
            )

        copied = []
        for index, sample in enumerate(samples):
            sample_id = sample.sample_id
            template_id = sample.template_id
            if template_id not in self._runtime.template_contexts:
                self._raise(
                    "UNKNOWN_TEMPLATE",
                    "exact template_id is absent from the bundle runtime",
                    sample_id=sample_id,
                    template_id=template_id,
                    solver_path=solver_path,
                    stage="template_lookup",
                )
            context = self._runtime.template_contexts[template_id]
            expected = self._runtime.template_fingerprints[template_id]
            try:
                context.validate_fingerprint()
            except Exception as error:
                self._raise(
                    "TEMPLATE_CONTEXT_FINGERPRINT_MISMATCH",
                    "template context content fingerprint validation failed",
                    sample_id=sample_id,
                    template_id=template_id,
                    solver_path=solver_path,
                    stage="template_preflight",
                    original_error=error,
                )
            if context.template_id != template_id or context.fingerprint != expected:
                self._raise(
                    "TEMPLATE_CONTEXT_FINGERPRINT_MISMATCH",
                    "template/context/full fingerprint binding differs",
                    sample_id=sample_id,
                    template_id=template_id,
                    solver_path=solver_path,
                    stage="template_preflight",
                )
            if batch_fingerprints is not None and batch_fingerprints[index] != expected:
                self._raise(
                    "TEMPLATE_FINGERPRINT_MISMATCH",
                    "StructureBatch fingerprint differs from bundle binding",
                    sample_id=sample_id,
                    template_id=template_id,
                    solver_path=solver_path,
                    stage="batch_fingerprint_preflight",
                )
            try:
                sample.validate()
                template = self._runtime.registry.resolve(template_id)
                if template.fingerprint != expected:
                    raise ValueError("registry template fingerprint differs from bundle")
                template.validate_structure(
                    sample.atomic_numbers,
                    cell=sample.cell if template.strict_domain is not None else None,
                    pbc=sample.pbc if template.strict_domain is not None else None,
                    sample_id=sample_id,
                )
            except Exception as error:
                message = str(error).lower()
                reason = (
                    "INVALID_N_GT_M"
                    if "n > m" in message or "exceeds reference-site" in message
                    else "UNSUPPORTED_SPECIES"
                    if "species" in message and ("unknown" in message or "unsupported" in message)
                    else "TEMPLATE_DOMAIN_MISMATCH"
                )
                self._raise(
                    reason,
                    "structure is incompatible with its explicit template",
                    sample_id=sample_id,
                    template_id=template_id,
                    solver_path=solver_path,
                    stage="structure_domain_preflight",
                    original_error=error,
                )
            copied.append(
                StructureSample(
                    sample_id=sample_id,
                    positions=_copy_tensor(
                        sample.positions, device=self._device, dtype=self._dtype
                    ),
                    atomic_numbers=_copy_tensor(
                        sample.atomic_numbers, device=self._device
                    ),
                    cell=_copy_tensor(
                        sample.cell, device=self._device, dtype=self._dtype
                    ),
                    pbc=_copy_tensor(sample.pbc, device=self._device),
                    origin=_copy_tensor(
                        sample.origin, device=self._device, dtype=self._dtype
                    ),
                    template_id=template_id,
                )
            )
        return tuple(copied)

    def _used_bindings(
        self,
        samples: tuple[StructureSample, ...],
        *,
        solver_path: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        used = sorted({sample.template_id for sample in samples})
        contexts = {
            template_id: self._runtime.template_contexts[template_id]
            for template_id in used
        }
        if solver_path == TRAIN_FIXED:
            return contexts, None
        policies = {}
        for template_id in used:
            if template_id not in self._runtime.evaluation_policies:
                sample = next(
                    value for value in samples if value.template_id == template_id
                )
                self._raise(
                    "POLICY_CONTEXT_MISMATCH",
                    "EVAL_ADAPTIVE requires an EvaluationPolicy for every used exact template_id",
                    sample_id=sample.sample_id,
                    template_id=template_id,
                    solver_path=solver_path,
                    stage="policy_lookup",
                )
            policy = self._runtime.evaluation_policies[template_id]
            try:
                policy.validate_fingerprint()
            except Exception as error:
                sample = next(
                    value for value in samples if value.template_id == template_id
                )
                self._raise(
                    "POLICY_CONTEXT_MISMATCH",
                    "EvaluationPolicy content fingerprint validation failed",
                    sample_id=sample.sample_id,
                    template_id=template_id,
                    solver_path=solver_path,
                    stage="policy_preflight",
                    original_error=error,
                )
            if (
                policy.template_id != template_id
                or policy.template_fingerprint
                != self._runtime.template_fingerprints[template_id]
            ):
                sample = next(
                    value for value in samples if value.template_id == template_id
                )
                self._raise(
                    "POLICY_CONTEXT_MISMATCH",
                    "EvaluationPolicy and full template fingerprint binding differ",
                    sample_id=sample.sample_id,
                    template_id=template_id,
                    solver_path=solver_path,
                    stage="policy_preflight",
                )
            policies[template_id] = policy
        return contexts, policies

    def _error_location(
        self,
        error: Exception,
        samples: tuple[StructureSample, ...],
    ) -> tuple[str | None, str | None]:
        sample_id = getattr(error, "sample_id", None)
        template_id = getattr(error, "template_id", None)
        if sample_id is not None:
            matching = next(
                (sample for sample in samples if sample.sample_id == sample_id), None
            )
            if matching is not None:
                return sample_id, matching.template_id
        if template_id is not None:
            matching = next(
                (sample for sample in samples if sample.template_id == template_id),
                None,
            )
            if matching is not None:
                return matching.sample_id, template_id
        return samples[0].sample_id, samples[0].template_id

    def _validate_finite(
        self,
        result,
        samples: tuple[StructureSample, ...],
        *,
        solver_path: str,
    ) -> None:
        for index, sample in enumerate(samples):
            site_start = int(result.site_ptr[index].detach().cpu())
            site_stop = int(result.site_ptr[index + 1].detach().cpu())
            atom_start = sum(value.num_atoms for value in samples[:index])
            atom_stop = atom_start + sample.num_atoms
            values = [
                result.energy[index],
                result.baseline_energy[index],
                result.residual_energy[index],
                result.site_energy[site_start:site_stop],
            ]
            if result.forces is not None:
                values.append(result.forces[atom_start:atom_stop])
            if result.stress is not None:
                values.extend((result.stress[index], result.stress_voigt[index]))
            if any(not bool(torch.all(torch.isfinite(value)).detach()) for value in values):
                self._raise(
                    "NONFINITE_PREDICTION",
                    "model returned a nonfinite prediction tensor",
                    sample_id=sample.sample_id,
                    template_id=sample.template_id,
                    solver_path=solver_path,
                    stage="output_validation",
                )

    def _snapshot_result(
        self,
        result,
        batch: StructureBatch,
        *,
        output_device: torch.device,
    ) -> BatchPrediction:
        diagnostics = (
            tuple(None for _ in batch.sample_ids)
            if result.auxiliary is None
            else tuple(
                _snapshot(value, device=output_device)
                for value in result.auxiliary
            )
        )
        states = None
        if result.candidate_neighbor_states is not None:
            states = MappingProxyType(
                {
                    sample_id: result.candidate_neighbor_states[sample_id].to(
                        device=output_device, dtype=self._dtype
                    )
                    for sample_id in batch.sample_ids
                }
            )
        decisions = None
        if result.candidate_reuse_decisions is not None:
            decisions = MappingProxyType(
                {
                    sample_id: result.candidate_reuse_decisions[sample_id]
                    for sample_id in batch.sample_ids
                }
            )

        def tensor(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.detach().to(device=output_device).contiguous().clone()

        return BatchPrediction(
            energy=tensor(result.energy),
            baseline_energy=tensor(result.baseline_energy),
            residual_energy=tensor(result.residual_energy),
            forces=tensor(result.forces),
            stress=tensor(result.stress),
            stress_voigt=tensor(result.stress_voigt),
            site_energy=tensor(result.site_energy),
            atom_ptr=tensor(batch.atom_ptr),
            site_ptr=tensor(result.site_ptr),
            sample_ids=result.sample_ids,
            template_ids=result.template_ids,
            diagnostics=diagnostics,
            candidate_neighbor_states=states,
            candidate_reuse_decisions=decisions,
        )

    def predict_samples(
        self,
        samples: StructureSample | Sequence[StructureSample] | StructureBatch,
        *,
        solver_path: str | None = None,
        compute_forces: bool | None = None,
        compute_stress: bool | None = None,
        return_aux: bool | None = None,
        candidate_neighbor_states: Mapping[
            str, CompactCandidateNeighborState
        ] | None = None,
        return_candidate_neighbor_states: bool | None = None,
    ) -> BatchPrediction:
        """Predict a deterministic ragged batch without consulting labels."""

        path = self._config.solver_path if solver_path is None else solver_path
        if path not in (TRAIN_FIXED, EVAL_ADAPTIVE):
            raise ValueError("solver_path must be TRAIN_FIXED or EVAL_ADAPTIVE")
        forces = _option(
            compute_forces, self._config.compute_forces, "compute_forces"
        )
        stress = _option(
            compute_stress, self._config.compute_stress, "compute_stress"
        )
        auxiliary = _option(return_aux, self._config.return_aux, "return_aux")
        return_states = _option(
            return_candidate_neighbor_states,
            self._config.return_candidate_neighbor_states,
            "return_candidate_neighbor_states",
        )
        if candidate_neighbor_states is not None and not isinstance(
            candidate_neighbor_states, Mapping
        ):
            raise TypeError("candidate_neighbor_states must be a mapping or None")

        records, fingerprints = self._input_records(samples, solver_path=path)
        if (forces or stress) and torch.is_inference_mode_enabled():
            first = records[0]
            self._raise(
                "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED",
                "force/stress prediction cannot run inside torch.inference_mode()",
                sample_id=first.sample_id,
                template_id=first.template_id,
                solver_path=path,
                stage="derivative_preflight",
            )
        prepared = self._preflight_and_copy(
            records, fingerprints, solver_path=path
        )
        contexts, policies = self._used_bindings(prepared, solver_path=path)
        try:
            batch = collate_structure_samples(prepared, self._runtime.registry)
            if forces:
                batch = replace(
                    batch,
                    positions=batch.positions.detach().requires_grad_(True),
                )
        except PredictorError:
            raise
        except Exception as error:
            first = prepared[0]
            self._raise(
                "BATCH_COLLATION_FAILED",
                "inference-only batch collation failed",
                sample_id=first.sample_id,
                template_id=first.template_id,
                solver_path=path,
                stage="collation",
                original_error=error,
            )

        self._runtime.model.eval()
        gradient_context = torch.enable_grad() if (forces or stress) else torch.no_grad()
        try:
            with gradient_context:
                result = evaluate_structure_batch(
                    self._runtime.model,
                    batch,
                    contexts,
                    solver_path=path,
                    evaluation_policies=policies,
                    compute_forces=forces,
                    compute_stress=stress,
                    create_graph=False,
                    return_aux=auxiliary,
                    candidate_neighbor_states=candidate_neighbor_states,
                    return_candidate_neighbor_states=return_states,
                )
        except PredictorError:
            raise
        except Exception as error:
            sample_id, template_id = self._error_location(error, prepared)
            self._raise(
                getattr(error, "reason_code", "PREDICTION_EXECUTION_FAILED"),
                "grouped model prediction failed",
                sample_id=sample_id,
                template_id=template_id,
                solver_path=path,
                stage=getattr(error, "stage", "model_evaluation"),
                original_error=error,
            )

        self._validate_finite(result, prepared, solver_path=path)
        output_device = (
            self._device
            if self._config.output_device == "runtime"
            else torch.device("cpu")
        )
        try:
            return self._snapshot_result(
                result, batch, output_device=output_device
            )
        except PredictorError:
            raise
        except Exception as error:
            first = prepared[0]
            self._raise(
                "OUTPUT_SNAPSHOT_FAILED",
                "detached prediction snapshot construction failed",
                sample_id=first.sample_id,
                template_id=first.template_id,
                solver_path=path,
                stage="output_snapshot",
                original_error=error,
            )

    def predict_batch(self, batch: StructureBatch, **kwargs: Any) -> BatchPrediction:
        if not isinstance(batch, StructureBatch):
            raise TypeError("predict_batch requires a StructureBatch")
        return self.predict_samples(batch, **kwargs)

    def predict_sample(
        self, sample: StructureSample, **kwargs: Any
    ) -> StructurePrediction:
        if not isinstance(sample, StructureSample):
            raise TypeError("predict_sample requires a StructureSample")
        return self.predict_samples((sample,), **kwargs).structure(0)

    def predict(
        self,
        values: StructureSample | Sequence[StructureSample] | StructureBatch,
        **kwargs: Any,
    ) -> StructurePrediction | BatchPrediction:
        if isinstance(values, StructureSample):
            return self.predict_sample(values, **kwargs)
        return self.predict_samples(values, **kwargs)


def load_reference_site_predictor(
    bundle_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    config: PredictorConfig | None = None,
) -> ReferenceSitePredictor:
    """Weights-only load exactly once, then reconstruct one reusable runtime."""

    bundle = load_reference_site_model_bundle(bundle_path, map_location="cpu")
    runtime = instantiate_reference_site_model_bundle(
        bundle, device=device, dtype=dtype
    )
    return ReferenceSitePredictor(runtime, config=config)


__all__ = [
    "PredictorConfig",
    "PredictorError",
    "ReferenceSitePredictor",
    "load_reference_site_predictor",
]
