"""Template-bound policy for stabilizer-aware production evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from numbers import Real
from typing import Any

import torch

from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import EVAL_ADAPTIVE


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _positive(value: Real, name: str, *, greater_than_one: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real")
    result = float(value)
    lower = 1.0 if greater_than_one else 0.0
    if not math.isfinite(result) or result <= lower:
        qualifier = "greater than one" if greater_than_one else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _schedule(values, name: str) -> tuple[float, ...]:
    result = tuple(_positive(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


@dataclass(frozen=True)
class EvaluationPolicy:
    """Absolute, template-bound acceptance policy for evaluation phase search.

    Thresholds deliberately carry ``_absolute`` in their names: reciprocal
    field and objective scales depend on the template and are not assumed to
    transfer between templates of different sizes.
    """

    template_id: str
    template_fingerprint: str
    candidate_offsets: torch.Tensor
    phase_step_schedule: tuple[float, ...]
    phase_damping_schedule: tuple[float, ...]
    minimum_objective_gap_absolute: float
    minimum_cross_amplitude_absolute: float
    minimum_atomic_amplitude_absolute: float
    minimum_reference_amplitude_absolute: float
    minimum_curvature: float
    maximum_condition: float
    maximum_gradient_norm: float
    equivalence_tolerance: float
    transport_path: str = EVAL_ADAPTIVE
    convention_version: str = "evaluation_policy_v1"
    content_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be nonempty")
        if not isinstance(self.template_fingerprint, str) or not _SHA256.fullmatch(
            self.template_fingerprint
        ):
            raise ValueError("template_fingerprint must be a lowercase SHA-256 string")
        if self.transport_path != EVAL_ADAPTIVE:
            raise ValueError("EvaluationPolicy transport_path must be EVAL_ADAPTIVE")
        if self.convention_version != "evaluation_policy_v1":
            raise ValueError("unsupported evaluation policy convention version")

        offsets = self.candidate_offsets
        if not isinstance(offsets, torch.Tensor):
            raise EvaluationPhaseError(
                "INVALID_CANDIDATES",
                "candidate_offsets must be a tensor",
                template_id=self.template_id,
            )
        offsets = offsets.detach().cpu().contiguous().clone()
        if offsets.dtype not in (torch.float32, torch.float64):
            raise EvaluationPhaseError(
                "INVALID_CANDIDATES",
                "candidate_offsets must use float32 or float64",
                template_id=self.template_id,
            )
        if offsets.ndim != 2 or offsets.shape[1] != 3 or offsets.shape[0] < 2:
            raise EvaluationPhaseError(
                "INVALID_CANDIDATES",
                "candidate_offsets must have shape [J,3] with J>=2",
                template_id=self.template_id,
            )
        if not bool(torch.all(torch.isfinite(offsets))):
            raise EvaluationPhaseError(
                "INVALID_CANDIDATES",
                "candidate_offsets must be finite",
                template_id=self.template_id,
            )
        object.__setattr__(self, "candidate_offsets", offsets)

        steps = _schedule(self.phase_step_schedule, "phase_step_schedule")
        damping = _schedule(self.phase_damping_schedule, "phase_damping_schedule")
        if len(steps) != len(damping):
            raise ValueError("phase step and damping schedules must have equal length")
        object.__setattr__(self, "phase_step_schedule", steps)
        object.__setattr__(self, "phase_damping_schedule", damping)
        for name in (
            "minimum_objective_gap_absolute",
            "minimum_cross_amplitude_absolute",
            "minimum_atomic_amplitude_absolute",
            "minimum_reference_amplitude_absolute",
            "minimum_curvature",
            "maximum_gradient_norm",
            "equivalence_tolerance",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "maximum_condition",
            _positive(self.maximum_condition, "maximum_condition", greater_than_one=True),
        )

        difference = offsets[:, None, :] - offsets[None, :, :]
        difference = difference - torch.round(difference)
        distance = torch.linalg.vector_norm(difference, dim=-1)
        duplicate = torch.triu(
            distance <= self.equivalence_tolerance, diagonal=1
        )
        if bool(torch.any(duplicate)):
            raise EvaluationPhaseError(
                "INVALID_CANDIDATES",
                "candidate_offsets contain duplicate torus representatives",
                template_id=self.template_id,
            )

        actual = self._compute_content_fingerprint()
        if self.content_fingerprint is None:
            object.__setattr__(self, "content_fingerprint", actual)
        elif self.content_fingerprint != actual:
            raise ValueError("evaluation policy content fingerprint mismatch")

    def _canonical_scalars(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "template_fingerprint": self.template_fingerprint,
            "phase_step_schedule": list(self.phase_step_schedule),
            "phase_damping_schedule": list(self.phase_damping_schedule),
            "minimum_objective_gap_absolute": self.minimum_objective_gap_absolute,
            "minimum_cross_amplitude_absolute": self.minimum_cross_amplitude_absolute,
            "minimum_atomic_amplitude_absolute": self.minimum_atomic_amplitude_absolute,
            "minimum_reference_amplitude_absolute": self.minimum_reference_amplitude_absolute,
            "minimum_curvature": self.minimum_curvature,
            "maximum_condition": self.maximum_condition,
            "maximum_gradient_norm": self.maximum_gradient_norm,
            "equivalence_tolerance": self.equivalence_tolerance,
            "transport_path": self.transport_path,
            "convention_version": self.convention_version,
        }

    def _compute_content_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self._canonical_scalars(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(str(self.candidate_offsets.dtype).encode("ascii"))
        digest.update(str(tuple(self.candidate_offsets.shape)).encode("ascii"))
        digest.update(self.candidate_offsets.numpy().tobytes())
        return digest.hexdigest()

    def validate_fingerprint(self) -> None:
        if self.content_fingerprint != self._compute_content_fingerprint():
            raise ValueError("evaluation policy content fingerprint mismatch")

    def materialize_candidate_offsets(
        self, *, device: torch.device | str, dtype: torch.dtype
    ) -> torch.Tensor:
        self.validate_fingerprint()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("evaluation candidate dtype must be float32 or float64")
        return self.candidate_offsets.to(device=device, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        self.validate_fingerprint()
        result = self._canonical_scalars()
        result.update(
            {
                "candidate_offsets": self.candidate_offsets.tolist(),
                "candidate_dtype": str(self.candidate_offsets.dtype).removeprefix(
                    "torch."
                ),
                "content_fingerprint": self.content_fingerprint,
            }
        )
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationPolicy":
        if not isinstance(value, dict):
            raise TypeError("evaluation policy payload must be a dictionary")
        dtype_name = value.get("candidate_dtype")
        dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype_name)
        if dtype is None:
            raise ValueError("unsupported evaluation candidate dtype")
        arguments = dict(value)
        arguments.pop("candidate_dtype")
        arguments["candidate_offsets"] = torch.tensor(
            arguments["candidate_offsets"], dtype=dtype
        )
        arguments["phase_step_schedule"] = tuple(arguments["phase_step_schedule"])
        arguments["phase_damping_schedule"] = tuple(
            arguments["phase_damping_schedule"]
        )
        return cls(**arguments)
