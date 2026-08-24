from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True)
class PotentialOutput:
    def __getitem__(self,key): return getattr(self,key)
    energy:torch.Tensor; site_energy:torch.Tensor; baseline_energy:torch.Tensor; residual_energy:torch.Tensor; site_features:torch.Tensor; raw_c:torch.Tensor; forces:torch.Tensor|None=None; stress:torch.Tensor|None=None; stress_voigt:torch.Tensor|None=None; auxiliary:dict[str,Any]|None=None


@dataclass(frozen=True)
class BatchedPotentialOutput:
    """Ragged model outputs restored to the input structure/atom order."""

    energy: torch.Tensor
    baseline_energy: torch.Tensor
    residual_energy: torch.Tensor
    site_energy: torch.Tensor
    site_ptr: torch.Tensor
    site_batch: torch.Tensor
    forces: torch.Tensor | None
    stress: torch.Tensor | None
    stress_voigt: torch.Tensor | None
    sample_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    auxiliary: tuple[dict[str, Any] | None, ...] | None = None

    def __getitem__(self, key):
        return getattr(self, key)
