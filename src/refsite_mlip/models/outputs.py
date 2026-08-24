from dataclasses import dataclass
from typing import Any
import torch
@dataclass(frozen=True)
class PotentialOutput:
    def __getitem__(self,key): return getattr(self,key)
    energy:torch.Tensor; site_energy:torch.Tensor; baseline_energy:torch.Tensor; residual_energy:torch.Tensor; site_features:torch.Tensor; raw_c:torch.Tensor; forces:torch.Tensor|None=None; stress:torch.Tensor|None=None; stress_voigt:torch.Tensor|None=None; auxiliary:dict[str,Any]|None=None
