"""Higher-body outputs and auditable metadata."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch

@dataclass(frozen=True)
class HigherBodyResult:
    h:torch.Tensor; c_raw:torch.Tensor; c_bar:torch.Tensor; edge_sh:torch.Tensor
    edge_weights:torch.Tensor; edge_messages:torch.Tensor|None; A:torch.Tensor
    C1:torch.Tensor; C2:torch.Tensor; C3:torch.Tensor
    Z1:torch.Tensor; Z2:torch.Tensor; Z3:torch.Tensor
    irreps_metadata:dict[str,str]; channel_metadata:dict[str,Any]
    graph_diagnostics:dict[str,Any]; parameter_diagnostics:dict[str,int]
