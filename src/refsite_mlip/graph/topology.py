"""Fixed periodic topology for canonical reference-site templates."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import torch


def _pbc(pbc: Sequence[bool]) -> tuple[bool, bool, bool]:
    values=tuple(pbc)
    if len(values)!=3 or any(not isinstance(v,bool) for v in values):
        raise ValueError("pbc must contain three booleans")
    return values


@dataclass(frozen=True)
class ReferenceGraphTopology:
    reference_fractional: torch.Tensor
    site_types: torch.Tensor
    edge_index: torch.Tensor
    shifts: torch.Tensor
    reference_cell: torch.Tensor
    cutoff: float
    skin: float
    maximum_strain: float
    minimum_edge_length: float
    pbc: tuple[bool,bool,bool]

    @property
    def num_sites(self)->int: return int(self.reference_fractional.shape[0])
    @property
    def num_edges(self)->int: return int(self.edge_index.shape[1])
    @property
    def candidate_cutoff(self)->float: return self.cutoff+self.skin

    def to(self, *, device=None, dtype=None) -> "ReferenceGraphTopology":
        floating_dtype=self.reference_fractional.dtype if dtype is None else dtype
        return ReferenceGraphTopology(
            self.reference_fractional.to(device=device,dtype=floating_dtype),
            self.site_types.to(device=device), self.edge_index.to(device=device),
            self.shifts.to(device=device), self.reference_cell.to(device=device,dtype=floating_dtype),
            self.cutoff,self.skin,self.maximum_strain,self.minimum_edge_length,self.pbc,
        )


def build_reference_graph_topology(
    reference_fractional: torch.Tensor,
    site_types: torch.Tensor,
    reference_cell: torch.Tensor,
    *, cutoff: Real, skin: Real, maximum_strain: Real=0.1,
    minimum_edge_length: Real=1.0e-8,
    pbc: Sequence[bool]=(True,True,True),
)->ReferenceGraphTopology:
    if reference_fractional.ndim!=2 or reference_fractional.shape[1]!=3:
        raise ValueError("reference_fractional must have shape [M,3]")
    if site_types.shape!=(reference_fractional.shape[0],) or site_types.dtype!=torch.long:
        raise ValueError("site_types must be long with shape [M]")
    if reference_cell.shape!=(3,3): raise ValueError("reference_cell must have shape [3,3]")
    if reference_fractional.dtype not in (torch.float32,torch.float64): raise ValueError("graph floating tensors must use float32 or float64")
    if reference_cell.dtype!=reference_fractional.dtype or reference_cell.device!=reference_fractional.device or site_types.device!=reference_fractional.device:
        raise ValueError("topology inputs must share dtype/device")
    values={"cutoff":cutoff,"skin":skin,"maximum_strain":maximum_strain,"minimum_edge_length":minimum_edge_length}
    for name,value in values.items():
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(float(value)) or float(value)<0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if float(cutoff)<=0 or float(minimum_edge_length)<=0 or not 0<=float(maximum_strain)<1:
        raise ValueError("cutoff/minimum_edge_length must be positive and maximum_strain in [0,1)")
    if (1.0-float(maximum_strain))*(float(cutoff)+float(skin)) < float(cutoff):
        raise ValueError("cutoff+skin cannot certify the requested strain domain")
    if not bool(torch.all(torch.isfinite(reference_fractional))) or not bool(torch.all(torch.isfinite(reference_cell))):
        raise ValueError("topology contains NaN or Inf")
    singular=torch.linalg.svdvals(reference_cell)
    if bool(singular.min()<=torch.finfo(reference_cell.dtype).eps): raise ValueError("reference cell is singular")
    periodic=_pbc(pbc); candidate=float(cutoff)+float(skin)
    max_difference=0.0
    if reference_fractional.numel():
        differences=reference_fractional[:,None,:]-reference_fractional[None,:,:]
        max_difference=float(torch.linalg.vector_norm(differences,dim=-1).max().detach().cpu())
    radius=math.ceil(candidate/float(singular.min().detach().cpu())+max_difference)+1
    axes=[range(-radius,radius+1) if enabled else (0,) for enabled in periodic]
    shifts_all=list(itertools.product(*axes))
    sources=[]; targets=[]; shifts=[]
    for target in range(reference_fractional.shape[0]):
        for source in range(reference_fractional.shape[0]):
            difference=reference_fractional[source]-reference_fractional[target]
            for shift_tuple in shifts_all:
                if source==target and shift_tuple==(0,0,0): continue
                shift=torch.tensor(shift_tuple,dtype=reference_fractional.dtype,device=reference_fractional.device)
                vector=(difference+shift)@reference_cell
                distance2=torch.sum(vector*vector)
                if bool(distance2 <= (candidate+1e-12)**2):
                    if bool(distance2 <= float(minimum_edge_length)**2):
                        raise ValueError("reference graph contains a zero-length nontrivial edge")
                    sources.append(source); targets.append(target); shifts.append(shift_tuple)
    edge_index=torch.tensor([sources,targets],dtype=torch.long,device=reference_fractional.device) if sources else torch.empty((2,0),dtype=torch.long,device=reference_fractional.device)
    shift_tensor=torch.tensor(shifts,dtype=torch.long,device=reference_fractional.device) if shifts else torch.empty((0,3),dtype=torch.long,device=reference_fractional.device)
    return ReferenceGraphTopology(reference_fractional,site_types,edge_index,shift_tensor,reference_cell,float(cutoff),float(skin),float(maximum_strain),float(minimum_edge_length),periodic)
