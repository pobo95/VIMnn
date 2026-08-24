"""Current-cell edge geometry for a fixed reference topology."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .topology import ReferenceGraphTopology

@dataclass(frozen=True)
class ReferenceEdgeGeometry:
    edge_vectors: torch.Tensor
    squared_lengths: torch.Tensor
    radial_coordinate: torch.Tensor
    cutoff_values: torch.Tensor
    active_mask: torch.Tensor
    deformation: torch.Tensor
    maximum_strain_seen: torch.Tensor


def c2_edge_cutoff(squared_lengths: torch.Tensor, cutoff: float)->torch.Tensor:
    u=squared_lengths/squared_lengths.new_tensor(cutoff*cutoff)
    polynomial=1-10*u.pow(3)+15*u.pow(4)-6*u.pow(5)
    return torch.where(u<1,polynomial,torch.zeros_like(u))


def update_reference_edge_geometry(topology: ReferenceGraphTopology,current_cell: torch.Tensor,*,edge_length_scale: float)->ReferenceEdgeGeometry:
    if current_cell.shape!=(3,3) or current_cell.dtype!=topology.reference_cell.dtype or current_cell.device!=topology.reference_cell.device:
        raise ValueError("current cell must match topology shape/dtype/device")
    if edge_length_scale<=0: raise ValueError("edge_length_scale must be positive")
    deformation=torch.linalg.solve(topology.reference_cell,current_cell)
    strain=torch.linalg.matrix_norm(deformation-torch.eye(3,dtype=current_cell.dtype,device=current_cell.device),ord=2)
    if bool(strain > topology.maximum_strain+32*torch.finfo(current_cell.dtype).eps):
        raise ValueError("current cell lies outside certified graph strain domain")
    source,target=topology.edge_index
    fractional=topology.reference_fractional[source]-topology.reference_fractional[target]+topology.shifts.to(current_cell.dtype)
    vectors=fractional@current_cell
    squared=torch.sum(vectors*vectors,dim=-1)
    if vectors.numel() and bool(torch.any(squared <= topology.minimum_edge_length**2)):
        raise ValueError("current reference graph contains a zero-length edge")
    cutoff_values=c2_edge_cutoff(squared,topology.cutoff)
    return ReferenceEdgeGeometry(vectors,squared,squared/(edge_length_scale**2),cutoff_values,squared<topology.cutoff**2,deformation,strain)
