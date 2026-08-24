"""Directed equivariant edge messages and target neighbor-density aggregation."""
from __future__ import annotations
import math
import torch
from torch import nn
from refsite_mlip.compatibility import import_e3nn_0_4_4

class EdgeRadialHead(nn.Module):
    def __init__(self,input_dim:int,hidden_dims:tuple[int,...],output_dim:int):
        super().__init__()
        if input_dim<=0 or output_dim<=0 or any(v<=0 for v in hidden_dims): raise ValueError("radial dimensions must be positive")
        dims=(input_dim,)+tuple(hidden_dims)+(output_dim,); layers=[]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if i<len(dims)-2: layers.append(nn.SiLU())
        self.network=nn.Sequential(*layers)
    def forward(self,radial,cutoff): return self.network(radial)*cutoff.unsqueeze(-1)


def squared_edge_radial_basis(radial_coordinate:torch.Tensor,dimension:int)->torch.Tensor:
    if radial_coordinate.ndim!=1 or dimension<=0: raise ValueError("radial coordinate must be [E] and dimension positive")
    values=[torch.ones_like(radial_coordinate)]
    for _ in range(1,dimension): values.append(values[-1]*radial_coordinate)
    return torch.stack(values,dim=-1)

class EdgeNeighborDensity(nn.Module):
    def __init__(self,irreps_source,irreps_A,lmax:int,radial_feature_dim:int,radial_hidden_dims:tuple[int,...],avg_num_neighbors:float):
        super().__init__(); _,o3=import_e3nn_0_4_4()
        if avg_num_neighbors<=0: raise ValueError("avg_num_neighbors must be fixed and positive")
        self.irreps_source=o3.Irreps(irreps_source); self.irreps_sh=o3.Irreps.spherical_harmonics(lmax); self.irreps_A=o3.Irreps(irreps_A)
        self.edge_tp=o3.FullyConnectedTensorProduct(self.irreps_source,self.irreps_sh,self.irreps_A,internal_weights=False,shared_weights=False,irrep_normalization="component",path_normalization="element")
        if {i.i_out for i in self.edge_tp.instructions}!=set(range(len(self.irreps_A))): raise ValueError("edge output contains an unreachable irrep")
        self.radial_head=EdgeRadialHead(radial_feature_dim,radial_hidden_dims,self.edge_tp.weight_numel)
        self.normalization=float(avg_num_neighbors)**-0.5
    def forward(self,source_features,edge_index,edge_vectors,edge_radial,edge_cutoff,num_nodes:int):
        if edge_index.shape[0]!=2 or edge_vectors.shape!=(edge_index.shape[1],3): raise ValueError("invalid directed edge shapes")
        if edge_vectors.numel() and bool(torch.any(torch.sum(edge_vectors*edge_vectors,dim=-1)<=0)): raise ValueError("edge angular basis cannot receive zero-length edges")
        _,o3=import_e3nn_0_4_4()
        edge_sh=o3.spherical_harmonics(self.irreps_sh,edge_vectors,normalize=True,normalization="component")
        weights=self.radial_head(edge_radial,edge_cutoff)
        source,target=edge_index
        messages=self.edge_tp(source_features[source],edge_sh,weights)
        density=messages.new_zeros((num_nodes,self.irreps_A.dim)).index_add(0,target,messages)*self.normalization
        return edge_sh,weights,messages,density
