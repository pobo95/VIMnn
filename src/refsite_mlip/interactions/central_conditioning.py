"""Immutable central chemistry, vacancy, and typed-site scalar conditioning."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from refsite_mlip.compatibility import import_e3nn_0_4_4

@dataclass(frozen=True)
class CentralChannelLayout:
    constant:slice; species:slice; vacancy:slice; site_type:slice; vacancy_site_type:slice

class CentralConditioner(nn.Module):
    def __init__(self,species_count:int,site_type_count:int,embedding_dim:int):
        super().__init__()
        if min(species_count,site_type_count,embedding_dim)<=0: raise ValueError("central channel counts must be positive")
        self.species_count=species_count; self.site_type_count=site_type_count; self.embedding_dim=embedding_dim
        self.embedding=nn.Embedding(site_type_count,embedding_dim)
        q=1+species_count; e=q+1; qe=e+embedding_dim
        self.layout=CentralChannelLayout(slice(0,1),slice(1,q),slice(q,q+1),slice(e,e+embedding_dim),slice(qe,qe+embedding_dim))
        self.num_channels=qe+embedding_dim
        _,o3=import_e3nn_0_4_4(); self.irreps=o3.Irreps([(self.num_channels,o3.Irrep("0e"))])
    def forward(self,c_raw:torch.Tensor,site_types:torch.Tensor)->torch.Tensor:
        if c_raw.ndim!=2 or c_raw.shape[1]!=self.species_count+1: raise ValueError("c_raw must be [M,species_count+1]")
        if site_types.shape!=(c_raw.shape[0],) or site_types.dtype!=torch.long: raise ValueError("site_types must be long [M]")
        if site_types.device!=c_raw.device: raise ValueError("site_types and c_raw must share device")
        if site_types.numel() and bool(torch.any((site_types<0)|(site_types>=self.site_type_count))): raise ValueError("unknown site type")
        embedding=self.embedding(site_types)
        if embedding.dtype!=c_raw.dtype: raise ValueError("module and c_raw dtype differ; move the module to input dtype")
        q=c_raw[:,-1:]
        return torch.cat((torch.ones_like(q),c_raw,embedding,q*embedding),dim=-1)
