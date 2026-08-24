"""Central-conditioned layer-local reference-site correlation prototype."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from numbers import Integral,Real
from typing import Any
import math
import torch
from torch import nn
from refsite_mlip.compatibility import import_e3nn_0_4_4
from .central_conditioning import CentralConditioner
from .correlations import CentralOuterProduct,DensityCorrelations
from .edge_density import EdgeNeighborDensity
from .instructions import instruction_metadata
from .node_encoder import EquivariantNodeEncoder
from .result import HigherBodyResult

@dataclass(frozen=True)
class HigherBodyConfig:
    irreps_feature:str
    species_count:int
    site_type_count:int
    site_type_embedding_dim:int=2
    n_correlation_channels:int=2
    lmax:int=2
    radial_feature_dim:int=3
    radial_hidden_dims:tuple[int,...]=(8,)
    avg_num_neighbors:float=4.0
    cutoff:float=3.0
    edge_length_scale:float=1.0
    correlation_mode:str="uuu"
    contract_version:str="central_conditioned_higher_body_v1"
    def validate(self):
        integer=(self.species_count,self.site_type_count,self.site_type_embedding_dim,self.n_correlation_channels,self.radial_feature_dim)
        if any(isinstance(v,bool) or not isinstance(v,Integral) or v<=0 for v in integer): raise ValueError("channel dimensions must be positive integers")
        if self.lmax not in (0,1,2): raise ValueError("prototype lmax must be 0, 1, or 2")
        if self.n_correlation_channels>16: raise ValueError("prototype correlation channel count is capped at 16")
        if any(isinstance(v,bool) or not isinstance(v,Integral) or v<=0 for v in self.radial_hidden_dims): raise ValueError("radial hidden dimensions must be positive")
        for name,v in (("avg_num_neighbors",self.avg_num_neighbors),("cutoff",self.cutoff),("edge_length_scale",self.edge_length_scale)):
            if isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(float(v)) or v<=0: raise ValueError(f"{name} must be finite and positive")
        if self.correlation_mode not in ("uuu","uvw"): raise ValueError("invalid correlation mode")
        if self.correlation_mode=="uvw" and self.n_correlation_channels>2: raise ValueError("dense uvw correlation is unit-test-only and capped at n_corr<=2")
        if self.contract_version!="central_conditioned_higher_body_v1": raise ValueError("unsupported higher-body contract version")
    def to_dict(self):
        self.validate(); d=asdict(self); d["radial_hidden_dims"]=list(self.radial_hidden_dims); return d
    @classmethod
    def from_dict(cls,d):
        values=dict(d); values["radial_hidden_dims"]=tuple(values["radial_hidden_dims"]); result=cls(**values); result.validate(); return result

class CentralConditionedHigherBody(nn.Module):
    def __init__(self,config:HigherBodyConfig):
        super().__init__(); config.validate(); self.config=config; _,o3=import_e3nn_0_4_4()
        self.irreps_feature=o3.Irreps(config.irreps_feature)
        self.irreps_h=o3.Irreps([(config.n_correlation_channels,o3.Irrep(l,1 if l%2==0 else -1)) for l in range(config.lmax+1)])
        self.irreps_A=self.irreps_h; self.irreps_C1=self.irreps_h; self.irreps_C2=self.irreps_h; self.irreps_C3=self.irreps_h
        self.central=CentralConditioner(config.species_count,config.site_type_count,config.site_type_embedding_dim)
        self.irreps_source=self.irreps_h+self.central.irreps
        self.node_encoder=EquivariantNodeEncoder(self.irreps_feature,self.irreps_h)
        self.edge_density=EdgeNeighborDensity(self.irreps_source,self.irreps_A,config.lmax,config.radial_feature_dim,config.radial_hidden_dims,config.avg_num_neighbors)
        self.correlations=DensityCorrelations(self.irreps_A,config.correlation_mode)
        self.central_products=nn.ModuleList([CentralOuterProduct(self.central.irreps,self.irreps_A) for _ in range(3)])
        self.irreps_Z1=self.central_products[0].irreps_out; self.irreps_Z2=self.central_products[1].irreps_out; self.irreps_Z3=self.central_products[2].irreps_out
        self.irreps_sh=self.edge_density.irreps_sh
    def parameter_diagnostics(self):
        count=lambda module:sum(p.numel() for p in module.parameters())
        return {"node_encoder":count(self.node_encoder),"edge_tensor_product":count(self.edge_density.edge_tp),"radial_head":count(self.edge_density.radial_head),"C2":count(self.correlations.C2_product),"C3":count(self.correlations.C3_product),"site_type_embedding":count(self.central.embedding),"total":count(self)}
    def _instructions(self):
        return {"edge":tuple(x.to_dict() for x in instruction_metadata(self.edge_density.edge_tp)),"C2":tuple(x.to_dict() for x in instruction_metadata(self.correlations.C2_product)),"C3":tuple(x.to_dict() for x in instruction_metadata(self.correlations.C3_product)),"Z":tuple(x.to_dict() for x in instruction_metadata(self.central_products[0].product))}
    def forward(self,probability_multipoles,c_raw,site_types,edge_index,edge_vectors,edge_radial,edge_cutoff,*,return_edge_messages:bool=True):
        if probability_multipoles.ndim!=2 or probability_multipoles.shape[1]!=self.irreps_feature.dim: raise ValueError("probability multipole shape does not match irreps_feature")
        if c_raw.shape[0]!=probability_multipoles.shape[0]: raise ValueError("central state site count mismatch")
        h=self.node_encoder(probability_multipoles); c_bar=self.central(c_raw,site_types)
        source=torch.cat((h,c_bar),dim=-1)
        edge_sh,edge_weights,messages,A=self.edge_density(source,edge_index,edge_vectors,edge_radial,edge_cutoff,probability_multipoles.shape[0])
        C1,C2,C3=self.correlations(A)
        Z1=self.central_products[0](c_bar,C1); Z2=self.central_products[1](c_bar,C2); Z3=self.central_products[2](c_bar,C3)
        irreps={name:str(value) for name,value in (("feature",self.irreps_feature),("h",self.irreps_h),("source",self.irreps_source),("sh",self.irreps_sh),("A",self.irreps_A),("C1",self.irreps_C1),("C2",self.irreps_C2),("C3",self.irreps_C3),("Z1",self.irreps_Z1),("Z2",self.irreps_Z2),("Z3",self.irreps_Z3))}
        layout=self.central.layout
        channels={"central":{"constant":(layout.constant.start,layout.constant.stop),"species":(layout.species.start,layout.species.stop),"vacancy":(layout.vacancy.start,layout.vacancy.stop),"site_type":(layout.site_type.start,layout.site_type.stop),"vacancy_site_type":(layout.vacancy_site_type.start,layout.vacancy_site_type.stop)},"instructions":self._instructions(),"edge_weight_numel":self.edge_density.edge_tp.weight_numel,"C2_weight_numel":self.correlations.C2_product.weight_numel,"C3_weight_numel":self.correlations.C3_product.weight_numel,"output_dimension":self.irreps_Z1.dim+self.irreps_Z2.dim+self.irreps_Z3.dim}
        graph={"num_nodes":probability_multipoles.shape[0],"num_edges":edge_index.shape[1],"avg_num_neighbors":self.config.avg_num_neighbors}
        return HigherBodyResult(h,c_raw,c_bar,edge_sh,edge_weights,messages if return_edge_messages else None,A,C1,C2,C3,Z1,Z2,Z3,irreps,channels,graph,self.parameter_diagnostics())
