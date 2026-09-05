"""Central-conditioned layer-local reference-site correlation prototype."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral,Real
from typing import Any, Mapping
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
from .symmetric_cg import SYMMETRIC_CG_BASIS_VERSION

LEGACY_HIGHER_BODY_CONTRACT_VERSION="central_conditioned_higher_body_v1"
SYMMETRIC_POWER_CONTRACT_VERSION="central_conditioned_symmetric_power_v2"


class HigherBodyArchitectureError(NotImplementedError):
    """Structured rejection of a parsed but not yet integrated architecture."""
    def __init__(self,reason_code:str,message:str):
        self.reason_code=reason_code
        self.message=message
        super().__init__(f"[{reason_code}] {message}")


@dataclass(frozen=True)
class SymmetricCorrelationConfig:
    correlation_order:int
    basis_kind:str="full_path"
    normalization:str="component"
    basis_version:str=SYMMETRIC_CG_BASIS_VERSION
    def __post_init__(self):
        if isinstance(self.correlation_order,bool) or not isinstance(self.correlation_order,Integral):
            raise TypeError("correlation_order must be an integer; bool is forbidden")
        order=int(self.correlation_order)
        if order not in (1,2,3): raise ValueError("correlation_order must be 1, 2, or 3")
        object.__setattr__(self,"correlation_order",order)
        if self.basis_kind!="full_path": raise ValueError("basis_kind must be 'full_path'")
        if self.normalization!="component": raise ValueError("normalization must be 'component'")
        if self.basis_version!=SYMMETRIC_CG_BASIS_VERSION: raise ValueError(f"basis_version must be {SYMMETRIC_CG_BASIS_VERSION!r}")
    def to_dict(self):
        return {"correlation_order":self.correlation_order,"basis_kind":self.basis_kind,"normalization":self.normalization,"basis_version":self.basis_version}
    @classmethod
    def from_dict(cls,values):
        if not isinstance(values,Mapping): raise TypeError("symmetric correlation config must be a mapping")
        if any(type(key) is not str for key in values): raise TypeError("symmetric correlation config keys must be strings")
        expected={"correlation_order","basis_kind","normalization","basis_version"}
        actual=set(values)
        if actual!=expected:
            missing=sorted(expected-actual); unknown=sorted(actual-expected)
            raise ValueError(f"invalid symmetric correlation config keys: missing={missing}, unknown={unknown}")
        return cls(**dict(values))
    def canonical_json(self):
        return json.dumps(self.to_dict(),sort_keys=True,separators=(",",":"),allow_nan=False)
    @property
    def content_fingerprint(self):
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

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
    correlation_mode:str|None="uuu"
    contract_version:str=LEGACY_HIGHER_BODY_CONTRACT_VERSION
    symmetric_correlation:SymmetricCorrelationConfig|None=None
    def validate(self):
        integer=(self.species_count,self.site_type_count,self.site_type_embedding_dim,self.n_correlation_channels,self.radial_feature_dim)
        if any(isinstance(v,bool) or not isinstance(v,Integral) or v<=0 for v in integer): raise ValueError("channel dimensions must be positive integers")
        if self.lmax not in (0,1,2): raise ValueError("prototype lmax must be 0, 1, or 2")
        if self.n_correlation_channels>16: raise ValueError("prototype correlation channel count is capped at 16")
        if any(isinstance(v,bool) or not isinstance(v,Integral) or v<=0 for v in self.radial_hidden_dims): raise ValueError("radial hidden dimensions must be positive")
        for name,v in (("avg_num_neighbors",self.avg_num_neighbors),("cutoff",self.cutoff),("edge_length_scale",self.edge_length_scale)):
            if isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(float(v)) or v<=0: raise ValueError(f"{name} must be finite and positive")
        if self.contract_version==LEGACY_HIGHER_BODY_CONTRACT_VERSION:
            if self.symmetric_correlation is not None: raise ValueError("v1 higher-body config must not contain symmetric_correlation")
            if self.correlation_mode not in ("uuu","uvw"): raise ValueError("invalid correlation mode")
            if self.correlation_mode=="uvw" and self.n_correlation_channels>2: raise ValueError("dense uvw correlation is unit-test-only and capped at n_corr<=2")
            return
        if self.contract_version==SYMMETRIC_POWER_CONTRACT_VERSION:
            if isinstance(self.lmax,bool) or not isinstance(self.lmax,Integral): raise TypeError("v2 lmax must be an integer; bool is forbidden")
            if self.correlation_mode is not None: raise ValueError("v2 higher-body config forbids v1-only correlation_mode")
            if not isinstance(self.symmetric_correlation,SymmetricCorrelationConfig): raise TypeError("v2 higher-body config requires SymmetricCorrelationConfig")
            return
        raise ValueError("unsupported higher-body contract version")
    def to_dict(self):
        self.validate()
        result={"irreps_feature":self.irreps_feature,"species_count":self.species_count,"site_type_count":self.site_type_count,"site_type_embedding_dim":self.site_type_embedding_dim,"n_correlation_channels":self.n_correlation_channels,"lmax":self.lmax,"radial_feature_dim":self.radial_feature_dim,"radial_hidden_dims":list(self.radial_hidden_dims),"avg_num_neighbors":self.avg_num_neighbors,"cutoff":self.cutoff,"edge_length_scale":self.edge_length_scale}
        if self.contract_version==LEGACY_HIGHER_BODY_CONTRACT_VERSION:
            result["correlation_mode"]=self.correlation_mode
            result["contract_version"]=self.contract_version
        else:
            result["contract_version"]=self.contract_version
            result["symmetric_correlation"]=self.symmetric_correlation.to_dict()
        return result
    @classmethod
    def from_dict(cls,d):
        if not isinstance(d,Mapping): raise TypeError("higher-body config must be a mapping")
        if any(type(key) is not str for key in d): raise TypeError("higher-body config keys must be strings")
        values=dict(d); version=values.get("contract_version",LEGACY_HIGHER_BODY_CONTRACT_VERSION)
        common={"irreps_feature","species_count","site_type_count","site_type_embedding_dim","n_correlation_channels","lmax","radial_feature_dim","radial_hidden_dims","avg_num_neighbors","cutoff","edge_length_scale","contract_version"}
        if version==LEGACY_HIGHER_BODY_CONTRACT_VERSION:
            allowed=common|{"correlation_mode"}
            if "symmetric_correlation" in values: raise ValueError("v1 higher-body dictionary forbids symmetric_correlation")
        elif version==SYMMETRIC_POWER_CONTRACT_VERSION:
            allowed=common|{"symmetric_correlation"}
            if "correlation_mode" in values: raise ValueError("v2 higher-body dictionary forbids correlation_mode")
            missing=allowed-set(values); unknown=set(values)-allowed
            if missing or unknown: raise ValueError(f"invalid v2 higher-body config keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
            values["symmetric_correlation"]=SymmetricCorrelationConfig.from_dict(values["symmetric_correlation"])
            values["correlation_mode"]=None
            allowed=allowed|{"correlation_mode"}
        else:
            raise ValueError("unsupported higher-body contract version")
        unknown=set(values)-allowed
        if unknown: raise ValueError(f"unknown higher-body config keys: {sorted(unknown)}")
        if "radial_hidden_dims" in values: values["radial_hidden_dims"]=tuple(values["radial_hidden_dims"])
        result=cls(**values); result.validate(); return result
    def canonical_json(self):
        return json.dumps(self.to_dict(),sort_keys=True,separators=(",",":"),allow_nan=False)
    @property
    def content_fingerprint(self):
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
    def require_legacy_execution(self,component:str):
        self.validate()
        if self.contract_version!=LEGACY_HIGHER_BODY_CONTRACT_VERSION:
            raise HigherBodyArchitectureError("SYMMETRIC_CORRELATION_NOT_INTEGRATED",f"{component} cannot execute {self.contract_version!r}; integration is deferred to Milestone 11B-2")

class CentralConditionedHigherBody(nn.Module):
    def __init__(self,config:HigherBodyConfig):
        config.require_legacy_execution("CentralConditionedHigherBody"); super().__init__(); self.config=config; _,o3=import_e3nn_0_4_4()
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
