"""Layer-local degree-1/2/3 density correlation algebra."""
from __future__ import annotations
from torch import nn
from refsite_mlip.compatibility import import_e3nn_0_4_4
from .instructions import legal_instructions

class DensityCorrelations(nn.Module):
    def __init__(self,irreps_corr,mode:str="uuu"):
        super().__init__(); _,o3=import_e3nn_0_4_4(); self.irreps_corr=o3.Irreps(irreps_corr)
        if mode not in ("uuu","uvw"): raise ValueError("correlation mode must be uuu or uvw")
        if mode=="uuu":
            multiplicities={mul for mul,_ in self.irreps_corr}
            if len(multiplicities)!=1: raise ValueError("uuu requires identical block multiplicities")
        instructions=legal_instructions(self.irreps_corr,self.irreps_corr,self.irreps_corr,mode,True)
        self.C2_product=o3.TensorProduct(self.irreps_corr,self.irreps_corr,self.irreps_corr,instructions,internal_weights=True,shared_weights=True,irrep_normalization="component",path_normalization="element")
        self.C3_product=o3.TensorProduct(self.irreps_corr,self.irreps_corr,self.irreps_corr,instructions,internal_weights=True,shared_weights=True,irrep_normalization="component",path_normalization="element")
        self.mode=mode
    def forward(self,A):
        C1=A; C2=self.C2_product(C1,A); C3=self.C3_product(C2,A); return C1,C2,C3

class CentralOuterProduct(nn.Module):
    def __init__(self,central_irreps,correlation_irreps):
        super().__init__(); _,o3=import_e3nn_0_4_4(); self.central_irreps=o3.Irreps(central_irreps); self.correlation_irreps=o3.Irreps(correlation_irreps)
        if len(self.central_irreps)!=1 or self.central_irreps[0].ir!=o3.Irrep("0e"): raise ValueError("central conditioner must be one semantic 0e block")
        nc=self.central_irreps[0].mul
        self.irreps_out=o3.Irreps([(nc*mul,ir) for mul,ir in self.correlation_irreps])
        instructions=[(0,i,i,"uvuv",False,1.0) for i in range(len(self.correlation_irreps))]
        self.product=o3.TensorProduct(self.central_irreps,self.correlation_irreps,self.irreps_out,instructions,internal_weights=False,shared_weights=True,irrep_normalization="component",path_normalization="element")
    def forward(self,central,correlation): return self.product(central,correlation)
