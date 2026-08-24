from __future__ import annotations
import math
import torch
from torch import nn
from refsite_mlip.compatibility import import_e3nn_0_4_4
import_e3nn_0_4_4()
from e3nn.nn import Gate
from refsite_mlip.interactions import EdgeNeighborDensity,DensityCorrelations,CentralOuterProduct

class ResidualInteractionBlock(nn.Module):
    def __init__(self,hidden_irreps,central_irreps,higher_config,residual_scale):
        super().__init__(); _,o3=import_e3nn_0_4_4(); self.irreps_h=o3.Irreps(hidden_irreps); self.residual_scale=float(residual_scale)
        self.edge=EdgeNeighborDensity(self.irreps_h+central_irreps,self.irreps_h,higher_config.lmax,higher_config.radial_feature_dim,higher_config.radial_hidden_dims,higher_config.avg_num_neighbors)
        self.corr=DensityCorrelations(self.irreps_h,higher_config.correlation_mode)
        self.outer=nn.ModuleList([CentralOuterProduct(central_irreps,self.irreps_h) for _ in range(3)])
        self.contract=nn.ModuleList([o3.Linear(x.irreps_out,self.irreps_h,biases=False) for x in self.outer])
        scalars=[]; gated=[]
        for mul,ir in self.irreps_h:
            if ir.l==0: scalars.append((mul,ir))
            else: gated.append((mul,ir))
        self.irreps_scalars=o3.Irreps(scalars); self.irreps_gated=o3.Irreps(gated)
        self.irreps_gates=o3.Irreps([(sum(m for m,_ in gated),o3.Irrep('0e'))]) if gated else o3.Irreps('')
        acts=[torch.nn.functional.silu if ir.p==1 else torch.tanh for _,ir in self.irreps_scalars]
        gate_acts=[torch.sigmoid for _ in self.irreps_gates]
        self.gate=Gate(self.irreps_scalars,acts,self.irreps_gates,gate_acts,self.irreps_gated)
        if self.gate.irreps_out!=self.irreps_h: raise ValueError('Gate output must equal hidden irreps')
        gate_in=self.gate.irreps_in
        self.self_projection=o3.Linear(self.irreps_h,gate_in,biases=False); self.message_projection=o3.Linear(self.irreps_h,gate_in,biases=False)
        self.raw_skip=o3.Linear(central_irreps,self.irreps_h,biases=False)
    def forward(self,h,c_bar,edge_index,edge_vectors,edge_radial,edge_cutoff):
        source=torch.cat((h,c_bar),dim=-1); edge_sh,weights,messages,A=self.edge(source,edge_index,edge_vectors,edge_radial,edge_cutoff,h.shape[0])
        C1,C2,C3=self.corr(A); Z=[self.outer[i](c_bar,C) for i,C in enumerate((C1,C2,C3))]
        u=sum(self.contract[i](Z[i]) for i in range(3))/math.sqrt(3.0)
        delta=self.gate(self.self_projection(h)+self.message_projection(u))
        updated=h+self.residual_scale*delta+self.raw_skip(c_bar)
        return updated,{'A':A,'C1':C1,'C2':C2,'C3':C3,'Z1':Z[0],'Z2':Z[1],'Z3':Z[2]}
