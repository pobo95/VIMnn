import math
import torch
from torch import nn
class SiteEnergyReadout(nn.Module):
    def __init__(self,scalar_dim,central_dim,hidden,energy_scale):
        super().__init__(); self.mlp=nn.Sequential(nn.Linear(scalar_dim+central_dim,hidden),nn.SiLU(),nn.Linear(hidden,1)); self.raw=nn.Linear(central_dim,1,bias=False); self.register_buffer('energy_scale',torch.tensor(float(energy_scale)))
        nn.init.xavier_uniform_(self.mlp[0].weight); nn.init.zeros_(self.mlp[0].bias); nn.init.normal_(self.mlp[-1].weight,0.,1e-2/math.sqrt(hidden)); nn.init.zeros_(self.mlp[-1].bias); nn.init.normal_(self.raw.weight,0.,1./math.sqrt(central_dim))
    def forward(self,scalars,central): return self.energy_scale.to(scalars)*(self.mlp(torch.cat((scalars,central),-1))+self.raw(central)).squeeze(-1)
