"""Legal equivariant projection into correlation irreps."""
from torch import nn
from refsite_mlip.compatibility import import_e3nn_0_4_4

class EquivariantNodeEncoder(nn.Module):
    def __init__(self,irreps_feature,irreps_h):
        super().__init__(); _,o3=import_e3nn_0_4_4()
        self.irreps_feature=o3.Irreps(irreps_feature); self.irreps_h=o3.Irreps(irreps_h)
        self.linear=o3.Linear(self.irreps_feature,self.irreps_h,biases=False)
        covered={ins.i_out for ins in self.linear.instructions}
        if covered!=set(range(len(self.irreps_h))): raise ValueError("node encoder output contains an unreachable irrep")
    def forward(self,features): return self.linear(features)
