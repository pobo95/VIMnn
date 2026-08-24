from __future__ import annotations
from dataclasses import dataclass
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig

@dataclass(frozen=True)
class PotentialConfig:
    species_vocabulary:tuple[int,...]
    num_layers:int
    feature:ProbabilityMultipoleConfig
    higher_body:HigherBodyConfig
    readout_hidden:int=16
    energy_scale:float=1.0
    epsilon_ot:float=.5
    ell_ot:float=1.5
    train_sinkhorn_iterations:int=256
    phase_steps:tuple[float,...]=(.7,.8,.9,1.)
    phase_damping:tuple[float,...]=(2.,1.,.5,.2)
    def validate(self):
        if self.num_layers<=0: raise ValueError('num_layers must be positive')
        if self.feature.species_vocabulary!=self.species_vocabulary: raise ValueError('feature species mismatch')
        if self.higher_body.species_count!=len(self.species_vocabulary): raise ValueError('higher-body species mismatch')
        if self.higher_body.irreps_feature!=str(self.feature_irreps): raise ValueError('higher-body feature irreps mismatch')
        if self.readout_hidden<=0 or self.energy_scale<=0: raise ValueError('readout/energy scale must be positive')
    @property
    def feature_irreps(self):
        from refsite_mlip.compatibility import import_e3nn_0_4_4
        _,o3=import_e3nn_0_4_4(); A=len(self.species_vocabulary); n=self.feature.n_radial
        return o3.Irreps(f'{A}x0e + {A*n}x0e + {A*n}x1o + {A*n}x2e')
