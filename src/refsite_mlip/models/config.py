from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.transport import TransportSupportConfig

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
    transport_support:TransportSupportConfig=field(default_factory=TransportSupportConfig)
    def validate(self):
        if self.num_layers<=0: raise ValueError('num_layers must be positive')
        if self.feature.species_vocabulary!=self.species_vocabulary: raise ValueError('feature species mismatch')
        if self.higher_body.species_count!=len(self.species_vocabulary): raise ValueError('higher-body species mismatch')
        if self.higher_body.irreps_feature!=str(self.feature_irreps): raise ValueError('higher-body feature irreps mismatch')
        if self.readout_hidden<=0 or self.energy_scale<=0: raise ValueError('readout/energy scale must be positive')
        if not isinstance(self.transport_support,TransportSupportConfig): raise TypeError('transport_support must be TransportSupportConfig')
    def to_dict(self)->dict[str,Any]:
        self.validate()
        return {
            'species_vocabulary':list(self.species_vocabulary),
            'num_layers':self.num_layers,
            'feature':self.feature.to_dict(),
            'higher_body':self.higher_body.to_dict(),
            'readout_hidden':self.readout_hidden,
            'energy_scale':self.energy_scale,
            'epsilon_ot':self.epsilon_ot,
            'ell_ot':self.ell_ot,
            'train_sinkhorn_iterations':self.train_sinkhorn_iterations,
            'phase_steps':list(self.phase_steps),
            'phase_damping':list(self.phase_damping),
            'transport_support':self.transport_support.to_dict(),
        }
    @classmethod
    def from_dict(cls,values:Mapping[str,Any])->'PotentialConfig':
        if not isinstance(values,Mapping): raise TypeError('potential config must be reconstructed from a mapping')
        data=dict(values)
        data['species_vocabulary']=tuple(data['species_vocabulary'])
        data['feature']=ProbabilityMultipoleConfig.from_dict(data['feature'])
        data['higher_body']=HigherBodyConfig.from_dict(data['higher_body'])
        data['phase_steps']=tuple(data.get('phase_steps',(.7,.8,.9,1.)))
        data['phase_damping']=tuple(data.get('phase_damping',(2.,1.,.5,.2)))
        data['transport_support']=TransportSupportConfig.from_dict(data.get('transport_support'))
        result=cls(**data); result.validate(); return result
    @property
    def feature_irreps(self):
        from refsite_mlip.compatibility import import_e3nn_0_4_4
        _,o3=import_e3nn_0_4_4(); A=len(self.species_vocabulary); n=self.feature.n_radial
        return o3.Irreps(f'{A}x0e + {A*n}x0e + {A*n}x1o + {A*n}x2e')
