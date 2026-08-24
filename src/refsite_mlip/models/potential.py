from __future__ import annotations
import math
import torch
from torch import nn
from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.features import build_probability_multipoles
from refsite_mlip.geometry.reference import aligned_reference_sites
from refsite_mlip.graph import update_reference_edge_geometry
from refsite_mlip.interactions import CentralConditioner,EquivariantNodeEncoder,squared_edge_radial_basis
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.transport import TRAIN_FIXED,EVAL_ADAPTIVE,TrainSinkhornConfig,EvalOTConfig,atom_site_displacements,solve_atom_vacancy_ot
from .config import PotentialConfig
from .outputs import PotentialOutput
from .readout import SiteEnergyReadout
from .residual_block import ResidualInteractionBlock

class ReferenceSitePotential(nn.Module):
    def __init__(self,config:PotentialConfig,topology,phase_modes,phase_mode_weights,species_alignment_weights,site_alignment_weights,phase_channel_weights,atomic_baseline=None):
        super().__init__(); config.validate(); self.config=config; self.topology=topology
        self.register_buffer('phase_modes',phase_modes); self.register_buffer('phase_mode_weights',phase_mode_weights); self.register_buffer('species_alignment_weights',species_alignment_weights); self.register_buffer('site_alignment_weights',site_alignment_weights); self.register_buffer('phase_channel_weights',phase_channel_weights)
        baseline=torch.zeros(len(config.species_vocabulary),dtype=topology.reference_cell.dtype) if atomic_baseline is None else torch.as_tensor(atomic_baseline,dtype=topology.reference_cell.dtype)
        if baseline.shape!=(len(config.species_vocabulary),): raise ValueError('atomic baseline shape mismatch')
        self.register_buffer('atomic_baseline',baseline)
        _,o3=import_e3nn_0_4_4(); higher=config.higher_body; self.central=CentralConditioner(higher.species_count,higher.site_type_count,higher.site_type_embedding_dim)
        self.irreps_hidden=o3.Irreps([(higher.n_correlation_channels,o3.Irrep(l,1 if l%2==0 else -1)) for l in range(higher.lmax+1)])
        self.probability_encoder=EquivariantNodeEncoder(config.feature_irreps,self.irreps_hidden); self.central_encoder=o3.Linear(self.central.irreps,self.irreps_hidden,biases=False)
        beta=1/math.sqrt(config.num_layers); self.layers=nn.ModuleList([ResidualInteractionBlock(self.irreps_hidden,self.central.irreps,higher,beta) for _ in range(config.num_layers)])
        scalar_dim=self.irreps_hidden[0].mul; self.scalar_slice=self.irreps_hidden.slices()[0]
        self.readout=SiteEnergyReadout(scalar_dim,self.central.num_channels,config.readout_hidden,config.energy_scale)
    def _species_indices(self,atomic_numbers):
        vocab=torch.tensor(self.config.species_vocabulary,dtype=torch.long,device=atomic_numbers.device); match=atomic_numbers[:,None]==vocab[None,:]
        if bool(torch.any(match.sum(-1)!=1)): raise ValueError('unknown atomic species')
        return torch.argmax(match.to(torch.long),-1)
    def energy(self,positions,atomic_numbers,cell,origin,*,solver_path=TRAIN_FIXED,return_aux=False):
        topology=self.topology.to(device=positions.device,dtype=positions.dtype); species=self._species_indices(atomic_numbers); atom_weights=self.species_alignment_weights.to(positions)[species]
        _,_,cross=typed_reciprocal_fields(positions,origin,cell,topology.reference_fractional,atom_weights,self.site_alignment_weights.to(positions),self.phase_modes,self.phase_channel_weights.to(positions))
        initial=primary_phase_initialization(cross[:3],self.phase_modes[:3]); phase=solve_training_phase(cross,self.phase_modes,self.phase_mode_weights.to(positions),initial,self.config.phase_steps,self.config.phase_damping).phase
        references=aligned_reference_sites(topology.reference_fractional,phase,origin,cell); displacements=atom_site_displacements(positions,references,cell,topology.pbc); cost=displacements.square().sum(-1)/(2*self.config.ell_ot**2)
        if solver_path==TRAIN_FIXED: ot=solve_atom_vacancy_ot(cost,self.config.epsilon_ot,TRAIN_FIXED,'sinkhorn',TrainSinkhornConfig(self.config.train_sinkhorn_iterations))
        elif solver_path==EVAL_ADAPTIVE: ot=solve_atom_vacancy_ot(cost,self.config.epsilon_ot,EVAL_ADAPTIVE,'hybrid',EvalOTConfig(sinkhorn_iterations=16,convergence_tolerance=1e-12))
        else: raise ValueError('unsupported solver path')
        features=build_probability_multipoles(ot.P,ot.q,atomic_numbers,displacements,self.config.feature,topology.site_types)
        c_raw=features.raw_probability_state; c_bar=self.central(c_raw,topology.site_types); h=self.probability_encoder(features.equivariant_features)+self.central_encoder(c_bar)
        geometry=update_reference_edge_geometry(topology,cell,edge_length_scale=self.config.higher_body.edge_length_scale); radial=squared_edge_radial_basis(geometry.radial_coordinate,self.config.higher_body.radial_feature_dim)
        correlations=[]
        for layer in self.layers:
            h,corr=layer(h,c_bar,topology.edge_index,geometry.edge_vectors,radial,geometry.cutoff_values)
            if return_aux: correlations.append(corr)
        site_energy=self.readout(h[:,self.scalar_slice],c_bar); residual=site_energy.sum(); baseline=self.atomic_baseline.to(positions)[species].sum(); total=baseline+residual
        aux={'phase':phase,'ot':ot,'q':ot.q,'multipoles':features,'correlations':correlations} if return_aux else None
        return PotentialOutput(total,site_energy,baseline,residual,h,c_raw,auxiliary=aux)
    def forward(self,positions,atomic_numbers,cell,origin,*,solver_path=TRAIN_FIXED,compute_forces=False,compute_stress=False,create_graph=False,return_aux=False):
        if not compute_forces and not compute_stress: return self.energy(positions,atomic_numbers,cell,origin,solver_path=solver_path,return_aux=return_aux)
        strain=torch.zeros((3,3),dtype=positions.dtype,device=positions.device,requires_grad=compute_stress)
        F=torch.eye(3,dtype=positions.dtype,device=positions.device)+strain
        p=positions@F; H=cell@F; o=origin@F
        out=self.energy(p,atomic_numbers,H,o,solver_path=solver_path,return_aux=return_aux)
        inputs=[]
        if compute_forces: inputs.append(positions)
        if compute_stress: inputs.append(strain)
        gradients=torch.autograd.grad(out.energy,inputs,create_graph=create_graph,retain_graph=create_graph)
        index=0; forces=None; stress=None; voigt=None
        if compute_forces: forces=-gradients[index]; index+=1
        if compute_stress:
            stress=gradients[index]/torch.linalg.det(cell).abs(); stress=.5*(stress+stress.T); voigt=stress[(0,1,2,1,0,0),(0,1,2,2,2,1)]
        return PotentialOutput(out.energy,out.site_energy,out.baseline_energy,out.residual_energy,out.site_features,out.raw_c,forces,stress,voigt,out.auxiliary)
