from __future__ import annotations
import copy
import torch
import pytest
from test_phase_ot_probability_multipoles import (_pipeline,_vacancy_inputs,_rotation,_symmetric_directions,TRAIN_FIXED,EVAL_ADAPTIVE)
from refsite_mlip.graph import build_reference_graph_topology,update_reference_edge_geometry,batch_reference_graphs
from refsite_mlip.interactions import CentralConditionedHigherBody,HigherBodyConfig,squared_edge_radial_basis


def _topology(data,sites=None,types=None):
    return build_reference_graph_topology(data["sites"] if sites is None else sites,data["site_types"] if types is None else types,data["cell"],cutoff=3.,skin=.5,maximum_strain=.1)

def _model(irreps):
    torch.manual_seed(23)
    return CentralConditionedHigherBody(HigherBodyConfig(str(irreps),2,2,2,1,2,3,(4,),6.,3.,1.)).double()

def _energy(result):
    return .017*result.C1.square().sum()+.023*result.C2.square().sum()+.031*result.C3.square().sum()+.011*result.Z1.square().sum()+.013*result.Z2.square().sum()+.019*result.Z3.square().sum()

def _full(data,positions,weights,model,topology,*,origin=None,cell=None,path=TRAIN_FIXED,sites=None,site_weights=None,site_types=None,edge_order=None):
    cell=data['cell'] if cell is None else cell; site_types=data['site_types'] if site_types is None else site_types
    upstream=_pipeline(data,positions,weights,origin=origin,cell=cell,path=path,sites=sites,site_weights=site_weights,site_types=site_types)
    geometry=update_reference_edge_geometry(topology,cell,edge_length_scale=1.)
    edge_index=topology.edge_index; vectors=geometry.edge_vectors; cutoff=geometry.cutoff_values
    if edge_order is not None: edge_index=edge_index[:,edge_order]; vectors=vectors[edge_order]; cutoff=cutoff[edge_order]
    radial=squared_edge_radial_basis(torch.sum(vectors*vectors,dim=-1),3)
    algebra=model(upstream['features'].equivariant_features,upstream['features'].raw_probability_state,site_types,edge_index,vectors,radial,cutoff)
    return upstream,algebra,_energy(algebra)


def test_train_eval_parity_and_edge_order(typed_crystal):
    pos,w=_vacancy_inputs(typed_crystal); top=_topology(typed_crystal); up=_pipeline(typed_crystal,pos,w); model=_model(up['features'].irreps_out)
    train,a,e=_full(typed_crystal,pos,w,model,top); adaptive,aa,ee=_full(typed_crystal,pos,w,model,top,path=EVAL_ADAPTIVE)
    assert max(float(train['ot'].row_residual),float(train['ot'].column_residual))<=1e-7
    for name in ('A','C1','C2','C3','Z1','Z2','Z3'): torch.testing.assert_close(getattr(a,name),getattr(aa,name),atol=2e-11,rtol=2e-11)
    order=torch.arange(top.num_edges-1,-1,-1); _,ap,ep=_full(typed_crystal,pos,w,model,top,edge_order=order)
    torch.testing.assert_close(ep,e,atol=3e-13,rtol=3e-13); torch.testing.assert_close(ap.Z3,a.Z3,atol=3e-13,rtol=3e-13)

@pytest.mark.parametrize('Q',[_rotation(),torch.diag(torch.tensor([-1.,1.,1.],dtype=torch.float64))])
def test_full_rotation_reflection_energy_force(Q,typed_crystal):
    pos,w=_vacancy_inputs(typed_crystal); top=_topology(typed_crystal); model=_model(_pipeline(typed_crystal,pos,w)['features'].irreps_out)
    p=pos.clone().requires_grad_(True); _,a,e=_full(typed_crystal,p,w,model,top); force=-torch.autograd.grad(e,p)[0]
    rotated=dict(typed_crystal); rotated['positions']=typed_crystal['positions']@Q.T; rotated['origin']=typed_crystal['origin']@Q.T; rotated['cell']=typed_crystal['cell']@Q.T
    rtop=_topology(rotated); rp=(p.detach()@Q.T).requires_grad_(True); _,ar,er=_full(rotated,rp,w,model,rtop); rf=-torch.autograd.grad(er,rp)[0]
    torch.testing.assert_close(er,e,atol=2e-9,rtol=2e-8); torch.testing.assert_close(rf,force@Q.T,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(ar.Z3,a.Z3@model.irreps_Z3.D_from_matrix(Q).T,atol=8e-9,rtol=3e-7)


def test_translation_wrapping_site_permutation_force_fd_and_double_backward(typed_crystal):
    pos,w=_vacancy_inputs(typed_crystal); top=_topology(typed_crystal); model=_model(_pipeline(typed_crystal,pos,w)['features'].irreps_out)
    p=pos.clone().requires_grad_(True); _,a,e=_full(typed_crystal,p,w,model,top); force=-torch.autograd.grad(e,p,create_graph=True)[0]
    torch.testing.assert_close(force.sum(0),torch.zeros(3,dtype=torch.float64),atol=2e-8,rtol=0)
    shift=torch.tensor([.73,-.41,.29],dtype=torch.float64); _,_,moved=_full(typed_crystal,p+shift,w,model,top); _,_,joint=_full(typed_crystal,p+shift,w,model,top,origin=typed_crystal['origin']+shift)
    lattice=torch.tensor([1.,-1.,2.],dtype=torch.float64)@typed_crystal['cell']; wrapped=p.detach().clone(); wrapped[0]+=lattice; _,_,wrape=_full(typed_crystal,wrapped,w,model,top)
    for other in (moved,joint,wrape): torch.testing.assert_close(other,e,atol=3e-8,rtol=3e-8)
    h=2e-6; direction=torch.zeros_like(p); direction[2,1]=h
    plus=_full(typed_crystal,p.detach()+direction,w,model,top)[2]; minus=_full(typed_crystal,p.detach()-direction,w,model,top)[2]
    torch.testing.assert_close(force[2,1],-(plus-minus)/(2*h),atol=3e-6,rtol=3e-5)
    grads=torch.autograd.grad(force.square().sum(),[model.node_encoder.linear.weight,model.correlations.C2_product.weight,model.central.embedding.weight])
    assert all(torch.all(torch.isfinite(g)) and torch.linalg.vector_norm(g)>0 for g in grads)


def _strain_energy(data,strain,model,top):
    F=torch.eye(3,dtype=torch.float64)+strain; transformed=dict(data); transformed['positions']=data['positions']@F; transformed['origin']=data['origin']@F; transformed['cell']=data['cell']@F
    return _full(transformed,transformed['positions'][:5],transformed['atom_weights'][:5],model,top)[2]

def test_full_stress_finite_difference_symmetry(typed_crystal):
    pos,w=_vacancy_inputs(typed_crystal); top=_topology(typed_crystal); model=_model(_pipeline(typed_crystal,pos,w)['features'].irreps_out)
    strain=torch.zeros((3,3),dtype=torch.float64,requires_grad=True); stress=torch.autograd.grad(_strain_energy(typed_crystal,strain,model,top),strain)[0]; h=2e-6
    for d in _symmetric_directions():
        fd=(_strain_energy(typed_crystal,h*d,model,top)-_strain_energy(typed_crystal,-h*d,model,top))/(2*h)
        torch.testing.assert_close(torch.sum(stress*d),fd,atol=5e-6,rtol=5e-5)
    torch.testing.assert_close(stress,stress.T,atol=3e-7,rtol=3e-7)


def test_full_position_gradcheck_gradgradcheck(typed_crystal):
    pos,w=_vacancy_inputs(typed_crystal); top=_topology(typed_crystal); model=_model(_pipeline(typed_crystal,pos,w)['features'].irreps_out)
    p=pos.clone().requires_grad_(True); fn=lambda value:_full(typed_crystal,value,w,model,top)[2]
    assert torch.autograd.gradcheck(fn,(p,),eps=1e-6,atol=8e-5,rtol=8e-4)
    assert torch.autograd.gradgradcheck(fn,(p,),eps=1e-6,atol=2e-4,rtol=2e-3)


def test_pristine_k1_k2_disjoint_batch_matches_single(typed_crystal):
    top=_topology(typed_crystal); counts=(6,5,4); upstream=[]
    for n in counts: upstream.append(_pipeline(typed_crystal,typed_crystal['positions'][:n],typed_crystal['atom_weights'][:n]))
    model=_model(upstream[0]['features'].irreps_out); singles=[]
    geometry=update_reference_edge_geometry(top,typed_crystal['cell'],edge_length_scale=1.); radial=squared_edge_radial_basis(geometry.squared_lengths,3)
    for u in upstream: singles.append(model(u['features'].equivariant_features,u['features'].raw_probability_state,typed_crystal['site_types'],top.edge_index,geometry.edge_vectors,radial,geometry.cutoff_values))
    batch=batch_reference_graphs([top,top,top]); features=torch.cat([u['features'].equivariant_features for u in upstream]); raw=torch.cat([u['features'].raw_probability_state for u in upstream]); types=typed_crystal['site_types'].repeat(3); vectors=geometry.edge_vectors.repeat(3,1); brad=radial.repeat(3,1); cutoff=geometry.cutoff_values.repeat(3)
    combined=model(features,raw,types,batch.edge_index,vectors,brad,cutoff)
    torch.testing.assert_close(combined.Z3,torch.cat([s.Z3 for s in singles]),atol=4e-13,rtol=4e-13)
    assert torch.count_nonzero(upstream[0]['ot'].q)==0 and abs(float(upstream[1]['ot'].q.sum())-1)<1e-12 and abs(float(upstream[2]['ot'].q.sum())-2)<1e-12


def test_stress_rotation_covariance(typed_crystal):
    pos,w=_vacancy_inputs(typed_crystal); top=_topology(typed_crystal); model=_model(_pipeline(typed_crystal,pos,w)['features'].irreps_out)
    strain=torch.zeros((3,3),dtype=torch.float64,requires_grad=True); stress=torch.autograd.grad(_strain_energy(typed_crystal,strain,model,top),strain)[0]
    Q=_rotation(); rotated=dict(typed_crystal); rotated['positions']=typed_crystal['positions']@Q.T; rotated['origin']=typed_crystal['origin']@Q.T; rotated['cell']=typed_crystal['cell']@Q.T
    rtop=_topology(rotated); rstrain=torch.zeros((3,3),dtype=torch.float64,requires_grad=True); rstress=torch.autograd.grad(_strain_energy(rotated,rstrain,model,rtop),rstrain)[0]
    torch.testing.assert_close(rstress,Q@stress@Q.T,atol=5e-7,rtol=5e-6)
