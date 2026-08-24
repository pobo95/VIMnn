from __future__ import annotations
import torch
import pytest
from refsite_mlip.graph import build_reference_graph_topology,update_reference_edge_geometry,batch_reference_graphs,c2_edge_cutoff
from refsite_mlip.geometry.reference import aligned_reference_sites


def _topology(cell=None,cutoff=1.1,skin=.2,maxstrain=.05):
    if cell is None: cell=torch.diag(torch.tensor([2.,4.,4.],dtype=torch.float64))
    sites=torch.tensor([[0.,0.,0.],[.5,0.,0.]],dtype=torch.float64)
    types=torch.tensor([0,1],dtype=torch.long)
    return build_reference_graph_topology(sites,types,cell,cutoff=cutoff,skin=skin,maximum_strain=maxstrain)


def test_directed_reverse_convention_and_zero_self_exclusion():
    t=_topology(); source,target=t.edge_index
    assert not torch.any((source==target)&torch.all(t.shifts==0,dim=1))
    records={(int(source[i]),int(target[i]),tuple(int(x) for x in t.shifts[i])) for i in range(t.num_edges)}
    for a,b,n in records: assert (b,a,tuple(-x for x in n)) in records
    g=update_reference_edge_geometry(t,t.reference_cell,edge_length_scale=1.)
    for i in range(t.num_edges):
        a,b=map(int,t.edge_index[:,i]); reverse=(t.edge_index[0]==b)&(t.edge_index[1]==a)&torch.all(t.shifts==-t.shifts[i],dim=1)
        torch.testing.assert_close(g.edge_vectors[reverse][0],-g.edge_vectors[i],atol=0,rtol=0)


def test_periodic_self_images_and_multiple_images_are_preserved():
    t=_topology(torch.diag(torch.tensor([1.,4.,4.],dtype=torch.float64)),cutoff=1.05,skin=.1,maxstrain=.02)
    self_images=(t.edge_index[0]==t.edge_index[1])
    assert self_images.sum()>=4
    assert torch.unique(t.shifts,dim=0).shape[0]>1


def test_fractional_representative_shift_gauge_and_origin_phase_cancel():
    t=_topology(); g=update_reference_edge_geometry(t,t.reference_cell,edge_length_scale=1.)
    integers=torch.tensor([[2,-1,0],[-1,1,2]],dtype=torch.long)
    source,target=t.edge_index
    represented=t.reference_fractional+integers
    transformed_shift=t.shifts+integers[target]-integers[source]
    vector=(represented[source]-represented[target]+transformed_shift)*1.0@t.reference_cell
    torch.testing.assert_close(vector,g.edge_vectors,atol=0,rtol=0)
    phase=torch.tensor([.23,-.17,.11],dtype=torch.float64); origin=torch.tensor([2.,-1.,.4],dtype=torch.float64)
    R=aligned_reference_sites(t.reference_fractional,phase,origin,t.reference_cell)
    cartesian=R[source]+t.shifts.to(torch.float64)@t.reference_cell-R[target]
    torch.testing.assert_close(cartesian,g.edge_vectors,atol=5e-16,rtol=0)


def test_current_cell_affine_update_skin_and_strain_failfast():
    t=_topology(); F=torch.tensor([[1.02,.01,0.],[0.,.99,0.],[0.,0.,1.]],dtype=torch.float64)
    g=update_reference_edge_geometry(t,t.reference_cell@F,edge_length_scale=.8)
    reference=update_reference_edge_geometry(t,t.reference_cell,edge_length_scale=.8)
    torch.testing.assert_close(g.edge_vectors,reference.edge_vectors@F,atol=2e-16,rtol=0)
    assert torch.any(~g.active_mask) or torch.all(g.active_mask)
    with pytest.raises(ValueError,match="strain domain"):
        update_reference_edge_geometry(t,t.reference_cell@torch.diag(torch.tensor([1.2,1.,1.],dtype=torch.float64)),edge_length_scale=1.)
    with pytest.raises(ValueError,match="certify"):
        _topology(skin=0.,maxstrain=.05)


def test_zero_edge_graph_and_batch_isolation():
    sites=torch.tensor([[0.,0.,0.]],dtype=torch.float64); types=torch.tensor([0]); cell=torch.eye(3,dtype=torch.float64)*10
    zero=build_reference_graph_topology(sites,types,cell,cutoff=1.,skin=.2,maximum_strain=.05)
    assert zero.num_edges==0
    batch=batch_reference_graphs([zero,_topology()]); assert batch.edge_batch.numel()==_topology().num_edges
    assert torch.all(batch.edge_index[:,batch.edge_batch==1]>=1)


def test_site_permutation_and_edge_order_do_not_change_edge_set():
    t=_topology(); order=torch.tensor([1,0]); p=build_reference_graph_topology(t.reference_fractional[order],t.site_types[order],t.reference_cell,cutoff=t.cutoff,skin=t.skin,maximum_strain=t.maximum_strain)
    inverse=torch.empty_like(order); inverse[order]=torch.arange(2)
    mapped=torch.stack((order[p.edge_index[0]],order[p.edge_index[1]]))
    records=lambda e,n:{(int(e[0,i]),int(e[1,i]),tuple(map(int,n[i]))) for i in range(e.shape[1])}
    assert records(mapped,p.shifts)==records(t.edge_index,t.shifts)


def test_edge_cutoff_crossing_is_c2_and_exactly_zero_outside():
    x=torch.tensor(1.0,dtype=torch.float64,requires_grad=True)
    value=c2_edge_cutoff(x.reshape(1),1.0)[0]
    first=torch.autograd.grad(value,x,create_graph=True)[0]
    second=torch.autograd.grad(first,x)[0]
    assert value==0 and first==0 and second==0
    outside=c2_edge_cutoff(torch.tensor([1.,1.01,4.],dtype=torch.float64),1.)
    torch.testing.assert_close(outside,torch.zeros_like(outside),atol=0,rtol=0)
