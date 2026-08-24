from __future__ import annotations
import ast,copy
from pathlib import Path
import pytest
import torch
from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.interactions import *


def _model(n=2,mode="uuu",dtype=torch.float64,device="cpu"):
    torch.manual_seed(7)
    cfg=HigherBodyConfig("2x0e+2x1o+2x2e",1,2,site_type_embedding_dim=2,n_correlation_channels=n,lmax=2,radial_feature_dim=3,radial_hidden_dims=(5,),avg_num_neighbors=2.,correlation_mode=mode)
    return CentralConditionedHigherBody(cfg).to(device=device,dtype=dtype)

def _inputs(model,dtype=torch.float64,device="cpu",requires=False):
    torch.manual_seed(11); M=3
    features=torch.randn(M,model.irreps_feature.dim,dtype=dtype,device=device,requires_grad=requires)
    raw=torch.tensor([[.8,.2],[.3,.7],[.6,.4]],dtype=dtype,device=device,requires_grad=requires)
    types=torch.tensor([0,1,0],dtype=torch.long,device=device)
    edge=torch.tensor([[0,1,1,2,2,0],[1,0,2,1,0,2]],dtype=torch.long,device=device)
    vectors=torch.tensor([[1.,.1,.2],[-1.,-.1,-.2],[.2,1.1,-.1],[-.2,-1.1,.1],[-.8,.3,1.], [.8,-.3,-1.]],dtype=dtype,device=device,requires_grad=requires)
    radial=squared_edge_radial_basis(torch.sum(vectors*vectors,dim=-1),3); cutoff=torch.full((6,),.73,dtype=dtype,device=device)
    return features,raw,types,edge,vectors,radial,cutoff

def _run(model,inputs): return model(*inputs)

def _rotation(dtype):
    q=torch.tensor([[.36,-.48,.8],[.8,.6,0.],[-.48,.64,.6]],dtype=dtype); return q


def test_instruction_modes_coverage_shapes_and_metadata_roundtrip():
    m=_model(); x=_inputs(m); r=_run(m,x)
    assert all(i.connection_mode=="uvw" for i in m.edge_density.edge_tp.instructions)
    assert all(i.connection_mode=="uuu" for i in m.correlations.C2_product.instructions)
    assert all(i.connection_mode=="uvuv" and not i.has_weight for i in m.central_products[0].product.instructions)
    assert {i.i_out for i in m.edge_density.edge_tp.instructions}==set(range(len(m.irreps_A)))
    assert r.edge_weights.shape[-1]==m.edge_density.edge_tp.weight_numel
    assert m.irreps_Z1.dim==m.central.num_channels*m.irreps_C1.dim
    assert HigherBodyConfig.from_dict(m.config.to_dict())==m.config
    assert r.channel_metadata["central"]["vacancy"]==(2,3)
    assert r.channel_metadata["central"]["vacancy_site_type"]==(5,7)


def test_scalar_polynomial_and_exact_central_q_oracle():
    corr=DensityCorrelations("1x0e").double(); corr.C2_product.weight.data.fill_(1); corr.C3_product.weight.data.fill_(1)
    A=torch.tensor([[2.],[3.]],dtype=torch.float64); C1,C2,C3=corr(A)
    torch.testing.assert_close(C2,A.square(),atol=0,rtol=0); torch.testing.assert_close(C3,A.pow(3),atol=0,rtol=0)
    outer=CentralOuterProduct("3x0e","1x0e").double(); q=torch.tensor([[.4],[.7]],dtype=torch.float64,requires_grad=True); central=torch.cat((torch.ones_like(q),torch.full_like(q,.2),q),dim=-1)
    for C,power in ((C1,1),(C2,2),(C3,3)):
        Z=outer(central,C); torch.testing.assert_close(Z[:,-1:],q*A.pow(power),atol=1e-15,rtol=1e-15)
        derivative=torch.autograd.grad(Z[:,-1:].sum(),q,retain_graph=True)[0]; torch.testing.assert_close(derivative,A.pow(power),atol=1e-15,rtol=1e-15)


def test_raw_source_path_pristine_constant_and_embedding_gradients():
    m=_model(); values=list(_inputs(m)); values[0]=torch.zeros_like(values[0]); raw=values[1].clone(); raw[:,1]=torch.tensor([0.,.5,.8],dtype=torch.float64); values[1]=raw
    before=raw.clone(); result=_run(m,tuple(values)); assert torch.linalg.vector_norm(result.A)>0
    assert result.c_raw is raw; torch.testing.assert_close(raw,before,atol=0,rtol=0)
    assert torch.linalg.vector_norm(result.Z1[0])>0
    loss=result.Z1.square().sum()+result.Z2.square().sum(); grad=torch.autograd.grad(loss,m.central.embedding.weight)[0]
    assert torch.all(torch.isfinite(grad)) and torch.linalg.vector_norm(grad)>0
    bad=list(values); bad[2]=torch.tensor([0,2,0]);
    with pytest.raises(ValueError,match="unknown site type"): _run(m,tuple(bad))


@pytest.mark.parametrize("Q",[_rotation(torch.float64),torch.diag(torch.tensor([-1.,1.,1.],dtype=torch.float64))])
def test_all_outputs_o3_equivariant(Q):
    m=_model(); x=_inputs(m); r=_run(m,x)
    feature,raw,types,edge,vectors,radial,cutoff=x
    transformed=(feature@m.irreps_feature.D_from_matrix(Q).T,raw,types,edge,vectors@Q.T,radial,cutoff)
    rt=_run(m,transformed)
    for name,irreps in (("h",m.irreps_h),("edge_sh",m.irreps_sh),("edge_messages",m.irreps_A),("A",m.irreps_A),("C1",m.irreps_C1),("C2",m.irreps_C2),("C3",m.irreps_C3),("Z1",m.irreps_Z1),("Z2",m.irreps_Z2),("Z3",m.irreps_Z3)):
        torch.testing.assert_close(getattr(rt,name),getattr(r,name)@irreps.D_from_matrix(Q).T,atol=5e-9,rtol=2e-7)


def test_edge_order_and_site_permutation_equivariance():
    m=_model(); x=_inputs(m); r=_run(m,x); order=torch.tensor([4,1,5,0,3,2]); xp=list(x); xp[3]=x[3][:,order]; xp[4]=x[4][order]; xp[5]=x[5][order]; xp[6]=x[6][order]; rp=_run(m,tuple(xp))
    for name in ("A","C1","C2","C3","Z1","Z2","Z3"): torch.testing.assert_close(getattr(rp,name),getattr(r,name),atol=2e-14,rtol=2e-14)
    site_order=torch.tensor([2,0,1]); inverse=torch.empty_like(site_order); inverse[site_order]=torch.arange(3); xs=list(x); xs[0]=x[0][site_order]; xs[1]=x[1][site_order]; xs[2]=x[2][site_order]; xs[3]=inverse[x[3]]; rs=_run(m,tuple(xs))
    for name in ("h","A","C1","C2","C3","Z1","Z2","Z3"): torch.testing.assert_close(getattr(rs,name),getattr(r,name)[site_order],atol=3e-14,rtol=3e-14)


def _invariant(result): return sum(getattr(result,n).square().sum()*(i+1)*.01 for i,n in enumerate(("A","C1","C2","C3","Z1","Z2","Z3")))

def test_coordinate_gradcheck_gradgradcheck_and_parameter_mixed_backward():
    m=_model(n=1); base=list(_inputs(m,requires=False)); vectors=base[4].clone().requires_grad_(True)
    def function(v):
        values=list(base); values[4]=v; values[5]=squared_edge_radial_basis(torch.sum(v*v,dim=-1),3); return _invariant(_run(m,tuple(values)))
    assert torch.autograd.gradcheck(function,(vectors,),eps=1e-6,atol=3e-5,rtol=3e-4)
    assert torch.autograd.gradgradcheck(function,(vectors,),eps=1e-6,atol=8e-5,rtol=8e-4)
    energy=function(vectors); force=-torch.autograd.grad(energy,vectors,create_graph=True)[0]; loss=force.square().sum()
    selected=[m.node_encoder.linear.weight,m.edge_density.radial_head.network[0].weight,m.correlations.C2_product.weight,m.correlations.C3_product.weight,m.central.embedding.weight]
    grads=torch.autograd.grad(loss,selected,allow_unused=False)
    assert all(torch.all(torch.isfinite(g)) for g in grads); assert all(torch.linalg.vector_norm(g)>0 for g in grads)


def test_dtype_device_state_roundtrip_determinism_and_scaling():
    for dtype in (torch.float32,torch.float64):
        cpu=_model(dtype=dtype); out=_run(cpu,_inputs(cpu,dtype=dtype)); assert out.Z3.dtype==dtype
        clone=_model(dtype=dtype); clone.load_state_dict(cpu.state_dict()); out2=_run(clone,_inputs(clone,dtype=dtype)); torch.testing.assert_close(out2.Z3,out.Z3,atol=0,rtol=0)
        if torch.cuda.is_available():
            gpu=copy.deepcopy(cpu).cuda()
            cpu_inputs=_inputs(cpu,dtype=dtype)
            gpu_inputs=tuple(value.cuda() if isinstance(value,torch.Tensor) else value for value in cpu_inputs)
            cpu_same=_run(cpu,cpu_inputs); gout=_run(gpu,gpu_inputs); assert gout.Z3.device.type=="cuda"
            if dtype==torch.float64: torch.testing.assert_close(gout.Z3.cpu(),cpu_same.Z3,atol=2e-12,rtol=2e-12)
    counts=[]
    for n in (2,4,8,16):
        model=_model(n=n); counts.append(model.parameter_diagnostics()); assert model.irreps_Z1.dim==model.central.num_channels*model.irreps_A.dim
    assert all(counts[i]["total"]<counts[i+1]["total"] for i in range(3))
    dense=_model(n=2,mode="uvw"); assert dense.parameter_diagnostics()["C2"]>counts[0]["C2"]


def test_no_mace_import_and_invalid_uuu_failfast():
    for path in Path('src/refsite_mlip/interactions').glob('*.py'):
        tree=ast.parse(path.read_text())
        names=[n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)]+[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(name=='mace' or name.startswith('mace.') for name in names)
    with pytest.raises(ValueError,match="identical"):
        DensityCorrelations("1x0e+2x1o","uuu")


def test_high_symmetry_odd_l_cancellation_and_two_fresh_backwards():
    m=_model(n=1); values=list(_inputs(m)); values[0]=torch.zeros_like(values[0]); values[1][:]=torch.tensor([.6,.4]); values[2][:]=0
    values[3]=torch.tensor([[1,1],[0,0]],dtype=torch.long); values[4]=torch.tensor([[1.,0.,0.],[-1.,0.,0.]],dtype=torch.float64); values[5]=squared_edge_radial_basis(torch.ones(2,dtype=torch.float64),3); values[6]=torch.ones(2,dtype=torch.float64)
    result=_run(m,tuple(values)); l1=m.irreps_A.slices()[1]; torch.testing.assert_close(result.A[0,l1],torch.zeros(3,dtype=torch.float64),atol=2e-15,rtol=0)
    for _ in range(2):
        fresh=list(_inputs(m)); fresh[4]=fresh[4].clone().requires_grad_(True); fresh[5]=squared_edge_radial_basis(torch.sum(fresh[4]*fresh[4],dim=-1),3); loss=_invariant(_run(m,tuple(fresh))); loss.backward(); assert torch.all(torch.isfinite(fresh[4].grad))
