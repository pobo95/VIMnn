from __future__ import annotations
import copy
import pytest
import torch
from refsite_mlip.data import StructureSample
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import PotentialConfig,ReferenceSitePotential
from refsite_mlip.training import AtomicBaselineConfig,apply_atomic_baseline_,fit_atomic_baseline
from refsite_mlip.transport import TRAIN_FIXED,EVAL_ADAPTIVE


def _numbers(data,n=5): return torch.tensor([6 if int(x)==0 else 41 for x in data['site_types'][:n]],dtype=torch.long,device=data['positions'].device)
def _model(data,baseline=(-1.,2.),layers=1):
    tolerance=1e-6 if data['cell'].dtype==torch.float32 else 1e-7; feature=ProbabilityMultipoleConfig((6,41),2,2,1.,3.,tolerance,site_type_vocabulary=(0,1)); irreps='2x0e+4x0e+4x1o+4x2e'; higher=HigherBodyConfig(irreps,2,2,2,1,2,3,(4,),6.,3.,1.); config=PotentialConfig((6,41),layers,feature,higher,8,1.)
    top=build_reference_graph_topology(data['sites'],data['site_types'],data['cell'],cutoff=3.,skin=.5,maximum_strain=.1)
    return ReferenceSitePotential(config,top,data['modes'],data['mode_weights'],torch.eye(2,dtype=data['cell'].dtype,device=data['cell'].device),data['site_weights'],data['channel_weights'],baseline).to(data['cell'])
def _rotation(dtype):
    return torch.tensor([[.36,-.48,.8],[.8,.6,0.],[-.48,.64,.6]],dtype=dtype)
def _rotated(data,Q):
    result=dict(data); result['positions']=data['positions']@Q.T; result['origin']=data['origin']@Q.T; result['cell']=data['cell']@Q.T; return result


def test_forward_shape_site_sum_baseline_and_zero_default(typed_crystal):
    m=_model(typed_crystal); p=typed_crystal['positions'][:5]; z=_numbers(typed_crystal); out=m(p,z,typed_crystal['cell'],typed_crystal['origin'],return_aux=True)
    assert out.energy.shape==() and out.site_energy.shape==(6,) and out.site_features.dtype==p.dtype
    torch.testing.assert_close(out.residual_energy,out.site_energy.sum(),atol=0,rtol=0); torch.testing.assert_close(out.energy,out.site_energy.sum()+torch.tensor(1.,dtype=p.dtype),atol=0,rtol=0)
    zero=_model(typed_crystal,baseline=None); assert torch.count_nonzero(zero.atomic_baseline)==0 and not zero.atomic_baseline.requires_grad
    assert out.raw_c is out.auxiliary['multipoles'].raw_probability_state


def test_atom_permutation_translation_and_forces(typed_crystal):
    m=_model(typed_crystal); p=typed_crystal['positions'][:5].clone().requires_grad_(True); z=_numbers(typed_crystal); out=m(p,z,typed_crystal['cell'],typed_crystal['origin'],compute_forces=True)
    order=torch.tensor([3,0,4,1,2]); perm=m(p.detach()[order].requires_grad_(True),z[order],typed_crystal['cell'],typed_crystal['origin'],compute_forces=True)
    torch.testing.assert_close(perm.energy,out.energy,atol=2e-10,rtol=2e-10); torch.testing.assert_close(perm.forces,out.forces[order],atol=2e-8,rtol=2e-8)
    shift=torch.tensor([.7,-.3,.9],dtype=p.dtype); moved=m((p.detach()+shift).requires_grad_(True),z,typed_crystal['cell'],typed_crystal['origin']+shift,compute_forces=True)
    torch.testing.assert_close(moved.energy,out.energy,atol=2e-10,rtol=2e-10); torch.testing.assert_close(moved.forces,out.forces,atol=2e-8,rtol=2e-8); torch.testing.assert_close(out.forces.sum(0),torch.zeros(3,dtype=p.dtype),atol=2e-8,rtol=0)

@pytest.mark.parametrize('Q',[_rotation(torch.float64),torch.diag(torch.tensor([-1.,1.,1.],dtype=torch.float64))])
def test_o3_energy_and_force_covariance(typed_crystal,Q):
    base=_model(typed_crystal); p=typed_crystal['positions'][:5].clone().requires_grad_(True); z=_numbers(typed_crystal); out=base(p,z,typed_crystal['cell'],typed_crystal['origin'],compute_forces=True)
    data=_rotated(typed_crystal,Q); rotated=_model(data); rotated.load_state_dict(base.state_dict()); rp=data['positions'][:5].clone().requires_grad_(True); rout=rotated(rp,z,data['cell'],data['origin'],compute_forces=True)
    torch.testing.assert_close(rout.energy,out.energy,atol=3e-8,rtol=3e-8); torch.testing.assert_close(rout.forces,out.forces@Q.T,atol=3e-6,rtol=3e-6)


def test_force_finite_difference_and_double_backward(typed_crystal):
    m=_model(typed_crystal); p=typed_crystal['positions'][:5].clone().requires_grad_(True); z=_numbers(typed_crystal); out=m(p,z,typed_crystal['cell'],typed_crystal['origin'],compute_forces=True,create_graph=True); h=2e-6; d=torch.zeros_like(p); d[2,1]=h
    plus=m(p.detach()+d,z,typed_crystal['cell'],typed_crystal['origin']).energy; minus=m(p.detach()-d,z,typed_crystal['cell'],typed_crystal['origin']).energy
    torch.testing.assert_close(out.forces[2,1],-(plus-minus)/(2*h),atol=5e-6,rtol=5e-5)
    grads=torch.autograd.grad(out.forces.square().sum(),[m.readout.mlp[-1].weight,m.layers[0].corr.C2_product.weight,m.central.embedding.weight]); assert all(torch.all(torch.isfinite(g)) for g in grads)


def _strain_energy(model,data,z,strain):
    F=torch.eye(3,dtype=strain.dtype,device=strain.device)+strain; return model(data['positions'][:5]@F,z,data['cell']@F,data['origin']@F).energy

def test_stress_symmetric_voigt_and_finite_difference(typed_crystal):
    m=_model(typed_crystal); p=typed_crystal['positions'][:5].clone().requires_grad_(True); z=_numbers(typed_crystal); out=m(p,z,typed_crystal['cell'],typed_crystal['origin'],compute_stress=True); assert out.stress_voigt.shape==(6,); torch.testing.assert_close(out.stress,out.stress.T,atol=0,rtol=0)
    strain=torch.zeros((3,3),dtype=torch.float64,requires_grad=True); derivative=torch.autograd.grad(_strain_energy(m,typed_crystal,z,strain),strain)[0]; h=2e-6
    dirs=[]
    for i in range(3): d=torch.zeros((3,3),dtype=torch.float64); d[i,i]=1; dirs.append(d)
    for i,j in ((1,2),(0,2),(0,1)): d=torch.zeros((3,3),dtype=torch.float64); d[i,j]=d[j,i]=.5; dirs.append(d)
    for d in dirs:
        fd=(_strain_energy(m,typed_crystal,z,h*d)-_strain_energy(m,typed_crystal,z,-h*d))/(2*h); torch.testing.assert_close(torch.sum(derivative*d),fd,atol=5e-6,rtol=5e-5)
    torch.testing.assert_close(out.stress,0.5*(derivative+derivative.T)/torch.linalg.det(typed_crystal['cell']).abs(),atol=3e-12,rtol=3e-12)


def test_eval_adaptive_semantic_migration_is_explicit(typed_crystal):
    """7A-2 replaces implicit fixed-phase evaluation with a bound policy."""

    model=_model(typed_crystal); p=typed_crystal['positions'][:5].clone().requires_grad_(True); z=_numbers(typed_crystal)
    with pytest.raises(ValueError,match='evaluation_policy'):
        model(p,z,typed_crystal['cell'],typed_crystal['origin'],solver_path=EVAL_ADAPTIVE)
    with pytest.raises(ValueError,match='energy-only'):
        model(p,z,typed_crystal['cell'],typed_crystal['origin'],solver_path=EVAL_ADAPTIVE,compute_forces=True)


def test_apply_fitted_atomic_baseline_preserves_model_state_contract(typed_crystal,tmp_path):
    def sample(sample_id,numbers,energy):
        count=len(numbers); dtype=torch.float64
        return StructureSample(sample_id,torch.zeros((count,3),dtype=dtype),torch.tensor(numbers,dtype=torch.long),torch.eye(3,dtype=dtype),torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=dtype),'template',energy=torch.tensor(energy,dtype=dtype))
    dataset=(sample('carbon',(6,),-1.5),sample('niobium',(41,),2.25),sample('mixed',(6,6,41),-0.75))
    fit=fit_atomic_baseline(dataset,range(3),(6,41),AtomicBaselineConfig())
    model=_model(typed_crystal,baseline=None); parameter_ids=tuple(id(value) for value in model.parameters()); parameter_count=sum(value.numel() for value in model.parameters()); state_keys=tuple(model.state_dict()); baseline_id=id(model.atomic_baseline)
    returned=apply_atomic_baseline_(model,fit)
    assert returned is model and id(model.atomic_baseline)==baseline_id and tuple(model.state_dict())==state_keys
    assert tuple(id(value) for value in model.parameters())==parameter_ids and sum(value.numel() for value in model.parameters())==parameter_count
    assert not model.atomic_baseline.requires_grad and 'atomic_baseline' in model._buffers
    torch.testing.assert_close(model.atomic_baseline,torch.tensor([-1.5,2.25],dtype=torch.float64),atol=2e-15,rtol=2e-15)
    z=_numbers(typed_crystal); output=model(typed_crystal['positions'][:5],z,typed_crystal['cell'],typed_crystal['origin']); indices=model._species_indices(z); expected=model.atomic_baseline[indices].sum()
    torch.testing.assert_close(output.baseline_energy,expected,atol=0,rtol=0); torch.testing.assert_close(output.residual_energy,output.site_energy.sum(),atol=0,rtol=0); torch.testing.assert_close(output.energy,output.baseline_energy+output.residual_energy,atol=0,rtol=0)
    path=tmp_path/'baseline-state.pt'; torch.save(model.state_dict(),path); loaded=torch.load(path,weights_only=True); clone=_model(typed_crystal,baseline=None); result=clone.load_state_dict(loaded,strict=True)
    assert result.missing_keys==[] and result.unexpected_keys==[] and torch.equal(clone.atomic_baseline,model.atomic_baseline)
    reversed_fit=fit_atomic_baseline(dataset,range(3),(41,6),AtomicBaselineConfig())
    with pytest.raises(ValueError,match='vocabulary/order'):
        apply_atomic_baseline_(model,reversed_fit)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_cpu_and_cuda_dtype_device(typed_crystal,dtype):
    data={k:(v.to(dtype=dtype) if isinstance(v,torch.Tensor) and v.is_floating_point() else v) for k,v in typed_crystal.items()}; cpu=_model(data); z=_numbers(data,6); out=cpu(data['positions'][:6],z,data['cell'],data['origin']); assert out.energy.dtype==dtype
    if torch.cuda.is_available():
        gpu_data={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in data.items()}; gpu=copy.deepcopy(cpu).cuda(); gout=gpu(gpu_data['positions'][:6],z.cuda(),gpu_data['cell'],gpu_data['origin']); assert gout.energy.device.type=='cuda'; torch.testing.assert_close(gout.energy.cpu(),out.energy,atol=3e-5 if dtype==torch.float32 else 3e-11,rtol=3e-5 if dtype==torch.float32 else 3e-11)
