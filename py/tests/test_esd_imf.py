"""Independent equations, gradient isolation, and CLI guards for boundary iMF."""
from types import SimpleNamespace
import pytest
import torch
from common import flow_map, interpolant, losses
from _equiv_utils import build_network_config, build_problem_config, perturb_params_
from test_stopped_flowmap import PolynomialMap, batch, heads, snapshots


def interp():return interpolant.setup_interpolant(build_problem_config())


def analytic_target(net,x0,x1,s,t):
    x=(1-s[:,None,None])*x0+s[:,None,None]*x1
    ss=s[:,None,None];gap=(t-s)[:,None,None];p=net.p.detach()
    a=heads({k:v.detach() for k,v in net.named_parameters()},s,s,x,False)[0]
    partial=p[1]*x.square()-2*p[2]*gap+p[3]*(gap-ss)
    return ((x1-x0)+gap*(partial+(p[0]+2*p[1]*ss*x)*a)).detach(),x


@pytest.mark.parametrize('weighted',[False,True])
def test_imf_equation_and_parameter_gradient_stop_tangent_and_boundary(weighted,monkeypatch):
    net=PolynomialMap(density=False,weighted=weighted);net.config.divergence_mode='pretrain_flowmap'
    x0,x1,s,t,h=batch();teacher,ema=snapshots(net)
    backwards=[];directions=[];original=losses._raw_jvp
    def record(net,ss,tt,x,direction):
        directions.append(direction);primal,tangent=original(net,ss,tt,x,direction)
        tangent.register_hook(lambda g:backwards.append(g));return primal,tangent
    monkeypatch.setattr(losses,'_raw_jvp',record)
    value,metrics=losses.esd_imf_term(net,teacher,x0,x1,None,s,t,interp=interp(),density_ema_params=ema)
    target,x=analytic_target(net,x0,x1,s,t)
    prediction=net.calc_phi(s,t,x);q=(prediction-target).square().sum((1,2));w=net.calc_weight(s,t)
    expected=torch.exp(-w)*q+w if torch.is_tensor(w) else q
    torch.testing.assert_close(value,expected)
    actual=torch.autograd.grad(value.sum(),(net.p,net.w,*teacher.values(),*ema.values()),allow_unused=True)
    reference=torch.autograd.grad(expected.sum(),(net.p,net.w),allow_unused=True)
    torch.testing.assert_close(actual[0],reference[0]);assert torch.count_nonzero(actual[0])>0
    if weighted:torch.testing.assert_close(actual[1],reference[1])
    else:assert actual[1] is None
    assert all(g is None for g in actual[2:]) and backwards==[]
    assert not directions[0].requires_grad
    torch.testing.assert_close(metrics['esd_imf/esd_imf_v_loss'],q.detach())
    changed,_=losses.esd_imf_term(net,{k:v+123 for k,v in teacher.items()},x0,x1,None,s,t,interp=interp())
    torch.testing.assert_close(value,changed)


def test_imf_diagonal_reduces_to_live_conditional_fm():
    net=PolynomialMap(density=False);net.config.divergence_mode='pretrain_flowmap'
    x0,x1,s,t,h=batch()
    imf,_=losses.esd_imf_term(net,None,x0,x1,None,s,s,interp=interp())
    fm,_=losses.diagonal_term(net,None,x0,x1,None,s,interp=interp())
    torch.testing.assert_close(imf,fm)
    g1,=torch.autograd.grad(imf.sum(),net.p);g2,=torch.autograd.grad(fm.sum(),net.p)
    torch.testing.assert_close(g1,g2);assert torch.count_nonzero(g1)>0


def test_imf_differs_from_conditional_meanflow_when_directions_differ():
    net=PolynomialMap(density=False);net.config.divergence_mode='pretrain_flowmap'
    x0,x1,s,t,h=batch()
    imf,_=losses.esd_imf_term(net,None,x0,x1,None,s,t,interp=interp())
    mf,_=losses.esd_term(net,None,x0,x1,None,s,t,interp=interp(),loss_type='esd_cond')
    torch.testing.assert_close(imf[0],mf[0]);assert not torch.allclose(imf[2:],mf[2:])


def test_imf_assembly_keeps_conditional_diagonal_even_if_teacher_is_passed():
    net=PolynomialMap(density=False);net.config.divergence_mode='pretrain_flowmap'
    cfg=SimpleNamespace(training=SimpleNamespace(loss_type='esd_imf',stopgrad_type='stopped',diag_teacher_source='conditional'),optimization=SimpleNamespace(diag_fraction=.5))
    x0,x1,s,t,h=batch();teacher,ema=snapshots(net);s=s.clone();s[:2]=t[:2]
    loss=losses.setup_loss(cfg,net,interp());args=(x0,x1,None,s,t,None,None)
    a,metrics=loss(teacher,teacher,*args);b,_=loss(None,None,*args);torch.testing.assert_close(a,b)
    target0=(x1[:2]-x0[:2]).detach();x=(1-t[:2,None,None])*x0[:2]+t[:2,None,None]*x1[:2]
    diagonal=(net.calc_b(t[:2],x)-target0).square().sum()/4
    target1,xoff=analytic_target(net,x0[2:],x1[2:],s[2:],t[2:])
    off=(net.calc_phi(s[2:],t[2:],xoff)-target1).square().sum()/4
    torch.testing.assert_close(a,diagonal+off);torch.testing.assert_close(metrics['diag/velocity_loss'],diagonal.detach())


def test_real_dit_imf_gradient_matches_independent_frozen_regression():
    cfg=build_network_config(n_particles=2,d_model=16,num_layers=1,nhead=2)
    cfg.predict_divergence=False;cfg.divergence_mode='pretrain_flowmap'
    net=perturb_params_(flow_map.TorchFlowMap(cfg),std=.03,seed=21)
    x0,x1,s,t,h=batch();x0,x1,s,t=[x.float() for x in (x0,x1,s,t)]
    x=(1-s[:,None,None])*x0+s[:,None,None]*x1
    with torch.no_grad():a=net.calc_b(s,x)
    _,derivative=torch.func.jvp(lambda xx,ss:net.calc_phi(ss,t,xx,train=False),(x,s),(a,torch.ones_like(s)))
    target=(x1-x0+(t-s)[:,None,None]*derivative).detach()
    value,_=losses.esd_imf_term(net,None,x0,x1,None,s,t,interp=interp())
    actual=torch.autograd.grad(value.sum(),tuple(net.parameters()),allow_unused=True)
    expected=(net.calc_phi(s,t,x,train=False)-target).square().sum()
    reference=torch.autograd.grad(expected,tuple(net.parameters()),allow_unused=True)
    for a,b in zip(actual,reference):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-4)


@pytest.mark.parametrize('mode',['pretrain_v','hutchinson','dlogpdt'])
def test_imf_rejects_unsupported_modes_in_cli(mode,monkeypatch):
    from launchers import learn
    monkeypatch.setattr('sys.argv',['learn.py','--cfg_path','configs.ace_a_nme','--loss_type','esd_imf','--mode',mode])
    with pytest.raises(ValueError,match='velocity-only pretrain_flowmap'):learn.setup_config_dict(learn.parse_command_line_arguments())


@pytest.mark.parametrize('flag,value,match',[('--diag_teacher_source','external','conditional diagonal'),('--offdiag_teacher_source','ema','student boundary'),('--teacher_load_path','fake.pt','external teacher')])
def test_imf_rejects_conflicting_teacher_settings(flag,value,match,monkeypatch):
    from launchers import learn
    monkeypatch.setattr('sys.argv',['learn.py','--cfg_path','configs.ace_a_nme','--loss_type','esd_imf','--mode','pretrain_flowmap',flag,value])
    with pytest.raises(ValueError,match=match):learn.setup_config_dict(learn.parse_command_line_arguments())


def test_imf_cli_and_forward_option_resolve_without_changing_other_losses(monkeypatch):
    from launchers import learn
    for family in ['lsd','esd','esd_cond','esd_imf']:
        monkeypatch.setattr('sys.argv',['learn.py','--cfg_path','configs.ace_a_nme','--loss_type',family,'--mode','pretrain_flowmap','--offdiag_time_order','forward'])
        cfg=learn.setup_config_dict(learn.parse_command_line_arguments())
        assert cfg.training.loss_type==family and cfg.training.offdiag_time_order=='forward'
        if family=='esd_imf':
            assert cfg.training.diag_teacher_source=='conditional' and cfg.training.offdiag_teacher_source=='self' and not cfg.teacher.load_path
        else:assert cfg.training.diag_teacher_source=='external'


def test_imf_direct_guards_density_nonlinear_and_wrong_anchor():
    x0,x1,s,t,h=batch();net=PolynomialMap(density=True)
    with pytest.raises(ValueError,match='velocity-only'):losses.esd_imf_term(net,None,x0,x1,None,s,t,interp=interp())
    net.config.predict_divergence=False;net.config.divergence_mode='pretrain_flowmap'
    with pytest.raises(ValueError,match='linear interpolant'):losses.esd_imf_term(net,None,x0,x1,None,s,t,interp=SimpleNamespace(kind='nonlinear'))
    cfg=SimpleNamespace(training=SimpleNamespace(loss_type='esd_imf',stopgrad_type='stopped',diag_teacher_source='external'))
    with pytest.raises(ValueError,match='conditional diagonal'):losses.setup_loss(cfg,net,interp())
