"""Independent trace/continuity, normalization, sampling, and EMA regressions."""
from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch

from common import flow_map, interpolant, loss_args, losses, state_utils, updates
from common.se3_utils import remove_mean
from _equiv_utils import build_network_config, build_problem_config
from test_stopped_flowmap import PolynomialMap

DEVICE=os.environ.get('IF2D2_TEST_DEVICE','cpu')


class Drift(torch.nn.Module):
    def __init__(self, matrix, com_aware, vectorized, mode='hutchinson', sigma=1.3):
        super().__init__()
        self.matrix=torch.nn.Parameter(matrix.clone())
        self.config=SimpleNamespace(com_aware=com_aware,vectorized_divergence=vectorized,
                                    divergence_mode=mode,rescale=sigma)
    def forward(self,t,x,method='calc_b',**kwargs):
        return (x.reshape(x.shape[0],-1)@self.matrix.T).reshape_as(x)


def geometry(full):
    eye=torch.eye(6,dtype=torch.float64,device=DEVICE)
    P=eye if full else torch.kron(torch.eye(2,dtype=eye.dtype,device=DEVICE)-.5,torch.eye(3,dtype=eye.dtype,device=DEVICE))
    eig,Q=torch.linalg.eigh(P)
    return P,Q[:,eig>.5]


@pytest.mark.parametrize('full',[False,True])
@pytest.mark.parametrize('vectorized',[False,True])
def test_hutchinson_basis_is_exact_restricted_trace_and_stopped(full,vectorized):
    P,Q=geometry(full)
    raw=torch.arange(36,dtype=P.dtype,device=DEVICE).reshape(6,6)/30
    A=P@(raw+.7*torch.eye(6,dtype=P.dtype,device=DEVICE))@P
    net=Drift(A,full,vectorized).to(DEVICE)
    x=torch.linspace(-.5,.9,6,dtype=P.dtype,device=DEVICE).reshape(1,2,3)
    t=x.new_tensor([.4])
    estimates=[]
    for direction in Q.T:
        # Average probes with covariance P, so this finite quadrature is exact.
        probe=(Q.shape[1]**.5*direction).reshape_as(x)
        phi,d=losses._hutchinson_divergence_with_phi(net,None,t,x,eps=probe)
        assert not phi.requires_grad and not d.requires_grad
        estimates.append(d)
    actual=torch.stack(estimates).mean(0)
    expected=A.diag().reshape_as(x) if vectorized else A.trace().reshape(1)
    torch.testing.assert_close(actual,expected,atol=2e-14,rtol=2e-14)
    assert net.matrix.grad is None


@pytest.mark.parametrize('full',[False,True])
@pytest.mark.parametrize('vectorized',[False,True])
@pytest.mark.parametrize('s',[.2,.65,.95])
def test_dlogpdt_gaussian_conditional_expectation_and_approximate_drift_limit(full,vectorized,s):
    P,Q=geometry(full)
    m=Q.shape[1]
    sigma0,sigma1=1.3,.8
    V=(1-s)**2*sigma0**2+s*s*sigma1**2
    k=(s*sigma1**2-(1-s)*sigma0**2)/V
    x=P@torch.linspace(-.7,.9,6,dtype=P.dtype,device=DEVICE)
    mu=(1-s)*sigma0**2/V*x
    variance=sigma0**2*s*s*sigma1**2/V
    # Exact second-moment quadrature of z | x_s=x on the modeled subspace.
    z=torch.cat([mu+(m*variance)**.5*Q.T,mu-(m*variance)**.5*Q.T]).reshape(2*m,2,3)
    xt=x.reshape(1,2,3).expand_as(z)
    data=(xt-(1-s)*z)/s
    times=x.new_full((2*m,),s)
    def estimate(offset):
        net=Drift((k+offset)*P,full,vectorized,mode='dlogpdt',sigma=sigma0)
        v,q=losses._calc_divergence_with_phi(net,None,times,xt,z,data)
        assert not v.requires_grad and not q.requires_grad
        return q.mean(0)
    expected=(k*P.diag()).reshape(2,3) if vectorized else k*m
    torch.testing.assert_close(estimate(0),torch.as_tensor(expected,device=DEVICE,dtype=x.dtype),atol=1e-11,rtol=1e-11)
    beta=.2
    biased=k*P.diag()+beta*x.square()/V
    expected_biased=biased.reshape(2,3) if vectorized else biased.sum()
    torch.testing.assert_close(estimate(beta),expected_biased,atol=1e-11,rtol=1e-11)
    # The continuity estimator with an approximate field is NOT its trace.
    trace_approx=((k+beta)*P.diag()).reshape(2,3) if vectorized else (k+beta)*m
    assert not torch.allclose(estimate(beta),torch.as_tensor(trace_approx,device=DEVICE,dtype=x.dtype))


def test_dlogpdt_rejects_singular_times_stale_labels_and_non_linear_interpolants():
    P,_=geometry(True)
    net=Drift(P,True,False,'dlogpdt')
    z=torch.zeros(1,2,3,dtype=P.dtype,device=DEVICE)
    data=torch.ones_like(z)
    for bad in (1.,1.1,-.1,float('nan')):
        with pytest.raises(ValueError,match='diagonal times'):
            losses._calc_divergence_with_phi(net,None,z.new_tensor([bad]),z,z,data)
    with pytest.raises(ValueError,match='original linear-interpolant'):
        losses._calc_divergence_with_phi(net,None,z.new_tensor([.3]),z,z,data)
    model=PolynomialMap().to(DEVICE)
    with pytest.raises(ValueError,match='linear interpolant'):
        losses.diagonal_term(model,None,z,data,None,z.new_tensor([.3]),
                             interp=interpolant.setup_interpolant(build_problem_config('trig')))


@pytest.mark.parametrize('weight',['uniform','linear','square'])
@pytest.mark.parametrize('vectorized',[False,True])
@pytest.mark.parametrize('learned_weight', [False, True])
def test_diagonal_density_factorization_weighting_and_scale(weight,vectorized,learned_weight):
    net=PolynomialMap(vectorized=vectorized,weighted=learned_weight).to(DEVICE)
    net.config.div_loss_weighting_type=weight
    x=torch.full((3,2,3),.2,dtype=torch.float64,device=DEVICE)
    t=x.new_tensor([.1,.5,.99])
    z=x-.4; data=x+.6
    pred=torch.full_like(x,.7) if vectorized else t+.7
    target=torch.full_like(pred,-.2)
    factor={'uniform':torch.ones_like(t),'linear':1-t,'square':(1-t)**2}[weight]
    raw = pred.square()-2*pred*target if weight == 'linear' else (pred-target).square()
    expected=raw.reshape(3,-1).sum(1)*factor/(6 if vectorized else 1)
    torch.testing.assert_close(losses._calc_div_loss(net,pred,target,t),expected)
    # Production diagonal loss uses the stopped, scaled estimator and ordinary student heads.
    xt=(1-t[:,None,None])*z+t[:,None,None]*data
    with torch.no_grad():
        _,q=losses._calc_divergence_with_phi(net,None,t,xt,z,data)
        q=q/net.config.divergence_rescale
    lv,metrics=losses.diagonal_term(net,None,z,data,None,t,
        interp=interpolant.setup_interpolant(build_problem_config()))
    u,d=net.calc_b(t,xt,return_div=True)
    raw_d = d.square()-2*d*q if weight == 'linear' else (d-q).square()
    div_expected = raw_d.reshape(3,-1).sum(1)*factor/(6 if vectorized else 1)
    expected=(u-(data-z)).square().sum((1,2))+div_expected
    if learned_weight:
        w = net.calc_weight(t,t)
        expected = torch.exp(-w)*expected+w
    torch.testing.assert_close(lv,expected)
    torch.testing.assert_close(metrics['diag/div_loss'],div_expected)
    params = (net.p,net.q,net.w) if learned_weight else (net.p,net.q)
    g=torch.autograd.grad(lv.sum(),params)
    ref=torch.autograd.grad(expected.sum(),params)
    for a,b in zip(g,ref):
        torch.testing.assert_close(a,b)


@pytest.mark.parametrize('distribution',['uniform','linear','square'])
def test_diagonal_time_sampling_respects_both_bounds_without_target_clamp(distribution):
    gen=torch.Generator().manual_seed(21)
    lo,hi=.23,.994
    t,_=loss_args._sample_diagonal(gen,10000,lo,hi,distribution)
    assert lo<=float(t.min())<lo+.02
    assert hi-.02<float(t.max())<hi
    # Check the inverse-CDF distribution, not just the endpoints.
    if distribution=='linear':
        uniform=(torch.log1p(-t)-torch.log(torch.tensor(1-lo)))/(torch.log(torch.tensor(1-hi))-torch.log(torch.tensor(1-lo)))
    elif distribution=='square':
        uniform=(1/(1-t)-1/(1-lo))/(1/(1-hi)-1/(1-lo))
    else:
        uniform=(t-lo)/(hi-lo)
    assert abs(float(uniform.mean())-.5)<.015
    cfg=SimpleNamespace(optimization=SimpleNamespace(bs=10000,diag_fraction=.5),
        network=SimpleNamespace(divergence_mode='dlogpdt'),
        training=SimpleNamespace(tmin=0.,tmax=1.,diag_tmax=.8,time_distr='uniform',loss_type='lsd',psd_type=None))
    s,t,_,_=loss_args.sample_times(cfg,gen)
    assert t[:5000].max()<.8 and t[5000:].max()>.99


@pytest.mark.parametrize('full',[False,True])
def test_density_accumulates_gap_scale_and_sign_once(full):
    cfg=build_network_config(n_particles=2,d_model=16,num_layers=1,nhead=2,divergence_rescale=2.5)
    cfg.com_aware=full
    net=flow_map.TorchFlowMap(cfg).to(DEVICE)
    cfg.unscale_factor=1.7
    # Constant nonzero raw d; total physical accumulated divergence is scale*d.
    def constant(s,t,x,**kw):
        return torch.zeros_like(x),x.new_full((x.shape[0],),.4)
    net.calc_phi=constant
    x=torch.zeros(4,2,3,device=DEVICE)
    for steps in (1,3):
        _,forward=net._integrate(.2,.8,x,n_steps=steps)
        _,reverse=net._integrate(.8,.2,x,n_steps=steps)
        torch.testing.assert_close(forward,x.new_full((4,),-.6))
        torch.testing.assert_close(reverse,-forward)
    dimension=6 if full else 3
    _,logp=net.unscale_samples(x,torch.zeros(4,device=DEVICE))
    torch.testing.assert_close(logp,x.new_full((4,),-dimension*torch.log(torch.tensor(1.7)).item()))
    _,restored=net.rescale_samples(x,logp)
    torch.testing.assert_close(restored,torch.zeros_like(restored))
    assert net.calc_weight(torch.zeros(4),torch.ones(4))==0


def test_partial_ema_restore_uses_loaded_student_and_does_not_alias_checkpoint():
    net=torch.nn.Linear(2,2).to(DEVICE)
    net.register_buffer('fixed',torch.tensor([3.],device=DEVICE),persistent=False)
    state=state_utils.TrainState(net, torch.optim.SGD(net.parameters(), lr=0.1), lambda _: 0.1, [0.9])
    sd=deepcopy(state.state_dict())
    sd['params']['weight'].fill_(7.)
    sd['params']['bias'].fill_(4.)
    # A compatible factor with one missing entry used to keep random pre-load weights.
    sd['ema_params'][.9]={'bias':sd['params']['bias']+1}
    sd['ema_params'][.8]={'bias':sd['params']['bias']+2}
    state.load_state_dict(sd,load_optimizer=False)
    with pytest.raises(ValueError,match="not updated"):
        state.ema_teacher(.8)
    torch.testing.assert_close(state.ema_teacher()['weight'],torch.full_like(net.weight,7.))
    torch.testing.assert_close(state.ema_teacher()['bias'],torch.full_like(net.bias,5.))
    torch.testing.assert_close(state.ema_teacher()['fixed'],net.fixed)
    sd['ema_params'][.9]['bias'].zero_()
    assert state.ema_teacher()['bias'].min()==5
    before=deepcopy(state.ema_params)
    state.update_ema()
    torch.testing.assert_close(state.ema_teacher()['bias'],torch.full_like(net.bias,4.9))
    assert all(not p.requires_grad for p in state.ema_teacher().values())


def test_train_step_passes_independent_ema_before_update():
    net=PolynomialMap().to(DEVICE)
    cfg=SimpleNamespace(network=net.config,training=SimpleNamespace(density_ema_fac=.9),optimization=SimpleNamespace(clip=10.))
    state=state_utils.TrainState(net, torch.optim.SGD(net.parameters(), lr=0.01), lambda _: 0.01, [0.9])
    with torch.no_grad(): net.q.add_(.5)
    original=state.ema_teacher()['q'].clone()
    observed=[]
    def fn(td,to,*args,density_ema_params,generator=None):
        observed.append(density_ema_params['q'].clone())
        assert not torch.equal(density_ema_params['q'],to['q'])
        return net.q.square().sum(),{}
    dummy=(torch.zeros(1,device=DEVICE),)*2+(None,torch.zeros(1,device=DEVICE),torch.zeros(1,device=DEVICE),None,None)
    updates.setup_train_step(cfg)(state,fn,state.self_teacher(),state.self_teacher(),dummy,torch.Generator(device=DEVICE))
    torch.testing.assert_close(observed[0],original)
    torch.testing.assert_close(state.ema_teacher()['q'],original)
    updates.setup_ema_update(cfg)(state)
    assert not torch.equal(state.ema_teacher()['q'],original)
