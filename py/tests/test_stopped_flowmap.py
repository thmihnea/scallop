"""Production loss values/gradients against independent, explicit frozen targets.

These tests intentionally do not treat the historical JAX gradients as an oracle.
Set IF2D2_TEST_DEVICE=cuda and CUDA_VISIBLE_DEVICES to exercise a local GPU.
"""
from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch

from common import flow_map, interpolant, losses, state_utils, updates
from _equiv_utils import build_network_config, build_problem_config, perturb_params_

DEVICE = os.environ.get('IF2D2_TEST_DEVICE', 'cpu')


class PolynomialMap(flow_map.TorchFlowMap):
    """Nonlinear raw heads using the real flow_call and method dispatcher."""
    def __init__(self, vectorized=False, density=True, weighted=False):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            predict_divergence=density, vectorized_divergence=vectorized,
            input_dims=(2, 3), n_particles=2, n_dimension=3, divergence_rescale=2.3,
            rescale=1.7, unscale_factor=2.0, com_aware=True,
            divergence_mode='dlogpdt', div_loss_weighting_type='uniform', from_scratch=False,
        )
        self.use_weight = weighted
        self.p = torch.nn.Parameter(torch.tensor([.3, -.2, .4, .13], dtype=torch.float64))
        self.q = torch.nn.Parameter(torch.tensor([-.15, .21, .33, -.24], dtype=torch.float64))
        self.w = torch.nn.Parameter(torch.tensor(.17, dtype=torch.float64))

    def calc_phi(self, s, t, x, train=False, return_div=False, **kwargs):
        u, d = heads(dict(self.named_parameters()), s, t, x, self.config.vectorized_divergence)
        return (u, d) if return_div and self.config.predict_divergence else u

    def calc_weight(self, s, t):
        return self.w * (s + .3 * t) if self.use_weight else 0.0


def heads(params, s, t, x, vectorized):
    p, q = params['p'], params['q']
    s, t = s[:, None, None], t[:, None, None]
    gap = t - s
    u = p[0]*x + p[1]*s*x.square() + p[2]*gap.square() + p[3]*s*gap
    d = q[0]*x.square() + q[1]*s*x + q[2]*gap + q[3]*t*gap
    return u, d if vectorized else d.sum((1, 2))


def batch():
    x0 = torch.linspace(-.8, .9, 24, dtype=torch.float64, device=DEVICE).reshape(4, 2, 3)
    x1 = torch.cos(2 * x0) * .7
    # Diagonal, tiny gap, forward and reverse finite gaps.
    s = x0.new_tensor([.3, .4, .15, .8])
    t = x0.new_tensor([.3, .40000001, .8, .1])
    h = x0.new_tensor([.2, .35, .7, .8])
    return x0, x1, s, t, h


def snapshots(net):
    teacher = {k: (v.detach().clone() + .31).requires_grad_() for k, v in net.named_parameters()}
    ema = {k: (v.detach().clone() - .19).requires_grad_() for k, v in net.named_parameters()}
    return teacher, ema


def reference_targets(net, family, source, teacher, ema, x, x0, x1, s, t, h):
    vec = net.config.vectorized_divergence
    p, q = net.p.detach(), net.q.detach()
    ss, tt = s[:, None, None], t[:, None, None]
    gap = tt - ss
    with torch.no_grad():
        u, d = heads(dict(net.named_parameters()), s, t, x, vec)
        if family == 'lsd':
            y = x + gap * u
            b = heads(teacher, t, t, y, vec)[0]
            g = heads(ema, t, t, y, vec)[1]
            du = (2*p[2]*gap + p[3]*ss).expand_as(x)
            dd = (q[2] + q[3]*(gap+tt)).expand_as(x)
            tu = b - gap*du
            td = g - losses._bcast(t-s, d)*(dd if vec else dd.sum((1, 2)))
        elif family in losses.ESD_LOSS_TYPES:
            b = x1 - x0 if family == 'esd_cond' else heads(teacher, s, s, x, vec)[0]
            density_b = x1 - x0 if family != 'esd' else b
            du = p[1]*x.square() - 2*p[2]*gap + p[3]*(gap-ss) + (p[0]+2*p[1]*ss*x)*b
            dd = q[1]*x - q[2] - q[3]*tt + (2*q[0]*x+q[1]*ss)*density_b
            g = heads(ema, s, s, x, vec)[1]
            tu = b + gap*du
            td = g + losses._bcast(t-s, d)*(dd if vec else dd.sum((1, 2)))
        else:
            a = h*s + (1-h)*t
            def composition(params):
                first_u, first_d = heads(params, s, a, x, vec)
                y = x + (a-s)[:, None, None]*first_u
                second_u, second_d = heads(params, a, t, y, vec)
                w = h[:, None, None]
                return (1-w)*first_u+w*second_u, (1-losses._bcast(h, first_d))*first_d+losses._bcast(h, first_d)*second_d
            tu = composition(teacher)[0]
            td = composition(ema)[1]
    return tu, td


def call_loss(net, family, teacher, ema, x0, x1, s, t, h, source='teacher', psd='uniform'):
    kw = dict(interp=interpolant.setup_interpolant(build_problem_config()), density_ema_params=ema)
    if family == 'psd':
        a = (s+t)/2 if psd == 'midpoint' else h*s+(1-h)*t
        return losses.psd_term(net, teacher, x0, x1, None, s, t, a, h, psd_type=psd, **kw)
    if family in losses.ESD_LOSS_TYPES:
        return losses.esd_term(net, teacher, x0, x1, None, s, t, loss_type=family, **kw)
    return losses.lsd_term(net, teacher, x0, x1, None, s, t, **kw)


@pytest.mark.parametrize('family,source,psd,density,vectorized', [
    (*case, density, vectorized)
    for case in [('lsd','teacher','uniform'), ('esd','teacher','uniform'),
                 ('esd_cond','conditional','uniform'), ('esd_cond_div_only','teacher','uniform'),
                 ('psd','teacher','uniform'), ('psd','teacher','midpoint')]
    for density, vectorized in [(False,False), (True,False), (True,True)]
    if density or case[0] != 'esd_cond_div_only'
])
@pytest.mark.parametrize('div_weight', ['uniform', 'linear', 'square'])
@pytest.mark.parametrize('weighted', [False, True])
def test_production_values_and_parameter_gradients(family, source, psd, density, vectorized, weighted, div_weight, monkeypatch):
    net = PolynomialMap(vectorized, density, weighted).to(DEVICE)
    net.config.div_loss_weighting_type = div_weight
    teacher, ema = snapshots(net)
    x0, x1, s, t, h = batch()
    x = (1-s[:,None,None])*x0+s[:,None,None]*x1
    ref_h = torch.full_like(h, .5) if psd == 'midpoint' else h
    tu, td = reference_targets(net, family, source, teacher, ema, x, x0, x1, s, t, ref_h)
    seen = []
    original = torch.func.jvp
    def watched(*args, **kwargs):
        primal, tangent = original(*args, **kwargs)
        for value in tangent if isinstance(tangent, tuple) else (tangent,):
            if value.requires_grad:
                value.register_hook(lambda grad: seen.append(grad))
        return primal, tangent
    monkeypatch.setattr(torch.func, 'jvp', watched)
    value, metrics = call_loss(net, family, teacher, ema, x0, x1, s, t, h, source, psd)
    params = tuple(net.parameters()) + tuple(teacher.values()) + tuple(ema.values())
    actual = torch.autograd.grad(value.sum(), params, allow_unused=True)
    assert seen == [], 'backward traversed a correction JVP'
    assert all(g is None for g in actual[3:]), 'teacher or EMA received gradients'
    u, d = heads(dict(net.named_parameters()), s, t, x, vectorized)
    raw = (u-tu).square().sum((1,2))
    if density:
        if family in ('esd_cond', 'esd_cond_div_only'):
            factor = {'uniform': torch.ones_like(s), 'linear': 1-s, 'square': (1-s)**2}[div_weight]
            dl = (d.square()-2*d*td).reshape(4,-1).sum(1)*factor
        else:
            dl = (d-td).square().reshape(4,-1).sum(1)
        raw = raw + dl/(6 if vectorized else 1)
    w = net.calc_weight(s,t)
    expected = torch.exp(-w)*raw+w if weighted else raw
    wanted = torch.autograd.grad(expected.sum(), tuple(net.parameters()), allow_unused=True)
    torch.testing.assert_close(value, expected, atol=2e-12, rtol=2e-12)
    for a,b in zip(actual[:3], wanted):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    assert actual[0].abs().max() > .01
    if density:
        assert actual[1].abs().max() > .01
    # Density depends on its own EMA even when an external velocity teacher exists.
    changed = {k: v.detach() + .2 for k,v in ema.items()}
    _, changed_metrics = call_loss(net, family, teacher, changed, x0, x1, s, t, h, source, psd)
    if density:
        assert not torch.allclose(metrics[f'{family}/{family}_div_loss'], changed_metrics[f'{family}/{family}_div_loss'])
        torch.testing.assert_close(metrics[f'{family}/{family}_v_loss'], changed_metrics[f'{family}/{family}_v_loss'])


@pytest.mark.parametrize('family', losses.DENSITY_LOSS_TYPES)
def test_density_ema_is_required_and_legacy_gradient_modes_rejected(family):
    net = PolynomialMap().to(DEVICE)
    x0,x1,s,t,h=batch()
    with pytest.raises(ValueError, match='density_ema_params'):
        call_loss(net,family,None,None,x0,x1,s,t,h)
    for old in ('none','full'):
        with pytest.raises(ValueError,match='stopped-target'):
            losses._validate_stopgrad(old)


@pytest.mark.parametrize('family', losses.DENSITY_LOSS_TYPES)
@pytest.mark.parametrize('vectorized', [False,True])
def test_real_dit_gradient_equals_frozen_regression(family,vectorized,monkeypatch):
    torch.manual_seed(12)
    cfg=build_network_config(n_particles=3,d_model=16,num_layers=1,nhead=2,
                             vectorized_divergence=vectorized,sigma_data=1.4)
    net=perturb_params_(flow_map.TorchFlowMap(cfg),std=.07,seed=15).to(DEVICE)
    teacher={k:v.detach().clone()+.01 for k,v in net.named_parameters()}
    ema={k:v.detach().clone()-.02 for k,v in net.named_parameters()}
    x0,x1,s,t,h=batch()
    x0,x1=x0.float(),x1.float()
    # Real DiT requires the configured atom count.
    x0=torch.cat((x0,x0[:,:1]),dim=1); x1=torch.cat((x1,x1[:,:1]),dim=1)
    s,t,h=s.float(),t.float(),h.float()
    captured={}
    original=losses._consistency_loss
    def capture(net,family,s,t,u,tu,d,td,**kwargs):
        captured.update(tu=tu.detach(),td=td.detach())
        return original(net,family,s,t,u,tu,d,td,**kwargs)
    monkeypatch.setattr(losses,'_consistency_loss',capture)
    loss,_=call_loss(net,family,teacher,ema,x0,x1,s,t,h)
    actual=torch.autograd.grad(loss.sum(),tuple(net.parameters()),allow_unused=True)
    x=(1-s[:,None,None])*x0+s[:,None,None]*x1
    u,d=net.calc_phi(s,t,x,train=False,return_div=True)
    dd = d.square()-2*d*captured['td'] if family in ('esd_cond','esd_cond_div_only') else (d-captured['td']).square()
    expected=(u-captured['tu']).square().sum()+dd.sum()/(9 if vectorized else 1)
    ref=torch.autograd.grad(expected,tuple(net.parameters()),allow_unused=True)
    for a,b in zip(actual,ref):
        assert (a is None)==(b is None)
        if a is not None:
            torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-4)


def test_conditional_esd_matches_reversed_meanflow_and_mean_gradient():
    net=PolynomialMap(density=True).to(DEVICE)
    teacher,ema=snapshots(net)
    x0,x1,s,t,h=batch()
    x=(1-s[:,None,None])*x0+s[:,None,None]*x1
    eta=x1-x0
    # Reverse coordinates r=1-t, tau=1-s and u_M=-u_F. Its material
    # derivative is partial_s u_F + J_x u_F*eta_F after time reversal.
    r,tau=1-t,1-s
    def meanflow(xx,rr,tt):
        return -net.calc_phi(1-tt,1-rr,xx)
    um,material=torch.func.jvp(meanflow,(x,r,tau),(-eta,torch.zeros_like(r),torch.ones_like(tau)))
    target=(-eta-(tau-r)[:,None,None]*material).detach()
    mf=(um-target).square().sum()
    grads_mf=torch.autograd.grad(mf,net.p)[0]
    net.config.predict_divergence=False
    esd,_=call_loss(net,'esd_cond',teacher,ema,x0,x1,s,t,h)
    torch.testing.assert_close(esd.sum(),mf,atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(esd.sum(),net.p)[0],grads_mf,atol=1e-11,rtol=1e-11)
    net.config.predict_divergence=True
    # At fixed x, two endpoint pairs with symmetric conditional-velocity noise.
    def grad_for(v):
        z=x-s[:,None,None]*v
        data=x+(1-s[:,None,None])*v
        val,_=call_loss(net,'esd_cond',teacher,ema,z,data,s,t,h)
        return val.detach(),torch.autograd.grad(val.sum(),(net.p,net.q))
    base,gb=grad_for(eta)
    plus,gp=grad_for(eta+.4)
    minus,gm=grad_for(eta-.4)
    for b,p,m in zip(gb,gp,gm):
        torch.testing.assert_close((p+m)/2,b,atol=1e-11,rtol=1e-11)
    assert not torch.allclose((plus+minus)/2,base)


@pytest.mark.parametrize('vectorized', [False, True])
def test_conditional_density_heads_share_targets_but_mixed_velocity_uses_teacher(vectorized):
    net = PolynomialMap(vectorized=vectorized).to(DEVICE)
    net.config.div_loss_weighting_type = 'linear'
    teacher, ema = snapshots(net)
    x0, x1, s, t, h = batch()
    def evaluate(family, source):
        value, metric = call_loss(net, family, source, ema, x0, x1, s, t, h)
        grads = torch.autograd.grad(value.sum(), (net.p, net.q))
        return metric[f'{family}/{family}_v_loss'], metric[f'{family}/{family}_div_loss'], grads
    v_cond, d_cond, g_cond = evaluate('esd_cond', teacher)
    v_mixed, d_mixed, g_mixed = evaluate('esd_cond_div_only', teacher)
    v_teacher, _, g_teacher = evaluate('esd', teacher)
    torch.testing.assert_close(v_mixed, v_teacher)
    torch.testing.assert_close(g_mixed[0], g_teacher[0])
    torch.testing.assert_close(d_mixed, d_cond)
    torch.testing.assert_close(g_mixed[1], g_cond[1])
    assert not torch.allclose(v_mixed, v_cond)
    changed = {k: v.detach()+.7 for k,v in teacher.items()}
    v_changed, d_changed, _ = evaluate('esd_cond_div_only', changed)
    assert not torch.allclose(v_changed, v_mixed)
    torch.testing.assert_close(d_changed, d_mixed)
    v_unused, d_unused, _ = evaluate('esd_cond', changed)
    torch.testing.assert_close(v_unused, v_cond)
    torch.testing.assert_close(d_unused, d_cond)


@pytest.mark.parametrize('family', ['esd_cond', 'esd_cond_div_only'])
@pytest.mark.parametrize('weight', ['uniform', 'linear', 'square'])
def test_conditional_density_factorization_matches_expected_target_not_noise_variance(family, weight):
    net = PolynomialMap(vectorized=True).to(DEVICE)
    net.config.div_loss_weighting_type = weight
    teacher, ema = snapshots(net)
    x0, x1, s, t, h = batch()
    x = (1-s[:,None,None])*x0+s[:,None,None]*x1
    eta = x1-x0
    def sample(v):
        z = x-s[:,None,None]*v
        data = x+(1-s[:,None,None])*v
        loss, metric = call_loss(net, family, teacher, ema, z, data, s, t, h)
        return metric[f'{family}/{family}_div_loss'], torch.autograd.grad(loss.sum(), net.q)[0]
    base, g = sample(eta)
    plus, gp = sample(eta+.4)
    minus, gm = sample(eta-.4)
    # The factorized density value is affine in the conditional velocity too;
    # restoring the target-square would fail this independent moment check.
    torch.testing.assert_close((plus+minus)/2, base, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((gp+gm)/2, g, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('family', ['esd_cond', 'esd_cond_div_only'])
def test_setup_routes_conditional_loss_and_rejects_non_linear_interpolant(family, monkeypatch):
    net = PolynomialMap().to(DEVICE)
    teacher, ema = snapshots(net)
    x0, x1, s, t, h = batch()
    cfg = SimpleNamespace(training=SimpleNamespace(loss_type=family, stopgrad_type='stopped'),
                          optimization=SimpleNamespace(diag_fraction=.5))
    interp = interpolant.setup_interpolant(build_problem_config())
    called = []
    original = losses.esd_term
    def record(*args, **kwargs):
        called.append(kwargs['loss_type'])
        return original(*args, **kwargs)
    monkeypatch.setattr(losses, 'esd_term', record)
    fn = losses.setup_loss(cfg, net, interp)
    value, metrics = fn(teacher, teacher, x0, x1, None, s, t, None, None, density_ema_params=ema)
    assert called == [family]
    assert torch.isfinite(value) and f'{family}/{family}_div_loss' in metrics
    with pytest.raises(ValueError, match='linear interpolant'):
        losses.setup_loss(cfg, net, interpolant.setup_interpolant(build_problem_config('trig')))


def test_mixed_density_variant_cannot_silently_train_without_density_head():
    net = PolynomialMap(density=False).to(DEVICE)
    teacher, ema = snapshots(net)
    x0, x1, s, t, h = batch()
    with pytest.raises(ValueError, match='requires a density head'):
        call_loss(net, 'esd_cond_div_only', teacher, ema, x0, x1, s, t, h)


def test_legacy_conditional_config_requires_explicit_migration():
    net = PolynomialMap().to(DEVICE)
    cfg = SimpleNamespace(training=SimpleNamespace(loss_type='esd', stopgrad_type='stopped',
                                                   esd_velocity_source='conditional'))
    with pytest.raises(ValueError, match="loss_type='esd_cond'"):
        losses.setup_loss(cfg, net, interpolant.setup_interpolant(build_problem_config()))
