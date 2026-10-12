"""Branched DiT: execution, gradient ownership and pretrained state continuity."""
from collections import Counter
from copy import deepcopy
import importlib

import pytest
import torch
from torch.func import functional_call

from common import interpolant, losses, state_utils, updates
from common.flow_map import TorchFlowMap
from configs.ace_a_nme import get_config


def config(mode='dlogpdt', *, legacy=False, vectorized=True):
    cfg = get_config(0)
    cfg.network.dit_kwargs = dict(d_model=24, num_layers=6, nhead=3,
                                 mlp_ratio=2, time_embedding_dim=16)
    if not legacy:
        cfg.network.dit_kwargs.update(num_shared_layers=4, num_div_layers=2)
    cfg.network.predict_divergence = mode not in ('pretrain_v', 'pretrain_flowmap')
    cfg.network.divergence_mode = mode
    cfg.network.vectorized_divergence = vectorized
    cfg.network.divergence_rescale = 1.
    cfg.network.div_loss_weighting_type = 'linear'
    cfg.network.from_scratch = False
    cfg.network.load_path = ''
    cfg.network.reset_optimizer = True
    cfg.teacher.load_path = ''
    cfg.optimization.bs = 4
    cfg.optimization.diag_fraction = 1. if mode == 'pretrain_v' else .5
    cfg.training.loss_type = 'lsd'
    cfg.training.stopgrad_type = 'convex'
    return cfg


def model(cfg=None):
    torch.manual_seed(27)
    net = TorchFlowMap((cfg or config()).network)
    # Exercise every attention block and both outputs away from zero init.
    with torch.no_grad():
        for name, p in net.named_parameters():
            if 'modulation' in name or 'final_linear' in name:
                p.normal_(std=.035)
    return net


def batch():
    gen = torch.Generator().manual_seed(84)
    x0 = torch.randn(4, 22, 3, generator=gen)
    x1 = torch.randn(4, 22, 3, generator=gen)
    s = torch.tensor([.2, .5, .1, .3])
    t = torch.tensor([.2, .5, .6, .9])
    return x0, x1, None, s, t, None, None


def grad_dict(net, loss):
    params = dict(net.named_parameters())
    values = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
    return {k: torch.zeros_like(p) if g is None else g for (k,p),g in zip(params.items(),values)}


def equal_dict(a, b):
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, msg=k)


@pytest.mark.parametrize('name', ['ace_a_nme', 'ace_aaa_nme', 'aaa', 'aaaaaa'])
def test_peptide_defaults_have_six_blocks_per_head(name):
    cfg = importlib.import_module('configs.' + name).get_config(0)
    kw = cfg.network.dit_kwargs
    assert (kw.num_layers, kw.num_shared_layers, kw.num_div_layers) == (6, 4, 2)


@pytest.mark.parametrize('which', ['velocity', 'divergence', 'both'])
def test_only_requested_blocks_and_projection_execute(which):
    net = model()
    x, _, _, s, t, _, _ = batch()
    calls = Counter()
    handles = []
    for name, m in net.dit.named_modules():
        if m.__class__.__name__ == 'DiTBlock' or name in ('final_linear', 'div_final_linear'):
            handles.append(m.register_forward_hook(lambda m,a,o,name=name: calls.update([name])))
    try:
        ss, tt, xx, _ = net.process_input(s, t, x)
        v, d = net.dit(xx, ss, tt-ss, output=which)
    finally:
        for h in handles:
            h.remove()
    expected = Counter({f'blocks.{i}': 1 for i in range(4)})
    if which != 'divergence':
        expected.update({'blocks.4': 1, 'blocks.5': 1, 'final_linear': 1})
        assert v.abs().max() > 0
    else:
        assert v is None
    if which != 'velocity':
        expected.update({'div_blocks.0': 1, 'div_blocks.1': 1, 'div_final_linear': 1})
        assert d.abs().max() > 0
    else:
        assert d is None
    assert calls == expected


@pytest.mark.parametrize('vectorized', [False, True])
@pytest.mark.parametrize('which', ['velocity', 'divergence'])
def test_selective_values_input_time_and_parameter_jvp_gradients_match_joint(which, vectorized):
    net = model(config(vectorized=vectorized))
    x, _, _, s, t, _, _ = batch()
    index = 0 if which == 'velocity' else 1
    def selected(tt):
        return net.calc_phi(s, tt, x) if index == 0 else net.calc_div(s, tt, x)
    def joint(tt):
        return net.calc_phi(s, tt, x, return_div=True)[index]
    pv, dv = torch.func.jvp(selected, (t,), (torch.ones_like(t),))
    pj, dj = torch.func.jvp(joint, (t,), (torch.ones_like(t),))
    torch.testing.assert_close(pv, pj, rtol=0, atol=0)
    torch.testing.assert_close(dv, dj, rtol=0, atol=0)
    assert pv.abs().max() > 1e-5 and dv.abs().max() > 1e-7
    gv = grad_dict(net, (pv + losses._bcast(t-s,pv)*dv).square().sum())
    gj = grad_dict(net, (pj + losses._bcast(t-s,pj)*dj).square().sum())
    equal_dict(gv, gj)
    assert gv['dit.blocks.0.attn.q.weight'].norm() > 0
    other = 'dit.div_blocks.' if index == 0 else 'dit.blocks.4.'
    assert all(g.count_nonzero() == 0 for k,g in gv.items() if k.startswith(other))
    # Spatial derivatives also remain identical for Hutchinson/material JVP use.
    a = torch.func.jvp(lambda xx: net.calc_phi(s,t,xx,return_div=True)[index], (x,), (x,))[1]
    b = torch.func.jvp(lambda xx: net.calc_phi(s,t,xx) if index == 0 else net.calc_div(s,t,xx), (x,), (x,))[1]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('single', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_density_only_api_weight_shape_and_functional_call(single, weighted):
    cfg = config()
    cfg.network.use_weight = weighted
    cfg.network.logvar_channels = 16 if weighted else None
    net = model(cfg)
    x, _, _, s, t, _, _ = batch()
    if single:
        x, s, t = x[0], s[0], t[0]
    v, d = net.calc_phi(s,t,x,return_div=True)
    only = net.calc_div(s,t,x)
    torch.testing.assert_close(only,d,rtol=0,atol=0)
    assert only.shape == ((22,3) if single else (4,22,3))
    result = net.calc_div(s,t,x,calc_weight=True)
    joint = net.calc_phi(s,t,x,return_div=True,calc_weight=True)
    torch.testing.assert_close(result[0],joint[1],rtol=0,atol=0)
    torch.testing.assert_close(result[1],joint[2],rtol=0,atol=0)
    out = functional_call(net, dict(net.named_parameters()), (s,t,x), {'method':'calc_div'})
    torch.testing.assert_close(out,d,rtol=0,atol=0)
    assert net.calc_phi(s,t,x,return_div=True,return_velocity=False)[0] is None


def test_invalid_selection_and_absent_density_fail_clearly():
    net = model(config('pretrain_v'))
    x,_,_,s,t,_,_ = batch()
    torch.testing.assert_close(net.calc_phi(s,t,x,return_div=True),net.calc_phi(s,t,x))
    with pytest.raises(ValueError,match='Density-only'):
        net.calc_div(s,t,x)
    with pytest.raises(ValueError,match='Density-only'):
        model().calc_phi(s,t,x,return_velocity=False)
    with pytest.raises(ValueError,match='Unknown DiT output'):
        net.dit(x,s[:,None,None],t[:,None,None],output='invalid')


@pytest.mark.parametrize('family', ['lsd','esd'])
@pytest.mark.parametrize('policy', ['convex','stopped'])
def test_teacher_head_selection_preserves_complete_loss_and_gradients(family, policy, monkeypatch):
    cfg = config()
    cfg.training.loss_type, cfg.training.stopgrad_type = family, policy
    net = model(cfg)
    teacher = {n: (p.detach().clone()+.01).requires_grad_() for n,p in net.named_parameters() if not n.startswith('dit.div_')}
    ema = {n: (p.detach().clone()-.01).requires_grad_() for n,p in net.named_parameters()}
    fn = losses.setup_loss(cfg,net,interpolant.setup_interpolant(cfg))
    calls = []
    handle = net.dit.register_forward_pre_hook(lambda m,a,kw: calls.append(kw['output']),with_kwargs=True)
    value, metrics = fn(teacher,teacher,*batch(),density_ema_params=ema)
    handle.remove()
    grads = grad_dict(net,value)
    assert calls.count('velocity') == 2
    assert calls.count('both') == 2
    assert calls.count('divergence') == 1
    # Force the unoptimized evaluation of both outputs as an independent control.
    old_forward = net.dit.forward
    monkeypatch.setattr(net.dit,'forward',lambda *a,**kw: old_forward(*a,**{**kw,'output':'both'}))
    reference, reference_metrics = fn(teacher,teacher,*batch(),density_ema_params=ema)
    equal_dict(grads,grad_dict(net,reference))
    torch.testing.assert_close(value,reference,rtol=0,atol=0)
    equal_dict(metrics,reference_metrics)
    assert all(p.grad is None for p in [*teacher.values(),*ema.values()])
    for key in ['dit.blocks.0.attn.q.weight','dit.blocks.4.attn.q.weight','dit.div_blocks.0.attn.q.weight']:
        assert torch.isfinite(grads[key]).all() and grads[key].norm() > 0


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('stage', ['pretrain_v','pretrain_flowmap'])
def test_pretrained_six_blocks_raw_ema_and_teacher_handoff(tmp_path, legacy, stage):
    cfg = config(stage,legacy=legacy)
    net = model(cfg)
    opt,sched = state_utils.setup_optimizer(cfg,net)
    state = state_utils.TrainState(net, opt, sched, [0.999])
    # Distinct saved EMA history must survive the handoff as well as raw weights.
    with torch.no_grad():
        for name,v in state.ema_params[.999].items():
            if name in dict(net.named_parameters()):
                v.add_(.003)
    saved = state.state_dict()
    if legacy:
        saved.pop('network_architecture')
    path = tmp_path/'pretrained.pt'
    torch.save(saved,path)
    target = config('dlogpdt')
    target.network.load_path = str(path)
    target.teacher.load_path = str(path)
    target.teacher.ema_fac = .999
    loaded,new_net,_ = state_utils.setup_training_state(target, 'cpu')
    teacher = state_utils.load_teacher_params(target,loaded,'cpu')
    assert len(new_net.dit.blocks) == 6 and len(new_net.dit.div_blocks) == 2
    assert new_net.dit.num_shared_layers == 4
    assert loaded.step == 0 and not loaded.optimizer.state
    for name,p in net.state_dict().items():
        torch.testing.assert_close(new_net.state_dict()[name],p,rtol=0,atol=0)
    for name,p in state.ema_params[.999].items():
        torch.testing.assert_close(loaded.ema_params[.999][name],p,rtol=0,atol=0)
    for name,p in teacher.items():
        torch.testing.assert_close(p,state.ema_params[.999][name],rtol=0,atol=0)
    for name,p in new_net.named_parameters():
        if name.startswith('dit.div_'):
            torch.testing.assert_close(loaded.ema_params[.999][name],p,rtol=0,atol=0)
    x,_,_,s,t,_,_ = batch()
    torch.testing.assert_close(new_net.calc_phi(s,t,x),net.calc_phi(s,t,x),rtol=0,atol=0)
    _,d = new_net.calc_phi(s,t,x,return_div=True)
    assert d.count_nonzero() == 0  # standard fresh output projection
    fn = losses.setup_loss(target,new_net,interpolant.setup_interpolant(target))
    before = new_net.dit.div_final_linear.weight.detach().clone()
    update = updates.setup_train_step(target)
    for _ in range(3):
        _,m = update(loaded,fn,teacher,teacher,batch(),torch.Generator().manual_seed(1))
        assert m['train/grads_finite'] and loaded.last_update_succeeded
        loaded.update_ema()
    assert not torch.equal(before,new_net.dit.div_final_linear.weight)
    assert not torch.equal(loaded.ema_params[.999]['dit.div_final_linear.weight'],before)


def test_density_checkpoint_layout_roundtrip_and_legacy_guard(tmp_path):
    for legacy in [False,True]:
        cfg = config(legacy=legacy)
        net = model(cfg)
        opt,sched = state_utils.setup_optimizer(cfg,net)
        state = state_utils.TrainState(net, opt, sched, [0.999])
        saved = state.state_dict()
        if legacy:
            saved.pop('network_architecture')
        path=tmp_path/f'density_{legacy}.pt'
        torch.save(saved,path)
        eval_cfg=config()
        state_utils.load_evaluation_model(eval_cfg, path, "cpu", use_ema=False)
        restored=model(eval_cfg)
        opt2,sched2=state_utils.setup_optimizer(eval_cfg,restored)
        second=state_utils.TrainState(restored, opt2, sched2, [0.999])
        second.load_state_dict(saved)
        equal_dict(restored.state_dict(),net.state_dict())
        x,_,_,s,t,_,_=batch()
        for a,b in zip(restored.calc_phi(s,t,x,return_div=True),net.calc_phi(s,t,x,return_div=True)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        if legacy:
            fresh=model(config())
            tx,sc=state_utils.setup_optimizer(config(),fresh)
            bad=state_utils.TrainState(fresh, tx, sc, [0.999])
            before=deepcopy(fresh.state_dict())
            with pytest.raises(ValueError,match='density architecture'):
                bad.load_state_dict(saved)
            equal_dict(before,fresh.state_dict())


def test_legacy_flax_velocity_is_rejected_before_loading(tmp_path):
    path = tmp_path / 'velocity.pkl'
    path.write_bytes(b'legacy checkpoint should not be decoded')
    cfg = config('pretrain_flowmap')
    cfg.network.load_path = str(path)
    with pytest.raises(ValueError, match='native .pt'):
        state_utils.setup_training_state(cfg, 'cpu')


def test_canonical_native_checkpoint_rejected_without_partial_restore():
    cfg = config()
    net = model(cfg)
    opt, schedule = state_utils.setup_optimizer(cfg, net)
    state = state_utils.TrainState(net, opt, schedule, [0.999])
    saved = deepcopy(state.state_dict())
    saved['model_version'] = 'canonical_atom_dit_v2'
    before = deepcopy(net.state_dict())
    with pytest.raises(ValueError, match='Incompatible backbone'):
        state.load_state_dict(saved, load_optimizer=False)
    equal_dict(before, net.state_dict())


@pytest.mark.parametrize('psd_type', ['uniform', 'midpoint'])
def test_psd_second_density_target_skips_velocity_with_exact_loss_and_gradients(psd_type, monkeypatch):
    cfg = config()
    cfg.training.loss_type = 'psd'
    cfg.training.stopgrad_type = 'stopped'
    cfg.training.psd_type = psd_type
    net = model(cfg)
    teacher = {n: p.detach().clone()+.01 for n,p in net.named_parameters()}
    ema = {n: p.detach().clone()-.01 for n,p in net.named_parameters()}
    x0,x1,label,s,t,_,_ = batch()
    h = torch.full_like(s, .5) if psd_type == 'midpoint' else torch.tensor([.2,.4,.6,.8])
    u = h*s+(1-h)*t
    args = x0,x1,label,s,t,u,h
    fn = losses.setup_loss(cfg,net,interpolant.setup_interpolant(cfg))
    calls = []
    handle = net.dit.register_forward_pre_hook(lambda m,a,kw: calls.append(kw['output']),with_kwargs=True)
    value, metrics = fn(teacher,teacher,*args,density_ema_params=ema)
    handle.remove()
    grads = grad_dict(net,value)
    # Joint diagonal/student/first density midpoint, three velocity teachers,
    # and one density-only second EMA call.
    assert Counter(calls) == Counter(both=3, velocity=3, divergence=1)
    old_forward = net.dit.forward
    monkeypatch.setattr(net.dit,'forward',lambda *a,**kw: old_forward(*a,**{**kw,'output':'both'}))
    reference, reference_metrics = fn(teacher,teacher,*args,density_ema_params=ema)
    equal_dict(grads,grad_dict(net,reference))
    torch.testing.assert_close(value,reference,rtol=0,atol=0)
    equal_dict(metrics,reference_metrics)
    assert grads['dit.div_blocks.0.attn.q.weight'].norm() > 0
    assert grads['dit.blocks.4.attn.q.weight'].norm() > 0
