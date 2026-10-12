"""Muon grouping, native updates, checkpoint continuation and finite-step policy."""
from copy import deepcopy

import pytest
import torch

from common import state_utils, updates
from common.flow_map import TorchFlowMap
from common.optimizers import MuonAdamW
from configs.ace_a_nme import get_config


def config():
    cfg = get_config(0)
    cfg.network.dit_kwargs = dict(d_model=12, num_layers=2, num_shared_layers=1,
                                 num_div_layers=1, nhead=3, mlp_ratio=2,
                                 time_embedding_dim=8)
    cfg.network.predict_divergence = True
    cfg.network.vectorized_divergence = True
    cfg.network.divergence_mode = 'dlogpdt'
    cfg.network.load_path = ''
    cfg.optimization.optimizer = 'muon_adamw'
    cfg.optimization.learning_rate = 1e-4
    cfg.optimization.decay_steps = 50000
    cfg.optimization.schedule_type = 'sqrt'
    return cfg


def model():
    torch.manual_seed(5)
    return TorchFlowMap(config().network)


def gradients(net, step):
    gen = torch.Generator().manual_seed(90 + step)
    for p in net.parameters():
        p.grad = torch.randn(p.shape, generator=gen)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b): assert_tree_equal(x,y)
    else: assert a == b


def test_groups_cover_all_parameters_and_protect_modulation_and_heads():
    net = model(); opt = MuonAdamW(net, lr=1e-4)
    names = opt.parameter_names
    assert set(names['muon']).isdisjoint(names['adamw'])
    assert set(names['muon'] + names['adamw']) == set(dict(net.named_parameters()))
    assert len(names['muon']) == 18  # Q, K, V, attention output and two MLP matrices per block
    assert all('modulation' not in n and not n.endswith('.bias') for n in names['muon'])
    for n in ['dit.x_embedder.weight', 'dit.final_linear.weight',
              'dit.div_final_linear.weight', 'dit.blocks.0.modulation.weight']:
        assert n in names['adamw']
    assert all(pg['weight_decay'] == 1e-4 for pg in opt.param_groups)


def test_hybrid_matches_independent_native_optimizers():
    a = model(); b = deepcopy(a); opt = MuonAdamW(a, lr=1e-4)
    bp = dict(b.named_parameters())
    muon = torch.optim.Muon([bp[n] for n in opt.parameter_names['muon']],
                           lr=1e-4, momentum=.95, nesterov=True, ns_steps=5,
                           eps=1e-7, adjust_lr_fn='match_rms_adamw', weight_decay=1e-4)
    adam = torch.optim.AdamW([bp[n] for n in opt.parameter_names['adamw']],
                            lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=1e-4)
    for i in range(4):
        gradients(a,i); gradients(b,i); opt.step(); muon.step(); adam.step()
    assert_tree_equal(a.state_dict(), b.state_dict())


def test_checkpoint_continuation_preserves_both_states_and_groups():
    cfg=config();a=model();opt,sched=state_utils.setup_optimizer(cfg,a)
    first=state_utils.TrainState(a, opt, sched, [0.999])
    for i in range(3):
        gradients(a,i);opt.step();first.step+=1;first.update_ema()
    saved=deepcopy(first.state_dict())
    b=model();other,sch=state_utils.setup_optimizer(cfg,b)
    second=state_utils.TrainState(b, other, sch, [0.999])
    second.load_state_dict(saved)
    for state in [first,second]:
        gradients(state.net,4);state.optimizer.step();state.step+=1;state.update_ema()
    assert_tree_equal(first.state_dict(),second.state_dict())
    second.optimizer.param_groups[0]['lr']=7e-5
    assert second.optimizer.muon.param_groups[0]['lr']==7e-5
    invalid=deepcopy(saved['opt_state']);invalid['parameter_names']['muon'].reverse()
    with pytest.raises(ValueError,match='grouping'):second.optimizer.load_state_dict(invalid)


@pytest.mark.parametrize('old,new',[('radam','muon_adamw'),('muon_adamw','radam')])
def test_optimizer_switch_requires_fresh_state(old,new):
    cfg=config();a=model();cfg.optimization.optimizer=old
    opt,sch=state_utils.setup_optimizer(cfg,a)
    first=state_utils.TrainState(a, opt, sch, [0.999])
    cfg.optimization.optimizer=new;b=model();opt,sch=state_utils.setup_optimizer(cfg,b)
    second=state_utils.TrainState(b, opt, sch, [0.999])
    with pytest.raises(ValueError,match='reset_optimizer'):second.load_state_dict(first.state_dict())
    second.load_state_dict(first.state_dict(),load_optimizer=False)
    assert not second.optimizer.state


def test_group_schedule_and_nonfinite_step_skip_both_optimizers_and_ema():
    cfg=config();cfg.network.predict_divergence=False
    cfg.optimization.adamw_lr_multiplier=.3
    net=model();opt,sch=state_utils.setup_optimizer(cfg,net)
    state=state_utils.TrainState(net, opt, sch, [0.999])
    train=updates.setup_train_step(cfg);ema=updates.setup_ema_update(cfg)
    def loss(*args,**kwargs):
        return sum(p.square().sum() for p in net.parameters()),{}
    state.step=100000
    train(state,loss,None,None,(None,)*7,torch.Generator());ema(state)
    assert opt.muon.param_groups[0]['lr']==pytest.approx(1e-4/2**.5)
    assert opt.adamw.param_groups[0]['lr']==pytest.approx(.3e-4/2**.5)
    saved=deepcopy(state.state_dict())
    def invalid(*args,**kwargs):return loss()[0]*float('nan'),{}
    _,metrics=train(state,invalid,None,None,(None,)*7,torch.Generator());ema(state)
    assert metrics['train/nan_detected'] and not state.last_update_succeeded
    # A skipped step may assign the schedule's next LR, but states/weights/EMA stay fixed.
    after=state.state_dict()
    for key in ['params','ema_params','step']:assert_tree_equal(saved[key],after[key])
    for child in ['muon','adamw']:
        assert_tree_equal(saved['opt_state'][child]['state'],after['opt_state'][child]['state'])
    assert sch(1000)==1e-4 and sch(50000)==1e-4
    assert sch(200000)==5e-5
