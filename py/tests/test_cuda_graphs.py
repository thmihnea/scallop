"""CUDA graph parity, fresh RNG, restore invalidation and rejected-step recovery."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from common import distributed, interpolant, loss_args, losses, state_utils, updates
from common.flow_map import TorchFlowMap
from configs.ace_a_nme import get_config


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def snapshot_before_skipped_step(state):
    """Only the candidate LR assignment may change during a skipped update."""
    saved = deepcopy(state.state_dict())
    optimizer = saved['opt_state']
    children = [optimizer[k] for k in ('muon', 'adamw')] if 'muon' in optimizer else [optimizer]
    for child in children:
        for group in child['param_groups']:
            group['lr'] = state.schedule(state.step) * group.get('lr_multiplier', 1.)
    return saved


def configuration(mode, family='lsd', psd=None, stopgrad='stopped'):
    cfg = get_config(0)
    cfg.network.dit_kwargs = dict(d_model=32, num_layers=3, num_shared_layers=2, num_div_layers=1, nhead=4,
                                 mlp_ratio=2, time_embedding_dim=16)
    cfg.network.divergence_mode = mode
    cfg.network.predict_divergence = mode not in ('pretrain_v', 'pretrain_flowmap')
    cfg.network.vectorized_divergence = cfg.network.predict_divergence
    cfg.network.divergence_rescale = 1.0
    cfg.network.div_loss_weighting_type = 'linear'
    cfg.network.from_scratch = mode == 'pretrain_v'
    cfg.training.loss_type = family
    cfg.training.psd_type = psd
    cfg.training.stopgrad_type = stopgrad
    cfg.training.diag_teacher_source = 'conditional'
    cfg.training.time_distr = 'uniform'
    cfg.optimization.bs = 16
    cfg.optimization.diag_fraction = 1.0 if mode == 'pretrain_v' else .5
    return cfg


class Pair:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda:0')
        torch.manual_seed(123)
        net = TorchFlowMap(cfg.network).to(self.device)
        with torch.no_grad():
            velocity_head = net.dit.final_linear if hasattr(net.dit, "final_linear") else net.dit.final_layer.linear
            velocity_head.weight.normal_(std=.02)
            if cfg.network.predict_divergence:
                density_head = net.dit.div_final_linear if hasattr(net.dit, "div_final_linear") else net.dit.div_final_layer.linear
                density_head.weight.normal_(std=.02)
        opt, sched = state_utils.setup_optimizer(cfg, net)
        self.eager = state_utils.TrainState(net, opt, sched, [0.999])
        self.graph = deepcopy(self.eager)
        self.interp = interpolant.setup_interpolant(cfg)
        self.eager_loss = losses.setup_loss(cfg, net, self.interp)
        self.graph_loss = losses.setup_loss(cfg, self.graph.net, self.interp)
        self.eager_step = updates.setup_train_step(cfg)
        graph_cfg = deepcopy(cfg)
        graph_cfg.training.cuda_graphs = True
        self.graph_step = updates.setup_train_step(graph_cfg)
        self.cpu_rng = torch.Generator().manual_seed(17)
        self.eager_rng = torch.Generator(device=self.device).manual_seed(23)
        self.graph_rng = torch.Generator(device=self.device).manual_seed(23)
        self.teacher = {n: p.detach().clone() for n, p in net.named_parameters()}

    def batch(self):
        x0 = torch.randn(self.cfg.optimization.bs, 22, 3, generator=self.cpu_rng).to(self.device)
        x1 = torch.randn(x0.shape, generator=self.cpu_rng).to(self.device)
        times = tuple(None if t is None else t.to(self.device)
                      for t in loss_args.sample_times(self.cfg, self.cpu_rng))
        return (x0, x1, None, *times)

    def teachers(self, state):
        """The configured (diag, offdiag) teachers, exactly as the training loop selects them."""
        from launchers.learn import _select_teacher
        return [_select_teacher(getattr(self.cfg.training, name), self.teacher,
                                state.self_teacher(), state.ema_teacher())
                for name in ('diag_teacher_source', 'offdiag_teacher_source')]

    def step(self, batch):
        values, flags = [], []
        for state, update, fn, gen in ((self.eager, self.eager_step, self.eager_loss, self.eager_rng),
                                        (self.graph, self.graph_step, self.graph_loss, self.graph_rng)):
            value, metrics = update(state, fn, *self.teachers(state), batch, gen)
            if state.last_update_succeeded:
                state.update_ema()
            values.append(value)
            flags.append(metrics['train/nan_detected'])
        assert_equal(values[0], values[1])
        assert_equal(self.eager.state_dict(), self.graph.state_dict())
        assert_equal([p.grad for p in self.eager.net.parameters()],
                     [p.grad for p in self.graph.net.parameters()])
        assert_equal(self.eager_rng.get_state(), self.graph_rng.get_state())
        assert flags[0] == flags[1]
        return values[0], flags[0]


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@requires_cuda
@pytest.mark.parametrize('mode,family,psd,stopgrad', [
    ('pretrain_v', 'lsd', None, 'stopped'),
    ('pretrain_flowmap', 'lsd', None, 'convex'),
    ('pretrain_flowmap', 'esd', None, 'stopped'),
    ('pretrain_flowmap', 'esd_imf', None, 'stopped'),
    ('pretrain_flowmap', 'psd', 'uniform', 'stopped'),
    ('pretrain_flowmap', 'psd', 'midpoint', 'stopped'),
    ('dlogpdt', 'lsd', None, 'stopped'),
    ('dlogpdt', 'lsd', None, 'convex'),
    ('dlogpdt', 'esd', None, 'stopped'),
    ('dlogpdt', 'esd_cond', None, 'stopped'),
    ('dlogpdt', 'esd_cond_div_only', None, 'stopped'),
    ('dlogpdt', 'psd', 'uniform', 'stopped'),
    ('dlogpdt', 'psd', 'midpoint', 'stopped'),
    ('cond_trace', 'lsd', None, 'stopped'),
    ('hutchinson', 'lsd', None, 'stopped'),
    ('hutchinson', 'lsd', None, 'convex'),
    ('hutchinson', 'psd', 'uniform', 'stopped'),
])
def test_replay_matches_eager_and_rng(mode, family, psd, stopgrad):
    pair = Pair(configuration(mode, family, psd, stopgrad))
    pair.eager.step = pair.graph.step = pair.cfg.optimization.decay_steps - 1
    for _ in range(4):
        _, skipped = pair.step(pair.batch())
        assert not skipped
    assert pair.graph_step.cuda_graph.capture_count == 1


@requires_cuda
def test_hutchinson_probes_change_and_recapture_does_not_consume_rng():
    pair = Pair(configuration('hutchinson'))
    batch = pair.batch()
    # Zero LR isolates changes from fresh probes, rather than parameter updates.
    pair.eager.schedule = pair.graph.schedule = lambda _: 0.0
    values = [float(pair.step(batch)[0]) for _ in range(3)]
    assert len(set(values)) > 1
    checkpoint = deepcopy(pair.eager.state_dict())
    pair.eager.load_state_dict(deepcopy(checkpoint))
    pair.graph.load_state_dict(deepcopy(checkpoint))
    pair.step(batch)
    assert pair.graph_step.cuda_graph.capture_count == 2


@requires_cuda
def test_density_restore_validation_and_nonfinite_recovery(tmp_path):
    pair = Pair(configuration('dlogpdt'))
    batch = pair.batch()
    pair.step(batch)
    checkpoint = tmp_path / 'state.pt'
    state_utils.save_state(pair.graph, str(checkpoint))
    saved = torch.load(checkpoint, weights_only=False)
    for name, value in saved['ema_params'][.999].items():
        if 'div' in name:
            value.add_(.01)
    for state in (pair.eager, pair.graph):
        state.load_state_dict(deepcopy(saved))
    pair.step(batch)
    assert pair.graph_step.cuda_graph.capture_count == 2
    bad = list(None if v is None else v.clone() for v in batch)
    bad[3][0] = bad[4][0] = 1.0
    for state, update, fn, gen in ((pair.eager, pair.eager_step, pair.eager_loss, pair.eager_rng),
                                    (pair.graph, pair.graph_step, pair.graph_loss, pair.graph_rng)):
        before = deepcopy(state.state_dict())
        # Same teachers as every other step: a teacher at new storage would rightly recapture.
        with pytest.raises(ValueError, match='diagonal times'):
            update(state, fn, *pair.teachers(state), tuple(bad), gen)
        assert_equal(before, state.state_dict())
    pair.step(batch)
    # NaN off-diagonal input bypasses the diagonal guard; finite-update policy skips it.
    bad = list(None if v is None else v.clone() for v in batch)
    bad[0][-1, 0, 0] = float('nan')
    before = snapshot_before_skipped_step(pair.eager)
    _, skipped = pair.step(tuple(bad))
    assert skipped
    assert_equal(before, pair.eager.state_dict())
    pair.step(batch)
    assert pair.graph_step.cuda_graph.capture_count == 2


@requires_cuda
def test_default_cuda_generator_device_alias():
    pair = Pair(configuration('hutchinson'))
    pair.eager_rng = torch.Generator(device='cuda').manual_seed(23)
    pair.graph_rng = torch.Generator(device='cuda').manual_seed(23)
    pair.step(pair.batch())
    pair.step(pair.batch())


@requires_cuda
def test_input_shape_change_is_rejected():
    pair = Pair(configuration('pretrain_flowmap'))
    batch = pair.batch()
    pair.step(batch)
    smaller = tuple(None if v is None else v[:8] for v in batch)
    with pytest.raises(ValueError, match='fixed input shapes'):
        pair.graph_step(pair.graph, pair.graph_loss, None, pair.graph.self_teacher(),
                        smaller, pair.graph_rng)


def test_cuda_graphs_reject_cpu_and_eager_is_default():
    cfg = configuration('pretrain_v')
    assert updates.setup_train_step(cfg).cuda_graph is None
    with pytest.raises(ValueError, match='CUDA device'):
        distributed.wrap_loss(None, None, torch.device('cpu'), cuda_graphs=True)


@pytest.mark.skipif(os.environ.get('IF2D2_TEST_DDP_CUDA') != '1',
                    reason='set IF2D2_TEST_DDP_CUDA=1 with two authorized GPUs')
def test_two_gpu_nccl_capture_and_recovery(tmp_path):
    assert torch.cuda.device_count() >= 2
    output = tmp_path / 'ddp_graphs.json'
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                             '--nproc_per_node=2',
                             str(Path(__file__).with_name('cuda_graph_ddp_worker.py')),
                             '--output', str(output)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=180)
    assert result.returncode == 0, result.stdout
    report = json.loads(output.read_text())
    assert report['world_size'] == 2
    assert len(report['results']) == 12


@requires_cuda
@pytest.mark.parametrize('family,psd,policy', [
    ('esd', None, 'stopped'), ('esd', None, 'convex'),
    ('psd', 'uniform', 'stopped'), ('psd', 'midpoint', 'stopped'),
])
def test_density_muon_graphs_restore_and_teacher_isolation(family, psd, policy, tmp_path):
    cfg = configuration('dlogpdt', family, psd, policy)
    cfg.optimization.optimizer = 'muon_adamw'
    cfg.optimization.decay_steps = 50000
    cfg.training.offdiag_time_order = 'forward'
    cfg.training.diag_teacher_source = 'external'
    cfg.training.offdiag_teacher_source = 'ema' if family == 'psd' else 'external'
    pair = Pair(cfg)
    teacher_before = deepcopy(pair.teacher)
    pair.eager.step = pair.graph.step = 49999
    initial_density = pair.eager.net.dit.div_final_linear.weight.detach().clone()
    for _ in range(4):
        _, skipped = pair.step(pair.batch())
        assert not skipped
    assert pair.graph_step.cuda_graph.capture_count == 1
    assert not torch.equal(initial_density, pair.graph.net.dit.div_final_linear.weight)
    assert pair.graph.optimizer.muon.state and pair.graph.optimizer.adamw.state
    path = tmp_path/'density_muon.pt'
    state_utils.save_state(pair.graph, str(path))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['network_architecture']['num_shared_layers'] == 2
    for state in (pair.eager, pair.graph):
        state.load_state_dict(deepcopy(saved))
    _, skipped = pair.step(pair.batch())
    assert not skipped and pair.graph_step.cuda_graph.capture_count == 2
    assert_equal(teacher_before, pair.teacher)


@requires_cuda
@pytest.mark.parametrize('mode', ['pretrain_v', 'dlogpdt', 'hutchinson'])
def test_v2_replay_restore_and_nonfinite_recovery(mode):
    cfg = configuration(mode, stopgrad='convex')
    cfg.network.dit_version = 'canonical_atom_dit_v2'
    pair = Pair(cfg)
    batch = pair.batch()
    for _ in range(3):
        assert not pair.step(batch)[1]
    saved = deepcopy(pair.eager.state_dict())
    pair.eager.load_state_dict(deepcopy(saved))
    pair.graph.load_state_dict(deepcopy(saved))
    pair.step(batch)
    assert pair.graph_step.cuda_graph.capture_count == 2
    bad = list(None if v is None else v.clone() for v in batch)
    bad[0][-1, 0, 0] = float('nan')
    before = snapshot_before_skipped_step(pair.eager)
    assert pair.step(tuple(bad))[1]
    assert_equal(before, pair.eager.state_dict())
    assert not pair.step(batch)[1]
