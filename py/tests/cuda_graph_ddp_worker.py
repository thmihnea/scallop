"""Two-or-more-GPU NCCL graph replay versus eager DDP, including skip recovery."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from common import distributed, interpolant, loss_args, losses, state_utils, updates
from common.flow_map import TorchFlowMap
from test_cuda_graphs import configuration, assert_equal, snapshot_before_skipped_step


def compare(eager, graphed):
    assert eager.step == graphed.step
    for left, right in zip(eager.net.parameters(), graphed.net.parameters()):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=5e-4, atol=3e-5,
                                       equal_nan=True)
    for name, value in eager.ema_params[.999].items():
        torch.testing.assert_close(value, graphed.ema_params[.999][name], rtol=2e-5, atol=2e-6)
    for values in (dict(graphed.net.named_parameters()), graphed.ema_params[.999]):
        for value in values.values():
            reference = value.detach().clone()
            dist.broadcast(reference, src=0)
            assert_equal(value, reference)


def run_case(device, mode, family, psd=None, stopgrad='stopped', version='original_atom_dit_v1'):
    cfg = configuration(mode, family, psd, stopgrad)
    cfg.network.dit_version = version
    cfg.optimization.bs *= distributed.world_size()
    distributed.configure_batch(cfg)
    graph_cfg = deepcopy(cfg)
    graph_cfg.training.cuda_graphs = True
    torch.manual_seed(123)
    net = TorchFlowMap(cfg.network).to(device)
    with torch.no_grad():
        head = net.dit.final_linear if hasattr(net.dit, "final_linear") else net.dit.final_layer.linear
        head.weight.normal_(std=.02)
        if cfg.network.predict_divergence:
            head = net.dit.div_final_linear if hasattr(net.dit, "div_final_linear") else net.dit.div_final_layer.linear
            head.weight.normal_(std=.02)
    optimizer, schedule = state_utils.setup_optimizer(cfg, net)
    eager = state_utils.TrainState(net, optimizer, schedule, [0.999])
    graphed = deepcopy(eager)
    gradient_errors = []
    if cfg.optimization.optimizer == 'muon_adamw':
        original_step = graphed.optimizer.step
        def same_gradient_step():
            # As in ddp_worker: check independent gradients before Muon's BF16 polar iteration.
            for actual, expected in zip(graphed.net.parameters(), eager.net.parameters()):
                assert (actual.grad is None) == (expected.grad is None)
                if actual.grad is not None:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-4, atol=3e-5)
                    gradient_errors.append(float((actual.grad - expected.grad).abs().max()))
                    actual.grad.copy_(expected.grad)
            original_step()
        graphed.optimizer.step = same_gradient_step
    interp = interpolant.setup_interpolant(cfg)
    fault = torch.ones(2, device=device)

    def with_fault(fn):
        def wrapped(*args, **kwargs):
            value, metrics = fn(*args, **kwargs)
            return value * fault[1] + fault[0].log(), metrics
        return wrapped

    eager_loss = distributed.wrap_loss(net, with_fault(losses.setup_loss(cfg, net, interp)), device)
    graph_loss = distributed.wrap_loss(graphed.net,
        with_fault(losses.setup_loss(cfg, graphed.net, interp)), device, cuda_graphs=True)
    eager_update = updates.setup_train_step(cfg)
    graph_update = updates.setup_train_step(graph_cfg)
    cpu_rng = torch.Generator().manual_seed(1000 + distributed.rank())
    eager_rng = torch.Generator(device=device).manual_seed(2000 + distributed.rank())
    graph_rng = torch.Generator(device=device).manual_seed(2000 + distributed.rank())

    def batch():
        shape = (cfg.optimization.local_bs, 22, 3)
        x0, x1 = [torch.randn(shape, generator=cpu_rng).to(device) for _ in range(2)]
        times = tuple(None if t is None else t.to(device)
                      for t in loss_args.sample_times(cfg, cpu_rng))
        return x0, x1, None, *times

    def step(args, expect_skip=False):
        outputs = []
        for state, fn, update, rng in ((eager, eager_loss, eager_update, eager_rng),
                                       (graphed, graph_loss, graph_update, graph_rng)):
            before = snapshot_before_skipped_step(state) if expect_skip else None
            value, metrics = update(state, fn, None, state.self_teacher(), args, rng)
            if state.last_update_succeeded:
                state.update_ema()
            assert metrics['train/nan_detected'] == expect_skip
            if expect_skip:
                assert_equal(before, state.state_dict())
            outputs.append(value)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-5, equal_nan=True)
        assert_equal(eager_rng.get_state(), graph_rng.get_state())
        compare(eager, graphed)

    for _ in range(3):
        step(batch())
    assert graph_update.cuda_graph.capture_count == 1
    args = batch()
    for gradient_fault in (False, True):
        if distributed.rank() == distributed.world_size() - 1:
            if gradient_fault:
                fault[1] = float('nan')
            else:
                fault[0] = 0.0  # Infinite local loss, finite derivatives.
        step(args, expect_skip=True)
        fault.fill_(1.0)
    step(args)
    assert graph_update.cuda_graph.capture_count == 1
    if mode == 'dlogpdt':
        bad = list(None if t is None else t.clone() for t in args)
        if distributed.rank() == distributed.world_size() - 1:
            bad[3][0] = bad[4][0] = 1.0
        before = deepcopy(graphed.state_dict())
        try:
            graph_update(graphed, graph_loss, None, graphed.self_teacher(), tuple(bad), graph_rng)
        except ValueError as error:
            assert 'diagonal times' in str(error)
        else:
            raise AssertionError('All ranks must reject invalid density times')
        assert_equal(before, graphed.state_dict())
    checkpoint = deepcopy(eager.state_dict())
    for state in (eager, graphed):
        state.load_state_dict(deepcopy(checkpoint))
    for _ in range(2):
        step(batch())
    assert graph_update.cuda_graph.capture_count == 2
    return dict(mode=mode, family=family, psd=psd, stopgrad=stopgrad,
                accepted_steps=graphed.step, captures=2,
                rank_parameters_and_ema_exact=True, skip_recovery=True,
                checkpoint_recapture=True, rng_matches_eager=True,
                max_independent_gradient_error=max(gradient_errors, default=0.),
                optimizer_compared_with_identical_gradients=bool(gradient_errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--single_rank', action='store_true', help='one-GPU NCCL capture smoke only')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = distributed.initialize('cuda', cuda_graphs=True)
    if args.single_rank:
        assert not distributed.active()
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '0'
        rendezvous = Path(args.output).resolve().with_suffix('.rendezvous')
        dist.init_process_group('nccl', init_method=rendezvous.as_uri(),
                                rank=0, world_size=1, device_id=device)
    try:
        results = []
        cases = [
            ('pretrain_v', 'lsd', None, 'stopped'),
            ('pretrain_flowmap', 'lsd', None, 'convex'),
            ('pretrain_flowmap', 'esd', None, 'stopped'),
            ('pretrain_flowmap', 'psd', 'uniform', 'stopped'),
            ('pretrain_flowmap', 'psd', 'midpoint', 'stopped'),
            ('dlogpdt', 'lsd', None, 'convex'),
            ('dlogpdt', 'esd_cond', None, 'stopped'),
            ('dlogpdt', 'psd', 'uniform', 'stopped'),
            ('hutchinson', 'lsd', None, 'stopped'),
        ]
        cases = [(*case, 'original_atom_dit_v1') for case in cases]
        cases += [(mode, 'lsd', None, 'convex', 'canonical_atom_dit_v2')
                  for mode in ('pretrain_v', 'dlogpdt', 'hutchinson')]
        for mode, family, psd, stopgrad, version in cases:
            result = run_case(device, mode, family, psd, stopgrad, version)
            result['version'] = version
            results.append(result)
            if distributed.is_main():
                print(json.dumps(result), flush=True)
                Path(args.output).write_text(json.dumps(dict(world_size=distributed.world_size(),
                    nccl=torch.cuda.nccl.version(), results=results), indent=2) + '\n')
            gc.collect()
            torch.cuda.empty_cache()
        if distributed.is_main():
            print('CUDA GRAPH DDP CORRECTNESS PASSED', flush=True)
    except BaseException:
        # Pending NCCL work can deadlock teardown after an assertion failure.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
