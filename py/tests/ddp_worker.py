"""Multi-process correctness probe; run with torchrun (CPU/Gloo or CUDA/NCCL)."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from configs.ace_a_nme import get_config
from common import distributed, interpolant, losses, loss_args, state_utils, updates
from common.flow_map import TorchFlowMap


def packed(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def assert_replicated(value):
    reference = value.clone()
    dist.broadcast(reference, src=0)
    torch.testing.assert_close(value, reference, rtol=0, atol=0)


def probe(device):
    results = []
    size, rank = distributed.world_size(), distributed.rank()
    stopped_cases = [("lsd", "pretrain_v", None),
                              ("lsd", "pretrain_flowmap", None),
                              ("esd", "pretrain_flowmap", None),
                              ("psd", "pretrain_flowmap", "uniform"),
                              ("psd", "pretrain_flowmap", "midpoint"),
                              ("lsd", "hutchinson", None),
                              ("esd", "dlogpdt", None),
                              ("psd", "dlogpdt", "uniform"),
                              ("esd_cond", "dlogpdt", None),
                              ("esd_cond_div_only", "hutchinson", None)]
    cases = [(family, mode, psd, "stopped") for family, mode, psd in stopped_cases]
    cases += [("lsd", mode, None, "convex")
              for mode in ("pretrain_flowmap", "dlogpdt", "hutchinson")]
    for family, mode, psd, stopgrad in cases:
        cfg = get_config(0)
        cfg.network.dit_kwargs = dict(d_model=32, num_layers=3, num_shared_layers=2, num_div_layers=1, nhead=4,
                                     mlp_ratio=2, time_embedding_dim=16)
        cfg.network.predict_divergence = mode not in ("pretrain_v", "pretrain_flowmap")
        cfg.network.divergence_mode = mode
        cfg.network.vectorized_divergence = mode == "dlogpdt"
        cfg.network.divergence_rescale = 1.0
        if family in ("esd_cond", "esd_cond_div_only") or (mode == "dlogpdt" and stopgrad == "convex"):
            cfg.network.div_loss_weighting_type = "linear"
        cfg.network.from_scratch = mode == "pretrain_v"
        cfg.training.loss_type = family
        cfg.training.psd_type = psd
        cfg.training.stopgrad_type = stopgrad
        cfg.training.time_distr = "uniform"
        cfg.optimization.bs = 8 * size
        cfg.optimization.diag_fraction = 1.0 if mode == "pretrain_v" else 0.5
        cfg.optimization.clip = 10.0
        global_cfg = deepcopy(cfg)
        distributed.configure_batch(cfg)
        torch.manual_seed(123)
        net = TorchFlowMap(cfg.network).to(device)
        with torch.no_grad():
            net.dit.final_linear.weight.normal_(std=0.02)
            if cfg.network.predict_divergence:
                net.dit.div_final_linear.weight.normal_(std=0.02)
        reference = deepcopy(net)
        opt, schedule = state_utils.setup_optimizer(cfg, net)
        ref_opt, _ = state_utils.setup_optimizer(global_cfg, reference)
        state = state_utils.TrainState(net, opt, schedule, [0.999])
        ref_state = state_utils.TrainState(reference, ref_opt, schedule, [0.999])
        interp = interpolant.setup_interpolant(cfg)
        raw_loss = losses.setup_loss(cfg, net, interp)
        ref_loss = losses.setup_loss(global_cfg, reference, interp)
        ddp_loss = distributed.wrap_loss(net, raw_loss, device)
        train_step = updates.setup_train_step(cfg)
        ema_update = updates.setup_ema_update(cfg)
        fixed_teacher = {k: v.detach().clone() for k, v in net.named_parameters()}
        cpu_rng = torch.Generator().manual_seed(1000 + rank)
        dev_rng = torch.Generator(device=device).manual_seed(2000 + rank)
        max_grad_error = 0.0
        max_param_error = 0.0
        for step in range(3):
            local_n = cfg.optimization.local_bs
            x0 = torch.randn(local_n, 22, 3, generator=cpu_rng).to(device)
            x1 = torch.randn(local_n, 22, 3, generator=cpu_rng).to(device)
            x0 -= x0.mean(1, keepdim=True)
            x1 -= x1.mean(1, keepdim=True)
            times = [None if v is None else v.to(device)
                     for v in loss_args.sample_times(cfg, cpu_rng)]
            args = (x0, x1, None, *times)
            diag_n = cfg.optimization.local_diag_bs
            def gather(value):
                if value is None:
                    return None
                values = [torch.empty_like(value) for _ in range(size)]
                dist.all_gather(values, value)
                return torch.cat([v[:diag_n] for v in values] + [v[diag_n:] for v in values])
            full_args = tuple(gather(v) for v in args)
            # Fixed probes let Hutchinson use the same samples in both paths.
            diag_eps = torch.randn_like(x0[:diag_n])
            torch.randn_like(x0[diag_n:])  # Preserve the historical RNG stream after retiring offdiag_eps.
            def gather_eps(value):
                values = [torch.empty_like(value) for _ in range(size)]
                dist.all_gather(values, value)
                return torch.cat(values)
            full_diag_eps = gather_eps(diag_eps)
            def wrapped(*a, **kw):
                return ddp_loss(*a, **kw, diag_eps=diag_eps)
            value, metrics = train_step(state, wrapped, fixed_teacher,
                                       state.self_teacher(), args, dev_rng)
            ema_update(state)
            mean_value, mean_metrics = distributed.mean_metrics(value, metrics)
            ref_opt.zero_grad(set_to_none=True)
            expected, expected_metrics = ref_loss(fixed_teacher, ref_state.self_teacher(),
                *full_args, diag_eps=full_diag_eps,
                density_ema_params=ref_state.ema_teacher())
            expected.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 10.0)
            torch.testing.assert_close(mean_value, expected, rtol=2e-5, atol=2e-5)
            for key, expected_metric in expected_metrics.items():
                torch.testing.assert_close(mean_metrics[key], expected_metric, rtol=3e-5, atol=3e-5)
            for actual, wanted in zip(net.parameters(), reference.parameters()):
                assert (actual.grad is None) == (wanted.grad is None)
                if actual.grad is not None:
                    max_grad_error = max(max_grad_error, float((actual.grad-wanted.grad).abs().max()))
                    torch.testing.assert_close(actual.grad, wanted.grad, rtol=5e-4, atol=3e-5)
            if cfg.optimization.optimizer == 'muon_adamw':
                # The independent global-batch gradients were checked above.
                # Muon's BF16 polar iteration amplifies tiny batch-reduction
                # roundoff; compare optimizer/EMA using identical reduced grads.
                for actual, wanted in zip(net.parameters(), reference.parameters()):
                    wanted.grad = None if actual.grad is None else actual.grad.clone()
            for group in ref_opt.param_groups:
                group['lr'] = schedule(ref_state.step) * group.get('lr_multiplier', 1.)
            ref_opt.step()
            ref_state.step += 1
            ref_state.update_ema()
            max_param_error = max(max_param_error, float((packed(net)-packed(reference)).abs().max()))
            torch.testing.assert_close(packed(net), packed(reference), rtol=1e-5, atol=2e-6)
            for key in state.ema_params[0.999]:
                torch.testing.assert_close(state.ema_params[0.999][key], ref_state.ema_params[0.999][key],
                                           rtol=1e-5, atol=2e-6)
            assert_replicated(packed(net))
            assert_replicated(torch.cat([v.flatten() for v in state.ema_params[0.999].values()]))
        # One rank has an infinite loss with FINITE derivatives. Everyone must skip.
        before = packed(net).clone()
        ema_before = deepcopy(state.ema_params)
        opt_before = deepcopy(dict(opt.state))
        for poison_gradient in (False, True):
            def poisoned(*a, **kw):
                val, metrics = ddp_loss(*a, **kw)
                if rank == size - 1:
                    val = val * float("nan") if poison_gradient else val + float("inf")
                return val, metrics
            _, flags = train_step(state, poisoned, fixed_teacher, state.self_teacher(), args, dev_rng)
            ema_update(state)
            assert flags['train/nan_detected'] and state.step == 3
            torch.testing.assert_close(packed(net), before, rtol=0, atol=0)
            for key in ema_before[0.999]:
                torch.testing.assert_close(state.ema_params[0.999][key], ema_before[0.999][key], rtol=0, atol=0)
            # Compare ordered state entries, independent of optimizer grouping.
            from test_cuda_graphs import assert_equal
            assert_equal(list(opt_before.values()), list(opt.state.values()))
        value, _ = train_step(state, ddp_loss, fixed_teacher, state.self_teacher(), args, dev_rng)
        assert state.step == 4 and torch.isfinite(value)
        assert_replicated(packed(net))
        results.append(dict(family=family, mode=mode, psd=psd, stopgrad=stopgrad, steps=state.step,
                            max_gradient_abs_error=max_grad_error,
                            max_parameter_abs_error=max_param_error))
        if family == 'lsd' and mode == 'hutchinson' and stopgrad == 'convex':
            from types import SimpleNamespace
            from unittest.mock import patch
            from launchers import learn
            cfg.optimization.total_steps = state.step + 1
            cfg.logging.update(log_freq=1, visual_freq=0, save_freq=0)
            cfg.training.diag_teacher_source = cfg.training.offdiag_teacher_source = 'external'
            statics = SimpleNamespace(loss=poisoned, teacher_params=fixed_teacher)
            ema_before = deepcopy(state.ema_params)
            with patch.object(learn.loss_args, 'get_loss_fn_args', return_value=args), \
                 patch.object(learn.logging, 'register_signal_handlers'), \
                 patch.object(learn.logging, 'log_metrics'), \
                 patch.object(learn.logging, 'save_state') as save:
                failed = 0
                try:
                    learn.train_loop(cfg, statics, state, device, 0)
                except RuntimeError as exc:
                    assert f'optimizer step 4, requested 5' in str(exc)
                    failed = 1
                failures = torch.tensor(failed, device=device)
                dist.all_reduce(failures)
                assert int(failures) == size
                assert not save.called and state.step == 4
                assert_equal(ema_before, state.ema_params)
            results[-1]['incomplete_on_all_ranks'] = True
        del ddp_loss
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = distributed.initialize(args.device)
    try:
        result = probe(device)
        if distributed.is_main():
            Path(args.output).write_text(json.dumps(dict(world_size=distributed.world_size(),
                                                        device=str(device), results=result), indent=2))
            print('DDP CORRECTNESS PASSED', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
