"""Independent density/weight oracles and entry-point coverage for the CoM lift."""
import importlib
import math
from types import SimpleNamespace

import pytest
import torch

from common.com_lift import com_log_prob, importance_source_energy
from common.flow_map import TorchFlowMap


@pytest.mark.parametrize("n", [5, 22, 33, 42, 63])
@pytest.mark.parametrize("scale", [1.0, 0.1629616767168045])
def test_lift_cancels_exactly_against_mean_free_density(n, scale):
    gen = torch.Generator().manual_seed(20)
    x = torch.randn(128, n, 3, generator=gen, dtype=torch.float64) * scale
    c = x.mean(1, keepdim=True)
    y = x - c
    logq_full = torch.distributions.Normal(0.0, scale).log_prob(x).sum((1, 2))
    logq_shape = -0.5 * (y / scale).square().sum((1, 2)) - 1.5 * (n - 1) * math.log(2 * math.pi * scale**2)
    # An independent translation-invariant target, different from the proposal.
    energy = 0.75 * (y / scale).square().sum((1, 2))
    full = importance_source_energy(x, logq_full, com_augmentation=True, unscale_factor=scale) - energy
    shape = -logq_shape - energy
    torch.testing.assert_close(full, shape + 1.5 * math.log(n))
    torch.testing.assert_close(full.softmax(0), shape.softmax(0))
    torch.testing.assert_close(1 / full.softmax(0).square().sum(), 1 / shape.softmax(0).square().sum())
    # The same computation in standardized units differs only by the shape Jacobian.
    standard = importance_source_energy(x / scale, logq_full + 3 * n * math.log(scale),
                                      com_augmentation=True)
    torch.testing.assert_close(full + energy - standard,
                               torch.full_like(full, 3 * (n - 1) * math.log(scale)))
    torch.testing.assert_close(importance_source_energy(x, logq_full), -logq_full, rtol=0, atol=0)


def test_zero_centroid_is_finite_and_matches_cartesian_distribution():
    x = torch.zeros(3, 22, 3, dtype=torch.float64)
    x[1] += torch.tensor([0.3, -0.2, 0.1])
    scale = 0.25
    oracle = torch.distributions.Normal(0.0, scale / math.sqrt(22)).log_prob(x.mean(1)).sum(-1)
    torch.testing.assert_close(com_log_prob(x, scale), oracle)
    assert torch.isfinite(com_log_prob(x, scale)).all()


@pytest.mark.parametrize("module,parser,required", [
    ("launchers.learn", "parse_command_line_arguments", []),
    ("launchers.eval_peptide_ts", "parse_args", ["--cfg_path", "configs.ace_a_nme", "--ckpt_path", "unused.pt", "--out_dir", "/tmp/com-lift-eval"]),
    ("launchers.eval_ode_int", "parse_args", ["--ckpt_path", "unused.pt", "--out_dir", "/tmp/com-lift-ode"]),
])
def test_cli_lift_defaults_on_with_explicit_mean_free_override(monkeypatch, module, parser, required):
    parse = getattr(importlib.import_module(module), parser)
    for option, expected in [(None, True), ("--com_lift", True), ("--com_augmentation", True),
                             ("--no_com_lift", False), ("--no_com_augmentation", False)]:
        monkeypatch.setattr("sys.argv", [module, *required, *([option] if option else [])])
        assert parse().com_augmentation is expected


@pytest.mark.parametrize("mode", ["dlogpdt", "hutchinson"])
@pytest.mark.parametrize("lift", [False, True])
def test_actual_flowmap_snis_cancels_centroid_once(monkeypatch, mode, lift):
    pytest.importorskip("transferable_samplers")
    pytest.importorskip("mdtraj")
    from common import ts_eval
    from common.tsx import TargetEnergy, SourceEnergy
    cfg = importlib.import_module("configs.ace_a_nme").get_config(0)
    cfg.network.divergence_mode = mode
    cfg.network.predict_divergence = True
    cfg.network.vectorized_divergence = mode == "dlogpdt"
    cfg.network.divergence_rescale = 1.0
    cfg.network.com_aware = lift
    cfg.network.unscale_factor = 0.25
    cfg.network.dit_kwargs = dict(d_model=24, num_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=16)
    cfg.logging.plot_bs = 32
    cfg.logging.sample_batch_size = 7  # multiple batches including a short last one
    cfg.logging.n_sampling_steps = [2]
    cfg.logging.importance_weight_clip = 0.0
    cfg.logging.do_plots = False
    net = TorchFlowMap(cfg.network)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()  # exact identity flow with zero divergence
    state = SimpleNamespace(net=net, ema_facs=[], ema_params={})
    scale = cfg.network.unscale_factor
    def energy(x):
        y = x - x.mean(1, keepdim=True)
        return 0.5 * (y / scale).square().sum((1, 2))
    norm = torch.tensor(scale if lift else 1.0)
    ctx = SimpleNamespace(target_energy=TargetEnergy(energy, norm), normalization_std=norm)
    captured = {}
    class Evaluator:
        def __init__(self, **kwargs): pass
        def evaluate(self, data, ctx, **kwargs):
            captured.update(data)
            return {}
    def forbidden(*args, **kwargs):
        raise AssertionError("Upstream radial CoM correction must never be invoked")
    monkeypatch.setattr(SourceEnergy, "com_energy_adjustment", forbidden)
    monkeypatch.setattr(ts_eval, "PeptideEnsembleEvaluator", Evaluator)
    monkeypatch.setattr(ts_eval, "_wandb_active", lambda: False)
    ts_eval.ts_evaluate(cfg, state, torch.device("cpu"), 1, ctx, com_augmentation=lift)
    expected = 1.5 * 21 * math.log(2 * math.pi * scale**2)
    if lift:
        expected += 1.5 * math.log(22)
    for name in ["proposal", "resampled"]:
        logw = captured[name].logw
        torch.testing.assert_close(logw, torch.full_like(logw, expected), atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(logw.softmax(0), torch.full_like(logw, 1 / len(logw)))
    centroid = captured["proposal"].samples.mean(1)
    assert bool(centroid.abs().max() > 0.01) is lift


@pytest.mark.parametrize("entry,mode", [("eval_peptide_ts", "dlogpdt"),
                                       ("eval_peptide_ts", "hutchinson"),
                                       ("eval_ode_int", "pretrain_v")])
@pytest.mark.parametrize("lift", [False, True])
def test_posthoc_entrypoint_prior_units_and_final_correction(monkeypatch, tmp_path, entry, mode, lift):
    pytest.importorskip("transferable_samplers")
    pytest.importorskip("mdtraj")
    from common import com_lift, datasets, state_utils, ts_systems
    launcher = importlib.import_module(f"launchers.{entry}")
    scale = 0.25
    class Energy:
        closed = False
        def numpy(self, x):
            return (0.5 * ((x - x.mean(1, keepdims=True)) / scale)**2).sum((1, 2))
        def close(self): self.closed = True
    energy = Energy()
    monkeypatch.setattr(ts_systems, "ParallelEnergy", lambda *a, **kw: energy)
    monkeypatch.setattr(ts_systems, "ensure_data", lambda *a, **kw: {"source": "analytic-test"})
    monkeypatch.setattr(ts_systems, "train_std", lambda *a, **kw: scale)
    setup_target = datasets.setup_target
    monkeypatch.setattr(datasets, "setup_target", lambda cfg, **kw: setup_target(cfg, train_std=scale))
    def setup(cfg, *args, **kwargs):
        assert cfg.problem.com_augmentation is lift and cfg.network.com_aware is lift
        cfg.network.dit_kwargs = dict(d_model=24, num_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=16)
        cfg.network.divergence_rescale = 1.0
        net = TorchFlowMap(cfg.network)
        with torch.no_grad():
            for p in net.parameters(): p.zero_()
        return net, 0, None
    monkeypatch.setattr(state_utils, "load_evaluation_model", setup)
    if entry == "eval_ode_int":
        monkeypatch.setattr(launcher, "_load_net", lambda args, cfg, device: (setup(cfg)[0], "raw", 0))
    class Checked(Exception): pass
    original = com_lift.importance_source_energy
    def check(x, logp, **kw):
        assert kw["com_augmentation"] is lift and kw["unscale_factor"] == scale
        assert bool(x.mean(1).abs().max() > 0.01) is lift
        corrected = original(x, logp, **kw)
        logw = corrected - torch.from_numpy(energy.numpy(x.numpy()))
        expected = 1.5 * 21 * math.log(2 * math.pi * scale**2) + (1.5 * math.log(22) if lift else 0)
        torch.testing.assert_close(logw, torch.full_like(logw, expected), atol=2e-5, rtol=2e-6)
        raise Checked
    monkeypatch.setattr(com_lift, "importance_source_energy", check)
    argv = [entry, "--cfg_path", "configs.ace_a_nme", "--ckpt_path", "ace_a_nme.pt",
            "--out_dir", str(tmp_path), "--device", "cpu", "--no_ema", "--no_plots",
            "--n_samples", "8", "--batch_size", "3", "--steps", "2"]
    if entry == "eval_peptide_ts": argv += ["--mode", mode]
    else: argv += ["--solver", "rk4", "--divergence_type", "hutchinson"]
    if not lift: argv += ["--no_com_augmentation"]  # lifted path exercises the default
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(Checked): launcher.main()
    assert energy.closed
