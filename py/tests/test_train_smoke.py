"""Real TS data, training, evaluation and native checkpoint smoke tests.

Uses cached official splits only (no downloads in pytest). Defaults to CPU and
Ace-A-Nme. Set IF2D2_TEST_DEVICE=cuda and IF2D2_TEST_TARGET to run on a chosen GPU
and peptide; CUDA_VISIBLE_DEVICES selects the physical GPU.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transferable_samplers")
pytest.importorskip("mdtraj")
pytest.importorskip("openmm")

from common import logging, loss_args, state_utils, ts_eval, ts_systems, updates
from launchers import learn

TARGET = os.environ.get("IF2D2_TEST_TARGET", "ace_a_nme")
DEVICE = os.environ.get("IF2D2_TEST_DEVICE", "cpu")


def _make_cfg(mode, tmp_path, evaluate=True):
    cfg = importlib.import_module(f"configs.{TARGET}").get_config(0, output_folder=str(tmp_path))
    cfg.optimization.bs = int(os.environ.get("IF2D2_TEST_BATCH_SIZE", "8"))
    cfg.optimization.total_steps = 3
    cfg.optimization.diag_fraction = 1.0 if mode == "pretrain_v" else 0.5
    if os.environ.get("IF2D2_TEST_FULL_DIT") != "1":
        cfg.network.dit_kwargs = dict(d_model=32, num_layers=3, num_shared_layers=2, num_div_layers=1, nhead=4, mlp_ratio=2, time_embedding_dim=16)
    cfg.network.load_path = ""
    cfg.teacher.load_path = ""
    cfg.network.divergence_mode = mode
    cfg.network.from_scratch = True
    cfg.network.predict_divergence = mode not in ("pretrain_v", "pretrain_flowmap")
    cfg.network.vectorized_divergence = mode == "dlogpdt"
    cfg.network.divergence_rescale = 1.0 if cfg.network.vectorized_divergence else float(cfg.problem.d)
    cfg.training.time_distr = "uniform"
    cfg.logging.visual_freq = 1 if evaluate else 0
    cfg.logging.save_freq = 0
    cfg.logging.plot_bs = 128
    cfg.logging.num_tica = 1000
    cfg.logging.do_plots = False
    cfg.logging.n_sampling_steps = [2]
    cfg.logging.output_name = f"{TARGET}_{mode}"
    return cfg


@pytest.fixture(autouse=True)
def cached_data(monkeypatch):
    cfg = importlib.import_module(f"configs.{TARGET}").get_config(0)
    system = ts_systems.resolve(cfg)
    paths = ts_systems.data_paths(system)
    if not all(Path(paths[k]).exists() for k in ("train", "val", "test", "pdb")):
        pytest.skip(f"Official TS data not cached for {system.sequence}")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    monkeypatch.setattr(logging, "_wandb_active", lambda: False)
    monkeypatch.setattr(ts_eval, "_wandb_active", lambda: False)


@pytest.mark.parametrize("mode,family,psd", [
    ("pretrain_v", "lsd", None),
    ("pretrain_flowmap", "esd_imf", None),
    *[(mode, family, psd)
      for mode in ("pretrain_flowmap", "dlogpdt", "hutchinson")
      for family, psd in (("lsd", None), ("esd", None), ("esd_cond", None),
          ("esd_cond_div_only", None), ("psd", "uniform"), ("psd", "midpoint"))
      if mode != "pretrain_flowmap" or family != "esd_cond_div_only"],
])
def test_train_and_upstream_evaluation(mode, family, psd, tmp_path, monkeypatch, stopgrad='stopped'):
    cfg = _make_cfg(mode, tmp_path)
    cfg.training.loss_type = family
    cfg.training.stopgrad_type = stopgrad
    if family == "esd_imf":
        cfg.training.diag_teacher_source = "conditional"
        cfg.training.offdiag_teacher_source = "self"
    with cfg.training.ignore_type():
        cfg.training.psd_type = psd
    if family in ('esd_cond', 'esd_cond_div_only') or (mode == 'dlogpdt' and stopgrad == 'convex'):
        cfg.network.div_loss_weighting_type = 'linear'
    cfg.logging.output_name += f"_{family}_{psd}_{stopgrad}"
    torch.manual_seed(123)
    device = torch.device(DEVICE)
    cfg, statics, state = learn.setup_state(cfg, device, seed=123)
    assert not hasattr(statics, "distr")
    assert state.ts_eval_ctx.topology.n_atoms == cfg.network.n_particles
    before = {name: p.detach().clone() for name, p in state.net.named_parameters()}
    train_step = updates.setup_train_step(cfg)
    update_ema = updates.setup_ema_update(cfg)
    cpu_rng = torch.Generator().manual_seed(321)
    device_rng = torch.Generator(device=device).manual_seed(456)
    seen = []
    for step in range(3):
        args = loss_args.get_loss_fn_args(cfg, statics, cpu_rng, device)
        teacher = state.self_teacher()
        value, metrics = train_step(state, statics.loss, teacher, teacher, args, device_rng)
        state = update_ema(state)
        assert torch.isfinite(value), (mode, step, value)
        seen.append(float(value))
    assert any(not torch.equal(before[n], p) for n, p in state.net.named_parameters())

    # Run the actual upstream evaluator through the normal logging entry point.
    eval_metrics = {}
    original = ts_eval.ts_evaluate
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        eval_metrics.update(result)
        return result
    monkeypatch.setattr(ts_eval, "ts_evaluate", record)
    logging.log_metrics(cfg, state, device, value, metrics, step=2, elapsed=0.0)
    prefix = f"val/{ts_systems.resolve(cfg).sequence}"
    for name in ("energy-w2", "torus-w2", "tica-w2"):
        key = f"{prefix}/proposal/{name}"
        assert key in eval_metrics, sorted(eval_metrics)
        assert torch.isfinite(torch.tensor(eval_metrics[key])), (key, eval_metrics[key])
    if cfg.network.predict_divergence:
        ess = eval_metrics[f"{prefix}/resampled/effective-sample-size"]
        assert 0 < ess <= 1.0 + 1e-6
    else:
        assert not any("effective-sample-size" in k or "/resampled/" in k for k in eval_metrics)
        assert "proposal_energy_w2" in state.best_metrics
        assert "ess" not in state.best_metrics
    print(f"TS smoke {TARGET} {mode} {family}/{psd} {device}: losses={seen}; "
          f"proposal_energy_w2={eval_metrics[f'{prefix}/proposal/energy-w2']}; "
          f"NFE={eval_metrics[f'{prefix}/sampling/nfe']}")


def test_native_pretraining_checkpoint_handoff(tmp_path):
    cfg = _make_cfg("pretrain_v", tmp_path, evaluate=False)
    device = torch.device(DEVICE)
    _, _, first = learn.setup_state(cfg, device, seed=3)
    with torch.no_grad():
        head = first.net.dit.final_linear if hasattr(first.net.dit, "final_linear") else first.net.dit.final_layer.linear
        head.weight.normal_(std=0.01)
    path = str(tmp_path / "velocity.pt")
    state_utils.save_state(first, path)
    next_cfg = _make_cfg("pretrain_flowmap", tmp_path, evaluate=False)
    next_cfg.network.load_path = path
    next_cfg.teacher.load_path = path
    next_cfg.teacher.ema_fac = None
    _, statics, second = learn.setup_state(next_cfg, device, seed=4)
    assert statics.teacher_params is not None
    for name, param in first.net.named_parameters():
        torch.testing.assert_close(second.net.state_dict()[name], param)
        torch.testing.assert_close(statics.teacher_params[name], param)


@pytest.mark.parametrize("mode", ["dlogpdt", "hutchinson"])
def test_mean_free_train_and_upstream_evaluation(mode, tmp_path, monkeypatch):
    original = _make_cfg
    def mean_free(*args, **kwargs):
        cfg = original(*args, **kwargs)
        cfg.problem.com_augmentation = False
        cfg.network.com_aware = False
        return cfg
    monkeypatch.setattr(sys.modules[__name__], "_make_cfg", mean_free)
    test_train_and_upstream_evaluation(mode, "lsd", None, tmp_path, monkeypatch)


@pytest.mark.parametrize("mode", ["pretrain_flowmap", "dlogpdt", "hutchinson"])
def test_convex_lsd_train_and_upstream_evaluation(mode, tmp_path, monkeypatch):
    test_train_and_upstream_evaluation(mode, "lsd", None, tmp_path, monkeypatch, stopgrad="convex")
