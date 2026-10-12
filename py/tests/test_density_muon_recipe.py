"""Density recipe precedence and ESD's live-prediction/stopped-derivative boundary."""
from types import SimpleNamespace

import pytest
import torch

from common import interpolant, losses
from launchers import learn
from test_stopped_flowmap import PolynomialMap, batch, snapshots


def resolve(monkeypatch, *extra):
    monkeypatch.setattr("sys.argv", ["learn.py", "--cfg_path", "configs.ace_a_nme",
        "--mode", "dlogpdt", "--training_recipe", "branched_muon", *extra])
    return learn.setup_config_dict(learn.parse_command_line_arguments())


@pytest.mark.parametrize("family,psd", [("lsd", None), ("esd", None),
                                      ("psd", "uniform"), ("psd", "midpoint")])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_recipe_selects_complete_density_stage(family, psd, device, monkeypatch):
    extra = ["--loss_type", family, "--device", device]
    if psd:
        extra += ["--psd_type", psd]
    cfg = resolve(monkeypatch, *extra)
    assert cfg.optimization.optimizer == "muon_adamw"
    assert cfg.optimization.decay_steps == 50000
    assert cfg.optimization.learning_rate == 1e-4 and cfg.optimization.bs == 1024
    assert cfg.optimization.schedule_type == "sqrt"
    assert cfg.optimization.warmup_fraction == 0
    assert cfg.optimization.weight_decay == 1e-4
    assert cfg.training.cuda_graphs == device.startswith("cuda")
    assert cfg.training.offdiag_teacher_source == ("ema" if family == "psd" else "external")
    assert cfg.training.diag_teacher_source == "external"
    assert cfg.training.density_ema_fac == .999
    assert cfg.training.offdiag_time_order == "forward"
    assert cfg.training.stopgrad_type == "stopped"
    assert cfg.network.vectorized_divergence
    assert cfg.network.div_loss_weighting_type == "linear"
    assert cfg.network.divergence_rescale == 1.
    assert cfg.network.dit_kwargs.num_shared_layers == 4
    assert cfg.network.dit_kwargs.num_div_layers == 2
    assert cfg.network.reset_optimizer
    assert cfg.logging.output_name.endswith("_branched_muon_ts_comaug")


def test_explicit_overrides_win_and_density_recipe_stays_optional(monkeypatch):
    cfg = resolve(monkeypatch, "--loss_type", "esd", "--device", "cuda",
        "--optimizer", "radam", "--decay_steps", "10000", "--no-cuda_graphs",
        "--offdiag_teacher_source", "self", "--offdiag_time_order", "square",
        "--no-vectorized_divergence", "--div_loss_weighting_type", "square",
        "--stopgrad_type", "convex")
    assert cfg.optimization.optimizer == "radam" and cfg.optimization.decay_steps == 10000
    assert not cfg.training.cuda_graphs and not cfg.network.vectorized_divergence
    assert cfg.network.div_loss_weighting_type == "square"
    assert cfg.network.divergence_rescale == cfg.problem.d
    assert cfg.training.offdiag_teacher_source == "self"
    assert cfg.training.offdiag_time_order == "square"
    assert cfg.training.stopgrad_type == "convex"
    monkeypatch.setattr("sys.argv", ["learn.py", "--cfg_path", "configs.ace_a_nme",
                                    "--mode", "dlogpdt", "--device", "cpu"])
    old = learn.setup_config_dict(learn.parse_command_line_arguments())
    assert old.optimization.optimizer == "muon_adamw"
    assert old.optimization.decay_steps == 50000
    assert old.network.vectorized_divergence
    assert old.network.div_loss_weighting_type == "linear"
    assert not old.training.cuda_graphs


@pytest.mark.parametrize("extra", [("--mode", "pretrain_flowmap"),
                                   ("--loss_type", "esd_cond")])
def test_recipe_rejects_unintended_objectives(extra, monkeypatch):
    with pytest.raises(ValueError, match="branched_muon requires"):
        resolve(monkeypatch, *extra)


class AnchoredMap(PolynomialMap):
    """At one batch, derivative coefficients have zero direct prediction effect.

    A leaked correction gradient changes p[1:]/q[1:], while ordinary prediction
    regression can only change p[0]/q[0]. The gap coefficient separately probes
    the destination derivative and its minus sign in a fixed-t source JVP.
    """
    def __init__(self, x, s, t, vectorized):
        super().__init__(vectorized=vectorized)
        self.register_buffer("anchor_x", x.clone())
        self.register_buffer("anchor_s", s[:, None, None].clone())
        self.register_buffer("anchor_gap", (t-s)[:, None, None].clone())

    def calc_phi(self, s, t, x, train=False, return_div=False, **kwargs):
        ss, gap = s[:, None, None], (t-s)[:, None, None]
        basis = (torch.ones_like(x), x-self.anchor_x,
                 (ss-self.anchor_s).expand_as(x), (gap-self.anchor_gap).expand_as(x))
        u = sum(p*b for p, b in zip(self.p, basis))
        d = sum(q*b for q, b in zip(self.q, basis))
        if not self.config.vectorized_divergence:
            d = d.sum((1, 2))
        return (u, d) if return_div else u


@pytest.mark.parametrize("policy", ["stopped", "convex"])
@pytest.mark.parametrize("vectorized", [False, True])
def test_esd_prediction_live_and_spatial_source_destination_derivatives_stopped(policy, vectorized):
    x0, x1, s, t, _ = batch()
    interp = interpolant.setup_interpolant(SimpleNamespace(problem=SimpleNamespace(interp_type="linear")))
    x = interp.calc_It(s, x0, x1)
    net = AnchoredMap(x, s, t, vectorized).to(x.device)
    teacher, ema = snapshots(net)
    # All derivative values must be nonzero, or a stopped-path check is vacuous.
    for ds, dt, dx in [(torch.ones_like(s), torch.zeros_like(t), torch.zeros_like(x)),
                       (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(x)),
                       (torch.zeros_like(s), torch.zeros_like(t), torch.ones_like(x))]:
        _, tangent = torch.func.jvp(lambda ss, tt, xx: net.calc_phi(ss, tt, xx, return_div=True),
                                   (s, t, x), (ds, dt, dx))
        assert all(v.abs().min() > .01 for v in tangent)
    value, _ = losses.esd_term(net, teacher, x0, x1, None, s, t, interp=interp,
                              density_ema_params=ema, stopgrad_type=policy)
    grads = torch.autograd.grad(value.sum(), (net.p, net.q, *teacher.values(), *ema.values()),
                                allow_unused=True)
    for g in grads[:2]:
        assert g[0].abs() > .01, "ordinary student prediction lost its gradient"
        torch.testing.assert_close(g[1:], torch.zeros_like(g[1:]), rtol=0, atol=1e-12)
    assert all(g is None for g in grads[2:]), "teacher or density EMA is not detached"
    # Negative control: differentiating the material correction must fail the
    # derivative-only parameter check above, despite identical forward values.
    with torch.no_grad():
        b = losses._apply(net, teacher, "calc_b", s, x, train=False)
    primal, tangent = losses._raw_jvp(net, s, t, x, b)
    wrong = (primal[0]-b-(t-s)[:, None, None]*tangent[0]).square().sum()
    leaked = torch.autograd.grad(wrong, net.p)[0]
    assert bool((leaked[1:].abs() > .01).all())


@pytest.mark.parametrize('resume', [False, True])
def test_recipe_fresh_stage_resets_best_metrics_but_resume_keeps_them(resume, monkeypatch, tmp_path):
    from common import state_utils
    from common.flow_map import TorchFlowMap
    cfg = resolve(monkeypatch, '--loss_type', 'esd', '--device', 'cpu')
    cfg.network.dit_kwargs = dict(d_model=12, num_layers=2, num_shared_layers=1,
        num_div_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=8)
    net = TorchFlowMap(cfg.network)
    opt, schedule = state_utils.setup_optimizer(cfg, net)
    original = state_utils.TrainState(net, opt, schedule, [0.999])
    original.step = 17
    original.best_metrics = {'ess': .34}
    path = tmp_path/'parent.pt'
    state_utils.save_state(original, str(path))
    cfg.network.load_path = str(path)
    cfg.network.reset_optimizer = not resume
    restored, _, _ = state_utils.setup_training_state(cfg, 'cpu')
    assert restored.step == (17 if resume else 0)
    assert restored.best_metrics == ({'ess': .34} if resume else {})
    for key, value in original.net.state_dict().items():
        torch.testing.assert_close(restored.net.state_dict()[key], value, rtol=0, atol=0)
