"""Direct Boltzmann MALA, FALCON grids, and the TS post-hoc pipeline."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.mala import mala_sample
from common.time_discretization import TIME_DISCRETIZATION_TYPES, time_grid
from launchers import eval_peptide_ts as launcher
from common.sample_stages import ENERGY_CUTOFFS_KT, sample_stages, resampling_indices


def gaussian(x):
    return -x.square().flatten(1).sum(1) / 2, -x


@pytest.mark.parametrize("kind", TIME_DISCRETIZATION_TYPES)
def test_paper_grids_and_both_density_directions(kind):
    from _equiv_utils import build_network_config
    from common.flow_map import TorchFlowMap

    i = np.arange(5)
    noise_time = {
        "linear": 1 - i / 4,
        "geometric": (2.0 ** (4 - i) - 1) / 15,
        "cosine": np.cos(np.pi * i / 8) ** 2,
        "chebyshev": (1 + np.cos(np.pi * (i + .5) / 5)) / 2,
        "edm": ((1 + i / 4 * (1e-3 ** (1 / 7) - 1)) ** 7 - 1e-3) / .999,
    }[kind]
    expected = 1 - noise_time
    expected[0], expected[-1] = 0, 1
    np.testing.assert_allclose(time_grid(4, kind), expected, atol=1e-15)
    np.testing.assert_array_equal(time_grid(1, kind), [0, 1])
    cfg = build_network_config(n_particles=3, predict_divergence=True,
                               divergence_mode="hutchinson", d_model=24, num_layers=1, nhead=3)
    cfg.divergence_rescale = 2.0
    net = TorchFlowMap(cfg)
    seen = []
    def field(s, t, x, **kw):
        seen.append((float(s[0]), float(t[0])))
        return s[:, None, None].expand_as(x), s
    net.calc_phi = field
    x = torch.zeros(2, 3, 3)
    for start, end, grid in [(0., 1., expected), (1., 0., expected[::-1])]:
        actual, log_change = net._integrate(start, end, x, n_steps=4,
                                            time_discretization_type=kind)
        integral = float(np.sum(np.diff(grid) * grid[:-1]))
        torch.testing.assert_close(actual, torch.full_like(actual, integral))
        torch.testing.assert_close(log_change, torch.full((2,), -2 * integral))
        np.testing.assert_allclose(seen[-4:], np.stack((grid[:-1], grid[1:]), 1), atol=3e-8)


def test_mala_ratio_fixed_target():
    x = torch.tensor([[2., -1.], [-.2, .5], [3., 2.]], dtype=torch.float64)
    tau = torch.tensor([.1, .3, .8], dtype=x.dtype)
    torch.manual_seed(123)
    candidate = x - tau[:, None] * x + (2 * tau[:, None]).sqrt() * torch.randn_like(x)
    forward = candidate - (1 - tau[:, None]) * x
    backward = x - (1 - tau[:, None]) * candidate
    ratio = gaussian(candidate)[0] - gaussian(x)[0] + (
        forward.square().sum(1) - backward.square().sum(1)) / (4 * tau)
    probability = ratio.clamp(max=0).exp()
    accepted = torch.rand_like(probability) < probability
    expected = torch.where(accepted[:, None], candidate, x)
    torch.manual_seed(123)
    actual, info = mala_sample(x, gaussian, 1, tau)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(info["accept_prob"][0], probability)
    torch.testing.assert_close(info["final_step_size"], tau)


def test_mala_rejects_nan_preserves_com_and_gaussian_stationarity():
    calls = 0
    def bad_proposals(x):
        nonlocal calls
        calls += 1
        if calls > 1:
            return torch.full((len(x),), -torch.inf), torch.full_like(x, torch.nan)
        return gaussian(x)
    initial = torch.ones(3, 2)
    actual, info = mala_sample(initial, bad_proposals, 3, .1)
    torch.testing.assert_close(actual, initial)
    assert not info["accept_prob"].any()
    torch.manual_seed(8)
    initial = torch.randn(12000, 3, 3, dtype=torch.float64)
    initial -= initial.mean(1, keepdim=True)
    actual, info = mala_sample(initial, gaussian, 12, .4, project_com=True,
                               checkpoint_steps=[0, 12])
    torch.testing.assert_close(actual.mean(1), torch.zeros_like(actual.mean(1)), atol=1e-15, rtol=0)
    # Nine Cartesian coordinates minus three centroid degrees of freedom.
    assert float(actual.square().sum((1, 2)).mean()) == pytest.approx(6, abs=.09)
    assert abs(float(actual.mean())) < .01
    torch.testing.assert_close(info["checkpoints"][12], actual)
    with pytest.raises(ValueError, match="step_size"):
        mala_sample(initial, gaussian, 1, -1)


def test_cli_defaults_and_validation():
    argv = ["--cfg_path", "configs.ace_a_nme", "--ckpt_path", "density.pt", "--out_dir", "/tmp/eval"]
    args = launcher.parse_args(argv)
    assert (args.n_generate_samples, args.n_eval_samples, args.seed, args.steps,
            args.time_discretization_type, args.posthoc_mala_steps,
            args.com_augmentation) == (100000, 10000, 42, [4], "linear", 128, True)
    assert args.energy_cutoff is None
    assert (args.resampling, args.mala_adapt_steps, args.mala_target_accept) == ("multinomial", 64, .6)
    fixed = launcher.parse_args(argv + ["--mala_adapt_steps", "0", "--resampling", "stratified"])
    assert fixed.mala_adapt_steps == 0 and fixed.resampling == "stratified"
    assert launcher.parse_args(argv + ["--posthoc_mala_steps", "0"]).mala_adapt_steps == 0
    assert ENERGY_CUTOFFS_KT == {
        "ace_a_nme": 10., "ace_aaa_nme": 50., "ala3_implicit": -120., "ala6_implicit": -20.}
    override = launcher.parse_args(argv + ["--energy_cutoff", "-30", "--seed", "17",
                                           "--n_generate_samples", "100", "--n_eval_samples", "20"])
    assert (override.energy_cutoff, override.seed, override.n_generate_samples, override.n_eval_samples) == (-30, 17, 100, 20)
    assert args.mode == "dlogpdt"
    assert not launcher.parse_args(argv + ["--no_com_augmentation"]).com_augmentation
    for options in [["--steps", "0"], ["--n_samples", "0"], ["--posthoc_mala_steps", "-1"],
                    ["--mala_step_size", "nan"], ["--logw_quantile_filter", "1"],
                    ["--n_eval_samples", "0"], ["--energy_cutoff", "nan"],
                    ["--mala_target_accept", "nan"], ["--mala_target_accept", "0"],
                    ["--mala_target_accept", "1"], ["--mala_adapt_steps", "-1"],
                    ["--mala_adapt_steps", "129"], ["--resampling", "unknown"]]:
        with pytest.raises(SystemExit):
            launcher.parse_args(argv + options)


def test_ts_resampling_ess_and_direct_relaxation():
    pytest.importorskip("transferable_samplers")
    from common.tsx import normalized_ess
    x = torch.arange(90, dtype=torch.float64).reshape(10, 3, 3) / 100
    e = x.square().sum((1, 2)) / 2
    logw = torch.arange(10, dtype=x.dtype) / 5
    e[-1] = torch.inf
    batches = []
    def energy_and_grad(x):
        batches.append(len(x))
        return -gaussian(x)[0], -gaussian(x)[1]
    target = SimpleNamespace(energy_and_grad=energy_and_grad)
    torch.manual_seed(13)
    stages, stats = sample_stages(x, e, logw, clip_fraction=.2,
        energy=target, mala_steps=4, mala_step_size=.1, n_eval_samples=3, energy_cutoff=float(e[6]))
    assert list(stages) == ["proposal", "resampled", "resampled_energy_cutoff", "relaxed", "relaxed_energy_cutoff"]
    assert all(len(data) == 3 for data in stages.values())
    assert stats["resampled"]["resampling-method"] == "multinomial"
    assert stats["resampled"]["resampling-order"] == "input"
    assert stats["proposal"]["n-nonfinite"] == 1
    assert stats["proposal"]["n-generated"] == 10 and stats["proposal"]["n-finite"] == 9
    assert stats["resampled"]["effective-sample-size"] == pytest.approx(float(normalized_ess(logw[:7])))
    assert stats["resampled_energy_cutoff"]["effective-sample-size"] == pytest.approx(float(normalized_ess(logw[:5])))
    assert stats["resampled_energy_cutoff"]["n-after-energy-cutoff"] == 6
    assert stats["resampled_energy_cutoff"]["n-after-filter"] == 5
    assert stats["proposal"]["ess-unfiltered"] == pytest.approx(float(normalized_ess(logw[:9])) * 9)
    for name, upper in [("resampled", 7), ("resampled_energy_cutoff", 5)]:
        for sample, energy in zip(stages[name].samples, stages[name].E_target):
            index = torch.where((x == sample).all(2).all(1))[0].item()
            assert index < upper and energy == e[index]
    for name in ("relaxed", "relaxed_energy_cutoff"):
        torch.testing.assert_close(stages[name].E_target, -gaussian(stages[name].samples)[0])
        assert stages[name].logw is None and "effective-sample-size" not in stats[name]
    assert batches == [6] * 5  # one joint initial evaluation + four MALA proposals, not 10 chains
    torch.manual_seed(13)
    repeated, repeated_stats = sample_stages(x, e, logw, clip_fraction=.2,
        energy=target, mala_steps=4, mala_step_size=.1, n_eval_samples=3, energy_cutoff=float(e[6]))
    for name in stages:
        torch.testing.assert_close(repeated[name].samples, stages[name].samples)
    assert repeated_stats == stats


@pytest.mark.parametrize("cutoff", [-1., 100.])
def test_cutoff_empty_or_identical_pool_avoids_extra_mala(cutoff):
    pytest.importorskip("transferable_samplers")
    x = torch.randn(12, 3, 3)
    energies = torch.arange(12, dtype=x.dtype)
    batches = []
    def energy_and_grad(x):
        batches.append(len(x))
        return -gaussian(x)[0], -gaussian(x)[1]
    stages, stats = sample_stages(x, energies, -energies, clip_fraction=0,
        energy=SimpleNamespace(energy_and_grad=energy_and_grad), mala_steps=2, mala_step_size=.1,
        n_eval_samples=5, energy_cutoff=cutoff)
    assert batches == [5, 5, 5]
    if cutoff < 0:
        assert len(stages["resampled_energy_cutoff"]) == len(stages["relaxed_energy_cutoff"]) == 0
        assert stats["resampled_energy_cutoff"]["effective-sample-size"] is None
        assert stats["relaxed_energy_cutoff"]["status"] == "empty-after-energy-cutoff"
    else:
        assert stages["resampled"] is stages["resampled_energy_cutoff"]
        assert stages["relaxed"] is stages["relaxed_energy_cutoff"]


@pytest.mark.parametrize("lift", [False, True])
def test_launcher_logs_all_stages_without_overwriting(monkeypatch, tmp_path, lift):
    pytest.importorskip("transferable_samplers")
    from common import datasets, state_utils, ts_systems, tsx
    from common.flow_map import TorchFlowMap
    class Energy:
        closed = False
        generation_batches = []
        mala_batches = []
        def numpy(self, x):
            if isinstance(x, np.ndarray): self.generation_batches.append(len(x))
            return ((x - x.mean(1, keepdims=True)) ** 2).sum((1, 2))
        def energy_and_grad(self, x):
            self.mala_batches.append(len(x))
            return self.numpy(x), 2 * (x - x.mean(1, keepdim=True))
        def close(self): self.closed = True
    energy = Energy()
    monkeypatch.setattr(ts_systems, "ParallelEnergy", lambda *a, **kw: energy)
    monkeypatch.setattr(ts_systems, "ensure_data", lambda *a, **kw: {"source": "analytic"})
    monkeypatch.setattr(ts_systems, "train_std", lambda *a, **kw: .25)
    setup_target = datasets.setup_target
    monkeypatch.setattr(datasets, "setup_target", lambda cfg, **kw: setup_target(cfg, train_std=.25))
    def setup(cfg, *args, **kwargs):
        assert cfg.network.com_aware == cfg.problem.com_augmentation == lift
        assert cfg.network.predict_divergence
        cfg.network.dit_kwargs = dict(d_model=24, num_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=16)
        assert cfg.network.divergence_rescale == 1.
        net = TorchFlowMap(cfg.network)
        return net, 0, None
    monkeypatch.setattr(state_utils, "load_evaluation_model", setup)
    def context(*a, **kw):
        assert kw["normalization_std"] == 1. and kw["num_eval_samples"] == 7
        return SimpleNamespace(true_data=tsx.SamplesData(torch.zeros(7, 22, 3), torch.arange(7.)))
    monkeypatch.setattr(ts_systems, "build_eval_context", context)
    class Evaluator:
        def __init__(self, **kw): assert kw["num_eval_samples"] == 7
        def evaluate(self, stages, ctx, prefix, **kw):
            assert set(stages) == {"proposal", "resampled", "resampled_energy_cutoff", "relaxed", "relaxed_energy_cutoff"}
            assert all(len(stage) == 7 for stage in stages.values())
            assert all(stage.logw is None for stage in stages.values())
            return {f"{prefix}/{name}/energy-w2": value for name, value in
                    [("proposal", 1.), ("resampled", 2.), ("resampled_energy_cutoff", 3.),
                     ("relaxed", 4.), ("relaxed_energy_cutoff", 5.)]}
    monkeypatch.setattr(tsx, "PeptideEnsembleEvaluator", Evaluator)
    monkeypatch.setattr("sys.argv", ["eval", "--cfg_path", "configs.ace_a_nme", "--ckpt_path", "unused.pt",
        "--out_dir", str(tmp_path), "--device", "cpu", "--no_ema", "--no_plots", "--n_generate_samples", "40",
        "--batch_size", "13", "--n_eval_samples", "7", "--posthoc_mala_steps", "3",
        "--time_discretization_type", "edm", "--save_energies"] + ([] if lift else ["--no_com_augmentation"]))
    launcher.main()
    result = json.loads((tmp_path / "ts_metrics.json").read_text())
    assert energy.closed and result["time_grids"]["4"] == time_grid(4, "edm").tolist()
    assert result["n_generate_samples"] == 40 and result["n_eval_samples"] == 7
    assert result["seed"] == 42 and result["energy_cutoff_kT"] == 10.
    assert result["mala_adapt"] and result["mala_adapt_steps"] == 1
    assert result["mala_target_accept"] == .6 and result["resampling"] == "multinomial"
    assert energy.generation_batches == [40]
    assert len(energy.mala_batches) == 4 and max(energy.mala_batches) <= 14
    for split in ("val", "test"):
        entry = result["splits"][split]["steps"]["4"]
        assert [entry[name]["energy-w2"] for name in ("proposal", "resampled", "resampled_energy_cutoff",
                                                      "relaxed", "relaxed_energy_cutoff")] == [1, 2, 3, 4, 5]
        assert entry["resampled"]["n-after-filter"] == 39
        assert entry["relaxed"]["mala-steps"] == 3
        assert entry["relaxed"]["mala-adapt-steps"] == 1
        assert entry["relaxed"]["mala-final-step-size-nm2"] > 1e-6
        assert 0 <= entry["relaxed"]["mala-post-adapt-mean-accept-prob"] <= 1
        assert entry["resampled"]["resampling-order"] == "input"
    assert set(np.load(tmp_path / "energies.npz").files) == {
        "steps_4_proposal", "steps_4_resampled", "steps_4_resampled_energy_cutoff",
        "steps_4_relaxed", "steps_4_relaxed_energy_cutoff", "reference_val", "reference_test"}


@pytest.mark.parametrize("target", ["ace_a_nme", "ace_aaa_nme", "ala3_implicit", "ala6_implicit"])
def test_cached_ts_openmm_force_sign_units_and_mala(target):
    pytest.importorskip("transferable_samplers")
    pytest.importorskip("openmm")
    from common import ts_systems
    system = ts_systems.resolve(target)
    paths = ts_systems.data_paths(system)
    if not all(Path(paths[k]).exists() for k in ("train", "val", "test", "pdb")):
        pytest.skip("Official TS cache is not available; this test never downloads")
    x = torch.from_numpy(np.load(paths["val"], mmap_mode="r")[:2].copy()).double()
    x -= x.mean(1, keepdim=True)
    torch.manual_seed(27)
    direction = torch.randn_like(x)
    direction -= direction.mean(1, keepdim=True)
    direction /= direction.square().sum((1, 2), keepdim=True).sqrt()
    with ts_systems.ParallelEnergy(system, n_workers=2, chunk=1) as energy:
        e, grad = energy.energy_and_grad(x)
        torch.testing.assert_close(e, torch.from_numpy(energy.numpy(x)), rtol=1e-6, atol=2e-5)
        h = 1e-4
        difference = (energy.numpy(x + h * direction) - energy.numpy(x - h * direction)) / (2 * h)
        np.testing.assert_allclose(difference, (grad * direction).sum((1, 2)).numpy(), rtol=.02, atol=.2)
        relaxed, info = mala_sample(x, lambda v: tuple(-a for a in energy.energy_and_grad(v)),
                                    3, 1e-6, project_com=True, adapt=True)
        assert torch.isfinite(relaxed).all()
        torch.testing.assert_close(-info["final_logp"], torch.from_numpy(energy.numpy(relaxed)),
                                   rtol=1e-6, atol=2e-5)
        assert (info["acceptance_rate"] > 0).any()


@pytest.mark.parametrize("vectorized", [False, True])
def test_launcher_resolves_nonzero_learned_density_scale(vectorized):
    from common.flow_map import TorchFlowMap
    from _equiv_utils import build_network_config
    cfg = importlib.import_module("configs.ace_a_nme").get_config(0)
    cfg.network.vectorized_divergence = vectorized
    assert cfg.network.divergence_rescale == 0
    args = SimpleNamespace(vectorized_divergence=None, divergence_rescale=None)
    launcher.configure_density_scale(cfg, args)
    assert cfg.network.divergence_rescale == (1 if vectorized else 66)
    config = build_network_config(n_particles=22, predict_divergence=True,
                                  vectorized_divergence=vectorized, divergence_mode="hutchinson",
                                  d_model=24, num_layers=1, nhead=3)
    config.divergence_rescale = cfg.network.divergence_rescale
    net = TorchFlowMap(config)
    with torch.no_grad():
        net.dit.div_final_linear.bias.fill_(.25)
    x = torch.zeros(2, 22, 3)
    _, logp, _ = net.sample_forward(x, n_steps=4)
    _, initial_logp = net.unscale_samples(x, net.base_log_prob(x))
    torch.testing.assert_close(logp - initial_logp, torch.full((2,), -.25 * 66 * config.divergence_rescale))
    args.divergence_rescale = 2.5
    launcher.configure_density_scale(cfg, args)
    assert cfg.network.divergence_rescale == 2.5


@pytest.mark.parametrize("kind", TIME_DISCRETIZATION_TYPES)
def test_velocity_sampling_uses_requested_grid(kind):
    pytest.importorskip("transferable_samplers")
    from common.ts_eval import sample_model
    calls = []
    class Velocity:
        config = SimpleNamespace(divergence_mode="pretrain_v")
        def calc_b(self, t, x, train=False):
            calls.append(float(t[0]))
            return .1 * x
        def unscale_samples(self, x): return .25 * x
    x = torch.ones(5, 3, 3)
    samples, logp, nfe = sample_model(Velocity(), x, n_steps=4, batch_size=2,
                                     velocity_solver="rk4", time_discretization_type=kind)
    torch.testing.assert_close(samples, x * np.exp(.1) * .25)
    np.testing.assert_allclose(calls[:16:4], time_grid(4, kind)[:-1], atol=3e-8)
    assert logp is None and nfe == len(calls) == 48


@pytest.mark.parametrize("divergence_type", ["hutchinson", "exact"])
@pytest.mark.parametrize("lift", [False, True])
def test_ode_dopri5_uses_shared_five_stage_pipeline(monkeypatch, tmp_path, divergence_type, lift):
    pytest.importorskip("transferable_samplers")
    from common import datasets, ts_systems, tsx
    from common.flow_map import TorchFlowMap
    from common.se3_utils import remove_mean
    from common.com_lift import importance_source_energy
    from launchers import eval_ode_int as ode
    from transferable_samplers.samplers.filtering import filter_by_logw_quantile

    class Energy:
        closed = False
        generation_batches = []
        mala_batches = []
        def numpy(self, x):
            self.generation_batches.append(len(x))
            return ((x - x.mean(1, keepdims=True)) ** 2).sum((1, 2)).astype(np.float64)
        def energy_and_grad(self, x):
            self.mala_batches.append(len(x))
            centered = remove_mean(x)
            return centered.square().sum((1, 2)), 2 * centered
        def close(self): self.closed = True
    energy = Energy()
    monkeypatch.setattr(ts_systems, "ParallelEnergy", lambda *a, **kw: energy)
    monkeypatch.setattr(ts_systems, "ensure_data", lambda *a, **kw: {"source": "analytic-test"})
    monkeypatch.setattr(ts_systems, "train_std", lambda *a, **kw: .25)
    setup_target = datasets.setup_target
    monkeypatch.setattr(datasets, "setup_target", lambda cfg, **kw: setup_target(cfg, train_std=.25))
    nets = []
    def load(args, cfg, device):
        assert cfg.network.divergence_mode == "pretrain_v" and not cfg.network.predict_divergence
        assert cfg.network.com_aware == cfg.problem.com_augmentation == lift
        cfg.network.dit_kwargs = dict(d_model=24, num_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=16)
        net = TorchFlowMap(cfg.network)
        net.calc_b = lambda t, x, **kw: .1 * (x if lift else remove_mean(x))
        nets.append(net)
        return net, "raw", 0
    monkeypatch.setattr(ode, "_load_net", load)
    def context(*a, **kw):
        assert kw["num_eval_samples"] == 7 and kw["normalization_std"] == 1.
        return SimpleNamespace(true_data=tsx.SamplesData(torch.zeros(7, 22, 3), torch.arange(7.)))
    monkeypatch.setattr(ts_systems, "build_eval_context", context)
    evaluated = []
    class Evaluator:
        def __init__(self, **kw): assert kw["num_eval_samples"] == 7
        def evaluate(self, stages, ctx, prefix, **kw):
            assert tuple(stages) == ode.STAGES
            assert all(len(data) == 7 and data.logw is None for data in stages.values())
            evaluated.append(stages)
            return {f"{prefix}/{name}/energy-w2": float(i) for i, name in enumerate(stages, 1)}
    monkeypatch.setattr(tsx, "PeptideEnsembleEvaluator", Evaluator)
    monkeypatch.setattr("sys.argv", ["eval_ode_int", "--system", "ace_a_nme",
        "--ckpt_path", "ace_a_nme_velocity.pt", "--out_dir", str(tmp_path), "--device", "cpu",
        "--no_ema", "--no_plots", "--n_generate_samples", "40", "--n_eval_samples", "7",
        "--batch_size", "13", "--posthoc_mala_steps", "3", "--energy_cutoff", "5",
        "--save_samples", "--save_energies", "--divergence_type", divergence_type]
        + ([] if lift else ["--no_com_augmentation"]))
    ode.main()
    result = json.loads(next(tmp_path.glob("ode_metrics_*.json")).read_text())
    raw = np.load(next(tmp_path.glob("ode_samples_*.npz")))
    saved_stages = np.load(next(tmp_path.glob("ode_stages_*.npz")))
    saved_energies = np.load(next(tmp_path.glob("ode_energies_*.npz")))
    assert energy.closed and energy.generation_batches == [40]
    assert energy.mala_batches == [14] * 4  # computed once, not once per reference split
    assert result["n_generate_samples"] == 40 and result["n_eval_samples"] == 7 and result["seed"] == 42
    assert result["energy_cutoff_kT"] == 5 and result["posthoc_mala_steps"] == 3
    assert result["argv"]["solver"] == ["dopri5"]
    assert result["mala_adapt"] and result["mala_adapt_steps"] == 1
    assert result["mala_target_accept"] == .6
    assert result["resampling"]["method"] == "multinomial" and result["resampling"]["order"] == "input"
    x0 = torch.randn(40, 22, 3, generator=torch.Generator().manual_seed(42))
    if not lift: x0 = remove_mean(x0)
    expected_x = .25 * np.exp(.1) * x0
    if divergence_type == "exact":
        trace = torch.full((40,), .1 * (66 if lift else 63))
    else:
        traces = []
        for i in range(0, 40, 13):
            eps = torch.randn(x0[i:i + 13].shape,
                              generator=torch.Generator().manual_seed(42 + 991 + i)).sign()
            traces.append(.1 * (eps * (eps if lift else remove_mean(eps))).sum((1, 2)))
        trace = torch.cat(traces)
    expected_logq = nets[0].base_log_prob(x0) - trace - nets[0].modeled_dimension * np.log(.25)
    torch.testing.assert_close(torch.from_numpy(raw["samples_dopri5_adaptive"]), expected_x, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(torch.from_numpy(raw["logq_dopri5_adaptive"]), expected_logq, rtol=2e-5, atol=1e-5)
    e = torch.from_numpy(raw["energies_dopri5_adaptive"])
    weights = torch.from_numpy(raw["logw_dopri5_adaptive"])
    expected_weights = importance_source_energy(expected_x, expected_logq,
        com_augmentation=lift, unscale_factor=.25) - e
    torch.testing.assert_close(weights, expected_weights, rtol=2e-5, atol=1e-5)
    assert 0 < int((e < 5).sum()) < 40
    cost = result["cost"]["dopri5/adaptive"]
    assert cost["adaptive"]["n_accepted_min"] > 0
    assert cost["n_jvp_per_sample"] == cost["nfe_velocity_per_sample"] * (66 if divergence_type == "exact" else 1)
    for name in ode.STAGES:
        assert evaluated[0][name] is evaluated[1][name]
        np.testing.assert_array_equal(saved_stages[f"samples_{name}"], evaluated[0][name].samples.numpy())
        np.testing.assert_array_equal(saved_energies[f"dopri5_adaptive_{name}"], evaluated[0][name].E_target.numpy())
    for split in ("val", "test"):
        entry = result["splits"][split]["runs"]["dopri5/adaptive"]
        assert [entry[name]["energy-w2"] for name in ode.STAGES] == [1, 2, 3, 4, 5]
        for name, mask in (("resampled", torch.ones(40, dtype=torch.bool)), ("resampled_energy_cutoff", e < 5)):
            _, filtered, _ = filter_by_logw_quantile(torch.where(mask)[0], weights[mask], torch.zeros_like(weights[mask]), .002)
            assert entry[name]["n-after-filter"] == len(filtered)
            assert entry[name]["ess"] == pytest.approx(float(tsx.normalized_ess(filtered)) * len(filtered))
        assert "ess" not in entry["relaxed"] and "ess" not in entry["relaxed_energy_cutoff"]


@pytest.mark.parametrize("initial_tau", [.3, 1.0])
def test_adaptive_mala_targets_acceptance_freezes_and_keeps_budget(initial_tau):
    calls = 0
    def target(x):
        nonlocal calls
        calls += 1
        return gaussian(x)
    torch.manual_seed(18)
    x = torch.randn(2048, 12, dtype=torch.float64)
    actual, info = mala_sample(x, target, 128, initial_tau, adapt=True)
    assert calls == 129 and info["adapt_steps"] == 64
    tau = info["step_size"]
    torch.testing.assert_close(tau[:, :1].expand_as(tau), tau)  # shared across chains
    torch.testing.assert_close(tau[64:], info["final_step_size"].expand_as(tau[64:]))
    assert (info["final_step_size"][0] > initial_tau) == (initial_tau == .3)
    assert float(info["accept_prob"][64:].mean()) == pytest.approx(.6, abs=.05)
    assert float(actual.square().mean()) == pytest.approx(1., abs=.03)
    # Explicit zero warmup reproduces the fixed kernel and its RNG stream exactly.
    torch.manual_seed(73)
    fixed, _ = mala_sample(x[:4], gaussian, 3, initial_tau)
    torch.manual_seed(73)
    zero_warmup, _ = mala_sample(x[:4], gaussian, 3, initial_tau, adapt=True, adapt_steps=0)
    torch.testing.assert_close(fixed, zero_warmup, rtol=0, atol=0)


def test_invalid_mala_adaptation_is_rejected():
    x = torch.randn(4, 3, 3)
    for kwargs in ({"adapt": True, "adapt_steps": -1}, {"adapt": True, "adapt_steps": 4},
                   {"adapt_steps": 1}, {"adapt": True, "metropolis": False},
                   {"target_accept": float("nan")}):
        with pytest.raises(ValueError):
            mala_sample(x, gaussian, 3, .1, **kwargs)


def test_stratified_counts_unbiasedness_and_lower_variance():
    torch.manual_seed(59)
    # Integer expected copy counts have no resampling variance here.
    indices = resampling_indices(torch.tensor([.5, .3, .2]).double().log(), 10, "stratified")
    assert torch.bincount(indices, minlength=3).tolist() == [5, 3, 2]
    # Noninteger counts remain unbiased, with lower count variance than multinomial.
    weights = torch.tensor([.52, .27, .21], dtype=torch.float64)
    means, variances = {}, {}
    for method in ("stratified", "multinomial"):
        counts = torch.stack([torch.bincount(resampling_indices(weights.log(), 7, method),
                                           minlength=3) for _ in range(1024)]).double()
        means[method], variances[method] = counts.mean(0), counts.var(0)
        torch.testing.assert_close(means[method], 7 * weights, atol=.12, rtol=0)
    assert torch.all(variances["stratified"] < variances["multinomial"])
    assert (resampling_indices(torch.tensor([-torch.inf, 9000., -torch.inf]), 100, "stratified") == 1).all()
    # The default is bit-for-bit identical to multinomial under the same seed.
    torch.manual_seed(5)
    old = torch.multinomial(weights, 40, replacement=True)
    torch.manual_seed(5)
    torch.testing.assert_close(resampling_indices(weights.log(), 40), old)
    for logw in (torch.empty(0), torch.ones(2, 2), torch.tensor([torch.nan]),
                 torch.tensor([torch.inf]), torch.tensor([-torch.inf])):
        with pytest.raises(ValueError):
            resampling_indices(logw, 3)


def test_energy_ordered_stratification_preserves_samples_weights_and_ess():
    pytest.importorskip("transferable_samplers")
    x = torch.arange(36, dtype=torch.float64).reshape(4, 3, 3)
    energies = torch.tensor([3., 0., 2., 1.])
    stages, stats = sample_stages(x, energies, torch.zeros(4), clip_fraction=0,
        energy=None, mala_steps=0, mala_step_size=1e-6, n_eval_samples=4, energy_cutoff=10., resampling="stratified")
    torch.testing.assert_close(stages["resampled"].E_target, torch.arange(4, dtype=energies.dtype))
    torch.testing.assert_close(stages["resampled"].samples, x[torch.tensor([1, 3, 2, 0])])
    assert stages["resampled"] is stages["resampled_energy_cutoff"]
    assert stats["resampled"]["ess"] == 4 and stats["resampled"]["effective-sample-size"] == 1


@pytest.mark.parametrize("lifted", [False, True])
@pytest.mark.parametrize("fraction", [0.0, 0.25])
def test_corrected_weights_stay_aligned_through_filtering(lifted, fraction):
    from common.com_lift import importance_source_energy
    x = torch.zeros(4, 2, 3, dtype=torch.float64)
    x[:, :, 0] = torch.arange(4, dtype=torch.float64)[:, None]
    logq = -torch.arange(4, dtype=torch.float64)
    energies = torch.tensor([0., .1, .2, .3], dtype=torch.float64)
    weights = importance_source_energy(x, logq, com_augmentation=lifted) - energies
    oracle = -logq - energies
    if lifted:
        oracle += torch.distributions.Normal(0., 2**-.5).log_prob(x.mean(1)).sum(-1)
        assert weights.argmax() != (-logq - energies).argmax()
    keep = oracle <= torch.quantile(oracle, 1-fraction) if fraction else torch.ones(4, dtype=torch.bool)
    artifacts = {}
    stages, _ = sample_stages(x, energies, weights, clip_fraction=fraction,
        energy=None, mala_steps=0, mala_step_size=.1, n_eval_samples=4,
        energy_cutoff=None, recipe_v1=True, artifacts=artifacts)
    assert torch.equal(artifacts['filtered_indices'], torch.where(keep)[0])
    index = artifacts['resampled_indices']
    assert keep[index].all()
    torch.testing.assert_close(stages['resampled'].samples, x[index])
    torch.testing.assert_close(stages['resampled'].E_target, energies[index])
    torch.testing.assert_close(weights, oracle)


def test_resampling_rejects_empty_finite_support():
    with pytest.raises(ValueError, match="No finite"):
        sample_stages(torch.zeros(2, 2, 3), torch.zeros(2),
            torch.full((2,), -float('inf')), clip_fraction=0., energy=None,
            mala_steps=0, mala_step_size=.1, n_eval_samples=2, energy_cutoff=100.)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('fraction', [0., .2, .5])
@pytest.mark.parametrize('equal', [False, True])
@pytest.mark.parametrize('cutoff', [-100., 3., 100.])
def test_one_quantile_per_distinct_pool_matches_upstream(monkeypatch, dtype, fraction, equal, cutoff):
    from unittest.mock import Mock
    from transferable_samplers.samplers.filtering import filter_by_logw_quantile
    x = torch.arange(12*3*3, dtype=dtype).reshape(12, 3, 3) / 100
    energy = torch.arange(12, dtype=dtype)
    weights = torch.ones(12, dtype=dtype) if equal else torch.arange(12, dtype=dtype) // 3
    expected, _, _ = filter_by_logw_quantile(torch.arange(12), weights, torch.zeros_like(weights), fraction)
    quantile = Mock(wraps=torch.quantile)
    monkeypatch.setattr(torch, 'quantile', quantile)
    artifacts = {}
    stages, stats = sample_stages(x, energy, weights, clip_fraction=fraction, energy=None,
        mala_steps=0, mala_step_size=.1, n_eval_samples=7, energy_cutoff=cutoff, artifacts=artifacts)
    assert stats['resampled']['n-after-filter'] == len(expected)
    assert quantile.call_count == ((2 if 0 < cutoff < 12 else 1) if fraction else 0)
    if cutoff > 12:
        assert torch.equal(artifacts['filtered_indices'], expected)
        assert stages['resampled'] is stages['resampled_energy_cutoff']
    if cutoff < 0:
        assert len(stages['resampled_energy_cutoff']) == 0
