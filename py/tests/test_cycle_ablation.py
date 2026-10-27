"""Cycle-loss ablation: each run is the README command plus its own flags, and checks catch drift."""
import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch

from common import distributed
from common.com_lift import com_log_prob
from common.flow_map import TorchFlowMap
from launchers import cycle_ablation, eval_round_trip, learn
from test_simplification import tiny_config

TEACHER = '/ckpt/teacher.pt'


def pipeline(tmp_path, smoke=False):
    opts = SimpleNamespace(device='cpu', data_dir='datasets/ts', smoke_steps=300)
    return cycle_ablation.Pipeline(opts, str(tmp_path), smoke=smoke)


def resolve(argv, monkeypatch):
    monkeypatch.setenv('WANDB_ENTITY', 'RKOY')
    monkeypatch.setattr(sys, 'argv', ['learn.py', *argv[2:]])
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    distributed.configure_batch(cfg)
    return cfg


def test_runs_cover_both_time_orders_with_and_without_cycle():
    assert {(order, cycle) for _, order, cycle in cycle_ablation.RUNS} == {
        ('square', False), ('forward', False), ('square', True), ('forward', True)}


@pytest.mark.parametrize('index', range(4))
def test_run_resolves_to_recipe_plus_its_flags(tmp_path, monkeypatch, index):
    name, order, cycle = cycle_ablation.RUNS[index]
    job = pipeline(tmp_path).run_jobs(TEACHER, 1)[index]
    argv = job.argv(None)
    added = ['--offdiag_time_order', order] + (['--cycle_loss'] if cycle else []) + ['--device', 'cpu']
    readme = ['--cfg_path', 'configs.ace_a_nme', '--ts_data_dir', 'datasets/ts', '--output_folder',
              str(tmp_path / 'runs' / name), '--mode', 'dlogpdt', '--network_load_path', TEACHER,
              '--teacher_load_path', TEACHER]
    assert sorted(argv[2:]) == sorted(readme + added)
    cfg = resolve(argv, monkeypatch).to_dict()
    assert cfg['training']['offdiag_time_order'] == order
    assert cfg['training'].get('cycle_loss', False) is cycle
    assert cfg['optimization']['total_steps'] == cycle_ablation.DISTILL_STEPS
    assert cfg['network']['load_path'] == cfg['teacher']['load_path'] == TEACHER
    for (section, key), value in cycle_ablation.RECIPE.items():
        assert cfg[section][key] == value, (section, key)


def test_resume_restores_student_but_keeps_teacher(tmp_path, monkeypatch):
    job = pipeline(tmp_path).run_jobs(TEACHER, 1)[3]
    cfg = resolve(job.argv('/ckpt/run4.pt'), monkeypatch)
    assert cfg.network.load_path == '/ckpt/run4.pt'
    assert cfg.teacher.load_path == TEACHER
    assert not cfg.network.reset_optimizer
    assert cfg.training.cycle_loss and cfg.training.offdiag_time_order == 'forward'


def test_smoke_only_shrinks_budgets(tmp_path, monkeypatch):
    cfg = resolve(pipeline(tmp_path, smoke=True).run_jobs(TEACHER, 1)[0].argv(None), monkeypatch)
    assert cfg.optimization.total_steps == 300
    assert cfg.logging.visual_freq == cfg.logging.save_freq == 150
    assert cfg.optimization.bs == 1024


@pytest.mark.parametrize('index', range(4))
def test_paper_runs_and_teacher_pass_every_recipe_check(tmp_path, monkeypatch, index):
    name, order, cycle = cycle_ablation.RUNS[index]
    opts = SimpleNamespace(device='cpu', data_dir='datasets/ts', smoke_steps=300, recipe='paper', max_attempts=4,
                           teacher_steps=0, distill_steps=0)
    paper = cycle_ablation.Pipeline(opts, str(tmp_path), smoke=False)
    cfg = resolve(paper.run_jobs(TEACHER, 1)[index].argv(None), monkeypatch).to_dict()
    teacher_cfg = resolve(paper.teacher_job().argv(None), monkeypatch).to_dict()
    assert cfg['optimization']['total_steps'] == teacher_cfg['optimization']['total_steps'] == 4_000_000
    run = {'config': cfg, 'records': [{'lsd/cycle_v_loss': .1}] if cycle else [],
           'checkpoint': None, 'ts': None, 'round_trip': None}
    checks = cycle_ablation.verify(run, order, cycle, TEACHER, 4_000_000, 'paper', teacher_cfg)
    assert [k for k, ok in checks.items() if not ok] == [
        'reached 4000000 optimizer steps', 'recipe-v1 eval of final checkpoint', 'round trip of final checkpoint']
    assert ('directional cycle stop-gradient, own learned weight' in checks) is cycle


def test_resume_keeps_the_budgets_an_ablation_started_with():
    manifest = {'teacher_steps': 400_000, 'distill_steps': 100_000}
    opts = SimpleNamespace(teacher_steps=0, distill_steps=0)
    cycle_ablation.pin_budgets(opts, manifest)
    assert (opts.teacher_steps, opts.distill_steps) == (400_000, 100_000)
    with pytest.raises(SystemExit, match='distill_steps=100000'):
        cycle_ablation.pin_budgets(SimpleNamespace(teacher_steps=0, distill_steps=4_000_000), manifest)


def test_snapshot_pruning_keeps_multiples_and_the_latest_state(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'logging': {'output_name': 'm'}}))
    kept = ['m.pt', 'm_signal.pt', 'm_best_ess.pt', 'm_step20.pt', 'm_step40.pt', 'm_step50.pt.tmp']
    for name in kept + ['m_step10.pt', 'm_step30.pt']:
        (tmp_path / name).write_text('')
    cycle_ablation.prune_snapshots(str(tmp_path), 20)
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix != '.json') == sorted(kept)


def test_out_of_memory_lowers_concurrency_without_using_an_attempt(tmp_path):
    opts = SimpleNamespace(device='cpu', n_gpus=1, poll=.05, progress_every=1e9, stall_warn=1e9, stall_restart=0)
    scheduler = cycle_ablation.Scheduler(opts, str(tmp_path), 'test')
    script = ("import pathlib, sys\n"
              "marker, done = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
              "if sys.argv[3] == 'oom' and not marker.exists():\n"
              "    marker.touch(); print('torch.OutOfMemoryError: CUDA out of memory.'); sys.exit(1)\n"
              "done.touch()\n")
    jobs = []
    for name in ('a', 'b', 'c'):
        done = tmp_path / f'{name}.done'
        argv = [sys.executable, '-c', script, str(tmp_path / 'oom-once'), str(done), 'oom' if name == 'a' else 'ok']
        jobs.append(cycle_ablation.Job(name, 'eval', str(tmp_path), str(tmp_path / f'{name}.log'),
                                       lambda resume, argv=argv: argv, done.exists, max_attempts=1))
    assert scheduler.run(jobs, 3) == []
    assert [job.status for job in jobs] == ['done'] * 3
    assert scheduler.per_gpu == 2
    assert jobs[0].oom and jobs[0].attempts == 1


def write_run(root, name, order, cycle, step):
    out = root / 'runs' / name
    out.mkdir(parents=True)
    config = {'logging': {'output_name': 'm'}, 'teacher': {'load_path': TEACHER},
              'training': {'offdiag_time_order': order, **({'cycle_loss': True} if cycle else {})}}
    for (section, key), value in cycle_ablation.RECIPE.items():
        config.setdefault(section, {})[key] = value
    (out / 'config.json').write_text(json.dumps(config))
    record = {'step': step, 'train/loss': 1.0, **({'lsd/cycle_v_loss': .1} if cycle else {})}
    (out / 'metrics.jsonl').write_text(json.dumps(record) + '\n')
    torch.save({'step': step}, out / 'm.pt')
    return out


def test_verification_flags_wrong_order_missing_cycle_and_short_runs(tmp_path):
    write_run(tmp_path, 'good', 'square', True, 10)
    write_run(tmp_path, 'bad', 'forward', False, 9)
    good = cycle_ablation.load_run(str(tmp_path), 'good', 10)
    bad = cycle_ablation.load_run(str(tmp_path), 'bad', 10)
    checks = cycle_ablation.verify(good, 'square', True, TEACHER, 10)
    assert [k for k, ok in checks.items() if not ok] == [
        'recipe-v1 eval of final checkpoint', 'round trip of final checkpoint']
    failed = {k for k, ok in cycle_ablation.verify(bad, 'square', True, TEACHER, 10).items() if not ok}
    assert {'offdiag_time_order is square', 'cycle_loss is on', 'cycle losses logged only if on',
            'reached 10 optimizer steps'} <= failed


def test_round_trip_of_identity_map_is_exact_and_nll_matches_gaussian():
    cfg = tiny_config()
    net = TorchFlowMap(cfg.network).eval()  # zero-initialised heads: every jump is the identity
    uf, n_atoms = cfg.network.unscale_factor, cfg.network.n_particles
    x0 = torch.randn(6, n_atoms, 3, generator=torch.Generator().manual_seed(3))
    frames = torch.randn(6, n_atoms, 3, generator=torch.Generator().manual_seed(4))
    frames = frames - frames.mean(dim=1, keepdim=True)
    out = eval_round_trip.round_trip(net, x0, frames, 4, 4, uf, True)
    assert out['model_x0_rms_error']['max'] == out['data_rms_error_nm']['max'] == 0
    assert out['model_logq_abs_gap_nats']['max'] == 0
    dim = 3 * n_atoms
    z = frames / uf
    logq = -0.5 * z.square().sum((1, 2)) - 0.5 * dim * math.log(2 * math.pi) - dim * math.log(uf)
    expected = -(logq - com_log_prob(frames, uf)).mean()
    assert out['data_inverse_nll_nats'] == pytest.approx(float(expected), rel=1e-5)
    assert out['data_inverse_nll_nats_per_dof'] == pytest.approx(float(expected) / (dim - 3), rel=1e-5)


def test_round_trip_detects_a_non_invertible_map():
    cfg = tiny_config()
    torch.manual_seed(5)
    net = TorchFlowMap(cfg.network).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(.05 * torch.randn_like(p))
    x0 = torch.randn(8, cfg.network.n_particles, 3)
    frames = torch.randn(8, cfg.network.n_particles, 3) * .2
    out = eval_round_trip.round_trip(net, x0, frames - frames.mean(1, keepdim=True), 4, 8,
                                     cfg.network.unscale_factor, True)
    assert out['model_x0_rms_error']['mean'] > 0 and out['model_logq_abs_gap_nats']['mean'] > 0
    assert math.isfinite(out['data_inverse_nll_nats'])
