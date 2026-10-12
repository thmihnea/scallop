"""Best-checkpoint tracking and ESS selection (audit N3 + N5).

N3: ``logging._BEST`` was a process-local dict seeded from +-inf, so ANY restarted run
made its first evaluation unconditionally "best" and overwrote ``{output_name}_best_ess.pt``
with a worse model. The trackers now live on ``state.best_metrics``, ride along in the
checkpoint, and remain local to that training state.

N5: the reported ``resampled`` ESS is computed after the vendored SNISSampler discards the
top ``logw_quantile_filter`` fraction of importance weights. That filter (0.002) is the
PROJECT STANDARD (user decision 2026-09-04), so selection uses the ``resampled`` key; the
unfiltered value ``ts_eval`` also emits is a diagnostic under an explicit ``-unfiltered``
suffix and must never be selected on.

These exercise ``log_metrics`` directly with a stubbed evaluator, so they need neither a
GPU, OpenMM, nor JAX.
"""

import os
import sys
import types

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_PY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)

import ml_collections
import pytest

import common.logging as L


def _cfg(out_name):
    c = ml_collections.ConfigDict()
    c.logging = ml_collections.ConfigDict()
    c.logging.output_name = out_name
    c.logging.visual_freq = 10
    c.logging.save_freq = 0  # only exercise the best-checkpoint path
    return c


@pytest.fixture
def harness(monkeypatch):
    """log_metrics with save_state and evaluate_samples stubbed out."""
    saved = []
    monkeypatch.setattr(L, "save_state", lambda state, cfg, name=None: saved.append(name))

    def run(cfg, state, eval_scalars, step=10):
        monkeypatch.setattr(L, "evaluate_samples", lambda *a, **k: dict(eval_scalars))
        L.log_metrics(cfg, state, "cpu", 1.0, {}, step, 0.01)

    return run, saved


def _state():
    return types.SimpleNamespace(best_metrics={}, ts_eval_ctx=object())


def test_stage_without_ess_persists_no_sentinels(harness):
    """pretrain_v / pretrain_flowmap emit no ESS; -inf/inf must not reach the checkpoint."""
    run, saved = harness
    st = _state()
    run(_cfg("stage1"), st, {"train_diag/energy_w2": 5.0, "train_diag/ess": 0.2})

    assert st.best_metrics == {}, f"sentinels leaked into the checkpoint: {st.best_metrics}"
    assert saved == [], "no best checkpoint should be written without a resampled ESS metric"


def test_finite_bests_are_persisted_and_saved(harness):
    run, saved = harness
    st = _state()
    run(_cfg("stage3"), st, {
        "val/S/resampled/effective-sample-size": 0.05,
        "val/S/resampled/energy-w2": 3.0,
    })

    assert st.best_metrics == {"ess": 0.05, "energy_w2": 3.0}
    assert saved == ["stage3_best_ess", "stage3_best_energy_w2"]


def test_restart_does_not_demote_the_best_checkpoint(harness):
    """The N3 regression: a resumed run must not overwrite a better *_best_*.pt."""
    run, saved = harness

    # Run A reaches ESS 0.05 / energy_w2 3.0 and checkpoints them.
    st_a = _state()
    run(_cfg("stageX"), st_a, {
        "val/S/resampled/effective-sample-size": 0.05,
        "val/S/resampled/energy-w2": 3.0,
    })
    assert saved == ["stageX_best_ess", "stageX_best_energy_w2"]

    # Run B restarts from that checkpoint
    # and its first evaluation is WORSE. Nothing may be overwritten.
    saved.clear()
    st_b = types.SimpleNamespace(best_metrics=dict(st_a.best_metrics), ts_eval_ctx=object())
    run(_cfg("stageX"), st_b, {
        "val/S/resampled/effective-sample-size": 0.01,
        "val/S/resampled/energy-w2": 9.0,
    })
    assert saved == [], f"a worse checkpoint overwrote the best: {saved}"
    assert st_b.best_metrics == {"ess": 0.05, "energy_w2": 3.0}

    # A genuinely better ESS still wins.
    saved.clear()
    st_c = types.SimpleNamespace(best_metrics=dict(st_b.best_metrics), ts_eval_ctx=object())
    run(_cfg("stageX"), st_c, {
        "val/S/resampled/effective-sample-size": 0.09,
        "val/S/resampled/energy-w2": 9.0,
    })
    assert saved == ["stageX_best_ess"]
    assert st_c.best_metrics["ess"] == 0.09


def test_selection_uses_the_filtered_resampled_ess(harness):
    """The 0.002-filtered ESS is the standard; the -unfiltered diagnostic must be ignored."""
    run, saved = harness
    st = _state()
    run(_cfg("stageY"), st, {
        "val/S/resampled/effective-sample-size": 0.40,               # standard (post-clip)
        "val/S/proposal/effective-sample-size-unfiltered": 0.04,     # diagnostic only
        "val/S/resampled/energy-w2": 1.0,
    })
    assert st.best_metrics["ess"] == 0.40, "selection used the unfiltered diagnostic"


def test_unfiltered_diagnostic_alone_selects_nothing(harness):
    """Selection must not silently fall back to a different ESS definition."""
    run, saved = harness
    st = _state()
    run(_cfg("stageZ"), st, {
        "val/S/proposal/effective-sample-size-unfiltered": 0.03,
        "val/S/resampled/energy-w2": 1.0,
    })
    assert "ess" not in st.best_metrics
    assert saved == ["stageZ_best_energy_w2"]


def test_find_metric_is_deterministic_and_deprioritises_mean_aggregates():
    """_find_metric returned the first dict-insertion match; two ESS keys now exist."""
    scalars = {
        "val/mean/resampled/effective-sample-size": 0.99,
        "val/S/resampled/effective-sample-size": 0.04,
    }
    assert L._find_metric(scalars, "resampled", "effective-sample-size") == 0.04
    # Insertion order must not change the answer.
    assert L._find_metric(dict(reversed(list(scalars.items()))),
                          "resampled", "effective-sample-size") == 0.04
    assert L._find_metric(scalars, "nonexistent") is None


def test_pretraining_tracks_proposal_quality_separately(harness):
    run, saved = harness
    cfg = _cfg("pretraining")
    cfg.network = ml_collections.ConfigDict({"predict_divergence": False})
    state = _state()
    run(cfg, state, {"val/S/proposal/energy-w2": 2.0})
    assert state.best_metrics == {"proposal_energy_w2": 2.0}
    assert saved == ["pretraining_best_proposal_energy_w2"]
    saved.clear()
    run(cfg, state, {"val/S/proposal/energy-w2": 3.0})
    assert saved == []
    assert "ess" not in state.best_metrics and "energy_w2" not in state.best_metrics


def test_each_best_save_contains_all_updated_trackers(monkeypatch, tmp_path):
    import torch
    from common.state_utils import TrainState
    net = torch.nn.Linear(2, 1)
    state = TrainState(net, torch.optim.SGD(net.parameters(), lr=0.1), lambda _: 0.1, [0.999])
    state.ts_eval_ctx = object()
    cfg = _cfg('persisted')
    cfg.logging.output_folder = str(tmp_path)
    monkeypatch.setattr(L, '_wandb_active', lambda: False)
    values = {'val/S/resampled/effective-sample-size': .4, 'val/S/resampled/energy-w2': 2.}
    monkeypatch.setattr(L, 'evaluate_samples', lambda *a, **kw: values)
    L.log_metrics(cfg, state, 'cpu', 1., {}, 10, .1)
    expected = {'ess': .4, 'energy_w2': 2.}
    for suffix in ('ess', 'energy_w2'):
        saved = torch.load(tmp_path / f'persisted_best_{suffix}.pt', weights_only=False)
        assert saved['best_metrics'] == expected
    # Restore an actual best file; worse metrics must leave every file untouched.
    state.load_state_dict(saved)
    before = {p: p.read_bytes() for p in tmp_path.glob('*.pt')}
    values.update({'val/S/resampled/effective-sample-size': .2, 'val/S/resampled/energy-w2': 4.})
    L.log_metrics(cfg, state, 'cpu', 1., {}, 20, .1)
    assert {p: p.read_bytes() for p in before} == before


def test_same_name_fresh_states_are_independent(harness):
    run, saved = harness
    run(_cfg('same'), _state(), {'val/S/resampled/effective-sample-size': .8})
    second = _state()
    run(_cfg('same'), second, {'val/S/resampled/effective-sample-size': .2})
    assert saved == ['same_best_ess', 'same_best_ess']
    assert second.best_metrics == {'ess': .2}


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_selection_metrics_are_not_persisted(harness, value):
    run, saved = harness
    state = _state()
    run(_cfg('nonfinite'), state, {'val/S/resampled/effective-sample-size': value,
                                 'val/S/resampled/energy-w2': value})
    assert saved == [] and state.best_metrics == {}
