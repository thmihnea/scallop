"""Distributed regression tests; numerical checks use real flow-map losses."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from common import distributed, loss_args
from configs.ace_a_nme import get_config


def test_global_batch_split_is_exact_and_repeatable(monkeypatch):
    monkeypatch.setattr(distributed, 'world_size', lambda: 4)
    cfg = get_config(0)
    cfg.optimization.bs = 1024
    distributed.configure_batch(cfg)
    distributed.configure_batch(cfg)
    assert cfg.optimization.bs == 1024
    assert cfg.optimization.local_bs == 256
    assert cfg.optimization.local_diag_bs == 128
    s, t, _, _ = loss_args.sample_times(cfg, torch.Generator().manual_seed(8))
    assert len(s) == 256
    assert torch.equal(s[:128], t[:128])
    assert (s[128:] != t[128:]).all()


@pytest.mark.parametrize('bs,fraction', [(1023, 0.5), (4, 0.5), (1024, 0.334), (0, 1.0)])
def test_reject_inexact_batch_split(monkeypatch, bs, fraction):
    monkeypatch.setattr(distributed, 'world_size', lambda: 4)
    cfg = get_config(0)
    cfg.optimization.bs = bs
    cfg.optimization.diag_fraction = fraction
    with pytest.raises(ValueError):
        distributed.configure_batch(cfg)


def test_data_shards_and_epoch_shuffle(monkeypatch):
    pytest.importorskip('transferable_samplers')
    from common import ts_data
    monkeypatch.setattr(distributed, 'active', lambda: True)
    monkeypatch.setattr(distributed, 'world_size', lambda: 4)
    data = torch.randn(64, 22, 3)
    shards = []
    for rank in range(4):
        monkeypatch.setattr(distributed, 'rank', lambda: rank)
        loader = ts_data.build_train_loader(data, 1., bs=4, num_workers=0, seed=13)
        first = list(loader.sampler)
        shards.append(set(first))
        stream = ts_data.InfiniteTSLoader(loader, 'cpu')
        for _ in range(len(loader) + 1):
            assert next(stream).shape == (4, 22, 3)
        assert stream.epoch == 1
        assert list(loader.sampler) != first
    assert len(set.union(*shards)) == 64
    assert sum(len(s) for s in shards) == 64
    for i in range(4):
        for j in range(i):
            assert shards[i].isdisjoint(shards[j])


def test_two_rank_numerical_equivalence_and_skip_recovery(tmp_path):
    output = tmp_path / 'result.json'
    worker = Path(__file__).with_name('ddp_worker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                             '--nproc_per_node=2', str(worker), '--output', str(output)],
                            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=180)
    assert result.returncode == 0, result.stdout
    report = json.loads(output.read_text())
    assert report['world_size'] == 2
    assert len(report['results']) == 13
    assert report['results'][-1]['incomplete_on_all_ranks']
    assert {r['mode'] for r in report['results'] if r['stopgrad'] == 'convex'} == {
        'pretrain_flowmap', 'dlogpdt', 'hutchinson',
    }
