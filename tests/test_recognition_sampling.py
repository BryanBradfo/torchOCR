"""references/recognition/train.py: per-epoch sampling of large word corpora."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "references" / "recognition"))

from train import epoch_sampler, parse_lmdb_spec  # noqa: E402


def test_parse_lmdb_spec():
    assert parse_lmdb_spec("/data/mj") == (Path("/data/mj"), None)
    assert parse_lmdb_spec("/data/mj:300000") == (Path("/data/mj"), 300000)


def test_epoch_sampler_keeps_real_data_whole_and_subsamples_large_corpora():
    # Sources of 100 and 50 words used whole, plus a 10_000-word corpus capped at 200 per epoch.
    sampler = epoch_sampler([100, 50, 10_000], [None, None, 200])
    assert sampler.num_samples == 350
    torch.manual_seed(0)
    drawn = torch.tensor(list(iter(sampler)))
    from_corpus = (drawn >= 150).float().mean().item()
    assert abs(from_corpus - 200 / 350) < 0.06


def test_epoch_sampler_without_caps_is_a_plain_shuffle():
    sampler = epoch_sampler([10, 20], [None, None])
    assert isinstance(sampler, torch.utils.data.RandomSampler) and len(list(iter(sampler))) == 30
