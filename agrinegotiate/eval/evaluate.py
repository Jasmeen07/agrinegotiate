"""Episode runner and fixed-seed evaluation."""
from __future__ import annotations

import numpy as np

from .. import metrics
from ..env.core import Market, F, B

EVAL_SEED_START = 50_000          # evaluation episodes: 50000..50999
TUNE_SEED_START = 0               # tuning: 0..99
TRAIN_SEED_START = 1000           # training seeds: 1000..1009


def run_episode(sc, farmer, buyer, seed: int, mechanism=None, log_events: bool = False,
                delay_aware: bool = True):
    env = Market(sc, mechanism=mechanism, delay_aware=delay_aware, log_events=log_events)
    env.reset(seed)
    farmer.reset(seed)
    buyer.reset(seed)
    while not env.done:
        obs, mask, active = env.observe()
        acts = np.stack([farmer.act(env, obs[F], mask[F], active[F]),
                         buyer.act(env, obs[B], mask[B], active[B])])
        env.step(acts, mask)
    return env.result()


def evaluate(sc, farmer, buyer, n_episodes: int = 1000, seed_start: int = EVAL_SEED_START,
             mechanism=None, delay_aware: bool = True) -> dict:
    results = [run_episode(sc, farmer, buyer, seed_start + k, mechanism, delay_aware=delay_aware)
               for k in range(n_episodes)]
    return metrics.aggregate(results)
