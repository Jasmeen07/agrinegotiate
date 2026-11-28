"""Tuned heuristic: pick the time-dependent beta that maximises the role's own
gain capture against a pool of standard opponents, on tuning seeds only (0..99)."""
from __future__ import annotations

from ..agents import scripted as S
from ..env.core import F, B
from .evaluate import TUNE_SEED_START, evaluate

BETAS = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
POOL = ["Boulware", "Linear", "Conceder", "TitForTat"]


def tune_role(sc, role: int, n_episodes: int = 100) -> tuple[float, dict]:
    scores = {}
    for beta in BETAS:
        total = 0.0
        for opp in POOL:
            me = S.make(f"TD:{beta}", role, sc)
            other = S.make(opp, 1 - role, sc)
            farmer, buyer = (me, other) if role == F else (other, me)
            r = evaluate(sc, farmer, buyer, n_episodes, seed_start=TUNE_SEED_START)
            total += r["FGC"] if role == F else r["BGC"]
        scores[beta] = total / len(POOL)
    best = max(scores, key=scores.get)
    return best, scores


def tune(sc, n_episodes: int = 100) -> dict:
    bf, sf = tune_role(sc, F, n_episodes)
    bb, sb = tune_role(sc, B, n_episodes)
    return {"farmer_beta": bf, "buyer_beta": bb, "farmer_scores": sf, "buyer_scores": sb}
