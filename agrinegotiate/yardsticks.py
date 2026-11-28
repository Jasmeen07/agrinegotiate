"""Closed-form reference points. Nothing here is learned.

Conventions: prices are in grid units. v = buyer value of a fresh lot,
o_f = farmer fallback price of a fresh lot, o_b = buyer fallback payoff,
rho = freshness kept per step, t = step at which the deal closes.
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def gains(v: float, o_f: float, o_b: float, rho: float = 1.0, t: int = 0) -> float:
    """Gains from trade G_t = v*rho^t - o_f*rho^t - o_b."""
    d = rho ** t
    return v * d - o_f * d - o_b


def nash_price(v: float, o_f: float, o_b: float, rho: float = 1.0, t: int = 0) -> Optional[float]:
    """Symmetric Nash bargaining price. None when no trade is possible."""
    g = gains(v, o_f, o_b, rho, t)
    if g <= 0:
        return None
    return o_f * rho ** t + 0.5 * g


def target_price(v: float, o_f: float, o_b: float, tau: float, rho: float = 1.0, t: int = 0) -> Optional[float]:
    """Price at which the farmer's gain share equals tau."""
    g = gains(v, o_f, o_b, rho, t)
    if g <= 0:
        return None
    return o_f * rho ** t + tau * g


def gain_share(p: float, v: float, o_f: float, o_b: float, rho: float = 1.0, t: int = 0) -> Optional[float]:
    """Farmer's share phi of the gains for a deal at price p. None if G_t <= 0."""
    g = gains(v, o_f, o_b, rho, t)
    if g <= 0:
        return None
    return (p - o_f * rho ** t) / g


def rubinstein_share(delta_f: float, delta_b: float, farmer_first: bool = True, k: int = 1) -> float:
    """Farmer's share in Rubinstein alternating offers.

    delta_* are per-step discount factors; each round takes k steps.
    First proposer's share = (1 - d_other) / (1 - d_f * d_b).
    """
    df, db = delta_f ** k, delta_b ** k
    denom = 1.0 - df * db
    if farmer_first:
        return (1.0 - db) / denom
    return 1.0 - (1.0 - df) / denom


def rubinstein_limit(delta_f: float, delta_b: float) -> float:
    """Short-round limit of the farmer's share: r_B / (r_F + r_B), r = -ln(delta)."""
    r_f, r_b = -np.log(delta_f), -np.log(delta_b)
    return float(r_b / (r_f + r_b))


def pair_gain_matrix(v: Sequence[float], o_f: Sequence[float], o_b: Sequence[float]) -> np.ndarray:
    """[N, M] matrix of max(0, v_j - o_f_i - o_b_j) at step 0."""
    v = np.asarray(v, float)[None, :]
    o_b = np.asarray(o_b, float)[None, :]
    o_f = np.asarray(o_f, float)[:, None]
    return np.maximum(0.0, v - o_f - o_b)


def max_gains(v: Sequence[float], o_f: Sequence[float], o_b: Sequence[float],
              capacity: Sequence[int]) -> tuple[float, list[tuple[int, int]]]:
    """Maximum total gains from trade G* and one optimal matching.

    One lot per farmer, at most capacity[j] lots per buyer, all at step 0.
    Returns (G*, [(farmer, buyer), ...]) listing only pairs with positive gain.
    """
    g = pair_gain_matrix(v, o_f, o_b)
    cols = [j for j, k in enumerate(capacity) for _ in range(int(k))]
    if g.size == 0 or not cols:
        return 0.0, []
    expanded = g[:, cols]
    rows, cidx = linear_sum_assignment(expanded, maximize=True)
    pairs = [(int(i), int(cols[c])) for i, c in zip(rows, cidx) if expanded[i, c] > 0]
    return float(sum(g[i, j] for i, j in pairs)), pairs


def max_gains_bruteforce(v, o_f, o_b, capacity) -> float:
    """Exhaustive search over all matchings. For tests on tiny cases only."""
    g = pair_gain_matrix(v, o_f, o_b)
    n = g.shape[0]
    slots = [j for j, k in enumerate(capacity) for _ in range(int(k))] + [None] * n
    best = 0.0
    for perm in set(itertools.permutations(slots, n)):
        best = max(best, sum(g[i, j] for i, j in enumerate(perm) if j is not None))
    return float(best)
