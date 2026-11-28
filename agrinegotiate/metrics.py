"""Evaluation metrics. Ratios of sums across episodes, never means of ratios."""
from __future__ import annotations

from typing import Iterable

import numpy as np

E_MIN_FOR_SHARE = 0.05


def gini(x) -> float:
    x = np.asarray(x, float)
    assert (x >= -1e-12).all(), "Gini needs non-negative inputs (use price received, not gains)"
    if x.size == 0 or x.mean() <= 0:
        return float("nan")
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * x.size ** 2 * x.mean()))


def jain(x) -> float:
    x = np.asarray(x, float)
    assert (x >= -1e-12).all(), "Jain's index needs non-negative inputs"
    if x.size == 0 or (x ** 2).sum() <= 0:
        return float("nan")
    return float(x.sum() ** 2 / (x.size * (x ** 2).sum()))


def aggregate(results: Iterable) -> dict:
    """Aggregate a list of EpisodeResult into the reported metrics."""
    results = list(results)
    n_total = len(results)
    used = [r for r in results if r.g_star > 1e-9]
    out = {"episodes": n_total, "episodes_no_trade_possible": n_total - len(used)}
    if not used:
        return out
    g_star = sum(r.g_star for r in used)
    f_gain = sum(float(r.farmer_gain.sum()) for r in used)
    b_gain = sum(float(r.buyer_gain.sum()) for r in used)
    fees = sum(float(r.buyer_fees.sum()) for r in used)
    E, FGC, BGC = (f_gain + b_gain) / g_star, f_gain / g_star, b_gain / g_star
    assert abs(E - (FGC + BGC)) < 1e-9, "identity E = FGC + BGC violated"
    deals = [d for r in used for d in r.deals]
    n_opt = sum(r.n_opt_pairs for r in used)
    phis = np.array([d["phi"] for d in deals if d["phi"] is not None], float)
    out.update(
        E=E, FGC=FGC, BGC=BGC, BGC_net_of_fees=(b_gain - fees) / g_star,
        S_F=(FGC / E) if (E >= E_MIN_FOR_SHARE and FGC >= 0 and BGC >= 0) else None,
        agreement_rate=len(deals) / n_opt if n_opt else None,
        deals=len(deals), fees_over_gstar=fees / g_star,
        failed_accepts_per_episode=sum(r.failed_accepts for r in used) / len(used),
        decay_loss=sum(r.decay_loss for r in used) / g_star,
        fallback_rate=float(np.mean(np.concatenate([~r.sold for r in used]))),
        price_share=float(np.mean(np.concatenate([r.price_received / r.level for r in used]))),
    )
    if deals:
        out.update(
            phi_mean=float(phis.mean()) if phis.size else None,
            phi_median=float(np.median(phis)) if phis.size else None,
            phi_p10=float(np.percentile(phis, 10)) if phis.size else None,
            nash_distance=float(np.abs(phis - 0.5).mean()) if phis.size else None,
            low_share_deal_rate=float((phis < 0.1).mean()) if phis.size else None,
            time_to_agreement=float(np.mean([d["t"] for d in deals])),
            j_c=float(np.mean([d["shortfall"] / d["g_t"] for d in deals if d["g_t"] > 1e-9] or [0.0])),
        )
    else:
        out.update(phi_mean=None, phi_median=None, phi_p10=None, nash_distance=None,
                   low_share_deal_rate=None, time_to_agreement=None, j_c=None)
    multi = [r for r in used if r.price_received.size > 1]
    if multi:
        out["gini_price_received"] = float(np.nanmean([gini(r.price_received) for r in multi]))
        out["jain_price_received"] = float(np.nanmean([jain(r.price_received) for r in multi]))
    return out
