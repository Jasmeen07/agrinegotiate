"""Aggregate per-seed scores: interquartile mean with a 95% stratified bootstrap
confidence interval (rliable's statistic), plus mean and standard deviation and a failed-run count."""
from __future__ import annotations

import json
import pathlib

import numpy as np

KEYS = ("E", "FGC", "BGC", "S_F", "agreement_rate", "price_share", "fees_over_gstar", "lam_fair", "j_c",
        "time_to_agreement", "phi_mean")
FAILED_E = 0.1


def load_runs(folder) -> list[dict]:
    runs = []
    for f in sorted(pathlib.Path(folder).glob("seed_*/metrics.json")):
        runs.append(json.loads(f.read_text()))
    return runs


def iqm(x) -> float:
    """Interquartile mean: mean of the middle 50% of runs (same statistic as rliable)."""
    from scipy.stats import trim_mean
    return float(trim_mean(np.asarray(x, float), 0.25))


def iqm_ci(values, reps: int = 5000, seed: int = 0):
    """IQM with a 95% percentile-bootstrap interval, resampling runs with replacement.

    For a single task this is the stratified bootstrap rliable uses. Implemented here because
    rliable 1.2.0 fails to import with pandas 3 (its `arch` dependency); see DECISIONS.md #16.
    """
    x = np.asarray(values, float)
    boots = _bootstrap_iqms(x, np.random.default_rng(seed), reps)
    return iqm(x), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def iqm_diff_ci(a, b, reps: int = 5000, seed: int = 0):
    """IQM(a) - IQM(b) with a 95% bootstrap interval; the two groups are resampled independently."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = _bootstrap_iqms(a, rng, reps) - _bootstrap_iqms(b, rng, reps)
    return iqm(a) - iqm(b), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _bootstrap_iqms(x, rng, reps):
    """IQM of each of `reps` resamples of x drawn with replacement."""
    samp = np.sort(rng.choice(x, size=(reps, x.size), replace=True), axis=1)
    cut = int(0.25 * x.size)                               # same trimming rule as scipy.stats.trim_mean
    return samp[:, cut:x.size - cut].mean(axis=1)


def summarise(folder) -> dict:
    runs = load_runs(folder)
    out = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
           "failed_runs": sum(1 for r in runs if (r["last10_checkpoint_mean"]["E"] or 0) < FAILED_E)}
    for k in KEYS:
        vals = [r["last10_checkpoint_mean"].get(k) for r in runs]
        defined = [v for v in vals if v is not None]
        out[k] = {"per_seed": vals, "undefined": len(vals) - len(defined)}
        if len(defined) >= 3:
            iqm, lo, hi = iqm_ci(defined)
            out[k].update(iqm=iqm, ci95=[lo, hi], mean=float(np.mean(defined)), std=float(np.std(defined, ddof=1)),
                          min=float(np.min(defined)), max=float(np.max(defined)))
    return out


if __name__ == "__main__":
    import sys
    for folder in sys.argv[1:]:
        s = summarise(folder)
        pathlib.Path(folder, "summary.json").write_text(json.dumps(s, indent=1))
        print(folder, "seeds:", s["n_seeds"], "failed:", s["failed_runs"])
        for k in KEYS[:5]:
            if "iqm" in s[k]:
                print(f"  {k:15s} IQM {s[k]['iqm']:.3f}  95% CI [{s[k]['ci95'][0]:.3f}, {s[k]['ci95'][1]:.3f}]  "
                      f"mean {s[k]['mean']:.3f} sd {s[k]['std']:.3f}  range [{s[k]['min']:.3f}, {s[k]['max']:.3f}]")
