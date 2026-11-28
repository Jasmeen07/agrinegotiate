"""Calibrate the scenarios from real Agmarknet mandi prices (data/real/lemon_daily_agmarknet.csv.gz).

    python experiments/calibrate_real.py

Writes data/real/calibration.json with three things the simulator needs:
  level_samples   quantiles of the national daily price relative to its median -> the market "price level" of an episode
  sigma_info      std of log price changes over 3 days at the same mandi -> how stale a farmer's price signal is
  theta_spread    same-day price spread between mandis of one state -> how different farmers' fallback options are
Prices are winsorised per mandi and the spreads use clipped or robust statistics, so a few mistyped prices cannot inflate them.
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data/real/lemon_daily_agmarknet.csv.gz", ROOT / "data/real/calibration.json"


def robust_std(x) -> float:
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


d = pd.read_csv(SRC, parse_dates=["date"])
d["mandi"] = d["state"] + "/" + d["market"]          # market names repeat across states
lo, hi = d.groupby("mandi")["modal"].transform(lambda s: s.quantile(.01)), d.groupby("mandi")["modal"].transform(lambda s: s.quantile(.99))
d["modal"] = d["modal"].clip(lo, hi)                       # winsorise each mandi at its 1st-99th percentile
md = d.groupby(["state", "mandi", "date"], as_index=False)["modal"].median()   # one price per mandi and day

# 1. price level of an episode = the national price relative to its own local (61-day) median, i.e. the short-run
#    price shock a negotiation takes place in (the slow seasonal swing is not part of one 30-step episode)
nat = md.groupby("date")["modal"].median().asfreq("D").interpolate(limit=3)
ratio = (nat / nat.rolling(61, center=True, min_periods=30).median()).dropna()
qs = np.quantile(ratio, np.linspace(0.02, 0.98, 25))
level_samples = [round(float(np.clip(0.9 * q, 0.6, 1.15)), 4) for q in qs]   # scaled so the typical level is 0.9 and stays on the grid

# 2. information staleness: log price change at the same mandi after k days (extreme 2.5% on each side clipped: typos)
wide = md.pivot(index="date", columns="mandi", values="modal").asfreq("D")
sigma = {}
for k in (1, 3, 7):
    r = np.log(wide / wide.shift(k)).to_numpy().ravel()
    r = r[np.isfinite(r)]
    c = np.clip(r, *np.quantile(r, [0.025, 0.975]))
    sigma[str(k)] = dict(std_clipped=round(float(c.std()), 4), share_unchanged=round(float((r == 0).mean()), 3), pairs=int(r.size))

# 3. same-day spread between mandis of a state (>= 5 mandis reporting)
g = md.groupby(["state", "date"])["modal"]
n, med = g.transform("size"), g.transform("median")
sub = md[n >= 5].assign(rel=lambda x: x["modal"] / med[n >= 5])
spread = sub.groupby(["state", "date"])["rel"].apply(robust_std)
rel_spread = float(spread.median())
theta = 0.45
theta_spread = round(float(min(0.20, rel_spread * np.sqrt(3) * theta)), 3)   # U(theta +- s) has relative std s / (sqrt(3) theta)

cal = dict(source="Agmarknet daily lemon prices, Government Open Data Licence India", rows=int(len(d)), mandis=int(d.mandi.nunique()),
           states=int(d.state.nunique()), days=int(d.date.nunique()), first=str(d.date.min().date()), last=str(d.date.max().date()),
           price_level_ratio_quantiles={f"p{q}": round(float(ratio.quantile(q / 100)), 3) for q in (2, 10, 50, 90, 98)},
           level_samples=level_samples, sigma_info=sigma, rel_spread_between_mandis=round(rel_spread, 4),
           theta_spread=theta_spread, sigma_info_used=sigma["3"]["std_clipped"])
OUT.write_text(json.dumps(cal, indent=1))
print(json.dumps({k: v for k, v in cal.items() if k != "level_samples"}, indent=1))
print("level_samples range", min(level_samples), max(level_samples))
