"""Calibration statistics from cleaned prices.

Per commodity: median modal price, standard deviation of lagged log returns (the farmer's
stale-information noise sigma_info at lags 1, 3 and 7 days) and the relative daily spread.
Gaps longer than 3 days are never bridged.
NOT YET RUN ON REAL DATA.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

LAGS = (1, 3, 7)
MAX_GAP_DAYS = 3


def calibrate(df: pd.DataFrame) -> dict:
    out = {}
    for commodity, g in df.groupby("commodity"):
        rets = {lag: [] for lag in LAGS}
        for _, s in g.groupby("market"):
            s = s.set_index("date")["modal"].asfreq("D")
            s = s.interpolate(limit=MAX_GAP_DAYS, limit_area="inside")
            for lag in LAGS:
                rets[lag].append(np.log(s / s.shift(lag)).dropna().to_numpy())
        out[str(commodity)] = {
            "rows": int(len(g)),
            "markets": int(g["market"].nunique()),
            "median_modal_rs_per_quintal": float(g["modal"].median()),
            "sigma_info_by_lag_days": {str(lag): float(np.concatenate(rets[lag]).std()) if rets[lag] and
                                       sum(len(r) for r in rets[lag]) > 1 else None for lag in LAGS},
            "relative_spread": float(((g["max"] - g["min"]) / g["modal"]).median()),
        }
    return out


def run(csv_in: str, parquet_out: str, json_out: str) -> dict:
    from .clean import clean
    df = clean(pd.read_csv(csv_in))
    df.to_parquet(parquet_out, index=False)
    cal = calibrate(df)
    with open(json_out, "w") as f:
        json.dump(cal, f, indent=1)
    return cal
