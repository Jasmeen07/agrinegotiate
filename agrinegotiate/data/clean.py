"""Cleaning rules for AGMARKNET mandi price rows.

Expected columns: State, District, Market, Commodity, Variety, Arrival_Date,
Min_Price, Max_Price, Modal_Price (rupees per quintal). Column names differ between
the data.gov.in API and the Kaggle mirrors; `normalise_columns` maps the variants we know of.
NOT YET RUN ON REAL DATA: this sandbox cannot reach data.gov.in or Kaggle.
"""
from __future__ import annotations

import pandas as pd

ALIASES = {
    "state": "state", "district": "district", "market": "market", "commodity": "commodity",
    "variety": "variety", "arrival_date": "date", "date": "date",
    "min_price": "min", "max_price": "max", "modal_price": "modal",
    "min price": "min", "max price": "max", "modal price": "modal",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: ALIASES.get(c.strip().lower().replace("_x0020_", "_"), c) for c in df.columns})
    need = ["market", "commodity", "date", "min", "max", "modal"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns after renaming: {missing}; have {list(df.columns)}")
    return df[need].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(df)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for c in ("min", "max", "modal"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "min", "max", "modal"])
    df = df[(df["modal"] > 0) & (df["min"] > 0) & (df["max"] > 0)]
    df = df[(df["modal"] >= df["min"]) & (df["modal"] <= df["max"])]
    df = df.drop_duplicates(subset=["market", "commodity", "date"])
    lo = df.groupby(["market", "commodity"])["modal"].transform(lambda s: s.quantile(0.01))
    hi = df.groupby(["market", "commodity"])["modal"].transform(lambda s: s.quantile(0.99))
    df["modal"] = df["modal"].clip(lo, hi)
    return df.sort_values(["commodity", "market", "date"]).reset_index(drop=True)
