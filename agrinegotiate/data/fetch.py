"""Fetch current mandi prices from data.gov.in.

UNVERIFIED AND NOT RUN: the resource id below comes from the literature review and must be
confirmed on data.gov.in. The API appears to serve current-day prices only; for history use a
Kaggle mirror and pass its CSV to `agrinegotiate.data.calibrate.run`. Check the licence first.

Usage: DATA_GOV_IN_KEY=... python -m agrinegotiate.data.fetch out.csv --commodity Tomato
"""
import argparse
import os

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"   # Verify before use
URL = "https://api.data.gov.in/resource/" + RESOURCE_ID


def fetch(commodity: str | None = None, limit: int = 1000, max_pages: int = 50):
    import pandas as pd
    import requests
    key = os.environ["DATA_GOV_IN_KEY"]
    rows, offset = [], 0
    for _ in range(max_pages):
        params = {"api-key": key, "format": "json", "limit": limit, "offset": offset}
        if commodity:
            params["filters[commodity]"] = commodity
        r = requests.get(URL, params=params, timeout=60)
        r.raise_for_status()
        recs = r.json().get("records", [])
        if not recs:
            break
        rows += recs
        offset += limit
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--commodity")
    a = ap.parse_args()
    fetch(a.commodity).to_csv(a.out, index=False)
