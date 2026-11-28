"""Cleaning and calibration logic on a tiny hand-made frame (test fixture, not real data)."""
import pandas as pd

from agrinegotiate.data.calibrate import calibrate
from agrinegotiate.data.clean import clean


def test_clean_drops_bad_rows_and_calibrates():
    days = pd.date_range("2025-01-01", periods=20, freq="D").strftime("%d/%m/%Y")
    df = pd.DataFrame({"State": "X", "District": "Y", "Market": "M1", "Commodity": "Tomato", "Variety": "Local",
                       "Arrival_Date": days, "Min_Price": 900.0, "Max_Price": 1250.0,
                       "Modal_Price": [1000.0 + 10 * k for k in range(20)]})
    bad = df.iloc[:3].copy()
    bad.loc[bad.index[0], "Modal_Price"] = 0            # zero price
    bad.loc[bad.index[1], "Modal_Price"] = 5000         # outside min..max
    out = clean(pd.concat([df, bad, df.iloc[:2]]))      # plus exact duplicates
    assert len(out) == 20 and list(out.columns) == ["market", "commodity", "date", "min", "max", "modal"]
    cal = calibrate(out)["Tomato"]
    assert cal["markets"] == 1 and cal["sigma_info_by_lag_days"]["1"] >= 0
    assert 0.25 < cal["relative_spread"] < 0.40
