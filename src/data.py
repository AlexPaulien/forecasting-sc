"""build the store-origin-target combinations and folds for backtesting"""
 
import numpy as np
import pandas as pd
 
from .config import HORIZON, N_TEST_ORIGINS, CATEGORICAL
from .features import build_asof_features, snapshot_at
 
 
def make_training_frame(df, origins, horizon=HORIZON, asof=None):
    """
    Stack d+1 to d+12 target enriched with 'asof_xx' and sales_ly features for each origin
    """
    if asof is None:
        asof = build_asof_features(df)
    asof_cols = [c for c in asof.columns if c.startswith("asof_")]
 
    # same day the previous year
    ly = df[["Store", "Date", "Sales"]].rename(columns={"Sales": "sales_ly"})
    ly["Date"] = ly["Date"] + pd.Timedelta(days=364)
 
    frames = []
    for origin in pd.to_datetime(sorted(origins)):
        tgt = df[(df["Date"] > origin)
                 & (df["Date"] <= origin + pd.Timedelta(days=horizon))].copy()
        snap = snapshot_at(asof, origin, asof_cols)
        tgt = tgt.merge(snap, on="Store", how="inner")   # stoes without snaphot are left out
        tgt = tgt.merge(ly, on=["Store", "Date"], how="left")  # missing sales_ly is fine
        tgt["origin"] = origin
        tgt["h"] = (tgt["Date"] - origin).dt.days
        tgt["h_week"] = np.ceil(tgt["h"] / 7).astype(int)
        frames.append(tgt)
 
    if not frames:
        raise ValueError(f"no workable origin among: {list(origins)}")
    out = pd.concat(frames, ignore_index=True)
    return out[out["Open"] == 1].reset_index(drop=True)
 
 
def build_origins(df, horizon=HORIZON, n_test=N_TEST_ORIGINS):
    """
    Builds a grid of potential origins (every 14 days) while keeping the n last monthly origin for testing
    """
    candidate = pd.date_range(
        start=df["Date"].min() + pd.Timedelta(days=400),   # need 364 days of historical data
        end=df["Date"].max() - pd.Timedelta(days=horizon),
        freq="14D",
    )
    test = pd.date_range(
        end=df["Date"].max() - pd.Timedelta(days=horizon), periods=n_test, freq="MS"
    )
    return candidate, test
 
 
def make_fold(df, asof, origin, candidate_origins, horizon=HORIZON, margin=True):
    """One fold = (train, test) for one test origin.
 
    margin=True => trains only on origins for which all 42 targets are all before the test origin
    margin=False => can lead to partial target block for a given origin if it extends into the test data (truncation)
    """
    cutoff = origin - pd.Timedelta(days=horizon) if margin else origin
    tr_origins = candidate_origins[candidate_origins < cutoff]
    tr = make_training_frame(df, tr_origins, horizon, asof)
    tr = tr[tr["Date"] < origin]
    te = make_training_frame(df, [origin], horizon, asof)
    assert tr["Date"].max() < origin < te["Date"].min(), f"overflow to {origin.date()}"
    return tr, te


def categorical_levels(df):
    """Category to integer mapping, frozen on all data.
    Sorted to ensure that train and service use the same encoding."""
    return {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))}
            for c in CATEGORICAL}
 
 
def prep(d, cat_maps):
    """
    Add target date-related features and freeze the categorical level using integers.
    Those integers makes the use of MLflow easier
    (compute once for all the date to ensure we get the same encoding for every fold and set)
    """
    d = d.copy()
    d["month"] = d["Date"].dt.month
    d["day"] = d["Date"].dt.day
    d["weekofyear"] = d["Date"].dt.isocalendar().week.astype(int)
    for c in CATEGORICAL:
        # -1 potentially unknown categories while using service
        d[c] = d[c].astype(str).map(cat_maps[c]).fillna(-1).astype("int32")
    return d