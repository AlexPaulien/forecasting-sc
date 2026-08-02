"""Service prediction : reconstructs features for a given origin
and predict d+1 to d+42

Client parameters (all optional) :
  - store   : on single store or all
  - horizon : 1..42 (défault 42)
  - origin  : origin or last know date

returns forecasts by (store, date) in euros
"""

import numpy as np
import pandas as pd

from .features import build_asof_features, snapshot_at
from .data import prep, categorical_levels
from .config import HORIZON, VARIANTS


class PredictError(ValueError):
    """Request error (parameter not accepted). API will return error 400."""


def resolve_origin(df, origin=None):
    """Validate/position the origin in the allowed range of dates"""
    dmin, dmax = df["Date"].min(), df["Date"].max()
    if origin is None:
        return dmax
    origin = pd.to_datetime(origin)
    # need 400 days of historical date for stable asof features
    earliest = dmin + pd.Timedelta(days=400)
    if origin < earliest:
        raise PredictError(
            f"origin {origin.date()} too early : min {earliest.date()} "
            f"(400 days of historical data required)")
    latest = dmax - pd.Timedelta(days=HORIZON)
    if origin > latest:
        raise PredictError(
            f"origin {origin.date()} date is too late for a backtest demo : "
            f"max {latest.date()} (need {HORIZON} j of actuals to build the comparison)")
    return origin


def resolve_horizon(horizon=None):
    if horizon is None:
        return HORIZON
    horizon = int(horizon)
    if not (1 <= horizon <= HORIZON):
        raise PredictError(f"horizon {horizon} outside of range : expects 1..{HORIZON}")
    return horizon


def build_inference_frame(df, calendar, asof, cat_maps, features,
                          origin=None, horizon=None, store=None):
    """Frame of features needed for prediction.

    df       : actuals (for asof and snapshot)
    calendar : feature known in advance for target dates
               (Store, Date, Open, Promo, SchoolHoliday, StateHoliday, DayOfWeek).
    """
    origin = resolve_origin(df, origin)
    horizon = resolve_horizon(horizon)
    asof_cols = [c for c in asof.columns if c.startswith("asof_")]

    # snapshot : last known state of each store at origin
    snap = snapshot_at(asof, origin, asof_cols)

    # targets :
    tgt = calendar[
        (calendar["Date"] > origin)
        & (calendar["Date"] <= origin + pd.Timedelta(days=horizon))
        & (calendar["Open"] == 1)
    ].copy()

    if store is not None:
        if store not in df["Store"].unique():
            raise PredictError(f"store {store} unknown")
        tgt = tgt[tgt["Store"] == store]
        snap = snap[snap["Store"] == store]

    if tgt.empty:
        raise PredictError(
            f"no open target for origin {origin.date()} / horizon {horizon}"
            + (f" / store {store}" if store is not None else ""))

    tgt = tgt.merge(snap, on="Store", how="inner")
    tgt["origin"] = origin
    tgt["h"] = (tgt["Date"] - origin).dt.days
    tgt["h_week"] = np.ceil(tgt["h"] / 7).astype(int)

    # keep the actual id from the data, instead of prep encoding of store into an integer
    tgt["store_id"] = tgt["Store"]

    tgt = prep(tgt, cat_maps)
    return tgt, origin, horizon


def predict(model, df, calendar, asof, cat_maps, features,
            origin=None, horizon=None, store=None):
    """Builds features, predicts, changes back to euros. 
    Returns a sorted dataframe (Store, Date) with `sales` column as prediction."""
    tgt, origin, horizon = build_inference_frame(
        df, calendar, asof, cat_maps, features,
        origin=origin, horizon=horizon, store=store)

    log_pred = model.predict(tgt[features])
    tgt["sales"] = np.clip(np.expm1(log_pred), 0, None)

    # store_id = actual id of the store in the original data (not the encoding
    # store_id to be used for display and merge (to avoid merging the two wrong stores)
    out = tgt[["store_id", "Date", "h", "sales"]].rename(columns={"store_id": "Store"})
    out = out.sort_values(["Store", "Date"])

    # backtest demo: join actuals when it is known
    actual = df[["Store", "Date", "Sales"]].rename(columns={"Sales": "actual"})
    out = out.merge(actual, on=["Store", "Date"], how="left")

    return out.reset_index(drop=True), origin, horizon