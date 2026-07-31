"""Features as known at the origin: moving averages"""
 
import numpy as np
import pandas as pd
 
 
def build_asof_features(df, windows=(7, 28, 91)):
    """Aggregate sales for a given windows ending at the current date.
 
    Pour une origine O, lire la ligne Date == O (ou la dernière avant O via
    snapshot_at) donne les features de départ des 42 cibles suivantes.

    For an origin 0, the row Date == 0 (or the last available snapshot before O)
    gives the features to be used for the 42 targers (forecating horizon)
 
    the close days are filtered out.
    """
    d = df.sort_values(["Store", "Date"]).copy()
    d["SalesOpen"] = d["Sales"].where(d["Open"] == 1)
    g = d.groupby("Store")["SalesOpen"]
 
    out = d[["Store", "Date"]].copy()
    for w in windows:
        mp = max(3, w // 3)
        out[f"asof_mean_{w}d"] = g.transform(
            lambda s, w=w, mp=mp: s.rolling(w, min_periods=mp).mean()
        )
        out[f"asof_std_{w}d"] = g.transform(
            lambda s, w=w, mp=mp: s.rolling(w, min_periods=mp).std()
        )
 
    out["asof_ratio_7_28"] = out["asof_mean_7d"] / out["asof_mean_28d"]
    out["asof_ratio_28_91"] = out["asof_mean_28d"] / out["asof_mean_91d"]
    out["asof_cv_28"] = out["asof_std_28d"] / out["asof_mean_28d"]
 
    return out.sort_values(["Store", "Date"]).reset_index(drop=True)
 

def snapshot_at(asof, origin, cols):
    """
    Last known state of each store at the origin date (included)
    This to avoid removing stores just because they were close at origin
    Supposes `asof` is already sorted by (Store, Date).   
    """
    s = asof[asof["Date"] <= origin].drop_duplicates("Store", keep="last")
    s = s.rename(columns={"Date": "snap_date"})
    s["days_since_snap"] = (origin - s["snap_date"]).dt.days
    return s[["Store", "days_since_snap"] + cols]