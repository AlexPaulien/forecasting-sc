"""Evaluation metrics and backtest with moving origins"""
 
import numpy as np
import pandas as pd
import lightgbm as lgb
 
from .config import PARAMS, CATEGORICAL, NUM_BOOST_ROUND, HORIZON
from .data import make_fold, prep, categorical_levels, build_origins
 
 
def wmape(y, yhat):
    return np.abs(y - yhat).sum() / y.sum()
 
 
def bias(y, yhat):
    return (yhat - y).sum() / y.sum()
 
 
def run_backtest(df, asof, features, num_boost_round=NUM_BOOST_ROUND):
    """train lightGBM model a fold a a time and evaluate on test origin.
 
    Returns (res, imp) :
      - res : one line by (origin, h_week) avec wmape / bias / naive seasonal baseline
      - imp : average gain by feature for all folds (pd.Series)
    """
    cat_levels = categorical_levels(df)
    candidate_origins, test_origins = build_origins(df)
 
    rows, imps = [], []
    for origin in test_origins:
        tr, te = make_fold(df, asof, origin, candidate_origins)
        tr, te = prep(tr, cat_levels), prep(te, cat_levels)
 
        dtrain = lgb.Dataset(tr[features], np.log1p(tr["Sales"]),
                             categorical_feature=CATEGORICAL)
        model = lgb.train(PARAMS, dtrain, num_boost_round=num_boost_round)
 
        te = te.assign(pred=np.clip(np.expm1(model.predict(te[features])), 0, None))
        imps.append(pd.Series(model.feature_importance(importance_type="gain"),
                              index=features))
 
        for hw, g in te.groupby("h_week"):
            m = g["sales_ly"].notna()
            rows.append(dict(
                origin=origin.date(), h_week=int(hw), n=len(g),
                wmape=wmape(g["Sales"], g["pred"]),
                bias=bias(g["Sales"], g["pred"]),
                wmape_ly=wmape(g.loc[m, "Sales"], g.loc[m, "sales_ly"]) if m.any() else np.nan,
                n_ly=int(m.sum()),
            ))
 
    res = pd.DataFrame(rows)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    return res, imp
 
 
def summarize(res):
    """Aggregate res by h_week: wmape, bias, fva vs naive seasonal baseline"""
    def wavg(g, col, weight):
        return np.average(g[col], weights=g[weight])

    rows = []
    for hw, g in res.groupby("h_week"):
        rows.append({
            "h_week": int(hw),
            "wmape": wavg(g, "wmape", "n"),
            "bias": wavg(g, "bias", "n"),
            "wmape_naif_ly": wavg(g, "wmape_ly", "n_ly"),
        })
    summary = pd.DataFrame(rows).set_index("h_week")
    summary["fva"] = 1 - summary["wmape"] / summary["wmape_naif_ly"]
    return summary