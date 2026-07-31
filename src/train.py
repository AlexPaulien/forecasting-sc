"""Entry point : runs a MLflow tracked backtest for a feature variant.
 
    python -m src.train --data data/rossmann.parquet --variant no_ly
    python -m src.train --data data/rossmann.parquet --variant full
    python -m src.train --data data/rossmann.parquet --variant all
"""
 
import argparse
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import subprocess
 
from .config import (PARAMS, HORIZON, NUM_BOOST_ROUND, VARIANTS,
                     TRACKING_URI, EXPERIMENT_NAME)
from .features import build_asof_features
from .data import build_origins
from .backtest import run_backtest, summarize


def git_info():
    """
    Current commit and state (dirty/clean) of the tree.
    Returns neutral value if not in git repo.
    """
    def run(cmd):
        return subprocess.check_output(cmd, cwd=".", stderr=subprocess.DEVNULL).decode().strip()
    try:
        commit = run(["git", "rev-parse", "HEAD"])
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        dirty = bool(run(["git", "status", "--porcelain"]))   # not empty = uncommitted modifications
        return {"git_commit": commit, "git_branch": branch, "git_dirty": str(dirty)}
    except Exception:
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": "unknown"}
 
 
def load_data(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
 
 
def run_and_log(df, asof, features, run_name):
    _, test_origins = build_origins(df)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(git_info())
        mlflow.log_params({
            "n_features": len(features),
            "horizon": HORIZON,
            "n_test_origins": len(test_origins),
            "num_boost_round": NUM_BOOST_ROUND,
            "has_sales_ly": "sales_ly" in features,
            **PARAMS,
        })
        mlflow.log_param("features", ", ".join(features))
 
        res, imp = run_backtest(df, asof, features)
        summary = summarize(res)
 
        # metrics by horizon (step = week used as plot in MLflow UI)
        for hw, row in summary.iterrows():
            for kpi in ["wmape", "bias", "fva"]:
                mlflow.log_metric(kpi, float(row[kpi]), step=int(hw))
 
        w = res.groupby("h_week")["n"].sum()
        mlflow.log_metric("wmape_global", float(np.average(summary["wmape"], weights=w)))
        mlflow.log_metric("bias_global", float(np.average(summary["bias"], weights=w)))
 
        # artefacts : detail, synthesis, importance
        res.to_csv("backtest_detail.csv", index=False)
        summary.round(4).to_csv("summary_by_horizon.csv")
        imp.to_csv("feature_importance.csv", header=["gain"])
        for f in ["backtest_detail.csv", "summary_by_horizon.csv", "feature_importance.csv"]:
            mlflow.log_artifact(f)
 
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.head(12).iloc[::-1].plot.barh(ax=ax)
        ax.set_xlabel("gain moyen (sur les folds)")
        ax.set_title(f"Feature importance — {run_name}")
        fig.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)
 
        print(f"[{run_name}] wmape par h_week : {summary['wmape'].round(4).to_dict()}")
        return summary
 
 
def main():
    p = argparse.ArgumentParser(description="LightGBM backtest with MLflow tracking")
    p.add_argument("--data", required=True, help="dataset path (.parquet ou .csv)")
    p.add_argument("--variant", default="no_ly", choices=list(VARIANTS) + ["all"])
    args = p.parse_args()
 
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
 
    df = load_data(args.data)
    asof = build_asof_features(df)   # computed once and reuse by all folds
 
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for v in variants:
        run_and_log(df, asof, VARIANTS[v], f"lgbm_{v}")
 
 
if __name__ == "__main__":
    main()
 