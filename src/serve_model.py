"""
Trains service model on all origins (no date set aside for testing) and logs it into mlflow
with a signature ready to be called by an API
"""
 
import argparse
import subprocess
 
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
 
from .config import (PARAMS, CATEGORICAL, NUM_BOOST_ROUND, VARIANTS,
                     TRACKING_URI, EXPERIMENT_NAME)
from .features import build_asof_features
from .data import make_training_frame, build_origins, prep, categorical_levels
 
 
def git_info():
    def run(cmd):
        return subprocess.check_output(cmd, cwd=".", stderr=subprocess.DEVNULL).decode().strip()
    try:
        return {
            "git_commit": run(["git", "rev-parse", "HEAD"]),
            "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": str(bool(run(["git", "status", "--porcelain"]))),
        }
    except Exception:
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": "unknown"}
 
 
def load_data(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
 
 
def train_service_model(df, asof, features, num_boost_round=NUM_BOOST_ROUND):
    """one model trained on all origins.
 
    Returns (model, X_sample): the model and a few samples to build the signature
    """
    cat_levels = categorical_levels(df)
    candidate_origins, _ = build_origins(df)
 
    # toute la donnée d'entraînement, aucune origine réservée au test
    frame = make_training_frame(df, candidate_origins, asof=asof)
    frame = prep(frame, cat_levels)
 
    dtrain = lgb.Dataset(frame[features], np.log1p(frame["Sales"]),
                         categorical_feature=CATEGORICAL)
    model = lgb.train(PARAMS, dtrain, num_boost_round=num_boost_round)


    # sample for the signature with NaN rows
    # sales ly / asof at the beginning so that the schema anticipates those cases
    X = frame[features]
    has_nan = X.isna().any(axis=1)
    if has_nan.any():
        sample = pd.concat([X[has_nan].head(50), X[~has_nan].head(50)])
    else:
        sample = X.head(100)
    return model, sample
 
 
def main():
    p = argparse.ArgumentParser(description="trains and logs service model")
    p.add_argument("--data", required=True)
    p.add_argument("--variant", default="no_ly", choices=list(VARIANTS))
    p.add_argument("--register", action="store_true",
                   help="save model to Model Registry MLflow")
    args = p.parse_args()
 
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
 
    df = load_data(args.data)
    asof = build_asof_features(df)
    features = VARIANTS[args.variant]
 
    with mlflow.start_run(run_name=f"service_{args.variant}"):
        mlflow.set_tags({**git_info(), "stage": "service"})
        mlflow.log_params({
            "variant": args.variant,
            "n_features": len(features),
            "num_boost_round": NUM_BOOST_ROUND,
            **PARAMS,
        })
        mlflow.log_param("features", ", ".join(features))
 
        model, X_sample = train_service_model(df, asof, features)
 
        # signature : input/output contract that the API must follow.
        preds = np.clip(np.expm1(model.predict(X_sample)), 0, None)
        signature = infer_signature(X_sample, preds)
 
        mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_sample.head(5),
            registered_model_name=("rossmann_forecaster" if args.register else None),
        )
 
        print(f"[service_{args.variant}] modèle logged — {len(features)} features, "
              f"{model.num_trees()} trees")
        print("run_id :", mlflow.active_run().info.run_id)
 
 
if __name__ == "__main__":
    main()