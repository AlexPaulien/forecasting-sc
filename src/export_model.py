"""
Exports service model from MLflow registry to standalone folder,
to be used with docker container, no tracking nor need for SQLite
    python export_model.py # exports models:/rossmann_forecaster/latest
    python export_model.py --uri runs:/<id>/model
./model folder has everything mlflow.pyfunc.load_model("model") needs
without mlflow backend
"""
import argparse
import shutil
from pathlib import Path
 
import mlflow
from mlflow.artifacts import download_artifacts
 
DEFAULT_URI = "models:/rossmann_forecaster/latest"
OUT_DIR = "model"
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default=DEFAULT_URI)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = p.parse_args()
 
    mlflow.set_tracking_uri(args.tracking_uri)
 
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
 
    local = download_artifacts(artifact_uri=args.uri)
    shutil.copytree(local, out)
    print(f"model exported to ./{args.out}/ depuis {args.uri}")
 
 
if __name__ == "__main__":
    main()