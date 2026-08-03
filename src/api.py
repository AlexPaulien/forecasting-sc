"""Prediction API — wrap src.predict.predict into a FastAPI endpoint.

To be launched from project root :
    uvicorn src.api:app --reload
testing UI : http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import mlflow.pyfunc

from .serve_model import load_data
from .features import build_asof_features
from .data import categorical_levels
from .predict import predict, aggregate_national, PredictError
from .config import VARIANTS, HORIZON

import os

DATA_PATH = os.environ.get("DATA_PATH", "data/rossmann.parquet")
# container: MODEL_PATH points to the exportation folder
# local environement: MODEL_PATH points to MLflow model registry
MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL_URI = MODEL_PATH if MODEL_PATH else "models:/rossmann_forecaster/latest"

app = FastAPI(title="Rossmann forecaster", version="1.0")

# --- launched once at start (never via request) ---
df = load_data(DATA_PATH)
asof = build_asof_features(df)
cat_maps = categorical_levels(df)
model = mlflow.pyfunc.load_model(MODEL_URI)
FEATURES = VARIANTS["no_ly"]


class PredictRequest(BaseModel):
    store: int | None = Field(None, description="single store ; all")
    horizon: int | None = Field(None, ge=1, le=HORIZON, description=f"1..{HORIZON}")
    origin: str | None = Field(None, description="origin date YYYY-MM-DD ; default = most recent")
    aggregate: bool = Field(False, description="sum of National sales by date instead of store level detail")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI, "n_features": len(FEATURES)}


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        out, origin, horizon = predict(
            model, df, df, asof, cat_maps, FEATURES,
            origin=req.origin, horizon=req.horizon, store=req.store,
        )
    except PredictError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.aggregate:
        out = aggregate_national(out)

    out = out.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return {
        "origin": str(origin.date()),
        "horizon": horizon,
        "aggregated": req.aggregate,
        "n": len(out),
        "predictions": out.to_dict(orient="records"),
    }