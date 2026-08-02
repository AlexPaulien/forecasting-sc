"""Prediction API — wrap src.predict.predict into a FastAPI endpoint.

To be launched from project root :
    uvicorn src.api:app --reload
testing UI : http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc

from .serve_model import load_data
from .features import build_asof_features
from .data import categorical_levels
from .predict import predict, PredictError
from .config import VARIANTS, HORIZON

DATA_PATH = "data/rossmann.parquet"
MODEL_URI = "models:/rossmann_forecaster/latest"

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

    out = out.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return {
        "origin": str(origin.date()),
        "horizon": horizon,
        "n": len(out),
        "predictions": out.to_dict(orient="records"),
    }