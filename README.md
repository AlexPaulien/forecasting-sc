# Demand Forecasting — Supply Chain

End-to-end demand forecasting project on retail sales data (Rossmann dataset).

## Stack
Python · Prophet · LightGBM · NeuralForecast (TFT) · MLflow · FastAPI · Docker · GCP Cloud Run

## Structure
- `notebooks/` — EDA, feature engineering, model benchmark
- `src/` — reusable modules (features, models)
- `api/` — FastAPI serving (coming)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
Download from [Kaggle Rossmann competition](https://www.kaggle.com/competitions/rossmann-store-sales/data) and place in `data/raw/`.


## Training & experiment tracking (MLflow)

### Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Prepare the data

The pipeline expects a single file (`train` + `store` already merged, filtered on
`Open == 1 & Sales > 0`). From the preparation notebook:

```python
df["StateHoliday"] = df["StateHoliday"].astype(str)   # mixed types lead to issues with Parquet
df.to_parquet("data/rossmann.parquet")
```

### Run a tracked backtest

From the project root (`forecasting-sc/`):

```bash
python -m src.train --data data/rossmann.parquet --variant no_ly   # reference model
python -m src.train --data data/rossmann.parquet --variant full    # variant with sales_ly
python -m src.train --data data/rossmann.parquet --variant all      # both, back to back
```

Each run creates an MLflow run (params, WMAPE / bias / FVA per horizon bucket, feature
importance as an artifact). Tracking is stored locally in `mlflow.db`.

### Open the MLflow UI

**From a separate terminal, at the project root** (where `mlflow.db` lives):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

To compare variants: select the `rossmann_04_production` experiment, tick the runs you
want, and click **Compare**. The `wmape` metric is logged by `step` (= horizon week,
D+1→D+42), which overlays the degradation curves across runs.

> **Two things that cost time:**
> - The UI must use the **same URI** as training (`sqlite:///mlflow.db`), otherwise it
>   opens an empty store.
> - The path is **relative**: launching the UI from any directory other than the one
>   containing `mlflow.db` will show no runs.


## Serving — FastAPI prediction API

The service model (see *Training & experiment tracking*) is exposed through a
FastAPI endpoint. Feature reconstruction at inference mirrors the training
pipeline exactly (same `snapshot_at`, same categorical encoding), so there is no
train/serve skew.

### Run the API locally

From the project root:

```bash
pip install fastapi uvicorn
uvicorn src.api:app --reload
```

The model is loaded **once at startup** from the MLflow Registry
(`models:/rossmann_forecaster/latest`), not per request. Startup takes a few
seconds while as-of features are computed over the full history.

Auto-generated interactive docs: http://localhost:8000/docs
Predicted vs Actuals dataviz: http://localhost:8000/

### Endpoints

`GET /health` — liveness check, returns the loaded model URI and feature count.

`POST /predict` — day-by-day forecast from a past origin, compared against the
actuals (interactive backtest). All body fields are optional:

| Field | Type | Default | Notes |
|---|---|---|---|
| `store` | int | all stores | rejected (400) if unknown |
| `horizon` | int | 42 | bounded 1..42 (training domain) |
| `origin` | str (YYYY-MM-DD) | latest usable date | must leave 42 days of actuals for comparison |

Example request:

```json
{ "store": 5, "horizon": 3, "origin": "2014-11-01" }
```

Response — one row per (store, open day), with predicted `sales` and observed
`actual` side by side:

```json
{
  "origin": "2014-11-01",
  "horizon": 3,
  "n": 2,
  "predictions": [
    { "Store": 5, "Date": "2014-11-03", "h": 2, "sales": 7857.5, "actual": 8274 },
    { "Store": 5, "Date": "2014-11-04", "h": 3, "sales": 6372.5, "actual": 6836 }
  ]
}
```

Only 2 days retrieved because "2014-11-02" was a closed day.

### Design notes

- **Backtest demo, not live forecasting.** Origins are restricted to the past so
  that actuals exist and predictions can be verified against them on the same
  chart. In a production setting the target-date calendar (promotions, holidays)
  would come from a planned business calendar rather than historical data.
- Closed days (e.g. Sundays) are absent from the output, so `n` can be lower than
  `horizon` — expected, not a bug.
- Out-of-range parameters are rejected with HTTP 400/422 rather than silently
  extrapolating beyond the model's trained horizon.
