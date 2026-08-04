# Demand Forecasting — Supply Chain

End-to-end demand forecasting project on retail sales data (Rossmann dataset).

## About

End-to-end demand forecasting for 1115 stores using the Rossmann dataset: 42-day forecasting horizon, predictions served through a live API with live backtest hosted on GCP Cloud Run.

GCP bactest demo: https://rossmann-forecaster-541488693264.europe-west1.run.app
<img src="img/Screenshot 2026-08-03 at 15.43.23.png" alt="Backtest screenshot" width="700">

## Problem & Methodology

### Problem statement

We want to build a forecasting model to predict up to the sales amount for up to 42 days ahead. The data is comprised of 1115 Rossmann stores with dayly sales recorded from Jan 1st 2013 to July 31st 2015.

### Exploratory Data Analysis

EDA (see notebook 1) shows that the distribution of sales is skewed to that right whereas the log distribution of sales is Gaussian. We will therefore train our models on log(sales).

We also demonstrated that December was a very strong month sales-wise and that there was also a weekly seasonality with Mondays and Sundays being stronger.

Finally, we saw that promotions tended to lift the sales by about 38% on average.

### Methodology

We first build a naive baseline (using averages) against which we put to test several algorithms:
- autoARIMA
- autoETS
- Prophet
- LightGBM
This model benchmark was done on a single reference store and LightGBM ended up outperforming the other ones although Prophet did well too. Here are the results from that single store benchmark:
<img src="img/21a178cb-0a5e-4908-8131-610c948245e9.png" alt="Model benchmark" width="700">


## Architecture



## Results



## Technicals

### Stack
Python · Prophet · LightGBM · MLflow · FastAPI · Docker · GCP Cloud Run

### Structure
- `notebooks/` — EDA, feature engineering, model benchmark
- `src/` — reusable modules (features, models)
- `api/` — FastAPI serving (coming)

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data
Download from [Kaggle Rossmann competition](https://www.kaggle.com/competitions/rossmann-store-sales/data) and place in `data/raw/`.


### Training & experiment tracking (MLflow)

#### Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

#### Prepare the data

The pipeline expects a single file (`train` + `store` already merged, filtered on
`Open == 1 & Sales > 0`). From the preparation notebook:

```python
df["StateHoliday"] = df["StateHoliday"].astype(str)   # mixed types lead to issues with Parquet
df.to_parquet("data/rossmann.parquet")
```

#### Run a tracked backtest

From the project root (`forecasting-sc/`):

```bash
python -m src.train --data data/rossmann.parquet --variant no_ly   # reference model
python -m src.train --data data/rossmann.parquet --variant full    # variant with sales_ly
python -m src.train --data data/rossmann.parquet --variant all      # both, back to back
```

Each run creates an MLflow run (params, WMAPE / bias / FVA per horizon bucket, feature
importance as an artifact). Tracking is stored locally in `mlflow.db`.

#### Open the MLflow UI

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


### Serving — FastAPI prediction API

The service model (see *Training & experiment tracking*) is exposed through a
FastAPI endpoint. Feature reconstruction at inference mirrors the training
pipeline exactly (same `snapshot_at`, same categorical encoding), so there is no
train/serve skew.

#### Run the API locally

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

#### Endpoints

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

#### Design notes

- **Backtest demo, not live forecasting.** Origins are restricted to the past so
  that actuals exist and predictions can be verified against them on the same
  chart. In a production setting the target-date calendar (promotions, holidays)
  would come from a planned business calendar rather than historical data.
- Closed days (e.g. Sundays) are absent from the output, so `n` can be lower than
  `horizon` — expected, not a bug.
- Out-of-range parameters are rejected with HTTP 400/422 rather than silently
  extrapolating beyond the model's trained horizon.


### Containerized deployment (Docker)

The API and its front-end are packaged into a single self-contained image. The
model is loaded from a local exported folder (no MLflow backend / SQLite needed
at runtime), so the container is fully standalone.

#### Export the service model

Ahead of the build, export the registered model from MLflow into a standalone
`model/` folder:

```bash
python src/export_model.py
```

This downloads `models:/rossmann_forecaster/latest` into `./model/`, which the
container loads directly by path. Re-run it whenever a new model version should
be shipped. The folder is gitignored (regenerable from the Registry).

#### Build the image

```bash
docker build -t rossmann-forecaster .
```

Notes:
- `libgomp1` is installed for LightGBM (missing it fails at runtime, not build).
- `requirements.txt` is copied before the source so dependency layers stay cached
  across code changes.
- `.dockerignore` keeps the build context small (excludes `venv/`, `mlflow.db`,
  `mlruns/`, notebooks).

#### Run locally

```bash
docker run -p 8000:8000 rossmann-forecaster
```

Then open http://localhost:8000/ — the front-end and API behave exactly as under
`uvicorn`, this time served from the container with the model loaded from `./model`.

#### Configuration

The image reads two environment variables (with sensible defaults):

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/app/model` | standalone model folder; if empty, falls back to the MLflow Registry |
| `DATA_PATH` | `/app/data/rossmann.parquet` | dataset loaded at startup |
| `PORT` | `8000` | server port — Cloud Run injects its own `$PORT` |

The same image runs unchanged locally and on a cloud host: `PORT` is read from the
environment, and the server binds `0.0.0.0` so it is reachable from outside the
container.


### Cloud deployment (Google Cloud Run)

The container is deployed to Cloud Run as a public, serverless service. Cloud Run
builds the image from the `Dockerfile` (via Cloud Build), pushes it to Artifact
Registry, and serves it — all from a single command.

**Live demo:** https://rossmann-forecaster-541488693264.europe-west1.run.app

#### One-time GCP setup

```bash
gcloud auth login
gcloud projects create <PROJECT_ID> --name="Rossmann forecaster"
gcloud config set project <PROJECT_ID>
gcloud billing projects link <PROJECT_ID> --billing-account=<BILLING_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Set a budget alert (Billing > Budgets & alerts) before deploying just in case although GCP free tier should be enough to run this demo.

#### Deploy

Export the standalone model first (if not already done), then deploy from source:

```bash
python src/export_model.py

gcloud run deploy rossmann-forecaster \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --port 8080
```

Flag rationale:

| Flag | Why |
|---|---|
| `--source .` | builds from the local Dockerfile via Cloud Build — no manual `docker push` |
| `--allow-unauthenticated` | public demo, reachable without a GCP login |
| `--memory 2Gi` | the app loads the dataset and rebuilds as-of features at startup; 512 MB (default) is not enough |
| `--timeout 300` | leaves room for a slow first boot (model load + feature build) |
| `--region europe-west1` | close to target users; Cloud Run's free tier applies in every region |

#### `.gcloudignore` trap

`--source` uploads the project directory to Cloud Build. With no `.gcloudignore`,
gcloud falls back to `.gitignore` — and since `model/` is gitignored (regenerable
from the Registry), it would be **excluded from the upload**, and the Dockerfile's
`COPY model/` would fail with *"not found in build context"*.

A standalone `.gcloudignore` breaks that inheritance so `model/` and `data/` are
uploaded even though they stay gitignored. Verify what will be sent:

```bash
gcloud meta list-files-for-upload | grep model
```

The `model/` files must appear in the output.

#### Notes

- **Cold starts.** With no traffic, Cloud Run scales to zero. The next request
  triggers a full boot (model load + feature build), adding a few seconds of
  latency. Acceptable for a demo; precomputing the as-of features into a Parquet
  file would shorten it.
- The same image runs unchanged locally (`docker run`) and on Cloud Run — `PORT`
  is read from the environment and the server binds `0.0.0.0`.

