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
df["StateHoliday"] = df["StateHoliday"].astype(str)   # mixed types leads to issues with Parquet
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
