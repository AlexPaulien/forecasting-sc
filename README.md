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
