FROM python:3.11-slim

# libgomp1 : LightGBM system dependancy
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependancies (cache Docker : only reinstall if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi "uvicorn[standard]"

# code, data, exported model, front
COPY src/ ./src/
COPY static/ ./static/
COPY data/ ./data/
COPY model/ ./model/

# model loaded frol local folder, not from registry
ENV MODEL_PATH=/app/model
ENV DATA_PATH=/app/data/rossmann.parquet

# Cloud Run forward $PORT ; default 8000 in local env
ENV PORT=8000
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT}
