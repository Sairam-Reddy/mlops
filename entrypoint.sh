#!/bin/bash
set -e  # Exit immediately if a command fails

echo "Starting training..."
python -m src.models.train

echo "Selecting and registering best model..."
python -m src.models.select_best_model

echo "Starting MLflow UI..."
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root /app/mlruns \
  --host 0.0.0.0 --port 5000 &

echo "Starting FastAPI..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --log-level info --workers 1
