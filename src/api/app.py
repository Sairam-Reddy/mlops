# src/api/app.py
from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import logging
import sys
import pandas as pd
from contextlib import asynccontextmanager

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------
# MLflow config
# -----------------------------
MODEL_NAME = "HeartDiseaseClassifier"
MODEL_STAGE = "Production"
MLFLOW_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_URI)

# -----------------------------
# Lifespan context for startup/shutdown
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        logger.info("Loading model from MLflow Registry...")
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model")
        raise RuntimeError("Model loading failed") from e
    yield
    logger.info("Application shutdown")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Heart Disease Prediction API", lifespan=lifespan)

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(features: dict):
    logger.info(f"Request received: {features}")

    # Validate input keys
    missing = set(FEATURE_COLUMNS) - set(features.keys())
    if missing:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}"
        )

    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prob = model.predict_proba(X)[0][1]

    return {"prediction": int(prob >= 0.5), "confidence": float(prob)}
