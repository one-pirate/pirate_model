import mlflow
import mlflow.sklearn
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd


# ============================================================
# 1 FASTAPI INIT
# ============================================================

app = FastAPI(
    title="Piracy Risk Prediction API",
    description="API pour prédire la probabilité de succès d'une attaque pirate.",
    version="1.0.0"
)


# ============================================================
# 2 LOAD BEST MODEL
# ============================================================

MODEL_PATH = "best_model"

print("[INFO] Loading model from best_model/ ...")
model = mlflow.sklearn.load_model(MODEL_PATH)
print("[INFO] Model loaded.")


# ============================================================
# 3 INPUT SCHEMA
# ============================================================

class AttackInput(BaseModel):
    longitude: float
    latitude: float


# ============================================================
# 4 HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


# ============================================================
# 5 PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(input_data: AttackInput):

    # Convert into DataFrame for sklearn
    X = pd.DataFrame([{
        "longitude": input_data.longitude,
        "latitude": input_data.latitude
    }])

    proba = model.predict_proba(X)[0][1]   # probability success
    pred = int(model.predict(X)[0])

    return {
        "prediction": pred,
        "probability_success": float(proba)
    }

# ============================================================
# 6 ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Piracy Prediction API is running.",
        "endpoints": ["/predict", "/health"]
    }
