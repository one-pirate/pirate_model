import mlflow
import mlflow.sklearn
import numpy as np
import shap

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

# SHAP explainer
print("[INFO] Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
print("[INFO] SHAP explainer ready.")

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
    try:
        test_df = pd.DataFrame([{"longitude": 115.25, "latitude": 19.67}])
        _ = model.predict(test_df)

        return {
            "status": "ok",
            "api": "running",
            "model_loaded": True,
            "model_path": MODEL_PATH
        }

    except Exception as e:
        return {
            "status": "error",
            "api": "running",
            "model_loaded": False,
            "error": str(e)
        }


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

@app.post("/explain")
def explain(point: AttackInput):
    X = np.array([[point.latitude, point.longitude]])

    shap_values = explainer.shap_values(X)

    return {
        "input": {
            "latitude": point.latitude,
            "longitude": point.longitude
        },
        # Valeur moyenne de la prédiction du modèle sans features
        # (baseline utilisée pour expliquer l'écart de la prédiction)
        "expected_value": (
            explainer.expected_value.tolist()
            if hasattr(explainer.expected_value, "tolist")
            else explainer.expected_value
        ),
        # Valeurs SHAP expliquant l'impact de chaque feature sur la prédiction finale
        "shap_values": (
            shap_values.tolist()
            if hasattr(shap_values, "tolist")
            else shap_values
        )
    }