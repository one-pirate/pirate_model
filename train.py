import pandas as pd
import numpy as np
import sys
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report
)

sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# 1 LOAD DATA
# ============================================================

print("[INFO] Loading datasets...")

df_attacks = pd.read_csv("pirate_attacks.csv")
df_indicators = pd.read_csv("country_indicators.csv")
df_codes = pd.read_csv("country_codes.csv")

print("[INFO] Files loaded.")


# ============================================================
# 2 CLEANING (exactement ton nettoyage)
# ============================================================

print("[INFO] Cleaning data...")

# Drop irrelevant columns
cols_to_drop = [
    "time", "attack_description", "vessel_name",
    "vessel_type", "eez_country"
]
df_attacks = df_attacks.drop(columns=cols_to_drop)

# Fill missing values
df_attacks["attack_type"] = df_attacks["attack_type"].fillna("Missing")
df_attacks["vessel_status"] = df_attacks["vessel_status"].fillna("Missing")
df_attacks["location_description"] = df_attacks["location_description"].fillna("Unknown")

# Convert date to datetime and extract year
df_attacks["date"] = pd.to_datetime(df_attacks["date"])
df_attacks["year"] = df_attacks["date"].dt.year

# Remove rows with missing nearest_country
df_attacks.dropna(subset=["nearest_country"], inplace=True)

# Drop missing values in country indicators
df_indicators.dropna(inplace=True)

print("[INFO] Cleaning complete.")


# ============================================================
# 3 FEATURE ENGINEERING
# ============================================================

print("[INFO] Feature engineering...")

data_ml_flow = df_attacks.copy()

# Binary target: success = 1 if boarding/hijacked
data_ml_flow["attack_success"] = (
    data_ml_flow["attack_type"].str.lower()
    .isin(["boarding", "hijacked"])
    .astype(int)
)

# Inputs: basic geospatial features for now
X = data_ml_flow[["longitude", "latitude"]].fillna(0)
y = data_ml_flow["attack_success"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"[INFO] Training size: {len(X_train)}, Test size: {len(X_test)}")


# ============================================================
# 4 MLflow SETUP
# ============================================================

mlflow.set_experiment("piracy_detection")
print("[INFO] MLflow experiment: piracy_detection")


# ============================================================
# 5 TRAIN TWO MODELS & COMPARE
# ============================================================

def evaluate_model(name, model, X_test, y_test):
    """Returns a dictionary of all metrics."""
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
    }

    print(f"\n=== {name} Metrics ===")
    for m, v in metrics.items():
        print(f"{m}: {v:.4f}")

    return metrics


results = {}
best_model = None
best_score = -1
best_name = None


# ------------------------------------------------------------
# MODEL 1 : RANDOM FOREST
# ------------------------------------------------------------
with mlflow.start_run(run_name="RandomForest"):
    print("\n[INFO] Training RandomForest...")

    rf = RandomForestClassifier(
        n_estimators=200, random_state=42
    )
    rf.fit(X_train, y_train)

    metrics = evaluate_model("RandomForestClassifier", rf, X_test, y_test)

    mlflow.log_params({"model": "RandomForest", "n_estimators": 200})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rf, "model")

    results["RandomForest"] = metrics

    if metrics["f1"] > best_score:
        best_model = rf
        best_score = metrics["f1"]
        best_name = "RandomForest"


# ------------------------------------------------------------
# MODEL 2 : LOGISTIC REGRESSION
# ------------------------------------------------------------
with mlflow.start_run(run_name="LogisticRegression"):
    print("\n[INFO] Training LogisticRegression...")

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)

    metrics = evaluate_model("LogisticRegression", lr, X_test, y_test)

    mlflow.log_params({"model": "LogisticRegression", "max_iter": 200})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(lr, "model")

    results["LogisticRegression"] = metrics

    if metrics["f1"] > best_score:
        best_model = lr
        best_score = metrics["f1"]
        best_name = "LogisticRegression"


# ============================================================
# 6 SAVE BEST MODEL
# ============================================================

print(f"\n[INFO] Best model is: {best_name} (F1={best_score:.4f})")

mlflow.sklearn.save_model(best_model, "best_model")

print("[INFO] Best model saved to: best_model/")
print("[INFO] Training pipeline completed successfully.")
