import pandas as pd
import numpy as np
import sys
import mlflow
import mlflow.sklearn
import os
import shutil

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
mlflow.set_tracking_uri("file:./mlruns")  # Tracking to stock runs, models...
mlflow.set_experiment("piracy_detection")
print("[INFO] MLflow experiment: piracy_detection")


# ============================================================
# 5 TRAIN MODELS WITH HYPERPARAMETER TUNING
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
# MODEL 1 : RANDOM FOREST + GRID SEARCH
# ------------------------------------------------------------
with mlflow.start_run(run_name="RandomForest_Tuning"):
    print("\n[INFO] Tuning RandomForest...")

    # Grille d’hyperparamètres à tester
    rf_params = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    # GridSearchCV = teste automatiquement toutes les combinaisons
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=10),
        rf_params,
        scoring="f1",
        cv=3,
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    # Meilleur modèle trouvé
    rf_best = rf_grid.best_estimator_

    print("[INFO] Best RF Params:", rf_grid.best_params_)

    metrics = evaluate_model("RandomForestClassifier", rf_best, X_test, y_test)

    mlflow.log_params(rf_grid.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rf_best, "model")

    results["RandomForest"] = metrics

    if metrics["f1"] > best_score:
        best_model = rf_best
        best_score = metrics["f1"]
        best_name = "RandomForest"


# ------------------------------------------------------------
# MODEL 2 : LOGISTIC REGRESSION + GRID SEARCH
# ------------------------------------------------------------
with mlflow.start_run(run_name="LogisticRegression_Tuning"):
    print("\n[INFO] Tuning LogisticRegression...")

    lr_params = {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [50, 100, 200, 500]
    }

    lr_grid = GridSearchCV(
        LogisticRegression(),
        lr_params,
        scoring="f1",
        cv=3,
        n_jobs=-1
    )

    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_

    print("[INFO] Best LR Params:", lr_grid.best_params_)

    metrics = evaluate_model("LogisticRegression", lr_best, X_test, y_test)

    mlflow.log_params(lr_grid.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(lr_best, "model")

    results["LogisticRegression"] = metrics

    if metrics["f1"] > best_score:
        best_model = lr_best
        best_score = metrics["f1"]
        best_name = "LogisticRegression"


# ------------------------------------------------------------
# MODEL 3 : XGBOOST + GRID SEARCH
# ------------------------------------------------------------
with mlflow.start_run(run_name="XGBoost_Tuning"):
    print("\n[INFO] Tuning XGBoost...")

    xgb_params = {
        "n_estimators": [200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8]
    }

    xgb_grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        xgb_params,
        scoring="f1",
        cv=3,
        n_jobs=-1
    )

    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_

    print("[INFO] Best XGB Params:", xgb_grid.best_params_)

    metrics = evaluate_model("XGBoost", xgb_best, X_test, y_test)

    mlflow.log_params(xgb_grid.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(xgb_best, "model")

    results["XGBoost"] = metrics

    if metrics["f1"] > best_score:
        best_model = xgb_best
        best_score = metrics["f1"]
        best_name = "XGBoost"


# ------------------------------------------------------------
# MODEL 4 : LightGBM + GRID SEARCH
# ------------------------------------------------------------
with mlflow.start_run(run_name="LightGBM_Tuning"):
    print("\n[INFO] Tuning LightGBM...")

    lgbm_params = {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 50],
        "subsample": [0.7, 0.8]
    }

    lgbm_grid = GridSearchCV(
        LGBMClassifier(random_state=42),
        lgbm_params,
        scoring="f1",
        cv=3,
        n_jobs=-1
    )

    lgbm_grid.fit(X_train, y_train)
    lgbm_best = lgbm_grid.best_estimator_

    print("[INFO] Best LGBM Params:", lgbm_grid.best_params_)

    metrics = evaluate_model("LightGBM", lgbm_best, X_test, y_test)

    mlflow.log_params(lgbm_grid.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(lgbm_best, "model")

    results["LightGBM"] = metrics

    if metrics["f1"] > best_score:
        best_model = lgbm_best
        best_score = metrics["f1"]
        best_name = "LightGBM"


# ============================================================
# 6 SAVE BEST MODEL
# ============================================================

print(f"\n[INFO] Best model is: {best_name} (F1={best_score:.4f})")

if os.path.exists("best_model"):
    print("[INFO] Removing old best_model/ ...")
    shutil.rmtree("best_model")

mlflow.sklearn.save_model(best_model, "best_model")

print("[INFO] Best model saved to: best_model/")
print("[INFO] Training pipeline completed successfully.")
