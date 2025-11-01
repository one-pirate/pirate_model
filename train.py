# Step 1: Upload datasets

# Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    accuracy_score, 
    f1_score, 
    recall_score, 
    precision_score
)
from sklearn.linear_model import LogisticRegression

import sys
sys.stdout.reconfigure(encoding='utf-8') 

import mlflow
import mlflow.sklearn


# Upload datasets
df_attacks = pd.read_csv("pirate_attacks.csv")
df_indicators = pd.read_csv("country_indicators.csv")
df_codes = pd.read_csv("country_codes.csv")

# Step 2: Clean datasets

# Drop irrelevant columns
df_attacks = df_attacks.drop('time', axis=1)
df_attacks = df_attacks.drop('attack_description', axis=1)
df_attacks = df_attacks.drop('vessel_name', axis=1)
df_attacks = df_attacks.drop('vessel_type', axis=1)
df_attacks = df_attacks.drop('eez_country', axis=1)

# Fill missing values
df_attacks["attack_type"] = df_attacks["attack_type"].fillna("Missing")
df_attacks["vessel_status"] = df_attacks["vessel_status"].fillna("Missing")
df_attacks["location_description"] = df_attacks["location_description"].fillna("Unknown")

# Convert date column to datetime and extract year
df_attacks["date"] = pd.to_datetime(df_attacks["date"])
df_attacks["year"] = df_attacks.date.dt.year

# Remove rows with missing nearest_country values (ignore eez_country)
df_attacks.dropna(subset=['nearest_country'], inplace=True)

# Drop missing values in country indicators dataset
df_indicators.dropna(inplace=True)


# Copy dataset for MLflow processing
data_ml_flow = df_attacks.copy()

# Target column: 1 if attack was successful (boarding or hijacked), else 0
data_ml_flow["attack_success"] = data_ml_flow["attack_type"].str.lower().isin(["boarding", "hijacked"]).astype(int)

# Input variables: longitude and latitude
X = data_ml_flow[["longitude", "latitude"]].fillna(0)
y = data_ml_flow["attack_success"]

# Split data into training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model and log with MLflow
with mlflow.start_run(run_name="Test_RandomForest_AttackPrediction_perfect"):

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {acc:.2f}")

    # Generate and save confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=["Failure", "Success"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Attack Prediction")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Log results to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("confusion_matrix.png")

print("\nDone.")


# Step 4: Quick test of the model with new coordinates
print("\nTesting the model with new coordinates:")
longitude = float(input("→ Longitude: "))
latitude = float(input("→ Latitude: "))

# Create a DataFrame for prediction
new_attack = pd.DataFrame([[longitude, latitude]], columns=["longitude", "latitude"])

# Predict result
prediction = model.predict(new_attack)[0]
probability = model.predict_proba(new_attack)[0]

print("\nPrediction result:")
print(f"Prediction (0 = Failure, 1 = Success): {prediction}")
print(f"Probabilities [Failure, Success]: {probability}")


# Step 5: Check MLflow experiments
client = mlflow.MlflowClient(tracking_uri="http://localhost:8080")
all_experiments = client.search_experiments()
print(all_experiments)
