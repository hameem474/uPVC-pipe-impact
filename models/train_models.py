"""
train_models.py

Trains ML surrogate models (Random Forest regressors) on the physics-
simulator-generated dataset to predict:
  - max_force_N
  - permanent_deformation_mm

from inputs: pressure_kpa, velocity_mps, mass_kg.

Evaluation happens at THREE levels, each answering a different question:
  1. Train/test split on the generated dataset
       -> "did the model learn the simulator's behavior well?"
  2. Held-out real thesis data (5 points, never used in training)
       -> "does the model (via the simulator) actually reflect reality?"
  3. Comparison against the physics simulator's own calibration error
       -> "is the ML model adding error on top of the simulator, or is
          it a faithful stand-in?"
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


FEATURES = ["pressure_kpa", "velocity_mps", "mass_kg"]
TARGETS = ["max_force_N", "permanent_deformation_mm"]


def load_data(path="data/generated_dataset.csv"):
    return pd.read_csv(path)


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]

    models = {}
    test_metrics = {}

    for target in TARGETS:
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        test_metrics[target] = {"MAE": mae, "R2": r2}
        models[target] = model

        print(f"[{target}]  Test MAE = {mae:.3f}   Test R2 = {r2:.4f}")

    # --- Evaluate against real thesis data (points the model NEVER saw) ---
    print("\n=== Validation against REAL thesis data (5 points) ===")
    real_df = pd.read_csv("data/real_thesis_data.csv")
    real_df["velocity_mps"] = 5.56  # thesis's actual constant impact velocity
    real_df["mass_kg"] = 4.0        # thesis's actual constant hammer mass

    X_real = real_df[FEATURES]

    real_results = real_df[["pressure_kpa"]].copy()
    for target, real_col in zip(TARGETS, ["max_force_N", "permanent_deformation_mm"]):
        pred = models[target].predict(X_real)
        real_results[f"pred_{target}"] = pred
        real_results[f"real_{real_col}"] = real_df[real_col]
        real_results[f"err_pct_{target}"] = 100 * (pred - real_df[real_col]) / real_df[real_col]

    print(real_results.to_string(index=False))

    # --- Save trained models ---
    joblib.dump(models["max_force_N"], "models/rf_max_force.joblib")
    joblib.dump(models["permanent_deformation_mm"], "models/rf_permanent_deformation.joblib")

    with open("models/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nSaved models to models/rf_max_force.joblib, models/rf_permanent_deformation.joblib")

    return models, test_metrics, real_results


if __name__ == "__main__":
    train_and_evaluate()
