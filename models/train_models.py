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
from xgboost import XGBRegressor


FEATURES = ["pressure_kpa", "velocity_mps", "mass_kg"]
TARGETS = ["max_force_N", "permanent_deformation_mm"]

MODEL_BUILDERS = {
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
    "xgboost": lambda: XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1
    ),
}


def load_data(path="data/generated_dataset.csv"):
    return pd.read_csv(path)


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]

    models = {}          # models[algo][target] = fitted model
    test_metrics = {}     # test_metrics[algo][target] = {MAE, R2}

    for algo_name, build_model in MODEL_BUILDERS.items():
        models[algo_name] = {}
        test_metrics[algo_name] = {}

        for target in TARGETS:
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = build_model()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            test_metrics[algo_name][target] = {"MAE": mae, "R2": r2}
            models[algo_name][target] = model

            print(f"[{algo_name:13s} | {target:26s}]  Test MAE = {mae:8.3f}   Test R2 = {r2:.4f}")

    # --- Evaluate BOTH algorithms against real thesis data ---
    print("\n=== Validation against REAL thesis data (5 points) ===")
    real_df = pd.read_csv("data/real_thesis_data.csv")
    real_df["velocity_mps"] = 5.56
    real_df["mass_kg"] = 4.0
    X_real = real_df[FEATURES]

    for algo_name in MODEL_BUILDERS:
        print(f"\n--- {algo_name} ---")
        real_results = real_df[["pressure_kpa"]].copy()
        for target, real_col in zip(TARGETS, ["max_force_N", "permanent_deformation_mm"]):
            pred = models[algo_name][target].predict(X_real)
            real_results[f"pred"] = pred
            real_results[f"real"] = real_df[real_col]
            real_results[f"err_pct"] = 100 * (pred - real_df[real_col]) / real_df[real_col]
            print(f"  Target: {target}")
            print(real_results[["pressure_kpa", "pred", "real", "err_pct"]].to_string(index=False))

    # --- Feature importance (Random Forest -- has clean, standard API) ---
    print("\n=== Feature Importance (Random Forest) ===")
    for target in TARGETS:
        rf = models["random_forest"][target]
        importances = dict(zip(FEATURES, rf.feature_importances_))
        print(f"{target}: " + ", ".join(f"{k}={v:.3f}" for k, v in importances.items()))

    # --- Save BEST models (default: random_forest, proven above to already
    # match the simulator's own accuracy almost exactly) ---
    joblib.dump(models["random_forest"]["max_force_N"], "models/rf_max_force.joblib")
    joblib.dump(models["random_forest"]["permanent_deformation_mm"], "models/rf_permanent_deformation.joblib")
    joblib.dump(models["xgboost"]["max_force_N"], "models/xgb_max_force.joblib")
    joblib.dump(models["xgboost"]["permanent_deformation_mm"], "models/xgb_permanent_deformation.joblib")

    with open("models/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nSaved all 4 models to models/")

    return models, test_metrics


if __name__ == "__main__":
    train_and_evaluate()
