"""
00_exploratory_regression.py

PRELIMINARY / MOTIVATING ANALYSIS (predates the surrogate model in this repo)

This is the original exploratory step: simple linear vs. quadratic
regression fit DIRECTLY against the 5 real thesis data points (no
physics simulator, no calibration, no ML training) -- just asking
"does pressure alone explain the trend, and is the relationship
linear or curved?"

FINDING: quadratic regression fits peak contact force much better than
linear (R2 = 0.998 vs 0.932), capturing a diminishing-return / saturating
relationship as pressure increases. This independently confirms, via a
completely different method, the same high-pressure saturation behavior
later diagnosed in simulator/calibrate.py (Lesson 4) -- two different
approaches converging on the same real physical effect is a good sign,
not a coincidence.

This analysis's own conclusion called for exactly what the rest of this
repo builds: "a genuinely predictive surrogate model... a multi-feature
regression or tree-based model (e.g., Random Forest)... running
additional simulations across pressure, velocity, and mass."
See simulator/, models/ for that follow-through.

LIMITATIONS (as originally noted): only 5 points, single-variable
(pressure only, velocity/mass held fixed), R2 describes fit quality on
these 5 points only -- not generalization to unseen conditions. No
train/test split, since 5 points cannot be meaningfully split.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error


def load_data():
    # Table 4.2 / 4.3 from the thesis
    return pd.DataFrame({
        "pressure_kpa": [689.48, 1034.21, 1378.95, 1723.69, 2068.43],
        "peak_force_N": [3527.20, 3776.05, 4004.31, 4138.00, 4174.00],
        "permanent_deformation_m": [0.01518, 0.01238, 0.01098, 0.00976, 0.00708],
        "plastic_area_mm2": [120.18, 134.01, 148.48, 213.82, 380.13],
    })


def fit_and_compare(df, target_col):
    X = df[["pressure_kpa"]].values
    y = df[target_col].values

    linear = LinearRegression().fit(X, y)
    y_pred_lin = linear.predict(X)

    quad = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X, y)
    y_pred_quad = quad.predict(X)

    return {
        "linear_R2": r2_score(y, y_pred_lin),
        "linear_MAE": mean_absolute_error(y, y_pred_lin),
        "quad_R2": r2_score(y, y_pred_quad),
        "quad_MAE": mean_absolute_error(y, y_pred_quad),
    }


if __name__ == "__main__":
    df = load_data()
    for target in ["peak_force_N", "permanent_deformation_m", "plastic_area_mm2"]:
        res = fit_and_compare(df, target)
        print(f"{target}:")
        print(f"  Linear:    R2={res['linear_R2']:.3f}  MAE={res['linear_MAE']:.5f}")
        print(f"  Quadratic: R2={res['quad_R2']:.3f}  MAE={res['quad_MAE']:.5f}")
        print()
