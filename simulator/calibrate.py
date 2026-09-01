"""
calibrate.py

Fits the impact simulator's free parameters (K0, beta, Fy0, gamma) to the
real thesis data (data/real_thesis_data.csv) using least-squares
optimization.

WHY THIS MATTERS: this is the step that turns our reduced-order physics
model from "a plausible guess" into "a model grounded in real measurements
from an actual lab test." Every claim in the README about this simulator
being "validated against real data" traces back to this script.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from simulator.impact_model import simulate_impact, PipeImpactParams


def load_real_data(path="data/real_thesis_data.csv"):
    return pd.read_csv(path)


def residuals(x, pressures, real_force, real_defm):
    """
    x = [K0, beta, Fy0, gamma]  (the 4 unknowns we're solving for)

    Returns a flat array of NORMALIZED residuals (predicted - real) for
    both force and deformation, for every pressure point. Normalizing is
    essential here: force is ~O(3000-4000) and deformation is ~O(0.01),
    so without normalizing, the optimizer would basically ignore
    deformation error entirely (it looks tiny in absolute terms).
    """
    K0, beta, Fy0, gamma = x
    params = PipeImpactParams(K0=K0, beta=beta, Fy0=Fy0, gamma=gamma)

    force_res = []
    defm_res = []
    for p, rf, rd in zip(pressures, real_force, real_defm):
        result = simulate_impact(pressure_kpa=p, params=params)
        pred_force = result["max_force"]
        pred_defm_mm = result["permanent_deformation"] * 1000

        # normalize by the real value's scale so both error types are
        # comparable (this makes it a "percent-ish" error)
        force_res.append((pred_force - rf) / rf)
        defm_res.append((pred_defm_mm - rd) / rd)

    return np.array(force_res + defm_res)


def calibrate():
    df = load_real_data()
    pressures = df["pressure_kpa"].values
    real_force = df["max_force_N"].values
    real_defm = df["permanent_deformation_mm"].values

    # initial guess = our current rough hand-picked values
    x0 = [2.0e6, 3.0e-4, 3200.0, 1.0e-4]

    # bounds keep the optimizer in physically sensible territory
    # (all four constants must be positive)
    lower = [1e4, 0.0, 500.0, 0.0]
    upper = [1e8, 1e-2, 2e4, 1e-2]

    result = least_squares(
        residuals, x0,
        args=(pressures, real_force, real_defm),
        bounds=(lower, upper),
        method="trf",
    )

    K0, beta, Fy0, gamma = result.x
    print("=== Calibration complete ===")
    print(f"K0    = {K0:.4e}  N/m")
    print(f"beta  = {beta:.4e}  1/kPa")
    print(f"Fy0   = {Fy0:.4e}  N")
    print(f"gamma = {gamma:.4e}  1/kPa")
    print(f"Final cost (sum of squared normalized residuals): {result.cost:.5f}")
    print()

    calibrated_params = PipeImpactParams(K0=K0, beta=beta, Fy0=Fy0, gamma=gamma)

    print(f"{'Pressure':>10} {'Pred Force':>11} {'Real Force':>11} {'Err%':>7}   "
          f"{'Pred Defm':>10} {'Real Defm':>10} {'Err%':>7}")
    for p, rf, rd in zip(pressures, real_force, real_defm):
        r = simulate_impact(pressure_kpa=p, params=calibrated_params)
        pf = r["max_force"]
        pd_mm = r["permanent_deformation"] * 1000
        ferr = 100 * (pf - rf) / rf
        derr = 100 * (pd_mm - rd) / rd
        print(f"{p:>10.2f} {pf:>11.1f} {rf:>11.1f} {ferr:>6.1f}%   "
              f"{pd_mm:>10.2f} {rd:>10.2f} {derr:>6.1f}%")

    # Save calibrated parameters so other scripts (dataset generation, app)
    # don't need to re-run this optimization every time.
    import json
    calib_dict = {"K0": K0, "beta": beta, "Fy0": Fy0, "gamma": gamma}
    with open("data/calibrated_params.json", "w") as f:
        json.dump(calib_dict, f, indent=2)
    print(f"\nSaved calibrated parameters to data/calibrated_params.json")

    return calibrated_params


if __name__ == "__main__":
    calibrate()
