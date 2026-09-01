"""
generate_dataset.py

Sweeps the CALIBRATED impact simulator across a grid of:
  - internal pressure  (kPa)
  - impact velocity    (m/s)
  - hammer mass        (kg)

and records the simulator's outputs (max contact force, permanent
deformation) for every combination. This becomes the training data for
the ML surrogate models in models/train_models.py.

SCOPE NOTE (read this before extending the ranges):
Pipe geometry (diameter, wall thickness) is held FIXED at the thesis's
actual specimen dimensions (48.1 mm OD, 2.75 mm wall) because K0, Fy0
etc. were calibrated only for that geometry (see calibrate.py). Velocity
and mass are swept because they enter the model as initial conditions,
not calibrated structural constants, so generalizing across them is
physically defensible without new calibration data.
"""

import json
import itertools
import numpy as np
import pandas as pd
from simulator.impact_model import simulate_impact, PipeImpactParams


def load_calibrated_params(path="data/calibrated_params.json"):
    with open(path) as f:
        d = json.load(f)
    return PipeImpactParams(K0=d["K0"], beta=d["beta"], Fy0=d["Fy0"], gamma=d["gamma"])


def generate_dataset(
    pressure_range=(0, 2500, 26),      # (min_kpa, max_kpa, n_points)
    velocity_range=(2.0, 10.0, 9),     # (min_mps, max_mps, n_points)
    mass_range=(2.0, 6.0, 5),          # (min_kg, max_kg, n_points)
    output_path="data/generated_dataset.csv",
):
    params = load_calibrated_params()

    pressures = np.linspace(*pressure_range[:2], pressure_range[2])
    velocities = np.linspace(*velocity_range[:2], velocity_range[2])
    masses = np.linspace(*mass_range[:2], mass_range[2])

    rows = []
    combos = list(itertools.product(pressures, velocities, masses))
    print(f"Running {len(combos)} simulations...")

    for i, (p, v, m) in enumerate(combos):
        run_params = PipeImpactParams(
            mass=m, velocity0=v,
            K0=params.K0, beta=params.beta, Fy0=params.Fy0, gamma=params.gamma,
        )
        result = simulate_impact(pressure_kpa=p, params=run_params)
        rows.append({
            "pressure_kpa": p,
            "velocity_mps": v,
            "mass_kg": m,
            "max_force_N": result["max_force"],
            "permanent_deformation_mm": result["permanent_deformation"] * 1000,
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(combos)} done")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(df.describe())
    return df


if __name__ == "__main__":
    generate_dataset()
