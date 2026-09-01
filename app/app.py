"""
app.py

Interactive Streamlit demo for the pipe impact ML surrogate model.

Run with:
    streamlit run app/app.py

Lets the user set pressure, impact velocity, and hammer mass with sliders,
and see live predictions of max contact force and permanent deformation
from the trained Random Forest models -- with the user's prediction
plotted alongside the 5 real thesis data points for context.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


@st.cache_resource
def load_models():
    force_model = joblib.load("models/rf_max_force.joblib")
    defm_model = joblib.load("models/rf_permanent_deformation.joblib")
    return force_model, defm_model


@st.cache_data
def load_real_data():
    return pd.read_csv("data/real_thesis_data.csv")


def main():
    st.set_page_config(page_title="Pipe Impact ML Surrogate", layout="centered")

    st.title("Pressurized uPVC Pipe — Impact Response Predictor")
    st.caption(
        "A physics-informed ML surrogate model, trained on a calibrated "
        "impact simulator and validated against real experimental/FEA data "
        "from an undergraduate thesis (Ahmed & Bashar, RUET, 2023)."
    )

    force_model, defm_model = load_models()
    real_df = load_real_data()

    st.subheader("Set impact parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        pressure = st.slider("Internal Pressure (kPa)", 0, 2500, 1000, step=10)
    with col2:
        velocity = st.slider("Impact Velocity (m/s)", 2.0, 10.0, 5.56, step=0.01)
    with col3:
        mass = st.slider("Hammer Mass (kg)", 2.0, 6.0, 4.0, step=0.1)

    X = pd.DataFrame([{
        "pressure_kpa": pressure,
        "velocity_mps": velocity,
        "mass_kg": mass,
    }])

    pred_force = force_model.predict(X)[0]
    pred_defm = defm_model.predict(X)[0]

    st.subheader("Predicted response")
    c1, c2 = st.columns(2)
    c1.metric("Max Contact Force", f"{pred_force:,.0f} N")
    c2.metric("Permanent Deformation", f"{pred_defm:.2f} mm")

    st.divider()
    st.subheader("Where does this sit relative to the real thesis data?")
    st.caption(
        "Grey points are the 5 real experimental/FEA results from the "
        "thesis (at 5.56 m/s, 4 kg -- the only conditions actually tested "
        "in the lab). The red star is your current prediction."
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(real_df["pressure_kpa"], real_df["max_force_N"],
                 "o-", color="gray", label="Real thesis data (5.56 m/s, 4 kg)")
    axes[0].plot(pressure, pred_force, "*", color="red", markersize=18,
                 label="Your prediction")
    axes[0].set_xlabel("Pressure (kPa)")
    axes[0].set_ylabel("Max Contact Force (N)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(real_df["pressure_kpa"], real_df["permanent_deformation_mm"],
                 "o-", color="gray", label="Real thesis data (5.56 m/s, 4 kg)")
    axes[1].plot(pressure, pred_defm, "*", color="red", markersize=18,
                 label="Your prediction")
    axes[1].set_xlabel("Pressure (kPa)")
    axes[1].set_ylabel("Permanent Deformation (mm)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("About this model & its limitations"):
        st.markdown("""
        - This is a **reduced-order physics simulator** (1-DOF elastoplastic
          impact oscillator with pressure-dependent stiffness/yield force),
          not a replacement for full 3D FEA (ABAQUS).
        - Calibrated against 5 real thesis data points; typical error is
          **2-15%**, comparable to the thesis's own experimental-vs-FEA
          error of 16.93%.
        - **Known limitation:** the model overshoots at high pressure
          (>1700 kPa) because it assumes a linear pressure-yield
          relationship, while the real pipe shows the force-pressure curve
          flattening near burst pressure (see thesis Fig 4.16b).
        - Pipe geometry (diameter, wall thickness) is fixed at the thesis's
          actual specimen dimensions; velocity and mass are swept because
          they enter as initial conditions, not calibrated constants.
        """)


if __name__ == "__main__":
    main()
