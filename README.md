# Pipe Impact ML — Physics-Informed Surrogate Model

Extends the BSc thesis "Behavior of Clamped uPVC Pipe with Different Internal
Pressure Subjected to Low Velocity Impact" (Ahmed & Bashar, RUET, 2023) into
a machine-learning surrogate model, using a Python physics simulator (in
place of ABAQUS) validated against the thesis's real experimental data.

## Status
🚧 Work in progress — currently on: **calibrating the impact simulator**
against the 5 real experimental data points from the thesis.

## Project structure
```
simulator/    physics model (Python, no external FEA software needed)
data/         real thesis data + generated parametric dataset
models/       ML training scripts (Random Forest / XGBoost / NN)
app/          Streamlit interactive demo
notebooks/    exploration & validation plots
tests/        pytest suite
```

## Running locally (Anaconda)
See setup instructions from the project tutor conversation. Quick version:

```bash
conda create -n pipe_impact python=3.11 -y
conda activate pipe_impact
pip install -r requirements.txt
python simulator/impact_model.py
```
