# Project: RUL Prediction with Stochastic Trajectories

## Goal
Train a single end-to-end model for C-MAPSS FD001 that:
1. predicts Remaining Useful Life (RUL),
2. then produces multiple plausible RUL trajectories through a small noise-conditioned residual,
3. so the final demo shows uncertainty, not just one line.

This is a time-constrained build for one A100 40GB run path:
- one smoke test to verify the pipeline,
- one backbone training run,
- one flow fine-tune run,
- then evaluation + plots.

Do not do hyperparameter sweeps, architecture search, or extra ablations unless the pipeline breaks.

---

## Runtime constraints
- Hardware: Nvidia A100 40GB on Modal
- Budget: about $30
- Goal: minimize wasted GPU time
- Priority: correctness, speed, clean output
- Training should be Modal-friendly and reproducible

---

## Dataset
- Dataset: C-MAPSS FD001
- Input: multivariate engine sensor time series
- Use only sensors with useful variance
- Drop constant / near-constant sensors
- Normalize with z-score statistics fit on train only
- Apply the same normalization to test

---

## Windowing
- Window size: `T = 40`
- Stride: `1`
- Each sample is a sliding window of sensor history
- Input shape: `(B, T, F)`
- `F` should end up around 14-ish after filtering

---

## Labels
- Compute RUL per timestep
- Cap RUL at `125`
- Target shape: `(B,)`

---

## Core model

### Backbone
Use:
- Conv1D frontend
- Mamba sequence model
- RUL regression head

### Conv frontend
- `Conv1d(F -> 64, kernel_size=3, padding=1)`
- ReLU
- `Conv1d(64 -> 128, kernel_size=3, padding=1)`
- ReLU
- Dropout `0.1`

### Mamba backbone
- `d_model = 128`
- `n_layers = 2`
- keep it small and stable

### Pooling
Use last timestep pooling first:
- `z_last = z[:, -1, :]`

### RUL head
- MLP: `128 -> 64 -> 1`

---

## Flow module

This is **not** full diffusion.
It is a small stochastic residual used to generate multiple plausible trajectories.

### Noise
- `noise_dim = 16`
- sample `ε ~ N(0, I)`

### Residual MLP
Input:
- concatenate `z_last` and `ε`

Structure:
- `(128 + 16) -> 128 -> 128 -> 128`

### Combine
- `z_final = z_last + delta`

### Final prediction
- `RUL = head(z_final)`

---

## Training plan

### Phase 0: smoke test
Run one short end-to-end training pass on a small subset to verify:
- dataset loading
- windowing
- model forward pass
- loss backward pass
- checkpoint save
- inference sampling
- plotting

This is not for performance. It is only to catch broken code early.

### Phase 1: backbone training
Train the SSM model without flow first.
This produces the base predictor.

### Phase 2: flow fine-tune
Load the backbone weights and train the residual noise module.
Fine-tune lightly. Do not restart from scratch.

---

## Loss
- Main loss: MSE
- No special diffusion loss
- No auxiliary losses unless required for shape/debugging

---

## Optimizer and training settings
- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: `128`
- Epochs: start with `10` to `15`
- Gradient clipping: `1.0`
- Mixed precision: enabled
- Save best checkpoint only

These are defaults, not a tuning playground.

---

## Inference behavior

### Deterministic mode
- Use zero noise or disable flow
- Produce a single RUL prediction

### Stochastic mode
- Sample `N = 10` noise vectors
- Produce multiple RUL predictions
- Report:
  - mean
  - min/max range
  - optional uncertainty band

---

## What must be shown in the final output
The main value of the project is not just RMSE.

The final demo must clearly show:
1. a single predicted RUL curve,
2. multiple sampled future trajectories,
3. uncertainty increasing near failure.

---

## Metrics
Track:
- RMSE
- MAE

Also track one simple operational metric:
- threshold crossing at `RUL < 20`

---

## Plots to generate
1. True vs predicted RUL over time
2. Multiple sampled trajectories for one engine
3. Mean trajectory with uncertainty band
4. Error vs time-to-failure

The trajectory plot is the main deliverable.

---

## Project structure
Keep the repository neat and easy to run on Modal.

Recommended layout:
- `config/`
- `data/`
- `models/`
- `trainer/`
- `scripts/`
- `infra/`
- `outputs/`

Modal-specific code should live in `infra/`.
Training logic should stay separate from infrastructure code.

---

## Non-goals
Do not build:
- a reusable ML framework
- a multi-agent system
- a diffusion system
- a hyperparameter search pipeline
- unnecessary abstraction layers

The goal is one strong, clean result.