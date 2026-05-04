# Project: RUL Prediction with Stochastic Trajectories (Single-Run, A100)

## Objective
Build a model that:
1. Predicts Remaining Useful Life (RUL) from C-MAPSS FD001
2. Generates multiple plausible future RUL predictions using stochastic residuals
3. Demonstrates uncertainty increasing near failure

Execution strategy:
- 1 smoke test
- 1 backbone training
- 1 flow fine-tune
- evaluation + plots

No experimentation loops.

---

## Dataset

### Source
- C-MAPSS FD001

### Preprocessing
- Drop low-variance sensors (based on std threshold)
- Keep ~14 sensors (log final selected indices)
- Normalize using z-score:
  - compute mean/std on TRAIN only
  - apply same stats to train + test
- Replace any NaNs after normalization with 0 (safety)

---

## Windowing

- Window size: `T = 40`
- Stride: `1`
- Generate sliding windows per engine_id
- Label = RUL at last timestep of window

### Shapes
- Input: `(B, T, F)`
- Output: `(B,)`

### Sanity Checks (must pass)
- No NaNs in X or y
- RUL strictly non-increasing within each engine
- Feature count consistent across splits

---

## RUL Computation

- `RUL = max_cycle - current_cycle`
- Cap:
  - `RUL = min(RUL, 125)`
- Normalize RUL optionally to [0,1] (only if training unstable; default = no)

---

## Model Architecture

### Input reshape
- `(B, T, F)` → `(B, F, T)` for Conv1D
- Back to `(B, T, C)` after conv

---

### Conv Encoder
- Conv1d(F → 64, kernel=3, padding=1)
- ReLU
- Conv1d(64 → 192, kernel=3, padding=1)
- ReLU
- Dropout(0.1)

Output:
- `(B, T, 192)`

---

### Mamba Backbone
- d_model = 192
- n_layers = 2
- Input: `(B, T, 192)`
- Output: `(B, T, 192)`

---

### Pooling
- `z_last = z[:, -1, :]` → `(B, 192)`
- (Optional fallback: mean pooling if unstable)

---

### RUL Head
- MLP: 192 → 64 → 1

---

## Flow Module (Stochastic Residual)

### Noise
- `noise_dim = 16`
- `ε ~ N(0, I)` → `(B, 16)`

### Noise Scaling (IMPORTANT for trajectories)
- Scale noise based on degradation stage:
  - `scale = sigmoid((125 - RUL_pred) / 125)`
  - `ε = ε * scale`
- Ensures:
  - low variance early
  - higher variance near failure

---

### Residual Input
- concat(z_last, ε) → `(B, 208)`

---

### Residual MLP
- 208 → 192 → 192 → 192
- ReLU between layers

---

### Combine
- `z_final = z_last + delta`

---

### Final Prediction
- `RUL = head(z_final)`

---

## Training

### Loss
- MSE

---

### Hyperparameters
- Batch size: 128–256 (prefer 256 on A100)
- Epochs: 10–15
- Optimizer: Adam
- LR: 1e-3
- Gradient clipping: 1.0
- Mixed precision: ON
- Early stop if loss plateaus (optional)

---

### Performance Constraints (must hit)
- One epoch ≤ 5 minutes on A100
- No GPU underutilization (check batch size)

---

## Training Phases

### Phase 0 — Smoke Test
- 10–20% data
- 1–2 epochs

Verify:
- forward + backward
- no NaNs
- checkpoint save/load
- stochastic inference works
- plot renders

---

### Phase 1 — Backbone Training
- Train Conv + Mamba + head
- No flow
- Save best checkpoint

---

### Phase 2 — Flow Fine-Tune
- Load backbone weights
- Enable flow
- Train 5–10 epochs
- Lower LR if unstable

---

## Inference

### Deterministic
- ε = 0

---

### Stochastic
- Sample N = 10
- Output:
  - predictions list
  - mean
  - min/max
  - (optional std)

---

## Output Requirement (CRITICAL)

For ONE engine:

Plot:
- true RUL
- predicted mean
- 5–10 sampled trajectories

Expected:
- tight early predictions
- spread increases near failure

---

## Metrics

- RMSE
- MAE

---

## Early Detection

- threshold: RUL < 20
- compare crossing timestep

---

## Failure Conditions (debug triggers)

If any occur:
- constant predictions → model collapse
- no variance in trajectories → flow broken
- exploding loss → normalization or LR issue
- NaNs → data or AMP issue

---

## Project Structure

config/
data/
models/
trainer/
scripts/
infra/
outputs/

---

## Non-Goals

- No diffusion models
- No architecture search
- No multi-run tuning
- No unnecessary abstraction

Goal:
→ one clean, interpretable result with visible uncertainty