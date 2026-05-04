# Tasks (Execution Plan — A100 Single-Run Strategy)

## Task 0 — Smoke Test (MANDATORY FIRST, NO EXCEPTIONS)

Goal:
Prevent wasting A100 time by validating the entire pipeline end-to-end.

Setup:
- Use only 10–20% of dataset
- Run for 1–2 epochs max

You MUST verify:
- dataset loads correctly (no missing files / parsing issues)
- sensor filtering works (correct feature count)
- normalization applied correctly (no NaNs)
- windowing produces correct shapes
- model forward pass works
- loss computes without issues
- backward pass runs (no graph errors)
- optimizer step works
- checkpoint saving works
- inference (with and without noise) runs
- at least one plot renders successfully

Success Criteria:
- No crashes
- No shape mismatches
- No device errors (CPU/GPU mismatch)
- No NaNs in loss or outputs

If ANY of these fail:
→ FIX immediately  
→ DO NOT proceed

---

## Task 1 — Data Pipeline

Implement:
- Load C-MAPSS FD001 dataset
- Group by engine_id
- Sort by cycle

Sensor Processing:
- Compute variance per sensor
- Drop low-variance sensors
- Keep ~14 meaningful sensors

Normalization:
- Compute mean/std on TRAIN split only
- Apply same normalization to:
  - train
  - test

Windowing:
- Sliding window with:
  - T = 40
  - stride = 1
- Each window aligned with last timestep label

RUL Computation:
- RUL = max_cycle - current_cycle
- Apply cap:
  - RUL = min(RUL, 125)

Output:
- X: (B, T, F)
- y: (B,)

Sanity checks:
- print shapes
- check no NaNs
- verify RUL decreasing over time

---

## Task 2 — DataLoader

Implement PyTorch Dataset + DataLoader

Settings:
- batch_size = 128 (increase to 256 if stable)
- shuffle = True (train)
- num_workers = 4
- pin_memory = True

Checks:
- iterate 1 batch
- verify GPU transfer works
- ensure no dataloader hangs

---

## Task 3 — Backbone Model (NO FLOW YET)

Implement:

### Conv Encoder
- reshape: (B, T, F) → (B, F, T)
- Conv1d(F → 64, k=3, padding=1)
- ReLU
- Conv1d(64 → 192, k=3, padding=1)
- ReLU
- Dropout(0.1)
- reshape back: (B, T, 192)

---

### Mamba Backbone
- d_model = 192
- n_layers = 2
- input: (B, T, 192)
- output: (B, T, 192)

---

### Pooling
- z_last = z[:, -1, :]
- shape: (B, 192)

---

### Head
- MLP: 192 → 64 → 1

---

Forward pass:
- input X → RUL prediction

Checks:
- forward pass runs
- output shape = (B, 1)
- no NaNs

---

## Task 4 — Training Loop

Implement minimal, stable loop:

- loss = MSE(pred, target)
- optimizer = Adam (lr=1e-3)

Enable:
- mixed precision (AMP)
- gradient clipping (1.0)

Loop:
- forward
- loss
- backward
- optimizer step

Logging:
- per epoch loss
- optional validation loss

Checkpoint:
- save BEST model only

Sanity:
- loss should decrease (even slightly)

---

## Task 5 — Backbone Training (MAIN RUN 1)

Run full dataset:

- epochs = 10–15
- batch size = 128–256

Track:
- training loss
- validation loss

Output:
- best checkpoint saved

IMPORTANT:
- do NOT modify architecture mid-run
- do NOT tweak hyperparameters repeatedly

Goal:
- stable RUL predictor

---

## Task 6 — Flow Module

Add stochastic residual:

Noise:
- ε ~ N(0, I)
- shape: (B, 16)

Optional (recommended for better visuals):
- scale noise based on predicted RUL:
  - higher uncertainty near failure

---

Residual Input:
- concat(z_last, ε)
- shape: (B, 208)

---

Residual MLP:
- (208 → 192)
- ReLU
- (192 → 192)
- ReLU
- (192 → 192)

---

Combine:
- z_final = z_last + delta

---

Final:
- RUL = head(z_final)

---

Checks:
- forward works with noise
- forward works with zero noise
- output stable

---

## Task 7 — Flow Fine-Tuning (MAIN RUN 2)

- load backbone weights
- enable flow module

Train:
- fewer epochs (5–10)
- lower LR if unstable

Goal:
- introduce controlled variability
- NOT destroy backbone performance

---

## Task 8 — Inference

Implement two modes:

### Deterministic
- ε = 0
- single prediction

---

### Stochastic
- sample N = 10
- run model multiple times

Return:
- list of predictions
- mean
- min/max

---

## Task 9 — Visualization (CRITICAL)

For a single engine:

Plot:
1. True RUL curve
2. Predicted mean curve
3. 5–10 sampled trajectories (light alpha)
4. Optional: shaded uncertainty band

Expected behavior:
- early life → tight predictions
- near failure → spread increases

If this fails:
→ flow is not working properly

---

## Task 10 — Evaluation

Compute:
- RMSE
- MAE

Early detection:
- threshold: RUL < 20
- compare prediction crossing time vs ground truth

---

## Final Deliverables

- trained backbone model
- trained flow model
- trajectory plots (MAIN RESULT)
- metrics

---

## Execution Order (STRICT — DO NOT CHANGE)

1. smoke test
2. data pipeline
3. dataloader
4. backbone model
5. training loop
6. backbone training
7. flow module
8. flow fine-tuning
9. inference
10. visualization
11. evaluation

---

## Critical Rules

- DO NOT skip smoke test
- DO NOT scale model further
- DO NOT add complexity mid-way
- DO NOT run multiple experiments

Goal:
→ one clean, working, convincing result