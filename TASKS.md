# Tasks

## Task 0: Modal smoke test
Goal: verify the entire stack works before spending serious GPU time.

Checklist:
- load data
- preprocess data
- create windows
- build model
- run one forward pass
- run one backward pass
- save one checkpoint
- run one inference sample
- generate one basic plot

Success criteria:
- no shape errors
- no device errors
- no data loading errors
- no checkpoint/save errors

---

## Task 1: Data pipeline
Implement:
- C-MAPSS FD001 loading
- sensor filtering
- z-score normalization
- sliding windows with `T = 40`
- RUL computation capped at `125`

Expected output:
- `X.shape = (B, T, F)`
- `y.shape = (B,)`

---

## Task 2: PyTorch dataset and dataloader
Implement:
- `Dataset`
- `DataLoader`

Use:
- batch size `128`
- `num_workers = 4`
- `pin_memory = True`

The loader should be stable under Modal execution.

---

## Task 3: Backbone model
Implement the main predictor:
- Conv1D frontend
- Mamba backbone
- last-timestep pooling
- MLP regression head

Forward pass should support:
- input `x`
- optional noise input `ε`
- output scalar RUL

For now, noise can be ignored in the forward path until flow is added.

---

## Task 4: Training loop
Implement a clean training loop with:
- MSE loss
- Adam optimizer
- mixed precision
- gradient clipping
- checkpoint save on best validation score

Keep logging minimal and useful.

---

## Task 5: Backbone training run
Train the backbone without flow.

Expected result:
- a stable RUL predictor
- one saved best checkpoint
- one validation metric report

Do not change architecture unless the pipeline fails.

---

## Task 6: Flow module
Implement the stochastic residual:
- sample noise `ε`
- concatenate with latent
- pass through small MLP
- add residual to latent
- produce final RUL

Keep it small and easy to fine-tune.

---

## Task 7: Flow fine-tuning
Load backbone weights and train the flow-enabled model.

Requirements:
- do not retrain from scratch
- keep learning rate conservative
- preserve backbone stability

Goal:
- generate multiple plausible outputs from the same input window

---

## Task 8: Inference
Implement:
- deterministic inference
- stochastic sampling inference

For stochastic inference:
- sample `N = 10`
- return:
  - mean prediction
  - min/max prediction
  - all sampled values if needed for plotting

---

## Task 9: Visualization
Implement plots for:
- true vs predicted RUL
- sampled trajectories
- uncertainty band
- error vs time-to-failure

The sampled trajectory plot is the main one.

---

## Task 10: Evaluation
Compute:
- RMSE
- MAE
- threshold crossing at `RUL < 20`

Report metrics for:
- backbone only
- backbone + flow

Do not compare many variants. Keep it to the final intended model.

---

## Final delivery
At the end, the repo should contain:
- a working Modal training path
- one backbone checkpoint
- one flow-finetuned checkpoint
- trajectory plots
- a short result summary

---

## Execution order
1. smoke test
2. data pipeline
3. backbone model
4. training loop
5. backbone training
6. flow module
7. flow fine-tuning
8. inference
9. plots
10. evaluation