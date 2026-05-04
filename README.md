# Uncertainty-Aware RUL Prediction (C-MAPSS FD001)

## Problem

Remaining Useful Life (RUL) models are often used to estimate how long an engine can keep operating before failure. A standard deterministic model gives one prediction per timestep, but that single number hides uncertainty exactly where it matters most. Near failure, the risk is higher and multiple future outcomes can be plausible, so we want predictions that reflect both expected RUL and confidence.

## Approach

- Dataset: NASA C-MAPSS FD001
- Input format: windowed multivariate time series with `T = 40`
- Feature encoder: Conv1D + LSTM backbone
- Baseline: deterministic RUL regression
- Extension: stochastic inference through noise sampling in latent space
- Output: multiple sampled trajectories instead of a single point estimate

## Key Results

- `RMSE_baseline ≈ 14.7`
- `RMSE_stochastic ≈ 15.5`
- `~5%` RMSE tradeoff for uncertainty-aware predictions
- Uncertainty clearly increases near failure

The stochastic model is slightly worse on point-error metrics, but it gives a much more useful picture of risk. Instead of one overconfident estimate, it shows a spread of plausible outcomes as degradation accelerates.

## Visual Results

### Trajectory with Uncertainty

![Trajectory](outputs/plots/trajectory_main.png)

The model tracks the overall degradation trend while the trajectory spread grows as failure approaches.

### Uncertainty Growth

![Uncertainty](outputs/plots/uncertainty_growth.png)

Prediction uncertainty rises as the engine gets closer to failure, which is the behavior we want from a risk-aware model.

### Error vs Time-to-Failure

![Error](outputs/plots/error_vs_time.png)

Prediction error also tends to increase near failure, reinforcing why uncertainty estimates matter in the late-life regime.

## Key Insight

Deterministic models collapse uncertainty into a single curve, even when the future is ambiguous. The stochastic model exposes multiple plausible futures, and that uncertainty widens in the same region where operational risk is highest.

## How to Run

```bash
pip install -r requirements.txt
python scripts/evaluate.py
python scripts/plots.py
```
