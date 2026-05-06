# Uncertainty-Aware RUL Prediction (C-MAPSS FD001)

## 1. Problem

Remaining Useful Life (RUL) models are often used to estimate how long an engine can keep operating before failure. Traditional deterministic models output a single point estimate per timestep. However, single-value predictions are insufficient, especially near degradation transitions where the risk of failure increases. In predictive maintenance, understanding uncertainty and the range of plausible future states is as critical as the expected outcome itself.

## 2. Evolution of the Project

### V1 — Stochastic Latent Sampling

The project initially explored probabilistic forecasting through latent noise injection. By passing stochastic embeddings through the network, the model generated a spread of trajectory samples to represent uncertainty. While this exploratory approach highlighted regions of prediction difficulty, the resulting uncertainty was not properly calibrated. Much of the variance was induced by the design rather than purely learned, leading to unreliable coverage.

### V2 — Quantile Regression + Conformal Calibration

To address the limitations of heuristic noise, the system evolved into a formal probabilistic quantile forecasting approach. The current version predicts specific quantiles (q10, q50, q90) directly.
* **q50** represents the median (expected) prediction.
* **q10** and **q90** define a prediction interval that captures 80% of the expected outcomes.

To ensure this interval is reliable, we apply **conformal calibration**. This post-processing step uses validation set statistics to correct the prediction intervals, ensuring the empirical coverage closely matches the target confidence level.

## 3. Architecture

The current system relies on:
* **Conv1D Encoder:** Extracts local temporal patterns from the multivariate sensor data.
* **LSTM Backbone:** Captures long-range temporal dependencies and degradation trends.
* **Quantile Prediction Head:** Outputs calibrated prediction intervals (q10, q50, q90) instead of single deterministic predictions.

## 4. Visual Comparison: V1 vs V2

V1 explored uncertainty heuristically using stochastic latent sampling. V2 transitioned to direct quantile forecasting combined with conformal calibration. This comparison demonstrates the evolution from heuristic uncertainty to empirically calibrated intervals.

| Feature | V1 (Stochastic Sampling) | V2 (Quantile Regression + Calibration) |
| ------- | ------------------------ | -------------------------------------- |
| **Trajectory Prediction** | ![V1 Trajectory](outputs/plots/trajectory_main.png)<br> *Stochastic trajectory spread exhibiting exploratory, uncalibrated uncertainty behavior.* | ![V2 Trajectory](outputs/plots/v2/trajectory_main.png)<br> *Calibrated q10-q90 intervals forming probabilistic prediction boundaries via conformal calibration.* |
| **Uncertainty Dynamics** | ![V1 Uncertainty Growth](outputs/plots/uncertainty_growth.png)<br> *Heuristic uncertainty induced through latent noise sampling.* | ![V2 Uncertainty Growth](outputs/plots/v2/uncertainty_growth.png)<br> *Uncertainty adapts dynamically across degradation stages, peaking during difficult transition regimes.* |

## 5. V2 Calibration Results

This section establishes V2 as the final, mature system. 

![Interval Misses](outputs/plots/v2/interval_misses.png)

This plot demonstrates the empirical coverage behavior after conformal calibration. The red points indicate interval misses. Most true values fall inside the calibrated intervals.

The intervals achieve strong empirical coverage while remaining reasonably sharp.

| Metric             | Value |
| ------------------ | ----- |
| RMSE               | 13.7  |
| MAE                | 9.4   |
| Coverage (q10–q90) | 0.88  |
| Quantile Crossing  | 0.00  |
| Avg Interval Width | 41.1  |

## 6. Key Insight

Transitioning from heuristic stochastic uncertainty to learned probabilistic forecasting demonstrates that uncertainty is not merely random noise. True predictive uncertainty should reflect the model's confidence and the inherent difficulty of the prediction at that specific time. Proper calibration matters just as much as point accuracy to inform maintenance decisions safely.

## 7. Earlier Exploratory Results (V1)

These were exploratory stochastic trajectory experiments that motivated the transition toward calibrated quantile forecasting. They are archived here to show the progression, but do not represent the final system.

### Engine 25

![Engine 25](outputs/demo/engine_25.png)

### Engine 31

![Engine 31](outputs/demo/engine_31.png)

## 8. Limitations / Future Work

* **Temporal Modeling:** The current architecture uses an LSTM, which has known limitations with very long-range dependencies. Exploring state-space models (e.g., Mamba) or transformers could improve long-horizon dynamics.
* **Architecture Extensions:** Further work could investigate multimodal extensions or richer probabilistic objectives to capture more complex predictive distributions.

## 9. How to Run

```bash
pip install -r requirements.txt
python scripts/evaluate.py
python scripts/plots.py
```
