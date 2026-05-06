# Uncertainty-Aware RUL Prediction (C-MAPSS FD001)

## 1. Problem

Remaining Useful Life (RUL) models are often used to estimate how long an engine can keep operating before failure. Traditional deterministic models output a single point estimate per timestep. However, single-value predictions are insufficient, especially near degradation transitions where the risk of failure increases. In predictive maintenance, understanding uncertainty and the range of plausible future states is as critical as the expected outcome itself.

## 2. Evolution of the Project

### V1 — Stochastic Latent Sampling

The project initially explored probabilistic forecasting through latent noise injection. By passing stochastic embeddings through the network, the model generated a spread of trajectory samples to represent uncertainty. While this exploratory approach highlighted regions of prediction difficulty, the resulting uncertainty was not properly calibrated. Much of the variance was induced by the design rather than purely learned, leading to unreliable coverage and overconfidence near critical failure regimes.

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

## 4. Final Results

Our calibrated intervals achieve strong empirical coverage while remaining reasonably sharp.

| Metric             | Value |
| ------------------ | ----- |
| RMSE               | 13.7  |
| MAE                | 9.4   |
| Coverage (q10–q90) | 0.88  |
| Quantile Crossing  | 0.00  |
| Avg Interval Width | 41.1  |

## 5. Visual Results

### Trajectory Plot

![Trajectory](outputs/plots/v2/trajectory_main.png)

The q50 median prediction successfully tracks the degradation trend. The uncertainty adapts dynamically across different degradation stages, and the conformal intervals widen the prediction bounds conservatively to ensure robust coverage.

### Calibration Coverage Plot

![Interval Misses](outputs/plots/v2/interval_misses.png)

This plot demonstrates the empirical coverage behavior of the calibrated model. The red points indicate interval misses (where the true RUL falls outside the conformal interval). The vast majority of true values fall safely inside the calibrated intervals.

### Uncertainty Width Plot

![Uncertainty Growth](outputs/plots/v2/uncertainty_growth.png)

Prediction uncertainty changes dynamically across degradation stages. Uncertainty often peaks during difficult transition regimes as the equipment begins to degrade. As terminal failure approaches and degradation becomes more predictable, the intervals begin to narrow.

## 6. Key Insight

Transitioning from heuristic stochastic uncertainty to learned probabilistic forecasting demonstrates that uncertainty is not merely random noise. True predictive uncertainty should reflect the model's confidence and the inherent difficulty of the prediction at that specific time. In predictive maintenance, proper calibration matters just as much as point accuracy to safely inform maintenance decisions.

## Earlier Exploratory Results (V1)

Earlier versions of this project explored uncertainty through stochastic latent-space sampling. Multiple plausible RUL trajectories were generated using noise-conditioned inference to highlight regions of ambiguity during degradation progression. These uncertainty estimates were purely exploratory and not formally calibrated. V2 later replaced this approach with direct quantile forecasting and conformal calibration.

### Engine 25

![Engine 25](outputs/demo/engine_25.png)

This trajectory illustrates the stochastic spread, exposing prediction ambiguity as the engine transitions into a degraded state.

### Engine 31

![Engine 31](outputs/demo/engine_31.png)

Another exploratory example demonstrating the uncalibrated but informative trajectory spread during degradation progression.

## 7. Limitations / Future Work

* **Temporal Modeling:** The current architecture uses an LSTM, which has known limitations with very long-range dependencies. Exploring state-space models (e.g., Mamba) or transformers could improve long-horizon dynamics.
* **Architecture Extensions:** Further work could investigate multimodal extensions or richer probabilistic objectives to capture more complex predictive distributions.

## 8. How to Run

```bash
pip install -r requirements.txt
python scripts/evaluate.py
python scripts/plots.py
```
