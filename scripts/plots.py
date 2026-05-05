from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import load_smoke_data
from model.rul_model import RULModel


CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "v2" / "v2_best.pt"
BACKBONE_CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "backbone_best.pt"
TEST_PATH = ROOT / "data" / "raw" / "test_FD001.txt"
PLOTS_DIR = ROOT / "outputs" / "plots" / "v2"
SEED = 42
WINDOW_SIZE = 40
STOCHASTIC_SAMPLES = 10
DISTRIBUTION_SAMPLES = 50
FAILURE_THRESHOLD = 20.0
SENSOR_START_COL = 5


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {CHECKPOINT_PATH}")
    if not BACKBONE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {BACKBONE_CHECKPOINT_PATH}"
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _set_plot_style()

    prep = load_smoke_data(
        ROOT / "data" / "raw", smoke_fraction=1.0, val_fraction=0.2, seed=SEED
    )
    flow_model = _load_model(
        feature_dim=len(prep.selected_sensors),
        checkpoint_path=CHECKPOINT_PATH,
        use_residual=True,
    )
    backbone_model = _load_model(
        feature_dim=len(prep.selected_sensors),
        checkpoint_path=BACKBONE_CHECKPOINT_PATH,
        use_residual=False,
    )
    engine_id, engine = _load_longest_engine(prep.selected_sensors, prep.mean, prep.std)
    results = _run_engine_inference(flow_model, backbone_model, engine["windows"])

    cycles = engine["cycles_valid"]
    true_rul = engine["true_rul_valid"]
    deterministic = results["deterministic"]
    use_quantiles = results.get("use_quantiles", False)

    if use_quantiles:
        stochastic_mean = results["q50"]
        stochastic_min = results["q10"]
        stochastic_max = results["q90"]
        stochastic = None
        stochastic_std = results["q90"] - results["q10"]
    else:
        stochastic = results["stochastic"]
        stochastic_mean = results["mean"]
        stochastic_min = results["min"]
        stochastic_max = results["max"]
        stochastic_std = results["std"]

    det_error = np.abs(deterministic - true_rul)
    mean_error = np.abs(stochastic_mean - true_rul)
    print(
        "Deterministic vs Stochastic diff:",
        np.mean(np.abs(deterministic - stochastic_mean)),
    )

    print(f"selected_engine_id={engine_id}")
    print(f"num_cycles={len(engine['cycles_full'])}")
    print("first_10_true_rul=", np.round(true_rul[:10], 3).tolist())
    print("first_10_deterministic=", np.round(deterministic[:10], 3).tolist())
    print("first_10_stochastic_mean=", np.round(stochastic_mean[:10], 3).tolist())

    _plot_trajectory_main(
        cycles=cycles,
        true_rul=true_rul,
        deterministic_pred=deterministic,
        stochastic_mean=stochastic_mean,
        stochastic=stochastic,
        stochastic_min=stochastic_min,
        stochastic_max=stochastic_max,
        path=PLOTS_DIR / "trajectory_main.png",
        use_quantiles=use_quantiles,
    )
    _plot_uncertainty_growth(
        cycles=cycles,
        true_rul=true_rul,
        stochastic_std=stochastic_std,
        path=PLOTS_DIR / "uncertainty_growth.png",
    )
    _plot_prediction_distribution(
        model=flow_model,
        window=engine["windows"][-1:],
        true_rul=float(true_rul[-1]),
        path=PLOTS_DIR / "prediction_distribution.png",
    )
    _plot_error_vs_time(
        true_rul=true_rul,
        det_error=det_error,
        mean_error=mean_error,
        path=PLOTS_DIR / "error_vs_time.png",
    )


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#d8d8d8",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_model(
    feature_dim: int, checkpoint_path: Path, use_residual: bool
) -> RULModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RULModel(feature_dim=feature_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.enable_residual(use_residual)
    model.eval()
    return model


def calibrate(pred: torch.Tensor) -> torch.Tensor:
    pred = torch.clamp(pred, 0.0, 125.0)
    return 0.95 * pred


def _load_longest_engine(
    selected_sensors: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[int, dict[str, np.ndarray | torch.Tensor]]:
    raw_test = np.loadtxt(TEST_PATH, dtype=np.float32)
    if raw_test.ndim != 2 or raw_test.shape[1] != 26:
        raise ValueError(f"Unexpected test file shape: {raw_test.shape}")

    engine_ids = raw_test[:, 0].astype(np.int64)
    cycles = raw_test[:, 1].astype(np.int64)
    sensors = raw_test[:, SENSOR_START_COL:].astype(np.float32)
    sensor_idx = np.asarray(
        [int(name.split("_")[1]) - 1 for name in selected_sensors], dtype=np.int64
    )

    unique_ids = np.unique(engine_ids)
    best_engine_id = None
    best_len = -1
    best_cycles = None
    best_features = None

    for engine_id in unique_ids:
        mask = engine_ids == engine_id
        engine_cycles = cycles[mask]
        engine_sensors = sensors[mask]
        order = np.argsort(engine_cycles)
        engine_cycles = engine_cycles[order]
        engine_features = engine_sensors[order][:, sensor_idx]
        if len(engine_cycles) >= WINDOW_SIZE and len(engine_cycles) > best_len:
            best_engine_id = int(engine_id)
            best_len = len(engine_cycles)
            best_cycles = engine_cycles
            best_features = engine_features

    if best_engine_id is None or best_cycles is None or best_features is None:
        raise RuntimeError(f"No test engine has at least {WINDOW_SIZE} cycles.")

    normalized = (best_features - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    _assert_finite_array(normalized, "normalized engine features")

    true_rul_full = np.minimum(best_cycles.max() - best_cycles, 125).astype(np.float32)
    _assert_finite_array(true_rul_full, "true RUL timeline")

    windows = []
    cycles_valid = []
    true_rul_valid = []
    for end in range(WINDOW_SIZE, len(best_cycles) + 1):
        windows.append(normalized[end - WINDOW_SIZE : end])
        cycles_valid.append(int(best_cycles[end - 1]))
        true_rul_valid.append(float(true_rul_full[end - 1]))

    window_tensor = torch.tensor(np.stack(windows), dtype=torch.float32)
    cycles_valid = np.asarray(cycles_valid, dtype=np.int64)
    true_rul_valid = np.asarray(true_rul_valid, dtype=np.float32)

    _assert_finite_tensor(window_tensor, "window tensor")
    _assert_finite_array(true_rul_valid, "valid true RUL")

    return best_engine_id, {
        "cycles_full": best_cycles,
        "windows": window_tensor,
        "cycles_valid": cycles_valid,
        "true_rul_valid": true_rul_valid,
    }


def _run_engine_inference(
    flow_model: RULModel, backbone_model: RULModel, windows: torch.Tensor
) -> dict[str, np.ndarray]:
    device = next(flow_model.parameters()).device
    windows = windows.to(device)
    _assert_finite_tensor(windows, "inference windows")

    # Detect quantile mode from a single probe forward pass
    with torch.no_grad():
        _probe = flow_model(windows[:1])
    use_quantiles = _probe.ndim == 2 and _probe.shape[-1] == 3

    deterministic_preds = []

    with torch.no_grad():
        for idx in range(windows.shape[0]):
            x = windows[idx : idx + 1]
            deterministic = calibrate(backbone_model(x))
            deterministic_preds.append(deterministic.squeeze(0))

    deterministic_tensor = torch.stack(deterministic_preds, dim=0)
    _assert_finite_tensor(deterministic_tensor, "deterministic predictions")
    deterministic_np = deterministic_tensor.cpu().numpy()

    if use_quantiles:
        with torch.no_grad():
            pred = flow_model(windows)  # (N, 3)
        _assert_finite_tensor(pred, "quantile predictions")
        q10 = pred[:, 0].cpu().numpy()
        q50 = pred[:, 1].cpu().numpy()
        q90 = pred[:, 2].cpu().numpy()
        return {
            "use_quantiles": True,
            "deterministic": deterministic_np,
            "q10": q10,
            "q50": q50,
            "q90": q90,
        }
    else:
        stochastic_preds = []
        with torch.no_grad():
            for idx in range(windows.shape[0]):
                x = windows[idx : idx + 1]
                sample_preds = []
                for _ in range(STOCHASTIC_SAMPLES):
                    epsilon = torch.randn(
                        1, flow_model.noise_dim, device=device, dtype=x.dtype
                    )
                    pred = calibrate(flow_model(x, epsilon))
                    sample_preds.append(pred.squeeze(0))
                stochastic_preds.append(torch.stack(sample_preds, dim=0))

        stochastic_tensor = torch.stack(stochastic_preds, dim=1)
        _assert_finite_tensor(stochastic_tensor, "stochastic predictions")
        stochastic_np = stochastic_tensor.cpu().numpy()
        return {
            "use_quantiles": False,
            "deterministic": deterministic_np,
            "stochastic": stochastic_np,
            "mean": stochastic_np.mean(axis=0),
            "min": stochastic_np.min(axis=0),
            "max": stochastic_np.max(axis=0),
            "std": stochastic_np.std(axis=0),
        }


def _plot_trajectory_main(
    cycles: np.ndarray,
    true_rul: np.ndarray,
    deterministic_pred: np.ndarray,
    stochastic_mean: np.ndarray,
    stochastic: np.ndarray | None,
    stochastic_min: np.ndarray,
    stochastic_max: np.ndarray,
    path: Path,
    use_quantiles: bool = False,
) -> None:
    fig, ax = plt.subplots()
    plot_deterministic = not np.allclose(deterministic_pred, stochastic_mean, atol=1e-3)
    failure_start = _first_crossing(cycles, true_rul, FAILURE_THRESHOLD)
    if failure_start is not None:
        ax.axvspan(
            failure_start, cycles[-1], color="#f7d6d9", alpha=0.25, label="RUL < 20"
        )

    if use_quantiles:
        # Quantile mode: fill q10–q90 band, no individual sample lines
        ax.fill_between(
            cycles,
            stochastic_min,  # q10
            stochastic_max,  # q90
            color="#4c78a8",
            alpha=0.20,
            label="uncertainty band (q10–q90)",
            zorder=1,
        )
        pred_label = "median (q50)"
    else:
        # Stochastic mode: spread fill + individual sample lines
        ax.fill_between(
            cycles,
            stochastic_min,
            stochastic_max,
            color="#4c78a8",
            alpha=0.16,
            label="stochastic spread (uncalibrated)",
            zorder=1,
        )
        if stochastic is not None:
            for idx in range(stochastic.shape[0]):
                ax.plot(
                    cycles,
                    stochastic[idx],
                    color="#7ea6d8",
                    linewidth=1.0,
                    alpha=0.22,
                    zorder=1.5,
                )
        pred_label = "stochastic mean"

    ax.plot(cycles, true_rul, color="black", linewidth=2, label="true RUL", zorder=3)
    ax.plot(
        cycles,
        stochastic_mean,
        color="#1f77b4",
        linewidth=2,
        label=pred_label,
        zorder=2,
    )
    if plot_deterministic:
        ax.plot(
            cycles,
            deterministic_pred,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label="baseline (deterministic)",
            zorder=4,
        )

    title = "Quantile RUL Trajectory" if use_quantiles else "Stochastic RUL Trajectory"
    annotation_label = (
        "uncertainty band widens near failure"
        if use_quantiles
        else "stochastic spread increases near failure"
    )
    ax.annotate(
        annotation_label,
        xy=(cycles[-1], stochastic_mean[-1]),
        xytext=(cycles[max(0, len(cycles) // 2)], min(110.0, float(np.max(true_rul)))),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#444444"},
        fontsize=12,
        color="#333333",
    )
    ax.set_title(title)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, max(130.0, float(np.max(true_rul) + 5.0)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_uncertainty_growth(
    cycles: np.ndarray,
    true_rul: np.ndarray,
    stochastic_std: np.ndarray,
    path: Path,
) -> None:
    fig, ax1 = plt.subplots()
    failure_start = _first_crossing(cycles, true_rul, FAILURE_THRESHOLD)
    if failure_start is not None:
        ax1.axvspan(failure_start, cycles[-1], color="#f7d6d9", alpha=0.25)

    smooth_std = _moving_average(stochastic_std, window=5)
    ax1.plot(cycles, smooth_std, color="#1f77b4", linewidth=2.8, label="stochastic std")
    ax1.set_title("Uncertainty Growth Over Time")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Prediction Std", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(
        cycles, true_rul, color="black", linewidth=2.0, alpha=0.75, label="true RUL"
    )
    ax2.set_ylabel("True RUL", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_prediction_distribution(
    model: RULModel,
    window: torch.Tensor,
    true_rul: float,
    path: Path,
) -> None:
    device = next(model.parameters()).device
    window = window.to(device)
    _assert_finite_tensor(window, "distribution window")

    with torch.no_grad():
        samples = []
        for _ in range(DISTRIBUTION_SAMPLES):
            epsilon = torch.randn(1, model.noise_dim, device=device, dtype=window.dtype)
            pred = calibrate(model(window, epsilon))
            samples.append(float(pred.item()))

    sample_array = np.asarray(samples, dtype=np.float32)
    _assert_finite_array(sample_array, "distribution predictions")
    mean_value = float(sample_array.mean())

    fig, ax = plt.subplots()
    ax.hist(
        sample_array,
        bins=12,
        density=True,
        color="#7ea6d8",
        alpha=0.85,
        edgecolor="white",
    )
    ax.axvline(true_rul, color="black", linewidth=2.5, label="true RUL")
    ax.axvline(
        mean_value, color="#1f77b4", linewidth=2.2, linestyle="--", label="sample mean"
    )
    ax.set_title("Prediction Distribution Near Failure")
    ax.set_xlabel("Predicted RUL")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_error_vs_time(
    true_rul: np.ndarray,
    det_error: np.ndarray,
    mean_error: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots()
    smooth_det = _moving_average(det_error, window=5)
    smooth_mean = _moving_average(mean_error, window=5)

    ax.plot(
        true_rul,
        smooth_det,
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="deterministic error",
    )
    ax.plot(
        true_rul,
        smooth_mean,
        color="#1f77b4",
        linewidth=2.5,
        label="stochastic mean error",
    )
    ax.axvspan(FAILURE_THRESHOLD, 0, color="#f7d6d9", alpha=0.22)
    ax.set_title("Absolute Error vs Time to Failure")
    ax.set_xlabel("Time to Failure (RUL)")
    ax.set_ylabel("Absolute Error")
    ax.invert_xaxis()
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _first_crossing(
    cycles: np.ndarray, values: np.ndarray, threshold: float
) -> int | None:
    idx = np.where(values < threshold)[0]
    if idx.size == 0:
        return None
    return int(cycles[idx[0]])


def _assert_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")


def _assert_finite_array(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")


if __name__ == "__main__":
    main()
