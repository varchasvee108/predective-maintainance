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
TEST_PATH = ROOT / "data" / "raw" / "test_FD001.txt"
PLOTS_DIR = ROOT / "outputs" / "plots" / "v2"
SEED = 42
WINDOW_SIZE = 40
FAILURE_THRESHOLD = 20.0
SENSOR_START_COL = 5
CALIBRATION_MARGIN = 5.9785


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"V2 checkpoint not found: {CHECKPOINT_PATH}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _set_plot_style()

    prep = load_smoke_data(
        ROOT / "data" / "raw", smoke_fraction=1.0, val_fraction=0.2, seed=SEED
    )
    flow_model = _load_model(
        feature_dim=len(prep.selected_sensors),
        checkpoint_path=CHECKPOINT_PATH,
    )
    engine_id, engine = _load_longest_engine(prep.selected_sensors, prep.mean, prep.std)
    results = _run_engine_inference(flow_model, engine["windows"])

    cycles = engine["cycles_valid"]
    true_rul = engine["true_rul_valid"]
    q10 = results["q10"]
    q50 = results["q50"]
    q90 = results["q90"]

    q10_conf = q10 - CALIBRATION_MARGIN
    q90_conf = q90 + CALIBRATION_MARGIN

    interval_width = q90 - q10
    abs_error = np.abs(q50 - true_rul)

    print(f"selected_engine_id={engine_id}")
    print(f"num_cycles={len(engine['cycles_full'])}")
    print("first_10_true_rul=", np.round(true_rul[:10], 3).tolist())
    print("first_10_q50=", np.round(q50[:10], 3).tolist())

    _plot_trajectory_main(
        cycles=cycles,
        true_rul=true_rul,
        q10=q10,
        q50=q50,
        q90=q90,
        q10_conf=q10_conf,
        q90_conf=q90_conf,
        path=PLOTS_DIR / "trajectory_main.png",
    )
    _plot_uncertainty_growth(
        cycles=cycles,
        true_rul=true_rul,
        interval_width=interval_width,
        path=PLOTS_DIR / "uncertainty_growth.png",
    )
    _plot_interval_misses(
        cycles=cycles,
        true_rul=true_rul,
        q10_conf=q10_conf,
        q50=q50,
        q90_conf=q90_conf,
        path=PLOTS_DIR / "interval_misses.png",
    )
    _plot_uncertainty_vs_error(
        interval_width=interval_width,
        abs_error=abs_error,
        path=PLOTS_DIR / "uncertainty_vs_error.png",
    )
    _plot_error_vs_time(
        true_rul=true_rul,
        mean_error=abs_error,
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


def _load_model(feature_dim: int, checkpoint_path: Path) -> RULModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RULModel(feature_dim=feature_dim, use_quantiles=True).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


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
    flow_model: RULModel,
    windows: torch.Tensor,
) -> dict[str, np.ndarray]:
    device = next(flow_model.parameters()).device
    windows = windows.to(device)
    _assert_finite_tensor(windows, "inference windows")

    with torch.no_grad():
        pred = flow_model(windows)  # (N, 3)

    if pred.ndim != 2 or pred.shape[-1] != 3:
        raise RuntimeError(
            f"Expected quantile output shape (N, 3), got {tuple(pred.shape)}."
        )

    _assert_finite_tensor(pred, "quantile predictions")
    q10 = pred[:, 0].cpu().numpy() * 125.0
    q50 = pred[:, 1].cpu().numpy() * 125.0
    q90 = pred[:, 2].cpu().numpy() * 125.0

    return {"q10": q10, "q50": q50, "q90": q90}


def _plot_trajectory_main(
    cycles: np.ndarray,
    true_rul: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    q10_conf: np.ndarray,
    q90_conf: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots()
    failure_start = _first_crossing(cycles, true_rul, FAILURE_THRESHOLD)
    if failure_start is not None:
        ax.axvspan(
            failure_start, cycles[-1], color="#f7d6d9", alpha=0.25, label="RUL < 20"
        )

    # Conformal interval (wider, lighter)
    ax.fill_between(
        cycles,
        q10_conf,
        q90_conf,
        color="#aec7e8",
        alpha=0.18,
        label="conformal interval",
        zorder=1,
    )

    # q10–q90 quantile band
    ax.fill_between(
        cycles,
        q10,
        q90,
        color="#4c78a8",
        alpha=0.28,
        label="uncertainty band (q10–q90)",
        zorder=2,
    )

    # True RUL and q50 prediction
    ax.plot(cycles, true_rul, color="black", linewidth=2, label="true RUL", zorder=4)
    ax.plot(
        cycles,
        q50,
        color="#1f77b4",
        linewidth=2,
        label="median (q50)",
        zorder=3,
    )

    ax.annotate(
        "uncertainty adapts across degradation stages",
        xy=(cycles[-1], q50[-1]),
        xytext=(cycles[max(0, len(cycles) // 2)], min(110.0, float(np.max(true_rul)))),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#444444"},
        fontsize=12,
        color="#333333",
    )
    ax.set_title("Calibrated Quantile RUL Trajectory")
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
    interval_width: np.ndarray,
    path: Path,
) -> None:
    fig, ax1 = plt.subplots()
    failure_start = _first_crossing(cycles, true_rul, FAILURE_THRESHOLD)
    if failure_start is not None:
        ax1.axvspan(failure_start, cycles[-1], color="#f7d6d9", alpha=0.25)

    smooth_width = _moving_average(interval_width, window=5)
    ax1.plot(
        cycles,
        smooth_width,
        color="#1f77b4",
        linewidth=2.8,
        label="q90 - q10 interval width",
    )
    ax1.set_title("Prediction Interval Width Over Time")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("q90 - q10 interval width", color="#1f77b4")
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


def _plot_interval_misses(
    cycles: np.ndarray,
    true_rul: np.ndarray,
    q10_conf: np.ndarray,
    q50: np.ndarray,
    q90_conf: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots()

    ax.fill_between(
        cycles,
        q10_conf,
        q90_conf,
        color="#4c78a8",
        alpha=0.25,
        label="q10–q90 band",
        zorder=1,
    )
    ax.plot(cycles, true_rul, color="black", linewidth=2, label="true RUL", zorder=3)
    ax.plot(
        cycles,
        q50,
        color="#1f77b4",
        linewidth=2,
        label="median (q50)",
        zorder=2,
    )

    miss_mask = (true_rul < q10_conf) | (true_rul > q90_conf)
    if miss_mask.any():
        ax.scatter(
            cycles[miss_mask],
            true_rul[miss_mask],
            color="red",
            s=28,
            zorder=5,
            label=f"interval miss ({miss_mask.sum()} pts)",
        )

    ax.set_title("Calibration Coverage: Interval Misses")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, max(130.0, float(np.max(true_rul) + 5.0)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_uncertainty_vs_error(
    interval_width: np.ndarray,
    abs_error: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.scatter(
        interval_width,
        abs_error,
        color="#4c78a8",
        alpha=0.4,
        s=20,
        edgecolors="none",
    )
    ax.set_title("Uncertainty vs Prediction Error")
    ax.set_xlabel("q90 - q10 interval width")
    ax.set_ylabel("Absolute q50 error")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_error_vs_time(
    true_rul: np.ndarray,
    mean_error: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots()
    smooth_mean = _moving_average(mean_error, window=5)

    ax.plot(
        true_rul,
        smooth_mean,
        color="#1f77b4",
        linewidth=2.5,
        label="q50 absolute error",
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
