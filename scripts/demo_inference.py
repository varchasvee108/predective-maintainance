from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.data import load_smoke_data
from model.rul_model import RULModel

CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "v2" / "v2_best.pt"
OUTPUT_DIR = ROOT / "outputs" / "demo" / "v2"
ENGINE_IDS = (25, 31)
SEED = 42
CALIBRATION_MARGIN = 5.9785


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"V2 checkpoint not found: {CHECKPOINT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_smoke_data(
        ROOT / "data" / "raw", smoke_fraction=1.0, val_fraction=0.2, seed=SEED
    )
    model = _load_model(feature_dim=data.test.x.shape[-1], device=device)

    for engine_id in ENGINE_IDS:
        mask = data.test.engine_id == engine_id
        if not np.any(mask):
            raise RuntimeError(f"Engine {engine_id} not found in test split.")

        windows = data.test.x[mask]
        cycles = data.test.cycle[mask]
        true_rul = data.test.y[mask].cpu().numpy()

        _assert_finite_tensor(windows, f"engine {engine_id} windows")
        _assert_finite_array(true_rul, f"engine {engine_id} true RUL")

        results = _run_inference(model, windows, device)
        _plot_engine(
            engine_id=engine_id,
            cycles=cycles,
            true_rul=true_rul,
            q50=results["q50"],
            q10=results["q10"],
            q90=results["q90"],
            q10_conf=results["q10_conf"],
            q90_conf=results["q90_conf"],
            path=OUTPUT_DIR / f"engine_{engine_id}_v2.png",
        )

        print(
            f"saved V2 demo plot for engine {engine_id} to {OUTPUT_DIR / f'engine_{engine_id}_v2.png'}"
        )


def _load_model(feature_dim: int, device: torch.device) -> RULModel:
    model = RULModel(feature_dim=feature_dim, use_quantiles=True).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _run_inference(
    model: RULModel, windows: torch.Tensor, device: torch.device
) -> dict[str, np.ndarray]:
    windows = windows.to(device)
    _assert_finite_tensor(windows, "inference windows")

    with torch.no_grad():
        preds = model(windows)
        preds = preds.cpu().numpy() * 125.0
    _assert_finite_array(preds, "quantile predictions")

    q10 = preds[:, 0]
    q50 = preds[:, 1]
    q90 = preds[:, 2]

    _assert_finite_array(q10, "q10 predictions")
    _assert_finite_array(q50, "q50 predictions")
    _assert_finite_array(q90, "q90 predictions")

    q10_conf = q10 - CALIBRATION_MARGIN
    q90_conf = q90 + CALIBRATION_MARGIN

    return {
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "q10_conf": q10_conf,
        "q90_conf": q90_conf,
    }


def _plot_engine(
    engine_id: int,
    cycles: np.ndarray,
    true_rul: np.ndarray,
    q50: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    q10_conf: np.ndarray,
    q90_conf: np.ndarray,
    path: Path,
) -> None:
    plt.figure(figsize=(10, 5), dpi=150)

    if len(cycles) > 20:
        plt.axvspan(cycles[-20], cycles[-1], color="#f7d6d9", alpha=0.2)

    plt.fill_between(
        cycles,
        q10_conf,
        q90_conf,
        color="#c6d4e1",
        alpha=0.3,
        label="conformal interval",
        zorder=1,
    )

    plt.fill_between(
        cycles,
        q10,
        q90,
        color="#4c78a8",
        alpha=0.5,
        label="q10-q90 interval",
        zorder=2,
    )

    plt.plot(cycles, true_rul, color="black", linewidth=2.5, label="true RUL", zorder=4)
    plt.plot(
        cycles,
        q50,
        color="#1f77b4",
        linewidth=3,
        label="median prediction (q50)",
        zorder=3,
    )

    plt.title(f"Engine {engine_id} Calibrated RUL Forecast")
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _assert_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")


def _assert_finite_array(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")


if __name__ == "__main__":
    main()
