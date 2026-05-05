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

CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "flow_best.pt"
OUTPUT_DIR = ROOT / "outputs" / "demo"
ENGINE_IDS = (25, 31)
SEED = 42
STOCHASTIC_SAMPLES = 30


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {CHECKPOINT_PATH}")

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
            mean_pred=results["mean"],
            pred_min=results["min"],
            pred_max=results["max"],
            samples=results["samples"],
            path=OUTPUT_DIR / f"engine_{engine_id}.png",
        )

        print(
            f"saved demo plot for engine {engine_id} to {OUTPUT_DIR / f'engine_{engine_id}.png'}"
        )


def _load_model(feature_dim: int, device: torch.device) -> RULModel:
    model = RULModel(feature_dim=feature_dim).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state["model_state"])
    model.enable_residual(True)
    model.eval()
    return model


def calibrate(pred: torch.Tensor) -> torch.Tensor:
    pred = torch.clamp(pred, 0.0, 125.0)
    return 0.95 * pred


def _run_inference(
    model: RULModel, windows: torch.Tensor, device: torch.device
) -> dict[str, np.ndarray]:
    windows = windows.to(device)
    _assert_finite_tensor(windows, "inference windows")

    with torch.no_grad():
        stochastic_preds = []
        for _ in range(STOCHASTIC_SAMPLES):
            epsilon = torch.randn(
                windows.shape[0], model.noise_dim, device=device, dtype=windows.dtype
            )
            pred = calibrate(model(windows, epsilon))
            # pred = pred + 0.05 * torch.randn_like(pred) * pred.mean()
            _assert_finite_tensor(pred, "stochastic predictions")
            stochastic_preds.append(pred)

    stochastic = torch.stack(stochastic_preds, dim=0).cpu().numpy()
    _assert_finite_array(stochastic, "stacked stochastic predictions")

    return {
        "samples": stochastic,
        "mean": stochastic.mean(axis=0),
        "min": stochastic.min(axis=0),
        "max": stochastic.max(axis=0),
    }


def _plot_engine(
    engine_id: int,
    cycles: np.ndarray,
    true_rul: np.ndarray,
    mean_pred: np.ndarray,
    pred_min: np.ndarray,
    pred_max: np.ndarray,
    samples: np.ndarray,
    path: Path,
) -> None:
    plt.figure(figsize=(10, 5), dpi=150)

    if len(cycles) > 20:
        plt.axvspan(cycles[-20], cycles[-1], color="#f7d6d9", alpha=0.2)

    plt.fill_between(
        cycles,
        pred_min,
        pred_max,
        color="#4c78a8",
        alpha=0.5,
        label="uncertainty band",
        zorder=1,
    )

    for idx in range(min(10, samples.shape[0])):
        plt.plot(
            cycles,
            samples[idx],
            color="#7ea6d8",
            linewidth=1.0,
            alpha=0.2,
            zorder=2,
        )

    plt.plot(cycles, true_rul, color="black", linewidth=2.5, label="true RUL", zorder=4)
    plt.plot(
        cycles,
        mean_pred,
        color="#1f77b4",
        linewidth=3,
        label="mean prediction",
        zorder=3,
    )

    plt.title(f"Engine {engine_id} RUL Prediction")
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(frameon=False)
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
