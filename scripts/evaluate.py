from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is missing. Install torch before running evaluation.") from exc
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import RULWindowDataset, load_smoke_data
from model.rul_model import RULModel


SEED = 42
BATCH_SIZE = 256
STOCHASTIC_SAMPLES = 10
BACKBONE_PATH = ROOT / "outputs" / "checkpoints" / "backbone_best.pt"
FLOW_PATH = ROOT / "outputs" / "checkpoints" / "flow_best.pt"


def main() -> None:
    if not BACKBONE_PATH.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {BACKBONE_PATH}")
    if not FLOW_PATH.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {FLOW_PATH}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_smoke_data(ROOT / "data" / "raw", smoke_fraction=1.0, val_fraction=0.2, seed=SEED)
    test_loader = DataLoader(
        RULWindowDataset(data.test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    baseline_model = _load_model(
        checkpoint_path=BACKBONE_PATH,
        feature_dim=data.test.x.shape[-1],
        device=device,
        use_residual=False,
    )
    flow_model = _load_model(
        checkpoint_path=FLOW_PATH,
        feature_dim=data.test.x.shape[-1],
        device=device,
        use_residual=True,
    )

    true_rul, baseline_pred, stochastic_mean = _run_inference(
        baseline_model=baseline_model,
        flow_model=flow_model,
        loader=test_loader,
        device=device,
    )

    rmse_baseline = _rmse(true_rul, baseline_pred)
    rmse_stochastic = _rmse(true_rul, stochastic_mean)
    mae_baseline = _mae(true_rul, baseline_pred)
    mae_stochastic = _mae(true_rul, stochastic_mean)
    rmse_pct_diff = ((rmse_stochastic - rmse_baseline) / rmse_baseline) * 100.0

    print("Evaluation on full FD001 test set")
    print(f"test_windows={len(true_rul)}")
    print(f"RMSE_baseline={rmse_baseline:.4f}")
    print(f"RMSE_stochastic={rmse_stochastic:.4f}")
    print(f"MAE_baseline={mae_baseline:.4f}")
    print(f"MAE_stochastic={mae_stochastic:.4f}")
    print(f"RMSE_percent_difference={rmse_pct_diff:.2f}%")

    if rmse_pct_diff < 0:
        print(f"Stochastic mean improves RMSE by {abs(rmse_pct_diff):.2f}%")
    elif rmse_pct_diff > 0:
        print(f"Stochastic mean worsens RMSE by {rmse_pct_diff:.2f}%")
    else:
        print("Stochastic mean matches baseline RMSE exactly")


def _load_model(
    checkpoint_path: Path,
    feature_dim: int,
    device: torch.device,
    use_residual: bool,
) -> RULModel:
    model = RULModel(feature_dim=feature_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.enable_residual(use_residual)
    model.eval()
    return model


def calibrate(pred: torch.Tensor) -> torch.Tensor:
    pred = torch.clamp(pred, 0.0, 125.0)
    return 0.95 * pred


def _run_inference(
    baseline_model: RULModel,
    flow_model: RULModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_all: list[torch.Tensor] = []
    baseline_all: list[torch.Tensor] = []
    stochastic_mean_all: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            baseline_pred = calibrate(baseline_model(x))
            _assert_finite(baseline_pred, "baseline predictions")

            stochastic_preds = []
            for _ in range(STOCHASTIC_SAMPLES):
                epsilon = torch.randn(x.shape[0], flow_model.noise_dim, device=device, dtype=x.dtype)
                pred = calibrate(flow_model(x, epsilon))
                _assert_finite(pred, "stochastic predictions")
                stochastic_preds.append(pred)
            stochastic_mean = torch.stack(stochastic_preds, dim=0).mean(dim=0)

            true_all.append(y.cpu())
            baseline_all.append(baseline_pred.cpu())
            stochastic_mean_all.append(stochastic_mean.cpu())

    true_rul = torch.cat(true_all).numpy()
    baseline_pred = torch.cat(baseline_all).numpy()
    stochastic_mean = torch.cat(stochastic_mean_all).numpy()

    _assert_finite_np(true_rul, "true RUL")
    _assert_finite_np(baseline_pred, "baseline predictions")
    _assert_finite_np(stochastic_mean, "stochastic mean predictions")

    return true_rul, baseline_pred, stochastic_mean


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _mae(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")


def _assert_finite_np(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")


if __name__ == "__main__":
    main()
