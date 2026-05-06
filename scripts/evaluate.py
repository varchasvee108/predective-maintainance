from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is missing. Install torch before running evaluation."
    ) from exc
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import RULWindowDataset, load_smoke_data
from model.rul_model import RULModel


SEED = 42
BATCH_SIZE = 256
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "v2" / "v2_best.pt"


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"V2 checkpoint not found: {CHECKPOINT_PATH}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_smoke_data(
        ROOT / "data" / "raw",
        smoke_fraction=1.0,
        val_fraction=0.2,
        seed=SEED,
    )

    val_loader = DataLoader(
        RULWindowDataset(data.val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        RULWindowDataset(data.test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = _load_model(
        CHECKPOINT_PATH,
        feature_dim=data.test.x.shape[-1],
        device=device,
    )

    y_val, q50_val, q10_val, q90_val = _run_inference(
        model=model,
        loader=val_loader,
        device=device,
    )

    q10_val *= 125.0
    q90_val *= 125.0

    calibration_margin = _compute_conformal_margin(
        q10_val,
        q90_val,
        y_val,
    )

    true_rul, q50_pred, q10_pred, q90_pred = _run_inference(
        model=model,
        loader=test_loader,
        device=device,
    )

    q10_pred *= 125.0
    q50_pred *= 125.0
    q90_pred *= 125.0

    q10_pred = q10_pred - calibration_margin
    q90_pred = q90_pred + calibration_margin

    rmse = _rmse(true_rul, q50_pred)
    mae = _mae(true_rul, q50_pred)
    coverage = np.mean((true_rul >= q10_pred) & (true_rul <= q90_pred))
    crossing_rate = np.mean((q10_pred > q50_pred) | (q50_pred > q90_pred))
    avg_width = (q90_pred - q10_pred).mean()

    print("Evaluation on full FD001 test set")
    print(true_rul[:5])
    print(q50_pred[:5])
    print(f"test_windows={len(true_rul)}")
    print(f"calibration_margin={calibration_margin:.4f}")
    print(f"RMSE={rmse:.4f}")
    print(f"MAE={mae:.4f}")
    print(f"coverage_q10_q90={coverage:.4f}")
    print(f"quantile_crossing_rate={crossing_rate:.4f}")
    print(f"avg_interval_width={avg_width:.4f}")


def _load_model(
    checkpoint_path: Path,
    feature_dim: int,
    device: torch.device,
) -> RULModel:
    model = RULModel(
        feature_dim=feature_dim,
        use_quantiles=True,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state["model_state"])
    model.eval()

    return model


def _run_inference(
    model: RULModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_all: list[torch.Tensor] = []
    q10_all: list[torch.Tensor] = []
    q50_all: list[torch.Tensor] = []
    q90_all: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)

            _assert_finite(pred, "quantile predictions")

            if pred.ndim != 2 or pred.shape[-1] != 3:
                raise RuntimeError(
                    f"Expected quantile output shape (B, 3), got {tuple(pred.shape)}."
                )

            q10_all.append(pred[:, 0].cpu())
            q50_all.append(pred[:, 1].cpu())
            q90_all.append(pred[:, 2].cpu())
            true_all.append(y.cpu())

    true_rul = torch.cat(true_all).numpy()
    q10_pred = torch.cat(q10_all).numpy()
    q50_pred = torch.cat(q50_all).numpy()
    q90_pred = torch.cat(q90_all).numpy()

    _assert_finite_np(true_rul, "true RUL")
    _assert_finite_np(q10_pred, "q10 predictions")
    _assert_finite_np(q50_pred, "q50 predictions")
    _assert_finite_np(q90_pred, "q90 predictions")

    return true_rul, q50_pred, q10_pred, q90_pred


def _compute_conformal_margin(
    q10_val: np.ndarray,
    q90_val: np.ndarray,
    y_val: np.ndarray,
    coverage_level: float = 0.9,
) -> float:
    errors = np.maximum.reduce(
        [
            q10_val - y_val,
            y_val - q90_val,
            np.zeros_like(y_val),
        ]
    )

    return float(np.quantile(errors, coverage_level))


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
