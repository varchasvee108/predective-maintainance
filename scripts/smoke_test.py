from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is missing. Install torch in the A100 environment before running Task 0.") from exc
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import RULWindowDataset, load_smoke_data
from core.training import train_model
from model.rul_model import RULModel


SEED = 42
BATCH_SIZE = 128
EPOCHS = 1


def main() -> None:
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = ROOT / "outputs"
    checkpoint_dir = outputs_dir / "checkpoints"
    plot_dir = outputs_dir / "plots"
    log_dir = outputs_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data = load_smoke_data(ROOT / "data" / "raw", smoke_fraction=0.2, seed=SEED)
    train_loader = DataLoader(
        RULWindowDataset(data.train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        RULWindowDataset(data.val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = RULModel(feature_dim=data.train.x.shape[-1]).to(device)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_checkpoint_path = checkpoint_dir / "smoke_backbone_best.pt"
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=1e-3,
        checkpoint_path=best_checkpoint_path,
        device=device,
    )

    model.enable_residual(True)
    residual_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    _train_one_batch(model, train_loader, residual_optimizer, loss_fn, scaler, device)

    checkpoint_path = checkpoint_dir / "smoke_test.pt"
    torch.save({"model_state": model.state_dict(), "feature_dim": data.train.x.shape[-1]}, checkpoint_path)

    reloaded = RULModel(feature_dim=data.train.x.shape[-1]).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    reloaded.eval()

    reloaded.enable_residual(True)
    deterministic = _predict(reloaded, data.test.x[:BATCH_SIZE], device, epsilon_zero=True)
    stochastic = _predict_stochastic(reloaded, data.test.x[:BATCH_SIZE], device, samples=20)
    if torch.allclose(stochastic[0], stochastic[1]):
        raise RuntimeError("Stochastic inference produced identical samples.")

    plot_path = _plot_one_engine(reloaded, data, device, plot_dir / "smoke_test_trajectory.png")

    summary = {
        "device": str(device),
        "selected_sensors": data.selected_sensors,
        "train_shape": list(data.train.x.shape),
        "val_shape": list(data.val.x.shape),
        "test_shape": list(data.test.x.shape),
        "deterministic_shape": list(deterministic.shape),
        "stochastic_shape": list(stochastic.shape),
        "best_backbone_checkpoint": str(best_checkpoint_path),
        "checkpoint": str(checkpoint_path),
        "plot": str(plot_path),
    }
    (log_dir / "smoke_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_batch(
    model: RULModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> None:
    model.train()
    x, y = next(iter(loader))
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        pred = model(x)
        loss = loss_fn(pred, y)
    _assert_finite(pred, "residual smoke predictions")
    if not torch.isfinite(loss):
        raise RuntimeError("Residual smoke loss is NaN or infinity.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()


def _predict(model: RULModel, x: torch.Tensor, device: torch.device, epsilon_zero: bool) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        epsilon = None
        if model.use_residual and epsilon_zero:
            epsilon = torch.zeros(x.shape[0], model.noise_dim, device=device)
        pred = model(x, epsilon=epsilon).clamp(0, 125).cpu()
    _assert_finite(pred, "inference predictions")
    return pred


def _predict_stochastic(model: RULModel, x: torch.Tensor, device: torch.device, samples: int) -> torch.Tensor:
    preds = []
    for _ in range(samples):
        preds.append(_predict(model, x, device, epsilon_zero=False))
    return torch.stack(preds, dim=0)


def _plot_one_engine(model: RULModel, data, device: torch.device, plot_path: Path) -> Path:
    engine_id = int(data.test.engine_id[0])
    mask = data.test.engine_id == engine_id
    x = data.test.x[mask]
    true_rul = data.test.y[mask].numpy()
    cycles = data.test.cycle[mask]

    model.enable_residual(True)
    deterministic = _predict(model, x, device, epsilon_zero=True).numpy()
    samples = _predict_stochastic(model, x, device, samples=10).numpy()
    mean = samples.mean(axis=0)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        svg_path = plot_path.with_suffix(".svg")
        _write_svg_plot(svg_path, cycles, true_rul, deterministic, mean, samples[:10])
        return svg_path

    plt.figure(figsize=(10, 5))
    plt.plot(cycles, true_rul, label="true RUL", linewidth=2)
    plt.plot(cycles, deterministic, label="deterministic", linewidth=1.5)
    plt.plot(cycles, mean, label="stochastic mean", linewidth=1.5)
    for idx in range(min(10, samples.shape[0])):
        plt.plot(cycles, samples[idx], color="tab:orange", alpha=0.2, linewidth=1)
    plt.xlabel("cycle")
    plt.ylabel("RUL")
    plt.ylim(0, 130)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def _write_svg_plot(
    path: Path,
    cycles: np.ndarray,
    true_rul: np.ndarray,
    deterministic: np.ndarray,
    mean: np.ndarray,
    samples: np.ndarray,
) -> None:
    width, height = 900, 450
    margin = 45
    x_min, x_max = float(cycles.min()), float(cycles.max())
    y_min, y_max = 0.0, 130.0

    def points(y_values: np.ndarray) -> str:
        xs = margin + (cycles - x_min) / max(x_max - x_min, 1.0) * (width - 2 * margin)
        ys = height - margin - (y_values - y_min) / (y_max - y_min) * (height - 2 * margin)
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>',
        f'<polyline points="{points(true_rul)}" fill="none" stroke="#1f77b4" stroke-width="2"/>',
        f'<polyline points="{points(deterministic)}" fill="none" stroke="#2ca02c" stroke-width="1.5"/>',
        f'<polyline points="{points(mean)}" fill="none" stroke="#d62728" stroke-width="1.5"/>',
    ]
    for sample in samples:
        lines.append(
            f'<polyline points="{points(sample)}" fill="none" stroke="#ff7f0e" stroke-width="1" opacity="0.25"/>'
        )
    lines.append('<text x="55" y="25" font-size="16">Smoke Test RUL Trajectory</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")


if __name__ == "__main__":
    main()
