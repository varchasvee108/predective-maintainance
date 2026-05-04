from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is missing. Install torch before running flow fine-tuning.") from exc
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import RULWindowDataset, load_smoke_data
from model.rul_model import RULModel


SEED = 42
BATCH_SIZE = 256
EPOCHS = 6
LR = 1e-4


def main() -> None:
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = ROOT / "outputs" / "checkpoints"
    backbone_path = checkpoint_dir / "backbone_best.pt"
    flow_path = checkpoint_dir / "flow_best.pt"

    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_path}")

    data = load_smoke_data(ROOT / "data" / "raw", smoke_fraction=1.0, val_fraction=0.2, seed=SEED)
    train_loader = DataLoader(
        RULWindowDataset(data.train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        RULWindowDataset(data.val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = RULModel(feature_dim=data.train.x.shape[-1]).to(device)
    state = torch.load(backbone_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.enable_residual(True)
    _freeze_backbone(model)

    optimizer = torch.optim.Adam(model.residual.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float("inf")

    print(f"device={device}")
    print(f"loading backbone checkpoint: {backbone_path}")
    print("flow fine-tuning started")

    for epoch in range(1, EPOCHS + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_loss = _evaluate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            flow_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                flow_path,
            )
            print(f"saved best flow checkpoint to {flow_path}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _freeze_backbone(model: RULModel) -> None:
    for module in (model.encoder, model.lstm, model.head):
        for param in module.parameters():
            param.requires_grad = False
    for param in model.residual.parameters():
        param.requires_grad = True


def _train_one_epoch(
    model: RULModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        epsilon = torch.randn(x.shape[0], model.noise_dim, device=device, dtype=x.dtype)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(x, epsilon)
            loss = loss_fn(pred, y)

        _assert_finite(pred, "training predictions")
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is NaN or infinity.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.residual.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    if total_count == 0:
        raise RuntimeError("Training dataloader produced zero batches.")
    return total_loss / total_count


def _evaluate(
    model: RULModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            epsilon = torch.randn(x.shape[0], model.noise_dim, device=device, dtype=x.dtype)
            pred = model(x, epsilon)
            _assert_finite(pred, "validation predictions")
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Validation loss is NaN or infinity.")
            batch_size = int(x.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    if total_count == 0:
        raise RuntimeError("Validation dataloader produced zero batches.")
    return total_loss / total_count


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")


if __name__ == "__main__":
    main()
