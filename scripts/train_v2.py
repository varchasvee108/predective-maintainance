from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is missing. Install torch before running V2 training."
    ) from exc
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import RULWindowDataset, load_smoke_data
from core.training import quantile_loss
from model.rul_model import RULModel


SEED = 42
BATCH_SIZE = 256
EPOCHS = 3
LR = 3e-4


def main() -> None:
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = ROOT / "outputs" / "checkpoints" / "v2"
    v2_path = checkpoint_dir / "v2_best.pt"

    data = load_smoke_data(
        ROOT / "data" / "raw", smoke_fraction=1.0, val_fraction=0.2, seed=SEED
    )
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

    model = RULModel(feature_dim=data.train.x.shape[-1], use_quantiles=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = quantile_loss
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float("inf")

    print(f"device={device}")
    print("V2 training started")

    for epoch in range(1, EPOCHS + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )
        val_loss = _evaluate(model, val_loader, loss_fn, device)
        print(
            f"[V2] epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            v2_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                v2_path,
            )
            print(f"[V2] saved best checkpoint to {v2_path}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(
    model: RULModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(x)
            y_norm = y / 125.0
            pred_norm = pred / 125.0
            loss = loss_fn(pred_norm, y_norm)

        _assert_finite(pred, "training predictions")
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is NaN or infinity.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    loss_fn,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            _assert_finite(pred, "validation predictions")
            y_norm = y / 125.0
            pred_norm = pred / 125.0
            loss = loss_fn(pred_norm, y_norm)
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
