from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 1,
    lr: float = 1e-3,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    if checkpoint_path is not None and val_loader is None:
        raise ValueError("val_loader is required when saving the best checkpoint by validation loss.")

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        row = {"epoch": float(epoch), "train_loss": train_loss}

        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, loss_fn, device)
            row["val_loss"] = val_loss
            print(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if checkpoint_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "val_loss": val_loss, "epoch": epoch}, path)
        else:
            print(f"epoch {epoch}: train_loss={train_loss:.6f}")

        history.append(row)

    return history


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module | None = None,
    device: torch.device | None = None,
) -> float:
    device = device or next(model.parameters()).device
    loss_fn = loss_fn or nn.MSELoss()
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            _assert_finite(pred, "validation predictions")
            if pred.ndim == 2 and pred.shape[1] == 3:
                loss = quantile_loss(pred, y)
            else:
                loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Validation loss is NaN or infinity.")
            batch_size = int(x.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    if total_count == 0:
        raise RuntimeError("Validation dataloader produced zero batches.")
    return total_loss / total_count


def _train_one_epoch(
    model: nn.Module,
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
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(x)
            loss = loss_fn(pred, y)

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


def quantile_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pinball (quantile) loss for three quantiles.

    pred:   (B, 3) -> [q10, q50, q90]
    target: (B,)
    """
    q = torch.tensor([0.1, 0.5, 0.9], device=pred.device, dtype=pred.dtype)
    target = target.unsqueeze(1)          # (B, 1)
    error = target - pred                 # (B, 3)
    loss = torch.maximum(q * error, (q - 1) * error)
    return loss.mean()


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contain NaN or infinity.")
