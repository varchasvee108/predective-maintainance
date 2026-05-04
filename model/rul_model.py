from __future__ import annotations

import torch
from torch import nn


class RULModel(nn.Module):
    def __init__(self, feature_dim: int, use_residual: bool = False, noise_dim: int = 16) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.noise_dim = noise_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=192,
            num_layers=2,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.residual = nn.Sequential(
            nn.Linear(192 + noise_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
        )
        self._zero_init_residual_final_layer()

    def forward(self, x: torch.Tensor, epsilon: torch.Tensor | None = None) -> torch.Tensor:
        z_last = self.encode(x)
        if self.use_residual:
            z_last = self.apply_residual(z_last, epsilon)
        return self.head(z_last).squeeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, F), got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        z = self.encoder(x)
        z = z.transpose(1, 2)
        z, _ = self.lstm(z)
        return z[:, -1, :]

    def apply_residual(self, z_last: torch.Tensor, epsilon: torch.Tensor | None) -> torch.Tensor:
        batch_size = z_last.shape[0]
        if epsilon is None:
            epsilon = torch.randn(batch_size, self.noise_dim, device=z_last.device, dtype=z_last.dtype)
        if epsilon.shape != (batch_size, self.noise_dim):
            raise ValueError(
                f"Expected epsilon shape {(batch_size, self.noise_dim)}, got {tuple(epsilon.shape)}"
            )

        with torch.no_grad():
            base_pred = self.head(z_last).squeeze(-1).clamp(0, 125)
            stage = 1 - base_pred / 125
            scale = 0.05 + 0.95 * stage
        epsilon_scaled = epsilon * scale.unsqueeze(-1)
        delta = self.residual(torch.cat([z_last, epsilon_scaled], dim=-1))
        return z_last + delta

    def enable_residual(self, enabled: bool = True) -> None:
        self.use_residual = enabled

    def _zero_init_residual_final_layer(self) -> None:
        final_layer = self.residual[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
