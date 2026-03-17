"""
GRU-based RNN decoder (Willett et al. baseline).

Maps neural sequences to character probabilities via recurrent layers.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base_decoder import BaseDecoder


class _RNNModel:
    """Pure PyTorch GRU model (imported lazily to avoid hard torch dependency at import time)."""

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self, n_inputs, n_outputs, hidden_size, n_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(
                    n_inputs, hidden_size, n_layers,
                    batch_first=True,
                    dropout=dropout if n_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, n_outputs)

            def forward(self, x):
                out, _ = self.gru(x)       # (B, T, hidden)
                return self.fc(out)        # (B, T, n_outputs)

        return _Model(*args, **kwargs)


class RNNDecoder(BaseDecoder):
    """
    GRU-based sequence decoder for neural → character mapping.

    Expects frame-level labels (one class per time-bin), produced by
    forced-alignment HMM.  Training uses cross-entropy loss.

    Args:
        n_inputs:     Number of neural channels.
        n_outputs:    Vocabulary size (number of character classes).
        hidden_size:  GRU hidden units per layer.
        n_layers:     Number of stacked GRU layers.
        dropout:      Dropout between GRU layers (applied only when n_layers > 1).
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_size: int = 512,
        n_layers: int = 2,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        self.n_inputs    = n_inputs
        self.n_outputs   = n_outputs
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.dropout     = dropout
        self.model       = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> "RNNDecoder":
        """
        Train the GRU decoder on aligned (neural, label) pairs.

        Args:
            X: (n_trials, T, n_channels) float array
            y: (n_trials, T) int array of per-timestep character class labels.
               Use -1 to mark padded positions (ignored in loss).
            epochs, batch_size, lr: standard training hyper-parameters.
        """
        import torch
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _RNNModel(
            self.n_inputs, self.n_outputs, self.hidden_size,
            self.n_layers, self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        y_arr = np.asarray(y)
        soft_mode = y_arr.ndim == 3  # (N, T, C) soft probability targets
        if soft_mode:
            y_t = torch.FloatTensor(y_arr.astype(np.float32))
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            y_t = torch.LongTensor(y_arr.astype(np.int64))

        X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32))
        n   = len(X_t)

        self.model.train()
        for epoch in range(epochs):
            perm       = np.random.permutation(n)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                xb  = X_t[idx].to(device)
                yb  = y_t[idx].to(device)

                optimizer.zero_grad()
                logits = self.model(xb)                      # (B, T, C)
                if soft_mode:
                    log_probs = torch.log_softmax(logits, dim=-1)
                    per_frame = -(yb * log_probs).sum(dim=-1)  # (B, T)
                    mask = yb.sum(dim=-1) > 0.5
                    loss = per_frame[mask].mean() if mask.any() else per_frame.mean()
                else:
                    loss = criterion(
                        logits.reshape(-1, self.n_outputs),
                        yb.reshape(-1),
                    )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"[RNN] epoch {epoch + 1}/{epochs}  loss={total_loss:.4f}")

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode neural sequences to per-timestep logits.

        Args:
            X: (n_trials, T, n_channels)

        Returns:
            logits: (n_trials, T, n_outputs)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        import torch

        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
            out = self.model(X_t)
        return out.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights and config to a .pt file."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        import torch

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "n_inputs":    self.n_inputs,
                    "n_outputs":   self.n_outputs,
                    "hidden_size": self.hidden_size,
                    "n_layers":    self.n_layers,
                    "dropout":     self.dropout,
                },
            },
            path,
        )

    def load(self, path: str) -> "RNNDecoder":
        """Load model weights and config from a .pt file."""
        import torch

        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt["config"]
        self.n_inputs    = cfg["n_inputs"]
        self.n_outputs   = cfg["n_outputs"]
        self.hidden_size = cfg["hidden_size"]
        self.n_layers    = cfg["n_layers"]
        self.dropout     = cfg["dropout"]
        self.model = _RNNModel(
            self.n_inputs, self.n_outputs, self.hidden_size,
            self.n_layers, self.dropout,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self
