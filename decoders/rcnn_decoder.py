"""
RCNN decoder: Recurrent Convolutional Neural Network.

1D temporal convolutions + recurrent layers for sequence decoding.
Adds local structure capture before recurrence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_decoder import BaseDecoder


def _build_rcnn_model(n_inputs, n_outputs, conv_channels, kernel_size,
                      hidden_size, n_layers, dropout):
    """Construct the RCNN nn.Module (lazy import)."""
    import torch.nn as nn

    class _Conv1dBlock(nn.Module):
        def __init__(self, in_ch, out_ch, ks, drop):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(drop),
            )

        def forward(self, x):
            return self.net(x)

    class _RCNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            convs    = []
            in_ch    = n_inputs
            for out_ch in conv_channels:
                convs.append(_Conv1dBlock(in_ch, out_ch, kernel_size, dropout))
                in_ch = out_ch
            self.convs = nn.Sequential(*convs)
            self.gru   = nn.GRU(
                in_ch, hidden_size, n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, n_outputs)

        def forward(self, x):
            # x: (B, T, n_inputs) → conv expects (B, C, T)
            x = x.permute(0, 2, 1)
            x = self.convs(x)
            x = x.permute(0, 2, 1)          # (B, T, last_ch)
            out, _ = self.gru(x)
            return self.fc(out)             # (B, T, n_outputs)

    return _RCNNModel()


class RCNNDecoder(BaseDecoder):
    """
    RCNN: Conv1D layers followed by GRU/LSTM.

    Captures local temporal patterns via convolutions before
    recurrent processing.  Expects frame-level labels from HMM alignment.

    Args:
        n_inputs:      Number of neural channels.
        n_outputs:     Vocabulary size.
        conv_channels: Tuple of output channel counts for each Conv1D block.
        kernel_size:   Convolutional kernel size (same padding applied).
        hidden_size:   GRU hidden units.
        n_layers:      Number of stacked GRU layers.
        dropout:       Dropout probability (Conv blocks + between GRU layers).
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        conv_channels: tuple = (32, 64, 128),
        kernel_size: int = 5,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        **kwargs: Any,
    ):
        self.n_inputs      = n_inputs
        self.n_outputs     = n_outputs
        self.conv_channels = conv_channels
        self.kernel_size   = kernel_size
        self.hidden_size   = hidden_size
        self.n_layers      = n_layers
        self.dropout       = dropout
        self.model         = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> "RCNNDecoder":
        """
        Train the RCNN decoder.

        Args:
            X: (n_trials, T, n_channels)
            y: (n_trials, T) int frame labels (-1 for padded positions).
        """
        import torch
        import torch.nn as nn

        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_rcnn_model(
            self.n_inputs, self.n_outputs, self.conv_channels,
            self.kernel_size, self.hidden_size, self.n_layers, self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32))
        y_t = torch.LongTensor(np.asarray(y,  dtype=np.int64))
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
                logits = self.model(xb)                        # (B, T, C)
                loss   = criterion(
                    logits.reshape(-1, self.n_outputs),
                    yb.reshape(-1),
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"[RCNN] epoch {epoch + 1}/{epochs}  loss={total_loss:.4f}")

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode neural sequences to per-timestep logits.

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
        """Save model weights and config."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        import torch

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "n_inputs":      self.n_inputs,
                    "n_outputs":     self.n_outputs,
                    "conv_channels": self.conv_channels,
                    "kernel_size":   self.kernel_size,
                    "hidden_size":   self.hidden_size,
                    "n_layers":      self.n_layers,
                    "dropout":       self.dropout,
                },
            },
            path,
        )

    def load(self, path: str) -> "RCNNDecoder":
        """Load model weights and config."""
        import torch

        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt["config"]
        self.n_inputs      = cfg["n_inputs"]
        self.n_outputs     = cfg["n_outputs"]
        self.conv_channels = tuple(cfg["conv_channels"])
        self.kernel_size   = cfg["kernel_size"]
        self.hidden_size   = cfg["hidden_size"]
        self.n_layers      = cfg["n_layers"]
        self.dropout       = cfg["dropout"]
        self.model = _build_rcnn_model(
            self.n_inputs, self.n_outputs, self.conv_channels,
            self.kernel_size, self.hidden_size, self.n_layers, self.dropout,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self
