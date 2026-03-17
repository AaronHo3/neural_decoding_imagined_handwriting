"""
Conformer decoder for neural handwriting decoding.

Combines multi-head self-attention with depthwise convolutions to capture
both long-range context and local temporal structure in neural time series.
Architecture follows Gulati et al. (2020) "Conformer: Convolution-augmented
Transformer for Speech Recognition", adapted for neural spike decoding.

This is a novel addition — Willett et al. (2021) used a GRU baseline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base_decoder import BaseDecoder


# ======================================================================
# PyTorch model components (lazy-imported)
# ======================================================================

def _build_conformer_model(
    n_inputs: int,
    n_outputs: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    conv_kernel_size: int,
    ff_dim: int,
    dropout: float,
):
    """Construct the Conformer nn.Module."""
    import torch
    import torch.nn as nn

    class _Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    class _PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding (Vaswani et al. 2017)."""

        def __init__(self, d_model: int, max_len: int = 10000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x):
            # x: (B, T, d_model)
            return x + self.pe[:, : x.size(1)]

    class _FeedForward(nn.Module):
        """Position-wise feed-forward with expansion factor."""

        def __init__(self, d_model, ff_dim, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, ff_dim),
                _Swish(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class _ConvModule(nn.Module):
        """
        Conformer convolution module:
        LayerNorm → Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout
        """

        def __init__(self, d_model, kernel_size, dropout):
            super().__init__()
            self.layer_norm = nn.LayerNorm(d_model)
            # Pointwise expand (×2 for GLU gate)
            self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
            self.glu = nn.GLU(dim=1)
            # Depthwise conv with same padding
            self.depthwise = nn.Conv1d(
                d_model, d_model, kernel_size,
                padding=kernel_size // 2, groups=d_model,
            )
            self.batch_norm = nn.BatchNorm1d(d_model)
            self.swish = _Swish()
            # Pointwise project back
            self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (B, T, d_model)
            x = self.layer_norm(x)
            x = x.permute(0, 2, 1)              # (B, d_model, T)
            x = self.pointwise1(x)               # (B, 2*d_model, T)
            x = self.glu(x)                      # (B, d_model, T)
            x = self.depthwise(x)                # (B, d_model, T)
            x = self.batch_norm(x)
            x = self.swish(x)
            x = self.pointwise2(x)               # (B, d_model, T)
            x = self.dropout(x)
            return x.permute(0, 2, 1)            # (B, T, d_model)

    class _MultiHeadSelfAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout):
            super().__init__()
            self.layer_norm = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x_norm = self.layer_norm(x)
            out, _ = self.attn(x_norm, x_norm, x_norm)
            return self.dropout(out)

    class _ConformerBlock(nn.Module):
        """
        Single Conformer block (Macaron-style):
            ½ FFN → MHSA → Conv → ½ FFN → LayerNorm
        """

        def __init__(self, d_model, n_heads, conv_kernel_size, ff_dim, dropout):
            super().__init__()
            self.ff1 = _FeedForward(d_model, ff_dim, dropout)
            self.attn = _MultiHeadSelfAttention(d_model, n_heads, dropout)
            self.conv = _ConvModule(d_model, conv_kernel_size, dropout)
            self.ff2 = _FeedForward(d_model, ff_dim, dropout)
            self.layer_norm = nn.LayerNorm(d_model)

        def forward(self, x):
            x = x + 0.5 * self.ff1(x)
            x = x + self.attn(x)
            x = x + self.conv(x)
            x = x + 0.5 * self.ff2(x)
            return self.layer_norm(x)

    class _ConformerDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_inputs, d_model)
            self.pos_enc = _PositionalEncoding(d_model)
            self.input_dropout = nn.Dropout(dropout)

            self.blocks = nn.ModuleList([
                _ConformerBlock(d_model, n_heads, conv_kernel_size, ff_dim, dropout)
                for _ in range(n_layers)
            ])

            self.output_proj = nn.Linear(d_model, n_outputs)

        def forward(self, x):
            # x: (B, T, n_inputs)
            x = self.input_proj(x)           # (B, T, d_model)
            x = self.pos_enc(x)
            x = self.input_dropout(x)
            for block in self.blocks:
                x = block(x)
            return self.output_proj(x)       # (B, T, n_outputs)

    return _ConformerDecoder()


# ======================================================================
# Public decoder class
# ======================================================================

class TransformerDecoder(BaseDecoder):
    """
    Conformer-based sequence decoder for neural → character mapping.

    Uses multi-head self-attention + depthwise convolutions (Conformer
    architecture) to capture both global context and local temporal patterns.
    Trained with frame-level cross-entropy on HMM-aligned labels, same as
    the RNN / RCNN decoders.

    Args:
        n_inputs:        Number of neural channels.
        n_outputs:       Vocabulary size (number of character classes).
        d_model:         Internal embedding dimension.
        n_heads:         Number of attention heads.
        n_layers:        Number of Conformer blocks.
        conv_kernel_size: Kernel size for the depthwise convolution module.
                          Larger values capture broader local context.
        ff_dim:          Feed-forward expansion dimension.
        dropout:         Dropout probability used throughout the model.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        conv_kernel_size: int = 31,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        self.n_inputs         = n_inputs
        self.n_outputs        = n_outputs
        self.d_model          = d_model
        self.n_heads          = n_heads
        self.n_layers         = n_layers
        self.conv_kernel_size = conv_kernel_size
        self.ff_dim           = ff_dim
        self.dropout          = dropout
        self.model            = None

    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 5e-4,
        warmup_steps: int = 500,
        **kwargs: Any,
    ) -> "TransformerDecoder":
        """
        Train the Conformer decoder on aligned (neural, label) pairs.

        Args:
            X: (n_trials, T, n_channels) float array.
            y: (n_trials, T) int array of per-timestep character labels.
               Use -1 for padded positions (ignored in loss).
            epochs, batch_size, lr: standard hyper-parameters.
            warmup_steps: Number of linear-warmup optimiser steps.
        """
        import torch
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_conformer_model(
            self.n_inputs, self.n_outputs, self.d_model, self.n_heads,
            self.n_layers, self.conv_kernel_size, self.ff_dim, self.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-2,
        )

        y_arr = np.asarray(y)
        soft_mode = y_arr.ndim == 3
        if soft_mode:
            y_t = torch.FloatTensor(y_arr.astype(np.float32))
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            y_t = torch.LongTensor(y_arr.astype(np.int64))

        # Linear warmup + cosine annealing schedule
        total_steps = max(1, (len(X) // batch_size) * epochs)

        def _lr_lambda(step):
            if step < warmup_steps:
                return max(step, 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32))
        n = len(X_t)

        self.model.train()
        global_step = 0
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0

            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                xb = X_t[idx].to(device)
                yb = y_t[idx].to(device)

                optimizer.zero_grad()
                logits = self.model(xb)  # (B, T, C)
                if soft_mode:
                    log_probs = torch.log_softmax(logits, dim=-1)
                    per_frame = -(yb * log_probs).sum(dim=-1)
                    mask = yb.sum(dim=-1) > 0.5
                    loss = per_frame[mask].mean() if mask.any() else per_frame.mean()
                else:
                    loss = criterion(
                        logits.reshape(-1, self.n_outputs),
                        yb.reshape(-1),
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                global_step += 1

            if (epoch + 1) % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[Conformer] epoch {epoch + 1}/{epochs}  "
                    f"loss={total_loss:.4f}  lr={current_lr:.2e}"
                )

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
                    "n_inputs":         self.n_inputs,
                    "n_outputs":        self.n_outputs,
                    "d_model":          self.d_model,
                    "n_heads":          self.n_heads,
                    "n_layers":         self.n_layers,
                    "conv_kernel_size": self.conv_kernel_size,
                    "ff_dim":           self.ff_dim,
                    "dropout":          self.dropout,
                },
            },
            path,
        )

    def load(self, path: str) -> "TransformerDecoder":
        """Load model weights and config from a .pt file."""
        import torch

        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        self.n_inputs         = cfg["n_inputs"]
        self.n_outputs        = cfg["n_outputs"]
        self.d_model          = cfg["d_model"]
        self.n_heads          = cfg["n_heads"]
        self.n_layers         = cfg["n_layers"]
        self.conv_kernel_size = cfg["conv_kernel_size"]
        self.ff_dim           = cfg["ff_dim"]
        self.dropout          = cfg["dropout"]
        self.model = _build_conformer_model(
            self.n_inputs, self.n_outputs, self.d_model, self.n_heads,
            self.n_layers, self.conv_kernel_size, self.ff_dim, self.dropout,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self
