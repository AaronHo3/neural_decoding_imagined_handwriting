"""
CTC decoder: CNN-BiLSTM with Connectionist Temporal Classification.

Alignment-free training inspired by offline handwriting OCR.
No explicit character boundaries required during training.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .base_decoder import BaseDecoder


def _build_ctc_model(n_inputs, n_outputs, conv_channels, kernel_size,
                     lstm_hidden, dropout):
    """Construct the CTC nn.Module (lazy import)."""
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

    class _CTCModel(nn.Module):
        def __init__(self):
            super().__init__()
            convs = []
            in_ch = n_inputs
            for out_ch in conv_channels:
                convs.append(_Conv1dBlock(in_ch, out_ch, kernel_size, dropout))
                in_ch = out_ch
            self.convs = nn.Sequential(*convs)
            self.lstm  = nn.LSTM(
                in_ch, lstm_hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            # n_outputs already includes the CTC blank token (index 0)
            self.fc = nn.Linear(lstm_hidden * 2, n_outputs)

        def forward(self, x):
            # x: (B, T, n_inputs)
            x = x.permute(0, 2, 1)           # (B, C, T)
            x = self.convs(x)
            x = x.permute(0, 2, 1)           # (B, T, lstm_in)
            out, _ = self.lstm(x)
            return self.fc(out)              # (B, T, n_outputs)

    return _CTCModel()


class CTCDecoder(BaseDecoder):
    """
    CNN-BiLSTM + CTC loss for neural sequence decoding.

    No HMM-aligned frame labels required.  Targets are just the character
    sequences (shorter than the neural time dimension).

    Convention: class index 0 is the CTC blank token.
    Character indices should therefore be 1-indexed (1 … n_chars).
    n_outputs = n_chars + 1  (to include blank).

    Args:
        n_inputs:      Number of neural channels.
        n_outputs:     Vocabulary size *including* CTC blank at index 0.
        conv_channels: Output channels for each Conv1D block.
        kernel_size:   Convolutional kernel size.
        lstm_hidden:   LSTM hidden units per direction.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        conv_channels: tuple = (64, 128, 256),
        kernel_size: int = 5,
        lstm_hidden: int = 256,
        dropout: float = 0.2,
        **kwargs: Any,
    ):
        self.n_inputs      = n_inputs
        self.n_outputs     = n_outputs
        self.conv_channels = conv_channels
        self.kernel_size   = kernel_size
        self.lstm_hidden   = lstm_hidden
        self.dropout       = dropout
        self.model         = None

    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> "CTCDecoder":
        """
        Train with CTC loss.

        Args:
            X: (n_trials, T, n_channels) — all trials padded to the same T.
            y: List of 1-D int arrays (character index sequences, 1-indexed).
               Lengths may vary across trials.
            epochs, batch_size, lr: hyper-parameters.
        """
        import torch
        import torch.nn as nn

        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_ctc_model(
            self.n_inputs, self.n_outputs, self.conv_channels,
            self.kernel_size, self.lstm_hidden, self.dropout,
        ).to(device)

        optimizer  = torch.optim.Adam(self.model.parameters(), lr=lr)
        ctc_loss   = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        X_arr = np.asarray(X, dtype=np.float32)
        n, T  = X_arr.shape[0], X_arr.shape[1]

        # Convert targets to list of tensors
        if isinstance(y, np.ndarray) and y.ndim == 2:
            # Padded array; infer lengths from first -1
            target_seqs = []
            for row in y:
                seq = row[row >= 0]
                target_seqs.append(torch.LongTensor(seq))
        else:
            target_seqs = [torch.LongTensor(np.asarray(s, dtype=np.int64)) for s in y]

        X_t = torch.FloatTensor(X_arr)

        self.model.train()
        for epoch in range(epochs):
            perm       = np.random.permutation(n)
            total_loss = 0.0

            for start in range(0, n, batch_size):
                idx  = perm[start: start + batch_size]
                xb   = X_t[idx].to(device)              # (B, T, C)
                B_sz = len(idx)

                # CTC targets: concatenated 1-D tensor
                tgt_batch  = [target_seqs[i] for i in idx]
                tgt_lens   = torch.LongTensor([len(t) for t in tgt_batch])
                tgt_concat = torch.cat(tgt_batch).to(device)

                # Input lengths: all T (no masking — if masked, use real lengths)
                inp_lens = torch.full((B_sz,), T, dtype=torch.long)

                optimizer.zero_grad()
                logits    = self.model(xb)             # (B, T, C)
                log_probs = logits.log_softmax(dim=-1) \
                                  .permute(1, 0, 2)    # (T, B, C) for CTCLoss
                loss = ctc_loss(log_probs, tgt_concat, inp_lens, tgt_lens)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"[CTC] epoch {epoch + 1}/{epochs}  loss={total_loss:.4f}")

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode via greedy CTC: argmax → collapse repeats → strip blanks.

        Args:
            X: (n_trials, T, n_channels)

        Returns:
            log_probs: (n_trials, T, n_outputs) — raw log-softmax outputs.
                       Use `ctc_greedy_decode` to convert to character sequences.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        import torch

        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            X_t    = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
            logits = self.model(X_t)
            out    = logits.log_softmax(dim=-1)
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
                    "lstm_hidden":   self.lstm_hidden,
                    "dropout":       self.dropout,
                },
            },
            path,
        )

    def load(self, path: str) -> "CTCDecoder":
        """Load model weights and config."""
        import torch

        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt["config"]
        self.n_inputs      = cfg["n_inputs"]
        self.n_outputs     = cfg["n_outputs"]
        self.conv_channels = tuple(cfg["conv_channels"])
        self.kernel_size   = cfg["kernel_size"]
        self.lstm_hidden   = cfg["lstm_hidden"]
        self.dropout       = cfg["dropout"]
        self.model = _build_ctc_model(
            self.n_inputs, self.n_outputs, self.conv_channels,
            self.kernel_size, self.lstm_hidden, self.dropout,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self


# ------------------------------------------------------------------
# Utility: greedy CTC decoding
# ------------------------------------------------------------------

def ctc_greedy_decode(log_probs: np.ndarray, blank: int = 0) -> List[List[int]]:
    """
    Greedy CTC decode: argmax → collapse repeats → remove blank.

    Args:
        log_probs: (n_trials, T, n_outputs) log-probability array.
        blank:     Index of the CTC blank token (default 0).

    Returns:
        List of decoded integer sequences (one per trial).
    """
    decoded = []
    for seq in log_probs:
        ids = seq.argmax(axis=-1)          # (T,)
        # Collapse consecutive identical tokens
        collapsed = [ids[0]]
        for tok in ids[1:]:
            if tok != collapsed[-1]:
                collapsed.append(tok)
        # Remove blank
        decoded.append([t for t in collapsed if t != blank])
    return decoded
