"""
CTC decoder: CNN-BiLSTM with Connectionist Temporal Classification.

Alignment-free training inspired by offline handwriting OCR.
No explicit character boundaries required during training.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_decoder import BaseDecoder


class CTCDecoder(BaseDecoder):
    """
    CNN-BiLSTM + CTC loss for neural sequence decoding.
    
    Learns alignment implicitly via CTC. No HMM-aligned labels
    strictly required for training (just character sequences).
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
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> CTCDecoder:
        """
        Train with CTC loss. y can be character sequences (no alignment needed).
        """
        raise NotImplementedError("Implement CTC training")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Decode via CTC beam search or greedy decoding."""
        raise NotImplementedError("Implement CTC inference")

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        raise NotImplementedError("Implement model serialization")

    def load(self, path: str) -> CTCDecoder:
        """Load model from checkpoint."""
        raise NotImplementedError("Implement model loading")
