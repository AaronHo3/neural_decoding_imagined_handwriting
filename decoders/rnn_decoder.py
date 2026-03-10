"""
GRU-based RNN decoder (Willett et al. baseline).

Maps neural sequences to character probabilities via recurrent layers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base_decoder import BaseDecoder


class RNNDecoder(BaseDecoder):
    """
    GRU-based sequence decoder for neural → character mapping.
    
    Baseline architecture following Willett et al. (Nature 2021).
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
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> RNNDecoder:
        """Train GRU decoder on aligned neural-character pairs."""
        raise NotImplementedError("Implement RNN training (e.g. with PyTorch/TensorFlow)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Decode neural sequences to character probabilities."""
        raise NotImplementedError("Implement RNN inference")

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        raise NotImplementedError("Implement model serialization")

    def load(self, path: str) -> RNNDecoder:
        """Load model from checkpoint."""
        raise NotImplementedError("Implement model loading")
