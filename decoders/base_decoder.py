"""
Base interface for sequence decoders.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseDecoder(ABC):
    """Abstract base class for neural sequence decoders."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> "BaseDecoder":
        """
        Train the decoder on aligned neural data.
        
        Args:
            X: Neural data (trials, time, channels) or (trials, time, channels) list
            y: Target character sequences (one-hot or indices)
            **kwargs: Additional training arguments (batch size, epochs, etc.)
            
        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode neural data to character sequences.
        
        Args:
            X: Neural data (trials, time, channels)
            
        Returns:
            Predictions: character indices or probabilities (batch, time, n_chars)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseDecoder":
        """Load model from disk."""
        pass
