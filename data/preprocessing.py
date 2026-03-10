"""
Preprocessing utilities for neural data.

Time warping, binning, normalization, etc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def bin_spikes(
    spike_counts: np.ndarray,
    bin_size_ms: int = 50,
    fs: Optional[float] = None,
) -> np.ndarray:
    """
    Bin spike counts into fixed-size windows.
    
    Args:
        spike_counts: (T_raw, N) or similar
        bin_size_ms: Bin size in milliseconds
        fs: Sampling rate (Hz) - required if spike_counts is in samples
        
    Returns:
        Binned array
    """
    raise NotImplementedError("Implement binning")


def normalize(
    data: np.ndarray,
    method: str = "zscore",
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize neural data (z-score, min-max, etc.).
    
    Args:
        data: Neural array
        method: 'zscore', 'minmax', 'robust'
        axis: Axis for statistics (default: last)
    """
    raise NotImplementedError("Implement normalization")


def prepare_for_decoder(
    neural_cube: np.ndarray,
    targets: np.ndarray,
    max_len: Optional[int] = None,
    padding_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare (neural, targets) for decoder training.
    
    Returns:
        neural_padded, targets_padded, sequence_lengths
    """
    raise NotImplementedError("Implement decoder input preparation")
