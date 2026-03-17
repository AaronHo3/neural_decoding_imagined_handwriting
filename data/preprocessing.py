"""
Preprocessing utilities for neural data.

Time warping, binning, normalization, etc.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


def bin_spikes(
    spike_counts: np.ndarray,
    bin_size_ms: int = 50,
    fs: Optional[float] = None,
) -> np.ndarray:
    """
    Bin spike counts into fixed-size windows by summing.

    Args:
        spike_counts: (T_raw, N) array of spike counts
        bin_size_ms: Bin size in milliseconds (or in samples if fs is None)
        fs: Sampling rate in Hz. If provided, bin_size_ms is treated as ms.
            If None, bin_size_ms is treated as number of samples per bin.

    Returns:
        Binned array of shape (n_bins, N)
    """
    arr = np.asarray(spike_counts, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    if fs is not None:
        samples_per_bin = max(1, int(round(bin_size_ms * fs / 1000.0)))
    else:
        samples_per_bin = max(1, int(bin_size_ms))

    T = arr.shape[0]
    n_bins = T // samples_per_bin
    arr = arr[: n_bins * samples_per_bin]

    # (n_bins, samples_per_bin, N) → sum over axis 1
    shape = (n_bins, samples_per_bin) + arr.shape[1:]
    return arr.reshape(shape).sum(axis=1)


def normalize(
    data: np.ndarray,
    method: str = "zscore",
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize neural data channel-wise or globally.

    Args:
        data: Neural array of any shape
        method: 'zscore', 'minmax', or 'robust'
        axis: Axis along which statistics are computed.
              Defaults to 0 (across time/trials).

    Returns:
        Normalized array of same shape.
    """
    data = np.asarray(data, dtype=float)
    if axis is None:
        axis = 0

    if method == "zscore":
        mean = data.mean(axis=axis, keepdims=True)
        std = data.std(axis=axis, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (data - mean) / std

    elif method == "minmax":
        mn = data.min(axis=axis, keepdims=True)
        mx = data.max(axis=axis, keepdims=True)
        rng = np.where(mx - mn == 0, 1.0, mx - mn)
        return (data - mn) / rng

    elif method == "robust":
        median = np.median(data, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        iqr = np.where(q75 - q25 == 0, 1.0, q75 - q25)
        return (data - median) / iqr

    else:
        raise ValueError(f"Unknown normalization method: {method!r}. "
                         f"Choose from 'zscore', 'minmax', 'robust'.")


def prepare_for_decoder(
    neural_cube: Union[np.ndarray, List[np.ndarray]],
    targets: Union[np.ndarray, List],
    max_len: Optional[int] = None,
    padding_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad variable-length (neural, target) pairs for batch decoder training.

    Args:
        neural_cube: Either a 3-D array (n_trials, T, n_channels) where all
                     trials already have the same length, or a list of 2-D
                     arrays (T_i, n_channels) with varying lengths.
        targets:     Either a 2-D array (n_trials, S) or a list of 1-D
                     integer arrays with varying target lengths.
        max_len:     Pad neural sequences to this length (defaults to the
                     longest sequence).
        padding_value: Fill value for neural padding (default 0.0).
                       Targets are padded with -1.

    Returns:
        neural_padded : (n_trials, max_len, n_channels)
        targets_padded: (n_trials, max_target_len) — int, -1 for padding
        seq_lengths   : (n_trials,) — original neural sequence lengths
    """
    # --- Normalize neural_cube to a list of 2-D arrays ---
    if isinstance(neural_cube, np.ndarray) and neural_cube.ndim == 3:
        sequences = [neural_cube[i] for i in range(len(neural_cube))]
    else:
        sequences = list(neural_cube)

    seq_lengths = np.array([s.shape[0] for s in sequences], dtype=int)
    n = len(sequences)
    n_channels = sequences[0].shape[1] if sequences[0].ndim > 1 else 1

    if max_len is None:
        max_len = int(seq_lengths.max())

    neural_padded = np.full((n, max_len, n_channels), padding_value, dtype=float)
    for i, seq in enumerate(sequences):
        L = min(seq.shape[0], max_len)
        neural_padded[i, :L] = seq[:L]

    # --- Normalize targets ---
    if isinstance(targets, np.ndarray) and targets.ndim == 2:
        targets_padded = targets.astype(int)
    else:
        target_list = [np.asarray(t, dtype=int) for t in targets]
        max_t = max(len(t) for t in target_list)
        targets_padded = np.full((n, max_t), -1, dtype=int)
        for i, t in enumerate(target_list):
            targets_padded[i, : len(t)] = t

    return neural_padded, targets_padded, seq_lengths
