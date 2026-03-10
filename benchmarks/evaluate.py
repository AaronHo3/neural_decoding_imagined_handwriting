"""
Evaluation metrics and benchmarking utilities.

Character error rate (CER), word error rate (WER), etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def compute_character_error_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute CER (edit distance / reference length) averaged over sequences.
    
    Args:
        predictions: List of decoded strings
        references: List of ground truth strings
        
    Returns:
        CER as fraction (0-1) or percentage
    """
    raise NotImplementedError("Implement CER (Levenshtein-based)")


def compute_word_error_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute WER (word-level edit distance) averaged over sequences.
    
    Args:
        predictions: List of decoded strings (space-separated words)
        references: List of ground truth strings
        
    Returns:
        WER as fraction (0-1) or percentage
    """
    raise NotImplementedError("Implement WER")


def evaluate_decoder(
    decoder: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    char_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run full evaluation: decode, compute CER/WER, return metrics dict.
    
    Args:
        decoder: Trained decoder with .predict()
        X_test: Test neural data
        y_test: Test labels (indices or one-hot)
        char_list: Character vocabulary for string conversion
        
    Returns:
        {'cer': ..., 'wer': ..., 'raw_accuracy': ...}
    """
    raise NotImplementedError("Implement full evaluation pipeline")


def run_alignment_benchmark(
    alignment_model: Any,
    obs_list: List[np.ndarray],
    sentences: List[str],
    templates: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Benchmark alignment quality (e.g. vs. ground truth character boundaries).
    
    Returns:
        Metrics on alignment accuracy if ground truth available
    """
    raise NotImplementedError("Implement alignment benchmark")
