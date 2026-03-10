"""
Comparative benchmarks: Gaussian vs Poisson HMM, RNN vs RCNN vs CTC.

Runs ablation studies and generates comparison tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def run_decoder_comparison(
    data_dir: str,
    decoders: List[tuple[str, Any]],
    partition: str = "HeldOutTrials",
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate multiple decoders, return metrics table.
    
    Args:
        data_dir: Path to Willett dataset
        decoders: [(name, decoder_instance), ...]
        partition: Train/test partition
        output_dir: Optional path to save results
        
    Returns:
        {decoder_name: {'cer': ..., 'wer': ...}}
    """
    raise NotImplementedError("Implement multi-decoder comparison")


def run_alignment_comparison(
    data_dir: str,
    alignment_models: List[tuple[str, Any]],
    decoder: Any,
    partition: str = "HeldOutTrials",
) -> Dict[str, Dict[str, float]]:
    """
    Compare different alignment methods with same decoder.
    
    Returns:
        {alignment_name: {'cer': ..., 'wer': ...}}
    """
    raise NotImplementedError("Implement alignment ablation")
