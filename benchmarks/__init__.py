"""
Benchmarks module: Evaluation and comparison of alignment + decoder models.
"""

from .evaluate import (
    evaluate_decoder,
    compute_character_error_rate,
    compute_word_error_rate,
    run_alignment_benchmark,
)
from .compare import run_decoder_comparison, run_alignment_comparison

__all__ = [
    "evaluate_decoder",
    "compute_character_error_rate",
    "compute_word_error_rate",
    "run_alignment_benchmark",
    "run_decoder_comparison",
    "run_alignment_comparison",
]
