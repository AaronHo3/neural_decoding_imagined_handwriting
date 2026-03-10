"""
Poisson HMM for forced alignment of neural data to character sequences.

Principled count-based emissions for neural spike data.
Differentiates from Willett's Gaussian HMM by using Poisson (or negative binomial)
emission distributions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class PoissonHMMForcedAlignment:
    """
    Forced-alignment HMM with Poisson emissions.
    
    Neural spike counts are modeled as P(k | λ) where λ is the expected
    firing rate for each state. More appropriate for count data than Gaussian.
    """

    def __init__(
        self,
        hmm_bin_size: int = 5,
        blank_prob: float = 0.1,
        stay_prob: float = 0.2,
        skip_prob: float = 0.2,
        use_negative_binomial: bool = False,
    ):
        self.hmm_bin_size = hmm_bin_size
        self.blank_prob = blank_prob
        self.stay_prob = stay_prob
        self.skip_prob = skip_prob
        self.use_negative_binomial = use_negative_binomial

    def fit_templates(self, tw_cubes: Dict[str, np.ndarray], char_def: Dict) -> Dict[str, np.ndarray]:
        """
        Initialize character templates (expected firing rates) from time-warped data.
        
        Args:
            tw_cubes: Time-warped data cubes per character
            char_def: Character definitions (names, lengths)
            
        Returns:
            templates: Expected rate template per character (for Poisson λ)
        """
        raise NotImplementedError("Implement template initialization for Poisson emissions")

    def align(
        self,
        obs: np.ndarray,
        sentence: str,
        templates: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Label neural data with character start times and durations via Viterbi.
        Uses Poisson (or negative binomial) emission likelihoods.
        
        Args:
            obs: Neural activity matrix (T x N) - binned spike counts
            sentence: Target sentence string
            templates: Character rate templates
            
        Returns:
            letter_starts: Start time per character
            letter_durations: Duration per character
        """
        raise NotImplementedError("Implement Poisson HMM forced alignment")
