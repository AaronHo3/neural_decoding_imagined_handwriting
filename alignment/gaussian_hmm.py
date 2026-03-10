"""
Gaussian HMM for forced alignment of neural data to character sequences.

Baseline implementation following Willett et al. (Nature 2021).
Uses Gaussian emission distributions (mean firing rates + diagonal covariance).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class GaussianHMMForcedAlignment:
    """
    Forced-alignment HMM with Gaussian emissions.
    
    Each state corresponds to a piece of a character. Emission probabilities
    use templates (mean firing rates) with diagonal covariance.
    """

    def __init__(
        self,
        hmm_bin_size: int = 5,
        blank_prob: float = 0.1,
        stay_prob: float = 0.2,
        skip_prob: float = 0.2,
    ):
        self.hmm_bin_size = hmm_bin_size
        self.blank_prob = blank_prob
        self.stay_prob = stay_prob
        self.skip_prob = skip_prob

    def fit_templates(self, tw_cubes: Dict[str, np.ndarray], char_def: Dict) -> Dict[str, np.ndarray]:
        """
        Initialize character templates from time-warped neural data.
        
        Args:
            tw_cubes: Time-warped data cubes per character
            char_def: Character definitions (names, lengths)
            
        Returns:
            templates: Mean firing rate template per character
        """
        raise NotImplementedError("Implement template initialization from time-warped data")

    def align(
        self,
        obs: np.ndarray,
        sentence: str,
        templates: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Label neural data with character start times and durations via Viterbi.
        
        Args:
            obs: Neural activity matrix (T x N)
            sentence: Target sentence string
            templates: Character templates
            
        Returns:
            letter_starts: Start time per character
            letter_durations: Duration per character
        """
        raise NotImplementedError("Implement Gaussian HMM forced alignment")
