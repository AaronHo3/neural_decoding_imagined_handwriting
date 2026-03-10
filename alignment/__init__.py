"""
Alignment module: HMM variants for forced alignment of neural data to character boundaries.

Based on Willett et al. (Nature 2021) - https://github.com/fwillett/handwritingBCI
"""

from .gaussian_hmm import GaussianHMMForcedAlignment
from .poisson_hmm import PoissonHMMForcedAlignment

__all__ = ["GaussianHMMForcedAlignment", "PoissonHMMForcedAlignment"]
