"""
Data module: Loading and preprocessing for Willett handwriting BCI dataset.

Dataset: https://doi.org/10.5061/dryad.wh70rxwmv
"""

from .loader import WillettDataset

__all__ = ["WillettDataset"]
