"""
Willett handwriting BCI dataset loader.

Reference: Willett et al. (Nature 2021) - High-Performance Brain-to-Text Communication via Handwriting
Data: https://doi.org/10.5061/dryad.wh70rxwmv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class WillettDataset:
    """
    Loader for Willett handwriting BCI dataset.
    
    Provides access to neural recordings, character labels, and train/test splits
    (HeldOutTrials, HeldOutBlocks).
    """

    def __init__(
        self,
        data_dir: str | Path,
        partition: str = "HeldOutTrials",
    ):
        """
        Args:
            data_dir: Path to extracted dataset (e.g. from Dryad)
            partition: 'HeldOutTrials' or 'HeldOutBlocks'
        """
        self.data_dir = Path(data_dir)
        self.partition = partition

    def load_session(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load neural data and labels for a session.
        
        Returns:
            dict with keys: neural_data, labels, sentences, cv_idx, etc.
        """
        raise NotImplementedError("Implement .mat/.pkl loading from Willett data format")

    def get_train_test_split(
        self,
        session_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test indices for the selected partition.
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        raise NotImplementedError("Implement split loading from dataset")

    def list_sessions(self) -> List[str]:
        """List available session IDs."""
        raise NotImplementedError("Implement session enumeration")
