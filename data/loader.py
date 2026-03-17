"""
Willett handwriting BCI dataset loader.

Reference: Willett et al. (Nature 2021) - High-Performance Brain-to-Text Communication via Handwriting
Data: https://doi.org/10.5061/dryad.wh70rxwmv

Expected directory layout (from Dryad download):
    <data_dir>/
        Datasets/
            t5.2019.05.08/
                singleLetters.mat
                sentences.mat (optional)
            t5.2019.11.25/
                ...
        RNNTrainingSteps/
            Step2_HMMLabels/     ← pre-computed Gaussian HMM labels
            Step4_RNNTraining/   ← pre-trained RNN weights
            trainTestPartitions_HeldOutTrials.mat
            trainTestPartitions_HeldOutBlocks.mat

Each .mat file may be MATLAB v7.3 (HDF5) or an older format.
The loader transparently handles both via h5py / scipy.io.

Data format (singleLetters.mat):
    neuralActivityCube_{char} : (n_trials, 201, 192) uint8 per character
    characterCues             : (n_total_trials,) object — char name per trial
    delayPeriodOnsetTimeBin   : (n_total_trials,) — maps trials to timeseries
    blockNumsTimeSeries       : (T_total,) — block ID per timebin
    meansPerBlock             : (n_blocks, 192) — per-block firing rate means
    stdAcrossAllData          : (192,) — global std for z-score
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# Character definitions (from Willett's characterDefinitions.py)
# ======================================================================

CHAR_LIST_FULL = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "greaterThan", "comma", "apostrophe", "tilde", "questionMark",
]

CHAR_ABBREV = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ">", ",", "'", "~", "?",
]

# Mapping from full char name → integer index (0-based)
CHAR_TO_IDX = {name: i for i, name in enumerate(CHAR_LIST_FULL)}


# ======================================================================
# MATLAB file helpers
# ======================================================================

def _load_mat(path: Path) -> Dict[str, Any]:
    """Load a .mat file, handling both legacy and v7.3 (HDF5) formats."""
    import scipy.io

    try:
        data = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        import h5py

        def _h5_to_dict(h5obj, root=None):
            if root is None:
                root = h5obj
            out = {}
            for key in h5obj.keys():
                item = h5obj[key]
                if isinstance(item, h5py.Dataset):
                    out[key] = item[()]
                elif isinstance(item, h5py.Group):
                    out[key] = _h5_to_dict(item, root)
            return out

        with h5py.File(str(path), "r") as f:
            return _h5_to_dict(f)


# ======================================================================
# Dataset class
# ======================================================================

class WillettDataset:
    """
    Loader for the Willett handwriting BCI dataset.

    Each session's singleLetters.mat contains per-character neural activity
    cubes (neuralActivityCube_{char}) with shape (n_trials, 201, 192),
    where 201 = time bins at 10 ms and 192 = electrodes.

    Supports two evaluation partitions:
        HeldOutTrials  : random trials within each block held out
        HeldOutBlocks  : entire recording blocks held out
    """

    def __init__(
        self,
        data_dir: str | Path,
        partition: str = "HeldOutTrials",
    ):
        self.data_dir = Path(data_dir)
        self.partition = partition
        self._sessions_root = self.data_dir / "Datasets"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sessions(self) -> List[str]:
        """Return sorted list of session IDs that have singleLetters.mat."""
        if not self._sessions_root.exists():
            raise FileNotFoundError(
                f"Datasets directory not found: {self._sessions_root}\n"
                "Download the Willett dataset from "
                "https://doi.org/10.5061/dryad.wh70rxwmv and point data_dir "
                "at the extracted root."
            )
        return sorted(
            p.name for p in self._sessions_root.iterdir()
            if p.is_dir() and (p / "singleLetters.mat").exists()
        )

    def load_session(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load neural data and labels for one session.

        Args:
            session_id: e.g. 't5.2019.05.08'. If None, uses first available.

        Returns:
            dict with keys:
                neural        : (n_trials, 201, 192) float32 — spike counts
                char_labels   : (n_trials,) int — 0-indexed character ids
                char_names    : (n_trials,) str — full character names
                char_list     : list[str] — full vocabulary (31 chars)
                char_abbrev   : list[str] — abbreviated vocabulary
                block_ids     : (n_trials,) int — recording block per trial
                means_per_block   : (n_blocks, 192) float32
                std_across_all    : (192,) float32
                session_id        : str
        """
        if session_id is None:
            session_id = self.list_sessions()[0]

        mat_path = self._sessions_root / session_id / "singleLetters.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Session file not found: {mat_path}")

        raw = _load_mat(mat_path)

        # --- Assemble neural cubes and labels from per-character keys ---
        char_cues = np.array(raw["characterCues"])  # (n_trials,) object
        n_total = len(char_cues)

        # Gather all per-character cubes
        cube_keys = sorted(
            k for k in raw if k.startswith("neuralActivityCube_")
        )

        # Build lookup: char_name → cube
        char_cubes: Dict[str, np.ndarray] = {}
        for key in cube_keys:
            char_name = key.replace("neuralActivityCube_", "")
            char_cubes[char_name] = np.array(raw[key], dtype=np.float32)

        # Stack trials in the original cue order
        # Track per-char trial counter
        char_counters: Dict[str, int] = {c: 0 for c in char_cubes}
        neural_list = []
        label_list = []
        name_list = []

        for cue in char_cues:
            cue_str = str(cue)
            if cue_str in char_cubes:
                idx = char_counters[cue_str]
                neural_list.append(char_cubes[cue_str][idx])
                char_counters[cue_str] = idx + 1
                label_list.append(CHAR_TO_IDX.get(cue_str, -1))
                name_list.append(cue_str)
            else:
                # Unknown character — skip or use zeros
                neural_list.append(np.zeros((201, 192), dtype=np.float32))
                label_list.append(-1)
                name_list.append(cue_str)

        neural = np.stack(neural_list, axis=0)        # (n_trials, 201, 192)
        char_labels = np.array(label_list, dtype=int)  # (n_trials,)

        # --- Block IDs per trial ---
        block_nums_ts = np.array(raw.get("blockNumsTimeSeries",
                                          np.zeros(1, dtype=int)))
        delay_bins = np.array(raw.get("delayPeriodOnsetTimeBin",
                                       np.zeros(n_total, dtype=int)), dtype=int)
        # Map each trial to its block by looking up the block at its delay onset
        if len(block_nums_ts) > 1 and delay_bins.max() < len(block_nums_ts):
            block_ids = block_nums_ts[delay_bins].astype(int)
        else:
            block_ids = np.zeros(n_total, dtype=int)

        means_per_block = np.array(
            raw.get("meansPerBlock", np.zeros((1, 192))), dtype=np.float32
        )
        std_across_all = np.array(
            raw.get("stdAcrossAllData", np.ones(192)), dtype=np.float32
        ).ravel()

        return {
            "neural": neural,
            "char_labels": char_labels,
            "char_names": np.array(name_list),
            "char_list": CHAR_LIST_FULL,
            "char_abbrev": CHAR_ABBREV,
            "block_ids": block_ids,
            "means_per_block": means_per_block,
            "std_across_all": std_across_all,
            "session_id": session_id,
        }

    def get_train_test_split(
        self,
        session_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X_train, y_train, X_test, y_test) for the selected partition.

        X arrays have shape (n_trials, 201, 192).
        y arrays have shape (n_trials,) with 0-indexed character labels.

        Split logic (mirrors Willett):
            HeldOutTrials  : last 20% of trials per block
            HeldOutBlocks  : last block in session
        """
        session = self.load_session(session_id)
        neural = session["neural"]
        labels = session["char_labels"]
        block_ids = session["block_ids"]

        # Remove "doNothing" trials (label -1 or explicit)
        mask = labels >= 0
        neural = neural[mask]
        labels = labels[mask]
        block_ids = block_ids[mask]

        if self.partition == "HeldOutTrials":
            train_idx, test_idx = self._split_held_out_trials(block_ids)
        elif self.partition == "HeldOutBlocks":
            train_idx, test_idx = self._split_held_out_blocks(block_ids)
        else:
            raise ValueError(f"Unknown partition: {self.partition!r}")

        return (
            neural[train_idx],
            labels[train_idx],
            neural[test_idx],
            labels[test_idx],
        )

    def get_time_warped_cubes(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return per-character neural cubes for HMM template fitting.

        Returns:
            {char_name: ndarray (n_trials, 201, 192)} for each character.
            Excludes 'doNothing'.
        """
        if session_id is None:
            session_id = self.list_sessions()[0]

        mat_path = self._sessions_root / session_id / "singleLetters.mat"
        raw = _load_mat(mat_path)

        cubes = {}
        for key in raw:
            if not key.startswith("neuralActivityCube_"):
                continue
            char_name = key.replace("neuralActivityCube_", "")
            if char_name == "doNothing":
                continue
            cubes[char_name] = np.array(raw[key], dtype=np.float32)

        return cubes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_held_out_trials(
        block_ids: np.ndarray,
        test_fraction: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Hold out the last `test_fraction` of trials within each block."""
        train_idx, test_idx = [], []
        for blk in np.unique(block_ids):
            idx = np.where(block_ids == blk)[0]
            n_test = max(1, int(len(idx) * test_fraction))
            test_idx.extend(idx[-n_test:])
            train_idx.extend(idx[:-n_test])
        return np.array(train_idx), np.array(test_idx)

    @staticmethod
    def _split_held_out_blocks(
        block_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Hold out all trials from the last recording block."""
        unique_blocks = np.unique(block_ids)
        last_block = unique_blocks[-1]
        test_idx = np.where(block_ids == last_block)[0]
        train_idx = np.where(block_ids != last_block)[0]
        return train_idx, test_idx
