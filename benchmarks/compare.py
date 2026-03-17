"""
Comparative benchmarks: Gaussian vs Poisson HMM, RNN vs RCNN vs CTC.

Runs ablation studies and generates comparison tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def run_decoder_comparison(
    data_dir: str,
    decoders: List[Tuple[str, Any]],
    partition: str = "HeldOutTrials",
    alignment_model: Optional[Any] = None,
    char_list: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    **fit_kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate multiple decoders on the Willett dataset.

    Each decoder in `decoders` is trained on the same train split and
    evaluated on the same test split.  Frame-level labels are derived from
    forced alignment (pass an already-fitted `alignment_model`), or from a
    simple majority-vote across time bins if no aligner is provided.

    Args:
        data_dir:        Path to the extracted Dryad dataset root.
        decoders:        [(name, decoder_instance), ...] — each instance must
                         implement BaseDecoder (fit / predict).
        partition:       'HeldOutTrials' or 'HeldOutBlocks'.
        alignment_model: Fitted HMM aligner (GaussianHMM / PoissonHMM) used to
                         generate frame-level labels for RNN / RCNN decoders.
                         If None, the raw character cue is broadcast across
                         all time bins (crude but functional baseline).
        char_list:       Vocabulary list (defaults to a–z + space).
        output_dir:      If provided, write a results .csv here.
        session_id:      Session to use (None → first available).
        **fit_kwargs:    Forwarded to each decoder's fit() call
                         (e.g. epochs=100, batch_size=32).

    Returns:
        {decoder_name: {'cer': float, 'wer': float, 'raw_accuracy': float}}
    """
    from data.loader import WillettDataset
    from data.preprocessing import normalize
    from benchmarks.evaluate import evaluate_decoder

    if char_list is None:
        char_list = list("abcdefghijklmnopqrstuvwxyz") + [" "]

    dataset = WillettDataset(data_dir, partition=partition)
    X_train, y_train_char, X_test, y_test_char = dataset.get_train_test_split(session_id)

    # Z-score normalise across training data, apply same stats to test
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True)
    std  = np.where(std == 0, 1.0, std)
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    T = X_train.shape[1]

    # Build frame-level labels (n_trials, T)
    def _make_frame_labels(char_labels: np.ndarray, T: int) -> np.ndarray:
        if alignment_model is not None:
            raise NotImplementedError(
                "Pass pre-computed frame labels; run alignment separately."
            )
        # Broadcast single character label across all time bins
        return np.tile(char_labels[:, np.newaxis], (1, T))

    y_train_frames = _make_frame_labels(y_train_char, T)
    y_test_frames  = _make_frame_labels(y_test_char,  T)

    results: Dict[str, Dict[str, float]] = {}

    for name, dec in decoders:
        from decoders.ctc_decoder import CTCDecoder
        is_ctc = isinstance(dec, CTCDecoder)

        print(f"\n{'='*60}")
        print(f"Training decoder: {name}")
        print(f"{'='*60}")

        if is_ctc:
            # CTC needs character sequences, not frame labels
            y_tr = [np.array([lbl + 1]) for lbl in y_train_char]  # 1-indexed
        else:
            y_tr = y_train_frames

        dec.fit(X_train, y_tr, **fit_kwargs)

        if is_ctc:
            y_te = y_test_frames  # still used for reference string construction
        else:
            y_te = y_test_frames

        metrics = evaluate_decoder(dec, X_test, y_te,
                                   char_list=char_list, is_ctc=is_ctc)
        results[name] = metrics
        print(f"  CER={metrics['cer']:.4f}  WER={metrics['wer']:.4f}  "
              f"Acc={metrics['raw_accuracy']:.4f}")

    if output_dir is not None:
        _save_results_csv(results, Path(output_dir) / "decoder_comparison.csv")

    return results


def run_alignment_comparison(
    data_dir: str,
    alignment_models: List[Tuple[str, Any]],
    decoder: Any,
    partition: str = "HeldOutTrials",
    char_list: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    **fit_kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    """
    Compare different alignment methods coupled with the same decoder.

    Each alignment model is fitted, used to generate frame-level labels,
    and then the shared `decoder` is re-trained from scratch for each one.

    Args:
        data_dir:          Path to the Dryad dataset root.
        alignment_models:  [(name, fitted_hmm), ...] — each must have a
                           .align(obs, sentence, templates) method and
                           .templates_ attribute.
        decoder:           A decoder class *instance* that will be re-fitted
                           for each alignment method.  Must implement BaseDecoder.
        partition:         'HeldOutTrials' or 'HeldOutBlocks'.
        char_list:         Vocabulary list.
        session_id:        Session to use.
        **fit_kwargs:      Forwarded to decoder.fit().

    Returns:
        {alignment_name: {'cer': float, 'wer': float, 'raw_accuracy': float}}
    """
    import copy
    from data.loader import WillettDataset
    from benchmarks.evaluate import evaluate_decoder

    if char_list is None:
        char_list = list("abcdefghijklmnopqrstuvwxyz") + [" "]

    dataset = WillettDataset(data_dir, partition=partition)
    session = dataset.load_session(session_id)
    X_train, y_train_char, X_test, y_test_char = dataset.get_train_test_split(session_id)

    # Normalise
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True)
    std  = np.where(std == 0, 1.0, std)
    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    T = X_train_n.shape[1]

    results: Dict[str, Dict[str, float]] = {}

    for name, aligner in alignment_models:
        print(f"\n{'='*60}")
        print(f"Alignment method: {name}")
        print(f"{'='*60}")

        templates = getattr(aligner, "templates_", {})
        if not templates:
            raise ValueError(
                f"Alignment model '{name}' has no fitted templates. "
                "Call fit_templates() before run_alignment_comparison()."
            )

        # Build per-trial frame labels via .align()
        # Here sentences are single characters (singleLetters protocol)
        y_train_frames = np.zeros((len(X_train_n), T), dtype=int)
        for trial_idx, (obs, char_idx) in enumerate(zip(X_train_n, y_train_char)):
            char = char_list[char_idx] if char_idx < len(char_list) else "?"
            starts, durations = aligner.align(obs, char, templates)
            # Fill frame labels with char_idx across the aligned span
            t0 = int(starts[0])
            t1 = min(t0 + int(durations[0]), T)
            y_train_frames[trial_idx, t0:t1] = char_idx
            # Mark unaligned positions as padding
            y_train_frames[trial_idx, :t0]  = -1
            y_train_frames[trial_idx, t1:]  = -1

        y_test_frames = np.full((len(X_test_n), T), -1, dtype=int)
        for trial_idx, (obs, char_idx) in enumerate(zip(X_test_n, y_test_char)):
            char = char_list[char_idx] if char_idx < len(char_list) else "?"
            starts, durations = aligner.align(obs, char, templates)
            t0 = int(starts[0])
            t1 = min(t0 + int(durations[0]), T)
            y_test_frames[trial_idx, t0:t1] = char_idx

        # Re-fit decoder from scratch for this alignment
        dec = copy.deepcopy(decoder)
        dec.fit(X_train_n, y_train_frames, **fit_kwargs)

        metrics = evaluate_decoder(dec, X_test_n, y_test_frames,
                                   char_list=char_list, is_ctc=False)
        results[name] = metrics
        print(f"  CER={metrics['cer']:.4f}  WER={metrics['wer']:.4f}  "
              f"Acc={metrics['raw_accuracy']:.4f}")

    return results


# ======================================================================
# Internal helpers
# ======================================================================

def _save_results_csv(
    results: Dict[str, Dict[str, float]],
    path: Path,
) -> None:
    """Write a simple CSV of metric results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [["model", "cer", "wer", "raw_accuracy"]]
    for name, m in results.items():
        rows.append([name,
                     f"{m.get('cer', float('nan')):.6f}",
                     f"{m.get('wer', float('nan')):.6f}",
                     f"{m.get('raw_accuracy', float('nan')):.6f}"])
    with open(path, "w") as f:
        f.write("\n".join(",".join(r) for r in rows) + "\n")
    print(f"Results saved to {path}")
