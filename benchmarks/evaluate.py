"""
Evaluation metrics and benchmarking utilities.

Character error rate (CER), word error rate (WER), etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ======================================================================
# Core string-distance metrics
# ======================================================================

def _levenshtein(a: List, b: List) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(a), len(b)
    # Use two rolling rows for O(n) memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def compute_character_error_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute mean CER = edit_distance(pred, ref) / len(ref), averaged over sequences.

    Empty references contribute 0 if the prediction is also empty, else 1.

    Args:
        predictions: List of decoded strings.
        references:  List of ground-truth strings.

    Returns:
        CER as a fraction in [0, ∞).  A value < 1 means fewer errors than chars.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    total = 0.0
    for pred, ref in zip(predictions, references):
        ref_len = len(ref)
        if ref_len == 0:
            total += 0.0 if len(pred) == 0 else 1.0
        else:
            total += _levenshtein(list(pred), list(ref)) / ref_len

    return total / max(len(predictions), 1)


def compute_word_error_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute mean WER = edit_distance(pred_words, ref_words) / len(ref_words).

    Args:
        predictions: Decoded strings (space-separated words).
        references:  Ground-truth strings.

    Returns:
        WER as a fraction.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    total = 0.0
    for pred, ref in zip(predictions, references):
        ref_words  = ref.split()
        pred_words = pred.split()
        ref_len    = len(ref_words)
        if ref_len == 0:
            total += 0.0 if len(pred_words) == 0 else 1.0
        else:
            total += _levenshtein(pred_words, ref_words) / ref_len

    return total / max(len(predictions), 1)


# ======================================================================
# Decoder evaluation
# ======================================================================

def _logits_to_strings(
    raw_output: np.ndarray,
    char_list: List[str],
    is_ctc: bool = False,
    blank: int = 0,
) -> List[str]:
    """
    Convert decoder output arrays to strings.

    For non-CTC decoders (RNN/RCNN): raw_output is (B, T, C) logits.
    Argmax per timestep → strip padding sentinel (-1) → join chars.

    For CTC decoder: raw_output is (B, T, C) log-probs.
    Greedy CTC decode (collapse + remove blank) → join chars.
    """
    strings = []
    for seq in raw_output:
        ids = seq.argmax(axis=-1)  # (T,)

        if is_ctc:
            # Collapse repeated tokens, then remove blank
            collapsed = [ids[0]]
            for tok in ids[1:]:
                if tok != collapsed[-1]:
                    collapsed.append(tok)
            ids = [t for t in collapsed if t != blank]
        else:
            # Remove padding and de-duplicate consecutive identical chars
            # (for frame-level outputs, argmax of each frame is the label)
            ids = [t for t in ids if 0 <= t < len(char_list)]

        chars = [char_list[t] for t in ids if 0 <= t < len(char_list)]
        strings.append("".join(chars))

    return strings


def _indices_to_strings(
    y: np.ndarray,
    char_list: List[str],
) -> List[str]:
    """Convert (B, T) label arrays to strings (skip -1 padding)."""
    strings = []
    for row in y:
        chars = [char_list[t] for t in row if 0 <= t < len(char_list)]
        strings.append("".join(chars))
    return strings


def evaluate_decoder(
    decoder: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    char_list: Optional[List[str]] = None,
    is_ctc: bool = False,
) -> Dict[str, float]:
    """
    Run full evaluation: decode X_test, compute CER / WER.

    Args:
        decoder:    Trained decoder with a .predict(X) method.
        X_test:     (n_trials, T, n_channels) neural test data.
        y_test:     (n_trials, T) or (n_trials, S) label array.
                    -1 values are treated as padding.
        char_list:  Vocabulary list indexed by label.  Defaults to
                    26 lower-case letters + space.
        is_ctc:     Set True for CTCDecoder outputs (enables greedy CTC decode).

    Returns:
        {'cer': float, 'wer': float, 'raw_accuracy': float}
    """
    if char_list is None:
        char_list = list("abcdefghijklmnopqrstuvwxyz") + [" "]

    raw_output   = decoder.predict(X_test)
    predictions  = _logits_to_strings(raw_output, char_list, is_ctc=is_ctc)
    references   = _indices_to_strings(np.asarray(y_test), char_list)

    cer = compute_character_error_rate(predictions, references)
    wer = compute_word_error_rate(predictions, references)

    # Raw token accuracy (per timestep, non-padded, non-CTC)
    if not is_ctc:
        pred_ids   = raw_output.argmax(axis=-1)   # (B, T)
        y_arr      = np.asarray(y_test)
        valid_mask = y_arr >= 0
        if valid_mask.any():
            raw_acc = float((pred_ids[valid_mask] == y_arr[valid_mask]).mean())
        else:
            raw_acc = 0.0
    else:
        raw_acc = 1.0 - cer  # approximate

    return {"cer": cer, "wer": wer, "raw_accuracy": raw_acc}


# ======================================================================
# Alignment benchmark
# ======================================================================

def run_alignment_benchmark(
    alignment_model: Any,
    obs_list: List[np.ndarray],
    sentences: List[str],
    templates: Dict[str, np.ndarray],
    gt_starts: Optional[List[np.ndarray]] = None,
    gt_durations: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Benchmark alignment quality against optional ground-truth boundaries.

    For each trial, the alignment model's .align() method is called.
    If ground-truth starts / durations are provided, mean absolute error
    (MAE) is reported.  Otherwise, statistics about the produced alignment
    (mean duration, std) are returned.

    Args:
        alignment_model: Fitted model with .align(obs, sentence, templates).
        obs_list:        List of (T, N) neural observation arrays.
        sentences:       Corresponding target sentences.
        templates:       Character templates (from fit_templates).
        gt_starts:       Optional list of ground-truth start arrays.
        gt_durations:    Optional list of ground-truth duration arrays.

    Returns:
        Metrics dict, e.g. {'start_mae': ..., 'duration_mae': ..., 'n_trials': ...}
    """
    pred_starts    = []
    pred_durations = []

    for obs, sentence in zip(obs_list, sentences):
        s, d = alignment_model.align(obs, sentence, templates)
        pred_starts.append(s)
        pred_durations.append(d)

    metrics: Dict[str, float] = {"n_trials": float(len(obs_list))}

    if gt_starts is not None:
        start_maes = [
            float(np.abs(ps.astype(float) - gs.astype(float)).mean())
            for ps, gs in zip(pred_starts, gt_starts)
            if len(ps) == len(gs)
        ]
        metrics["start_mae"] = float(np.mean(start_maes)) if start_maes else float("nan")

    if gt_durations is not None:
        dur_maes = [
            float(np.abs(pd_.astype(float) - gd.astype(float)).mean())
            for pd_, gd in zip(pred_durations, gt_durations)
            if len(pd_) == len(gd)
        ]
        metrics["duration_mae"] = float(np.mean(dur_maes)) if dur_maes else float("nan")

    # Always report mean / std of predicted durations
    all_durs = np.concatenate([d.astype(float) for d in pred_durations]) \
               if pred_durations else np.array([0.0])
    metrics["mean_duration"] = float(all_durs.mean())
    metrics["std_duration"]  = float(all_durs.std())

    return metrics
