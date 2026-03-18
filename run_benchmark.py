#!/usr/bin/env python3
"""
Full benchmark: compare alignment methods × decoder architectures.

Loads sentence data from the Willett dataset, generates frame-level labels
using both Willett's pre-computed Gaussian HMM and our Poisson HMM, then
trains and evaluates all decoders.

Usage:
    python3 run_benchmark.py                        # default (1 session, fast)
    python3 run_benchmark.py --session t5.2019.11.25
    python3 run_benchmark.py --full                 # all sessions, more epochs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Character definitions
# ---------------------------------------------------------------------------

from data.loader import CHAR_LIST_FULL, CHAR_ABBREV, CHAR_TO_IDX, _load_mat

# Map from abbreviated char → full name (for sentence parsing)
_ABBREV_TO_FULL = {a: f for a, f in zip(CHAR_ABBREV, CHAR_LIST_FULL)}


def _sentence_to_char_indices(text: str) -> List[int]:
    """Convert sentence text like 'hello>world~' to list of char indices."""
    indices = []
    for c in text:
        full = _ABBREV_TO_FULL.get(c, c)
        idx = CHAR_TO_IDX.get(full, -1)
        if idx >= 0:
            indices.append(idx)
    return indices


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sentence_data(
    data_dir: Path,
    session: str,
    partition: str,
    max_len: int = 0,
) -> Dict:
    """
    Load sentence neural data, Willett's HMM labels, and train/test partition.

    Returns dict with keys:
        neural       : (n_sentences, T_padded, 192) float32
        seq_lens     : (n_sentences,) int — actual timesteps per sentence
        gauss_labels : (n_sentences, T_padded) int — hard frame labels from Gaussian HMM
        ignore_mask  : (n_sentences, T_padded) bool — True where loss should be ignored
        sentences    : list[str] — prompt text
        char_seqs    : list[list[int]] — character index sequences per sentence
        train_idx    : ndarray — sentence indices for training
        test_idx     : ndarray — sentence indices for test
        std_all      : (192,) — global std for z-scoring
        means_block  : (n_blocks, 192) — per-block means
    """
    # --- Sentence neural data ---
    sent_raw = _load_mat(data_dir / "Datasets" / session / "sentences.mat")
    neural = np.array(sent_raw["neuralActivityCube"], dtype=np.float32)  # (N, T, 192)
    seq_lens = np.array(sent_raw["numTimeBinsPerSentence"], dtype=int)
    prompts = list(sent_raw["sentencePrompt"])

    # --- Willett's pre-computed Gaussian HMM labels ---
    hmm_path = (data_dir / "RNNTrainingSteps" / "Step2_HMMLabels" /
                partition / f"{session}_timeSeriesLabels.mat")
    hmm_raw = _load_mat(hmm_path)
    char_prob_target = np.array(hmm_raw["charProbTarget"], dtype=np.float32)
    ignore_error = np.array(hmm_raw["ignoreErrorHere"], dtype=np.float32)

    gauss_labels = char_prob_target.argmax(axis=-1).astype(int)  # (N, T)
    ignore_mask = ignore_error > 0.5  # True = ignore in loss

    # --- Train/test partition ---
    part_path = data_dir / "RNNTrainingSteps" / f"trainTestPartitions_{partition}.mat"
    part_raw = _load_mat(part_path)
    train_idx = np.array(part_raw[f"{session}_train"], dtype=int).ravel()
    test_idx = np.array(part_raw[f"{session}_test"], dtype=int).ravel()

    # --- Normalization stats from singleLetters ---
    sl_raw = _load_mat(data_dir / "Datasets" / session / "singleLetters.mat")
    std_all = np.array(sl_raw.get("stdAcrossAllData", np.ones(192)),
                       dtype=np.float32).ravel()
    means_block = np.array(sl_raw.get("meansPerBlock", np.zeros((1, 192))),
                           dtype=np.float32)

    # --- Char sequences for CTC ---
    char_seqs = [_sentence_to_char_indices(str(p)) for p in prompts]

    # --- Optional truncation ---
    if max_len > 0 and neural.shape[1] > max_len:
        neural = neural[:, :max_len, :]
        gauss_labels = gauss_labels[:, :max_len]
        char_prob_target = char_prob_target[:, :max_len, :]
        ignore_mask = ignore_mask[:, :max_len]
        seq_lens = np.minimum(seq_lens, max_len)

    # --- Prepare soft targets (zero out ignored positions) ---
    soft_targets = char_prob_target.copy()
    soft_targets[ignore_mask] = 0.0

    return {
        "neural": neural,
        "seq_lens": seq_lens,
        "gauss_labels": gauss_labels,
        "soft_targets": soft_targets,
        "ignore_mask": ignore_mask,
        "sentences": prompts,
        "char_seqs": char_seqs,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "std_all": std_all,
        "means_block": means_block,
    }


def prepare_labels(gauss_labels: np.ndarray, ignore_mask: np.ndarray) -> np.ndarray:
    """Convert Gaussian HMM labels to training targets (ignore → -1)."""
    labels = gauss_labels.copy()
    labels[ignore_mask] = -1
    return labels


# ---------------------------------------------------------------------------
# Poisson HMM alignment on sentences
# ---------------------------------------------------------------------------

def run_poisson_alignment(
    data_dir: Path,
    session: str,
    neural_raw: np.ndarray,
    seq_lens: np.ndarray,
    sentences: List[str],
) -> np.ndarray:
    """
    Fit Poisson HMM templates from single-letter data, then align sentences.

    IMPORTANT: Poisson emissions model integer spike counts, so we pass
    raw (un-normalised) neural data — NOT z-scored data.

    Returns:
        poisson_labels: (n_sentences, T_padded) int — frame-level char indices
    """
    from data.loader import WillettDataset
    from alignment.poisson_hmm import PoissonHMMForcedAlignment

    print("  Fitting Poisson HMM templates from single-letter data...")
    ds = WillettDataset(str(data_dir))
    cubes = ds.get_time_warped_cubes(session)

    # hmm_bin_size=20 gives ~20 states per character.
    # For a 20-char sentence → 400 states across ~1500 timesteps,
    # so each state covers ~3-4 bins on average.
    # stay_prob=0.6 gives expected dwell time of 2.5 bins/state.
    phmm = PoissonHMMForcedAlignment(
        hmm_bin_size=20, stay_prob=0.6, skip_prob=0.05,
    )
    templates = phmm.fit_templates(cubes, {})

    print(f"  Aligning {len(sentences)} sentences with Poisson HMM...")
    T_padded = neural_raw.shape[1]
    poisson_labels = np.full((len(sentences), T_padded), -1, dtype=int)

    for i, (prompt, slen) in enumerate(zip(sentences, seq_lens)):
        chars = []
        for c in str(prompt):
            full = _ABBREV_TO_FULL.get(c, c)
            if full in templates:
                chars.append(full)
        if not chars:
            continue

        # Pass RAW spike counts — Poisson needs integer count data
        obs = neural_raw[i, :slen, :].astype(float)

        starts, durations = _align_sentence(phmm, obs, chars, templates)

        # Fill frame labels
        for j, (s, d, ch) in enumerate(zip(starts, durations, chars)):
            idx = CHAR_TO_IDX.get(ch, -1)
            if idx >= 0 and s < slen:
                end = min(int(s) + int(d), slen)
                poisson_labels[i, int(s):end] = idx

        if (i + 1) % 20 == 0:
            print(f"    Aligned {i + 1}/{len(sentences)} sentences")

    print(f"  Poisson alignment complete.")
    return poisson_labels


def _align_sentence(
    hmm,
    obs: np.ndarray,
    chars: List[str],
    templates: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a sentence using the HMM's Viterbi algorithm.

    We build a composite sentence string and template dict for alignment.
    Since align() expects a string where each character maps to a template,
    we use single-char keys and build a mapping.
    """
    # Build a unique-key template dict for this sentence
    # The HMM's align() iterates over each char in the sentence string,
    # looking up templates[char]. For multi-char names like 'greaterThan',
    # we need to use abbreviated single-char keys.
    abbrev_map = {f: a for f, a in zip(CHAR_LIST_FULL, CHAR_ABBREV)}
    local_templates = {}
    for ch in chars:
        abbr = abbrev_map.get(ch, ch[0])
        if abbr not in local_templates and ch in templates:
            local_templates[abbr] = templates[ch]

    sentence_str = "".join(abbrev_map.get(ch, ch[0]) for ch in chars)
    starts, durations = hmm.align(obs, sentence_str, local_templates)
    return starts, durations


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def normalize_neural(
    neural: np.ndarray,
    train_idx: np.ndarray,
) -> np.ndarray:
    """Z-score normalise using training set statistics."""
    train_data = neural[train_idx]
    mean = train_data.mean(axis=(0, 1), keepdims=True)
    std = train_data.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (neural - mean) / std


def augment_data(X: np.ndarray, y, n_augments: int = 2) -> tuple:
    """
    Simple data augmentation: Gaussian noise + random time scaling.

    Works with both hard labels (2D) and soft labels (3D).
    Returns augmented X, y concatenated with originals.
    """
    all_X = [X]
    all_y = [y]

    for _ in range(n_augments):
        # Gaussian noise (σ = 10% of data std)
        noise_std = 0.1 * X.std()
        X_noisy = X + np.random.randn(*X.shape).astype(np.float32) * noise_std
        all_X.append(X_noisy)
        all_y.append(y.copy())

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# ---------------------------------------------------------------------------
# Simple bigram language model for rescoring
# ---------------------------------------------------------------------------

class BigramLM:
    """
    Character-level bigram LM trained on English text.
    Used to rescore decoder outputs via beam search.
    """

    def __init__(self, vocab: list, smoothing: float = 0.1):
        self.vocab = vocab
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.n = len(vocab)
        self.smoothing = smoothing
        self.log_bigram = None
        self.log_unigram = None

    def fit(self, texts: list):
        """Train on list of character sequences (using CHAR_ABBREV encoding)."""
        counts = np.full((self.n, self.n), self.smoothing)
        unigram = np.full(self.n, self.smoothing)
        for text in texts:
            for i, c in enumerate(text):
                idx = self.char2idx.get(c, -1)
                if idx >= 0:
                    unigram[idx] += 1
                    if i > 0:
                        prev = self.char2idx.get(text[i - 1], -1)
                        if prev >= 0:
                            counts[prev, idx] += 1
        # Normalise to log probabilities
        self.log_unigram = np.log(unigram / unigram.sum())
        row_sums = counts.sum(axis=1, keepdims=True)
        self.log_bigram = np.log(counts / row_sums)

    def score(self, text: str) -> float:
        """Log-probability of a character string."""
        if not text:
            return -100.0
        score = 0.0
        for i, c in enumerate(text):
            idx = self.char2idx.get(c, -1)
            if idx < 0:
                score -= 10.0
                continue
            if i == 0:
                score += self.log_unigram[idx]
            else:
                prev = self.char2idx.get(text[i - 1], -1)
                if prev >= 0:
                    score += self.log_bigram[prev, idx]
                else:
                    score += self.log_unigram[idx]
        return score

    def rescore_beam(self, candidates: list, lm_weight: float = 0.3) -> str:
        """Pick best candidate from list of (string, acoustic_score) tuples."""
        if not candidates:
            return ""
        best, best_score = candidates[0][0], -float("inf")
        for text, ac_score in candidates:
            combined = ac_score + lm_weight * self.score(text)
            if combined > best_score:
                best_score = combined
                best = text
        return best


def _beam_decode(logits: np.ndarray, active_mask: np.ndarray,
                 beam_width: int = 10) -> list:
    """
    Simple frame-level beam search.
    Returns list of (string, score) candidates.
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.special import softmax, log_softmax

    if not active_mask.any():
        return [("", 0.0)]

    probs = softmax(logits[active_mask], axis=-1)
    T = probs.shape[0]
    if T < 10:
        return [("", 0.0)]

    window = min(51, T // 3)
    if window < 3:
        window = 3
    smoothed = uniform_filter1d(probs, size=window, axis=0)
    log_sm = np.log(smoothed + 1e-10)

    # Get top-k per frame
    top_k = min(3, smoothed.shape[1])
    min_run = max(5, T // 100)

    # Simple: generate candidates by varying the argmax at ambiguous positions
    pred_ids = smoothed.argmax(axis=1)

    # Base candidate from argmax
    candidates = []
    for shift in range(min(beam_width, top_k)):
        # Use nth-best at each frame
        sorted_ids = np.argsort(-smoothed, axis=1)
        candidate_ids = sorted_ids[:, min(shift, sorted_ids.shape[1] - 1)]

        # Collapse runs
        runs = []
        cur_val, cur_start = candidate_ids[0], 0
        score = log_sm[0, candidate_ids[0]]
        for i in range(1, len(candidate_ids)):
            score += log_sm[i, candidate_ids[i]]
            if candidate_ids[i] != cur_val:
                runs.append((cur_val, i - cur_start))
                cur_val, cur_start = candidate_ids[i], i
        runs.append((cur_val, len(candidate_ids) - cur_start))

        chars = []
        for val, length in runs:
            if length >= min_run and 0 <= val < len(CHAR_ABBREV):
                chars.append(CHAR_ABBREV[val])
        text = "".join(chars)
        avg_score = score / T
        candidates.append((text, float(avg_score)))

    # Also add the original argmax-based candidate
    runs = []
    cur_val, cur_start = pred_ids[0], 0
    for i in range(1, len(pred_ids)):
        if pred_ids[i] != cur_val:
            runs.append((cur_val, i - cur_start))
            cur_val, cur_start = pred_ids[i], i
    runs.append((cur_val, len(pred_ids) - cur_start))
    chars = []
    for val, length in runs:
        if length >= min_run and 0 <= val < len(CHAR_ABBREV):
            chars.append(CHAR_ABBREV[val])
    base_text = "".join(chars)
    base_score = float(np.mean([log_sm[t, pred_ids[t]] for t in range(T)]))
    candidates.append((base_text, base_score))

    return candidates


def _smooth_and_decode(logits: np.ndarray, active_mask: np.ndarray) -> str:
    """
    Convert frame-level logits to a character string using probability
    smoothing and peak detection.

    Instead of argmax → smooth → collapse (which fails with noisy predictions),
    we smooth the raw softmax probabilities over a large window, then detect
    character segments by finding where the dominant class changes.

    With ~75 frames per character on average, we use a 51-frame window so
    smoothing spans roughly 2/3 of a character segment — enough to stabilise
    without blurring adjacent characters together.
    """
    from scipy.ndimage import uniform_filter1d

    if not active_mask.any():
        return ""

    # Work with softmax probabilities (more informative than hard argmax)
    from scipy.special import softmax
    probs = softmax(logits[active_mask], axis=-1)  # (T_active, n_classes)
    T = probs.shape[0]

    if T < 10:
        return ""

    # Heavy smoothing on probability distributions
    window = min(51, T // 3)
    if window < 3:
        window = 3
    smoothed = uniform_filter1d(probs, size=window, axis=0)
    pred_ids = smoothed.argmax(axis=1)

    # Collapse with minimum run length proportional to expected char width
    # Estimate: ~75 frames/char for 1500 frames / 20 chars
    # Use min_run = 15 to filter noise (20% of expected segment)
    min_run = max(5, T // 100)

    runs = []
    cur_val, cur_start = pred_ids[0], 0
    for i in range(1, len(pred_ids)):
        if pred_ids[i] != cur_val:
            runs.append((cur_val, i - cur_start))
            cur_val, cur_start = pred_ids[i], i
    runs.append((cur_val, len(pred_ids) - cur_start))

    chars = []
    for val, length in runs:
        if length >= min_run and 0 <= val < len(CHAR_ABBREV):
            chars.append(CHAR_ABBREV[val])
    return "".join(chars)


def train_and_evaluate(
    name: str,
    decoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    char_seqs_test: List[List[int]],
    ref_sentences: List[str],
    is_ctc: bool = False,
    lm: BigramLM = None,
    y_test_hard: np.ndarray = None,
    **fit_kwargs,
) -> Dict[str, float]:
    """Train a decoder and evaluate on the test set.

    Args:
        y_test_hard: hard labels (2D) for frame accuracy when y_test is soft (3D).
        lm: optional BigramLM for rescoring decoded sequences.
    """
    from benchmarks.evaluate import (
        compute_character_error_rate,
        compute_word_error_rate,
    )

    print(f"\n  Training {name}...")
    t0 = time.time()
    decoder.fit(X_train, y_train, **fit_kwargs)
    dt = time.time() - t0
    print(f"  Trained in {dt:.1f}s")

    # Predict
    logits = decoder.predict(X_test)  # (B, T, C)

    # For frame accuracy, always use hard labels
    y_eval = y_test_hard if y_test_hard is not None else y_test

    if is_ctc:
        from decoders.ctc_decoder import ctc_greedy_decode
        log_probs = logits  # already log-softmax
        decoded_seqs = ctc_greedy_decode(log_probs, blank=0)
        # Convert indices back (CTC uses 1-indexed, subtract 1)
        pred_strings = []
        for seq in decoded_seqs:
            chars = [CHAR_ABBREV[i - 1] if 0 < i <= len(CHAR_ABBREV) else "?"
                     for i in seq]
            pred_strings.append("".join(chars))
    else:
        # Frame-level: beam decode with LM rescoring, or smooth decode
        pred_strings = []
        for i in range(logits.shape[0]):
            if y_eval.ndim == 2:
                active = y_eval[i] >= 0
            else:
                active = y_eval[i].sum(axis=-1) > 0.5

            if lm is not None:
                candidates = _beam_decode(logits[i], active)
                pred_strings.append(lm.rescore_beam(candidates, lm_weight=0.3))
            else:
                pred_strings.append(_smooth_and_decode(logits[i], active))

    # Reference strings from actual sentence prompts (NOT from collapsing
    # frame labels — that merges repeated characters like "ll" in "hello")
    ref_strings = []
    for sent in ref_sentences:
        s = str(sent)
        chars = []
        for c in s:
            full = _ABBREV_TO_FULL.get(c, c)
            if full in CHAR_TO_IDX:
                chars.append(c)
        ref_strings.append("".join(chars))

    # Show a few examples
    for k in range(min(3, len(pred_strings))):
        print(f"    [{k}] ref:  {ref_strings[k][:60]}")
        print(f"         pred: {pred_strings[k][:60]}")

    cer = compute_character_error_rate(pred_strings, ref_strings)
    wer = compute_word_error_rate(
        [s.replace(">", " ").replace("~", ".") for s in pred_strings],
        [s.replace(">", " ").replace("~", ".") for s in ref_strings],
    )

    # Frame accuracy on active timesteps (always use hard labels)
    if not is_ctc:
        pred_all = logits.argmax(axis=-1)
        if y_eval.ndim == 2:
            active_mask = y_eval >= 0
            if active_mask.any():
                frame_acc = float((pred_all[active_mask] == y_eval[active_mask]).mean())
            else:
                frame_acc = 0.0
        else:
            # y_eval is soft (3D) — convert to hard for accuracy
            hard = y_eval.argmax(axis=-1)
            active_mask = y_eval.sum(axis=-1) > 0.5
            if active_mask.any():
                frame_acc = float((pred_all[active_mask] == hard[active_mask]).mean())
            else:
                frame_acc = 0.0
    else:
        frame_acc = 1.0 - cer

    return {"cer": cer, "wer": wer, "frame_acc": frame_acc, "train_time": dt}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: alignment methods × decoder architectures")
    parser.add_argument("--data-dir", default="./handwritingBCIData",
                        help="Path to extracted Dryad dataset")
    parser.add_argument("--session", default="t5.2019.05.08",
                        help="Session ID to benchmark")
    parser.add_argument("--partition", default="HeldOutTrials",
                        choices=["HeldOutTrials", "HeldOutBlocks"])
    parser.add_argument("--max-len", type=int, default=1500,
                        help="Truncate sentences to this many time bins "
                             "(0 = no truncation, default 1500 ≈ 15 seconds)")
    parser.add_argument("--full", action="store_true",
                        help="Full run: more epochs, larger models")
    parser.add_argument("--decoders", nargs="+",
                        default=["gru", "rcnn", "conformer", "ctc"],
                        help="Which decoders to benchmark")
    parser.add_argument("--skip-poisson", action="store_true",
                        help="Skip Poisson HMM alignment (faster)")
    parser.add_argument("--multi-session", action="store_true",
                        help="Train on all available sessions (test on primary)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fast = not args.full

    # Hyperparameters
    if fast:
        epochs, batch_size = 20, 8
        hidden, d_model = 128, 64
        n_layers_rnn, n_layers_conf = 1, 2
        n_heads, ff_dim = 4, 128
        conv_ch_rcnn = (32, 64)
        conv_ch_ctc = (32, 64)
        lstm_hidden = 64
    else:
        epochs, batch_size = 80, 16
        hidden, d_model = 256, 128
        n_layers_rnn, n_layers_conf = 2, 4
        n_heads, ff_dim = 4, 512
        conv_ch_rcnn = (32, 64, 128)
        conv_ch_ctc = (64, 128)
        lstm_hidden = 128

    N_CHARS = len(CHAR_LIST_FULL)  # 31

    print("=" * 70)
    print("Neural Handwriting Decoding Benchmark")
    print("=" * 70)
    print(f"  Session:    {args.session}")
    print(f"  Partition:  {args.partition}")
    print(f"  Max length: {args.max_len} bins ({args.max_len * 10}ms)")
    print(f"  Mode:       {'full' if args.full else 'fast'}")
    print(f"  Multi-sess: {args.multi_session}")
    print(f"  Decoders:   {args.decoders}")
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/5] Loading sentence data...")
    data = load_sentence_data(data_dir, args.session, args.partition,
                              max_len=args.max_len)

    neural = data["neural"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    gauss_labels = prepare_labels(data["gauss_labels"], data["ignore_mask"])

    print(f"  Primary session: {neural.shape}")
    print(f"  Train: {len(train_idx)} sentences, Test: {len(test_idx)} sentences")

    # --- Multi-session: load other sessions for additional training data ---
    if args.multi_session:
        print("\n  Loading additional sessions for multi-session training...")
        datasets_dir = data_dir / "Datasets"
        hmm_dir = data_dir / "RNNTrainingSteps" / "Step2_HMMLabels" / args.partition
        part_path = data_dir / "RNNTrainingSteps" / f"trainTestPartitions_{args.partition}.mat"
        part_raw = _load_mat(part_path)

        extra_neural = []
        extra_gauss = []
        extra_soft = []
        extra_sentences = []
        extra_char_seqs = []
        n_extra = 0

        for sess_dir in sorted(datasets_dir.iterdir()):
            sess = sess_dir.name
            if not sess_dir.is_dir() or sess == args.session:
                continue
            sent_path = sess_dir / "sentences.mat"
            hmm_path = hmm_dir / f"{sess}_timeSeriesLabels.mat"
            # Check that both files exist and partition has this session
            if not sent_path.exists() or not hmm_path.exists():
                continue
            train_key = f"{sess}_train"
            if train_key not in part_raw:
                continue

            try:
                sess_data = load_sentence_data(data_dir, sess, args.partition,
                                               max_len=args.max_len)
                s_train = sess_data["train_idx"]
                # Pad/truncate to match primary session time dimension
                s_neural = sess_data["neural"]
                s_gauss = prepare_labels(sess_data["gauss_labels"], sess_data["ignore_mask"])
                s_soft = sess_data["soft_targets"]
                T_primary = neural.shape[1]
                T_sess = s_neural.shape[1]

                if T_sess < T_primary:
                    # Pad with zeros
                    pad_n = T_primary - T_sess
                    s_neural = np.pad(s_neural, ((0,0),(0,pad_n),(0,0)))
                    s_gauss = np.pad(s_gauss, ((0,0),(0,pad_n)), constant_values=-1)
                    s_soft = np.pad(s_soft, ((0,0),(0,pad_n),(0,0)))
                elif T_sess > T_primary:
                    s_neural = s_neural[:, :T_primary, :]
                    s_gauss = s_gauss[:, :T_primary]
                    s_soft = s_soft[:, :T_primary, :]

                # Only take training sentences from extra sessions
                extra_neural.append(s_neural[s_train])
                extra_gauss.append(s_gauss[s_train])
                extra_soft.append(s_soft[s_train])
                for i in s_train:
                    extra_sentences.append(str(sess_data["sentences"][i]))
                    extra_char_seqs.append(sess_data["char_seqs"][i])

                n_extra += len(s_train)
                print(f"    {sess}: +{len(s_train)} training sentences")
            except Exception as e:
                print(f"    {sess}: skipped ({e})")
                continue

        if extra_neural:
            # Concatenate extra training data
            extra_neural_all = np.concatenate(extra_neural, axis=0)
            extra_gauss_all = np.concatenate(extra_gauss, axis=0)
            extra_soft_all = np.concatenate(extra_soft, axis=0)

            # Store for later use — we'll prepend these to training arrays
            data["_extra_neural"] = extra_neural_all
            data["_extra_gauss"] = extra_gauss_all
            data["_extra_soft"] = extra_soft_all
            data["_extra_sentences"] = extra_sentences
            data["_extra_char_seqs"] = extra_char_seqs
            print(f"  Total extra training data: {n_extra} sentences from {len(extra_neural)} sessions")
        else:
            print("  No additional sessions found.")

    # ------------------------------------------------------------------
    # 2. Normalize
    # ------------------------------------------------------------------
    print("\n[2/5] Normalizing neural data...")
    neural_norm = normalize_neural(neural, train_idx)

    # ------------------------------------------------------------------
    # 3. Poisson HMM alignment 
    # ------------------------------------------------------------------
    alignment_conditions = {
        "Gaussian Soft (Willett)": data["soft_targets"],  # 3D soft probs
        "Gaussian Hard (Willett)": gauss_labels,           # 2D hard labels
    }

    if not args.skip_poisson:
        print("\n[3/5] Running Poisson HMM alignment...")
        poisson_raw = run_poisson_alignment(
            data_dir, args.session, neural, data["seq_lens"],
            data["sentences"],
        )
        # Apply ignore mask from Willett (same pre/post-sentence idle periods)
        poisson_labels = poisson_raw.copy()
        poisson_labels[data["ignore_mask"]] = -1
        alignment_conditions["Poisson HMM (ours)"] = poisson_labels
    else:
        print("\n[3/5] Skipping Poisson HMM alignment")

    # ------------------------------------------------------------------
    # 4. Train and evaluate decoders
    # ------------------------------------------------------------------
    print("\n[4/5] Training decoders...")

    X_train = neural_norm[train_idx]
    X_test = neural_norm[test_idx]
    N_CH = X_train.shape[2]  # 192

    # Normalise extra session data using same stats, if available
    if "_extra_neural" in data:
        train_data = neural[train_idx]
        mean = train_data.mean(axis=(0, 1), keepdims=True)
        std = train_data.std(axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1.0, std)
        extra_neural_norm = (data["_extra_neural"] - mean) / std
    else:
        extra_neural_norm = None

    # Hard labels for frame accuracy evaluation (always needed)
    gauss_hard_test = gauss_labels[test_idx]

    results = {}  # {(alignment, decoder): metrics}

    # --- Train bigram LM on training sentence prompts ---
    print("\n  Training bigram language model on training sentences...")
    lm = BigramLM(vocab=list(CHAR_ABBREV))
    train_texts = []
    for i in train_idx:
        s = str(data["sentences"][i])
        chars = []
        for c in s:
            full = _ABBREV_TO_FULL.get(c, c)
            if full in CHAR_TO_IDX:
                chars.append(c)
        train_texts.append("".join(chars))
    # Also add extra session texts to LM
    if "_extra_sentences" in data:
        for s in data["_extra_sentences"]:
            chars = []
            for c in s:
                full = _ABBREV_TO_FULL.get(c, c)
                if full in CHAR_TO_IDX:
                    chars.append(c)
            train_texts.append("".join(chars))
    lm.fit(train_texts)
    print(f"  LM trained on {len(train_texts)} sentences")

    # Reference sentences for the test set (actual prompt text)
    ref_sentences_test = [str(data["sentences"][i]) for i in test_idx]

    for align_name, labels in alignment_conditions.items():
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Prepend extra session data if available
        X_train_combined = X_train
        y_train_combined = y_train
        if extra_neural_norm is not None and "Poisson" not in align_name:
            # Determine which extra labels to use based on alignment type
            if labels.ndim == 3:  # soft targets
                extra_labels = data["_extra_soft"]
            else:  # hard labels
                extra_labels = data["_extra_gauss"]
            X_train_combined = np.concatenate([X_train, extra_neural_norm], axis=0)
            y_train_combined = np.concatenate([y_train, extra_labels], axis=0)
            print(f"\n  Multi-session training: {X_train.shape[0]} + {extra_neural_norm.shape[0]} = {X_train_combined.shape[0]} sentences")

        # Data augmentation for training
        n_aug = 2 if not fast else 1
        X_train_aug, y_train_aug = augment_data(X_train_combined, y_train_combined, n_augments=n_aug)
        print(f"\n  Augmented training: {X_train_combined.shape[0]} → {X_train_aug.shape[0]} sentences")

        print(f"\n{'=' * 60}")
        print(f"Alignment: {align_name}")
        print(f"{'=' * 60}")

        for dec_name in args.decoders:
          try:
            if dec_name == "gru":
                from decoders.rnn_decoder import RNNDecoder
                dec = RNNDecoder(
                    n_inputs=N_CH, n_outputs=N_CHARS,
                    hidden_size=hidden, n_layers=n_layers_rnn,
                )
                metrics = train_and_evaluate(
                    f"GRU ({align_name})", dec,
                    X_train_aug, y_train_aug, X_test, y_test, [],
                    ref_sentences_test, lm=lm,
                    y_test_hard=gauss_hard_test,
                    epochs=epochs, batch_size=batch_size, lr=1e-3,
                )
                results[(align_name, "GRU")] = metrics

            elif dec_name == "rcnn":
                from decoders.rcnn_decoder import RCNNDecoder
                dec = RCNNDecoder(
                    n_inputs=N_CH, n_outputs=N_CHARS,
                    conv_channels=conv_ch_rcnn, kernel_size=5,
                    hidden_size=hidden, n_layers=n_layers_rnn,
                )
                metrics = train_and_evaluate(
                    f"RCNN ({align_name})", dec,
                    X_train_aug, y_train_aug, X_test, y_test, [],
                    ref_sentences_test, lm=lm,
                    y_test_hard=gauss_hard_test,
                    epochs=epochs, batch_size=batch_size, lr=1e-3,
                )
                results[(align_name, "RCNN")] = metrics

            elif dec_name == "conformer":
                from decoders.transformer_decoder import TransformerDecoder
                # Conformer self-attention is O(T^2) memory — use smaller
                # batch size to avoid OOM on long sequences
                conf_batch = max(1, batch_size // 4)
                dec = TransformerDecoder(
                    n_inputs=N_CH, n_outputs=N_CHARS,
                    d_model=d_model, n_heads=n_heads,
                    n_layers=n_layers_conf, conv_kernel_size=15,
                    ff_dim=ff_dim, dropout=0.1,
                )
                metrics = train_and_evaluate(
                    f"Conformer ({align_name})", dec,
                    X_train_aug, y_train_aug, X_test, y_test, [],
                    ref_sentences_test, lm=lm,
                    y_test_hard=gauss_hard_test,
                    epochs=epochs, batch_size=conf_batch, lr=5e-4,
                    warmup_steps=100,
                )
                results[(align_name, "Conformer")] = metrics

            elif dec_name == "ctc":
                # CTC only needs to run once (alignment-free)
                if align_name != list(alignment_conditions.keys())[0]:
                    continue
                from decoders.ctc_decoder import CTCDecoder
                dec = CTCDecoder(
                    n_inputs=N_CH, n_outputs=N_CHARS + 1,  # +1 for blank
                    conv_channels=conv_ch_ctc, kernel_size=5,
                    lstm_hidden=lstm_hidden, dropout=0.2,
                )
                # CTC targets: list of char index arrays (1-indexed)
                y_tr_ctc = [np.array(data["char_seqs"][i], dtype=np.int64) + 1
                            for i in train_idx]
                # Add extra session CTC targets if multi-session
                if "_extra_char_seqs" in data:
                    for seq in data["_extra_char_seqs"]:
                        y_tr_ctc.append(np.array(seq, dtype=np.int64) + 1)
                y_tr_ctc_aug = y_tr_ctc * (n_aug + 1)  # replicate for augmented data
                metrics = train_and_evaluate(
                    "CTC (alignment-free)", dec,
                    X_train_aug, y_tr_ctc_aug, X_test, y_test, [],
                    ref_sentences_test, lm=lm,
                    y_test_hard=gauss_hard_test,
                    is_ctc=True,
                    epochs=epochs, batch_size=batch_size, lr=1e-3,
                )
                results[("No alignment", "CTC")] = metrics

          except Exception as e:
                print(f"\n  WARNING: {dec_name} failed: {e}")
                print(f"  Skipping and continuing...")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Results table
    # ------------------------------------------------------------------
    print("\n\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Collect unique alignments and decoders
    alignments = sorted(set(a for a, _ in results.keys()))
    decoders = sorted(set(d for _, d in results.keys()))

    # Header
    header = f"{'Decoder':<12s}"
    for a in alignments:
        short = a.split("(")[0].strip() if "(" in a else a
        header += f" | {short:>16s} CER  {short:>8s} WER  {'Acc':>8s}"
    print(header)
    print("-" * len(header))

    for dec in decoders:
        row = f"{dec:<12s}"
        for align in alignments:
            key = (align, dec)
            if key in results:
                m = results[key]
                row += f" | {m['cer']*100:>14.2f}%  {m['wer']*100:>8.2f}%  {m['frame_acc']*100:>7.1f}%"
            else:
                row += f" | {'—':>15s}  {'—':>9s}  {'—':>8s}"
        print(row)

    print()
    print("CER = Character Error Rate (lower is better)")
    print("WER = Word Error Rate (lower is better)")
    print("Acc = Frame-level accuracy (higher is better)")

    # Also print a simpler comparison
    print("\n\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for (align, dec), m in sorted(results.items()):
        print(f"  {dec:<12s} + {align:<28s}  "
              f"CER={m['cer']*100:5.2f}%  WER={m['wer']*100:5.2f}%  "
              f"Acc={m['frame_acc']*100:5.1f}%  ({m['train_time']:.0f}s)")

    return results


if __name__ == "__main__":
    main()
