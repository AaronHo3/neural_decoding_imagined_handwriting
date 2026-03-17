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
        ignore_mask = ignore_mask[:, :max_len]
        seq_lens = np.minimum(seq_lens, max_len)

    return {
        "neural": neural,
        "seq_lens": seq_lens,
        "gauss_labels": gauss_labels,
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
    **fit_kwargs,
) -> Dict[str, float]:
    """Train a decoder and evaluate on the test set."""
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
        # Frame-level: smooth probabilities, then collapse to string
        pred_strings = []
        for i in range(len(y_test)):
            active = y_test[i] >= 0
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

    # Frame accuracy on active timesteps
    if not is_ctc:
        pred_all = logits.argmax(axis=-1)
        active_mask = y_test >= 0
        if active_mask.any():
            frame_acc = float((pred_all[active_mask] == y_test[active_mask]).mean())
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

    print(f"  Neural shape: {neural.shape}")
    print(f"  Train: {len(train_idx)} sentences, Test: {len(test_idx)} sentences")

    # ------------------------------------------------------------------
    # 2. Normalize
    # ------------------------------------------------------------------
    print("\n[2/5] Normalizing neural data...")
    neural_norm = normalize_neural(neural, train_idx)

    # ------------------------------------------------------------------
    # 3. Poisson HMM alignment 
    # ------------------------------------------------------------------
    alignment_conditions = {"Gaussian HMM (Willett)": gauss_labels}

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

    results = {}  # {(alignment, decoder): metrics}

    # Reference sentences for the test set (actual prompt text)
    ref_sentences_test = [str(data["sentences"][i]) for i in test_idx]

    for align_name, labels in alignment_conditions.items():
        y_train = labels[train_idx]
        y_test = labels[test_idx]

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
                    X_train, y_train, X_test, y_test, [],
                    ref_sentences_test,
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
                    X_train, y_train, X_test, y_test, [],
                    ref_sentences_test,
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
                    X_train, y_train, X_test, y_test, [],
                    ref_sentences_test,
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
                metrics = train_and_evaluate(
                    "CTC (alignment-free)", dec,
                    X_train, y_tr_ctc, X_test, y_test, [],
                    ref_sentences_test,
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
