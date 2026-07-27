#!/usr/bin/env python3
"""
RQ1: What is the alignment error budget?

Sweeps controlled corruption of the HMM frame labels and measures the decoding
response. See docs/RESEARCH_PLAN.md section 2 for hypotheses.

Two corruption models, chosen to mimic the two ways forced alignment fails:

    boundary jitter    - right characters, wrong transition times
    segment corruption - wrong character identities

Corruption is applied to TRAINING labels only. Test evaluation always scores
against the true prompt text, so the metric itself is never perturbed.

Usage:
    # single condition
    python3 experiments/exp1_alignment_sensitivity.py \
        --corruption jitter --level 10 --decoder rcnn --seed 0

    # full sweep (resumable: skips conditions whose artifact already exists)
    python3 experiments/exp1_alignment_sensitivity.py --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_benchmark import (  # noqa: E402
    BigramLM,
    augment_data,
    git_sha,
    load_sentence_data,
    normalize_neural,
    prepare_labels,
    set_seed,
    train_and_evaluate,
    _ABBREV_TO_FULL,
)
from data.loader import CHAR_ABBREV, CHAR_LIST_FULL, CHAR_TO_IDX  # noqa: E402

N_CHARS = len(CHAR_LIST_FULL)

# Sweep grids, fixed by the research plan.
JITTER_LEVELS = [0, 5, 10, 20, 40]        # sigma, in 10 ms frames
CORRUPT_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.40]  # per-segment relabel probability
DECODERS = ["gru", "rcnn", "conformer"]
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Corruption models
# ---------------------------------------------------------------------------

def _segments(row: np.ndarray) -> List[Tuple[int, int, int]]:
    """Split a frame-label row into (start, end, label) runs, skipping -1 padding."""
    segs = []
    start = 0
    for i in range(1, len(row) + 1):
        if i == len(row) or row[i] != row[start]:
            if row[start] >= 0:
                segs.append((start, i, int(row[start])))
            start = i
    return segs


def jitter_boundaries(labels: np.ndarray, sigma: float, rng) -> np.ndarray:
    """
    Displace each character-segment boundary by N(0, sigma) frames.

    Segment identity and order are preserved; only transition times move. This
    models an aligner that reads the sequence correctly but mislocates when each
    character starts. Padding (-1) positions are left untouched.
    """
    if sigma <= 0:
        return labels.copy()

    out = labels.copy()
    for i, row in enumerate(labels):
        segs = _segments(row)
        if len(segs) < 2:
            continue

        # Perturb interior boundaries, then repair ordering so segments stay
        # non-overlapping and monotone.
        bounds = [s[0] for s in segs] + [segs[-1][1]]
        noise = rng.normal(0.0, sigma, size=len(bounds)).round().astype(int)
        noise[0] = 0        # keep the sentence onset fixed
        noise[-1] = 0       # keep the sentence offset fixed
        new_bounds = np.array(bounds) + noise
        new_bounds = np.maximum.accumulate(new_bounds)
        new_bounds = np.clip(new_bounds, bounds[0], bounds[-1])

        out[i, bounds[0]:bounds[-1]] = -1
        for (b0, b1), (_, _, lab) in zip(zip(new_bounds[:-1], new_bounds[1:]), segs):
            if b1 > b0:
                out[i, b0:b1] = lab
    return out


def corrupt_segments(labels: np.ndarray, p: float, rng) -> np.ndarray:
    """
    Replace each character segment's label with a uniformly random character
    with probability p. Boundaries are preserved; only identities change.
    """
    if p <= 0:
        return labels.copy()

    out = labels.copy()
    for i, row in enumerate(labels):
        for (s, e, lab) in _segments(row):
            if rng.random() < p:
                choices = [c for c in range(N_CHARS) if c != lab]
                out[i, s:e] = rng.choice(choices)
    return out


# ---------------------------------------------------------------------------
# Single condition
# ---------------------------------------------------------------------------

def run_condition(
    data_dir: Path,
    session: str,
    partition: str,
    corruption: str,
    level: float,
    decoder: str,
    seed: int,
    max_len: int,
    epochs: int,
    out_dir: Path,
) -> dict:
    """Train one decoder under one corruption level and write a JSON artifact."""
    tag = f"{corruption}_{level}_{decoder}_seed{seed}"
    out_path = out_dir / f"{tag}.json"
    if out_path.exists():
        print(f"[skip] {tag}: artifact exists")
        return json.loads(out_path.read_text())

    set_seed(seed)
    rng = np.random.default_rng(seed)

    data = load_sentence_data(data_dir, session, partition, max_len=max_len)
    train_idx, test_idx = data["train_idx"], data["test_idx"]
    gauss_labels = prepare_labels(data["gauss_labels"], data["ignore_mask"])
    neural_norm = normalize_neural(data["neural"], train_idx)

    # Corrupt TRAINING labels only.
    y_train_clean = gauss_labels[train_idx]
    if corruption == "jitter":
        y_train = jitter_boundaries(y_train_clean, float(level), rng)
    elif corruption == "corrupt":
        y_train = corrupt_segments(y_train_clean, float(level), rng)
    else:
        raise ValueError(f"unknown corruption model: {corruption!r}")

    # How far the corrupted labels actually moved, so the x-axis is grounded in
    # a measured quantity rather than only the nominal knob setting.
    active = y_train_clean >= 0
    label_agreement = float((y_train[active] == y_train_clean[active]).mean())

    X_train, X_test = neural_norm[train_idx], neural_norm[test_idx]
    X_aug, y_aug = augment_data(X_train, y_train, n_augments=2)

    lm = BigramLM(vocab=list(CHAR_ABBREV))
    lm.fit([
        "".join(c for c in str(data["sentences"][i])
                if _ABBREV_TO_FULL.get(c, c) in CHAR_TO_IDX)
        for i in train_idx
    ])

    if decoder == "gru":
        from decoders.rnn_decoder import RNNDecoder
        dec = RNNDecoder(n_inputs=X_train.shape[2], n_outputs=N_CHARS,
                         hidden_size=256, n_layers=2)
        fit_kwargs = dict(epochs=epochs, batch_size=16, lr=1e-3)
    elif decoder == "rcnn":
        from decoders.rcnn_decoder import RCNNDecoder
        dec = RCNNDecoder(n_inputs=X_train.shape[2], n_outputs=N_CHARS,
                          conv_channels=(32, 64, 128), kernel_size=5,
                          hidden_size=256, n_layers=2)
        fit_kwargs = dict(epochs=epochs, batch_size=16, lr=1e-3)
    elif decoder == "conformer":
        from decoders.transformer_decoder import TransformerDecoder
        dec = TransformerDecoder(n_inputs=X_train.shape[2], n_outputs=N_CHARS,
                                 d_model=128, n_heads=4, n_layers=4,
                                 conv_kernel_size=15, ff_dim=512, dropout=0.1)
        fit_kwargs = dict(epochs=epochs, batch_size=4, lr=5e-4, warmup_steps=100)
    else:
        raise ValueError(f"unknown decoder: {decoder!r}")

    metrics = train_and_evaluate(
        f"{decoder} ({corruption}={level})", dec,
        X_aug, y_aug, X_test, gauss_labels[test_idx], [],
        [str(data["sentences"][i]) for i in test_idx],
        lm=lm, y_test_hard=gauss_labels[test_idx], **fit_kwargs,
    )

    artifact = {
        "experiment": "exp1_alignment_sensitivity",
        "research_question": "RQ1",
        "corruption_model": corruption,
        "corruption_level": level,
        "label_agreement_with_clean": label_agreement,
        "decoder": decoder,
        "seed": seed,
        "session": session,
        "partition": partition,
        "max_len": max_len,
        "epochs": epochs,
        "n_train_sentences": int(X_train.shape[0]),
        "n_test_sentences": int(len(test_idx)),
        "git_sha": git_sha(),
        **metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"[done] {tag}  CER={metrics['cer']*100:.2f}%  → {out_path}")
    return artifact


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./handwritingBCIData")
    p.add_argument("--session", default="t5.2019.05.08")
    p.add_argument("--partition", default="HeldOutTrials",
                   choices=["HeldOutTrials", "HeldOutBlocks"])
    p.add_argument("--corruption", choices=["jitter", "corrupt"], default="jitter")
    p.add_argument("--level", type=float, default=0.0)
    p.add_argument("--decoder", choices=DECODERS, default="rcnn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-len", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--out-dir", default="results/exp1_alignment_sensitivity")
    p.add_argument("--sweep", action="store_true",
                   help="Run the full pre-registered grid (resumable)")
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    common = dict(data_dir=data_dir, session=args.session,
                  partition=args.partition, max_len=args.max_len,
                  epochs=args.epochs, out_dir=out_dir)

    if not args.sweep:
        run_condition(corruption=args.corruption, level=args.level,
                      decoder=args.decoder, seed=args.seed, **common)
        return

    grid = [("jitter", lv) for lv in JITTER_LEVELS] + \
           [("corrupt", lv) for lv in CORRUPT_LEVELS]
    total = len(grid) * len(DECODERS) * len(SEEDS)
    done = 0
    for corruption, level in grid:
        for decoder in DECODERS:
            for seed in SEEDS:
                done += 1
                print(f"\n=== [{done}/{total}] {corruption}={level} "
                      f"{decoder} seed={seed} ===")
                try:
                    run_condition(corruption=corruption, level=level,
                                  decoder=decoder, seed=seed, **common)
                except Exception as e:
                    # Record the failure rather than silently dropping the cell,
                    # so a gap in the sweep is visible in the artifacts.
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fail = out_dir / f"FAILED_{corruption}_{level}_{decoder}_seed{seed}.json"
                    fail.write_text(json.dumps(
                        {"error": repr(e), "corruption_model": corruption,
                         "corruption_level": level, "decoder": decoder,
                         "seed": seed}, indent=2))
                    print(f"  FAILED: {e}  (recorded at {fail})")


if __name__ == "__main__":
    main()
