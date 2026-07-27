#!/usr/bin/env python3
"""
Generate every figure and summary table from committed JSON artifacts.

No number in the writeup is typed by hand: this script is the only path from
results to figures. If an artifact is missing, the corresponding point is
absent from the plot rather than interpolated.

Usage:
    python3 analysis/make_figures.py                  # all figures
    python3 analysis/make_figures.py --experiment exp1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

CORRUPTION_LABELS = {
    "jitter": "Boundary jitter σ (10 ms frames)",
    "corrupt": "Segment relabel probability p",
}


def load_artifacts(subdir: str) -> List[Dict]:
    """Load every non-failed artifact from results/<subdir>/."""
    d = RESULTS / subdir
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("FAILED_"):
            print(f"  ! skipping recorded failure: {p.name}")
            continue
        out.append(json.loads(p.read_text()))
    return out


def summarise(runs: List[Dict], key_fields: tuple) -> Dict[tuple, Dict]:
    """Group runs by key_fields and reduce seeds to mean / min / max."""
    groups = defaultdict(list)
    for r in runs:
        groups[tuple(r[f] for f in key_fields)].append(r)

    summary = {}
    for key, rs in groups.items():
        cers = [r["cer"] * 100 for r in rs]
        summary[key] = {
            "mean": sum(cers) / len(cers),
            "min": min(cers),
            "max": max(cers),
            "n_seeds": len(cers),
            "seeds": sorted(r["seed"] for r in rs),
            "cers": cers,
        }
    return summary


def figure_exp1(runs: List[Dict], stem: str = "exp1_alignment_sensitivity",
                label: str = "") -> None:
    """RQ1: CER dose-response against each corruption model."""
    import matplotlib.pyplot as plt

    summary = summarise(runs, ("corruption_model", "corruption_level", "decoder"))
    models = sorted({k[0] for k in summary})
    if not models:
        print("  no exp1 artifacts yet, skipping figure")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5),
                             squeeze=False)

    for ax, model in zip(axes[0], models):
        decoders = sorted({k[2] for k in summary if k[0] == model})
        for dec in decoders:
            pts = sorted(
                (k[1], v) for k, v in summary.items()
                if k[0] == model and k[2] == dec
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            means = [p[1]["mean"] for p in pts]
            lo = [p[1]["mean"] - p[1]["min"] for p in pts]
            hi = [p[1]["max"] - p[1]["mean"] for p in pts]

            ax.errorbar(xs, means, yerr=[lo, hi], marker="o", capsize=3,
                        label=dec, linewidth=1.8)
            # Individual seeds, so the reader sees the raw spread (n=3).
            for x, v in pts:
                ax.scatter([x] * len(v["cers"]), v["cers"], s=12, alpha=0.35)

        ax.set_xlabel(CORRUPTION_LABELS.get(model, model))
        ax.set_ylabel("Character error rate (%)")
        ax.set_title(f"Decoding response to {model}")
        ax.legend(title="Decoder", frameon=False)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    suffix = f" [{label}]" if label else ""
    fig.suptitle(
        "RQ1: label quality vs. decoder architecture "
        f"(n=10 test sentences, 3 seeds){suffix}",
        y=1.02,
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{stem}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")

    _report_h1c(summary)


def _report_h1c(summary: Dict[tuple, Dict]) -> None:
    """
    H1c is the project's central claim, so it gets computed explicitly rather
    than eyeballed off the plot: is the architecture spread at fixed label
    quality smaller than the label-quality spread at fixed architecture?
    """
    arch_spreads, label_spreads = [], []

    for model in {k[0] for k in summary}:
        levels = sorted({k[1] for k in summary if k[0] == model})
        decoders = sorted({k[2] for k in summary if k[0] == model})

        for lv in levels:
            means = [summary[(model, lv, d)]["mean"]
                     for d in decoders if (model, lv, d) in summary]
            if len(means) > 1:
                arch_spreads.append(max(means) - min(means))

        for d in decoders:
            means = [summary[(model, lv, d)]["mean"]
                     for lv in levels if (model, lv, d) in summary]
            if len(means) > 1:
                label_spreads.append(max(means) - min(means))

    if not arch_spreads or not label_spreads:
        return

    arch = sum(arch_spreads) / len(arch_spreads)
    label = sum(label_spreads) / len(label_spreads)
    print("\n  H1c: central thesis check:")
    print(f"    mean CER spread across architectures (fixed labels): {arch:.2f} pp")
    print(f"    mean CER spread across label quality (fixed arch):   {label:.2f} pp")
    verdict = "SUPPORTED" if label > arch else "NOT SUPPORTED"
    print(f"    -> H1c {verdict}")


# RQ1 result sets. The 1500-bin grid is retained as evidence for the truncation
# confound; the 3000-bin grid is the canonical result.
EXP1_SETS = [
    ("exp1_alignment_sensitivity", "exp1_alignment_sensitivity", "1500 bins"),
    ("exp1_sweep_3000", "exp1_sweep_3000", "3000 bins (full length)"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="all", choices=["all", "exp1"])
    p.add_argument("--subdir", default=None,
                   help="Plot one specific results/ subdirectory instead of all")
    args = p.parse_args()

    if args.experiment in ("all", "exp1"):
        sets = ([(args.subdir, args.subdir, args.subdir)] if args.subdir
                else EXP1_SETS)
        for subdir, stem, label in sets:
            print(f"{subdir}:")
            runs = load_artifacts(subdir)
            print(f"  loaded {len(runs)} artifacts")
            if runs:
                figure_exp1(runs, stem=stem, label=label)


if __name__ == "__main__":
    main()
