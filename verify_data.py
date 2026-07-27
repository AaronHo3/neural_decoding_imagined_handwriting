#!/usr/bin/env python3
"""
Verify a Willett dataset directory has everything the experiments need.

The dataset cannot be fetched automatically: Dryad returns 403 for the legacy
file_stream URL and 401 for the v2 API, so the 1.3 GB archive has to be
downloaded once through a browser. What a replicator actually needs is not a
download wrapper but a way to confirm the extracted tree is complete before
committing hours of GPU time, which is what this does.

Run it before any sweep. It exits non-zero if anything required is missing, and
names the specific missing path rather than failing deep inside training.

Usage:
    python verify_data.py
    python verify_data.py --data-dir C:\\path\\to\\handwritingBCIData
    python verify_data.py --partition HeldOutBlocks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Session used by the pre-registered experiments (docs/RESEARCH_PLAN.md 3).
PRIMARY_SESSION = "t5.2019.05.08"

DATASET_DOI = "https://doi.org/10.5061/dryad.wh70rxwmv"
ARCHIVE = "handwritingBCIData.tar.gz"  # 1.31 GB, expands to ~5.7 GB


def check(data_dir: Path, partition: str) -> int:
    problems: list[str] = []
    notes: list[str] = []

    print(f"Verifying: {data_dir}")
    print(f"Partition: {partition}\n")

    if not data_dir.exists():
        print(f"FAIL  directory does not exist: {data_dir}")
        _how_to_get_it()
        return 1

    # A common mistake is pointing at the folder *containing* the dataset
    # rather than the dataset root, since the archive nests one level deep.
    if not (data_dir / "Datasets").is_dir():
        nested = data_dir / "handwritingBCIData"
        if (nested / "Datasets").is_dir():
            print(f"FAIL  'Datasets/' not here, but found one level down.")
            print(f"      Use --data-dir {nested}")
            return 1
        problems.append("Datasets/ not found")

    # --- Partition file ---
    part = data_dir / "RNNTrainingSteps" / f"trainTestPartitions_{partition}.mat"
    if part.is_file():
        print(f"OK    partition file  {part.name}")
    else:
        problems.append(f"missing {part}")

    # --- Sessions ---
    sessions_dir = data_dir / "Datasets"
    sessions = sorted(p.name for p in sessions_dir.iterdir() if p.is_dir()) \
        if sessions_dir.is_dir() else []
    print(f"OK    sessions        {len(sessions)} found")

    if PRIMARY_SESSION not in sessions:
        problems.append(
            f"primary session {PRIMARY_SESSION} missing (experiments default to it)"
        )

    # --- Per-session files. Sentence data and HMM labels are what the
    #     frame-supervised decoders train on; single letters feed the aligner.
    hmm_dir = data_dir / "RNNTrainingSteps" / "Step2_HMMLabels" / partition
    usable = []
    for s in sessions:
        has_sent = (sessions_dir / s / "sentences.mat").is_file()
        has_lets = (sessions_dir / s / "singleLetters.mat").is_file()
        has_hmm = (hmm_dir / f"{s}_timeSeriesLabels.mat").is_file()

        if has_sent and has_hmm:
            usable.append(s)
        elif not has_sent:
            # Willett's t5.2019.06.03 is single-letters only. Expected, not a fault.
            notes.append(f"{s}: no sentences.mat (single-letter session)")
        elif not has_hmm:
            notes.append(f"{s}: sentences present but no HMM labels for {partition}")

        if s == PRIMARY_SESSION:
            for label, ok in [("sentences.mat", has_sent),
                              ("singleLetters.mat", has_lets),
                              (f"HMM labels ({partition})", has_hmm)]:
                if ok:
                    print(f"OK    {PRIMARY_SESSION}  {label}")
                else:
                    problems.append(f"{PRIMARY_SESSION} missing {label}")

    print(f"OK    usable sessions {len(usable)} (sentences + HMM labels)")

    if len(usable) < 2:
        problems.append(
            f"only {len(usable)} usable session(s); multi-session work needs more"
        )

    if notes:
        print("\nNotes (not failures):")
        for n in notes:
            print(f"  - {n}")

    if problems:
        print(f"\nFAILED with {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        _how_to_get_it()
        return 1

    print("\nAll required files present. Safe to run the experiments:")
    print(f"  python experiments/exp1_alignment_sensitivity.py --sweep "
          f"--data-dir {data_dir}")
    return 0


def _how_to_get_it() -> None:
    print("\n" + "-" * 68)
    print("Obtaining the dataset (one manual step, ~1.3 GB)")
    print("-" * 68)
    print(f"  1. Open {DATASET_DOI}")
    print(f"  2. Click 'Download dataset' and save {ARCHIVE}")
    print("     Automated download is blocked: the legacy file_stream URL")
    print("     returns 403 and the v2 API requires a bearer token.")
    print(f"  3. tar -xzf {ARCHIVE}        (tar ships with Windows 10 1803+)")
    print("  4. Re-run this script against the extracted folder.")
    print()
    print("Expected layout:")
    print("  handwritingBCIData/")
    print("    Datasets/<session>/sentences.mat, singleLetters.mat")
    print("    RNNTrainingSteps/Step2_HMMLabels/<partition>/"
          "<session>_timeSeriesLabels.mat")
    print("    RNNTrainingSteps/trainTestPartitions_<partition>.mat")
    print()
    print("Keep it outside the repo if the repo is in a synced folder such as")
    print("OneDrive or Dropbox; it expands to roughly 5.7 GB.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", default="./handwritingBCIData",
                   help="Dataset root (default ./handwritingBCIData)")
    p.add_argument("--partition", default="HeldOutTrials",
                   choices=["HeldOutTrials", "HeldOutBlocks"])
    args = p.parse_args()
    return check(Path(args.data_dir).expanduser(), args.partition)


if __name__ == "__main__":
    sys.exit(main())
