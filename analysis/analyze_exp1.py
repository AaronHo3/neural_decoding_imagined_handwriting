#!/usr/bin/env python3
"""
RQ1 analysis: dose-response tables, noise floor, and the H1c regime comparison.

Recomputes every RQ1 number cited in docs/RESEARCH_PLAN.md from the committed
artifacts, and writes results/exp1_summary.json so those numbers have a machine
-readable source rather than living only in prose. tests/test_results_integrity.py
checks the plan against this output.

The regime comparison (1500 vs 3000 bins) is restricted to MATCHED conditions:
the corruption endpoints p=0.0 and p=0.40, which are the only cells the A1
control ran. Comparing the full 1500-bin grid against the 18-run control would
mix different condition sets and is not a like-for-like contrast. The H1c figure
in make_figures.py uses the full grid instead, so the two scripts legitimately
report different spreads; both are labelled with their scope.

Usage:
    python analysis/analyze_exp1.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

DECODERS = ["gru", "rcnn", "conformer"]
ENDPOINTS = [0.0, 0.40]  # matched conditions for the regime comparison

REGIMES = [
    ("1500", "exp1_alignment_sensitivity", 1500),
    ("3000", "exp1_maxlen3000", 3000),
]


def load(subdir: str) -> dict:
    d = RESULTS / subdir
    g = defaultdict(list)
    if not d.exists():
        return g
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("FAILED_"):
            continue
        r = json.loads(p.read_text())
        g[(r["corruption_model"], r["corruption_level"], r["decoder"])].append(r)
    return g


def cers(runs: list) -> list:
    return [r["cer"] * 100 for r in runs]


def summarise_regime(g: dict, levels: list) -> dict:
    """Architecture spread, label effect and seed noise floor over `levels`."""
    arch_spreads, per_level = [], {}
    for lv in levels:
        means = {d: st.mean(cers(g[("corrupt", lv, d)]))
                 for d in DECODERS if g.get(("corrupt", lv, d))}
        if len(means) > 1:
            spread = max(means.values()) - min(means.values())
            arch_spreads.append(spread)
            per_level[str(lv)] = {"means": means, "arch_spread": spread}

    label_effects = {}
    for d in DECODERS:
        lo, hi = g.get(("corrupt", levels[0], d)), g.get(("corrupt", levels[-1], d))
        if lo and hi:
            label_effects[d] = st.mean(cers(hi)) - st.mean(cers(lo))

    seed_spreads = [max(cers(v)) - min(cers(v))
                    for k, v in g.items()
                    if k[0] == "corrupt" and k[1] in levels and len(v) >= 2]

    arch = st.mean(arch_spreads) if arch_spreads else float("nan")
    label = st.mean(label_effects.values()) if label_effects else float("nan")
    noise = st.mean(seed_spreads) if seed_spreads else float("nan")

    return {
        "per_level": per_level,
        "label_effects": label_effects,
        "mean_arch_spread": arch,
        "mean_label_effect": label,
        "seed_noise_floor": noise,
        "max_seed_spread": max(seed_spreads) if seed_spreads else float("nan"),
        "label_over_arch": (label / arch) if arch else float("nan"),
        "arch_below_noise": bool(arch < noise) if arch == arch else None,
        "n_runs": sum(len(v) for v in g.values()),
    }


def dose_response(g: dict) -> dict:
    """Full dose-response table for whichever corruption models are present."""
    out = {}
    for model in sorted({k[0] for k in g}):
        levels = sorted({k[1] for k in g if k[0] == model})
        rows = {}
        for lv in levels:
            entry = {}
            for d in DECODERS:
                runs = g.get((model, lv, d), [])
                if not runs:
                    continue
                c = cers(runs)
                entry[d] = {"mean": st.mean(c), "min": min(c), "max": max(c),
                            "n_seeds": len(c)}
            agree = [r["label_agreement_with_clean"]
                     for d in DECODERS for r in g.get((model, lv, d), [])]
            entry["_label_agreement"] = st.mean(agree) if agree else None
            rows[str(lv)] = entry
        out[model] = rows
    return out


def print_dose_response(name: str, table: dict) -> None:
    for model, rows in table.items():
        print(f"\n  {name} / {model}")
        print(f"    {'level':>7} {'agree':>7}  " +
              "".join(f"{d:>20}" for d in DECODERS))
        for lv, entry in rows.items():
            agree = entry.get("_label_agreement")
            line = f"    {lv:>7} {agree:>7.3f}  " if agree is not None else \
                   f"    {lv:>7} {'--':>7}  "
            for d in DECODERS:
                e = entry.get(d)
                line += (f"{e['mean']:>8.1f} [{e['min']:.1f}-{e['max']:.1f}]".rjust(20)
                         if e else f"{'--':>20}")
            print(line)


def main() -> int:
    summary = {"regimes": {}, "dose_response": {}}

    print("=" * 78)
    print("RQ1 ANALYSIS")
    print("=" * 78)

    for label, subdir, max_len in REGIMES:
        g = load(subdir)
        if not g:
            print(f"\n[{label} bins] no artifacts at results/{subdir}/ - skipped")
            continue

        s = summarise_regime(g, ENDPOINTS)
        s["max_len"] = max_len
        s["source"] = f"results/{subdir}"
        summary["regimes"][label] = s
        summary["dose_response"][label] = dose_response(g)

        print(f"\n[{label} bins]  {s['n_runs']} runs  ({s['source']})")
        print_dose_response(f"{label} bins", summary["dose_response"][label])
        print(f"\n    mean architecture spread : {s['mean_arch_spread']:6.2f} pp")
        print(f"    mean label effect        : {s['mean_label_effect']:6.2f} pp")
        print(f"    seed noise floor         : {s['seed_noise_floor']:6.2f} pp "
              f"(max {s['max_seed_spread']:.2f})")
        print(f"    label / architecture     : {s['label_over_arch']:6.2f}x")
        print(f"    -> architecture spread is "
              f"{'BELOW' if s['arch_below_noise'] else 'ABOVE'} the noise floor")

    # --- H1c verdict, applying the pre-committed rule from RESEARCH_PLAN A1 ---
    if "3000" in summary["regimes"]:
        s = summary["regimes"]["3000"]
        h1c = "SUPPORTED" if s["arch_below_noise"] else "REFUTED"
        summary["h1c_verdict_full_length"] = h1c
        summary["h1c_rule"] = (
            "architecture spread below seed noise floor at 3000 bins -> H1c stands; "
            "above -> H1c holds only in the truncated regime"
        )
        print("\n" + "=" * 78)
        print(f"H1c AT FULL SEQUENCE LENGTH: {h1c}")
        print("=" * 78)
        if "1500" in summary["regimes"]:
            a = summary["regimes"]["1500"]
            print(f"  {'':<26}{'1500 bins':>13}{'3000 bins':>13}")
            for lab, key in [("architecture spread", "mean_arch_spread"),
                             ("label effect", "mean_label_effect"),
                             ("seed noise floor", "seed_noise_floor"),
                             ("label / architecture", "label_over_arch")]:
                print(f"  {lab:<26}{a[key]:>13.2f}{s[key]:>13.2f}")

    out = RESULTS / "exp1_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
