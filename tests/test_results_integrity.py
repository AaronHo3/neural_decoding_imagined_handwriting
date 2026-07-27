"""
Every number claimed in the research plan must trace to a committed artifact.

docs/RESEARCH_PLAN.md section 4 promises exactly this check. Without it, the
plan's statistics are hand-transcribed prose that can silently drift from the
data as artifacts are added or re-run, which is the specific failure the
preliminary work in docs/RESULTS.md suffered from.

This recomputes the RQ1 statistics from results/ and asserts the plan's numbers
match. If a claim in the plan has no backing artifact, or the artifacts have
changed since the plan was written, this fails and names the discrepancy.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.analyze_exp1 import (  # noqa: E402
    ENDPOINTS,
    REGIMES,
    load,
    summarise_regime,
)

PLAN = REPO / "docs" / "RESEARCH_PLAN.md"

# (regime label, summary key, format spec, suffix as written in the plan)
CLAIMS = [
    ("1500", "mean_arch_spread", ".2f", " pp"),
    ("1500", "mean_label_effect", ".2f", " pp"),
    ("1500", "seed_noise_floor", ".2f", " pp"),
    ("1500", "label_over_arch", ".2f", "x"),
    ("3000", "mean_arch_spread", ".2f", " pp"),
    ("3000", "mean_label_effect", ".2f", " pp"),
    ("3000", "seed_noise_floor", ".2f", " pp"),
    ("3000", "label_over_arch", ".2f", "x"),
]

EXPECTED_RUNS = {"1500": 90, "3000": 18}


@pytest.fixture(scope="module")
def regimes():
    out = {}
    for label, subdir, _ in REGIMES:
        g = load(subdir)
        if g:
            out[label] = summarise_regime(g, ENDPOINTS)
    if not out:
        pytest.skip("no RQ1 artifacts committed yet")
    return out


@pytest.fixture(scope="module")
def plan_text():
    if not PLAN.exists():
        pytest.skip("research plan not found")
    return PLAN.read_text(encoding="utf-8")


def test_expected_run_counts(regimes):
    """A partial sweep would silently change every statistic below."""
    for label, expected in EXPECTED_RUNS.items():
        if label not in regimes:
            continue
        actual = regimes[label]["n_runs"]
        assert actual == expected, (
            f"{label}-bin regime has {actual} artifacts, expected {expected}. "
            "Re-run the sweep or update EXPECTED_RUNS."
        )


def test_plan_statistics_match_artifacts(regimes, plan_text):
    """Each statistic cited in the plan must equal the recomputed value."""
    missing = []
    for label, key, fmt, suffix in CLAIMS:
        if label not in regimes:
            continue
        value = regimes[label][key]
        rendered = f"{value:{fmt}}{suffix}"
        if rendered not in plan_text:
            missing.append(
                f"{label}-bin {key} = {rendered} (recomputed) not found in the plan"
            )

    assert not missing, (
        "The research plan cites statistics that do not match the artifacts.\n"
        "Re-run `python analysis/analyze_exp1.py` and update docs/RESEARCH_PLAN.md:\n  "
        + "\n  ".join(missing)
    )


def test_h1c_verdict_matches_the_precommitted_rule(regimes, plan_text):
    """
    Section 8 fixed the rule before the control ran: architecture spread below
    the seed noise floor at 3000 bins means H1c stands, above means it holds
    only in the truncated regime. The plan's stated outcome must follow it.
    """
    if "3000" not in regimes:
        pytest.skip("no full-length control artifacts")

    s = regimes["3000"]
    refuted = s["mean_arch_spread"] > s["seed_noise_floor"]

    if refuted:
        assert "refuted" in plan_text.lower(), (
            f"architecture spread ({s['mean_arch_spread']:.2f} pp) exceeds the noise "
            f"floor ({s['seed_noise_floor']:.2f} pp), so the plan must record H1c as "
            "refuted at full length"
        )
    else:
        assert "H1c **stands**" in plan_text or "H1c stands" in plan_text, (
            "architecture spread is below the noise floor, so the plan should "
            "record H1c as standing"
        )


def test_summary_json_is_current(regimes):
    """
    results/exp1_summary.json is the machine-readable source for the plan's
    numbers. If it drifts from the artifacts, the audit trail is broken.
    """
    import json

    path = REPO / "results" / "exp1_summary.json"
    if not path.exists():
        pytest.skip("run analysis/analyze_exp1.py to generate the summary")

    saved = json.loads(path.read_text())
    for label, s in regimes.items():
        assert label in saved["regimes"], f"{label} missing from exp1_summary.json"
        for key in ["mean_arch_spread", "mean_label_effect", "seed_noise_floor"]:
            a, b = s[key], saved["regimes"][label][key]
            assert abs(a - b) < 1e-6, (
                f"{label} {key}: artifacts give {a:.4f}, summary says {b:.4f}. "
                "Re-run analysis/analyze_exp1.py."
            )
