# Lab Notebook

Chronological record of what was done, why, and what it showed. Dated entries, appended
never rewritten. Where a decision turned out to be wrong, the original entry stays and a
later entry corrects it.

This complements the other two documents rather than duplicating them:

- `RESEARCH_PLAN.md` holds hypotheses and design, fixed before running.
- `results/*.json` hold the measurements.
- This file holds the **reasoning**: why each step was taken and what it changed.

---

## 2026-07-26 - Audit of the existing repository

Read the whole repo before changing anything. The project was a comparison of four decoder
architectures across three alignment strategies, with results reported as markdown tables.

**Implementations found to be genuine and correct:** GRU, RCNN, Conformer (faithful
Macaron-style blocks, not an `nn.Transformer` wrapper), CTC (CNN-BiLSTM with `nn.CTCLoss`),
Poisson HMM, Gaussian HMM, and the CER/WER metrics. Roughly 3,700 lines. This code was
reused unchanged and remains the backbone of the project.

**Defects found:**

1. **No evaluation results committed anywhere.** Every number lived only as prose in two
   markdown files. `run_benchmark.py` printed to stdout and never wrote to disk; the Colab
   notebook had all 11 cells cleared with `execution_count: null`. Nothing could be verified.
2. **No seeds anywhere.** No `np.random.seed` or `torch.manual_seed` in the repo, while
   augmentation and every training shuffle used `np.random`. No run was reproducible.
3. **CTC "frame accuracy" was fabricated.** Computed as `1 - CER`. Confirmed against the
   published tables: 100 - 85.75 = 14.25, reported as 14.3%.
4. **Poisson frame accuracy was circular.** `y_test_hard=gauss_hard_test` was passed
   unconditionally, so Poisson-trained models were scored against Gaussian labels. The
   writeup then cited that gap as proof Gaussian alignment was better.
5. **The Gaussian HMM class was never called.** All "Gaussian" results came from Willett's
   precomputed `charProbTarget`. Only the shared Viterbi helpers were exercised, via the
   Poisson path.
6. **n = 10 test sentences, one run per condition.** Every ranking claim rested on
   differences well inside the noise.

**Decision.** The interesting observation was buried as "finding #4": alignment quality
appeared to dominate architecture choice. An architecture bake-off is not a research
question, but that is. Reframed the project around it. Preliminary results retained at
`docs/RESULTS.md` with a status banner listing the defects, rather than deleted.

---

## 2026-07-26 - Infrastructure

Built the machinery needed for any claim to be checkable, before running anything.

- `docs/RESEARCH_PLAN.md`: three research questions, hypotheses, sweep grids, compute
  budget, falsification criteria. Written first so results cannot be reverse-engineered
  into a story.
- Seeding across `random`, `numpy`, `torch` via `--seed`.
- `--output` writes a JSON artifact per run: metrics, full config, seed, git SHA, and the
  decoded string for all 10 test sentences.
- `git_sha()` appends `-dirty` on a modified tree, since a bare SHA would claim provenance
  the code does not have.
- `analysis/make_figures.py` as the only path from artifacts to figures.
- `tests/` covering the metrics and the corruption models, the latter because they define
  the experiment's independent variable.
- CTC frame accuracy now reports `n/a`.

**Design decision that mattered most.** With 10 test sentences fixed by the dataset, no
pairwise comparison can be trusted. So every experiment is a **dose-response sweep**: a
monotone trend across five pre-specified levels is evidence even when adjacent points are
individually indistinguishable. The sweep, not the comparison, is the unit of evidence.

---

## 2026-07-27 - Environment and data

- Local GPU is an **RTX 4070 Ti SUPER (16 GB)**, roughly 4-5x a Colab T4. Abandoned the
  Colab plan; local has no session limits and no disconnect risk. The `handwriting` conda
  env already had torch 2.7.1+cu118 with CUDA working.
- Relaxed `numpy<2.0` to `numpy>=1.20` after confirming the repo runs on numpy 2.1.2.
- **The dataset was already on disk, twice**, from November 2025. A third copy got
  extracted by mistake into the PowerShell profile folder inside OneDrive, because the
  download script trusted `$PSScriptRoot`. Reclaimed about 8.3 GB. Dataset now lives at
  `C:\Users\aaron\handwritingBCIData\handwritingBCIData`, outside OneDrive, referenced by
  `--data-dir`.
- **Deleted both download scripts.** Verified that Dryad blocks automation: the legacy
  `file_stream` URL returns 403, the v2 API returns 401 without a bearer token. Neither
  script could download anything; they were extractors with misleading names. Replaced with
  `verify_data.py`, which checks that a dataset tree has the specific files the experiments
  open and names what is missing. That addresses the real failure mode, which is a wrong or
  truncated `--data-dir` failing hours into a sweep.

**Smoke test on real data.** The RQ1 script had passed 33 tests but never opened a `.mat`
file. One condition at 5 epochs: label agreement 0.914, CER 83.96%, frame accuracy 0.478
(chance is 1/31), predictions garbled but structured. Pipeline confirmed before committing
GPU hours.

---

## 2026-07-27 - RQ1 sweep at 1500 bins (90 runs)

All 90 completed, zero failures.

| | |
|---|---|
| Architecture spread | 1.96 pp |
| Label effect | 7.80 pp |
| Seed noise floor | 2.22 pp |

**H1c appeared supported.** Architecture spread sat *below* the seed noise floor, meaning
the differences between GRU, RCNN and Conformer were not measurable at all, while label
corruption moved CER by 7 to 9 pp.

**H1b supported, and cleanly.** At matched label damage, wrong-identity corruption cost
roughly two to three times what wrong-timing jitter cost (at ~19% of frames changed:
jitter +1.3 pp, corruption +4.6 pp). Consistent with the mechanism: 51-frame smoothing
absorbs boundary noise, a wrong character is unrecoverable.

**H1d partly failed.** The general claim held, rankings scrambled across levels. But the
specific prediction, that the RCNN's convolutional front-end would degrade more gracefully
under jitter than the GRU, was wrong in direction: GRU +2.9 pp, RCNN +4.5 pp.

**Concern raised immediately.** Clean-label CER came out at 77-79%, but the preliminary
work reached 65.97% for RCNN at 3000 bins. Truncating to 1500 bins cuts the tail off longer
sentences, so predictions cannot cover the full reference. That inflates CER and compresses
the dynamic range for *both* effects, which offers an alternative explanation for the
headline result: architectures might be indistinguishable because there is no headroom, not
because they are equivalent.

---

## 2026-07-27 - A1 truncation control (18 runs) - H1c REFUTED

Added as a post-hoc control, recorded in `RESEARCH_PLAN.md` section 8 with the
interpretation rule fixed **before** running it. Corruption endpoints only, all three
decoders, three seeds, at 3000 bins. 18/18 completed.

| | 1500 bins | 3000 bins |
|---|---|---|
| Architecture spread | 1.96 pp | **13.64 pp** |
| Label effect | 7.80 pp | 13.45 pp |
| Seed noise floor | 2.22 pp | 3.57 pp |
| Label / architecture | 3.98x | **0.99x** |

At full length the architectures separate enormously on clean labels: RCNN 67.3%,
GRU 74.2%, Conformer 86.5%. A 19.2 pp spread against a 3.6 pp noise floor, where the same
spread at 1500 bins was 1.98 pp.

**The truncation was destroying the architecture signal. The H1c result was an artifact.**

Two consistency checks say the 3000-bin numbers are the trustworthy ones, not a new bug.
RCNN's 67.3% reproduces the preliminary 65.97%, and the Conformer's poor showing reproduces
the preliminary single-session Conformer at 85.50%. The 1500-bin numbers are the anomaly.

**Consequences, applied as pre-committed:**

1. H1c is refuted at full length. The central thesis, that label quality dominates
   architecture, does not hold as stated. Left unedited in section 1.
2. What survives is weaker but real: label quality has a large effect (13.45 pp, 3.8x the
   noise floor). It does not dominate; architecture is comparable in size.
3. Every RQ1 number from the 1500-bin sweep is quantitatively compromised. The
   dose-response *shape* may survive; magnitudes are compressed by roughly 40% and the
   architecture comparison is invalid.
4. Section 3's "1500 bins for RQ1" tradeoff was a mistake and is superseded.

**A methodological finding in its own right.** Sequence truncation is a confound that can
invert an architecture conclusion on this dataset. Anyone benchmarking decoders on
Willett's sentences at reduced sequence length may be measuring the ceiling rather than the
model. The paired 1500/3000 evidence for this is committed.

**Caveat on the Conformer, revised.** First reading was that it got *worse* with more
context (77.9% to 86.5%), suggesting an optimisation failure at 3000 steps with batch size
4. On reflection that framing was wrong. At 1500 bins *everything* sat at the ceiling near
78%, so the Conformer's 77.9% was not a good score, it was a masked bad one. Lifting the
ceiling let GRU and RCNN improve to 74.2% and 67.3% while the Conformer stayed at 86.5%.
It did not degrade; it failed to benefit.

That reading is corroborated by the preliminary work, which independently found the
single-session Conformer at 85.50% and attributed it to overfitting on 89 sentences. Two
independent runs agreeing makes a genuine data-hunger effect more likely than a training
bug. Downgraded from a blocking caveat to open question 2; the full-length grid is worth
running with the Conformer included.

---

## 2026-07-27 - Capturing the analysis itself

Noticed that the statistics above were computed by throwaway scripts and hand-typed into
the plan, which is precisely the failure mode identified in the 2026-07-26 audit. Fixed:

- `analysis/analyze_exp1.py` recomputes every RQ1 statistic from the artifacts and writes
  `results/exp1_summary.json` as a machine-readable source.
- `tests/test_results_integrity.py` asserts the plan's numbers match the recomputed values,
  that run counts are complete, that the H1c verdict follows the pre-committed rule, and
  that the summary JSON has not drifted. This is the check `RESEARCH_PLAN.md` section 4
  promised but which did not previously exist.

Note on scope: the regime comparison uses matched conditions (corruption endpoints only),
since that is all the control ran. `make_figures.py` reports the H1c check over the full
grid instead, so the two scripts legitimately give different spreads. Both are labelled.

---

## 2026-07-27 - Full RQ1 grid at 3000 bins (90 runs) - the canonical result

Re-ran the whole pre-registered grid at full sequence length into
`results/exp1_sweep_3000`. 90/90 completed, zero failures, 7.7 h of training.

### Determinism check (free, from the 18 overlapping conditions)

The A1 control and this sweep independently ran the same 18 conditions. All 18 CERs are
**bit-identical**, maximum difference 0.000000 pp. The pipeline is fully deterministic
given a seed, which means the seed noise floor measures initialisation variance only, with
no GPU nondeterminism mixed in. Worth knowing before quoting any noise floor.

### H1c: refuted, and inverted

Over the full grid (all 10 level-by-model cells, both corruption models):

| | |
|---|---|
| Architecture spread | **17.55 pp** |
| Label-quality spread | 9.41 pp |
| Seed noise floor | 4.17 pp |
| Label / architecture | **0.54x** |

Architecture does not merely fail to be dominated by label quality; it dominates label
quality by roughly two to one. Both effects clear the noise floor (4.2x and 2.3x
respectively), so both are real. The central thesis in section 1 is wrong.

RCNN is best at every level in both corruption models, at 67.3% on clean labels versus
74.2% for GRU and 86.5% for the Conformer.

### H1b: supported, and much stronger at full length

This is now the most interesting result in the study. Clean-to-worst CER change:

| Decoder | Jitter | Corruption | Ratio |
|---|---|---|---|
| GRU | +5.5 pp | +18.3 pp | 3.3x |
| RCNN | +7.1 pp | +17.1 pp | 2.4x |
| Conformer | +0.5 pp | +5.0 pp | 10.8x |

The RCNN jitter curve is the striking part. Displacing boundaries so that 4%, 8% and 17%
of frames change their label moves CER by -1.0, -0.7 and -0.7 pp respectively: no effect at
all, within noise. Only at 31% of frames does it rise, by +7.1 pp. Identity corruption over
the same range costs +2.8, +4.6, +6.8 and +17.1 pp, rising immediately and roughly linearly.

**Alignment does not need to be temporally precise. It needs to be identity-correct.** For
a forced aligner that is an actionable design target: spend effort on getting the character
sequence right, not on tightening boundary placement.

### H1a: split by corruption model

Monotone under identity corruption for GRU and RCNN. Not monotone under jitter for any
decoder, but that is the H1b result restated rather than a failure: the jitter curve is
flat within noise until the extreme, so ordering among the flat points is arbitrary. The
Conformer is non-monotone under corruption too, because it sits near its own ceiling
throughout.

### H1d: refuted

H1d predicted the ranking would be unstable across perturbation levels. At full length it
is stable: RCNN < GRU < Conformer at nine of ten levels. The single exception is maximum
corruption (p = 0.40), where the GRU degrades worst of all and the Conformer edges past it.
The apparent instability at 1500 bins was noise around a compressed mean, not a real
interaction. Both the general claim and its specific mechanism are now refuted.

### Caveat

One condition (RCNN, corruption p = 0.05) has an 11.4 pp seed spread, far above the 4.17 pp
mean. With three seeds and ten test sentences, individual cells remain unstable even though
the sweep-level trends are clear. Quote trends, not cells.

---

## 2026-07-27 - Analysis pipeline corrected

`analyze_exp1.py` was reporting endpoint-only statistics for every regime, since that is
what the A1 control could support. Applied to the full 3000-bin grid this understated the
architecture effect as 13.64 pp instead of 17.55 pp. Added `summarise_full_grid()` and
`error_type_asymmetry()`, so a regime with the complete grid now reports both the
endpoint-matched numbers (for cross-regime comparison) and the full-grid numbers (the ones
to quote for that regime alone). `make_figures.py` now emits one figure per result set.

---

## Where RQ1 stands

Three findings, in decreasing order of how well they are supported:

1. **Alignment error *type* matters more than alignment error *rate*.** Timing errors are
   nearly free up to roughly 17% of frames; identity errors cost immediately. Survived the
   regime change and got cleaner. This is the substantive contribution.
2. **Sequence truncation is a confound that can invert an architecture conclusion** on this
   dataset. Demonstrated with paired 1500 and 3000-bin grids, 180 runs total. This is the
   methodological contribution.
3. **Label quality does not dominate architecture.** The original thesis is refuted and in
   fact inverted, 0.54x. Reported as a negative result.

---

## Open questions

1. **Re-run RQ1 at 3000 bins.** The full grid, roughly 4 to 6 hours. Required before any of
   RQ1 is written up. Does the H1b jitter-versus-corruption asymmetry survive at full
   length, or was it also a truncation effect?
2. **Is the Conformer result an optimisation failure?** Test with gradient accumulation or
   a shorter attention window before drawing any conclusion about attention.
3. **Does the label effect hold once architecture is free to vary?** At 3000 bins the two
   effects are the same size. The interesting question becomes whether they interact.
4. **RQ2 and RQ3 remain unrun.** RQ2 in particular should use 3000 bins from the start.
