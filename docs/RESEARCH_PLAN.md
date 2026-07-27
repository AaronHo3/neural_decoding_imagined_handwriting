# Research Plan

**Working title:** Label quality, not decoder architecture, is the binding constraint in handwriting BCI decoding

Aaron Ho · Plan version 1.0 · Written before the experiments below were run

---

## 0. Why this document exists

This plan is written *before* running the experiments it describes. Hypotheses, sample
sizes, metrics, and the analysis procedure are fixed in advance so that the results
cannot be reverse-engineered into a story after the fact. Where a prediction turns out
wrong, the plan is left unedited and the discrepancy is reported in `RESULTS.md`.

An earlier iteration of this repository ran an uncontrolled architecture comparison and
drew conclusions from single-run differences on a 10-sentence test set. Those results are
retained in `docs/RESULTS.md` as preliminary work, explicitly labelled as such. This plan
supersedes them.

---

## 1. Background and gap

Willett et al. (2021) decode imagined handwriting from intracortical recordings in
participant T5 at >90 characters/minute. Their pipeline has three stages:

1. **Forced alignment.** A Gaussian-emission HMM, fit to single-letter trials, is run over
   sentence recordings via Viterbi to assign a character label to every 10 ms time bin.
2. **Frame-level decoding.** A GRU is trained with cross-entropy on those frame labels.
3. **Language modelling.** An RNN language model rescores the character posteriors.

Stage 2 is supervised by the output of stage 1. The frame labels are *not ground truth*:
no one observed which character T5 was imagining at bin 4,217. They are the output of a
generative model fit to a different task (isolated letters) and transferred to a harder one
(continuous sentences). Every downstream decoder inherits whatever error stage 1 makes.

**The gap.** The literature has explored stage 2 extensively (architectures, augmentation,
regularisation) and stage 3 extensively (n-gram, RNN, and more recently transformer LMs).
Stage 1 is treated as fixed infrastructure. Nobody has published, for this paradigm:

- how much decoding error is attributable to alignment error rather than decoder capacity;
- how alignment error behaves when the aligner is transferred to a session it was not fit on;
- whether architecture rankings reported in the literature are stable under alignment noise,
  or whether they are artefacts of one particular label set.

This matters practically. Alignment must be re-derived for every new recording session,
using single-letter calibration data the participant has to sit through. If alignment is the
binding constraint, then calibration protocol design, not decoder architecture, is where
clinical effort should go.

**Central thesis.** In this paradigm, frame-label quality dominates decoder architecture.
Architecture differences that appear meaningful under clean labels shrink or invert under
realistic alignment noise, and cross-session alignment degradation accounts for more error
than any architectural choice tested.

---

## 2. Research questions and hypotheses

### RQ1: What is the alignment error budget?

*How does decoding accuracy respond to controlled degradation of the frame labels, and
where does the current pipeline sit on that curve?*

The Gaussian HMM labels from `Step2_HMMLabels` are treated as the reference condition. We
inject two kinds of controlled corruption, chosen to mimic the two ways forced alignment
actually fails:

- **Boundary jitter** (σ ∈ {0, 5, 10, 20, 40} frames). Each character-segment boundary is
  displaced by a Gaussian perturbation. Models the case where the aligner identifies the
  right character sequence but mislocates transitions.
- **Segment corruption** (p ∈ {0, 0.05, 0.10, 0.20, 0.40}). Each character segment's label
  is replaced with a uniformly random character with probability p. Models the case where
  the aligner assigns the wrong identity.

Both perturbations are applied to training labels only. Test-set evaluation always uses the
true prompt text, so the metric is unaffected by the perturbation.

**H1a.** CER degrades monotonically with both perturbation types.

**H1b.** The CER response to boundary jitter is *shallow* for σ up to roughly one third of
mean character duration (~25 frames), because the decoding pipeline's 51-frame smoothing
absorbs boundary noise. Segment corruption is *steep* from p = 0, because it injects
systematic rather than zero-mean error.

**H1c.** The between-architecture CER spread at any fixed perturbation level is smaller
than the within-architecture spread induced by moving one step along the corruption axis.
This is the quantitative form of the central thesis, and the primary result of RQ1.

**H1d.** Architecture *ranking* is not stable across perturbation levels. Specifically, the
convolutional front-end of the RCNN, which imposes local temporal smoothness, degrades more
gracefully under boundary jitter than the GRU does.

Falsification: if H1c fails, meaning architecture spread exceeds label-quality spread, the
central thesis is wrong and the project reverts to a (properly powered) architecture study.

### RQ2: How does alignment quality transfer across sessions?

*When the aligner is fit on session A's single-letter data and applied to session B's
sentences, how much accuracy is lost, and how much new-session calibration data recovers it?*

The dataset spans ten sessions from May 2019 to February 2020, with substantial neural drift
across that window. For a held-out target session, we compare adaptation strategies at
calibration budgets of k ∈ {0, 5, 10, 20, 40, 89} sentences:

- **S0: No adaptation.** Train on pooled source sessions, apply directly. Lower bound.
- **S1: Statistics recalibration.** Re-estimate per-channel z-scoring on the target
  session's calibration sentences; decoder weights frozen. Cheapest possible adaptation.
- **S2: Session-specific input layer.** Willett's approach: a per-session affine input
  transform is learned while the shared trunk stays frozen.
- **S3: Full fine-tune.** All weights updated on the calibration sentences.

**H2a.** S0 performs far worse than a within-session model, because cross-session drift is the
dominant error source, exceeding the multi-session pooling gains reported in the preliminary
results.

**H2b.** S2 dominates S3 at small k (k ≤ 20) because it has far fewer free parameters, and
S3 catches up or overtakes by k = 89.

**H2c.** S1 captures a substantial fraction of S2's benefit at k = 5, which would be a
practically useful finding: most of the drift is first-order channel statistics, not
representational change.

### RQ3: Does alignment-free training escape the constraint? (secondary)

CTC needs no frame labels, so it is immune to RQ1's corruption entirely. In the preliminary
work it performed worst, but it was also the only condition denied beam search and LM
rescoring, so the comparison was confounded.

**H3.** Given a matched decoding pipeline (same beam search, same LM), CTC is worse than
frame-supervised decoders at clean labels, but the gap closes as label corruption increases,
with a crossover somewhere in the tested range. If a crossover exists, its location is a
concrete recommendation: *below this alignment quality, stop using forced alignment.*

---

## 3. Experimental design

### Fixed choices

| Parameter | Value | Rationale |
|---|---|---|
| Primary session | t5.2019.05.08 | Matches preliminary work and Willett's examples |
| Partition | HeldOutTrials | Pre-specified by dataset authors |
| Seeds | 3 per condition (0, 1, 2) | Minimum for a variance estimate |
| Epochs | 80 | Matches preliminary work |
| Sequence length | 1500 bins (RQ1), 3000 (RQ2) | RQ1 trades length for run count |
| Primary metric | CER, mean over test sentences | Standard in the field |
| Reported statistic | mean ± range over seeds | Range, not SD, at n=3 |

### Statistical treatment

The test set is **10 sentences**. This is the hardest constraint on the project and it is
non-negotiable, since the partition is fixed by the dataset. Consequences, accepted in advance:

- No differences smaller than the seed-to-seed range will be claimed as real. Based on the
  preliminary runs, that floor is likely 3–5 CER points.
- Effects are reported as curves across a swept parameter, not as pairwise comparisons.
  A monotone trend across five perturbation levels is evidence even when adjacent points
  are individually indistinguishable; a single A-vs-B gap is not.
- Every figure shows individual seed points, not just means.
- Per-sentence CER is reported alongside the mean, since one pathological sentence can move
  a 10-sentence mean by several points.

This is why RQ1 and RQ2 are designed as dose-response sweeps rather than bake-offs. **The
sweep is the unit of evidence.** That design choice is a direct response to the sample-size
limit, and it is the main methodological improvement over the preliminary work.

### Compute budget

| Experiment | Runs | Est. per run | Total |
|---|---|---|---|
| RQ1 boundary jitter | 5 levels × 3 arch × 3 seeds = 45 | ~5 min | ~4 h |
| RQ1 segment corruption | 5 × 3 × 3 = 45 | ~5 min | ~4 h |
| RQ2 calibration sweep | 6 k × 4 strategies × 3 seeds = 72 | ~10 min | ~12 h |
| RQ3 CTC matched pipeline | 5 × 3 = 15 | ~5 min | ~1.5 h |

Roughly 22 GPU-hours on a T4, splittable across Colab sessions. Every run writes a JSON
artifact to `results/` immediately on completion, so partial progress is never lost and the
sweep can be resumed.

---

## 4. What gets committed

Reproducibility is a claim this project makes, so it has to be mechanically checkable:

- **Every run** emits `results/<experiment>/<condition>_seed<n>.json` containing metrics,
  the full config, the git SHA, the seed, wall-clock time, and the decoded strings for all
  10 test sentences. These are committed.
- **Every figure** is generated by `analysis/make_figures.py` from those JSON files alone.
  No hand-entered numbers anywhere in the pipeline.
- **Every number in the writeup** traces to a committed artifact. The writeup is checked by
  `tests/test_results_integrity.py`, which fails if a claimed metric has no backing file.

---

## 5. Relationship to prior work

This project does not attempt to beat Willett et al.'s 5.32% CER, and any comparison to that
number would be meaningless: it depends on an RNN language model, thousands of training
epochs, and an augmentation pipeline far beyond what is reproduced here. The absolute CERs
in this work are high and are not the contribution.

The contribution is the *shape of the response*: how decoding accuracy moves as label quality
and session distance are varied under otherwise matched conditions. That is measurable at
high absolute error, and it is a question the original work did not ask.

---

## 6. Known limitations, stated up front

1. **One participant.** T5 only. Nothing here establishes generality across subjects,
   implant sites, or aetiologies.
2. **Ten test sentences.** Addressed by sweep design, not solved. Every claim is bounded by it.
3. **Synthetic corruption is a proxy.** Boundary jitter and segment corruption are stylised
   models of alignment failure, not measured failure modes of the real aligner. RQ1 measures
   sensitivity to a *model* of alignment error.
4. **Bigram LM.** Weaker than Willett's RNNLM, which compresses the dynamic range of CER and
   may mask effects a stronger LM would reveal.
5. **No hyperparameter search.** Architectures are compared at reasonable but untuned
   settings. A tuned Conformer might behave differently; the claim is about the untuned
   regime a practitioner would actually start from.

---

## 7. Milestones

| # | Deliverable | Gate to proceed |
|---|---|---|
| M0 | Reproduce preliminary baseline with seeds and committed artifacts | 3 seeds agree within stated range |
| M1 | RQ1 boundary jitter sweep complete | H1a holds; monotone curve obtained |
| M2 | RQ1 segment corruption sweep + H1c evaluated | Central thesis supported or refuted |
| M3 | RQ2 calibration sweep complete | Curves separate the four strategies |
| M4 | RQ3 matched-pipeline CTC | Crossover located or ruled out |
| M5 | Writeup revised against plan; discrepancies documented | Every number traces to an artifact |

M2 is the decision point. If the central thesis is refuted there, the honest paper is
"architecture matters more than expected under label noise", which is also publishable, and
the plan is not rewritten to hide the prediction that failed.

---

## 8. Amendments after data collection

Everything above this line was fixed before any experiment ran. Anything added below was
decided *after* seeing results and is therefore exploratory, not confirmatory. It is
recorded here rather than folded into section 2 so the distinction stays visible.

### A1. Truncation control for RQ1 (added after the RQ1 sweep)

**Why.** The RQ1 sweep ran at `--max-len 1500` to afford 90 runs, as section 3 specified.
It returned a clean H1c result: architecture spread 1.64 pp against a seed noise floor of
2.32 pp, versus label-quality spread 6.21 pp. But baseline CER at clean labels came out at
77-79%, well above the 65.97% the preliminary work reached at 3000 bins. Truncating to 15
seconds cuts the tail off longer sentences, so predictions cannot cover the full reference
and CER is inflated.

That compresses the dynamic range for *both* effects, which raises an alternative
explanation for the headline result: architectures may be indistinguishable because there
is no headroom to distinguish them, not because they are genuinely equivalent.

**Design.** Corruption endpoints only (p = 0.0 and p = 0.40), all three decoders, three
seeds, at `--max-len 3000`. Eighteen runs. Endpoints alone are enough to measure
architecture spread with the ceiling lifted; the intermediate levels are not needed to
answer this specific question.

**Interpretation, fixed before running the control.** If architecture spread at 3000 bins
remains below the seed noise floor, H1c stands and the truncation objection is answered.
If architectures separate by more than the noise floor once the ceiling is lifted, then
H1c holds only in the truncated regime, and every claim from RQ1 must be qualified to that
regime. Neither outcome is edited into section 2 afterwards.

Run with `python experiments/exp1_alignment_sensitivity.py --control`.

### A2. RQ1 re-run at full sequence length (consequence of A1)

Following A1, the full pre-registered grid was re-run at `--max-len 3000` into
`results/exp1_sweep_3000`. 90/90 completed. This supersedes the 1500-bin grid as the
canonical RQ1 result; the 1500-bin grid is retained as evidence for the truncation confound.

Over the full grid: architecture spread **17.55 pp**, label-quality spread **9.41 pp**,
seed noise floor **4.17 pp**, ratio **0.54x**. H1c is refuted and inverted; architecture
dominates label quality roughly two to one, with both effects clearing the noise floor.

Hypothesis outcomes at full length, none of them edited above:

| | Outcome |
|---|---|
| H1a monotone degradation | Split: holds under identity corruption, not under jitter (jitter is flat within noise until the extreme) |
| H1b jitter shallow, corruption steep | **Supported, and stronger.** 2.4x to 10.8x steeper by decoder |
| H1c label quality dominates architecture | **Refuted and inverted** (0.54x) |
| H1d ranking unstable across levels | **Refuted.** RCNN < GRU < Conformer at 9 of 10 levels |

**Reproducibility check.** A1 and A2 independently ran the same 18 conditions. All 18 CERs
are bit-identical, so the pipeline is deterministic given a seed and the quoted noise floor
reflects initialisation variance alone.

**Outcome: the control refuted H1c.** 18/18 runs completed. Applying the rule above without
modification:

| | 1500 bins | 3000 bins |
|---|---|---|
| Mean architecture spread | 1.96 pp | **13.64 pp** |
| Mean label effect | 7.80 pp | 13.45 pp |
| Seed noise floor | 2.22 pp | 3.57 pp |
| Label / architecture ratio | 3.98x | **0.99x** |

At full sequence length the architectures separate enormously at clean labels: RCNN 67.3%,
GRU 74.2%, Conformer 86.5%, a 19.2 pp spread against a 3.6 pp noise floor. At 1500 bins that
same spread was 1.98 pp. Truncation was compressing CER toward a ceiling and destroying the
architecture signal, and the "architectures are indistinguishable" result was an artifact of
it.

Two consistency checks support this reading rather than a bug. RCNN's 67.3% at 3000 bins
reproduces the preliminary work's 65.97%, and the Conformer's poor showing reproduces the
preliminary single-session Conformer result (85.50%). The 1500-bin numbers, not the
3000-bin ones, are the anomaly.

**Consequences, applied as pre-committed:**

1. H1c is **refuted** in the full-length regime. The central thesis in section 1, that
   label quality dominates architecture, does not hold as stated. It is left unedited above.
2. What survives is weaker but still real: label quality has a large effect (13.45 pp, 3.8x
   the noise floor). It simply does not *dominate*; architecture is comparable in size.
3. Every RQ1 number from the 1500-bin sweep is quantitatively compromised. The dose-response
   *shape* may survive, but magnitudes are compressed by roughly 40% and the
   architecture comparison is invalid.
4. RQ1 must be re-run at `--max-len 3000` before any of it is written up.
5. Section 3's "1500 bins for RQ1" tradeoff was a mistake and is superseded.

**A methodological finding worth reporting in its own right:** sequence truncation is a
confound that can invert an architecture conclusion on this dataset. Anyone benchmarking
decoders on Willett's sentences at reduced sequence length may be measuring the ceiling
rather than the model.

**Caveat on the Conformer.** Its 86.5% at 3000 bins is worse than its 77.9% at 1500, which
is the opposite of the expected direction for more context. Self-attention over 3000 steps
at batch size 4 may be an optimisation failure rather than an architectural limit. This is
not resolved here and should not be reported as evidence about attention per se.
