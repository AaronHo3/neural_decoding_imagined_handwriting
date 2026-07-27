# Alignment Error Type, Not Error Rate, Limits Decoder Accuracy in Handwriting BCI

Aaron Ho

Every number in this document is recomputed from committed artifacts by
`analysis/analyze_exp1.py`. `tests/test_results_integrity.py` fails if any of them drift.

---

## Abstract

Brain-to-text decoding from intracortical recordings depends on a supervision signal that is
not ground truth. In the pipeline established by Willett et al. (2021), a hidden Markov model
performs forced alignment to assign a character label to every 10 ms time bin, and the decoder
is then trained on those labels. Nobody observed which character the participant was actually
imagining at any given bin, so every downstream decoder inherits whatever error the aligner
makes. The literature has explored decoder architectures and language models extensively while
treating the alignment stage as fixed infrastructure.

I measure how much decoding accuracy depends on that stage by injecting controlled corruption
into the frame labels and observing the response, across two corruption models, five severity
levels, three decoder architectures, three seeds and two sequence-length regimes: 198 training
runs in total. The headline result is an asymmetry. Displacing character boundaries so that up
to 17% of frame labels change has no measurable effect on the best decoder, while relabelling
the same proportion of frames to the wrong character costs 6.8 percentage points of character
error rate and rises steeply thereafter. Corruption is 2.4 to 10.8 times steeper than jitter
depending on architecture. **A forced aligner does not need to be temporally precise; it needs
to be identity-correct.**

The study also contains a cautionary methodological result. Run at truncated 1500-bin
sequences, decoder architectures appeared statistically indistinguishable, with a 1.64 pp
spread against a 2.32 pp seed noise floor. Re-run at full 3000-bin length, the identical grid
produced a 17.55 pp architecture spread. Truncation compresses character error rate toward a
ceiling and can invert an architecture conclusion.

The pre-registered hypothesis, that label quality dominates architecture, was refuted and in
fact inverted: architecture spread exceeds label-quality spread by roughly two to one at full
sequence length. That prediction is reported unmodified.

---

## 1. Introduction

Willett et al. (2021) demonstrated brain-to-text communication at over 90 characters per
minute by decoding imagined handwriting from two 96-channel intracortical arrays in the motor
cortex of a participant with tetraplegia. Their pipeline has three stages: a Gaussian-emission
HMM performs forced alignment of sentence recordings against the prompted text, producing a
character label for every 10 ms bin; a recurrent network is trained with cross-entropy on
those frame labels; and a language model rescores the resulting character posteriors.

Stage two is supervised entirely by the output of stage one, and the frame labels stage one
produces are estimates rather than observations. They come from a generative model fit to a
different task, isolated single letters, and transferred to a harder one, continuous
sentences. Nobody knows the true label of any individual bin.

Subsequent work has explored stage two extensively, comparing recurrent, convolutional and
attention-based decoders, and stage three extensively, moving from n-gram to neural language
models. Stage one is generally adopted as given. For this paradigm the literature does not
report how much decoding error is attributable to alignment error rather than decoder
capacity, nor which *kind* of alignment error is expensive.

That second question turns out to be the useful one. Forced alignment can fail in two
distinct ways: it can identify the correct character sequence but mislocate the transitions
between characters, or it can assign the wrong character identity altogether. These are
different failure modes with different plausible consequences, and they suggest different
priorities for anyone designing an aligner.

This study measures the decoding response to both, under otherwise matched conditions.

---

## 2. Methods

### 2.1 Data

All experiments use session `t5.2019.05.08` of the publicly available Willett et al. dataset
from Dryad, with the `HeldOutTrials` partition specified by the dataset authors: **89 training
sentences and 10 held-out test sentences**. Neural activity is threshold-crossing spike counts
in 10 ms bins across 192 channels. The vocabulary is 31 characters (26 letters plus space,
comma, apostrophe, period and question mark). Spike counts are z-scored per channel using
training-set statistics.

Frame-level character labels come from Willett's precomputed `Step2_HMMLabels`, which are the
output of their Gaussian HMM forced alignment. These serve as the **reference condition**: the
best available labels, treated as the zero-corruption baseline. They are not treated as
ground truth, because they are not.

### 2.2 Decoders

Three frame-supervised architectures, trained with cross-entropy on frame labels, 80 epochs,
Adam or AdamW:

| Decoder | Architecture |
|---|---|
| **GRU** | 2 layers, 256 hidden units. The Willett baseline. |
| **RCNN** | Conv1D blocks (32, 64, 128), kernel 5, then 2 GRU layers of 256. |
| **Conformer** | 4 Macaron-style blocks, d_model 128, 4 heads, depthwise conv kernel 15, FF 512, warmup plus cosine schedule, batch size 4 for attention memory. |

Training data is augmented with two additive-Gaussian-noise copies per sentence.

### 2.3 Controlled label corruption: the measuring instrument

The central methodological device is that **corruption is an instrument, not an attempt to
improve anything**. Ground truth for frame labels does not exist, so the absolute cost of
alignment error cannot be measured directly. What can be measured is the *derivative*: degrade
the labels by a known amount and observe the response.

Two corruption models were chosen to mirror the two ways forced alignment actually fails.

**Boundary jitter** (σ ∈ {0, 5, 10, 20, 40} frames). Each character-segment boundary is
displaced by a Gaussian perturbation, then boundaries are repaired to stay monotone and
non-overlapping. Character identity and order are preserved; only transition times move. This
models an aligner that reads the sequence correctly but mislocates when each character begins.

**Segment corruption** (p ∈ {0, 0.05, 0.10, 0.20, 0.40}). Each character segment is relabelled
to a uniformly random different character with probability p. Boundaries are preserved exactly;
only identities change. This models an aligner that assigns the wrong character.

Both are applied to **training labels only**. Test evaluation always scores against the true
prompt text, so the metric itself is never perturbed and remains comparable across all
conditions. Each run records the *measured* fraction of frames whose label actually changed,
so the x-axis is grounded in an observed quantity rather than only the nominal knob setting.
This matters because the two models produce different frame-change rates at their nominal
settings, and comparisons between them are made at matched measured damage.

The corruption functions are the experiment's independent variable, so they are covered by
unit tests asserting that jitter preserves the character sequence and never extends beyond the
original span, and that corruption preserves boundaries exactly and never touches padding.

### 2.4 Decoding and metrics

Frame logits are converted to text by softmax probability smoothing over a 51-frame window,
temporal collapse of constant-argmax runs, minimum-duration filtering, and beam search
rescored by a character bigram language model trained on the training prompts. Character error
rate is Levenshtein distance normalised by reference length, averaged over the 10 test
sentences.

### 2.5 Design: why a sweep rather than a comparison

The test set contains **10 sentences**, fixed by the dataset partition and not negotiable.
This is the hardest constraint on the study, and it drove the design.

With 10 sentences and 3 seeds, no pairwise comparison is trustworthy: the measured seed-to-seed
spread is 4.17 pp at full length, and one pathological sentence moves the mean by several
points. The response is to make **the sweep, rather than the comparison, the unit of
evidence**. A monotone trend across five pre-specified severity levels, reproduced across three
seeds and three architectures, is evidence even when adjacent points are individually
indistinguishable, because noise does not produce monotone trends in a pre-specified direction.
A single A-beats-B gap is not evidence.

Every figure shows individual seed points rather than means alone, and no difference smaller
than the seed spread is claimed as real.

Hypotheses, sweep grids and falsification criteria were fixed in writing before any experiment
ran (`docs/RESEARCH_PLAN.md`).

---

## 3. Results

Two complete grids were run: 90 runs at 1500-bin truncated sequences and 90 runs at full
3000-bin sequences, plus an 18-run control. The **3000-bin grid is the canonical result**;
section 3.3 explains why the 1500-bin grid is reported separately rather than discarded.

### 3.1 Error type asymmetry (the main result)

Character error rate change from clean labels to maximum corruption, at full sequence length:

| Decoder | Boundary jitter | Identity corruption | Corruption / jitter |
|---|---|---|---|
| GRU | +5.5 pp | +18.3 pp | **3.3x** |
| RCNN | +7.1 pp | +17.1 pp | **2.4x** |
| Conformer | +0.5 pp | +5.0 pp | **10.8x** |

The per-level detail for the RCNN, the best decoder, is the clearest form of the result.
Percentages are the *measured* fraction of frame labels changed:

| Frames changed | Jitter, ΔCER | Corruption, ΔCER |
|---|---|---|
| 4-5% | **-1.0 pp** | +2.8 pp |
| 8-10% | **-0.7 pp** | +4.6 pp |
| 17-20% | **-0.7 pp** | +6.8 pp |
| 31-40% | +7.1 pp | +17.1 pp |

Displacing boundaries so that up to 17% of frame labels change has **no measurable effect**:
the three deltas are negative and well inside the 4.17 pp noise floor. Only when 31% of frames
are displaced does error rise. Corrupting identities over the same range costs accuracy
immediately and roughly linearly, then accelerates.

The mechanism is straightforward. The decoding pipeline smooths posteriors over a 51-frame
window before collapsing runs into characters, and mean character duration is roughly 75
frames. A boundary displaced by 10 or 20 frames is absorbed by that smoothing; the character
sequence still collapses correctly. A character labelled `q` when it should be `a` is
unrecoverable at any smoothing width, and it also teaches the network a wrong neural-to-
character mapping that generalises to test data.

The Conformer's 10.8x ratio should be read with care: it is a poor decoder here in absolute
terms (86.5% CER on clean labels) and sits near its own ceiling throughout, so both deltas are
compressed and the ratio is inflated. The GRU and RCNN ratios are the trustworthy ones.

### 3.2 Architecture effects

At full sequence length, on clean labels: **RCNN 67.3%, GRU 74.2%, Conformer 86.5%** CER. The
RCNN is best at 9 of 10 corruption levels across both models. The single exception is maximum
identity corruption, where the GRU degrades worst of all three (92.5%) and the Conformer edges
past it.

The RCNN's convolutional front-end extracting local temporal structure before recurrent
integration appears well matched to this task. The Conformer's poor showing is consistent with
the known data-hunger of attention: with 89 training sentences it has insufficient diversity,
and this reproduces an independent earlier observation on the same session (85.50% CER).

### 3.3 The truncation confound

The 1500-bin grid was run first, trading sequence length for run count. It produced a
qualitatively different and **incorrect** picture:

| | 1500 bins | 3000 bins |
|---|---|---|
| Architecture spread | 1.64 pp | **17.55 pp** |
| Label-quality spread | 6.21 pp | 9.41 pp |
| Seed noise floor | 2.32 pp | 4.17 pp |
| Label / architecture | 3.77x | **0.54x** |

At 1500 bins the architecture spread sat *below* the seed noise floor: the three decoders
appeared statistically indistinguishable. At 3000 bins the same grid separates them by 17.55
pp, more than four times the noise floor.

Truncating to 1500 bins (15 seconds) cuts the tail off longer test sentences, so predictions
cannot cover the full reference and CER is inflated toward a ceiling near 78% for every
condition. That ceiling compresses all effects and destroys the architecture signal.

The confound was caught because clean-label CER (77-79%) disagreed with an earlier
observation on the same data at full length (65.97% for the RCNN). A control was then run at
3000 bins with the interpretation rule fixed in advance, and it inverted the conclusion. The
1500-bin grid is retained as the paired evidence for this effect rather than discarded.

**Benchmarks on this dataset run at reduced sequence length may be measuring the ceiling
rather than the model.**

### 3.4 Hypothesis outcomes

| Hypothesis | Prediction | Outcome at full length |
|---|---|---|
| H1a | CER degrades monotonically with both corruption types | **Split.** Monotone under identity corruption for GRU and RCNN; not under jitter, because the jitter curve is flat within noise until the extreme |
| H1b | Jitter shallow, corruption steep | **Supported, and stronger than predicted** (2.4x to 10.8x) |
| H1c | Label quality dominates architecture | **Refuted and inverted** (0.54x) |
| H1d | Architecture ranking unstable across levels | **Refuted.** RCNN best at 9 of 10 levels |

H1c was the central thesis. Both effects clear the noise floor, so both are real, but
architecture dominates label quality by roughly two to one. The apparent instability behind
H1d at 1500 bins was noise around a compressed mean rather than a genuine interaction.

### 3.5 Reproducibility

The 18 conditions shared between the control and the full grid were run independently, six
hours apart, at different commits, into different output directories. They produced
**bit-identical** results: identical character error rate to full float precision, identical
word error rate, identical frame accuracy, and identical decoded strings for all 10 test
sentences in all 18 conditions. Training wall-clock differed between the two runs (for
example 546.2 s versus 554.7 s for the same Conformer condition), confirming the work was
genuinely re-executed rather than cached. Across the grid, different seeds do produce
different error rates in 30 of 30 conditions, so seeding is doing real work rather than
collapsing the runs.

The quoted seed noise floor therefore reflects weight-initialisation variance alone, with no
run-to-run hardware nondeterminism mixed in.

**Caveat on the scope of this claim.** `torch.use_deterministic_algorithms` and
`torch.backends.cudnn.deterministic` are *not* set in `set_seed()`. Determinism here was
observed empirically on one machine (RTX 4070 Ti SUPER, torch 2.7.1+cu118) with fixed batch
shapes, not enforced by configuration. A different GPU, driver, cuDNN version or batch shape
could select different kernels and break it. The correct reading is that these particular
results reproduce exactly on this configuration, not that the code carries a portable
determinism guarantee.

---

## 4. Discussion

### 4.1 What this means for aligner design

The practical implication is a reordering of priorities. Effort spent tightening boundary
placement in a forced aligner buys little: the decoder tolerates roughly 17% of frames being
mistimed with no measurable loss. Effort spent ensuring the aligner recovers the correct
character sequence buys a great deal, since identity errors cost immediately and compound.

For HMM-based alignment specifically, this argues for prioritising emission-model quality and
template estimation, which determine which character wins at each frame, over transition-model
tuning, which determines where boundaries fall. It also suggests that alignment quality
metrics reported purely as boundary displacement may be measuring the wrong thing.

### 4.2 Truncation as a benchmarking hazard

The 1500-bin result was not a measurement error. It was a correct measurement of a regime that
does not generalise, and the inference from that regime to the general case was wrong. The
distinction matters, because the failure is silent: every individual number was reproducible,
the seeds agreed, the trends were monotone, and the conclusion was still an artifact.

The lesson generalises beyond this dataset. Whenever a metric is bounded and the experimental
configuration pushes results toward that bound, effect sizes compress and differences vanish
in a way that looks like a null result. Checking that a baseline condition agrees with an
independently obtained value for the same condition is a cheap guard, and it is what caught
this.

### 4.3 Why the original hypothesis failed

H1c was motivated by the observation that alignment is unglamorous infrastructure that the
field has not scrutinised. That motivation was sound; the quantitative claim attached to it
was not. Label quality does have a large effect, 9.41 pp of spread at 3.8 times the noise
floor. It simply is not larger than the effect of choosing a different architecture, which is
17.55 pp.

The finding that survived is more specific and more useful than the one predicted. "Label
quality matters more than architecture" would have been a claim about relative magnitudes,
sensitive to which architectures were in the comparison set. "Timing errors are nearly free
while identity errors are expensive" is a claim about mechanism, and it held across every
architecture tested and strengthened when the truncation confound was removed.

---

## 5. Limitations

1. **One participant, one session.** All results come from session `t5.2019.05.08` of
   participant T5. Nothing here establishes generality across subjects, implant sites, or
   sessions.
2. **Ten test sentences.** Addressed by sweep design, not solved. Individual cells remain
   unstable: one condition (RCNN at p = 0.05) shows an 11.4 pp seed spread against a 4.17 pp
   mean. Trends are reportable; individual cells are not.
3. **Synthetic corruption is a proxy.** Boundary jitter and segment corruption are stylised
   models of alignment failure, not measured failure modes of a real aligner. This study
   measures sensitivity to a *model* of alignment error, and a real aligner's errors are
   likely correlated with the neural data in ways this does not capture.
4. **The reference condition is not ground truth.** Zero corruption means Willett's HMM
   labels, which contain unknown error of their own. All effects are measured relative to
   that baseline, so the true cost of alignment error is underestimated by whatever the
   baseline already contains.
5. **Weak language model.** A character bigram, far weaker than Willett's RNN language model.
   A stronger LM would lower absolute CER and might change the dynamic range over which these
   effects are visible.
6. **No hyperparameter search.** Architectures were compared at reasonable but untuned
   settings. The Conformer in particular may be undertrained rather than fundamentally
   unsuited; its result should not be read as evidence about attention mechanisms in general.
7. **Absolute error rates are high** and are not the contribution. No comparison to Willett's
   5.32% CER is meaningful, since that depends on an RNN language model, thousands of epochs
   and an augmentation pipeline not reproduced here.

---

## 6. Reproducing this

```bash
python verify_data.py --data-dir /path/to/handwritingBCIData
python experiments/exp1_alignment_sensitivity.py --sweep --max-len 3000 \
    --out-dir results/exp1_sweep_3000 --data-dir /path/to/handwritingBCIData
python analysis/analyze_exp1.py
python analysis/make_figures.py --experiment exp1
pytest tests/ -q
```

Roughly 7.7 hours of training on an RTX 4070 Ti SUPER. The sweep is resumable and skips
conditions whose artifact already exists. All 180 run artifacts are committed under
`results/`, each carrying its metrics, full configuration, seed, git SHA and the decoded
string for all 10 test sentences.

---

## 7. Future directions

**Cross-session calibration efficiency.** The clinically important question this project does
not answer. Intracortical recordings drift over weeks; the dataset spans nine months across
eleven sessions. When a decoder trained on prior sessions meets a new one, how little
new-session data is needed to recover accuracy, and which adaptation strategy is most
efficient at small budgets? A natural design compares no adaptation, per-channel statistics
recalibration, a session-specific affine input layer (Willett's approach), and full
fine-tuning, across calibration budgets from 0 to 89 sentences. If statistics recalibration
alone captures most of the benefit at 5 sentences, that would mean drift is largely
first-order and roughly two minutes of calibration would suffice, which is a meaningful
quality-of-life result for someone using such a device daily. This was planned and descoped;
it is a separate study rather than an extension of this one.

**Alignment-free training.** CTC requires no frame labels and is therefore immune to the
corruption studied here. Given a *matched* decoding pipeline, the same beam search and
language model afforded to the frame-supervised decoders, does a label-quality threshold exist
below which abandoning forced alignment is the better choice? Section 3.1 implies the
threshold, if it exists, is governed by identity error rate rather than boundary precision.
This is the most direct extension of the present work and reuses its harness.

**Measuring a real aligner's error modes.** The corruption models here are stylised. Running
the Poisson and Gaussian HMM aligners implemented in this repository against Willett's
reference labels would characterise how real alignment errors are distributed between timing
and identity, and therefore where on the curves in section 3.1 a real aligner actually sits.

**Resolving the Conformer.** Its poor performance is corroborated by an independent run, but
undertraining has not been excluded. Gradient accumulation to raise the effective batch size,
a longer warmup, or a restricted attention window would separate a data-hunger explanation
from an optimisation one.

---

## 8. Acknowledgements

This work is built entirely on data and prior methodology from Willett et al. (2021), and the
dependency runs deeper than reuse of a public dataset.

The zero-corruption reference condition in every experiment here **is Willett et al.'s own
forced-alignment output**, taken from the `Step2_HMMLabels` directory of their published
release. Those labels are a derived research product representing substantial methodological
work: fitting Gaussian emission templates from single-letter trials, Viterbi alignment of
sentence recordings, and session-specific calibration. This study does not reproduce that
pipeline; it treats its output as the baseline against which controlled degradation is
measured. Every result reported here is therefore measured *relative to their alignment*, and
the study could not exist without it.

The dataset is released under CC0, which waives copyright but does not waive scholarly
credit. Both the primary article and the dataset are cited separately below, as Dryad's
independent DOI intends.

Character class definitions follow `characterDefinitions.py` from the authors' reference
implementation. The decoder architectures, corruption models, analysis and evaluation code in
this repository are original to this project.

---

## 9. References

### Primary source

- Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., & Shenoy, K. V. (2021).
  High-performance brain-to-text communication via handwriting. *Nature*, 593(7858), 249-254.
  https://doi.org/10.1038/s41586-021-03506-2
- Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., & Shenoy, K. V. (2021).
  *Data from: High-performance brain-to-text communication via handwriting* [Dataset]. Dryad.
  https://doi.org/10.5061/dryad.wh70rxwmv
- Willett, F. R. (2021). *handwritingBCI* [Software]. GitHub.
  https://github.com/fwillett/handwritingBCI

### Architectures and methods

- Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., &
  Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical
  machine translation. *EMNLP 2014*, 1724-1734. (GRU)
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*,
  9(8), 1735-1780. (BiLSTM used in the CTC decoder)
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., &
  Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*, 5998-6008.
  (Multi-head self-attention and sinusoidal positional encoding)
- Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang,
  Z., Wu, Y., & Pang, R. (2020). Conformer: Convolution-augmented Transformer for Speech
  Recognition. *Interspeech 2020*, 5036-5040.
- Graves, A., Fernandez, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal
  classification: labelling unsegmented sequence data with recurrent neural networks.
  *ICML 2006*, 369-376.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
  (AdamW, used for the Conformer)
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions and
  reversals. *Soviet Physics Doklady*, 10(8), 707-710. (Edit distance underlying CER and WER)

### Software

- Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning
  library. *NeurIPS 2019*, 8024-8035.
- Harris, C. R. et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362.
- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in
  Python. *Nature Methods*, 17, 261-272.
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science &
  Engineering*, 9(3), 90-95.
