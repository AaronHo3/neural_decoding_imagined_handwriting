# Is alignment quality, not decoder architecture, the binding constraint in handwriting BCI decoding?

Brain-to-text decoding from intracortical recordings, building on
[Willett et al. (Nature 2021)](https://www.nature.com/articles/s41586-021-03506-2).

The Willett pipeline trains its decoder on frame-level character labels produced by a
Gaussian HMM forced aligner. Those labels are **not ground truth**: nobody observed which
character the participant was imagining at a given 10 ms bin. They are the output of a
generative model fit to isolated letters and transferred to continuous sentences, and every
downstream decoder inherits whatever error they contain.

The literature has explored decoder architectures and language models extensively. The
alignment stage is treated as fixed infrastructure. **This project asks how much of the
decoding error that stage is actually responsible for.**

📄 **[Research plan](docs/RESEARCH_PLAN.md)**: questions, hypotheses, and analysis
procedure, written before the experiments were run.

---

## Status

| | |
|---|---|
| **Research plan** | Complete, hypotheses and analysis fixed in advance |
| **Infrastructure** | Complete: 4 decoders, 2 aligners, seeded, artifact-emitting, 33 tests passing |
| **RQ1** alignment error budget | Experiment implemented; sweep not yet run |
| **RQ2** cross-session calibration | Not yet implemented |
| **RQ3** alignment-free (CTC) crossover | Not yet implemented |
| **Preliminary results** | [`docs/RESULTS.md`](docs/RESULTS.md), exploratory only, see caveats |

This README makes no accuracy claims. Results appear here only once they are backed by
committed artifacts in `results/` and reproduced across three seeds. Earlier exploratory
numbers are in [`docs/RESULTS.md`](docs/RESULTS.md), labelled with their known defects.

---

## Research questions

**RQ1: What is the alignment error budget?** Inject controlled corruption into the frame
labels (boundary jitter; segment relabelling) and measure the decoding response. The
central hypothesis is that the CER spread induced by moving one step along the label-quality
axis exceeds the spread between architectures at any fixed label quality.

**RQ2: How does alignment quality transfer across sessions?** The dataset spans ten
sessions over nine months, with substantial neural drift. How much calibration data does a
new session need, and which adaptation strategy (statistics recalibration, a session-specific
input layer, or full fine-tuning) is most data-efficient at small budgets?

**RQ3: Does alignment-free training escape the constraint?** CTC needs no frame labels and
so is immune to RQ1's corruption. Given a *matched* decoding pipeline, is there a label-quality
threshold below which abandoning forced alignment is the better choice?

Hypotheses, sweep grids, and falsification criteria are in
[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).

---

## Methodological note: 10 test sentences

The dataset's held-out partition contains ten test sentences, and this is fixed by the
dataset authors. Consequently every experiment here is designed as a **dose-response sweep
rather than a pairwise comparison**. A monotone trend across five corruption levels is
evidence; a single A-beats-B gap on ten sentences is not. All results report three seeds
with individual points shown, and no difference smaller than the seed-to-seed range is
claimed as real.

This constraint drove the experimental design, and it is the main methodological difference
from the preliminary work.

---

## Repository layout

```
├── docs/
│   ├── RESEARCH_PLAN.md      # questions, hypotheses, design (written first)
│   └── RESULTS.md            # preliminary writeup (caveated)
├── experiments/              # one script per research question
│   └── exp1_alignment_sensitivity.py
├── analysis/
│   └── make_figures.py       # the ONLY path from artifacts to figures
├── results/                  # committed JSON, one per run (config + git SHA + seed)
├── figures/                  # generated; never hand-edited
├── tests/                    # 33 tests
├── verify_data.py            # check a dataset tree before spending GPU time
│
├── alignment/                # forced alignment
│   ├── gaussian_hmm.py       # Gaussian emissions + shared log-domain Viterbi
│   └── poisson_hmm.py        # Poisson / negative-binomial emissions
├── decoders/                 # shared BaseDecoder interface
│   ├── rnn_decoder.py        # GRU (Willett baseline)
│   ├── rcnn_decoder.py       # Conv1D + GRU
│   ├── transformer_decoder.py # Conformer (Gulati et al. 2020)
│   └── ctc_decoder.py        # CNN-BiLSTM + CTC
├── data/                     # Willett dataset loader, preprocessing
├── benchmarks/               # CER / WER / Levenshtein
└── run_benchmark.py          # baseline architecture sweep
```

---

## Reproducibility

Every run emits a JSON artifact containing its metrics, full config, git SHA, seed, and the
decoded string for all ten test sentences. Figures are generated from those artifacts alone
by `analysis/make_figures.py`. No number is transcribed by hand at any point.

### Getting the data

One manual step. Dryad blocks automated download (the legacy `file_stream` URL returns
403, the v2 API requires a bearer token), so there is no download script to run.

1. Open [doi:10.5061/dryad.wh70rxwmv](https://doi.org/10.5061/dryad.wh70rxwmv) and
   download `handwritingBCIData.tar.gz` (1.31 GB).
2. `tar -xzf handwritingBCIData.tar.gz`, which expands to roughly 5.7 GB. Keep it outside
   the repo if the repo lives in a synced folder such as OneDrive or Dropbox.
3. Confirm the tree is complete before spending GPU time on it:

```bash
python3 verify_data.py --data-dir /path/to/handwritingBCIData
```

`verify_data.py` checks the specific files the experiments open (per-session
`sentences.mat`, the `Step2_HMMLabels` for your partition, the partition file) and names
whatever is missing, rather than letting a truncated extraction fail hours into a sweep.
It exits non-zero on any problem.

### Running

```bash
pip install -r requirements.txt
pytest tests/ -q

# RQ1: one condition, or the full pre-registered sweep (resumable)
python3 experiments/exp1_alignment_sensitivity.py --corruption jitter --level 10 --seed 0
python3 experiments/exp1_alignment_sensitivity.py --sweep

python3 analysis/make_figures.py

# Baseline architecture sweep
python3 run_benchmark.py --full --max-len 3000 --seed 0 --output results/baseline_seed0.json
```

Pass `--data-dir` to any of these if the dataset is not at `./handwritingBCIData`.

The sweep skips conditions whose artifact already exists, so it resumes cleanly after an
interruption. Failed conditions are recorded as `FAILED_*.json` rather than silently
dropped, so gaps in coverage stay visible.

`benchmark_colab.ipynb` runs the pipeline on a Colab GPU, though a local NVIDIA card is
substantially faster and has no session limits.

---

## Scope

This project does **not** attempt to beat Willett et al.'s 5.32% CER, and no comparison to
that number would be meaningful, because it depends on an RNN language model, thousands of training
epochs, and an augmentation pipeline well beyond what is reproduced here. Absolute error
rates in this work are high and are not the contribution.

The contribution is the *shape of the response*: how accuracy moves as label quality and
session distance vary under otherwise matched conditions. That is measurable at high absolute
error, and it is a question the original work did not ask.

**Data:** [Dryad](https://doi.org/10.5061/dryad.wh70rxwmv) ·
**Reference implementation:** [handwritingBCI](https://github.com/fwillett/handwritingBCI)

---

## References

- Willett, F. R. et al. (2021). High-performance brain-to-text communication via handwriting. *Nature*, 593, 249–254.
- Gulati, A. et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Interspeech 2020*, 5036–5040.
- Graves, A. et al. (2006). Connectionist temporal classification. *ICML 2006*, 369–376.
