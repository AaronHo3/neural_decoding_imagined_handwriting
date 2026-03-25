# Neural Decoding of Imagined Handwriting

A systematic comparison of decoder architectures and alignment strategies for brain-to-text communication from intracortical neural recordings. Building on [Willett et al. (Nature 2021)](https://www.nature.com/articles/s41586-021-03506-2), I evaluate four neural sequence decoders across three alignment methods to understand what drives decoding accuracy under clinical data constraints.

**Dataset**: [Willett et al. Dryad Repository](https://doi.org/10.5061/dryad.wh70rxwmv) | **Reference Code**: [handwritingBCI](https://github.com/fwillett/handwritingBCI)

---

## Key Results

### Multi-Session Training (574 sentences from 10 sessions)

| Decoder | Alignment | CER (%) | WER (%) | Frame Acc (%) |
|---------|-----------|---------|---------|---------------|
| **Conformer** | **Gaussian Hard** | **55.86** | **71.98** | **80.4** |
| RCNN | Gaussian Hard | 60.04 | 85.05 | 73.5 |
| RCNN | Gaussian Soft | 60.69 | 80.60 | 72.7 |
| Conformer | Gaussian Soft | 60.79 | 81.10 | 80.4 |
| GRU | Gaussian Hard | 71.91 | 94.07 | 60.2 |
| CTC | None (alignment-free) | 85.75 | 98.62 | 14.3 |

### Single-Session Training (89 sentences)

| Decoder | Alignment | CER (%) | WER (%) | Frame Acc (%) |
|---------|-----------|---------|---------|---------------|
| **RCNN** | **Gaussian Hard** | **65.97** | **88.06** | 64.7 |
| RCNN | Gaussian Soft | 67.63 | 92.95 | **66.5** |
| GRU | Gaussian Soft | 73.69 | 90.78 | 61.3 |
| Conformer | Gaussian Hard | 85.50 | 100.00 | 56.9 |

*80 epochs, NVIDIA T4 GPU. See [RESULTS.md](RESULTS.md) for the full paper.*

### Key Findings

1. **Architecture ranking reverses with more data** — RCNN is best with 89 sentences (65.97% CER), but Conformer becomes best with 574 sentences (55.86% CER), improving by 29.6 pp
2. **Multi-session training yields large gains** — best CER improved from 65.97% to 55.86% (10.1 pp) by aggregating data across 10 recording sessions
3. **Soft probability targets** from HMM alignment improve frame accuracy by 1.8-5.0 pp over hard labels at zero additional cost
4. **Alignment quality dominates** — Poisson HMM underperforms Gaussian by 17+ pp, showing template quality matters more than emission model choice
5. **Conformer's data-scaling curve is dramatically steeper** than RCNN's (-29.6 pp vs -5.9 pp), confirming attention-based models are data-hungry but data-responsive

---

## Architecture Overview

```
Neural Activity (192 channels, 10ms bins)
         │
         ├──→ Gaussian HMM Alignment (Willett) ──→ Frame-level character labels
         ├──→ Poisson HMM Alignment (novel)    ──→ Frame-level character labels
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  Decoder Architectures                      │
    │                                             │
    │  GRU:       Input → GRU → Linear → Logits  │
    │  RCNN:      Input → Conv1D → GRU → Logits  │
    │  Conformer: Input → [½FFN→Attn→Conv→½FFN]  │
    │  CTC:       Input → Conv1D → BiLSTM → CTC  │
    └─────────────────────────────────────────────┘
         │
         ▼
    Smoothing → Collapse → Beam Search + Bigram LM
         │
         ▼
    Decoded Text: "you>want>me>to>sing?"
```

---

## Project Structure

```
neural_decoding_imagined_handwriting/
├── alignment/                  # HMM forced alignment
│   ├── gaussian_hmm.py         # Willett-style Gaussian emissions
│   └── poisson_hmm.py          # Novel Poisson emissions for count data
├── decoders/                   # Neural sequence decoders
│   ├── rnn_decoder.py          # GRU baseline (Willett)
│   ├── rcnn_decoder.py         # Conv1D + GRU hybrid
│   ├── transformer_decoder.py  # Conformer (novel for neural decoding)
│   ├── ctc_decoder.py          # CNN-BiLSTM + CTC loss
│   └── base_decoder.py         # Abstract interface
├── benchmarks/                 # Evaluation metrics
│   ├── evaluate.py             # CER, WER, Levenshtein distance
│   └── compare.py              # Comparison orchestration
├── data/                       # Data pipeline
│   ├── loader.py               # Willett dataset loader
│   └── preprocessing.py        # Binning, normalization
├── run_benchmark.py            # Main experiment script
├── benchmark_colab.ipynb       # Google Colab notebook (GPU)
├── download_data.sh            # Dataset download helper
└── RESULTS.md                  # Full research paper
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch scipy scikit-learn h5py numpy
```

### 2. Download the dataset

```bash
bash download_data.sh
```

Or download manually from [Dryad](https://doi.org/10.5061/dryad.wh70rxwmv) (~1.4 GB) and extract to `./handwritingBCIData/`.

### 3. Run the benchmark

```bash
# Fast run (~5 min on GPU)
python3 run_benchmark.py --max-len 1500 --skip-poisson --decoders gru rcnn

# Full run (~30 min on GPU, single session)
python3 run_benchmark.py --full --max-len 3000

# Full run with multi-session training (~11 hours on GPU)
python3 run_benchmark.py --full --max-len 3000 --multi-session --skip-poisson
```

### 4. Run on Google Colab (recommended)

Open `benchmark_colab.ipynb` in Colab with a T4 GPU runtime. Upload `handwritingBCIData.tar.gz` to Google Drive, then run all cells.

---

## Novelty Beyond Willett et al.

| Aspect | Willett et al. (2021) | This Work |
|--------|----------------------|-----------|
| Decoder | GRU only | GRU, RCNN, Conformer, CTC |
| Alignment | Gaussian HMM | Gaussian + Poisson HMM |
| Labels | Hard (argmax) | Hard + soft probability targets |
| Language model | RNN-LM | Bigram LM with beam search |
| Training | Multi-session, 1000s of epochs | Multi-session (10 sessions), 80 epochs |
| Focus | Maximum accuracy | Controlled architectural comparison |

---

## References

- Willett, F. R. et al. (2021). High-performance brain-to-text communication via handwriting. *Nature*, 593, 249-254.
- Gulati, A. et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Interspeech 2020*.
- Graves, A. et al. (2006). Connectionist temporal classification. *ICML 2006*.
