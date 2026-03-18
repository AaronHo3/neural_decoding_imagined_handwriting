# Neural Decoding of Imagined Handwriting

A systematic comparison of decoder architectures and alignment strategies for brain-to-text communication from intracortical neural recordings. Building on [Willett et al. (Nature 2021)](https://www.nature.com/articles/s41586-021-03506-2), I evaluate four neural sequence decoders across three alignment methods to understand what drives decoding accuracy under clinical data constraints.

**Dataset**: [Willett et al. Dryad Repository](https://doi.org/10.5061/dryad.wh70rxwmv) | **Reference Code**: [handwritingBCI](https://github.com/fwillett/handwritingBCI)

---

## Key Results

| Decoder | Alignment | CER (%) | WER (%) | Frame Acc (%) |
|---------|-----------|---------|---------|---------------|
| **RCNN** | **Gaussian Hard** | **65.97** | **88.06** | 64.7 |
| RCNN | Gaussian Soft | 67.63 | 92.95 | **66.5** |
| GRU | Gaussian Soft | 73.69 | 90.78 | 61.3 |
| GRU | Gaussian Hard | 78.51 | 93.76 | 56.3 |
| CTC | None (alignment-free) | 82.77 | 110.98 | 17.2 |
| Conformer | Gaussian Soft | 85.06 | 98.00 | 55.6 |
| Conformer | Gaussian Hard | 85.50 | 100.00 | 56.9 |
| RCNN | Poisson HMM | 83.35 | 100.00 | 25.1 |

*Single session (89 train / 10 test sentences), 80 epochs, NVIDIA T4 GPU. See [RESULTS.md](RESULTS.md) for the full paper.*

### Key Findings

1. **RCNN outperforms all architectures** in the low-data regime — 12.5 pp better CER than GRU with the same alignment
2. **Soft probability targets** from HMM alignment improve frame accuracy by 1.8-5.0 pp over hard labels at zero additional cost
3. **Conformer overfits** with limited data — competitive frame accuracy (57%) but poor sequence-level output, an important negative result
4. **Alignment quality dominates** — Poisson HMM underperforms Gaussian by 17+ pp, showing template quality matters more than emission model choice

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
│   └── preprocessing.py        # Binning, normalisation
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

# Full run (~30 min on GPU)
python3 run_benchmark.py --full --max-len 3000
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
| Focus | Maximum accuracy | Controlled architectural comparison |

---

## References

- Willett, F. R. et al. (2021). High-performance brain-to-text communication via handwriting. *Nature*, 593, 249-254.
- Gulati, A. et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Interspeech 2020*.
- Graves, A. et al. (2006). Connectionist temporal classification. *ICML 2006*.
