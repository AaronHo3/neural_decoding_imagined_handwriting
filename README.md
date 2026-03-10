# Neural Decoding of Imagined Handwriting

Benchmarking alignment models (Gaussian vs Poisson HMM) and sequence decoders (RNN, RCNN, CTC) for brain-to-text communication from intracortical neural signals.

Based on [Willett et al. (Nature 2021)](https://www.nature.com/articles/s41586-021-03506-2) and the [handwritingBCI](https://github.com/fwillett/handwritingBCI) codebase. Dataset: [Dryad](https://doi.org/10.5061/dryad.wh70rxwmv).

## Project Structure

```
neural_decoding_imagined_handwriting/
├── alignment/           # HMM variants for forced alignment
│   ├── gaussian_hmm.py   # Willett-style baseline
│   └── poisson_hmm.py    # Count-based emissions
├── decoders/             # Sequence decoders
│   ├── rnn_decoder.py    # GRU baseline
│   ├── rcnn_decoder.py   # Recurrent conv + RNN
│   └── ctc_decoder.py    # CNN-BiLSTM + CTC
├── benchmarks/           # Evaluation
│   ├── evaluate.py      # CER, WER, metrics
│   └── compare.py       # Ablation / comparison runs
├── data/                 # Data loading
│   ├── loader.py        # Willett dataset
│   └── preprocessing.py # Binning, normalization
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Download the Willett dataset from [Dryad](https://doi.org/10.5061/dryad.wh70rxwmv).
2. Implement the stubs in each module.
3. Run benchmarks:

```python
from data import WillettDataset
from alignment import GaussianHMMForcedAlignment, PoissonHMMForcedAlignment
from decoders import RNNDecoder, RCNNDecoder, CTCDecoder
from benchmarks import run_decoder_comparison, run_alignment_comparison
```

## Key Differentiations from Willett

- **Poisson HMM**: Count-based emissions for neural spike data
- **RCNN**: Convolutional + recurrent architecture
- **CTC decoder**: Alignment-free training option
- **Comparative framework**: Benchmark alignment methods and decoders
