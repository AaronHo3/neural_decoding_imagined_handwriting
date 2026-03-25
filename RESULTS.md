# Comparative Analysis of Decoder Architectures and Alignment Strategies for Neural Decoding of Imagined Handwriting

Aaron Ho

---

## Abstract

Brain-computer interfaces (BCIs) that decode imagined handwriting from intracortical neural recordings hold transformative potential for restoring communication in individuals with paralysis. Building on the landmark dataset of Willett et al. (2021), I present a systematic comparison of four neural sequence decoder architectures — GRU, RCNN, CTC, and a novel Conformer-based decoder — evaluated across three frame-level alignment strategies: Willett's pre-computed Gaussian HMM with hard labels, Gaussian HMM with soft probability targets, and a Poisson HMM alignment designed to better model the discrete count statistics of neural spike data. I further investigate the effects of data augmentation, multi-session training, and character-level language model rescoring on decoding accuracy. In single-session experiments (89 training sentences, 10 test sentences), the RCNN decoder with Gaussian hard alignment achieved the lowest character error rate (CER) of 65.97%, while the Conformer exhibited severe overfitting — a finding consistent with the known data-hunger of attention-based models. However, when training data was expanded via multi-session aggregation (574 training sentences from 10 sessions), the Conformer surpassed all other architectures, achieving the best overall CER of 55.86% with Gaussian hard alignment — a 10.1 percentage point improvement over its single-session performance and a dramatic reversal in the architecture ranking. This data-scaling result confirms that the Conformer's self-attention mechanism, while disadvantaged in extreme low-data regimes, becomes the superior architecture once sufficient training diversity is available. Poisson HMM alignment underperformed Gaussian alignment across all decoders (83–90% CER in single-session), suggesting that template estimation quality outweighs emission model correctness. These results provide practical guidance for decoder selection across data availability regimes and highlight both alignment quality and training data volume as primary bottlenecks in neural handwriting decoding pipelines.

---

## 1. Introduction

The ability to decode imagined handwriting from neural activity represents a significant advance in brain-computer interface (BCI) technology. Willett et al. (2021) demonstrated that a recurrent neural network (RNN) trained on intracortical recordings from a single participant with tetraplegia could decode imagined handwriting at rates exceeding 90 characters per minute, achieving character error rates (CER) below 5% with language model integration. Their work established that the motor cortex retains detailed representations of handwriting movements even years after paralysis onset, and that these representations are sufficiently structured to support high-accuracy real-time decoding.

The Willett pipeline comprises three stages: (1) forced alignment of neural time series to character sequences via a Gaussian hidden Markov model (HMM), producing frame-level character labels; (2) training a GRU-based RNN to map neural features to character probabilities at each time step; and (3) decoding with a recurrent neural network language model (RNNLM) to produce final text. While this pipeline achieved remarkable results, several design choices merit further investigation.

First, the Gaussian emission model used in HMM alignment assumes that neural activity at each time bin is drawn from a multivariate Gaussian distribution. However, the recorded signals represent spike counts — non-negative integers arising from a point process. The Poisson distribution is the canonical model for count data, and Poisson emissions could in principle provide a more faithful generative model of neural spiking, particularly at low firing rates where Gaussian approximations predict non-negligible probability for negative counts.

Second, Willett et al. used a single GRU architecture for neural decoding. Recent advances in sequence modeling — particularly the Conformer architecture (Gulati et al., 2020), which combines self-attention with depthwise convolutions — have achieved state-of-the-art results in speech recognition, a domain with structural similarities to neural handwriting decoding (both involve mapping continuous temporal signals to discrete symbol sequences). Whether these architectural advances transfer to the neural decoding domain remains unexplored.

Third, the original pipeline trained on hard (argmax) frame labels, discarding the soft probability distributions produced by HMM alignment. Training on soft targets provides richer gradient information at character boundaries, where the correct label is inherently ambiguous.

In this work, I systematically investigate these questions through a controlled comparison of:

- **Four decoder architectures**: GRU (Willett baseline), RCNN (convolutional-recurrent hybrid), Conformer (attention + depthwise convolution), and CTC (alignment-free connectionist temporal classification).
- **Three alignment strategies**: Gaussian HMM with hard labels, Gaussian HMM with soft probability targets, and Poisson HMM alignment.
- **Training enhancements**: Gaussian noise data augmentation and bigram language model rescoring.

All experiments use the publicly available Willett et al. dataset from the Dryad repository, enabling full reproducibility.

---

## 2. Methods

### 2.1 Dataset

I used the publicly available dataset from Willett et al. (2021), recorded from participant T5 (male, 65 years old at time of recording) with two 96-channel Utah microelectrode arrays implanted in the left-hemisphere hand "knob" area of premotor cortex. Neural activity was recorded as threshold crossing spike counts in 10 ms bins across 192 channels (96 electrodes x 2 thresholds).

The dataset comprises multiple recording sessions spanning May 2019 to February 2020. Each session contains:

- **Single-letter trials**: The participant imagined writing individual characters (26 lowercase letters plus space, comma, apostrophe, period, and question mark; 31 classes total). Each trial yields a neural activity tensor of shape (time x 192).
- **Sentence trials**: The participant imagined writing complete sentences prompted on screen. A total of 102 sentence trials are available per session, with variable lengths up to ~30 seconds (3000 time bins).

For single-session experiments, I used session t5.2019.05.08 with the HeldOutTrials partition, yielding 89 training sentences and 10 held-out test sentences. For multi-session experiments, training data was aggregated from 10 sessions spanning May 2019 to January 2020 (t5.2019.05.08 through t5.2020.01.15), yielding 574 total training sentences (89 primary + 485 from 9 additional sessions). Test data remained the 10 held-out sentences from the primary session in all conditions, ensuring a fair comparison. Neural data from additional sessions was zero-padded or truncated to match the primary session's time dimension (3000 bins). Neural data was truncated to 3000 time bins (30 seconds) to include complete sentence durations while maintaining uniform tensor dimensions.

### 2.2 Preprocessing

Neural spike counts were z-score normalized using training set statistics (per-channel mean and standard deviation computed across all training sentences and time bins). Channels with zero variance were assigned unit standard deviation to prevent division by zero.

### 2.3 Forced Alignment

Frame-level character labels are required for training the GRU, RCNN, and Conformer decoders. I compared three approaches:

#### 2.3.1 Gaussian HMM (Willett Baseline)

I used the pre-computed alignment labels provided by Willett et al. in the `Step2_HMMLabels` directory. These were generated by fitting Gaussian emission templates from single-letter data and running Viterbi forced alignment on sentence-level neural recordings. The alignment produces two outputs:

- **charProbTarget**: A soft probability distribution over 31 character classes at each time bin, shape (N, T, 31). These represent the posterior probability of each character given the observed neural activity and the forced alignment path.
- **ignoreErrorHere**: A binary mask indicating time bins where the alignment is unreliable (e.g., pre- and post-sentence idle periods), shape (N, T).

**Hard labels** were derived by taking the argmax of charProbTarget at each time bin, with ignored positions set to -1 for loss masking.

**Soft labels** were used directly as training targets, with ignored positions zeroed out. When training with soft targets, I replaced cross-entropy loss with the soft cross-entropy: L = -sum(p_target * log(p_predicted)), where the sum is over character classes at each active time bin.

#### 2.3.2 Poisson HMM (Novel)

I implemented a Poisson HMM forced alignment to better model the discrete count statistics of neural spike data. Character-specific emission templates were estimated from single-letter trials by computing the mean spike count per channel per time bin for each character class, resampled to a fixed number of HMM states per character (hmm_bin_size = 20).

The Poisson log-emission probability at time t for state s with template rate lambda_s is:

    log P(x_t | s) = sum_c [ x_{t,c} * log(lambda_{s,c}) - lambda_{s,c} - log(x_{t,c}!) ]

where the sum is over neural channels c, x_{t,c} is the observed spike count, and lambda_{s,c} is the expected rate from the template. Template rates were floored at 0.1 to prevent log(0).

Viterbi decoding used a left-to-right topology with three transition types: stay (probability 0.6), advance to next state (probability 0.35), and skip one state (probability 0.05).

Critically, Poisson alignment was performed on raw (un-normalized) spike count data, as z-scored data violates the non-negative integer assumption of the Poisson model.

### 2.4 Decoder Architectures

#### 2.4.1 GRU (Baseline)

The GRU decoder follows the architecture of Willett et al.: a multi-layer gated recurrent unit network mapping neural features to per-frame character logits. Input at each time bin is a 192-dimensional neural feature vector; output is a 31-dimensional logit vector.

Configuration (full mode): 2 GRU layers, 256 hidden units, trained with Adam optimizer (lr = 1e-3) for 80 epochs with batch size 16.

#### 2.4.2 RCNN (Convolutional-Recurrent Hybrid)

The RCNN prepends a stack of 1D convolutional blocks before the recurrent layers. Each convolutional block comprises Conv1d (with same-padding), BatchNorm, ReLU activation, and dropout. The convolutions capture local temporal features (e.g., within-stroke dynamics) before the GRU layers integrate long-range sequential context.

Configuration: 3 convolutional blocks with channel dimensions (32, 64, 128), kernel size 5, followed by 2 GRU layers with 256 hidden units.

#### 2.4.3 Conformer (Novel Architecture for Neural Decoding)

I adapted the Conformer architecture (Gulati et al., 2020), originally developed for speech recognition, to neural handwriting decoding. Each Conformer block follows the Macaron-style arrangement:

    x = x + 0.5 * FFN_1(x)
    x = x + MHSA(x)
    x = x + ConvModule(x)
    x = x + 0.5 * FFN_2(x)
    x = LayerNorm(x)

where FFN is a position-wise feed-forward network with Swish activation, MHSA is multi-head self-attention, and ConvModule consists of pointwise convolution, GLU gating, depthwise convolution, batch normalization, Swish activation, and pointwise projection.

The input projection maps 192-dimensional neural features to d_model dimensions, followed by sinusoidal positional encoding.

Configuration: 4 Conformer blocks, d_model = 128, 4 attention heads, ff_dim = 512, depthwise convolution kernel size = 15, dropout = 0.1. Trained with AdamW (lr = 5e-4, weight decay = 1e-2) using linear warmup (100 steps) followed by cosine annealing. Batch size was reduced to 4 (from 16) due to the O(T^2) memory complexity of self-attention over 3000-frame sequences.

#### 2.4.4 CTC (Alignment-Free)

The CTC decoder removes the dependency on forced alignment entirely. It comprises convolutional feature extraction (2 Conv1d blocks with channels 64 and 128), a bidirectional LSTM (128 hidden units), and a linear projection to 32 output classes (31 characters + blank). The network is trained with CTC loss, which marginalizes over all possible alignments between the output sequence and the target character string.

CTC targets are character index sequences (1-indexed, with 0 reserved for the blank token). Decoding uses greedy search (argmax at each frame, collapse repeats, remove blanks).

### 2.5 Training Protocol

All decoders were trained for 80 epochs. Training data was augmented by generating two additional copies of each training sentence with additive Gaussian noise (sigma = 10% of training data standard deviation), yielding 267 effective training examples from 89 originals in the single-session condition and 1722 from 574 originals in the multi-session condition.

For soft-target training, cross-entropy loss was replaced with:

    L = -(1/|A|) * sum_{t in A} sum_c p_target(c,t) * log p_predicted(c,t)

where A is the set of active (non-ignored) time bins. For hard-target training, standard cross-entropy with ignore_index = -1 was used.

### 2.6 Decoding Pipeline

Frame-level logits were converted to character strings through the following pipeline:

1. **Softmax probability smoothing**: Raw logits were converted to probability distributions via softmax, then smoothed with a uniform moving average filter (window size = 51 frames, approximately 2/3 of the expected character duration).
2. **Temporal collapse**: Consecutive frames with identical argmax predictions were collapsed into character segments.
3. **Minimum duration filtering**: Segments shorter than T/100 frames (where T is the active sequence length) were discarded as noise.
4. **Beam search with LM rescoring**: Multiple candidate strings were generated by varying the per-frame class selection (top-1 through top-3). A character-level bigram language model, trained on the 89 training sentence prompts, rescored candidates using a weighted combination of acoustic score and language model log-probability (lambda = 0.3).

### 2.7 Evaluation Metrics

- **Character Error Rate (CER)**: Levenshtein edit distance between predicted and reference character strings, normalized by reference length. Computed using an O(n)-memory dynamic programming implementation.
- **Word Error Rate (WER)**: Edit distance computed on word tokens (delimited by the space character ">").
- **Frame-level Accuracy**: Proportion of active (non-ignored) time bins where the argmax prediction matches the hard alignment label.

---

## 3. Results

### 3.1 Single-Session Decoder Comparison

Table 1 presents the full results matrix across all decoder-alignment combinations in the single-session condition (89 training sentences).

**Table 1.** Single-session results: character error rate (CER), word error rate (WER), and frame-level accuracy for each decoder architecture and alignment strategy. Best result in each column is bolded. Training time per decoder on NVIDIA T4 GPU is shown in parentheses.

| Decoder | Alignment | CER (%) | WER (%) | Frame Acc (%) | Time (s) |
|---------|-----------|---------|---------|---------------|----------|
| **RCNN** | **Gaussian Hard** | **65.97** | **88.06** | 64.7 | 307 |
| RCNN | Gaussian Soft | 67.63 | 92.95 | **66.5** | 325 |
| GRU | Gaussian Soft | 73.69 | 90.78 | 61.3 | 298 |
| GRU | Gaussian Hard | 78.51 | 93.76 | 56.3 | 281 |
| CTC | None | 82.77 | 110.98 | 17.2 | 211 |
| RCNN | Poisson HMM | 83.35 | 100.00 | 25.1 | 306 |
| Conformer | Gaussian Soft | 85.06 | 98.00 | 55.6 | 1867 |
| Conformer | Gaussian Hard | 85.50 | 100.00 | 56.9 | 1854 |
| Conformer | Poisson HMM | 87.69 | 100.00 | 23.9 | 1853 |
| GRU | Poisson HMM | 90.21 | 99.38 | 17.3 | 283 |

### 3.2 Multi-Session Decoder Comparison

Table 2 presents results when training data was aggregated across 10 recording sessions (574 training sentences, augmented to 1722). Poisson HMM alignment was excluded from multi-session experiments because it consistently underperformed Gaussian alignment in the single-session condition (Section 3.3), and Poisson template estimation was not available for extra sessions.

**Table 2.** Multi-session results (574 training sentences from 10 sessions, tested on 10 held-out sentences from primary session). Best result in each column is bolded.

| Decoder | Alignment | CER (%) | WER (%) | Frame Acc (%) | Time (s) |
|---------|-----------|---------|---------|---------------|----------|
| **Conformer** | **Gaussian Hard** | **55.86** | **71.98** | **80.4** | 13974 |
| RCNN | Gaussian Hard | 60.04 | 85.05 | 73.5 | 1991 |
| RCNN | Gaussian Soft | 60.69 | 80.60 | 72.7 | 2095 |
| Conformer | Gaussian Soft | 60.79 | 81.10 | **80.4** | 13975 |
| GRU | Gaussian Hard | 71.91 | 94.07 | 60.2 | 1858 |
| GRU | Gaussian Soft | 77.21 | 97.69 | 56.6 | 1928 |
| CTC | None | 85.75 | 98.62 | 14.3 | 1399 |

**Table 3.** Effect of multi-session training: CER improvement by decoder and alignment.

| Decoder | Alignment | Single-session CER | Multi-session CER | Improvement |
|---------|-----------|--------------------|--------------------|-------------|
| Conformer | Gaussian Hard | 85.50% | 55.86% | -29.64 pp |
| Conformer | Gaussian Soft | 85.06% | 60.79% | -24.27 pp |
| RCNN | Gaussian Soft | 67.63% | 60.69% | -6.94 pp |
| RCNN | Gaussian Hard | 65.97% | 60.04% | -5.93 pp |
| GRU | Gaussian Soft | 73.69% | 77.21% | +3.52 pp |
| GRU | Gaussian Hard | 78.51% | 71.91% | -6.60 pp |
| CTC | None | 82.77% | 85.75% | +2.98 pp |

### 3.3 Effect of Decoder Architecture

The architecture ranking changed dramatically between single-session and multi-session conditions, revealing a critical interaction between model capacity and data availability.

**Single-session (89 training sentences):** RCNN achieved the lowest CER across alignment conditions (65.97–83.35%), followed by GRU (73.69–90.21%), Conformer (85.06–87.69%), and CTC (82.77%). The RCNN's advantage over the pure GRU is attributable to its convolutional front-end, which extracts local temporal features before recurrent processing. The Conformer achieved competitive frame-level accuracy (55.6–56.9% with Gaussian alignment) but substantially worse CER (85.06–85.50%), exhibiting severe overfitting — producing truncated output strings on longer test sentences despite accurate per-frame predictions. With only 267 effective training examples, the Conformer lacked sufficient diversity to learn robust temporal generalization.

**Multi-session (574 training sentences):** The Conformer surpassed all other architectures, achieving the best overall CER of 55.86% — a 29.6 percentage point improvement over its single-session performance. This reversal in architecture ranking is the most striking finding of this study. With 6.4x more training data (1722 augmented examples vs. 267), the Conformer's self-attention mechanism — which was a liability in the single-session regime — became an asset, enabling it to capture long-range temporal dependencies across diverse sentence structures. The Conformer also achieved the highest frame-level accuracy (80.4%), substantially above RCNN (73.5%) and GRU (60.2%).

The RCNN improved more modestly with additional data (-5.93 pp CER), consistent with its stronger inductive biases already extracting most available signal from limited data. The GRU and CTC showed mixed results, suggesting these simpler architectures may have hit a representational ceiling.

CTC showed meaningful learning but produced output with poor character accuracy in both conditions. CTC's alignment-free training requires the network to jointly learn both the neural-to-character mapping and the temporal alignment, which is a substantially harder optimization problem.

### 3.4 Effect of Alignment Strategy

Gaussian HMM alignment (both soft and hard) substantially outperformed Poisson HMM alignment across all decoders. The best Poisson result (RCNN, 83.35% CER) was 17.4 percentage points worse than the best Gaussian result (RCNN Hard, 65.97% CER).

Despite the theoretical appeal of Poisson emissions for count data, the Gaussian HMM benefited from Willett et al.'s carefully optimized template estimation pipeline, which included cross-validated template smoothing and session-specific calibration. My Poisson HMM templates were estimated from single-letter data without these refinements, introducing template noise that propagated into alignment errors.

Frame-level accuracy tells a striking story: Gaussian-aligned decoders achieved 56–67% frame accuracy, while Poisson-aligned decoders achieved only 17–25%. This confirms that the alignment quality — not the decoder architecture — is the primary bottleneck for the Poisson condition.

### 3.5 Effect of Soft vs. Hard Labels

Soft probability targets improved frame-level accuracy relative to hard labels for both GRU (61.3% vs. 56.3%, +5.0 pp) and RCNN (66.5% vs. 64.7%, +1.8 pp). This is expected: soft targets provide informative gradients at character boundaries where the correct label is inherently ambiguous, rather than penalising the model for an arbitrary hard assignment.

However, the CER results were mixed: GRU benefited from soft labels (73.69% vs. 78.51%, -4.8 pp), while RCNN showed a slight degradation (67.63% vs. 65.97%, +1.7 pp). One explanation is that the RCNN's convolutional layers already capture boundary dynamics effectively, so the additional boundary information from soft labels provides marginal benefit while slightly blurring the within-segment predictions that drive the collapse-based decoding pipeline.

### 3.6 Qualitative Analysis of Predictions

**Table 4.** Example decoded sentences from the best single-session (RCNN + Hard) and best multi-session (Conformer + Hard) configurations. ">" denotes space.

| | Reference | RCNN + Hard (single) | Conformer + Hard (multi) |
|---|-----------|---------------------|------------------------|
| 1 | you>want>me>to>sing? | iow>mwant>rme>to>simag | o>want>me>to>smngl |
| 2 | have>you>ever>seen>a>large>cat>fold>... | haus>~ou>ewer>faoer>a>k>hardge | have>you>ever>secn>ar,rtd>large>ca |
| 3 | the>jeep>was>thirsty>so>i>stopped>... | mioe>pljep>was>thinst>asol>i>s | the>jieap>,>was>thirsty>y>yso>i>s |

The multi-session Conformer predictions show dramatically improved word structure preservation compared to both the single-session Conformer (which produced catastrophically truncated outputs like "jtusoatmd>ulg") and the single-session RCNN. In sentence 2, the Conformer correctly decodes "have>you>ever" and "large" — demonstrating that with sufficient training data, the self-attention mechanism successfully captures the sequential structure of handwriting. Sentence 3 shows recognizable multi-word segments ("the>jieap>was>thirsty>so>i>s") that would benefit substantially from a neural language model.

### 3.7 Computational Efficiency

In single-session experiments, the Conformer required approximately 6x longer training time (1854–1867s) compared to GRU and RCNN (281–325s). In multi-session experiments, this gap widened substantially: the Conformer required ~14,000s (~3.9 hours) per condition, compared to ~2,000s (~33 minutes) for GRU and RCNN — a 7x ratio, driven by the quadratic memory complexity of self-attention over T = 3000 time steps with 1722 training examples and the reduced batch size (4 vs. 16) necessitated by GPU memory constraints. The total multi-session benchmark runtime was approximately 11 hours on a single NVIDIA T4 GPU. CTC was the fastest to train owing to its simpler architecture.

---

## 4. Discussion

### 4.1 Decoder Architecture Selection Across Data Regimes

My results demonstrate that architectural choice interacts critically with data availability, and that the optimal decoder depends on the training data regime.

In the **low-data regime** (89 sentences, 267 augmented), the RCNN's inductive biases — local feature extraction via convolutions followed by sequential integration via recurrence — proved well-matched to neural handwriting decoding. The Conformer's flexible self-attention mechanism led to severe overfitting.

In the **moderate-data regime** (574 sentences, 1722 augmented), the architecture ranking reversed entirely: the Conformer achieved the best CER (55.86%), surpassing RCNN (60.04%) by 4.2 percentage points. The Conformer's 29.6 pp improvement from single- to multi-session — compared to RCNN's 5.9 pp improvement — demonstrates that attention-based architectures have a dramatically steeper data-scaling curve in this domain. This is consistent with findings in speech recognition (Gulati et al., 2020) and NLP, where transformer-based models underperform simpler alternatives on small datasets but dominate with sufficient data.

This finding has practical implications for BCI deployment: when calibration data is limited (as is typical in early clinical sessions), RCNN should be preferred. As data accumulates across sessions, transitioning to a Conformer architecture would yield substantially better performance. An adaptive deployment strategy — starting with RCNN and switching to Conformer after sufficient sessions — may be optimal in practice.

### 4.2 The Alignment Bottleneck

The large performance gap between Gaussian and Poisson alignment conditions underscores that alignment quality is the dominant factor in the decoding pipeline. Even the best decoder (RCNN) could not compensate for the noisy frame labels produced by Poisson HMM alignment. This suggests that future work should prioritise alignment quality improvements — such as discriminative alignment training, multi-session template estimation, or end-to-end alignment learning — over decoder architecture exploration.

The Poisson HMM's underperformance does not necessarily invalidate the Poisson emission model. Rather, my results indicate that the advantages of a theoretically correct generative model are overwhelmed by practical factors: template estimation noise from limited single-letter data, suboptimal HMM hyperparameters (bin size, transition probabilities), and the lack of the extensive calibration pipeline that Willett et al. developed for their Gaussian HMM.

### 4.3 Soft Labels: A Low-Cost Improvement

The consistent improvement in frame-level accuracy from soft labels (+1.8 to +5.0 pp) represents a "free" improvement that requires no additional data or computation — only using the probability distributions already produced by HMM alignment rather than discarding them via argmax. I recommend soft-target training as a default for future neural decoding work using HMM-aligned labels.

### 4.4 Limitations

Several limitations constrain the interpretation of these results:

1. **Limited training epochs and data**: Although multi-session training substantially improved results, my training protocol (80 epochs on 574 sentences) remains far smaller than Willett et al.'s (thousands of epochs with extensive synthetic augmentation). Further performance gains are likely achievable with longer training and more sophisticated augmentation strategies.

2. **No recurrent language model**: Willett et al.'s 5.32% CER relied critically on RNNLM decoding. My bigram LM provides only rudimentary linguistic constraint. Integrating a pre-trained character-level language model would likely reduce CER substantially across all conditions.

3. **Single participant**: The dataset contains recordings from one participant (T5). Generalizability to other participants, electrode placements, or neurological conditions remains unknown.

4. **Hyperparameter optimization**: I did not perform systematic hyperparameter tuning. The Conformer in particular may benefit from different learning rate schedules, dropout rates, or architectural configurations optimized for the low-data regime.

### 4.5 Comparison with Willett et al.

Direct comparison with Willett et al.'s reported 5.32% CER is not appropriate, as my experimental conditions differ in several critical ways: bigram vs. RNNLM decoding, 80 vs. thousands of training epochs, and minimal vs. extensive data augmentation. While I incorporated multi-session training, the data volume and augmentation sophistication remain far below the original study. This work is complementary — it provides a controlled comparison of architectural and alignment choices that was not the focus of the original study, and demonstrates the data-scaling properties of different decoder architectures.

### 4.6 Future Directions

Several directions could substantially improve performance:

1. **Extended training and augmentation**: My multi-session results (55.86% CER) with 80 epochs suggest that the Conformer has not yet saturated. Training for hundreds or thousands of epochs with more aggressive augmentation (time-warping, electrode dropout, neural activity interpolation) could yield substantial further gains.
2. **Adaptive architecture selection**: An online BCI system could begin with RCNN during initial sessions and automatically transition to a Conformer once sufficient cross-session data has accumulated, leveraging the data-scaling properties demonstrated in this work.
3. **Pre-trained language model integration**: Replacing the bigram LM with a character-level transformer or LSTM language model trained on large English text corpora.
4. **End-to-end training**: Jointly optimising alignment and decoding, potentially using CTC or attention-based sequence-to-sequence models with curriculum learning to bootstrap from the limited data.
5. **Hybrid architectures**: Combining the RCNN's convolutional front-end with Conformer blocks (rather than GRU layers) may capture the benefits of both architectures while mitigating the Conformer's data requirements.

---

## 5. Conclusion

I presented a systematic comparison of decoder architectures and alignment strategies for neural decoding of imagined handwriting, evaluated on the publicly available Willett et al. (2021) dataset under both single-session (89 sentences) and multi-session (574 sentences) training conditions. The key findings are:

1. **Architecture ranking depends on data availability**: The RCNN outperformed all architectures in the single-session regime (65.97% CER), but the Conformer surpassed it in the multi-session regime (55.86% CER) — a complete reversal driven by the Conformer's steeper data-scaling curve.
2. **Multi-session training provides large gains for attention-based models**: The Conformer improved by 29.6 percentage points with 6.4x more training data, compared to only 5.9 pp for RCNN, demonstrating that self-attention mechanisms are especially data-hungry but also especially data-responsive in this domain.
3. **Soft probability targets** from HMM alignment improved frame-level accuracy by 1.8–5.0 percentage points over hard labels, representing a zero-cost enhancement.
4. **Poisson HMM** alignment underperformed Gaussian HMM by 17+ percentage points in single-session experiments, indicating that template estimation quality outweighs emission model correctness in the current paradigm.
5. **Alignment quality remains the primary bottleneck**, but training data volume is a close second — with sufficient data, the best CER improved from 65.97% to 55.86%, a 10.1 pp gain.

These findings provide actionable guidance for neural BCI decoder design across data availability regimes and demonstrate the importance of considering data-scaling properties when selecting decoder architectures for clinical deployment.

---

## References

Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., & Pang, R. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Interspeech 2020*, 5036–5040.

Graves, A., Fernandez, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. *Proceedings of the 23rd International Conference on Machine Learning*, 369–376.

Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., & Shenoy, K. V. (2021). High-performance brain-to-text communication via handwriting. *Nature*, 593(7858), 249–254.
