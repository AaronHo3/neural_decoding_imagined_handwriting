# Comparative Analysis of Decoder Architectures and Alignment Strategies for Neural Decoding of Imagined Handwriting

Aaron Ho

---

## Abstract

Brain-computer interfaces (BCIs) that decode imagined handwriting from intracortical neural recordings hold transformative potential for restoring communication in individuals with paralysis. Building on the landmark dataset of Willett et al. (2021), I present a systematic comparison of four neural sequence decoder architectures — GRU, RCNN, CTC, and a novel Conformer-based decoder — evaluated across three frame-level alignment strategies: Willett's pre-computed Gaussian HMM with hard labels, Gaussian HMM with soft probability targets, and a Poisson HMM alignment designed to better model the discrete count statistics of neural spike data. I further investigate the effects of data augmentation and character-level language model rescoring on decoding accuracy. On single-session data (89 training sentences, 10 test sentences), the RCNN decoder with Gaussian hard alignment achieved the lowest character error rate (CER) of 65.97%, while soft probability targets improved GRU frame-level accuracy from 56.3% to 61.3%. The Conformer architecture, despite achieving competitive frame accuracy (56.9%), exhibited severe overfitting to training data, producing degraded sequence-level outputs — a finding consistent with the known data-hunger of attention-based models. Poisson HMM alignment underperformed Gaussian alignment across all decoders (83–90% CER), suggesting that the theoretical advantages of count-based emission models are insufficient to overcome template estimation noise in the single-session regime. These results provide practical guidance for decoder selection under data-limited conditions and highlight the critical role of alignment quality in neural handwriting decoding pipelines.

---

## 1. Introduction

The ability to decode imagined handwriting from neural activity represents a significant advance in brain-computer interface (BCI) technology. Willett et al. (2021) demonstrated that a recurrent neural network (RNN) trained on intracortical recordings from a single participant with tetraplegia could decode imagined handwriting at rates exceeding 90 characters per minute, achieving character error rates (CER) below 5% with language model integration. Their work established that the motor cortex retains detailed representations of handwriting movements even years after paralysis onset, and that these representations are sufficiently structured to support high-accuracy real-time decoding.

The Willett pipeline comprises three stages: (1) forced alignment of neural time series to character sequences via a Gaussian hidden Markov model (HMM), producing frame-level character labels; (2) training a GRU-based RNN to map neural features to character probabilities at each time step; and (3) decoding with a recurrent neural network language model (RNNLM) to produce final text. While this pipeline achieved remarkable results, several design choices merit further investigation.

First, the Gaussian emission model used in HMM alignment assumes that neural activity at each time bin is drawn from a multivariate Gaussian distribution. However, the recorded signals represent spike counts — non-negative integers arising from a point process. The Poisson distribution is the canonical model for count data, and Poisson emissions could in principle provide a more faithful generative model of neural spiking, particularly at low firing rates where Gaussian approximations predict non-negligible probability for negative counts.

Second, Willett et al. used a single GRU architecture for neural decoding. Recent advances in sequence modelling — particularly the Conformer architecture (Gulati et al., 2020), which combines self-attention with depthwise convolutions — have achieved state-of-the-art results in speech recognition, a domain with structural similarities to neural handwriting decoding (both involve mapping continuous temporal signals to discrete symbol sequences). Whether these architectural advances transfer to the neural decoding domain remains unexplored.

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

For all experiments reported here, I used session t5.2019.05.08 with the HeldOutTrials partition, yielding 89 training sentences and 10 held-out test sentences. Neural data was truncated to 3000 time bins (30 seconds) to include complete sentence durations while maintaining uniform tensor dimensions.

### 2.2 Preprocessing

Neural spike counts were z-score normalised using training set statistics (per-channel mean and standard deviation computed across all training sentences and time bins). Channels with zero variance were assigned unit standard deviation to prevent division by zero.

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

Critically, Poisson alignment was performed on raw (un-normalised) spike count data, as z-scored data violates the non-negative integer assumption of the Poisson model.

### 2.4 Decoder Architectures

#### 2.4.1 GRU (Baseline)

The GRU decoder follows the architecture of Willett et al.: a multi-layer gated recurrent unit network mapping neural features to per-frame character logits. Input at each time bin is a 192-dimensional neural feature vector; output is a 31-dimensional logit vector.

Configuration (full mode): 2 GRU layers, 256 hidden units, trained with Adam optimiser (lr = 1e-3) for 80 epochs with batch size 16.

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

where FFN is a position-wise feed-forward network with Swish activation, MHSA is multi-head self-attention, and ConvModule consists of pointwise convolution, GLU gating, depthwise convolution, batch normalisation, Swish activation, and pointwise projection.

The input projection maps 192-dimensional neural features to d_model dimensions, followed by sinusoidal positional encoding.

Configuration: 4 Conformer blocks, d_model = 128, 4 attention heads, ff_dim = 512, depthwise convolution kernel size = 15, dropout = 0.1. Trained with AdamW (lr = 5e-4, weight decay = 1e-2) using linear warmup (100 steps) followed by cosine annealing. Batch size was reduced to 4 (from 16) due to the O(T^2) memory complexity of self-attention over 3000-frame sequences.

#### 2.4.4 CTC (Alignment-Free)

The CTC decoder removes the dependency on forced alignment entirely. It comprises convolutional feature extraction (2 Conv1d blocks with channels 64 and 128), a bidirectional LSTM (128 hidden units), and a linear projection to 32 output classes (31 characters + blank). The network is trained with CTC loss, which marginalises over all possible alignments between the output sequence and the target character string.

CTC targets are character index sequences (1-indexed, with 0 reserved for the blank token). Decoding uses greedy search (argmax at each frame, collapse repeats, remove blanks).

### 2.5 Training Protocol

All decoders were trained for 80 epochs. Training data was augmented by generating two additional copies of each training sentence with additive Gaussian noise (sigma = 10% of training data standard deviation), yielding 267 effective training examples from 89 originals.

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

- **Character Error Rate (CER)**: Levenshtein edit distance between predicted and reference character strings, normalised by reference length. Computed using an O(n)-memory dynamic programming implementation.
- **Word Error Rate (WER)**: Edit distance computed on word tokens (delimited by the space character ">").
- **Frame-level Accuracy**: Proportion of active (non-ignored) time bins where the argmax prediction matches the hard alignment label.

---

## 3. Results

### 3.1 Overall Decoder Comparison

Table 1 presents the full results matrix across all decoder-alignment combinations.

**Table 1.** Character error rate (CER), word error rate (WER), and frame-level accuracy for each decoder architecture and alignment strategy. Best result in each column is bolded. Training time per decoder on NVIDIA T4 GPU is shown in parentheses.

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

### 3.2 Effect of Decoder Architecture

Among the four architectures tested, RCNN consistently achieved the lowest CER across alignment conditions (65.97–83.35%), followed by GRU (73.69–90.21%), Conformer (85.06–87.69%), and CTC (82.77%).

The RCNN's advantage over the pure GRU is attributable to its convolutional front-end, which extracts local temporal features before recurrent processing. In the neural handwriting domain, local temporal structure — the shape of individual pen strokes spanning ~50–200 ms — carries substantial discriminative information that 1D convolutions are well-suited to capture.

The Conformer achieved competitive frame-level accuracy (55.6–56.9% with Gaussian alignment, comparable to GRU's 56.3%) but substantially worse CER (85.06–85.50% vs. 73.69–78.51%). Qualitative inspection of predictions reveals the failure mode: while the Conformer learned accurate frame-level representations, it produced severely truncated output strings on longer test sentences (e.g., predicting "gterseo" for a 60-character reference). This suggests that the model overfit to the training sentence length distribution and failed to generalise temporal scaling to held-out sentences of varying duration. This finding is consistent with the known data-hunger of attention-based architectures — with only 267 effective training examples (89 original + augmentation), the Conformer lacks sufficient diversity to learn robust temporal generalisation.

CTC showed meaningful learning (loss decreased from 50.7 to 23.9 over 80 epochs) but produced output with poor character accuracy. CTC's alignment-free training requires the network to jointly learn both the neural-to-character mapping and the temporal alignment, which is a substantially harder optimisation problem with limited data.

### 3.3 Effect of Alignment Strategy

Gaussian HMM alignment (both soft and hard) substantially outperformed Poisson HMM alignment across all decoders. The best Poisson result (RCNN, 83.35% CER) was 17.4 percentage points worse than the best Gaussian result (RCNN Hard, 65.97% CER).

Despite the theoretical appeal of Poisson emissions for count data, the Gaussian HMM benefited from Willett et al.'s carefully optimised template estimation pipeline, which included cross-validated template smoothing and session-specific calibration. My Poisson HMM templates were estimated from single-letter data without these refinements, introducing template noise that propagated into alignment errors.

Frame-level accuracy tells a striking story: Gaussian-aligned decoders achieved 56–67% frame accuracy, while Poisson-aligned decoders achieved only 17–25%. This confirms that the alignment quality — not the decoder architecture — is the primary bottleneck for the Poisson condition.

### 3.4 Effect of Soft vs. Hard Labels

Soft probability targets improved frame-level accuracy relative to hard labels for both GRU (61.3% vs. 56.3%, +5.0 pp) and RCNN (66.5% vs. 64.7%, +1.8 pp). This is expected: soft targets provide informative gradients at character boundaries where the correct label is inherently ambiguous, rather than penalising the model for an arbitrary hard assignment.

However, the CER results were mixed: GRU benefited from soft labels (73.69% vs. 78.51%, -4.8 pp), while RCNN showed a slight degradation (67.63% vs. 65.97%, +1.7 pp). One explanation is that the RCNN's convolutional layers already capture boundary dynamics effectively, so the additional boundary information from soft labels provides marginal benefit while slightly blurring the within-segment predictions that drive the collapse-based decoding pipeline.

### 3.5 Qualitative Analysis of Predictions

Table 2 shows representative predictions from the best-performing configuration (RCNN + Gaussian Hard) and the novel Conformer (Gaussian Soft).

**Table 2.** Example decoded sentences. ">" denotes space.

| | Reference | RCNN + Gauss Hard | Conformer + Gauss Soft |
|---|-----------|-------------------|----------------------|
| 1 | you>want>me>to>sing? | iow>mwant>rme>to>simag | ou>arant>me>to>s>'neop |
| 2 | have>you>ever>seen>a>large>cat>fold>... | haus>~ou>ewer>faoer>a>k>hardge | jtusoatmd>ulg |
| 3 | the>jeep>was>thirsty>so>i>stopped>... | mioe>pljep>was>thinst>asol>i>s | irnpynnpmwnlmvmoae |

The RCNN predictions preserve word structure and are partially readable — "mwant" (want), "rme>to" (me>to), "pljep>was>thinst" (jeep>was>thirst) — while the Conformer degrades catastrophically on longer sentences. This pattern of strong short-sequence performance but poor length generalisation is characteristic of attention-based models trained on limited data.

### 3.6 Computational Efficiency

The Conformer required approximately 6x longer training time (1854–1867s) compared to GRU and RCNN (281–325s), due to the quadratic memory complexity of self-attention over T = 3000 time steps and the reduced batch size (4 vs. 16) necessitated by GPU memory constraints. CTC was the fastest to train (211s) owing to its simpler architecture and lack of frame-level loss computation.

---

## 4. Discussion

### 4.1 Decoder Architecture Selection Under Data Constraints

My results demonstrate that architectural choice interacts critically with data availability. The RCNN's inductive biases — local feature extraction via convolutions followed by sequential integration via recurrence — proved well-matched to neural handwriting decoding under severe data limitations. In contrast, the Conformer's flexible self-attention mechanism, which has driven advances in data-rich speech recognition, failed to provide benefits and instead led to overfitting.

This finding has practical implications for BCI deployment: when calibration data is limited (as is typical in clinical settings where recording time is constrained by participant fatigue), architectures with stronger inductive biases should be preferred over more flexible alternatives.

### 4.2 The Alignment Bottleneck

The large performance gap between Gaussian and Poisson alignment conditions underscores that alignment quality is the dominant factor in the decoding pipeline. Even the best decoder (RCNN) could not compensate for the noisy frame labels produced by Poisson HMM alignment. This suggests that future work should prioritise alignment quality improvements — such as discriminative alignment training, multi-session template estimation, or end-to-end alignment learning — over decoder architecture exploration.

The Poisson HMM's underperformance does not necessarily invalidate the Poisson emission model. Rather, my results indicate that the advantages of a theoretically correct generative model are overwhelmed by practical factors: template estimation noise from limited single-letter data, suboptimal HMM hyperparameters (bin size, transition probabilities), and the lack of the extensive calibration pipeline that Willett et al. developed for their Gaussian HMM.

### 4.3 Soft Labels: A Low-Cost Improvement

The consistent improvement in frame-level accuracy from soft labels (+1.8 to +5.0 pp) represents a "free" improvement that requires no additional data or computation — only using the probability distributions already produced by HMM alignment rather than discarding them via argmax. I recommend soft-target training as a default for future neural decoding work using HMM-aligned labels.

### 4.4 Limitations

Several limitations constrain the interpretation of these results:

1. **Single session**: All experiments used one recording session (89 training sentences). Willett et al. trained on data aggregated across 10+ sessions with synthetic data augmentation, yielding effectively thousands of training examples. Performance differences between architectures may change substantially with more data — in particular, the Conformer may become competitive or superior with sufficient training data.

2. **No recurrent language model**: Willett et al.'s 5.32% CER relied critically on RNNLM decoding. My bigram LM provides only rudimentary linguistic constraint. Integrating a pre-trained character-level language model would likely reduce CER substantially across all conditions.

3. **Single participant**: The dataset contains recordings from one participant (T5). Generalisability to other participants, electrode placements, or neurological conditions remains unknown.

4. **Hyperparameter optimisation**: I did not perform systematic hyperparameter tuning. The Conformer in particular may benefit from different learning rate schedules, dropout rates, or architectural configurations optimised for the low-data regime.

### 4.5 Comparison with Willett et al.

Direct comparison with Willett et al.'s reported 5.32% CER is not appropriate, as my experimental conditions differ in several critical ways: single-session vs. multi-session training, bigram vs. RNNLM decoding, 80 vs. thousands of training epochs, and minimal vs. extensive data augmentation. This work is complementary — it provides a controlled comparison of architectural and alignment choices that was not the focus of the original study.

### 4.6 Future Directions

Several directions could substantially improve performance:

1. **Multi-session training**: Aggregating data across all available sessions would increase training data by approximately 10x.
2. **Synthetic data augmentation**: Time-warping, electrode dropout, and neural activity interpolation could further expand the effective training set.
3. **Pre-trained language model integration**: Replacing the bigram LM with a character-level transformer or LSTM language model trained on large English text corpora.
4. **End-to-end training**: Jointly optimising alignment and decoding, potentially using CTC or attention-based sequence-to-sequence models with curriculum learning to bootstrap from the limited data.
5. **Hybrid architectures**: Combining the RCNN's convolutional front-end with Conformer blocks (rather than GRU layers) may capture the benefits of both architectures while mitigating the Conformer's data requirements.

---

## 5. Conclusion

I presented a systematic comparison of decoder architectures and alignment strategies for neural decoding of imagined handwriting, evaluated on the publicly available Willett et al. (2021) dataset. The key findings are:

1. The **RCNN** (convolutional-recurrent hybrid) outperformed all other architectures in the low-data regime, achieving 65.97% CER — a 12.5 percentage point improvement over the GRU baseline with the same alignment.
2. **Soft probability targets** from HMM alignment improved frame-level accuracy by 1.8–5.0 percentage points over hard labels, representing a zero-cost enhancement.
3. The **Conformer** architecture, despite strong frame-level performance, failed to generalise at the sequence level due to overfitting — an important negative result for the BCI community.
4. **Poisson HMM** alignment underperformed Gaussian HMM by 17+ percentage points, indicating that template estimation quality outweighs emission model correctness in the current paradigm.

These findings provide actionable guidance for neural BCI decoder design under clinical data constraints and identify alignment quality as the primary bottleneck for future improvement.

---

## References

Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., & Pang, R. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Interspeech 2020*, 5036–5040.

Graves, A., Fernandez, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. *Proceedings of the 23rd International Conference on Machine Learning*, 369–376.

Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., & Shenoy, K. V. (2021). High-performance brain-to-text communication via handwriting. *Nature*, 593(7858), 249–254.
