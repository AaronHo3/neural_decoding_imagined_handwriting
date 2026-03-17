"""
Gaussian HMM for forced alignment of neural data to character sequences.

Baseline implementation following Willett et al. (Nature 2021).
Uses Gaussian emission distributions (mean firing rates + diagonal covariance).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d


class GaussianHMMForcedAlignment:
    """
    Forced-alignment HMM with Gaussian emissions.

    Each character in the target sentence is modelled by `hmm_bin_size`
    consecutive HMM states, each corresponding to one time-bin of the
    character template.  Viterbi decoding enforces a strict left-to-right
    traversal (every character must be produced exactly once) while
    allowing the decoder to:
        - stay in the current state  (stay_prob)
        - skip one state             (skip_prob)
        - advance normally           (1 − stay_prob − skip_prob)

    blank_prob modulates the emission probability for blank / inter-character
    states (currently used as an additive floor on the log-likelihood).
    """

    def __init__(
        self,
        hmm_bin_size: int = 5,
        blank_prob: float = 0.1,
        stay_prob: float = 0.2,
        skip_prob: float = 0.2,
    ):
        self.hmm_bin_size = hmm_bin_size
        self.blank_prob = blank_prob
        self.stay_prob = stay_prob
        self.skip_prob = skip_prob

        # Filled by fit_templates
        self.templates_: Dict[str, np.ndarray] = {}
        self.sigma_: float = 1.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_templates(
        self,
        tw_cubes: Dict[str, np.ndarray],
        char_def: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Build per-character mean-rate templates from time-warped data.

        Args:
            tw_cubes: {char: ndarray (n_trials, T_warped, N_channels)}
                      Time-warped neural activity for each character.
            char_def: Character definitions dict (may contain 'charList',
                      'numTimeBins', etc.).  Not strictly required — the
                      method processes every key present in tw_cubes.

        Returns:
            templates: {char: ndarray (hmm_bin_size, N_channels)}
        """
        templates: Dict[str, np.ndarray] = {}
        all_residuals = []

        for char, cube in tw_cubes.items():
            cube = np.asarray(cube, dtype=float)
            if cube.ndim == 2:
                cube = cube[np.newaxis]  # (1, T, N)

            # Mean across trials → (T_warped, N)
            mean_template = cube.mean(axis=0)
            N = mean_template.shape[1]

            # Resample T_warped → hmm_bin_size if needed
            T_orig = mean_template.shape[0]
            if T_orig != self.hmm_bin_size:
                t_old = np.linspace(0.0, 1.0, T_orig)
                t_new = np.linspace(0.0, 1.0, self.hmm_bin_size)
                f = interp1d(t_old, mean_template, axis=0, kind="linear",
                             fill_value="extrapolate")
                mean_template = f(t_new)  # (hmm_bin_size, N)

            templates[char] = mean_template

            # Collect residuals for global sigma estimate
            # Resample each trial and compute residual
            for trial in cube:
                if trial.shape[0] != self.hmm_bin_size:
                    t_old = np.linspace(0.0, 1.0, trial.shape[0])
                    t_new = np.linspace(0.0, 1.0, self.hmm_bin_size)
                    f = interp1d(t_old, trial, axis=0, kind="linear",
                                 fill_value="extrapolate")
                    trial_r = f(t_new)
                else:
                    trial_r = trial
                all_residuals.append((trial_r - mean_template).ravel())

        if all_residuals:
            residuals = np.concatenate(all_residuals)
            self.sigma_ = max(float(np.std(residuals)), 1e-3)
        else:
            self.sigma_ = 1.0

        self.templates_ = templates
        return templates

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def align(
        self,
        obs: np.ndarray,
        sentence: str,
        templates: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find character boundaries via Viterbi forced alignment.

        Args:
            obs:       Neural activity matrix (T, N_channels).
            sentence:  Target sentence string (each character will be aligned).
            templates: {char: (hmm_bin_size, N)} mean templates.

        Returns:
            letter_starts    : (len(sentence),) int — start time index per char
            letter_durations : (len(sentence),) int — duration in time bins
        """
        obs = np.asarray(obs, dtype=float)
        T, N = obs.shape
        L = len(sentence)
        B = self.hmm_bin_size
        n_states = L * B

        # --- Build emission matrix (T, n_states) ---
        # Stack templates into (n_states, N)
        mu = np.zeros((n_states, N))
        for i, ch in enumerate(sentence):
            tmpl = templates.get(ch, templates.get(" ", np.zeros((B, N))))
            if tmpl.shape[0] != B:
                t_old = np.linspace(0, 1, tmpl.shape[0])
                t_new = np.linspace(0, 1, B)
                f = interp1d(t_old, tmpl, axis=0, kind="linear",
                             fill_value="extrapolate")
                tmpl = f(t_new)
            mu[i * B: (i + 1) * B] = tmpl

        sigma2 = self.sigma_ ** 2
        # log p(obs[t] | state s) = -0.5 * ||obs[t] - mu[s]||^2 / sigma^2 + const
        # Vectorised: (T, 1, N) - (1, n_states, N) → (T, n_states)
        diff = obs[:, np.newaxis, :] - mu[np.newaxis, :, :]  # (T, S, N)
        log_em = -0.5 * np.sum(diff ** 2, axis=-1) / sigma2  # (T, S)

        # Add a small floor for blank probability
        log_em += np.log(max(self.blank_prob, 1e-10))

        # --- Viterbi ---
        path = _viterbi(log_em, L, B, self.stay_prob, self.skip_prob)

        return _extract_segmentation(path, L, B)


# ======================================================================
# Shared helpers (used by both HMM variants)
# ======================================================================

def _viterbi(
    log_em: np.ndarray,
    L: int,
    B: int,
    stay_prob: float,
    skip_prob: float,
) -> np.ndarray:
    """
    Log-domain Viterbi for a left-to-right forced-alignment HMM.

    States:  s = i*B + j  (character i, bin j)
    Transitions (from state s = i*B + j):
        stay  → s              prob = stay_prob
        adv   → s+1            prob depends on j (see below)
        skip  → s+2            prob = skip_prob  (when valid)

    Args:
        log_em  : (T, n_states) log emission probabilities
        L, B    : number of characters and bins-per-character
        stay_prob, skip_prob: transition parameters

    Returns:
        path: (T,) state indices
    """
    T, n_states = log_em.shape
    advance_prob = max(1.0 - stay_prob - skip_prob, 1e-10)

    log_stay    = np.log(stay_prob    + 1e-10)
    log_advance = np.log(advance_prob + 1e-10)
    log_skip    = np.log(skip_prob    + 1e-10)
    # When skip is impossible, all remaining probability goes to advance
    log_adv_only = np.log(max(1.0 - stay_prob, 1e-10))

    viterbi   = np.full((T, n_states), -np.inf)
    backtrack = np.zeros((T, n_states), dtype=np.int32)

    # Initialise: must start in state 0
    viterbi[0, 0] = log_em[0, 0]

    for t in range(1, T):
        for s in range(n_states):
            i = s // B
            j = s % B

            best_score = -np.inf
            best_prev  = s  # default: stay

            # 1) Stay
            v = viterbi[t - 1, s] + log_stay
            if v > best_score:
                best_score = v
                best_prev  = s

            # 2) Advance from s-1
            if s > 0:
                prev_s = s - 1
                pi = prev_s // B
                pj = prev_s % B

                # Within same character
                if pi == i and pj == j - 1:
                    lp = log_adv_only if pj >= B - 2 else log_advance
                    v  = viterbi[t - 1, prev_s] + lp
                    if v > best_score:
                        best_score = v
                        best_prev  = prev_s

                # Cross character boundary (last bin → first bin)
                elif pi == i - 1 and pj == B - 1 and j == 0:
                    v = viterbi[t - 1, prev_s] + log_adv_only
                    if v > best_score:
                        best_score = v
                        best_prev  = prev_s

            # 3) Skip from s-2
            if s > 1:
                prev_s = s - 2
                pi = prev_s // B
                pj = prev_s % B

                valid = False
                if pi == i and j == pj + 2 and pj < B - 2:
                    # Skip within character
                    valid = True
                elif pi == i - 1:
                    if pj == B - 2 and j == 0:
                        # Skip from second-to-last bin → first bin of next char
                        valid = True
                    elif pj == B - 1 and j == 1 and B > 1:
                        # Skip from last bin → second bin of next char
                        valid = True

                if valid:
                    v = viterbi[t - 1, prev_s] + log_skip
                    if v > best_score:
                        best_score = v
                        best_prev  = prev_s

            viterbi[t, s]   = best_score + log_em[t, s]
            backtrack[t, s] = best_prev

    # Force backtrack from the last state
    path = np.empty(T, dtype=np.int32)
    path[T - 1] = n_states - 1
    for t in range(T - 1, 0, -1):
        path[t - 1] = backtrack[t, path[t]]

    return path


def _extract_segmentation(
    path: np.ndarray,
    L: int,
    B: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a Viterbi state path to per-character start times and durations."""
    char_indices = path // B  # which character is active at each timestep

    letter_starts    = np.zeros(L, dtype=int)
    letter_durations = np.zeros(L, dtype=int)

    for i in range(L):
        times = np.where(char_indices == i)[0]
        if len(times) > 0:
            letter_starts[i]    = int(times[0])
            letter_durations[i] = int(len(times))
        else:
            # Fallback: equally space missing characters
            letter_starts[i]    = i * (len(path) // L)
            letter_durations[i] = max(1, len(path) // L)

    return letter_starts, letter_durations
