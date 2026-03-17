"""
Poisson HMM for forced alignment of neural data to character sequences.

Principled count-based emissions for neural spike data.
Differentiates from Willett's Gaussian HMM by using Poisson (or negative binomial)
emission distributions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import gammaln

# Re-use the shared Viterbi and segmentation helpers from the Gaussian module
from .gaussian_hmm import _viterbi, _extract_segmentation


class PoissonHMMForcedAlignment:
    """
    Forced-alignment HMM with Poisson emissions.

    Neural spike counts are modelled as P(k | λ) where λ is the expected
    firing rate for each state.  More appropriate for count data than Gaussian.

    Optionally uses a Negative Binomial distribution (controlled by
    `use_negative_binomial`) which adds a per-channel dispersion parameter r,
    letting the variance exceed the mean.
    """

    def __init__(
        self,
        hmm_bin_size: int = 5,
        blank_prob: float = 0.1,
        stay_prob: float = 0.2,
        skip_prob: float = 0.2,
        use_negative_binomial: bool = False,
    ):
        self.hmm_bin_size = hmm_bin_size
        self.blank_prob = blank_prob
        self.stay_prob = stay_prob
        self.skip_prob = skip_prob
        self.use_negative_binomial = use_negative_binomial

        # Filled by fit_templates
        self.templates_: Dict[str, np.ndarray] = {}
        # For NB: dispersion r per channel (scalar or array)
        self.dispersion_: np.ndarray = np.array([1.0])

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_templates(
        self,
        tw_cubes: Dict[str, np.ndarray],
        char_def: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate per-character rate templates (λ) from time-warped data.

        Args:
            tw_cubes: {char: ndarray (n_trials, T_warped, N_channels)}
            char_def: Character definitions dict (informational; unused directly).

        Returns:
            templates: {char: ndarray (hmm_bin_size, N_channels)} — mean rates λ
        """
        templates: Dict[str, np.ndarray] = {}
        all_rates  = []   # for NB dispersion estimation
        all_vars   = []

        for char, cube in tw_cubes.items():
            cube = np.asarray(cube, dtype=float)
            if cube.ndim == 2:
                cube = cube[np.newaxis]

            # Mean across trials → expected rate (T_warped, N)
            mean_rate = np.maximum(cube.mean(axis=0), 1e-6)
            N = mean_rate.shape[1]

            # Resample to hmm_bin_size
            T_orig = mean_rate.shape[0]
            if T_orig != self.hmm_bin_size:
                t_old = np.linspace(0.0, 1.0, T_orig)
                t_new = np.linspace(0.0, 1.0, self.hmm_bin_size)
                f = interp1d(t_old, mean_rate, axis=0, kind="linear",
                             fill_value="extrapolate")
                mean_rate = np.maximum(f(t_new), 1e-6)

            templates[char] = mean_rate

            if self.use_negative_binomial:
                # Estimate per-character variance for NB dispersion
                var_rate = cube.var(axis=0)
                if T_orig != self.hmm_bin_size:
                    fv = interp1d(
                        np.linspace(0, 1, T_orig), var_rate, axis=0,
                        kind="linear", fill_value="extrapolate"
                    )
                    var_rate = np.maximum(fv(t_new), 1e-6)
                all_rates.append(mean_rate.ravel())
                all_vars.append(var_rate.ravel())

        self.templates_ = templates

        if self.use_negative_binomial and all_rates:
            mu  = np.concatenate(all_rates)
            var = np.maximum(np.concatenate(all_vars), mu + 1e-6)
            # Method of moments: r = mu^2 / (var - mu)
            r = np.maximum(mu ** 2 / (var - mu), 1e-3)
            self.dispersion_ = np.array([float(r.mean())])

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
        Find character boundaries via Viterbi forced alignment with Poisson
        (or Negative Binomial) emissions.

        Args:
            obs:       (T, N_channels) binned spike counts (non-negative integers).
            sentence:  Target sentence string.
            templates: {char: (hmm_bin_size, N)} expected rate templates λ.

        Returns:
            letter_starts    : (len(sentence),) int
            letter_durations : (len(sentence),) int
        """
        obs = np.asarray(obs, dtype=float)
        T, N = obs.shape
        L = len(sentence)
        B = self.hmm_bin_size
        n_states = L * B

        # --- Build rate matrix λ : (n_states, N) ---
        lam = np.zeros((n_states, N))
        for i, ch in enumerate(sentence):
            tmpl = templates.get(ch, templates.get(" ", np.ones((B, N)) * 1e-6))
            if tmpl.shape[0] != B:
                t_old = np.linspace(0, 1, tmpl.shape[0])
                t_new = np.linspace(0, 1, B)
                f = interp1d(t_old, tmpl, axis=0, kind="linear",
                             fill_value="extrapolate")
                tmpl = f(t_new)
            lam[i * B: (i + 1) * B] = np.maximum(tmpl, 1e-6)

        # --- Compute log-emission matrix (T, n_states) ---
        if self.use_negative_binomial:
            log_em = self._nb_log_emission(obs, lam)
        else:
            log_em = self._poisson_log_emission(obs, lam)

        # Add blank floor
        log_em += np.log(max(self.blank_prob, 1e-10))

        # --- Viterbi (shared implementation) ---
        path = _viterbi(log_em, L, B, self.stay_prob, self.skip_prob)

        return _extract_segmentation(path, L, B)

    # ------------------------------------------------------------------
    # Emission helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_log_emission(
        obs: np.ndarray,
        lam: np.ndarray,
    ) -> np.ndarray:
        """
        log P(obs[t] | λ[s]) = Σ_n [ k_n log λ_n − λ_n − log(k_n!) ]

        Args:
            obs : (T, N)
            lam : (S, N)

        Returns:
            log_em : (T, S)
        """
        # log(k!) via gammaln(k+1)
        log_k_fact = gammaln(obs + 1).sum(axis=-1, keepdims=True)  # (T, 1)

        # k * log(λ) summed over neurons: (T, 1, N) * (1, S, N) → (T, S)
        k_log_lam = (obs[:, np.newaxis, :] * np.log(lam)[np.newaxis, :, :]).sum(axis=-1)

        # sum_n λ_n over states: (1, S)
        sum_lam = lam.sum(axis=-1, keepdims=True).T  # (1, S)

        return k_log_lam - sum_lam - log_k_fact

    def _nb_log_emission(
        self,
        obs: np.ndarray,
        lam: np.ndarray,
    ) -> np.ndarray:
        """
        Negative Binomial log-likelihood.

        NB(k | r, μ): log P = log C(k+r-1, k) + r·log(r/(r+μ)) + k·log(μ/(r+μ))

        Uses a scalar global dispersion r estimated in fit_templates.

        Args:
            obs : (T, N)
            lam : (S, N)  — mean rates μ

        Returns:
            log_em : (T, S)
        """
        r   = float(self.dispersion_[0])
        mu  = lam  # (S, N)

        # Per-neuron log NB, summed across neurons
        # log C(k+r-1, k) = gammaln(k+r) - gammaln(k+1) - gammaln(r)
        # (T, N, 1) for obs broadcasts with (1, N, S) for mu → tricky
        # Work in (T, S) by summation

        T, N = obs.shape
        S    = mu.shape[0]

        log_p  = np.log(mu / (r + mu + 1e-10))  # (S, N)
        log_1p = np.log(r   / (r + mu + 1e-10))  # (S, N)

        # k * log(p)  summed over N: (T, S)
        k_log_p = (obs[:, np.newaxis, :] * log_p[np.newaxis, :, :]).sum(axis=-1)
        # r * log(1-p) summed: (T, S)  — actually constant across T per state
        r_log_1p = (r * log_1p).sum(axis=-1)  # (S,)

        # Binomial coefficient term: sum over N
        log_binom = (
            gammaln(obs + r) - gammaln(obs + 1) - gammaln(r)
        ).sum(axis=-1, keepdims=True)  # (T, 1)

        return log_binom + k_log_p + r_log_1p[np.newaxis, :]
