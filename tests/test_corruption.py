"""
Tests for the RQ1 label-corruption models.

These functions define the independent variable of the whole experiment, so
their behaviour has to be pinned down: if `jitter_boundaries` silently changed
character identities, or `corrupt_segments` moved boundaries, the sweep would
be measuring something other than what it claims to measure.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp1_alignment_sensitivity import (  # noqa: E402
    _segments,
    corrupt_segments,
    jitter_boundaries,
)


def make_labels():
    """Two sentences: 3 segments each, flanked by -1 padding."""
    row0 = np.array([-1] * 5 + [0] * 20 + [1] * 20 + [2] * 20 + [-1] * 5)
    row1 = np.array([-1] * 5 + [7] * 30 + [8] * 15 + [9] * 15 + [-1] * 5)
    return np.stack([row0, row1])


class TestSegments:
    def test_finds_all_runs_and_skips_padding(self):
        segs = _segments(make_labels()[0])
        assert segs == [(5, 25, 0), (25, 45, 1), (45, 65, 2)]

    def test_all_padding_yields_nothing(self):
        assert _segments(np.array([-1, -1, -1])) == []

    def test_single_frame_segment(self):
        assert _segments(np.array([-1, 4, -1])) == [(1, 2, 4)]

    def test_repeated_label_in_separate_runs_stays_separate(self):
        # "ll" in "hello" must not collapse into one segment
        segs = _segments(np.array([3, 3, 5, 5, 3, 3]))
        assert segs == [(0, 2, 3), (2, 4, 5), (4, 6, 3)]


class TestJitterBoundaries:
    def test_sigma_zero_is_identity(self):
        labels = make_labels()
        assert np.array_equal(jitter_boundaries(labels, 0, np.random.default_rng(0)),
                              labels)

    def test_preserves_character_sequence(self):
        # The whole point of jitter: identities and order survive, times move.
        labels = make_labels()
        out = jitter_boundaries(labels, 8, np.random.default_rng(1))
        for orig_row, new_row in zip(labels, out):
            orig_seq = [lab for _, _, lab in _segments(orig_row)]
            new_seq = [lab for _, _, lab in _segments(new_row)]
            # Segments may be squeezed to zero width, so the new sequence is a
            # subsequence of the original, never a reordering or substitution.
            assert new_seq == [c for c in orig_seq if c in new_seq]
            assert set(new_seq) <= set(orig_seq)

    def test_does_not_extend_beyond_original_span(self):
        labels = make_labels()
        out = jitter_boundaries(labels, 20, np.random.default_rng(2))
        for orig_row, new_row in zip(labels, out):
            orig_active = np.where(orig_row >= 0)[0]
            new_active = np.where(new_row >= 0)[0]
            assert new_active.min() >= orig_active.min()
            assert new_active.max() <= orig_active.max()

    def test_boundaries_stay_monotone(self):
        labels = make_labels()
        out = jitter_boundaries(labels, 15, np.random.default_rng(3))
        for row in out:
            starts = [s for s, _, _ in _segments(row)]
            assert starts == sorted(starts)

    def test_larger_sigma_moves_more_labels(self):
        labels = make_labels()
        active = labels >= 0

        def disagreement(sigma, seed):
            out = jitter_boundaries(labels, sigma, np.random.default_rng(seed))
            return (out[active] != labels[active]).mean()

        small = np.mean([disagreement(3, s) for s in range(12)])
        large = np.mean([disagreement(25, s) for s in range(12)])
        assert large > small


class TestCorruptSegments:
    def test_p_zero_is_identity(self):
        labels = make_labels()
        assert np.array_equal(corrupt_segments(labels, 0.0, np.random.default_rng(0)),
                              labels)

    def test_p_one_relabels_every_segment(self):
        labels = make_labels()
        out = corrupt_segments(labels, 1.0, np.random.default_rng(4))
        for orig_row, new_row in zip(labels, out):
            for (s, e, lab) in _segments(orig_row):
                assert np.all(new_row[s:e] != lab)

    def test_preserves_boundaries_exactly(self):
        # Only identities change; the segmentation must be untouched.
        labels = make_labels()
        out = corrupt_segments(labels, 1.0, np.random.default_rng(5))
        for orig_row, new_row in zip(labels, out):
            assert ([(s, e) for s, e, _ in _segments(orig_row)]
                    == [(s, e) for s, e, _ in _segments(new_row)])

    def test_never_touches_padding(self):
        labels = make_labels()
        out = corrupt_segments(labels, 1.0, np.random.default_rng(6))
        assert np.array_equal(labels == -1, out == -1)

    def test_segment_stays_internally_constant(self):
        labels = make_labels()
        out = corrupt_segments(labels, 0.5, np.random.default_rng(7))
        for orig_row, new_row in zip(labels, out):
            for (s, e, _) in _segments(orig_row):
                assert len(np.unique(new_row[s:e])) == 1

    def test_corruption_rate_tracks_p(self):
        labels = make_labels()
        rng = np.random.default_rng(8)
        changed = [
            np.mean([
                lab != new_row[s]
                for row, new_row in zip(labels, corrupt_segments(labels, 0.5, rng))
                for (s, _, lab) in _segments(row)
            ])
            for _ in range(200)
        ]
        assert np.mean(changed) == pytest.approx(0.5, abs=0.06)

    def test_is_deterministic_given_seed(self):
        labels = make_labels()
        a = corrupt_segments(labels, 0.3, np.random.default_rng(99))
        b = corrupt_segments(labels, 0.3, np.random.default_rng(99))
        assert np.array_equal(a, b)
