"""Tests for the error-rate metrics used in every reported result."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.evaluate import (  # noqa: E402
    _levenshtein,
    compute_character_error_rate,
    compute_word_error_rate,
)


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein(list("kitten"), list("kitten")) == 0

    def test_classic_example(self):
        # kitten → sitten → sittin → sitting
        assert _levenshtein(list("kitten"), list("sitting")) == 3

    def test_empty_against_nonempty(self):
        assert _levenshtein([], list("abc")) == 3
        assert _levenshtein(list("abc"), []) == 3

    def test_symmetric(self):
        a, b = list("flaw"), list("lawn")
        assert _levenshtein(a, b) == _levenshtein(b, a)

    def test_pure_insertion(self):
        assert _levenshtein(list("ac"), list("abc")) == 1


class TestCER:
    def test_perfect_prediction_is_zero(self):
        assert compute_character_error_rate(["hello"], ["hello"]) == 0.0

    def test_single_substitution(self):
        # one edit over a 5-char reference
        assert compute_character_error_rate(["hallo"], ["hello"]) == pytest.approx(0.2)

    def test_empty_prediction_is_total_loss(self):
        assert compute_character_error_rate([""], ["hello"]) == pytest.approx(1.0)

    def test_can_exceed_one_when_prediction_is_too_long(self):
        # CER is normalised by reference length, so over-generation exceeds 1.0.
        # Guards the claim in RESULTS.md that a >100% error rate is legitimate
        # rather than a bug. 15-char prediction vs 5-char reference = 10 edits.
        assert compute_character_error_rate(["hellohelloworld"], ["hello"]) == pytest.approx(2.0)

    def test_averages_over_sequences_not_characters(self):
        # 0.0 and 1.0 → 0.5, regardless of differing reference lengths
        cer = compute_character_error_rate(["ab", ""], ["ab", "abcdefgh"])
        assert cer == pytest.approx(0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_character_error_rate(["a"], ["a", "b"])


class TestWER:
    def test_perfect(self):
        assert compute_word_error_rate(["the cat sat"], ["the cat sat"]) == 0.0

    def test_one_wrong_word_of_three(self):
        wer = compute_word_error_rate(["the dog sat"], ["the cat sat"])
        assert wer == pytest.approx(1 / 3)

    def test_operates_on_words_not_characters(self):
        # Every word differs by one character, so char-level would give a small
        # rate; word-level must give 1.0.
        assert compute_word_error_rate(["aa bb"], ["ab bc"]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_word_error_rate(["a"], ["a", "b"])
