"""Forced alignment of words onto the voice.

The property that matters is the one measured on a real track: a word must
never be highlighted while nobody is singing. Whisper placed more than a third
of its words in silence, which is what made the karaoke feel wrong regardless
of any constant offset.

Synthetic voiced regions here, so the tests state the guarantee directly rather
than depending on what a model produced today.
"""

from __future__ import annotations

import numpy as np
import pytest

from singcoach.analysis import lyrics as L


def contour_with_gaps(spans: list[tuple[float, float]], hop: float = 0.01,
                      total: float = 10.0) -> np.ndarray:
    """A pitch contour that is voiced only inside ``spans``."""
    n = int(total / hop)
    out = np.full(n, np.nan)
    for start, end in spans:
        out[int(start / hop):int(end / hop)] = 60.0
    return out


def W(text: str, start: float, end: float) -> L.Word:
    return L.Word(text=text, start=start, end=end)


class TestVoicedSpans:
    def test_finds_the_singing(self):
        c = contour_with_gaps([(1.0, 2.0), (4.0, 5.5)])
        spans = L.voiced_spans(c, 0.01)
        assert len(spans) == 2
        assert spans[0][0] == pytest.approx(1.0, abs=0.05)
        assert spans[1][1] == pytest.approx(5.5, abs=0.05)

    def test_merges_tiny_gaps(self):
        """A consonant briefly stops phonation mid-word; that is not a new phrase."""
        c = contour_with_gaps([(1.0, 1.5), (1.58, 2.2)])
        assert len(L.voiced_spans(c, 0.01, merge_gap_s=0.18)) == 1

    def test_discards_blips(self):
        c = contour_with_gaps([(1.0, 1.03), (4.0, 5.0)])
        spans = L.voiced_spans(c, 0.01, min_span_s=0.12)
        assert len(spans) == 1
        assert spans[0][0] == pytest.approx(4.0, abs=0.05)

    def test_silence_yields_nothing(self):
        assert L.voiced_spans(np.full(500, np.nan), 0.01) == []


class TestAlignToVoice:
    def test_no_word_is_ever_placed_in_silence(self):
        """The guarantee. On the real track this went from 64% to 100%."""
        spans = [(1.0, 2.0), (4.0, 5.5)]
        words = [W("a", 0.2, 0.4), W("b", 2.9, 3.1), W("c", 4.2, 4.4), W("d", 7.0, 7.2)]
        aligned = L.align_to_voice(words, spans)
        for w in aligned:
            assert any(s <= w.start < e for s, e in spans), (
                f"{w.text!r} placed at {w.start} which is not inside any voiced span"
            )

    def test_word_order_is_preserved(self):
        spans = [(1.0, 2.0), (4.0, 5.5)]
        words = [W(str(i), i * 0.7, i * 0.7 + 0.3) for i in range(6)]
        aligned = L.align_to_voice(words, spans)
        assert [w.text for w in aligned] == [str(i) for i in range(6)]
        for a, b in zip(aligned, aligned[1:]):
            assert a.start <= b.start

    def test_a_line_is_stretched_to_fill_its_phrase(self):
        spans = [(1.0, 2.0)]
        words = [W("aa", 1.0, 1.2), W("bb", 1.3, 1.5), W("cc", 1.6, 1.9)]
        aligned = L.align_to_voice(words, spans)
        assert aligned[0].start == pytest.approx(1.0, abs=0.02)
        assert aligned[-1].end == pytest.approx(2.0, abs=0.02)

    def test_the_transcriber_s_rhythm_is_preserved(self):
        """The whole reason for this design.

        Spreading words evenly across the voiced timeline placed them
        correctly but destroyed pacing — median word length went to 0.98 s
        because a held vowel is one long voiced stretch. Whisper's *relative*
        spacing within a phrase is good; only its placement of the phrase is
        bad. So gaps between words must survive re-anchoring in proportion.
        """
        spans = [(10.0, 12.0)]
        # Second word twice as far from the first as the third is from the
        # second. Gaps stay under the line-break threshold so this is one line.
        words = [W("a", 0.0, 0.2), W("b", 0.5, 0.7), W("c", 0.75, 0.95)]
        aligned = L.align_to_voice(words, spans)
        first_gap = aligned[1].start - aligned[0].start
        second_gap = aligned[2].start - aligned[1].start
        assert first_gap / second_gap == pytest.approx(2.0, rel=0.15)

    def test_words_do_not_crawl(self):
        """No word should be stretched across an entire held note."""
        spans = [(0.0, 4.0)]
        words = [W(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(8)]
        aligned = L.align_to_voice(words, spans)
        durations = [w.end - w.start for w in aligned]
        assert max(durations) < 1.0

    def test_no_words_or_no_spans_is_a_no_op(self):
        assert L.align_to_voice([], [(1.0, 2.0)]) == []
        words = [W("a", 1.0, 1.2)]
        assert L.align_to_voice(words, []) == words

    def test_records_how_far_each_word_moved(self):
        aligned = L.align_to_voice([W("a", 5.0, 5.2)], [(1.0, 2.0)])
        assert aligned[0].snapped_by == pytest.approx(-4.0, abs=0.01)


class TestSyllables:
    @pytest.mark.parametrize(
        "word,expected",
        [("a", 1), ("the", 1), ("whiskey", 2), ("beautiful", 3), ("rhythm", 1), ("", 1)],
    )
    def test_counts(self, word, expected):
        assert L._syllables(word) == expected

    def test_never_returns_zero(self):
        """A zero would divide by zero when sharing out a phrase."""
        for token in ("", "!!!", "123", "shh"):
            assert L._syllables(token) >= 1
