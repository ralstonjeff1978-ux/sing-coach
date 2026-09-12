"""Lyric timing logic, tested on synthetic words.

Deliberately no real song here. The behaviour under test is the timing maths —
onset snapping, line breaking, .lrc round-tripping — none of which cares what
the words actually say. Synthetic tokens make the tests fast, deterministic,
and independent of whatever a transcription model happens to produce today.
"""

from __future__ import annotations

from singcoach.analysis import lyrics as L


def W(text: str, start: float, end: float, prob: float = 0.9) -> L.Word:
    return L.Word(text=text, start=start, end=end, probability=prob)


class TestOnsetSnapping:
    def test_word_snaps_to_a_nearby_note_onset(self):
        words = [W("aaa", 1.08, 1.50)]
        snapped = L.snap_to_onsets(words, [1.00, 2.00])
        assert snapped[0].start == 1.00
        assert snapped[0].snapped_by == -0.08

    def test_distant_onset_is_ignored(self):
        """A word must never be dragged onto an unrelated note."""
        words = [W("aaa", 1.80, 2.20)]
        snapped = L.snap_to_onsets(words, [1.00, 3.00])
        assert snapped[0].start == 1.80
        assert snapped[0].snapped_by == 0.0

    def test_snapping_never_reorders_words(self):
        words = [W("aaa", 1.00, 1.40), W("bbb", 1.45, 1.90)]
        snapped = L.snap_to_onsets(words, [1.00, 1.30])
        assert snapped[0].start <= snapped[1].start
        assert all(w.start < w.end for w in snapped)

    def test_snapping_never_inverts_a_word(self):
        """An onset past the word's own end must be refused."""
        words = [W("aaa", 1.00, 1.10)]
        snapped = L.snap_to_onsets(words, [1.20])
        assert snapped[0].start == 1.00

    def test_no_onsets_is_a_no_op(self):
        words = [W("aaa", 1.0, 1.5), W("bbb", 2.0, 2.5)]
        assert L.snap_to_onsets(words, []) == words

    def test_word_text_and_confidence_survive(self):
        words = [W("aaa", 1.08, 1.50, prob=0.42)]
        snapped = L.snap_to_onsets(words, [1.00])
        assert snapped[0].text == "aaa"
        assert snapped[0].probability == 0.42


class TestLineGrouping:
    def test_breaks_on_a_pause(self):
        words = [W("a", 0.0, 0.3), W("b", 0.3, 0.6), W("c", 2.0, 2.3)]
        lines = L.group_lines(words, break_s=0.7)
        assert len(lines) == 2
        assert len(lines[0].words) == 2
        assert len(lines[1].words) == 1

    def test_does_not_break_mid_phrase(self):
        words = [W(str(i), i * 0.3, i * 0.3 + 0.25) for i in range(5)]
        lines = L.group_lines(words, break_s=0.7, max_words=10)
        assert len(lines) == 1

    def test_caps_line_length(self):
        words = [W(str(i), i * 0.3, i * 0.3 + 0.25) for i in range(25)]
        lines = L.group_lines(words, break_s=5.0, max_words=8)
        assert all(len(ln.words) <= 8 for ln in lines)
        assert sum(len(ln.words) for ln in lines) == 25

    def test_line_bounds_match_its_words(self):
        words = [W("a", 1.0, 1.3), W("b", 1.4, 1.9)]
        line = L.group_lines(words)[0]
        assert line.start == 1.0
        assert line.end == 1.9

    def test_empty_input(self):
        assert L.group_lines([]) == []


class TestLookup:
    def _lyrics(self) -> L.Lyrics:
        words = [W("a", 1.0, 1.4), W("b", 1.4, 1.9), W("c", 5.0, 5.5)]
        return L.Lyrics(model="test", source="transcribed", lines=L.group_lines(words))

    def test_word_at_finds_the_active_word(self):
        assert self._lyrics().word_at(1.5).text == "b"

    def test_word_at_returns_nothing_between_phrases(self):
        """Silence must highlight nothing, not the last word sung."""
        assert self._lyrics().word_at(3.0) is None

    def test_line_at_spans_its_words(self):
        assert self._lyrics().line_at(1.2) is not None


class TestLrcRoundTrip:
    def test_write_then_read_preserves_lines_and_timing(self, tmp_path):
        words = [W("aa", 1.0, 1.4), W("bb", 1.4, 1.9), W("cc", 5.0, 5.5)]
        original = L.Lyrics(model="test", source="transcribed", lines=L.group_lines(words))
        path = tmp_path / "x.lrc"
        L.write_lrc(original, path)
        back = L.read_lrc(path)

        assert back.source == "lrc"
        assert len(back.lines) == len(original.lines)
        assert back.lines[0].start == 1.0
        assert back.lines[0].text == "aa bb"

    def test_read_ignores_metadata_and_blank_lines(self, tmp_path):
        path = tmp_path / "x.lrc"
        path.write_text("[re:SingCoach]\n[ve:small.en]\n\n[00:01.00]hello there\n", "utf-8")
        parsed = L.read_lrc(path)
        assert len(parsed.lines) == 1
        assert parsed.lines[0].text == "hello there"

    def test_words_are_spread_across_a_line(self, tmp_path):
        """An .lrc has no word timing, so words share the line evenly and must
        still come out ordered and non-overlapping."""
        path = tmp_path / "x.lrc"
        path.write_text("[00:00.00]one two three four\n[00:04.00]next\n", "utf-8")
        line = L.read_lrc(path).lines[0]
        assert len(line.words) == 4
        assert line.words[0].start == 0.0
        for a, b in zip(line.words, line.words[1:]):
            assert a.end <= b.start


class TestSerialisation:
    def test_round_trips_through_dict(self):
        words = [W("aa", 1.0, 1.4), W("bb", 1.4, 1.9)]
        original = L.Lyrics(model="small.en", source="transcribed", lines=L.group_lines(words))
        back = L.Lyrics.from_dict(original.to_dict())
        assert back.model == "small.en"
        assert [w.text for w in back.words] == ["aa", "bb"]
        assert back.words[0].start == 1.0

    def test_stats_on_empty_lyrics(self):
        assert L.Lyrics(model="x", source="transcribed").stats()["words"] == 0
