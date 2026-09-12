"""Headless construction and paint tests for the UI.

These cannot judge whether the app looks good — that needs eyes. What they do
catch is the whole class of failures that would otherwise only show up when a
person launches it: import errors, bad signal connections, widgets that throw
while painting, and crashes on the empty state before any song is loaded.

The empty-state coverage matters most. Every paint path here runs with no song,
no lyrics, and no pitch readings, because that is exactly the state the window
is in for the first few seconds of every launch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from singcoach.analysis.lyrics import Lyrics, Word, group_lines  # noqa: E402
from singcoach.config import PracticeToggles  # noqa: E402
from singcoach.pitch.compare import Verdict, compare  # noqa: E402
from singcoach.ui.controls import PracticePanel, StatusStrip, StemMixer, TransportBar  # noqa: E402
from singcoach.ui.highway import HighwayWidget, note_name  # noqa: E402
from singcoach.ui.lyrics_view import LyricsWidget  # noqa: E402
from singcoach.ui.tuner import LevelMeter, TunerWidget  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def render(widget, width: int = 900, height: int = 320) -> QImage:
    """Force a real paint pass and return the result.

    Passing the QImage rather than a QPainter: the QPainter overload of
    QWidget.render requires an explicit target offset, and omitting it is a
    TypeError rather than a default.
    """
    widget.resize(width, height)
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(0)
    widget.render(image)
    return image


def is_painted(image: QImage) -> bool:
    """True if anything was actually drawn.

    A widget that silently draws nothing would pass a "did not crash" check
    while being useless, so the paint tests assert real pixels.
    """
    return not image.isNull() and any(
        image.pixelColor(x, y).alpha() > 0
        for x in range(0, image.width(), 17)
        for y in range(0, image.height(), 17)
    )


def sample_notes(count: int = 40) -> list[dict]:
    return [
        {"start": i * 0.6, "end": i * 0.6 + 0.45, "midi": 60 + (i % 8),
         "cents_offset": 0.0, "confidence": 0.9, "vibrato_rate_hz": None,
         "slide_semitones": 0.0, "scoop_cents": 0.0}
        for i in range(count)
    ]


class TestHighway:
    def test_paints_empty(self, qapp):
        assert is_painted(render(HighwayWidget()))

    def test_paints_with_a_song(self, qapp):
        w = HighwayWidget()
        contour = 60 + 4 * np.sin(np.linspace(0, 30, 3000))
        w.set_song(sample_notes(), contour, 0.0116)
        w.set_position(5.0)
        assert is_painted(render(w))

    def test_paints_with_gaps_in_the_contour(self, qapp):
        """Unreadable stretches are NaN; painting must skip, not crash."""
        w = HighwayWidget()
        contour = np.full(3000, 60.0)
        contour[500:900] = np.nan
        w.set_song(sample_notes(), contour, 0.0116)
        w.set_position(7.0)
        assert is_painted(render(w))

    def test_paints_the_singer_trace(self, qapp):
        w = HighwayWidget()
        w.set_song(sample_notes(), None, 0.0116)
        for i in range(400):
            w.add_reading(i * 0.0116, 60 + np.sin(i / 20) * 2, np.sin(i / 15) * 40)
        w.set_position(3.0)
        assert is_painted(render(w))

    def test_paints_expressive_and_harmony(self, qapp):
        w = HighwayWidget()
        w.set_song(sample_notes(), np.full(2000, 62.0), 0.0116, expressive=True)
        w.state.harmony_offsets = (4, 7)
        w.state.loop = (2.0, 6.0)
        w.set_position(4.0)
        assert is_painted(render(w))

    def test_view_widens_for_an_out_of_range_singer(self, qapp):
        w = HighwayWidget()
        w.set_song(sample_notes(), None, 0.0116)
        before = w._target_hi
        w.add_reading(1.0, 90.0, 0.0)
        assert w._target_hi > before

    def test_note_names(self):
        assert note_name(60) == "C4"
        assert note_name(69) == "A4"
        assert note_name(61) == "C#4"


class TestTuner:
    def test_paints_empty(self, qapp):
        assert is_painted(render(TunerWidget(), 400, 180))

    @pytest.mark.parametrize(
        "sung,target",
        [(60.0, 60.0), (59.0, 60.0), (61.0, 60.0), (48.0, 60.0), (59.7, 60.0)],
    )
    def test_paints_every_verdict(self, qapp, sung, target):
        w = TunerWidget()
        w.set_comparison(compare(sung, target))
        assert is_painted(render(w, 400, 180))

    def test_paints_no_target_and_silence(self, qapp):
        w = TunerWidget()
        for c in (compare(60.0, None), compare(None, 60.0)):
            w.set_comparison(c)
            assert is_painted(render(w, 400, 180))

    def test_needle_moves_toward_the_reading(self, qapp):
        w = TunerWidget()
        w.set_comparison(compare(60.0, 60.0))
        start = w._needle
        for _ in range(20):
            w.set_comparison(compare(60.5, 60.0))
        assert w._needle > start

    def test_needle_pins_rather_than_running_off(self, qapp):
        w = TunerWidget()
        for _ in range(40):
            w.set_comparison(compare(65.0, 60.0))
        assert abs(w._needle) <= 100.0

    def test_level_meter(self, qapp):
        m = LevelMeter()
        for dbfs, clip in ((-60.0, False), (-12.0, False), (0.0, True)):
            m.set_level(dbfs, clip)
            assert is_painted(render(m, 120, 8))


class TestLyricsView:
    def _lyrics(self) -> Lyrics:
        words = [Word(text=f"w{i}", start=i * 0.4, end=i * 0.4 + 0.35) for i in range(6)]
        words += [Word(text=f"x{i}", start=5 + i * 0.4, end=5 + i * 0.4 + 0.35) for i in range(4)]
        return Lyrics(model="t", source="transcribed", lines=group_lines(words))

    def test_paints_without_lyrics(self, qapp):
        assert is_painted(render(LyricsWidget(), 600, 110))

    def test_paints_active_line(self, qapp):
        w = LyricsWidget()
        w.set_lyrics(self._lyrics())
        w.set_position(0.8)
        assert is_painted(render(w, 600, 110))

    def test_paints_countdown_before_a_line(self, qapp):
        w = LyricsWidget()
        w.set_lyrics(self._lyrics())
        w.set_position(3.5)          # between phrases
        assert is_painted(render(w, 600, 110))

    def test_nothing_highlighted_between_phrases(self, qapp):
        """A highlight stuck on the last word through an instrumental break is
        worse than no highlight."""
        lyrics = self._lyrics()
        assert lyrics.word_at(3.5) is None


class TestControls:
    def test_stem_mixer_presets_move_the_slider(self, qapp):
        m = StemMixer()
        received = []
        m.vocal_changed.connect(received.append)
        m.set_vocal(1.0)
        assert received and received[-1] == pytest.approx(1.0)
        m.set_vocal(0.0)
        assert received[-1] == pytest.approx(0.0)

    def test_practice_toggles_round_trip(self, qapp):
        p = PracticePanel()
        p.apply(PracticeToggles(transpose=True, scoring=True))
        toggles = p.toggles()
        assert toggles.transpose and toggles.scoring
        assert not toggles.harmony_guides

    def test_dependent_controls_disabled_until_enabled(self, qapp):
        p = PracticePanel()
        assert not p.speed_box.isEnabled()
        p.checks["slowdown_loop"].setChecked(True)
        assert p.speed_box.isEnabled()

    def test_transport_formats_time(self, qapp):
        t = TransportBar()
        t.set_duration(293.0)
        assert t.duration_label.text() == "4:53"
        t.set_position(64.0)
        assert t.time_label.text() == "1:04"

    def test_transport_records_state(self, qapp):
        t = TransportBar()
        t.set_recording(True)
        assert "Stop" in t.record_button.text()

    def test_status_strip_flags_unmeasured_latency(self, qapp):
        s = StatusStrip()
        s.set_latency(None, measured=False)
        assert "not measured" in s.latency_label.text()
        s.set_latency(96.0, measured=True)
        assert "96" in s.latency_label.text()


class TestVerdictMessages:
    """The words are the product; assert them directly."""

    def test_actionable_wording(self):
        assert compare(59.0, 60.0).message == "GO HIGHER"
        assert compare(61.0, 60.0).message == "GO LOWER"
        assert compare(60.0, 60.0).message == "ON PITCH"

    def test_octave_message_is_readable(self):
        assert compare(48.0, 60.0).message == "an octave low"

    def test_no_target_does_not_read_as_failure(self):
        c = compare(60.0, None)
        assert c.verdict is Verdict.NO_TARGET
        assert c.message == "—"
