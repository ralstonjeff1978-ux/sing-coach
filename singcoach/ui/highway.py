"""The note highway: what you actually watch while singing.

Time runs left to right, pitch runs bottom to top, and the playhead sits about
a third in from the left so there is always more future than past on screen —
you need to see the note *coming* in time to prepare for it, not learn about it
as it arrives.

Three layers, in order of importance:

**Target notes** — blocks at the pitch to sing. In expressive mode these are
drawn faintly, because on that material they are a simplification rather than
the truth.

**The ghost contour** — the original singer's actual pitch, frame by frame.
This is the layer that teaches phrasing: the scoop up into a note, the slide
between two, the shape of the vibrato. On blues-inflected material it *is* the
target, and the note blocks are only scaffolding.

**Your pitch** — a trailing trace coloured by accuracy, with the live point
emphasised.

Regions where the melody could not be read honestly show as a flat neutral
band. Drawing a guessed note there would teach the wrong thing.

Painted directly with QPainter rather than through a plotting library: at 60 fps
with a moving viewport, a general-purpose chart widget spends most of its time
on machinery we do not need.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from . import theme

#: Seconds of song visible across the full width.
DEFAULT_SPAN_S = 6.0
#: Fraction of the width behind the playhead.
PLAYHEAD_FRACTION = 0.33

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: float) -> str:
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


@dataclass
class HighwayState:
    """Everything the widget needs to draw one frame."""

    position: float = 0.0
    notes: list[dict] | None = None
    contour: np.ndarray | None = None
    contour_hop: float = 0.0116
    expressive: bool = False
    harmony_offsets: tuple[int, ...] = ()
    loop: tuple[float, float] | None = None
    #: Semitones the targets are drawn shifted by, when the singer has settled
    #: into a different octave. Keeps their trace and the target on the same
    #: line instead of a screen apart.
    octave_shift: int = 0
    #: Regions where the original vocal is actually sounding. Shaded so the
    #: highway shows "sing here" directly, and shared with the lyric timing so
    #: the two views cannot contradict each other.
    vocal_spans: list[tuple[float, float]] = field(default_factory=list)


class HighwayWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setAutoFillBackground(False)

        self.state = HighwayState()
        self.span_s = DEFAULT_SPAN_S
        self.tolerance_cents = 20.0

        #: (song_time, midi, cents_error|None) — the singer's trace.
        #: Sized to a little more than one screenful. Holding more costs paint
        #: time on points that are scrolled off and cannot be seen.
        self.trace: deque[tuple[float, float, float | None]] = deque(maxlen=640)

        # Pitch window, in MIDI. Adapts to the song and to where you sing, but
        # slowly — a view that jumps around is unreadable.
        self._lo, self._hi = 52.0, 76.0
        self._target_lo, self._target_hi = self._lo, self._hi

        # Pens built once. Constructing them inside the paint loop was
        # measurably expensive at 60 fps.
        self._grid_pen = QPen(theme.GRIDLINE, 1)
        self._octave_pen = QPen(theme.GRIDLINE_OCTAVE, 1)
        self._label_pen = QPen(theme.TEXT_FAINT, 1)
        self._playhead_pen = QPen(theme.PLAYHEAD, 1.5)
        self._trace_pen = QPen(theme.GHOST_CONTOUR, 2.6, Qt.PenStyle.SolidLine,
                               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        # Built once. Qt resolves a font by family name on construction, and
        # doing that every frame cost more than every line and label combined.
        self._grid_font = theme.mono_font(8)

    # -- data -------------------------------------------------------------

    def set_song(
        self,
        notes: list[dict] | None,
        contour: np.ndarray | None,
        hop: float,
        *,
        expressive: bool = False,
    ) -> None:
        self.state.notes = notes or []
        self.state.contour = contour
        self.state.contour_hop = hop
        self.state.expressive = expressive
        self.trace.clear()

        if notes:
            pitches = [n["midi"] for n in notes]
            self._target_lo = min(pitches) - 4
            self._target_hi = max(pitches) + 4
            self._lo, self._hi = self._target_lo, self._target_hi
        self.update()

    def set_position(self, t: float) -> None:
        self.state.position = t
        self.update()

    def add_reading(self, song_time: float, midi: float | None, cents: float | None) -> None:
        if midi is not None:
            self.trace.append((song_time, midi, cents))
            self._accommodate(midi)

    def _accommodate(self, midi: float) -> None:
        """Widen the view if the singer is outside it — gently."""
        changed = False
        if midi < self._target_lo + 2:
            self._target_lo = midi - 3
            changed = True
        if midi > self._target_hi - 2:
            self._target_hi = midi + 3
            changed = True
        if changed:
            self.update()

    def clear_trace(self) -> None:
        self.trace.clear()
        self.update()

    # -- geometry ---------------------------------------------------------

    def _x(self, t: float) -> float:
        origin = self.state.position - self.span_s * PLAYHEAD_FRACTION
        return (t - origin) / self.span_s * self.width()

    def _y(self, midi: float) -> float:
        span = max(self._hi - self._lo, 1.0)
        return self.height() * (1.0 - (midi - self._lo) / span)

    # -- painting ---------------------------------------------------------

    def paintEvent(self, _event) -> None:
        # Ease the viewport toward its target so range changes glide.
        self._lo += (self._target_lo - self._lo) * 0.12
        self._hi += (self._target_hi - self._hi) * 0.12

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), theme.BG)

        self._draw_grid(p)
        self._draw_loop(p)
        self._draw_vocal_bands(p)
        self._draw_notes(p)
        if self.state.harmony_offsets:
            self._draw_harmony(p)
        self._draw_contour(p)
        self._draw_trace(p)
        self._draw_playhead(p)
        p.end()

    def _draw_grid(self, p: QPainter) -> None:
        """Horizontal semitone lines.

        Antialiasing is switched off here and the pens are built once. This
        looked like the cheapest thing on the widget and profiled as the most
        expensive — 6.1 ms of a 10 ms frame, for about thirty straight lines.
        Antialiasing does nothing for an axis-aligned line but still shades
        every pixel along it, and a QPen was being constructed per row.
        """
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setFont(self._grid_font)
        width = self.width()

        octaves: list[int] = []
        p.setPen(self._grid_pen)
        for midi in range(int(self._lo), int(self._hi) + 1):
            if midi % 12 == 0:
                octaves.append(midi)
                continue
            y = int(self._y(midi))
            p.drawLine(0, y, width, y)

        p.setPen(self._octave_pen)
        for midi in octaves:
            y = int(self._y(midi))
            p.drawLine(0, y, width, y)

        p.setPen(self._label_pen)
        for midi in octaves:
            p.drawText(4, int(self._y(midi)) - 3, note_name(midi))

        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def _draw_loop(self, p: QPainter) -> None:
        if not self.state.loop:
            return
        a, b = self.state.loop
        rect = QRectF(self._x(a), 0, self._x(b) - self._x(a), self.height())
        p.fillRect(rect, QColor(56, 189, 248, 22))
        p.setPen(QPen(theme.ACCENT, 1, Qt.PenStyle.DashLine))
        p.drawLine(int(rect.left()), 0, int(rect.left()), self.height())
        p.drawLine(int(rect.right()), 0, int(rect.right()), self.height())

    def _draw_vocal_bands(self, p: QPainter) -> None:
        """Shade the stretches where the singer is actually singing.

        This used to shade the opposite — the gaps — on the reasoning that
        marking "nothing to sing here" reassures the singer that the silence
        belongs to the song. In use it read exactly backwards: the lighter
        bands were taken to mean *this is where the voice goes*, which is the
        more useful thing to know and the thing people look for.

        So presence is marked rather than absence. The bands come from the same
        voiced regions that time the lyrics, which also removes a genuine
        inconsistency: the shading was previously derived from segmented notes
        while the tuner read the raw contour, so the readout could say "no
        target here" in the middle of a lighter band.
        """
        spans = self.state.vocal_spans
        if not spans:
            return
        left = self.state.position - self.span_s * PLAYHEAD_FRACTION
        right = left + self.span_s

        for a, b in spans:
            if b < left:
                continue
            if a > right:
                break
            x0 = self._x(max(a, left))
            x1 = self._x(min(b, right))
            if x1 - x0 < 1.0:
                continue
            p.fillRect(QRectF(x0, 0, x1 - x0, self.height()), theme.VOCAL_BAND)

    def _draw_notes(self, p: QPainter) -> None:
        notes = self.state.notes
        if not notes:
            return
        left = self.state.position - self.span_s * PLAYHEAD_FRACTION
        right = left + self.span_s
        now = self.state.position

        # In expressive mode the grid is scaffolding, not the target.
        alpha = 70 if self.state.expressive else 255
        height = max(6.0, self.height() / max(self._hi - self._lo, 1.0) * 0.8)

        shift = self.state.octave_shift
        p.setPen(Qt.PenStyle.NoPen)
        for n in notes:
            if n["end"] < left or n["start"] > right:
                continue
            x0, x1 = self._x(n["start"]), self._x(n["end"])
            y = self._y(n["midi"] + shift)
            active = n["start"] <= now < n["end"]

            colour = QColor(theme.TARGET_NOTE_ACTIVE if active else theme.TARGET_NOTE)
            colour.setAlpha(255 if active else alpha)
            p.setBrush(QBrush(colour))
            p.drawRoundedRect(QRectF(x0, y - height / 2, max(x1 - x0, 2.0), height), 3, 3)

    def _draw_harmony(self, p: QPainter) -> None:
        notes = self.state.notes
        if not notes:
            return
        left = self.state.position - self.span_s * PLAYHEAD_FRACTION
        right = left + self.span_s
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(theme.HARMONY_LINE))
        for n in notes:
            if n["end"] < left or n["start"] > right:
                continue
            x0, x1 = self._x(n["start"]), self._x(n["end"])
            for offset in self.state.harmony_offsets:
                y = self._y(n["midi"] + offset + self.state.octave_shift)
                p.drawRoundedRect(QRectF(x0, y - 2, max(x1 - x0, 2.0), 4), 2, 2)

    def _draw_contour(self, p: QPainter) -> None:
        """The original singer's actual pitch — the phrasing to learn."""
        contour = self.state.contour
        if contour is None or len(contour) == 0:
            return
        hop = self.state.contour_hop
        left = self.state.position - self.span_s * PLAYHEAD_FRACTION
        right = left + self.span_s
        i0 = max(0, int(left / hop))
        i1 = min(len(contour), int(right / hop) + 2)
        if i1 <= i0:
            return

        width = 3.0 if self.state.expressive else 1.6
        p.setPen(QPen(theme.GHOST_CONTOUR, width, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

        shift = self.state.octave_shift
        path = QPainterPath()
        pen_down = False
        for i in range(i0, i1):
            v = contour[i]
            if np.isnan(v):
                pen_down = False
                continue
            pt = QPointF(self._x(i * hop), self._y(float(v) + shift))
            if pen_down:
                path.lineTo(pt)
            else:
                path.moveTo(pt)
                pen_down = True
        p.drawPath(path)

    def _draw_trace(self, p: QPainter) -> None:
        """Your pitch, coloured by how close it was.

        Segments are bucketed by colour and drawn as one path per bucket. The
        obvious implementation — setPen and drawLine per segment — costs two
        Python-to-C++ calls for every point, and at 60 fps over a few hundred
        points that measured 11.6 ms per frame. Which happens to be exactly one
        audio block period: the paint held the GIL long enough to starve the
        audio callback, and the result was choppy playback. Four pen changes
        instead of hundreds fixes it.
        """
        if not self.trace:
            return
        left = self.state.position - self.span_s * PLAYHEAD_FRACTION

        buckets: dict[int, QPainterPath] = {}
        colours: dict[int, QColor] = {}
        prev: tuple[float, float] | None = None

        for song_time, midi, cents in self.trace:
            if song_time < left:
                prev = None
                continue
            pt = (self._x(song_time), self._y(midi))
            if prev is not None and abs(pt[0] - prev[0]) < 40:
                colour = theme.accuracy_color(cents, self.tolerance_cents)
                key = colour.rgba()
                path = buckets.get(key)
                if path is None:
                    path = buckets[key] = QPainterPath()
                    colours[key] = colour
                path.moveTo(QPointF(*prev))
                path.lineTo(QPointF(*pt))
            prev = pt

        for key, path in buckets.items():
            p.setPen(QPen(colours[key], 2.6, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(path)

        # Emphasise the live point so the eye can find itself instantly.
        song_time, midi, cents = self.trace[-1]
        if song_time >= left:
            colour = theme.accuracy_color(cents, self.tolerance_cents)
            centre = QPointF(self._x(song_time), self._y(midi))
            glow = QColor(colour)
            glow.setAlpha(60)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(centre, 11, 11)
            p.setBrush(QBrush(colour))
            p.drawEllipse(centre, 5, 5)

    def _draw_playhead(self, p: QPainter) -> None:
        x = self.width() * PLAYHEAD_FRACTION
        gradient = QLinearGradient(x - 12, 0, x, 0)
        gradient.setColorAt(0.0, QColor(248, 250, 252, 0))
        gradient.setColorAt(1.0, QColor(248, 250, 252, 40))
        p.fillRect(QRectF(x - 12, 0, 12, self.height()), QBrush(gradient))
        p.setPen(self._playhead_pen)
        p.drawLine(int(x), 0, int(x), self.height())
