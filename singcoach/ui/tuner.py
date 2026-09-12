"""The big readout: are you high, low, or right?

This is the only element allowed to shout. While singing you get a fraction of
a second of attention to spare, so the verdict has to be readable at a glance
from across a room.

Three redundant channels carry the same message, so none of them is load-
bearing alone:

* **Words** — "GO HIGHER", not a number to interpret.
* **Position** — the needle's distance from centre.
* **Colour** — reinforcement only, never the sole signal, because red/green
  deficiency is common and the app must work without it.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QWidget

from ..pitch.compare import Comparison, Verdict
from . import theme
from .highway import note_name

#: Cents at the edge of the meter. Beyond this the needle pins.
METER_RANGE_CENTS = 100.0


class TunerWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.comparison: Comparison | None = None
        self.tolerance = 20.0
        self._needle = 0.0          # smoothed, for a calm meter

    def set_comparison(self, comparison: Comparison | None, tolerance: float = 20.0) -> None:
        self.comparison = comparison
        self.tolerance = tolerance
        target = 0.0
        if comparison is not None and comparison.cents is not None:
            target = max(-METER_RANGE_CENTS, min(METER_RANGE_CENTS, comparison.cents))
        # Ease toward the reading. The underlying data is already median
        # filtered; this is purely so the needle does not twitch.
        self._needle += (target - self._needle) * 0.35
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), theme.PANEL)

        w, h = self.width(), self.height()
        centre_x = w / 2
        scale_y = h * 0.62

        c = self.comparison
        verdict = c.verdict if c else Verdict.NO_TARGET

        self._draw_scale(p, centre_x, scale_y, w)
        if verdict not in (Verdict.NO_TARGET, Verdict.SILENT):
            self._draw_needle(p, centre_x, scale_y, w)
        self._draw_message(p, w, h, c, verdict)
        self._draw_notes(p, w, h, c)
        p.end()

    def _draw_scale(self, p: QPainter, centre_x: float, y: float, w: float) -> None:
        half = w * 0.42
        # In-tune zone, drawn as a band so "close enough" is visible as an area.
        band = half * (self.tolerance / METER_RANGE_CENTS)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(52, 211, 153, 34)))
        p.drawRoundedRect(QRectF(centre_x - band, y - 16, band * 2, 32), 4, 4)

        p.setPen(QPen(theme.BORDER, 2))
        p.drawLine(int(centre_x - half), int(y), int(centre_x + half), int(y))

        p.setFont(theme.mono_font(8))
        for cents in (-100, -50, -20, 0, 20, 50, 100):
            x = centre_x + half * (cents / METER_RANGE_CENTS)
            tall = cents == 0
            p.setPen(QPen(theme.TEXT_DIM if tall else theme.TEXT_FAINT, 2 if tall else 1))
            p.drawLine(int(x), int(y - (12 if tall else 6)), int(x), int(y + (12 if tall else 6)))

        p.setPen(QPen(theme.TEXT_FAINT, 1))
        p.setFont(theme.ui_font(8))
        p.drawText(int(centre_x - half), int(y + 28), "FLAT")
        p.drawText(int(centre_x + half - 26), int(y + 28), "SHARP")

    def _draw_needle(self, p: QPainter, centre_x: float, y: float, w: float) -> None:
        half = w * 0.42
        x = centre_x + half * (self._needle / METER_RANGE_CENTS)
        colour = theme.accuracy_color(self._needle, self.tolerance)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(colour))
        p.drawPolygon(
            QPolygonF([QPointF(x, y - 20), QPointF(x - 8, y - 34), QPointF(x + 8, y - 34)])
        )
        p.setPen(QPen(colour, 3))
        p.drawLine(int(x), int(y - 18), int(x), int(y + 18))

    def _draw_message(self, p: QPainter, w: float, h: float, c, verdict) -> None:
        if verdict is Verdict.NO_TARGET:
            message, colour = "no target here", theme.TEXT_FAINT
        elif verdict is Verdict.SILENT:
            message, colour = "waiting for you", theme.TEXT_FAINT
        else:
            message = c.message
            colour = theme.accuracy_color(c.cents, self.tolerance)

        p.setPen(QPen(colour, 1))
        p.setFont(theme.ui_font(19, bold=True))
        p.drawText(QRectF(0, h * 0.02, w, h * 0.22),
                   Qt.AlignmentFlag.AlignCenter, message)

        if c is not None and c.cents is not None and verdict.is_scored:
            p.setPen(QPen(theme.TEXT_DIM, 1))
            p.setFont(theme.mono_font(10))
            p.drawText(QRectF(0, h * 0.24, w, h * 0.12),
                       Qt.AlignmentFlag.AlignCenter, f"{c.cents:+.0f} cents")

    def _draw_notes(self, p: QPainter, w: float, h: float, c) -> None:
        if c is None:
            return
        p.setFont(theme.mono_font(10))
        if c.target_midi is not None:
            p.setPen(QPen(theme.TARGET_NOTE_ACTIVE, 1))
            p.drawText(QRectF(10, h - 26, w / 2 - 10, 20),
                       Qt.AlignmentFlag.AlignLeft, f"target  {note_name(c.target_midi)}")
        if c.sung_midi is not None:
            p.setPen(QPen(theme.TEXT, 1))
            p.drawText(QRectF(w / 2, h - 26, w / 2 - 10, 20),
                       Qt.AlignmentFlag.AlignRight, f"you  {note_name(c.sung_midi)}")


class LevelMeter(QWidget):
    """Microphone input level, with a clipping warning.

    Worth its space: a mic that is too quiet makes pitch detection unreliable
    and a mic that is clipping makes it wrong, and in both cases the singer
    would otherwise blame themselves for the app's confusion.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(8)
        self.setMinimumWidth(90)
        self._level = 0.0
        self._clipping = False

    def set_level(self, dbfs: float, clipping: bool = False) -> None:
        # Map -60..0 dBFS onto 0..1.
        self._level = max(0.0, min(1.0, (dbfs + 60.0) / 60.0))
        self._clipping = clipping
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(theme.PANEL_LIGHT))
        p.drawRoundedRect(self.rect(), 4, 4)

        if self._level > 0:
            if self._clipping:
                colour = theme.OFF
            elif self._level > 0.85:
                colour = theme.NEAR
            else:
                colour = theme.ON_PITCH
            p.setBrush(QBrush(colour))
            p.drawRoundedRect(
                QRectF(0, 0, self.width() * self._level, self.height()), 4, 4
            )
        p.end()
