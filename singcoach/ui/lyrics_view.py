"""Karaoke lyric display: the current line, with the current word lit.

Rendering rules, all of them driven by the fact that you are reading this while
singing rather than while sitting still:

* **The current line is large and centred.** The next line is shown smaller
  beneath it, because knowing what is coming is most of what makes karaoke
  singable.
* **The active word is highlighted, and words already sung stay dimmed but
  visible.** Erasing them removes your place in the line.
* **A countdown appears before a line starts**, so an entry after a long
  instrumental passage does not arrive unannounced.
* **Nothing is highlighted when nothing is being sung.** A stuck highlight
  during an instrumental break is worse than no highlight at all.

Text comes from the user's own local transcript for their own recording; this
widget only draws whatever that file contains.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..analysis.lyrics import Line, Lyrics
from . import theme

#: Show a count-in when the next line is this close.
COUNTDOWN_LEAD_S = 3.0


class LyricsWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(96)
        self.lyrics: Lyrics | None = None
        self._position = 0.0

    def set_lyrics(self, lyrics: Lyrics | None) -> None:
        self.lyrics = lyrics
        self.update()

    def set_position(self, t: float) -> None:
        self._position = t
        self.update()

    # -- helpers ----------------------------------------------------------

    def _current_and_next(self) -> tuple[Line | None, Line | None]:
        if not self.lyrics or not self.lyrics.lines:
            return None, None
        current = self.lyrics.line_at(self._position)
        upcoming = None
        for line in self.lyrics.lines:
            if line.start > self._position:
                upcoming = line
                break
        return current, upcoming

    # -- painting ---------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), theme.PANEL)

        if not self.lyrics or not self.lyrics.lines:
            p.setPen(QPen(theme.TEXT_FAINT, 1))
            p.setFont(theme.ui_font(10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no lyrics for this song")
            p.end()
            return

        current, upcoming = self._current_and_next()

        if current is not None:
            self._draw_active_line(p, current)
        elif upcoming is not None:
            self._draw_countdown(p, upcoming)

        if upcoming is not None and upcoming is not current:
            self._draw_next_line(p, upcoming)
        p.end()

    def _draw_active_line(self, p: QPainter, line: Line) -> None:
        font = theme.ui_font(20, bold=True)
        p.setFont(font)
        metrics = QFontMetrics(font)
        gap = metrics.horizontalAdvance(" ")

        widths = [metrics.horizontalAdvance(w.text) for w in line.words]
        total = sum(widths) + gap * max(0, len(widths) - 1)
        x = (self.width() - total) / 2
        y = self.height() * 0.42

        for word, width in zip(line.words, widths):
            if word.start <= self._position < word.end:
                colour = theme.ACCENT                    # singing this now
            elif word.end <= self._position:
                colour = theme.TEXT_DIM                  # already sung
            else:
                colour = theme.TEXT                      # still to come
            p.setPen(QPen(colour, 1))
            p.drawText(QRectF(x, y - metrics.height(), width + 2, metrics.height() * 1.4),
                       Qt.AlignmentFlag.AlignCenter, word.text)

            # Underline the active word: a second, colour-independent signal.
            if word.start <= self._position < word.end:
                p.setPen(QPen(theme.ACCENT, 2))
                p.drawLine(int(x), int(y + 5), int(x + width), int(y + 5))
            x += width + gap

    def _draw_countdown(self, p: QPainter, upcoming: Line) -> None:
        remaining = upcoming.start - self._position
        if remaining > COUNTDOWN_LEAD_S or remaining < 0:
            return
        p.setPen(QPen(theme.TEXT_DIM, 1))
        p.setFont(theme.ui_font(13, bold=True))
        beats = int(remaining) + 1
        p.drawText(QRectF(0, self.height() * 0.12, self.width(), self.height() * 0.34),
                   Qt.AlignmentFlag.AlignCenter, "•  " * beats)

    def _draw_next_line(self, p: QPainter, line: Line) -> None:
        font = theme.ui_font(12)
        p.setFont(font)
        colour = QColor(theme.TEXT_FAINT)
        p.setPen(QPen(colour, 1))
        p.drawText(QRectF(0, self.height() * 0.68, self.width(), self.height() * 0.3),
                   Qt.AlignmentFlag.AlignCenter, line.text)
