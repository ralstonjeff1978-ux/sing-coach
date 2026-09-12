"""Tap-to-sync: mark each line's start by ear.

Every automatic approach tried here inferred *when* a line is sung from a
speech-trained transcript, and all of them landed somewhere between 50% and 88%
correct depending on which metric you weighed. The failure mode was always the
same: some sections right, some a whole line out.

This removes inference from the timing path. The song plays; you press SPACE
when a line starts. What you mark is what you actually hear, so the result is
correct by construction and it stays correct.

Two things make it quick rather than tedious:

**It starts from the automatic result**, so lines already in the right place
can be skipped — press SPACE only where it is wrong, and everything else keeps
its existing time.

**Taps are snapped to the singer's onsets.** Human reaction time puts a tap
150-250 ms late, consistently. Rather than ask anyone to compensate by feel,
each tap moves to the nearest moment the vocal actually starts, within a window
tight enough that it can only ever be a correction. The bias disappears.
"""

from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from ..analysis import lyrics as lyrics_mod
from . import theme

#: How far a tap may be moved to reach a vocal onset. Wide enough to absorb
#: reaction time, tight enough that it can never reach a different phrase.
SNAP_WINDOW_S = 0.40


class SyncDialog(QDialog):
    def __init__(self, song, engine, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sync lyrics — tap along")
        self.setMinimumSize(700, 640)
        self.song = song
        self.engine = engine
        self.saved = False

        self.lyrics = self._load_lyrics()
        self.spans = self._load_spans()
        #: New start time per line, or None to keep the existing one.
        self.tapped: list[float | None] = [None] * len(self.lyrics.lines)
        self.armed = 0

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(
            self._dim(
                "Press SPACE the instant each line starts. The line waiting for "
                "your tap is highlighted below.\n\n"
                "You do not have to do all of them — lines you skip keep the "
                "time they already have. Use ↑ ↓ to move the marker, and "
                "Enter to play or pause."
            )
        )

        self.now_label = QLabel("Press Play, then tap along")
        self.now_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.now_label.setFont(theme.ui_font(20, bold=True))
        self.now_label.setWordWrap(True)
        self.now_label.setMinimumHeight(64)
        layout.addWidget(self.now_label)

        self.list = QListWidget()
        self.list.setFont(theme.ui_font(11))
        self.list.currentRowChanged.connect(self._row_changed)
        layout.addWidget(self.list, 1)
        self._rebuild_list()

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("primary")
        self.play_button.setFixedWidth(96)
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)

        self.position_label = QLabel("0:00")
        self.position_label.setFont(theme.mono_font(11))
        controls.addWidget(self.position_label)

        controls.addStretch(1)

        tap = QPushButton("Tap  (SPACE)")
        tap.setFixedWidth(140)
        tap.clicked.connect(self._tap)
        controls.addWidget(tap)

        back = QPushButton("Undo tap")
        back.clicked.connect(self._undo)
        controls.addWidget(back)

        jump = QPushButton("Play from here")
        jump.setToolTip("Seek to the highlighted line, a couple of seconds early")
        jump.clicked.connect(self._play_from_here)
        controls.addWidget(jump)
        layout.addLayout(controls)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

    @staticmethod
    def _dim(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        return label

    # -- loading ----------------------------------------------------------

    def _load_lyrics(self) -> lyrics_mod.Lyrics:
        paths = self.song.paths
        try:
            if paths.lyrics_override.exists():
                return lyrics_mod.read_lrc(paths.lyrics_override)
            if paths.lyrics.exists():
                return lyrics_mod.load(paths.lyrics)
        except (OSError, json.JSONDecodeError):
            pass
        return lyrics_mod.Lyrics(model="none", source="user", lines=[])

    def _load_spans(self) -> list[tuple[float, float]]:
        """Voiced regions, used to cancel reaction-time bias out of each tap."""
        try:
            analysis = json.loads(self.song.paths.analysis.read_text("utf-8"))
            melody = analysis["melody"]
            contour = np.array(
                [np.nan if v is None else v for v in melody["contour"]["midi"]],
                dtype=float,
            )
            return lyrics_mod.voiced_spans(contour, melody["hop_s"])
        except (OSError, KeyError, json.JSONDecodeError):
            return []

    # -- list -------------------------------------------------------------

    def _rebuild_list(self) -> None:
        current = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for index, line in enumerate(self.lyrics.lines):
            when = self.tapped[index]
            marker = "●" if when is not None else "○"
            shown = when if when is not None else line.start
            item = QListWidgetItem(f"{marker}  {self._mmss(shown)}   {line.text}")
            if when is not None:
                item.setForeground(theme.ON_PITCH)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.list.setCurrentRow(min(max(current, 0), len(self.lyrics.lines) - 1)
                                if self.lyrics.lines else -1)

    @staticmethod
    def _mmss(seconds: float) -> str:
        m, s = divmod(max(0.0, seconds), 60)
        return f"{int(m)}:{s:05.2f}"

    def _row_changed(self, row: int) -> None:
        if row >= 0:
            self.armed = row

    # -- actions ----------------------------------------------------------

    def _toggle_play(self) -> None:
        self.engine.mixer.toggle()
        self.play_button.setText("Pause" if self.engine.mixer.playing else "Play")

    def _play_from_here(self) -> None:
        if not self.lyrics.lines:
            return
        line = self.lyrics.lines[self.armed]
        when = self.tapped[self.armed]
        target = (when if when is not None else line.start) - 2.0
        self.engine.mixer.seek(max(0.0, target))
        if not self.engine.mixer.playing:
            self._toggle_play()

    def _tap(self) -> None:
        if not self.lyrics.lines or self.armed >= len(self.lyrics.lines):
            return
        # The moment the user *heard*, not the mixer's playhead — the audio in
        # the output buffer has not reached them yet.
        raw = self.engine.heard_position
        snapped = self._snap(raw)
        self.tapped[self.armed] = snapped

        moved = (snapped - raw) * 1000.0
        self.status.setText(
            f"Line {self.armed + 1} set to {self._mmss(snapped)}"
            + (f"  (moved {moved:+.0f} ms onto the singer's onset)" if abs(moved) > 1 else "")
        )
        self.status.setStyleSheet(f"color: {theme.ON_PITCH.name()};")

        self.armed = min(self.armed + 1, len(self.lyrics.lines) - 1)
        self._rebuild_list()
        self.list.setCurrentRow(self.armed)
        self.list.scrollToItem(self.list.currentItem())

    def _snap(self, t: float) -> float:
        """Move a tap onto the nearest vocal onset, if one is close enough."""
        if not self.spans:
            return t
        candidates = [start for start, _ in self.spans if abs(start - t) <= SNAP_WINDOW_S]
        return min(candidates, key=lambda s: abs(s - t)) if candidates else t

    def _undo(self) -> None:
        index = max(0, self.armed - 1 if self.tapped[self.armed] is None else self.armed)
        self.tapped[index] = None
        self.armed = index
        self._rebuild_list()
        self.list.setCurrentRow(index)
        self.status.setText(f"Line {index + 1} reset to its previous time.")
        self.status.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")

    # -- keyboard ---------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space:
            self._tap()
            event.accept()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._toggle_play()
            event.accept()
        elif key == Qt.Key.Key_Backspace:
            self._undo()
            event.accept()
        else:
            super().keyPressEvent(event)

    # -- per-frame --------------------------------------------------------

    def _tick(self) -> None:
        position = self.engine.heard_position
        self.position_label.setText(self._mmss(position))
        self.play_button.setText("Pause" if self.engine.mixer.playing else "Play")

        if self.lyrics.lines and self.armed < len(self.lyrics.lines):
            line = self.lyrics.lines[self.armed]
            self.now_label.setText(line.text or "(blank line)")
            self.now_label.setStyleSheet(f"color: {theme.ACCENT.name()};")

    # -- saving -----------------------------------------------------------

    def _save(self) -> None:
        if not self.lyrics.lines:
            self.reject()
            return

        starts = [
            self.tapped[i] if self.tapped[i] is not None else line.start
            for i, line in enumerate(self.lyrics.lines)
        ]
        # Keep it strictly increasing. A mistimed tap should not be able to
        # make the highlight jump backwards.
        for i in range(1, len(starts)):
            if starts[i] <= starts[i - 1]:
                starts[i] = starts[i - 1] + 0.3

        rebuilt: list[lyrics_mod.Line] = []
        for index, line in enumerate(self.lyrics.lines):
            tokens = [w.text for w in line.words]
            if not tokens:
                continue
            start = starts[index]
            # A line runs until the next one begins, bounded by the voiced
            # regions inside that window so words never light up in silence.
            limit = starts[index + 1] if index + 1 < len(starts) else start + 6.0
            window = lyrics_mod._spans_within(self.spans, start, limit) if self.spans \
                else [(start, limit)]
            words = lyrics_mod._spread_over_spans(tokens, window)
            if words:
                rebuilt.append(
                    lyrics_mod.Line(start=words[0].start, end=words[-1].end, words=words)
                )

        result = lyrics_mod.Lyrics(model="tapped", source="user", lines=rebuilt)
        lyrics_mod.write_lrc(result, self.song.paths.lyrics_override)
        lyrics_mod.save(result, self.song.paths.lyrics)
        self.saved = True
        self.accept()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        super().closeEvent(event)
