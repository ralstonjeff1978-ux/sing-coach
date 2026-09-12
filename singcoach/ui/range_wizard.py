"""Pitch-match warm-up: find the range you can actually sing in.

The earlier version asked you to slide down and up and measured whatever came
out. That measures what your voice *did*, not what it can reliably *do* — a
scrape at the bottom of a slide counts the same as a note you could hold for a
phrase, and the resulting range was optimistic at both ends.

This one asks a fairer question. The app plays a note; you sing it back. A note
only counts if you actually matched it and held it. Walking outward from a
comfortable anchor until you miss twice in a row finds the edge of your usable
range rather than the edge of what your vocal cords can be made to emit.

It doubles as a warm-up, which is why it starts in the middle and works
outward: that is how you would warm up anyway.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..pitch.detector import midi_to_hz
from . import theme
from .highway import note_name

#: Where we start. Comfortable for most untrained voices of any type, and the
#: anchor is re-derived from what the singer actually produces anyway.
ANCHOR_MIDI = 57.0                    # A3

LISTEN_SECONDS = 1.6                  # tone sounds alone
SING_SECONDS = 2.6                    # your turn
#: Within this of the target counts as a match. Generous on purpose — this is
#: measuring reach, not precision.
MATCH_TOLERANCE_CENTS = 90.0
#: Fraction of your sung frames that must be on target.
MATCH_FRACTION = 0.45
#: Consecutive misses that end a direction.
MISSES_TO_STOP = 2
#: Absolute safety rails, so a runaway loop cannot ask for impossible notes.
FLOOR_MIDI, CEILING_MIDI = 33.0, 84.0


class Phase(Enum):
    IDLE = auto()
    LISTEN = auto()
    SING = auto()
    DONE = auto()


class RangeWizard(QDialog):
    def __init__(self, engine, settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pitch match — find your range")
        self.setMinimumWidth(560)
        self.engine = engine
        self.settings = settings

        self.phase = Phase.IDLE
        self._elapsed = 0.0
        self._target = ANCHOR_MIDI
        self._samples: list[float] = []
        self._misses = 0
        self._going_down = True
        self._matched: list[float] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        self.instruction = QLabel(
            "This finds the notes you can actually hold, so songs can be put "
            "in a key that suits your voice.\n\n"
            "You will hear a note. When the prompt says SING, sing it back on "
            "'ah' and hold it. The app works outward from a comfortable middle "
            "until you run out of room in each direction.\n\n"
            "Do not strain. If a note is uncomfortable, just do not sing it — "
            "missing is how the test finds your edge."
        )
        self.instruction.setWordWrap(True)
        layout.addWidget(self.instruction)

        self.prompt = QLabel("Ready")
        self.prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt.setFont(theme.ui_font(22, bold=True))
        layout.addWidget(self.prompt)

        row = QHBoxLayout()
        self.target_label = QLabel("—")
        self.target_label.setFont(theme.mono_font(15, bold=True))
        self.target_label.setStyleSheet(f"color: {theme.TARGET_NOTE_ACTIVE.name()};")
        self.you_label = QLabel("—")
        self.you_label.setFont(theme.mono_font(15, bold=True))
        self.you_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(QLabel("target"))
        row.addWidget(self.target_label, 1)
        row.addWidget(self.you_label, 1)
        row.addWidget(QLabel("you"))
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.result_label = QLabel(self._existing_text())
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        layout.addWidget(self.result_label)

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start)
        layout.addWidget(self.start_button)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
        layout.addWidget(self.buttons)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

        self._low: float | None = None
        self._high: float | None = None

    # -- helpers ----------------------------------------------------------

    def _existing_text(self) -> str:
        low, high = self.settings.vocal_range_low, self.settings.vocal_range_high
        if low is None or high is None:
            return "No range recorded yet."
        return (
            f"Currently recorded: {note_name(low)} to {note_name(high)} "
            f"({high - low:.0f} semitones)."
        )

    def _start(self) -> None:
        self._matched = []
        self._misses = 0
        self._going_down = True
        self._target = ANCHOR_MIDI
        self.start_button.setEnabled(False)
        self.result_label.setText("Warming up from the middle…")
        self.result_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        self._enter(Phase.LISTEN)

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._elapsed = 0.0
        self._samples = []
        self.engine.readings.clear()

        if phase is Phase.LISTEN:
            self.engine.mixer.set_tone(midi_to_hz(self._target))
            self.prompt.setText("Listen")
            self.prompt.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
            self.target_label.setText(note_name(self._target))
        elif phase is Phase.SING:
            # Tone continues, quieter, so there is something to tune against.
            self.engine.mixer.set_tone(midi_to_hz(self._target), amplitude=0.07)
            self.prompt.setText("SING")
            self.prompt.setStyleSheet(f"color: {theme.ACCENT.name()};")
        else:
            self.engine.mixer.set_tone(0.0)

    # -- the loop ---------------------------------------------------------

    def _tick(self) -> None:
        live: list[float] = []
        while self.engine.readings:
            _t, reading, smoothed = self.engine.readings.popleft()
            if smoothed is not None and reading.confidence > 0.4:
                live.append(smoothed)

        if live:
            self.you_label.setText(note_name(live[-1]))
        if self.phase in (Phase.IDLE, Phase.DONE):
            return

        if self.phase is Phase.SING:
            self._samples.extend(live)

        self._elapsed += 0.033
        span = LISTEN_SECONDS if self.phase is Phase.LISTEN else SING_SECONDS
        self.progress.setValue(int(min(1.0, self._elapsed / span) * 100))

        if self._elapsed < span:
            return

        if self.phase is Phase.LISTEN:
            self._enter(Phase.SING)
        else:
            self._judge()

    def _judge(self) -> None:
        matched = False
        if self._samples:
            errors = np.abs((np.asarray(self._samples) - self._target) * 100.0)
            # Octave-forgiving: singing the right pitch class an octave away is
            # a correct match for range purposes, not a miss.
            folded = np.minimum(errors, np.abs(errors - 1200.0))
            matched = float(np.mean(folded <= MATCH_TOLERANCE_CENTS)) >= MATCH_FRACTION

        if matched:
            self._matched.append(self._target)
            self._misses = 0
        else:
            self._misses += 1

        self._advance()

    def _advance(self) -> None:
        if self._misses >= MISSES_TO_STOP:
            self._misses = 0
            if self._going_down:
                # Switch to walking up, starting just above the anchor.
                self._going_down = False
                self._target = ANCHOR_MIDI + 1
                self.result_label.setText("Now working upward…")
                self._enter(Phase.LISTEN)
                return
            self._finish()
            return

        self._target += -1.0 if self._going_down else 1.0
        if not (FLOOR_MIDI <= self._target <= CEILING_MIDI):
            if self._going_down:
                self._going_down = False
                self._misses = 0
                self._target = ANCHOR_MIDI + 1
                self._enter(Phase.LISTEN)
            else:
                self._finish()
            return
        self._enter(Phase.LISTEN)

    def _finish(self) -> None:
        self._enter(Phase.DONE)
        self.prompt.setText("Done")
        self.prompt.setStyleSheet(f"color: {theme.TEXT.name()};")
        self.start_button.setEnabled(True)
        self.start_button.setText("Run again")
        self.progress.setValue(100)

        if len(self._matched) < 4:
            self.result_label.setStyleSheet(f"color: {theme.NEAR.name()};")
            self.result_label.setText(
                "Not enough notes were matched to measure a range. Check the "
                "microphone is selected and the level meter moves when you "
                "sing, then run it again."
            )
            return

        self._low, self._high = float(min(self._matched)), float(max(self._matched))
        self.result_label.setStyleSheet(f"color: {theme.ON_PITCH.name()};")
        self.result_label.setText(
            f"You reliably matched {note_name(self._low)} to {note_name(self._high)} "
            f"— {self._high - self._low:.0f} semitones, {len(self._matched)} notes held.\n\n"
            "These are notes you actually hit and sustained, not the extremes "
            "you can reach. Songs will be transposed to sit inside them."
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)

    # -- teardown ---------------------------------------------------------

    def _save(self) -> None:
        if self._low is not None and self._high is not None:
            self.settings.vocal_range_low = round(self._low, 2)
            self.settings.vocal_range_high = round(self._high, 2)
        self.engine.mixer.set_tone(0.0)
        self.accept()

    def reject(self) -> None:
        self.engine.mixer.set_tone(0.0)
        super().reject()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.engine.mixer.set_tone(0.0)
        super().closeEvent(event)
