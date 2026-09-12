"""Transport, stem mix, and practice controls.

The stem mixer here is the control the whole app was built around, so it gets
named presets rather than an anonymous slider: "Karaoke", "Guide", "Duet" say
what they do, and the slider stays available for anything in between.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..config import PracticeToggles
from . import theme


def _panel() -> QFrame:
    frame = QFrame()
    frame.setObjectName("panel")
    return frame


def _heading(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("heading")
    return label


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(max(0, seconds)), 60)
    return f"{m}:{s:02d}"


class StemMixer(QFrame):
    """Vocal / backing balance, with the three presets that matter."""

    vocal_changed = Signal(float)
    backing_changed = Signal(float)

    PRESETS = (
        ("Karaoke", 0.0, "The original lead is gone. You are the lead."),
        ("Guide", 0.20, "Original kept quiet underneath, so you cannot get lost."),
        ("Duet", 1.00, "Full original vocal — sing a harmony against it."),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        layout.addWidget(_heading("Mix"))

        presets = QHBoxLayout()
        presets.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for i, (name, value, tip) in enumerate(self.PRESETS):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked, v=value: self.set_vocal(v))
            self._group.addButton(button, i)
            presets.addWidget(button)
        self._group.button(0).setChecked(True)
        layout.addLayout(presets)

        self.vocal_slider, self.vocal_label = self._slider("Original vocal", 0)
        self.vocal_slider.valueChanged.connect(self._on_vocal)
        layout.addLayout(self._row(self.vocal_slider, self.vocal_label, "Original vocal"))

        self.backing_slider, self.backing_label = self._slider("Music", 100)
        self.backing_slider.valueChanged.connect(self._on_backing)
        layout.addLayout(self._row(self.backing_slider, self.backing_label, "Music"))

    @staticmethod
    def _slider(_name: str, value: int) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        label = QLabel(f"{value}%")
        label.setFont(theme.mono_font(9))
        label.setFixedWidth(38)
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        return slider, label

    @staticmethod
    def _row(slider: QSlider, value_label: QLabel, name: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        caption = QLabel(name)
        caption.setFixedWidth(94)
        caption.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        row.addWidget(caption)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        return row

    def _on_vocal(self, value: int) -> None:
        self.vocal_label.setText(f"{value}%")
        # Un-check presets when the slider lands somewhere else.
        exact = next((i for i, (_n, v, _t) in enumerate(self.PRESETS)
                      if abs(v * 100 - value) < 0.5), None)
        self._group.setExclusive(False)
        for i, button in enumerate(self._group.buttons()):
            button.setChecked(i == exact)
        self._group.setExclusive(True)
        self.vocal_changed.emit(value / 100.0)

    def _on_backing(self, value: int) -> None:
        self.backing_label.setText(f"{value}%")
        self.backing_changed.emit(value / 100.0)

    def set_vocal(self, value: float) -> None:
        self.vocal_slider.setValue(int(round(value * 100)))


class TransportBar(QFrame):
    play_pause = Signal()
    seek = Signal(float)
    record_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._duration = 0.0
        self._scrubbing = False
        self._shown_time = ""
        self._shown_playing: bool | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("primary")
        self.play_button.setFixedWidth(88)
        self.play_button.clicked.connect(self.play_pause.emit)
        layout.addWidget(self.play_button)

        self.time_label = QLabel("0:00")
        self.time_label.setFont(theme.mono_font(10))
        self.time_label.setFixedWidth(44)
        layout.addWidget(self.time_label)

        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, 1000)
        self.scrubber.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self.scrubber.sliderReleased.connect(self._released)
        layout.addWidget(self.scrubber, 1)

        self.duration_label = QLabel("0:00")
        self.duration_label.setFont(theme.mono_font(10))
        self.duration_label.setFixedWidth(44)
        self.duration_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        layout.addWidget(self.duration_label)

        self.record_button = QPushButton("● Record")
        self.record_button.setCheckable(True)
        self.record_button.setFixedWidth(96)
        self.record_button.setToolTip("Record a take you can mix into a cover")
        self.record_button.toggled.connect(self.record_toggled.emit)
        layout.addWidget(self.record_button)

    def _released(self) -> None:
        self._scrubbing = False
        if self._duration:
            self.seek.emit(self.scrubber.value() / 1000.0 * self._duration)

    def set_duration(self, seconds: float) -> None:
        self._duration = seconds
        self.duration_label.setText(_fmt_time(seconds))

    def set_position(self, seconds: float) -> None:
        # Both of these are called every frame; skip the work when the visible
        # result would be identical. Widget text changes trigger a relayout.
        text = _fmt_time(seconds)
        if text != self._shown_time:
            self._shown_time = text
            self.time_label.setText(text)
        if not self._scrubbing and self._duration:
            value = int(seconds / self._duration * 1000)
            if value != self.scrubber.value():
                self.scrubber.setValue(value)

    def set_playing(self, playing: bool) -> None:
        if playing == self._shown_playing:
            return
        self._shown_playing = playing
        self.play_button.setText("Pause" if playing else "Play")

    def set_recording(self, recording: bool) -> None:
        self.record_button.setObjectName("record" if recording else "")
        self.record_button.setText("■ Stop" if recording else "● Record")
        self.record_button.style().polish(self.record_button)


class PracticePanel(QFrame):
    """The four practice features, each independently switchable.

    Defaults are all off: with nothing enabled this is a plain sing-along, and
    that is a legitimate way to use the app rather than a degraded one.
    """

    toggled = Signal(str, bool)
    speed_changed = Signal(float)
    transpose_changed = Signal(int)
    loop_requested = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)
        layout.addWidget(_heading("Practice"))

        self.checks: dict[str, QCheckBox] = {}
        for key, label, tip in (
            ("transpose", "Fit to my range",
             "Shift the whole song so the melody sits where you sing comfortably."),
            ("slowdown_loop", "Slow down and loop",
             "Practise a hard phrase slower, on repeat, without changing its pitch."),
            ("scoring", "Score me",
             "Track accuracy per note and diagnose recurring habits over time."),
            ("harmony_guides", "Show harmony lines",
             "Draw a third and a fifth so you can practise harmonising."),
        ):
            box = QCheckBox(label)
            box.setToolTip(tip)
            box.toggled.connect(lambda on, k=key: self._on_toggle(k, on))
            self.checks[key] = box
            layout.addWidget(box)

        # Controls that only make sense when their feature is on.
        self.speed_row = QHBoxLayout()
        self.speed_box = QComboBox()
        for pct in (60, 70, 80, 90, 100):
            self.speed_box.addItem(f"{pct}%", pct / 100.0)
        self.speed_box.setCurrentIndex(4)
        self.speed_box.currentIndexChanged.connect(
            lambda _i: self.speed_changed.emit(self.speed_box.currentData())
        )
        self.loop_button = QPushButton("Set loop")
        self.loop_button.setCheckable(True)
        self.loop_button.toggled.connect(self.loop_requested.emit)
        self.speed_row.addWidget(QLabel("Speed"))
        self.speed_row.addWidget(self.speed_box, 1)
        self.speed_row.addWidget(self.loop_button)
        layout.addLayout(self.speed_row)

        self.transpose_row = QHBoxLayout()
        self.transpose_box = QComboBox()
        for n in range(-7, 8):
            self.transpose_box.addItem("original key" if n == 0 else f"{n:+d} semitones", n)
        self.transpose_box.setCurrentIndex(7)
        self.transpose_box.currentIndexChanged.connect(
            lambda _i: self.transpose_changed.emit(self.transpose_box.currentData())
        )
        self.transpose_row.addWidget(QLabel("Key"))
        self.transpose_row.addWidget(self.transpose_box, 1)
        layout.addLayout(self.transpose_row)

        self._sync_enabled()

    def _on_toggle(self, key: str, on: bool) -> None:
        self._sync_enabled()
        self.toggled.emit(key, on)

    def _sync_enabled(self) -> None:
        slow = self.checks["slowdown_loop"].isChecked()
        self.speed_box.setEnabled(slow)
        self.loop_button.setEnabled(slow)
        self.transpose_box.setEnabled(self.checks["transpose"].isChecked())

    def toggles(self) -> PracticeToggles:
        return PracticeToggles(**{k: box.isChecked() for k, box in self.checks.items()})

    def apply(self, toggles: PracticeToggles) -> None:
        for key, box in self.checks.items():
            box.blockSignals(True)
            box.setChecked(getattr(toggles, key))
            box.blockSignals(False)
        self._sync_enabled()


class StatusStrip(QFrame):
    """Quiet, persistent truth about the setup.

    Latency and input level live here because when something is wrong with
    them the singer will otherwise assume the fault is theirs.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(16)

        self.song_label = QLabel("No song loaded")
        self.song_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        layout.addWidget(self.song_label, 1)

        self.key_label = QLabel("")
        self.key_label.setFont(theme.mono_font(9))
        self.key_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        layout.addWidget(self.key_label)

        self.mode_label = QLabel("")
        self.mode_label.setFont(theme.mono_font(9))
        layout.addWidget(self.mode_label)

        self.latency_label = QLabel("latency: not measured")
        self.latency_label.setFont(theme.mono_font(9))
        self.latency_label.setStyleSheet(f"color: {theme.NEAR.name()};")
        self._latency_state: tuple | None = (None, False)
        layout.addWidget(self.latency_label)

        self.dropout_label = QLabel("")
        self.dropout_label.setFont(theme.mono_font(9))
        self.dropout_label.setToolTip(
            "Audio dropouts since the song was loaded. Anything above zero "
            "means playback is glitching and the pitch feedback is unreliable."
        )
        self._dropouts_shown = -1
        layout.addWidget(self.dropout_label)

        from .tuner import LevelMeter

        layout.addWidget(QLabel("mic"))
        self.level = LevelMeter()
        layout.addWidget(self.level)

    def set_dropouts(self, count: int) -> None:
        if count == self._dropouts_shown:
            return
        self._dropouts_shown = count
        if count == 0:
            self.dropout_label.setText("audio: clean")
            self.dropout_label.setStyleSheet(f"color: {theme.TEXT_FAINT.name()};")
        else:
            self.dropout_label.setText(f"dropouts: {count}")
            self.dropout_label.setStyleSheet(f"color: {theme.OFF.name()};")

    def set_latency(self, ms: float | None, measured: bool) -> None:
        """Update the latency readout, but only when it actually changed.

        This is called every UI frame. ``setStyleSheet`` forces Qt to reparse
        the sheet and repolish the widget's whole style cascade — doing that 60
        times a second, on the same unchanged string, is pure waste and it
        holds the GIL while it happens, which steals time from the audio
        callback.
        """
        state = (round(ms) if ms is not None else None, measured)
        if state == self._latency_state:
            return
        self._latency_state = state

        if not measured:
            self.latency_label.setText("latency: not measured")
            self.latency_label.setStyleSheet(f"color: {theme.NEAR.name()};")
        else:
            self.latency_label.setText(f"latency: {ms:.0f} ms")
            self.latency_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
