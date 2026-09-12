"""Audio device selection and latency calibration.

The calibration step here is not optional housekeeping — it is what makes every
other piece of feedback in the app truthful. Until it has run, the app is
comparing your voice against the wrong moment of the song by an unknown amount.
The dialog says so plainly rather than letting you skip past it unaware.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from ..audio import calibrate
from . import theme


class SetupDialog(QDialog):
    def __init__(self, settings, engine, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audio setup")
        self.setMinimumWidth(560)
        self.settings = settings
        self.engine = engine

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        devices = engine.devices()

        layout.addWidget(self._label("Output (your headphones)", heading=True))
        self.output_box = QComboBox()
        for dev in devices["output"]:
            self.output_box.addItem(f"{dev['name']}  [{dev['hostapi']}]", dev["name"])
        self._preselect(self.output_box, settings.output_device)
        layout.addWidget(self.output_box)

        layout.addWidget(self._label("Input (your microphone)", heading=True))
        self.input_box = QComboBox()
        for dev in devices["input"]:
            self.input_box.addItem(f"{dev['name']}  [{dev['hostapi']}]", dev["name"])
        self._preselect(self.input_box, settings.input_device)
        layout.addWidget(self.input_box)

        layout.addSpacing(6)
        layout.addWidget(self._label("Round-trip latency", heading=True))
        layout.addWidget(
            self._label(
                "Your voice reaches the app later than you sang it. Measuring "
                "that delay is what lets the app compare you against the right "
                "moment of the song. Put your headphones on and keep the room "
                "quiet, then press Measure.",
                wrap=True, dim=True,
            )
        )

        row = QHBoxLayout()
        self.measure_button = QPushButton("Measure")
        self.measure_button.setObjectName("primary")
        self.measure_button.clicked.connect(self._measure)
        row.addWidget(self.measure_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        row.addWidget(self.progress, 1)
        layout.addLayout(row)

        self.result_label = self._label(self._current_latency_text(), wrap=True)
        layout.addWidget(self.result_label)

        layout.addWidget(self._label("Manual adjustment", heading=True))
        layout.addWidget(
            self._label(
                "If highlighting still feels a touch early or late, nudge it here.",
                dim=True, wrap=True,
            )
        )
        nudge_row = QHBoxLayout()
        self.nudge = QSlider(Qt.Orientation.Horizontal)
        self.nudge.setRange(-100, 100)
        self.nudge.setValue(int(settings.latency_nudge_ms))
        self.nudge_label = QLabel(f"{settings.latency_nudge_ms:+.0f} ms")
        self.nudge_label.setFont(theme.mono_font(9))
        self.nudge_label.setFixedWidth(64)
        self.nudge.valueChanged.connect(
            lambda v: (self.nudge_label.setText(f"{v:+d} ms"),
                       setattr(self.settings, "latency_nudge_ms", float(v)))
        )
        nudge_row.addWidget(self.nudge, 1)
        nudge_row.addWidget(self.nudge_label)
        layout.addLayout(nudge_row)

        layout.addWidget(self._label("Picture / sound sync", heading=True))
        layout.addWidget(
            self._label(
                "If the highlighted word and the scrolling notes run ahead of "
                "or behind what you hear, correct it here. Drag right if the "
                "picture is early, left if it lags. Judge it by watching while "
                "the song plays — no measurement can do this for you.",
                dim=True, wrap=True,
            )
        )
        visual_row = QHBoxLayout()
        self.visual = QSlider(Qt.Orientation.Horizontal)
        self.visual.setRange(-300, 300)
        self.visual.setValue(int(settings.visual_offset_ms))
        self.visual_label = QLabel(f"{settings.visual_offset_ms:+.0f} ms")
        self.visual_label.setFont(theme.mono_font(9))
        self.visual_label.setFixedWidth(64)
        self.visual.valueChanged.connect(
            lambda v: (self.visual_label.setText(f"{v:+d} ms"),
                       setattr(self.settings, "visual_offset_ms", float(v)))
        )
        visual_row.addWidget(self.visual, 1)
        visual_row.addWidget(self.visual_label)
        layout.addLayout(visual_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _label(text: str, *, heading: bool = False, dim: bool = False,
               wrap: bool = False) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(wrap)
        if heading:
            label.setObjectName("heading")
        elif dim:
            label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        return label

    @staticmethod
    def _preselect(box: QComboBox, name: str | None) -> None:
        if not name:
            return
        index = box.findData(name)
        if index >= 0:
            box.setCurrentIndex(index)

    def _current_latency_text(self) -> str:
        device = self.settings.output_device or ""
        measured = self.settings.latency_ms_by_device.get(device)
        if measured is None:
            return "Not measured yet — feedback timing will be approximate."
        return f"Measured: {measured:.0f} ms for this output device."

    def _measure(self) -> None:
        self.settings.output_device = self.output_box.currentData()
        self.settings.input_device = self.input_box.currentData()

        self.measure_button.setEnabled(False)
        self.progress.show()
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        try:
            result = calibrate.measure(self.engine.play_and_record)
        except Exception as exc:
            self.progress.hide()
            self.measure_button.setEnabled(True)
            QMessageBox.warning(self, "Measurement failed", str(exc))
            return

        self.progress.hide()
        self.measure_button.setEnabled(True)

        if result.ok:
            self.settings.latency_ms_by_device[self.settings.output_device or ""] = result.latency_ms
            self.result_label.setStyleSheet(f"color: {theme.ON_PITCH.name()};")
            self.result_label.setText(
                f"{result.message}  Confidence {result.confidence:.0%} "
                f"from {len(result.measurements_ms)} measurements."
            )
        else:
            self.result_label.setStyleSheet(f"color: {theme.NEAR.name()};")
            self.result_label.setText(result.message)

    def _accept(self) -> None:
        self.settings.output_device = self.output_box.currentData()
        self.settings.input_device = self.input_box.currentData()
        self.accept()
