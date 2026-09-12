"""Mixing your recorded takes into a finished cover."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from ..audio import recorder
from ..audio.export import MixSettings, VocalChain, export
from . import theme


class ExportDialog(QDialog):
    def __init__(self, song, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export cover")
        self.setMinimumWidth(560)
        self.song = song
        self.takes_dir = song.paths.root / "takes"

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(self._heading("Takes to include"))
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        takes = recorder.list_takes(self.takes_dir)
        for take in takes:
            item = QListWidgetItem(
                f"{take.name}   {take.duration:.0f}s   from {take.song_offset:.0f}s"
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, take)
            self.list.addItem(item)
        layout.addWidget(self.list)

        if not takes:
            layout.addWidget(
                self._dim("No takes recorded yet. Press Record while the song "
                          "plays, and your performance is saved here.")
            )

        layout.addWidget(
            self._dim("Multiple takes are placed on one timeline by where they "
                      "were recorded, so you can sing the verse and the chorus "
                      "in separate passes and combine them.")
        )

        self.vocal_slider, vocal_row = self._slider("Your voice", 0, -12, 12)
        layout.addLayout(vocal_row)
        self.backing_slider, backing_row = self._slider("Backing track", -1, -24, 6)
        layout.addLayout(backing_row)

        self.chain_check = QCheckBox("Apply vocal polish (EQ, compression, a little reverb)")
        self.chain_check.setChecked(True)
        self.chain_check.setToolTip(
            "Helps a close-mic vocal sit in the mix. It will not fix pitch."
        )
        layout.addWidget(self.chain_check)

        self.duet_check = QCheckBox("Keep the original singer in the mix (duet)")
        layout.addWidget(self.duet_check)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        buttons = QDialogButtonBox()
        self.export_button = QPushButton("Export…")
        self.export_button.setObjectName("primary")
        self.export_button.setEnabled(bool(takes))
        self.export_button.clicked.connect(self._export)
        buttons.addButton(self.export_button, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("heading")
        return label

    @staticmethod
    def _dim(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        return label

    def _slider(self, name: str, value: int, low: int, high: int):
        row = QHBoxLayout()
        caption = QLabel(name)
        caption.setFixedWidth(110)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        readout = QLabel(f"{value:+d} dB")
        readout.setFont(theme.mono_font(9))
        readout.setFixedWidth(56)
        slider.valueChanged.connect(lambda v: readout.setText(f"{v:+d} dB"))
        row.addWidget(caption)
        row.addWidget(slider, 1)
        row.addWidget(readout)
        return slider, row

    def _selected(self):
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _export(self) -> None:
        takes = self._selected()
        if not takes:
            QMessageBox.information(self, "Nothing selected", "Tick at least one take.")
            return

        suggested = str(Path.home() / "Music" / f"{self.song.meta.display_name} (cover).mp3")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cover", suggested, "MP3 (*.mp3);;WAV (*.wav);;FLAC (*.flac)"
        )
        if not path:
            return

        settings = MixSettings(
            your_vocal_db=float(self.vocal_slider.value()),
            backing_db=float(self.backing_slider.value()),
            original_vocal_db=-6.0 if self.duet_check.isChecked() else None,
            chain=VocalChain(enabled=self.chain_check.isChecked()),
        )

        self.progress.show()
        self.export_button.setEnabled(False)
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()
        try:
            written = export(
                takes,
                self.song.paths.accompaniment,
                Path(path),
                original_vocal_path=self.song.paths.vocals,
                settings=settings,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            self.progress.hide()
            self.export_button.setEnabled(True)

        QMessageBox.information(self, "Cover exported", f"Saved to:\n{written}")
        self.accept()
