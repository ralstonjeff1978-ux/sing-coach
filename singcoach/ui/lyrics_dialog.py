"""Paste your own lyrics and let the app time them.

The transcription model gets the words roughly right and the timing roughly
right. When you already have the correct words — from the album booklet, the
digital purchase, or any licensed lyrics source — pasting them in removes half
the problem entirely, and the half that remains (working out *when* each word
happens) is what the pitch tracker is actually good at.

Line breaks matter and are respected. A lyric sheet breaks where the singer
breathes, and so do the voiced regions in the audio, so the two line up
naturally.

The result is saved as a standard ``.lrc`` beside the song. It takes priority
over anything the transcriber produced, permanently, and survives re-analysis.
"""

from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ..analysis import lyrics as lyrics_mod
from . import theme


class LyricsDialog(QDialog):
    def __init__(self, song, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lyrics")
        self.setMinimumSize(620, 560)
        self.song = song
        self.result_lyrics: lyrics_mod.Lyrics | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        heading = QLabel("PASTE THE LYRICS")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        layout.addWidget(
            self._dim(
                "One line per sung phrase — exactly as a lyric sheet is laid "
                "out. The app matches your lines to the phrases it can hear in "
                "the vocal, so your line breaks are what make the timing work.\n\n"
                "Blank lines and [section] markers are ignored, so you can "
                "paste a whole sheet as-is."
            )
        )

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "Paste the words here, one phrase per line…"
        )
        self.editor.setFont(theme.ui_font(12))
        layout.addWidget(self.editor, 1)

        existing = self._load_existing()
        if existing:
            self.editor.setPlainText(existing)
            layout.addWidget(
                self._dim(
                    "Loaded what is currently in use. Correct it and re-time, "
                    "or replace it entirely."
                )
            )

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        row = QHBoxLayout()
        self.time_button = QPushButton("Time these lyrics to the song")
        self.time_button.setObjectName("primary")
        self.time_button.clicked.connect(self._apply)
        row.addWidget(self.time_button)

        clear = QPushButton("Clear")
        clear.clicked.connect(self.editor.clear)
        row.addWidget(clear)
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _dim(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        return label

    def _load_existing(self) -> str:
        """Show whatever lyrics are currently in force, as plain text."""
        paths = self.song.paths
        try:
            if paths.lyrics_override.exists():
                current = lyrics_mod.read_lrc(paths.lyrics_override)
            elif paths.lyrics.exists():
                current = lyrics_mod.load(paths.lyrics)
            else:
                return ""
        except (json.JSONDecodeError, OSError):
            return ""
        return "\n".join(line.text for line in current.lines)

    def _apply(self) -> None:
        text = self.editor.toPlainText().strip()
        if not text:
            self.status.setText("Nothing to time — paste the words first.")
            self.status.setStyleSheet(f"color: {theme.NEAR.name()};")
            return

        try:
            analysis = json.loads(self.song.paths.analysis.read_text("utf-8"))
            melody = analysis["melody"]
            contour = np.array(
                [np.nan if v is None else v for v in melody["contour"]["midi"]],
                dtype=float,
            )
            hop = melody["hop_s"]
        except (OSError, KeyError, json.JSONDecodeError):
            contour, hop = None, 0.0

        # The machine transcript is used purely as a positional index: it knows
        # roughly where in the song each phrase occurs, which is what stops a
        # chorus line landing on a verse.
        heard = None
        raw_path = self.song.paths.vocals.with_name("words_raw.json")
        try:
            if raw_path.exists():
                words = [
                    lyrics_mod.Word(**w)
                    for w in json.loads(raw_path.read_text("utf-8"))
                ]
                heard = lyrics_mod.Lyrics(
                    model="asr", source="transcribed",
                    lines=lyrics_mod.group_lines(words),
                )
        except (OSError, json.JSONDecodeError, TypeError):
            heard = None

        lyrics = lyrics_mod.from_text(text, contour, hop, heard=heard)
        if not lyrics.lines:
            self.status.setText("Could not read any lines from that.")
            self.status.setStyleSheet(f"color: {theme.NEAR.name()};")
            return

        lyrics_mod.write_lrc(lyrics, self.song.paths.lyrics_override)
        lyrics_mod.save(lyrics, self.song.paths.lyrics)
        self.result_lyrics = lyrics

        words = len(lyrics.words)
        self.status.setText(
            f"Timed {words} words across {len(lyrics.lines)} lines and saved to "
            f"{self.song.paths.lyrics_override.name}. These now take priority "
            "over the transcript, permanently.\n\n"
            "If a phrase highlights early or late, adjusting where you break "
            "that line is the fastest fix — the breaks are what anchor it."
        )
        self.status.setStyleSheet(f"color: {theme.ON_PITCH.name()};")

    def accept(self) -> None:
        super().accept()
