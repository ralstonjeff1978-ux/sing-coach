"""The main window: everything wired together.

Threading contract, which is the part that matters:

* The audio callback runs on PortAudio's thread and touches only the mixer,
  capture buffer, and recorder.
* Pitch analysis runs on its own worker and appends to a deque.
* This class runs on the Qt thread and *polls* at 60 fps. It never blocks, and
  nothing else ever touches a widget.

Polling rather than signalling from the audio thread is deliberate: emitting a
Qt signal 86 times a second from a realtime thread invites queue build-up and
priority inversion. Reading the newest state each frame is simpler and cannot
starve the audio.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..analysis import lyrics as lyrics_mod
from ..analysis import pipeline
from ..audio.engine import AudioEngine
from ..audio.mixer import StemSet
from ..audio.render import render_both, transform_notes, transform_times
from ..library import importer
from ..library.song import Song
from ..pitch.compare import ContourTarget, NoteTarget, OctaveTracker, compare
from ..scoring import SessionScorer, save_session
from . import theme
from .controls import PracticePanel, StatusStrip, StemMixer, TransportBar
from .highway import HighwayWidget, note_name
from .lyrics_view import LyricsWidget
from .tuner import TunerWidget

UI_FPS = 60


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SingCoach")
        self.resize(1280, 860)

        self.settings = config.load_settings()
        self.engine = AudioEngine(self.settings)

        self.song: Song | None = None
        self.analysis: dict | None = None
        self.lyrics: lyrics_mod.Lyrics | None = None
        self.target = None              # NoteTarget or ContourTarget
        self.scorer: SessionScorer | None = None
        self.octaves = OctaveTracker()
        self._shown_octave = 0
        self.expressive = False
        self.transpose = 0
        self.speed = 1.0
        self._loop_anchor: float | None = None

        self._build_ui()
        self._build_menu()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / UI_FPS))

        try:
            self.engine.start()
        except Exception as exc:  # device problems must not kill the window
            QMessageBox.warning(
                self, "Audio device",
                f"Could not open the audio device:\n\n{exc}\n\n"
                "Open Setup to choose a different one.",
            )

        self._check_first_run()

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.status_strip = StatusStrip()
        root.addWidget(self.status_strip)

        self.highway = HighwayWidget()
        root.addWidget(self.highway, 3)

        middle = QHBoxLayout()
        middle.setSpacing(10)
        self.tuner = TunerWidget()
        middle.addWidget(self.tuner, 2)
        self.lyrics_view = LyricsWidget()
        middle.addWidget(self.lyrics_view, 3)
        root.addLayout(middle, 1)

        self.transport = TransportBar()
        self.transport.play_pause.connect(self._toggle_play)
        self.transport.seek.connect(self._seek)
        self.transport.record_toggled.connect(self._toggle_record)
        root.addWidget(self.transport)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self.mixer_panel = StemMixer()
        self.mixer_panel.vocal_changed.connect(
            lambda v: self.engine.mixer.set_gain("vocals", v)
        )
        self.mixer_panel.backing_changed.connect(
            lambda v: self.engine.mixer.set_gain("accompaniment", v)
        )
        bottom.addWidget(self.mixer_panel, 1)

        self.practice = PracticePanel()
        self.practice.toggled.connect(self._on_practice_toggle)
        self.practice.speed_changed.connect(self._on_speed)
        self.practice.transpose_changed.connect(self._on_transpose)
        self.practice.loop_requested.connect(self._on_loop_button)
        bottom.addWidget(self.practice, 1)

        self.findings_label = QLabel("Load a song to begin.")
        self.findings_label.setWordWrap(True)
        self.findings_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.findings_label.setStyleSheet(
            f"color: {theme.TEXT_DIM.name()}; background: {theme.PANEL.name()};"
            f" border: 1px solid {theme.BORDER.name()}; border-radius: 8px; padding: 12px;"
        )
        bottom.addWidget(self.findings_label, 1)
        root.addLayout(bottom)

        self.setCentralWidget(central)

    def _build_menu(self) -> None:
        song_menu = self.menuBar().addMenu("&Song")
        self._action(song_menu, "&Open…", QKeySequence.StandardKey.Open, self.open_song)
        self._action(song_menu, "&Lyrics…", "Ctrl+L", self.open_lyrics)
        self._action(song_menu, "&Sync lyrics by ear…", "Ctrl+T", self.open_sync)
        song_menu.addSeparator()
        self._action(song_menu, "Export &cover…", "Ctrl+E", self.export_cover)

        setup_menu = self.menuBar().addMenu("&Setup")
        self._action(setup_menu, "Audio devices and &latency…", "Ctrl+,", self.open_setup)
        self._action(setup_menu, "Pitch match &warm-up…", "Ctrl+W", self.open_range_wizard)

        play_menu = self.menuBar().addMenu("&Play")
        self._action(play_menu, "Play / Pause", "Space", self._toggle_play)
        self._action(play_menu, "Back 5 seconds", "Left", lambda: self._nudge(-5))
        self._action(play_menu, "Forward 5 seconds", "Right", lambda: self._nudge(5))
        self._action(play_menu, "Restart", "Home", lambda: self._seek(0.0))

    def _action(self, menu, text, shortcut, slot) -> None:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)

    # -- song loading -----------------------------------------------------

    def open_song(self) -> None:
        patterns = " ".join(f"*{s}" for s in sorted(importer.SUPPORTED_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a song", str(Path.home() / "Music"), f"Audio ({patterns})"
        )
        if path:
            self.load_song(Path(path))

    def load_song(self, path: Path) -> None:
        progress = QProgressDialog("Preparing…", None, 0, 100, self)
        progress.setWindowTitle("SingCoach")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)

        def report(stage: str, fraction: float) -> None:
            progress.setLabelText(stage)
            progress.setValue(int(fraction * 100))
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()

        try:
            song = importer.import_song(path, progress=report)
            pipeline.analyse_song(song, progress=report)
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Could not prepare this song", str(exc))
            return
        progress.close()
        self._activate(song)

    def _activate(self, song: Song) -> None:
        self.song = song
        self.analysis = pipeline.load_analysis(song)
        melody = self.analysis["melody"]
        self.expressive = melody.get("suggested_mode") == "expressive"

        self._load_lyrics()
        self._apply_variant(reload_audio=True)

        self.settings.last_song_hash = song.meta.hash
        config.save_settings(self.settings)

        kb = self.analysis["key_beats"]
        self.status_strip.song_label.setText(song.meta.display_name)
        self.status_strip.key_label.setText(f"{kb['key']} {kb['mode']}")
        self.status_strip.mode_label.setText(
            "expressive — following the original's phrasing"
            if self.expressive else "note targets"
        )
        self.status_strip.mode_label.setStyleSheet(
            f"color: {(theme.ACCENT if self.expressive else theme.TEXT_DIM).name()};"
        )

        self.engine.arm_recorder(song.meta.hash, song.paths.root / "takes")
        self.practice.apply(song.practice_toggles(self.settings.practice))
        self._describe_song()

    def _load_lyrics(self) -> None:
        assert self.song is not None
        paths = self.song.paths
        try:
            if paths.lyrics_override.exists():
                self.lyrics = lyrics_mod.read_lrc(paths.lyrics_override)
            elif paths.lyrics.exists():
                self.lyrics = lyrics_mod.load(paths.lyrics)
            else:
                self.lyrics = None
        except (json.JSONDecodeError, OSError):
            self.lyrics = None
        self.lyrics_view.set_lyrics(self.lyrics)

    def _apply_variant(self, *, reload_audio: bool) -> None:
        """Rebuild targets (and audio) for the current transpose/speed."""
        if not self.song or not self.analysis:
            return
        melody = self.analysis["melody"]

        notes = transform_notes(melody["notes"], self.transpose, self.speed)
        contour = np.array(
            [np.nan if v is None else v for v in melody["contour"]["midi"]], dtype=np.float64
        )
        if self.transpose:
            contour = contour + self.transpose
        hop = melody["hop_s"] / self.speed

        self.target = (
            ContourTarget(contour, hop) if self.expressive else NoteTarget(notes)
        )
        harmony = ()
        if self.practice.checks["harmony_guides"].isChecked():
            third = self.analysis["key_beats"].get("third_semitones", 4)
            harmony = (third, 7)
        self.highway.state.harmony_offsets = harmony
        self.highway.set_song(notes, contour, hop, expressive=self.expressive)
        # Same voiced regions the lyrics are timed against, so the shaded
        # "sing here" bands and the word highlighting can never disagree.
        self.highway.state.vocal_spans = [
            (a / self.speed, b / self.speed)
            for a, b in lyrics_mod.voiced_spans(contour, melody["hop_s"])
        ]

        if self.lyrics is not None and self.speed != 1.0:
            # Keep the words with the music when the music is slowed.
            for line in self.lyrics.lines:
                line.start, line.end = line.start / self.speed, line.end / self.speed
                for w in line.words:
                    w.start, w.end = w.start / self.speed, w.end / self.speed
            self.lyrics_view.set_lyrics(self.lyrics)

        if reload_audio:
            paths = self.song.paths
            if self.transpose or self.speed != 1.0:
                vocals, accomp = render_both(paths, self.transpose, self.speed)
            else:
                vocals, accomp = paths.vocals, paths.accompaniment
            self.engine.load_song(StemSet.load(vocals, accomp))
            self.transport.set_duration(self.engine.mixer.duration)

        if self.practice.checks["scoring"].isChecked():
            self.scorer = SessionScorer(notes, self.settings.scoring_tolerance_cents)

    def _describe_song(self) -> None:
        if not self.analysis:
            return
        melody = self.analysis["melody"]
        low, high = melody.get("range_low_midi"), melody.get("range_high_midi")
        lines = []
        if self.expressive:
            lines.append(
                "This vocal is sung expressively — scoops, slides and blue notes. "
                "Targets follow the original's actual pitch line rather than a "
                "note grid, so bending a note is not marked wrong."
            )
        if low and high and self.settings.vocal_range_low and self.settings.vocal_range_high:
            from ..audio.render import suggest_transpose

            shift, message = suggest_transpose(
                low, high, self.settings.vocal_range_low, self.settings.vocal_range_high
            )
            lines.append(message)
        elif low and high:
            lines.append(
                f"This melody spans {note_name(low)} to {note_name(high)}. "
                "Run Setup → Pitch match warm-up (Ctrl+W) and the app will "
                "tell you which key puts it in your voice."
            )
        self.findings_label.setText("\n\n".join(lines))

    # -- transport --------------------------------------------------------

    def _toggle_play(self) -> None:
        if not self.engine.mixer.loaded:
            return
        self.engine.mixer.toggle()
        self.transport.set_playing(self.engine.mixer.playing)

    def _seek(self, seconds: float) -> None:
        self.engine.mixer.seek(seconds)
        self.highway.clear_trace()

    def _nudge(self, delta: float) -> None:
        self._seek(max(0.0, self.engine.mixer.position + delta))

    def _toggle_record(self, on: bool) -> None:
        if not self.song:
            self.transport.record_button.setChecked(False)
            return
        if on:
            if not self.settings.latency_ms_by_device:
                QMessageBox.information(
                    self, "Measure latency first",
                    "Recording needs the round-trip latency so your voice lines "
                    "up with the music.\n\nSetup → Audio devices and latency.",
                )
                self.transport.record_button.setChecked(False)
                return
            self.engine.start_take(speed=self.speed, transpose=self.transpose)
        else:
            take = self.engine.stop_take()
            if take is not None:
                self.findings_label.setText(
                    f"Saved take '{take.name}' ({take.duration:.0f}s).\n\n"
                    "Song → Export cover to mix it with the backing track."
                )
        self.transport.set_recording(on)

    # -- practice ---------------------------------------------------------

    def _on_practice_toggle(self, key: str, on: bool) -> None:
        if self.song:
            self.song.meta.practice_overrides[key] = on
            self.song.save_meta()
        if key == "scoring":
            if on and self.analysis:
                notes = transform_notes(
                    self.analysis["melody"]["notes"], self.transpose, self.speed
                )
                self.scorer = SessionScorer(notes, self.settings.scoring_tolerance_cents)
            elif not on:
                self._finish_scoring()
        elif key == "harmony_guides":
            self._apply_variant(reload_audio=False)
        elif key == "transpose" and not on and self.transpose:
            self.practice.transpose_box.setCurrentIndex(7)
        elif key == "slowdown_loop" and not on:
            self.practice.speed_box.setCurrentIndex(4)
            self.engine.mixer.set_loop(None, None)
            self.highway.state.loop = None

    def _on_speed(self, speed: float) -> None:
        if speed == self.speed or not self.song:
            return
        self.speed = speed
        self._rerender()

    def _on_transpose(self, semitones: int) -> None:
        if semitones == self.transpose or not self.song:
            return
        self.transpose = semitones
        self._rerender()

    def _rerender(self) -> None:
        was_playing = self.engine.mixer.playing
        self.engine.mixer.pause()
        progress = QProgressDialog("Rendering audio…", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)

        def report(stage: str, fraction: float) -> None:
            progress.setLabelText(stage)
            progress.setValue(int(fraction * 100))
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()

        try:
            self._load_lyrics()          # reload at original timing, then rescale
            self._apply_variant(reload_audio=True)
        finally:
            progress.close()
        if was_playing:
            self.engine.mixer.play()

    def _on_loop_button(self, on: bool) -> None:
        if not on:
            self.engine.mixer.set_loop(None, None)
            self.highway.state.loop = None
            self._loop_anchor = None
            self.practice.loop_button.setText("Set loop")
            return
        # First press marks A, second marks B.
        self._loop_anchor = self.engine.mixer.position
        self.practice.loop_button.setText("Loop set")
        beats = self.analysis["key_beats"]["beats"] if self.analysis else []
        beats = transform_times(beats, self.speed)
        from ..analysis.key_beats import snap_to_beat

        start = snap_to_beat(self._loop_anchor, beats)
        end = snap_to_beat(start + 8.0, beats)
        self.engine.mixer.set_loop(start, end)
        self.highway.state.loop = (start, end)

    # -- per-frame update -------------------------------------------------

    def _tick(self) -> None:
        status = self.engine.status()
        position = status.position

        # The visuals follow what is being *heard*, not what the mixer has
        # produced — that audio is still in the output buffer. Without this the
        # highway and the lyric highlighting both run early by the buffer
        # depth, which reads as the karaoke being out of time.
        heard = self.engine.heard_position

        self.transport.set_position(position)
        self.transport.set_playing(status.playing)
        self.highway.set_position(heard)
        self.lyrics_view.set_position(heard)
        self.status_strip.level.set_level(status.input_peak_dbfs, status.clipping)
        self.status_strip.set_dropouts(status.underruns + status.overruns)

        latency = self.settings.latency_ms()
        self.status_strip.set_latency(latency, bool(self.settings.latency_ms_by_device))

        # Drain new pitch readings and judge each against the target.
        #
        # Each reading already carries the song position it belongs to,
        # computed from its own microphone sample index and latency-corrected
        # by the engine. Recomputing it here from the current playhead — as an
        # earlier version did — stamped every reading in a drained burst with
        # the same instant, which piled the trace into a vertical line and made
        # onset timing meaningless.
        latest = None
        while self.engine.readings:
            song_time, reading, smoothed = self.engine.readings.popleft()
            if self.target is None:
                continue
            target_midi = (
                self.target.smoothed_at(song_time)
                if isinstance(self.target, ContourTarget)
                else self.target.at(song_time)
            )

            # Follow the singer's octave rather than arguing with it. Moving a
            # song into your own register is normal musicianship, not an error,
            # and reporting "an octave low" on every note would make the app
            # useless to anyone whose voice does not match the record.
            shift = self.octaves.push(smoothed, target_midi)
            effective = target_midi + shift if target_midi is not None else None

            judgement = compare(
                smoothed,
                effective,
                tolerance_cents=self.settings.display_tolerance_cents,
                confidence=reading.confidence,
            )
            if smoothed is not None:
                self.highway.add_reading(song_time, smoothed, judgement.cents)
            if self.scorer is not None and status.playing:
                self.scorer.push(song_time, judgement)
            latest = judgement

        if latest is not None:
            self.tuner.set_comparison(latest, self.settings.display_tolerance_cents)

        # Tell the singer plainly when the app has followed them, so a shifted
        # highway never looks like a bug.
        if self.octaves.offset != self._shown_octave:
            self._shown_octave = self.octaves.offset
            self.highway.state.octave_shift = self._shown_octave
            if self._shown_octave:
                direction = "down" if self._shown_octave < 0 else "up"
                count = abs(self._shown_octave) // 12
                self.status_strip.mode_label.setText(
                    f"following you {count} octave{'s' if count > 1 else ''} {direction}"
                )
                self.status_strip.mode_label.setStyleSheet(
                    f"color: {theme.ACCENT.name()};"
                )
            else:
                self.status_strip.mode_label.setText(
                    "expressive — following the original's phrasing"
                    if self.expressive else "note targets"
                )

    def _finish_scoring(self) -> None:
        if self.scorer is None or not self.song:
            return
        if len(self.scorer.attempted) < 5:
            self.scorer = None
            return
        summary = self.scorer.summary()
        save_session(
            self.song.meta.hash, self.scorer,
            speed=self.speed, transpose=self.transpose,
            vocal_gain=self.engine.mixer.get_gain("vocals"),
            mode="expressive" if self.expressive else "quantized",
        )
        parts = [f"Score {summary['score']:.0f}%  ({summary['notes_attempted']} notes)"]
        for finding in summary["findings"][:2]:
            parts.append(f"\n{finding['headline']}\n{finding['detail']}")
        self.findings_label.setText("\n".join(parts))
        self.scorer = None

    # -- dialogs ----------------------------------------------------------

    def open_setup(self) -> None:
        from .setup_dialog import SetupDialog

        dialog = SetupDialog(self.settings, self.engine, self)
        if dialog.exec():
            config.save_settings(self.settings)

    def open_range_wizard(self) -> None:
        from .range_wizard import RangeWizard

        wizard = RangeWizard(self.engine, self.settings, self)
        if wizard.exec():
            config.save_settings(self.settings)
            self._describe_song()

    def open_lyrics(self) -> None:
        from .lyrics_dialog import LyricsDialog

        if not self.song:
            QMessageBox.information(self, "No song", "Load a song first.")
            return
        dialog = LyricsDialog(self.song, self)
        dialog.exec()
        if dialog.result_lyrics is not None:
            # Re-read through the normal path so the .lrc override is what
            # actually gets used, not a copy held in the dialog.
            self._load_lyrics()

    def open_sync(self) -> None:
        from .sync_dialog import SyncDialog

        if not self.song:
            QMessageBox.information(self, "No song", "Load a song first.")
            return
        if not self.lyrics or not self.lyrics.lines:
            QMessageBox.information(
                self, "No lyrics yet",
                "Add the words first (Song → Lyrics), then sync them by ear.",
            )
            return
        dialog = SyncDialog(self.song, self.engine, self)
        dialog.exec()
        if dialog.saved:
            self._load_lyrics()

    def export_cover(self) -> None:
        from .export_dialog import ExportDialog

        if not self.song:
            QMessageBox.information(self, "No song", "Load a song first.")
            return
        ExportDialog(self.song, self).exec()

    def _check_first_run(self) -> None:
        if self.settings.headphone_warning_acknowledged:
            return
        QMessageBox.information(
            self, "Wear headphones",
            "SingCoach listens to your microphone while playing the song.\n\n"
            "On speakers, the microphone hears the backing track and the app "
            "ends up analysing the record instead of you — the feedback becomes "
            "meaningless.\n\nHeadphones are required, not recommended.",
        )
        self.settings.headphone_warning_acknowledged = True
        config.save_settings(self.settings)

    def closeEvent(self, event) -> None:
        self._finish_scoring()
        self.engine.stop()
        config.save_settings(self.settings)
        super().closeEvent(event)
