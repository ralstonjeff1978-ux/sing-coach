"""The duplex audio engine: one stream that plays and records at once.

Everything realtime meets here. A single `sounddevice` duplex stream calls
:meth:`AudioEngine._callback` on a high-priority thread; that callback pulls a
block from the :class:`~singcoach.audio.mixer.Mixer`, hands the microphone
input to :class:`~singcoach.audio.capture.MicCapture` and, if a take is
running, to the recorder.

The callback obeys three rules without exception:

1. **No allocation.** Buffers are pre-allocated in :meth:`start`.
2. **No locks.** Cross-thread state passes through counters and plain
   attribute stores, which are atomic on CPython.
3. **No I/O.** Nothing touches the disk, the network, or the UI.

Breaking any of them produces dropouts, which are audible as clicks and, worse,
corrupt the pitch analysis that the coaching depends on.

Analysis runs on a separate worker thread that drains the microphone ring
buffer. It is deliberately allowed to fall behind — losing old microphone
samples is far better than glitching playback, and the ring buffer reports when
it happens rather than silently skipping.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import PITCH_HOP, SAMPLE_RATE, Settings
from ..pitch.detector import PitchReading, YinDetector
from ..pitch.smoothing import PitchSmoother, StabilityTracker
from .capture import MicCapture
from .mixer import Mixer, StemSet
from .recorder import TakeRecorder


@dataclass
class EngineStatus:
    playing: bool
    position: float
    duration: float
    input_peak_dbfs: float
    clipping: bool
    overruns: int
    underruns: int
    recording: bool
    record_elapsed: float


class AudioEngine:
    """Owns the audio device and the analysis worker."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mixer = Mixer()
        self.capture = MicCapture()
        self.recorder: TakeRecorder | None = None

        self.detector = YinDetector(gate_dbfs=settings.noise_gate_dbfs)
        self.smoother = PitchSmoother(size=3)
        self.stability = StabilityTracker(hop_s=PITCH_HOP / SAMPLE_RATE)

        self._stream = None
        self._scratch: dict | None = None
        self._analysis_thread: threading.Thread | None = None
        self._running = threading.Event()
        self.underruns = 0

        #: Most recent readings, newest last. Read by the UI each frame.
        self.readings: deque[tuple[float, PitchReading, float | None]] = deque(maxlen=2048)
        #: Called with (audio_time, reading, smoothed_midi) on the worker thread.
        self.on_pitch: Callable[[float, PitchReading, float | None], None] | None = None

        # Constant offset between the microphone sample counter and the song
        # playhead. Both are advanced by the same duplex callback, so once
        # established the difference holds until a seek changes it.
        #
        #     song_frame = mic_sample_index + _sync_offset
        #
        # Without this, readings drained together all take the playhead's
        # value at drain time, which stacks a whole burst onto one instant and
        # destroys both the trace and the timing measurements.
        self._sync_offset = 0
        self._last_playback: np.ndarray | None = None

    # -- clocks -------------------------------------------------------------

    @property
    def output_latency(self) -> float:
        """Seconds between the mixer producing a sample and you hearing it.

        Reported by PortAudio for the open stream, so it reflects the real
        device and buffer size rather than an assumption.

        This matters for the visuals. The playhead says where the mixer has got
        to, but that audio is still sitting in the buffer — so drawing the
        highway and lighting up lyrics at the raw playhead runs the picture
        *ahead* of the sound by exactly this much. Subtracting it puts what you
        see in step with what you hear.
        """
        if self._stream is None:
            return 0.0
        latency = getattr(self._stream, "latency", None)
        if isinstance(latency, (tuple, list)) and len(latency) >= 2:
            return float(latency[1])
        return float(latency) if isinstance(latency, (int, float)) else 0.0

    @property
    def heard_position(self) -> float:
        """The song position the listener is hearing at this instant.

        The driver's reported output latency plus the user's own correction —
        see ``Settings.visual_offset_ms`` for why a manual term is warranted.
        """
        return max(
            0.0,
            self.mixer.position
            - self.output_latency
            + self.settings.visual_offset_ms / 1000.0,
        )

    # -- devices ------------------------------------------------------------

    @staticmethod
    def devices() -> dict[str, list[dict]]:
        import sounddevice as sd

        inputs, outputs = [], []
        for i, dev in enumerate(sd.query_devices()):
            entry = {
                "index": i,
                "name": dev["name"],
                "channels_in": dev["max_input_channels"],
                "channels_out": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
                "hostapi": sd.query_hostapis(dev["hostapi"])["name"],
            }
            if dev["max_input_channels"] > 0:
                inputs.append(entry)
            if dev["max_output_channels"] > 0:
                outputs.append(entry)
        return {"input": inputs, "output": outputs}

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return

        block = self.settings.block_size
        self._scratch = Mixer.make_scratch(block * 4)
        self._out_buf = np.zeros((block * 4, 2), dtype=np.float32)

        self._stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=block,
            channels=(1, 2),                       # mono in, stereo out
            dtype="float32",
            device=(self.settings.input_device, self.settings.output_device),
            callback=self._callback,
            latency="low",
        )
        self._stream.start()

        self._running.set()
        self._analysis_thread = threading.Thread(
            target=self._analyse_loop, name="singcoach-pitch", daemon=True
        )
        self._analysis_thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._analysis_thread is not None:
            self._analysis_thread.join(timeout=1.0)
            self._analysis_thread = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "AudioEngine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- the audio callback -------------------------------------------------

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            # Underflow/overflow flags from PortAudio. Count them; never print
            # from the audio thread.
            self.underruns += 1

        buf = self._out_buf[:frames]
        song_frame = self.mixer.read(frames, buf, self._scratch)
        outdata[:] = buf

        # Peg the two clocks together before the microphone counter advances.
        self._sync_offset = song_frame - self.capture.total_written

        self.capture.push(indata)
        if self.recorder is not None and self.recorder.recording:
            self.recorder.push(indata[:, 0])

        self._last_playback = buf[:, 0]

    # -- analysis worker ----------------------------------------------------

    def _analyse_loop(self) -> None:
        hop_s = PITCH_HOP / SAMPLE_RATE
        while self._running.is_set():
            produced = False
            for samples, mic_index in self.capture.frames():
                produced = True
                # Each reading carries the song position it actually belongs
                # to, computed from its own sample index — not from wherever
                # the playhead happens to be when the UI gets round to it.
                song_time = self.song_time_for_mic_index(mic_index)
                reading = self.detector.process(samples, song_time)
                smoothed = self.smoother.push(reading)
                self.stability.push(smoothed)
                self.readings.append((song_time, reading, smoothed))
                if self.on_pitch is not None:
                    self.on_pitch(song_time, reading, smoothed)
            if not produced:
                # Nothing ready; sleep less than one hop so we never add
                # meaningful latency of our own.
                self._running.wait(hop_s / 2)

    # -- song time ----------------------------------------------------------

    def song_time_for_mic_index(self, mic_index: float) -> float:
        """Which moment of the song a microphone sample is a response to.

        Two corrections, and both are needed:

        1. ``_sync_offset`` maps the microphone sample counter onto the song
           playhead, so a reading is anchored to when it was captured rather
           than when it was processed.
        2. The round-trip latency is subtracted, because what the singer was
           responding to left the speakers one round trip before the microphone
           heard the reply.
        """
        song_frame = mic_index + self._sync_offset
        return song_frame / SAMPLE_RATE - self.settings.latency_ms() / 1000.0

    # -- takes --------------------------------------------------------------

    def arm_recorder(self, song_hash: str, takes_dir) -> None:
        self.recorder = TakeRecorder(song_hash, takes_dir)

    def start_take(self, **context):
        if self.recorder is None:
            raise RuntimeError("Recorder not armed for a song.")
        self.recorder.start(
            song_offset=self.mixer.position,
            latency_ms=self.settings.latency_ms(),
            vocal_gain=self.mixer.get_gain("vocals"),
            **context,
        )

    def stop_take(self, name: str | None = None):
        return self.recorder.stop(name) if self.recorder is not None else None

    # -- calibration --------------------------------------------------------

    def play_and_record(self, signal: np.ndarray, listen_frames: int) -> np.ndarray:
        """Emit ``signal`` and capture what comes back. Used by calibration.

        Runs its own short-lived stream rather than borrowing the main one, so
        calibration cannot be contaminated by music still playing.
        """
        import sounddevice as sd

        padded = np.zeros(listen_frames, dtype=np.float32)
        padded[: len(signal)] = signal
        recorded = sd.playrec(
            np.column_stack([padded, padded]),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=(self.settings.input_device, self.settings.output_device),
        )
        sd.wait()
        return recorded[:, 0]

    # -- status -------------------------------------------------------------

    def status(self) -> EngineStatus:
        return EngineStatus(
            playing=self.mixer.playing,
            position=self.mixer.position,
            duration=self.mixer.duration,
            input_peak_dbfs=self.capture.peak_dbfs,
            clipping=self.capture.clipping,
            overruns=self.capture.overruns,
            underruns=self.underruns,
            recording=bool(self.recorder and self.recorder.recording),
            record_elapsed=self.recorder.elapsed if self.recorder else 0.0,
        )

    def load_song(self, stems: StemSet) -> None:
        self.mixer.set_stems(stems, keep_position=False)
        self.detector.reset()
        self.smoother.reset()
        self.stability.reset()
        self.readings.clear()
