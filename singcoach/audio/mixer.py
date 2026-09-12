"""Stem mixing — the feature the whole app was asked for.

Because separation gave us the vocal and the backing as independent signals,
"remove the singer" and "sing along with the singer" are the same control at
two positions:

    vocal 0.00  karaoke — the lead is gone, you are the lead
    vocal 0.20  guide   — quiet enough to lead, loud enough not to get lost
    vocal 1.00  duet    — the original is there; sing a harmony against it

Everything here is called from the audio callback, so the rules are strict: no
allocation, no locks, no file I/O, no Python-level surprises. Stems are decoded
to memory up front — a five-minute stereo song is about 100 MB as float32, and
with 64 GB of RAM there is no reason to stream from disk and risk a dropout.

Gains are ramped rather than applied instantly. Jumping a gain between callback
blocks puts a step discontinuity in the waveform, which is audible as a click.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import SAMPLE_RATE

#: Seconds over which a gain change is applied. Long enough to be inaudible,
#: short enough that the slider still feels immediate.
GAIN_RAMP_S = 0.05


@dataclass
class StemSet:
    """The audio for one (transposition, speed) variant of a song."""

    vocals: np.ndarray          # (2, N) float32
    accompaniment: np.ndarray   # (2, N) float32

    def __post_init__(self) -> None:
        n = min(self.vocals.shape[1], self.accompaniment.shape[1])
        self.vocals = np.ascontiguousarray(self.vocals[:, :n], dtype=np.float32)
        self.accompaniment = np.ascontiguousarray(self.accompaniment[:, :n], dtype=np.float32)

    @property
    def frames(self) -> int:
        return self.vocals.shape[1]

    @property
    def duration(self) -> float:
        return self.frames / SAMPLE_RATE

    @classmethod
    def load(cls, vocals: Path, accompaniment: Path) -> "StemSet":
        v, sr_v = sf.read(str(vocals), dtype="float32", always_2d=True)
        a, sr_a = sf.read(str(accompaniment), dtype="float32", always_2d=True)
        if sr_v != SAMPLE_RATE or sr_a != SAMPLE_RATE:
            raise ValueError(
                f"Stems must be {SAMPLE_RATE} Hz (got {sr_v}/{sr_a}). "
                "Re-run analysis for this song."
            )
        return cls(vocals=v.T, accompaniment=a.T)


class Mixer:
    """Sample-accurate playhead plus per-stem gain, safe to drive from the UI."""

    def __init__(self, stems: StemSet | None = None) -> None:
        self._stems = stems
        self._position = 0
        self._playing = False

        # Target gains are set by the UI thread; current gains are what the
        # audio thread is actually applying, ramping toward the target.
        self._target = {"vocals": 0.0, "accompaniment": 1.0, "master": 1.0}
        self._current = dict(self._target)
        self._ramp = 1.0 / max(1, int(GAIN_RAMP_S * SAMPLE_RATE))

        # Loop region in frames; None means play through.
        self._loop: tuple[int, int] | None = None

        # Reference tone, used by the pitch-match warm-up. Generated here
        # rather than on a second output stream so it shares the one clock the
        # microphone is already aligned to.
        self._tone_hz = 0.0
        self._tone_amp = 0.0
        self._tone_phase = 0.0

        # Guards only the swap of whole stem objects, never per-block reads.
        self._lock = threading.Lock()

    # -- loading ------------------------------------------------------------

    def set_stems(self, stems: StemSet, *, keep_position: bool = True) -> None:
        """Swap in a different variant (transposed or time-stretched).

        Position is preserved proportionally so switching speed mid-phrase
        keeps you in the same musical place rather than jumping.
        """
        with self._lock:
            fraction = (
                self._position / self._stems.frames
                if keep_position and self._stems and self._stems.frames
                else 0.0
            )
            self._stems = stems
            self._position = int(np.clip(fraction * stems.frames, 0, stems.frames - 1))

    @property
    def stems(self) -> StemSet | None:
        return self._stems

    @property
    def loaded(self) -> bool:
        return self._stems is not None

    # -- transport ----------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        self._playing = True

    def pause(self) -> None:
        self._playing = False

    def toggle(self) -> None:
        self._playing = not self._playing

    @property
    def position_frames(self) -> int:
        return self._position

    @property
    def position(self) -> float:
        """Playhead in seconds."""
        return self._position / SAMPLE_RATE

    def seek(self, seconds: float) -> None:
        if not self._stems:
            return
        self._position = int(np.clip(seconds * SAMPLE_RATE, 0, self._stems.frames - 1))

    @property
    def duration(self) -> float:
        return self._stems.duration if self._stems else 0.0

    # -- loop ---------------------------------------------------------------

    def set_loop(self, start: float | None, end: float | None) -> None:
        if start is None or end is None or end <= start:
            self._loop = None
            return
        self._loop = (int(start * SAMPLE_RATE), int(end * SAMPLE_RATE))

    @property
    def loop(self) -> tuple[float, float] | None:
        if self._loop is None:
            return None
        a, b = self._loop
        return a / SAMPLE_RATE, b / SAMPLE_RATE

    # -- gains --------------------------------------------------------------

    def set_gain(self, stem: str, value: float) -> None:
        if stem not in self._target:
            raise KeyError(f"unknown stem {stem!r}")
        self._target[stem] = float(np.clip(value, 0.0, 2.0))

    def get_gain(self, stem: str) -> float:
        return self._target[stem]

    def _ramped(self, name: str, frames: int, out: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Per-sample gain curve moving current toward target.

        Writes into a caller-supplied buffer using a pre-built index ramp.
        ``np.arange`` would allocate, and allocating on the audio thread is how
        you get a dropout under memory pressure.
        """
        cur, tgt = self._current[name], self._target[name]
        if cur == tgt:
            out[:] = cur
            return out
        step = self._ramp * (1 if tgt > cur else -1)
        np.multiply(idx[:frames], step, out=out)
        out += cur
        if tgt > cur:
            np.minimum(out, tgt, out=out)
        else:
            np.maximum(out, tgt, out=out)
        self._current[name] = float(out[-1])
        return out

    # -- the audio callback's entry point -----------------------------------

    def set_tone(self, hz: float, amplitude: float = 0.18) -> None:
        """Sound a reference pitch, or silence it with ``hz=0``."""
        self._tone_hz = max(0.0, float(hz))
        self._tone_amp = float(np.clip(amplitude, 0.0, 0.5))
        if not self._tone_hz:
            self._tone_phase = 0.0

    @property
    def tone_hz(self) -> float:
        return self._tone_hz

    def _render_tone(self, frames: int, out: np.ndarray) -> None:
        """Add the reference tone. Phase is carried between blocks so the
        waveform stays continuous — restarting it each block would click."""
        step = 2.0 * np.pi * self._tone_hz / SAMPLE_RATE
        phase = self._tone_phase + step * np.arange(frames)
        # Fundamental plus a soft octave: easier to pitch-match than a bare
        # sine, which many singers find hard to place.
        tone = (np.sin(phase) + 0.28 * np.sin(2.0 * phase)) * self._tone_amp
        out[:, 0] += tone
        out[:, 1] += tone
        self._tone_phase = float((self._tone_phase + step * frames) % (2.0 * np.pi))

    def read(self, frames: int, out: np.ndarray, scratch: dict) -> int:
        """Fill ``out`` ((frames, 2) float32) with the next block.

        Returns the frame index this block started at, so the caller can map
        audio time to analysis time without asking the mixer again (which would
        race with the position advancing).
        """
        out[:] = 0.0
        stems = self._stems

        if self._tone_hz > 0.0:
            self._render_tone(frames, out[:frames])

        if stems is None or not self._playing:
            return self._position

        start = self._position
        end = start + frames

        if self._loop is not None:
            loop_a, loop_b = self._loop
            if start >= loop_b:
                start = self._position = loop_a
                end = start + frames

        available = min(end, stems.frames) - start
        if available <= 0:
            self._playing = False
            return start

        idx = scratch["idx"]
        gv = self._ramped("vocals", available, scratch["gv"][:available], idx)
        ga = self._ramped("accompaniment", available, scratch["ga"][:available], idx)
        gm = self._ramped("master", available, scratch["gm"][:available], idx)

        block = out[:available]
        block += (stems.vocals[:, start : start + available] * gv).T
        block += (stems.accompaniment[:, start : start + available] * ga).T
        block *= gm[:, None]

        # Guard the output. A user pushing both stems to unity on a loud master
        # can exceed full scale, and hard clipping on headphones is unpleasant.
        np.clip(block, -1.0, 1.0, out=block)

        self._position = start + available

        if self._loop is not None and self._position >= self._loop[1]:
            self._position = self._loop[0]
        elif self._position >= stems.frames:
            self._playing = False

        return start

    @staticmethod
    def make_scratch(max_frames: int) -> dict:
        """Pre-allocated working buffers, so the callback never allocates."""
        return {
            "gv": np.zeros(max_frames, dtype=np.float32),
            "ga": np.zeros(max_frames, dtype=np.float32),
            "gm": np.zeros(max_frames, dtype=np.float32),
            # Pre-built 0,1,2,... so gain ramps need no allocation per block.
            "idx": np.arange(max_frames, dtype=np.float32),
        }
