"""Smoothing and stability analysis for the live pitch stream.

Two consumers with opposite needs share this module:

* **The display** wants a calm needle. Raw YIN output jitters a few cents frame
  to frame, and a readout that twitches is unreadable and makes a singer chase
  noise.
* **Scoring** wants the raw truth. Smoothing before scoring would flatter the
  singer by hiding exactly the instability the app exists to reveal.

So the smoothing here is offered, never imposed: the UI asks for it, the scorer
does not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .detector import PitchReading


class PitchSmoother:
    """Short median filter over recent voiced readings.

    Median rather than mean on purpose — a single octave slip would drag a mean
    by 600 cents, while a median of five ignores it entirely.
    """

    def __init__(self, size: int = 3) -> None:
        if size < 1 or size % 2 == 0:
            raise ValueError("smoothing window must be a positive odd number")
        self.size = size
        self._buf: deque[float] = deque(maxlen=size)

    def push(self, reading: PitchReading) -> float | None:
        if not reading.voiced or reading.midi is None:
            # A gap ends the run; do not smooth across silence, or the needle
            # would drift toward a note that is no longer being sung.
            self._buf.clear()
            return None
        self._buf.append(reading.midi)
        return float(np.median(self._buf))

    def reset(self) -> None:
        self._buf.clear()


@dataclass(frozen=True)
class Stability:
    """How steady the last stretch of singing was."""

    steady_cents: float          # spread of the recent pitch, in cents
    drift_cents_per_s: float     # sustained rise or fall
    vibrato_rate_hz: float | None
    vibrato_depth_cents: float | None
    frames: int

    @property
    def is_steady(self) -> bool:
        return self.steady_cents < 35.0

    @property
    def has_vibrato(self) -> bool:
        return self.vibrato_rate_hz is not None


class StabilityTracker:
    """Rolling analysis of how well a note is being held.

    This is what turns a tuner into a coach. "You're on pitch" is worth much
    less than "you're on pitch but drifting flat", or "that's vibrato, not
    wobble" — and the difference between those two is measurable, not a matter
    of opinion.
    """

    def __init__(self, hop_s: float, window_s: float = 1.0) -> None:
        self.hop_s = hop_s
        self.maxlen = max(4, int(window_s / hop_s))
        self._buf: deque[float] = deque(maxlen=self.maxlen)

    def push(self, midi: float | None) -> None:
        if midi is None:
            self._buf.clear()
        else:
            self._buf.append(midi)

    def analyse(self) -> Stability | None:
        n = len(self._buf)
        if n < max(4, int(0.25 / self.hop_s)):
            return None

        cents = np.asarray(self._buf, dtype=np.float64) * 100.0
        t = np.arange(n) * self.hop_s

        slope, intercept = np.polyfit(t, cents, 1)
        detrended = cents - (slope * t + intercept)

        spread = float(np.percentile(detrended, 95) - np.percentile(detrended, 5))
        rate, depth = self._vibrato(detrended)

        return Stability(
            steady_cents=round(spread, 1),
            drift_cents_per_s=round(float(slope), 1),
            vibrato_rate_hz=rate,
            vibrato_depth_cents=depth,
            frames=n,
        )

    def _vibrato(self, detrended: np.ndarray) -> tuple[float | None, float | None]:
        n = len(detrended)
        if n < int(0.35 / self.hop_s):
            return None, None
        spec = np.abs(np.fft.rfft(detrended * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=self.hop_s)
        band = (freqs >= 3.5) & (freqs <= 9.0)
        if not band.any() or spec[band].max() <= 0:
            return None, None
        k = np.flatnonzero(band)[np.argmax(spec[band])]
        context = (freqs >= 1.0) & (freqs <= 15.0)
        floor = float(np.median(spec[context])) if context.any() else 0.0
        if floor <= 0 or spec[k] / floor < 3.0:
            return None, None
        depth = float(np.percentile(detrended, 95) - np.percentile(detrended, 5))
        if depth < 30.0:
            return None, None
        return round(float(freqs[k]), 2), round(depth, 1)

    def reset(self) -> None:
        self._buf.clear()
