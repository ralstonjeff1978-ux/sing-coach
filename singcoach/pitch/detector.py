"""Realtime pitch detection (YIN), in plain numpy.

This runs on every microphone block, so it has a hard budget: a 2048-sample
window must be analysed in well under the 11.6 ms it takes to record the next
hop. FFT-based autocorrelation gets us there with room to spare — no numba, no
model, no GPU.

YIN in four steps (Cheveigné & Kawahara, 2002):

1. **Difference function** d(τ) — how unlike itself the signal is when shifted
   by τ samples. Computed via autocorrelation, because the naive form is O(N²).
2. **Cumulative mean normalisation** d'(τ) — divides by the running mean, which
   suppresses the zero-lag trivial minimum and makes a single absolute
   threshold work across loud and quiet input alike.
3. **Absolute threshold** — take the *first* dip below threshold, not the
   deepest. This is the step that avoids octave errors: the deepest dip is
   often at twice the true period, because a wave that repeats every N samples
   also repeats every 2N.
4. **Parabolic interpolation** — the true minimum rarely lands exactly on a
   sample, and without this the reported pitch quantises audibly at high
   frequencies.

What comes out is a pitch *and* an aperiodicity figure, which is what lets the
app distinguish "you sang a wrong note" from "that was a consonant, or a
breath, or nothing at all". Reporting a confident pitch for a breath would put
a false error on the singer's score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import F0_MAX_HZ, F0_MIN_HZ, PITCH_WINDOW, SAMPLE_RATE

#: Below this aperiodicity a frame counts as reliably pitched. YIN's own paper
#: suggests 0.1-0.15 for clean speech; singing into a headset mic is cleaner
#: than that, but reverb and consonants push it up, so we allow a little more.
DEFAULT_THRESHOLD = 0.15

#: Frames quieter than this are silence, not singing.
DEFAULT_GATE_DBFS = -48.0


@dataclass(frozen=True)
class PitchReading:
    """One frame's worth of analysis."""

    hz: float | None            # None when nothing pitched was found
    midi: float | None
    aperiodicity: float         # 0 = perfectly periodic, 1 = noise
    rms_dbfs: float
    timestamp: float = 0.0

    @property
    def voiced(self) -> bool:
        return self.hz is not None

    @property
    def confidence(self) -> float:
        """0..1, for driving UI opacity and scoring weights."""
        if self.hz is None:
            return 0.0
        return float(np.clip(1.0 - self.aperiodicity / DEFAULT_THRESHOLD, 0.0, 1.0))


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-10))


class YinDetector:
    """Stateful YIN detector. One instance per input stream."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window: int = PITCH_WINDOW,
        fmin: float = F0_MIN_HZ,
        fmax: float = F0_MAX_HZ,
        threshold: float = DEFAULT_THRESHOLD,
        gate_dbfs: float = DEFAULT_GATE_DBFS,
    ) -> None:
        self.sample_rate = sample_rate
        self.window = window
        self.threshold = threshold
        self.gate_dbfs = gate_dbfs

        # Period bounds, in samples. Guard against a window too short to
        # resolve fmin — otherwise the low end silently stops working.
        self.tau_min = max(2, int(sample_rate / fmax))
        self.tau_max = min(int(sample_rate / fmin), window // 2)
        if self.tau_max <= self.tau_min:
            raise ValueError(
                f"A {window}-sample window cannot resolve {fmin} Hz at "
                f"{sample_rate} Hz. Need at least {int(2 * sample_rate / fmin)} samples."
            )

        # Last confidently-voiced pitch, used to break octave ambiguity.
        self._last_midi: float | None = None
        self._frames_since_voiced = 0

    # -- core ---------------------------------------------------------------

    def _difference(self, x: np.ndarray) -> np.ndarray:
        """YIN's cumulative mean normalised difference function."""
        n = len(x)
        # d(tau) = sum (x[j] - x[j+tau])^2, expanded so it can use an FFT:
        #   = power_first + power_shifted - 2 * autocorr(tau)
        size = 1 << (2 * n - 1).bit_length()
        fft = np.fft.rfft(x, size)
        acf = np.fft.irfft(fft * np.conjugate(fft), size)[: self.tau_max + 1]

        cumsum = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
        taus = np.arange(self.tau_max + 1)
        power_first = cumsum[n - taus] - cumsum[0]
        power_shift = cumsum[n] - cumsum[taus]
        diff = power_first + power_shift - 2.0 * acf
        np.maximum(diff, 0.0, out=diff)

        # Cumulative mean normalisation; d'(0) is defined as 1.
        cmnd = np.ones_like(diff)
        running = np.cumsum(diff[1:])
        nonzero = running > 0
        idx = np.arange(1, len(diff))
        cmnd[1:][nonzero] = diff[1:][nonzero] * idx[nonzero] / running[nonzero]
        return cmnd

    def _pick_tau(self, cmnd: np.ndarray) -> tuple[int, float] | None:
        """First dip below threshold, else the global minimum if it is decent."""
        search = cmnd[self.tau_min : self.tau_max + 1]
        below = np.flatnonzero(search < self.threshold)

        if below.size:
            # Walk to the bottom of this dip rather than taking its first sample.
            start = int(below[0])
            i = start
            while i + 1 < len(search) and search[i + 1] < search[i]:
                i += 1
            return self.tau_min + i, float(search[i])

        # Nothing crossed the threshold. Report the best candidate anyway so
        # that callers can decide, but its aperiodicity will be high and the
        # frame will read as unvoiced.
        i = int(np.argmin(search))
        return self.tau_min + i, float(search[i])

    @staticmethod
    def _refine(cmnd: np.ndarray, tau: int) -> float:
        """Parabolic interpolation around the minimum, for sub-sample accuracy."""
        if tau <= 0 or tau >= len(cmnd) - 1:
            return float(tau)
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = 2.0 * (2.0 * b - a - c)
        if abs(denom) < 1e-12:
            return float(tau)
        return float(tau + (c - a) / denom)

    # -- public -------------------------------------------------------------

    def __call__(self, frame: np.ndarray, timestamp: float = 0.0) -> PitchReading:
        return self.process(frame, timestamp)

    def process(self, frame: np.ndarray, timestamp: float = 0.0) -> PitchReading:
        x = np.asarray(frame, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if len(x) < self.window:
            x = np.pad(x, (0, self.window - len(x)))
        elif len(x) > self.window:
            x = x[-self.window :]

        level = rms_dbfs(x)
        if level < self.gate_dbfs:
            self._frames_since_voiced += 1
            if self._frames_since_voiced > 40:      # ~0.5 s of quiet
                self._last_midi = None              # stop anchoring to stale pitch
            return PitchReading(None, None, 1.0, level, timestamp)

        # Remove DC before analysis; a bias offset skews the difference function.
        x = x - x.mean()

        cmnd = self._difference(x)
        picked = self._pick_tau(cmnd)
        if picked is None:
            self._frames_since_voiced += 1
            return PitchReading(None, None, 1.0, level, timestamp)

        tau, aperiodicity = picked
        if aperiodicity > self.threshold:
            self._frames_since_voiced += 1
            if self._frames_since_voiced > 40:
                self._last_midi = None
            return PitchReading(None, None, aperiodicity, level, timestamp)

        hz = self.sample_rate / self._refine(cmnd, tau)
        midi = hz_to_midi(hz)
        midi = self._repair_octave(midi, cmnd)

        self._last_midi = midi
        self._frames_since_voiced = 0
        return PitchReading(midi_to_hz(midi), midi, aperiodicity, level, timestamp)

    def _repair_octave(self, midi: float, cmnd: np.ndarray) -> float:
        """Prefer the octave nearest the last confident pitch.

        Even with the first-dip rule, YIN occasionally reports a harmonic. A
        voice does not jump a clean octave and back within ~12 ms, so when the
        reading is about an octave from where we just were and the alternative
        octave is also a plausible dip, the continuous reading is the right one.
        """
        if self._last_midi is None:
            return midi

        best = midi
        best_err = abs(midi - self._last_midi)
        for shift in (-12.0, 12.0):
            candidate = midi + shift
            err = abs(candidate - self._last_midi)
            if err >= best_err - 2.0:
                continue
            # Only accept if that period is genuinely periodic too.
            tau_c = self.sample_rate / midi_to_hz(candidate)
            k = int(round(tau_c))
            if self.tau_min <= k <= self.tau_max and cmnd[k] < self.threshold * 2.0:
                best, best_err = candidate, err
        return best

    def reset(self) -> None:
        self._last_midi = None
        self._frames_since_voiced = 0
