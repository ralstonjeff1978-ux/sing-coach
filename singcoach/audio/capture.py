"""Microphone capture: a lock-free ring buffer between two threads.

The audio callback thread writes; the pitch worker reads. They must never block
each other — a callback that waits on a lock produces a dropout, and dropouts
are both audible and destructive to pitch tracking.

The design is a single-producer/single-consumer ring buffer with monotonically
increasing counters. The producer only ever advances ``_written``; the consumer
only ever advances ``_read``. Neither modifies the other's counter, so no lock
is required for correctness on CPython, where an attribute store is atomic.

Overrun is handled by dropping the oldest audio rather than blocking the
producer. If the analysis thread falls behind, losing old microphone samples is
much better than glitching playback — and the consumer is told it happened
rather than silently receiving a discontinuity.
"""

from __future__ import annotations

import numpy as np

from ..config import PITCH_HOP, PITCH_WINDOW, SAMPLE_RATE


class RingBuffer:
    """Mono float32 ring buffer sized in samples."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._written = 0      # total samples ever written
        self._read = 0         # total samples ever consumed
        self.overruns = 0

    # -- producer side (audio callback) -------------------------------------

    def write(self, data: np.ndarray) -> None:
        """Append samples. Never blocks; drops oldest data if full."""
        n = len(data)
        if n == 0:
            return
        if n > self.capacity:
            # A single block bigger than the whole buffer: samples are lost no
            # matter what we do, so keep the newest and record the loss rather
            # than dropping it silently.
            data = data[-self.capacity :]
            n = len(data)
            self.overruns += 1

        start = self._written % self.capacity
        end = start + n
        if end <= self.capacity:
            self._buf[start:end] = data
        else:
            split = self.capacity - start
            self._buf[start:] = data[:split]
            self._buf[: end - self.capacity] = data[split:]

        self._written += n

        # If the writer has lapped the reader, the oldest unread samples are
        # gone. Move the read cursor up and record it.
        behind = self._written - self._read
        if behind > self.capacity:
            self._read = self._written - self.capacity
            self.overruns += 1

    # -- consumer side (pitch worker) ---------------------------------------

    @property
    def available(self) -> int:
        return self._written - self._read

    @property
    def total_written(self) -> int:
        """Monotonic count of samples ever captured.

        This is the clock everything downstream is pegged to. It advances in
        lockstep with the output stream because both are serviced by the same
        duplex callback, which is what makes a constant offset between
        microphone samples and song position valid.
        """
        return self._written

    def read_window(self, window: int, hop: int) -> tuple[np.ndarray, int] | None:
        """Take one analysis window, advancing by ``hop``.

        Returns (samples, index_of_first_sample) or None if not enough audio
        has arrived. The absolute index is what lets the caller convert to a
        wall-clock timestamp without a separate clock that could drift.
        """
        if self.available < window:
            return None

        start = self._read
        offset = start % self.capacity
        end = offset + window
        if end <= self.capacity:
            out = self._buf[offset:end].copy()
        else:
            split = self.capacity - offset
            out = np.concatenate([self._buf[offset:], self._buf[: end - self.capacity]])

        self._read += hop
        return out, start

    def clear(self) -> None:
        self._read = self._written


class MicCapture:
    """Microphone side of the duplex stream, plus level metering."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window: int = PITCH_WINDOW,
        hop: int = PITCH_HOP,
        seconds: float = 4.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.window = window
        self.hop = hop
        self.buffer = RingBuffer(int(seconds * sample_rate))
        self._peak = 0.0
        self._clip_frames = 0

    def push(self, indata: np.ndarray) -> None:
        """Called from the audio callback with (frames, channels) input."""
        mono = indata[:, 0] if indata.ndim > 1 else indata
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        self._peak = max(self._peak * 0.92, peak)   # fast attack, slow decay
        if peak >= 0.999:
            self._clip_frames += 1
        self.buffer.write(np.ascontiguousarray(mono, dtype=np.float32))

    @property
    def total_written(self) -> int:
        return self.buffer.total_written

    def frames(self):
        """Yield (samples, centre_sample_index) for every complete window.

        A raw sample index rather than a timestamp, because the caller needs to
        map it onto song position and doing that arithmetic in seconds would
        introduce rounding for no reason.

        The index refers to the *centre* of the window: that is the moment the
        pitch estimate actually describes. Using the start would report every
        note about 23 ms early.
        """
        while (item := self.buffer.read_window(self.window, self.hop)) is not None:
            samples, index = item
            yield samples, index + self.window / 2

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def peak_dbfs(self) -> float:
        return 20.0 * np.log10(max(self._peak, 1e-10))

    @property
    def clipping(self) -> bool:
        return self._clip_frames > 0

    def reset_clip(self) -> None:
        self._clip_frames = 0

    @property
    def overruns(self) -> int:
        return self.buffer.overruns
