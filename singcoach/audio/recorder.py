"""Recording your own takes, for building a cover.

The subtle part is not capturing audio — it is putting it back in the right
place. What you sang at playhead 1:04 did not arrive at the soundcard at 1:04:
it went out through the buffer, out the headphones, into the air, into the mic,
and back through the input buffer. On a typical setup that round trip is
80-150 ms. Mix the raw recording against the backing track and every word lands
late, which sounds like poor timing on your part when it is nothing of the
kind.

So a take stores the playhead position it began at, and export shifts it back
by the measured round-trip latency. The correction is a measurement (see
:mod:`singcoach.audio.calibrate`), not a guess, and it is stored per take so a
later change to your audio setup cannot retroactively misalign old recordings.

Takes are written as lossless WAV. Encoding happens once, at export.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import SAMPLE_RATE


@dataclass
class Take:
    """One recorded performance."""

    name: str
    path: str
    song_hash: str
    created: str
    #: Playhead position (seconds into the song) when recording started.
    song_offset: float
    #: Round-trip latency in force at record time, in milliseconds.
    latency_ms: float
    duration: float
    #: Mix settings during recording, so you can remember what you sang against.
    vocal_gain: float = 0.0
    speed: float = 1.0
    transpose: int = 0
    #: Populated by scoring, when it ran.
    score: float | None = None
    notes: str = ""

    @property
    def file(self) -> Path:
        return Path(self.path)

    @property
    def alignment_offset(self) -> float:
        """Seconds to shift this take *earlier* to sit against the backing."""
        return self.latency_ms / 1000.0


class TakeRecorder:
    """Accumulates microphone audio during a take.

    Audio arrives on the callback thread and is appended to a plain list of
    blocks — no resizing, no file I/O, nothing that could stall the callback.
    Concatenation and writing happen on stop, off the audio thread.
    """

    def __init__(self, song_hash: str, takes_dir: Path) -> None:
        self.song_hash = song_hash
        self.takes_dir = Path(takes_dir)
        self.takes_dir.mkdir(parents=True, exist_ok=True)

        self._blocks: list[np.ndarray] = []
        self._recording = False
        self._start_offset = 0.0
        self._context: dict = {}
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def elapsed(self) -> float:
        # Cheap enough to compute without the lock; used only for a UI counter.
        return sum(len(b) for b in self._blocks) / SAMPLE_RATE

    def start(
        self,
        song_offset: float,
        latency_ms: float,
        *,
        vocal_gain: float = 0.0,
        speed: float = 1.0,
        transpose: int = 0,
    ) -> None:
        with self._lock:
            self._blocks = []
            self._start_offset = song_offset
            self._context = {
                "latency_ms": latency_ms,
                "vocal_gain": vocal_gain,
                "speed": speed,
                "transpose": transpose,
            }
            self._recording = True

    def push(self, mono: np.ndarray) -> None:
        """Called from the audio callback. Must stay trivial."""
        if self._recording:
            self._blocks.append(mono.copy())

    def stop(self, name: str | None = None) -> Take | None:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            blocks = self._blocks
            self._blocks = []

        if not blocks:
            return None

        audio = np.concatenate(blocks).astype(np.float32)
        if len(audio) < SAMPLE_RATE * 0.5:
            return None                     # discard accidental taps

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = name or f"take-{stamp}"
        path = self.takes_dir / f"{name}.wav"
        sf.write(str(path), audio, SAMPLE_RATE, subtype="FLOAT")

        take = Take(
            name=name,
            path=str(path),
            song_hash=self.song_hash,
            created=datetime.now().isoformat(timespec="seconds"),
            song_offset=round(self._start_offset, 3),
            duration=round(len(audio) / SAMPLE_RATE, 3),
            **self._context,
        )
        _append_index(self.takes_dir, take)
        return take


# ---------------------------------------------------------------------------
# take index
# ---------------------------------------------------------------------------


def _index_path(takes_dir: Path) -> Path:
    return Path(takes_dir) / "takes.json"


def _append_index(takes_dir: Path, take: Take) -> None:
    takes = list_takes(takes_dir)
    takes.append(take)
    save_index(takes_dir, takes)


def list_takes(takes_dir: Path) -> list[Take]:
    path = _index_path(takes_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        return []
    out = []
    for item in data:
        known = {k: v for k, v in item.items() if k in Take.__dataclass_fields__}
        take = Take(**known)
        if take.file.exists():          # skip entries whose audio was deleted
            out.append(take)
    return out


def save_index(takes_dir: Path, takes: list[Take]) -> None:
    path = _index_path(takes_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([asdict(t) for t in takes], indent=2), "utf-8")
    tmp.replace(path)


def delete_take(takes_dir: Path, name: str) -> bool:
    takes = list_takes(takes_dir)
    remaining = [t for t in takes if t.name != name]
    if len(remaining) == len(takes):
        return False
    for t in takes:
        if t.name == name:
            t.file.unlink(missing_ok=True)
    save_index(takes_dir, remaining)
    return True
