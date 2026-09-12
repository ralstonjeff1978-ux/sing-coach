"""The :class:`Song` record: what we know about one imported track."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import PracticeToggles
from . import cache
from .cache import SongPaths


@dataclass
class SongMeta:
    """Written to ``meta.json`` at import time. Cheap to read, never derived."""

    hash: str
    source_path: str          # where the file lived when imported (may move later)
    display_name: str
    title: str | None
    artist: str | None
    duration_s: float
    source_codec: str
    source_sample_rate: int
    source_channels: int

    #: EBU R128 measurements of the source, used to level playback.
    integrated_lufs: float | None = None
    true_peak_dbfs: float | None = None
    loudness_range: float | None = None

    #: Filled in by later pipeline stages so the UI can show progress per song.
    stages_done: list[str] = field(default_factory=list)

    #: Per-song overrides. These beat the global defaults when set.
    practice_overrides: dict[str, Any] = field(default_factory=dict)
    transpose_semitones: int = 0
    speed: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SongMeta":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Song:
    meta: SongMeta
    paths: SongPaths

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, hash_: str) -> "Song":
        paths = cache.paths_for(hash_)
        if not paths.meta.exists():
            raise FileNotFoundError(f"No cached song {hash_!r}. Import it first.")
        meta = SongMeta.from_dict(json.loads(paths.meta.read_text("utf-8")))
        return cls(meta=meta, paths=paths)

    def save_meta(self) -> None:
        tmp = self.paths.meta.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta.to_dict(), indent=2), "utf-8")
        tmp.replace(self.paths.meta)

    # -- pipeline bookkeeping ----------------------------------------------

    def mark_stage(self, stage: str) -> None:
        if stage not in self.meta.stages_done:
            self.meta.stages_done.append(stage)
            self.save_meta()

    @property
    def is_separated(self) -> bool:
        return self.paths.vocals.exists() and self.paths.accompaniment.exists()

    @property
    def is_analysed(self) -> bool:
        return self.paths.analysis.exists()

    @property
    def has_lyrics(self) -> bool:
        return self.paths.lyrics_override.exists() or self.paths.lyrics.exists()

    @property
    def is_ready_to_practice(self) -> bool:
        """Separation + melody is the minimum bar. Lyrics are a bonus."""
        return self.is_separated and self.is_analysed

    # -- settings resolution ------------------------------------------------

    def practice_toggles(self, defaults: PracticeToggles) -> PracticeToggles:
        """Per-song overrides layered on top of the global defaults."""
        merged = asdict(defaults)
        merged.update(
            {
                k: v
                for k, v in self.meta.practice_overrides.items()
                if k in PracticeToggles.__dataclass_fields__
            }
        )
        return PracticeToggles(**merged)

    def playback_gain(self, target_lufs: float) -> float:
        """Linear gain that brings this song to the target loudness.

        Clamped so a very quiet source cannot be boosted into clipping.
        """
        if self.meta.integrated_lufs is None:
            return 1.0
        gain_db = target_lufs - self.meta.integrated_lufs
        peak = self.meta.true_peak_dbfs
        if peak is not None:
            # Leave 1 dB of headroom below full scale.
            gain_db = min(gain_db, -1.0 - peak)
        return float(10.0 ** (gain_db / 20.0))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Song {self.meta.hash} {self.meta.display_name!r} {self.meta.duration_s:.0f}s>"
