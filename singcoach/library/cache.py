"""Content-addressed cache for imported songs.

Every artifact we derive from a song — decoded audio, stems, melody, lyrics,
pitch-shifted and time-stretched renders — lives in one directory keyed by a
hash of the *source file contents*. Consequences worth relying on:

* Re-importing the same file is instant, no matter where it moved to or what it
  got renamed to.
* Two copies of the same song in different folders share one analysis.
* Editing a file's tags changes the hash, so a re-tagged file re-analyses. That
  is a rare, cheap, and correct outcome.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import CACHE_DIR

_CHUNK = 1 << 20  # 1 MiB
_HASH_LEN = 16    # hex chars; 64 bits is ample for a personal library


def song_hash(path: Path) -> str:
    """Stable hash of a file's contents, plus its size as a cheap guard."""
    h = hashlib.blake2b(digest_size=16)
    h.update(str(path.stat().st_size).encode())
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()[:_HASH_LEN]


@dataclass(frozen=True)
class SongPaths:
    """Every path belonging to one cached song."""

    root: Path

    # -- stage 1: import ----------------------------------------------------
    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def source_wav(self) -> Path:
        """Decoded 44.1k stereo float32 — the input to everything else."""
        return self.root / "source.wav"

    # -- stage 2: separation ------------------------------------------------
    @property
    def vocals(self) -> Path:
        return self.root / "vocals.flac"

    @property
    def accompaniment(self) -> Path:
        return self.root / "accompaniment.flac"

    # -- stage 3+: analysis -------------------------------------------------
    @property
    def analysis(self) -> Path:
        """Melody notes, f0 contour, confidence mask, key, tempo, beats."""
        return self.root / "analysis.json"

    @property
    def lyrics(self) -> Path:
        return self.root / "lyrics.json"

    @property
    def lyrics_override(self) -> Path:
        """Hand-edited .lrc. When this exists it wins over the transcript."""
        return self.root / "lyrics.lrc"

    # -- derived renders ----------------------------------------------------
    def render(self, stem: str, semitones: int, speed: float) -> Path:
        """Path for a transposed and/or time-stretched copy of a stem.

        ``semitones=0, speed=1.0`` is the original, so callers can ask for a
        render unconditionally and get the untouched file back.
        """
        renders = self.root / "renders"
        return renders / f"{stem}_t{semitones:+d}_s{int(round(speed * 100)):03d}.flac"

    def size_on_disk(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


def paths_for(hash_: str) -> SongPaths:
    root = CACHE_DIR / hash_
    root.mkdir(parents=True, exist_ok=True)
    return SongPaths(root)


def is_imported(hash_: str) -> bool:
    p = paths_for(hash_)
    return p.meta.exists() and p.source_wav.exists()


def is_separated(hash_: str) -> bool:
    p = paths_for(hash_)
    return p.vocals.exists() and p.accompaniment.exists()


def is_analysed(hash_: str) -> bool:
    return paths_for(hash_).analysis.exists()


def list_cached() -> list[str]:
    if not CACHE_DIR.exists():
        return []
    return sorted(d.name for d in CACHE_DIR.iterdir() if d.is_dir() and (d / "meta.json").exists())


def purge(hash_: str) -> None:
    """Delete one song's cache entry. Never touches the user's source file."""
    root = CACHE_DIR / hash_
    if root.exists():
        shutil.rmtree(root)
