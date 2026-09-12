"""Stage 1 of the pipeline: turn any audio file into a cached, known quantity.

Import is deliberately dumb and fast. It decodes, measures loudness, and writes
metadata — nothing that needs a model. The expensive stages (separation, melody,
lyrics) live in :mod:`singcoach.analysis` and run afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..config import CHANNELS, SAMPLE_RATE
from . import cache, ffmpeg
from .song import Song, SongMeta

#: Containers ffmpeg will happily open for us. This list gates the file dialog;
#: it is not a decoding constraint.
SUPPORTED_SUFFIXES = frozenset(
    {".mp3", ".mp4", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus",
     ".wma", ".aiff", ".aif", ".alac", ".mkv", ".webm", ".m4b"}
)

ProgressFn = Callable[[str, float], None]


def _noop(_stage: str, _frac: float) -> None:
    pass


class UnsupportedAudio(ValueError):
    pass


def import_song(
    src: Path,
    *,
    progress: ProgressFn = _noop,
    force: bool = False,
) -> Song:
    """Import ``src`` into the cache and return the resulting :class:`Song`.

    Idempotent: importing an already-cached file just loads it, unless ``force``.
    """
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedAudio(
            f"{src.suffix or '(no extension)'} is not a recognised audio container. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    progress("hashing", 0.0)
    hash_ = cache.song_hash(src)
    paths = cache.paths_for(hash_)

    if cache.is_imported(hash_) and not force:
        progress("done", 1.0)
        song = Song.load(hash_)
        # The file may have moved since last time; keep the pointer fresh.
        if song.meta.source_path != str(src):
            song.meta.source_path = str(src)
            song.save_meta()
        return song

    progress("probing", 0.1)
    info = ffmpeg.probe(src)
    if info.duration_s <= 0:
        raise UnsupportedAudio(f"{src.name} reports zero duration; is it a valid audio file?")

    progress("decoding", 0.2)
    ffmpeg.decode_to_wav(src, paths.source_wav, SAMPLE_RATE, CHANNELS)

    progress("measuring loudness", 0.75)
    loudness = ffmpeg.measure_loudness(paths.source_wav)

    meta = SongMeta(
        hash=hash_,
        source_path=str(src),
        display_name=info.display_name if info.title else src.stem,
        title=info.title,
        artist=info.artist,
        duration_s=info.duration_s,
        source_codec=info.codec,
        source_sample_rate=info.sample_rate,
        source_channels=info.channels,
        integrated_lufs=loudness.get("integrated_lufs"),
        true_peak_dbfs=loudness.get("true_peak_dbfs"),
        loudness_range=loudness.get("loudness_range"),
    )
    song = Song(meta=meta, paths=paths)
    song.save_meta()
    song.mark_stage("import")

    progress("done", 1.0)
    return song
