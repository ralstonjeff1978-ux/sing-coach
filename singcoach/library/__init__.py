"""Song import, caching, and metadata."""

from .cache import SongPaths, is_analysed, is_imported, is_separated, paths_for, song_hash
from .importer import SUPPORTED_SUFFIXES, UnsupportedAudio, import_song
from .song import Song, SongMeta

__all__ = [
    "Song",
    "SongMeta",
    "SongPaths",
    "SUPPORTED_SUFFIXES",
    "UnsupportedAudio",
    "import_song",
    "is_analysed",
    "is_imported",
    "is_separated",
    "paths_for",
    "song_hash",
]
