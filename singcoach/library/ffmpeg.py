"""Thin wrapper around the ffmpeg/ffprobe binaries.

ffmpeg is our decoder for every input format, which is why SingCoach needs no
codec libraries of its own. It is already on PATH on this machine; if it ever
is not, :func:`require_ffmpeg` fails with an actionable message instead of a
``FileNotFoundError`` from deep inside a subprocess call.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Windows: keep the console window from flashing on every subprocess call.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class FFmpegMissing(RuntimeError):
    pass


class FFmpegFailed(RuntimeError):
    def __init__(self, args: list[str], stderr: str) -> None:
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        super().__init__(f"ffmpeg failed: {' '.join(args[:3])} ...\n{tail}")
        self.stderr = stderr


def _find(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise FFmpegMissing(
            f"{name} was not found on PATH. Install it with "
            f"`winget install Gyan.FFmpeg` and reopen your terminal."
        )
    return exe


def require_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg, ffprobe) paths, raising if either is missing."""
    return _find("ffmpeg"), _find("ffprobe")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    if proc.returncode != 0:
        raise FFmpegFailed(args, proc.stderr)
    return proc


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    duration_s: float
    sample_rate: int
    channels: int
    codec: str
    title: str | None
    artist: str | None

    @property
    def display_name(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or "Unknown"


def probe(path: Path) -> ProbeResult:
    """Read stream metadata without decoding the file."""
    _, ffprobe = require_ffmpeg()
    proc = _run(
        [
            ffprobe, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration:format_tags=title,artist",
            "-of", "json",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise FFmpegFailed(["ffprobe"], f"{path.name} contains no audio stream.")
    stream = streams[0]
    fmt = data.get("format") or {}
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    return ProbeResult(
        duration_s=float(fmt.get("duration") or 0.0),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        codec=str(stream.get("codec_name") or "?"),
        title=tags.get("title"),
        artist=tags.get("artist"),
    )


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def decode_to_wav(src: Path, dst: Path, sample_rate: int, channels: int) -> None:
    """Decode any supported container to 32-bit float PCM WAV.

    Float output is deliberate: the separation model wants unclipped input, and
    we would rather spend disk than lose headroom on a loud master.
    """
    ffmpeg, _ = require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.wav")
    _run(
        [
            ffmpeg, "-hide_banner", "-nostdin", "-y",
            "-i", str(src),
            "-map", "0:a:0",
            "-vn", "-sn", "-dn",           # audio only: ignore video/subs/data
            "-ac", str(channels),
            "-ar", str(sample_rate),
            "-c:a", "pcm_f32le",
            str(tmp),
        ]
    )
    tmp.replace(dst)  # atomic: a half-written cache entry is never observable


# ---------------------------------------------------------------------------
# loudness
# ---------------------------------------------------------------------------

_LOUDNESS_KEYS = {
    "I": "integrated_lufs",
    "LRA": "loudness_range",
    "Peak": "true_peak_dbfs",
}


def measure_loudness(path: Path) -> dict[str, float]:
    """Measure EBU R128 integrated loudness and true peak.

    We measure rather than normalise so the cached audio stays bit-faithful to
    the source; the gain is applied at playback time instead.
    """
    ffmpeg, _ = require_ffmpeg()
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-nostdin",
            "-i", str(path),
            "-af", "ebur128=peak=true:framelog=quiet",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=_NO_WINDOW,
    )
    # ebur128 prints its summary block to stderr, e.g.  "    I:         -8.3 LUFS"
    out: dict[str, float] = {}
    for key, name in _LOUDNESS_KEYS.items():
        m = re.search(rf"^\s*{re.escape(key)}:\s*(-?\d+(?:\.\d+)?)", proc.stderr, re.M)
        if m:
            out[name] = float(m.group(1))
    return out
