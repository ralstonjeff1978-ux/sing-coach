"""One-time model downloads.

This module is the *only* place SingCoach touches the network. Everything else —
separation, melody extraction, transcription, scoring — runs offline against
files already on disk.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import MODELS_DIR

ProgressFn = Callable[[str, float], None]

_UA = "SingCoach/0.1 (local practice tool)"
_CHUNK = 1 << 18  # 256 KiB


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    size_bytes: int
    description: str

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename


#: Mirror maintained by the Ultimate Vocal Remover project. MDX-Net models are
#: 4-stem-free 2-stem separators: they output the target and the residual, which
#: is exactly what we need (vocal for analysis, accompaniment for playback).
_UVR_BASE = (
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
)

MODELS: dict[str, ModelSpec] = {
    "separator": ModelSpec(
        key="separator",
        filename="UVR-MDX-NET-Voc_FT.onnx",
        url=f"{_UVR_BASE}/UVR-MDX-NET-Voc_FT.onnx",
        size_bytes=66_762_045,
        description=(
            "MDX-Net vocal separator (fine-tuned). Chosen over the instrumental "
            "variants because we need a clean *vocal* for pitch tracking, and "
            "the accompaniment we play back tolerates bleed better than the "
            "melody extractor does."
        ),
    ),
    "separator_alt": ModelSpec(
        key="separator_alt",
        filename="UVR-MDX-NET-Inst_HQ_3.onnx",
        url=f"{_UVR_BASE}/UVR-MDX-NET-Inst_HQ_3.onnx",
        size_bytes=66_762_045,
        description=(
            "Instrumental-optimised alternative. Better backing track, slightly "
            "muddier vocal. Selectable in settings when a song separates poorly."
        ),
    ),
}

MODELS["aligner"] = ModelSpec(
    key="aligner",
    filename="wav2vec2-base-960h.onnx",
    url=(
        "https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/"
        "resolve/main/onnx/model.onnx"
    ),
    size_bytes=377_898_212,
    description=(
        "wav2vec2 CTC acoustic model, used for forced alignment. Whisper is a "
        "transcriber: it recovers what was sung but only guesses when. This "
        "model emits per-frame character probabilities, which lets us align "
        "known text to the audio and get real word timings instead."
    ),
)

#: Whisper size used for lyric transcription. CTranslate2 downloads and caches
#: this itself on first use, so it is not in MODELS.
WHISPER_MODEL = "small.en"


class DownloadFailed(RuntimeError):
    pass


def is_present(key: str) -> bool:
    spec = MODELS[key]
    if not spec.path.exists():
        return False
    # A truncated download is worse than a missing one: ONNX would fail with an
    # opaque protobuf error. Treat a wrong-sized file as absent.
    actual = spec.path.stat().st_size
    return abs(actual - spec.size_bytes) <= max(4096, spec.size_bytes // 100)


def ensure(key: str, *, progress: ProgressFn | None = None) -> Path:
    """Return the local path to a model, downloading it if needed."""
    spec = MODELS[key]
    if is_present(key):
        return spec.path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = spec.path.with_suffix(spec.path.suffix + ".partial")
    req = urllib.request.Request(spec.url, headers={"User-Agent": _UA})

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or spec.size_bytes)
            done = 0
            with tmp.open("wb") as fh:
                while chunk := resp.read(_CHUNK):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(f"Downloading {spec.filename}", min(done / total, 1.0))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadFailed(
            f"Could not download {spec.filename}.\n"
            f"  URL: {spec.url}\n"
            f"  Cause: {exc}\n"
            f"You can also download it by hand and drop it in {MODELS_DIR}."
        ) from exc

    tmp.replace(spec.path)
    if progress:
        progress(f"{spec.filename} ready", 1.0)
    return spec.path


def sha256(path: Path) -> str:
    """Utility for pinning a model file once we've verified one we trust."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()
