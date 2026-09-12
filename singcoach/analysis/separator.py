"""Pluggable vocal-separation backends and the logic that chooses one.

SingCoach separates the lead vocal from a song so it can (a) extract the melody
the original singer sang and (b) offer a karaoke/guide/duet mix. Two backends
implement the same tiny contract:

* :class:`~singcoach.analysis.separate.MDXSeparator` — the shipped MDX-Net ONNX
  model. Small (~66 MB), fast, always present, downloaded on first run.
* :class:`~singcoach.analysis.roformer.RoformerSeparator` — an optional, much
  stronger 2026-class **HTDemucs / BS-Roformer** ONNX model the user drops in on
  a capable (GPU) box. Large; never downloaded automatically.

:func:`build_separator` picks the best backend that is *actually available* and
falls back to MDX-Net rather than ever failing:

    backend "auto"      -> HQ model if its file is present, else MDX-Net
    backend "mdx"       -> always MDX-Net
    backend "htdemucs"  -> HQ model, but still falls back to MDX-Net if the file
                           is missing or fails to load

Every path returns a :class:`SeparatorSelection` that says which backend is
actually active, on which ONNX execution provider, and — if it fell back — why.
That string is meant to be surfaced in the UI/CLI so the choice is never silent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .. import config, modelstore
from ..config import SeparatorConfig
from .roformer import RoformerSeparator, WaveformParams
from .separate import (
    MDXParams,
    MDXSeparator,
    ProgressFn,
    SeparationError,
    _noop,
    load_mix,
    write_stem,
)

#: MDX n_fft is a property of the checkpoint, determined once by measurement.
#: See scripts/probe_nfft.py and singcoach.analysis.pipeline.SEPARATOR_N_FFT.
DEFAULT_MDX_N_FFT = 7680


@runtime_checkable
class Separator(Protocol):
    """What every separation backend must provide.

    Deliberately minimal so a new model is cheap to add: identify yourself,
    say where you are running, and turn a stereo mix into two stems that sum
    back to it.
    """

    #: Stable backend id, e.g. "mdx-net" or "htdemucs".
    backend: str
    #: The ONNX execution provider actually in use, e.g. "DmlExecutionProvider".
    provider: str

    @property
    def on_gpu(self) -> bool:
        ...

    def separate(
        self, mix: np.ndarray, *, progress: ProgressFn = ...
    ) -> tuple[np.ndarray, np.ndarray]:
        """(2, N) float32 mix -> (vocals, accompaniment), each (2, N) float32."""
        ...


@dataclass(frozen=True)
class SeparatorSelection:
    """The outcome of :func:`build_separator`: which backend won, and why."""

    separator: Separator
    backend: str          #: the backend actually constructed
    model_path: Path      #: the model file it loaded
    provider: str         #: ONNX execution provider in use
    requested: str        #: the normalised backend the config asked for
    fell_back: bool       #: True if an HQ model was wanted but MDX-Net was used
    reason: str           #: human-readable explanation, safe to show a user

    @property
    def on_gpu(self) -> bool:
        return self.provider == "DmlExecutionProvider"

    def summary(self) -> str:
        where = "GPU" if self.on_gpu else "CPU"
        return f"{self.backend} on {where} ({self.provider}) - {self.reason}"


def hq_model_path(cfg: SeparatorConfig, models_dir: Path | None = None) -> Path:
    """Where the high-quality model file is expected to live."""
    base = models_dir if models_dir is not None else config.MODELS_DIR
    return Path(base) / cfg.hq_model_file


def hq_present(cfg: SeparatorConfig, models_dir: Path | None = None) -> bool:
    """True only if the HQ file exists and is large enough to be a real model.

    A truncated or placeholder file is treated as absent, so a half-finished
    copy quietly falls back to MDX-Net instead of handing ONNX a corrupt graph.
    """
    path = hq_model_path(cfg, models_dir)
    try:
        return path.is_file() and path.stat().st_size >= cfg.hq_min_bytes
    except OSError:
        return False


def _build_hq(
    cfg: SeparatorConfig, path: Path, *, prefer_gpu: bool
) -> RoformerSeparator:
    params = WaveformParams(
        segment_samples=cfg.hq_segment_samples,  # 0 -> read from the ONNX shape
        overlap=cfg.hq_overlap,
        vocals_index=cfg.hq_vocals_index,
        num_stems=cfg.hq_num_stems,
    )
    return RoformerSeparator(
        path, params, prefer_gpu=prefer_gpu, backend_name="htdemucs"
    )


def _build_mdx(
    *, prefer_gpu: bool, n_fft: int, model_path: Path | None, progress: ProgressFn
) -> MDXSeparator:
    # Preserve the one-time-download behaviour: if the small MDX model is not on
    # disk yet, fetch it. This is the same call the pipeline has always made.
    path = model_path or modelstore.ensure("separator", progress=progress)
    return MDXSeparator(path, MDXParams(n_fft=n_fft), prefer_gpu=prefer_gpu)


def build_separator(
    *,
    settings: "config.Settings | None" = None,
    separator_config: SeparatorConfig | None = None,
    prefer_gpu: bool = True,
    mdx_n_fft: int = DEFAULT_MDX_N_FFT,
    mdx_model_path: Path | None = None,
    models_dir: Path | None = None,
    progress: ProgressFn = _noop,
) -> SeparatorSelection:
    """Choose and construct the best available separation backend.

    Never raises for a *missing HQ model*: that is the expected common case and
    falls back to MDX-Net. It only raises :class:`SeparationError` if even
    MDX-Net cannot be obtained — a genuine, unrecoverable setup problem.
    """
    cfg = separator_config
    if cfg is None:
        cfg = settings.separator if settings is not None else SeparatorConfig()
    requested = cfg.normalised_backend()

    # -- try the high-quality backend first, unless the user forced MDX -------
    if requested in ("auto", "htdemucs"):
        path = hq_model_path(cfg, models_dir)
        if hq_present(cfg, models_dir):
            try:
                sep = _build_hq(cfg, path, prefer_gpu=prefer_gpu)
            except Exception as exc:  # noqa: BLE001 - ORT raises bare Exceptions
                reason = (
                    f"high-quality model {path.name} failed to load ({exc}); "
                    "using MDX-Net instead"
                )
            else:
                return SeparatorSelection(
                    separator=sep,
                    backend=sep.backend,
                    model_path=path,
                    provider=sep.provider,
                    requested=requested,
                    fell_back=False,
                    reason=f"using high-quality model {path.name}",
                )
        else:
            reason = (
                f"high-quality model not found at {path} "
                f"(backend='{requested}'); using MDX-Net"
            )
    else:
        reason = "using MDX-Net (configured)"

    # -- MDX-Net: the always-available fallback ------------------------------
    mdx = _build_mdx(
        prefer_gpu=prefer_gpu,
        n_fft=mdx_n_fft,
        model_path=mdx_model_path,
        progress=progress,
    )
    return SeparatorSelection(
        separator=mdx,
        backend=mdx.backend,
        model_path=mdx.model_path,
        provider=mdx.provider,
        requested=requested,
        fell_back=requested in ("auto", "htdemucs"),
        reason=reason,
    )


def separate_to_files(
    source_wav: Path,
    vocals_out: Path,
    accompaniment_out: Path,
    *,
    settings: "config.Settings | None" = None,
    separator_config: SeparatorConfig | None = None,
    prefer_gpu: bool = True,
    mdx_n_fft: int = DEFAULT_MDX_N_FFT,
    progress: ProgressFn = _noop,
) -> dict[str, float | str | bool]:
    """Full separation stage: pick a backend, run it, write both stems.

    Returns a detail dict the pipeline records, including which backend ran and
    whether it fell back — so "which model made this stem" is never a mystery.
    """
    selection = build_separator(
        settings=settings,
        separator_config=separator_config,
        prefer_gpu=prefer_gpu,
        mdx_n_fft=mdx_n_fft,
        progress=lambda s, f: progress(s, f * 0.05),
    )

    mix = load_mix(source_wav)
    started = time.perf_counter()
    vocals, accompaniment = selection.separator.separate(
        mix, progress=lambda s, f: progress(s, 0.05 + f * 0.9)
    )
    elapsed = time.perf_counter() - started

    progress("writing stems", 0.95)
    write_stem(vocals_out, vocals)
    write_stem(accompaniment_out, accompaniment)
    progress("done", 1.0)

    duration = mix.shape[1] / config.SAMPLE_RATE
    return {
        "backend": selection.backend,
        "provider": selection.provider,
        "fell_back": selection.fell_back,
        "reason": selection.reason,
        "seconds": round(elapsed, 1),
        "realtime_factor": round(duration / elapsed, 2) if elapsed else 0.0,
    }
