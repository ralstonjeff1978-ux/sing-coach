"""Orchestrates the offline stages and caches each one independently.

Stages are separately resumable on purpose. Separation is fast on the GPU but
melody extraction is not, and re-running a five-minute pYIN pass because the
app was closed would be an insult. Each stage checks for its own output first.

    import  ->  separate  ->  melody  ->  key/beats  ->  lyrics
                    |            |            |            |
              vocals.flac  analysis.json  (same file)  lyrics.json
              accomp.flac
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .. import config
from ..library.song import Song
from . import key_beats, lyrics as lyrics_mod, melody
from .separator import separate_to_files

ProgressFn = Callable[[str, float], None]


def _noop(_s: str, _f: float) -> None:
    pass


#: Determined by measurement, not by trust — see scripts/probe_nfft.py.
SEPARATOR_N_FFT = 7680

STAGES = ("separate", "melody", "key_beats", "lyrics")


@dataclass
class StageResult:
    name: str
    skipped: bool
    detail: dict


def _scoped(progress: ProgressFn, stage: str, lo: float, hi: float) -> ProgressFn:
    """Map a stage's own 0..1 progress onto its slice of the overall bar."""

    def inner(label: str, frac: float) -> None:
        progress(f"{stage}: {label}", lo + (hi - lo) * max(0.0, min(1.0, frac)))

    return inner


def analyse_song(
    song: Song,
    *,
    progress: ProgressFn = _noop,
    force: bool = False,
    prefer_gpu: bool = True,
    skip_lyrics: bool = False,
    lyrics_model: str = lyrics_mod.DEFAULT_MODEL,
    settings: "config.Settings | None" = None,
) -> list[StageResult]:
    """Run every offline stage that has not already been cached."""
    results: list[StageResult] = []
    paths = song.paths
    if settings is None:
        settings = config.load_settings()

    # -- separation ---------------------------------------------------------
    if song.is_separated and not force:
        results.append(StageResult("separate", True, {}))
        progress("separate: cached", 0.35)
    else:
        # Picks the high-quality backend if its model file is present, otherwise
        # falls back to MDX-Net. The chosen backend is recorded in the detail.
        detail = separate_to_files(
            paths.source_wav,
            paths.vocals,
            paths.accompaniment,
            settings=settings,
            prefer_gpu=prefer_gpu,
            mdx_n_fft=SEPARATOR_N_FFT,
            progress=_scoped(progress, "separate", 0.0, 0.35),
        )
        song.mark_stage("separate")
        results.append(StageResult("separate", False, detail))

    # -- melody + key/beats share one analysis.json -------------------------
    existing: dict = {}
    if paths.analysis.exists() and not force:
        try:
            existing = json.loads(paths.analysis.read_text("utf-8"))
        except json.JSONDecodeError:
            existing = {}

    if "melody" in existing:
        results.append(StageResult("melody", True, {}))
        progress("melody: cached", 0.85)
        mel_dict = existing["melody"]
    else:
        mel = melody.analyse_vocal(
            paths.vocals, progress=_scoped(progress, "melody", 0.35, 0.85)
        )
        mel_dict = mel.to_dict()
        song.mark_stage("melody")
        results.append(
            StageResult(
                "melody",
                False,
                {
                    "notes": len(mel.notes),
                    "expressiveness": mel.expressiveness,
                    "suggested_mode": mel.suggested_mode,
                    "range": [mel.range_low_midi, mel.range_high_midi],
                },
            )
        )

    if "key_beats" in existing:
        results.append(StageResult("key_beats", True, {}))
        kb_dict = existing["key_beats"]
    else:
        kb = key_beats.analyse(
            paths.accompaniment, progress=_scoped(progress, "key", 0.85, 0.99)
        )
        kb_dict = kb.to_dict()
        song.mark_stage("key_beats")
        results.append(
            StageResult(
                "key_beats",
                False,
                {"key": kb.name, "confidence": kb.key_confidence, "tempo": kb.tempo_bpm},
            )
        )

    tmp = paths.analysis.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"melody": mel_dict, "key_beats": kb_dict}), "utf-8")
    tmp.replace(paths.analysis)

    # -- lyrics -------------------------------------------------------------
    # Last because it is the slowest stage and the only optional one: a song
    # with no usable lyrics is still fully practisable.
    if skip_lyrics:
        results.append(StageResult("lyrics", True, {"reason": "skipped"}))
    elif paths.lyrics.exists() and not force:
        results.append(StageResult("lyrics", True, {}))
    else:
        # The pitch contour is what times the words: it says exactly when the
        # voice was on, which a speech model cannot reliably tell us about
        # singing. See lyrics.align_to_voice.
        import numpy as np

        onsets = [n["start"] for n in mel_dict.get("notes", [])]
        contour = np.array(
            [np.nan if v is None else v
             for v in mel_dict.get("contour", {}).get("midi", [])],
            dtype=float,
        )
        try:
            lyr = lyrics_mod.analyse_lyrics(
                paths.vocals,
                onsets,
                model_size=lyrics_model,
                lrc_override=paths.lyrics_override,
                contour=contour if contour.size else None,
                contour_hop=mel_dict.get("hop_s", 0.0),
                progress=_scoped(progress, "lyrics", 0.0, 0.99),
            )
        except (ImportError, RuntimeError, OSError) as exc:
            results.append(StageResult("lyrics", False, {"error": str(exc)[:200]}))
        else:
            lyrics_mod.save(lyr, paths.lyrics)
            if lyr.source == "transcribed" and not paths.lyrics_override.exists():
                # Ship an editable copy so corrections are possible from day one.
                lyrics_mod.write_lrc(lyr, paths.lyrics_override)
            song.mark_stage("lyrics")
            results.append(StageResult("lyrics", False, lyr.stats()))

    progress("done", 1.0)
    return results


def load_analysis(song: Song) -> dict:
    if not song.paths.analysis.exists():
        raise FileNotFoundError(
            f"{song.meta.display_name} has not been analysed yet."
        )
    return json.loads(song.paths.analysis.read_text("utf-8"))
