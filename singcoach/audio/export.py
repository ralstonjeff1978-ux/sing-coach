"""Mixing your takes with the backing track and rendering a cover.

Three things stand between a raw headset recording and something you would
actually play for someone, and all three are handled here.

**Alignment.** The take is shifted earlier by the measured round-trip latency
so your words land where you sang them. Without this, everything is
80-150 ms late and reads as sloppy timing.

**A vocal chain.** A dry close-mic vocal sounds thin and sits oddly against a
produced backing track. The default chain is deliberately conservative — a
high-pass to clear rumble, gentle compression to even out the loud and quiet
lines, a little reverb for space, and a limiter to catch peaks. It is meant to
make the take sit in the mix, not to disguise the singing.

**Loudness.** The mix is normalised so it plays back at a comparable level to
commercial music instead of arriving suspiciously quiet.

pedalboard supplies the processing (it wraps the same Rubber Band and JUCE DSP
used elsewhere in the app), and ffmpeg does the final encode.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import SAMPLE_RATE
from ..library import ffmpeg
from .recorder import Take


@dataclass
class VocalChain:
    """Processing applied to your recorded vocal.

    Defaults are chosen to be flattering but honest: they will not fix pitch,
    and they will not hide a bad take. Set ``enabled=False`` to hear exactly
    what the microphone captured.
    """

    enabled: bool = True
    highpass_hz: float = 90.0        # below a male fundamental; clears rumble
    compress_threshold_db: float = -18.0
    compress_ratio: float = 3.0
    reverb_wet: float = 0.12         # a hint of room, not a cathedral
    reverb_room: float = 0.35
    gain_db: float = 0.0

    def build(self):
        from pedalboard import Compressor, Gain, HighpassFilter, Limiter, Pedalboard, Reverb

        if not self.enabled:
            return Pedalboard([Gain(gain_db=self.gain_db)])
        return Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=self.highpass_hz),
                Compressor(
                    threshold_db=self.compress_threshold_db,
                    ratio=self.compress_ratio,
                    attack_ms=8.0,
                    release_ms=140.0,
                ),
                Reverb(room_size=self.reverb_room, wet_level=self.reverb_wet, dry_level=1.0),
                Gain(gain_db=self.gain_db),
                Limiter(threshold_db=-1.0),
            ]
        )


@dataclass
class MixSettings:
    your_vocal_db: float = 0.0
    backing_db: float = -1.0
    #: Include the original singer too — for checking a harmony line, or for a
    #: duet. Off by default: a cover usually means your voice, not both.
    original_vocal_db: float | None = None
    chain: VocalChain = None            # type: ignore[assignment]
    target_lufs: float = -14.0          # streaming-ish loudness

    def __post_init__(self) -> None:
        if self.chain is None:
            self.chain = VocalChain()


def _db(x: float) -> float:
    return float(10.0 ** (x / 20.0))


def _place(take_audio: np.ndarray, take: Take, total: int) -> np.ndarray:
    """Put a mono take onto a full-length timeline, latency-compensated."""
    out = np.zeros(total, dtype=np.float32)
    start = int(round((take.song_offset - take.alignment_offset) * SAMPLE_RATE))

    src_from = max(0, -start)                 # take began before the song did
    dst_from = max(0, start)
    n = min(len(take_audio) - src_from, total - dst_from)
    if n > 0:
        out[dst_from : dst_from + n] = take_audio[src_from : src_from + n]
    return out


def mix_take(
    takes: list[Take],
    accompaniment: np.ndarray,
    original_vocal: np.ndarray | None,
    settings: MixSettings | None = None,
) -> np.ndarray:
    """Render (2, N) float32 of your cover.

    Multiple takes are summed onto one timeline, so you can comp a chorus from
    one pass and a verse from another simply by recording them separately.
    """
    settings = settings or MixSettings()
    total = accompaniment.shape[1]

    vocal = np.zeros(total, dtype=np.float32)
    for take in takes:
        audio, sr = sf.read(str(take.file), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa

            mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)
        vocal += _place(mono, take, total)

    board = settings.chain.build()
    processed = board(vocal.reshape(1, -1), SAMPLE_RATE)[0]

    mix = accompaniment * _db(settings.backing_db)
    mix = mix + np.vstack([processed, processed]) * _db(settings.your_vocal_db)

    if settings.original_vocal_db is not None and original_vocal is not None:
        mix = mix + original_vocal[:, :total] * _db(settings.original_vocal_db)

    return _normalise(mix, settings.target_lufs)


def _normalise(mix: np.ndarray, target_lufs: float) -> np.ndarray:
    """Approximate loudness normalisation with a true-peak safety margin.

    A full EBU R128 implementation is overkill here — we are levelling a single
    mix we just built, not matching a broadcast spec — but leaving it unlevelled
    would produce a track that plays far quieter than anything around it.
    """
    rms = float(np.sqrt(np.mean(mix.astype(np.float64) ** 2)))
    if rms <= 0:
        return mix.astype(np.float32)

    # -0.691 is the R128 weighting constant; close enough for a mono-ish sum.
    approx_lufs = -0.691 + 10.0 * np.log10(rms**2)
    gain = _db(target_lufs - approx_lufs)

    peak = float(np.max(np.abs(mix)))
    if peak * gain > 0.98:
        gain = 0.98 / peak

    return np.clip(mix * gain, -1.0, 1.0).astype(np.float32)


def export(
    takes: list[Take],
    accompaniment_path: Path,
    out_path: Path,
    *,
    original_vocal_path: Path | None = None,
    settings: MixSettings | None = None,
    bitrate: str = "320k",
) -> Path:
    """Render a cover to mp3, wav, or flac — chosen by ``out_path`` suffix."""
    if not takes:
        raise ValueError("No takes selected. Record something first.")

    acc, sr = sf.read(str(accompaniment_path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Accompaniment must be {SAMPLE_RATE} Hz, got {sr}.")
    accompaniment = acc.T

    original = None
    if original_vocal_path is not None and (settings and settings.original_vocal_db is not None):
        ov, _ = sf.read(str(original_vocal_path), dtype="float32", always_2d=True)
        original = ov.T

    mix = mix_take(takes, accompaniment, original, settings)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() in (".wav", ".flac"):
        sf.write(str(out_path), mix.T, SAMPLE_RATE,
                 subtype="FLOAT" if out_path.suffix.lower() == ".wav" else "PCM_24")
        return out_path

    # Encode via ffmpeg, writing an intermediate wav it can read.
    tmp_wav = out_path.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), mix.T, SAMPLE_RATE, subtype="FLOAT")
    ffmpeg_bin, _ = ffmpeg.require_ffmpeg()
    try:
        subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", str(tmp_wav), "-c:a", "libmp3lame", "-b:a", bitrate, str(out_path)],
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        tmp_wav.unlink(missing_ok=True)
    return out_path
