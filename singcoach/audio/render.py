"""Offline transposition and time-stretching.

Two practice features need altered audio: dropping a song into your range, and
slowing a hard phrase down. Both are done **offline and cached**, never live.

That is a deliberate choice. A realtime phase vocoder in the audio callback
would risk dropouts, and its artefacts smear exactly the transients a singer
uses to place a note. Pre-rendering costs a few seconds once, then switching
speed is instantaneous and the audio is as clean as the algorithm can make it.

The corresponding melody data needs no re-analysis: transposing by *n*
semitones adds *n* to every target note, and stretching by factor *f* divides
every timestamp by *f*. Both are exact, so the coaching stays perfectly aligned
with audio that has been altered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from ..config import SAMPLE_RATE
from ..library.cache import SongPaths

ProgressFn = Callable[[str, float], None]


def _noop(_s: str, _f: float) -> None:
    pass


def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Shift pitch, preserving duration. ``audio`` is (2, N) float32."""
    if abs(semitones) < 1e-6:
        return audio
    from pedalboard import Pedalboard, PitchShift

    board = Pedalboard([PitchShift(semitones=float(semitones))])
    return board(audio.astype(np.float32), SAMPLE_RATE)


# --- formant correction ----------------------------------------------------

_FORMANT_NFFT = 2048
_FORMANT_HOP = 512
#: Cepstral cutoff separating the slow spectral envelope (vocal tract shape)
#: from the fine harmonic structure (pitch). ~40 quefrency bins at this FFT
#: size sits comfortably between the two for a human voice.
_CEPSTRUM_CUTOFF = 30
#: Correction is clamped: an unbounded gain would amplify noise in near-silent
#: bands into audible artefacts.
_MAX_CORRECTION_DB = 30.0


def _spectral_envelope(spec: np.ndarray) -> np.ndarray:
    """Smooth magnitude envelope via cepstral liftering, per frame.

    The envelope is what the vocal tract imposes; the ripple on top of it is
    the pitch. Separating them in the cepstral domain is the standard way to
    change one without the other.
    """
    log_mag = np.log(np.maximum(np.abs(spec), 1e-10))
    cepstrum = np.fft.irfft(log_mag, axis=0)
    cepstrum[_CEPSTRUM_CUTOFF:-_CEPSTRUM_CUTOFF] = 0.0
    return np.exp(np.real(np.fft.rfft(cepstrum, axis=0)))


def pitch_shift_formant_safe(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch-shift a voice without turning the singer into a different person.

    Plain pitch shifting moves the vocal-tract resonances along with the
    fundamental. But a person singing lower does not grow a longer throat —
    their formants stay where they are. Shifting both together is what produces
    the "monster" voice downward and the "chipmunk" upward: in tune, obviously
    unnatural, and immediately recognisable as processed.

    So we shift as usual, then measure the spectral envelope before and after
    and put the original envelope back. The harmonics keep their new spacing;
    the resonances return to where a real voice would have them.
    """
    if abs(semitones) < 1e-6:
        return audio

    ratio = 2.0 ** (semitones / 12.0)
    shifted = pitch_shift(audio, semitones)
    out = np.empty_like(shifted)
    limit = 10.0 ** (_MAX_CORRECTION_DB / 20.0)
    bins = np.arange(_FORMANT_NFFT // 2 + 1)

    for channel in range(shifted.shape[0]):
        shifted_spec = librosa.stft(
            np.ascontiguousarray(shifted[channel]),
            n_fft=_FORMANT_NFFT,
            hop_length=_FORMANT_HOP,
        )
        have = _spectral_envelope(shifted_spec)

        # The shifter scaled the envelope in frequency by exactly ``ratio``, so
        # the envelope now sitting at bin k is what used to sit at k/ratio.
        # The original envelope at bin k is therefore what now sits at k*ratio.
        # Reading the target off the *shifted* signal this way needs no
        # cross-comparison with the original — which is what the first attempt
        # did, and why it only recovered part of the shift: the shifter gives
        # no guarantee that its output frames line up in time with its input,
        # so that ratio was contaminated by misalignment.
        source_bins = np.clip(bins * ratio, 0, len(bins) - 1)
        lower = np.floor(source_bins).astype(int)
        upper = np.minimum(lower + 1, len(bins) - 1)
        frac = (source_bins - lower)[:, None]
        wanted = have[lower] * (1.0 - frac) + have[upper] * frac

        gain = np.clip(wanted / np.maximum(have, 1e-10), 1.0 / limit, limit)

        corrected = librosa.istft(
            shifted_spec * gain,
            hop_length=_FORMANT_HOP,
            n_fft=_FORMANT_NFFT,
            length=shifted.shape[1],
        )
        out[channel] = corrected

    # Preserve the original level; the correction changes overall energy a little.
    original_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    new_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if new_rms > 1e-9:
        out *= original_rms / new_rms
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def time_stretch(audio: np.ndarray, speed: float) -> np.ndarray:
    """Change tempo without changing pitch. ``speed`` < 1 is slower."""
    if abs(speed - 1.0) < 1e-6:
        return audio
    import pedalboard

    # pedalboard exposes Rubber Band's stretcher directly in recent versions;
    # fall back to the resample-plus-shift identity where it is unavailable.
    if hasattr(pedalboard, "time_stretch"):
        return pedalboard.time_stretch(
            audio.astype(np.float32), SAMPLE_RATE, stretch_factor=float(speed)
        )

    # Fallback: resampling changes both rate and pitch; shifting back by the
    # equivalent number of semitones restores the original pitch.
    import librosa

    resampled = librosa.resample(
        audio.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE * speed)
    )
    semitones = 12.0 * np.log2(1.0 / speed)
    return pitch_shift(resampled, semitones)


def render_variant(
    paths: SongPaths,
    stem: str,
    semitones: int,
    speed: float,
    *,
    progress: ProgressFn = _noop,
) -> Path:
    """Produce (and cache) one stem at a given transposition and speed.

    ``semitones=0, speed=1.0`` returns the original file untouched, so callers
    can request a variant unconditionally without special-casing the default.
    """
    source = paths.vocals if stem == "vocals" else paths.accompaniment
    if semitones == 0 and abs(speed - 1.0) < 1e-6:
        return source

    target = paths.render(stem, semitones, speed)
    if target.exists():
        return target

    progress(f"rendering {stem}", 0.0)
    audio, sr = sf.read(str(source), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}.")
    data = audio.T

    # Stretch first, then shift. Doing it the other way round makes the shifter
    # work on material whose length it will then change, which compounds the
    # artefacts of both stages.
    if abs(speed - 1.0) > 1e-6:
        progress(f"stretching {stem}", 0.3)
        data = time_stretch(data, speed)
    if semitones:
        progress(f"transposing {stem}", 0.6)
        # Formant correction matters on the vocal and not on the backing: a
        # guitar transposed down really is a bigger guitar, but a singer is
        # not a bigger person. Skipping it on the accompaniment also saves the
        # STFT round trip on the longer of the two stems.
        data = (
            pitch_shift_formant_safe(data, semitones)
            if stem == "vocals"
            else pitch_shift(data, semitones)
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial.flac")
    sf.write(str(tmp), data.T, SAMPLE_RATE, subtype="PCM_24", format="FLAC")
    tmp.replace(target)
    progress(f"rendered {stem}", 1.0)
    return target


def render_both(
    paths: SongPaths,
    semitones: int,
    speed: float,
    *,
    progress: ProgressFn = _noop,
) -> tuple[Path, Path]:
    vocals = render_variant(paths, "vocals", semitones, speed,
                            progress=lambda s, f: progress(s, f * 0.5))
    accomp = render_variant(paths, "accompaniment", semitones, speed,
                            progress=lambda s, f: progress(s, 0.5 + f * 0.5))
    return vocals, accomp


# ---------------------------------------------------------------------------
# matching transformations for the analysis data
# ---------------------------------------------------------------------------


def transform_notes(notes: list[dict], semitones: int, speed: float) -> list[dict]:
    """Apply the same transformation to the target melody.

    Exact by construction — no re-analysis, and therefore no risk that the
    coaching drifts out of step with altered audio.
    """
    if semitones == 0 and abs(speed - 1.0) < 1e-6:
        return notes
    out = []
    for n in notes:
        m = dict(n)
        m["midi"] = n["midi"] + semitones
        m["start"] = n["start"] / speed
        m["end"] = n["end"] / speed
        if n.get("vibrato_rate_hz"):
            # Vibrato happens in time, so slowing the track slows the wobble.
            m["vibrato_rate_hz"] = n["vibrato_rate_hz"] * speed
        out.append(m)
    return out


def transform_contour(
    contour: np.ndarray, hop_s: float, semitones: int, speed: float
) -> tuple[np.ndarray, float]:
    """Transpose the contour and report its new frame spacing."""
    shifted = contour + semitones if semitones else contour
    return shifted, hop_s / speed


def transform_times(times: list[float], speed: float) -> list[float]:
    """Rescale any list of timestamps (lyrics, beats, loop points)."""
    if abs(speed - 1.0) < 1e-6:
        return times
    return [t / speed for t in times]


def suggest_transpose(
    song_low: float,
    song_high: float,
    voice_low: float,
    voice_high: float,
) -> tuple[int, str]:
    """Recommend a shift that centres the song in the singer's range.

    Returns (semitones, explanation). Being told "drop it two semitones" is not
    a failure — it is what every touring singer does with material written for
    someone else's voice.
    """
    song_span = song_high - song_low
    voice_span = voice_high - voice_low

    if song_span > voice_span + 0.5:
        # Cannot fit. Centre it and say so plainly rather than pretending.
        shift = int(round(((voice_low + voice_high) - (song_low + song_high)) / 2))
        return shift, (
            f"This song spans {song_span:.0f} semitones but your comfortable "
            f"range is {voice_span:.0f}. It will not fit in full; shifting "
            f"{shift:+d} centres it, and the extremes will be a stretch."
        )

    # Room to spare: centre the song within the available headroom.
    shift = int(round(((voice_low + voice_high) - (song_low + song_high)) / 2))
    if shift == 0:
        return 0, "This song already sits comfortably in your range."

    direction = "up" if shift > 0 else "down"
    return shift, (
        f"Shifting {direction} {abs(shift)} semitone{'s' if abs(shift) != 1 else ''} "
        f"centres the melody in your comfortable range."
    )
