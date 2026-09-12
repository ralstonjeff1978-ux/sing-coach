"""Extract the sung melody from an isolated vocal stem.

Two representations come out of here, and the difference between them is the
whole design argument of this app:

**The contour** — pitch at every 11.6 ms frame, exactly as sung. Scoops into
notes, slides between them, vibrato, blue notes sitting deliberately between
two piano keys. This is what singing actually is.

**The notes** — that contour carved into discrete events with one pitch each.
Tidy, easy to draw, easy to score against, and a lie about any singer with
style.

For something like a hymn or a pop topline the notes are a fine target. For a
blues-inflected vocal they are actively harmful: a bent third reads as "40
cents flat" and the app would coach the soul out of the performance. So we
compute both, measure how expressive the vocal is (see
:func:`expressiveness`), and let the app target the contour instead when the
song calls for it.

A note on honesty: pYIN reports a confidence per frame, and we keep it. Regions
where the tracker is not sure — rap, spoken word, stacked harmonies, heavy
effects — are marked unvoiced rather than assigned a best guess. A confidently
wrong target teaches the wrong thing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from ..config import ANALYSIS_SR, F0_MAX_HZ, F0_MIN_HZ

ProgressFn = Callable[[str, float], None]


def _noop(_s: str, _f: float) -> None:
    pass


# --- tuning constants ------------------------------------------------------

HOP = 256                      # 11.6 ms at 22.05 kHz
FRAME_LENGTH = 2048

#: A voiced gap longer than this ends a note.
MAX_GAP_S = 0.08
#: Sustained departure from the running centre that counts as a new note.
NOTE_SPLIT_CENTS = 80.0
#: ...but only if it persists this long. Shorter excursions are vibrato/scoops.
NOTE_SPLIT_MIN_S = 0.06
#: Anything briefer than this is a transient, not a note.
MIN_NOTE_S = 0.10
#: Frames below this confidence are treated as "we don't know".
#:
#: Chosen by sweep (scripts/tune_melody.py) rather than by feel. Loosening from
#: 0.30 to 0.20 raised voiced coverage from 18.4% to 23.2% while the share of
#: implausible sub-E3 frames moved only 5.0% -> 5.6%, so the frames recovered
#: are overwhelmingly real singing rather than breath and bleed. Below 0.15 the
#: error rate starts climbing faster than the coverage.
MIN_VOICED_PROB = 0.20

#: Human vibrato lives here. Outside this band it is drift or tremor, not vibrato.
VIBRATO_MIN_HZ = 3.5
VIBRATO_MAX_HZ = 9.0
#: Peak-to-peak extent below which the modulation is wobble, not vibrato.
#: A trained vibrato typically spans 50-150 cents peak-to-peak.
VIBRATO_MIN_DEPTH_CENTS = 30.0
#: How far the spectral peak must stand above the surrounding noise floor
#: before we are willing to call it periodic.
VIBRATO_MIN_PROMINENCE = 3.0


@dataclass
class Note:
    start: float
    end: float
    midi: float                     # fractional: 60.4 means 40 cents above C4
    cents_offset: float             # distance from the nearest equal-tempered semitone
    confidence: float
    peak_db: float
    vibrato_rate_hz: float | None = None
    vibrato_depth_cents: float | None = None
    #: Net pitch travel from the note's head to its tail. Large => a slide.
    slide_semitones: float = 0.0
    #: Pitch travel during the first 15% — a scoop into the note.
    scoop_cents: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def semitone(self) -> int:
        return int(round(self.midi))


@dataclass
class MelodyAnalysis:
    """Everything we learned about the sung line."""

    hop_s: float
    times: np.ndarray = field(repr=False)
    f0_hz: np.ndarray = field(repr=False)          # NaN where unvoiced
    midi: np.ndarray = field(repr=False)           # NaN where unvoiced
    voiced_prob: np.ndarray = field(repr=False)
    notes: list[Note] = field(default_factory=list)

    expressiveness: float = 0.0
    suggested_mode: str = "quantized"              # or "expressive"
    range_low_midi: float | None = None
    range_high_midi: float | None = None
    voiced_fraction: float = 0.0
    #: How many frames the octave-error repair had to move. A high count is a
    #: hint that the separation was poor, not just that the tracker slipped.
    octaves_fixed: int = 0

    def to_dict(self) -> dict:
        """JSON-safe. The contour is rounded — 0.01 semitone is 1 cent, and
        nobody hears a tenth of that, so full float64 is wasted bytes."""
        return {
            "hop_s": self.hop_s,
            "notes": [asdict(n) for n in self.notes],
            "expressiveness": round(self.expressiveness, 4),
            "suggested_mode": self.suggested_mode,
            "range_low_midi": self.range_low_midi,
            "range_high_midi": self.range_high_midi,
            "voiced_fraction": round(self.voiced_fraction, 4),
            "octaves_fixed": self.octaves_fixed,
            "contour": {
                "midi": [None if np.isnan(v) else round(float(v), 3) for v in self.midi],
                "voiced_prob": [round(float(v), 3) for v in self.voiced_prob],
            },
        }

    def note_at(self, t: float) -> Note | None:
        for n in self.notes:  # linear is fine; the UI keeps its own index
            if n.start <= t < n.end:
                return n
        return None


# ---------------------------------------------------------------------------
# pitch tracking
# ---------------------------------------------------------------------------


def track_f0(
    audio: np.ndarray,
    sr: int,
    *,
    progress: ProgressFn = _noop,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run pYIN. Returns raw (f0_hz, voiced_flag, voiced_prob), NaN where unvoiced.

    Deliberately unfiltered: the confidence floor is applied by the caller so
    that it stays a tunable knob rather than being baked into the cached
    pitch track.
    """
    progress("tracking pitch", 0.0)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP,
        fill_na=np.nan,
    )
    progress("tracking pitch", 1.0)
    return f0, voiced_flag, voiced_prob


def apply_confidence_floor(
    f0: np.ndarray, prob: np.ndarray, floor: float = MIN_VOICED_PROB
) -> np.ndarray:
    """Blank frames the tracker was not confident about.

    pYIN's own voiced flag is generous. Frames below the floor become honest
    gaps — the app shows "no target" there rather than inventing a pitch.
    """
    out = f0.copy()
    out[prob < floor] = np.nan
    return out


def _smooth_midi(midi: np.ndarray) -> np.ndarray:
    """Light 3-frame median. Kills single-frame slips without touching vibrato,
    which spans ~8 frames per cycle at 5 Hz."""
    out = midi.copy()
    valid = ~np.isnan(midi)
    idx = np.flatnonzero(valid)
    if idx.size < 3:
        return out
    vals = midi[idx]
    med = np.median(np.stack([vals[:-2], vals[1:-1], vals[2:]]), axis=0)
    out[idx[1:-1]] = med
    return out


#: Half-width of the window used to establish "where the voice currently is".
_OCTAVE_CONTEXT_S = 0.75
#: Only frames this far from the local centre are candidates for correction.
_OCTAVE_SUSPECT_SEMITONES = 7.0


def fix_octaves(midi: np.ndarray, hop_s: float, passes: int = 2) -> tuple[np.ndarray, int]:
    """Repair pYIN's octave errors.

    Autocorrelation pitch trackers systematically confuse a pitch with its
    half or double, because a waveform that repeats every N samples also
    repeats every 2N. On a breathy or reverberant vocal this happens often
    enough to matter: measured on this song, sub-E3 frames carried noticeably
    lower confidence than the rest, which is the signature of exactly this
    failure rather than of genuinely low singing.

    The fix uses continuity. A voice does not leap an octave and come straight
    back within a few tens of milliseconds, so when a frame sits far from the
    local centre of the melody and shifting it by ±12 semitones would bring it
    much closer, that shift is the better reading.

    Returns the corrected contour and how many frames were moved.
    """
    out = midi.copy()
    half = max(1, int(_OCTAVE_CONTEXT_S / hop_s))
    fixed = 0

    for _ in range(passes):
        moved = 0
        # Local centre from a rolling median of the *current* estimate.
        centre = np.full_like(out, np.nan)
        valid = ~np.isnan(out)
        idx = np.flatnonzero(valid)
        if idx.size < 5:
            break
        for i in idx:
            lo, hi = max(0, i - half), min(len(out), i + half + 1)
            window = out[lo:hi]
            window = window[~np.isnan(window)]
            if window.size >= 3:
                centre[i] = np.median(window)

        for i in idx:
            c = centre[i]
            if np.isnan(c):
                continue
            err = abs(out[i] - c)
            if err <= _OCTAVE_SUSPECT_SEMITONES:
                continue
            # Try both octave shifts; take one only if it is a clear improvement.
            best = min((abs(out[i] + s - c), s) for s in (-12.0, 12.0, 0.0))
            if best[1] and best[0] < err - 3.0:
                out[i] += best[1]
                moved += 1

        fixed += moved
        if moved == 0:
            break

    return out, fixed


# ---------------------------------------------------------------------------
# note segmentation
# ---------------------------------------------------------------------------


def _segments(midi: np.ndarray, hop_s: float) -> list[tuple[int, int]]:
    """Carve the contour into candidate note spans (start, end) frame indices."""
    gap_frames = max(1, int(MAX_GAP_S / hop_s))
    hold_frames = max(1, int(NOTE_SPLIT_MIN_S / hop_s))

    # Centre is a median over a trailing window, NOT an exponential average.
    # An EMA chases the new pitch: at a one-semitone step it closes a third of
    # the 100-cent gap within the 60 ms confirmation window, so the deviation
    # never stays above threshold and adjacent scale steps silently merge into
    # one note. A median over ~0.4 s holds still until the new pitch actually
    # dominates the window, and it is unmoved by vibrato swinging either side.
    window = deque(maxlen=max(3, int(0.40 / hop_s)))

    spans: list[tuple[int, int]] = []
    start: int | None = None
    silent = 0
    off_centre = 0

    for i, v in enumerate(midi):
        if np.isnan(v):
            silent += 1
            if start is not None and silent >= gap_frames:
                spans.append((start, i - silent + 1))
                start = None
                window.clear()
            continue

        silent = 0
        if start is None:
            start = i
            off_centre = 0
            window.clear()
            window.append(v)
            continue

        centre = float(np.median(window)) if window else v
        if abs(v - centre) * 100.0 > NOTE_SPLIT_CENTS:
            off_centre += 1
            if off_centre >= hold_frames:
                # A sustained move: close the old note, open a new one at the
                # first frame that departed.
                split = i - off_centre + 1
                spans.append((start, split))
                start = split
                off_centre = 0
                window.clear()
                window.append(v)
        else:
            off_centre = 0
            window.append(v)

    if start is not None:
        spans.append((start, len(midi)))

    min_frames = max(1, int(MIN_NOTE_S / hop_s))
    return [(a, b) for a, b in spans if b - a >= min_frames]


def _vibrato(cents: np.ndarray, hop_s: float) -> tuple[float | None, float | None]:
    """Measure vibrato rate and peak-to-peak depth from a cents-deviation series.

    Three guards, each earning its place:

    * **Linear detrend.** A note that slides has a strong low-frequency
      component. Without removing the trend, a portamento reads as very slow,
      very deep vibrato.
    * **Spectral prominence.** It is not enough that *some* energy sits in the
      vibrato band — noise puts energy everywhere. The peak must clearly stand
      above the surrounding spectrum, or it is not a periodic modulation.
    * **Peak-to-peak depth, measured in the time domain.** The FFT peak
      magnitude understates depth badly when the vibrato ramps in or varies
      over the note, which real vibrato always does. A 5th-to-95th percentile
      spread is robust to both, and peak-to-peak extent is what singing
      teachers actually talk about.
    """
    n = len(cents)
    if n < int(0.25 / hop_s):          # need ~2 cycles at 5 Hz before judging
        return None, None

    # Remove any linear slide before looking for periodicity.
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, cents, 1)
    x = cents - (slope * t + intercept)
    if np.allclose(x, 0):
        return None, None

    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=hop_s)
    band = (freqs >= VIBRATO_MIN_HZ) & (freqs <= VIBRATO_MAX_HZ)
    if not band.any() or spec[band].max() <= 0:
        return None, None

    k = np.flatnonzero(band)[np.argmax(spec[band])]
    # Compare the peak against the general level from 1-15 Hz. A genuine
    # vibrato towers over it; broadband wobble does not.
    context = (freqs >= 1.0) & (freqs <= 15.0)
    floor = float(np.median(spec[context])) if context.any() else 0.0
    if floor <= 0 or spec[k] / floor < VIBRATO_MIN_PROMINENCE:
        return None, None

    depth = float(np.percentile(x, 95) - np.percentile(x, 5))
    if depth < VIBRATO_MIN_DEPTH_CENTS:
        return None, None
    return round(float(freqs[k]), 2), round(depth, 1)


def _build_note(
    midi: np.ndarray,
    prob: np.ndarray,
    rms_db: np.ndarray,
    a: int,
    b: int,
    hop_s: float,
) -> Note | None:
    seg = midi[a:b]
    good = ~np.isnan(seg)
    if good.sum() < 2:
        return None
    vals = seg[good]

    # Representative pitch from the note's stable core. Using the whole span
    # would let a scoop at the front or a fall at the end drag the pitch away
    # from what the singer actually sustained.
    lo = int(len(vals) * 0.20)
    hi = max(lo + 1, int(len(vals) * 0.85))
    core = vals[lo:hi] if hi > lo else vals
    centre = float(np.median(core))

    cents_dev = (vals - centre) * 100.0
    rate, depth = _vibrato(cents_dev, hop_s)

    head = float(np.median(vals[: max(1, len(vals) // 6)]))
    tail = float(np.median(vals[-max(1, len(vals) // 6) :]))

    return Note(
        start=round(a * hop_s, 4),
        end=round(b * hop_s, 4),
        midi=round(centre, 3),
        cents_offset=round((centre - round(centre)) * 100.0, 1),
        confidence=round(float(np.mean(prob[a:b])), 3),
        peak_db=round(float(np.max(rms_db[a:b])) if b > a else -120.0, 1),
        vibrato_rate_hz=rate,
        vibrato_depth_cents=depth,
        slide_semitones=round(tail - head, 3),
        scoop_cents=round((float(vals[0]) - centre) * 100.0, 1),
    )


# ---------------------------------------------------------------------------
# expressiveness
# ---------------------------------------------------------------------------


def expressiveness(midi: np.ndarray, notes: list[Note], hop_s: float) -> float:
    """How badly would quantising this vocal to semitones misrepresent it?

    Three signals, each in 0..1, averaged:

    * **off-grid time** — fraction of voiced frames sitting more than a quarter
      tone from any equal-tempered semitone. Blues thirds and sevenths live here.
    * **motion** — how fast the pitch moves within notes. Melisma and slides
      score high; a hymn scores near zero.
    * **slide prevalence** — fraction of notes whose head and tail differ by
      more than a semitone.
    """
    valid = midi[~np.isnan(midi)]
    if valid.size < 10:
        return 0.0

    off_grid = np.abs(valid - np.round(valid)) * 100.0
    off_score = float(np.mean(off_grid > 25.0))

    d = np.abs(np.diff(valid)) / hop_s          # semitones per second
    motion_score = float(np.clip(np.mean(d) / 6.0, 0.0, 1.0))

    slide_score = (
        float(np.mean([abs(n.slide_semitones) > 1.0 for n in notes])) if notes else 0.0
    )

    return float(np.clip((off_score + motion_score + slide_score) / 3.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

#: Above this, quantised notes misrepresent the performance badly enough that
#: the app should score against the contour instead.
EXPRESSIVE_THRESHOLD = 0.33


def _raw_cache_path(vocal_path: Path) -> Path:
    """Where the unprocessed pYIN output is kept.

    pYIN costs ~a minute on a full song; every post-processing knob after it is
    microseconds. Caching the raw result means tuning segmentation or octave
    repair is instant instead of a coffee break, and it makes re-analysis after
    a code change nearly free.
    """
    return vocal_path.with_name("f0_raw.npz")


def analyse_vocal(
    vocal_path: Path,
    *,
    progress: ProgressFn = _noop,
    use_raw_cache: bool = True,
    confidence_floor: float = MIN_VOICED_PROB,
) -> MelodyAnalysis:
    progress("loading vocal", 0.0)
    raw_path = _raw_cache_path(vocal_path)

    if use_raw_cache and raw_path.exists():
        cached = np.load(raw_path)
        f0, prob, sr = cached["f0"], cached["prob"], int(cached["sr"])
        progress("using cached pitch track", 0.5)
    else:
        audio, sr = sf.read(str(vocal_path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if sr != ANALYSIS_SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
            sr = ANALYSIS_SR
        f0, _voiced, prob = track_f0(mono, sr, progress=progress)
        np.savez_compressed(raw_path, f0=f0, prob=prob, sr=sr)

    # RMS is cheap, but it needs the audio; load only if we skipped it above.
    audio, file_sr = sf.read(str(vocal_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if file_sr != sr:
        mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr)

    hop_s = HOP / sr
    f0 = apply_confidence_floor(f0, prob, confidence_floor)

    midi = np.full_like(f0, np.nan, dtype=np.float64)
    ok = ~np.isnan(f0)
    midi[ok] = librosa.hz_to_midi(f0[ok])
    midi = _smooth_midi(midi)
    midi, octaves_fixed = fix_octaves(midi, hop_s)

    rms = librosa.feature.rms(y=mono, frame_length=FRAME_LENGTH, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
    n = min(len(midi), len(rms_db), len(prob))
    midi, prob, rms_db = midi[:n], prob[:n], rms_db[:n]

    progress("segmenting notes", 0.6)
    notes = [
        note
        for a, b in _segments(midi, hop_s)
        if (note := _build_note(midi, prob, rms_db, a, b, hop_s)) is not None
    ]

    expr = expressiveness(midi, notes, hop_s)
    low, high = _range_from_notes(notes)

    progress("done", 1.0)
    return MelodyAnalysis(
        hop_s=hop_s,
        times=np.arange(n) * hop_s,
        f0_hz=f0[:n],
        midi=midi,
        voiced_prob=prob,
        notes=notes,
        expressiveness=expr,
        suggested_mode="expressive" if expr >= EXPRESSIVE_THRESHOLD else "quantized",
        range_low_midi=low,
        range_high_midi=high,
        voiced_fraction=round(float(np.mean(~np.isnan(midi))), 4),
        octaves_fixed=octaves_fixed,
    )


def _range_from_notes(notes: list[Note]) -> tuple[float | None, float | None]:
    """The song's singing range, derived from notes rather than raw frames.

    Taking a percentile of every voiced frame lets a handful of tracking errors
    define the extremes — the first attempt on this song reported a low of Bb2,
    which no one sang. Notes have already survived a 100 ms duration filter, so
    a spurious frame cannot become one. We additionally weight by duration and
    require reasonable confidence, because the range is used to recommend
    transposition and being wrong there sends the singer to the wrong key.
    """
    good = [n for n in notes if n.confidence >= 0.5 and n.duration >= MIN_NOTE_S]
    if not good:
        good = notes
    if not good:
        return None, None
    pitches = np.array([n.midi for n in good])
    weights = np.array([n.duration for n in good])
    order = np.argsort(pitches)
    pitches, weights = pitches[order], weights[order]
    cdf = np.cumsum(weights) / weights.sum()
    lo = float(pitches[np.searchsorted(cdf, 0.02)])
    hi = float(pitches[min(np.searchsorted(cdf, 0.98), len(pitches) - 1)])
    return round(lo, 2), round(hi, 2)
