"""Key and tempo detection.

Two small jobs that both feed practice features rather than pitch feedback:

* **Key** decides whether a harmony guide's third is major or minor. Drawing a
  major third over a minor-key song would teach the singer a wrong note, so
  this has to be right — and when it is not confident, the app should offer the
  fifth (which is key-agnostic) rather than guess a third.
* **Beats** give loop points somewhere musical to snap to. Looping from 1:04.37
  to 1:12.91 starts mid-word; looping bar to bar does not.

Both run on the *accompaniment* stem, not the mix. The instrumental is where
the harmony actually lives, and removing the vocal takes a lot of pitched noise
out of the chroma estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from ..config import ANALYSIS_SR

ProgressFn = Callable[[str, float], None]


def _noop(_s: str, _f: float) -> None:
    pass


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles: how strongly each scale degree is expected
# to be present in a major and a minor key.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

#: Correlation gap below which a relative major/minor pair counts as tied and
#: the tonic-salience tie-break takes over. Measured on this song the pair came
#: in 0.008 apart, so anything on this scale is a coin flip.
RELATIVE_TIE_MARGIN = 0.15


@dataclass
class KeyBeats:
    key: str                  # e.g. "A"
    mode: str                 # "major" | "minor"
    key_confidence: float     # 0..1, gap between best and runner-up
    tonic_pc: int             # 0..11
    tempo_bpm: float
    beats: list[float]
    downbeats: list[float]
    #: Equally valid readings of the same pulse (halves/thirds of the grid).
    #: Non-empty means "this could also reasonably be counted at these speeds".
    tempo_alternatives: list[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.key} {self.mode}"

    #: Semitone offsets for a diatonic third above the tonic in this mode.
    @property
    def third_semitones(self) -> int:
        return 4 if self.mode == "major" else 3

    def to_dict(self) -> dict:
        return asdict(self)


def _relative_of(pc: int, mode: str) -> tuple[int, str]:
    """The relative major of a minor key, or relative minor of a major one."""
    if mode == "major":
        return (pc + 9) % 12, "minor"
    return (pc + 3) % 12, "major"


def _correlate_profiles(
    chroma_mean: np.ndarray, bass_chroma: np.ndarray | None = None
) -> tuple[int, str, float]:
    """Rotate both profiles through all 12 tonics and take the best match.

    There is a well-known trap here. A major key and its relative minor contain
    exactly the same seven pitch classes — A major and F# minor are built from
    identical notes — so a histogram of pitch content correlates almost equally
    with both, and which one wins is close to a coin flip. On this song the
    first implementation picked F# minor over A major with a margin of 0.011,
    i.e. noise.

    That distinction is not cosmetic: it decides whether a harmony guide draws
    a major or a minor third, and the wrong one teaches a wrong note.

    Pitch-class content cannot resolve it, so we bring in evidence that can —
    the **bass**. Whatever the mode, the tonic is where the low end lives, and
    the two candidates in a relative pair have different tonics (A vs F#). So
    when the top two are a relative pair, the bass decides.

    ``confidence`` deliberately measures the margin over *unrelated* keys only.
    Relative ambiguity is expected and separately resolved, and folding it into
    the score would make every song look uncertain.
    """
    x = chroma_mean - chroma_mean.mean()
    scores: list[tuple[float, int, str]] = []
    for pc in range(12):
        for profile, mode in ((_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")):
            p = np.roll(profile, pc)
            p = p - p.mean()
            denom = np.linalg.norm(x) * np.linalg.norm(p)
            scores.append((float(np.dot(x, p) / denom) if denom else 0.0, pc, mode))
    scores.sort(reverse=True)

    best_score, best_pc, best_mode = scores[0]
    rel_pc, rel_mode = _relative_of(best_pc, best_mode)

    # Margin against the best candidate that is NOT the relative pair.
    unrelated = [
        s for s, pc, mode in scores[1:] if not (pc == rel_pc and mode == rel_mode)
    ]
    runner_up = unrelated[0] if unrelated else 0.0
    confidence = float(
        np.clip((best_score - runner_up) / (abs(best_score) + 1e-9), 0.0, 1.0)
    )

    if bass_chroma is not None:
        rel_score = next(
            (s for s, pc, mode in scores if pc == rel_pc and mode == rel_mode), None
        )
        # Only intervene when pitch content genuinely cannot separate them.
        if rel_score is not None and abs(best_score - rel_score) < RELATIVE_TIE_MARGIN:
            # Between exactly two candidates, pick the more salient tonic
            # outright rather than demanding some arbitrary margin. Salience
            # combines overall presence with bass presence, each normalised so
            # neither view dominates purely by scale.
            full_n = chroma_mean / (chroma_mean.max() + 1e-9)
            bass_n = bass_chroma / (bass_chroma.max() + 1e-9)
            salience = full_n + bass_n
            if salience[rel_pc] > salience[best_pc]:
                best_pc, best_mode = rel_pc, rel_mode

    return best_pc, best_mode, confidence


def analyse(
    accompaniment_path: Path,
    *,
    progress: ProgressFn = _noop,
) -> KeyBeats:
    progress("loading accompaniment", 0.0)
    audio, sr = sf.read(str(accompaniment_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != ANALYSIS_SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        sr = ANALYSIS_SR

    progress("estimating key", 0.3)
    # CQT chroma tracks pitch classes better than STFT chroma on real
    # instruments, which is what we have here.
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    # A second chroma restricted to the low register, to locate the tonic.
    bass_chroma = librosa.feature.chroma_cqt(
        y=mono, sr=sr, fmin=librosa.note_to_hz("C2"), n_octaves=3
    ).mean(axis=1)
    tonic_pc, mode, confidence = _correlate_profiles(chroma.mean(axis=1), bass_chroma)

    progress("tracking beats", 0.6)
    tempo_bpm, beats, alternatives = _track_beats(mono, sr)

    # Assume 4/4 and take every fourth beat as a downbeat. Good enough to snap
    # loop points to; we are not doing metrical analysis.
    downbeats = beats[::4]

    progress("done", 1.0)
    return KeyBeats(
        key=PITCH_CLASSES[tonic_pc],
        mode=mode,
        key_confidence=round(confidence, 3),
        tonic_pc=tonic_pc,
        tempo_bpm=round(tempo_bpm, 2),
        tempo_alternatives=alternatives,
        beats=[round(b, 4) for b in beats],
        downbeats=[round(b, 4) for b in downbeats],
    )


#: Tempos a human would actually count a song at. Outside this, we are almost
#: certainly reporting a subdivision or a multiple rather than the pulse.
_TEMPO_SANE_LOW = 55.0
_TEMPO_SANE_HIGH = 145.0


def _track_beats(mono: np.ndarray, sr: int) -> tuple[float, list[float], list[float]]:
    """Beat positions, the grid's tempo, and the metrically plausible readings.

    Beat trackers routinely lock onto a subdivision rather than the pulse a
    person would tap. This song comes back at ~152 BPM; counted in its actual
    slow 6/8 feel it is nearer 51.

    An earlier version silently divided that down to look sensible, which was
    the wrong call: the reported number then contradicted the beat grid handed
    to the loop feature, and a number that disagrees with the data is worse
    than an honestly ambiguous one. So the tempo returned is the one the grid
    actually has, accompanied by the halves and thirds that are equally valid
    readings of the same pulse. The UI can offer them; it must not pretend
    there is only one.
    """
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    grid_bpm = float(np.atleast_1d(tempo)[0])

    alternatives = [
        round(grid_bpm / d, 1)
        for d in (2.0, 3.0, 4.0)
        if _TEMPO_SANE_LOW * 0.6 <= grid_bpm / d <= _TEMPO_SANE_HIGH
    ]
    return grid_bpm, beats, alternatives


def snap_to_beat(t: float, beats: list[float], *, prefer_downbeat: bool = False) -> float:
    """Move a loop marker to the nearest beat. Used by the A/B loop UI."""
    if not beats:
        return t
    arr = np.asarray(beats)
    return float(arr[int(np.argmin(np.abs(arr - t)))])
