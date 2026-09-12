"""Turning a pitch reading into coaching.

The output of this module is what the singer actually reads off the screen, so
the wording matters as much as the maths. Two principles:

**Say which way to move.** "You are 62 cents flat" is a measurement; "go higher"
is an instruction. Singers correct pitch by moving, not by arithmetic.

**Never guess.** When the target is unknown or the microphone heard nothing,
the verdict is :attr:`Verdict.NO_TARGET` or :attr:`Verdict.SILENT`, never a
default of "correct" or "wrong". Silence is not failure, and a passage the
melody extractor could not read is not the singer's problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Verdict(str, Enum):
    ON_PITCH = "on_pitch"
    SLIGHTLY_SHARP = "slightly_sharp"
    SLIGHTLY_FLAT = "slightly_flat"
    SHARP = "sharp"
    FLAT = "flat"
    WRONG_OCTAVE = "wrong_octave"
    NO_TARGET = "no_target"        # nothing to sing here (instrumental, or unreadable)
    SILENT = "silent"              # target exists, singer is not singing

    @property
    def is_scored(self) -> bool:
        """Whether this verdict should count toward an accuracy figure."""
        return self not in (Verdict.NO_TARGET, Verdict.SILENT)


#: Beyond this the singer is in the wrong octave, and saying "you are 1150
#: cents flat" would be useless where "you're an octave low" is actionable.
OCTAVE_THRESHOLD_CENTS = 600.0


@dataclass(frozen=True)
class Comparison:
    verdict: Verdict
    cents: float | None            # signed: positive = sung above target
    target_midi: float | None
    sung_midi: float | None
    confidence: float = 0.0

    @property
    def direction(self) -> int:
        """-1 sing lower, +1 sing higher, 0 hold."""
        if self.cents is None or self.verdict in (Verdict.NO_TARGET, Verdict.SILENT):
            return 0
        if self.verdict is Verdict.ON_PITCH:
            return 0
        return -1 if self.cents > 0 else 1

    @property
    def message(self) -> str:
        """Short instruction for the big readout."""
        match self.verdict:
            case Verdict.NO_TARGET:
                return "—"
            case Verdict.SILENT:
                return "waiting"
            case Verdict.ON_PITCH:
                return "ON PITCH"
            case Verdict.SLIGHTLY_SHARP:
                return "a touch sharp"
            case Verdict.SLIGHTLY_FLAT:
                return "a touch flat"
            case Verdict.SHARP:
                return "GO LOWER"
            case Verdict.FLAT:
                return "GO HIGHER"
            case Verdict.WRONG_OCTAVE:
                if self.cents is None:
                    return "wrong octave"
                octaves = abs(self.cents) / 1200.0
                where = "high" if self.cents > 0 else "low"
                n = max(1, int(round(octaves)))
                return f"an octave {where}" if n == 1 else f"{n} octaves {where}"
        return "—"

    @property
    def detail(self) -> str:
        if self.cents is None:
            return ""
        return f"{self.cents:+.0f}¢"


def cents_between(sung_midi: float, target_midi: float) -> float:
    """Signed distance in cents. Positive means the singer is above the target."""
    return (sung_midi - target_midi) * 100.0


def compare(
    sung_midi: float | None,
    target_midi: float | None,
    *,
    tolerance_cents: float = 20.0,
    loose_cents: float = 50.0,
    confidence: float = 1.0,
    ignore_octave: bool = False,
) -> Comparison:
    """Judge one frame.

    ``ignore_octave`` matters more than it looks: a bass practising a melody
    written for a soprano is singing correctly an octave down, and marking that
    wrong on every note would make the app unusable for half its users.
    """
    if target_midi is None:
        return Comparison(Verdict.NO_TARGET, None, None, sung_midi, confidence)
    if sung_midi is None:
        return Comparison(Verdict.SILENT, None, target_midi, None, 0.0)

    cents = cents_between(sung_midi, target_midi)

    if ignore_octave:
        # Fold onto the nearest octave of the target before judging.
        folded = cents - 1200.0 * round(cents / 1200.0)
        cents = folded

    if abs(cents) > OCTAVE_THRESHOLD_CENTS:
        return Comparison(Verdict.WRONG_OCTAVE, cents, target_midi, sung_midi, confidence)

    if abs(cents) <= tolerance_cents:
        verdict = Verdict.ON_PITCH
    elif abs(cents) <= loose_cents:
        verdict = Verdict.SLIGHTLY_SHARP if cents > 0 else Verdict.SLIGHTLY_FLAT
    else:
        verdict = Verdict.SHARP if cents > 0 else Verdict.FLAT

    return Comparison(verdict, cents, target_midi, sung_midi, confidence)


# ---------------------------------------------------------------------------
# expressive mode
# ---------------------------------------------------------------------------


class OctaveTracker:
    """Detects when the singer is deliberately singing in a different octave.

    A bass practising a soprano line an octave down is singing it *correctly*.
    Left unhandled the app calls every note "an octave low", which is both
    useless and demoralising — it turns the normal, universal act of moving a
    song into your own register into a permanent error state.

    So we watch the offset between what is sung and what is written. When it
    settles near a whole number of octaves for long enough, the app follows the
    singer rather than arguing with them.

    Deliberately slow to engage and slow to release: flipping octave mid-phrase
    because of a couple of stray frames would be worse than not doing it at all.
    """

    def __init__(self, window: int = 120, engage: float = 0.7) -> None:
        self.window = window
        self.engage = engage
        self._offsets: list[float] = []
        self._current = 0

    @property
    def offset(self) -> int:
        """Octave shift being applied, in semitones (0, ±12, ±24)."""
        return self._current

    def push(self, sung_midi: float | None, target_midi: float | None) -> int:
        if sung_midi is None or target_midi is None:
            return self._current

        self._offsets.append(sung_midi - target_midi)
        if len(self._offsets) > self.window:
            del self._offsets[: len(self._offsets) - self.window]
        if len(self._offsets) < self.window // 2:
            return self._current

        values = np.asarray(self._offsets)
        # Which octave shift explains the most frames?
        best_shift, best_share = 0, 0.0
        for shift in (-24, -12, 0, 12, 24):
            share = float(np.mean(np.abs(values - shift) < 3.0))
            if share > best_share:
                best_shift, best_share = shift, share

        if best_share >= self.engage:
            self._current = best_shift
        return self._current

    def reset(self) -> None:
        self._offsets.clear()
        self._current = 0


class ContourTarget:
    """Target pitch sampled from the original singer's contour.

    For expressive material the note grid is the wrong reference — see
    :mod:`singcoach.analysis.melody`. Here the target at time *t* is simply
    what the original singer was doing at time *t*, blue notes and all.
    """

    def __init__(self, midi_contour: np.ndarray, hop_s: float) -> None:
        self.contour = np.asarray(midi_contour, dtype=np.float64)
        self.hop_s = hop_s

    def at(self, t: float) -> float | None:
        if t < 0:
            return None
        i = int(round(t / self.hop_s))
        if not (0 <= i < len(self.contour)):
            return None
        v = self.contour[i]
        return None if np.isnan(v) else float(v)

    #: How far either side we will look for a pitch before declaring that
    #: there is genuinely nothing to sing. Sung phrases are interrupted
    #: constantly by consonants, which stop phonation for 50-150 ms at a time.
    #: A narrow window blanks the target on every one of them, so the readout
    #: flickers "no target here" in the middle of a word the singer is very
    #: much in the middle of singing.
    BRIDGE_S = 0.22

    def smoothed_at(self, t: float, width_s: float = 0.10) -> float | None:
        """Contour with vibrato averaged out, bridging short unvoiced gaps.

        Chasing someone else's vibrato cycle-for-cycle is neither possible nor
        desirable. Smoothing leaves the phrase shape — the scoops and slides
        worth learning — while removing the wobble the singer should be
        producing themselves rather than imitating frame by frame.

        Gaps shorter than :attr:`BRIDGE_S` are bridged from the surrounding
        pitch. Longer ones still return ``None``: a rest is a rest, and the app
        must keep saying so honestly.
        """
        if width_s <= 0:
            return self.at(t)

        i = int(round(t / self.hop_s))
        if not (0 <= i < len(self.contour)):
            return None

        half = max(1, int(width_s / self.hop_s / 2))
        lo, hi = max(0, i - half), min(len(self.contour), i + half + 1)
        window = self.contour[lo:hi]
        window = window[~np.isnan(window)]
        if window.size:
            return float(np.median(window))

        # Nothing in the smoothing window — look a little wider before giving
        # up, so a consonant does not read as a rest.
        bridge = max(1, int(self.BRIDGE_S / self.hop_s))
        lo, hi = max(0, i - bridge), min(len(self.contour), i + bridge + 1)
        wider = self.contour[lo:hi]
        wider = wider[~np.isnan(wider)]
        if wider.size == 0:
            return None
        return float(np.median(wider))


class NoteTarget:
    """Target pitch from quantised notes — the right reference for material
    that really is sung on the grid."""

    def __init__(self, notes: list[dict]) -> None:
        self.notes = sorted(notes, key=lambda n: n["start"])
        self._starts = np.array([n["start"] for n in self.notes]) if notes else np.empty(0)

    def at(self, t: float) -> float | None:
        if self._starts.size == 0:
            return None
        i = int(np.searchsorted(self._starts, t, side="right")) - 1
        if i < 0:
            return None
        note = self.notes[i]
        return float(note["midi"]) if note["start"] <= t < note["end"] else None

    def note_at(self, t: float) -> dict | None:
        if self._starts.size == 0:
            return None
        i = int(np.searchsorted(self._starts, t, side="right")) - 1
        if i < 0:
            return None
        note = self.notes[i]
        return note if note["start"] <= t < note["end"] else None
