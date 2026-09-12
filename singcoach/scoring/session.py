"""Scoring a performance, and — more usefully — diagnosing it.

A number on its own teaches nothing. "72%" tells a singer they were imperfect,
which they already knew, and gives them nothing to practise. So scoring here
produces two things: a figure for tracking progress over weeks, and a set of
**findings** that name specific, fixable habits.

The findings are the point. There is a real difference between:

* being flat on everything (usually breath support, or the song is too low)
* being flat only on the highest notes (reaching the top of your range)
* being accurate but unsteady (support and control, not pitch perception)
* being accurate but late (timing, not pitch at all)

Those are four different practice sessions, and a percentage cannot tell them
apart.

Every finding must be earned by enough evidence to be real; see
:data:`MIN_EVIDENCE`. Telling someone they have a habit on the strength of
three notes would be worse than saying nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import numpy as np

from ..pitch.compare import Comparison, Verdict

#: A finding needs at least this many scored notes behind it.
MIN_EVIDENCE = 8

#: Consistent bias beyond this many cents is a habit worth naming.
BIAS_CENTS = 12.0

#: Spread beyond this within a held note counts as unsteady.
UNSTEADY_CENTS = 45.0

#: Rests longer than this separate one phrase from the next.
PHRASE_GAP_S = 0.60


@dataclass
class NoteScore:
    """How one target note was sung."""

    index: int
    start: float
    end: float
    target_midi: float
    frames: int = 0
    in_tune_frames: int = 0
    cents_sum: float = 0.0
    cents_values: list[float] = field(default_factory=list)
    first_voiced_at: float | None = None
    sung: bool = False

    @property
    def accuracy(self) -> float:
        """Fraction of the note spent within tolerance, 0..1."""
        return self.in_tune_frames / self.frames if self.frames else 0.0

    @property
    def mean_cents(self) -> float:
        return self.cents_sum / self.frames if self.frames else 0.0

    @property
    def spread_cents(self) -> float:
        if len(self.cents_values) < 3:
            return 0.0
        return float(np.percentile(self.cents_values, 95) - np.percentile(self.cents_values, 5))

    @property
    def onset_error(self) -> float | None:
        """Positive means you came in late."""
        return None if self.first_voiced_at is None else self.first_voiced_at - self.start

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Finding:
    """One diagnosed habit, with the evidence behind it."""

    code: str
    headline: str
    detail: str
    severity: str            # "info" | "notable" | "major"
    evidence: int

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"[{self.severity}] {self.headline} — {self.detail}"


@dataclass
class Phrase:
    start: float
    end: float
    notes: list[NoteScore]

    @property
    def accuracy(self) -> float:
        scored = [n for n in self.notes if n.frames]
        if not scored:
            return 0.0
        weight = sum(n.frames for n in scored)
        return sum(n.accuracy * n.frames for n in scored) / weight


class SessionScorer:
    """Accumulates frame-by-frame judgements into a scored performance."""

    def __init__(self, notes: list[dict], tolerance_cents: float = 35.0) -> None:
        self.tolerance = tolerance_cents
        self.note_scores: list[NoteScore] = [
            NoteScore(
                index=i,
                start=float(n["start"]),
                end=float(n["end"]),
                target_midi=float(n["midi"]),
            )
            for i, n in enumerate(notes)
        ]
        self._starts = np.array([n.start for n in self.note_scores])
        self.started = datetime.now()

    def _note_at(self, t: float) -> NoteScore | None:
        if self._starts.size == 0:
            return None
        i = int(np.searchsorted(self._starts, t, side="right")) - 1
        if i < 0:
            return None
        note = self.note_scores[i]
        return note if note.start <= t < note.end else None

    def push(self, song_time: float, comparison: Comparison) -> None:
        """Record one frame's judgement at a given position in the song."""
        if not comparison.verdict.is_scored or comparison.cents is None:
            return
        note = self._note_at(song_time)
        if note is None:
            return

        note.frames += 1
        note.sung = True
        note.cents_sum += comparison.cents
        note.cents_values.append(comparison.cents)
        if abs(comparison.cents) <= self.tolerance:
            note.in_tune_frames += 1
        if note.first_voiced_at is None:
            note.first_voiced_at = song_time

    # -- results ------------------------------------------------------------

    @property
    def attempted(self) -> list[NoteScore]:
        return [n for n in self.note_scores if n.frames]

    def score(self) -> float:
        """Overall accuracy 0..100, weighted by how long each note lasts.

        Duration weighting matters: a three-second sustained note is a bigger
        test than a passing sixteenth, and counting them equally would let a
        singer coast through the hard parts.
        """
        scored = self.attempted
        if not scored:
            return 0.0
        total = sum(n.frames for n in scored)
        return round(100.0 * sum(n.accuracy * n.frames for n in scored) / total, 1)

    @property
    def coverage(self) -> float:
        """Fraction of the song's notes you actually attempted."""
        if not self.note_scores:
            return 0.0
        return round(len(self.attempted) / len(self.note_scores), 3)

    def phrases(self) -> list[Phrase]:
        out: list[Phrase] = []
        current: list[NoteScore] = []
        for note in self.note_scores:
            if current and note.start - current[-1].end >= PHRASE_GAP_S:
                out.append(Phrase(current[0].start, current[-1].end, current))
                current = []
            current.append(note)
        if current:
            out.append(Phrase(current[0].start, current[-1].end, current))
        return out

    def weakest_phrases(self, limit: int = 5) -> list[Phrase]:
        attempted = [p for p in self.phrases() if any(n.frames for n in p.notes)]
        return sorted(attempted, key=lambda p: p.accuracy)[:limit]

    # -- diagnosis ----------------------------------------------------------

    def findings(self) -> list[Finding]:
        scored = self.attempted
        out: list[Finding] = []
        if len(scored) < MIN_EVIDENCE:
            return out

        out.extend(self._pitch_bias(scored))
        out.extend(self._range_dependent_bias(scored))
        out.extend(self._steadiness(scored))
        out.extend(self._timing(scored))
        out.extend(self._sustained_notes(scored))

        order = {"major": 0, "notable": 1, "info": 2}
        return sorted(out, key=lambda f: (order[f.severity], -f.evidence))

    def _pitch_bias(self, scored: list[NoteScore]) -> Iterable[Finding]:
        means = [n.mean_cents for n in scored]
        bias = float(np.mean(means))
        if abs(bias) < BIAS_CENTS:
            return
        # Only call it a habit if it is consistent, not an average of extremes.
        same_side = np.mean([np.sign(m) == np.sign(bias) for m in means])
        if same_side < 0.65:
            return
        direction = "sharp" if bias > 0 else "flat"
        yield Finding(
            code=f"bias_{direction}",
            headline=f"You sing consistently {direction}",
            detail=(
                f"Average {abs(bias):.0f} cents {direction} across "
                f"{len(scored)} notes, on the same side {same_side:.0%} of the time. "
                + (
                    "Singing flat throughout usually means breath support rather "
                    "than pitch perception — try more air, and check the song is "
                    "not sitting too low for you."
                    if bias < 0
                    else "Singing sharp throughout often means pushing. Try easing "
                    "off the volume and letting the note settle."
                )
            ),
            severity="major" if abs(bias) > 25 else "notable",
            evidence=len(scored),
        )

    def _range_dependent_bias(self, scored: list[NoteScore]) -> Iterable[Finding]:
        """Is the error concentrated at one end of your range?

        This separates "the song is hard" from "the top of my range is hard",
        which need completely different responses — transposition versus
        technique.
        """
        if len(scored) < MIN_EVIDENCE * 2:
            return
        pitches = np.array([n.target_midi for n in scored])
        errors = np.array([n.mean_cents for n in scored])
        high = pitches >= np.percentile(pitches, 70)
        low = pitches <= np.percentile(pitches, 30)
        if high.sum() < 4 or low.sum() < 4:
            return

        high_err, low_err = float(np.mean(errors[high])), float(np.mean(errors[low]))
        if abs(high_err - low_err) < 15.0:
            return

        if high_err < -BIAS_CENTS and high_err < low_err:
            yield Finding(
                code="flat_up_high",
                headline="You go flat on the high notes specifically",
                detail=(
                    f"High notes average {abs(high_err):.0f} cents flat while lower "
                    f"ones sit within {abs(low_err):.0f}. That is reaching, not tone "
                    "deafness — either the song wants transposing down, or those "
                    "notes need more support underneath them."
                ),
                severity="notable",
                evidence=int(high.sum()),
            )
        elif low_err < -BIAS_CENTS and low_err < high_err:
            yield Finding(
                code="flat_down_low",
                headline="Your low notes drop under pitch",
                detail=(
                    f"Low notes average {abs(low_err):.0f} cents flat. Low notes "
                    "lose support easily; they usually need more air, not less."
                ),
                severity="notable",
                evidence=int(low.sum()),
            )

    def _steadiness(self, scored: list[NoteScore]) -> Iterable[Finding]:
        held = [n for n in scored if n.duration >= 0.5 and len(n.cents_values) >= 10]
        if len(held) < 5:
            return
        spreads = [n.spread_cents for n in held]
        typical = float(np.median(spreads))
        if typical <= UNSTEADY_CENTS:
            return
        accurate = float(np.mean([n.accuracy for n in held]))
        yield Finding(
            code="unsteady",
            headline="Your held notes wander",
            detail=(
                f"On sustained notes the pitch moves about {typical:.0f} cents "
                f"while you hold them"
                + (
                    ", even though you are landing on the right note. That is "
                    "control rather than pitch — long steady tones on a single "
                    "vowel are the drill for it."
                    if accurate > 0.6
                    else ". Try slowing the song down and holding each note "
                    "deliberately."
                )
            ),
            severity="notable",
            evidence=len(held),
        )

    def _timing(self, scored: list[NoteScore]) -> Iterable[Finding]:
        errors = [n.onset_error for n in scored if n.onset_error is not None]
        if len(errors) < MIN_EVIDENCE:
            return
        median = float(np.median(errors))
        if abs(median) < 0.06:
            return
        late = median > 0
        yield Finding(
            code="late_entry" if late else "early_entry",
            headline=f"You come in {'late' if late else 'early'}",
            detail=(
                f"Entries average {abs(median) * 1000:.0f} ms "
                f"{'behind' if late else 'ahead of'} the original across "
                f"{len(errors)} notes. This is timing, not pitch — try counting "
                "the bar in before each phrase."
            ),
            severity="notable" if abs(median) > 0.12 else "info",
            evidence=len(errors),
        )

    def _sustained_notes(self, scored: list[NoteScore]) -> Iterable[Finding]:
        held = [n for n in scored if n.duration >= 1.0]
        short = [n for n in scored if n.duration < 0.5]
        if len(held) < 4 or len(short) < 4:
            return
        held_acc = float(np.mean([n.accuracy for n in held]))
        short_acc = float(np.mean([n.accuracy for n in short]))
        if held_acc >= short_acc - 0.15:
            return
        yield Finding(
            code="weak_sustains",
            headline="Long notes are where you lose accuracy",
            detail=(
                f"Sustained notes score {held_acc:.0%} against {short_acc:.0%} on "
                "short ones. Starting a note right and holding it right are "
                "different skills; the second is breath."
            ),
            severity="notable",
            evidence=len(held),
        )

    # -- summary ------------------------------------------------------------

    def summary(self) -> dict:
        scored = self.attempted
        return {
            "score": self.score(),
            "coverage": self.coverage,
            "notes_attempted": len(scored),
            "notes_total": len(self.note_scores),
            "mean_cents": round(float(np.mean([n.mean_cents for n in scored])), 1) if scored else 0.0,
            "findings": [
                {"code": f.code, "headline": f.headline, "detail": f.detail,
                 "severity": f.severity, "evidence": f.evidence}
                for f in self.findings()
            ],
            "weakest_phrases": [
                {"start": round(p.start, 2), "end": round(p.end, 2),
                 "accuracy": round(p.accuracy, 3)}
                for p in self.weakest_phrases()
            ],
        }
