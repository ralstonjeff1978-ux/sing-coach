"""Scoring and diagnosis, driven by simulated performances.

Each test sings a song in a specific, deliberate way — always flat, unsteady,
late — and asserts the app names that habit and does not name the others. A
diagnostic that fires on everything is as useless as one that fires on nothing,
so the negative assertions matter as much as the positive ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from singcoach.pitch.compare import compare
from singcoach.scoring import SessionScorer
from singcoach.scoring.store import (
    persistent_problem_notes,
    progress_trend,
    recent_sessions,
    save_session,
)


def make_notes(count: int = 20, spacing: float = 1.0, duration: float = 0.8,
               base_midi: float = 60.0, spread: float = 0.0) -> list[dict]:
    """Build a target melody.

    Duration is clamped below spacing because melody notes never overlap — the
    segmenter produces contiguous spans. Generating overlapping notes here once
    produced a false test failure: frames from one note were attributed to the
    next, which quietly destroyed the onset measurements.
    """
    duration = min(duration, spacing * 0.9)
    return [
        {
            "start": i * spacing,
            "end": i * spacing + duration,
            "midi": base_midi + (spread * (i % 5)),
        }
        for i in range(count)
    ]


def sing(scorer: SessionScorer, notes: list[dict], *, offset_cents=0.0,
         jitter_cents=0.0, onset_delay=0.0, frames_per_note=30, seed=0,
         sung_fraction=1.0) -> None:
    """Simulate singing, frame by frame, with a controllable flaw."""
    rng = np.random.default_rng(seed)
    for i, note in enumerate(notes):
        if i / max(1, len(notes)) >= sung_fraction:
            break
        span = note["end"] - note["start"] - onset_delay
        if span <= 0:
            continue
        for f in range(frames_per_note):
            t = note["start"] + onset_delay + span * f / frames_per_note
            bias = offset_cents(i, note) if callable(offset_cents) else offset_cents
            cents = bias + (rng.normal(0, jitter_cents) if jitter_cents else 0.0)
            sung = note["midi"] + cents / 100.0
            scorer.push(t, compare(sung, note["midi"], tolerance_cents=35.0))


class TestScore:
    def test_perfect_performance_scores_100(self):
        notes = make_notes()
        s = SessionScorer(notes)
        sing(s, notes)
        assert s.score() == 100.0
        assert s.coverage == 1.0

    def test_consistently_wrong_scores_zero(self):
        notes = make_notes()
        s = SessionScorer(notes)
        sing(s, notes, offset_cents=200.0)
        assert s.score() == 0.0

    def test_not_singing_is_not_scored_as_wrong(self):
        """Silence must not count against you — it is absence of evidence."""
        notes = make_notes()
        s = SessionScorer(notes)
        sing(s, notes, sung_fraction=0.5)
        assert s.score() == 100.0          # what was sung, was sung well
        assert s.coverage == pytest.approx(0.5, abs=0.06)

    def test_longer_notes_weigh_more(self):
        notes = [
            {"start": 0.0, "end": 3.0, "midi": 60.0},   # long, sung well
            {"start": 4.0, "end": 4.2, "midi": 62.0},   # short, sung badly
        ]
        s = SessionScorer(notes)
        for f in range(300):
            s.push(0.0 + 3.0 * f / 300, compare(60.0, 60.0))
        for f in range(20):
            s.push(4.0 + 0.2 * f / 20, compare(64.0, 62.0))
        assert s.score() > 80.0


class TestDiagnosis:
    def test_flat_singing_is_named(self):
        notes = make_notes(30)
        s = SessionScorer(notes)
        sing(s, notes, offset_cents=-30.0)
        codes = [f.code for f in s.findings()]
        assert "bias_flat" in codes
        finding = next(f for f in s.findings() if f.code == "bias_flat")
        assert "breath support" in finding.detail

    def test_sharp_singing_is_named(self):
        notes = make_notes(30)
        s = SessionScorer(notes)
        sing(s, notes, offset_cents=30.0)
        assert "bias_sharp" in [f.code for f in s.findings()]

    def test_accurate_singing_yields_no_bias_finding(self):
        notes = make_notes(30)
        s = SessionScorer(notes)
        sing(s, notes, jitter_cents=5.0)
        codes = [f.code for f in s.findings()]
        assert "bias_flat" not in codes and "bias_sharp" not in codes

    def test_flat_only_high_is_distinguished_from_flat_everywhere(self):
        """The whole point: reaching for high notes is a different problem
        from being flat throughout, and needs a different fix."""
        notes = make_notes(40, spread=3.0, base_midi=58.0)
        s = SessionScorer(notes)
        pitches = [n["midi"] for n in notes]
        threshold = np.percentile(pitches, 70)
        sing(s, notes, offset_cents=lambda i, n: -45.0 if n["midi"] >= threshold else -2.0)
        codes = [f.code for f in s.findings()]
        assert "flat_up_high" in codes

    def test_unsteady_notes_are_named(self):
        notes = make_notes(20, duration=1.5)
        s = SessionScorer(notes)
        sing(s, notes, jitter_cents=40.0, frames_per_note=60, seed=3)
        assert "unsteady" in [f.code for f in s.findings()]

    def test_late_entries_are_named_as_timing_not_pitch(self):
        notes = make_notes(25, duration=1.2)
        s = SessionScorer(notes)
        sing(s, notes, onset_delay=0.25)
        finding = next((f for f in s.findings() if f.code == "late_entry"), None)
        assert finding is not None
        assert "timing, not pitch" in finding.detail

    def test_no_findings_without_enough_evidence(self):
        """Three notes is not a habit."""
        notes = make_notes(3)
        s = SessionScorer(notes)
        sing(s, notes, offset_cents=-50.0)
        assert s.findings() == []

    def test_findings_are_ordered_by_severity(self):
        notes = make_notes(30)
        s = SessionScorer(notes)
        sing(s, notes, offset_cents=-40.0)
        findings = s.findings()
        severities = [f.severity for f in findings]
        order = {"major": 0, "notable": 1, "info": 2}
        assert severities == sorted(severities, key=lambda x: order[x])


class TestPhrases:
    def test_phrases_split_on_rests(self):
        notes = [
            {"start": 0.0, "end": 0.5, "midi": 60.0},
            {"start": 0.6, "end": 1.1, "midi": 62.0},
            {"start": 3.0, "end": 3.5, "midi": 64.0},
        ]
        s = SessionScorer(notes)
        assert len(s.phrases()) == 2

    def test_weakest_phrase_is_identified(self):
        notes = [
            {"start": 0.0, "end": 0.5, "midi": 60.0},
            {"start": 5.0, "end": 5.5, "midi": 62.0},
        ]
        s = SessionScorer(notes)
        for f in range(20):
            s.push(0.0 + 0.5 * f / 20, compare(60.0, 60.0))
        for f in range(20):
            s.push(5.0 + 0.5 * f / 20, compare(64.0, 62.0))
        weakest = s.weakest_phrases(limit=1)[0]
        assert weakest.start == 5.0


class TestHistory:
    def test_session_round_trips(self, tmp_path):
        db = tmp_path / "h.db"
        notes = make_notes(15)
        s = SessionScorer(notes)
        sing(s, notes)
        sid = save_session("abc", s, path=db)
        assert sid > 0
        rows = recent_sessions("abc", path=db)
        assert len(rows) == 1
        assert rows[0].score == 100.0

    def test_trend_excludes_easier_conditions(self, tmp_path):
        """A run at 70% speed must not masquerade as improvement."""
        db = tmp_path / "h.db"
        notes = make_notes(15)
        for speed in (1.0, 0.7):
            s = SessionScorer(notes)
            sing(s, notes)
            save_session("abc", s, speed=speed, path=db)
        assert len(progress_trend("abc", comparable_only=True, path=db)) == 1
        assert len(progress_trend("abc", comparable_only=False, path=db)) == 2

    def test_conditions_are_described(self, tmp_path):
        db = tmp_path / "h.db"
        notes = make_notes(15)
        s = SessionScorer(notes)
        sing(s, notes)
        save_session("abc", s, speed=0.8, transpose=-2, path=db)
        assert "80% speed" in recent_sessions("abc", path=db)[0].conditions

    def test_persistent_problems_need_repeats(self, tmp_path):
        db = tmp_path / "h.db"
        notes = make_notes(12)
        for _ in range(3):
            s = SessionScorer(notes)
            sing(s, notes, offset_cents=lambda i, n: -80.0 if i == 4 else 0.0)
            save_session("abc", s, path=db)
        problems = persistent_problem_notes("abc", min_sessions=3, path=db)
        assert problems
        assert problems[0]["note_index"] == 4

    def test_one_off_mistake_is_not_a_persistent_problem(self, tmp_path):
        db = tmp_path / "h.db"
        notes = make_notes(12)
        s = SessionScorer(notes)
        sing(s, notes, offset_cents=lambda i, n: -80.0 if i == 4 else 0.0)
        save_session("abc", s, path=db)
        assert persistent_problem_notes("abc", min_sessions=3, path=db) == []
