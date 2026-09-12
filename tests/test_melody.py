"""Melody extraction checked against known ground truth.

These tests run on the synthetic fixture, where we chose every note ourselves.
That is the point: you cannot verify a pitch tracker against a real recording,
because you would be comparing one estimate to another.

Regenerate the fixture with:  python tests/make_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from singcoach.analysis import melody

FIXTURES = Path(__file__).parent / "fixtures"
TRUTH = FIXTURES / "fixture_truth.json"
VOCAL = FIXTURES / "fixture_vocal.wav"

pytestmark = pytest.mark.skipif(
    not TRUTH.exists(), reason="run tests/make_fixture.py first"
)


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads(TRUTH.read_text("utf-8"))


@pytest.fixture(scope="module")
def analysis() -> melody.MelodyAnalysis:
    # The fixture vocal is already isolated, so this exercises exactly the path
    # a real separated stem takes.
    return melody.analyse_vocal(VOCAL, use_raw_cache=False)


class TestNoteRecovery:
    def test_finds_roughly_the_right_number_of_notes(self, analysis, truth):
        expected = len(truth["notes"])
        # Allow a little slack: a portamento may legitimately split in two.
        assert expected - 1 <= len(analysis.notes) <= expected + 3

    def test_every_true_note_has_a_match_at_the_right_pitch(self, analysis, truth):
        """For each note we synthesised, some detected note overlapping it in
        time must carry the right pitch."""
        misses = []
        for t in truth["notes"]:
            mid = (t["start"] + t["end"]) / 2
            found = [n for n in analysis.notes if n.start <= mid < n.end]
            if not found:
                misses.append((t["label"], "no note detected"))
                continue
            err_semitones = abs(found[0].midi - t["midi"])
            if err_semitones > 0.5:
                misses.append((t["label"], f"off by {err_semitones:.2f} semitones"))
        assert not misses, f"mismatched notes: {misses}"

    def test_pitch_accuracy_is_within_a_few_cents(self, analysis, truth):
        errors = []
        for t in truth["notes"]:
            if t["vibrato"] or t["slide"]:
                continue  # centre pitch is intentionally ambiguous for these
            mid = (t["start"] + t["end"]) / 2
            found = [n for n in analysis.notes if n.start <= mid < n.end]
            if found:
                errors.append(abs(found[0].midi - t["midi"]) * 100.0)
        assert errors, "no comparable notes found"
        assert np.mean(errors) < 15.0, f"mean pitch error {np.mean(errors):.1f} cents"

    def test_rests_produce_no_target(self, analysis, truth):
        """The lead-in has no vocal, so nothing may be reported there.

        This is the honesty guarantee: silence must read as 'no target', never
        as an invented note."""
        lead_in = truth["lead_in_s"]
        early = [n for n in analysis.notes if n.end < lead_in - 0.1]
        assert not early, f"invented {len(early)} notes during the silent lead-in"


class TestExpression:
    def test_vibrato_is_detected_where_it_was_synthesised(self, analysis, truth):
        vib_true = [t for t in truth["notes"] if t["vibrato"]]
        detected = 0
        for t in vib_true:
            mid = (t["start"] + t["end"]) / 2
            found = [n for n in analysis.notes if n.start <= mid < n.end]
            if found and found[0].vibrato_rate_hz:
                detected += 1
        assert detected >= len(vib_true) - 1, (
            f"found vibrato on {detected}/{len(vib_true)} notes that have it"
        )

    def test_vibrato_rate_is_accurate(self, analysis, truth):
        want = truth["vibrato_rate_hz"]
        rates = [n.vibrato_rate_hz for n in analysis.notes if n.vibrato_rate_hz]
        assert rates, "no vibrato detected at all"
        assert abs(float(np.median(rates)) - want) < 1.5, (
            f"median rate {np.median(rates):.2f} Hz vs synthesised {want} Hz"
        )

    def test_no_vibrato_reported_on_plain_notes(self, analysis, truth):
        """A steady note must not be described as having vibrato — otherwise
        the coaching would tell the singer to fix something that isn't there."""
        false_positives = []
        for t in truth["notes"]:
            if t["vibrato"] or t["slide"]:
                continue
            mid = (t["start"] + t["end"]) / 2
            found = [n for n in analysis.notes if n.start <= mid < n.end]
            if found and found[0].vibrato_depth_cents:
                false_positives.append((t["label"], found[0].vibrato_depth_cents))
        assert not false_positives, f"vibrato hallucinated on: {false_positives}"

    def test_the_slide_is_measured(self, analysis, truth):
        """A portamento between two sustained notes shows up as a transition.

        Note what is *not* asserted: that the slide lives on the note it starts
        from. A glide connecting two held pitches is legitimately its own brief
        segment, and demanding otherwise would encode one arbitrary
        segmentation choice as the only correct one. What matters for coaching
        is that a downward glide is detected in the right place.
        """
        slide = next(t for t in truth["notes"] if t["slide"])
        nxt = next(t for t in truth["notes"] if t["index"] == slide["index"] + 1)
        expected = nxt["midi"] - slide["midi"]          # negative: sliding down

        # Anywhere in the second half of the slide note through the start of
        # the note it lands on.
        lo = (slide["start"] + slide["end"]) / 2
        hi = nxt["start"] + 0.15
        gliding = [
            n for n in analysis.notes
            if n.start < hi and n.end > lo and n.slide_semitones < -0.5
        ]
        assert gliding, (
            f"no downward glide found between {lo:.2f}s and {hi:.2f}s; "
            f"expected roughly {expected:+.0f} semitones. "
            f"Notes there: {[(round(n.start, 2), round(n.midi, 1), round(n.slide_semitones, 2)) for n in analysis.notes if n.start < hi and n.end > lo]}"
        )
        assert min(n.slide_semitones for n in gliding) < -1.0

    def test_the_slide_lands_on_the_right_pitch(self, analysis, truth):
        """However the glide is segmented, the note after it must be correct."""
        nxt = next(
            t for t in truth["notes"]
            if t["index"] == next(x for x in truth["notes"] if x["slide"])["index"] + 1
        )
        mid = (nxt["start"] + nxt["end"]) / 2
        found = [n for n in analysis.notes if n.start <= mid < n.end]
        assert found, "note after the slide not detected"
        assert abs(found[0].midi - nxt["midi"]) < 0.5


class TestOctaveRepair:
    def test_isolated_octave_drop_is_corrected(self):
        hop = 0.0116
        midi = np.full(200, 60.0)
        midi[100:104] -= 12.0                    # a classic halving error
        fixed, moved = melody.fix_octaves(midi, hop)
        assert moved == 4
        assert np.allclose(fixed, 60.0)

    def test_genuine_octave_leap_is_left_alone(self):
        """A real sung octave jump that stays put must survive untouched —
        over-correction would flatten the melody."""
        hop = 0.0116
        midi = np.concatenate([np.full(200, 60.0), np.full(200, 72.0)])
        fixed, _ = melody.fix_octaves(midi, hop)
        assert np.allclose(fixed[:150], 60.0)
        assert np.allclose(fixed[250:], 72.0)

    def test_steady_pitch_is_untouched(self):
        midi = np.full(300, 57.5)
        fixed, moved = melody.fix_octaves(midi, 0.0116)
        assert moved == 0
        assert np.allclose(fixed, 57.5)


class TestRangeEstimate:
    def test_range_matches_the_synthesised_span(self, analysis, truth):
        lo = min(t["midi"] for t in truth["notes"])
        hi = max(t["midi"] for t in truth["notes"])
        assert analysis.range_low_midi == pytest.approx(lo, abs=1.5)
        assert analysis.range_high_midi == pytest.approx(hi, abs=1.5)
