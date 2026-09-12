"""Pitch detection against synthesised signals of known frequency.

Every assertion here has a ground truth we generated, so a failure means the
detector is wrong — never that the reference was debatable.

The accuracy bar is 5 cents. That is roughly the smallest pitch difference a
trained ear reliably hears, so a detector inside it cannot be the limiting
factor in the app's feedback.
"""

from __future__ import annotations

import numpy as np
import pytest

from singcoach.config import SAMPLE_RATE
from singcoach.pitch import (
    PitchSmoother,
    StabilityTracker,
    Verdict,
    YinDetector,
    compare,
    hz_to_midi,
    midi_to_hz,
)


def tone(hz: float, seconds: float = 0.1, harmonics: int = 5, amp: float = 0.3) -> np.ndarray:
    """A voice-like tone: fundamental plus decaying harmonics.

    A pure sine is unrealistically easy — real voices are harmonically rich,
    and harmonics are exactly what tempts a pitch tracker into octave errors.
    """
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    out = np.zeros_like(t)
    for n in range(1, harmonics + 1):
        out += (1.0 / n) * np.sin(2 * np.pi * hz * n * t)
    return (out / np.max(np.abs(out)) * amp).astype(np.float32)


def cents_error(detected_midi: float, true_hz: float) -> float:
    return abs(detected_midi - hz_to_midi(true_hz)) * 100.0


class TestAccuracy:
    @pytest.mark.parametrize(
        "note,hz",
        [
            ("E2", 82.41), ("A2", 110.0), ("E3", 164.81), ("A3", 220.0),
            ("C4", 261.63), ("E4", 329.63), ("A4", 440.0), ("C5", 523.25),
            ("E5", 659.25), ("A5", 880.0),
        ],
    )
    def test_detects_notes_across_the_vocal_range(self, note, hz):
        det = YinDetector()
        reading = det.process(tone(hz, 0.15))
        assert reading.voiced, f"{note} ({hz} Hz) not detected at all"
        assert cents_error(reading.midi, hz) < 5.0, (
            f"{note}: detected {reading.hz:.2f} Hz, wanted {hz} Hz "
            f"({cents_error(reading.midi, hz):.1f} cents off)"
        )

    def test_accurate_on_quarter_tones(self):
        """Real singing sits between the keys; the detector must not snap."""
        hz = midi_to_hz(60.5)
        reading = YinDetector().process(tone(hz, 0.15))
        assert reading.voiced
        assert cents_error(reading.midi, hz) < 5.0

    def test_tracks_a_slow_glide(self):
        det = YinDetector()
        errors = []
        for midi in np.linspace(55, 67, 12):
            hz = midi_to_hz(midi)
            r = det.process(tone(hz, 0.12))
            if r.voiced:
                errors.append(cents_error(r.midi, hz))
        assert len(errors) >= 10
        assert max(errors) < 10.0


class TestHonesty:
    """The detector must never invent a pitch. A false reading becomes a false
    error on the singer's score."""

    def test_silence_reports_nothing(self):
        r = YinDetector().process(np.zeros(2048, dtype=np.float32))
        assert not r.voiced
        assert r.midi is None

    def test_white_noise_reports_nothing(self):
        rng = np.random.default_rng(0)
        det = YinDetector()
        voiced = [det.process(rng.normal(0, 0.2, 2048).astype(np.float32)).voiced
                  for _ in range(10)]
        assert not any(voiced), "noise was reported as a pitch"

    def test_very_quiet_input_is_gated(self):
        r = YinDetector().process(tone(220.0, 0.1, amp=0.0005))
        assert not r.voiced

    def test_confidence_is_low_for_noisy_input(self):
        rng = np.random.default_rng(1)
        signal = tone(220.0, 0.1) + rng.normal(0, 0.25, int(0.1 * SAMPLE_RATE)).astype(np.float32)
        clean = YinDetector().process(tone(220.0, 0.1))
        noisy = YinDetector().process(signal)
        assert noisy.confidence < clean.confidence


class TestOctaveErrors:
    def test_low_note_with_strong_harmonics_is_not_doubled(self):
        """The classic YIN failure: reporting a harmonic instead of the root."""
        det = YinDetector()
        r = det.process(tone(110.0, 0.15, harmonics=8))
        assert r.voiced
        assert cents_error(r.midi, 110.0) < 15.0, (
            f"expected ~110 Hz, got {r.hz:.1f} Hz — likely an octave error"
        )

    def test_missing_fundamental_still_reads_correctly(self):
        """Small speakers and thin mics roll off the fundamental; the pitch we
        perceive is still the one implied by the harmonic series."""
        t = np.arange(int(0.15 * SAMPLE_RATE)) / SAMPLE_RATE
        sig = sum(np.sin(2 * np.pi * 150.0 * n * t) / n for n in (2, 3, 4, 5))
        sig = (sig / np.max(np.abs(sig)) * 0.3).astype(np.float32)
        r = YinDetector().process(sig)
        assert r.voiced
        # 150 Hz itself, or its octave, are both defensible; a fifth is not.
        assert min(cents_error(r.midi, 150.0), cents_error(r.midi, 300.0)) < 30.0


class TestComparison:
    def test_on_pitch_within_tolerance(self):
        c = compare(60.1, 60.0, tolerance_cents=20.0)
        assert c.verdict is Verdict.ON_PITCH
        assert c.direction == 0

    def test_flat_says_go_higher(self):
        c = compare(59.0, 60.0)
        assert c.verdict is Verdict.FLAT
        assert c.message == "GO HIGHER"
        assert c.direction == 1

    def test_sharp_says_go_lower(self):
        c = compare(61.0, 60.0)
        assert c.verdict is Verdict.SHARP
        assert c.message == "GO LOWER"
        assert c.direction == -1

    def test_slightly_flat_is_distinguished_from_flat(self):
        assert compare(59.7, 60.0).verdict is Verdict.SLIGHTLY_FLAT
        assert compare(59.0, 60.0).verdict is Verdict.FLAT

    def test_octave_error_is_named_as_such(self):
        c = compare(48.0, 60.0)
        assert c.verdict is Verdict.WRONG_OCTAVE
        assert "octave" in c.message

    def test_octave_can_be_forgiven(self):
        """A bass singing a soprano line an octave down is not making a mistake."""
        c = compare(48.0, 60.0, ignore_octave=True)
        assert c.verdict is Verdict.ON_PITCH

    def test_no_target_is_not_an_error(self):
        c = compare(60.0, None)
        assert c.verdict is Verdict.NO_TARGET
        assert not c.verdict.is_scored

    def test_silence_against_a_target_is_not_scored(self):
        c = compare(None, 60.0)
        assert c.verdict is Verdict.SILENT
        assert not c.verdict.is_scored

    def test_cents_sign_convention(self):
        assert compare(61.0, 60.0).cents == pytest.approx(100.0)
        assert compare(59.0, 60.0).cents == pytest.approx(-100.0)


class TestSmoothing:
    def test_median_rejects_a_single_octave_slip(self):
        from singcoach.pitch.detector import PitchReading

        sm = PitchSmoother(size=3)
        out = None
        for midi in (60.0, 48.0, 60.0):
            out = sm.push(PitchReading(midi_to_hz(midi), midi, 0.05, -20.0))
        assert out == pytest.approx(60.0)

    def test_silence_clears_the_window(self):
        from singcoach.pitch.detector import PitchReading

        sm = PitchSmoother(size=3)
        sm.push(PitchReading(440.0, 69.0, 0.05, -20.0))
        assert sm.push(PitchReading(None, None, 1.0, -80.0)) is None
        out = sm.push(PitchReading(midi_to_hz(60.0), 60.0, 0.05, -20.0))
        assert out == pytest.approx(60.0), "smoothed across a gap"

    def test_rejects_even_window(self):
        with pytest.raises(ValueError):
            PitchSmoother(size=4)


class TestStability:
    def test_steady_note_reads_as_steady(self):
        tr = StabilityTracker(hop_s=0.0116)
        for _ in range(90):
            tr.push(60.0)
        s = tr.analyse()
        assert s is not None and s.is_steady
        assert abs(s.drift_cents_per_s) < 1.0

    def test_drift_is_measured(self):
        """Sliding flat across a held note is a specific, coachable fault."""
        tr = StabilityTracker(hop_s=0.0116)
        for i in range(90):
            tr.push(60.0 - i * 0.004)
        s = tr.analyse()
        assert s is not None
        assert s.drift_cents_per_s < -20.0

    def test_vibrato_is_recognised_not_called_wobble(self):
        tr = StabilityTracker(hop_s=0.0116)
        for i in range(120):
            t = i * 0.0116
            tr.push(60.0 + 0.4 * np.sin(2 * np.pi * 5.5 * t))
        s = tr.analyse()
        assert s is not None and s.has_vibrato
        assert 4.0 < s.vibrato_rate_hz < 7.0

    def test_gap_resets_the_window(self):
        tr = StabilityTracker(hop_s=0.0116)
        for _ in range(90):
            tr.push(60.0)
        tr.push(None)
        assert tr.analyse() is None
