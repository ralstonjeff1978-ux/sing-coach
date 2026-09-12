"""Audio layer tests that need no sound device.

Everything here is the logic that has to be right before a speaker is involved:
ring buffer arithmetic, mix maths, latency estimation against a simulated
loopback, and the transformations that keep the melody aligned with altered
audio. Device handling itself is exercised by running the app.
"""

from __future__ import annotations

import numpy as np
import pytest

from singcoach.audio import RingBuffer
from singcoach.audio.calibrate import estimate_delay, make_chirp, measure, detect_bleed
from singcoach.audio.mixer import Mixer, StemSet
from singcoach.audio.render import suggest_transpose, transform_notes, transform_times
from singcoach.config import SAMPLE_RATE


def stems(seconds: float = 2.0, vocal_amp: float = 0.5, backing_amp: float = 0.25) -> StemSet:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    v = (np.sin(2 * np.pi * 440 * t) * vocal_amp).astype(np.float32)
    b = (np.sin(2 * np.pi * 110 * t) * backing_amp).astype(np.float32)
    return StemSet(vocals=np.vstack([v, v]), accompaniment=np.vstack([b, b]))


class TestRingBuffer:
    def test_reads_back_what_was_written(self):
        rb = RingBuffer(1000)
        rb.write(np.arange(100, dtype=np.float32))
        out, index = rb.read_window(64, 32)
        assert index == 0
        assert np.allclose(out, np.arange(64))

    def test_hop_advances_by_hop_not_window(self):
        rb = RingBuffer(1000)
        rb.write(np.arange(300, dtype=np.float32))
        _, i0 = rb.read_window(64, 32)
        _, i1 = rb.read_window(64, 32)
        assert i1 - i0 == 32

    def test_returns_none_when_short(self):
        rb = RingBuffer(1000)
        rb.write(np.zeros(10, dtype=np.float32))
        assert rb.read_window(64, 32) is None

    def test_wraps_correctly(self):
        rb = RingBuffer(128)
        rb.write(np.arange(100, dtype=np.float32))
        rb.read_window(64, 64)
        rb.write(np.arange(100, 160, dtype=np.float32))
        out, _ = rb.read_window(32, 32)
        assert np.allclose(out, np.arange(64, 96))

    def test_overrun_drops_oldest_and_is_reported(self):
        """Falling behind must lose old audio, never block the audio thread."""
        rb = RingBuffer(128)
        rb.write(np.arange(500, dtype=np.float32))
        assert rb.overruns >= 1
        assert rb.available <= 128


class TestMixer:
    def test_silent_until_playing(self):
        m = Mixer(stems())
        out = np.zeros((512, 2), dtype=np.float32)
        m.read(512, out, Mixer.make_scratch(512))
        assert np.allclose(out, 0.0)

    def test_vocal_gain_zero_removes_the_lead(self):
        """Karaoke mode: the vocal stem must genuinely be gone."""
        m = Mixer(stems())
        m.set_gain("vocals", 0.0)
        m.set_gain("accompaniment", 1.0)
        m.play()
        out = np.zeros((4096, 2), dtype=np.float32)
        m.read(4096, out, Mixer.make_scratch(4096))
        # Only the 110 Hz backing should remain; check via spectrum.
        spec = np.abs(np.fft.rfft(out[:, 0]))
        freqs = np.fft.rfftfreq(len(out), 1 / SAMPLE_RATE)
        at_440 = spec[np.argmin(np.abs(freqs - 440))]
        at_110 = spec[np.argmin(np.abs(freqs - 110))]
        assert at_440 < at_110 * 0.1

    def test_position_advances(self):
        m = Mixer(stems())
        m.play()
        out = np.zeros((512, 2), dtype=np.float32)
        scratch = Mixer.make_scratch(512)
        m.read(512, out, scratch)
        assert m.position_frames == 512

    def test_seek(self):
        m = Mixer(stems(seconds=4.0))
        m.seek(2.0)
        assert m.position == pytest.approx(2.0, abs=0.01)

    def test_stops_at_the_end(self):
        m = Mixer(stems(seconds=0.05))
        m.play()
        out = np.zeros((8192, 2), dtype=np.float32)
        m.read(8192, out, Mixer.make_scratch(8192))
        assert not m.playing

    def test_loop_wraps_back(self):
        m = Mixer(stems(seconds=4.0))
        m.set_loop(1.0, 1.05)
        m.seek(1.0)
        m.play()
        out = np.zeros((4096, 2), dtype=np.float32)
        scratch = Mixer.make_scratch(4096)
        for _ in range(4):
            m.read(4096, out, scratch)
        assert 1.0 <= m.position < 1.06

    def test_output_never_clips(self):
        m = Mixer(stems(vocal_amp=0.95, backing_amp=0.95))
        m.set_gain("vocals", 2.0)
        m.set_gain("accompaniment", 2.0)
        m.play()
        out = np.zeros((4096, 2), dtype=np.float32)
        m.read(4096, out, Mixer.make_scratch(4096))
        assert np.max(np.abs(out)) <= 1.0

    def test_gain_changes_are_ramped_not_stepped(self):
        """An instant gain jump puts a step in the waveform, which clicks."""
        m = Mixer(stems())
        m.set_gain("vocals", 0.0)
        m.play()
        out = np.zeros((256, 2), dtype=np.float32)
        scratch = Mixer.make_scratch(256)
        m.read(256, out, scratch)
        m.set_gain("vocals", 1.0)
        m.read(256, out, scratch)
        # Mid-ramp the applied gain should be between the two, not already 1.
        assert m._current["vocals"] < 1.0


class TestCalibration:
    def test_recovers_a_known_delay(self):
        chirp = make_chirp()
        delay_samples = 4410                       # exactly 100 ms
        recorded = np.zeros(int(0.6 * SAMPLE_RATE))
        recorded[delay_samples : delay_samples + len(chirp)] = chirp
        found = estimate_delay(recorded, chirp)
        assert found is not None
        delay_s, ratio = found
        assert delay_s == pytest.approx(0.1, abs=0.002)
        assert ratio > 4.0

    def test_survives_noise_and_attenuation(self):
        rng = np.random.default_rng(0)
        chirp = make_chirp()
        recorded = rng.normal(0, 0.01, int(0.6 * SAMPLE_RATE))
        recorded[3000 : 3000 + len(chirp)] += chirp * 0.05   # quiet bleed
        found = estimate_delay(recorded, chirp)
        assert found is not None
        assert found[0] == pytest.approx(3000 / SAMPLE_RATE, abs=0.002)

    def test_measure_agrees_over_repeats(self):
        chirp_len = len(make_chirp())

        def loopback(signal, listen_frames):
            out = np.zeros(listen_frames, dtype=np.float32)
            d = 5000
            out[d : d + chirp_len] = signal[:chirp_len] * 0.3
            return out

        result = measure(loopback, repeats=5)
        assert result.ok
        assert result.latency_ms == pytest.approx(1000 * 5000 / SAMPLE_RATE, abs=1.0)
        assert result.confidence > 0.5

    def test_reports_failure_rather_than_guessing(self):
        """Silence must produce an honest failure, never a fabricated number."""
        result = measure(lambda s, n: np.zeros(n, dtype=np.float32), repeats=3)
        assert not result.ok
        assert result.latency_ms == 0.0
        assert "Could not hear" in result.message

    def test_flags_implausible_latency(self):
        chirp_len = len(make_chirp())

        def slow(signal, listen_frames):
            out = np.zeros(max(listen_frames, 30000), dtype=np.float32)
            out[25000 : 25000 + chirp_len] = signal[:chirp_len] * 0.4
            return out

        result = measure(slow, repeats=3)
        assert not result.ok
        assert "Bluetooth" in result.message


class TestBleedDetection:
    def test_identical_signals_correlate(self):
        t = np.arange(4096) / SAMPLE_RATE
        sig = np.sin(2 * np.pi * 300 * t)
        assert detect_bleed(sig, sig) > 0.9

    def test_unrelated_signals_do_not(self):
        rng = np.random.default_rng(2)
        assert detect_bleed(rng.normal(0, 1, 4096), rng.normal(0, 1, 4096)) < 0.3


class TestAnalysisTransforms:
    def test_transpose_shifts_notes_exactly(self):
        notes = [{"midi": 60.0, "start": 1.0, "end": 2.0}]
        out = transform_notes(notes, 3, 1.0)
        assert out[0]["midi"] == 63.0
        assert out[0]["start"] == 1.0

    def test_slowdown_scales_time_not_pitch(self):
        notes = [{"midi": 60.0, "start": 1.0, "end": 2.0}]
        out = transform_notes(notes, 0, 0.5)
        assert out[0]["midi"] == 60.0
        assert out[0]["start"] == 2.0
        assert out[0]["end"] == 4.0

    def test_slowdown_also_slows_vibrato(self):
        notes = [{"midi": 60.0, "start": 0.0, "end": 1.0, "vibrato_rate_hz": 6.0}]
        assert transform_notes(notes, 0, 0.5)[0]["vibrato_rate_hz"] == 3.0

    def test_identity_transform_returns_input(self):
        notes = [{"midi": 60.0, "start": 1.0, "end": 2.0}]
        assert transform_notes(notes, 0, 1.0) is notes

    def test_times_rescale_consistently(self):
        assert transform_times([0.0, 2.0, 4.0], 0.5) == [0.0, 4.0, 8.0]


class TestTransposeSuggestion:
    def test_no_shift_when_it_already_fits(self):
        shift, msg = suggest_transpose(60, 72, 58, 74)
        assert shift == 0
        assert "comfortably" in msg

    def test_suggests_dropping_a_high_song(self):
        shift, msg = suggest_transpose(65, 77, 55, 67)
        assert shift < 0
        assert "down" in msg

    def test_suggests_raising_a_low_song(self):
        shift, msg = suggest_transpose(45, 55, 55, 67)
        assert shift > 0
        assert "up" in msg

    def test_says_so_when_the_song_cannot_fit(self):
        """Being honest beats silently recommending an impossible key."""
        shift, msg = suggest_transpose(50, 80, 58, 70)
        assert "will not fit" in msg
