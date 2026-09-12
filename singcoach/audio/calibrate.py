"""Measuring round-trip latency.

Every piece of feedback in this app depends on one number: how long after the
app plays a sound does that sound come back through the microphone. Output
buffer, DAC, headphone, air, microphone, ADC, input buffer — on a typical
Windows setup the total is 80-150 ms, and it varies by device and driver.

If we ignore it, a perfectly in-tune singer is compared against the wrong
moment of the melody and told they are wrong. If we guess it, we are wrong by
an unknown amount. So we measure it.

**Method.** Play a short logarithmic chirp while recording. Cross-correlate the
recording against the chirp we sent; the correlation peak sits exactly at the
round-trip delay. A chirp beats a click for this because its energy is spread
across time and frequency, giving a sharp, unambiguous correlation peak even
over a noisy room and a cheap microphone.

**On leakage.** This works on headphones (the mic picks up a little bleed,
which is plenty) and on speakers. On headphones with very good isolation the
signal may be too quiet to detect, which is reported honestly rather than
guessed at — the user can then raise the volume and retry, or enter a value by
hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SAMPLE_RATE

CHIRP_SECONDS = 0.08
CHIRP_LOW_HZ = 200.0
CHIRP_HIGH_HZ = 8000.0

#: How long to listen after playing. Comfortably longer than any sane latency.
LISTEN_SECONDS = 0.6
#: Repeats, so a single cough cannot decide the result.
DEFAULT_REPEATS = 5

#: Correlation peak must stand this far above the background to be believed.
MIN_PEAK_RATIO = 4.0


def make_chirp(seconds: float = CHIRP_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Logarithmic sweep with a raised-cosine envelope (no click at the edges)."""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    k = np.log(CHIRP_HIGH_HZ / CHIRP_LOW_HZ)
    phase = 2 * np.pi * CHIRP_LOW_HZ * seconds / k * (np.exp(k * t / seconds) - 1.0)
    envelope = np.hanning(n)
    return (np.sin(phase) * envelope * 0.5).astype(np.float32)


@dataclass(frozen=True)
class CalibrationResult:
    latency_ms: float
    confidence: float           # 0..1, from peak sharpness and agreement
    measurements_ms: list[float]
    spread_ms: float
    ok: bool
    message: str


def estimate_delay(
    recorded: np.ndarray, reference: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> tuple[float, float] | None:
    """Cross-correlate and return (delay_seconds, peak_ratio), or None."""
    if len(recorded) < len(reference):
        return None

    rec = recorded - np.mean(recorded)
    ref = reference - np.mean(reference)
    if np.allclose(rec, 0) or np.allclose(ref, 0):
        return None

    size = 1 << (len(rec) + len(ref)).bit_length()
    corr = np.fft.irfft(
        np.fft.rfft(rec, size) * np.conjugate(np.fft.rfft(ref, size)), size
    )[: len(rec)]

    corr = np.abs(corr)
    peak = int(np.argmax(corr))
    peak_value = corr[peak]
    if peak_value <= 0:
        return None

    # Compare against the typical level away from the peak.
    mask = np.ones(len(corr), dtype=bool)
    lo, hi = max(0, peak - 200), min(len(corr), peak + 200)
    mask[lo:hi] = False
    background = float(np.median(corr[mask])) if mask.any() else 0.0
    ratio = peak_value / background if background > 0 else float("inf")

    return peak / sample_rate, float(ratio)


def measure(
    play_and_record,
    *,
    repeats: int = DEFAULT_REPEATS,
    sample_rate: int = SAMPLE_RATE,
) -> CalibrationResult:
    """Run the measurement.

    ``play_and_record(signal, listen_frames) -> np.ndarray`` is supplied by the
    audio engine, which keeps this module free of any device handling and
    therefore testable with a simulated loopback.
    """
    chirp = make_chirp(sample_rate=sample_rate)
    listen_frames = int(LISTEN_SECONDS * sample_rate)

    results: list[float] = []
    ratios: list[float] = []
    for _ in range(repeats):
        recorded = play_and_record(chirp, listen_frames)
        found = estimate_delay(np.asarray(recorded, dtype=np.float64), chirp, sample_rate)
        if found is None:
            continue
        delay, ratio = found
        if ratio >= MIN_PEAK_RATIO:
            results.append(delay * 1000.0)
            ratios.append(ratio)

    if len(results) < max(2, repeats // 2):
        return CalibrationResult(
            latency_ms=0.0,
            confidence=0.0,
            measurements_ms=results,
            spread_ms=0.0,
            ok=False,
            message=(
                "Could not hear the test tone. Turn the volume up, make sure the "
                "right output device is selected, and try again. If you are on "
                "well-isolated headphones, cup one earcup near the microphone."
            ),
        )

    arr = np.asarray(results)
    # Median over mean: one bad reading should not move the answer.
    latency = float(np.median(arr))
    spread = float(np.percentile(arr, 90) - np.percentile(arr, 10))

    # Confidence falls off as the readings disagree. Above ~10 ms of spread
    # something is unstable and the number should not be trusted blindly.
    agreement = float(np.clip(1.0 - spread / 20.0, 0.0, 1.0))
    sharpness = float(np.clip(np.median(ratios) / 20.0, 0.0, 1.0))
    confidence = round(0.6 * agreement + 0.4 * sharpness, 3)

    if latency > 400.0:
        return CalibrationResult(
            latency, confidence, results, round(spread, 2), False,
            f"Measured {latency:.0f} ms, which is implausibly high. Check for "
            "Bluetooth headphones — they add far too much delay for singing. "
            "A wired headset is strongly recommended.",
        )

    return CalibrationResult(
        latency_ms=round(latency, 1),
        confidence=confidence,
        measurements_ms=[round(r, 1) for r in results],
        spread_ms=round(spread, 2),
        ok=True,
        message=f"Round-trip latency {latency:.0f} ms (±{spread / 2:.0f} ms).",
    )


def detect_bleed(mic: np.ndarray, playback: np.ndarray) -> float:
    """How much of the backing track is leaking into the microphone, 0..1.

    This is the headphone check. On open speakers the microphone hears the song
    and the pitch detector tracks the recording rather than the singer — the app
    would grade the record. High correlation here means "put headphones on",
    and it is worth saying loudly, because everything downstream silently
    becomes meaningless.
    """
    n = min(len(mic), len(playback))
    if n < 1024:
        return 0.0
    a = mic[:n] - np.mean(mic[:n])
    b = playback[:n] - np.mean(playback[:n])
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    return float(np.clip(abs(np.dot(a, b)) / denom, 0.0, 1.0))
