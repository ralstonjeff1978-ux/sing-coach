"""Generate a synthetic test song with known ground truth.

Real music is useless as a test fixture: you cannot assert that the melody
extractor is right when you do not independently know what the melody was. So
we synthesise a "song" whose every note we chose ourselves, and write:

    fixture_vocal.wav   the lead line alone      (ideal separation output)
    fixture_backing.wav the chord bed alone      (ideal separation output)
    fixture_mix.mp3     the two summed, encoded  (what the app imports)
    fixture_truth.json  every note, exactly

That gives us ground truth for two different stages at once:

* melody extraction — run pYIN on the mix's separated vocal and compare the
  recovered notes against ``fixture_truth.json``
* separation quality — compare the separated stems against the true stems

The lead deliberately includes the things that break naive pitch trackers:
vibrato, a slow portamento slide between two notes, a wide leap, a rest, and a
quiet note near the noise floor.

Usage:  python tests/make_fixture.py [outdir]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
A4 = 440.0


def midi_to_hz(midi: float) -> float:
    return A4 * 2.0 ** ((midi - 69) / 12.0)


# (midi note, duration in seconds, label). None = rest.
# A singable phrase in C major sitting in a comfortable baritone/tenor range.
MELODY: list[tuple[float | None, float, str]] = [
    (60, 0.60, "C4 plain"),
    (62, 0.40, "D4 short"),
    (64, 0.80, "E4 vibrato"),
    (65, 0.50, "F4"),
    (None, 0.35, "rest"),
    (67, 1.00, "G4 vibrato long"),
    (64, 0.45, "E4"),
    (72, 0.70, "C5 leap up"),
    (None, 0.30, "rest"),
    (69, 0.90, "A4 slide-from"),   # portamento into the next note
    (65, 0.75, "F4 slide-to"),
    (60, 1.20, "C4 quiet ending"),
]

#: Notes that get vibrato, by index into MELODY.
VIBRATO_NOTES = {2, 5, 9}
#: Note index that slides into the following note.
SLIDE_FROM = 9
#: Note index rendered quietly, to exercise the noise gate.
QUIET_NOTE = 11

VIBRATO_RATE_HZ = 5.5     # typical human vibrato is 5–7 Hz
VIBRATO_DEPTH_CENTS = 40.0


def _adsr(n: int, attack=0.02, release=0.06) -> np.ndarray:
    """Gentle envelope so notes do not click at the boundaries."""
    env = np.ones(n, dtype=np.float64)
    a = min(int(attack * SR), n // 2)
    r = min(int(release * SR), n // 2)
    if a:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def _voice_wave(freq_curve: np.ndarray) -> np.ndarray:
    """A vowel-ish tone: fundamental plus decaying harmonics.

    A pure sine is unrealistically easy for a pitch tracker. Harmonics make the
    fixture a fair test — and give the separator something to actually separate.
    """
    phase = 2.0 * np.pi * np.cumsum(freq_curve) / SR
    harmonics = [(1, 1.00), (2, 0.42), (3, 0.24), (4, 0.13), (5, 0.07), (6, 0.04)]
    out = np.zeros_like(phase)
    for n, amp in harmonics:
        out += amp * np.sin(n * phase)
    return out / sum(a for _, a in harmonics)


def build_lead() -> tuple[np.ndarray, list[dict]]:
    chunks: list[np.ndarray] = []
    truth: list[dict] = []
    t = 0.0

    for i, (midi, dur, label) in enumerate(MELODY):
        n = int(dur * SR)
        if midi is None:
            chunks.append(np.zeros(n))
            t += dur
            continue

        base = midi_to_hz(midi)
        freq = np.full(n, base, dtype=np.float64)

        if i in VIBRATO_NOTES:
            # Delay vibrato onset slightly, the way a singer actually does it.
            onset = int(0.15 * SR)
            tt = np.arange(n) / SR
            depth = np.clip((np.arange(n) - onset) / (0.2 * SR), 0.0, 1.0)
            cents = VIBRATO_DEPTH_CENTS * depth * np.sin(2 * np.pi * VIBRATO_RATE_HZ * tt)
            freq *= 2.0 ** (cents / 1200.0)

        if i == SLIDE_FROM:
            # Portamento over the last 25% into the next note's pitch.
            nxt = MELODY[i + 1][0]
            if nxt is not None:
                k = int(n * 0.75)
                glide = np.linspace(0.0, nxt - midi, n - k)
                freq[k:] = midi_to_hz(midi + glide)

        amp = 0.12 if i == QUIET_NOTE else 0.5
        chunks.append(_voice_wave(freq) * _adsr(n) * amp)

        truth.append(
            {
                "index": i,
                "start": round(t, 4),
                "end": round(t + dur, 4),
                "midi": midi,
                "hz": round(base, 3),
                "label": label,
                "vibrato": i in VIBRATO_NOTES,
                "slide": i == SLIDE_FROM,
                "quiet": i == QUIET_NOTE,
            }
        )
        t += dur

    return np.concatenate(chunks), truth


# Chord bed: I - V - vi - IV, the backing the lead sits on top of.
CHORDS = [
    ([48, 55, 64], 2.0),   # C
    ([43, 50, 59], 1.85),  # G
    ([45, 52, 60], 2.0),   # Am
    ([41, 48, 57], 2.0),   # F
]


def build_backing(total_samples: int) -> np.ndarray:
    out = np.zeros(total_samples)
    pos = 0
    while pos < total_samples:
        for notes, dur in CHORDS:
            n = min(int(dur * SR), total_samples - pos)
            if n <= 0:
                break
            seg = np.zeros(n)
            for midi in notes:
                f = midi_to_hz(midi)
                tt = np.arange(n) / SR
                # Slightly detuned saw-ish pad — harmonically dense, so the
                # separator has real work to do.
                seg += np.sin(2 * np.pi * f * tt) * 0.6
                seg += np.sin(2 * np.pi * f * 2 * tt) * 0.2
                seg += np.sin(2 * np.pi * f * 1.005 * tt) * 0.3
            seg *= _adsr(n, attack=0.05, release=0.15) / (len(notes) * 1.1)
            out[pos:pos + n] += seg * 0.35
            pos += n
            if pos >= total_samples:
                break
    return out


def main(outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)

    lead, truth = build_lead()
    backing = build_backing(len(lead))

    # Two seconds of backing-only lead-in, so the fixture starts without a
    # vocal — exactly the "no target here" case the highway must handle.
    lead_in = int(2.0 * SR)
    lead = np.concatenate([np.zeros(lead_in), lead])
    backing = build_backing(len(lead))
    for note in truth:
        note["start"] = round(note["start"] + 2.0, 4)
        note["end"] = round(note["end"] + 2.0, 4)

    mix = lead + backing
    peak = np.max(np.abs(mix))
    if peak > 0.99:
        mix *= 0.99 / peak

    def stereo(x: np.ndarray) -> np.ndarray:
        return np.column_stack([x, x]).astype(np.float32)

    sf.write(outdir / "fixture_vocal.wav", stereo(lead), SR, subtype="FLOAT")
    sf.write(outdir / "fixture_backing.wav", stereo(backing), SR, subtype="FLOAT")
    mix_wav = outdir / "fixture_mix.wav"
    sf.write(mix_wav, stereo(mix), SR, subtype="FLOAT")

    (outdir / "fixture_truth.json").write_text(
        json.dumps(
            {
                "sample_rate": SR,
                "duration_s": round(len(lead) / SR, 4),
                "lead_in_s": 2.0,
                "vibrato_rate_hz": VIBRATO_RATE_HZ,
                "vibrato_depth_cents": VIBRATO_DEPTH_CENTS,
                "notes": truth,
            },
            indent=2,
        ),
        "utf-8",
    )

    # Encode to MP3 so the import path gets exercised on a lossy container,
    # which is what the user will actually be feeding it.
    mp3 = outdir / "fixture_mix.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(mix_wav), "-c:a", "libmp3lame", "-b:a", "192k",
         "-metadata", "title=SingCoach Test Fixture",
         "-metadata", "artist=Synthetic",
         str(mp3)],
        check=True,
    )

    print(f"Wrote fixture to {outdir}")
    print(f"  duration {len(lead) / SR:.2f}s, {len(truth)} notes")
    for p in sorted(outdir.glob("fixture_*")):
        print(f"  {p.name:<24}{p.stat().st_size / 1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fixtures"
    raise SystemExit(main(target))
