"""Generate a sung phrase whose word timings we chose ourselves.

Built for the same reason as the melody fixture: you cannot judge an aligner
against a real recording, because you would be comparing one estimate to
another. Here the answer is known in advance because we placed every word.

Windows ships a speech synthesiser (SAPI), which gives us real phonetic content
rather than the tones used elsewhere. Each word is rendered separately and
placed at a time of our choosing, with musically realistic gaps — including a
couple of deliberately long held notes, which is exactly what defeats a
speech-trained model's guess at word boundaries.

    python tests/make_align_fixture.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000

#: (word, start_seconds, spoken_duration_hint)
#: Deliberately uneven: short runs of words, long gaps, and two sustained
#: words standing alone the way a held note does in a ballad.
SCRIPT: list[tuple[str, float]] = [
    ("hello", 1.00),
    ("there", 1.55),
    ("my", 2.10),
    ("friend", 2.45),
    # long rest — an instrumental gap
    ("sing", 5.00),
    ("with", 5.50),
    ("me", 5.90),
    ("now", 6.40),
    # a held word alone
    ("stay", 9.00),
    # another cluster
    ("the", 12.00),
    ("morning", 12.35),
    ("light", 13.00),
    ("is", 13.50),
    ("gone", 13.90),
    ("away", 14.50),
]

TOTAL_S = 17.0


def synth_word(word: str, out: Path) -> bool:
    """Render one word to a wav with the Windows speech synthesiser."""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = -2; "
        f"$s.SetOutputToWaveFile('{out}'); "
        f"$s.Speak('{word}'); "
        "$s.Dispose()"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and out.exists() and out.stat().st_size > 1000


def main(outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    timeline = np.zeros(int(TOTAL_S * SR), dtype=np.float32)
    truth: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        for word, start in SCRIPT:
            wav = Path(tmp) / f"{word}.wav"
            if not synth_word(word, wav):
                print(f"Speech synthesis unavailable (failed on {word!r}).")
                return 1

            audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if sr != SR:
                import librosa

                mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)

            # Trim leading/trailing silence so the word starts where we say.
            envelope = np.abs(mono)
            loud = np.flatnonzero(envelope > envelope.max() * 0.02)
            if loud.size:
                mono = mono[loud[0] : loud[-1] + 1]

            at = int(start * SR)
            n = min(len(mono), len(timeline) - at)
            if n <= 0:
                continue
            timeline[at : at + n] += mono[:n]
            truth.append(
                {
                    "word": word.upper(),
                    "start": round(start, 3),
                    "end": round(start + n / SR, 3),
                }
            )

    peak = float(np.max(np.abs(timeline))) or 1.0
    timeline = timeline / peak * 0.9

    sf.write(outdir / "align_fixture.wav", timeline, SR, subtype="FLOAT")
    (outdir / "align_truth.json").write_text(
        json.dumps(
            {
                "sample_rate": SR,
                "duration_s": TOTAL_S,
                "text": " ".join(w for w, _ in SCRIPT),
                "words": truth,
            },
            indent=2,
        ),
        "utf-8",
    )
    print(f"Wrote alignment fixture to {outdir}")
    print(f"  {len(truth)} words over {TOTAL_S:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).parent / "fixtures"))
