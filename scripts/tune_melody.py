"""Sweep melody post-processing parameters against a cached pitch track.

pYIN costs about a minute per song; everything after it costs milliseconds. So
we run the tracker once, cache it, and then evaluate the tunable knobs
instantly. Judgement here is by measured behaviour, not by taste:

* **voiced %** — too low means real singing is being discarded, too high means
  breath and bleed are being called notes.
* **sub-E3 %** — a male lead in this repertoire almost never sings below E3, so
  a large tail down there is octave error, not singing.
* **notes** — a plausible count for a 5-minute vocal is low hundreds. Thousands
  means the segmenter is shattering sustained notes.
* **median note length** — should land near the length of a sung syllable.

    python scripts/tune_melody.py <song-hash>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singcoach.analysis import melody  # noqa: E402
from singcoach.library import cache  # noqa: E402


def summarise(m: melody.MelodyAnalysis) -> dict:
    v = m.midi[~np.isnan(m.midi)]
    durs = [n.duration for n in m.notes]
    return {
        "voiced%": 100 * m.voiced_fraction,
        "sub_e3%": 100 * float(np.mean(v < 52)) if v.size else 0.0,
        "notes": len(m.notes),
        "med_len": float(np.median(durs)) if durs else 0.0,
        "range": f"{m.range_low_midi}-{m.range_high_midi}",
        "octfix": m.octaves_fixed,
        "expr": m.expressiveness,
        "vib": sum(1 for n in m.notes if n.vibrato_rate_hz),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    paths = cache.paths_for(sys.argv[1])
    if not paths.vocals.exists():
        print("Separate the song first.")
        return 1

    # Prime the pitch-track cache (slow, once).
    print("priming pitch track cache...")
    melody.analyse_vocal(paths.vocals)
    print("done.\n")

    hdr = f"{'floor':>6} {'voiced%':>8} {'sub_e3%':>8} {'notes':>6} {'med_len':>8} {'octfix':>7} {'vib':>5} {'expr':>6}  range"
    print(hdr)
    print("-" * len(hdr))
    for floor in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        m = melody.analyse_vocal(paths.vocals, confidence_floor=floor)
        s = summarise(m)
        print(f"{floor:6.2f} {s['voiced%']:8.1f} {s['sub_e3%']:8.1f} {s['notes']:6d} "
              f"{s['med_len']:8.3f} {s['octfix']:7d} {s['vib']:5d} {s['expr']:6.3f}  {s['range']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
