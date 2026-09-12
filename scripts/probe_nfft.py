"""Determine the separation model's true n_fft by measurement.

Run once per model. The result is what gets hardcoded in the pipeline.
See the docstring on singcoach.analysis.separate.infer_n_fft for the reasoning.

    python scripts/probe_nfft.py <song-hash> [start_s] [dur_s]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from singcoach import modelstore  # noqa: E402
from singcoach.analysis.separate import MDXParams, MDXSeparator, load_mix  # noqa: E402
from singcoach.config import SAMPLE_RATE  # noqa: E402
from singcoach.library import cache  # noqa: E402


def score(vocals: np.ndarray) -> dict[str, float]:
    """Metrics that distinguish a real separation from a smeared one."""
    env = np.abs(vocals).mean(axis=0)
    frame = SAMPLE_RATE // 20  # 50 ms
    usable = (len(env) // frame) * frame
    energy = env[:usable].reshape(-1, frame).mean(axis=1)

    lo = float(np.percentile(energy, 10)) + 1e-9
    hi = float(np.percentile(energy, 90))
    return {
        "contrast": hi / lo,          # decisive separation -> large
        "p10": lo,                    # near-silence floor between phrases
        "p90": hi,
        "rms": float(np.sqrt(np.mean(vocals ** 2))),
        "peak": float(np.max(np.abs(vocals))),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    hash_ = sys.argv[1]
    start_s = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    dur_s = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    paths = cache.paths_for(hash_)
    if not paths.source_wav.exists():
        print(f"No decoded audio for {hash_}. Import it first.")
        return 1

    model = modelstore.ensure("separator")
    mix = load_mix(paths.source_wav)
    a, b = int(start_s * SAMPLE_RATE), int((start_s + dur_s) * SAMPLE_RATE)
    excerpt = np.ascontiguousarray(mix[:, a:b])
    print(f"Probing on {dur_s:.0f}s from {start_s:.0f}s  ({excerpt.shape[1]} samples)\n")

    results = {}
    for n_fft in (7680, 6144):
        sep = MDXSeparator(model, MDXParams(n_fft=n_fft), denoise=False)
        t0 = time.perf_counter()
        vocals, accomp = sep.separate(excerpt)
        elapsed = time.perf_counter() - t0
        m = score(vocals)
        results[n_fft] = m
        print(f"n_fft={n_fft}  provider={sep.provider}  {elapsed:.1f}s "
              f"({dur_s / elapsed:.1f}x realtime)")
        print(f"   contrast {m['contrast']:9.1f}   (higher = cleaner separation)")
        print(f"   p10      {m['p10']:9.6f}   p90 {m['p90']:9.6f}")
        print(f"   rms      {m['rms']:9.6f}   peak {m['peak']:9.4f}")
        # Sanity: stems must sum back to the mix.
        err = float(np.max(np.abs((vocals + accomp) - excerpt)))
        print(f"   reconstruction error {err:.2e}\n")

    best = max(results, key=lambda k: results[k]["contrast"])
    ratio = results[best]["contrast"] / max(
        results[k]["contrast"] for k in results if k != best
    )
    print(f"=> n_fft={best} wins by {ratio:.2f}x contrast")
    if ratio < 1.15:
        print("   WARNING: margin is small. Listen to both before trusting this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
