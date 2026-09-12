"""Vocal/accompaniment separation with an MDX-Net ONNX model.

Why this is hand-rolled rather than pulled from a library: every published
wrapper for these models depends on PyTorch (plus onnx2torch and friends) —
roughly 2.5 GB, most of which exists to support CUDA we do not have. The actual
inference is a short, well-understood piece of DSP, so we do it directly against
ONNX Runtime and keep the DirectML path that makes the Radeon useful.

How MDX-Net works, briefly:

    audio -> STFT -> [L.real, L.imag, R.real, R.imag] -> UNet -> masked spectrum
          -> inverse STFT -> the isolated stem

The network sees a fixed tile of 3072 frequency bins x 256 time frames, so the
song is processed in overlapping chunks. Each chunk is padded by half an FFT on
both sides and those edges are discarded afterwards, which is what keeps chunk
boundaries from ticking.

The residual (mix minus vocal) becomes the accompaniment. Doing it by
subtraction rather than running a second model means the two stems sum back to
the original exactly, which matters: the app crossfades between them and any
mismatch would be audible as a phase artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

from ..config import SAMPLE_RATE

ProgressFn = Callable[[str, float], None]


def _noop(_s: str, _f: float) -> None:
    pass


@dataclass(frozen=True)
class MDXParams:
    """Geometry of an MDX-Net checkpoint.

    ``dim_f`` and ``dim_t`` are readable straight off the ONNX input shape.
    ``n_fft`` is not — it is a property of how the model was trained, and the
    published values differ per checkpoint (7680 for the vocal models, 6144 for
    several instrumental ones). Getting it wrong does not crash: it produces
    plausible-looking audio that is subtly, badly wrong. See
    :func:`infer_n_fft`, which determines it by measurement instead of trust.
    """

    n_fft: int = 7680
    dim_f: int = 3072
    dim_t: int = 256
    hop: int = 1024
    #: MDX models output a slightly quiet stem; UVR ships a per-model scalar.
    compensate: float = 1.021

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def chunk_size(self) -> int:
        # center=True STFT of this length yields exactly dim_t frames.
        return self.hop * (self.dim_t - 1)

    @property
    def trim(self) -> int:
        return self.n_fft // 2

    @property
    def gen_size(self) -> int:
        """Usable samples per chunk once both padded edges are discarded."""
        return self.chunk_size - 2 * self.trim


class SeparationError(RuntimeError):
    pass


class MDXSeparator:
    #: Stable identifier for this backend, surfaced in the UI/CLI and used by
    #: :mod:`singcoach.analysis.separator` to report which model actually ran.
    backend: str = "mdx-net"

    def __init__(
        self,
        model_path: Path,
        params: MDXParams | None = None,
        *,
        prefer_gpu: bool = True,
        batch_size: int = 1,
        denoise: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise SeparationError(f"Model not found: {self.model_path}")

        self.params = params or MDXParams()
        self.batch_size = max(1, batch_size)
        self.denoise = denoise

        available = ort.get_available_providers()
        wanted = ["DmlExecutionProvider", "CPUExecutionProvider"] if prefer_gpu else ["CPUExecutionProvider"]
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # DirectML dislikes ORT's own multithreading; it manages its own queue.
        if providers[0] == "DmlExecutionProvider":
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(str(self.model_path), opts, providers=providers)
        self.provider = self.session.get_providers()[0]
        self._input = self.session.get_inputs()[0].name
        self._output = self.session.get_outputs()[0].name

        shape = self.session.get_inputs()[0].shape
        if list(shape[1:]) != [4, self.params.dim_f, self.params.dim_t]:
            raise SeparationError(
                f"Model expects {shape[1:]} but params describe "
                f"[4, {self.params.dim_f}, {self.params.dim_t}]."
            )

    @property
    def on_gpu(self) -> bool:
        return self.provider == "DmlExecutionProvider"

    # -- spectral transforms ------------------------------------------------

    def _stft(self, waves: np.ndarray) -> np.ndarray:
        """(B, 2, chunk) float32 -> (B, 4, dim_f, dim_t) float32."""
        p = self.params
        spec = librosa.stft(
            waves, n_fft=p.n_fft, hop_length=p.hop,
            window="hann", center=True, pad_mode="reflect",
        )  # (B, 2, n_bins, dim_t)
        spec = spec[..., : p.dim_f, :]
        b = waves.shape[0]
        out = np.empty((b, 4, p.dim_f, p.dim_t), dtype=np.float32)
        # Channel order must match training: L.re, L.im, R.re, R.im.
        out[:, 0] = spec[:, 0].real
        out[:, 1] = spec[:, 0].imag
        out[:, 2] = spec[:, 1].real
        out[:, 3] = spec[:, 1].imag
        return out

    def _istft(self, spec: np.ndarray) -> np.ndarray:
        """(B, 4, dim_f, dim_t) float32 -> (B, 2, chunk) float32."""
        p = self.params
        b = spec.shape[0]
        full = np.zeros((b, 2, p.n_bins, p.dim_t), dtype=np.complex64)
        full[:, 0, : p.dim_f] = spec[:, 0] + 1j * spec[:, 1]
        full[:, 1, : p.dim_f] = spec[:, 2] + 1j * spec[:, 3]
        wave = librosa.istft(
            full, hop_length=p.hop, n_fft=p.n_fft,
            window="hann", center=True, length=p.chunk_size,
        )
        return wave.astype(np.float32)

    # -- inference ----------------------------------------------------------

    def _run(self, spec: np.ndarray) -> np.ndarray:
        if not self.denoise:
            return self.session.run([self._output], {self._input: spec})[0]
        # Averaging the model's response to x and to -x cancels a chunk of the
        # additive artifact the network hallucinates in near-silence. Costs 2x
        # compute and is clearly worth it on a quiet intro.
        pos = self.session.run([self._output], {self._input: spec})[0]
        neg = self.session.run([self._output], {self._input: -spec})[0]
        return (pos - neg) * 0.5

    def separate(
        self,
        mix: np.ndarray,
        *,
        progress: ProgressFn = _noop,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split ``mix`` (2, N) float32 into (vocals, accompaniment)."""
        p = self.params
        if mix.ndim != 2 or mix.shape[0] != 2:
            raise SeparationError(f"Expected a (2, N) stereo array, got {mix.shape}.")

        n_samples = mix.shape[1]
        pad = p.gen_size - (n_samples % p.gen_size)
        padded = np.concatenate(
            [np.zeros((2, p.trim), np.float32), mix,
             np.zeros((2, pad + p.trim), np.float32)],
            axis=1,
        )

        starts = list(range(0, n_samples + pad, p.gen_size))
        chunks = [padded[:, s : s + p.chunk_size] for s in starts]

        out: list[np.ndarray] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = np.stack(chunks[i : i + self.batch_size]).astype(np.float32)
            spec = self._stft(batch)
            wave = self._istft(self._run(spec))
            # Discard the padded edges; only the interior is trustworthy.
            out.append(wave[:, :, p.trim : -p.trim].reshape(2, -1)
                       if wave.shape[0] == 1 else
                       np.concatenate(list(wave[:, :, p.trim : -p.trim]), axis=1))
            progress("separating", min((i + self.batch_size) / len(chunks), 1.0))

        vocals = np.concatenate(out, axis=1)[:, :n_samples] * p.compensate
        np.clip(vocals, -1.0, 1.0, out=vocals)
        accompaniment = mix - vocals
        return vocals.astype(np.float32), accompaniment.astype(np.float32)


# ---------------------------------------------------------------------------
# n_fft determination
# ---------------------------------------------------------------------------


def infer_n_fft(
    model_path: Path,
    probe_mix: np.ndarray,
    candidates: Iterable[int] = (7680, 6144),
    *,
    progress: ProgressFn = _noop,
) -> tuple[int, dict[int, float]]:
    """Pick the model's true ``n_fft`` by trying each and scoring the result.

    The wrong value still produces audio, so we cannot detect it by exception.
    Instead we exploit what a *correct* vocal separation looks like: the
    isolated vocal should be strongly bimodal in time — loud where someone is
    singing, near-silent where nobody is. A wrong transform smears energy
    everywhere and flattens that distribution.

    Score = ratio of the 90th-percentile frame energy to the 10th. Higher is a
    more decisive separation.
    """
    scores: dict[int, float] = {}
    for n_fft in candidates:
        params = MDXParams(n_fft=n_fft)
        sep = MDXSeparator(model_path, params, denoise=False)
        vocals, _ = sep.separate(probe_mix)
        env = np.abs(vocals).mean(axis=0)
        frame = SAMPLE_RATE // 20
        usable = (len(env) // frame) * frame
        energy = env[:usable].reshape(-1, frame).mean(axis=1)
        lo = np.percentile(energy, 10) + 1e-9
        hi = np.percentile(energy, 90)
        scores[n_fft] = float(hi / lo)
        progress(f"probing n_fft={n_fft}", 1.0)
    best = max(scores, key=scores.__getitem__)
    return best, scores


# ---------------------------------------------------------------------------
# file-level helpers
# ---------------------------------------------------------------------------


def load_mix(path: Path) -> np.ndarray:
    """Read a wav as (2, N) float32 at SAMPLE_RATE."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        data = librosa.resample(data.T, orig_sr=sr, target_sr=SAMPLE_RATE).T
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return np.ascontiguousarray(data[:, :2].T)


def write_stem(path: Path, audio: np.ndarray) -> None:
    """Write a (2, N) stem as 24-bit FLAC — inaudible loss, one third the size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.flac")
    sf.write(str(tmp), audio.T, SAMPLE_RATE, subtype="PCM_24", format="FLAC")
    tmp.replace(path)


def separate_song(
    source_wav: Path,
    vocals_out: Path,
    accompaniment_out: Path,
    model_path: Path,
    *,
    n_fft: int = 7680,
    prefer_gpu: bool = True,
    progress: ProgressFn = _noop,
) -> dict[str, float | str]:
    progress("loading audio", 0.0)
    mix = load_mix(source_wav)

    sep = MDXSeparator(model_path, MDXParams(n_fft=n_fft), prefer_gpu=prefer_gpu)
    started = time.perf_counter()
    vocals, accompaniment = sep.separate(mix, progress=progress)
    elapsed = time.perf_counter() - started

    progress("writing stems", 0.95)
    write_stem(vocals_out, vocals)
    write_stem(accompaniment_out, accompaniment)
    progress("done", 1.0)

    duration = mix.shape[1] / SAMPLE_RATE
    return {
        "provider": sep.provider,
        "seconds": round(elapsed, 1),
        "realtime_factor": round(duration / elapsed, 2) if elapsed else 0.0,
        "n_fft": n_fft,
    }
