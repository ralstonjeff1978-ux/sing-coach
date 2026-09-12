"""High-quality vocal separation with a waveform-domain ONNX model.

This is the optional 2026-class upgrade over the shipped MDX-Net model. It drives
an **HTDemucs** or **BS-Roformer / Mel-Roformer** ONNX export that works in the
*time domain*: it takes a stereo waveform chunk and returns separated stems as
waveforms, so — unlike the MDX path in :mod:`singcoach.analysis.separate` — there
is no STFT/iSTFT to run here; the model does its own spectral work internally.

    mix (2, N) float32
      -> overlapping fixed-length chunks (1, 2, segment)
      -> ONNX  ->  stems (1, S, 2, segment)   [drums, bass, other, vocals]
      -> take the vocals row, overlap-add back to (2, N)
      -> accompaniment = mix - vocals

Contract of the export we target (HTDemucs-FT ONNX, and the same shape family
used by waveform BS-Roformer exports):

    input   name "mix"    shape (1, 2, 343980)  float32, 44.1 kHz, [-1, 1]
    output  name "stems"  shape (1, 4, 2, 343980) float32
            stem order [drums, bass, other, vocals]  -> vocals is index 3

Segment length and stem layout are read off the model / config rather than
hardcoded, so a 2-stem vocal Roformer export (output (1, 2, 2, seg) with vocals
at index 0) works through the same code by setting ``num_stems``/``vocals_index``.

Why accompaniment is ``mix - vocals`` rather than a second model output: the two
stems then sum back to the original exactly, which is what lets the app crossfade
between "karaoke" and "sing with the original" with no phase artefact. This
matches the guarantee the MDX path makes, so the rest of the pipeline — melody
extraction and the live pitch overlay — is unaffected by which backend produced
the vocal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

from .separate import SeparationError

ProgressFn = Callable[[str, float], None]


def _noop(_s: str, _f: float) -> None:
    pass


@dataclass(frozen=True)
class WaveformParams:
    """Geometry of a waveform-domain separation model.

    Defaults describe the HTDemucs-FT ONNX export (7.8 s segment at 44.1 kHz,
    four stems in Demucs order). ``segment_samples`` is normally overridden by
    the value read from the ONNX input shape at load time.
    """

    #: Fixed input length in samples. 343980 = 7.8 s @ 44.1 kHz (HTDemucs-FT).
    segment_samples: int = 343980
    #: Overlap fraction between consecutive chunks (Demucs default is 0.25).
    overlap: float = 0.25
    #: Which stem row is the vocal. Demucs order [drums, bass, other, vocals].
    vocals_index: int = 3
    #: Total stems the model emits. 4 for HTDemucs, 2 for a vocal Roformer.
    num_stems: int = 4

    @property
    def hop(self) -> int:
        step = int(round(self.segment_samples * (1.0 - self.overlap)))
        return max(1, min(step, self.segment_samples))


def _pick_providers(prefer_gpu: bool) -> list[str]:
    available = ort.get_available_providers()
    wanted = (
        ["DmlExecutionProvider", "CPUExecutionProvider"]
        if prefer_gpu
        else ["CPUExecutionProvider"]
    )
    return [p for p in wanted if p in available] or ["CPUExecutionProvider"]


class RoformerSeparator:
    """Run a waveform-domain ONNX stem model (HTDemucs / BS-Roformer class).

    ``session`` exists for tests: pass a duck-typed ONNX session (anything with
    ``get_inputs``/``get_outputs``/``get_providers``/``run``) to exercise the
    chunking and overlap-add without a real multi-GB model on disk. In normal
    use it is ``None`` and a real :class:`onnxruntime.InferenceSession` is built
    from ``model_path``.
    """

    #: Default backend id; the instance value is set from ``backend_name``.
    backend: str = "htdemucs"

    def __init__(
        self,
        model_path: Path,
        params: WaveformParams | None = None,
        *,
        prefer_gpu: bool = True,
        backend_name: str = "htdemucs",
        session: "ort.InferenceSession | None" = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.params = params or WaveformParams()
        self.backend = backend_name

        if session is None:
            if not self.model_path.exists():
                raise SeparationError(f"Model not found: {self.model_path}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = _pick_providers(prefer_gpu)
            if providers[0] == "DmlExecutionProvider":
                # DirectML manages its own queue; ORT's own threading fights it.
                opts.enable_mem_pattern = False
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            try:
                self.session = ort.InferenceSession(
                    str(self.model_path), opts, providers=providers
                )
            except Exception as exc:  # noqa: BLE001 - ORT raises bare Exceptions
                raise SeparationError(
                    f"Could not load {self.model_path.name}: {exc}"
                ) from exc
        else:
            self.session = session

        self.provider = self.session.get_providers()[0]
        self._input = self.session.get_inputs()[0].name
        self._output = self.session.get_outputs()[0].name

        # Trust the model's own declared input length over the configured guess:
        # feeding a waveform model the wrong segment size produces silent garbage.
        seg = self._static_segment_length()
        if seg:
            self.params = replace(self.params, segment_samples=seg)
        if self.params.segment_samples <= 0:
            raise SeparationError(
                "Could not determine the model's input length; set "
                "SeparatorConfig.hq_segment_samples explicitly."
            )

    # -- introspection ------------------------------------------------------

    @property
    def on_gpu(self) -> bool:
        return self.provider == "DmlExecutionProvider"

    def _static_segment_length(self) -> int | None:
        """The fixed sample count from the ONNX input shape, if it is static."""
        shape = list(self.session.get_inputs()[0].shape)
        if not shape:
            return None
        last = shape[-1]
        return int(last) if isinstance(last, int) and last > 0 else None

    # -- inference ----------------------------------------------------------

    def _vocals_from_output(self, out: np.ndarray) -> np.ndarray:
        """Pull the (2, segment) vocal chunk out of one model output.

        Accepts the common export shapes:
          (1, S, 2, T) stems tensor  -> row ``vocals_index``
          (1, 2, T)    a single stem -> taken as the vocal directly
          (S, 2, T) / (2, T)         -> batch dim already squeezed
        """
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 4:  # (B, S, 2, T)
            return arr[0, self.params.vocals_index]
        if arr.ndim == 3:
            # Either (B, 2, T) single stem or (S, 2, T) stems without batch.
            if arr.shape[0] == 2 and arr.shape[1] != 2:
                return arr  # (2, T)
            if arr.shape[1] == 2 and arr.shape[0] not in (2,):
                return arr[self.params.vocals_index]  # (S, 2, T)
            # Ambiguous 2xT vs Sx2xT where S==2: prefer the single-stem reading.
            return arr if arr.shape[0] == 2 else arr[self.params.vocals_index]
        if arr.ndim == 2:  # (2, T)
            return arr
        raise SeparationError(f"Unexpected model output rank {arr.ndim}: {arr.shape}")

    def separate(
        self,
        mix: np.ndarray,
        *,
        progress: ProgressFn = _noop,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split ``mix`` (2, N) float32 into (vocals, accompaniment)."""
        if mix.ndim != 2 or mix.shape[0] != 2:
            raise SeparationError(f"Expected a (2, N) stereo array, got {mix.shape}.")

        mix = np.ascontiguousarray(mix, dtype=np.float32)
        n = mix.shape[1]
        seg = self.params.segment_samples
        hop = self.params.hop

        # Triangular fade so chunk edges — where the model is least sure —
        # contribute less to the overlap. Dividing by the accumulated weight
        # makes the reconstruction correct for any positive window; the floor
        # keeps the very first/last sample from dividing by zero.
        window = np.bartlett(seg).astype(np.float32) if seg > 1 else np.ones(seg, np.float32)
        window += 1e-3

        vocals = np.zeros((2, n), dtype=np.float32)
        weight = np.zeros(n, dtype=np.float32)

        starts = list(range(0, max(n, 1), hop))
        for i, s in enumerate(starts):
            end = min(s + seg, n)
            valid = end - s
            chunk = np.zeros((1, 2, seg), dtype=np.float32)
            chunk[0, :, :valid] = mix[:, s:end]

            out = self.session.run([self._output], {self._input: chunk})[0]
            voc = self._vocals_from_output(out)  # (2, seg)
            voc = np.asarray(voc, dtype=np.float32)
            if voc.shape[-1] < valid:
                raise SeparationError(
                    f"Model returned {voc.shape[-1]} samples for a {seg}-sample "
                    "chunk; the export is not waveform-domain as configured."
                )

            w = window[:valid]
            vocals[:, s:end] += voc[:, :valid] * w
            weight[s:end] += w
            progress("separating", min((i + 1) / len(starts), 1.0))

        weight[weight == 0.0] = 1.0
        vocals /= weight
        np.clip(vocals, -1.0, 1.0, out=vocals)
        accompaniment = (mix - vocals).astype(np.float32)
        return vocals.astype(np.float32), accompaniment
