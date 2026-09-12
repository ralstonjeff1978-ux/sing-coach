"""CTC forced alignment: finding where known words actually occur.

This exists because transcription and alignment are different problems, and
using one tool for both was the wrong call.

Whisper answers "what was sung". It is good at that. Asking it "when was each
word sung" gets a guess, because it was trained on speech — where syllables are
short and consonants mark boundaries — and singing has neither. Measured on a
real track, barely half its word onsets fell anywhere near the singer actually
phonating.

Forced alignment answers the other question directly. Given the audio *and* the
text, an acoustic model emits a probability for every character at every 20 ms
frame, and Viterbi finds the single most likely path through those frames that
spells out exactly the text we already know. The timings come out as a
by-product, and they are measurements rather than guesses.

Two consequences worth having:

* Word timings get dramatically better, because the model is not choosing the
  words — only placing them.
* If the user supplies correct lyrics (by editing the ``.lrc``), alignment uses
  *those*. The words are then right by construction, and their timings are
  derived from the audio. Nothing is left to a transcriber's opinion.

Runs on ONNX Runtime, so it uses the same DirectML acceleration as separation
and needs no PyTorch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

ProgressFn = Callable[[str, float], None]


def _noop(_s: str, _f: float) -> None:
    pass


#: wav2vec2 works at 16 kHz.
ALIGN_SR = 16000

#: The model's output vocabulary, in index order. Index 0 is the CTC blank,
#: and "|" is the word separator. This ordering is wav2vec2-base-960h's own —
#: it is frequency-ordered rather than alphabetical, and getting it wrong would
#: silently align to the wrong letters.
VOCAB = [
    "<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H",
    "S", "R", "D", "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V",
    "K", "'", "X", "J", "Q", "Z",
]
BLANK = 0
#: wav2vec2-base emits one frame per 320 audio samples at 16 kHz — 20 ms.
FRAME_STRIDE = 320

#: CTC reports an onset once the model is confident, which is slightly after
#: the sound actually starts. Measured against a fixture with known word
#: times, every one of fifteen words came in late, by a median of 73 ms and
#: never early. Removing that bias is a calibration, not a fudge — the sign is
#: consistent because the cause is.
ONSET_BIAS_S = 0.070

_TOKEN = {ch: i for i, ch in enumerate(VOCAB)}


@dataclass
class AlignedWord:
    text: str
    start: float
    end: float
    score: float


class AlignmentFailed(RuntimeError):
    pass


def normalise(text: str) -> str:
    """Reduce text to the model's alphabet: A-Z, apostrophe, word separator."""
    text = text.upper().replace("’", "'")
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ForcedAligner:
    def __init__(self, model_path: Path, *, prefer_gpu: bool = True) -> None:
        available = ort.get_available_providers()
        wanted = ["DmlExecutionProvider", "CPUExecutionProvider"] if prefer_gpu else ["CPUExecutionProvider"]
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        if providers[0] == "DmlExecutionProvider":
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(str(model_path), options, providers=providers)
        self.provider = self.session.get_providers()[0]
        self._input = self.session.get_inputs()[0].name
        self._output = self.session.get_outputs()[0].name

    # -- emissions --------------------------------------------------------

    def emissions(
        self, audio: np.ndarray, chunk_s: float = 20.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame log probabilities and the absolute time of each frame.

        Returns ``(log_probs (frames, vocab), times (frames,))``.

        Both the obvious approaches are wrong, and each failed differently:

        * **One pass over the whole song.** wav2vec2 was trained on utterances
          of tens of seconds. Given 14,664 frames it degraded completely and
          emitted the CTC blank on *every* frame — the same model handled a
          15-second slice perfectly.
        * **Chunking and concatenating frames.** The convolutional front end
          consumes part of each chunk's edges, so a chunk yields slightly fewer
          than N/320 frames. Assuming a fixed frame rate then accumulates a
          time deficit that grows through the song.

        So we chunk *and* carry each chunk's real time base: every frame's
        absolute time is computed from its own chunk's audio offset and that
        chunk's measured frame count. Boundary effects stay local instead of
        compounding.
        """
        if len(audio) < FRAME_STRIDE * 4:
            raise AlignmentFailed("Audio too short to align.")

        chunk = int(chunk_s * ALIGN_SR)
        parts: list[np.ndarray] = []
        times: list[np.ndarray] = []

        for start in range(0, len(audio), chunk):
            segment = audio[start : start + chunk]
            if len(segment) < FRAME_STRIDE * 4:
                break
            logits = self.session.run(
                [self._output], {self._input: segment[None, :].astype(np.float32)}
            )[0][0]
            n_frames = logits.shape[0]
            span = len(segment) / ALIGN_SR
            offset = start / ALIGN_SR
            parts.append(logits)
            times.append(offset + (np.arange(n_frames) + 0.5) * (span / n_frames))

        if not parts:
            raise AlignmentFailed("Audio too short to align.")

        logits = np.concatenate(parts, axis=0)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        return log_probs, np.concatenate(times)

    # -- Viterbi ----------------------------------------------------------

    @staticmethod
    def _viterbi(log_probs: np.ndarray, tokens: list[int]) -> np.ndarray:
        """Most likely alignment of ``tokens`` to frames. Returns frame index
        per token.

        Standard CTC forced alignment: the target sequence is interleaved with
        blanks, and each frame may stay on the current symbol or advance. Run
        in log space, which is what keeps a five-minute song from underflowing
        to zero probability.
        """
        # Interleave blanks: _ t1 _ t2 _ ... _
        extended = [BLANK]
        for token in tokens:
            extended.extend([token, BLANK])
        n_states = len(extended)
        n_frames = log_probs.shape[0]

        if n_frames < n_states:
            raise AlignmentFailed(
                f"Not enough audio ({n_frames} frames) for {len(tokens)} characters."
            )

        neg_inf = -1e30
        scores = np.full(n_states, neg_inf, dtype=np.float64)
        scores[0] = log_probs[0, extended[0]]
        if n_states > 1:
            scores[1] = log_probs[0, extended[1]]

        backpointers = np.zeros((n_frames, n_states), dtype=np.int8)

        for t in range(1, n_frames):
            frame = log_probs[t]
            stay = scores
            step = np.concatenate(([neg_inf], scores[:-1]))
            # A skip over a blank is allowed only between two *different* labels.
            skip = np.full(n_states, neg_inf)
            if n_states > 2:
                allowed = np.array(
                    [
                        s >= 2 and extended[s] != BLANK and extended[s] != extended[s - 2]
                        for s in range(n_states)
                    ]
                )
                candidate = np.concatenate(([neg_inf, neg_inf], scores[:-2]))
                skip = np.where(allowed, candidate, neg_inf)

            options = np.vstack([stay, step, skip])
            choice = np.argmax(options, axis=0)
            backpointers[t] = choice
            scores = options[choice, np.arange(n_states)] + frame[extended]

        # Finish on the last real symbol or the trailing blank.
        state = n_states - 1 if scores[-1] >= scores[-2] else n_states - 2
        path = np.zeros(n_frames, dtype=np.int32)
        for t in range(n_frames - 1, -1, -1):
            path[t] = state
            state -= int(backpointers[t][state])

        # Frame at which each original (non-blank) token begins.
        onsets = np.zeros(len(tokens), dtype=np.int64)
        seen = -1
        for t, state in enumerate(path):
            if state % 2 == 1:                       # odd states are real tokens
                index = state // 2
                if index > seen:
                    onsets[index] = t
                    seen = index
        return onsets

    # -- public -----------------------------------------------------------

    def align(
        self,
        audio: np.ndarray,
        text: str,
        *,
        offset_s: float = 0.0,
        progress: ProgressFn = _noop,
    ) -> list[AlignedWord]:
        cleaned = normalise(text)
        if not cleaned:
            return []

        progress("running acoustic model", 0.1)
        log_probs, frame_times = self.emissions(audio)

        transcript = cleaned.replace(" ", "|")
        tokens = [_TOKEN[ch] for ch in transcript if ch in _TOKEN]
        if not tokens:
            return []

        progress("aligning", 0.7)
        onsets = self._viterbi(log_probs, tokens)

        def at(frame: int) -> float:
            return float(frame_times[min(int(frame), len(frame_times) - 1)])

        words: list[AlignedWord] = []
        cursor = 0
        for word in cleaned.split(" "):
            length = len(word)
            if length == 0 or cursor + length > len(onsets):
                cursor += length + 1
                continue
            start = max(0.0, at(onsets[cursor]) - ONSET_BIAS_S)
            # End at the following separator's onset where possible, so a word
            # occupies the time up to the next one rather than a guessed span.
            follower = cursor + length
            end = at(onsets[follower]) if follower < len(onsets) else start + 0.35
            words.append(
                AlignedWord(
                    text=word,
                    start=round(start + offset_s, 3),
                    end=round(max(end, start + 0.05) + offset_s, 3),
                    score=1.0,
                )
            )
            cursor += length + 1

        progress("aligned", 1.0)
        return words


def load_audio(path: Path) -> np.ndarray:
    """Read a stem as 16 kHz mono, normalised the way wav2vec2 expects.

    The normalisation is zero-mean, unit-variance — not peak normalisation.
    That distinction is not cosmetic: feeding peak-normalised audio put the
    input outside the distribution the model was trained on, and it responded
    by emitting the CTC blank on 100% of frames. Every word then collapsed to
    the start of the song. A model that hears nothing fails silently, which is
    exactly why the diagnostic that counts non-blank frames is worth keeping.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != ALIGN_SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=ALIGN_SR)
    mono = mono - mono.mean()
    std = float(mono.std())
    return (mono / (std + 1e-7)).astype(np.float32)
