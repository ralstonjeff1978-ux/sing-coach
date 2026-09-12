"""Word-level lyric timing for karaoke-style highlighting.

Transcription runs locally (faster-whisper / CTranslate2) against the
**isolated vocal stem**. That last part matters more than the model choice:
speech recognition degrades badly when a full band is playing over the voice,
and separating first buys more accuracy than any amount of model size.

The transcript is written to the song's own cache directory and never leaves
this machine.

Two refinements make this usable as a karaoke track rather than just a
transcript:

**Onset snapping.** Whisper's word timings drift on sung audio — it was trained
on speech, where syllables are short and consonants are crisp, and a held vowel
confuses its sense of where a word began. But we already know exactly where the
singer started phonating, because pYIN gave us note onsets. Snapping each word
to a nearby note onset replaces an estimate with a measurement.

**Honest line breaks.** Lines are split on the singer's own phrasing (the gaps
where they stopped singing), not on a fixed word count, so a line on screen
corresponds to a breath.

If a hand-edited ``lyrics.lrc`` sits next to the transcript it wins outright,
so any correction you make is permanent.
"""

from __future__ import annotations

import bisect
import json
import re

import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from ..config import MODELS_DIR

ProgressFn = Callable[[str, float], None]


def _noop(_s: str, _f: float) -> None:
    pass


#: Default transcription model. `.en` variants are markedly better on English
#: than the multilingual ones at the same size, and singing needs the help.
#:
#: medium.en over small.en, decided by measurement on a real song: 83s vs 27s
#: to transcribe, but 19 low-confidence words instead of 26, and — the telling
#: figure — its raw word onsets landed 92ms from the measured vocal onsets
#: versus 128ms. That second number is independent of the model's own opinion
#: of itself, so it is real accuracy rather than mere confidence. A minute of
#: one-time cost is cheap against words the user would otherwise hand-correct.
DEFAULT_MODEL = "medium.en"

#: A pause longer than this ends a lyric line — that is the singer breathing.
LINE_BREAK_S = 0.70
#: Never let a line run longer than this on screen, however long the phrase.
MAX_LINE_WORDS = 10

#: A word onset within this distance of a vocal note onset is assumed to *be*
#: that onset. Wider than this and we would drag words onto unrelated notes.
SNAP_WINDOW_S = 0.25


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0
    #: How far onset snapping moved this word, in seconds. Diagnostic only.
    snapped_by: float = 0.0


@dataclass
class Line:
    start: float
    end: float
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


@dataclass
class Lyrics:
    model: str
    source: str                      # "transcribed" | "lrc"
    lines: list[Line] = field(default_factory=list)

    @property
    def words(self) -> list[Word]:
        return [w for line in self.lines for w in line.words]

    def word_at(self, t: float) -> Word | None:
        """The word that should be lit up at time ``t``."""
        for w in self.words:
            if w.start <= t < w.end:
                return w
        return None

    def line_at(self, t: float) -> Line | None:
        for line in self.lines:
            if line.start <= t < line.end:
                return line
        return None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "source": self.source,
            "lines": [
                {"start": ln.start, "end": ln.end, "words": [asdict(w) for w in ln.words]}
                for ln in self.lines
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lyrics":
        lines = [
            Line(
                start=ln["start"],
                end=ln["end"],
                words=[Word(**w) for w in ln["words"]],
            )
            for ln in data.get("lines", [])
        ]
        return cls(model=data.get("model", "?"), source=data.get("source", "transcribed"), lines=lines)

    # -- statistics, for verification without printing the text -------------

    def stats(self) -> dict:
        words = self.words
        if not words:
            return {"words": 0, "lines": 0}
        durations = [w.end - w.start for w in words]
        snapped = [w for w in words if w.snapped_by]
        probs = [w.probability for w in words]
        return {
            "words": len(words),
            "lines": len(self.lines),
            "sung_span_s": round(words[-1].end - words[0].start, 1),
            "mean_word_s": round(sum(durations) / len(durations), 3),
            "mean_confidence": round(sum(probs) / len(probs), 3),
            "low_confidence_words": sum(1 for p in probs if p < 0.5),
            "snapped": len(snapped),
            "mean_snap_ms": (
                round(1000 * sum(abs(w.snapped_by) for w in snapped) / len(snapped), 1)
                if snapped else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# transcription
# ---------------------------------------------------------------------------


def transcribe(
    vocal_path: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    progress: ProgressFn = _noop,
) -> list[Word]:
    """Run Whisper over the isolated vocal and return timed words."""
    from faster_whisper import WhisperModel

    progress("loading model", 0.0)
    model = WhisperModel(
        model_size,
        device="cpu",             # CTranslate2 has no DirectML backend; CPU int8
        compute_type="int8",      # is fast enough for a one-off offline pass
        download_root=str(MODELS_DIR / "whisper"),
    )

    progress("transcribing", 0.05)
    segments, info = model.transcribe(
        str(vocal_path),
        word_timestamps=True,
        # The stem is silent through instrumental sections; VAD skips them
        # instead of letting the model hallucinate words into the quiet.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        beam_size=5,
        condition_on_previous_text=False,  # stops one bad line derailing the rest
    )

    words: list[Word] = []
    total = max(info.duration, 1e-6)
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=round(float(w.start), 3),
                    end=round(float(w.end), 3),
                    probability=round(float(w.probability), 3),
                )
            )
        progress("transcribing", min(0.05 + 0.9 * (seg.end / total), 0.95))

    progress("transcribing", 1.0)
    return words


# ---------------------------------------------------------------------------
# refinement
# ---------------------------------------------------------------------------


def snap_to_onsets(
    words: list[Word], note_onsets: Iterable[float], window: float = SNAP_WINDOW_S
) -> list[Word]:
    """Pull word onsets onto nearby vocal note onsets.

    Whisper estimates when a word began; pYIN *measured* when the voice began.
    Where the two nearly agree, the measurement is the better number, and being
    tens of milliseconds early or late is exactly what makes karaoke
    highlighting feel wrong.

    Only the start moves, and only within a tight window — a word is never
    dragged onto an unrelated note, and word order is preserved.
    """
    onsets = sorted(note_onsets)
    if not onsets:
        return words

    import bisect

    out: list[Word] = []
    prev_end = -1.0
    for w in words:
        i = bisect.bisect_left(onsets, w.start)
        candidates = [onsets[j] for j in (i - 1, i) if 0 <= j < len(onsets)]
        best = min(candidates, key=lambda o: abs(o - w.start), default=None)

        new = w
        if best is not None and abs(best - w.start) <= window:
            # Never let snapping reorder words or invert a word's own span.
            if best > prev_end and best < w.end:
                new = Word(
                    text=w.text,
                    start=round(best, 3),
                    end=w.end,
                    probability=w.probability,
                    snapped_by=round(best - w.start, 3),
                )
        out.append(new)
        prev_end = new.end
    return out


def _syllables(word: str) -> int:
    """Rough syllable count, for sharing a phrase out between words.

    A vowel-group heuristic — not linguistics, but the only thing it decides is
    the relative width of each word inside a phrase whose start and end are
    already known from the audio. Being one syllable out shifts a word by a few
    tens of milliseconds; getting the phrase boundaries right, which the audio
    does, matters far more.
    """
    text = "".join(ch for ch in word.lower() if ch.isalpha())
    if not text:
        return 1
    groups = 0
    previous_vowel = False
    for ch in text:
        vowel = ch in "aeiouy"
        if vowel and not previous_vowel:
            groups += 1
        previous_vowel = vowel
    if text.endswith("e") and groups > 1:
        groups -= 1                      # silent terminal e
    return max(1, groups)


def voiced_spans(
    contour, hop_s: float, *, merge_gap_s: float = 0.18, min_span_s: float = 0.12
) -> list[tuple[float, float]]:
    """Contiguous stretches where the vocal stem is actually phonating.

    Derived from the pitch contour, so these are measurements of when the
    singer's voice was on — far more reliable than a speech model's opinion
    about where a sung word began.
    """
    import numpy as np

    voiced = ~np.isnan(np.asarray(contour, dtype=float))
    if not voiced.any():
        return []

    edges = np.diff(voiced.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if voiced[0]:
        starts.insert(0, 0)
    if voiced[-1]:
        ends.append(len(voiced))

    spans = [(s * hop_s, e * hop_s) for s, e in zip(starts, ends)]

    merged: list[tuple[float, float]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= merge_gap_s:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return [s for s in merged if s[1] - s[0] >= min_span_s]


def align_to_voice(
    words: list[Word], spans: list[tuple[float, float]]
) -> list[Word]:
    """Re-time words onto the stretches where the voice is actually singing.

    Whisper reliably recovers *which* words were sung and in *what order*. It
    is much weaker on *when*, because it was trained on speech: a held vowel or
    a melisma has no crisp boundaries to latch onto. Measured on a real track,
    only a third of its word onsets landed within 100 ms of a vocal onset, and
    more than a third of all words were placed in passages where nobody was
    singing at all.

    So the word order is kept and the timing is rebuilt. Words are assigned to
    voiced spans in sequence, then laid out inside each span in proportion to
    their syllable count. The result cannot place a word in silence, because
    silence is not a span.
    """
    if not words or not spans:
        return words

    # Two failed approaches informed this one, and both failures were
    # instructive.
    #
    # Asking each word which span it belonged to used the model's own
    # timestamps — the very thing we do not trust — and left 54 of 122 spans
    # empty while crowding nine words into one.
    #
    # Spreading every word evenly across the voiced timeline fixed placement
    # but destroyed rhythm: median word length went to 0.98 s, because a held
    # vowel is one long voiced stretch and uniform spreading gives every word
    # a share of it.
    #
    # What Whisper is genuinely good at is *relative* timing inside a phrase —
    # it hears the syllables of a line in roughly the right proportions. What
    # it is bad at is placing that phrase in the song. So we keep its rhythm
    # and re-anchor it: lines are matched to sung phrases in order, then each
    # line's own word timings are linearly rescaled to fill its phrase.
    phrases = _merge_spans(spans, gap=PHRASE_MERGE_S)
    lines = group_lines(words)
    if not phrases or not lines:
        return words

    # Map line k onto phrase k. When the counts differ — the transcriber split
    # a breath differently than the pitch tracker did — scale the index so the
    # two sequences stay in step end to end instead of drifting apart after the
    # first mismatch.
    assigned: dict[int, list] = {}
    for index, line in enumerate(lines):
        phrase_index = min(
            int(round(index * (len(phrases) - 1) / max(1, len(lines) - 1))),
            len(phrases) - 1,
        )
        assigned.setdefault(phrase_index, []).append(line)

    out: list[Word] = []
    for phrase_index, group in sorted(assigned.items()):
        start, end = phrases[phrase_index]
        # Several lines can land on one phrase when there are more lines than
        # phrases. Subdivide rather than stacking them, or the highlighting
        # would show two lines at once.
        share = (end - start) / len(group)
        for offset, line in enumerate(group):
            out.extend(
                _rescale_line(
                    line.words, start + offset * share, start + (offset + 1) * share
                )
            )
    return out


#: Voiced stretches closer together than this belong to one sung phrase.
#: Wider than the merge used to find spans, because a phrase spans the small
#: silences between its own syllables.
PHRASE_MERGE_S = 0.55


def _merge_spans(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return merged


def _rescale_line(words: list[Word], start: float, end: float) -> list[Word]:
    """Stretch a line's existing word timings to fill [start, end]."""
    if not words:
        return []
    origin = words[0].start
    extent = max(words[-1].end - origin, 1e-6)
    scale = (end - start) / extent

    out: list[Word] = []
    for word in words:
        new_start = start + (word.start - origin) * scale
        new_end = start + (word.end - origin) * scale
        if new_end <= new_start:
            new_end = new_start + 0.08
        out.append(
            Word(
                text=word.text,
                start=round(new_start, 3),
                end=round(min(new_end, end), 3),
                probability=word.probability,
                snapped_by=round(new_start - word.start, 3),
            )
        )
    return out


def from_text(
    text: str,
    contour,
    contour_hop: float,
    *,
    heard: "Lyrics | None" = None,
    model_label: str = "your lyrics",
) -> Lyrics:
    """Time lyrics the user supplied, against the singing in the audio.

    This is the best path to correct karaoke and it is not close. A
    transcription model is guessing at both *what* was sung and *when*; when
    the words are known, half the problem disappears and the remaining half —
    placing them — is exactly what the pitch tracker is good at.

    The user's own line breaks are treated as meaningful, because they are: a
    lyric sheet breaks where the singer breathes, which is precisely where the
    voiced regions break too. So line *k* is matched to sung phrase *k*, and
    the words inside are laid out by syllable count.

    Nothing here is generated. The words are whatever was pasted in.
    """
    spans = voiced_spans(contour, contour_hop) if contour is not None else []
    raw_lines = [ln.strip() for ln in text.splitlines()]
    raw_lines = [ln for ln in raw_lines if ln and not ln.startswith("[")]
    if not raw_lines:
        return Lyrics(model=model_label, source="user", lines=[])

    phrases = _merge_spans(spans, PHRASE_MERGE_S) if spans else []
    if not phrases:
        # No audio guidance: spread evenly so at least the words are right.
        return Lyrics(
            model=model_label,
            source="user",
            lines=[
                Line(
                    start=i * 4.0,
                    end=(i + 1) * 4.0,
                    words=_spread(ln.split(), i * 4.0, (i + 1) * 4.0),
                )
                for i, ln in enumerate(raw_lines)
            ],
        )

    # Preferred: locate each written line by matching it against what the
    # transcriber heard, which knows roughly where in the song each phrase is.
    # Positional allocation is the fallback, and it is materially worse — it
    # drifts a whole section when the audio contains phrases the lyric sheet
    # does not.
    # First choice: cut the vocal into exactly as many sung phrases as there
    # are written lines, then map them 1:1 in order. Every line then starts
    # when the singer starts that phrase — measured from the audio, with no
    # transcript involved and therefore nothing to drift.
    if contour is not None and contour_hop > 0:
        tuned = segment_to_count(contour, contour_hop, len(raw_lines))
        # Only trust this when the segmentation genuinely converged. A large
        # mismatch means the phrase structure does not correspond to the sheet
        # (an instrumental-heavy song, or a sheet with repeats written out
        # differently), and forcing a 1:1 map would be worse than inferring.
        if tuned and abs(len(tuned) - len(raw_lines)) <= max(1, len(raw_lines) // 10):
            out_lines: list[Line] = []
            for index, text_line in enumerate(raw_lines):
                if index >= len(tuned):
                    break
                words = _spread_over_spans(text_line.split(), [tuned[index]])
                if words:
                    out_lines.append(
                        Line(start=words[0].start, end=words[-1].end, words=words)
                    )
            if out_lines:
                return Lyrics(model=model_label, source="user", lines=out_lines)

    # Second choice: align the written words directly against the heard words.
    # Six times as many anchors as line matching gives, and each word's time is
    # borrowed rather than inferred from its line's start.
    if heard is not None and heard.words:
        tokens_per_line = [line.split() for line in raw_lines]
        flat = [token for line in tokens_per_line for token in line]
        matched = align_words_to_transcript(flat, heard.words)
        hits = sum(1 for t in matched if t is not None)

        if hits >= max(4, len(flat) // 5):
            starts = snap_into_voice(_fill_word_times(matched), spans)
            out_lines: list[Line] = []
            cursor = 0
            for tokens in tokens_per_line:
                if not tokens:
                    continue
                slice_starts = starts[cursor : cursor + len(tokens)]
                cursor += len(tokens)
                words = _words_from_starts(tokens, slice_starts, spans)
                if words:
                    words = _repair_stretched_line(words, tokens, spans)
                    out_lines.append(
                        Line(start=words[0].start, end=words[-1].end, words=words)
                    )
            if out_lines:
                return Lyrics(model=model_label, source="user", lines=out_lines)

    # Fallback: positional allocation across sung phrases.
    out_lines = []
    for text_line, claimed in _allocate_phrases(raw_lines, phrases):
        words = _spread_over_spans(text_line.split(), claimed)
        if words:
            out_lines.append(Line(start=words[0].start, end=words[-1].end, words=words))

    return Lyrics(model=model_label, source="user", lines=out_lines)


#: A sung line longer than this did not really take that long; the alignment
#: lost the thread somewhere inside it. Measured lines here sit around 4-8s.
MAX_PLAUSIBLE_LINE_S = 14.0


def _repair_stretched_line(
    words: list[Word], tokens: list[str], spans: list[tuple[float, float]]
) -> list[Word]:
    """Re-lay a line whose alignment clearly failed.

    Word alignment is right most of the time and occasionally loses its place —
    on this song it produced one line stretched across forty seconds, which no
    one sang. Rather than let a single bad line spoil the display, any line
    longer than :data:`MAX_PLAUSIBLE_LINE_S` is discarded and rebuilt the
    reliable way: anchored at its first word and laid across the voiced regions
    that follow at speaking pace.

    The first word's time is kept because that is the anchor alignment gets
    right; it is the *inside* of a long line that drifts.
    """
    if not words or (words[-1].end - words[0].start) <= MAX_PLAUSIBLE_LINE_S:
        return words

    start = words[0].start
    needed = sum(max(_syllables(t) * SECONDS_PER_SYLLABLE, MIN_WORD_S) for t in tokens)
    window = _spans_within(spans, start, start + needed + MAX_SUSTAIN_S)
    rebuilt = _spread_over_spans(tokens, window)
    return rebuilt or words


def snap_into_voice(
    starts: list[float], spans: list[tuple[float, float]], window: float = 0.6
) -> list[float]:
    """Pull word starts onto the nearest moment the voice is actually sounding.

    The two halves of the problem are solved by different tools and this is
    where they meet. Word alignment against the transcript gets the *structure*
    right — which word, in which order, in roughly which part of the song — but
    inherits the transcriber's per-word timing noise, which on this material put
    only 39% of words on the voice. The pitch tracker knows precisely when the
    voice was on but nothing about words.

    So each aligned word is nudged to the nearest voiced instant, within a
    window tight enough that it can only be a correction and never a
    relocation. Order is preserved throughout.
    """
    if not spans or not starts:
        return starts

    span_starts = [s for s, _ in spans]
    out: list[float] = []
    for value in starts:
        inside = any(a <= value < b for a, b in spans)
        if inside:
            out.append(value)
            continue

        index = bisect.bisect_left(span_starts, value)
        candidates: list[float] = []
        if index < len(spans):
            candidates.append(spans[index][0])       # start of the next span
        if index > 0:
            candidates.append(spans[index - 1][1])   # end of the previous one
        near = [c for c in candidates if abs(c - value) <= window]
        out.append(min(near, key=lambda c: abs(c - value)) if near else value)

    # Keep it strictly increasing: snapping must never reorder the line.
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 0.05
    return out


def _words_from_starts(
    tokens: list[str], starts: list[float], spans: list[tuple[float, float]]
) -> list[Word]:
    """Build words from per-word start times, ending each at the next.

    A word runs until the following word begins, capped so the last one in a
    phrase releases instead of sitting lit through a long held note or an
    instrumental passage.
    """
    out: list[Word] = []
    for index, (token, start) in enumerate(zip(tokens, starts)):
        if index + 1 < len(starts):
            end = min(starts[index + 1], start + MAX_SUSTAIN_S)
        else:
            end = start + min(MAX_SUSTAIN_S, 1.0)
        out.append(
            Word(text=token, start=round(start, 3), end=round(max(end, start + 0.08), 3))
        )
    return out


def _spans_within(
    spans: list[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    """Voiced regions overlapping a time window, clipped to it.

    Words are laid onto these rather than onto the raw window, so a line spread
    across a window containing silence still only lights up where the voice is.
    """
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if b <= start or a >= end:
            continue
        out.append((max(a, start), min(b, end)))
    if not out:
        # The transcriber placed this line where the pitch tracker heard
        # nothing. Trust the window rather than dropping the line.
        out = [(start, max(end, start + 0.4))]
    return out


#: Syllables per second of actual singing. Only used to decide how much voiced
#: time a line needs; the real durations come from the audio.
_FALLBACK_RATE = 3.0

#: How long a sung syllable takes at conversational pace. Around 4 syllables a
#: second is typical for lyric delivery; slower than speech, far faster than
#: the average you get if you divide a phrase by its word count, because that
#: average is inflated by the note held at the end.
SECONDS_PER_SYLLABLE = 0.24
#: Floor, so a one-letter word still gets long enough to be readable.
MIN_WORD_S = 0.16
#: Longest a single word stays lit while its note is held. Beyond this the
#: highlight stops rather than implying the word is still being articulated.
MAX_SUSTAIN_S = 1.4


def _allocate_phrases(
    lines: list[str], phrases: list[tuple[float, float]]
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Give each line as many consecutive sung phrases as its words need.

    One line to one phrase is the simple mapping and it leaves words stranded:
    a long line does not fit in a short phrase, so its tail spills into the
    silence after it. Measured on a real sheet, roughly a quarter of words
    ended up in gaps between phrases.

    Instead, lines claim consecutive phrases until they have enough *voiced*
    time for their syllable count, at the rate implied by the song as a whole.
    A line with twice the words takes roughly twice the singing, which is what
    actually happens when someone sings it.
    """
    weights = [max(1, sum(_syllables(t) for t in line.split())) for line in lines]
    total_weight = float(sum(weights))
    voiced_total = float(sum(end - start for start, end in phrases)) or 1.0
    rate = total_weight / voiced_total or _FALLBACK_RATE

    out: list[tuple[str, list[tuple[float, float]]]] = []
    index = 0
    for line, weight in zip(lines, weights):
        needed = weight / rate
        claimed: list[tuple[float, float]] = []
        gathered = 0.0

        # Always take at least one phrase, so every line lands somewhere.
        while index < len(phrases) and (not claimed or gathered < needed):
            start, end = phrases[index]
            claimed.append((start, end))
            gathered += end - start
            index += 1
            # Stop before over-claiming: if the next phrase would overshoot by
            # more than it helps, leave it for the following line.
            if gathered >= needed:
                break

        if not claimed:
            # Ran out of phrases; share the final one rather than dropping text.
            claimed = [phrases[-1]]
        out.append((line, claimed))

    return out


def _spread_over_spans(
    tokens: list[str], spans: list[tuple[float, float]]
) -> list[Word]:
    """Lay words across several voiced spans, skipping the silences between.

    This is what keeps a word from being highlighted while nobody is singing:
    the gaps between phrases simply have no length on the timeline the words
    are laid onto.
    """
    if not tokens or not spans:
        return []

    durations = [end - start for start, end in spans]
    voiced = sum(durations) or 1e-3

    offsets: list[float] = []
    running = 0.0
    for duration in durations:
        offsets.append(running)
        running += duration

    def to_clock(position: float) -> float:
        position = min(max(position, 0.0), voiced)
        i = bisect.bisect_right(offsets, position) - 1
        i = min(max(i, 0), len(spans) - 1)
        return spans[i][0] + min(position - offsets[i], durations[i])

    # Words move at roughly speaking pace; the phrase's final note holds.
    #
    # Sharing a phrase out evenly by syllable is the obvious approach and it
    # feels wrong for exactly the reason a ballad feels the way it does: the
    # phrase ends on a long sustained vowel, and dividing that sustain among
    # every word stretches all of them. The highlight then creeps through words
    # the singer has already finished and arrives late — which is precisely the
    # lag being reported. Giving the surplus to the last word instead matches
    # how the line is actually sung.
    nominal = [max(_syllables(t) * SECONDS_PER_SYLLABLE, MIN_WORD_S) for t in tokens]
    needed = sum(nominal)

    if needed >= voiced:
        # Phrase is tighter than speaking pace: compress everything to fit.
        scale = voiced / needed
        nominal = [d * scale for d in nominal]
    else:
        # Surplus is the sustain. It belongs to the final word — but only up to
        # a point. An unbounded share produced a 17-second held word, and a
        # highlight parked on one word for 17 seconds is as wrong as one that
        # lags. Past the cap the line is simply over and nothing is lit, which
        # is the honest reading: the singer is holding a vowel, not still
        # working through words.
        nominal[-1] = min(nominal[-1] + (voiced - needed), MAX_SUSTAIN_S)

    out: list[Word] = []
    cumulative = 0.0
    for token, duration in zip(tokens, nominal):
        start = to_clock(cumulative)
        cumulative += duration
        end = to_clock(cumulative)
        # Clamp on the *wall clock*, not on voiced time. A word whose voiced
        # allocation straddles a gap between phrases still occupies that gap on
        # screen, so a 4-second voiced share became a 21-second highlight. What
        # the eye sees is the thing that needs bounding.
        end = min(max(end, start + 0.05), start + MAX_SUSTAIN_S)
        out.append(Word(text=token, start=round(start, 3), end=round(end, 3)))
    return out


def _match_lines_to_phrases(
    lines: list[str], phrases: list[tuple[float, float]]
) -> dict[int, list[str]]:
    """Decide which sung phrase each written line belongs to.

    Matching line *k* to phrase *k* by position is the obvious approach and it
    drifts: a lyric sheet rarely has exactly as many lines as the pitch tracker
    finds phrases, and once the two counts disagree every subsequent line is
    off by a little more.

    Weighting fixes that. A long line takes longer to sing than a short one, so
    lines are placed by their cumulative *syllable* position against the
    phrases' cumulative *duration*. A line with three times the words claims
    roughly three times the singing time, and the two sequences stay in step
    from beginning to end rather than only at the start.
    """
    weights = [max(1, sum(_syllables(t) for t in line.split())) for line in lines]
    total_weight = float(sum(weights))
    durations = [end - start for start, end in phrases]
    total_duration = float(sum(durations)) or 1.0

    # Cumulative fraction of total singing time at which each phrase ends.
    boundaries: list[float] = []
    running = 0.0
    for duration in durations:
        running += duration
        boundaries.append(running / total_duration)

    grouped: dict[int, list[str]] = {}
    cumulative = 0.0
    for line, weight in zip(lines, weights):
        midpoint = (cumulative + weight / 2.0) / total_weight
        cumulative += weight
        index = bisect.bisect_left(boundaries, midpoint)
        grouped.setdefault(min(index, len(phrases) - 1), []).append(line)
    return grouped


def _similarity(a: str, b: str) -> float:
    """0..1 text similarity, ignoring case and punctuation."""
    import difflib

    clean = lambda s: re.sub(r"[^a-z ]+", "", s.lower()).strip()
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


#: Similarity below which a match is not believed at all.
_MATCH_FLOOR = 0.45


def anchor_to_transcript(
    user_lines: list[str], heard: list[Line]
) -> list[tuple[float, float] | None]:
    """Locate each written line in the song, by matching it to what was heard.

    Positional allocation — line *k* to phrase *k* — is what produced the
    reported failure of a chorus line appearing during a verse. It assumes the
    detected phrases correspond one-for-one with the written ones, and a single
    spurious phrase in an instrumental passage throws every later line out of
    step permanently.

    Transcription solves the part that positional matching cannot. Whisper's
    word timings are poor, but it does know roughly *where in the song* a given
    phrase was sung. So each of the user's (correct) lines is matched to the
    heard line it most resembles, and takes its position from there.

    The matching is a monotonic alignment, not a nearest-neighbour search: a
    chorus that repeats four times has four near-identical candidates, and only
    the ordering constraint stops the third repeat matching the first. Lines
    with no confident match return ``None`` and are interpolated by the caller.
    """
    if not user_lines or not heard:
        return [None] * len(user_lines)

    n, m = len(user_lines), len(heard)
    heard_text = [line.text for line in heard]

    # Needleman-Wunsch style alignment: both sides may skip, order preserved.
    gap = -0.15
    score = np.full((n + 1, m + 1), -1e9)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    score[0, :] = np.arange(m + 1) * gap
    score[:, 0] = np.arange(n + 1) * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = score[i - 1, j - 1] + _similarity(user_lines[i - 1], heard_text[j - 1])
            skip_user = score[i - 1, j] + gap
            skip_heard = score[i, j - 1] + gap
            best = max(diagonal, skip_user, skip_heard)
            score[i, j] = best
            back[i, j] = 0 if best == diagonal else (1 if best == skip_user else 2)

    matches: list[tuple[float, float] | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        move = back[i, j]
        if move == 0:
            if _similarity(user_lines[i - 1], heard_text[j - 1]) >= _MATCH_FLOOR:
                matches[i - 1] = (heard[j - 1].start, heard[j - 1].end)
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return matches


#: Sung regions shorter than this are not phrases. Breaths, bleed from the
#: backing, and tracker artefacts in instrumental passages all land here, and
#: each one that survives steals a line's allocation and pushes every later
#: line out of step.
MIN_PHRASE_S = 0.45


def segment_to_count(
    contour,
    hop_s: float,
    target_count: int,
    *,
    min_phrase_s: float = MIN_PHRASE_S,
) -> list[tuple[float, float]]:
    """Segment the vocal into as close to ``target_count`` sung phrases as possible.

    This is what ties the karaoke to the singer rather than to a language model.

    A lyric sheet has one line per sung phrase. The vocal stem tells us exactly
    when the singer is phonating. So if the audio is cut into the same number of
    phrases as there are written lines, the mapping is 1:1 and in order, and
    every line begins precisely when the singer begins that phrase — no
    inference, no transcript, nothing to drift.

    The obstacle was that the merge threshold was a fixed 0.55 s, which happened
    to yield 67 phrases for 26 written lines. Nothing made those agree. Here the
    threshold is searched instead: a wider gap merges more aggressively and
    yields fewer phrases, so the count falls monotonically as the gap grows,
    and a binary search converges quickly.

    Short regions are discarded first. An instrumental passage still produces a
    few flickers of apparent voicing, and those are not phrases.
    """
    base = voiced_spans(contour, hop_s, merge_gap_s=0.12, min_span_s=0.08)
    if not base or target_count <= 0:
        return base

    def phrases_at(gap: float) -> list[tuple[float, float]]:
        merged = _merge_spans(base, gap)
        return [s for s in merged if s[1] - s[0] >= min_phrase_s]

    low, high = 0.05, 6.0
    best = phrases_at(PHRASE_MERGE_S)
    best_error = abs(len(best) - target_count)

    for _ in range(40):
        mid = (low + high) / 2.0
        candidate = phrases_at(mid)
        error = abs(len(candidate) - target_count)
        if error < best_error:
            best, best_error = candidate, error
        if error == 0:
            return candidate
        if len(candidate) > target_count:
            low = mid          # too many phrases: merge harder
        else:
            high = mid         # too few: merge less
        if high - low < 1e-3:
            break

    return best


def align_words_to_transcript(
    user_words: list[str], heard: list[Word]
) -> list[float | None]:
    """Give each written word the time of the heard word it matches.

    Line-level anchoring proved too coarse. A line covers several seconds, so
    inheriting its start leaves every word after the first to be guessed at,
    and errors in the line's own placement are inherited whole. Successive
    attempts to fix it kept trading one artefact for another: lines were either
    too long and lagged, or too short and left the screen blank.

    Word-level alignment is the finer instrument. The transcriber's per-word
    timing is noisy — a few hundred milliseconds either way — but it is
    *unbiased*, and there are six times as many anchors. The user's text stays
    authoritative for the words; only the times are borrowed.

    Matching is a monotonic alignment, so the four near-identical repeats of a
    chorus cannot swap places. Words that match nothing are interpolated
    between their nearest matched neighbours.
    """
    if not user_words or not heard:
        return [None] * len(user_words)

    clean = [re.sub(r"[^a-z']+", "", w.lower()) for w in user_words]
    heard_clean = [re.sub(r"[^a-z']+", "", w.text.lower()) for w in heard]

    n, m = len(clean), len(heard_clean)
    gap = -0.4
    score = np.zeros((n + 1, m + 1))
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    score[0, :] = np.arange(m + 1) * gap
    score[:, 0] = np.arange(n + 1) * gap

    for i in range(1, n + 1):
        a = clean[i - 1]
        row_prev, row = score[i - 1], score[i]
        for j in range(1, m + 1):
            b = heard_clean[j - 1]
            # Exact match is cheap and covers most of it; fall back to a ratio
            # only when the tokens differ, which keeps this loop fast.
            similarity = 1.0 if a == b else (0.5 if a and b and a[0] == b[0] else 0.0)
            diagonal = row_prev[j - 1] + similarity
            up = row_prev[j] + gap
            left = row[j - 1] + gap
            best = max(diagonal, up, left)
            row[j] = best
            back[i, j] = 0 if best == diagonal else (1 if best == up else 2)

    times: list[float | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        move = back[i, j]
        if move == 0:
            if clean[i - 1] == heard_clean[j - 1]:
                times[i - 1] = heard[j - 1].start
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return times


def _fill_word_times(times: list[float | None], fallback_gap: float = 0.35) -> list[float]:
    """Interpolate words that matched nothing, keeping the order strictly
    increasing so the highlight can never jump backwards."""
    out: list[float] = [0.0] * len(times)
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        return [i * fallback_gap for i in range(len(times))]

    for position, index in enumerate(known):
        out[index] = float(times[index])
        if position:
            previous = known[position - 1]
            span = out[index] - out[previous]
            steps = index - previous
            for k in range(1, steps):
                out[previous + k] = out[previous] + span * k / steps

    first, last = known[0], known[-1]
    for k in range(first - 1, -1, -1):
        out[k] = out[k + 1] - fallback_gap
    for k in range(last + 1, len(times)):
        out[k] = out[k - 1] + fallback_gap

    # Strictly increasing, so a repeated timestamp cannot stall the cursor.
    for k in range(1, len(out)):
        if out[k] <= out[k - 1]:
            out[k] = out[k - 1] + 0.05
    return out


def _interpolate_gaps(
    matches: list[tuple[float, float] | None], duration: float
) -> list[tuple[float, float]]:
    """Fill in lines that matched nothing, between their confident neighbours."""
    out: list[tuple[float, float]] = []
    for index, match in enumerate(matches):
        if match is not None:
            out.append(match)
            continue
        before = next(
            (matches[k] for k in range(index - 1, -1, -1) if matches[k]), None
        )
        after = next(
            (matches[k] for k in range(index + 1, len(matches)) if matches[k]), None
        )
        if before and after:
            span = after[0] - before[1]
            steps = sum(1 for k in range(index, len(matches)) if matches[k] is None) + 1
            width = max(span / steps, 0.5)
            start = before[1] + (index - matches.index(before)) * 0.0
            out.append((start, start + width))
        elif before:
            out.append((before[1], min(before[1] + 3.0, duration)))
        elif after:
            out.append((max(after[0] - 3.0, 0.0), after[0]))
        else:
            out.append((index * 4.0, index * 4.0 + 3.5))
    return out


def _spread(tokens: list[str], start: float, end: float) -> list[Word]:
    """Lay words across a span in proportion to their syllables."""
    if not tokens:
        return []
    weights = [_syllables(t) for t in tokens]
    total = sum(weights) or 1
    duration = max(end - start, 1e-3)
    cursor = start
    out: list[Word] = []
    for token, weight in zip(tokens, weights):
        width = duration * weight / total
        out.append(
            Word(text=token, start=round(cursor, 3), end=round(cursor + width, 3))
        )
        cursor += width
    return out


def group_lines(
    words: list[Word],
    *,
    break_s: float = LINE_BREAK_S,
    max_words: int = MAX_LINE_WORDS,
) -> list[Line]:
    """Break words into displayable lines at the singer's own pauses."""
    lines: list[Line] = []
    current: list[Word] = []

    for w in words:
        if current:
            gap = w.start - current[-1].end
            if gap >= break_s or len(current) >= max_words:
                lines.append(Line(current[0].start, current[-1].end, current))
                current = []
        current.append(w)

    if current:
        lines.append(Line(current[0].start, current[-1].end, current))
    return lines


# ---------------------------------------------------------------------------
# .lrc sidecar (hand corrections win)
# ---------------------------------------------------------------------------

_LRC_LINE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")


def write_lrc(lyrics: Lyrics, path: Path) -> None:
    """Write a standard .lrc the user can open in any text editor and fix."""
    out = ["[re:SingCoach]", f"[ve:{lyrics.model}]"]
    for line in lyrics.lines:
        m, s = divmod(line.start, 60)
        out.append(f"[{int(m):02d}:{s:05.2f}]{line.text}")
    path.write_text("\n".join(out) + "\n", "utf-8")


def read_lrc(path: Path) -> Lyrics:
    """Parse a hand-edited .lrc.

    Word-level timing is lost — an .lrc only carries line starts — so words in
    a line are spread evenly across it. Line highlighting stays exact, which is
    the part that matters most, and the user gets correct words.
    """
    entries: list[tuple[float, str]] = []
    for raw in path.read_text("utf-8").splitlines():
        m = _LRC_LINE.match(raw.strip())
        if not m:
            continue
        t = int(m.group(1)) * 60 + float(m.group(2))
        text = m.group(3).strip()
        if text:
            entries.append((t, text))

    entries.sort()
    lines: list[Line] = []
    for i, (start, text) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else start + 4.0
        tokens = text.split()
        if not tokens:
            continue
        step = (end - start) / len(tokens)
        words = [
            Word(text=tok, start=round(start + j * step, 3), end=round(start + (j + 1) * step, 3))
            for j, tok in enumerate(tokens)
        ]
        lines.append(Line(start=start, end=end, words=words))
    return Lyrics(model="lrc", source="lrc", lines=lines)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def analyse_lyrics(
    vocal_path: Path,
    note_onsets: Iterable[float] = (),
    *,
    model_size: str = DEFAULT_MODEL,
    lrc_override: Path | None = None,
    contour=None,
    contour_hop: float = 0.0,
    progress: ProgressFn = _noop,
) -> Lyrics:
    """Transcribe and time the lyrics.

    When a pitch contour is supplied, word timing is rebuilt from the voiced
    regions it describes rather than taken from the transcription model — see
    :func:`align_to_voice`. Without one we fall back to nudging the model's own
    onsets toward nearby notes, which is much weaker.
    """
    if lrc_override is not None and lrc_override.exists():
        progress("reading hand-edited lyrics", 1.0)
        return read_lrc(lrc_override)

    # The raw transcript is cached separately from the aligned result.
    # Transcription costs a minute; alignment costs milliseconds. Keeping only
    # the aligned output meant re-tuning alignment silently re-aligned its own
    # previous output, compounding the error each pass.
    raw_path = vocal_path.with_name("words_raw.json")
    if raw_path.exists():
        words = [Word(**w) for w in json.loads(raw_path.read_text("utf-8"))]
        progress("using cached transcript", 0.9)
    else:
        words = transcribe(vocal_path, model_size=model_size, progress=progress)
        raw_path.write_text(
            json.dumps([asdict(w) for w in words], ensure_ascii=False), "utf-8"
        )

    words = time_words(
        vocal_path, words, contour, contour_hop, note_onsets, progress=progress
    )
    return Lyrics(model=model_size, source="transcribed", lines=group_lines(words))


def time_words(
    vocal_path: Path,
    words: list[Word],
    contour=None,
    contour_hop: float = 0.0,
    note_onsets: Iterable[float] = (),
    *,
    prefer_forced: bool = False,
    progress: ProgressFn = _noop,
) -> list[Word]:
    """Decide when each word happens.

    Two approaches were built and measured against each other; the result was
    not the one the architecture predicted.

    **CTC forced alignment** (:mod:`singcoach.analysis.align`) is the textbook
    answer: give an acoustic model the text and let it find the audio. Against
    a fixture of synthesised *speech* with known word times it is excellent —
    73 ms median error, every word inside 200 ms. But on an actual sung vocal
    it placed only a third of the words on the voice. The available acoustic
    models are trained on read speech, and a sustained vowel with vibrato and a
    drawl is nothing like read speech.

    **Anchoring the transcript to sung phrases** (:func:`align_to_voice`) uses
    pitch tracking to find where the voice is genuinely sounding, which does
    not care whether the singing resembles speech. On the same real vocal it
    put 79% of words on the voice and 97% within 150 ms of it.

    So the heuristic is the default, and forced alignment is available via
    ``prefer_forced`` for material closer to speech. Measurements over
    elegance.
    """
    if contour is not None and contour_hop > 0 and not prefer_forced:
        progress("aligning words to the voice", 0.97)
        return align_to_voice(words, voiced_spans(contour, contour_hop))

    if prefer_forced:
        try:
            from .. import modelstore
            from . import align as align_mod

            progress("forced alignment", 0.9)
            aligner = align_mod.ForcedAligner(modelstore.ensure("aligner"))
            audio = align_mod.load_audio(vocal_path)
            aligned = aligner.align(audio, " ".join(w.text for w in words))
            if aligned:
                # Only the timing comes from the aligner; the transcriber's
                # casing and punctuation are what the singer reads.
                return [
                    Word(
                        text=original.text,
                        start=timed.start,
                        end=timed.end,
                        probability=original.probability,
                        snapped_by=round(timed.start - original.start, 3),
                    )
                    for original, timed in zip(words, aligned)
                ]
        except Exception:
            # Never let an alignment failure cost the user their lyrics.
            pass
        if contour is not None and contour_hop > 0:
            return align_to_voice(words, voiced_spans(contour, contour_hop))

    return snap_to_onsets(words, note_onsets)


def load(path: Path) -> Lyrics:
    return Lyrics.from_dict(json.loads(path.read_text("utf-8")))


def save(lyrics: Lyrics, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lyrics.to_dict(), ensure_ascii=False), "utf-8")
    tmp.replace(path)
