"""Filesystem layout and persisted user settings.

Everything SingCoach writes lives under ``ROOT``. Nothing leaves this machine
except the one-time model download in :mod:`singcoach.modelstore`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

APP_NAME = "SingCoach"

ROOT = Path(os.environ.get("SINGCOACH_ROOT", r"F:\sing-coach"))
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data"
SETTINGS_PATH = DATA_DIR / "settings.json"
HISTORY_DB = DATA_DIR / "history.db"

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------

#: Working rate for playback and live pitch detection.
SAMPLE_RATE = 44100
CHANNELS = 2

#: Analysis rate. pYIN and Whisper gain nothing from 44.1k on a voice, and both
#: are markedly faster at half rate.
ANALYSIS_SR = 22050

#: Live pitch detection window/hop @ SAMPLE_RATE.
#: 2048 samples = 46 ms, enough to resolve a 65 Hz bass note (needs ~1360).
#: 512-sample hop gives ~86 readings/sec, comfortably faster than the eye.
PITCH_WINDOW = 2048
PITCH_HOP = 512

#: Vocal search range for melody extraction: C2 (65.4 Hz) to C6 (1046.5 Hz).
F0_MIN_HZ = 65.4
F0_MAX_HZ = 1046.5

#: Playback speeds we are willing to pre-render (see audio/render.py).
SPEED_STEPS = (0.60, 0.70, 0.80, 0.90, 1.00)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class PracticeToggles:
    """The four practice features. All off by default — plain sing-along.

    Stored per song as well as globally; the per-song copy wins when present.
    """

    transpose: bool = False
    slowdown_loop: bool = False
    scoring: bool = False
    harmony_guides: bool = False


@dataclass
class SeparatorConfig:
    """Which vocal-separation backend to use, and where its model file lives.

    SingCoach ships one small MDX-Net ONNX model (~66 MB, always present) and
    can optionally use a much stronger — but much larger — 2026-class model:
    an **HTDemucs** or **BS-Roformer / Mel-Roformer** ONNX export. The big model
    is *not* downloaded automatically; you drop the ``.onnx`` file into
    ``models/`` yourself (see the README). If it is not there, separation falls
    back to MDX-Net rather than failing.

    ``backend`` values:

    * ``"auto"``     — use the HQ model if its file is present, otherwise MDX-Net.
    * ``"mdx"``      — always MDX-Net, even if an HQ file is present.
    * ``"htdemucs"`` — require the HQ (waveform) model; still falls back to
                       MDX-Net if the file is missing or fails to load, so a
                       typo in the filename can never brick separation.

    The remaining fields describe the HQ model's I/O so the loader is correct
    without having to guess. Defaults match the HTDemucs-FT ONNX export
    (input ``mix`` ``(1, 2, 343980)`` f32 @ 44.1 kHz, output ``stems``
    ``(1, 4, 2, 343980)`` ordered ``[drums, bass, other, vocals]``). For a
    2-stem BS-Roformer vocal export, set ``hq_num_stems=2`` and
    ``hq_vocals_index=0``.
    """

    backend: str = "auto"
    #: Filename (under ``models/``) of the HQ ONNX model to look for.
    hq_model_file: str = "htdemucs_ft_vocals.onnx"
    #: Index of the vocals stem in the model's output. Demucs order is
    #: [drums, bass, other, vocals] -> 3. A vocals-only or 2-stem export is 0.
    hq_vocals_index: int = 3
    #: Number of stems the model outputs (4 for HTDemucs, 2 for a vocal Roformer).
    hq_num_stems: int = 4
    #: Fixed input length (samples) the model expects. 0 means "read it from the
    #: ONNX input shape at load time", which is the safest choice.
    hq_segment_samples: int = 0
    #: Overlap fraction between consecutive chunks for overlap-add reconstruction.
    hq_overlap: float = 0.25
    #: Smallest plausible size (bytes) for the HQ file to count as present. A
    #: smaller file is treated as a truncated/placeholder download and ignored,
    #: so a half-finished copy falls back to MDX instead of crashing ONNX.
    hq_min_bytes: int = 1_000_000

    #: Backends we know how to build. Anything else is coerced to "auto".
    _VALID = ("auto", "mdx", "htdemucs")

    def normalised_backend(self) -> str:
        b = (self.backend or "auto").strip().lower()
        # A few friendly aliases for the same waveform backend.
        if b in ("roformer", "bs-roformer", "bs_roformer", "mel-roformer",
                 "mel_band_roformer", "demucs", "hq"):
            return "htdemucs"
        if b == "mdx-net":
            return "mdx"
        return b if b in self._VALID else "auto"


@dataclass
class Settings:
    # -- audio devices ------------------------------------------------------
    output_device: str | None = None
    input_device: str | None = None
    #: Audio callback block size. See MIN_BLOCK_SIZE for why this is generous:
    #: the callback is Python and must take the GIL, so it needs slack to
    #: survive a busy UI frame.
    block_size: int = 2048

    #: Measured round-trip latency in ms, keyed by output device name. The
    #: figure is device-specific, so switching headphones must not silently
    #: reuse the wrong number.
    latency_ms_by_device: dict[str, float] = field(default_factory=dict)
    #: Manual nudge applied on top of the measured value.
    latency_nudge_ms: float = 0.0

    #: Extra shift applied to the *picture* only — highway and lyrics — on top
    #: of the output latency the driver reports. Positive moves the display
    #: later (use it when the words light up early); negative moves it earlier
    #: (use it when they lag). Drivers report buffer latency inconsistently, so
    #: this is a judgement the person watching the screen can make instantly
    #: and no measurement can make for them.
    visual_offset_ms: float = 0.0

    # -- pitch judgement ----------------------------------------------------
    #: Half-width of the "on pitch" band shown in the UI.
    display_tolerance_cents: float = 20.0
    #: Half-width used when scoring. Looser than display on purpose.
    scoring_tolerance_cents: float = 35.0
    #: Mic RMS below this (dBFS) is treated as silence, not a wrong note.
    noise_gate_dbfs: float = -48.0

    # -- mix ----------------------------------------------------------------
    vocal_gain: float = 0.0          # 0.0 karaoke .. 1.0 sing with the original
    accompaniment_gain: float = 1.0
    master_gain: float = 1.0
    #: Normalise every song to this integrated loudness at playback time.
    target_lufs: float = -18.0

    # -- separation ---------------------------------------------------------
    #: Vocal-separation backend selection and HQ-model geometry. See
    #: :class:`SeparatorConfig`.
    separator: SeparatorConfig = field(default_factory=SeparatorConfig)

    # -- practice -----------------------------------------------------------
    practice: PracticeToggles = field(default_factory=PracticeToggles)
    #: Comfortable range from the Range Wizard, as MIDI note numbers.
    vocal_range_low: float | None = None
    vocal_range_high: float | None = None
    #: Detected register break(s), MIDI. Used to warn before a song crosses it.
    passaggio: list[float] = field(default_factory=list)

    # -- misc ---------------------------------------------------------------
    headphone_warning_acknowledged: bool = False
    last_song_hash: str | None = None

    # -- (de)serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        data = dict(data)
        practice = data.pop("practice", {}) or {}
        separator = data.pop("separator", {}) or {}
        nested = {"practice", "separator"}
        known = {f for f in cls.__dataclass_fields__ if f not in nested}
        # Drop unknown keys so an older settings file never crashes a newer app.
        clean = {k: v for k, v in data.items() if k in known}
        toggles = PracticeToggles(
            **{
                k: v
                for k, v in practice.items()
                if k in PracticeToggles.__dataclass_fields__
            }
        )
        # Same forward-compat filtering for the separator block: an unknown key
        # (e.g. one written by a newer build) is ignored rather than fatal.
        sep = SeparatorConfig(
            **{
                k: v
                for k, v in separator.items()
                if k in SeparatorConfig.__dataclass_fields__
            }
        )
        return cls(practice=toggles, separator=sep, **clean)

    def latency_ms(self) -> float:
        """Total offset to subtract from 'now' when looking up the target note."""
        base = self.latency_ms_by_device.get(self.output_device or "", 0.0)
        return base + self.latency_nudge_ms


def ensure_dirs() -> None:
    for d in (MODELS_DIR, CACHE_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


#: Smallest audio block we are willing to run. Below this a Python audio
#: callback has too little slack to survive a busy UI frame.
#:
#: 2048 samples is 46 ms of headroom per callback. The usual objection to a
#: large block — added output latency — does not apply here: feedback timing is
#: handled by measurement, and the visuals are aligned to the stream's reported
#: output latency rather than to the raw playhead. So the extra buffer costs us
#: nothing we care about and buys a lot of robustness.
MIN_BLOCK_SIZE = 2048


def load_settings() -> Settings:
    ensure_dirs()
    if not SETTINGS_PATH.exists():
        return Settings()
    try:
        settings = Settings.from_dict(json.loads(SETTINGS_PATH.read_text("utf-8")))
    except (json.JSONDecodeError, TypeError, ValueError):
        # A corrupt settings file must never block startup. Keep the bad copy
        # for forensics and carry on with defaults.
        SETTINGS_PATH.replace(SETTINGS_PATH.with_suffix(".json.bad"))
        return Settings()

    # Raise a stored block size that predates the current floor. There is no UI
    # for this value, so anything on disk is an old default rather than a
    # deliberate choice — and leaving it would silently deny the fix to exactly
    # the users who already hit the problem.
    if settings.block_size < MIN_BLOCK_SIZE:
        settings.block_size = MIN_BLOCK_SIZE
        save_settings(settings)
    return settings


def save_settings(settings: Settings) -> None:
    ensure_dirs()
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2), "utf-8")
    tmp.replace(SETTINGS_PATH)
