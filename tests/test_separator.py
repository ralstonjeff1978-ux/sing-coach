"""Tests for the pluggable vocal-separation layer.

Three things are checked here, none of which need the real multi-GB HQ weights
or any audio hardware:

* **selection / fallback** — build_separator picks the high-quality backend when
  its model file is present and falls back to MDX-Net when it is not (or when the
  HQ model fails to load), and reports which backend is active and why.
* **config parsing** — SeparatorConfig round-trips through settings JSON and is
  forward/backward compatible.
* **output-stem contract** — RoformerSeparator's chunking + overlap-add produce
  (2, N) float32 stems that sum back to the mix, driven by a tiny fake ONNX
  session that emulates the documented HTDemucs I/O contract. A final test runs
  the *real* MDX-Net model end to end, but only if it is already on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from singcoach import modelstore
from singcoach.analysis import separator as sepmod
from singcoach.analysis.roformer import RoformerSeparator, WaveformParams
from singcoach.analysis.separate import SeparationError
from singcoach.config import SAMPLE_RATE, SeparatorConfig, Settings


# ---------------------------------------------------------------------------
# fakes: a minimal ONNX-session stand-in and stub separators
# ---------------------------------------------------------------------------


class _FakeIO:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape


class FakeSession:
    """Emulates the slice of the onnxruntime InferenceSession API we use.

    Returns a stems tensor (1, num_stems, 2, seg) where stem ``k`` is the input
    scaled by ``0.1 * (k + 1)`` — distinct per stem so tests can prove the right
    row is selected, and an exact linear function of the input so overlap-add can
    be checked against ground truth.
    """

    def __init__(self, *, seg: int, num_stems: int = 4, provider: str = "CPUExecutionProvider",
                 input_shape: list | None = None) -> None:
        self.seg = seg
        self.num_stems = num_stems
        self._provider = provider
        self._in_shape = input_shape if input_shape is not None else [1, 2, seg]
        self.calls: list[tuple] = []

    def get_inputs(self):
        return [_FakeIO("mix", self._in_shape)]

    def get_outputs(self):
        return [_FakeIO("stems", [1, self.num_stems, 2, self.seg])]

    def get_providers(self):
        return [self._provider]

    def run(self, output_names, feed):
        chunk = feed["mix"]
        # The separator must feed the model exactly (1, 2, seg) float32.
        assert chunk.shape == (1, 2, self.seg), chunk.shape
        assert chunk.dtype == np.float32, chunk.dtype
        self.calls.append(chunk.shape)
        stems = np.zeros((1, self.num_stems, 2, self.seg), dtype=np.float32)
        for k in range(self.num_stems):
            stems[0, k] = chunk[0] * (0.1 * (k + 1))
        return [stems]


class StubMDX:
    backend = "mdx-net"

    def __init__(self, model_path, params=None, *, prefer_gpu=True):
        self.model_path = Path(model_path)
        self.params = params
        self.prefer_gpu = prefer_gpu
        self.provider = "CPUExecutionProvider"

    @property
    def on_gpu(self) -> bool:
        return False

    def separate(self, mix, *, progress=None):
        v = (mix * 0.5).astype(np.float32)
        return v, (mix - v).astype(np.float32)


class StubRoformer:
    backend = "htdemucs"

    def __init__(self, model_path, params=None, *, prefer_gpu=True,
                 backend_name="htdemucs", session=None):
        self.model_path = Path(model_path)
        self.params = params
        self.backend = backend_name
        self.provider = "DmlExecutionProvider"
        StubRoformer.constructed = getattr(StubRoformer, "constructed", 0) + 1

    @property
    def on_gpu(self) -> bool:
        return True

    def separate(self, mix, *, progress=None):
        v = (mix * 0.9).astype(np.float32)
        return v, (mix - v).astype(np.float32)


class ExplodingRoformer:
    def __init__(self, *a, **k):
        raise SeparationError("boom: incompatible graph")


@pytest.fixture
def hq_config(tmp_path: Path):
    """A models dir plus a config whose presence floor is tiny for testing."""
    cfg = SeparatorConfig(hq_model_file="hq.onnx", hq_min_bytes=8, backend="auto")
    return tmp_path, cfg


def _write_hq(models_dir: Path, cfg: SeparatorConfig, size: int) -> Path:
    p = models_dir / cfg.hq_model_file
    p.write_bytes(b"\x00" * size)
    return p


# ---------------------------------------------------------------------------
# selection / fallback logic
# ---------------------------------------------------------------------------


class TestSelection:
    def test_auto_uses_hq_when_present(self, monkeypatch, hq_config):
        models_dir, cfg = hq_config
        _write_hq(models_dir, cfg, 4096)
        monkeypatch.setattr(sepmod, "RoformerSeparator", StubRoformer)
        monkeypatch.setattr(sepmod, "MDXSeparator", StubMDX)

        sel = sepmod.build_separator(separator_config=cfg, models_dir=models_dir)

        assert sel.backend == "htdemucs"
        assert sel.fell_back is False
        assert sel.provider == "DmlExecutionProvider"
        assert sel.on_gpu is True
        assert "high-quality" in sel.reason

    def test_auto_falls_back_when_hq_absent(self, monkeypatch, hq_config):
        models_dir, cfg = hq_config  # nothing written
        monkeypatch.setattr(sepmod, "RoformerSeparator", ExplodingRoformer)
        monkeypatch.setattr(sepmod, "MDXSeparator", StubMDX)

        sel = sepmod.build_separator(
            separator_config=cfg, models_dir=models_dir,
            mdx_model_path=models_dir / "mdx.onnx",
        )

        assert sel.backend == "mdx-net"
        assert sel.fell_back is True
        assert "not found" in sel.reason

    def test_forced_mdx_ignores_present_hq(self, monkeypatch, hq_config):
        models_dir, cfg = hq_config
        cfg = SeparatorConfig(hq_model_file="hq.onnx", hq_min_bytes=8, backend="mdx")
        _write_hq(models_dir, cfg, 4096)
        # If the HQ backend were even *considered*, this would raise.
        monkeypatch.setattr(sepmod, "RoformerSeparator", ExplodingRoformer)
        monkeypatch.setattr(sepmod, "MDXSeparator", StubMDX)

        sel = sepmod.build_separator(
            separator_config=cfg, models_dir=models_dir,
            mdx_model_path=models_dir / "mdx.onnx",
        )

        assert sel.backend == "mdx-net"
        assert sel.fell_back is False
        assert "configured" in sel.reason

    def test_htdemucs_falls_back_when_hq_load_fails(self, monkeypatch, hq_config):
        models_dir, cfg = hq_config
        cfg = SeparatorConfig(hq_model_file="hq.onnx", hq_min_bytes=8, backend="htdemucs")
        _write_hq(models_dir, cfg, 4096)
        monkeypatch.setattr(sepmod, "RoformerSeparator", ExplodingRoformer)
        monkeypatch.setattr(sepmod, "MDXSeparator", StubMDX)

        sel = sepmod.build_separator(
            separator_config=cfg, models_dir=models_dir,
            mdx_model_path=models_dir / "mdx.onnx",
        )

        assert sel.backend == "mdx-net"
        assert sel.fell_back is True
        assert "failed to load" in sel.reason

    def test_truncated_hq_file_is_ignored(self, monkeypatch, tmp_path):
        # File exists but is below the presence floor -> treated as absent.
        cfg = SeparatorConfig(hq_model_file="hq.onnx", hq_min_bytes=1000, backend="auto")
        (tmp_path / "hq.onnx").write_bytes(b"\x00" * 8)
        monkeypatch.setattr(sepmod, "RoformerSeparator", ExplodingRoformer)
        monkeypatch.setattr(sepmod, "MDXSeparator", StubMDX)

        sel = sepmod.build_separator(
            separator_config=cfg, models_dir=tmp_path,
            mdx_model_path=tmp_path / "mdx.onnx",
        )
        assert sel.backend == "mdx-net"
        assert sel.fell_back is True

    def test_reports_which_backend_is_active(self, monkeypatch, hq_config):
        models_dir, cfg = hq_config
        _write_hq(models_dir, cfg, 4096)
        monkeypatch.setattr(sepmod, "RoformerSeparator", StubRoformer)
        monkeypatch.setattr(sepmod, "MDXSeparator", StubMDX)

        sel = sepmod.build_separator(separator_config=cfg, models_dir=models_dir)
        summary = sel.summary()
        assert "htdemucs" in summary
        assert "DmlExecutionProvider" in summary
        assert "GPU" in summary

    def test_hard_error_only_when_even_mdx_unavailable(self, monkeypatch, tmp_path):
        """A missing HQ model is normal; a missing MDX model is a real error."""
        cfg = SeparatorConfig(backend="mdx")

        def boom(*a, **k):
            raise SeparationError("MDX model not found")

        monkeypatch.setattr(sepmod, "MDXSeparator", boom)
        with pytest.raises(SeparationError):
            sepmod.build_separator(
                separator_config=cfg, models_dir=tmp_path,
                mdx_model_path=tmp_path / "mdx.onnx",
            )

    def test_backend_alias_normalisation(self):
        assert SeparatorConfig(backend="BS-Roformer").normalised_backend() == "htdemucs"
        assert SeparatorConfig(backend="demucs").normalised_backend() == "htdemucs"
        assert SeparatorConfig(backend="mdx-net").normalised_backend() == "mdx"
        assert SeparatorConfig(backend="nonsense").normalised_backend() == "auto"


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_defaults(self):
        cfg = Settings().separator
        assert cfg.backend == "auto"
        assert cfg.hq_model_file.endswith(".onnx")
        assert cfg.hq_vocals_index == 3
        assert cfg.hq_num_stems == 4

    def test_round_trip_through_json(self):
        s = Settings()
        s.separator.backend = "htdemucs"
        s.separator.hq_vocals_index = 0
        s.separator.hq_num_stems = 2
        restored = Settings.from_dict(json.loads(json.dumps(s.to_dict())))
        assert restored.separator == s.separator

    def test_old_settings_without_separator_still_load(self):
        data = Settings().to_dict()
        data.pop("separator")
        restored = Settings.from_dict(data)
        assert restored.separator == SeparatorConfig()

    def test_unknown_separator_key_is_dropped(self):
        data = Settings().to_dict()
        data["separator"]["future_flag"] = 123
        data["separator"]["backend"] = "mdx"
        restored = Settings.from_dict(data)
        assert restored.separator.backend == "mdx"
        assert not hasattr(restored.separator, "future_flag")


# ---------------------------------------------------------------------------
# output-stem contract, via a fake ONNX session (no real weights)
# ---------------------------------------------------------------------------


def _sine_mix(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    left = 0.4 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    return np.vstack([left, right]).astype(np.float32)


class TestRoformerContract:
    def test_output_shape_dtype_and_sum(self):
        seg = 20000
        mix = _sine_mix(1.5)  # longer than one segment -> multiple chunks
        sess = FakeSession(seg=seg, num_stems=4)
        sep = RoformerSeparator(
            "unused.onnx", WaveformParams(segment_samples=0, vocals_index=3, num_stems=4),
            session=sess,
        )
        vocals, accomp = sep.separate(mix)

        assert vocals.shape == mix.shape
        assert accomp.shape == mix.shape
        assert vocals.dtype == np.float32
        assert accomp.dtype == np.float32
        assert np.isfinite(vocals).all()
        # The exact-sum guarantee the crossfade depends on.
        assert np.allclose(vocals + accomp, mix, atol=1e-5)
        # More than one chunk actually ran.
        assert len(sess.calls) > 1

    def test_reads_segment_length_from_onnx_shape(self):
        sess = FakeSession(seg=12345, num_stems=4)
        sep = RoformerSeparator(
            "unused.onnx", WaveformParams(segment_samples=0), session=sess,
        )
        assert sep.params.segment_samples == 12345

    def test_dynamic_shape_without_config_is_rejected(self):
        sess = FakeSession(seg=8000, input_shape=[1, 2, "frames"])
        with pytest.raises(SeparationError):
            RoformerSeparator(
                "unused.onnx", WaveformParams(segment_samples=0), session=sess,
            )

    def test_overlap_add_reconstructs_a_linear_model(self):
        """A model that returns input*0.4 must reconstruct to input*0.4 after
        windowed overlap-add — proving the reconstruction is unbiased."""
        seg = 16000
        mix = _sine_mix(1.2)
        sess = FakeSession(seg=seg, num_stems=4)  # vocals row = input * 0.4
        sep = RoformerSeparator(
            "unused.onnx",
            WaveformParams(segment_samples=0, vocals_index=3, num_stems=4, overlap=0.25),
            session=sess,
        )
        vocals, _ = sep.separate(mix)
        assert np.allclose(vocals, mix * 0.4, atol=1e-4)

    def test_vocals_index_selects_the_right_stem(self):
        seg = 16000
        mix = _sine_mix(0.5)  # single chunk
        sess = FakeSession(seg=seg, num_stems=4)
        # index 0 -> *0.1, index 3 -> *0.4
        sep0 = RoformerSeparator(
            "u.onnx", WaveformParams(segment_samples=0, vocals_index=0), session=sess,
        )
        v0, _ = sep0.separate(mix)
        assert np.allclose(v0, mix * 0.1, atol=1e-4)

        sep3 = RoformerSeparator(
            "u.onnx", WaveformParams(segment_samples=0, vocals_index=3), session=sess,
        )
        v3, _ = sep3.separate(mix)
        assert np.allclose(v3, mix * 0.4, atol=1e-4)

    def test_two_stem_export(self):
        seg = 16000
        mix = _sine_mix(0.5)
        sess = FakeSession(seg=seg, num_stems=2)
        sep = RoformerSeparator(
            "u.onnx", WaveformParams(segment_samples=0, vocals_index=0, num_stems=2),
            session=sess,
        )
        vocals, accomp = sep.separate(mix)
        assert vocals.shape == mix.shape
        assert np.allclose(vocals + accomp, mix, atol=1e-5)

    def test_short_audio_single_chunk(self):
        seg = 100000
        mix = _sine_mix(0.3)  # shorter than one segment
        sess = FakeSession(seg=seg, num_stems=4)
        sep = RoformerSeparator(
            "u.onnx", WaveformParams(segment_samples=0), session=sess,
        )
        vocals, accomp = sep.separate(mix)
        assert vocals.shape == mix.shape
        assert len(sess.calls) == 1

    def test_rejects_non_stereo_input(self):
        sess = FakeSession(seg=8000)
        sep = RoformerSeparator("u.onnx", WaveformParams(segment_samples=0), session=sess)
        with pytest.raises(SeparationError):
            sep.separate(np.zeros((1, 5000), dtype=np.float32))
        with pytest.raises(SeparationError):
            sep.separate(np.zeros(5000, dtype=np.float32))

    def test_on_gpu_reflects_provider(self):
        sess = FakeSession(seg=8000, provider="DmlExecutionProvider")
        sep = RoformerSeparator("u.onnx", WaveformParams(segment_samples=0), session=sess)
        assert sep.on_gpu is True
        assert sep.backend == "htdemucs"


# ---------------------------------------------------------------------------
# orchestration: separate_to_files writes stems and reports the backend
# ---------------------------------------------------------------------------


class TestSeparateToFiles:
    def test_writes_stems_and_reports_backend(self, monkeypatch, tmp_path):
        src = tmp_path / "source.wav"
        mix = _sine_mix(0.2)
        sf.write(str(src), mix.T, SAMPLE_RATE, subtype="FLOAT")

        monkeypatch.setattr(sepmod, "MDXSeparator", StubMDX)
        voc, acc = tmp_path / "vocals.flac", tmp_path / "accompaniment.flac"
        detail = sepmod.separate_to_files(
            src, voc, acc,
            settings=Settings(separator=SeparatorConfig(backend="mdx")),
        )
        assert voc.exists() and acc.exists()
        assert detail["backend"] == "mdx-net"
        assert detail["fell_back"] is False
        assert "reason" in detail
        data, _ = sf.read(str(voc), always_2d=True)
        assert data.shape[1] == 2


# ---------------------------------------------------------------------------
# real MDX-Net end to end — only if the (small, ~66 MB) model is already on disk
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not modelstore.is_present("separator"),
    reason="MDX-Net model not downloaded; run: python -m singcoach.cli fetch-models",
)
class TestMDXEndToEnd:
    def test_real_mdx_output_contract(self):
        cfg = SeparatorConfig(backend="mdx")
        sel = sepmod.build_separator(separator_config=cfg, prefer_gpu=True)
        assert sel.backend == "mdx-net"

        mix = _sine_mix(3.0)
        vocals, accomp = sel.separator.separate(mix)

        assert vocals.shape == mix.shape
        assert vocals.dtype == np.float32
        assert accomp.dtype == np.float32
        assert np.isfinite(vocals).all()
        # Vocal + accompaniment must reconstruct the mix (the crossfade relies
        # on this), and nothing may clip.
        assert np.allclose(vocals + accomp, mix, atol=1e-4)
        assert np.max(np.abs(vocals)) <= 1.0 + 1e-6
