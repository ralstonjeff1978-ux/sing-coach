"""Command-line entry point.

Mostly a development and verification surface — the real product is the GUI —
but ``import`` and ``info`` are genuinely useful for batch-preparing a library
before a practice session, since analysis is the slow part.

    python -m singcoach.cli import "F:\\music\\song.mp3"
    python -m singcoach.cli list
    python -m singcoach.cli info <hash>
    python -m singcoach.cli doctor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, config, modelstore
from .library import cache, importer
from .library.song import Song


_last_bar: tuple[str, int] = ("", -1)


def _bar(stage: str, frac: float) -> None:
    """Redraw only when something visibly changed.

    Without this a long download repaints hundreds of identical lines, which is
    unreadable and, when stdout is a pipe rather than a terminal, enormous.
    """
    global _last_bar
    pct = int(frac * 100)
    if (stage, pct) == _last_bar:
        return
    _last_bar = (stage, pct)

    width = 28
    filled = int(frac * width)
    sys.stdout.write(f"\r  {stage:<26} [{'#' * filled}{'.' * (width - filled)}] {pct:3d}%")
    sys.stdout.flush()
    if frac >= 1.0:
        sys.stdout.write("\n")
        _last_bar = ("", -1)


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_import(args: argparse.Namespace) -> int:
    for raw in args.paths:
        path = Path(raw)
        print(f"\n{path.name}")
        try:
            song = importer.import_song(path, progress=_bar, force=args.force)
        except (FileNotFoundError, importer.UnsupportedAudio, RuntimeError) as exc:
            print(f"  ERROR: {exc}")
            continue
        m = song.meta
        print(f"  hash      {m.hash}")
        print(f"  name      {m.display_name}")
        print(f"  duration  {_fmt_duration(m.duration_s)}")
        print(f"  source    {m.source_codec} {m.source_sample_rate} Hz, {m.source_channels} ch")
        if m.integrated_lufs is not None:
            print(f"  loudness  {m.integrated_lufs:.1f} LUFS "
                  f"(peak {m.true_peak_dbfs:.1f} dBTP) "
                  f"-> playback gain {song.playback_gain(config.load_settings().target_lufs):.2f}x")
        print(f"  cache     {song.paths.root}  ({_fmt_size(song.paths.size_on_disk())})")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    hashes = cache.list_cached()
    if not hashes:
        print("No songs imported yet.  python -m singcoach.cli import <file>")
        return 0
    print(f"{'hash':<18}{'dur':>6}  {'stages':<28}name")
    print("-" * 90)
    for h in hashes:
        try:
            song = Song.load(h)
        except (FileNotFoundError, ValueError):
            print(f"{h:<18}{'?':>6}  {'(unreadable metadata)':<28}")
            continue
        stages = ",".join(song.meta.stages_done) or "-"
        print(f"{h:<18}{_fmt_duration(song.meta.duration_s):>6}  {stages:<28}{song.meta.display_name}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    try:
        song = Song.load(args.hash)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    m = song.meta
    print(f"{m.display_name}")
    print(f"  hash          {m.hash}")
    print(f"  source        {m.source_path}")
    print(f"  duration      {_fmt_duration(m.duration_s)}")
    print(f"  loudness      {m.integrated_lufs} LUFS, peak {m.true_peak_dbfs} dBTP")
    print(f"  stages done   {', '.join(m.stages_done) or '-'}")
    print(f"  separated     {song.is_separated}")
    print(f"  analysed      {song.is_analysed}")
    print(f"  lyrics        {song.has_lyrics}")
    print(f"  cache size    {_fmt_size(song.paths.size_on_disk())}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import pipeline

    for hash_ in args.hashes:
        try:
            song = Song.load(hash_)
        except FileNotFoundError as exc:
            print(exc)
            continue

        print(f"\n{song.meta.display_name}  ({_fmt_duration(song.meta.duration_s)})")
        try:
            results = pipeline.analyse_song(
                song,
                progress=_bar,
                force=args.force,
                prefer_gpu=not args.cpu,
                skip_lyrics=args.no_lyrics,
                **({"lyrics_model": args.lyrics_model} if args.lyrics_model else {}),
            )
        except (RuntimeError, OSError) as exc:
            print(f"  ERROR: {exc}")
            continue

        for r in results:
            if r.skipped:
                print(f"  {r.name:<12} cached")
            else:
                detail = "  ".join(f"{k}={v}" for k, v in r.detail.items())
                print(f"  {r.name:<12} {detail}")
        print(f"  cache        {_fmt_size(song.paths.size_on_disk())}")
    return 0


def cmd_melody(args: argparse.Namespace) -> int:
    """Print the extracted melody — the verification surface for M1."""
    import numpy as np

    from .analysis import pipeline

    try:
        song = Song.load(args.hash)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    data = pipeline.load_analysis(song)
    mel, kb = data["melody"], data["key_beats"]
    notes = mel["notes"]

    print(f"{song.meta.display_name}")
    print(f"  key            {kb['key']} {kb['mode']} (confidence {kb['key_confidence']})")
    alts = kb.get("tempo_alternatives") or []
    alt_txt = f"  (also countable as {', '.join(str(a) for a in alts)})" if alts else ""
    print(f"  tempo          {kb['tempo_bpm']} BPM grid, {len(kb['beats'])} beats{alt_txt}")
    print(f"  notes          {len(notes)}")
    print(f"  voiced         {mel['voiced_fraction']:.1%}")
    print(f"  range          MIDI {mel['range_low_midi']} - {mel['range_high_midi']}")
    print(f"  expressiveness {mel['expressiveness']:.3f}  -> target the "
          f"{'CONTOUR (expressive)' if mel['suggested_mode'] == 'expressive' else 'NOTE GRID'}")

    vib = [n for n in notes if n.get("vibrato_rate_hz")]
    if vib:
        print(f"  vibrato        {len(vib)} notes, "
              f"{np.mean([n['vibrato_rate_hz'] for n in vib]):.1f} Hz avg, "
              f"{np.mean([n['vibrato_depth_cents'] for n in vib]):.0f} cents deep")
    slides = [n for n in notes if abs(n.get("slide_semitones", 0)) > 1.0]
    print(f"  slides         {len(slides)} notes move >1 semitone within the note")

    if args.limit:
        print(f"\n  {'start':>7} {'dur':>6} {'note':>6} {'off':>6} {'conf':>5}  detail")
        for n in notes[: args.limit]:
            name = librosa_note(n["midi"])
            extra = []
            if n.get("vibrato_rate_hz"):
                extra.append(f"vib {n['vibrato_rate_hz']}Hz/{n['vibrato_depth_cents']:.0f}c")
            if abs(n.get("slide_semitones", 0)) > 1.0:
                extra.append(f"slide {n['slide_semitones']:+.1f}")
            if abs(n.get("scoop_cents", 0)) > 40:
                extra.append(f"scoop {n['scoop_cents']:+.0f}c")
            print(f"  {n['start']:7.2f} {n['end'] - n['start']:6.2f} {name:>6} "
                  f"{n['cents_offset']:+6.0f} {n['confidence']:5.2f}  {', '.join(extra)}")
    return 0


def librosa_note(midi: float) -> str:
    """Note name without the unicode sharp, which cp1252 consoles cannot print."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    m = int(round(midi))
    return f"{names[m % 12]}{m // 12 - 1}"


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Check that the environment can actually do the work."""
    ok = True
    print(f"SingCoach {__version__}")
    print(f"  root                {config.ROOT}")

    try:
        ffmpeg_path, _ = __import__(
            "singcoach.library.ffmpeg", fromlist=["require_ffmpeg"]
        ).require_ffmpeg()
        print(f"  ffmpeg              OK  ({ffmpeg_path})")
    except RuntimeError as exc:
        print(f"  ffmpeg              MISSING — {exc}")
        ok = False

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            print(f"  ONNX runtime        OK  {ort.__version__}, DirectML (GPU) available")
        else:
            print(f"  ONNX runtime        {ort.__version__}, CPU only — separation will be slow.")
            print("                      Fix: scripts/setup.ps1 (DirectML must install last)")
            ok = False
    except ImportError:
        print("  ONNX runtime        MISSING")
        ok = False

    try:
        import numba
        import numpy

        if tuple(int(x) for x in numpy.__version__.split(".")[:2]) >= (2, 5):
            print(f"  numpy/numba         numpy {numpy.__version__} is too new for numba "
                  f"{numba.__version__}; pYIN will be very slow.")
            ok = False
        else:
            print(f"  numpy/numba         OK  (numpy {numpy.__version__}, numba {numba.__version__})")
    except ImportError as exc:
        print(f"  numpy/numba         MISSING — {exc}")
        ok = False

    try:
        import sounddevice as sd

        default_out = sd.query_devices(kind="output")["name"]
        default_in = sd.query_devices(kind="input")["name"]
        print(f"  audio out           {default_out}")
        print(f"  audio in            {default_in}")
    except Exception as exc:  # sounddevice raises bare Exception subclasses
        print(f"  audio devices       PROBLEM — {exc}")
        ok = False

    for key in ("separator",):
        state = "present" if modelstore.is_present(key) else "not downloaded yet"
        print(f"  model[{key}]    {state}")

    print()
    print("All good." if ok else "Problems found — see above.")
    return 0 if ok else 1


def cmd_fetch_models(args: argparse.Namespace) -> int:
    for key in args.keys or ["separator"]:
        print(f"{key}: {modelstore.MODELS[key].filename}")
        try:
            path = modelstore.ensure(key, progress=_bar)
        except modelstore.DownloadFailed as exc:
            print(f"  ERROR: {exc}")
            return 1
        print(f"  -> {path}  ({_fmt_size(path.stat().st_size)})")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="singcoach", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="import one or more audio files into the cache")
    imp.add_argument("paths", nargs="+")
    imp.add_argument("--force", action="store_true", help="re-import even if cached")
    imp.set_defaults(func=cmd_import)

    lst = sub.add_parser("list", help="list cached songs")
    lst.set_defaults(func=cmd_list)

    inf = sub.add_parser("info", help="show details for one cached song")
    inf.add_argument("hash")
    inf.set_defaults(func=cmd_info)

    ana = sub.add_parser("analyze", help="run separation + melody + key/beats")
    ana.add_argument("hashes", nargs="+")
    ana.add_argument("--force", action="store_true", help="redo cached stages")
    ana.add_argument("--cpu", action="store_true", help="skip the GPU provider")
    ana.add_argument("--no-lyrics", action="store_true", help="skip transcription")
    ana.add_argument("--lyrics-model", default=None,
                     help="whisper size override, e.g. small.en (faster, less accurate)")
    ana.set_defaults(func=cmd_analyze)

    mel = sub.add_parser("melody", help="show the extracted melody for a song")
    mel.add_argument("hash")
    mel.add_argument("--limit", type=int, default=25, help="notes to list (0 for none)")
    mel.set_defaults(func=cmd_melody)

    doc = sub.add_parser("doctor", help="check the environment is set up correctly")
    doc.set_defaults(func=cmd_doctor)

    fm = sub.add_parser("fetch-models", help="download the separation model(s)")
    fm.add_argument("keys", nargs="*", choices=sorted(modelstore.MODELS) or None)
    fm.set_defaults(func=cmd_fetch_models)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.ensure_dirs()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
