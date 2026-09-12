# SingCoach

A local, real-time singing coach. Load a song you own; SingCoach separates the
vocal from the music, works out the melody the original singer actually sang,
and scrolls it past you while your microphone pitch is drawn on top — with an
unambiguous readout of whether to go **higher**, **lower**, or hold.

One slider covers both ways you'd want to practise:

| Vocal stem | What you get |
|---|---|
| 0 % | Karaoke. The lead is gone; you are the lead. |
| 15–30 % | A guide track. Quiet enough that you lead, loud enough that you can't get lost. |
| 100 % | The original is there — sing a harmony against it. |

**Everything runs on this machine.** The only network access in the entire app
is a one-time model download.

## Requirements

- Windows, Python 3.12
- ffmpeg on PATH (`winget install Gyan.FFmpeg`)
- **Headphones.** Not optional — see below.
- A GPU helps but isn't required. On an AMD card, separation runs through
  DirectML; the CPU fallback is the same model, just slower.

### Why headphones are mandatory

On open speakers, your microphone hears the backing track. The pitch detector
then tracks *the song* instead of *you* — the app would end up grading the
record. SingCoach checks for this at startup and warns you, but the fix is
always the same: wear headphones.

## Setup

```powershell
.\scripts\setup.ps1
```

Use that script rather than `pip install -r requirements.txt`. `faster-whisper`
depends on stock `onnxruntime`, which installs over the top of
`onnxruntime-directml` — same module name, last one wins. If stock wins, GPU
separation silently disappears and everything just runs ~8× slower with no
warning. The script installs the DirectML build last and then verifies it.

Check your environment any time:

```powershell
.venv\Scripts\python -m singcoach.cli doctor
```

## Usage

```powershell
.\run.ps1
```

Then **Song → Open** and pick a track. First load runs the analysis pipeline
(a couple of minutes); every load after that is instant.

Before your first real session, do these two things once:

1. **Setup → Audio devices and latency.** Until this is measured, the app is
   comparing your voice against the wrong moment of the song by an unknown
   amount. It takes about five seconds.
2. **Setup → Find my vocal range.** Slide down to your lowest comfortable note,
   then up to your highest. This is what lets the app tell you which key a song
   should be in for *your* voice.

### Keyboard

| Key | Action |
|---|---|
| `Space` | play / pause |
| `←` `→` | back / forward 5 seconds |
| `Home` | restart |
| `Ctrl+O` | open a song |
| `Ctrl+E` | export a cover |
| `Ctrl+,` | audio setup |

### Command line

Useful for preparing a batch of songs in advance, since analysis is the slow part.

```powershell
.venv\Scripts\python -m singcoach.cli import "F:\music\song.mp3"
.venv\Scripts\python -m singcoach.cli analyze <hash>
.venv\Scripts\python -m singcoach.cli melody <hash>     # what it heard
.venv\Scripts\python -m singcoach.cli list
.venv\Scripts\python -m singcoach.cli doctor            # check the setup
```

Supported inputs: mp3, mp4/m4a, aac, wav, flac, ogg, opus, wma, aiff, mkv, webm.

## Recording your own cover

Press **Record** while the song plays and your take is saved. Record as many
as you like — a verse in one pass, a chorus in another — then **Song → Export
cover** to mix the ones you tick against the backing track and render an MP3.

Each take stores the round-trip latency in force when it was recorded, and
export shifts it back by that amount. Without this your voice would sit
80–150 ms behind the music and sound like poor timing rather than the
measurement error it actually is.

The optional vocal chain (high-pass, gentle compression, a little reverb, a
limiter) helps a close-mic take sit in the mix. It will not fix pitch, and it is
not trying to.

## The practice features

All four are off by default — with nothing enabled this is a plain sing-along,
which is a perfectly good way to use it.

- **Fit to my range** — shifts the whole song so the melody sits where you sing
  comfortably. Audio and targets move together, so nothing drifts out of step.
- **Slow down and loop** — practise a phrase at 60–90% speed without changing
  its pitch. Speeds are pre-rendered and cached, so switching is instant and
  free of phase-vocoder artefacts.
- **Score me** — per-note accuracy plus diagnosis of *habits*. Sessions record
  the conditions they were sung under, so a run at 70% speed with a guide vocal
  is never compared against a cold run at full tempo.
- **Show harmony lines** — draws a third and a fifth above the melody, using the
  detected key so the third is the right one. When you pick a harmony line,
  scoring re-targets to it; otherwise singing a correct harmony would count as
  failure.

## How it works

**Offline, once per song, then cached forever** (keyed by a hash of the file
contents, so re-importing or renaming costs nothing):

1. **Import** — ffmpeg decodes to 44.1 kHz stereo float32. Loudness is
   *measured*, not applied, so cached audio stays faithful to the source and
   levelling happens at playback.
2. **Separate** — an MDX-Net ONNX model splits vocal from accompaniment.
3. **Melody** — pYIN on the isolated vocal produces an f0 contour, which is
   segmented into notes.
4. **Key & beats** — drives harmony guides and snaps loop points to the bar.
5. **Lyrics** — Whisper on the isolated vocal (far more accurate than on the
   full mix) gives word-level timings.

**Live:** a YIN pitch detector reads your microphone ~86×/second, guarded by a
noise gate, a confidence gate, and octave-error repair.

### Three things this gets right that most tuners don't

**It knows when the note grid is a lie.** Every song is measured for how
expressively it is sung. On material built from blue notes, scoops and melisma,
the target becomes the original singer's actual pitch contour rather than a
grid of semitones — otherwise a correctly sung bent third reads as "40 cents
flat" and the app coaches the style out of you. Measured on the first real song
tested: 43% of its notes sit more than 25 cents off the grid.

**It diagnoses, it doesn't just score.** "72%" tells you nothing you didn't
already know. Being flat on everything, flat only up high, accurate but
unsteady, and accurate but late are four different problems needing four
different practice sessions, and the app names which one you have — but only
when there is enough evidence to be sure.


**It teaches shape, not just notes.** Alongside the quantised note blocks, the
app draws the original singer's raw pitch contour — the scoops into notes, the
vibrato, the slides. That's the difference between hitting a pitch and singing
a phrase.

**It says "I don't know" when it doesn't.** Rap, spoken word, dense backing
harmonies and heavy effects all defeat pitch tracking. Those regions are marked
*no target* and shown as a neutral band. A confidently wrong target is worse
than no target — it teaches you the wrong thing.

## Testing

`tests/make_fixture.py` synthesises a song whose every note we chose, so the
melody extractor can be checked against ground truth rather than against a
guess. It deliberately includes what breaks naive pitch trackers: vibrato, a
portamento slide, a wide leap, rests, and a note near the noise floor.

```powershell
.venv\Scripts\python tests\make_fixture.py
.venv\Scripts\python -m pytest
```

## Layout

```
singcoach/
  config.py        paths + persisted settings
  library/         import, content-addressed cache, song metadata
  analysis/        offline: separation, melody, key/beats, lyrics
  audio/           playback engine, mixer, capture, latency calibration
  pitch/           realtime detection, smoothing, comparison
  scoring/         session scoring and history
  ui/              Qt front end
models/            downloaded on first run (gitignored)
cache/             per-song derived audio + analysis (gitignored)
data/              settings.json, history.db (gitignored)
```
