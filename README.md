# SubtitleMaker

A small Tk desktop app that turns a batch of audio or video files into `.srt`
subtitles using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Pick files, pick a model, press start — it works through the queue and writes
each `.srt` next to its source file.

Runs on Windows, Linux, and macOS.

## Features

- **Batch processing** — select any number of files; progress is reported per
  file and per segment.
- **Transcribe or translate** — Whisper's translate task outputs English.
- **Finds the language by listening to the whole file** — not just its
  opening, which is usually music or silence and is how a Japanese film
  gets transcribed as Norwegian from end to end.
- **Drops what Whisper invents over silence** — the stock "Thank you for
  watching." held for thirty seconds, and lines it repeats until the screen
  is full.
- **Cues cut on the words, not on the segment** — Whisper returns whatever it
  closed a window on, which is often a whole sentence. Word timings say where
  the pauses and the full stops were, so a long one is divided between cues
  that each begin and end on real speech.
- **Readable cues** — folded to subtitle width, never more lines than a frame
  should carry, never on screen too briefly to read, never overlapping.
- **GPU or CPU** — CUDA when available, with automatic fallback to a compute
  type the hardware actually supports.
- **Survives corrupt media** — packets are demuxed and decoded one at a time,
  so a single bad packet drops that packet instead of silently truncating the
  rest of the transcription
  ([faster-whisper#988](https://github.com/SYSTRAN/faster-whisper/issues/988)).
- **Resumable batches** — files that already have subtitles can be skipped, and
  a running batch can be stopped after the current file.
- **Built-in log panel** — collapsible, with a rotating `error.log` on disk.
- **Remembers your choices** — model, device, precision, language and the
  option switches all come back as you left them.

## Requirements

- Python 3.13
- tkinter (bundled on Windows and macOS; a separate package on Linux)
- Optional: an NVIDIA GPU with CUDA 12 for a large speedup

On Linux both X11 and Wayland sessions work. Tk has no native Wayland
backend, so under Wayland the window is served by XWayland — which is present
by default on every mainstream desktop, and is the only extra requirement.

## Installation

Clone the repository, then create the virtual environment.

### Windows

```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux

tkinter cannot be installed by pip. Install it from your distribution first:

```
sudo apt install python3-tk        # Debian/Ubuntu
sudo dnf install python3-tkinter   # Fedora
sudo pacman -S tk                  # Arch
```

Then:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### GPU support

CUDA needs cuBLAS and cuDNN 9, neither of which pip installs as part of the
requirements. On Windows they come from the CUDA 12.8 toolkit on `PATH` and
from `C:\Windows\System32`. On Linux, install your distribution's CUDA runtime
and `libcudnn9` packages, or add the wheels:

```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

None of this is needed for CPU-only use — choose **CPU** in the device
dropdown.

## Running

| Platform | Command |
| --- | --- |
| Windows | `SubtitleMaker.bat` (or double-click it) |
| Linux / macOS | `./SubtitleMaker.sh` |

Both launchers use the project's own `.venv` and print recovery instructions if
it is missing. To run it directly instead: `python SubtitleMaker.py`.

## Options

### Models

The first run of any model downloads it from Hugging Face and caches it.

| Model | Capability |
| --- | --- |
| `large-v3` (default), `large-v2`, `large-v1`, `medium`, `small`, `base`, `tiny` | Multilingual — transcribe or translate to English |
| `large-v3-turbo` | Multilingual transcription, but never trained to translate |
| `medium.en`, `small.en`, `base.en`, `tiny.en` | English audio only |
| `distil-large-v3.5`, `distil-large-v3`, `distil-large-v2`, `distil-medium.en`, `distil-small.en` | English audio only — faster than the sizes they are distilled from |
| `kotoba-tech/kotoba-whisper-v2.0-faster` | Japanese audio only, and not trained to translate |
| `nyrahealth/faster_CrisperWhisper` | English audio only — transcribes verbatim, disfluencies included |

The box is editable: anything else that CTranslate2 can load works too, either
a Hugging Face repository id or the path to a folder holding a converted
model. Capability is read off the name, so a model that is neither on this
list nor recognisable from it is left alone rather than guessed at.

The app shows the selected model's capability underneath the dropdown and warns
before starting if the choice conflicts with the requested task.

### Device and precision

**Device** is `Auto`, `GPU (CUDA)`, or `CPU`. **Precision** defaults to `Auto`
— `float16` on GPU, `int8` on CPU — and can be pinned to `float16`,
`int8_float16`, `int8`, or `float32`. An unsupported choice falls back to one
the hardware reports as available, with a warning in the log rather than a
crash.

### Language

Auto-detect, or one of English, Japanese, Indonesian, Chinese, Korean, French,
Spanish, German, Italian, Portuguese, Russian, Arabic, Hindi, Dutch.

This selects the **spoken** language of the source, not the output language.
Whisper's translate task only ever produces English; there is no model support
for translating into any other language.

Auto-detect listens at six points spread across the file, choosing them from
where speech was actually found, and takes the majority verdict. Whisper itself
reads the language off the beginning of whatever it is handed, and the
beginning of a film is a logo sting, a music bed or silence — which is how a
Japanese film gets identified as Norwegian and then transcribed as Norwegian
from end to end. On two of three test files the opening thirty seconds gave the
wrong language outright (`nn` at 66%, `en` at 54%); sampling across the file
gave `ja` at 71% and 99%.

The log always says which language was used and how sure it was, and warns when
it is under 50%. If a transcript comes back as nonsense, that line is the first
place to look — set the language by hand and re-run.

### Processing options

| Option | Default | Effect |
| --- | --- | --- |
| Batched inference | on | Batch size 8. Much faster; trades VRAM for speed. |
| Skip files that already have subtitles | on | Lets an interrupted batch be re-run cheaply. |

## Output

Each subtitle file is written beside its source: `video.mp4` produces
`video.srt`. When two sources in one batch would collide — `video.mp4` and
`video.mkv` in the same folder — the second keeps its extension and becomes
`video.mkv.srt` so neither result is lost.

Cues are tidied on the way out:

- **Divided where the speech divides.** A segment too wide or too long to be
  one cue is split between several, cut at a pause or a full stop where there
  is one near the even division and at the even division where there is not.
  Each piece is timed from its own first and last word.
- **Folded to width.** Lines are balanced onto as few lines as will hold them
  at 42 columns, counting East Asian characters as the two columns they are
  drawn in, so Japanese folds at 21 characters rather than running off the
  side of the frame. A cue is never allowed more than two lines; a third is
  what splitting exists to avoid.
- **Long enough to read.** A cue Whisper closes almost as soon as it opens is
  given at least a second, and a denser one as long as its width takes to read
  at 20 columns a second. Both come out of the gap after the cue and never out
  of the next one, so neither is guaranteed when speech is continuous.
- **Never overlapping.** No subtitle outlives the start of the next, and a
  segment with no usable word timings is capped at 10 seconds.
- **Nothing invented.** See [Troubleshooting](#troubleshooting) for the three
  kinds of made-up line that are removed automatically.

On a 196-second sample of continuous synthesized speech read by `large-v3`,
cutting on words rather than on segments took the cues that needed three or
more lines from 6 of 11 to 0 of 20 — six lines at worst before, two after —
and removed both cases where the 10-second cap had taken a line off the screen
while its own words were still being spoken. The transcript is untouched by
any of it: 185 words either way. Asking for word timings cost 2% of
transcription time on that run (median of five, batched, CUDA, `float16`).

## Files

| Path | Purpose |
| --- | --- |
| `SubtitleMaker.py` | The whole application |
| `test_SubtitleMaker.py` | Tests for the cue and hallucination rules |
| `SubtitleMaker.bat` / `SubtitleMaker.sh` | Launchers |
| `requirements.txt` | Pinned dependencies |
| `error.log` | Rotating log, 1 MB × 3 (git-ignored) |
| `settings.json` | Model, device, language and option choices (git-ignored) |

`error.log` records the full path and filename of every file processed. It is
git-ignored for that reason — check before sharing it.

## Tests

```
.venv/Scripts/python.exe -m unittest -v        # Windows
.venv/bin/python -m unittest -v                # Linux, macOS
```

No model is loaded and nothing is downloaded, so the suite finishes in under a
second. It covers what decides the contents of an `.srt`: how a cue is cut,
folded and timed, which lines are thrown out for having been invented rather
than heard, and what each model can be asked to do. Those are all thresholds, and the awkward cases are real ones measured
from transcripts — the tests say where each came from, because the numbers mean
nothing without it.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tkinter'` on Linux** — tkinter is a
system package, not a pip one. See [Installation](#linux).

**`no display name and no $DISPLAY environment variable` on Linux** — the app
reached a session with neither X11 nor XWayland available. On a deliberately
Wayland-only setup, install your distribution's `xorg-xwayland` package.

**Blurry or undersized text on a HiDPI Wayland desktop** — XWayland does not
inherit the compositor's fractional scaling, so Tk sizes itself for 96 DPI.
Launch with a scaling factor to compensate:

```
GDK_DPI_SCALE=1 ./SubtitleMaker.sh
```

or set Tk's own factor by adding `root.tk.call('tk', 'scaling', 1.5)` in
`main()`.

**CUDA is not used even with a GPU present** — the log states the resolved
device and compute type on every run. A missing cuDNN 9 is the usual cause;
faster-whisper cannot load CUDA without it.

**Out of memory on GPU** — turn off batched inference, or use a smaller model
or `int8` precision.

**The whole transcript is in the wrong language, or reads as nonsense** — the
language was detected wrongly. It is picked by listening across the file rather
than to its opening, which fixes the common case, but quiet or heavily mixed
speech can still fool it. The log line beginning `heard` says which language was
used and how sure it was. Set **Spoken language** by hand and re-run.

Note that a wrong language does not produce gibberish — it produces fluent,
plausible text in the wrong language, which is easy to mistake for a bad
transcription. If the output reads like a different film, check that line.

**Subtitles stop partway through, or whole stretches are empty** — every second
of audio is transcribed, so this is not the app skipping anything. It is either
audio that genuinely stops (the log warns when less audio was decoded than the
container promised, which points at a damaged file), or speech quiet enough
that Whisper itself returns nothing for it. A larger model is the only real
remedy for the latter.

**Repeated or nonsense lines, or one line repeated until it fills the screen**
— a known Whisper failure mode on silence, not a decoding bug. Three kinds are
removed automatically, with no option to turn off and nothing lost when there
is nothing to remove:

- A line smeared over a long span - "Thank you for watching." held for the
  whole thirty seconds, or its Japanese equivalent - is Whisper filling a
  window it heard nothing in. Any cue lasting eight seconds or more at under
  one character a second is dropped. Over 579 cues from three files in two
  languages that caught 18 invented lines and no real ones; the slowest
  genuine line ran at 1.5 characters a second.

  The eight-second gate is also what lets a shorter invention through, so
  Whisper's own reading of the window is used to lower it. Where it reports
  that nothing was said there, three seconds is enough. That reading is never
  a reason to drop a line by itself — it belongs to the whole window rather
  than to one cue, and a line still has to fail the rate test above to go —
  and the log says how many were caught only because of it, so the threshold
  can be judged against a real run.
- A line that is one short unit repeated to the end of the window - the same
  syllable or phrase over and over - is Whisper stuck in a loop. A cue of
  eight seconds or more that is 90% one repeated unit is dropped. That caught
  11 more over the same 579 cues. Length is what makes this safe: real speech
  repeats too, and a word said three times in a row scores just as high, but
  it lasts under two seconds and is kept.
- Runs of three or more identical consecutive lines are collapsed to one.

The log says how many of each were dropped, and debug logging prints them.
None of the three ever skips audio: every second of the file is still
transcribed, and only the lines themselves are judged.
