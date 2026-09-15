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

The box is editable: anything else that CTranslate2 can load works too, either
a Hugging Face repository id or the path to a folder holding a converted
model. Capability is read off the name, so a model that is neither on this
list nor recognisable from it is left alone rather than guessed at.

#### Which one to pick

Measured against the published English subtitles of two feature films, 146 and
149 minutes, every model through the same cue rules. The error rate is
computed over time windows, so a model that puts the right words at the wrong
moment is charged twice — once where the word is missing and once where it
reappears. Widening the window to fifteen minutes forgives displacement but
not a wrong word, which is what separates the two error columns.

| Model | Speed | Placed | Transcript | Onsets within 0.5s |
| --- | --- | --- | --- | --- |
| `large-v3-turbo` | 74× / 63× | **0.148 / 0.233** | 0.126 / 0.208 | **61.7% / 62.2%** |
| `large-v3` | 49× / 48× | 0.176 / 0.264 | 0.136 / 0.241 | 51.5% / 57.0% |
| `distil-large-v3.5` | 70× / 58× | 0.405 / 0.369 | **0.117 / 0.189** | 25.6% / 32.1% |
| `medium.en` | 60× / — | 0.187 / — | 0.158 / — | 61.4% / — |

`large-v3-turbo` leads on speed, on placement and on cue onsets on both films,
and is still not the default. There is one model box rather than one per task,
and turbo was never trained to translate — so defaulting to it would leave
half of what the app offers broken until you noticed the warning. `large-v3`
is the one that does both.

**If you only transcribe English, pick `large-v3-turbo` from the list.** It is
faster and better at it on both films measured, and the translate task it
cannot do is one you are not asking for.

`distil-large-v3.5` heard more words than anything else on both films and
placed them worst by a wide margin. Its cue onsets scatter across 1.6 seconds
against turbo's 0.5, and that is scatter rather than a constant lag, so no
offset corrects it. It is the wrong choice here specifically because every cue
boundary is now cut from word timings.

#### Translating Japanese

Whisper's translate task only ever outputs English, and `large-v3-turbo` was
never trained on it, so a translation has to start somewhere else — the app
warns and offers `large-v3`.

Measured on a 110-minute Japanese film against its published English subtitle,
which is a human translation of the same dialogue:

| Model | Speed | chrF | Onsets within 0.5s | Words |
| --- | --- | --- | --- | --- |
| `large-v2` | 39× | 0.456 | 71.6% | 8,063 |
| `large-v3` | 40× | 0.426 | 57.1% | 7,455 |

That gap did not survive a second film. A 125-minute Japanese one carries a
published translation into a third language — no use for scoring content, but
cue onsets are where the speech is whatever language the words are in, and
there the two models come out level: 50.8% against 48.9% within half a second,
with `large-v2` the wider spread of the two at 1.45 seconds against 1.00.

So the translation default stays `large-v3`. One film showed a wide margin,
the next showed none, and a difference that does not reproduce is not a
difference worth changing a default for. `large-v2` is in the list for anyone
who wants to try it on their own material.

Word error rate is the wrong tool for a translation — two good translations of
one line share meaning and almost no exact wording, and both of these score
about 0.98 against the reference, which says nothing. chrF is what machine
translation is normally judged by; it forgives rewording and still punishes
content that is not there. It is scored here in 30-second windows, because
scored across a whole film it mostly reports that both texts are English: an
*unrelated* English film reaches 0.660 that way, against 0.203 windowed. The
same model transcribing rather than translating scores 0.010, which is the
check that the measure discriminates at all.

Neither number means anything on its own. A translation that agrees with
another translation at 0.456 may be perfectly good.

#### Models that are not offered

Both of these load and both are real; neither is in the dropdown, and both can
still be typed into it deliberately.

- `kotoba-tech/kotoba-whisper-v2.0-faster` — **crashes the process** when word
  timestamps are asked for, which this app always does. Its configuration
  carries the alignment heads of the model it was distilled from, pointing at
  decoder layer 25 of a decoder that no longer has one, and the lookup runs
  off the end. It transcribes normally with word timestamps turned off. The
  fault is in how the model is packaged and there is nothing here that can
  detect it in advance: CTranslate2 does not expose a layer count, and the
  crash is a segmentation fault, which cannot be caught.
- `nyrahealth/faster_CrisperWhisper` — transcribes verbatim, which is what it
  is for, but on feature-film audio it returned 8,583 words where the others
  returned about 13,000 against a reference carrying 14,098, at an error rate
  of 0.564 against their 0.12 to 0.16.

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

Measured against the published English subtitles of two feature films, and
the published translation of a third, all read by `large-v3`:

| | reference | before | after |
| --- | --- | --- | --- |
| cues needing 3+ lines, 146-min film | 1 | 156 | 0 |
| cues needing 3+ lines, 149-min film | 1 | 120 | 0 |
| most lines in any one cue | 3 | 22 | 2 |
| cues faster than 20 columns/s | 80 | 91 | 61 |
| reference cue onsets matched within 0.5s | — | 47.0% | 51.5% |
| word error rate against the reference | — | 0.198 | 0.176 |

The transcript itself is untouched by any of it; the lower error rate is the
same words landing in the right time window. That reference is a published
subtitle rather than a verbatim transcript, so the rate is a floor on
agreement and not a count of mistakes — useful for comparing two builds
against one yardstick, not as an absolute.

Japanese is the case where none of this changes much: its segments already fit
two lines at 21 characters, so almost nothing needs splitting, and the cue
onsets moved by hundredths of a second. Asking for word timings cost 2% of
transcription time (median of five runs, batched, CUDA, `float16`).

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

  A loop can also come back short and dense - one unbroken token hundreds of
  columns wide, held for about a second - and the length gate is exactly what
  lets it through. Such a cue is dropped on its rate instead, above 100
  columns a second. Over 4887 segments from three feature films in two
  languages, eleven were both short and 90% one repeated unit: ten were
  genuine and none exceeded 26 columns a second, and the one invented line ran
  at 636.
- Runs of three or more identical consecutive lines are collapsed to one.

The log says how many of each were dropped, and debug logging prints them.
None of the three ever skips audio: every second of the file is still
transcribed, and only the lines themselves are judged.
