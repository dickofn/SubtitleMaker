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
- **GPU or CPU** — CUDA when available, with automatic fallback to a compute
  type the hardware actually supports.
- **Survives corrupt media** — packets are demuxed and decoded one at a time,
  so a single bad packet drops that packet instead of silently truncating the
  rest of the transcription
  ([faster-whisper#988](https://github.com/SYSTRAN/faster-whisper/issues/988)).
- **Resumable batches** — files that already have subtitles can be skipped, and
  a running batch can be stopped after the current file.
- **Built-in log panel** — collapsible, with a rotating `error.log` on disk.

## Requirements

- Python 3.13
- tkinter (bundled on Windows and macOS; a separate package on Linux)
- Optional: an NVIDIA GPU with CUDA 12 for a large speedup

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
| `large-v3` (default), `large-v2`, `medium`, `small`, `base`, `tiny` | Multilingual — transcribe or translate to English |
| `large-v3-turbo` | Multilingual transcription, but never trained to translate |
| `distil-large-v3.5` | English audio only |

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

### Processing options

| Option | Default | Effect |
| --- | --- | --- |
| Filter silence (VAD) | on | Skips silent stretches. Strongly recommended — it is what prevents Whisper from hallucinating repeated lines over silence. |
| Batched inference | on | Batch size 8. Much faster; trades VRAM for speed. |
| Skip files that already have subtitles | on | Lets an interrupted batch be re-run cheaply. |

## Output

Each subtitle file is written beside its source: `video.mp4` produces
`video.srt`. When two sources in one batch would collide — `video.mp4` and
`video.mkv` in the same folder — the second keeps its extension and becomes
`video.mkv.srt` so neither result is lost.

Overlapping segments are trimmed so no subtitle outlives the start of the next,
and any segment left open by Whisper is capped at 10 seconds.

## Files

| Path | Purpose |
| --- | --- |
| `SubtitleMaker.py` | The whole application |
| `SubtitleMaker.bat` / `SubtitleMaker.sh` | Launchers |
| `requirements.txt` | Pinned dependencies |
| `error.log` | Rotating log, 1 MB × 3 (git-ignored) |
| `settings.json` | Persisted UI preferences (git-ignored) |

`error.log` records the full path and filename of every file processed. It is
git-ignored for that reason — check before sharing it.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tkinter'` on Linux** — tkinter is a
system package, not a pip one. See [Installation](#linux).

**CUDA is not used even with a GPU present** — the log states the resolved
device and compute type on every run. A missing cuDNN 9 is the usual cause;
faster-whisper cannot load CUDA without it.

**Out of memory on GPU** — turn off batched inference, or use a smaller model
or `int8` precision.

**Repeated or nonsense lines over quiet passages** — leave "Filter silence" on.
This is a known Whisper failure mode on silence, not a decoding bug.
