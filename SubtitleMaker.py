"""SubtitleMaker - batch subtitle generation with faster-whisper.

Transcribes audio/video files to .srt, optionally translating to English.

Note on translation: Whisper's "translate" task only ever outputs English.
There is no model support for translating into any other language, so the
language selector below picks the *source* language, not the target.
"""

import io
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
import tkinter as tk
from dataclasses import dataclass
from datetime import timedelta
from math import ceil
from logging.handlers import RotatingFileHandler
from tkinter import filedialog, font as tkfont, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import av
import ctranslate2
import numpy as np
import srt
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(APP_DIR, "error.log")
CONFIG_PATH = os.path.join(APP_DIR, "settings.json")

# Every size faster-whisper knows by name. The dropdown is editable, so any
# other CTranslate2 repository id or local directory can be typed into it
# instead of picked from this list - but only what is listed here has been run
# against a feature film and measured, and two that were are deliberately
# absent. See "Models that are not offered" in the README.
MODELS = (
    "large-v3",
    "large-v3-turbo",
    "large-v2",
    "large-v1",
    "medium",
    "small",
    "base",
    "tiny",
    "medium.en",
    "small.en",
    "base.en",
    "tiny.en",
    "distil-large-v3.5",
    "distil-large-v3",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
)
# large-v3-turbo measured better on both films this was tested against -
# faster, closer to the reference, and better at placing its cues - and it is
# still not the default, because it was never trained to translate and there
# is one model box rather than one per task. A default that silently cannot do
# half of what the app offers is worse than a slower one that can do all of
# it. Turbo is first among the alternatives for anyone transcribing English
# and not translating; the README carries the figures.
DEFAULT_MODEL = "large-v3"

# Named models whose capability cannot be read off the name. Everything else
# is recognised by its suffix or prefix below. These two are not offered in
# the dropdown but are still recognised, because the box is editable and
# someone may well type one in.
ENGLISH_ONLY_MODELS = {"nyrahealth/faster_CrisperWhisper"}
JAPANESE_ONLY_MODELS = {"kotoba-tech/kotoba-whisper-v2.0-faster"}
# Capabilities that may be asked for a translation without the request being
# quietly ignored. A custom model is included because nothing here knows what
# it was trained for - the warning would be a guess.
CAN_BE_ASKED_TO_TRANSLATE = {"full", "custom"}

MODEL_NOTES = {
    "english_only": "English audio only - cannot handle Japanese or other languages.",
    "japanese_only": "Japanese audio only, and not trained to translate.",
    "no_translation": "Multilingual transcription, but not trained to translate.",
    "custom": "Custom model - whichever languages and tasks it was trained for.",
    "full": "Multilingual. Can transcribe or translate to English.",
}


def model_capability(model_size):
    """What a model can be asked to do, read off its name.

    The suffix and prefix rules cover the whole stock list: ".en" weights were
    trained on English audio alone, every distil-whisper release is
    English-only whatever its size, and the turbo weights skipped the
    translation task entirely. A name that matches none of them and looks like
    a repository id or a path is somebody else's model, and nothing here can
    say what it does.
    """
    name = model_size.strip()
    lowered = name.lower()
    if name in JAPANESE_ONLY_MODELS or "kotoba" in lowered:
        return "japanese_only"
    if (name in ENGLISH_ONLY_MODELS
            or lowered.endswith(".en")
            or lowered.startswith("distil-")):
        return "english_only"
    # OpenAI: "the turbo model will return the original language even if
    # --task translate is specified."
    # https://github.com/openai/whisper#available-models
    if "turbo" in lowered:
        return "no_translation"
    if "/" in name or os.sep in name:
        return "custom"
    return "full"

# Label -> (ctranslate2 device, default compute type for that device)
DEVICES = {
    "Auto": ("auto", "default"),
    "GPU (CUDA)": ("cuda", "float16"),
    "CPU": ("cpu", "int8"),
}
DEFAULT_DEVICE = "Auto"

PRECISIONS = ("Auto", "float16", "int8_float16", "int8", "float32")

AUTO_DETECT = "Auto-detect"
LANGUAGES = {
    AUTO_DETECT: None,
    "English": "en",
    "Japanese": "ja",
    "Indonesian": "id",
    "Chinese": "zh",
    "Korean": "ko",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Dutch": "nl",
}

MEDIA_EXTENSIONS = (
    "mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "wma",
    "mp4", "mkv", "avi", "mov", "webm", "wmv", "flv", "ts", "m4v",
)


def _media_filetypes():
    """Build the open dialog's filters, which are platform-sensitive.

    On Windows and macOS Tk calls the native file dialog, which matches
    case-insensitively and reads "*.*" as "everything". On Unix it draws its
    own dialog and matches with Tcl's glob, which is case-sensitive and only
    matches "*.*" against names containing a dot - so a plain "*.mp4" hides
    MOVIE.MP4 and "*.*" hides extensionless files. Character classes fix the
    first, "*" the second.

    This follows the Tk build, not the display server: Tk has no native
    Wayland backend and runs through XWayland, so a Wayland session gets the
    same Unix dialog and needs the same patterns.
    """
    if sys.platform == "win32":
        return [
            ("Audio/Video files",
             " ".join("*." + ext for ext in MEDIA_EXTENSIONS)),
            ("All files", "*.*"),
        ]

    def any_case(ext):
        return "".join(
            "[%s%s]" % (c, c.upper()) if c.isalpha() else c for c in ext
        )

    return [
        ("Audio/Video files",
         " ".join("*." + any_case(ext) for ext in MEDIA_EXTENSIONS)),
        ("All files", "*"),
    ]


MEDIA_TYPES = _media_filetypes()

# Batches trade VRAM for speed; 8 is comfortable on 8GB+ cards.
BATCH_SIZE = 8
# Whisper's encoder window. Batched inference only transcribes the chunks it is
# handed, and none of them may be longer than this.
CHUNK_SECONDS = 30.0
# Don't repaint the GUI more often than this (seconds) - progress fires per segment.
UI_REFRESH_INTERVAL = 0.1
# How often the main thread drains queued log records and GUI updates, and how
# many lines the log panel keeps before trimming the oldest.
UI_POLL_INTERVAL_MS = 100
LOG_PANEL_MAX_LINES = 2000

# Whisper always works at 16 kHz mono. The fifo size only trades resampler
# call overhead against memory, and mirrors what faster-whisper uses.
AUDIO_SAMPLE_RATE = 16000
AUDIO_FIFO_SAMPLES = 500_000
# Warn if decoded audio falls short of the container duration by more than this.
DURATION_TOLERANCE = 0.02
# Longest a single subtitle may stay on screen, in seconds.
MAX_SUBTITLE_DURATION = 10.0
# Shortest, too. Whisper can close a cue almost as soon as it opens it, and a
# quarter-second flash is unreadable however correct it is. A cue is only
# stretched into the gap that follows it, never over the next line.
MIN_SUBTITLE_DURATION = 1.0
# And a cue carrying more than a moment's reading is held for at least as
# long as it takes to read, in columns a second. Like the minimum above, this
# is only ever satisfied out of the gap after a cue, so it reaches a line that
# ends before a pause and cannot reach one in the middle of continuous speech
# - there is no time there to give it. Counting columns rather than characters
# makes one number serve both scripts, East Asian text being read at about
# half the character rate and drawn at twice the width. Broadcast guidance
# sits between 15 and 20.
MAX_READING_RATE = 20.0
# Columns a subtitle line may occupy before it is folded, counting East Asian
# characters as the two columns they are drawn in. 42 is the usual broadcast
# figure and about what a player will show without shrinking the text.
SUBTITLE_LINE_COLUMNS = 42
# Lines a single cue may occupy. Two is the broadcast convention; a third
# starts covering the picture and outruns the time the cue is on screen for.
SUBTITLE_MAX_LINES = 2
# What those lines hold if nothing is wasted at the end of them, which is
# only ever used to estimate how many cues a segment needs. Words do not
# divide evenly into lines, so a cue this wide can still fold onto one line
# too many; what a cue may actually hold is settled by folding it and
# counting, in overflows() below.
CUE_MAX_COLUMNS = SUBTITLE_LINE_COLUMNS * SUBTITLE_MAX_LINES
# A gap this long between two words is somewhere the speaker stopped, and so
# somewhere a cue may end without cutting a phrase in half. Measured against
# ordinary delivery, a shorter gap than this is just the space between words.
CUE_BREAK_PAUSE = 0.35
# Punctuation that closes a thought, and so also ends a cue well. The clause
# marks are a weaker signal than the sentence ones but still beat cutting at
# whatever word happens to reach the column limit.
SENTENCE_END = frozenset(".。．!！?？…")
CLAUSE_END = frozenset(",、，;；:：")
# How short of its even share a cue may be cut when a pause or a full stop
# offers itself there. A break one word early at a full stop reads better than
# a break at the column limit; a break at the first comma in a long sentence
# does not, because it leaves the rest with everything still to carry.
NATURAL_BREAK_SHARE = 0.5
# Punctuation that may never be left stranded at the start of a folded line.
NEVER_STARTS_A_LINE = frozenset("、。，．！？・）」』］｝,.!?)]}")
# Length of a run of identical consecutive cues that counts as Whisper looping
# rather than someone genuinely repeating themselves.
REPEAT_RUN_MIN = 3
# Neither invention test below judges a cue that ran for less than this. Both
# read a cue's span as evidence, and a short cue has too little to give: real
# speech is full of brief lines that look wrong by either measure.
SUSPECT_MIN_SECONDS = 8.0
# A hallucination over silence is a short phrase smeared across a long span:
# Whisper fills the window with something plausible and stretches it to fit.
# Measured over 579 cues from three files in two languages, every cue this slow
# and this long was invented - "Thank you for watching." at 0.67 characters a
# second, the notorious one, and 17 others - while the slowest real line ran at
# 1.5. Spaces are not counted, so the rate means the same thing in Japanese as
# in English.
HALLUCINATION_MAX_RATE = 1.0
# The other kind: Whisper latches onto a unit and repeats it until the window
# is full. Over the same 579 cues, everything at or above this share was a loop
# and nothing real came close once the length gate had its say - the most
# repetitive genuine line scored 0.85 and lasted four seconds.
LOOP_MIN_REPETITION = 0.9
# Below this many characters a repetition score means nothing either way.
LOOP_MIN_CHARS = 8
# The length gate above is also a hole. A loop can come back short and dense -
# one token hundreds of columns wide, held for a second - and pass for speech
# on the strength of not having run long. Nobody says that much that fast, and
# the gap is not a close one: over 4887 segments from three films in two
# languages, eleven were both short and at least 90% one repeated unit, ten of
# them genuine at no more than 26 columns a second, and the eleventh, which
# was a single token of 890 columns, at 636. This bar sits between the two
# with four times the headroom below it and six times above.
IMPOSSIBLE_RATE = 100.0
# Whisper reports, for each window it decodes, how sure it is that nothing was
# said in it. A window it calls silent still comes back with text whenever the
# decoder had to produce something, and that text was invented. The figure is
# too coarse to act on alone - it belongs to the whole window rather than to
# one cue - so it is never a reason to drop a line by itself. It only lowers
# the bar the test below already applies, which is what makes it safe: two
# independent signals have to agree before anything is thrown away.
NO_SPEECH_SUSPECT = 0.6
# With that agreement a stretched line does not have to run the full eight
# seconds to be recognisable, which is the whole point - the eight-second gate
# is what lets shorter inventions through.
SUSPECT_MIN_SECONDS_UNHEARD = 3.0
# Windows of speech sampled to identify the language, spread evenly across a
# file. Whisper reads the language off the start of whatever it is handed, and
# the start of a file is very often music, a logo sting or silence - which is
# how a Japanese film ends up transcribed as Norwegian from end to end.
LANGUAGE_SAMPLE_WINDOWS = 6
# Below this confidence the guess is worth saying out loud: everything that
# follows is written in whichever language was picked.
LANGUAGE_CONFIDENCE_WARN = 0.5
# Speech detection settings. These only decide where to listen from when
# naming the language - nothing is ever skipped on the detector's say-so.
# Measured on this material it hears as little as a tenth of the speech in a
# file, so it is trusted to find some speech and never to find all of it.
# Pauses shorter than this stay inside a region rather than splitting it.
SILENCE_GAP_SECONDS = 4.0
# Kept either side of a region, so a soft onset survives into the sample.
SPEECH_PAD_SECONDS = 0.5
# Silero's own default is 0.5, which writes off quiet and sung speech.
VAD_SPEECH_THRESHOLD = 0.3

WINDOW_WIDTH = 620
WINDOW_HEIGHT = 800
MIN_WIDTH = 560
# Lines reserved for the status text, and the length past which a filename in
# it is elided. Together they keep the status line a constant height.
STATUS_LINES = 2
STATUS_NAME_MAX = 60


def setup_logging():
    handlers = [
        RotatingFileHandler(
            LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    ]

    # Only log to the console when there is one. Under pythonw.exe there is no
    # console, and the stream is cp1252 anyway - media filenames containing CJK
    # characters would raise UnicodeEncodeError on every write.
    if sys.stderr is not None:
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    # Keep the log panel readable: these libraries log every HTTP call at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config():
    """Read persisted UI preferences. Never fails - falls back to defaults."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
        logging.warning("%s is not a JSON object; ignoring it.", CONFIG_PATH)
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        logging.warning("Could not read %s (%s); using defaults.", CONFIG_PATH, exc)
    return {}


def save_config(config):
    """Persist UI preferences. A failure here must never break the app."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
    except OSError as exc:
        logging.warning("Could not save %s (%s).", CONFIG_PATH, exc)


@dataclass(frozen=True)
class Settings:
    """A snapshot of the GUI controls.

    Tk variables may only be read from the main thread, so the worker gets
    plain values captured at start time instead of live widget references.
    This also means changing a dropdown mid-batch can't affect a running job.
    """

    model_size: str
    device_label: str
    precision_label: str
    language: "str | None"
    translate: bool
    batched: bool
    skip_existing: bool


class QueueLogHandler(logging.Handler):
    """Hand log records to the GUI thread without touching Tk from a worker.

    logging.Handler.emit runs on whichever thread logged the message, so it
    cannot write to a widget directly. It only enqueues here; the Tk main loop
    drains the queue on a timer.
    """

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put_nowait((record.levelno, self.format(record)))
        except Exception:  # noqa: BLE001 - logging must never break the app
            self.handleError(record)


def resolve_compute_type(device, precision_label):
    """Pick a compute type the current hardware actually supports."""
    if precision_label != "Auto":
        requested = precision_label
    else:
        requested = "float16" if device in ("cuda", "auto") else "int8"

    probe = device
    if probe == "auto":
        probe = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    supported = ctranslate2.get_supported_compute_types(probe)
    if requested in supported:
        return requested

    fallback = "int8" if "int8" in supported else "float32"
    logging.warning(
        "Compute type %s unsupported on %s (available: %s); using %s instead.",
        requested, probe, sorted(supported), fallback,
    )
    return fallback


def unique_srt_path(media_path, taken):
    """Pick an .srt path, disambiguating video.mp4 vs video.mkv collisions."""
    stem, ext = os.path.splitext(media_path)
    candidate = stem + ".srt"
    if candidate not in taken:
        return candidate
    # Fold the source extension in: "video.mkv" -> "video.mkv.srt"
    return f"{stem}{ext}.srt"


def shorten(name, limit=STATUS_NAME_MAX):
    """Elide the middle of a long filename, keeping the start and extension."""
    if len(name) <= limit:
        return name
    head = (limit - 3) // 2
    return name[:head] + "..." + name[len(name) - (limit - 3 - head):]


def speech_regions(audio):
    """Locate speech in decoded audio, as {start, end} pairs in seconds.

    Used only to decide where to listen from when naming the language. It is
    not used to choose what gets transcribed, and must not be: on quiet or
    heavily mixed audio this detector hears a fraction of what is said, so
    skipping whatever it misses throws away a fifth of the dialogue. Picking
    listening posts tolerates that, because it only needs to find some speech.

    Returns an empty list when it hears nothing at all.
    """
    chunks = get_speech_timestamps(
        audio,
        VadOptions(
            threshold=VAD_SPEECH_THRESHOLD,
            min_silence_duration_ms=int(SILENCE_GAP_SECONDS * 1000),
            speech_pad_ms=int(SPEECH_PAD_SECONDS * 1000),
        ),
    )
    return [
        {
            "start": chunk["start"] / AUDIO_SAMPLE_RATE,
            "end": chunk["end"] / AUDIO_SAMPLE_RATE,
        }
        for chunk in chunks
    ]


def split_clips(regions):
    """Cut regions into windows the batched pipeline can swallow.

    BatchedInferencePipeline only transcribes the clips it is given, and
    without them it refuses to guess: any file longer than one encoder window
    raises "No clip timestamps found". It also transcribes just the first
    CHUNK_SECONDS of any clip and warns about the rest, so the timeline has to
    arrive pre-cut. Windows stay absolute, and faster-whisper leaves
    caller-supplied clips where they are, so no timestamp remapping follows.

    A region is divided evenly rather than sliced into full windows with the
    remainder trailing behind. Both cost the same number of passes, but an even
    split never leaves a two-second sliver padded out with silence, which is
    exactly the input Whisper invents dialogue for.
    """
    clips = []
    for region in regions:
        span = region["end"] - region["start"]
        windows = max(1, ceil(span / CHUNK_SECONDS))
        for window in range(windows):
            clips.append(
                {
                    "start": region["start"] + span * window / windows,
                    "end": region["start"] + span * (window + 1) / windows,
                }
            )
    return clips


def language_montage(audio, regions):
    """Windows of speech taken at even spacing through a file, joined up.

    Only ever used to name the language. Whisper reads that off the start of
    whatever it is given, and the start of a file is the least representative
    part of it - a logo sting, a music bed, a quiet establishing shot. Hand it
    speech drawn from the whole running time instead. Splicing does no harm
    here: this audio is never transcribed or timed, only listened to.
    """
    window = int(CHUNK_SECONDS * AUDIO_SAMPLE_RATE)
    if not regions:
        return audio[: LANGUAGE_SAMPLE_WINDOWS * window]

    speech = sum(region["end"] - region["start"] for region in regions)
    picks = []
    for index in range(LANGUAGE_SAMPLE_WINDOWS):
        # Walk this far into the speech, counting only the speech.
        target, seen = speech * (index + 0.5) / LANGUAGE_SAMPLE_WINDOWS, 0.0
        for region in regions:
            length = region["end"] - region["start"]
            if seen + length >= target:
                at = int((region["start"] + target - seen) * AUDIO_SAMPLE_RATE)
                picks.append(audio[at:at + window])
                break
            seen += length
    return np.concatenate(picks) if picks else audio[:window]


def detect_language(model, audio, regions, name):
    """Name the spoken language, listening across the whole file."""
    language, confidence, _ = model.detect_language(
        audio=language_montage(audio, regions),
        language_detection_segments=LANGUAGE_SAMPLE_WINDOWS,
        # Score every sample and let the majority carry it. The default settles
        # on the first window to clear a low bar, which over a quiet opening is
        # exactly how a confident wrong answer gets in.
        language_detection_threshold=1.0,
    )
    if confidence < LANGUAGE_CONFIDENCE_WARN:
        logging.warning(
            "%s: heard %s, but only %.0f%% sure. Everything is transcribed as "
            "that language, so set the language by hand if the result reads "
            "like nonsense.", name, language, confidence * 100,
        )
    else:
        logging.info(
            "%s: heard %s (%.0f%% confidence).", name, language, confidence * 100,
        )
    return language


def decode_audio(path, should_stop=None):
    """Decode to 16 kHz mono float32, skipping corrupt packets.

    faster-whisper decodes with a single PyAV generator, and its
    _ignore_invalid_frames helper cannot actually recover: once the generator
    raises InvalidDataError it is exhausted, so the `continue` immediately hits
    StopIteration and the rest of the file is dropped without an error. A
    single bad packet therefore truncates the whole transcription silently.
    See https://github.com/SYSTRAN/faster-whisper/issues/988

    MP4s remuxed from MPEG-TS hit this constantly, because stream glitches in
    the original broadcast survive the remux. Demuxing and decoding one packet
    at a time lets us drop only the bad packets and keep the rest.

    Returns (audio, skipped_packets, container_duration), or None if stopped.
    """
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=AUDIO_SAMPLE_RATE
    )
    fifo = av.audio.fifo.AudioFifo()
    buffer = io.BytesIO()
    skipped = 0

    def drain(force=False):
        if force or fifo.samples >= AUDIO_FIFO_SAMPLES:
            frame = fifo.read()
            if frame is not None:
                for out in resampler.resample(frame):
                    buffer.write(out.to_ndarray())

    with av.open(path, mode="r", metadata_errors="ignore") as container:
        container_duration = (
            container.duration / av.time_base if container.duration else 0.0
        )
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        for packet in container.demux(stream):
            if should_stop is not None and should_stop():
                return None
            try:
                frames = packet.decode()
            except av.FFmpegError:
                skipped += 1
                continue
            for frame in frames:
                frame.pts = None  # the fifo rejects non-contiguous timestamps
                fifo.write(frame)
            drain()
        drain(force=True)
        # The resampler buffers internally too; flushing it keeps the last
        # fraction of a second of audio.
        for out in resampler.resample(None):
            buffer.write(out.to_ndarray())

    audio = np.frombuffer(buffer.getbuffer(), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0, skipped, container_duration


def drop_repeat_runs(subtitles):
    """Keep only the first cue of each run of identical consecutive lines.

    Over audio with no clear speech - crowd noise or music - Whisper latches
    onto one phrase and emits it again and again for as long as the noise
    lasts. Two identical lines in a row can be real; a dozen never is.
    """
    kept = []
    run_start = 0
    for position in range(len(subtitles) + 1):
        same = (
            position < len(subtitles)
            and subtitles[position].content == subtitles[run_start].content
        )
        if same:
            continue
        run = subtitles[run_start:position]
        kept.extend(run[:1] if len(run) >= REPEAT_RUN_MIN else run)
        run_start = position
    return kept


def is_stretched(content, seconds, no_speech_prob=0.0):
    """Is this too few characters to have taken that long to say?

    Whisper closes a segment at the next speech boundary, so the span of a cue
    is really the span of the silence after it, and a line invented to fill a
    window carries the whole window's worth. Real speech keeps up a rate even
    when the cue runs long; an invented one cannot.

    The length gate is what stops the rate test judging the brief lines real
    speech is full of, and it is also what lets a shorter invention through.
    Whisper's own reading of the window - how sure it was that nothing was
    said there - lowers that gate when it agrees. It never raises it and it is
    never enough on its own, so a line still has to fail the rate test to go.
    """
    floor = (SUSPECT_MIN_SECONDS_UNHEARD if no_speech_prob >= NO_SPEECH_SUSPECT
             else SUSPECT_MIN_SECONDS)
    if seconds < floor:
        return False
    return len("".join(content.split())) / seconds < HALLUCINATION_MAX_RATE


def repetition(content):
    """What share of a line is one short unit repeated back to back, 0 to 1.

    Comparing the line against itself shifted by every plausible unit length
    finds a loop without knowing anything about the language: text that is one
    thing over and over matches itself at some shift nearly everywhere, and
    ordinary speech does not come close.
    """
    letters = "".join(content.split())
    if len(letters) < LOOP_MIN_CHARS:
        return 0.0
    # A unit has to fit at least three times over to count as repetition.
    return max(
        sum(a == b for a, b in zip(letters, letters[unit:])) / (len(letters) - unit)
        for unit in range(1, len(letters) // 3 + 1)
    )


def is_looping(content, seconds):
    """Has Whisper latched onto a phrase and filled the window with it?

    Real speech repeats too - a word said three times in a row scores a flat
    1.0 - so length is what separates the two. Saying a word three times over
    takes a second or two. A loop runs to the end of the window, however long
    that is.

    Length is not the only thing that separates them, though, and waiting for
    it lets a short, dense loop through: a token hundreds of columns wide held
    for a second reads as speech to every test here, and then folds onto
    twenty lines. Rate catches that one without touching the brief repetitions
    real speech is full of, which are short because there is little of them
    rather than because they were said impossibly fast.
    """
    if repetition(content) < LOOP_MIN_REPETITION:
        return False
    if seconds >= SUSPECT_MIN_SECONDS:
        return True
    return seconds > 0 and (
        display_width("".join(content.split())) / seconds > IMPOSSIBLE_RATE
    )


def display_width(text):
    """Columns this text occupies, counting East Asian characters as two."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _chop(unit):
    """Cut a run with no break in it down to line-sized pieces.

    Nothing said aloud looks like this, but a URL or a long identifier does,
    and one of those must not be allowed to run off the side of the frame.
    """
    while display_width(unit) > SUBTITLE_LINE_COLUMNS:
        taken = width = 0
        while taken < len(unit):
            step = display_width(unit[taken])
            if width + step > SUBTITLE_LINE_COLUMNS:
                break
            width += step
            taken += 1
        yield unit[:taken]
        unit = unit[taken:]
    if unit:
        yield unit


def _units(text):
    """Split into the smallest pieces a line may be broken between.

    A Latin word travels with its trailing space and is kept whole; an East
    Asian character is its own unit, because that writing has no spaces to
    break at and breaking between characters is what readers expect.
    """
    unit = ""
    for char in text:
        if unicodedata.east_asian_width(char) in "WF":
            if unit:
                yield from _chop(unit)
                unit = ""
            yield char
        elif char == " ":
            yield from _chop(unit + char)
            unit = ""
        else:
            unit += char
    if unit:
        yield from _chop(unit)


def wrap_caption(text):
    """Fold a cue onto as few lines as will hold it, balanced evenly.

    Whisper returns a sentence as one long run. Left alone, a line wider than
    the frame is either shrunk or cut off by the player, and East Asian text
    hits that at half the character count because each character is drawn
    double width. Lines are balanced rather than filled greedily, so a cue
    never breaks as one full line and one stray word.
    """
    text = " ".join(text.split())
    width = display_width(text)
    if width <= SUBTITLE_LINE_COLUMNS:
        return text

    lines = ceil(width / SUBTITLE_LINE_COLUMNS)
    target = width / lines
    folded, line, filled = [], "", 0
    for unit in _units(text):
        span = display_width(unit)
        # Break to stay inside the frame, or to keep the lines even. The width
        # test is not optional, so it ignores the planned line count.
        if line and (
            filled + span > SUBTITLE_LINE_COLUMNS
            or (filled >= target and len(folded) < lines - 1)
        ):
            # Closing punctuation never starts a line: let it ride past the
            # edge instead. The allowance is one full-width mark and no more,
            # so neither a run of punctuation nor a word that merely begins
            # with some can drag a line out.
            riding = (
                unit[:1] in NEVER_STARTS_A_LINE
                and filled + span <= SUBTITLE_LINE_COLUMNS + 2
            )
            if not riding:
                folded.append(line.strip())
                line, filled = "", 0
        line += unit
        filled += span
    if line.strip():
        folded.append(line.strip())
    return "\n".join(folded)


def breaks_between(earlier, later):
    """Is the join between these two words somewhere a cue may end?

    Either the speaker stopped long enough for the pause to be audible, or the
    earlier word closed a thought. Both are places a reader expects the line
    to change; the column limit is not.
    """
    if later.start - earlier.end >= CUE_BREAK_PAUSE:
        return True
    tail = earlier.word.strip()[-1:]
    return tail in SENTENCE_END or tail in CLAUSE_END


def overflows(text):
    """Would this cue need more lines than it is allowed, once folded?

    Asked of the folding itself rather than of a column count, because the
    two do not agree: a line ends at the last word that fits, so a cue of
    exactly two lines' worth of columns routinely folds onto three. Only the
    fold knows, and it is the fold that reaches the screen.
    """
    return len(wrap_caption(text).split("\n")) > SUBTITLE_MAX_LINES


def split_cue(segment):
    """Cut one segment into cues that begin and end where its words do.

    A segment is whatever Whisper closed on, which is frequently a whole
    sentence - too wide for the frame and too long to read at once. Folding it
    onto a third and fourth line is not the answer; ending one cue and
    starting another is. Word timings say where to cut and what time to give
    each piece, so a cue that used to be placed by dividing a span now lands
    on the words themselves.

    The timing matters just as much as the cut. A segment that ran longer than
    a cue is allowed to used to be clamped at ten seconds, which took the line
    off the screen while its own words were still being spoken - measured on a
    run of dense speech, two cues in eleven lost their ending that way. A cue
    cut from the words ends on the last of them, so it can neither outlast the
    speech it carries nor be taken down in the middle of it.

    Word timings are asked for but not guaranteed, and alignment can come back
    covering less than the segment said. Either way the segment is returned
    whole, timed and clamped exactly as it was before any of this.
    """
    text = segment.text.strip()
    words = [word for word in (getattr(segment, "words", None) or []) if word.word]
    if not words:
        return [(segment.start, segment.end, text)]

    # Alignment that lost or altered a character cannot be trusted to place a
    # cut, and the transcript matters more than the timing.
    spoken = "".join("".join(word.word.split()) for word in words)
    if spoken != "".join(text.split()):
        return [(segment.start, segment.end, text)]

    width = display_width(text)
    span = words[-1].end - words[0].start
    # However many cues it takes to satisfy whichever limit binds harder.
    pieces = max(1, ceil(width / CUE_MAX_COLUMNS),
                 ceil(span / MAX_SUBTITLE_DURATION))
    # Aim for even pieces rather than filling each one up: a cue should not
    # break as one full screen and one trailing word, for the same reason a
    # line should not. Filling to the limit and hoping for a pause is what
    # leaves the scrap - continuous speech offers no pause, so the break lands
    # at the margin and the remainder is a word or two with no time of its own.
    target = width / pieces

    cues, taken, filled, pending = [], [], 0, ""
    for word in words:
        if taken and (
            overflows(pending + word.word)
            or word.end - taken[0].start > MAX_SUBTITLE_DURATION
            or filled >= target
            or (filled >= target * NATURAL_BREAK_SHARE
                and breaks_between(taken[-1], word))
        ):
            cues.append((taken, pending))
            taken, filled, pending = [], 0, ""
        taken.append(word)
        filled += display_width(word.word)
        pending += word.word
    if taken:
        cues.append((taken, pending))

    # Cutting at a full stop near the end of a segment can leave a scrap - a
    # word or two, on screen for a fraction of a second, with nowhere to
    # borrow time from because the next cue begins where it ends. Hand it back
    # to the cue it was cut from whenever that one still has the room.
    joined = [cues[0]]
    for words_in, text_of in cues[1:]:
        before_words, before_text = joined[-1]
        if (words_in[-1].end - words_in[0].start < MIN_SUBTITLE_DURATION
                and not overflows(before_text + text_of)
                and words_in[-1].end - before_words[0].start
                <= MAX_SUBTITLE_DURATION):
            joined[-1] = (before_words + words_in, before_text + text_of)
        else:
            joined.append((words_in, text_of))

    # A single word too wide for the lines it is allowed is left alone here:
    # there is nowhere to cut it, and wrap_caption breaks it as a last resort.
    return [
        (words_in[0].start, words_in[-1].end, text_of.strip())
        for words_in, text_of in joined
    ]


def readable_duration(content):
    """How long this cue has to be on screen to be read at all.

    Long enough not to be a flash, and long enough that its characters do not
    go past faster than they can be taken in. Both are floors the timing pass
    reaches for inside the gap after a cue; neither is ever met by taking time
    from the cue that follows, so neither is guaranteed.
    """
    return timedelta(seconds=max(
        MIN_SUBTITLE_DURATION,
        display_width("".join(content.split())) / MAX_READING_RATE,
    ))


def build_subtitles(segments):
    """Convert Whisper segments to srt.Subtitle, dropping empty ones."""
    subtitles = []
    stretched = looping = unheard = 0
    for segment in segments:
        content = segment.text.strip()
        if not content:
            continue

        # Both tests read the segment's own span, not the span of the cues cut
        # out of it below: an invented line is invented as a whole, and it is
        # the window it was stretched across that gives it away.
        span = segment.end - segment.start
        silent = getattr(segment, "no_speech_prob", 0.0) or 0.0
        if is_stretched(content, span, silent):
            logging.debug(
                "Dropping %.0fs of silence written up as %r (no_speech_prob "
                "%.2f)", span, content, silent,
            )
            stretched += 1
            # Count what only the silence reading caught, so the threshold can
            # be judged against a real run rather than argued about.
            unheard += span < SUSPECT_MIN_SECONDS
            continue
        if is_looping(content, span):
            logging.debug("Dropping %.0fs of %r on a loop", span, content)
            looping += 1
            continue

        for begins, ends, text in split_cue(segment):
            if not text:
                continue
            start = timedelta(seconds=begins)
            end = timedelta(seconds=ends)
            # A segment with no usable word timings still arrives whole, and a
            # short line isolated in a long silence can stretch for minutes.
            # Clamping only shortens a cue, so it cannot introduce overlaps.
            if ends - begins > MAX_SUBTITLE_DURATION:
                end = start + timedelta(seconds=MAX_SUBTITLE_DURATION)
            subtitles.append(
                srt.Subtitle(index=0, start=start, end=end,
                             content=wrap_caption(text))
            )

    if stretched or looping:
        counted = [
            f"{count} {reason}"
            for count, reason in (
                (stretched, "stretched over silence"),
                (looping, "repeated until the window was full"),
            )
            if count
        ]
        logging.info(
            "Dropped %s - lines Whisper invented rather than heard; run with "
            "debug logging to see them.", " and ".join(counted),
        )
        if unheard:
            logging.info(
                "%d of those ran shorter than %.0fs and were only recognised "
                "because Whisper reported the window as silent.",
                unheard, SUSPECT_MIN_SECONDS,
            )
    subtitles = drop_repeat_runs(subtitles)

    # One pass for timing, sweeping forward. Whisper both overlaps cues by a
    # fraction of a second and closes some of them almost immediately, so every
    # cue is pulled back off the next one and then given as much of the gap
    # after it as it needs to be readable.
    sliver = timedelta(milliseconds=1)
    for position, subtitle in enumerate(subtitles):
        # Whisper can also stamp a run of segments with one instant and no
        # duration between them - measured on one 149-minute film, 23 segments
        # had no duration and 20 shared a start with the segment before. They
        # cannot all be shown at that instant, and giving each a sliver where
        # it stands puts every one of them on top of the next. Laying them out
        # one after another instead costs a millisecond apiece and keeps the
        # promise that no subtitle outlives the start of the next.
        if position and subtitle.start < subtitles[position - 1].end:
            subtitle.start = subtitles[position - 1].end
        following = subtitles[position + 1] if position + 1 < len(subtitles) else None
        needed = readable_duration(subtitle.content)
        ceiling = following.start if following else subtitle.end + needed
        # Never behind where this cue now starts: every cue gets some duration,
        # even where that leaves it only the sliver.
        ceiling = max(ceiling, subtitle.start + sliver)
        if subtitle.end > ceiling:
            subtitle.end = ceiling
        if subtitle.end - subtitle.start < needed:
            subtitle.end = min(ceiling, subtitle.start + needed)
        if subtitle.end <= subtitle.start:
            subtitle.end = subtitle.start + sliver

    # Numbered only now, so dropped repeats leave no gaps in the sequence.
    for position, subtitle in enumerate(subtitles, start=1):
        subtitle.index = position

    return subtitles


def write_srt(path, subtitles):
    """Write atomically so an interrupted run can't leave a half-written file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(srt.compose(subtitles))
    os.replace(tmp_path, path)


class SubtitleMakerApp:
    def __init__(self, root):
        self.root = root
        self.stop_event = threading.Event()
        self.worker = None
        self._last_ui_update = 0.0
        self.log_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self.config = load_config()
        self.log_visible = True
        self._expanded_height = WINDOW_HEIGHT
        self._drain_job = None
        self.log_handler = None

        root.title("SubtitleMaker - Whisper batch subtitles")
        root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_widgets()
        # minsize is derived from the real required height in here, so the
        # buttons can never be clipped in either state.
        self._on_language_change(chosen=False)
        self._apply_log_visibility(bool(self.config.get("log_visible", True)))
        self._attach_log_handler()
        self._drain_job = self.root.after(UI_POLL_INTERVAL_MS, self._drain_queues)

    # ------------------------------------------------------------------ UI

    def _build_widgets(self):
        pad = {"padx": 16, "fill": "x"}

        settings = ttk.LabelFrame(self.root, text="Model")
        settings.pack(pady=(12, 6), **pad)

        # No list of allowed values here: the box takes any CTranslate2
        # repository id or local directory, so a saved name that is not in
        # MODELS is a custom model rather than a stale setting.
        self.model_var = tk.StringVar(value=self._remembered("model", DEFAULT_MODEL))
        self._labelled_combo(
            settings, "Model:", self.model_var, MODELS, editable=True
        )
        # Traced rather than bound to <<ComboboxSelected>>, so the note keeps
        # up with a name being typed as well as one picked off the list.
        self.model_var.trace_add("write", lambda *_: self._update_model_note())

        self.model_note = ttk.Label(settings, foreground="gray", wraplength=520)
        self.model_note.pack(anchor="w", padx=16, pady=(0, 0))
        ttk.Label(
            settings,
            text="Any CTranslate2 model id or folder can be typed in here.",
            foreground="gray",
            wraplength=520,
        ).pack(anchor="w", padx=16, pady=(0, 6))

        self.device_var = tk.StringVar(
            value=self._remembered("device", DEFAULT_DEVICE, DEVICES)
        )
        self._labelled_combo(settings, "Device:", self.device_var, tuple(DEVICES))

        self.precision_var = tk.StringVar(
            value=self._remembered("precision", "Auto", PRECISIONS)
        )
        self._labelled_combo(settings, "Precision:", self.precision_var, PRECISIONS)

        language = ttk.LabelFrame(self.root, text="Language")
        language.pack(pady=6, **pad)

        self.language_var = tk.StringVar(
            value=self._remembered("language", AUTO_DETECT, LANGUAGES)
        )
        language_combo = self._labelled_combo(
            language, "Spoken language:", self.language_var, tuple(LANGUAGES)
        )
        language_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_language_change()
        )

        self.translate_var = tk.BooleanVar(value=self._remembered("translate", True))
        self.translate_check = ttk.Checkbutton(
            language,
            text="Translate to English",
            variable=self.translate_var,
            command=self._update_model_note,
        )
        self.translate_check.pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(
            language,
            text="Whisper can only translate into English.",
            foreground="gray",
        ).pack(anchor="w", padx=28, pady=(0, 8))

        options = ttk.LabelFrame(self.root, text="Options")
        options.pack(pady=6, **pad)

        self.batched_var = tk.BooleanVar(value=self._remembered("batched", True))
        ttk.Checkbutton(
            options,
            text=f"Batched inference (much faster, batch size {BATCH_SIZE})",
            variable=self.batched_var,
        ).pack(anchor="w", padx=8, pady=(6, 0))

        self.skip_existing_var = tk.BooleanVar(
            value=self._remembered("skip_existing", True)
        )
        ttk.Checkbutton(
            options,
            text="Skip files that already have subtitles",
            variable=self.skip_existing_var,
        ).pack(anchor="w", padx=8, pady=(0, 8))

        self.start_button = ttk.Button(
            self.root, text="Select files and start", command=self.start_batch
        )
        self.start_button.pack(pady=(12, 4))

        self.stop_button = ttk.Button(
            self.root, text="Stop", command=self.request_stop, state="disabled"
        )
        self.stop_button.pack()

        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(
            self.root, variable=self.progress_var, maximum=100
        ).pack(fill="x", padx=16, pady=12)

        # Reserve a fixed height for the status text. It is the only widget
        # whose length changes while a batch runs, and letting it reflow eats
        # the space the button row below needs: pack hands out what is left in
        # packing order, so with the log panel hidden the buttons are what
        # falls off the bottom of the window.
        status_area = ttk.Frame(
            self.root,
            height=(tkfont.nametofont("TkDefaultFont").metrics("linespace")
                    * STATUS_LINES),
        )
        status_area.pack(padx=16, fill="x")
        status_area.pack_propagate(False)

        self.status_label = ttk.Label(
            status_area, text="Idle", foreground="gray",
            wraplength=560, justify="left", anchor="nw",
        )
        self.status_label.pack(fill="both", expand=True)

        self.log_frame = ttk.LabelFrame(self.root, text="Log")

        log_font = self._monospace_font(9)

        self.log_text = ScrolledText(
            self.log_frame, height=10, wrap="word", state="disabled",
            font=log_font, background="#1e1e1e", foreground="#d4d4d4",
            insertbackground="#d4d4d4", borderwidth=0,
        )
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text.tag_configure("WARNING", foreground="#e5c07b")
        self.log_text.tag_configure("ERROR", foreground="#e06c75")

        # Pinned to the bottom so the log panel, packed afterwards, always
        # lands between the status line and these buttons - including when it
        # is hidden and re-shown.
        self.buttons = ttk.Frame(self.root)
        self.buttons.pack(side="bottom", pady=(0, 12))

        self.log_toggle = ttk.Button(self.buttons, command=self.toggle_log)
        self.log_toggle.pack(side="left", padx=4)
        ttk.Button(self.buttons, text="Clear", command=self.clear_log).pack(side="left", padx=4)
        ttk.Button(self.buttons, text="Open log file", command=self.open_log).pack(side="left", padx=4)

    # Ordered by preference; the first one actually installed wins.
    MONOSPACE_FAMILIES = (
        "Consolas",           # Windows
        "SF Mono", "Menlo",   # macOS
        "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono",  # Linux
    )

    def _remembered(self, key, default, allowed=None):
        """A saved choice, or the default if it is missing or no longer valid.

        A settings file can outlive the build that wrote it - a model dropped
        from the list, a hand-edited value, a file from a newer version - so
        anything that is not still on offer falls back rather than leaving the
        app pointing at something it cannot use.
        """
        value = self.config.get(key, default)
        if type(value) is not type(default):
            return default
        if allowed is not None and value not in allowed:
            logging.info("Ignoring saved %s %r; it is no longer offered.", key, value)
            return default
        return value

    def _remember(self):
        """Persist every control, so the next run opens where this one left."""
        self.config.update(
            model=self.model_var.get(),
            device=self.device_var.get(),
            precision=self.precision_var.get(),
            language=self.language_var.get(),
            translate=self.translate_var.get(),
            batched=self.batched_var.get(),
            skip_existing=self.skip_existing_var.get(),
            log_visible=self.log_visible,
        )
        save_config(self.config)

    def _monospace_font(self, size):
        """Resolve a monospace font that exists on this machine.

        Naming one family outright is unsafe: Tk silently substitutes a
        proportional face wherever that family is missing, so the log panel
        would quietly stop lining up. TkFixedFont always exists but resolves
        to Courier New on Windows, so it is the last resort rather than the
        first choice.
        """
        available = set(tkfont.families(self.root))
        for family in self.MONOSPACE_FAMILIES:
            if family in available:
                return (family, size)

        fallback = tkfont.nametofont("TkFixedFont").copy()
        fallback.configure(size=size)
        return fallback

    def _labelled_combo(self, parent, label, variable, values, editable=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=16).pack(side="left")
        combo = ttk.Combobox(
            row, textvariable=variable, state="normal" if editable else "readonly"
        )
        combo["values"] = values
        combo.pack(side="left", fill="x", expand=True)
        return combo

    # ------------------------------------------------- thread-safe GUI calls

    def post(self, func, *args):
        """Queue func to run on the Tk main thread. Safe from any thread.

        This deliberately avoids root.after(), which registers a command on the
        Tcl interpreter and is not thread-safe. Only the drain loop, which runs
        on the main thread, ever touches a widget.
        """
        self.ui_queue.put_nowait((func, args))

    def set_status(self, text):
        self.status_label.config(text=text)

    def set_progress(self, percent):
        self.progress_var.set(percent)

    def set_running(self, running):
        self.start_button.config(state="disabled" if running else "normal")
        self.stop_button.config(state="normal" if running else "disabled")
        if not running:
            self.progress_var.set(0)

    # ---------------------------------------------------------- log panel

    def _attach_log_handler(self):
        self.log_handler = QueueLogHandler(self.log_queue)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(self.log_handler)

    def _drain_queues(self):
        """Apply queued work from the worker. Runs on the Tk main thread."""
        try:
            for _ in range(200):  # bounded so a burst can't freeze the GUI
                levelno, message = self.log_queue.get_nowait()
                self._append_log(levelno, message)
        except queue.Empty:
            pass

        try:
            while True:
                func, args = self.ui_queue.get_nowait()
                try:
                    func(*args)
                except Exception:  # noqa: BLE001 - keep the drain loop alive
                    logging.error("GUI update failed:\n%s", traceback.format_exc())
        except queue.Empty:
            pass

        self._drain_job = self.root.after(UI_POLL_INTERVAL_MS, self._drain_queues)

    def _append_log(self, levelno, message):
        widget = self.log_text
        # Only auto-scroll if already at the bottom, so scrolling up to read
        # something doesn't get yanked away by the next message.
        at_bottom = widget.yview()[1] >= 0.999

        widget.configure(state="normal")
        if levelno >= logging.ERROR:
            tag = "ERROR"
        elif levelno >= logging.WARNING:
            tag = "WARNING"
        else:
            tag = ""
        widget.insert("end", message + "\n", tag)

        excess = int(widget.index("end-1c").split(".")[0]) - LOG_PANEL_MAX_LINES
        if excess > 0:
            widget.delete("1.0", f"{excess + 1}.0")
        widget.configure(state="disabled")

        if at_bottom:
            widget.see("end")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def toggle_log(self):
        if self.log_visible:
            # Remember the current height so re-showing restores it.
            self._expanded_height = self.root.winfo_height()
        self._apply_log_visibility(not self.log_visible)

        self.config["log_visible"] = self.log_visible
        save_config(self.config)

    def _apply_log_visibility(self, visible):
        self.log_visible = visible
        if visible:
            self.log_frame.pack(padx=16, pady=(10, 4), fill="both", expand=True)
            self.log_toggle.config(text="Hide log")
        else:
            self.log_frame.pack_forget()
            self.log_toggle.config(text="Show log")

        self.root.update_idletasks()
        required = self.root.winfo_reqheight()
        width = max(self.root.winfo_width(), self.root.winfo_reqwidth(), MIN_WIDTH)

        self.root.minsize(MIN_WIDTH, required)
        self.root.geometry(
            f"{width}x{max(self._expanded_height, required) if visible else required}"
        )
        # With the panel hidden nothing expands vertically, so growing the
        # window would just add dead space.
        self.root.resizable(True, visible)

    # ------------------------------------------------------------ lifecycle

    def _on_language_change(self, chosen=True):
        """Translating English into English is a no-op, so tie the two together.

        Any other source language (including auto-detect) defaults to
        translating, which is the common case for foreign-language media.
        That default belongs to picking a language, though, not to starting
        up: on startup a saved choice has to survive. English is the exception
        either way, because the box is about to be disabled.
        """
        is_english = LANGUAGES[self.language_var.get()] == "en"
        if chosen or is_english:
            self.translate_var.set(not is_english)
        self.translate_check.config(state="disabled" if is_english else "normal")
        self._update_model_note()

    def _update_model_note(self):
        capability = model_capability(self.model_var.get())
        conflict = (self.translate_var.get()
                    and capability not in CAN_BE_ASKED_TO_TRANSLATE)
        self.model_note.config(
            text=("! " if conflict else "") + MODEL_NOTES[capability],
            foreground="#b26a00" if conflict else "gray",
        )

    def _confirm_model_choice(self):
        """Warn before a model/translate combination that silently misbehaves."""
        model_size = self.model_var.get()
        capability = model_capability(model_size)
        if (not self.translate_var.get()
                or capability in CAN_BE_ASKED_TO_TRANSLATE):
            return True

        if capability == "english_only":
            detail = (
                f"'{model_size}' was trained on English audio only. On any other "
                "language it produces garbage."
            )
        elif capability == "japanese_only":
            detail = (
                f"'{model_size}' was trained on Japanese audio only, and not on "
                "the translation task at all."
            )
        else:
            detail = (
                f"'{model_size}' is not trained for translation. It returns the "
                "original language even when translation is requested."
            )

        switch = messagebox.askyesno(
            "Model cannot translate",
            f"{detail}\n\nSwitch to '{DEFAULT_MODEL}'?\n\n"
            f"Yes - switch and continue\nNo - continue with '{model_size}'",
        )
        if switch:
            logging.info("Switching from %s to %s for translation.",
                         model_size, DEFAULT_MODEL)
            self.model_var.set(DEFAULT_MODEL)
            self._update_model_note()
        return True

    def start_batch(self):
        if self.worker and self.worker.is_alive():
            return

        # The box is free text now, so it can be left empty or full of spaces.
        # Catch that here rather than in the worker, where it would surface as
        # a download failure several steps from the cause.
        if not self.model_var.get().strip():
            messagebox.showwarning(
                "No model",
                "Pick a model from the list, or type the id of a CTranslate2 "
                "model or the path to a folder holding one.",
            )
            return

        if not self._confirm_model_choice():
            return

        paths = filedialog.askopenfilenames(
            title="Select audio/video files", filetypes=MEDIA_TYPES
        )
        if not paths:
            return

        self.stop_event.clear()
        self.set_running(True)
        self.set_status("Loading model...")

        self.worker = threading.Thread(
            target=self._run_batch,
            args=(list(paths), self._collect_settings()),
            daemon=True,
        )
        self.worker.start()

    def _collect_settings(self):
        """Snapshot every control. Must run on the Tk main thread."""
        return Settings(
            model_size=self.model_var.get().strip(),
            device_label=self.device_var.get(),
            precision_label=self.precision_var.get(),
            language=LANGUAGES[self.language_var.get()],
            translate=self.translate_var.get(),
            batched=self.batched_var.get(),
            skip_existing=self.skip_existing_var.get(),
        )

    def request_stop(self):
        self.stop_event.set()
        self.stop_button.config(state="disabled")
        self.set_status("Stopping after the current file...")
        logging.info("User requested stop.")

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askokcancel(
                "Quit", "Transcription is still running. Stop it and quit?"
            ):
                return
            self.stop_event.set()

        # Cancel the pending drain before tearing down the interpreter,
        # otherwise Tk fires it against a destroyed widget tree.
        if self._drain_job is not None:
            self.root.after_cancel(self._drain_job)
            self._drain_job = None
        if self.log_handler is not None:
            logging.getLogger().removeHandler(self.log_handler)

        self._remember()
        self.root.destroy()

    def open_log(self):
        if not os.path.isfile(LOG_PATH):
            messagebox.showinfo("Log", "No log file yet.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(LOG_PATH)
            elif sys.platform == "darwin":
                subprocess.run(["open", LOG_PATH], check=False)
            else:
                subprocess.run(["xdg-open", LOG_PATH], check=False)
        except OSError as exc:
            messagebox.showerror("Error", f"Cannot open log file:\n{exc}")

    # --------------------------------------------------------- worker thread

    def _run_batch(self, paths, settings):
        done = skipped = failed = 0
        try:
            model = self._load_model(settings)
            transcribe = self._make_transcriber(model, settings)

            taken = set()
            total = len(paths)

            for index, path in enumerate(paths):
                if self.stop_event.is_set():
                    break

                name = os.path.basename(path)
                srt_path = unique_srt_path(path, taken)
                taken.add(srt_path)

                if settings.skip_existing and os.path.exists(srt_path):
                    logging.info("Skipping %s - subtitles already exist.", name)
                    skipped += 1
                    continue

                self.post(self.set_status, f"[{index + 1}/{total}] {shorten(name)}")
                try:
                    if self._transcribe_one(transcribe, path, srt_path, index, total):
                        done += 1
                    else:
                        skipped += 1
                except Exception:
                    # One bad file shouldn't abort the whole batch.
                    failed += 1
                    logging.error(
                        "Failed to process %s:\n%s", name, traceback.format_exc()
                    )

            self.post(self._finish, done, skipped, failed)

        except Exception as exc:
            logging.error("Batch failed:\n%s", traceback.format_exc())
            self.post(self._fail, str(exc))

    def _load_model(self, settings):
        device, _ = DEVICES[settings.device_label]
        compute_type = resolve_compute_type(device, settings.precision_label)

        logging.info(
            "Loading model %s (device=%s, compute_type=%s)",
            settings.model_size, device, compute_type,
        )
        return WhisperModel(
            settings.model_size, device=device, compute_type=compute_type
        )

    def _make_transcriber(self, model, settings):
        """Return a callable with a single signature for both pipelines."""
        task = "translate" if settings.translate else "transcribe"

        def prepare(audio, name):
            """Settle the language before anything is decoded."""
            language = settings.language
            if language is None:
                # Listening posts only - every second of audio is transcribed
                # either way.
                language = detect_language(
                    model, audio, speech_regions(audio), name
                )
            else:
                logging.info("%s: transcribing as %s, as set.", name, language)
            return language

        # Both pipelines take the windows planned here instead of running
        # their own silence filter, which would splice the surviving speech
        # together before Whisper ever saw it.
        #
        # Both are also asked for word timestamps, which faster-whisper reads
        # off the decoder's own cross-attention. A segment's bounds are only
        # where Whisper closed the window; the words are where the speech was,
        # and split_cue spends them on cue boundaries and cue ends.
        if settings.batched:
            pipeline = BatchedInferencePipeline(model=model)

            def batched(audio, name):
                language = prepare(audio, name)
                duration = len(audio) / AUDIO_SAMPLE_RATE

                return pipeline.transcribe(
                    audio,
                    language=language,
                    task=task,
                    clip_timestamps=split_clips(
                        [{"start": 0.0, "end": duration}]
                    ),
                    batch_size=BATCH_SIZE,
                    # Batched mode defaults to without_timestamps=True, which
                    # makes segment bounds come from merged chunks instead of
                    # from Whisper. On sparse speech that yields single
                    # subtitles many minutes long. Asking for timestamps costs
                    # ~25% speed and brings segment length back in line with
                    # sequential mode.
                    without_timestamps=False,
                    word_timestamps=True,
                )

            return batched

        def sequential(audio, name):
            return model.transcribe(
                audio,
                language=prepare(audio, name),
                task=task,
                # "0" is faster-whisper's own way of saying the whole file.
                # Sequential mode needs no windows: it advances to wherever the
                # last segment ended, so it stops on a sentence rather than
                # every thirty seconds.
                clip_timestamps="0",
                condition_on_previous_text=False,
                word_timestamps=True,
            )

        return sequential

    def _transcribe_one(self, transcribe, path, srt_path, index, total):
        name = os.path.basename(path)
        # The full name goes to the log; the status line gets an elided one so
        # it always fits the height reserved for it.
        label = shorten(name)
        logging.info("Processing file %d/%d: %s", index + 1, total, name)

        self.post(self.set_status, f"[{index + 1}/{total}] Decoding audio: {label}")
        decoded = decode_audio(path, should_stop=self.stop_event.is_set)
        if decoded is None:
            logging.info("Stopped while decoding %s", name)
            return False
        audio, skipped, container_duration = decoded

        audio_duration = len(audio) / AUDIO_SAMPLE_RATE
        if skipped:
            logging.warning(
                "%s: skipped %d corrupt audio packet(s) - common in MP4s "
                "remuxed from MPEG-TS.", name, skipped,
            )
        if container_duration and audio_duration < container_duration * (
            1 - DURATION_TOLERANCE
        ):
            logging.warning(
                "%s: decoded only %s of %s of audio - subtitles will be "
                "incomplete.", name,
                timedelta(seconds=int(audio_duration)),
                timedelta(seconds=int(container_duration)),
            )

        self.post(self.set_status, f"[{index + 1}/{total}] Transcribing: {label}")
        segment_iter, info = transcribe(audio, name)
        duration = info.duration or audio_duration

        segments = []
        for segment in segment_iter:
            if self.stop_event.is_set():
                logging.info("Stopped mid-file; discarding partial output for %s", name)
                return False
            segments.append(segment)
            self._report_progress(label, segment.end, duration, index, total)

        if not segments:
            logging.warning("No speech found in %s; no subtitles written.", name)
            return False

        subtitles = build_subtitles(segments)
        if not subtitles:
            logging.warning("Only empty segments in %s; no subtitles written.", name)
            return False

        write_srt(srt_path, subtitles)
        logging.info("Saved %d subtitles to %s", len(subtitles), srt_path)
        return True

    def _report_progress(self, label, position, duration, index, total):
        now = time.monotonic()
        if now - self._last_ui_update < UI_REFRESH_INTERVAL:
            return
        self._last_ui_update = now

        if duration > 0:
            ratio = min(position / duration, 1.0)
            self.post(self.set_progress, ((index + ratio) / total) * 100)
            self.post(
                self.set_status,
                f"[{index + 1}/{total}] {label} - {int(ratio * 100)}%",
            )

    # ------------------------------------------------- completion (main thread)

    def _finish(self, done, skipped, failed):
        self.set_running(False)
        stopped = self.stop_event.is_set()

        parts = [f"{done} transcribed"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed")
        summary = ", ".join(parts)

        if stopped:
            self.set_status(f"Stopped. {summary}.")
            logging.info("Batch stopped by user. %s", summary)
            return

        self.set_status(f"Done. {summary}.")
        logging.info("Batch complete. %s", summary)

        if failed:
            messagebox.showwarning(
                "Finished with errors", f"{summary}.\n\nDetails in error.log"
            )
        else:
            messagebox.showinfo("Done", f"{summary}.")

    def _fail(self, message):
        self.set_running(False)
        self.set_status("Error")
        messagebox.showerror("Error", f"{message}\n\nDetails in error.log")


def main():
    setup_logging()
    root = tk.Tk()
    SubtitleMakerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
