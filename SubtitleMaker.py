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

MODELS = (
    "large-v3",
    "large-v2",
    "large-v3-turbo",
    "medium",
    "small",
    "base",
    "tiny",
    "distil-large-v3.5",
)
DEFAULT_MODEL = "large-v3"

# Trained on English audio only - useless for anything else.
ENGLISH_ONLY_MODELS = {"distil-large-v3.5"}
# Multilingual for transcription, but never trained on the translate task.
# OpenAI: "the turbo model will return the original language even if
# --task translate is specified." https://github.com/openai/whisper#available-models
NO_TRANSLATION_MODELS = {"large-v3-turbo", "turbo"} | ENGLISH_ONLY_MODELS

MODEL_NOTES = {
    "english_only": "English audio only - cannot handle Japanese or other languages.",
    "no_translation": "Multilingual transcription, but not trained to translate.",
    "full": "Multilingual. Can transcribe or translate to English.",
}


def model_capability(model_size):
    if model_size in ENGLISH_ONLY_MODELS:
        return "english_only"
    if model_size in NO_TRANSLATION_MODELS:
        return "no_translation"
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
# The silence filter only cuts a stretch out when the detector hears nothing
# for this long. Shorter pauses stay in, so a line keeps the rhythm around it
# instead of being spliced against a moment from minutes away.
SILENCE_GAP_SECONDS = 4.0
# Kept either side of every stretch of speech, so a soft onset or a trailing
# word never gets clipped off by the detector.
SPEECH_PAD_SECONDS = 0.5
# Silero's own default is 0.5, which writes off whispering, singing and speech
# under music. A false positive only costs a little time; a false negative
# loses subtitles outright, so lean towards keeping the audio.
VAD_SPEECH_THRESHOLD = 0.3
# Keeping less than this share of a file means the detector failed rather than
# found a quiet file, so the filter stands down instead.
VAD_MIN_KEEP_RATIO = 0.05
# Past this share skipped, say so loudly: it is the usual reason a transcript
# covers the opening minutes and then stops.
VAD_REMOVAL_WARN_RATIO = 0.6

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
    vad_filter: bool
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
    """Locate the speech in decoded audio, as {start, end} pairs in seconds.

    Only silences longer than SILENCE_GAP_SECONDS separate one region from the
    next; Silero is told to wait that long before closing a region, so ordinary
    pauses stay inside it. Returns an empty list when it hears nothing.
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


def widen_regions(regions, duration):
    """Grow speech regions until each is worth a pass, then merge what overlaps.

    A clip costs a full encoder pass whatever its length, because the features
    are padded out to a whole window either way. Handing the model a one-second
    burst therefore costs as much as a thirty-second one, and a file of
    scattered bursts would run slower with the filter on than off. Widening
    each region with the audio that actually surrounds it costs nothing,
    absorbs neighbouring bursts into the same pass, and gives Whisper the
    run-up it needs to decode a line in context.

    Nothing is rearranged: regions stay on the original timeline, so widening
    only ever takes in real audio from either side.
    """
    widened = []
    for region in regions:
        start = max(0.0, region["start"])
        end = min(duration, region["end"])
        if end - start < CHUNK_SECONDS:
            # Take the audio that follows, then whatever precedes, rather than
            # spending a pass on mostly padding.
            end = min(duration, start + CHUNK_SECONDS)
            start = max(0.0, end - CHUNK_SECONDS)
        if widened and start <= widened[-1]["end"]:
            widened[-1]["end"] = max(widened[-1]["end"], end)
            continue
        widened.append({"start": start, "end": end})
    return widened


def split_clips(regions):
    """Cut regions into windows the batched pipeline can swallow.

    BatchedInferencePipeline transcribes only the first CHUNK_SECONDS of any
    clip it is handed and warns about the rest, so regions have to arrive
    pre-cut. Windows stay absolute, and faster-whisper leaves caller-supplied
    clips on the original timeline, so no timestamp remapping follows.

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


def plan_regions(audio, name, use_vad):
    """Choose which stretches of a file to transcribe, or None for all of it.

    The silence filter stops here rather than being handed to faster-whisper.
    Its own filter concatenates the speech it keeps and maps the timestamps
    back afterwards, so Whisper decodes windows stitched together from moments
    minutes apart - which is what sends it into repeat loops and lets one cue
    run on until the next burst of speech. Here the detector only says where to
    point the model: the audio it hears is real and contiguous, and it picks up
    again at the next speech by itself.
    """
    if not use_vad:
        return None

    duration = len(audio) / AUDIO_SAMPLE_RATE
    regions = speech_regions(audio)
    heard = sum(region["end"] - region["start"] for region in regions)
    if heard < duration * VAD_MIN_KEEP_RATIO:
        logging.warning(
            "%s: the silence filter heard almost no speech, which means the "
            "audio is mixed too quietly for the detector rather than that the "
            "file is silent. Transcribing all of it instead.", name,
        )
        return None

    widened = widen_regions(regions, duration)
    covered = sum(region["end"] - region["start"] for region in widened)
    skipped = duration - covered
    if not widened or skipped <= 0:
        return None

    # One silence between each pair of regions, plus any at either end.
    stretches = (
        len(widened) - 1
        + (widened[0]["start"] > 0)
        + (widened[-1]["end"] < duration)
    )
    level = (
        logging.WARNING
        if skipped > duration * VAD_REMOVAL_WARN_RATIO
        else logging.INFO
    )
    logging.log(
        level,
        "%s: skipping %s of %s, in %d stretch(es) with no speech in them. "
        'Untick "Skip long silences" if those stretches should have had '
        "subtitles.", name,
        timedelta(seconds=int(skipped)),
        timedelta(seconds=int(duration)),
        stretches,
    )
    return widened


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


def is_stretched(content, seconds):
    """Is this too few characters to have taken that long to say?

    Whisper closes a segment at the next speech boundary, so the span of a cue
    is really the span of the silence after it, and a line invented to fill a
    window carries the whole window's worth. Real speech keeps up a rate even
    when the cue runs long; an invented one cannot.
    """
    if seconds < SUSPECT_MIN_SECONDS:
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
    """
    if seconds < SUSPECT_MIN_SECONDS:
        return False
    return repetition(content) >= LOOP_MIN_REPETITION


def build_subtitles(segments):
    """Convert Whisper segments to srt.Subtitle, dropping empty ones."""
    subtitles = []
    stretched = looping = 0
    for segment in segments:
        content = segment.text.strip()
        if not content:
            continue

        start = timedelta(seconds=segment.start)
        end = timedelta(seconds=segment.end)
        # Both tests read the span, so both come before the clamp below, which
        # would destroy the evidence.
        span = (end - start).total_seconds()
        if is_stretched(content, span):
            logging.debug("Dropping %.0fs of silence written up as %r", span, content)
            stretched += 1
            continue
        if is_looping(content, span):
            logging.debug("Dropping %.0fs of %r on a loop", span, content)
            looping += 1
            continue

        # A short line isolated in a long silence can still stretch for minutes.
        # Clamping only ever shortens a cue, so it cannot introduce overlaps.
        if span > MAX_SUBTITLE_DURATION:
            end = start + timedelta(seconds=MAX_SUBTITLE_DURATION)

        subtitles.append(
            srt.Subtitle(index=0, start=start, end=end, content=content)
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
    subtitles = drop_repeat_runs(subtitles)

    # Whisper sometimes emits segments that overlap by a fraction of a second.
    # Trim the earlier cue so a player never shows two lines at once. The
    # start comparison keeps a badly ordered pair from getting a negative
    # duration.
    for current, following in zip(subtitles, subtitles[1:]):
        if current.start < following.start < current.end:
            current.end = following.start

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
        self._on_language_change()
        self._apply_log_visibility(bool(self.config.get("log_visible", True)))
        self._attach_log_handler()
        self._drain_job = self.root.after(UI_POLL_INTERVAL_MS, self._drain_queues)

    # ------------------------------------------------------------------ UI

    def _build_widgets(self):
        pad = {"padx": 16, "fill": "x"}

        settings = ttk.LabelFrame(self.root, text="Model")
        settings.pack(pady=(12, 6), **pad)

        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        model_combo = self._labelled_combo(settings, "Model:", self.model_var, MODELS)
        model_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_model_note())

        self.model_note = ttk.Label(settings, foreground="gray", wraplength=520)
        self.model_note.pack(anchor="w", padx=16, pady=(0, 6))

        self.device_var = tk.StringVar(value=DEFAULT_DEVICE)
        self._labelled_combo(settings, "Device:", self.device_var, tuple(DEVICES))

        self.precision_var = tk.StringVar(value="Auto")
        self._labelled_combo(settings, "Precision:", self.precision_var, PRECISIONS)

        language = ttk.LabelFrame(self.root, text="Language")
        language.pack(pady=6, **pad)

        self.language_var = tk.StringVar(value=AUTO_DETECT)
        language_combo = self._labelled_combo(
            language, "Spoken language:", self.language_var, tuple(LANGUAGES)
        )
        language_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_language_change()
        )

        self.translate_var = tk.BooleanVar(value=True)
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

        # Off by default: skipping a stretch is still a judgement call made
        # by a detector that cannot hear everything - see the note below.
        self.vad_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options,
            text="Skip long silences (faster on sparse audio)",
            variable=self.vad_var,
        ).pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Label(
            options,
            text=f"Stretches with no speech for over {SILENCE_GAP_SECONDS:.0f} "
                 "seconds are passed over; everything else is transcribed in "
                 "place, pauses and all. Quiet speech under music can still be "
                 "mistaken for silence and skipped.",
            foreground="gray",
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=28, pady=(0, 4))

        self.batched_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options,
            text=f"Batched inference (much faster, batch size {BATCH_SIZE})",
            variable=self.batched_var,
        ).pack(anchor="w", padx=8)

        self.skip_existing_var = tk.BooleanVar(value=True)
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

    def _labelled_combo(self, parent, label, variable, values):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=16).pack(side="left")
        combo = ttk.Combobox(row, textvariable=variable, state="readonly")
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

    def _on_language_change(self):
        """Translating English into English is a no-op, so tie the two together.

        Any other source language (including auto-detect) defaults to
        translating, which is the common case for foreign-language media.
        """
        is_english = LANGUAGES[self.language_var.get()] == "en"
        self.translate_var.set(not is_english)
        self.translate_check.config(state="disabled" if is_english else "normal")
        self._update_model_note()

    def _update_model_note(self):
        capability = model_capability(self.model_var.get())
        conflict = self.translate_var.get() and capability != "full"
        self.model_note.config(
            text=("! " if conflict else "") + MODEL_NOTES[capability],
            foreground="#b26a00" if conflict else "gray",
        )

    def _confirm_model_choice(self):
        """Warn before a model/translate combination that silently misbehaves."""
        model_size = self.model_var.get()
        capability = model_capability(model_size)
        if not self.translate_var.get() or capability == "full":
            return True

        if capability == "english_only":
            detail = (
                f"'{model_size}' was trained on English audio only. On any other "
                "language it produces garbage."
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
            model_size=self.model_var.get(),
            device_label=self.device_var.get(),
            precision_label=self.precision_var.get(),
            language=LANGUAGES[self.language_var.get()],
            translate=self.translate_var.get(),
            vad_filter=self.vad_var.get(),
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

        # Both pipelines take the windows planned here instead of running
        # their own silence filter, which would splice the surviving speech
        # together before Whisper ever saw it.
        if settings.batched:
            pipeline = BatchedInferencePipeline(model=model)

            def batched(audio, name):
                duration = len(audio) / AUDIO_SAMPLE_RATE
                regions = plan_regions(audio, name, settings.vad_filter)
                if regions is None:
                    regions = [{"start": 0.0, "end": duration}]

                return pipeline.transcribe(
                    audio,
                    language=settings.language,
                    task=task,
                    clip_timestamps=split_clips(regions),
                    batch_size=BATCH_SIZE,
                    # Batched mode defaults to without_timestamps=True, which
                    # makes segment bounds come from merged chunks instead of
                    # from Whisper. On sparse speech that yields single
                    # subtitles many minutes long. Asking for timestamps costs
                    # ~25% speed and brings segment length back in line with
                    # sequential mode.
                    without_timestamps=False,
                )

            return batched

        def sequential(audio, name):
            regions = plan_regions(audio, name, settings.vad_filter)
            return model.transcribe(
                audio,
                language=settings.language,
                task=task,
                # Flattened into start, end, start, end... in seconds. Regions
                # are left whole here rather than cut into windows: within one
                # the model advances to wherever the last segment ended, so it
                # stops on a sentence instead of every thirty seconds. "0" is
                # faster-whisper's own way of saying the entire file.
                clip_timestamps=(
                    [
                        bound
                        for region in regions
                        for bound in (region["start"], region["end"])
                    ]
                    if regions
                    else "0"
                ),
                condition_on_previous_text=False,
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
        logging.info(
            "Detected language for %s: %s (%.0f%% confidence)",
            name, info.language, (info.language_probability or 0) * 100,
        )

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
