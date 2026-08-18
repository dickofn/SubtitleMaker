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
from logging.handlers import RotatingFileHandler
from tkinter import filedialog, font as tkfont, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import av
import ctranslate2
import numpy as np
import srt
from faster_whisper import BatchedInferencePipeline, WhisperModel

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

    Tk delegates glob matching to the native dialog: Windows matches
    case-insensitively and reads "*.*" as "everything", while X11 matches
    case-sensitively and only matches "*.*" against names containing a dot.
    So on Linux a plain "*.mp4" hides MOVIE.MP4, and "*.*" hides
    extensionless files. Character classes fix the first, "*" the second.
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

WINDOW_WIDTH = 620
WINDOW_HEIGHT = 800
MIN_WIDTH = 560


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

    audio = np.frombuffer(buffer.getbuffer(), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0, skipped, container_duration


def build_subtitles(segments):
    """Convert Whisper segments to srt.Subtitle, dropping empty ones."""
    subtitles = []
    for segment in segments:
        content = segment.text.strip()
        if not content:
            continue

        start = timedelta(seconds=segment.start)
        end = timedelta(seconds=segment.end)
        # Whisper closes a segment at the next speech boundary, so a short line
        # isolated in a long silence can stretch for many minutes. Clamping only
        # ever shortens a cue, so it cannot introduce overlaps.
        if (end - start).total_seconds() > MAX_SUBTITLE_DURATION:
            end = start + timedelta(seconds=MAX_SUBTITLE_DURATION)

        subtitles.append(
            srt.Subtitle(
                index=len(subtitles) + 1,
                start=start,
                end=end,
                content=content,
            )
        )

    # Whisper sometimes emits segments that overlap by a fraction of a second.
    # Trim the earlier cue so a player never shows two lines at once. The
    # start comparison keeps a badly ordered pair from getting a negative
    # duration.
    for current, following in zip(subtitles, subtitles[1:]):
        if current.start < following.start < current.end:
            current.end = following.start

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

        self.vad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options,
            text="Filter silence (recommended - avoids hallucinated repeats)",
            variable=self.vad_var,
        ).pack(anchor="w", padx=8, pady=(6, 0))

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

        self.status_label = ttk.Label(
            self.root, text="Idle", foreground="gray",
            wraplength=560, justify="left",
        )
        self.status_label.pack(padx=16, fill="x")

        self.log_frame = ttk.LabelFrame(self.root, text="Log")

        # TkFixedFont resolves to whatever monospace font the platform
        # actually has; naming Consolas outright silently falls back to a
        # proportional face anywhere it is not installed, Linux included.
        log_font = tkfont.nametofont("TkFixedFont").copy()
        log_font.configure(size=9)

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

                self.post(self.set_status, f"[{index + 1}/{total}] {name}")
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

        if settings.batched:
            pipeline = BatchedInferencePipeline(model=model)
            return lambda audio: pipeline.transcribe(
                audio,
                language=settings.language,
                task=task,
                vad_filter=settings.vad_filter,
                batch_size=BATCH_SIZE,
                # Batched mode defaults to without_timestamps=True, which makes
                # segment bounds come from merged VAD chunks instead of from
                # Whisper. On sparse speech that yields single subtitles many
                # minutes long. Asking for timestamps costs ~25% speed and
                # brings segment length back in line with sequential mode.
                without_timestamps=False,
            )

        return lambda audio: model.transcribe(
            audio,
            language=settings.language,
            task=task,
            vad_filter=settings.vad_filter,
            condition_on_previous_text=False,
        )

    def _transcribe_one(self, transcribe, path, srt_path, index, total):
        name = os.path.basename(path)
        logging.info("Processing file %d/%d: %s", index + 1, total, name)

        self.post(self.set_status, f"[{index + 1}/{total}] Decoding audio: {name}")
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

        segment_iter, info = transcribe(audio)
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
            self._report_progress(name, segment.end, duration, index, total)

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

    def _report_progress(self, name, position, duration, index, total):
        now = time.monotonic()
        if now - self._last_ui_update < UI_REFRESH_INTERVAL:
            return
        self._last_ui_update = now

        if duration > 0:
            ratio = min(position / duration, 1.0)
            self.post(self.set_progress, ((index + ratio) / total) * 100)
            self.post(
                self.set_status,
                f"[{index + 1}/{total}] {name} - {int(ratio * 100)}%",
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
