import re
import sys
import time
from collections import defaultdict, deque


class OutputCapture:
    """Capture stdout/stderr to a buffer while still printing to console."""

    _instance = None
    _max_lines = 200

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lines = deque(maxlen=cls._max_lines)
            cls._instance._original_stdout = None
            cls._instance._original_stderr = None
            cls._instance._capturing = False
        return cls._instance

    def start(self):
        """Start capturing stdout/stderr."""
        if self._capturing:
            return
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._original_stdout, self._lines, "stdout")
        sys.stderr = _TeeWriter(self._original_stderr, self._lines, "stderr")
        self._capturing = True


class _TeeWriter:
    """Write to both the original stream and a capture buffer."""

    _ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")
    _BOX_REPLACEMENTS = {
        "█": "#",
        "▏": "|",
        "▎": "|",
        "▍": "|",
        "▌": "|",
        "▋": "|",
        "▊": "|",
        "▉": "|",
        "░": "-",
        "▒": "=",
        "▓": "#",
        "━": "-",
        "┃": "|",
        "╸": ">",
        "╺": "<",
    }
    _TQDM_PATTERNS = [
        re.compile(r"\d+%\|"),
        re.compile(r"it/s"),
        re.compile(r"[0-9]+/[0-9]+\s*\["),
    ]

    def __init__(self, original, buffer: deque, stream_name: str):
        self._original = original
        self._buffer = buffer
        self._stream_name = stream_name

    def _clean(self, text: str) -> str | None:
        text = self._ANSI_ESCAPE.sub("", text).replace("\r", "")
        if not text.strip():
            return None
        for old, new in self._BOX_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text.strip()

    def _tqdm_key(self, text: str) -> str | None:
        for pattern in self._TQDM_PATTERNS:
            if pattern.search(text):
                match = re.match(r"^([^:]+):", text)
                return f"tqdm_{match.group(1).strip() if match else 'tqdm'}"
        return None

    def write(self, text):
        if self._original:
            self._original.write(text)

        if not text:
            return

        cleaned = self._clean(text)
        if cleaned is None:
            return

        timestamp = time.strftime("%H:%M:%S")
        full_key = self._tqdm_key(cleaned)

        if full_key:
            # update last entry with same key in place to avoid log spam
            for i in range(len(self._buffer) - 1, -1, -1):
                entry = self._buffer[i]
                if len(entry) >= 4 and entry[3] == full_key:
                    self._buffer[i] = (timestamp, self._stream_name, cleaned, full_key)
                    return
            self._buffer.append((timestamp, self._stream_name, cleaned, full_key))
        else:
            self._buffer.append((timestamp, self._stream_name, cleaned, ""))

    def flush(self):
        if self._original:
            self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


_output_capture = OutputCapture()

_progress_state = defaultdict(
    lambda: {
        "hide_time": None,
        "is_showing_done": False,
        "done_shown_once": False,
        "done_cleared": False,
    }
)


def reset_progress_state(key: str):
    """Reset progress state for a given key to allow re-display."""
    if key in _progress_state:
        _progress_state[key] = {
            "hide_time": None,
            "is_showing_done": False,
            "done_shown_once": False,
            "done_cleared": False,
        }


def _get_active_progress_items(self) -> list[dict]:
    """Collect active progress operations from the widget state.

    Returns list of dicts with: key, text, progress, done.
    """
    items = []

    save_as = getattr(self, "save_as", None)
    saveas_running = save_as.running if save_as is not None else False
    saveas_progress = save_as.progress if save_as is not None else 0.0
    saveas_current = save_as.current_index if save_as is not None else 0
    saveas_done = save_as.done if save_as is not None else False

    if saveas_running or (0.0 < saveas_progress < 1.0):
        text = (
            "Starting save..."
            if saveas_progress == 0.0
            else f"Saving z-plane {saveas_current}"
        )
        items.append(
            {
                "key": "saveas",
                "text": text,
                "progress": max(0.01, saveas_progress),
                "done": False,
            }
        )
    elif saveas_done:
        items.append(
            {
                "key": "saveas",
                "text": "Save complete",
                "progress": 1.0,
                "done": True,
            }
        )

    num_graphics = getattr(self, "num_graphics", 1)
    zstats = getattr(self, "zstats", None)
    zstats_running = zstats.running if zstats is not None else []
    zstats_progress = zstats.progress if zstats is not None else []

    for i in range(num_graphics):
        running = (
            zstats_running[i]
            if isinstance(zstats_running, list) and i < len(zstats_running)
            else False
        )
        progress = (
            zstats_progress[i]
            if isinstance(zstats_progress, list) and i < len(zstats_progress)
            else 0.0
        )

        if running or (0.0 < progress < 1.0):
            # report the monotonic overall fraction, not the per-plane
            # index — the latter resets each channel/camera and reads as
            # the bar jumping backward.
            text = (
                f"Z-stats {i + 1}: starting..."
                if progress == 0.0
                else f"Z-stats: computing {progress * 100:.0f}%"
            )
            items.append(
                {
                    "key": f"zstats_{i}",
                    "text": text,
                    "progress": max(0.01, progress),
                    "done": False,
                }
            )

    register_running = save_as.register_running if save_as is not None else False
    register_progress = save_as.register_progress if save_as is not None else 0.0
    register_msg = save_as.register_msg if save_as is not None else None
    register_done = save_as.register_done if save_as is not None else False

    if register_running or (0.0 < register_progress < 1.0):
        msg = register_msg if register_msg else "Starting..."
        items.append(
            {
                "key": "register_z",
                "text": f"Z-Reg: {msg}",
                "progress": max(0.01, register_progress),
                "done": False,
            }
        )
    elif register_done and register_msg:
        items.append(
            {
                "key": "register_z",
                "text": f"Z-Reg: {register_msg}",
                "progress": 1.0,
                "done": True,
            }
        )

    return items


def start_output_capture():
    """Start capturing stdout/stderr."""
    _output_capture.start()
