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
            # a tqdm bar redraws in place: replace its last line instead of adding one
            for i in range(len(self._buffer) - 1, -1, -1):
                entry = self._buffer[i]
                if entry[3] == full_key:
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


def _get_active_progress_items(parent) -> list[dict]:
    """Collect the viewer's running Save As, z-stats and z-registration jobs.

    Returns one ``{key, text, progress, done}`` dict per job.
    """
    items = []

    saveas_progress = parent._saveas_progress
    if parent._saveas_running or (0.0 < saveas_progress < 1.0):
        text = (
            "Starting save..."
            if saveas_progress == 0.0
            else f"Saving z-plane {parent._saveas_current_index}"
        )
        items.append(
            {
                "key": "saveas",
                "text": text,
                "progress": max(0.01, saveas_progress),
                "done": False,
            }
        )
    elif parent._saveas_done:
        items.append(
            {
                "key": "saveas",
                "text": "Save complete",
                "progress": 1.0,
                "done": True,
            }
        )

    for i, (running, progress) in enumerate(
        zip(parent._zstats_running, parent._zstats_progress)
    ):
        if running or (0.0 < progress < 1.0):
            # the overall fraction: a per-plane index restarts at each
            # channel or camera and the bar would jump back
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

    register_progress = parent._register_z_progress
    register_msg = parent._register_z_current_msg
    if parent._register_z_running or (0.0 < register_progress < 1.0):
        msg = register_msg if register_msg else "Starting..."
        items.append(
            {
                "key": "register_z",
                "text": f"Z-Reg: {msg}",
                "progress": max(0.01, register_progress),
                "done": False,
            }
        )
    elif parent._register_z_done and register_msg:
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
