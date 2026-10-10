"""Windows that run in a process of their own, so they never share the Studio's imgui context."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

from mbo_utilities.preferences import get_mbo_dirs


def launch_window(module: str, args: list[str], log_name: str) -> int:
    """Run ``python -m <module> <args>`` detached and return its pid.

    The window outlives the viewer that opened it; its output goes to
    ``~/.mbo/logs/<stamp>_<log_name>.log``.
    """
    python = sys.executable
    if sys.platform == "win32" and python.endswith("python.exe"):
        # no console window beside the new one
        pythonw = python[:-10] + "pythonw.exe"
        if Path(pythonw).exists():
            python = pythonw
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = get_mbo_dirs()["logs"] / f"{stamp}_{log_name}.log"
    cmd = [python, "-m", module, *args]
    with log_file.open("a", encoding="utf-8") as out:
        if sys.platform == "win32":
            proc = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=out,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=out,
            )
    return proc.pid


class LaunchedWindow:
    """A window :func:`launch_window` started, watched from a draw loop.

    ``what`` names it in the message a crash leaves (``QC viewer``).
    """

    def __init__(self, pid: int, log_name: str, what: str):
        self.pid, self.log_name, self.what = pid, log_name, what
        self._checked = 0.0

    def poll(self) -> str | None:
        """None while the window runs; once it has exited, ``""`` or, when it
        died on a traceback, its last line and log. Looks at most once a second.
        """
        if time.monotonic() - self._checked < 1.0:
            return None
        self._checked = time.monotonic()
        try:
            if psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            pass
        logs = sorted(get_mbo_dirs()["logs"].glob(f"*_{self.log_name}.log"))
        text = logs[-1].read_text(encoding="utf-8", errors="replace") if logs else ""
        if "Traceback" not in text:
            return ""
        last = [line for line in text.splitlines() if line.strip()][-1]
        return f"{self.what} failed: {last} (log: {logs[-1]})"
