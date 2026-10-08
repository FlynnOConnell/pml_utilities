"""Windows that run in a process of their own, so they never share the Studio's imgui context."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
