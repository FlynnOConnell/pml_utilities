"""
worker entry point for background subprocess tasks.

this module is invoked via:
    python -m mbo_utilities.gui._worker <task_type> <args_json>

it runs independently of the gui and can survive gui closure.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from mbo_utilities.gui.tasks import TASKS

MAX_STALL_MINUTES = (
    120  # no progress change and no log write for this long kills the worker
)

_JOB_HANDLE = None  # NOTE: the Windows Job Object handle lives for the whole worker: closing it kills every contained child


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Setup logging for worker process.

    routes through mbo_utilities.log so the package's single non-propagating
    'mbo' handler is the only stdout sink — prevents the duplicate-line
    pattern caused by adding a second handler on a propagating child logger.
    """
    from mbo_utilities import log

    logger = log.get("worker")
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for h in logger.handlers + logging.getLogger("mbo").handlers:
        h.setFormatter(fmt)

    if log_file:
        try:
            from logging.handlers import RotatingFileHandler

            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(  # caps a chatty run's log
                log_path, mode="a", maxBytes=5_000_000, backupCount=2, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Failed to setup file logging: {e}", file=sys.stderr)

    return logger


def _update_status(
    pid: int,
    status: str,
    message: str | None = None,
    details: str | dict | None = None,
    uuid: str | None = None,
):
    """Ensure process status is reported to sidecar file."""
    try:
        from mbo_utilities.preferences import get_mbo_dirs

        log_dir = get_mbo_dirs()["logs"]
        if uuid:
            sidecar = log_dir / f"progress_{uuid}.json"
        else:
            sidecar = log_dir / f"progress_{pid}.json"

        current_data = {}
        if sidecar.exists():
            try:
                with sidecar.open() as f:
                    current_data = json.load(f)
            except Exception:
                pass

        if status == "completed" and current_data.get("status") == "completed":
            return

        data = current_data.copy()
        data["pid"] = pid
        data["uuid"] = uuid
        data["timestamp"] = time.time()
        data["status"] = status

        if message:
            data["message"] = message
        if details:
            data["details"] = details

        if status == "completed":
            data["progress"] = 1.0

        tmp_file = sidecar.with_suffix(".tmp")
        with tmp_file.open("w") as f:
            json.dump(data, f)
        _atomic_replace(tmp_file, sidecar)

    except Exception as e:
        print(f"Failed to update status sidecar: {e}", file=sys.stderr)


def _atomic_replace(src: Path, dst: Path, attempts: int = 10, delay: float = 0.05):
    """Path.replace with retry for windows sharing-violation races.

    the gui opens the sidecar for read with default share flags, which on
    windows denies rename until the handle closes. a short retry loop
    clears the contention without blocking the worker meaningfully.
    """
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
    if last_err is not None:
        raise last_err


def _start_watchdog(
    uuid: str | None, logger: logging.Logger, log_file: str | None = None
):
    """Daemon thread that kills this process if progress stalls."""
    threading.Thread(
        target=_watchdog, args=(uuid, logger, log_file), daemon=True
    ).start()


def _watchdog(uuid, logger, log_file) -> None:
    from mbo_utilities.preferences import get_mbo_dirs

    log_dir = get_mbo_dirs()["logs"]
    last_progress = 0.0
    last_log_mtime = 0.0
    last_change = time.time()
    stall_seconds = MAX_STALL_MINUTES * 60

    while True:
        time.sleep(60)
        try:
            if uuid:
                sidecar = log_dir / f"progress_{uuid}.json"
            else:
                sidecar = log_dir / f"progress_{os.getpid()}.json"

            if sidecar.exists():
                with sidecar.open() as f:
                    data = json.load(f)
                progress = data.get("progress", 0.0)
                status = data.get("status", "running")

                if status in ("completed", "error"):
                    return

                if progress != last_progress:
                    last_progress = progress
                    last_change = time.time()
        except Exception:  # a half-written sidecar is read again next minute
            pass

        # NOTE: a blocking task logs without moving progress, so only a worker that is both progress-frozen and log-silent counts as hung
        try:
            if log_file:
                m = Path(log_file).stat().st_mtime
                if m > last_log_mtime:
                    last_log_mtime = m
                    last_change = time.time()
        except OSError:
            pass

        if time.time() - last_change > stall_seconds:
            msg = f"watchdog: no progress for {MAX_STALL_MINUTES} min, terminating"
            logger.error(msg)
            _update_status(os.getpid(), "error", message=msg, uuid=uuid)
            os._exit(1)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _resolve_mem_settings() -> tuple[float, float, float]:
    """(tick_s, log_s, warn_pct) for this worker's memory monitor.

    Tick 0 means off. Env vars win so an HPC submission can dial the rates
    without touching the gui preferences:

      MBO_MEM_LOG_INTERVAL  sample (tick) interval, seconds
      MBO_MEM_LOG_EVERY     seconds between log lines (0 = every tick)
      MBO_MEM_WARN_PCT      warn at/above this system-memory percent (0 = off)
    """
    tick = _env_float("MBO_MEM_LOG_INTERVAL")
    log_s = _env_float("MBO_MEM_LOG_EVERY")
    warn = _env_float("MBO_MEM_WARN_PCT")

    if tick is None or log_s is None or warn is None:
        try:
            from mbo_utilities.preferences import (
                get_mem_log_interval,
                get_mem_monitor,
                get_mem_monitor_interval,
                get_mem_warn_pct,
            )

            if tick is None:
                tick = float(get_mem_monitor_interval()) if get_mem_monitor() else 0.0
            if log_s is None:
                log_s = float(get_mem_log_interval())
            if warn is None:
                warn = float(get_mem_warn_pct())
        except Exception:
            pass

    return (tick or 0.0), (log_s if log_s is not None else 0.0), (warn or 0.0)


def _start_mem_monitor(uuid: str | None, logger: logging.Logger):
    """Sample system + process-tree memory to logs/mem_<uuid>.csv.

    Every tick is a csv row (flushed, so the last sample survives a hard OOM
    kill); log lines are throttled to their own rate, with new pipeline-memory
    peaks and high-memory warnings still getting through immediately.
    """
    tick, log_s, warn_pct = _resolve_mem_settings()
    if tick <= 0:
        return None

    from mbo_utilities._sysmem import MemoryMonitor
    from mbo_utilities.preferences import get_mbo_dirs

    log_dir = get_mbo_dirs()["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    return MemoryMonitor(
        tick_s=tick,
        log_s=log_s,
        warn_pct=warn_pct,
        csv_path=log_dir / f"mem_{uuid or os.getpid()}.csv",
        logger=logger,
    ).start()


def _contain_in_job():
    """Windows: assign this worker to a Job Object that terminates all
    members when the job handle closes.

    Processes a task spawns (ProcessPoolExecutor / multiprocessing Manager /
    suite2p plane workers) inherit the job, so they are reaped when the worker
    exits for any reason — clean exit, crash, taskkill, or the watchdog's
    os._exit. The job is created and held by the worker (not the gui), so it
    does not affect the worker's own survival of gui closure.

    Returns the job handle (caller must keep it alive) or None when
    unavailable: non-Windows, or the process is already in a job that forbids
    nesting (pre-Windows 8). Best-effort — never raises.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IO(ctypes.Structure):
            _fields_ = [
                (n, ctypes.c_ulonglong)
                for n in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class _EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            k32.CloseHandle(job)
            return None
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            k32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


class _CRCollapseWriter(io.TextIOBase):
    r"""Collapse carriage-return progress redraws to one log line per bar.

    tqdm (masknmf, imwrite) and pytorch-lightning ignore TQDM_DISABLE and
    emit every refresh; with stdio redirected to the log file each ``\\r``
    redraw lands as its own line, flooding the log. Only the text after
    the last ``\\r`` of each completed line survives, so a bar logs once,
    at its final state.
    """

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "\r" in line:
                segs = [s for s in line.split("\r") if s.strip()]
                line = segs[-1] if segs else ""
            if line.strip():
                out.append(line)
        if out:
            self._stream.write("\n".join(out) + "\n")
            self._stream.flush()
        if (
            len(self._buf) > 65536 and "\r" in self._buf
        ):  # a bar that redraws forever without a newline
            self._buf = self._buf.rsplit("\r", 1)[-1]
        return len(text)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


def main():
    """Main entry point for worker subprocess."""
    # NOTE: line buffering so prints reach the log as they happen; utf-8 because redirected stdio on Windows is cp1252, which the lsp arrows (U+2192) crash
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

    os.environ["TQDM_DISABLE"] = (
        "1"  # no terminal, so no \r redraws; libraries that ignore it are squashed by _CRCollapseWriter
    )

    if sys.stdout is not None:
        sys.stdout = _CRCollapseWriter(sys.stdout)
    if sys.stderr is not None:
        sys.stderr = _CRCollapseWriter(sys.stderr)

    print(f"Worker starting (pid={os.getpid()})", file=sys.stderr, flush=True)

    if len(sys.argv) < 3:
        print(
            "Usage: python -m mbo_utilities.gui._worker <task_type> <args_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    task_type = sys.argv[1]
    args_source = sys.argv[2]

    args_file = None
    try:
        candidate = Path(
            args_source
        )  # a file (the usual: Windows' command-line limit) or inline JSON
        if candidate.is_file() and candidate.suffix == ".json":
            args_file = candidate
            with args_file.open(encoding="utf-8") as f:
                args = json.load(f)
        else:
            args = json.loads(args_source)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to load args: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if args_file is not None:  # uuid-named and read by nobody else
            try:
                args_file.unlink(missing_ok=True)
            except OSError:
                pass

    log_file = args.get("_log_file")
    uuid = args.get("_uuid")
    logger = setup_logging(log_file)

    logger.info(f"Worker started: task={task_type}, pid={os.getpid()}")

    global _JOB_HANDLE
    _JOB_HANDLE = (
        _contain_in_job()
    )  # task-spawned pools and plane workers die with this process, os._exit included
    if sys.platform == "win32":
        logger.debug(f"job containment: {'on' if _JOB_HANDLE else 'unavailable'}")

    _start_watchdog(uuid, logger, log_file)

    _start_mem_monitor(uuid, logger)

    if (
        task_type not in TASKS
    ):  # a plugin pipeline's task arrives through its entry point
        try:
            from mbo_utilities.gui.tasks import load_plugin_tasks

            load_plugin_tasks()
        except Exception:
            logger.exception("Could not load plugin tasks")

    if task_type not in TASKS:
        logger.error(f"Unknown task type: {task_type}")
        print(f"Unknown task type: {task_type}", file=sys.stderr)
        _update_status(
            os.getpid(), "error", f"Unknown task type: {task_type}", uuid=uuid
        )
        sys.exit(1)

    task_func = TASKS[task_type]

    try:
        task_func(args, logger)
        logger.info("Task completed successfully")
        _update_status(os.getpid(), "completed", uuid=uuid)
        sys.exit(0)
    except Exception as e:
        try:
            from mbo_utilities._sysmem import format_mem_line, mem_snapshot

            logger.error("memory at failure: " + format_mem_line(mem_snapshot()))
        except Exception:
            pass
        logger.exception(f"Task failed: {e}")
        _update_status(
            os.getpid(),
            "error",
            message=str(e),
            details=traceback.format_exc(),
            uuid=uuid,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
