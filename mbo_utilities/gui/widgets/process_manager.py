"""
process manager for background operations.

tracks subprocesses that can survive gui closure. stores process info
to disk so orphaned processes can be detected on next gui launch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from mbo_utilities import log
from mbo_utilities.preferences import get_mbo_dirs

logger = log.get("gui.process_manager")


COMPLETED_RETENTION_SECONDS = 300  # how long a completed row stays in the console


@dataclass
class ProcessInfo:
    pid: int
    description: str
    start_time: float
    task_type: str
    output_path: str | None = None
    args: dict | None = None
    status: str = "running"
    progress: float = 0.0
    status_message: str = ""
    error_details: dict | str | None = None
    completed_time: float | None = None

    def elapsed_seconds(self) -> float:
        """Seconds since process started, frozen once it reaches a terminal state."""
        end = self.completed_time if self.completed_time is not None else time.time()
        return end - self.start_time

    def elapsed_str(self) -> str:
        """human-readable elapsed time."""
        elapsed = self.elapsed_seconds()
        if elapsed < 60:
            return f"{int(elapsed)} seconds ago"
        if elapsed < 3600:
            return f"{int(elapsed / 60)} minutes ago"
        return f"{elapsed / 3600:.1f} hours ago"

    def update_from_sidecar(self, verbose=False):
        """Read progress_{uuid}.json (preferred) or progress_{pid}.json."""
        log_dir = get_mbo_dirs()["logs"]

        uuid = self.args.get("_uuid") if self.args else None

        sidecar = None
        if uuid:
            candidate = log_dir / f"progress_{uuid}.json"
            if candidate.exists():
                sidecar = candidate

        if not sidecar:
            candidate = log_dir / f"progress_{self.pid}.json"
            if candidate.exists():
                sidecar = candidate

        if verbose:
            logger.debug(f"Checking sidecar for PID {self.pid} (UUID={uuid})")
            if uuid:
                logger.debug(
                    f"  UUID path: {log_dir / f'progress_{uuid}.json'} (Exists: {(log_dir / f'progress_{uuid}.json').exists()})"
                )
            logger.debug(
                f"  PID path:  {log_dir / f'progress_{self.pid}.json'} (Exists: {(log_dir / f'progress_{self.pid}.json').exists()})"
            )

        if sidecar and sidecar.exists():
            try:
                with sidecar.open() as f:
                    data = json.load(f)

                    valid = False
                    if (uuid and data.get("uuid") == uuid) or data.get(
                        "pid"
                    ) == self.pid:
                        valid = True

                    if valid:
                        self.status = data.get("status", self.status)
                        self.progress = data.get("progress", self.progress)
                        self.status_message = data.get("message", self.status_message)
                        self.error_details = data.get("details", self.error_details)
                        if (
                            self.status in ("completed", "error")
                            and self.completed_time is None
                        ):
                            self.completed_time = data.get("timestamp", time.time())
                    else:
                        logger.debug(
                            f"Identity mismatch in sidecar {sidecar}. Found pid={data.get('pid')}, uuid={data.get('uuid')}"
                        )
            except Exception as e:
                logger.debug(f"Failed to read sidecar for pid {self.pid}: {e}")
        elif verbose:
            logger.debug(f"No sidecar found for PID {self.pid}.")

    def is_alive(self) -> bool:
        """Check if process is still running."""
        try:
            if sys.platform == "win32":
                import ctypes

                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = k32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, self.pid
                )
                if handle:
                    k32.CloseHandle(handle)
                    return True
                err = ctypes.get_last_error()
                if (
                    err == 5
                ):  # ERROR_ACCESS_DENIED: the process exists and cannot be queried
                    return True
                if err != 87:  # ERROR_INVALID_PARAMETER: no such pid
                    logger.debug(
                        f"PID {self.pid} is_alive check failed. OpenProcess err={err}"
                    )
                return False
            os.kill(self.pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def kill(self) -> bool:
        """Kill the process and its descendants (task-spawned pool / plane
        workers, multiprocessing Manager).

        Terminate gracefully first, then force-kill survivors. Children are
        signalled before the parent so they are not reparented mid-kill.
        """
        try:
            parent = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return True
        try:
            procs = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            procs = []
        procs.append(parent)
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(procs, timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        return True

    def tail_log(self, n: int = 50) -> list[str]:
        """Read the last n lines from the log file."""
        if not self.output_path:
            return []
        p = Path(self.output_path)
        if not p.is_file():
            return []
        with p.open(encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]


@dataclass
class LocalJob:
    """An in-process background job, tracked beside spawned subprocesses.

    Work that runs on a thread inside the GUI — extracting a trace from an
    ROI, say — never gets a pid, so it cannot go through
    :meth:`ProcessManager.spawn`. It registers here instead and shows up in
    the same status button and process console, which is the only way the
    user can tell a click did anything.

    The worker thread owns this object: call :meth:`set_progress` as it
    goes, then exactly one of :meth:`done` or :meth:`fail`.
    """

    job_id: int
    task_type: str
    description: str
    start_time: float
    status: str = "running"
    progress: float = 0.0
    status_message: str = ""
    error_details: str | None = None
    completed_time: float | None = None

    def elapsed_seconds(self) -> float:
        end = self.completed_time if self.completed_time is not None else time.time()
        return max(0.0, end - self.start_time)

    def elapsed_str(self) -> str:
        seconds = int(self.elapsed_seconds())
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"

    def is_alive(self) -> bool:
        return self.status == "running"

    def set_progress(self, value: float, message: str = "") -> None:
        self.progress = float(min(max(value, 0.0), 1.0))
        if message:
            self.status_message = message

    def done(self, message: str = "") -> None:
        self.status = "completed"
        self.progress = 1.0
        self.status_message = message or self.status_message
        self.completed_time = time.time()

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error_details = str(error)
        self.status_message = str(error)
        self.completed_time = time.time()


class ProcessManager:
    """
    manages background subprocesses that can survive gui closure.

    spawns processes using subprocess.Popen with no connection to parent.
    tracks process info in a json file so processes can be monitored
    even after gui restart.

    also tracks in-process :class:`LocalJob` work, which lives only as long
    as the GUI and is never written to the process file.
    """

    PROCESS_FILE = get_mbo_dirs()["cache"] / "running_processes.json"

    def __init__(self):
        self._processes: dict[int, ProcessInfo] = {}
        self._jobs: dict[int, LocalJob] = {}
        self._next_job_id = 1
        self._job_lock = threading.Lock()
        self._load()
        prune_logs()

    def start_job(self, task_type: str, description: str) -> LocalJob:
        """Register an in-process job and return its handle."""
        with self._job_lock:
            job_id = self._next_job_id
            self._next_job_id += 1
            job = LocalJob(
                job_id=job_id,
                task_type=task_type,
                description=description,
                start_time=time.time(),
            )
            self._jobs[job_id] = job
        logger.debug(f"started local job {job_id}: {description}")
        return job

    def get_jobs(self) -> list[LocalJob]:
        """Every in-process job still being shown, newest last."""
        with self._job_lock:
            return sorted(self._jobs.values(), key=lambda j: j.job_id)

    def clear_job(self, job_id: int) -> None:
        with self._job_lock:
            self._jobs.pop(job_id, None)

    def _cleanup_jobs(self) -> int:
        """Drop finished jobs once they have been visible long enough."""
        now = time.time()
        with self._job_lock:
            stale = [
                job_id
                for job_id, job in self._jobs.items()
                if job.completed_time is not None
                and now - job.completed_time > COMPLETED_RETENTION_SECONDS
            ]
            for job_id in stale:
                del self._jobs[job_id]
        return len(stale)

    def _load(self) -> None:
        """Load process info from disk."""
        if not self.PROCESS_FILE.exists():
            return

        try:
            with self.PROCESS_FILE.open() as f:
                data = json.load(f)

            now = time.time()
            max_age_seconds = (
                24 * 3600
            )  # a tracked pid older than this is from another boot or long gone

            for entry in data.get("processes", []):
                info = ProcessInfo(**entry)
                age = now - info.start_time
                if age > max_age_seconds:
                    logger.debug(
                        f"Discarding old process {info.pid} (age: {age / 3600:.1f}h)"
                    )
                    continue
                self._processes[info.pid] = info

            if len(self._processes) < len(data.get("processes", [])):
                self._save()

        except Exception as e:
            logger.warning(f"Failed to load process file: {e}")

    def _save(self) -> None:
        """Save process info to disk."""
        self.PROCESS_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {"processes": [asdict(p) for p in self._processes.values()]}

        try:
            with self.PROCESS_FILE.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save process file: {e}")

    def spawn(
        self,
        task_type: str,
        args: dict[str, Any],
        description: str,
        output_path: str | None = None,
    ) -> int | None:
        """
        Spawn a subprocess for a background task.

        Parameters
        ----------
        task_type : str
            type of task: "save_zarr", "suite2p", etc.
        args : dict
            arguments to pass to the worker.
        description : str
            human-readable description for display.
        output_path : str, optional
            output path for display purposes.

        Returns
        -------
        int or None
            pid of spawned process, or None if spawn failed.
        """
        try:
            python_exe = sys.executable
            if sys.platform == "win32" and python_exe.endswith("python.exe"):
                pythonw = python_exe[:-10] + "pythonw.exe"
                if Path(pythonw).exists():
                    python_exe = pythonw

            from datetime import datetime
            from uuid import uuid4

            task_uuid = str(uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            log_dir = get_mbo_dirs()["logs"]

            log_file = log_dir / f"{timestamp}_{task_type}_{task_uuid[:8]}.log"

            args["_uuid"] = task_uuid
            args["_log_file"] = str(log_file)

            # NOTE: args go in a file named by the task uuid: a large frame selection exceeds the 32768-char Windows command line (WinError 206); the worker deletes it after loading
            args_file = log_dir / f"args_{task_uuid}.json"
            with args_file.open("w", encoding="utf-8") as af:
                json.dump(args, af)

            cmd = [
                python_exe,
                "-m",
                "mbo_utilities.gui._worker",
                task_type,
                str(args_file),
            ]

            f_out = log_file.open("a")

            child_env = os.environ.copy()
            child_env.setdefault(
                "PYTHONIOENCODING", "utf-8"
            )  # NOTE: redirected stdio on Windows defaults to cp1252, which raises on the unicode lsp prints
            child_env.setdefault("PYTHONUTF8", "1")

            if sys.platform == "win32":
                # NOTE: CREATE_NO_WINDOW hides the console and CREATE_NEW_PROCESS_GROUP detaches it from the GUI's Ctrl-C group; adding DETACHED_PROCESS pops a blank console window
                creationflags = (
                    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                )
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                proc = subprocess.Popen(
                    cmd,
                    creationflags=creationflags,
                    startupinfo=startupinfo,
                    stdin=subprocess.DEVNULL,
                    stdout=f_out,
                    stderr=f_out,
                    env=child_env,
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=f_out,
                    stderr=f_out,
                    env=child_env,
                )

            f_out.close()  # the child owns its duplicated handle

            info = ProcessInfo(
                pid=proc.pid,
                description=description,
                start_time=time.time(),
                task_type=task_type,
                output_path=str(log_file),
                args=args,
            )
            self._processes[proc.pid] = info
            self._save()

            logger.info(f"Spawned background process {proc.pid}: {description}")
            logger.debug(f"Process log: {log_file}")
            return proc.pid

        except Exception as e:
            logger.exception(f"Failed to spawn process: {e}")
            return None

    def get_all(self) -> list[ProcessInfo]:
        """Get all tracked processes."""
        return list(self._processes.values())

    def get_running(self) -> list[ProcessInfo]:
        """Get processes to display (running, completed, or errored)."""
        active = []
        for p in self._processes.values():
            p.update_from_sidecar()
            active.append(p)
        return active

    def cleanup_finished(self) -> int:
        """
        Remove entries for processes that have finished successfully and
        exceeded the retention period. failed processes are kept until
        dismissed or cleared.

        returns number of entries removed.
        """
        now = time.time()
        to_remove = []
        changed = False

        for pid, p in list(self._processes.items()):
            p.update_from_sidecar()

            if not p.is_alive():
                p.update_from_sidecar()

                if p.status == "error":
                    continue

                if p.status != "completed":
                    p.status = "error"
                    p.status_message = "Process crashed unexpectedly"
                    p.error_details = {
                        "traceback": "Process exited without reporting results. Check worker logs."
                    }
                    changed = True
                    continue

                if p.completed_time is None:
                    p.completed_time = now
                    changed = True

                if now - p.completed_time > COMPLETED_RETENTION_SECONDS:
                    to_remove.append(pid)

        for pid in to_remove:
            del self._processes[pid]

        if to_remove or changed:
            self._save()
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} finished processes")

        return len(to_remove) + self._cleanup_jobs()

    def kill(self, pid: int) -> bool:
        """Kill a tracked process by pid."""
        if pid not in self._processes:
            return False

        info = self._processes[pid]
        if info.kill():
            del self._processes[pid]
            self._save()
            return True
        return False

    def kill_all(self) -> int:
        """Kill all tracked processes. returns count killed."""
        killed = 0
        for pid in list(self._processes.keys()):
            if self.kill(pid):
                killed += 1
        return killed


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def prune_logs() -> None:
    """Trim ~/.mbo/logs so task logs don't accumulate unbounded.

    Deletes transient sidecars (progress_*/args_*/*.tmp) past a 24h grace, and
    keeps the newest ``MBO_LOG_KEEP_RECENT`` (25) task logs while dropping the
    rest once older than ``MBO_LOG_RETENTION_DAYS`` (14). Best-effort: a running
    task's recent files fall inside the grace/keep window and are spared.
    """
    try:
        log_dir = get_mbo_dirs()["logs"]
        now = time.time()
        retention = _env_int("MBO_LOG_RETENTION_DAYS", 14) * 86400
        keep_recent = _env_int("MBO_LOG_KEEP_RECENT", 25)
        grace = 24 * 3600

        removed = 0
        for pat in ("progress_*.json", "args_*.json", "*.tmp"):
            for p in log_dir.glob(pat):
                try:
                    if now - p.stat().st_mtime > grace:
                        p.unlink(missing_ok=True)
                        removed += 1
                except OSError:
                    pass

        logs = sorted(
            (p for p in log_dir.glob("*.log*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        cutoff = now - retention
        for p in logs[keep_recent:]:
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass

        if removed:
            logger.debug(f"prune_logs: removed {removed} stale files from {log_dir}")
    except Exception as e:
        logger.debug(f"prune_logs failed: {e}")


_manager: ProcessManager | None = None


def get_process_manager() -> ProcessManager:
    """Get the global process manager instance."""
    global _manager
    if _manager is None:
        _manager = ProcessManager()
    return _manager
