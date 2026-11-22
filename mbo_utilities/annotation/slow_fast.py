"""Curation of one trace's fast peaks, found by the slow / fast detector.

The second curation method beside vnoiser's (``vnoiser.session``): the
candidates are the fast peaks :func:`~mbo_utilities.analysis.slow_fast.detect_slow_fast`
finds inside slow events, each labeled yes or no by hand; there are no auto
calls. Labels are keyed by a peak's sample index, so they survive a change of
settings wherever the same peak is found again. One JSON file per curation
folder, ``.curation/slow_fast_curation.json``, holds every recording's
settings, labels and whether the trace is excluded; :func:`export_tables`
writes the detection's tables beside it as CSV. Needs neither vnoiser nor a
GUI.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from mbo_utilities.analysis.dff import dfof_zscore
from mbo_utilities.analysis.slow_fast import (
    FAST_COLUMNS,
    SLOW_COLUMNS,
    SlowFastEvents,
    SlowFastSettings,
    detect_slow_fast,
)
from mbo_utilities.analysis.trace_readouts import event_dprime
from mbo_utilities.results import (
    CURATION_DIR,
    ResultsArray,
    recording_id,
    trace_label,
)

__all__ = [
    "CURATION_FILE",
    "LABEL_RGBA",
    "MODE",
    "TABLE_FILES",
    "SlowFastSession",
    "curated_trace",
    "events_positive_dff",
    "export_run_tables",
    "export_tables",
    "kept_dprime",
    "kept_times",
]

MODE = "slow_fast"
CURATION_FILE = "slow_fast_curation.json"
TABLE_FILES = {
    "recordings": "slow_fast_recordings.csv",
    "slow": "slow_fast_slow.csv",
    "fast": "slow_fast_fast.csv",
}
LABELS = ("yes", "no", "unlabeled")
VIEW_FILTERS = ("all", "yes", "no", "unlabeled")
## NOTE: vnoiser.curation.LABEL_COLORS, copied so this module runs without vnoiser
LABEL_RGBA = {
    "yes": (0x1B / 255, 0x99 / 255, 0x8B / 255, 1.0),
    "no": (0xD7 / 255, 0x26 / 255, 0x3D / 255, 1.0),
    "auto_yes": (0x91 / 255, 0xD7 / 255, 0xC9 / 255, 1.0),
    "auto_no": (0xF3 / 255, 0xA0 / 255, 0xAA / 255, 1.0),
    "unlabeled": (0x52 / 255, 0x58 / 255, 0x66 / 255, 1.0),
}


def events_positive_dff(arr: ResultsArray, unit: str, roi: str) -> np.ndarray:
    """One ROI's dF/F of a run with its events pointing up.

    The run's ``dff`` (``dfof_raw``) is negated unless the run inverted F
    before dF/F or recorded a brightening indicator (``polarity`` ``invert_f``
    or ``none``); an archive folder records no polarity and is negated, as the
    detector's own script does. A run without ``dff`` has it computed from
    ``raw`` with the pipeline's default dF/F.
    """
    found = arr.results.units[unit]
    polarity = arr.results.provenance.get("polarity", "flip_z")
    if "dff" in found.traces:
        dff = np.asarray(arr.trace(roi, "dff", unit), dtype=np.float64)
    elif "raw" in found.traces:
        polarity = "flip_z"
        dff = dfof_zscore(arr.trace(roi, "raw", unit), found.fs)[0][0]
    else:
        raise KeyError(f"{arr.path} has no dff or raw trace for {unit} / {roi}")
    return dff if polarity in ("invert_f", "none") else -dff


def kept_times(session) -> np.ndarray:
    """Times of a session's candidates not called no, for either method."""
    keep = [i for i in range(session.n) if session.label(i) not in ("no", "auto_no")]
    return np.asarray(session.times_s, dtype=float)[keep]


def curated_trace(session) -> np.ndarray:
    """The trace a session's candidates were found on, for either method: the
    baseline-subtracted dF/F, or the denoised trace.
    """
    return session.events.x if session.mode == MODE else session.denoised


def kept_dprime(session) -> tuple[float, int]:
    """``(d', n)`` of a session's candidates not called no, on their own trace,
    so vnoiser's d' runs higher by the noise it took out.
    """
    peaks = np.round(kept_times(session) * session.fs).astype(np.int64)
    return event_dprime(curated_trace(session), peaks, session.fs)


class SlowFastSession:
    """Curation of the fast peaks of one trace.

    Parameters
    ----------
    settings : SlowFastSettings, optional
        Detection settings for a recording the file has none for.
    """

    mode = MODE

    def __init__(self, settings: SlowFastSettings | None = None):
        self.settings = settings or SlowFastSettings()
        self.events: SlowFastEvents | None = None
        self.trace = np.zeros(0)
        self.fs = 0.0
        self.recording_id = ""
        self.recording_label = ""
        self.block = None
        self.scan_labels: list[str] = []
        self.label_path: Path | None = None
        self.labels: dict[int, str] = {}
        self.excluded = False
        self.current = 0
        self.view_filter = "all"
        self.status = ""

    def load_run(self, arr: ResultsArray, unit: str, roi: str) -> str:
        """Curate one ROI of one unit of a run; labels in ``<run>/.curation``."""
        unit, roi = str(unit), str(roi)
        found = arr.results.units.get(unit)
        if found is None or roi not in found.roi_names:
            raise KeyError(f"{arr.path} has no trace for {unit} / {roi}")
        self.block = found.attrs.get("block")
        self.scan_labels = list(found.attrs.get("labels") or [])
        return self._open(
            events_positive_dff(arr, unit, roi),
            found.fs,
            recording_id(found, roi),
            trace_label(found, roi),
            arr.path / CURATION_DIR,
        )

    def load_trace(
        self,
        trace,
        fs_hz: float,
        *,
        recording_id: str,
        label: str,
        source_path,
        curation_dir=None,
    ) -> str:
        """Curate a raw trace held in memory (a line-scan ROI, events already
        pointing up): its dF/F over the pipeline's default baseline. Labels go
        to ``curation_dir`` (``.curation`` beside the source by default).
        """
        trace = np.asarray(trace, dtype=np.float64).ravel()
        if trace.size < 2:
            raise ValueError(f"{label} has fewer than two samples")
        if not np.isfinite(trace).all():
            raise ValueError(f"{label} contains NaN or infinite values")
        folder = (
            Path(curation_dir)
            if curation_dir is not None
            else Path(source_path).parent / CURATION_DIR
        )
        dff = dfof_zscore(trace, float(fs_hz))[0][0]
        return self._open(dff, fs_hz, recording_id, label, folder)

    def _open(self, trace, fs, rid, label, folder: Path) -> str:
        self.trace = np.asarray(trace, dtype=np.float64)
        self.fs = float(fs)
        self.recording_id = str(rid)
        self.recording_label = str(label)
        self.label_path = Path(folder) / CURATION_FILE
        entry = self._read_file().get("recordings", {}).get(self.recording_id, {})
        if "settings" in entry:
            self.settings = SlowFastSettings.from_dict(entry["settings"])
        self.excluded = bool(entry.get("excluded", False))
        self.labels = {int(k): v for k, v in entry.get("labels", {}).items()}
        self._detect()
        self.current = 0
        return self.status

    def _detect(self) -> None:
        self.events = detect_slow_fast(self.trace, self.fs, self.settings)
        self.status = (
            f"loaded {self.recording_label}; excluded"
            if self.excluded
            else f"loaded {self.recording_label}; {self.events.n_slow} slow events, "
            f"{self.events.n_fast} fast peaks"
        )

    @property
    def loaded(self) -> bool:
        return self.events is not None

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.trace.size, dtype=float) / self.fs

    @property
    def sample_index(self) -> np.ndarray:
        """Sample of each candidate; none while the trace is excluded."""
        if self.events is None or self.excluded:
            return np.zeros(0, dtype=np.int64)
        return self.events.fast["sample_index"]

    @property
    def n(self) -> int:
        return int(self.sample_index.size)

    @property
    def times_s(self) -> np.ndarray:
        return self.sample_index / self.fs

    @property
    def peak_values(self) -> np.ndarray:
        """The fast trace at each candidate, the height panel A draws it at."""
        return self.events.fast_trace[self.sample_index]

    def set_settings(self, **changes) -> None:
        """Detect again with ``changes`` applied and save; the focus stays on
        the candidate nearest the one focused before.
        """
        settings = replace(self.settings, **changes)
        if settings == self.settings or not self.loaded:
            return
        before = self.times_s[self.current] if self.n else None
        self.settings = settings
        self._detect()
        if before is not None and self.n:
            self.current = int(np.argmin(np.abs(self.times_s - before)))
        else:
            self.current = 0
        self._save()

    def set_excluded(self, excluded: bool) -> None:
        """Exclude the whole trace (no candidates) or bring it back, and save."""
        if bool(excluded) == self.excluded or not self.loaded:
            return
        self.excluded = bool(excluded)
        self.current = 0
        self._detect()
        self._save()

    def select(self, index: int) -> None:
        if 0 <= index < self.n:
            self.current = int(index)

    def step(self, delta: int) -> None:
        """Focus the previous / next candidate passing the view filter."""
        visible = self.visible
        if not visible.size:
            return
        if delta > 0:
            after = visible[visible > self.current]
            self.current = int(after[0] if after.size else visible[-1])
        else:
            before = visible[visible < self.current]
            self.current = int(before[-1] if before.size else visible[0])

    def set_view_filter(self, value: str) -> None:
        if value in VIEW_FILTERS:
            self.view_filter = value

    @property
    def visible(self) -> np.ndarray:
        """Candidate indices passing the view filter."""
        if self.view_filter == "all":
            return np.arange(self.n)
        return np.array(
            [i for i in range(self.n) if self.label(i) == self.view_filter],
            dtype=int,
        )

    def label(self, index: int) -> str:
        return self.labels.get(int(self.sample_index[int(index)]), "unlabeled")

    def set_label(self, label: str) -> None:
        """Label the focused candidate and save."""
        self.set_labels([self.current], label)

    def set_labels(self, indices, label: str) -> int:
        """Label several candidates (a box) and save; returns how many."""
        if label not in LABELS or not self.n:
            return 0
        samples = [
            int(self.sample_index[int(i)]) for i in indices if 0 <= int(i) < self.n
        ]
        for sample in samples:
            if label == "unlabeled":
                self.labels.pop(sample, None)
            else:
                self.labels[sample] = label
        if samples:
            self._save()
        return len(samples)

    def clear_labels(self) -> int:
        """Drop every label of the candidates found now; returns how many."""
        labeled = [i for i in range(self.n) if self.label(i) != "unlabeled"]
        return self.set_labels(labeled, "unlabeled") if labeled else 0

    def counts(self) -> tuple[int, int, int]:
        """``(yes, no, unlabeled)`` over the candidates found now."""
        shown = [self.label(i) for i in range(self.n)]
        return shown.count("yes"), shown.count("no"), shown.count("unlabeled")

    def colors(self, indices=None) -> np.ndarray:
        """``(n, 4)`` rgba per candidate (or per ``indices``)."""
        indices = range(self.n) if indices is None else indices
        return np.array(
            [LABEL_RGBA[self.label(int(i))] for i in indices], dtype=np.float32
        ).reshape(-1, 4)

    def slow_bounds(self) -> np.ndarray:
        """``(n_slow, 2)`` start and end of each slow event, in seconds."""
        if self.events is None or self.excluded:
            return np.zeros((0, 2))
        return np.column_stack([self.events.slow["start_s"], self.events.slow["end_s"]])

    def event_info(self, index: int) -> dict:
        """The focused candidate and the slow event it sits in."""
        i = int(index)
        fast, slow = self.events.fast, self.events.slow
        k = int(fast["slow_id"][i])
        return {
            "index": i,
            "label": self.label(i),
            "time_s": float(fast["time_s"][i]),
            "amplitude": float(fast["amplitude"][i]),
            "prominence_noise": float(fast["prominence_noise"][i]),
            "width_ms": float(fast["width_ms"][i]),
            "slow_id": k,
            "slow_start_s": float(slow["start_s"][k]),
            "slow_end_s": float(slow["end_s"][k]),
            "slow_amplitude_sd": float(slow["amplitude_sd"][k]),
            "slow_n_fast": int(slow["n_fast"][k]),
        }

    def tables(self) -> tuple[dict, dict]:
        """The slow table with its ``slow_id`` and the fast one with a ``label``
        column; both empty for an excluded trace, as the original script keeps it.
        """
        slow = {"slow_id": np.arange(self.events.n_slow), **self.events.slow}
        fast = dict(self.events.fast)
        fast["label"] = np.array(
            [self.labels.get(int(s), "unlabeled") for s in fast["sample_index"]],
            dtype=object,
        )
        if self.excluded:
            slow = {k: v[:0] for k, v in slow.items()}
            fast = {k: v[:0] for k, v in fast.items()}
        return slow, fast

    def _read_file(self) -> dict:
        if self.label_path is None or not self.label_path.exists():
            return {}
        return json.loads(self.label_path.read_text())

    def _save(self) -> None:
        """Write this recording's entry; every other recording's stays."""
        data = self._read_file()
        data["method"] = MODE
        data.setdefault("recordings", {})[self.recording_id] = {
            "label": self.recording_label,
            "fs": self.fs,
            "settings": self.settings.to_dict(),
            "excluded": self.excluded,
            "labels": {str(k): v for k, v in sorted(self.labels.items())},
        }
        self.label_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.label_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.label_path)


def export_tables(sessions, folder) -> list[Path]:
    """Write loaded sessions' tables as three CSVs in ``folder`` (:data:`TABLE_FILES`).

    ``recordings`` has one row per trace (block, its scan's labels, excluded,
    counts, noise, d', settings), ``slow`` and ``fast`` one row per event, keyed
    by ``recording_id``; the fast table's ``label`` is the curation's yes / no /
    unlabeled.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    settings_names = list(SlowFastSettings().to_dict())
    recordings, slow_rows, fast_rows = [], [], []
    for session in sessions:
        slow, fast = session.tables()
        yes, no, unlabeled = session.counts()
        dprime, dprime_n = kept_dprime(session)
        key = {
            "recording_id": session.recording_id,
            "recording": session.recording_label,
        }
        recordings.append(
            {
                **key,
                "fs": session.fs,
                "block": session.block,
                "scan_labels": ";".join(session.scan_labels),
                "excluded": session.excluded,
                "n_slow": len(slow["slow_id"]),
                "n_fast": len(fast["sample_index"]),
                "yes": yes,
                "no": no,
                "unlabeled": unlabeled,
                "sd": session.events.sd,
                "noise": session.events.noise,
                "dprime": dprime,
                "dprime_n": dprime_n,
                **asdict(session.settings),
            }
        )
        slow_rows += [
            {**key, **{k: v[i] for k, v in slow.items()}}
            for i in range(len(slow["slow_id"]))
        ]
        fast_rows += [
            {**key, **{k: v[i] for k, v in fast.items()}}
            for i in range(len(fast["sample_index"]))
        ]
    key_columns = ["recording_id", "recording"]
    columns = {
        "recordings": [
            *key_columns,
            "fs",
            "block",
            "scan_labels",
            "excluded",
            "n_slow",
            "n_fast",
            "yes",
            "no",
            "unlabeled",
            "sd",
            "noise",
            "dprime",
            "dprime_n",
            *settings_names,
        ],
        "slow": [*key_columns, "slow_id", *SLOW_COLUMNS],
        "fast": [*key_columns, *FAST_COLUMNS, "label"],
    }
    paths = []
    for name, rows in (
        ("recordings", recordings),
        ("slow", slow_rows),
        ("fast", fast_rows),
    ):
        path = folder / TABLE_FILES[name]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns[name])
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths


def export_run_tables(
    arr: ResultsArray, folder=None, settings: SlowFastSettings | None = None
) -> list[Path]:
    """Detect every ROI of a run with its saved settings and labels and write
    :func:`export_tables` to ``folder`` (``<run>/.curation`` by default).

    A trace with nothing saved uses ``settings`` (the defaults without).
    """
    sessions = []
    for unit_name, unit in arr.results.units.items():
        if "dff" not in unit.traces and "raw" not in unit.traces:
            continue
        for roi in unit.roi_names:
            session = SlowFastSession(settings)
            session.load_run(arr, unit_name, roi)
            sessions.append(session)
    return export_tables(
        sessions, arr.path / CURATION_DIR if folder is None else folder
    )
