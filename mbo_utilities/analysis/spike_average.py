"""Spike-triggered averages: what a recording does, on average, in the frames
around each spike.

One ROI of a run's results (any pipeline that writes the results file,
AGENTS.md §7.5) over the movie it was measured on. The spikes are the run's
detected events for that ROI, the events a curation file beside the run
accepts, or the peaks of one of its traces over a threshold. Around them: the
movie (and a MaskNMF run's compressed movie), the ROI's trace and every motion
correction the movie went through (masknmf's shifts, the AOD's RTMC).
Activity that is real shows in the movies and leaves the motion flat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.signal import find_peaks

from mbo_utilities.arrays.masknmf_run import (
    RESULTS_FILE,
    MasknmfRunArray,
    is_masknmf_run,
)
from mbo_utilities.lazy_array import base_array
from mbo_utilities.reader import imread
from mbo_utilities.results import (
    CURATION_DIR,
    ResultUnit,
    recording_id,
    results_dir_of,
)
from mbo_utilities.roi_workflow import PlaneMovie

__all__ = [
    "EVENTS",
    "MEAN_F",
    "THRESHOLD",
    "SpikeAverage",
    "SpikeSource",
    "accepted_events",
    "threshold_peaks",
    "triggered_average",
]

# where spikes come from besides a source's own: the peaks of the trace over a line
THRESHOLD = "threshold"
# the run's detected events, as the results file holds them
EVENTS = "events"
MEAN_F = "mean F (a.u.)"
# the trace spikes are found on, most processed first
TRACE_ORDER = ("denoised", "dff", "zscore", "raw")


@dataclass
class SpikeAverage:
    """One ROI averaged over its spikes.

    ``movies`` maps a name to ``(lags, Y, X)``. ``traces`` maps a y label to
    ``{line: (mean, sem)}``, each ``(lags,)``; the trace and motion lines are
    about each window's own mean.
    """

    lags: np.ndarray
    fs: float | None
    n_spikes: int
    movies: dict[str, np.ndarray]
    traces: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
    label: str = ""


def accepted_events(label_path) -> dict[str, np.ndarray]:
    """The events a curation file accepts, as sorted samples per recording id.

    ``label_path`` is the ``<mode>_template_curation.json`` the curation
    window saves. An event counts when its label is ``yes``, set by hand or
    auto-passed; its sample is the aligned peak on the curated trace.
    """
    events = json.loads(Path(label_path).read_text(encoding="utf-8"))["events"]
    samples: dict[str, list[int]] = {}
    for event in events.values():
        if event["label"] == "yes":
            samples.setdefault(event["recording"], []).append(
                event["source_aligned_index"]
            )
    return {recording: np.unique(found) for recording, found in samples.items()}


def threshold_peaks(trace, threshold: float, distance: int = 5) -> np.ndarray:
    """Frames of the peaks of ``trace`` at or over ``threshold``, in the
    trace's units, at least ``distance`` frames apart.
    """
    return find_peaks(trace, height=threshold, distance=distance)[0]


def triggered_average(
    source,
    frames,
    before: int = 10,
    after: int = 10,
    center: bool = False,
    block: int = 1024,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and standard error of ``source`` over the windows ``frame - before``
    to ``frame + after``.

    ``source`` is indexed along its first axis with an array of frames: a
    trace, a movie in memory, or a lazy movie that answers numpy or a CPU
    tensor, read ``block`` frames at a time. ``center`` takes each window's
    own mean off it first, so the error is of the shape within the window and
    not of the drift between windows. Returns ``(mean, sem, n)``, both
    ``(before + after + 1, ...)``, over the ``n`` frames whose window fits.
    """
    frames = np.asarray(frames, dtype=np.int64)
    given = frames.size
    frames = frames[(frames >= before) & (frames + after < source.shape[0])]
    if frames.size == 0:
        raise ValueError(
            f"none of {given} spikes has {before} frames before and {after} after "
            f"it in {source.shape[0]} frames"
        )
    width = before + after + 1
    step = max(block // width, 1)
    total = squares = 0.0
    for start in range(0, frames.size, step):
        windows = frames[start : start + step, None] + np.arange(-before, after + 1)
        unique, inverse = np.unique(windows, return_inverse=True)
        data = source[unique]
        data = data.cpu().numpy() if hasattr(data, "cpu") else data
        data = np.asarray(data, dtype=np.float64)[inverse.reshape(windows.shape)]
        if center:
            data = data - data.mean(axis=1, keepdims=True)
        total = total + data.sum(axis=0)
        squares = squares + np.square(data).sum(axis=0)
    mean = total / frames.size
    sem = np.sqrt(np.maximum(squares / frames.size - mean**2, 0.0) / frames.size)
    return mean, sem, int(frames.size)


@dataclass
class SpikeSource:
    """One ROI of a run's results over the movie it was measured on.

    Every array counts the frames of the ROI's traces. ``movies`` maps a name
    to a ``(T, Y, X)`` movie, ``traces`` a trace kind to the ROI's ``(T,)``
    trace, ``spikes`` where spikes come from (``events``, ``curated fast``)
    to their frames, and ``motion`` a y label to ``{line: (T,)}``. ``arr`` is
    the array the run was opened as, so another ROI of it opens without
    reading it again (:meth:`with_roi`).
    """

    arr: Any
    unit: ResultUnit
    roi: str
    c: int
    label: str
    fs: float | None
    movies: dict
    traces: dict[str, np.ndarray]
    spikes: dict[str, np.ndarray]
    motion: dict[str, dict[str, np.ndarray]]

    @classmethod
    def open(
        cls, path, unit: str | None = None, roi=None, c: int | None = None
    ) -> SpikeSource:
        """One ROI of the run at ``path``: a MaskNMF run folder, a results file
        or a folder holding one, a suite2p plane dir, anything ``imread``
        opens with ``results``.

        ``unit`` and ``roi`` are the results file's names (``zplane01``,
        ``scan35``; ``roi3``), the first of each by default. ``c`` is the
        movie's channel, 0-based; the run's own by default.
        """
        path = Path(path)
        # a results file a MaskNMF run wrote is read through its run folder,
        # whose movie is the registered one the traces came from
        if is_masknmf_run(path.parent):
            path = path.parent
        kwargs = (
            {"unit": unit}
            if unit and results_dir_of(path) is not None and not is_masknmf_run(path)
            else {}
        )
        return cls.of(imread(path, **kwargs), unit, roi, c)

    @classmethod
    def of(
        cls, arr, unit: str | None = None, roi=None, c: int | None = None
    ) -> SpikeSource:
        """One ROI of ``arr.results`` over ``arr``'s movie; see :meth:`open`."""
        results = arr.results
        if results is None or not results.units:
            raise ValueError(
                f"{Path(arr.source_path).name} has no results: run a pipeline that "
                "writes them (MaskNMF with demixing, Voltage, suite2p) first"
            )
        base = base_array(arr)
        name = unit or getattr(base, "unit", None) or next(iter(results.units))
        if name not in results.units:
            raise KeyError(f"no unit {name!r}; the results hold {list(results.units)}")
        found = results.units[name]
        roi = found.roi_names[0] if roi is None else str(roi)
        if roi not in found.roi_names:
            raise KeyError(f"{name} has no ROI {roi!r}; it has {found.roi_names}")
        k = found.roi_names.index(roi)
        c = int(results.source.get("channel") or 0) if c is None else int(c)
        # a line scan's ROI is read on its first line, the recording's Z
        z = (
            int(found.members[k][0])
            if found.member_kind == "line"
            else int(found.attrs.get("z") or 0)
        )
        n = found.n_timepoints
        first = int((results.source.get("frames") or [0])[0])
        movie = PlaneMovie(arr, z=z, c=c)
        if movie.shape[0] < first + n:
            raise ValueError(
                f"{name} has {n} frames from frame {first}, the movie {movie.shape[0]}"
            )
        if (first, n) != (0, movie.shape[0]):
            movie = movie.window(first, first + n)

        movies = {"movie": movie}
        if isinstance(base, MasknmfRunArray):
            movies = {"registered": movie}
            with h5py.File(base.run_folder / RESULTS_FILE, "r") as f:
                compressed = "CompressionArray" in f
            if compressed:
                import masknmf

                movies["compressed"] = masknmf.CompressionArray.from_hdf5(
                    base.run_folder / RESULTS_FILE
                )

        spikes = {}
        if len(found.events.get(roi, ())):
            spikes[EVENTS] = np.asarray(found.events[roi], np.int64)
        curation = Path(results.path or "") / CURATION_DIR
        for label_path in sorted(curation.glob("*_template_curation.json")):
            accepted = accepted_events(label_path).get(recording_id(found, roi))
            if accepted is not None and accepted.size:
                mode = label_path.name.removesuffix("_template_curation.json")
                spikes[f"curated {mode}"] = accepted

        # each stage's shifts on its own clock: a run's on its frames, its
        # recording's on the recording frames the run read
        clocks = [(arr, first + np.arange(n))]
        if isinstance(base, MasknmfRunArray):
            clocks.append((base.recording, base.frames[:n]))
        motion = {}
        for owner, frames in clocks:
            mc = base_array(owner).motion_correction
            if not mc:
                continue
            t = frames / owner.fs if owner.fs else frames.astype(np.float64)
            lines = {
                line: shift
                for line, shift in mc.at(t).items()
                if mc.planes.get(line, z) == z
            }
            if lines:
                motion[f"{mc.source} shift ({mc.unit})"] = lines

        return cls(
            arr=arr,
            unit=found,
            roi=roi,
            c=c,
            label=f"{Path(arr.source_path).name}  {name}  {roi}",
            fs=float(found.fs or arr.fs) if (found.fs or arr.fs) else None,
            movies=movies,
            traces={
                kind: np.asarray(found.traces[kind][k], np.float64)
                for kind in sorted(
                    found.traces,
                    key=lambda kind: TRACE_ORDER.index(kind)
                    if kind in TRACE_ORDER
                    else len(TRACE_ORDER),
                )
            },
            spikes=spikes,
            motion=motion,
        )

    def with_roi(self, roi) -> SpikeSource:
        """Another ROI of the same unit, on the array already open."""
        return SpikeSource.of(self.arr, self.unit.name, roi, self.c)

    def average(
        self, spikes, kind: str, before: int = 10, after: int = 10
    ) -> SpikeAverage:
        """The movies, the ROI's ``kind`` trace and the motion averaged around
        ``spikes``, frames of the ROI's traces.
        """
        movies = {}
        for name, movie in self.movies.items():
            mean, _sem, n_spikes = triggered_average(movie, spikes, before, after)
            movies[name] = mean.astype(np.float32)
        lines = {MEAN_F: {name: (m.mean(axis=(1, 2)), None) for name, m in movies.items()}}
        traces = {
            f"{self.roi} {kind}": {
                kind: triggered_average(self.traces[kind], spikes, before, after, True)[
                    :2
                ]
            },
            **lines,
        }
        for y_label, motion in self.motion.items():
            traces[y_label] = {
                line: triggered_average(shift, spikes, before, after, True)[:2]
                for line, shift in motion.items()
            }
        return SpikeAverage(
            lags=np.arange(-before, after + 1),
            fs=self.fs,
            n_spikes=n_spikes,
            movies=movies,
            traces=traces,
            label=self.label,
        )
