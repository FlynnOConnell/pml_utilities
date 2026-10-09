"""Spike-triggered averages: what a recording does, on average, in the frames
around each spike.

The spikes are the peaks of one of masknmf's demixed traces over a threshold
(or the events the curation window accepted), the movies are masknmf's
registered and compressed movies of the same recording, and the traces are
the motion each stage measured: the AOD's real-time motion correction and
masknmf's rigid shifts. Activity that is real shows in the movies and leaves
the motion averages flat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks

from mbo_utilities.lazy_array import base_array
from mbo_utilities.reader import imread

__all__ = [
    "MasknmfUnit",
    "SpikeAverage",
    "accepted_events",
    "threshold_peaks",
    "triggered_average",
]


@dataclass
class SpikeAverage:
    """One recording averaged over its spikes.

    ``movies`` maps a name to ``(lags, Y, X)``. ``traces`` maps a y label to
    ``{line: (mean, sem)}``, each ``(lags,)`` about the window's own mean.
    """

    lags: np.ndarray
    fs: float
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
        data = np.asarray(source[unique], dtype=np.float64)[
            inverse.reshape(windows.shape)
        ]
        if center:
            data = data - data.mean(axis=1, keepdims=True)
        total = total + data.sum(axis=0)
        squares = squares + np.square(data).sum(axis=0)
    mean = total / frames.size
    sem = np.sqrt(np.maximum(squares / frames.size - mean**2, 0.0) / frames.size)
    return mean, sem, int(frames.size)


@dataclass
class MasknmfUnit:
    """One ``.mesc`` unit with its masknmf run, read once so any set of
    spikes can be averaged over it.

    Every array counts frames from ``first_frame`` of the unit. ``movies``
    maps a name to a ``(frames, Y, X)`` movie, ``traces`` a y label to
    ``{line: (frames,)}`` and ``signals`` is the run's ``temporal_demixed``
    as ``(frames, signals)``. ``recording`` is the id the curation window
    saves this ROI's events under.
    """

    label: str
    recording: str
    fs: float
    first_frame: int
    movies: dict
    traces: dict[str, dict[str, np.ndarray]]
    signals: np.ndarray

    @classmethod
    def open(
        cls,
        mesc_path,
        unit: str,
        masknmf_path,
        channel: int = 0,
        roi: int = 0,
        first_frame: int | None = None,
        negative: bool = True,
    ) -> MasknmfUnit:
        """Read one ROI of a unit and a masknmf results hdf5 run on it,
        registered rigidly.

        The run's shifts are replayed on the unit's frames for the registered
        movie and its compressed movie is read as stored. ``channel`` and
        ``roi`` are 0-based. ``first_frame`` is the unit's frame the run
        starts on: the file's ``retained_frames[0]`` when None, so a run
        handed a movie that was already cut needs it given. ``negative``
        flips the frames about their mean, as the run's input was for an
        indicator that dims on a spike. A run folder the MaskNMF pipeline
        wrote knows all of this: open it with :meth:`from_run`.
        """
        import masknmf

        arr = imread(mesc_path, unit=unit)
        with h5py.File(masknmf_path, "r") as f:
            signals = f["DemixingResults/temporal_demixed"][()]
            if first_frame is None:
                first_frame = (
                    int(f["retained_frames"][0]) if "retained_frames" in f else 0
                )
        n_frames = signals.shape[0]
        raw = arr[first_frame : first_frame + n_frames, channel, roi]
        if raw.shape[0] != n_frames:
            raise ValueError(
                f"{Path(masknmf_path).name} holds {n_frames} frames, "
                f"{Path(mesc_path).name} {unit} has {raw.shape[0]} from frame {first_frame}"
            )
        registered = masknmf.RigidRegistrationArray.from_hdf5(
            masknmf_path,
            input_movie=masknmf.OphysArray(
                raw, negative_indicator=negative, include_mean=True, device="cpu"
            ),
        )
        return cls._read(
            arr,
            roi,
            first_frame + np.arange(n_frames),
            registered,
            registered.shifts.cpu().numpy(),
            masknmf_path,
            signals,
        )

    @classmethod
    def from_run(cls, run) -> MasknmfUnit:
        """Read a run folder the MaskNMF pipeline wrote, demixing included.

        The recording, its plane, channel, frames and whether it was inverted
        are the ones ``config.json`` records; the registered movie is
        :class:`~mbo_utilities.arrays.masknmf_run.MasknmfRunArray`'s.
        """
        from mbo_utilities.arrays.masknmf_run import RESULTS_FILE, MasknmfRunArray

        run_arr = MasknmfRunArray(run)
        results_path = run_arr.run_folder / RESULTS_FILE
        with h5py.File(results_path, "r") as f:
            if "DemixingResults" not in f:
                raise ValueError(f"{run_arr.run_folder.name} did not demix: no spikes to find")
            signals = f["DemixingResults/temporal_demixed"][()]
        movie = run_arr.config["inputs"]["movie"]
        if movie.get("frames") is not None:
            frames = np.arange(*movie["frames"])
        elif movie.get("tp_indices") is not None:
            frames = np.asarray(movie["tp_indices"])
        else:
            frames = np.arange(signals.shape[0])
        shifts = run_arr.shifts
        shifts = (
            np.zeros((signals.shape[0], 2), np.float32)
            if shifts is None
            else shifts.reshape(shifts.shape[0], -1, 2).mean(axis=1)
        )
        return cls._read(
            run_arr.raw.arr,
            int(movie["z"]),
            frames,
            run_arr.registered,
            shifts,
            results_path,
            signals,
        )

    @classmethod
    def _read(cls, arr, roi, frames, registered, shifts, results_path, signals):
        """The movies and traces of ``registered`` and the compressed movie in
        ``results_path``, on the frames of ``arr`` the run read.
        """
        import masknmf

        n_frames = signals.shape[0]
        movies = {
            "registered": registered,
            "compressed": masknmf.CompressionArray.from_hdf5(results_path),
        }
        traces = {
            "mean F (a.u.)": {
                name: np.concatenate(
                    [
                        np.asarray(movie[start : min(start + 4096, n_frames)]).mean(
                            axis=(1, 2)
                        )
                        for start in range(0, n_frames, 4096)
                    ]
                )
                for name, movie in movies.items()
            }
        }
        motion = base_array(arr).motion_correction
        if motion is not None:
            frame_times = np.asarray(frames, dtype=np.float64) / arr.fs
            traces[f"{motion.source} shift ({motion.unit})"] = {
                line: np.interp(frame_times, t, shift)
                for line, (t, shift) in motion.traces.items()
            }
        traces["masknmf shift (px)"] = {"Y": shifts[:, 0], "X": shifts[:, 1]}
        source = Path(base_array(arr).source_path)
        # the curation window's id: <stem>/<MUnit>/roi=<roi> for a .mesc unit
        munit = base_array(arr).metadata.get("mesc_unit", "").split("/")[-1]
        name = "/".join(filter(None, (source.stem, munit)))
        return cls(
            label=" ".join(filter(None, (source.name, munit, f"ROI {roi}"))),
            recording=f"{name}/roi={roi}",
            fs=float(arr.fs),
            first_frame=int(frames[0]),
            movies=movies,
            traces=traces,
            signals=signals,
        )

    def average(self, spikes, before: int = 10, after: int = 10) -> SpikeAverage:
        """The movies and traces averaged around ``spikes``, frames counted
        from ``first_frame``.
        """
        movies = {}
        for name, movie in self.movies.items():
            mean, _sem, n_spikes = triggered_average(movie, spikes, before, after)
            movies[name] = mean.astype(np.float32)
        return SpikeAverage(
            lags=np.arange(-before, after + 1),
            fs=self.fs,
            n_spikes=n_spikes,
            movies=movies,
            traces={
                y_label: {
                    line: triggered_average(trace, spikes, before, after, True)[:2]
                    for line, trace in lines.items()
                }
                for y_label, lines in self.traces.items()
            },
            label=self.label,
        )
