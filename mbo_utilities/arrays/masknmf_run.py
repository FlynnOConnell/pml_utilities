"""A masknmf run folder as its registered movie, rebuilt from the recording.

The MaskNMF pipeline writes one run folder per plane, masknmf's own layout:
``<yyyymmddTHHMMSS>_masknmf_zplaneNN[_tpAAAAA-BBBBB]/`` with ``results.hdf5``
(one group per stage), ``config.json``, the log and, when registration ran on
a denoised copy, ``alignment.hdf5``. No movie is written. The raw movie is the
recording itself, re-opened from ``config.json``'s ``inputs["movie"]`` (path,
reader kwargs, read features, plane, channel, frames), and the registered
movie replays the stored shifts on it with masknmf.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import ReductionMixin, _normalize_key
from mbo_utilities.arrays.features._frame_average import INVERT_DEFLECTION_KEY
from mbo_utilities.arrays.features._motion import MotionCorrection
from mbo_utilities.lazy_array import LazyArray
from mbo_utilities.masknmf.params import ALIGN_FILE
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.masknmf_run")

RUN_CONFIG = "config.json"
RESULTS_FILE = "results.hdf5"
# the registration groups masknmf stores, by array class name
REGISTRATION_GROUPS = (
    "RigidRegistrationArray",
    "PiecewiseRigidRegistrationArray",
    "GradientRegistrationArray",
)

register_pipeline(
    PipelineInfo(
        name="masknmf_run",
        description="masknmf run folder",
        input_patterns=["**/config.json"],
        output_patterns=[],
        input_extensions=[],
        output_extensions=[],
        marker_files=[RUN_CONFIG, RESULTS_FILE],
        category="reader",
    )
)


def is_masknmf_run(path: Path | str) -> bool:
    """Whether ``path`` is a folder holding ``config.json`` and ``results.hdf5``."""
    p = Path(path)
    return (p / RUN_CONFIG).is_file() and (p / RESULTS_FILE).is_file()


def run_demixing(path: Path | str) -> Path | None:
    """A run folder's ``results.hdf5`` when the run demixed, else None."""
    p = Path(path)
    if not is_masknmf_run(p):
        return None
    with h5py.File(p / RESULTS_FILE, "r") as f:
        return p / RESULTS_FILE if "DemixingResults" in f else None


def run_config(run: Path | str) -> dict:
    """A run folder's ``config.json``."""
    return json.loads((Path(run) / RUN_CONFIG).read_text(encoding="utf-8"))


def run_raw_movie(run: Path | str, invert: bool = True):
    """The ``(T, Y, X)`` recording a run read, re-opened from its ``config.json``.

    A ``PlaneMovie`` (``roi_workflow``) over ``imread`` of the recorded path,
    with the recorded read features applied (Invert Deflection only when
    ``invert``) and the run's plane, channel and frames selected. None when
    the run was given an in-memory array.
    """
    from mbo_utilities.arrays.features import apply_read_features
    from mbo_utilities.reader import imread
    from mbo_utilities.roi_workflow import PlaneMovie

    movie = run_config(run)["inputs"]["movie"]
    if not movie.get("path"):
        return None
    arr = imread(movie["path"], **(movie.get("reader_kwargs") or {}))
    features = dict(movie.get("read_features") or {})
    if not invert:
        features.pop(INVERT_DEFLECTION_KEY, None)
    arr, _ = apply_read_features(arr, features)
    frames = movie.get("frames")
    if frames is not None:
        frames = range(*frames)
    return PlaneMovie(arr, z=int(movie["z"]), c=int(movie["c"])).select(
        frames if frames is not None else movie.get("tp_indices")
    )


def registration_group(path: Path | str) -> str | None:
    """The registration group a results file holds, or None."""
    with h5py.File(path, "r") as f:
        return next((name for name in REGISTRATION_GROUPS if name in f), None)


class MasknmfRunArray(ReductionMixin, LazyArray):
    """One masknmf run folder as its registered movie, ``(T, 1, 1, Y, X)``.

    ``raw`` is the recording the run read; without a registration in
    ``results.hdf5`` the registered movie is the raw one. With
    ``on_alignment_copy`` the shifts are replayed on the run's
    ``alignment.hdf5`` (the denoised copy they were estimated on) instead of
    the recording, never inverted. ``device`` is where masknmf applies the shifts.
    """

    # above ResultsArray (70), which claims any folder holding a results file
    PRIORITY = 75

    def __init__(
        self,
        filenames: Path | str,
        device: str = "cpu",
        on_alignment_copy: bool = False,
    ):
        run = Path(filenames)
        self.run_folder = run
        self.filenames = [run]
        self.device = device
        self.on_alignment_copy = on_alignment_copy
        self._registered = None
        self._results_read = False
        self.config = run_config(run)
        self.raw = run_raw_movie(run)
        if self.raw is None:
            raise ValueError(
                f"{run} was run on an in-memory array; there is no recording to read"
            )
        movie = self.config["inputs"]["movie"]
        self._metadata = dict(getattr(self.raw.arr, "metadata", None) or {})
        self._metadata.update(
            {
                "num_timepoints": self.raw.shape[0],
                "Ly": self.raw.shape[1],
                "Lx": self.raw.shape[2],
            }
        )
        if movie.get("fs"):
            self._metadata["fs"] = float(movie["fs"])
        self._group = registration_group(run / RESULTS_FILE)

    @classmethod
    def can_open(cls, path: Path | str) -> bool:
        return is_masknmf_run(path)

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        t, y, x = self.raw.shape
        return (t, 1, 1, y, x)

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def reader_kwargs(self) -> dict:
        return {}

    @property
    def registered(self):
        """The registration replayed on ``raw`` or the alignment copy (masknmf), or ``raw`` when the run did not register."""
        if self._registered is None:
            if self._group is None:
                self._registered = self.raw
            else:
                from masknmf.io import REGISTRATION_ARRAYS

                from masknmf import OphysArray

                after = self.config["inputs"]["movie"].get(
                    "registered_before_inversion"
                )
                if self.on_alignment_copy:
                    from masknmf import CompressionArray

                    movie = CompressionArray.from_hdf5(self.run_folder / ALIGN_FILE)
                    after = False
                elif after:
                    movie = run_raw_movie(self.run_folder, invert=False)
                else:
                    movie = self.raw
                self._registered = REGISTRATION_ARRAYS[self._group].from_hdf5(
                    self.run_folder / RESULTS_FILE,
                    input_movie=movie,
                    device=self.device,
                )
                if after:
                    self._registered = OphysArray(
                        self._registered,
                        negative_indicator=True,
                        include_mean=True,
                        device=self.device,
                    )
        return self._registered

    @property
    def frames(self) -> np.ndarray:
        """The recording's frame under each of the run's, 0-based."""
        movie = self.config["inputs"]["movie"]
        if movie.get("frames") is not None:
            return np.arange(*movie["frames"])
        if movie.get("tp_indices") is not None:
            return np.asarray(movie["tp_indices"])
        return np.arange(self.raw.shape[0])

    @property
    def recording(self):
        """The recording the run read, as ``imread`` opened it (read features applied)."""
        return self.raw.arr

    @property
    def shifts(self) -> np.ndarray | None:
        """The stored shifts, ``(T, 2)`` (y, x) in px, or blockwise for piecewise rigid."""
        if self._group is None:
            return None
        with h5py.File(self.run_folder / RESULTS_FILE, "r") as f:
            return np.asarray(f[self._group]["shifts"][()], np.float32)

    @property
    def motion_correction(self) -> MotionCorrection | None:
        """The run's shifts as y and x traces in px; blockwise shifts are averaged over blocks."""
        shifts = self.shifts
        if shifts is None:
            return None
        shifts = shifts.reshape(shifts.shape[0], -1, 2).mean(axis=1)
        fs = self._metadata.get("fs")
        t = (
            np.arange(shifts.shape[0]) / fs
            if fs
            else np.arange(shifts.shape[0], dtype=float)
        )
        return MotionCorrection(
            source="masknmf",
            unit="px",
            traces={"Y": (t, shifts[:, 0]), "X": (t, shifts[:, 1])},
        )

    @property
    def results(self):
        """The demixing this run wrote (its results zarr), None without one. Read once."""
        if not self._results_read:
            from mbo_utilities.results import Results, newest_results

            self._results_read = True
            found = newest_results(self.run_folder)
            self._results = None if found is None else Results.read(found)
            for unit in [] if self._results is None else self._results.units.values():
                unit.attrs["z"] = 0
        return self._results

    @results.setter
    def results(self, value) -> None:
        self._results, self._results_read = value, True

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = key + (slice(None),) * (5 - len(key))
        t_key, c_key, z_key, y_key, x_key = key
        nt = self.raw.shape[0]
        ts = np.atleast_1d(np.arange(nt)[t_key]).tolist()
        frames = self.registered[ts]
        frames = frames.cpu().numpy() if hasattr(frames, "cpu") else np.asarray(frames)
        out = frames.reshape(len(ts), 1, 1, *self.raw.shape[1:]).astype(np.float32)
        out = out[:, c_key, z_key, y_key, x_key]
        if isinstance(t_key, (int, np.integer)):
            out = out[0]
        return out

    def __array__(self, dtype=None, copy=None):
        frame = self[0, 0, 0]
        return frame if dtype is None else frame.astype(dtype)

    def close(self) -> None:
        self._registered = None
