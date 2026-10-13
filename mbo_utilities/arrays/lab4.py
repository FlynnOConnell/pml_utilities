"""The standardized imaging H5 that every lab4 converter writes."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import DIMS, ReductionMixin, _imwrite_base
from mbo_utilities.arrays.bruker import _plain
from mbo_utilities.arrays.h5 import _index_5d_into_labeled
from mbo_utilities.arrays.suite2p import _extract_plane_number, registration_shifts
from mbo_utilities.file_io import has_plane_ops, load_plane_ops
from mbo_utilities.lazy_array import LazyArray, register_array_class
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.lab4")

DATASET = "imaging/data"

# lab4/convert/imaging.md: (T, Z, Y, X, C), or (T, R, Y, X, C) for an AOD scan
RAW_DIMS = ("T", "Z", "Y", "X", "C")

# MESc MethodType values the AOD converter writes with ROIs on the Z slot
AOD_ROI_MODALITIES = (6, 7, 8, 9, 10)

_LAB4_INFO = PipelineInfo(
    name="lab4",
    description="lab4 standardized imaging H5 (/imaging/data)",
    input_patterns=["**/*.h5", "**/*.hdf5"],
    output_patterns=[],
    input_extensions=["h5", "hdf5"],
    output_extensions=[],
    marker_files=[],
    category="reader",
)
register_pipeline(_LAB4_INFO)


def suite2p_run_planes(recording: Path) -> list[Path]:
    """The plane dirs of the newest suite2p run beside a lab4 recording, in plane order.

    lab4 writes each run as ``<params uuid>/suite2p/plane<N>/`` next to the
    recording's H5; the newest run is the one whose plane dirs changed last.
    """
    runs = []
    for root in Path(recording).parent.glob("*/suite2p"):
        planes = [
            d
            for d in root.iterdir()
            if d.is_dir() and _extract_plane_number(d.name) is not None and has_plane_ops(d)
        ]
        if planes:
            runs.append((max(d.stat().st_mtime for d in planes), planes))
    if not runs:
        return []
    if len(runs) > 1:
        logger.info(
            "%d suite2p runs beside %s; showing the newest, %s",
            len(runs),
            Path(recording).name,
            max(runs, key=lambda r: r[0])[1][0].parent.parent.name,
        )
    return sorted(max(runs, key=lambda r: r[0])[1], key=lambda d: _extract_plane_number(d.name))


def _is_lab4(f: h5py.File) -> bool:
    d = f.get(DATASET)
    return (
        isinstance(d, h5py.Dataset)
        and d.ndim == 5
        and "frame_rate_hz" in f["imaging"].attrs
    )


class Lab4H5Array(ReductionMixin, LazyArray):
    """One lab4 recording (``python -m lab4.convert``) as (T, C, Z, Y, X).

    ``/imaging/data`` is stored ``(T, Z, Y, X, C)``; HDF5 dimension labels, when
    the writer set them (the SLM route), decide the axes instead. An AOD scan
    keeps its ROIs on Z, as ``MescArray`` does. The ``/imaging`` group's attrs
    are the metadata, ``device_specifics`` parsed from its JSON. The newest
    suite2p run beside the file gives ``results`` and ``motion_correction``.
    """

    PRIORITY = 65

    def __init__(self, filenames: Path | str):
        self.filenames = [Path(filenames)]
        path = self.filenames[0]
        self._f = h5py.File(path, "r")
        try:
            if not _is_lab4(self._f):
                raise ValueError(f"{path} is not a lab4 imaging H5 (no {DATASET})")
            self._d = self._f[DATASET]
            labels = tuple(self._d.dims[i].label.upper() for i in range(self._d.ndim))
            self._raw_dims = labels if sorted(labels) == sorted(DIMS) else RAW_DIMS
            self._metadata = self._build_metadata()
            self._planes = None
            self._results, self._results_read = None, False
            self._motion, self._motion_read = None, False
        except Exception:
            self._f.close()
            raise

    def _build_metadata(self) -> dict:
        md = {k: _plain(v) for k, v in self._f.attrs.items()}
        md.update({k: _plain(v) for k, v in self._f["imaging"].attrs.items()})
        specifics = md.pop("device_specifics", "")
        if specifics:
            md["device_specifics"] = json.loads(specifics)
        nt, nc, nz, ny, nx = self._shape5d()
        # mini2p stamps z_step_um = 0.0 when it has no step; bruker leaves start_time ""
        if nz == 1 or not md.get("z_step_um"):
            md.pop("z_step_um", None)
        if not md.get("start_time"):
            md.pop("start_time", None)
        md.update(
            {
                "num_timepoints": nt,
                "num_color_channels": nc,
                "num_zplanes": nz,
                "Ly": ny,
                "Lx": nx,
                "dtype": self._d.dtype.name,
                "h5_dataset": DATASET,
                "h5_raw_dims": "".join(self._raw_dims),
            }
        )
        return md

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        p = Path(file)
        if not (p.is_file() and p.suffix.lower() in (".h5", ".hdf5")):
            return False
        try:
            with h5py.File(p, "r") as f:
                return _is_lab4(f)
        except OSError:
            return False

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        sizes = dict(zip(self._raw_dims, self._d.shape))
        return tuple(sizes.get(d, 1) for d in DIMS)

    @property
    def dtype(self):
        return self._d.dtype

    @property
    def suite2p_planes(self) -> list[Path]:
        """Plane dirs of the newest suite2p run beside this file (``suite2p_run_planes``), found once."""
        if self._planes is None:
            self._planes = suite2p_run_planes(self.filenames[0])
        return self._planes

    @property
    def results(self):
        """The newest suite2p run's detection as ``results.Results``, read once; None without one."""
        if not self._results_read:
            from mbo_utilities.results import Results

            self._results_read = True
            planes = [
                d
                for d in self.suite2p_planes
                if (d / "stat.npy").is_file() and (d / "F.npy").is_file()
            ]
            if planes:
                self._results = Results.from_suite2p(planes[0].parent, plane_dirs=planes)
        return self._results

    @results.setter
    def results(self, value) -> None:
        self._results, self._results_read = value, True

    @property
    def motion_correction(self):
        """The newest suite2p run's rigid shifts (``registration_shifts``), read once."""
        if not self._motion_read:
            self._motion_read = True
            planes = self.suite2p_planes
            self._motion = registration_shifts(
                [(d.name, load_plane_ops(d)) for d in planes], self.fs, len(planes) > 1
            )
        return self._motion

    @property
    def z_is_roi(self) -> bool:
        """True for an AOD scan, whose Z slot holds its ROIs, not depths."""
        specifics = self._metadata.get("device_specifics", {})
        return specifics.get("modality") in AOD_ROI_MODALITIES

    @property
    def slider_dim_labels(self) -> tuple[str, ...]:
        nt, nc, nz, _, _ = self._shape5d()
        labels = []
        if nt > 1:
            labels.append("Timepoint")
        if nc > 1:
            labels.append("Channel")
        if nz > 1:
            labels.append("ROI" if self.z_is_roi else "Z-plane")
        return tuple(labels)

    def __len__(self) -> int:
        return self.nt

    def __getitem__(self, key):
        return _index_5d_into_labeled(self._d, key, self._raw_dims)

    def __array__(self, dtype=None, copy=None):
        data = np.asarray(self[0, 0, 0])
        return data if dtype is None else data.astype(dtype)

    @property
    def reader_kwargs(self) -> dict:
        return {}

    def close(self):
        self._f.close()

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            **kwargs,
        )


register_array_class(Lab4H5Array)
