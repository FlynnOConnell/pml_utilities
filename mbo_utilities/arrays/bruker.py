"""Bruker two-photon recordings exported to HDF5."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import DIMS, ReductionMixin, _imwrite_base
from mbo_utilities.arrays.h5 import _index_5d_into_labeled, _iter_h5_datasets
from mbo_utilities.lazy_array import LazyArray, register_array_class
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.bruker")

_BRUKER_INFO = PipelineInfo(
    name="bruker",
    description="Bruker two-photon recordings exported to HDF5",
    input_patterns=["**/*.h5", "**/*.hdf5"],
    output_patterns=[],
    input_extensions=["h5", "hdf5"],
    output_extensions=[],
    marker_files=[],
    category="reader",
)
register_pipeline(_BRUKER_INFO)


def _plain(value):
    """An h5 attribute as a python value: bytes decoded, arrays as lists."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _bruker_datasets(f: h5py.File) -> list[str]:
    """Datasets in an open file whose ``imaging_system`` attr says bruker."""
    return [
        d["key"]
        for d in _iter_h5_datasets(f)
        if str(_plain(f[d["key"]].attrs.get("imaging_system", ""))).lower() == "bruker"
    ]


class BrukerArray(ReductionMixin, LazyArray):
    """One Bruker recording in an HDF5 file as (T, C, Z, Y, X).

    The dataset carries ``imaging_system = "bruker"`` and names its axes with
    HDF5 dimension labels (``t z y x c`` in the files seen so far); those labels,
    not the dataset's rank, decide which axis is which. ``element_size_um``
    lists the pixel size of the spatial axes in their stored order, and
    ``frame_period`` is the time between samples on T.
    """

    PRIORITY = 60

    def __init__(self, filenames: Path | str, dataset: str | None = None):
        self.filenames = [Path(filenames)]
        path = self.filenames[0]
        self._f = h5py.File(path, "r")
        try:
            if dataset is None:
                found = _bruker_datasets(self._f)
                if not found:
                    raise ValueError(f"no dataset marked imaging_system=bruker in {path}")
                dataset = found[0]
            self._d = self._f[dataset]
            self.dataset_name = dataset
            labels = tuple(self._d.dims[i].label.upper() for i in range(self._d.ndim))
            if (
                any(label not in DIMS for label in labels)
                or len(set(labels)) != len(labels)
                or not {"Y", "X"} <= set(labels)
            ):
                raise ValueError(
                    f"dataset '{dataset}' in {path} has dimension labels "
                    f"{labels}; expected distinct letters from TCZYX including Y and X"
                )
            self._raw_dims = labels
            self._metadata = self._build_metadata()
        except Exception:
            self._f.close()
            raise

    def _build_metadata(self) -> dict:
        md = {k: _plain(v) for k, v in self._f.attrs.items()}
        md.update(
            {k: _plain(v) for k, v in self._d.attrs.items() if k != "DIMENSION_LABELS"}
        )
        nt, nc, nz, ny, nx = self._shape5d()
        spatial = [d for d in self._raw_dims if d in "ZYX"]
        sizes = md.get("element_size_um")
        if isinstance(sizes, list) and len(sizes) == len(spatial):
            size_of = dict(zip(spatial, sizes))
            md["dx"] = float(size_of["X"])
            md["dy"] = float(size_of["Y"])
            # a single plane has no z-step; its z entry is a placeholder
            if nz > 1:
                md["dz"] = float(size_of["Z"])
        elif sizes is not None:
            logger.warning(
                "element_size_um %s does not match spatial axes %s in %s; "
                "pixel size left unset",
                sizes,
                "".join(spatial),
                self.filenames[0].name,
            )
        md.update(
            {
                "num_timepoints": nt,
                "num_color_channels": nc,
                "num_zplanes": nz,
                "Ly": ny,
                "Lx": nx,
                "dtype": self._d.dtype.name,
                "h5_dataset": self.dataset_name,
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
                return bool(_bruker_datasets(f))
        except OSError:
            return False

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        sizes = dict(zip(self._raw_dims, self._d.shape))
        return tuple(sizes.get(d, 1) for d in DIMS)

    @property
    def dtype(self):
        return self._d.dtype

    def __len__(self) -> int:
        return self.nt

    def __getitem__(self, key):
        return _index_5d_into_labeled(self._d, key, self._raw_dims)

    def __array__(self, dtype=None, copy=None):
        data = np.asarray(self[0, 0, 0])
        return data if dtype is None else data.astype(dtype)

    @property
    def reader_kwargs(self) -> dict:
        return {"dataset": self.dataset_name}

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


register_array_class(BrukerArray)
