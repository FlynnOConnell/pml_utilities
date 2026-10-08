"""A masknmf compressed movie (a ``CompressionArray`` hdf5) as a lazy 5D array.

The file holds the movie as factors, ``spatial_compressed temporal_compressed``
plus the per-pixel mean and noise scale it was standardized by. Frames are
rebuilt on read with numpy, in the units of the movie that was compressed.
The MaskNMF pipeline's ``roi_workflow`` caches write ``compression.hdf5``;
a run folder's ``results.hdf5`` holds the same group and, when registration
ran on a denoised copy, ``alignment.hdf5`` holds that copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import ReductionMixin, _normalize_key
from mbo_utilities.arrays.demixing import _read_sparse
from mbo_utilities.lazy_array import LazyArray
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.compression")

GROUP = "CompressionArray"

register_pipeline(
    PipelineInfo(
        name="compression",
        description="masknmf compressed movie",
        input_patterns=["**/compression.hdf5", "**/alignment.hdf5"],
        output_patterns=[],
        input_extensions=["hdf5", "h5"],
        output_extensions=[],
        marker_files=[],
        category="reader",
    )
)


def has_compressed_movie(path: Path | str) -> bool:
    """Whether ``path`` is an hdf5 file with a whole ``CompressionArray`` group."""
    p = Path(path)
    if not (p.is_file() and p.suffix.lower() in (".h5", ".hdf5")):
        return False
    try:
        with h5py.File(p, "r") as f:
            return f"{GROUP}/shape" in f and f"{GROUP}/temporal_compressed" in f
    except OSError:
        return False


class CompressedMovieArray(ReductionMixin, LazyArray):
    """One masknmf ``CompressionArray`` file as ``(T, 1, 1, Y, X)`` float32."""

    PRIORITY = 60

    def __init__(self, filenames: Path | str):
        path = Path(filenames)
        self.filenames = [path]
        self._factors = None
        with h5py.File(path, "r") as f:
            g = f[GROUP]
            self._shape3 = tuple(int(x) for x in g["shape"][()])
            self._mean_img = np.asarray(g["mean_image"][()], dtype=np.float32).reshape(
                self._shape3[1:]
            )
            prov = (
                json.loads(f.attrs["mbo_provenance"])
                if "mbo_provenance" in f.attrs
                else {}
            )
        t, y, x = self._shape3
        self._metadata = {
            "num_timepoints": t,
            "num_color_channels": 1,
            "num_zplanes": 1,
            "Ly": y,
            "Lx": x,
            "dtype": "float32",
            "compression_provenance": prov,
        }
        if prov.get("fs"):
            self._metadata["fs"] = float(prov["fs"])

    @classmethod
    def can_open(cls, path: Path | str) -> bool:
        return has_compressed_movie(path)

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        t, y, x = self._shape3
        return (t, 1, 1, y, x)

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def reader_kwargs(self) -> dict:
        return {}

    def _frames(self, ts: list[int]) -> np.ndarray:
        """Frames ``ts`` as ``(len(ts), Y * X)``."""
        if self._factors is None:
            logger.info(f"loading {self.filenames[0].name} factors")
            with h5py.File(self.filenames[0], "r") as f:
                g = f[GROUP]
                self._factors = (
                    _read_sparse(g["spatial_compressed"]).tocsr(),
                    np.asarray(g["temporal_compressed"][()], dtype=np.float32),
                    np.asarray(g["noise_variance_image"][()], np.float32).reshape(-1),
                    self._mean_img.reshape(-1),
                )
        u, v, scale, mean = self._factors
        return np.asarray((u @ v[:, ts]).T, dtype=np.float32) * scale + mean

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = key + (slice(None),) * (5 - len(key))
        t_key, c_key, z_key, y_key, x_key = key
        nt, _, _, ny, nx = self._shape5d()
        ts = np.atleast_1d(np.arange(nt)[t_key]).tolist()
        out = self._frames(ts).reshape(len(ts), 1, 1, ny, nx)
        out = out[:, c_key, z_key, y_key, x_key]
        if isinstance(t_key, (int, np.integer)):
            out = out[0]
        return out

    def __array__(self, dtype=None, copy=None):
        # the stored mean image stands in for the representative frame
        return self._mean_img if dtype is None else self._mean_img.astype(dtype)

    def close(self) -> None:
        self._factors = None
