"""``H5Array``: one HDF5 dataset as a lazy (T, C, Z, Y, X) array."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import (
    DIMS,
    ReductionMixin,
    Shape5DMixin,
    _imwrite_base,
    _index_5d_into_raw,
    _normalize_key,
)
from mbo_utilities.arrays.numpy import _canonicalize_to_5d
from mbo_utilities.lazy_array import register_array_class
from mbo_utilities.metadata import get_param
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.h5")

# dataset names probed (in order) when the caller doesn't pick one
_PREFERRED_KEYS = ("mov", "data", "scan_corrections", "imaging/data", "raw")

# the rank guess (AGENTS §5.2); a dataset labeled this way reads through `_index_5d_into_raw`'s front padding
_DEFAULT_RAW_DIMS = {
    1: ("X",),
    2: ("Y", "X"),
    3: ("T", "Y", "X"),
    4: ("T", "Z", "Y", "X"),
    5: ("T", "C", "Z", "Y", "X"),
}


def _iter_h5_datasets(f) -> list[dict]:
    """Describe every dataset in an open HDF5 file, groups walked recursively."""
    names: list[str] = []
    f.visit(names.append)
    objs = [f[name] for name in names]
    # a null dataspace (h5py.Empty) has shape None and nothing to read
    return [
        {
            "key": name,
            "shape": tuple(obj.shape),
            "dtype": obj.dtype,
            "ndim": obj.ndim,
            "nbytes": int(obj.nbytes),
        }
        for name, obj in zip(names, objs, strict=True)
        if isinstance(obj, h5py.Dataset) and obj.shape is not None
    ]


def _index_5d_into_labeled(data, key, raw_dims: tuple[str, ...]) -> np.ndarray:
    """Index a dataset whose axes are labeled ``raw_dims`` with a 5D TCZYX key.

    Maps the key onto the labeled raw axes, reads, permutes the kept axes back
    to TCZYX, then drops the axes that were integer-indexed so the result keeps
    numpy 5D semantics.
    """
    key = _normalize_key(key, 5)
    if len(key) > 5:
        raise IndexError(
            f"too many indices for array: array is 5-dimensional, "
            f"but {len(key)} were indexed"
        )
    key = key + (slice(None),) * (5 - len(key))
    key_by_dim = dict(zip(DIMS, key))
    raw_key = tuple(key_by_dim.get(d, slice(None)) for d in raw_dims)
    out = np.asarray(data[raw_key])
    kept = tuple(
        d for d in raw_dims if not isinstance(key_by_dim[d], (int, np.integer))
    )
    out = _canonicalize_to_5d(out, kept)
    for axis in reversed(range(5)):
        if isinstance(key[axis], (int, np.integer)):
            out = np.squeeze(out, axis=axis)
    return out


def list_h5_datasets(path: Path | str) -> list[dict]:
    """One dict per dataset in an HDF5 file, nested groups included, read from headers only.

    Keys: ``key`` (full path, ``imaging/data``), ``shape``, ``dtype``, ``ndim``,
    ``nbytes``; in ``visit`` order.
    """
    with h5py.File(Path(path), "r") as f:
        return _iter_h5_datasets(f)


_H5_INFO = PipelineInfo(
    name="hdf5",
    description="HDF5 datasets",
    input_patterns=[
        "**/*.h5",
        "**/*.hdf5",
        "**/*.hdf",
    ],
    output_patterns=[
        "**/*.h5",
        "**/*.hdf5",
    ],
    input_extensions=["h5", "hdf5", "hdf"],
    output_extensions=["h5", "hdf5"],
    marker_files=[],
    category="reader",
)
register_pipeline(_H5_INFO)


class H5Array(ReductionMixin, Shape5DMixin):
    """One HDF5 dataset as (T, C, Z, Y, X).

    ``dataset`` names it, nested paths accepted (``imaging/data``); without
    one, the first of ``mov``, ``data``, ``scan_corrections``, ``imaging/data``,
    ``raw`` that exists opens, else the largest >=3D dataset anywhere in the
    file. ``list_h5_datasets`` lists the choices without reading pixels.
    """

    def __init__(self, filenames: Path | str, dataset: str | None = None):
        # a list like every reader's: lbm_suite2p_python.run_plane reads filenames[0]
        self.filenames = [Path(filenames)]
        path = self.filenames[0]
        self._f = h5py.File(path, "r")
        # the file is read-only; metadata writes (imwrite stamping the source) shadow its attrs here
        self._metadata_overlay: dict = {}

        try:
            self._datasets = _iter_h5_datasets(self._f)

            if dataset is None:
                dataset = self._autodetect_dataset(path)

            try:
                d = self._f[dataset]
            except KeyError:
                d = None
            if not isinstance(d, h5py.Dataset):
                available = [entry["key"] for entry in self._datasets]
                raise KeyError(
                    f"Dataset '{dataset}' not found in {path}. "
                    f"Available datasets: {available}"
                ) from None

            self._d = d
            self.dataset_name = dataset
            self._raw_shape = tuple(self._d.shape)
            self._raw_dims = self._resolve_raw_dims()
            self._dtype = self._d.dtype
            self._target_dtype = None
        except Exception:
            self._f.close()
            raise

    def _autodetect_dataset(self, path: Path) -> str:
        """Pick a dataset: preferred names first, else the largest >=3D one."""
        for key in _PREFERRED_KEYS:
            obj = self._f.get(key)
            if isinstance(obj, h5py.Dataset):
                if key == "scan_corrections":
                    logger.info(f"Detected pollen calibration file: {path.name}")
                return key

        if not self._datasets:
            raise ValueError(f"No datasets found in {path}")

        candidates = [d for d in self._datasets if d["ndim"] >= 3] or self._datasets
        chosen = min(candidates, key=lambda d: (-d["nbytes"], d["key"]))
        available = [d["key"] for d in self._datasets]
        logger.warning(
            f"Auto-selected dataset '{chosen['key']}' (shape {chosen['shape']}) "
            f"in {path.name}. Available: {available}. "
            f"Pass dataset=<name> to open a different one."
        )
        return chosen["key"]

    def _resolve_raw_dims(self) -> tuple[str, ...]:
        """Label the raw axes so _shape5d/__getitem__ can map onto TCZYX."""
        ndim = len(self._raw_shape)
        if ndim > 5:
            raise ValueError(
                f"dataset '{self.dataset_name}' has {ndim} dimensions; "
                f"H5Array supports up to 5"
            )
        if self.dataset_name.lstrip("/") == "imaging/data" and ndim == 5:
            # MINI2P h5 converter layout
            return ("T", "Z", "Y", "X", "C")
        if ndim == 4:
            attrs = self._d.attrs
            n_channel = attrs.get("n_channel")
            if attrs.get("scan_mode") is not None and n_channel is not None:
                try:
                    if int(n_channel) == self._raw_shape[-1]:
                        return ("T", "Y", "X", "C")
                except (TypeError, ValueError):
                    pass
        return _DEFAULT_RAW_DIMS.get(ndim, ())

    PRIORITY = 50

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        p = Path(file)
        if p.name.endswith("_pollen.h5"):
            return False  # pollen calibration output, not source data
        return p.is_file() and p.suffix.lower() in (".h5", ".hdf5", ".hdf")

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        sizes = dict(zip(self._raw_dims, self._raw_shape))
        return tuple(sizes.get(d, 1) for d in DIMS)

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    @property
    def num_planes(self) -> int:
        """Number of Z-planes in the dataset (index 2 in 5D TCZYX)."""
        nplanes = get_param(self.metadata, "nplanes")
        if nplanes is not None:
            return int(nplanes)
        if self.dataset_name == "scan_corrections" and len(self._d.shape) == 1:
            return int(self._d.shape[0])

        return self._shape5d()[2]

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        raw_ndim = len(self._raw_shape)
        if self._raw_dims == _DEFAULT_RAW_DIMS.get(raw_ndim):
            out = _index_5d_into_raw(self._d, key, raw_ndim)
        else:
            out = _index_5d_into_labeled(self._d, key, self._raw_dims)
        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def __array__(self, dtype=None, copy=None):
        # one representative (Y, X) frame, never the whole dataset (AGENTS §5.3)
        data = np.asarray(self[0, 0, 0])
        if self._target_dtype is not None:
            data = data.astype(self._target_dtype)
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def close(self):
        """Close the HDF5 file."""
        self._f.close()

    @property
    def reader_kwargs(self) -> dict:
        """Kwargs `imread` needs to re-open this exact dataset in another process."""
        return {"dataset": self.dataset_name}

    @property
    def metadata(self) -> dict:
        """On-disk attributes merged with any in-memory overrides. Always a dict.

        File attrs first, then each parent group's attrs along the dataset
        path (root-first), then the dataset's own attrs, so the most specific
        source wins. Sibling groups are not consulted.
        """
        md = dict(self._f.attrs)
        parts = [p for p in self.dataset_name.split("/") if p]
        for i in range(1, len(parts)):
            md.update(self._f["/".join(parts[:i])].attrs)
        md.update(self._d.attrs)
        md["h5_dataset"] = self.dataset_name
        md["h5_datasets"] = [
            {"key": d["key"], "shape": d["shape"], "dtype": str(d["dtype"])}
            for d in self._datasets
        ]
        md["h5_raw_dims"] = "".join(self._raw_dims)
        md.update(self._metadata_overlay)
        return md

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata_overlay.update(value)

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
        """Write H5Array to disk in various formats."""
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


register_array_class(H5Array, priority=50)
