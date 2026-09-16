"""masknmf demixing results (a ``DemixingResults`` hdf5) as a lazy 5D array.

The file holds factors, not pixels: the PMD movie ``u v``, the demixed
signals ``a c`` and the background terms. The C axis picks which
reconstruction to render (``VIEWS``); masknmf rebuilds frames on read.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse

from mbo_utilities import log
from mbo_utilities.arrays._base import ReductionMixin, _normalize_key
from mbo_utilities.lazy_array import LazyArray
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.demixing")

GROUP = "DemixingResults"
VIEWS = ("pmd", "demixed", "residual")
VIEW_LABELS = {
    "pmd": "PMD movie (u v): the compressed, motion-corrected data",
    "demixed": "demixed signals (a c): every ROI's footprint times its trace",
    "residual": "residual: PMD minus demixed signals and background",
}
CHANNELS = ("calcium", "glutamate")

_DEMIXING_INFO = PipelineInfo(
    name="demixing",
    description="masknmf demixing results",
    input_patterns=["**/*demixing*.hdf5", "**/*demixing*.h5"],
    output_patterns=[],
    input_extensions=["hdf5", "h5"],
    output_extensions=[],
    marker_files=[],
    category="reader",
)
register_pipeline(_DEMIXING_INFO)


def has_demixing_results(path: Path | str) -> bool:
    """Whether ``path`` is an hdf5 file with a ``DemixingResults`` group."""
    p = Path(path)
    if not (p.is_file() and p.suffix.lower() in (".h5", ".hdf5")):
        return False
    try:
        with h5py.File(p, "r") as f:
            return GROUP in f
    except OSError:
        return False


def _describe(path: Path, label: str) -> dict:
    stem = path.stem.lower()
    with h5py.File(path, "r") as f:
        g = f[GROUP]
        shape = tuple(int(x) for x in g["shape"][()])
        a = g["a"]
        num_rois = int(a["size"][1]) if isinstance(a, h5py.Group) else int(a.shape[1])
    return {
        "path": path,
        "label": label,
        "channel": next((c for c in CHANNELS if c in stem), None),
        "shape": shape,
        "num_rois": num_rois,
    }


def list_demixing_results(path: Path | str) -> list[dict]:
    """The demixing result files that belong with ``path``.

    Every ``DemixingResults`` hdf5 in the same folder (a glutamate/calcium
    run writes one per channel and pass), plus the same-named file in
    sibling folders (one ``demixing_results.hdf5`` per ``zplaneNN``). Each
    entry has ``path``, ``label``, ``channel`` (``"calcium"``,
    ``"glutamate"`` or None), ``shape`` (T, Y, X) and ``num_rois``.
    """
    p = Path(path)
    folder = p.parent
    entries = []
    for f in sorted(folder.iterdir()):
        if has_demixing_results(f):
            label = f.stem.removesuffix("_demixing")
            if label == "demixing_results":
                label = folder.name
            entries.append(_describe(f, label))
    if folder.parent != folder:
        here = folder.resolve()
        for sib in sorted(folder.parent.iterdir()):
            # a link back to this folder (pytest's `current` junction) is not a sibling
            if sib.is_dir() and sib.resolve() != here and has_demixing_results(sib / p.name):
                entries.append(_describe(sib / p.name, sib.name))
    return entries


class DemixingArray(ReductionMixin, LazyArray):
    """One masknmf ``DemixingResults`` file as ``(T, 3, 1, Y, X)``.

    C is the view: 0 the PMD movie, 1 the demixed signals, 2 the residual
    (``VIEWS``). ROI footprints and traces are read straight from the file;
    pixels are rebuilt by masknmf on first read, on the CUDA device when
    there is one.
    """

    PRIORITY = 60

    def __init__(self, filenames: Path | str, device: str | None = None):
        path = Path(filenames)
        self.filenames = [path]
        self._device = device
        self._results = None
        self._traces = None
        self._footprints = None
        with h5py.File(path, "r") as f:
            g = f[GROUP]
            self._shape3 = tuple(int(x) for x in g["shape"][()])
            a = g["a"]
            self.num_rois = int(a["size"][1]) if isinstance(a, h5py.Group) else int(a.shape[1])
            self.label_names = [
                s.decode() if isinstance(s, bytes) else str(s) for s in g["label_names"][()]
            ] if "label_names" in g else []
            self.class_labels = (
                np.asarray(g["class_labels"][()], dtype=int)
                if "class_labels" in g
                else np.zeros(self.num_rois, dtype=int)
            )
            self.iscell = (
                np.asarray(g["iscell"][()], dtype=bool)
                if "iscell" in g
                else np.ones(self.num_rois, dtype=bool)
            )
            self._mean_img = np.asarray(g["mean_img"][()], dtype=np.float32).reshape(self._shape3[1:])
            prov = json.loads(f.attrs["mbo_provenance"]) if "mbo_provenance" in f.attrs else {}
        t, y, x = self._shape3
        self._metadata = {
            "num_timepoints": t,
            "num_color_channels": len(VIEWS),
            "num_zplanes": 1,
            "Ly": y,
            "Lx": x,
            "dtype": "float32",
            "demixing_views": VIEWS,
            "demixing_num_rois": self.num_rois,
            "demixing_label_names": tuple(self.label_names),
            "demixing_provenance": prov,
        }
        if prov.get("fs"):
            self._metadata["fs"] = float(prov["fs"])

    @classmethod
    def can_open(cls, path: Path | str) -> bool:
        return has_demixing_results(path)

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        t, y, x = self._shape3
        return (t, len(VIEWS), 1, y, x)

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def slider_dim_labels(self) -> tuple[str, ...]:
        labels = ["Timepoint"] if self._shape3[0] > 1 else []
        labels.append("View")
        return tuple(labels)

    @property
    def reader_kwargs(self) -> dict:
        return {}

    @property
    def roi_labels(self) -> list[str]:
        """One class name per ROI (``"-"`` when the file carries no labels)."""
        names = self.label_names
        return [names[k] if 0 <= k < len(names) else "-" for k in self.class_labels]

    @property
    def traces(self) -> np.ndarray:
        """The demixed temporal components ``c`` as ``(T, num_rois)``."""
        if self._traces is None:
            with h5py.File(self.filenames[0], "r") as f:
                self._traces = np.asarray(f[GROUP]["c"][()], dtype=np.float32)
        return self._traces

    @property
    def footprints(self) -> scipy.sparse.csc_matrix:
        """The spatial footprints ``a`` as a ``(Y * X, num_rois)`` sparse matrix."""
        if self._footprints is None:
            with h5py.File(self.filenames[0], "r") as f:
                a = f[GROUP]["a"]
                if isinstance(a, h5py.Group):
                    idx = a["indices"][()]
                    self._footprints = scipy.sparse.csc_matrix(
                        (a["values"][()], (idx[0], idx[1])),
                        shape=tuple(int(s) for s in a["size"][()]),
                    )
                else:
                    self._footprints = scipy.sparse.csc_matrix(a[()])
        return self._footprints

    def footprint(self, k: int) -> np.ndarray:
        """ROI ``k``'s footprint as a dense ``(Y, X)`` image."""
        _, y, x = self._shape3
        return np.asarray(self.footprints[:, k].todense(), dtype=np.float32).reshape(y, x)

    def _view(self, c: int):
        if self._results is None:
            import torch
            from masknmf import DemixingResults

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"loading {self.filenames[0].name} on {device}")
            self._results = DemixingResults.from_hdf5(self.filenames[0], device=device)
        res = self._results
        return (res.pmd_array, res.ac_array, res.residual_array)[c]

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = key + (slice(None),) * (5 - len(key))
        t_key, c_key, z_key, y_key, x_key = key
        nt, nc, _, ny, nx = self._shape5d()
        ts = np.atleast_1d(np.arange(nt)[t_key]).tolist()
        cs = np.atleast_1d(np.arange(nc)[c_key]).tolist()
        stack = np.stack(
            [np.asarray(self._view(c)[ts], dtype=np.float32).reshape(len(ts), ny, nx) for c in cs],
            axis=1,
        )
        out = stack[:, :, None][:, :, z_key, y_key, x_key]
        if isinstance(c_key, (int, np.integer)):
            out = out[:, 0]
        if isinstance(t_key, (int, np.integer)):
            out = out[0]
        return out

    def __array__(self, dtype=None, copy=None):
        # the stored mean image stands in for the representative frame, so
        # `mbo info` and the histogram never have to rebuild pixels
        return self._mean_img if dtype is None else self._mean_img.astype(dtype)

    def close(self) -> None:
        self._results = None
