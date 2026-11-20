"""TIFF readers: raw ScanImage tiffs (LBM, piezo, pollen, single plane) and ``TiffArray`` for every other tiff."""

from __future__ import annotations

import copy
import json
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tifffile import TiffFile

from mbo_utilities import log
from mbo_utilities.analysis.phasecorr import _apply_offset, bidir_phasecorr
from mbo_utilities.arrays._base import (
    ReductionMixin,
    Shape5DMixin,
    TiffReaderMixin,
    _normalize_key,
)
from mbo_utilities.arrays.features import (
    PhaseCorrectionFeature,
    PhaseCorrectionMixin,
    RoiFeatureMixin,
)
from mbo_utilities.arrays.features._slicing import index_length, listify_index
from mbo_utilities.file_io import expand_paths
from mbo_utilities.lazy_array import LazyArray, register_array_class
from mbo_utilities.metadata import (
    METADATA_PARAMS,
    extract_roi_slices,
    get_metadata,
    get_metadata_single,
    get_param,
    is_raw_scanimage,
)
from mbo_utilities.metadata.scanimage import (
    StackType,
    detect_stack_type,
    get_log_average_factor,
    layout_metadata,
    scan_layout,
)
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = log.get("arrays.tiff")

_TIFF_INFO = PipelineInfo(
    name="tiff",
    description="Generic TIFF files (BigTIFF, OME-TIFF)",
    input_patterns=[
        "**/*.tif",
        "**/*.tiff",
    ],
    output_patterns=[
        "**/*.tif",
        "**/*.tiff",
    ],
    input_extensions=["tif", "tiff"],
    output_extensions=["tif", "tiff"],
    marker_files=[],
    category="reader",
)
register_pipeline(_TIFF_INFO)

_SCANIMAGE_INFO = PipelineInfo(
    name="ScanImage TIFF",
    description="Raw ScanImage TIFF files with multi-ROI support",
    input_patterns=[
        "**/*.tif",
        "**/*.tiff",
    ],
    output_patterns=[],
    input_extensions=["tif", "tiff"],
    output_extensions=[],
    marker_files=[],
    category="reader",
)
register_pipeline(_SCANIMAGE_INFO)


def _convert_range_to_slice(k):
    if isinstance(k, range):
        return slice(k.start, k.stop, k.step)
    return k


def _is_full_slice(k) -> bool:
    """True for ``slice(None)``; safe for array keys, where ``k != slice(None)``
    would return an array.
    """
    return isinstance(k, slice) and k == slice(None)


def _extract_tiff_plane_number(name: str) -> int | None:
    """The N of ``planeN`` anywhere in ``name`` (``plane01.tiff``, ``plane14_stitched.tif``)."""
    match = re.search(r"plane(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _read_tiff_metadata(path: Path) -> dict:
    """The file's metadata from tag 50839, the ImageJ Info, shaped metadata or the page description; empty when none parses."""
    from mbo_utilities.metadata.io import get_metadata_single

    try:
        return get_metadata_single(path)
    except Exception:
        return {}


def find_tiff_plane_files(directory: Path) -> list[Path]:
    """The ``planeNN`` tiffs in ``directory``, sorted by plane number."""
    plane_files = [
        f
        for f in directory.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".tif", ".tiff")
        and _extract_tiff_plane_number(f.stem) is not None
    ]
    return sorted(plane_files, key=lambda p: _extract_tiff_plane_number(p.stem))


def _group_plane_files(files: list[Path]) -> list[list[Path]]:
    """Group ``planeNN``-named files into one sorted list per plane number."""
    groups: dict[int, list[Path]] = {}
    for p in files:
        pnum = _extract_tiff_plane_number(p.name)
        if pnum is not None:
            groups.setdefault(pnum, []).append(p)
    return [sorted(groups[k]) for k in sorted(groups)]


def _imagej_layout(meta: dict, path: Path) -> tuple[int, ...] | None:
    """(frames, slices, channels) when a single file is a multi-frame ImageJ
    hyperstack — timeseries, z-stack, or both — else None.

    Prefers the counts already deposited by ``get_metadata_single``; reads them
    straight from the file only when caller-supplied metadata omits them.

    ImageJ writes ``slices=N`` for *any* plain stack, including a movie saved
    without going through Stack -> Hyperstack. Only a real hyperstack carries
    ``hyperstack=true`` (and a z-stack normally a ``spacing``), so a plain
    single-channel stack with no ``frames`` count is read as a time series -
    the page axis is what every downstream step (traces, binning, pipelines)
    needs on T. ``imread(path, dims="ZYX")`` forces the z reading.
    """
    counts = tuple(meta.get(k) for k in ("frames", "slices", "channels"))
    hyperstack = meta.get("hyperstack")
    spacing = meta.get("spacing")
    if any(c is not None for c in counts):
        frames, slices, channels = (int(c or 1) for c in counts)
    else:
        from mbo_utilities.metadata.io import imagej_hyperstack_counts

        with TiffFile(path) as tf:
            if not tf.is_imagej:
                return None
            frames, slices, channels = imagej_hyperstack_counts(tf)
            ij = tf.imagej_metadata or {}
            hyperstack = ij.get("hyperstack")
            spacing = ij.get("spacing")
    if frames * slices * channels <= 1:
        return None
    if frames == 1 and channels == 1 and slices > 1 and not hyperstack and not spacing:
        logger.info(
            f"ImageJ stack with slices={slices} and no hyperstack/spacing tags: "
            "reading the page axis as time. Pass dims='ZYX' to keep it as z."
        )
        frames, slices = slices, 1
    return (frames, slices, channels)


def _apply_dims_override(
    layout: tuple[int, ...], dims: str | Sequence[str] | None
) -> tuple[int, ...]:
    """Move a single-file page axis to the one the caller named.

    A layout whose whole non-spatial extent sits on one of T, Z or C and a
    three-letter ``dims`` (``"TYX"``, ``"ZYX"``, ``"CYX"``) put that extent
    on the named axis. A file with two populated axes, or a ``dims`` naming
    more than one, is left alone.
    """
    if not dims:
        return layout
    letters = "".join(dims).upper()
    populated = [n for n in layout if n > 1]
    if len(letters) != 3 or letters[0] not in "TZC" or len(populated) != 1:
        return layout
    n = populated[0]
    return {"T": (n, 1, 1), "Z": (1, n, 1), "C": (1, 1, n)}[letters[0]]


def _ome_layout(meta: dict) -> tuple[int, int, int] | None:
    """(frames, slices, channels) from OME-XML SizeT/SizeZ/SizeC counts
    deposited by ``get_metadata_single``, else None.
    """
    counts = tuple(meta.get(k) for k in ("SizeT", "SizeZ", "SizeC"))
    if all(c is None for c in counts):
        return None
    frames, slices, channels = (int(c or 1) for c in counts)
    return (frames, slices, channels) if frames * slices * channels > 1 else None


def _shape_layout(meta: dict) -> tuple[int, ...] | None:
    """(frames, planes[, channels]) from a TZCYX ``shape`` for shaped/legacy
    metadata that predates the ImageJ count keys, else None.
    """
    shape = meta.get("shape")
    if shape and len(shape) == 5 and shape[1] > 1 and shape[2] > 1:
        return shape[0], shape[1], shape[2]
    if shape and len(shape) == 4 and shape[1] > 1:
        return shape[0], shape[1]
    return None


def _declared_layout(
    path: Path, dims: str | Sequence[str] | None
) -> tuple[int, int, int] | None:
    """``(frames, slices, channels)`` for a plain single file whose page axis
    the caller named: ``dims="ZYX"`` puts the pages on Z, ``"CYX"`` on C.
    None for ``"TYX"`` or no ``dims``, which leaves them on T, and for a
    one-page file.
    """
    if not dims:
        return None
    letters = "".join(dims).upper()
    if len(letters) != 3 or letters[0] not in "ZC":
        return None
    with TiffFile(path) as tf:
        pages = len(tf.pages)
    if pages <= 1:
        return None
    return (1, pages, 1) if letters[0] == "Z" else (1, 1, pages)


class _InterleavedTiffReader:
    """Internal reader for ImageJ-style interleaved TZCYX hyperstacks."""

    def __init__(
        self,
        path: Path,
        n_frames: int,
        n_planes: int,
        n_channels: int = 1,
        dim_order: str = "XYCZT",
    ):
        self._path = path
        self._tiff = TiffFile(path)
        self._lock = threading.Lock()
        self._n_frames = n_frames
        self._n_planes = n_planes
        self._n_channels = n_channels

        # OME DimensionOrder: after XY, the first letter varies fastest across pages
        order = (dim_order or "XYCZT").upper()[2:]
        if set(order) != {"C", "Z", "T"}:
            order = "CZT"
        sizes = {"T": n_frames, "Z": n_planes, "C": n_channels}
        self._page_strides = {}
        stride = 1
        for d in order:
            self._page_strides[d] = stride
            stride *= sizes[d]

        page0 = self._tiff.pages.first
        self._page_shape = page0.shape
        self._dtype = page0.dtype
        self._zstore = None  # the series' zarr store once a contiguous hyperstack is detected (_frame)

    @property
    def nframes(self) -> int:
        return self._n_frames

    @property
    def n_planes(self) -> int:
        return self._n_planes

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def Ly(self) -> int:
        return self._page_shape[0]

    @property
    def Lx(self) -> int:
        return self._page_shape[1]

    @property
    def dtype(self):
        from mbo_utilities.arrays._base import get_dtype

        return get_dtype(self._dtype)

    def _frame(self, flat_idx: int) -> np.ndarray:
        """One (Y, X) frame by flat page index.

        A contiguous ImageJ hyperstack keeps every frame in one physical IFD
        page, so ``pages[i]`` raises IndexError; from then on frames come
        from the series' zarr store, one chunk each.
        """
        if self._zstore is None:
            try:
                return self._tiff.pages[flat_idx].asarray()
            except IndexError:
                import zarr

                self._zstore = zarr.open(self._tiff.series[0].aszarr(), mode="r")
        lead = self._zstore.shape[:-2]
        total = int(np.prod(lead)) if lead else 1
        if flat_idx >= total:
            # a corrupt file collapsed to one plane would otherwise answer frame 0 for every index
            raise IndexError(
                f"frame {flat_idx} out of range ({total} frames) in {self._path}"
            )
        if not lead:
            return np.asarray(self._zstore)
        midx = tuple(int(i) for i in np.unravel_index(flat_idx, lead))
        return np.asarray(self._zstore[midx])

    def read_tzyx(
        self,
        t_indices: list[int],
        z_indices: list[int],
        c_indices: list[int] | None = None,
    ) -> np.ndarray:
        """(T, C, Z, Y, X) when ``c_indices`` is given on a multi-channel file, else (T, Z, Y, X).

        The page index follows the ``dim_order`` strides; for XYCZT it is
        ``t * Z * C + z * C + c``.
        """
        st, sz, sc = (self._page_strides[d] for d in "TZC")
        if c_indices is not None and self._n_channels > 1:
            buf = np.empty(
                (len(t_indices), len(c_indices), len(z_indices), self.Ly, self.Lx),
                dtype=self._dtype,
            )
            with self._lock:
                for ti, t_idx in enumerate(t_indices):
                    for ci, c_idx in enumerate(c_indices):
                        for zi, z_idx in enumerate(z_indices):
                            page_idx = t_idx * st + z_idx * sz + c_idx * sc
                            buf[ti, ci, zi] = self._frame(page_idx)
            return buf
        buf = np.empty(
            (len(t_indices), len(z_indices), self.Ly, self.Lx), dtype=self._dtype
        )
        with self._lock:
            for ti, t_idx in enumerate(t_indices):
                for zi, z_idx in enumerate(z_indices):
                    buf[ti, zi] = self._frame(t_idx * st + z_idx * sz)
        return buf

    def close(self):
        self._tiff.close()


class _SingleTiffPlaneReader:
    """One plane's (T, Y, X) pages, spread over one or more files."""

    def __init__(self, files: list[Path]):
        from mbo_utilities.metadata import query_tiff_pages

        self.filenames = files
        self.tiff_files = [TiffFile(f) for f in files]
        self._tiff_lock = threading.Lock()

        tf = self.tiff_files[0]
        page0 = tf.pages.first
        self._page_shape = page0.shape
        self._dtype = page0.dtype

        self._frames_per_file = []
        self._num_frames = 0

        for i, (tfile, fpath) in enumerate(
            zip(self.tiff_files, self.filenames, strict=False)
        ):
            nframes = None

            desc = page0.description if i == 0 else tfile.pages.first.description

            if desc:
                try:
                    meta = json.loads(desc)
                    if "shape" in meta and isinstance(meta["shape"], list):
                        if len(meta["shape"]) >= 3:
                            nframes = meta["shape"][0]
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

            # the page estimate divides the file size by one page's stride, which only holds for
            # ScanImage's uniform pages; any other file walks its IFDs
            if nframes is None and tfile.is_scanimage and query_tiff_pages(fpath) > 1:
                nframes = query_tiff_pages(fpath)

            if nframes is None:
                nframes = len(tfile.pages)

            self._frames_per_file.append(nframes)
            self._num_frames += nframes

    @property
    def shape(self) -> tuple:
        return (self._num_frames, self._page_shape[0], self._page_shape[1])

    @property
    def nframes(self) -> int:
        return self._num_frames

    @property
    def Ly(self) -> int:
        return self._page_shape[0]

    @property
    def Lx(self) -> int:
        return self._page_shape[1]

    @property
    def dtype(self):
        from mbo_utilities.arrays._base import get_dtype

        return get_dtype(self._dtype)

    def __getitem__(self, key):
        key = _normalize_key(key, 3)

        t_key = key[0] if len(key) > 0 else slice(None)
        y_key = key[1] if len(key) > 1 else slice(None)
        x_key = key[2] if len(key) > 2 else slice(None)

        t_key = _convert_range_to_slice(t_key)

        frames = listify_index(t_key, self._num_frames)
        if not frames:
            return np.empty((0, *self._page_shape), dtype=self._dtype)

        out = self._read_frames(frames)

        if y_key != slice(None) or x_key != slice(None):
            out = out[:, y_key, x_key]

        if isinstance(t_key, (int, np.integer)):
            out = out[0]

        return out

    def _read_frames(self, frames: list[int]) -> np.ndarray:
        buf = np.empty(
            (len(frames), self._page_shape[0], self._page_shape[1]), dtype=self._dtype
        )

        start = 0
        frame_to_buf_idx = {f: i for i, f in enumerate(frames)}

        for tf, nframes in zip(self.tiff_files, self._frames_per_file, strict=False):
            end = start + nframes
            file_frames = [f for f in frames if start <= f < end]
            if not file_frames:
                start = end
                continue

            local_indices = [f - start for f in file_frames]

            with self._tiff_lock:
                chunk = tf.asarray(key=local_indices)
            if chunk.ndim == 2:
                chunk = chunk[np.newaxis, ...]

            for chunk_idx, (_local_idx, global_frame) in enumerate(
                zip(local_indices, file_frames, strict=False)
            ):
                buf_idx = frame_to_buf_idx[global_frame]
                buf[buf_idx] = chunk[chunk_idx]

            start = end

        return buf

    def close(self):
        for tf in self.tiff_files:
            tf.close()


class TiffArray(TiffReaderMixin, ReductionMixin, Shape5DMixin):
    """Any tiff that is not a raw ScanImage file, as (T, C, Z, Y, X).

    One file reads by its ImageJ, OME or shaped layout, else as pages on T
    (``dims`` names another page axis, §5.2); ``planeNN`` files stack on Z;
    other file lists concatenate on T.
    """

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """True for any ``.tif`` or ``.tiff``; the last-resort tiff reader."""
        if not file:
            return False
        path = Path(file)
        return path.suffix.lower() in (".tif", ".tiff")

    def __init__(
        self,
        files: str | Path | list[str] | list[Path],
        dims: str | Sequence[str] | None = None,
        metadata: dict | None = None,
    ):
        self._planes: list[_SingleTiffPlaneReader] = []
        self._is_volumetric = False
        self._target_dtype = None
        self._interleaved_reader = None
        self._nc = 1
        self._dims_override = dims

        if isinstance(files, (str, Path)):
            path = Path(files)
            if path.is_dir():
                plane_files = find_tiff_plane_files(path)
                if plane_files:
                    file_list = plane_files
                else:
                    file_list = sorted(
                        list(path.glob("*.tif")) + list(path.glob("*.tiff"))
                    )
                if not file_list:
                    raise ValueError(f"No TIFF files found in {path}")
            else:
                file_list = expand_paths(files)
        else:
            file_list = [Path(f) for f in files]

        if not file_list:
            raise ValueError("No TIFF files provided")

        if metadata is not None:
            self._metadata = metadata
        else:
            self._metadata = _read_tiff_metadata(file_list[0])

        # ImageJ counts win over a legacy TZCYX shape; a plain file has only the page axis the caller declared
        if len(file_list) == 1:
            layout = (
                _imagej_layout(self._metadata, file_list[0])
                or _ome_layout(self._metadata)
                or _shape_layout(self._metadata)
                or _declared_layout(file_list[0], dims)
            )
            if layout:
                self._init_interleaved(
                    file_list[0], *_apply_dims_override(layout, dims)
                )
                return
        self._init_from_file_structure(file_list)

    @property
    def reader_kwargs(self) -> dict:
        """The ``dims`` override, when given, so a worker re-opening the path reads the page axis the same way."""
        if not self._dims_override:
            return {}
        dims = self._dims_override
        return {"dims": dims if isinstance(dims, str) else "".join(dims)}

    def _init_volume_from_groups(self, plane_groups: list[list[Path]]):
        """One ``_SingleTiffPlaneReader`` per plane group, stacked on Z."""
        self._is_volumetric = True
        self._planes = [_SingleTiffPlaneReader(files) for files in plane_groups]

        shapes = [(p.Ly, p.Lx) for p in self._planes]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent spatial shapes across planes: {shapes}")

        nframes = [p.nframes for p in self._planes]
        if len(set(nframes)) != 1:
            logger.warning(
                f"Inconsistent frame counts across planes: {nframes}. "
                f"Using minimum: {min(nframes)}"
            )

        self._nframes = min(nframes)
        self._nz = len(self._planes)
        self._ly, self._lx = shapes[0]
        self._dtype = self._planes[0].dtype
        self.filenames = [f for group in plane_groups for f in group]
        self._metadata.update(
            {
                "shape": self.shape,
                "dtype": str(self._dtype),
                "num_timepoints": self._nframes,
                "num_zplanes": self._nz,
                "file_paths": [str(p) for p in self.filenames],
            }
        )
        self.num_rois = 1

    def _init_single_plane(self, files: list[Path]):
        self._is_volumetric = False

        reader = _SingleTiffPlaneReader(files)
        self._planes = [reader]

        self._nframes = reader.nframes
        self._nz = 1
        self._ly = reader.Ly
        self._lx = reader.Lx
        self._dtype = reader.dtype
        self.filenames = files
        self._metadata.update(
            {
                "shape": self.shape,
                "dtype": str(self._dtype),
                "num_timepoints": self._nframes,
                "file_paths": [str(p) for p in files],
                "num_files": len(files),
            }
        )
        self.num_rois = 1

    def _init_from_file_structure(self, files: list[Path]):
        """``planeNN`` file sets stack on Z; anything else concatenates on T."""
        plane_groups = _group_plane_files(files)
        if len(plane_groups) > 1:
            self._init_volume_from_groups(plane_groups)
        else:
            self._init_single_plane(files)

    def _init_interleaved(
        self, path: Path, n_frames: int, n_planes: int, n_channels: int = 1
    ):
        self._is_volumetric = True
        self._interleaved_reader = _InterleavedTiffReader(
            path,
            n_frames,
            n_planes,
            n_channels,
            dim_order=self._metadata.get("DimensionOrder") or "XYCZT",
        )
        self._planes = []
        self._nframes = n_frames
        self._nz = n_planes
        self._nc = n_channels
        self._ly = self._interleaved_reader.Ly
        self._lx = self._interleaved_reader.Lx
        self._dtype = self._interleaved_reader.dtype
        self.filenames = [path]
        self._metadata.update(
            {
                "shape": self.shape,
                "dtype": str(self._dtype),
                "num_timepoints": self._nframes,
                "num_zplanes": self._nz,
                "num_color_channels": self._nc,
                "file_path": str(path),
            }
        )
        self.num_rois = 1
        logger.info(
            f"Loaded interleaved TCZYX: {n_frames} frames, {n_channels} channels, "
            f"{n_planes} planes, {self._ly}x{self._lx} px"
        )

    @property
    def is_volumetric(self) -> bool:
        return self._is_volumetric

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        return (self._nframes, self._nc, self._nz, self._ly, self._lx)

    @property
    def dtype(self):
        from mbo_utilities.arrays._base import get_dtype

        return (
            self._target_dtype
            if self._target_dtype is not None
            else get_dtype(self._dtype)
        )

    def __len__(self) -> int:
        return self._nframes

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = tuple(_convert_range_to_slice(k) for k in key) + (slice(None),) * (
            5 - len(key)
        )
        t_key, c_key, z_key, y_key, x_key = key
        t_indices = listify_index(t_key, self._nframes)
        z_indices = listify_index(z_key, self._nz)
        c_indices = listify_index(c_key, self._nc) if self._nc > 1 else None

        if self._interleaved_reader is not None:
            out = self._interleaved_reader.read_tzyx(t_indices, z_indices, c_indices)
            if self._nc == 1:
                out = out[:, np.newaxis]  # (T, 1, Z, Y, X)
            if not (_is_full_slice(y_key) and _is_full_slice(x_key)):
                out = out[:, :, :, y_key, x_key]
        else:
            arrs = []
            for z_idx in z_indices:
                plane_data = self._planes[z_idx][t_indices, y_key, x_key]
                if plane_data.ndim == 2:
                    plane_data = plane_data[np.newaxis, ...]
                arrs.append(plane_data)
            out = np.stack(arrs, axis=1)[:, np.newaxis]  # (T, 1, Z, Y, X)

        # np.integer too: np.linspace(dtype=int) hands back np.int64 keys
        squeeze_axes = [i for i in range(3) if isinstance(key[i], (int, np.integer))]
        for ax in sorted(squeeze_axes, reverse=True):
            out = np.squeeze(out, axis=ax)

        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def close(self):
        if self._interleaved_reader is not None:
            self._interleaved_reader.close()
        for plane in self._planes:
            plane.close()


class ScanImageArray(
    TiffReaderMixin, RoiFeatureMixin, ReductionMixin, PhaseCorrectionMixin, LazyArray
):
    """A raw ScanImage acquisition as (T, C, Z, Y, X), with mROI stitching or selection and bidirectional scan-phase correction.

    ``metadata.scanimage.ScanLayout`` places every (t, c, z) among the pages:
    T is frames, volumes or piezo slices, C the PMT colors, Z the beamlets or
    slices. ``roi`` None stitches the mROIs side by side, 0 splits them, n picks
    one (1-based), a list picks several. ``average_frames`` averages the repeats
    of each stack slice into one float32 frame instead of keeping them on T.
    ``open_scanimage`` picks the subclass for the stack type.
    """

    STACK_TYPES: tuple[StackType, ...] = ("lbm", "piezo", "pollen", "single_plane")
    METADATA_CONTEXT: dict[str, str] = {  # the gui's tooltips; a subclass overrides per stack type
        "Ly": "Raw TIFF page height in pixels.",
        "Lx": "Raw TIFF page width in pixels.",
        "num_zplanes": "Number of z-planes in the stack.",
        "dz": "Z-step size in micrometers.",
    }
    REQUIRED_METADATA: list[str] = []  # canonical keys the file cannot supply; the user fills them in

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        """True for a raw ScanImage tiff whose stack type is one of ``STACK_TYPES``."""
        if not is_raw_scanimage(file):
            return False
        try:
            return detect_stack_type(get_metadata_single(Path(file))) in cls.STACK_TYPES
        except Exception:
            return False

    def __init__(
        self,
        files: str | Path | list,
        roi: int | Sequence[int] | None = None,
        fix_phase: bool = True,
        phasecorr_method: str = "mean",
        border: int | tuple[int, int, int, int] = 3,
        max_offset: int = 4,
        use_fft: bool = True,
        average_frames: bool = False,
        metadata: dict | None = None,
        dims: str | Sequence[str] | None = None,
    ):
        self.filenames = [files] if isinstance(files, (str, Path)) else list(files)
        self.tiff_files = [TiffFile(f) for f in self.filenames]
        self._tiff_lock = threading.Lock()
        self._metadata = metadata if metadata is not None else get_metadata(self.filenames)
        if dims is not None:
            logger.warning(f"dims={dims!r} ignored: a ScanImage header says what each page is")

        page_shape = (self._metadata["page_height"], self._metadata["page_width"])
        for f, tf in zip(self.filenames, self.tiff_files, strict=True):
            if tf.pages.first.shape != page_shape:
                raise ValueError(
                    f"{Path(f).name} holds {tf.pages.first.shape} pages, not {page_shape}: "
                    "another acquisition is mixed in"
                )

        pages_per_frame = int(self._metadata.get("nchannels", 1))
        self._pages_per_file = [
            n * pages_per_frame for n in self._metadata["frames_per_file"]
        ]
        self.layout = scan_layout(self._metadata, sum(self._pages_per_file))
        if self.layout.pages_per_frame != pages_per_frame:
            raise ValueError(
                f"{self.filenames[0]}: {pages_per_frame} saved channels per frame, but "
                f"{self.layout.colors} colors x {self.layout.beamlets} beamlets"
            )
        if self.layout.volumes == 0:
            raise ValueError(
                f"{self.filenames[0]}: fewer frames than one volume "
                f"({self.layout.frames_per_volume})"
            )
        self._source_dtype = np.dtype(self._metadata.get("dtype", "int16"))
        self._target_dtype = None
        self._offset_cache: dict[int, float] = {}

        self._rois = extract_roi_slices(self._metadata)
        self._roi = None
        if roi is not None:
            self.roi = roi  # the mixin's setter validates it

        self._mean_subtraction = False
        self._average_frames = False
        self.average_frames = average_frames

        self.phase_correction = PhaseCorrectionFeature(
            enabled=fix_phase,
            method=phasecorr_method,
            shift=None,
            use_fft=use_fft,
            border=border if isinstance(border, int) else 3,
            max_offset=max_offset,
        )

    def close(self):
        for tf in self.tiff_files:
            tf.close()

    def get_required_metadata(self) -> list[dict]:
        """One record per ``REQUIRED_METADATA`` key: canonical, description, label, unit, dtype and the current value."""
        fields = []
        for param in self.REQUIRED_METADATA:
            mp = METADATA_PARAMS.get(param)
            fields.append(
                {
                    "canonical": param,
                    "description": self.get_param_description(param),
                    "label": mp.label if mp else param,
                    "unit": mp.unit if mp else "",
                    "dtype": mp.dtype if mp else float,
                    "value": get_param(self._metadata, param, default=None),
                }
            )
        return fields

    def get_param_description(self, param: str) -> str:
        """The nearest class's ``METADATA_CONTEXT`` entry for ``param``, else the registry's description."""
        for cls in type(self).__mro__:
            ctx = getattr(cls, "METADATA_CONTEXT", None)
            if ctx and param in ctx:
                return ctx[param]
        mp = METADATA_PARAMS.get(param)
        return mp.description if mp else ""

    @property
    def stack_type(self) -> StackType:
        return self.layout.stack_type

    @property
    def num_channels(self) -> int:
        """Pages saved per frame: colors times beamlets."""
        return self.layout.pages_per_frame

    @property
    def num_color_channels(self) -> int:
        return self.layout.colors

    @property
    def num_slices(self) -> int:
        return self.layout.slices

    @property
    def num_volumes(self) -> int:
        return self.layout.volumes

    @property
    def frames_per_slice(self) -> int:
        """Pages saved at each slice, after ScanImage's own averaging."""
        return self.layout.reps

    @property
    def log_average_factor(self) -> int:
        """ScanImage's acquisition-time averaging; above 1 every page is already that many frames averaged."""
        return get_log_average_factor(self._metadata)

    @property
    def can_average(self) -> bool:
        return self.frames_per_slice > 1

    @property
    def average_frames(self) -> bool:
        return self._average_frames

    @average_frames.setter
    def average_frames(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("average_frames must be a boolean value.")
        if value and not self.can_average:
            logger.warning("Frame averaging disabled: one saved frame per slice.")
            value = False
        self._average_frames = value
        md = layout_metadata(self._metadata, self.layout, value)
        md.pop("dz", None)  # the header's step, or the user's; averaging leaves it alone
        self._metadata.pop("fs", None)
        self._metadata.update(md)
        self._offset_cache.clear()
        self.invalidate_dimension_specs()

    @property
    def reader_kwargs(self) -> dict:
        """``average_frames`` when on, so a worker re-opening the files reads the T on screen."""
        return {"average_frames": True} if self._average_frames else {}

    @property
    def dtype(self):
        if self._target_dtype is not None:
            return self._target_dtype
        return np.dtype(np.float32) if self._average_frames else self._source_dtype

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata.update(value)

    @property
    def rois(self):
        return self.roi_slices

    @property
    def mean_subtraction(self):
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("mean_subtraction must be a boolean value.")
        self._mean_subtraction = value

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        t, c, z = self.layout.shape(self._average_frames)
        if isinstance(self.roi, int) and self.roi > 0:
            roi = self._rois[self.roi - 1]
            return t, c, z, roi["height"], roi["width"]
        height = max(r["height"] for r in self._rois)
        width = sum(r["width"] for r in self._rois)
        return t, c, z, height, width

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = tuple(_convert_range_to_slice(k) for k in key) + (slice(None),) * (5 - len(key))
        t_key, c_key, z_key, y_key, x_key = key
        nt, nc, nz = self.layout.shape(self._average_frames)
        t, c, z = listify_index(t_key, nt), listify_index(c_key, nc), listify_index(z_key, nz)
        if not t or not c or not z:
            return np.empty((len(t), len(c), len(z), *self.shape[3:]), dtype=self.dtype)

        pages = self.layout.pages(t, c, z, self._average_frames)  ## (T, R, C, Z)
        out = self._read_rois(pages.ravel())
        if isinstance(out, tuple):
            out = tuple(self._finish(x, pages.shape, key, y_key, x_key) for x in out)
        else:
            out = self._finish(out, pages.shape, key, y_key, x_key)
        return out

    def _finish(self, frames, pages_shape, key, y_key, x_key):
        """(T*R*C*Z, Y, X) stitched frames -> the averaged, cropped, squeezed and cast result."""
        frames = frames.reshape(*pages_shape, *frames.shape[-2:])
        frames = frames.mean(axis=1, dtype=np.float32) if self._average_frames else frames[:, 0]
        # whole frames are read and stitched, so y and x are cropped here
        if not (_is_full_slice(y_key) and _is_full_slice(x_key)):
            frames = frames[..., y_key, x_key]
        # np.integer too: np.linspace(dtype=int) hands back np.int64 keys
        squeeze = tuple(i for i in range(3) if isinstance(key[i], (int, np.integer)))
        if squeeze:
            frames = frames.squeeze(axis=squeeze)
        return frames.astype(self.dtype, copy=False)

    def _read_rois(self, pages: np.ndarray):
        """The ``pages`` with the mROI selection applied: one stitched (N, Y, X), one ROI, or a tuple per listed ROI."""
        if isinstance(self.roi, int) and self.roi > 0:
            return self._read_pages(pages, yslice=self._rois[self.roi - 1]["slice"])
        full = self._read_pages(pages)
        if isinstance(self.roi, (list, tuple)):
            return tuple(full[:, self._rois[r - 1]["slice"], :] for r in self.roi)
        height = max(r["height"] for r in self._rois)
        out = np.zeros((len(pages), height, sum(r["width"] for r in self._rois)), dtype=full.dtype)
        x = 0
        for r in self._rois:
            out[:, : r["height"], x : x + r["width"]] = full[:, r["slice"], :]
            x += r["width"]
        return out

    def _read_pages(self, pages: np.ndarray, yslice=slice(None)) -> np.ndarray:
        """Raw pages as (N, Y, X) in the order given, scan-phase corrected per file when ``fix_phase`` is on."""
        width = self._metadata["page_width"]
        height = index_length(yslice, self._metadata["page_height"])
        buf = np.empty((len(pages), height, width), dtype=self._source_dtype)
        start = 0
        for tf, num_pages in zip(self.tiff_files, self._pages_per_file, strict=True):
            in_file = np.flatnonzero((pages >= start) & (pages < start + num_pages))
            if len(in_file):
                local = pages[in_file] - start
                unique, inverse = np.unique(local, return_inverse=True)
                with self._tiff_lock:
                    chunk = tf.asarray(key=unique.tolist()).reshape(len(unique), -1, width)
                chunk = chunk[:, yslice, :]
                if self.fix_phase:
                    chunk = self._correct_phase(chunk, unique + start)
                else:
                    self._offset_cache.update(dict.fromkeys((unique + start).tolist(), 0.0))
                buf[in_file] = chunk[inverse]
            start += num_pages
        return buf

    def _correct_phase(self, chunk: np.ndarray, pages: np.ndarray) -> np.ndarray:
        shift = self.phase_correction.effective_shift
        if shift is not None:
            corrected, offset = _apply_offset(chunk, shift, use_fft=self.use_fft), shift
        else:
            corrected, offset = bidir_phasecorr(
                chunk,
                method=self.phasecorr_method,
                max_offset=self.max_offset,
                border=self.border,
                use_fft=self.use_fft,
            )
        offsets = np.broadcast_to(np.asarray(offset, dtype=float), pages.shape)
        self._offset_cache.update(zip(pages.tolist(), offsets.tolist(), strict=True))
        return corrected

    def get_offset_at(self, t: int, c: int = 0, z: int = 0):
        """The scan-phase offset applied to (t, c, z), the mean over averaged repeats; None before it is read."""
        pages = self.layout.pages([t], [c], [z], self._average_frames).ravel()
        offsets = [self._offset_cache.get(int(p)) for p in pages]
        return None if None in offsets else float(np.mean(offsets))

    def _invalidate_offset_cache(self) -> None:
        self._offset_cache.clear()

    def imshow(self, **kwargs):
        arrays = []
        names = []
        for roi in self.iter_rois():
            arr = copy.copy(self)
            arr.roi = roi
            arr.fix_phase = False
            arrays.append(arr)
            names.append(f"ROI {roi}" if roi else "Stitched mROIs")

        sample_frame = arrays[0][0, 0, 0]
        from mbo_utilities.gui._ndviewer import MboNDViewer

        return MboNDViewer(
            data=arrays,
            names=names,
            histogram_widget=kwargs.get("histogram_widget", True),
            figure_kwargs=kwargs.get("figure_kwargs", {"size": (600, 600)}),
            figure_shape=(1, len(arrays)),
            graphic_kwargs={
                "vmin": float(sample_frame.min()),
                "vmax": float(sample_frame.max()),
            },
            window_funcs=kwargs.get("window_funcs"),
        )


class LBMArray(ScanImageArray):
    """A Light Beads Microscopy scan: the beamlets ScanImage saved as channels are the z-planes."""

    STACK_TYPES = ("lbm",)
    METADATA_CONTEXT: dict[str, str] = {
        "Ly": (
            "For LBM with multi-ROIs, this is the total vertical height of all ROI strips "
            "including fly-to deadspace between them (num_fly_to_lines pixels)."
        ),
        "Lx": "ROI width in pixels (same for all mROIs in the scan).",
        "num_zplanes": "Number of z-planes encoded as ScanImage channels for LBM.",
        "dz": "Z-step size (µm). Must be user-supplied for LBM - not in ScanImage metadata.",
        "fs": "Volume rate in Hz: every beamlet records each frame.",
    }
    REQUIRED_METADATA: list[str] = ["dz"]  # ScanImage does not record the beamlet spacing


class PiezoArray(ScanImageArray):
    """A piezo or fast-z stack: volumes on T, slices on Z.

    Each slice saves ``framesPerSlice / logAverageFactor`` frames (one for
    fast-z); with more than one they follow their slice on T unless
    ``average_frames`` averages them. Fast-z flyback frames are dropped.
    """

    STACK_TYPES = ("piezo",)
    METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Frame height in pixels.",
        "Lx": "Frame width in pixels.",
        "num_zplanes": "Slices per volume (hStackManager.numSlices).",
        "num_timepoints": "Volumes saved, counted from the pages (times frames per slice unless averaged).",
        "dz": "Z-step size in µm (hStackManager.actualStackZStepSize).",
        "fs": "Volume rate in Hz (hRoiManager.scanVolumeRate).",
    }


class SinglePlaneArray(ScanImageArray):
    """A single-plane time series: frames on T, PMT colors on C."""

    STACK_TYPES = ("single_plane",)
    METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Frame height in pixels.",
        "Lx": "Frame width in pixels.",
        "num_zplanes": "Single plane (no z-stack).",
        "fs": "Frame rate in Hz.",
    }


class LBMPiezoArray(ScanImageArray):
    """A pollen calibration stack: the piezo steps through depth while every LBM beamlet records.

    Piezo positions go on T (they are not time, so ``fs`` is unset) and beamlets
    on Z, which is what ``pollen_calibration_mbo.py`` and the calibration viewer
    unpack after dropping C. Open it with ``average_frames=True`` to get one
    frame per position when each slice saved several.
    """

    STACK_TYPES = ("pollen",)
    METADATA_CONTEXT: dict[str, str] = {
        "Ly": "Frame height in pixels.",
        "Lx": "Frame width in pixels.",
        "num_timepoints": "Piezo positions (times frames per slice unless averaged).",
        "num_zplanes": "LBM beamlets.",
        "dz": "Beamlet spacing in µm; not in the header. The piezo step is hStackManager.stackZStepSize.",
    }
    REQUIRED_METADATA: list[str] = ["dz"]

    @property
    def num_beamlets(self) -> int:
        return self.layout.beamlets


def open_scanimage(files: str | Path | list, **kwargs) -> ScanImageArray:
    """Open ScanImage tiffs as the subclass for their stack type."""
    file_list = [files] if isinstance(files, (str, Path)) else list(files)
    metadata = get_metadata(file_list)
    cls = {
        "lbm": LBMArray,
        "piezo": PiezoArray,
        "pollen": LBMPiezoArray,
        "single_plane": SinglePlaneArray,
    }[detect_stack_type(metadata)]
    return cls(file_list, metadata=metadata, **kwargs)


# NOTE: ScanImage files are pieces of one acquisition, so they stay at the suffix-level 50: above it imread treats every file as a whole recording and opens only the first
register_array_class(LBMArray, priority=50)
register_array_class(LBMPiezoArray, priority=50)
register_array_class(PiezoArray, priority=50)
register_array_class(SinglePlaneArray, priority=50)
register_array_class(ScanImageArray, priority=40)
register_array_class(TiffArray, priority=30)
