"""Shared helpers for the array readers: 5D key mapping, the writer entry point, reductions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.exceptions import AxisError

from mbo_utilities import log
from mbo_utilities.lazy_array import LazyArray

if TYPE_CHECKING:
    from collections.abc import Callable

logger = log.get("arrays._base")

CHUNKS_4D = {0: 1, 1: "auto", 2: -1, 3: -1}
CHUNKS_3D = {0: 1, 1: -1, 2: -1}


def get_dtype(dtype):
    if isinstance(dtype, np.dtype):
        return dtype
    try:
        return np.dtype(dtype)
    except TypeError:
        return np.dtype(str(dtype))


DIMS = ("T", "C", "Z", "Y", "X")


class Shape5DMixin(LazyArray):
    """Compatibility alias of `LazyArray`, kept until every reader subclasses it directly."""


def _normalize_key(key, ndim):
    """Wraps ``key`` in a tuple and expands an Ellipsis to ``ndim`` axes."""
    if not isinstance(key, tuple):
        key = (key,)
    # identity, not ==: an ndarray in the key would compare elementwise
    idx = next((i for i, k in enumerate(key) if k is Ellipsis), None)
    if idx is not None:
        n_missing = ndim - (len(key) - 1)
        key = key[:idx] + (slice(None),) * max(n_missing, 0) + key[idx + 1 :]
    return key


_SKIP_BY_RAW_NDIM = {  # 5D axes a natural-rank source lacks, by its ndim
    5: (),
    4: (1,),
    3: (1, 2),
    2: (0, 1, 2),
    1: (0, 1, 2, 3),
}


def _index_5d_into_raw(data, key, raw_ndim):
    """Indexes natural-rank ``data`` with a 5D key, keeping numpy 5D semantics on the result.

    The singleton axes ``data`` lacks are dropped from the key and re-inserted
    on the result wherever the key sliced rather than integer-indexed them.
    """
    key = _normalize_key(key, 5)
    key = key + (slice(None),) * (5 - len(key))
    skip = _SKIP_BY_RAW_NDIM.get(raw_ndim, ())
    raw_key = tuple(k for i, k in enumerate(key) if i not in skip)
    out = np.asarray(data[raw_key])
    out_axis = 0
    for i in range(5):
        if isinstance(key[i], (int, np.integer)):
            continue
        if i in skip:
            out = np.expand_dims(out, axis=out_axis)
        out_axis += 1
    return out


def supports_roi(obj):
    """Deprecated; ``hasattr(obj, "roi_mode")`` is the check."""
    if hasattr(obj, "roi_mode"):
        return True
    return hasattr(obj, "roi") and hasattr(obj, "num_rois")


def normalize_roi(value):
    """Returns the ROI selection as None, an int or a list of ints; True is 0, split all."""
    if value in (None, (), [], False):
        return None
    if value is True:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def iter_rois(obj):
    """Deprecated; ``arr.iter_rois()`` yields None (stitched), 1..num_rois (``roi=0``) or the ROIs selected."""
    if hasattr(obj, "roi_mode") and hasattr(obj, "iter_rois"):
        yield from obj.iter_rois()
        return

    if not supports_roi(obj):
        yield None
        return

    roi = getattr(obj, "roi", None)
    num_rois = getattr(obj, "num_rois", 1)

    if roi is None:
        yield None
    elif roi == 0:
        yield from range(1, num_rois + 1)
    elif isinstance(roi, int):
        yield roi
    elif isinstance(roi, (list, tuple)):
        for r in roi:
            if r == 0:
                yield from range(1, num_rois + 1)
            else:
                yield r


def _normalize_planes(planes, num_planes: int) -> list[int]:
    """1-based plane selection (None is every plane) to a 0-based list."""
    if planes is None:
        return list(range(num_planes))
    if isinstance(planes, int):
        return [planes - 1]
    return [p - 1 for p in planes]


def _sanitize_suffix(suffix: str) -> str:
    """Makes a user suffix safe in a filename: no path characters, no trailing extension, one leading underscore."""
    if not suffix:
        return ""
    for char in '<>:"/\\|?*':
        suffix = suffix.replace(char, "")
    suffix = re.sub(r"\.[a-zA-Z0-9]+$", "", suffix)
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    while "__" in suffix:
        suffix = suffix.replace("__", "_")
    return suffix.rstrip("_")


def _sanitize_value(v):
    if isinstance(v, dict):
        return {k: _sanitize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_value(x) for x in v]
    if isinstance(v, (Path, np.dtype, type)):
        return str(v)
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _sanitize_metadata(md: dict) -> dict:
    """Metadata with every numpy, Path and dtype value turned into a JSON-serializable one."""
    return _sanitize_value(md)


def _selection_list(val):
    """1-based selection kwarg as a list of ints; None stays None (every index)."""
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return [int(val)]
    return [int(v) for v in val if v is not None]


def _imwrite_base(
    arr,
    outpath: Path | str,
    planes: int | list | tuple | None = None,
    frames: int | list | tuple | None = None,
    channels: int | list | tuple | None = None,
    ext: str = ".tiff",
    overwrite: bool = False,
    target_chunk_mb: int = 50,
    progress_callback: Callable | None = None,
    debug: bool = False,
    show_progress: bool = True,
    **kwargs,
) -> Path:
    """Writes ``arr`` into ``outpath`` in the format ``ext`` names (AGENTS.md §5.6).

    Selections are 1-based. tiff, zarr, h5 and mp4 write the volume in one
    pass; bin and npy write one file per (channel, plane). A split ROI
    selection fans out into one ``roiNN/`` directory per ROI.
    """
    from mbo_utilities._writers import (
        _scanphase_record,
        _write_plane,
        _write_scanphase_sidecar,
        _write_volumetric_tiff,
        _write_volumetric_zarr,
    )

    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    ext_clean = ext.lower().lstrip(".")

    _roi_sel = getattr(arr, "roi", None)
    _split = isinstance(_roi_sel, (list, tuple)) or (
        _roi_sel == 0 and getattr(arr, "num_rois", 1) > 1
    )
    if _split:
        roi_indices = (
            list(_roi_sel)
            if isinstance(_roi_sel, (list, tuple))
            else list(range(1, arr.num_rois + 1))
        )
        for r in roi_indices:
            arr.roi = r
            _imwrite_base(
                arr,
                outpath / f"roi{r:02d}",
                planes=planes,
                frames=frames,
                channels=channels,
                ext=ext,
                overwrite=overwrite,
                target_chunk_mb=target_chunk_mb,
                progress_callback=progress_callback,
                debug=debug,
                show_progress=show_progress,
                **kwargs,
            )
        arr.roi = _roi_sel
        return outpath

    md = dict(arr.metadata) if arr.metadata else {}

    if debug:
        n_shifts = len(md.get("plane_shifts", []))
        logger.info(f"_imwrite_base: plane_shifts={n_shifts} planes")

    if "metadata_overrides" in kwargs:
        overrides = kwargs.pop("metadata_overrides", None)
        if overrides and isinstance(overrides, dict):
            md.update(overrides)

    md = _sanitize_metadata(md)

    num_planes = arr._shape5d()[2]
    # NOTE: IsoView's num_color_channels counts wavelengths; its C axis is num_views
    num_channels = getattr(arr, "num_views", None)
    if num_channels is None:
        num_channels = getattr(arr, "num_color_channels", None)
    if num_channels is None:
        num_channels = arr.nc

    planes_list = _selection_list(planes)
    frames_list = _selection_list(frames)
    channels_list = _selection_list(channels)

    # the volumetric writers take num_frames only as an explicit frames list
    if ext_clean in ("tiff", "tif", "zarr", "h5", "hdf5") and frames_list is None:
        _num_frames = kwargs.get("num_frames")
        if _num_frames is not None:
            total_T = int(arr._shape5d()[0])
            frames_list = list(range(1, min(int(_num_frames), total_T) + 1))

    if ext_clean in ("tiff", "tif"):
        output_suffix = kwargs.pop("output_suffix", None)
        result = _write_volumetric_tiff(
            arr,
            outpath,
            metadata=md,
            planes=planes_list,
            frames=frames_list,
            channels=channels_list,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            show_progress=show_progress,
            debug=debug,
            output_suffix=output_suffix,
        )
        return result

    if ext_clean == "zarr":
        output_suffix = kwargs.pop("output_suffix", None)
        sharded = kwargs.pop("sharded", True)
        # NOTE: uncompressed by default; gzip-1 cost ~5 ms per frame when scrubbing
        compression_level = kwargs.pop("compression_level", 0)
        compressor = kwargs.pop("compressor", "none")
        shuffle = kwargs.pop("shuffle", None)
        pyramid = kwargs.pop("pyramid", False)
        pyramid_max_layers = kwargs.pop("pyramid_max_layers", 4)
        pyramid_method = kwargs.pop("pyramid_method", "mean")
        result = _write_volumetric_zarr(
            arr,
            outpath,
            metadata=md,
            planes=planes_list,
            frames=frames_list,
            channels=channels_list,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            show_progress=show_progress,
            debug=debug,
            output_suffix=output_suffix,
            sharded=sharded,
            compression_level=compression_level,
            compressor=compressor,
            shuffle=shuffle,
            pyramid=pyramid,
            pyramid_max_layers=pyramid_max_layers,
            pyramid_method=pyramid_method,
        )
        return result

    if ext_clean in ("h5", "hdf5"):
        from mbo_utilities._writers import _write_volumetric_h5

        output_suffix = kwargs.pop("output_suffix", None)
        dataset_name = kwargs.pop("dataset_name", None) or "mov"
        h5_compression = kwargs.pop("compression", "gzip")
        h5_compression_level = kwargs.pop("compression_level", 1)
        result = _write_volumetric_h5(
            arr,
            outpath,
            metadata=md,
            planes=planes_list,
            frames=frames_list,
            channels=channels_list,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            show_progress=show_progress,
            debug=debug,
            output_suffix=output_suffix,
            dataset_name=dataset_name,
            compression=h5_compression,
            compression_level=h5_compression_level,
        )
        return result

    if ext_clean == "mp4":
        from mbo_utilities.arrays.mp4 import MP4Array

        output_suffix = kwargs.pop("output_suffix", None)
        mean_subtract_stack = kwargs.pop("mean_subtract_stack", None)
        mean_subtract_path = kwargs.pop("mean_subtract_path", None)
        if mean_subtract_stack is None and mean_subtract_path:
            mean_subtract_stack = np.load(mean_subtract_path)
        return MP4Array.write_video(
            arr,
            outpath,
            metadata=md,
            planes=planes_list,
            frames=frames_list,
            channels=channels_list,
            ext=ext_clean,
            overwrite=overwrite,
            output_suffix=output_suffix,
            progress_callback=progress_callback,
            show_progress=show_progress,
            fps=int(kwargs.pop("fps", 30)),
            speed_factor=float(kwargs.pop("speed_factor", 1.0)),
            vmin=kwargs.pop("vmin", None),
            vmax=kwargs.pop("vmax", None),
            vmin_percentile=float(kwargs.pop("vmin_percentile", 1.0)),
            vmax_percentile=float(kwargs.pop("vmax_percentile", 99.5)),
            temporal_smooth=int(kwargs.pop("temporal_smooth", 0)),
            spatial_smooth=float(kwargs.pop("spatial_smooth", 0.0)),
            gamma=float(kwargs.pop("gamma", 1.0)),
            cmap=kwargs.pop("cmap", None),
            quality=kwargs.pop("quality", "visually lossless"),
            codec=str(kwargs.pop("codec", "libx264")),
            mean_subtract_stack=mean_subtract_stack,
            temporal_mode=str(kwargs.pop("temporal_mode", "mean")),
            time_overlay=bool(kwargs.pop("time_overlay", False)),
            scalebar=bool(kwargs.pop("scalebar", False)),
        )

    if channels_list is not None:
        channels_0idx = [c - 1 for c in channels_list]
    else:
        channels_0idx = list(range(num_channels))

    s5 = arr._shape5d()  # BinArray and _ChannelView report another rank on .shape
    nframes = s5[0]
    Ly, Lx = s5[3], s5[4]
    num_planes = min(num_planes, s5[2])

    from mbo_utilities.metadata import OutputMetadata

    frame_indices_0 = None
    plane_indices_0 = None
    if frames_list is not None:
        frame_indices_0 = [f - 1 for f in frames_list]
    if planes_list is not None:
        plane_indices_0 = [p - 1 for p in planes_list]

    out_meta = OutputMetadata(
        source=md,
        frame_indices=frame_indices_0,
        plane_indices=plane_indices_0,
        source_num_frames=nframes,
        source_num_planes=num_planes,
    )

    md = out_meta.to_dict()

    truncated_n = kwargs.get("num_frames")
    has_frame_selection = bool(frame_indices_0)
    if has_frame_selection and out_meta.num_frames is not None:
        effective_nt = int(out_meta.num_frames)
    elif truncated_n is not None and int(truncated_n) > 0:
        effective_nt = int(truncated_n)
    elif out_meta.num_frames is not None:
        effective_nt = int(out_meta.num_frames)
    else:
        effective_nt = int(nframes)

    # NOTE: Ly/Lx stay OutputMetadata's; restamping them loses lsp's padded dims
    md["num_timepoints"] = effective_nt
    md["nframes"] = effective_nt
    md["num_frames"] = effective_nt
    md["n_frames"] = effective_nt
    md["T"] = effective_nt
    md["nt"] = effective_nt
    md["timepoints"] = effective_nt

    planes_0idx = _normalize_planes(planes_list, num_planes)

    if frames_list is not None:
        written_frames_0 = [f - 1 for f in frames_list]
    else:
        written_frames_0 = list(range(effective_nt))

    output_name = kwargs.pop("output_name", None)

    for c_idx in channels_0idx:
        for plane_idx in planes_0idx:
            from mbo_utilities.arrays.features import (
                TAG_REGISTRY,
                DimensionTag,
                OutputFilename,
            )

            z_tag = DimensionTag.from_dim_size(
                TAG_REGISTRY["Z"], num_planes, [plane_idx + 1]
            )
            t_tag = DimensionTag.from_dim_size(TAG_REGISTRY["T"], nframes, frames_list)

            if output_name:
                target = outpath / output_name
            elif ext_clean == "bin":
                # suite2p reads data_raw.bin by exact name, so only the dir carries tags
                plane_dir = outpath / f"{z_tag.to_string()}_{t_tag.to_string()}"
                plane_dir.mkdir(parents=True, exist_ok=True)
                bin_name = "data_chan2.bin" if c_idx > 0 else "data_raw.bin"
                target = plane_dir / bin_name
            else:
                tags = [t_tag, z_tag] if nframes > 1 else [z_tag]
                filename = OutputFilename(tags).build(f".{ext_clean}")
                target = outpath / filename

            if target.exists() and not overwrite:
                logger.warning(f"File {target} already exists. Skipping write.")
                continue

            plane_md = md.copy()
            plane_md["plane"] = plane_idx + 1

            _write_plane(
                arr,
                target,
                overwrite=overwrite,
                target_chunk_mb=target_chunk_mb,
                metadata=plane_md,
                progress_callback=progress_callback,
                debug=debug,
                show_progress=show_progress,
                dshape=(nframes, Ly, Lx),
                plane_index=plane_idx,
                channel_index=c_idx,
                frames=frames_list,
                **kwargs,
            )

            if ext_clean == "bin" and c_idx == 0:
                _write_scanphase_sidecar(
                    target.parent,
                    _scanphase_record(arr, written_frames_0, c_idx, plane_idx),
                )

    if progress_callback:
        progress_callback(1.0, "Complete")

    return outpath


def _axis_slice(ndim, axis, start, end):
    slices = [slice(None)] * ndim
    slices[axis] = slice(start, end)
    return tuple(slices)


def temporal_mean(arr, progress_callback=None) -> np.ndarray:
    """``arr``'s per-pixel mean over T as ``(C, Z, Y, X)`` float32.

    A reader's own ``temporal_mean`` wins (`MescArray` reads each channel once);
    otherwise T is streamed in 64 MB blocks.
    """
    own = getattr(arr, "temporal_mean", None)
    if own is not None:
        return own(progress_callback=progress_callback)
    nt, nc, nz, ny, nx = arr._shape5d()
    frame_bytes = nc * nz * ny * nx * np.dtype(arr.dtype).itemsize
    step = max(1, (64 * 1024 * 1024) // max(1, frame_bytes))
    acc = np.zeros((nc, nz, ny, nx), dtype=np.float64)
    for t0 in range(0, nt, step):
        t1 = min(t0 + step, nt)
        block = np.asarray(arr[t0:t1]).reshape(t1 - t0, nc, nz, ny, nx)
        acc += block.sum(axis=0, dtype=np.float64)
        if progress_callback is not None:
            progress_callback(t1 / nt)
    return (acc / max(nt, 1)).astype(np.float32)


class TiffReaderMixin:
    """Lazy ``astype``, a one-frame ``__array__``, ``_imwrite`` and ``imshow`` for the TIFF readers."""

    _target_dtype = None

    def astype(self, dtype, copy=True):
        """Sets the dtype reads are converted to; nothing is read."""
        self._target_dtype = np.dtype(dtype)
        return self

    def __array__(self, dtype=None, copy=None):
        """The first (Y, X) frame, never the whole array."""
        ndim = self.ndim
        if ndim == 5:
            data = np.asarray(self[0, 0, 0])
        elif ndim == 2:
            data = np.asarray(self[:])
        else:
            data = np.asarray(self[0])
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def _imwrite(
        self,
        outpath,
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

    def save(self, outpath, **kwargs):
        return self._imwrite(outpath, **kwargs)

    def imshow(self, **kwargs):
        """Opens the array in the Studio's fastplotlib viewer."""
        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        window_funcs = kwargs.get("window_funcs")
        from mbo_utilities.gui._ndviewer import MboNDViewer

        return MboNDViewer(
            data=self,
            cmap="gnuplot2",
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            graphic_kwargs={"vmin": self.vmin, "vmax": self.vmax},
            window_funcs=window_funcs,
        )


class ReductionMixin:
    """numpy's ``mean`` ``max`` ``min`` ``std`` ``var`` ``sum`` for any array-like with ``shape``, ``dtype`` and ``__getitem__``.

    Above 100M elements the reduction streams chunks along the reduced axis;
    a numpy array or memmap goes straight to numpy. Also holds the cached
    display range ``vmin`` / ``vmax`` of the representative frame.
    """

    __array_ufunc__ = None  # numpy and xarray must not coerce a lazy array
    __array_function__ = None

    _cached_vmin = None
    _cached_vmax = None

    def _compute_frame_vminmax(self):
        # NOTE: np.asarray(self) is each class's own representative frame; never index here
        if self._cached_vmin is None:
            frame = np.asarray(self)
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        self._compute_frame_vminmax()
        return self._cached_vmax

    def _is_numpy_like(self) -> bool:
        return isinstance(self, (np.ndarray, np.memmap))

    def _get_array_info(self) -> tuple[tuple, np.dtype, int]:
        """``(shape, dtype, ndim)``; dtype is float64 when the array reports None."""
        shape = self.shape
        dtype = self.dtype
        if dtype is None:
            dtype = np.float64
        return shape, dtype, len(shape)

    def _reduce(
        self,
        func: str,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        keepdims: bool = False,
        out: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Applies numpy's ``func`` with numpy's semantics, chunked above 100M elements."""
        shape, _arr_dtype, ndim = self._get_array_info()

        if self._is_numpy_like():
            np_func = getattr(np, func)
            if func in ("max", "min"):  # they take no dtype
                return np_func(self, axis=axis, keepdims=keepdims, out=out, **kwargs)
            return np_func(
                self, axis=axis, dtype=dtype, keepdims=keepdims, out=out, **kwargs
            )

        if axis is not None:
            if isinstance(axis, int):
                if axis < 0:
                    axis = ndim + axis
                if axis < 0 or axis >= ndim:
                    raise AxisError(axis, ndim)
            else:
                axis = tuple(ax if ax >= 0 else ndim + ax for ax in axis)
                for ax in axis:
                    if ax < 0 or ax >= ndim:
                        raise AxisError(ax, ndim)

        if int(np.prod(shape)) <= 100_000_000:
            data = self[:]  # np.asarray(self) is one frame only
            np_func = getattr(np, func)
            if func in ("max", "min"):
                return np_func(data, axis=axis, keepdims=keepdims, out=out, **kwargs)
            return np_func(
                data, axis=axis, dtype=dtype, keepdims=keepdims, out=out, **kwargs
            )

        return self._chunked_reduce(
            func, axis=axis, dtype=dtype, keepdims=keepdims, out=out, **kwargs
        )

    def _chunked_reduce(
        self,
        func: str,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        keepdims: bool = False,
        out: np.ndarray | None = None,
        chunk_size: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """``func`` over ``axis`` in blocks of ``chunk_size`` along it (along axis 0 for a full reduction)."""
        from tqdm.auto import tqdm

        shape, arr_dtype, ndim = self._get_array_info()

        if axis is None:
            n = shape[0]
            if func == "mean":
                total_sum = 0.0
                total_count = 0
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end])
                    total_sum += np.sum(chunk, dtype=np.float64)
                    total_count += chunk.size
                result = total_sum / total_count
            elif func == "sum":
                result = 0
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end])
                    result += np.sum(chunk, dtype=dtype)
            elif func == "max":
                result = None
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end])
                    chunk_max = np.max(chunk)
                    result = chunk_max if result is None else max(result, chunk_max)
            elif func == "min":
                result = None
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end])
                    chunk_min = np.min(chunk)
                    result = chunk_min if result is None else min(result, chunk_min)
            elif func in ("std", "var"):
                mean_val = self._chunked_reduce(
                    "mean", axis=None, chunk_size=chunk_size
                )
                variance_sum = 0.0
                total_count = 0
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end]).astype(np.float64)
                    variance_sum += np.sum((chunk - mean_val) ** 2)
                    total_count += chunk.size
                ddof = kwargs.get("ddof", 0)
                variance = variance_sum / (total_count - ddof)
                result = np.sqrt(variance) if func == "std" else variance
            else:
                raise ValueError(f"Unknown reduction function: {func}")

            if dtype is not None:
                result = np.dtype(dtype).type(result)
            if keepdims:
                result = np.array(result).reshape((1,) * ndim)
            if out is not None:
                out[...] = result
                return out
            return result

        if isinstance(axis, int):
            ax = axis
            n = shape[ax]
            out_shape = shape[:ax] + shape[ax + 1 :]
            if dtype is None:
                if func in ("mean", "std", "var"):
                    reduce_dtype = np.float64
                else:
                    reduce_dtype = arr_dtype
            else:
                reduce_dtype = dtype

            if func == "mean":
                accumulator = np.zeros(out_shape, dtype=np.float64)
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[_axis_slice(ndim, ax, start, end)])
                    accumulator += np.sum(chunk, axis=ax, dtype=np.float64)
                result = (accumulator / n).astype(reduce_dtype)

            elif func == "sum":
                accumulator = np.zeros(out_shape, dtype=reduce_dtype)
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[_axis_slice(ndim, ax, start, end)])
                    accumulator += np.sum(chunk, axis=ax)
                result = accumulator

            elif func == "max":
                chunk = np.asarray(self[_axis_slice(ndim, ax, 0, min(chunk_size, n))])
                accumulator = np.max(chunk, axis=ax)
                for start in tqdm(
                    range(chunk_size, n, chunk_size), desc=f"Computing {func}"
                ):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[_axis_slice(ndim, ax, start, end)])
                    accumulator = np.maximum(accumulator, np.max(chunk, axis=ax))
                result = accumulator.astype(reduce_dtype)

            elif func == "min":
                chunk = np.asarray(self[_axis_slice(ndim, ax, 0, min(chunk_size, n))])
                accumulator = np.min(chunk, axis=ax)
                for start in tqdm(
                    range(chunk_size, n, chunk_size), desc=f"Computing {func}"
                ):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[_axis_slice(ndim, ax, start, end)])
                    accumulator = np.minimum(accumulator, np.min(chunk, axis=ax))
                result = accumulator.astype(reduce_dtype)

            elif func in ("std", "var"):
                mean_val = self._chunked_reduce("mean", axis=ax, chunk_size=chunk_size)
                variance = np.zeros(out_shape, dtype=np.float64)
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[_axis_slice(ndim, ax, start, end)]).astype(
                        np.float64
                    )
                    mean_expanded = np.expand_dims(mean_val, axis=ax)
                    variance += np.sum((chunk - mean_expanded) ** 2, axis=ax)
                ddof = kwargs.get("ddof", 0)
                variance = variance / (n - ddof)
                result = (np.sqrt(variance) if func == "std" else variance).astype(
                    reduce_dtype
                )

            else:
                raise ValueError(f"Unknown reduction function: {func}")

            if keepdims:
                result = np.expand_dims(result, axis=ax)
            if out is not None:
                out[...] = result
                return out
            return result

        axes = sorted(axis, reverse=True)  # highest first so the lower indices hold
        result = self
        for ax in axes:
            result = (
                result._chunked_reduce(func, axis=ax, dtype=dtype, **kwargs)
                if hasattr(result, "_chunked_reduce")
                else getattr(np, func)(result, axis=ax, dtype=dtype, **kwargs)
            )
        if keepdims:
            for ax in sorted(axis):
                result = np.expand_dims(result, axis=ax)
        if out is not None:
            out[...] = result
            return out
        return result

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """``numpy.mean`` over ``axis``; None reduces every element."""
        return self._reduce(
            "mean", axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kwargs
        )

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """``numpy.max`` over ``axis``; None reduces every element."""
        return self._reduce("max", axis=axis, out=out, keepdims=keepdims, **kwargs)

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """``numpy.min`` over ``axis``; None reduces every element."""
        return self._reduce("min", axis=axis, out=out, keepdims=keepdims, **kwargs)

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        out: np.ndarray | None = None,
        ddof: int = 0,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """``numpy.std`` over ``axis``; None reduces every element."""
        return self._reduce(
            "std",
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            **kwargs,
        )

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        out: np.ndarray | None = None,
        ddof: int = 0,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """``numpy.var`` over ``axis``; None reduces every element."""
        return self._reduce(
            "var",
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            **kwargs,
        )

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """``numpy.sum`` over ``axis``; None reduces every element."""
        return self._reduce(
            "sum", axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kwargs
        )
