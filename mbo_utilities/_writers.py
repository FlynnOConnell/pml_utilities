import functools
import math
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tifffile import TiffWriter
from tifffile import imwrite as tiff_imwrite
from tqdm.auto import tqdm

from . import log
from .file_io import load_npy
from .metadata.base import normalize_ops_arrays
from .metadata.io import _build_ome_metadata

logger = log.get("writers")


def _is_disabled_si_module(value) -> bool:
    """Check if a scanimage module dict has enable=false."""
    if not isinstance(value, dict):
        return False
    enable_val = value.get("enable")
    if enable_val is False:
        return True
    return bool(isinstance(enable_val, str) and enable_val.lower() in ("false", "0"))


def _filter_disabled_modules(metadata: dict, recursive: bool = True) -> dict:
    """Filter out disabled scanimage modules (hXxx with enable=false) from metadata."""
    if not isinstance(metadata, dict):
        return metadata
    result = {}
    for k, v in metadata.items():
        if isinstance(k, str) and k.startswith("h") and _is_disabled_si_module(v):
            continue
        if recursive and isinstance(v, dict):
            v = _filter_disabled_modules(v, recursive=True)
        result[k] = v
    return result


_MAX_NDARRAY_NBYTES = 1 * 1024 * 1024


def _make_json_serializable(obj, filter_disabled: bool = True):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        if filter_disabled:
            obj = _filter_disabled_modules(obj, recursive=True)
        return {
            k: _make_json_serializable(v, filter_disabled=False) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v, filter_disabled=False) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.nbytes >= _MAX_NDARRAY_NBYTES:
            return None
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def _convert_paths_to_strings(obj, filter_disabled: bool = True):
    """Recursively convert pathlib.Path objects to strings in a nested structure."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        if filter_disabled:
            obj = _filter_disabled_modules(obj, recursive=True)
        return {
            k: _convert_paths_to_strings(v, filter_disabled=False)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            _convert_paths_to_strings(v, filter_disabled=False) for v in obj
        )
    if isinstance(obj, np.ndarray):
        return obj
    return obj


warnings.filterwarnings("ignore")

CHUNKS = {0: "auto", 1: -1, 2: -1}


def _get_mbo_version():
    from . import __version__

    return __version__


def add_processing_step(
    metadata: dict,
    step_name: str,
    input_files: list | str | None = None,
    output_files: list | str | None = None,
    duration_seconds: float | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Add a processing step to metadata["processing_history"].

    Each step is appended to the history list, preserving previous runs.
    This allows tracking of re-runs and incremental processing across
    both mbo_utilities and downstream tools like lbm_suite2p_python.

    Parameters
    ----------
    metadata : dict
        The metadata dictionary to update.
    step_name : str
        Name of the processing step (e.g., "imwrite", "scan_phase_correction",
        "format_conversion", "z_registration").
    input_files : list of str or str, optional
        List of input file paths for this step.
    output_files : list of str or str, optional
        List of output file paths for this step.
    duration_seconds : float, optional
        How long this step took.
    extra : dict, optional
        Additional metadata for this step (e.g., scan-phase parameters,
        output format, compression settings).

    Returns
    -------
    dict
        The updated metadata dictionary.

    Examples
    --------
    >>> metadata = {}
    >>> add_processing_step(
    ...     metadata,
    ...     "imwrite",
    ...     input_files=["raw.tif"],
    ...     output_files=["output.zarr"],
    ...     extra={"output_format": ".zarr", "fix_phase": True, "use_fft": True}
    ... )
    """
    if "processing_history" not in metadata:
        metadata["processing_history"] = []

    step_record = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "mbo_utilities_version": _get_mbo_version(),
    }

    if input_files is not None:
        if isinstance(input_files, (str, Path)):
            step_record["input_files"] = [str(input_files)]
        else:
            step_record["input_files"] = [str(f) for f in input_files]

    if output_files is not None:
        if isinstance(output_files, (str, Path)):
            step_record["output_files"] = [str(output_files)]
        else:
            step_record["output_files"] = [str(f) for f in output_files]

    if duration_seconds is not None:
        step_record["duration_seconds"] = round(duration_seconds, 2)

    if extra is not None:
        step_record.update(extra)

    metadata["processing_history"].append(step_record)
    return metadata


_BIN_WRITERS: dict[str, Any] = {}
_BIN_OFFSETS: dict[str, int] = {}
_NPY_ARRAYS: dict[str, np.memmap] = {}
_NPY_OFFSETS: dict[str, int] = {}
_NPY_METADATA: dict[str, dict] = {}
_H5_INITIALIZED: dict[Path, bool] = {}
_H5_OFFSETS: dict[Path, int] = {}
_H5_DATASET_NAMES: dict[Path, str] = {}
_TIFF_WRITERS: dict[Path, TiffWriter] = {}
_TIFF_FIRST_WRITE: dict[Path, bool] = {}
_TIFF_IMAGEJ_MODE: dict[Path, bool] = {}
_ZARR_ARRAYS: dict = {}
_ZARR_OFFSETS: dict = {}
_ZARR_GROUPS: dict = {}


def _close_specific_bin_writer(filepath):
    key = str(Path(filepath))
    if key in _BIN_WRITERS:
        _BIN_WRITERS.pop(key).close()
        _BIN_OFFSETS.pop(key, None)


def _close_specific_tiff_writer(filepath):
    key = Path(filepath).with_suffix(".tif")
    if key in _TIFF_WRITERS:
        _TIFF_WRITERS.pop(key).close()
        _TIFF_FIRST_WRITE.pop(key, None)
        _TIFF_IMAGEJ_MODE.pop(key, None)


def _close_specific_npy_writer(filepath):
    """Package a finished .npy memmap with its metadata into one np.savez file."""
    key = str(Path(filepath).with_suffix(".npy"))
    if key not in _NPY_ARRAYS:
        return
    mmap = _NPY_ARRAYS.pop(key)
    metadata = _NPY_METADATA.pop(key, {})
    _NPY_OFFSETS.pop(key, None)
    data = np.array(mmap)
    mmap.flush()
    if mmap._mmap is not None:
        mmap._mmap.close()
    temp_path = Path(key).with_suffix(".npy.tmp")
    if temp_path.exists():
        temp_path.unlink()
    final_path = Path(key)
    npz_path = final_path.with_suffix(".npz")
    np.savez(npz_path, data=data, metadata=np.array(metadata, dtype=object))
    if final_path.exists():
        final_path.unlink()
    npz_path.rename(
        final_path
    )  # NOTE: np.savez writes .npz; the rename keeps the .npy suffix np.load expects


def _write_plane(
    data: np.ndarray | Any,
    filename: Path,
    *,
    overwrite=False,
    metadata=None,
    target_chunk_mb=20,
    progress_callback=None,
    debug=False,
    show_progress=True,
    dshape=None,
    plane_index=None,
    channel_index=None,
    frames=None,
    **kwargs,
):
    if dshape is None:
        dshape = data.shape

    metadata = metadata or {}

    if plane_index is not None:
        if not isinstance(plane_index, (int, np.integer)):
            raise TypeError(f"plane_index must be an integer, got {type(plane_index)}")
        metadata["plane"] = int(plane_index) + 1

    if frames is not None:
        frames_0 = [int(f) - 1 for f in frames]
        nframes_target = len(frames_0)
    else:
        frames_0 = None
        nframes_target = (
            kwargs.get("nframes")
            or kwargs.get("num_frames")
            or metadata.get("nframes")
            or metadata.get("num_frames")
        )

    if nframes_target is None or nframes_target <= 0:
        nframes_target = data.shape[0]

    nframes_target = int(nframes_target)
    metadata["nframes"] = nframes_target
    metadata["num_frames"] = nframes_target

    dshape = (nframes_target, *dshape[1:])
    metadata["shape"] = dshape

    fname = filename
    writer = _get_file_writer(fname.suffix, overwrite=overwrite)

    itemsize = np.dtype(data.dtype).itemsize
    ntime = int(nframes_target)

    if ntime == 0:
        raise ValueError(
            f"Cannot write file with 0 frames. Data shape: {data.shape}, nframes_target: {nframes_target}"
        )

    bytes_per_t = int(np.prod(dshape[1:], dtype=np.int64)) * int(itemsize)
    chunk_size = int(target_chunk_mb) * 1024 * 1024

    if chunk_size <= 0:
        chunk_size = 20 * 1024 * 1024

    total_bytes = int(ntime) * int(bytes_per_t)
    nchunks = max(1, math.ceil(total_bytes / chunk_size))
    nchunks = min(nchunks, ntime)

    base = ntime // nchunks
    extra = ntime % nchunks

    if show_progress and not debug:
        pbar = tqdm(total=nchunks, desc=f"Saving {fname.name}")
    else:
        pbar = None

    start = 0
    for i in range(nchunks):
        end = start + base + (1 if i < extra else 0)

        c_idx = channel_index if channel_index is not None else 0
        z_idx = plane_index if plane_index is not None else 0
        if frames_0 is not None:
            sel = frames_0[start:end]
            if sel and sel == list(range(sel[0], sel[-1] + 1)):
                chunk = data[sel[0] : sel[-1] + 1, c_idx, z_idx, :, :]
            else:
                chunk = np.stack(
                    [np.asarray(data[fi, c_idx, z_idx, :, :]) for fi in sel]
                )
        else:
            chunk = data[start:end, c_idx, z_idx, :, :]

        if hasattr(chunk, "compute"):
            chunk = chunk.compute()
        elif isinstance(chunk, np.memmap):
            chunk = np.array(chunk)
        elif not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk)

        writer(fname, chunk, metadata=metadata, **kwargs)

        if pbar:
            pbar.update(1)
        if progress_callback:
            progress_callback(pbar.n / pbar.total, current_plane=plane_index)
        start = end
    if pbar:
        pbar.close()

    if fname.suffix in [".tiff", ".tif"]:
        _close_specific_tiff_writer(fname)
    elif fname.suffix in [".bin"]:
        _close_specific_bin_writer(fname)
    elif fname.suffix in [".npy"]:
        _close_specific_npy_writer(fname)


def _scanphase_raw_source(arr):
    """(dir, file_list) the source array read from, for replay re-open."""
    files = [str(f) for f in getattr(arr, "filenames", []) or []]
    src = None
    if files:
        parents = {str(Path(f).parent) for f in files}
        if len(parents) == 1:
            src = next(iter(parents))
    return src, files


def _scanphase_record(arr, written_frames_0, channel_index, plane_index):
    """Per-frame scan-phase offsets the writer applied to (channel, plane).

    Returns a replay record (offsets aligned to written frame order, raw
    source, frame map) or None when the source applied no phase correction.
    """
    get_off = getattr(arr, "get_offset_at", None)
    if get_off is None or not getattr(arr, "fix_phase", False):
        return None
    offs = [
        float(get_off(int(t), int(channel_index), int(plane_index)) or 0.0)
        for t in written_frames_0
    ]
    src, files = _scanphase_raw_source(arr)
    return {
        "scanphase_applied": True,
        "offsets": np.asarray(offs, dtype=np.float32),
        "frame_map": np.asarray(written_frames_0, dtype=np.int64),
        "use_fft": bool(getattr(arr, "use_fft", True)),
        "raw_source": src,
        "raw_files": files,
        "plane_index": int(plane_index),
        "channel_index": int(channel_index),
    }


def _write_scanphase_sidecar(plane_dir, record):
    """Persist a scan-phase replay record next to the binaries (scanphase.npy)."""
    if record is None:
        return
    np.save(Path(plane_dir) / "scanphase.npy", record, allow_pickle=True)


def _scanphase_zarr_attrs(data, frames, planes, channels):
    """JSON-able per-(channel, plane) scan-phase offsets for zarr group attrs.

    Provenance of the bidirectional offset baked into the zarr pixels; returns
    None when the source applied no phase correction.
    """
    get_off = getattr(data, "get_offset_at", None)
    if get_off is None or not getattr(data, "fix_phase", False):
        return None
    s5 = data._shape5d()
    T, C, Z = int(s5[0]), int(s5[1]), int(s5[2])
    fr0 = [f - 1 for f in frames] if frames else list(range(T))
    z0 = [p - 1 for p in planes] if planes else list(range(Z))
    c0 = [c - 1 for c in channels] if channels else list(range(C))
    recs = []
    for ci in c0:
        for zi in z0:
            offs = [float(get_off(int(t), int(ci), int(zi)) or 0.0) for t in fr0]
            recs.append(
                {"channel_index": int(ci), "plane_index": int(zi), "offsets": offs}
            )
    src, files = _scanphase_raw_source(data)
    return {
        "scanphase_applied": True,
        "use_fft": bool(getattr(data, "use_fft", True)),
        "frame_map": [int(f) for f in fr0],
        "raw_source": src,
        "raw_files": files,
        "planes": recs,
    }


def _get_file_writer(ext, overwrite):
    if ext.startswith("."):
        ext = ext.lstrip(".")
    if ext in ["tif", "tiff"]:
        return functools.partial(
            _write_tiff,
            overwrite=overwrite,
        )
    if ext in ["h5", "hdf5"]:
        return functools.partial(
            _write_h5,
            overwrite=overwrite,
        )
    if ext in ["zarr"]:
        return functools.partial(
            _write_zarr,
            overwrite=overwrite,
        )
    if ext == "bin":
        return functools.partial(
            _write_bin,
            overwrite=overwrite,
        )
    if ext == "npy":
        return functools.partial(
            _write_npy,
            overwrite=overwrite,
        )
    raise ValueError(f"Unsupported file extension: {ext}")


def _write_bin(path, data, *, overwrite: bool = False, metadata=None, **kwargs):
    from .arrays.bin import BinArray

    if metadata is None:
        metadata = {}

    fname = Path(path)
    fname.parent.mkdir(exist_ok=True)

    key = str(fname)
    first_write = False

    if key in _BIN_WRITERS and not Path(key).exists():
        _BIN_WRITERS.pop(key, None)
        _BIN_OFFSETS.pop(key, None)

    if overwrite and key not in _BIN_WRITERS and fname.exists():
        fname.unlink()

    if key not in _BIN_WRITERS:
        Ly, Lx = data.shape[-2], data.shape[-1]
        nframes = metadata.get("nframes")
        if nframes is None:
            nframes = metadata.get("num_frames")
        if nframes is None:
            raise ValueError("Metadata must contain 'nframes' or 'num_frames'.")

        metadata["Ly"] = Ly
        metadata["Lx"] = Lx
        metadata["shape"] = (
            nframes,
            Ly,
            Lx,
        )  # NOTE: the chunk's own dims: ScanImage reports the ROI height (550) where the stitched frame is 542

        _BIN_WRITERS[key] = BinArray(
            filename=key,
            shape=(nframes, Ly, Lx),
            dtype=np.int16,
        )
        _BIN_OFFSETS[key] = 0
        first_write = True

    bf = _BIN_WRITERS[key]
    off = _BIN_OFFSETS[key]

    bf[off : off + data.shape[0]] = data
    bf.flush()
    _BIN_OFFSETS[key] = off + data.shape[0]

    if first_write:
        write_ops(metadata, fname, **kwargs)


def _write_npy(path, data, *, overwrite: bool = False, metadata=None, **kwargs):
    """
    Write data to a .npy file with chunked/streaming support.

    Uses memory-mapped file for efficient chunked writing.
    Metadata is embedded in the file using np.savez format (stored as .npy).
    """
    if metadata is None:
        metadata = {}

    fname = Path(path).with_suffix(".npy")
    fname.parent.mkdir(parents=True, exist_ok=True)

    key = str(fname)

    if key in _NPY_ARRAYS and not fname.exists():
        _NPY_ARRAYS.pop(key, None)
        _NPY_OFFSETS.pop(key, None)
        _NPY_METADATA.pop(key, None)

    if overwrite and key not in _NPY_ARRAYS and fname.exists():
        fname.unlink()

    if key not in _NPY_ARRAYS:
        nframes = metadata.get("nframes") or metadata.get("num_frames")
        if nframes is None:
            raise ValueError("Metadata must contain 'nframes' or 'num_frames'.")

        h, w = data.shape[-2], data.shape[-1]
        shape = (int(nframes), h, w)

        temp_fname = fname.with_suffix(".npy.tmp")

        mmap = np.lib.format.open_memmap(
            temp_fname,
            mode="w+",
            dtype=data.dtype,
            shape=shape,
        )
        _NPY_ARRAYS[key] = mmap
        _NPY_OFFSETS[key] = 0
        _NPY_METADATA[key] = _make_json_serializable(metadata)

    mmap = _NPY_ARRAYS[key]
    off = _NPY_OFFSETS[key]

    mmap[off : off + data.shape[0]] = data
    mmap.flush()
    _NPY_OFFSETS[key] = off + data.shape[0]


def _write_h5(path, data, *, overwrite=True, metadata=None, **kwargs):
    if metadata is None:
        metadata = {}

    filename = Path(path).with_suffix(".h5")
    dataset_name = kwargs.get("dataset_name") or "mov"

    if filename not in _H5_INITIALIZED:
        nframes = metadata.get("num_frames")
        if nframes is None:
            raise ValueError("Metadata must contain 'nframes' or 'nun_frames'.")
        h, w = data.shape[-2:]
        with h5py.File(filename, "w" if overwrite else "a") as f:
            f.create_dataset(
                dataset_name,
                shape=(nframes, h, w),
                maxshape=(None, h, w),
                chunks=(
                    1,
                    h,
                    w,
                ),  # one Y x X plane per chunk, one plane per fixed-(t, c, z) read
                dtype=data.dtype,
                compression=None,
            )
            if metadata:
                for k, v in metadata.items():
                    f.attrs[k] = v if np.isscalar(v) else str(v)

        _H5_INITIALIZED[filename] = True
        _H5_OFFSETS[filename] = 0
        _H5_DATASET_NAMES[filename] = dataset_name

    offset = _H5_OFFSETS[filename]
    active_name = _H5_DATASET_NAMES[filename]

    with h5py.File(filename, "a") as f:
        f[active_name][offset : offset + data.shape[0]] = data

    _H5_OFFSETS[filename] = offset + data.shape[0]


def _build_imagej_metadata(metadata: dict, shape: tuple) -> tuple[dict, tuple]:
    """
    Build ImageJ-compatible metadata dict and resolution tuple.

    ImageJ expects metadata in a specific format in the ImageDescription tag.
    The key fields are:
    - spacing: z-step size in units
    - unit: physical unit (e.g., 'um')
    - finterval: frame interval in seconds
    - axes: dimension order (e.g., 'TYX', 'TZYX')
    - min/max: display range (optional)
    - loop: animation loop flag (optional)

    The resolution tuple is (pixels_per_unit_x, pixels_per_unit_y), which is
    the inverse of micrometers per pixel.

    Parameters
    ----------
    metadata : dict
        Source metadata dict with imaging parameters.
    shape : tuple
        Array shape (T, Y, X) or (T, Z, Y, X).

    Returns
    -------
    tuple[dict, tuple]
        (imagej_metadata, resolution) ready for tifffile.imwrite(imagej=True).
    """
    from mbo_utilities.metadata import get_param, get_voxel_size

    vs = get_voxel_size(metadata)
    dx, dy, dz = vs.dx, vs.dy, vs.dz

    res_x = 1.0 / dx if dx and dx > 0 else 1.0  # ImageJ resolution is pixels per unit
    res_y = 1.0 / dy if dy and dy > 0 else 1.0
    resolution = (res_x, res_y)

    ij_meta = {
        "unit": "um",
        "loop": False,
    }

    ij_meta["frames"] = shape[0]
    ij_meta["slices"] = shape[2]
    ij_meta["channels"] = shape[1]

    if dz is not None:
        ij_meta["spacing"] = dz

    fs = get_param(metadata, "fs")
    if fs and fs > 0:
        ij_meta["finterval"] = 1.0 / float(fs)

    if "min" in metadata:
        ij_meta["min"] = float(metadata["min"])
    if "max" in metadata:
        ij_meta["max"] = float(metadata["max"])

    return ij_meta, resolution


def _write_tiff(path, data, overwrite=True, metadata=None, imagej=True, **kwargs):
    """
    Write data to TIFF file with optional ImageJ hyperstack compatibility.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : np.ndarray
        Image data to write.
    overwrite : bool
        Whether to overwrite existing file.
    metadata : dict
        Metadata dict containing imaging parameters.
    imagej : bool
        If True (default), write ImageJ-compatible TIFF with proper metadata
        that Fiji/ImageJ can auto-detect (resolution, spacing, frame interval).
        If False, write standard tifffile format with JSON metadata.
    """
    if metadata is None:
        metadata = {}

    filename = Path(path).with_suffix(".tif")

    is_new_session = filename not in _TIFF_WRITERS

    if is_new_session:
        if filename.exists() and not overwrite:
            logger.warning(
                f"File {filename} already exists and overwrite=False. Skipping write."
            )
            return

        if filename.exists() and overwrite:
            import time

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    filename.unlink()
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                        import gc

                        gc.collect()
                    else:
                        raise

        _TIFF_IMAGEJ_MODE[filename] = imagej

        if imagej:
            _TIFF_WRITERS[filename] = TiffWriter(filename, bigtiff=True, imagej=True)
        else:
            _TIFF_WRITERS[filename] = TiffWriter(filename, bigtiff=True)
        _TIFF_FIRST_WRITE[filename] = True

    writer = _TIFF_WRITERS[filename]
    is_first = _TIFF_FIRST_WRITE.get(filename, True)
    use_imagej = _TIFF_IMAGEJ_MODE.get(filename, imagej)

    if use_imagej:
        ij_meta = None
        resolution = None
        extratags = None

        if is_first:
            target_shape = metadata.get("shape", data.shape)
            ij_meta, resolution = _build_imagej_metadata(metadata, target_shape)

            import json

            from .metadata.base import strip_for_export

            # NOTE: regPC alone would balloon the Info tag past 500 MB
            json_meta = _make_json_serializable(strip_for_export(metadata))
            json_bytes = json.dumps(json_meta).encode("utf-8")
            extratags = [(50839, 2, len(json_bytes), json_bytes, True)]

        data_5d = np.moveaxis(data, 1, 2)

        for frame in data_5d:
            writer.write(
                frame,
                contiguous=True,
                photometric="minisblack",
                resolution=resolution if is_first else None,
                metadata=ij_meta if is_first else None,
                extratags=extratags if is_first else None,
            )
            is_first = False
    else:
        for frame in data:
            writer.write(
                frame,
                contiguous=True,
                photometric="minisblack",
                metadata=_make_json_serializable(metadata) if is_first else {},
            )
            is_first = False

    _TIFF_FIRST_WRITE[filename] = False


def _write_volumetric_tiff(
    data,
    path: Path,
    metadata: dict | None = None,
    planes: list | None = None,
    frames: list | None = None,
    channels: list | None = None,
    overwrite: bool = True,
    target_chunk_mb: int = 50,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
    output_suffix: str | None = None,
    pyramid: bool = False,
    pyramid_max_layers: int = 4,
    pyramid_method: str = "mean",
):
    """
    Write volumetric TZYX or TCYX data as single ImageJ hyperstack tiff.

    Parameters
    ----------
    data : array-like
        data with shape (T, Z, Y, X), (T, C, Y, X), (T, Y, X), or (Z, Y, X)
    path : Path
        output directory (filename auto-generated from dims)
    metadata : dict
        imaging metadata for resolution, spacing, etc.
    planes : list | None
        z-plane selection (1-based indices). None = all planes.
    frames : list | None
        timepoint selection (1-based indices). None = all frames.
    channels : list | None
        channel selection (1-based indices). None = all channels.
    overwrite : bool
        overwrite existing files
    target_chunk_mb : int
        chunk size for streaming writes
    progress_callback : callable
        progress callback(fraction, message)
    show_progress : bool
        show tqdm progress bar
    debug : bool
        verbose logging
    """
    from mbo_utilities.arrays.features import (
        ArraySlicing,
        OutputFilename,
        read_chunk,
    )
    from mbo_utilities.arrays.features._pyramid import (
        PyramidConfig,
        compute_pyramid_shapes,
        downsample_block,
    )

    if metadata is None:
        metadata = {}

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    selections = {}
    if frames is not None:
        selections["T"] = [frames] if isinstance(frames, int) else frames
    if planes is not None:
        selections["Z"] = [planes] if isinstance(planes, int) else planes
    if channels is not None:
        selections["C"] = [channels] if isinstance(channels, int) else channels

    slicing = ArraySlicing.from_array(data, selections=selections, one_based=True)

    suffix = output_suffix or ""
    output_fn = OutputFilename.from_array(
        data, planes=planes, frames=frames, channels=channels, suffix=suffix
    )
    filename = path / output_fn.build(".tif")

    if filename.exists() and not overwrite:
        logger.warning(f"File {filename} exists and overwrite=False. Skipping.")
        return filename

    if filename.exists():
        filename.unlink()

    n_frames = slicing.selections["T"].count if "T" in slicing.selections else 1
    n_planes = slicing.selections["Z"].count if "Z" in slicing.selections else 1
    n_channels = slicing.selections["C"].count if "C" in slicing.selections else 1
    Ly, Lx = slicing.spatial_shape

    target_shape = (n_frames, n_planes, n_channels, Ly, Lx)

    from mbo_utilities.arrays.features import get_dims
    from mbo_utilities.metadata import OutputMetadata

    source_dims = get_dims(data)

    output_selections = {}
    if "T" in slicing.selections:
        output_selections["T"] = slicing.selections["T"].indices
    if "Z" in slicing.selections:
        output_selections["Z"] = slicing.selections["Z"].indices
    if "C" in slicing.selections:
        output_selections["C"] = slicing.selections["C"].indices

    out_meta = OutputMetadata(
        source=metadata or {},
        source_shape=data.shape,
        source_dims=source_dims,
        selections=output_selections,
    )

    md = out_meta.to_dict()
    md["shape"] = target_shape
    md["Lx"] = Lx
    md["Ly"] = Ly

    ij_meta, resolution = out_meta.to_imagej(target_shape)

    import json

    from tifffile import imagej_metadata_tag

    from .metadata.base import strip_for_export

    # NOTE: regPC alone would balloon the Info tag past 500 MB
    json_meta = _make_json_serializable(strip_for_export(md))
    json_str = json.dumps(json_meta)
    ij_extratags = imagej_metadata_tag({"Info": json_str}, "<")
    extratags = list(ij_extratags) if ij_extratags else []

    if debug:
        logger.info(f"Writing volumetric tiff: {filename}")
        logger.info(f"  Shape: {target_shape} (TZCYX)")
        logger.info(
            f"  ImageJ meta: frames={ij_meta.get('frames')}, slices={ij_meta.get('slices')}, channels={ij_meta.get('channels')}"
        )
        logger.info(
            f"  Output metadata: dz={out_meta.dz}, fs={out_meta.fs}, contiguous={out_meta.is_contiguous}"
        )
        if out_meta.z_step_factor > 1:
            logger.info(
                f"  Z-step factor: {out_meta.z_step_factor}x (saving every {out_meta.z_step_factor} plane)"
            )

    if pyramid:
        pyramid_config = PyramidConfig(
            max_layers=pyramid_max_layers,
            scale_factors=(1, 1, 1, 2, 2) if n_channels > 1 else (1, 1, 2, 2),
            method=pyramid_method,
            min_size=64,
        )
        pyramid_levels = compute_pyramid_shapes(target_shape, pyramid_config)
        if debug:
            logger.info(f"  Pyramid: {len(pyramid_levels)} levels")
    else:
        pyramid_levels = None

    with TiffWriter(filename, bigtiff=True, imagej=not pyramid, ome=pyramid) as writer:
        first_write = True

        pbar = None
        if show_progress:
            total_pages = n_frames * n_planes * n_channels
            pbar = tqdm(total=total_pages, desc="Writing TIFF", unit="pg")

        for chunk_info in slicing.iter_chunks(chunk_dim="T", target_mb=target_chunk_mb):
            chunk_data = read_chunk(data, chunk_info, slicing.dims)

            chunk_data = chunk_data.transpose(
                0, 2, 1, 3, 4
            )  # TCZYX -> TZCYX, ImageJ page order
            chunk_data = np.ascontiguousarray(chunk_data)

            chunk_5d = chunk_data

            for t in range(chunk_5d.shape[0]):
                frame_data = chunk_5d[t]

                if pyramid and pyramid_levels and len(pyramid_levels) > 1:
                    frame_pyramid = []
                    current = frame_data
                    scale_factors_frame = (1, 1, 2, 2)

                    for _ in range(1, len(pyramid_levels)):
                        current = downsample_block(
                            current, scale_factors_frame, pyramid_method
                        )
                        if current.dtype != frame_data.dtype:
                            current = current.astype(frame_data.dtype)
                        frame_pyramid.append(current)

                    writer.write(
                        frame_data,
                        contiguous=True,
                        photometric="minisblack",
                        resolution=resolution if first_write else None,
                        subifds=len(frame_pyramid),
                    )

                    for downsampled in frame_pyramid:
                        writer.write(
                            downsampled,
                            contiguous=True,
                            photometric="minisblack",
                            subfiletype=1,
                        )
                else:
                    writer.write(
                        frame_data,
                        contiguous=True,
                        photometric="minisblack",
                        resolution=resolution if first_write else None,
                        metadata=ij_meta if first_write else None,
                        extratags=extratags if first_write else None,
                    )
                first_write = False

            if pbar:
                frames_in_chunk = len(chunk_info.selections.get("T", [1]))
                pbar.update(frames_in_chunk * n_planes * n_channels)

            if progress_callback:
                progress_callback(chunk_info.progress)

        if pbar:
            pbar.close()

    if debug:
        logger.info(f"Wrote {filename} ({filename.stat().st_size / 1e9:.2f} GB)")

    return filename


def _write_volumetric_h5(
    data,
    path: Path,
    metadata: dict | None = None,
    planes: list | None = None,
    frames: list | None = None,
    channels: list | None = None,
    overwrite: bool = True,
    target_chunk_mb: int = 50,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
    output_suffix: str | None = None,
    dataset_name: str = "mov",
    compression: str | None = "gzip",
    compression_level: int = 1,
):
    """Write volumetric TZYX (or TCZYX) data into a single HDF5 file.

    This is the h5 analogue of `_write_volumetric_zarr`. Previously, h5
    output went through the per-plane streaming writer (`_write_plane`),
    which produced one `.h5` file per z-plane (`zplane01.h5`,
    `zplane03.h5`, ...). That mismatched zarr's behaviour, where the
    full volume lives in a single store. This writer puts everything into
    one `.h5` with a 4D `(T, Z, Y, X)` dataset under `dataset_name`,
    matching what mbo's H5Array reader expects on the read side.

    Parameters
    ----------
    data : array-like
        Source array. The 5D TCZYX shape is read via `read_chunk`. A singleton
        C dim is dropped before writing, giving a 4D TZYX dataset; two or more
        channels are written as 5D TCZYX instead, which `H5Array` reads back
        unchanged.
    path : Path
        Output directory. Filename is auto-generated from dim tags.
    metadata : dict | None
        Imaging metadata; serializable values land on the file's
        root attrs and on the dataset attrs.
    planes, frames, channels : list | None
        1-based selections that subset the source on the way out.
    overwrite : bool
        Replace existing file rather than failing.
    target_chunk_mb : int
        Sizing hint for the streaming read loop. Each h5 chunk on disk
        is fixed at one frame `(1, Z, Y, X)` for fast random access.
    dataset_name : str
        H5 dataset name (default 'mov' — same default as `_write_h5`
        and what mbo's H5Array reader auto-detects). Pass via the
        save-as dialog's H5 dataset-name field.
    compression : str | None
        H5 compression filter ('gzip', 'lzf', or None).
    compression_level : int
        Gzip level 0-9 (ignored for lzf/None).
    """
    import h5py

    from mbo_utilities.arrays.features import (
        ArraySlicing,
        OutputFilename,
        get_dims,
        read_chunk,
    )
    from mbo_utilities.metadata import OutputMetadata

    if metadata is None:
        metadata = {}

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    selections = {}
    if frames is not None:
        selections["T"] = [frames] if isinstance(frames, int) else frames
    if planes is not None:
        selections["Z"] = [planes] if isinstance(planes, int) else planes
    if channels is not None:
        selections["C"] = [channels] if isinstance(channels, int) else channels

    slicing = ArraySlicing.from_array(data, selections=selections, one_based=True)

    suffix = output_suffix or ""
    output_fn = OutputFilename.from_array(
        data, planes=planes, frames=frames, channels=channels, suffix=suffix
    )
    filename = path / output_fn.build(".h5")

    if filename.exists() and not overwrite:
        logger.warning(f"File {filename} exists and overwrite=False. Skipping.")
        return filename
    if filename.exists():
        filename.unlink()

    n_frames = slicing.selections["T"].count if "T" in slicing.selections else 1
    n_planes = slicing.selections["Z"].count if "Z" in slicing.selections else 1
    n_channels = slicing.selections["C"].count if "C" in slicing.selections else 1

    Ly, Lx = slicing.spatial_shape
    Ly_out, Lx_out = int(Ly), int(Lx)
    n_frames = int(n_frames)
    n_planes = int(n_planes)
    n_channels = int(n_channels)

    multichannel = n_channels > 1
    if multichannel:
        out_dims = ("T", "C", "Z", "Y", "X")
        target_shape = (n_frames, n_channels, n_planes, Ly_out, Lx_out)
        inner_chunk = (1, 1, 1, Ly_out, Lx_out)
    else:
        out_dims = ("T", "Z", "Y", "X")
        target_shape = (n_frames, n_planes, Ly_out, Lx_out)
        inner_chunk = (1, 1, Ly_out, Lx_out)

    source_dims = get_dims(data)
    output_selections: dict[str, list[int]] = {}
    if "T" in slicing.selections:
        output_selections["T"] = list(slicing.selections["T"].indices)
    if "Z" in slicing.selections:
        output_selections["Z"] = list(slicing.selections["Z"].indices)
    if "C" in slicing.selections:
        output_selections["C"] = list(slicing.selections["C"].indices)

    out_meta = OutputMetadata(
        source=metadata or {},
        source_shape=data.shape,
        source_dims=source_dims,
        selections=output_selections,
    )

    md = out_meta.to_dict()
    md["shape"] = target_shape
    md["dataset_name"] = dataset_name
    md["dims"] = out_dims

    if debug:
        logger.info(f"Writing volumetric h5: {filename}")
        logger.info(f"  Shape: {target_shape} ({''.join(out_dims)})")
        logger.info(f"  Dataset: /{dataset_name}")
        logger.info(
            f"  Output metadata: dz={out_meta.dz}, fs={out_meta.fs}, contiguous={out_meta.is_contiguous}"
        )

    h5_compression = None
    h5_compression_opts = None
    if compression is not None and compression_level > 0:
        if compression.lower() == "gzip":
            h5_compression = "gzip"
            h5_compression_opts = int(compression_level)
        elif compression.lower() == "lzf":
            h5_compression = "lzf"
        else:
            logger.warning(
                f"Unknown h5 compression '{compression}'; falling back to none."
            )

    pbar = None
    if show_progress:
        pbar = tqdm(total=n_frames, desc="Writing H5", unit="frames")

    try:
        with h5py.File(filename, "w") as f:
            dset = f.create_dataset(
                dataset_name,
                shape=target_shape,
                maxshape=(None, *target_shape[1:]),
                chunks=inner_chunk,
                dtype=data.dtype,
                compression=h5_compression,
                compression_opts=h5_compression_opts,
            )

            from .metadata.base import strip_for_export

            serializable_md = _make_json_serializable(strip_for_export(md))
            for k, v in serializable_md.items():
                if v is None:
                    continue
                try:
                    f.attrs[k] = v if np.isscalar(v) else str(v)
                except (TypeError, ValueError):
                    f.attrs[k] = str(v)
            dset.attrs["dims"] = list(out_dims)

            t_offset = 0
            for chunk_info in slicing.iter_chunks(
                chunk_dim="T", target_mb=target_chunk_mb
            ):
                chunk_data = read_chunk(data, chunk_info, slicing.dims)
                if chunk_data.ndim == 5 and not multichannel:
                    chunk_data = chunk_data[:, 0, :, :, :]
                chunk_data = np.ascontiguousarray(chunk_data)

                t_start = t_offset
                t_end = t_start + chunk_data.shape[0]
                dset[t_start:t_end] = chunk_data
                t_offset = t_end

                if pbar:
                    pbar.update(chunk_data.shape[0])
                if progress_callback:
                    progress_callback(chunk_info.progress)
    finally:
        if pbar:
            pbar.close()

    if debug:
        logger.info(f"Wrote {filename} ({filename.stat().st_size / 1e9:.2f} GB)")

    return filename


def _build_inner_codecs(
    name: str, level: int, itemsize: int, shuffle: str | None = None
) -> list:
    from zarr.codecs import BytesCodec, GzipCodec

    name = (name or "none").lower()
    if name == "none" or level == 0:
        return [BytesCodec()]
    if name == "gzip":
        return [BytesCodec(), GzipCodec(level=level)]
    if name == "zstd":
        from zarr.codecs import ZstdCodec

        return [BytesCodec(), ZstdCodec(level=level)]
    if name in ("blosc-lz4", "blosc-zstd"):
        from zarr.codecs import BloscCodec

        cname = "lz4" if name == "blosc-lz4" else "zstd"
        blosc_kwargs = {
            "cname": cname,
            "clevel": level,
            "typesize": itemsize,
        }
        if shuffle is not None:
            blosc_kwargs["shuffle"] = shuffle
        return [BytesCodec(), BloscCodec(**blosc_kwargs)]
    raise ValueError(
        f"unknown compressor {name!r}; expected one of "
        "'none', 'gzip', 'zstd', 'blosc-lz4', 'blosc-zstd'"
    )


def _write_volumetric_zarr(
    data,
    path: Path,
    metadata: dict | None = None,
    planes: list | None = None,
    frames: list | None = None,
    channels: list | None = None,
    overwrite: bool = True,
    target_chunk_mb: int = 2048,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
    output_suffix: str | None = None,
    sharded: bool = True,
    compression_level: int = 1,
    compressor: str = "gzip",
    shuffle: str | None = None,
    pyramid: bool = False,
    pyramid_max_layers: int = 4,
    pyramid_method: str = "median",
):
    """
    Write volumetric TZYX data as single OME-NGFF zarr.

    Parameters
    ----------
    data : array-like
        data with shape (T, Z, Y, X), (T, C, Y, X), (T, Y, X), or (Z, Y, X)
    path : Path
        output directory (filename auto-generated from dims)
    metadata : dict
        imaging metadata for resolution, spacing, etc.
    planes : list | None
        z-plane selection (1-based indices). None = all planes.
    frames : list | None
        timepoint selection (1-based indices). None = all frames.
    channels : list | None
        channel selection (1-based indices). None = all channels.
    overwrite : bool
        overwrite existing files
    target_chunk_mb : int
        Maximum shard byte budget in MB. The shard layout aims for one shard
        per ``(c, z)`` (i.e. ``shards=(T, 1, 1, Y, X)`` for 5D, ``(T, 1, Y, X)``
        for 4D) so a fixed-``(c, z)`` T-scrub opens a single file. If
        ``T * Y * X * itemsize`` would exceed ``target_chunk_mb``, the shard's
        T-extent is truncated and additional shards are added along T. The
        2 GB default keeps typical multi-thousand-frame recordings to one
        shard per ``(c, z)``.
    progress_callback : callable
        progress callback(fraction, message)
    show_progress : bool
        show tqdm progress bar
    debug : bool
        verbose logging
    output_suffix : str | None
        suffix for output filename
    sharded : bool
        use zarr v3 sharding codec
    compression_level : int
        compression level. meaning depends on `compressor`:
        gzip: 0=none, 1-9. zstd: 1-22. blosc: 1-9. ignored for "none".
    compressor : str
        compression codec. one of "none", "gzip", "zstd", "blosc-lz4",
        "blosc-zstd". default "gzip".
    pyramid : bool
        generate multi-resolution pyramid (default False).
        enables napari multiscale viewing and faster navigation.
    pyramid_max_layers : int
        max additional resolution levels (default 4 = levels 0-4).
        per-level factors follow the webknossos anisotropic-mag algorithm
        (Z is downsampled at deep levels for anisotropic volumes).
    pyramid_method : str
        downsampling reducer: "median" (default, webknossos intensity),
        "mode" (webknossos labels/masks), "mean", "nearest", "gaussian".
    """
    import zarr
    from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec

    from mbo_utilities.arrays.features import (
        ArraySlicing,
        OutputFilename,
        get_dims,
        read_chunk,
    )
    from mbo_utilities.arrays.features._pyramid import downsample_block
    from mbo_utilities.arrays.isoview.consolidate import _compute_anisotropic_mags
    from mbo_utilities.metadata import OutputMetadata

    if metadata is None:
        metadata = {}

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    selections = {}
    if frames is not None:
        selections["T"] = [frames] if isinstance(frames, int) else frames
    if planes is not None:
        selections["Z"] = [planes] if isinstance(planes, int) else planes
    if channels is not None:
        selections["C"] = [channels] if isinstance(channels, int) else channels

    slicing = ArraySlicing.from_array(data, selections=selections, one_based=True)

    suffix = output_suffix or ""
    output_fn = OutputFilename.from_array(
        data, planes=planes, frames=frames, channels=channels, suffix=suffix
    )
    filename = path / output_fn.build(".zarr")

    if filename.exists() and not overwrite:
        logger.warning(f"File {filename} exists and overwrite=False. Skipping.")
        return filename

    if filename.exists():
        shutil.rmtree(filename)

    n_frames = slicing.selections["T"].count if "T" in slicing.selections else 1
    n_planes = slicing.selections["Z"].count if "Z" in slicing.selections else 1
    n_channels = slicing.selections["C"].count if "C" in slicing.selections else 1
    Ly, Lx = slicing.spatial_shape

    Ly_out, Lx_out = int(Ly), int(Lx)
    n_frames = int(n_frames)
    n_planes = int(n_planes)
    n_channels = int(n_channels)

    output_5d = "C" in slicing.selections and n_channels > 1
    if output_5d:
        target_shape = (n_frames, n_channels, n_planes, Ly_out, Lx_out)
    else:
        target_shape = (n_frames, n_planes, Ly_out, Lx_out)

    source_dims = get_dims(data)

    output_selections: dict[str, list[int]] = {}
    if "T" in slicing.selections:
        output_selections["T"] = list(slicing.selections["T"].indices)
    if "Z" in slicing.selections:
        output_selections["Z"] = list(slicing.selections["Z"].indices)
    if "C" in slicing.selections:
        output_selections["C"] = list(slicing.selections["C"].indices)

    out_meta = OutputMetadata(
        source=metadata or {},
        source_shape=data.shape,
        source_dims=source_dims,
        selections=output_selections,
    )

    md = out_meta.to_dict()
    md["shape"] = target_shape

    if debug:
        logger.info(f"Writing volumetric zarr: {filename}")
        logger.info(f"  Shape: {target_shape} ({'TCZYX' if output_5d else 'TZYX'})")
        logger.info(
            f"  Output metadata: dz={out_meta.dz}, fs={out_meta.fs}, contiguous={out_meta.is_contiguous}"
        )
        if out_meta.z_step_factor > 1:
            logger.info(
                f"  Z-step factor: {out_meta.z_step_factor}x (saving every {out_meta.z_step_factor} plane)"
            )

    itemsize = np.dtype(data.dtype).itemsize
    if output_5d:
        inner_chunk = (1, 1, 1, Ly_out, Lx_out)
    else:
        inner_chunk = (1, 1, Ly_out, Lx_out)

    bytes_per_yx = Ly_out * Lx_out * itemsize

    inner_codecs = _build_inner_codecs(
        compressor, compression_level, itemsize, shuffle=shuffle
    )

    if sharded:
        target_bytes = target_chunk_mb * 1024 * 1024
        max_shard_t = max(1, target_bytes // max(1, bytes_per_yx))
        shard_t = min(n_frames, max_shard_t)
        if (
            output_5d
        ):  # one shard per (c, z) over all T: a fixed-(c, z) scrub opens one file
            shard_shape = (shard_t, 1, 1, Ly_out, Lx_out)
        else:
            shard_shape = (shard_t, 1, Ly_out, Lx_out)

        codec = ShardingCodec(
            chunk_shape=inner_chunk,
            codecs=inner_codecs,
            index_codecs=[BytesCodec(), Crc32cCodec()],
        )
        codecs = [codec]
        chunks = shard_shape
    else:
        codecs = inner_codecs
        chunks = inner_chunk

    root = zarr.open_group(str(filename), mode="w", zarr_format=3)

    output_dims = ("T", "C", "Z", "Y", "X") if output_5d else ("T", "Z", "Y", "X")
    ome_meta = out_meta.to_ome_ngff(dims=output_dims)
    base_scale = ome_meta["coordinateTransformations"][0]["scale"]
    dimension_names = [d.lower() for d in output_dims]

    md["dims"] = output_dims

    if pyramid:
        nz_p, ny_p, nx_p = (int(s) for s in target_shape[-3:])
        voxel_zyx = tuple(float(v) for v in base_scale[-3:])
        pyramid_mags = _compute_anisotropic_mags(
            voxel_zyx, (nz_p, ny_p, nx_p), pyramid_max_layers, min_size=64
        )
        level_spatial = [
            (max(1, nz_p // mz), max(1, ny_p // my), max(1, nx_p // mx))
            for mz, my, mx in pyramid_mags
        ]
        if debug:
            logger.info(f"  Pyramid: {len(pyramid_mags)} levels, mags {pyramid_mags}")
    else:
        pyramid_mags = None
        level_spatial = None

    z = zarr.create(
        store=root.store,
        path="0",
        shape=target_shape,
        chunks=chunks,
        dtype=data.dtype,
        codecs=codecs,
        dimension_names=dimension_names,
        overwrite=True,
    )

    multiscale_metadata = {
        "method": pyramid_method,
        "description": (
            f"Downsampled with mbo_utilities using {pyramid_method} method"
            if pyramid_mags
            else "Single resolution level (no pyramid)"
        ),
    }

    if pyramid_mags:
        datasets = []
        for lvl_idx, (mz, my, mx) in enumerate(pyramid_mags):
            physical_scale = list(base_scale)
            physical_scale[-3] = base_scale[-3] * mz
            physical_scale[-2] = base_scale[-2] * my
            physical_scale[-1] = base_scale[-1] * mx
            datasets.append(
                {
                    "path": str(lvl_idx),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": physical_scale}
                    ],
                }
            )

        multiscales = [
            {
                "name": metadata.get("name", filename.stem),
                "axes": ome_meta["axes"],
                "datasets": datasets,
                "type": pyramid_method,
                "metadata": multiscale_metadata,
            }
        ]
    else:
        multiscales = [
            {
                "name": metadata.get("name", filename.stem),
                "axes": ome_meta["axes"],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": ome_meta[
                            "coordinateTransformations"
                        ],
                    }
                ],
                "type": "none",
                "metadata": multiscale_metadata,
            }
        ]

    ome_content = {
        "version": "0.5",  # NOTE: version lives here only; a copy in the multiscale entry is invalid in 0.5
        "multiscales": multiscales,
    }

    root.attrs["ome"] = ome_content

    from .metadata.base import strip_for_export

    serializable_md = _make_json_serializable(strip_for_export(md))
    for k, v in serializable_md.items():
        if (
            k in ("ome", "multiscales", "_ome_time_scale")
        ):  # NOTE: a carried source OME block would keep the source time scale on a decimated re-save
            continue
        root.attrs[k] = v

    z.attrs["scale"] = base_scale

    pbar = None
    total_work = n_frames * (len(pyramid_mags) if pyramid_mags else 1)
    if show_progress:
        pbar = tqdm(total=total_work, desc="Writing Zarr", unit="frames")

    t_offset = 0

    try:
        for chunk_info in slicing.iter_chunks(chunk_dim="T", target_mb=target_chunk_mb):
            chunk_data = read_chunk(data, chunk_info, slicing.dims)

            if not output_5d and chunk_data.ndim == 5:
                chunk_data = chunk_data[:, 0, :, :, :]

            chunk_data = np.ascontiguousarray(chunk_data)

            t_start = t_offset
            t_end = t_start + chunk_data.shape[0]

            z[t_start:t_end] = chunk_data

            t_offset = t_end

            if pbar:
                pbar.update(chunk_data.shape[0])

            if progress_callback:
                progress_callback(
                    chunk_info.progress / (len(pyramid_mags) if pyramid_mags else 1)
                )
    finally:
        if pbar:
            pbar.close()

    _sp_attrs = _scanphase_zarr_attrs(data, frames, planes, channels)
    if _sp_attrs is not None:
        root.attrs["scanphase"] = _make_json_serializable(_sp_attrs)

    if pyramid_mags and len(pyramid_mags) > 1:
        if show_progress:
            pbar = tqdm(
                total=len(pyramid_mags) - 1,
                desc="Building pyramid",
                unit="levels",
            )

        for lvl_idx in range(1, len(pyramid_mags)):
            prev_path = str(lvl_idx - 1)

            prev_z = root[prev_path]
            prev_data = prev_z[:]

            pz, py, px = level_spatial[lvl_idx - 1]
            cz, cy, cx = level_spatial[lvl_idx]
            zf, yf, xf = max(1, pz // cz), max(1, py // cy), max(1, px // cx)
            factor = (1, 1, zf, yf, xf) if output_5d else (1, zf, yf, xf)

            level_data = downsample_block(prev_data, factor, pyramid_method)

            lvl_shape = level_data.shape
            lvl_Y = lvl_shape[-2]
            lvl_X = lvl_shape[-1]
            if output_5d:
                lvl_inner = (1, 1, 1, lvl_Y, lvl_X)
            else:
                lvl_inner = (1, 1, lvl_Y, lvl_X)

            lvl_inner_codecs = _build_inner_codecs(
                compressor, compression_level, itemsize, shuffle=shuffle
            )

            if sharded:
                lvl_bytes_per_yx = lvl_Y * lvl_X * itemsize
                lvl_target_bytes = target_chunk_mb * 1024 * 1024
                lvl_max_shard_t = max(1, lvl_target_bytes // max(1, lvl_bytes_per_yx))
                lvl_shard_t = min(lvl_shape[0], lvl_max_shard_t)
                if output_5d:
                    lvl_shard = (lvl_shard_t, 1, 1, lvl_Y, lvl_X)
                else:
                    lvl_shard = (lvl_shard_t, 1, lvl_Y, lvl_X)
                lvl_codec = ShardingCodec(
                    chunk_shape=lvl_inner,
                    codecs=lvl_inner_codecs,
                    index_codecs=[BytesCodec(), Crc32cCodec()],
                )
                lvl_codecs = [lvl_codec]
                lvl_chunks = lvl_shard
            else:
                lvl_codecs = lvl_inner_codecs
                lvl_chunks = lvl_inner

            lvl_z = zarr.create(
                store=root.store,
                path=str(lvl_idx),
                shape=lvl_shape,
                chunks=lvl_chunks,
                dtype=data.dtype,
                codecs=lvl_codecs,
                dimension_names=dimension_names,
                overwrite=True,
            )

            lvl_z[:] = level_data

            mz, my, mx = pyramid_mags[lvl_idx]
            physical_scale = list(base_scale)
            physical_scale[-3] = base_scale[-3] * mz
            physical_scale[-2] = base_scale[-2] * my
            physical_scale[-1] = base_scale[-1] * mx
            lvl_z.attrs["scale"] = physical_scale

            if debug:
                logger.info(f"  Wrote level {lvl_idx}: {lvl_shape}")

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    if debug:
        total_size = sum(f.stat().st_size for f in filename.rglob("*") if f.is_file())
        logger.info(f"Wrote {filename} ({total_size / 1e9:.2f} GB)")

    return filename


def _write_zarr(
    path,
    data,
    *,
    overwrite=True,
    metadata=None,
    **kwargs,
):
    sharded = kwargs.get("sharded", True)
    ome = kwargs.get("ome", True)
    level = kwargs.get("level", 1)
    shard_frames = kwargs.get("shard_frames")
    chunk_shape = kwargs.get("chunk_shape")

    if metadata is None:
        metadata = {}

    filename = Path(path)

    if overwrite and filename not in _ZARR_ARRAYS and filename.exists():
        shutil.rmtree(filename)

    if filename not in _ZARR_ARRAYS:
        import zarr
        from zarr.codecs import BytesCodec, Crc32cCodec, GzipCodec, ShardingCodec

        nframes = int(metadata["num_frames"])
        h, w = data.shape[-2:]

        if level == 0:
            inner_codecs = [BytesCodec()]
        else:
            inner_codecs = [BytesCodec(), GzipCodec(level=level)]

        if sharded:
            if chunk_shape is not None:
                inner = chunk_shape
            else:
                inner = (1, h, w)

            inner_t = inner[0]

            if shard_frames is not None:
                shard_t = min(nframes, shard_frames)
            else:
                shard_t = nframes

            if (
                inner_t > 1 and shard_t % inner_t != 0
            ):  # zarr requires the shard to be a multiple of the inner chunk
                shard_t = (shard_t // inner_t) * inner_t
                if shard_t == 0:
                    shard_t = inner_t

            outer = (shard_t, h, w)

            codec = ShardingCodec(
                chunk_shape=inner,
                codecs=inner_codecs,
                index_codecs=[BytesCodec(), Crc32cCodec()],
            )
            codecs = [codec]
            chunks = outer
        else:
            codecs = inner_codecs
            chunks = chunk_shape if chunk_shape is not None else (1, h, w)

        if ome:
            root = zarr.open_group(str(filename), mode="w", zarr_format=3)

            array_codecs = codecs
            array_chunks = chunks

            z = zarr.create(
                store=root.store,
                path="0",
                shape=(nframes, h, w),
                chunks=array_chunks,
                dtype=data.dtype,
                codecs=array_codecs,
                dimension_names=["t", "y", "x"],
                overwrite=True,
            )

            ome_metadata = _build_ome_metadata(
                shape=(nframes, h, w),
                dtype=data.dtype,
                metadata=metadata or {},
            )

            for key, value in ome_metadata.items():
                root.attrs[key] = value

            _ZARR_GROUPS[filename] = root
        else:
            z = zarr.create(
                store=str(filename),
                shape=(nframes, h, w),
                chunks=chunks,
                dtype=data.dtype,
                codecs=codecs,
                overwrite=True,
            )

            serializable_metadata = _make_json_serializable(metadata or {})
            for k, v in serializable_metadata.items():
                z.attrs[k] = v

        _ZARR_ARRAYS[filename] = z
        _ZARR_OFFSETS[filename] = 0

    z = _ZARR_ARRAYS[filename]
    offset = _ZARR_OFFSETS[filename]

    z[offset : offset + data.shape[0]] = data
    _ZARR_OFFSETS[filename] = offset + data.shape[0]


def _try_generic_writers(
    data: Any,
    outpath: str | Path,
    overwrite: bool = True,
    metadata: dict | None = None,
    dataset_name: str | None = None,
):
    import gc
    import shutil
    import time

    if metadata is None:
        metadata = {}

    outpath = Path(outpath)
    if outpath.exists():
        if not overwrite:
            raise FileExistsError(f"{outpath} already exists and overwrite=False")
        if outpath.is_dir():
            shutil.rmtree(outpath)
        else:
            gc.collect()
            try:
                outpath.unlink()
            except PermissionError:
                time.sleep(0.1)
                gc.collect()
                outpath.unlink()

    if outpath.suffix.lower() in {".npy", ".npz"}:
        if metadata is None:
            np.save(outpath, data)
        else:
            np.savez(outpath, data=data, metadata=_convert_paths_to_strings(metadata))
    elif outpath.suffix.lower() in {".tif", ".tiff"}:
        target_shape = metadata.get("shape", data.shape)
        ij_meta, resolution = _build_imagej_metadata(metadata, target_shape)
        tiff_imwrite(
            outpath,
            data,
            imagej=True,
            resolution=resolution,
            metadata=ij_meta,
            photometric="minisblack",
        )
    elif outpath.suffix.lower() in {".h5", ".hdf5"}:
        h5_dataset = dataset_name if dataset_name is not None else "mov"
        with h5py.File(outpath, "w" if overwrite else "a") as f:
            f.create_dataset(h5_dataset, data=data)
            if metadata:
                for k, v in metadata.items():
                    f.attrs[k] = v if np.isscalar(v) else str(v)
    elif outpath.suffix.lower() == ".bin":
        arr = np.asarray(data)
        if arr.dtype != np.int16:
            arr = arr.astype(np.int16)

        with open(outpath, "wb") as f:
            arr.tofile(f)

        if metadata:
            ops = metadata.copy()
            ops["Ly"] = arr.shape[-2]
            ops["Lx"] = arr.shape[-1]
            nt = arr.shape[0]
            ops["nframes"] = nt
            ops["num_frames"] = nt
            ops["num_timepoints"] = nt
            ops["n_frames"] = nt
            ops["timepoints"] = nt
            ops["T"] = nt
            ops["nt"] = nt
            np.save(
                outpath.parent / "ops.npy",
                normalize_ops_arrays(_convert_paths_to_strings(ops)),
            )
    elif outpath.suffix.lower() == ".zarr":
        import zarr
        from zarr.codecs import (
            BytesCodec,
            Crc32cCodec,
            GzipCodec,
            ShardingCodec,
        )

        arr = np.asarray(data)

        if arr.ndim < 3:
            create_kwargs = dict(
                chunks=arr.shape,
                codecs=[BytesCodec(), GzipCodec(level=5)],
            )
        else:
            Y, X = arr.shape[-2], arr.shape[-1]
            inner = (1,) * (arr.ndim - 2) + (Y, X)
            shard = (arr.shape[0],) + (1,) * (arr.ndim - 3) + (Y, X)
            create_kwargs = dict(
                chunks=shard,
                codecs=[
                    ShardingCodec(
                        chunk_shape=inner,
                        codecs=[BytesCodec()],
                        index_codecs=[BytesCodec(), Crc32cCodec()],
                    )
                ],
            )

        z = zarr.create(
            store=str(outpath),
            shape=arr.shape,
            dtype=arr.dtype,
            overwrite=True,
            zarr_format=3,
            **create_kwargs,
        )
        z[:] = arr

        if metadata:
            for k, v in metadata.items():
                try:
                    z.attrs[k] = (
                        v
                        if np.isscalar(v) or isinstance(v, (list, dict, str))
                        else str(v)
                    )
                except Exception:
                    z.attrs[k] = str(v)
    else:
        raise ValueError(f"Unsupported file extension: {outpath.suffix}")


def write_ops(metadata, raw_filename, **kwargs):
    """
    Write metadata to an ops file alongside the given filename.

    This creates a Suite2p-compatible ops.npy file from the provided metadata.
    The ops file is used by Suite2p for processing configuration.

    Parameters
    ----------
    metadata : dict
        Must contain 'shape' key with (T, Y, X) dimensions.
        Optional keys: 'pixel_resolution', 'frame_rate', 'fs', 'dx', 'dy', 'dz'.
    raw_filename : str or Path
        Path to the data file (e.g., data_raw.bin). The ops.npy will be
        written to the same directory.
    **kwargs
        Additional arguments. 'structural=True' indicates channel 2 data.
    """
    logger.debug(f"Writing ops file for {raw_filename} with metadata: {metadata}")
    if not isinstance(raw_filename, (str, Path)):
        raise TypeError(f"raw_filename must be str or Path, got {type(raw_filename)}")
    filename = Path(raw_filename).expanduser().resolve()

    structural = kwargs.get("structural", False)
    chan = 2 if structural or "data_chan2.bin" in str(filename) else 1
    logger.debug(f"Detected channel {chan}")

    # NOTE: never is_file(): a file just created and not yet flushed reports False
    root = filename.parent if filename.suffix or not filename.is_dir() else filename
    ops_path = root / "ops.npy"
    logger.info(f"Writing ops file to {ops_path}")

    shape = metadata["shape"]
    nt, Ly, Lx = (
        shape[0],
        shape[-2],
        shape[-1],
    )

    if "num_frames" in metadata:
        nt_metadata = int(metadata["num_frames"])
        if nt_metadata != shape[0]:
            raise ValueError(
                f"Inconsistent frame count in metadata!\n"
                f"metadata['num_frames'] = {nt_metadata}\n"
                f"metadata['shape'][0] = {shape[0]}\n"
                f"These must match. Check your data and metadata."
            )
        nt = nt_metadata
        logger.debug(f"Using explicit num_frames={nt} from metadata")
    elif "nframes" in metadata:
        nt_metadata = int(metadata["nframes"])
        if nt_metadata != shape[0]:
            raise ValueError(
                f"Inconsistent frame count in metadata!\n"
                f"metadata['nframes'] = {nt_metadata}\n"
                f"metadata['shape'][0] = {shape[0]}\n"
                f"These must match. Check your data and metadata."
            )
        nt = nt_metadata
        logger.debug(f"Using explicit nframes={nt} from metadata")

    from mbo_utilities.metadata import get_param, get_voxel_size

    fs = get_param(metadata, "fs")
    if fs is None:
        finterval = get_param(metadata, "finterval")
        if finterval is not None and finterval > 0:
            fs = 1.0 / finterval
        else:
            logger.warning("No frame rate found; defaulting fs=10")
            fs = 10
    metadata["fs"] = fs
    voxel_size = get_voxel_size(metadata)
    dx, dy, dz = voxel_size.dx, voxel_size.dy, voxel_size.dz

    if ops_path.exists():
        ops = load_npy(ops_path).item()
    else:
        from mbo_utilities.metadata import default_ops

        ops = default_ops()

    ops.update(
        {
            "Ly": Ly,
            "Lx": Lx,
            "fs": metadata["fs"],
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "umPerPixX": dx,
            "umPerPixY": dy,
            "umPerPixZ": dz,
            "pixel_resolution": (dx, dy),
            "z_step": dz,
            "ops_path": str(ops_path),
        }
    )

    if chan == 1:
        ops["nframes_chan1"] = nt
        ops["raw_file"] = str(filename)
    else:
        ops["nframes_chan2"] = nt
        ops["chan2_file"] = str(filename)

    ops["align_by_chan"] = chan

    # NOTE: every frame-count alias is set from nt and shielded from the metadata merge below; a stale source value here reaches ops.npy
    ops["nframes"] = nt
    ops["num_frames"] = nt
    ops["num_timepoints"] = nt
    ops["n_frames"] = nt
    ops["timepoints"] = nt
    ops["T"] = nt
    ops["nt"] = nt

    protected_keys = {
        "nframes",
        "nframes_chan1",
        "nframes_chan2",
        "num_frames",
        "num_timepoints",
        "n_frames",
        "timepoints",
        "T",
        "nt",
        "dx",
        "dy",
        "dz",
        "umPerPixX",
        "umPerPixY",
        "umPerPixZ",
        "pixel_resolution",
        "z_step",
    }
    for key, value in metadata.items():
        if key not in protected_keys:
            ops[key] = value

    # NOTE: metadata merged from JSON carries images as lists; suite2p indexes them with .shape
    np.save(ops_path, normalize_ops_arrays(_convert_paths_to_strings(ops)))
    logger.debug(
        f"Ops file written to {ops_path} with nframes={ops['nframes']}, nframes_chan1={ops.get('nframes_chan1')}"
    )


def to_video(*args, **kwargs):
    """Encode array data to an mp4 video file.

    The implementation moved to ``mbo_utilities.arrays.mp4``; this shim keeps
    the ``mbo_utilities._writers.to_video`` import path stable.
    """
    from mbo_utilities.arrays.mp4 import to_video as _to_video

    return _to_video(*args, **kwargs)
