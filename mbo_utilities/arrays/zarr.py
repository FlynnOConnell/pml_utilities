"""
Zarr array reader.

This module provides ZarrArray for reading Zarr v3 stores, including OME-Zarr.
Presents data in TZYX format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import (
    ReductionMixin,
    Shape5DMixin,
    _imwrite_base,
    _normalize_key,
)
from mbo_utilities.arrays.suite2p import _add_suite2p_labels
from mbo_utilities.file_io import HAS_ZARR
from mbo_utilities.lazy_array import register_array_class
from mbo_utilities.metadata import _build_ome_metadata, get_param, get_voxel_size
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = log.get("arrays.zarr")

_ZARR_INFO = PipelineInfo(
    name="zarr",
    description="Zarr v3 stores (including OME-Zarr)",
    input_patterns=[
        "**/*.zarr",
        "**/*.zarr/",
        "**/zarr.json",
    ],
    output_patterns=[
        "**/*.zarr",
        "**/*.zarr/",
    ],
    input_extensions=["zarr"],
    output_extensions=["zarr"],
    marker_files=["zarr.json"],
    category="reader",
)
register_pipeline(_ZARR_INFO)


def _ome_time_scale_attr(md: dict) -> float | None:
    """Level-0 time-axis scale from an OME multiscales attr, if present."""
    ome = md.get("ome")
    multiscales = ome.get("multiscales") if isinstance(ome, dict) else None
    if multiscales is None:
        multiscales = md.get("multiscales")
    try:
        entry = multiscales[0]
        idx = next(
            i
            for i, ax in enumerate(entry["axes"])
            if isinstance(ax, dict) and ax.get("type") == "time"
        )
        return float(entry["datasets"][0]["coordinateTransformations"][0]["scale"][idx])
    except (TypeError, KeyError, IndexError, ValueError, StopIteration):
        return None


def _read_index(idx):
    if isinstance(idx, range):
        if len(idx) == 0:
            return slice(0, 0)
        return slice(idx.start, idx.stop, idx.step)
    if isinstance(idx, list) and len(idx) > 0:
        if all(idx[i] + 1 == idx[i + 1] for i in range(len(idx) - 1)):
            return slice(idx[0], idx[-1] + 1)
        return np.array(idx)
    return idx


def _keep_slice(k):
    if isinstance(k, (int, np.integer)):
        return slice(int(k), int(k) + 1)
    return _read_index(k)


class ZarrArray(ReductionMixin, Shape5DMixin):
    """
    Reader for Zarr stores (including OME-Zarr).

    Presents data as (T, Z, H, W) with Z=1..nz. Supports both standard
    zarr arrays and OME-Zarr groups with "0" arrays.

    Parameters
    ----------
    filenames : str, Path, or sequence
        Path(s) to zarr store(s).
    compressor : str, optional
        Compressor name (not currently used for reading).
    rois : list[int] or int, optional
        ROI filter (not currently used).
    dims : str | tuple | None, optional
        Dimension labels. Defaults to "TZYX" for 4D data.
        Supports custom orderings like "ZTYX", "sTZYX", etc.

    Attributes
    ----------
    shape : tuple[int, int, int, int]
        Shape as (T, Z, H, W).
    dtype : np.dtype
        Data type.
    dims : tuple[str, ...]
        Dimension labels (e.g., ('T', 'Z', 'Y', 'X')).
    zs : list
        List of zarr arrays.

    Examples
    --------
    >>> arr = ZarrArray("data.zarr")
    >>> arr.shape
    (10000, 1, 512, 512)
    >>> arr.dims
    ('T', 'Z', 'Y', 'X')
    >>> frame = arr[0, 0]  # Get first frame of first z-plane

    >>> # custom dimension ordering
    >>> arr = ZarrArray("data.zarr", dims="ZTYX")
    >>> arr.dims
    ('Z', 'T', 'Y', 'X')
    """

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        p = Path(file)
        if p.suffix != ".zarr" and not (p.is_dir() and (p / "zarr.json").exists()):
            return False
        from mbo_utilities.results import results_pipeline

        return (
            results_pipeline(p) is None
        )  # a results file is a zarr group without an image

    def __init__(
        self,
        filenames: str | Path | Sequence[str | Path],
        compressor: str | None = "default",
        rois: list[int] | int | None = None,
        dims: str | tuple | None = None,
    ):
        import zarr

        if isinstance(filenames, (str, Path)):
            filenames = [filenames]

        self.filenames = [Path(p).with_suffix(".zarr") for p in filenames]
        self.rois = rois
        for p in self.filenames:
            if not p.exists():
                raise FileNotFoundError(f"No zarr store at {p}")

        opened = [zarr.open(p, mode="r") for p in self.filenames]

        self.zs = []
        self._groups = []
        for i, z in enumerate(opened):
            if isinstance(z, zarr.Group):
                if "0" not in z:
                    raise ValueError(
                        f"OME-Zarr group missing '0' array in {self.filenames[i]}"
                    )
                self.zs.append(z["0"])
                self._groups.append(z)
            else:
                self.zs.append(z)
                self._groups.append(None)

        shapes = [z.shape for z in self.zs]
        if len(set(shapes)) != 1:
            listing = "\n".join(
                f"  - {p.name}: {tuple(s)}" for p, s in zip(self.filenames, shapes)
            )
            parent = self.filenames[0].parent
            raise ValueError(
                f"{parent} contains {len(self.zs)} zarr stores with different "
                f"shapes, so they are separate datasets and cannot be opened "
                f"together:\n{listing}\n"
                f'Open one explicitly, e.g. imread(r"{self.filenames[0]}").'
            )

        self._metadata = []
        for i, z in enumerate(self.zs):
            if self._groups[i] is not None:
                self._metadata.append(dict(self._groups[i].attrs))
            else:
                self._metadata.append(dict(z.attrs))
        self.compressor = compressor
        self._target_dtype = None

    @property
    def metadata(self) -> dict:
        """Return metadata as dict. Always returns dict, never None."""
        if not self._metadata:
            md = {}
        else:
            md = self._metadata[0].copy() if self._metadata[0] else {}

        md["dtype"] = self.dtype
        if "num_timepoints" not in md and self.zs:
            tp = int(self.zs[0].shape[0])
            md["num_timepoints"] = tp
            md["nframes"] = tp
            md["num_frames"] = tp

        time_scale = _ome_time_scale_attr(md)
        if time_scale is not None:
            md.setdefault("_ome_time_scale", time_scale)

        return md

    @property
    def zstats(self) -> dict | None:
        """
        Return pre-computed z-statistics from metadata if available.

        Returns
        -------
        dict | None
            Dictionary with keys 'mean', 'std', 'snr' (each a list of floats),
            or None if not available.

        Notes
        -----
        Deprecated: use the `stats` property instead.
        """
        md = self.metadata
        if "zstats" in md:
            return md["zstats"]
        if "stats" in md:
            return md["stats"]
        return None

    @zstats.setter
    def zstats(self, value: dict):
        """Store z-statistics in metadata for persistence."""
        if not isinstance(value, dict):
            raise TypeError(f"zstats must be a dict, got {type(value)}")
        if not all(k in value for k in ("mean", "std", "snr")):
            raise ValueError("zstats must contain 'mean', 'std', and 'snr' keys")

        if not self._metadata:
            self._metadata = [{}]
        self._metadata[0]["zstats"] = value
        self._metadata[0]["stats"] = value

    @property
    def stats(self) -> dict | None:
        """
        Return pre-computed statistics from metadata if available.

        Returns
        -------
        dict | None
            Dictionary with keys 'mean', 'std', 'snr', 'slice_label'
            (each a list of floats), or None if not available.
        """
        md = self.metadata
        if "stats" in md:
            return md["stats"]
        if "zstats" in md:
            zs = md["zstats"]
            return {**zs, "slice_label": "z"}
        return None

    @stats.setter
    def stats(self, value: dict):
        """Store statistics in metadata for persistence."""
        if not isinstance(value, dict):
            raise TypeError(f"stats must be a dict, got {type(value)}")
        if not all(k in value for k in ("mean", "std", "snr")):
            raise ValueError("stats must contain 'mean', 'std', and 'snr' keys")

        if not self._metadata:
            self._metadata = [{}]
        self._metadata[0]["stats"] = value
        self._metadata[0]["zstats"] = value

    @metadata.setter
    def metadata(self, value: dict):
        """Set metadata. Updates the first zarr file's metadata."""
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")

        if not self._metadata:
            self._metadata = [value]
        else:
            self._metadata[0] = value

    def _summary_stats_store_path(self) -> str | None:
        """Cache summary stats in the first backing zarr store."""
        return str(self.filenames[0]) if self.filenames else None

    @property
    def _shape_tzyx(self) -> tuple[int, int, int, int]:
        """Internal 4D shape for zarr indexing (used by 2D/3D/4D source paths;
        5D sources skip this and feed _shape5d directly).
        """
        first_shape = self.zs[0].shape
        if len(first_shape) == 4:
            return first_shape
        if len(first_shape) == 3:
            t, h, w = first_shape
            return t, len(self.zs), h, w
        if len(first_shape) == 2:
            h, w = first_shape
            return 1, len(self.zs), h, w
        raise ValueError(
            f"Unexpected zarr shape: {first_shape}. "
            f"Expected 2D (Y, X), 3D (T, Y, X), 4D (T, Z, Y, X), or 5D (T, C, Z, Y, X)"
        )

    @property
    def dtype(self):
        from mbo_utilities.arrays._base import get_dtype

        return (
            self._target_dtype
            if self._target_dtype is not None
            else get_dtype(self.zs[0].dtype)
        )

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    @property
    def size(self):
        return np.prod(self.shape)

    def __array__(self, dtype=None, copy=None):
        data = self[0]
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        first_shape = self.zs[0].shape
        if len(first_shape) == 5:
            return first_shape
        s = self._shape_tzyx
        return (s[0], 1, s[1], s[2], s[3])

    def __len__(self) -> int:
        return self.nt

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = key + (slice(None),) * (5 - len(key))
        t_key, c_key, z_key, y_key, x_key = key

        # NOTE: integer T/C/Z keys are read as slices of one and squeezed at the end, so the C insert below never lands on the wrong axis
        t_read = _keep_slice(t_key)
        c_read = _keep_slice(c_key)
        z_read = _keep_slice(z_key)
        y_read = _read_index(y_key)
        x_read = _read_index(x_key)

        first_ndim = len(self.zs[0].shape)
        is_single_5d = len(self.zs) == 1 and first_ndim == 5
        is_single_4d = len(self.zs) == 1 and first_ndim == 4
        is_single_2d = len(self.zs) == 1 and first_ndim == 2

        if is_single_5d:
            out = self.zs[0][t_read, c_read, z_read, y_read, x_read]
        elif is_single_4d:
            out = self.zs[0][t_read, z_read, y_read, x_read]
        elif is_single_2d:
            if isinstance(t_key, (int, np.integer)) and int(t_key) != 0:
                raise IndexError("T dimension has size 1, only index 0 is valid")
            if isinstance(z_key, (int, np.integer)) and int(z_key) != 0:
                raise IndexError("Z dimension has size 1, only index 0 is valid")
            data = self.zs[0][y_read, x_read]
            out = data[np.newaxis, np.newaxis, ...]
        elif len(self.zs) == 1:
            if isinstance(z_key, (int, np.integer)) and int(z_key) != 0:
                raise IndexError("Z dimension has size 1, only index 0 is valid")
            data = self.zs[0][t_read, y_read, x_read]
            out = np.expand_dims(data, axis=1)
        elif isinstance(z_key, (int, np.integer)):
            z_int = int(z_key)
            data = self.zs[z_int][t_read, y_read, x_read]
            out = np.expand_dims(data, axis=1)
        else:
            if isinstance(z_key, slice):
                z_indices = range(len(self.zs))[z_key]
            elif isinstance(z_key, (np.ndarray, list)):
                z_indices = list(z_key)
            else:
                z_indices = range(len(self.zs))
            arrs = [self.zs[i][t_read, y_read, x_read] for i in z_indices]
            out = np.stack(arrs, axis=1)

        if not is_single_5d:
            out = np.expand_dims(
                out, axis=1
            )  # 4D sources give TZYX; the stores never carry C

        squeeze_axes = []
        if isinstance(t_key, (int, np.integer)):
            squeeze_axes.append(0)
        if isinstance(c_key, (int, np.integer)):
            squeeze_axes.append(1)
        if isinstance(z_key, (int, np.integer)):
            squeeze_axes.append(2)
        for ax in sorted(squeeze_axes, reverse=True):
            out = np.squeeze(out, axis=ax)

        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite: bool = False,
        target_chunk_mb: int = 50,
        ext: str = ".tiff",
        progress_callback=None,
        debug: bool = False,
        planes: list[int] | int | None = None,
        **kwargs,
    ):
        """Write ZarrArray to disk in various formats."""
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
        """
        Save array to disk.

        Parameters
        ----------
        outpath : str | Path
            Output path.
        **kwargs
            Arguments passed to _imwrite (format, overwrite, etc).
        """
        return self._imwrite(outpath, **kwargs)


def merge_zarr_zplanes(
    zarr_paths: list[str | Path],
    output_path: str | Path,
    *,
    suite2p_dirs: list[str | Path] | None = None,
    metadata: dict | None = None,
    overwrite: bool = True,
    compression_level: int = 1,
) -> Path:
    """
    Merge multiple single z-plane Zarr files into a single OME-Zarr volume.

    Creates an OME-NGFF v0.5 compliant Zarr store with shape (T, Z, Y, X) by
    stacking individual z-plane Zarr files. Optionally includes Suite2p segmentation
    masks as OME-Zarr labels.

    Parameters
    ----------
    zarr_paths : list of str or Path
        List of paths to single-plane Zarr stores. Should be ordered by z-plane.
        Each Zarr should have shape (T, Y, X).
    output_path : str or Path
        Path for the output merged Zarr store.
    suite2p_dirs : list of str or Path, optional
        List of Suite2p output directories corresponding to each z-plane.
        If provided, ROI masks will be added as OME-Zarr labels.
        Must match length of zarr_paths.
    metadata : dict, optional
        Comprehensive metadata dictionary. Coordinate-related keys are used for
        OME-NGFF transformations, while additional keys are preserved as custom
        metadata. Supported keys:

        **Coordinate transformations:**
        - pixel_resolution : tuple (x, y) in micrometers
        - frame_rate : float, Hz (or 'fs')
        - dz : float, z-step in micrometers (or 'z_step')
        - name : str, volume name

        **ScanImage metadata:**
        - si : dict, complete ScanImage metadata structure
        - roi_groups : list, ROI definitions with scanfield info
        - objective_resolution : float, objective NA
        - zoom_factor : float

        **Acquisition metadata:**
        - acquisition_date : str, ISO format
        - experimenter : str
        - description : str
        - specimen : str

        **Microscope metadata:**
        - objective : str, objective name
        - emission_wavelength : float, nm
        - excitation_wavelength : float, nm
        - numerical_aperture : float

        **Processing metadata:**
        - fix_phase : bool
        - phasecorr_method : str
        - use_fft : bool
        - register_z : bool

        **Channel metadata:**
        - channel_names : list of str
        - num_planes : int, number of channels/planes

        All metadata is organized into structured groups (scanimage, acquisition,
        microscope, processing) in the output OME-Zarr attributes.
    overwrite : bool, default=True
        If True, overwrite existing output Zarr store.
    compression_level : int, default=1
        Gzip compression level (0-9). Higher = better compression, slower.

    Returns
    -------
    Path
        Path to the created OME-Zarr store.

    Raises
    ------
    ValueError
        If zarr_paths is empty or shapes are incompatible.
    FileNotFoundError
        If any input Zarr or Suite2p directory doesn't exist.

    Examples
    --------
    Merge z-plane Zarr files into a volume:

    >>> zarr_files = [
    ...     "session1/plane01.zarr",
    ...     "session1/plane02.zarr",
    ...     "session1/plane03.zarr",
    ... ]
    >>> merge_zarr_zplanes(zarr_files, "session1/volume.zarr")

    Include Suite2p segmentation masks:

    >>> s2p_dirs = [
    ...     "session1/plane01_suite2p",
    ...     "session1/plane02_suite2p",
    ...     "session1/plane03_suite2p",
    ... ]
    >>> merge_zarr_zplanes(
    ...     zarr_files,
    ...     "session1/volume.zarr",
    ...     suite2p_dirs=s2p_dirs,
    ...     metadata={"pixel_resolution": (0.5, 0.5), "frame_rate": 30.0, "dz": 5.0}
    ... )

    See Also
    --------
    imwrite : Write imaging data to various formats including OME-Zarr
    """
    if not HAS_ZARR:
        raise ImportError("zarr package required. Install with: pip install zarr")

    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    zarr_paths = [Path(p) for p in zarr_paths]
    output_path = Path(output_path)

    if not zarr_paths:
        raise ValueError("zarr_paths cannot be empty")

    for zp in zarr_paths:
        if not zp.exists():
            raise FileNotFoundError(f"Zarr store not found: {zp}")

    if suite2p_dirs is not None:
        suite2p_dirs = [Path(p) for p in suite2p_dirs]
        if len(suite2p_dirs) != len(zarr_paths):
            raise ValueError(
                f"suite2p_dirs length ({len(suite2p_dirs)}) must match "
                f"zarr_paths length ({len(zarr_paths)})"
            )
        for s2p_dir in suite2p_dirs:
            if not s2p_dir.exists():
                raise FileNotFoundError(f"Suite2p directory not found: {s2p_dir}")

    logger.info(f"Reading first Zarr to determine dimensions: {zarr_paths[0]}")
    z0 = zarr.open(str(zarr_paths[0]), mode="r")
    logger.debug(f"Zarr type: {type(z0)}")

    if isinstance(z0, zarr.Array):
        T, Y, X = z0.shape
        dtype = z0.dtype
        logger.debug(f"Detected direct array with shape {(T, Y, X)}, dtype {dtype}")
    else:
        logger.debug(f"Detected group with keys: {list(z0.keys())}")
        if "0" in z0:
            arr = z0["0"]
            T, Y, X = arr.shape
            dtype = arr.dtype
            logger.debug(f"Using '0' subarray with shape {(T, Y, X)}, dtype {dtype}")
        else:
            raise ValueError(
                f"Cannot determine shape of {zarr_paths[0]}. "
                f"Expected direct array or group with '0' subarray. "
                f"Got group with keys: {list(z0.keys())}"
            )

    Z = len(zarr_paths)
    logger.info(f"Creating merged Zarr volume with shape (T={T}, Z={Z}, Y={Y}, X={X})")

    if output_path.exists() and overwrite:
        import shutil

        shutil.rmtree(output_path)

    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    from zarr.codecs import Crc32cCodec, ShardingCodec

    inner_codecs = [BytesCodec(), GzipCodec(level=compression_level)]
    image_codecs = [
        ShardingCodec(
            chunk_shape=(1, 1, Y, X),
            codecs=inner_codecs,
            index_codecs=[BytesCodec(), Crc32cCodec()],
        )
    ]
    image = zarr.create(
        store=root.store,
        path="0",
        shape=(T, Z, Y, X),
        chunks=(
            T,
            1,
            Y,
            X,
        ),  # NOTE: one shard per z holding all T: same ~5.5 ms/frame as unsharded, ~50x fewer files
        dtype=dtype,
        codecs=image_codecs,
        overwrite=True,
    )

    logger.info("Copying z-plane data...")
    for zi, zpath in enumerate(zarr_paths):
        logger.debug(f"Reading z-plane {zi + 1}/{Z} from {zpath}")
        z_arr = zarr.open(str(zpath), mode="r")

        if isinstance(z_arr, zarr.Array):
            plane_data = z_arr[:]
            logger.debug(f"  Read direct array with shape {plane_data.shape}")
        elif "0" in z_arr:
            plane_data = z_arr["0"][:]
            logger.debug(f"  Read '0' subarray with shape {plane_data.shape}")
        else:
            raise ValueError(
                f"Cannot read data from {zpath}. "
                f"Got group with keys: {list(z_arr.keys())}"
            )

        if plane_data.shape != (T, Y, X):
            raise ValueError(
                f"Shape mismatch at z={zi} (file: {zpath.name}): "
                f"expected {(T, Y, X)}, got {plane_data.shape}"
            )

        logger.debug(f"  Writing to output volume at z={zi}")
        image[:, zi, :, :] = plane_data
        logger.info(f"Copied z-plane {zi + 1}/{Z} from {zpath.name}")

    if suite2p_dirs is not None:
        logger.info("Adding Suite2p segmentation masks as labels...")
        _add_suite2p_labels(root, suite2p_dirs, T, Z, Y, X, dtype, compression_level)

    metadata = metadata or {}
    ome_attrs = _build_ome_metadata(
        shape=(T, Z, Y, X),
        dtype=dtype,
        metadata=metadata,
    )

    for key, value in ome_attrs.items():
        root.attrs[key] = value

    vs = get_voxel_size(metadata)
    frame_rate = get_param(metadata, "fs", default=1.0)
    time_scale = 1.0 / float(frame_rate) if frame_rate else 1.0

    image.attrs["scale"] = [
        time_scale,
        vs.dz,
        vs.dy,
        vs.dx,
    ]  # napari reads (t, z, y, x) from the array attrs

    logger.info(f"Successfully created merged OME-Zarr at {output_path}")
    logger.info(f"Napari scale (t,z,y,x): {image.attrs['scale']}")
    return output_path


register_array_class(ZarrArray, priority=80)
