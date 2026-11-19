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
    DIMS,
    ReductionMixin,
    Shape5DMixin,
    _imwrite_base,
    _normalize_key,
)
from mbo_utilities.arrays.suite2p import _add_suite2p_labels
from mbo_utilities.file_io import HAS_ZARR
from mbo_utilities.lazy_array import register_array_class
from mbo_utilities.arrays.features._dim_labels import DEFAULT_DIMS
from mbo_utilities.metadata import _build_ome_metadata, get_param, get_voxel_size
from mbo_utilities.metadata.io import _OME_LENGTH_TO_UM
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
        return slice(int(k), int(k) + 1 or None)  # -1 reads the last element, not slice(-1, 0)
    return _read_index(k)


def _ngff_multiscale(attrs: dict) -> dict | None:
    """The first OME-NGFF multiscales entry of a group's attrs (0.5 under ``ome``, 0.4 at the top)."""
    ome = attrs.get("ome")
    multiscales = ome.get("multiscales") if isinstance(ome, dict) else attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales and isinstance(multiscales[0], dict):
        return multiscales[0]
    return None


def _store_labels(z, attrs: dict) -> tuple[str, ...] | None:
    """The axis labels a store declares: NGFF axes, then zarr v3 ``dimension_names``, then an mbo ``dims`` attr."""
    entry = _ngff_multiscale(attrs)
    if entry is not None and isinstance(entry.get("axes"), list):
        return tuple(str(ax["name"] if isinstance(ax, dict) else ax) for ax in entry["axes"])
    names = getattr(z.metadata, "dimension_names", None)
    if names and all(n is not None for n in names):
        return tuple(names)
    if attrs.get("dims"):
        return tuple(attrs["dims"])
    return None


def _valid_labels(labels, ndim: int) -> tuple[str, ...] | None:
    """``labels`` upper-cased when they name ``ndim`` distinct axes of TCZYX, else None."""
    labels = tuple(str(c).upper() for c in labels)
    if len(labels) == ndim and len(set(labels)) == ndim and set(labels) <= set(DIMS):
        return labels
    return None


def _ngff_voxel_size(attrs: dict) -> dict:
    """``dx``, ``dy``, ``dz`` in um from the level-0 NGFF scale, for the space axes that carry one."""
    entry = _ngff_multiscale(attrs)
    if entry is None:
        return {}
    try:
        scale = next(t["scale"] for t in entry["datasets"][0]["coordinateTransformations"] if t.get("type") == "scale")
    except (KeyError, IndexError, TypeError, StopIteration):
        return {}
    out = {}
    for ax, value in zip(entry.get("axes", []), scale, strict=False):
        if not isinstance(ax, dict) or ax.get("type") != "space":
            continue
        to_um = _OME_LENGTH_TO_UM.get(ax.get("unit", "micrometer"))
        if to_um is not None and str(ax.get("name", "")).lower() in ("x", "y", "z"):
            out["d" + ax["name"].lower()] = float(value) * to_um
    return out


def _index_by_labels(data, key: tuple, labels: tuple[str, ...]) -> np.ndarray:
    """Reads ``data``, whose axes are ``labels`` (a subset of TCZYX in any order), with a full 5D key; numpy 5D semantics on the result."""
    by_dim = dict(zip(DIMS, key, strict=True))
    out = np.asarray(data.oindex[tuple(_keep_slice(by_dim[d]) for d in labels)])
    order = sorted(range(len(labels)), key=lambda i: DIMS.index(labels[i]))
    out = np.transpose(out, order)  ## axes now in TCZYX order of the labels present
    present = {labels[i] for i in order}
    for i, d in enumerate(DIMS):
        if d not in present:
            out = np.expand_dims(out, axis=i)
    ints = tuple(i for i, k in enumerate(key) if isinstance(k, (int, np.integer)))
    return out.reshape([n for i, n in enumerate(out.shape) if i not in ints]) if ints else out


class ZarrArray(ReductionMixin, Shape5DMixin):
    """One zarr array, OME-Zarr group, or a folder of same-shape stores as (T, C, Z, Y, X).

    Axis labels come from ``dims``, else the store's NGFF ``axes``, zarr v3
    ``dimension_names`` or ``dims`` attr, else the rank (§5.2). Several stores
    stack onto Z, so their own labels must not include Z. Pixel sizes come from
    the level-0 NGFF scale when the attrs carry no ``dx``/``dy``/``dz``.
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
        self._labels = self._resolve_labels(dims)
        for md in self._metadata:
            for key, value in _ngff_voxel_size(md).items():
                md.setdefault(key, value)

    @property
    def metadata(self) -> dict:
        """Return metadata as dict. Always returns dict, never None."""
        if not self._metadata:
            md = {}
        else:
            md = self._metadata[0].copy() if self._metadata[0] else {}

        md["dtype"] = self.dtype
        nt, nc, nz, ny, nx = self._shape5d()
        md.update(num_timepoints=nt, num_color_channels=nc, num_zplanes=nz, Ly=ny, Lx=nx)

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

    def _resolve_labels(self, dims) -> tuple[str, ...]:
        """Axis labels of one store: declared, else what the store says, else the rank guess."""
        z = self.zs[0]
        declared = _valid_labels(dims, z.ndim) if dims is not None else None
        if dims is not None and declared is None:
            logger.warning(f"dims={dims!r} does not name the {z.ndim} axes of {self.filenames[0].name}; using the store's own labels")
        stored = _store_labels(z, self._metadata[0] if self._metadata else {})
        labels = declared or (_valid_labels(stored, z.ndim) if stored else None)
        if labels is None:
            if z.ndim not in DEFAULT_DIMS:
                raise ValueError(f"{self.filenames[0]}: a {z.ndim}D zarr array has no axis labels to read it by")
            labels = DEFAULT_DIMS[z.ndim]
        if len(self.zs) > 1:
            if "Z" in labels:
                raise ValueError(
                    f"{len(self.zs)} stores stack onto Z, but each already has a Z axis ({''.join(labels)}); open one store"
                )
        return labels

    @property
    def input_dims(self) -> tuple[str, ...]:
        """The store's own axis labels, as read."""
        return self._labels

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
        data = self[0, 0, 0]
        return data if dtype is None else data.astype(dtype)

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        sizes = dict(zip(self._labels, self.zs[0].shape, strict=True))
        if len(self.zs) > 1:
            sizes["Z"] = len(self.zs)
        return tuple(int(sizes.get(d, 1)) for d in DIMS)

    def __len__(self) -> int:
        return self.nt

    def __getitem__(self, key):
        key = _normalize_key(key, 5)
        key = key + (slice(None),) * (5 - len(key))
        if len(self.zs) == 1:
            out = _index_by_labels(self.zs[0], key, self._labels)
        else:
            z_key = key[2]
            per_store = key[:2] + (0,) + key[3:]
            if isinstance(z_key, (int, np.integer)):
                out = _index_by_labels(self.zs[int(z_key)], per_store, self._labels)
            else:
                stores = range(len(self.zs))[z_key] if isinstance(z_key, slice) else [int(i) for i in np.atleast_1d(z_key)]
                z_axis = sum(not isinstance(k, (int, np.integer)) for k in key[:2])
                out = np.stack([_index_by_labels(self.zs[i], per_store, self._labels) for i in stores], axis=z_axis)
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
