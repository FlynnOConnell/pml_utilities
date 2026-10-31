"""``imread``: open imaging data of any supported format as a lazy array."""

from __future__ import annotations

import importlib.util
import inspect
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays import (
    BinArray,
    H5Array,
    LBMArray,
    LBMPiezoArray,
    MP4Array,
    NumpyArray,
    PiezoArray,
    ScanImageArray,
    SinglePlaneArray,
    Suite2pArray,
    TiffArray,
    ZarrArray,
    _extract_tiff_plane_number,
)
from mbo_utilities.arrays.isoview import (
    IsoviewArray,
    detect_isoview_kind,
)
from mbo_utilities.lazy_array import _dispatch

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = log.get("reader")

MBO_SUPPORTED_FTYPES = [
    ".tiff",
    ".zarr",
    ".bin",
    ".h5",
    ".klb",
    ".mp4",
]  # the save-as dropdown
MBO_READABLE_FTYPES = [
    ".tiff",
    ".tif",
    ".zarr",
    ".bin",
    ".h5",
    ".mesc",
    ".npy",
    ".klb",
    ".mp4",
]

_OPTIONAL_PKG_BY_EXT = {".klb": "pyklb"}

MBO_AVAILABLE_FTYPES = [  # the dropdown without the formats whose optional package is missing
    ext
    for ext in MBO_SUPPORTED_FTYPES
    if _OPTIONAL_PKG_BY_EXT.get(ext) is None
    or importlib.util.find_spec(_OPTIONAL_PKG_BY_EXT[ext]) is not None
]


def source_reader_kwargs(arr) -> dict:
    """Extra `imread` kwargs needed to re-create `arr` from its `source_path`.

    Most arrays are fully determined by their path and return ``{}``. Formats
    whose file holds several independent datasets (``.mesc``, one array per
    measurement unit; ``.h5``, one array per HDF5 dataset) return the
    selector, so a task that re-opens the path in a worker process gets the
    same data the user is looking at rather than the file's default.
    """
    return dict(getattr(arr, "reader_kwargs", None) or {})


def widget_reader_kwargs(image_widget) -> dict:
    """`source_reader_kwargs` for the array an ImageWidget is displaying.

    Returns ``{}`` for a missing or empty widget, so a task's argument dict can
    always ask without guarding.
    """
    data = getattr(image_widget, "data", None) or []
    return source_reader_kwargs(data[0]) if len(data) else {}


@lru_cache(maxsize=32)
def _get_init_params(cls: type) -> set[str]:
    """The named parameters of ``cls.__init__``, without ``self``."""
    sig = inspect.signature(cls.__init__)
    return {
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _filter_kwargs(cls, kwargs):
    allowed = _get_init_params(cls)
    return {k: v for k, v in kwargs.items() if k in allowed}


def imread(
    inputs: str | Path | np.ndarray | Sequence[str | Path],
    **kwargs,
):
    """
    Lazy load imaging data from supported file types.

    Currently supported file types:
    - .bin: Suite2p binary files (.bin + ops.npy)
    - .tif/.tiff: TIFF files (BigTIFF, OME-TIFF and raw ScanImage TIFFs)
    - .h5: HDF5 files
    - .h5 with an ``imaging_system = bruker`` dataset: Bruker exports, axes from its dimension labels
    - .h5 with ``/imaging/data`` and ``frame_rate_hz``: lab4 converter output, stored (T, Z, Y, X, C)
    - .hdf5 with a DemixingResults group: masknmf demixing results, C = view (the GUI opens these in masknmf's viewers)
      (PMD movie, demixed signals, residual)
    - .hdf5 with a CompressionArray group: a masknmf compressed movie (compression.hdf5, alignment.hdf5)
    - .mesc: Femtonics MESc acquisitions (one MUnit per array)
    - .zarr: Zarr v3
    - .npy: NumPy arrays
    - np.ndarray: In-memory numpy arrays (wrapped as NumpyArray)

    Parameters
    ----------
    inputs : str, Path, ndarray, or sequence of str/Path
        Input source. Can be:
        - Path to a file or directory
        - List/tuple of file paths
        - A numpy array (will be wrapped as NumpyArray for full imwrite support)
        - An existing lazy array (passed through unchanged)
    channel : int, optional
        Zero-based color-channel index. When given, the returned array
        is wrapped as a 4D TZYX view of that single channel — useful for
        feeding multi-channel sources into pipelines that expect TZYX
        input. Subprocess workers can re-create the same view by passing
        ``reader_kwargs={"channel": N}`` to ``imread``.
    frame_average : int, optional
        Average every N consecutive timepoints into one (temporal binning).
        The returned array is a ``FrameAveragedView`` with ``T // N`` frames
        and a frame rate divided by N. Round-trips through
        ``source_reader_kwargs(arr)`` like ``channel`` does.
    invert_deflection : bool, optional
        Flip every frame about the per-pixel temporal mean (``2 * mean -
        frame``), so an indicator that dims on activity reads positive-going.
        The returned array is an ``InvertedDeflectionView``; round-trips
        through ``source_reader_kwargs(arr)``.
    dataset : str, optional
        HDF5 dataset to open (``.h5`` inputs only), nested paths accepted
        (e.g. ``"imaging/data"``). When omitted, common names are probed
        and the largest >=3D dataset is the fallback. Like the ``.mesc``
        ``unit`` selector, it round-trips through
        ``source_reader_kwargs(arr)`` so a worker process re-opens the
        same dataset the user picked.
    **kwargs
        Extra keyword arguments passed to specific array readers.

    Returns
    -------
    array_like
        A lazy array appropriate for the input format. Use `mbo formats` CLI
        command to list all supported formats and their array types.

    Examples
    --------
    >>> from mbo_utilities import imread, imwrite
    >>> arr = imread("/data/raw")  # directory with supported files
    >>> arr = imread("data.tiff")  # single file
    >>> arr = imread(["file1.tiff", "file2.tiff"])  # multiple files

    >>> # Wrap numpy array for imwrite compatibility
    >>> data = np.random.randn(100, 512, 512)
    >>> arr = imread(data)  # Returns NumpyArray
    >>> imwrite(arr, "output", ext=".zarr")  # Full write support
    """
    channel = kwargs.pop("channel", None)
    squeeze = kwargs.pop("squeeze", False)
    frame_average = kwargs.pop("frame_average", None)
    inverted = kwargs.pop("invert_deflection", False)
    arr = _imread_impl(inputs, **kwargs)
    if frame_average is not None and int(frame_average) > 1:
        from mbo_utilities.arrays._average_view import average_frames

        arr = average_frames(arr, int(frame_average))
    if inverted:
        from mbo_utilities.arrays._inverted_view import invert_deflection

        arr = invert_deflection(arr)
    # the channel view goes on last so its 4D surface stays on top
    if channel is not None:
        from mbo_utilities.arrays._channel_view import _ChannelView

        if len(arr.shape) < 5:
            logger.debug(
                "imread(channel=%r): underlying array is %dD, returning unwrapped",
                channel,
                getattr(arr, "ndim", "?"),
            )
        else:
            arr = _ChannelView(arr, int(channel))
    if squeeze and hasattr(arr, "squeeze"):
        arr = arr.squeeze()
    return arr


def _open_first_recording(files, where, kwargs):
    """Of ``files``, in the order given, the ones a format-specific class
    (``PRIORITY`` above 50) claims are whole recordings each (a Bruker h5, a
    MESc): the first opens through its class and the rest are logged by name;
    None when no file is one. A suffix reader would guess such a file's axes,
    and several never concatenate: they are separate sessions the user picks
    by file.
    """
    recordings = [
        (f, cls)
        for f in files
        if f.is_file()
        for cls in [_dispatch(f)]
        if cls is not None and cls.PRIORITY > 50
    ]
    if not recordings:
        return None
    first, cls = recordings[0]
    if len(recordings) > 1:
        logger.info(
            f"{where} holds {len(recordings)} recordings; opening {first.name} with "
            f"{cls.__name__}. Open a file to see another: "
            f"{', '.join(f.name for f, _ in recordings[1:])}"
        )
    return cls(first, **_filter_kwargs(cls, kwargs))


def _imread_impl(
    inputs: str | Path | np.ndarray | Sequence[str | Path],
    **kwargs,
):
    """The lazy array for ``inputs`` before the read-time views (§5.5 dispatch, then the legacy chain)."""
    if isinstance(inputs, np.ndarray):
        return NumpyArray(inputs, **_filter_kwargs(NumpyArray, kwargs))
    from mbo_utilities.squeeze import SqueezedView

    if isinstance(inputs, SqueezedView):
        return inputs.base
    if hasattr(inputs, "_imwrite") and hasattr(inputs, "shape"):
        return inputs

    if isinstance(inputs, (str, Path)):
        p = Path(inputs)
        if not p.exists():
            raise ValueError(f"Input path does not exist: {p}")

        # file pickers hand back the zarr.json or a chunk file inside a store
        if not p.is_dir():
            for ancestor in p.parents:
                if ancestor.suffix.lower() == ".zarr" and ancestor.is_dir():
                    logger.debug(
                        f"Redirecting {p.name} -> parent zarr store {ancestor}"
                    )
                    p = ancestor
                    break

        cls = _dispatch(p)
        if cls is not None:
            logger.debug(f"Dispatch selected {cls.__name__} for {p}")
            return cls(p, **_filter_kwargs(cls, kwargs))

        # suite2p markers win over isoview: detect_isoview_kind walks ancestors, so a
        # plane folder under `<root>.corrected/` would otherwise open the source tree
        if p.is_dir():
            if (p / "ops.npy").exists():
                logger.info(f"Detected Suite2p directory at {p}")
                return Suite2pArray(p)
            if p.name == "reg_tif" and (p.parent / "ops.npy").exists():
                logger.info(f"Detected Suite2p reg_tif folder at {p}")
                return Suite2pArray(p.parent / "ops.npy", use_reg_tif=True)
            plane_subdirs = [
                d for d in p.iterdir() if d.is_dir() and (d / "ops.npy").exists()
            ]
            if plane_subdirs:
                logger.info(
                    f"Detected Suite2p volume with {len(plane_subdirs)} planes in {p}"
                )
                return Suite2pArray(p)

        # detect_isoview_kind walks up to the stack root from a file, an SPM## folder or the root itself
        iso_kind = detect_isoview_kind(p)
        if iso_kind is not None:
            logger.info(f"Detected isoview-{iso_kind} tree at {p}")
            return IsoviewArray(p, kind=iso_kind)

        if p.suffix.lower() == ".zarr" and p.is_dir():
            paths = [p]
        elif p.is_dir():
            logger.debug(f"Input is a directory, searching for supported files in {p}")

            zarrs = list(p.glob("*.zarr"))
            if zarrs:
                logger.debug(
                    f"Found {len(zarrs)} zarr stores in {p}, loading as ZarrArray."
                )
                paths = zarrs
            else:
                plane_tiffs = sorted(p.glob("plane*.tif*"))
                if plane_tiffs:
                    logger.info(
                        f"Detected TIFF volume with {len(plane_tiffs)} planes in {p}"
                    )
                    return TiffArray(p)

                # a folder of whole recordings (a Bruker h5, a MESc each): the first by name opens
                opened = _open_first_recording(sorted(p.iterdir()), p, kwargs)
                if opened is not None:
                    return opened

                paths = [Path(f) for f in p.glob("*") if f.is_file()]
                logger.debug(f"Found {len(paths)} files in {p}")
        else:
            paths = [p]
    elif isinstance(inputs, (list, tuple)):
        if not inputs:
            raise ValueError("Input list is empty")
        if all(isinstance(item, np.ndarray) for item in inputs):
            return inputs
        if not all(isinstance(item, (str, Path)) for item in inputs):
            raise TypeError(
                f"Mixed input types in list. Expected all paths or all ndarrays. "
                f"Got: {[type(item).__name__ for item in inputs]}"
            )

        paths = [Path(p) for p in inputs]
        opened = _open_first_recording(paths, "the list", kwargs)
        if opened is not None:
            return opened
    else:
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    if not paths:
        raise ValueError("No input files found.")

    filtered = [p for p in paths if p.suffix.lower() in MBO_READABLE_FTYPES]
    if not filtered:
        raise ValueError(
            f"No supported files in {inputs}. \n"
            f"Supported file types are: {MBO_READABLE_FTYPES}"
        )
    paths = [
        p for p in filtered if not p.name.endswith("_pollen.h5")
    ]  # pollen results are outputs
    if not paths:
        raise ValueError(
            f"No source data files found in {inputs}. "
            f"Only pollen calibration result files (*_pollen.h5) were found."
        )

    ops_file = paths[0].parent / "ops.npy"
    if ops_file.exists():
        if len(paths) == 1 and paths[0].suffix.lower() == ".bin":
            logger.debug(f"Ops.npy detected - reading specific binary {paths[0]}.")
            return Suite2pArray(paths[0])
        logger.debug(f"Ops.npy detected - reading from {ops_file}.")
        return Suite2pArray(ops_file)

    exts = {p.suffix.lower() for p in paths}
    first = paths[0]

    if len(exts) > 1:
        if exts == {".bin", ".npy"}:
            npy_file = first.parent / "ops.npy"
            logger.debug(f"Reading {npy_file} from {npy_file}.")
            return Suite2pArray(npy_file)
        raise ValueError(f"Multiple file types found in input: {exts!r}")

    if first.suffix in [".tif", ".tiff"]:
        # planeNN files stack on Z whatever their kind
        if len(paths) > 1:
            plane_nums = {_extract_tiff_plane_number(p.name) for p in paths}
            plane_nums.discard(None)
            if len(plane_nums) > 1:
                logger.debug(
                    "Detected multiple planes in file list, loading as volumetric TiffArray."
                )
                return TiffArray(paths, **_filter_kwargs(TiffArray, kwargs))

        TIFF_ARRAY_CLASSES = [  # most specific first; TiffArray takes whatever is left
            (LBMArray, "LBM stack"),
            (PiezoArray, "piezo stack"),
            (LBMPiezoArray, "LBM+piezo stack"),
            (SinglePlaneArray, "single-plane ScanImage"),
            (ScanImageArray, "raw ScanImage"),
            (TiffArray, "TIFF"),
        ]

        for array_cls, description in TIFF_ARRAY_CLASSES:
            if array_cls.can_open(first):
                # saved outputs or unrelated tiffs in the same folder are left out
                if len(paths) > 1:
                    valid = [p for p in paths if array_cls.can_open(p)]
                    if not valid:
                        continue
                    if len(valid) < len(paths):
                        logger.info(
                            f"Filtered {len(paths) - len(valid)} non-{description} "
                            f"file(s) from directory load"
                        )
                    paths = valid
                logger.debug(
                    f"Detected {description}, loading as {array_cls.__name__}."
                )
                return array_cls(paths, **_filter_kwargs(array_cls, kwargs))

    if first.suffix == ".bin":
        if isinstance(inputs, (str, Path)) and Path(inputs).suffix == ".bin":
            logger.debug(f"Reading binary file as BinArray: {first}")
            return BinArray(first, **_filter_kwargs(BinArray, kwargs))

        npy_file = first.parent / "ops.npy"
        if npy_file.exists():
            logger.debug(f"Reading Suite2p directory from {npy_file}.")
            return Suite2pArray(npy_file)

        raise ValueError(
            "Cannot read .bin file without ops.npy or shape parameter. "
            "Provide shape=(nframes, Ly, Lx) as kwarg or ensure ops.npy exists."
        )

    if first.suffix == ".h5":
        logger.debug(f"Reading HDF5 files from {first}.")
        return H5Array(first, **_filter_kwargs(H5Array, kwargs))

    if first.suffix == ".mesc":
        from mbo_utilities.arrays.mesc import MescArray

        if len(paths) > 1:
            logger.warning(
                f"{len(paths)} .mesc files in {inputs}; opening {first.name}. "
                f"Each file holds its own units - open them one at a time."
            )
        logger.debug(f"Reading MESc file: {first}")
        return MescArray(first, **_filter_kwargs(MescArray, kwargs))

    if first.suffix == ".mp4":
        logger.debug(f"Reading MP4 file as MP4Array: {first}")
        return MP4Array(first, **_filter_kwargs(MP4Array, kwargs))

    if first.suffix == ".zarr":
        sub_zarrs = list(first.glob("*.zarr"))
        if sub_zarrs:
            logger.info("Detected nested zarr stores, loading as ZarrArray.")
            return ZarrArray(sub_zarrs, **_filter_kwargs(ZarrArray, kwargs))
        if (first / "zarr.json").exists():
            logger.info("Detected zarr.json, loading as ZarrArray.")
            return ZarrArray(paths, **_filter_kwargs(ZarrArray, kwargs))

        raise ValueError(
            f"Zarr path {first} is not a valid store. "
            "Expected nested *.zarr dirs or a zarr.json inside."
        )

    if first.suffix == ".json":
        logger.debug(f"Reading JSON files from {first}.")
        return ZarrArray(first.parent, **_filter_kwargs(ZarrArray, kwargs))

    if first.suffix == ".npy":
        if (first.parent / "pmd_demixer.npy").is_file():
            raise NotImplementedError("PMD Arrays are not yet supported.")

        logger.debug(f"Loading .npy file as NumpyArray: {first}")
        return NumpyArray(first, **_filter_kwargs(NumpyArray, kwargs))

    if first.suffix == ".klb":
        import pyklb

        if len(paths) == 1:
            logger.info(f"Loading KLB file: {first}")
            data = pyklb.readfull(str(first))
            return NumpyArray(data)

        # clusterPT tree: TM######/SPM##_TM######_CM##_CHN##.klb
        parent = first.parent
        if parent.name.startswith("TM"):
            logger.info("Detected KLB file in TM folder, loading as isoview-clusterpt.")
            return IsoviewArray(parent.parent, kind="clusterpt")

        tm_folders = [
            d for d in parent.iterdir() if d.is_dir() and d.name.startswith("TM")
        ]
        if tm_folders:
            logger.info(
                "Detected KLB file in clusterPT root, loading as isoview-clusterpt."
            )
            return IsoviewArray(parent, kind="clusterpt")

        logger.info(f"Loading standalone KLB file: {first}")
        data = pyklb.readfull(str(first))
        return NumpyArray(data)

    raise TypeError(f"Unsupported file type: {first.suffix}")
