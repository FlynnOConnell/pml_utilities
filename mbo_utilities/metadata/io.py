"""
metadata file I/O - reading/writing metadata from TIFF files.

functions for extracting metadata from ScanImage TIFFs and other file formats.
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import tifffile

from mbo_utilities import log
from mbo_utilities.file_io import get_files, load_npy

from .params import get_param, normalize_resolution

__all__ = [
    "_build_ome_metadata",
    "clean_scanimage_metadata",
    "default_ops",
    "get_metadata",
    "get_metadata_batch",
    "get_metadata_single",
    "imagej_hyperstack_counts",
    "is_raw_scanimage",
    "query_tiff_pages",
]

logger = log.get("metadata")


def _metadata_from_ops(ops: dict) -> dict:
    """
    Build metadata dict from ops.npy using the metadata API for consistent extraction.

    Parameters
    ----------
    ops : dict
        The ops dictionary loaded from ops.npy.

    Returns
    -------
    dict
        Metadata dictionary with standard keys.
    """
    result = {
        "num_planes": get_param(ops, "nplanes", default=1),
        "Lx": get_param(ops, "Lx"),
        "Ly": get_param(ops, "Ly"),
        "fov": (
            get_param(ops, "Lx", default=0),
            get_param(ops, "Ly", default=0),
        ),
        "frame_rate": get_param(ops, "fs"),
        "fs": get_param(ops, "fs"),
        "zoom_factor": ops.get("zoom"),
        "dx": get_param(ops, "dx", default=1.0),
        "dy": get_param(ops, "dy", default=1.0),
        "pixel_resolution": (
            get_param(ops, "dx", default=1.0),
            get_param(ops, "dy", default=1.0),
        ),
        "dtype": "int16",
        "source": "ops_fallback",
    }
    return {k: v for k, v in result.items() if v is not None}


def is_raw_scanimage(file: os.PathLike | str) -> bool:
    """
    Check if a TIFF file is a raw ScanImage TIFF (unprocessed acquisition data).

    A file is considered "raw" if it has ScanImage metadata AND has not been
    processed by mbo_utilities (no custom tag 50839 with our JSON metadata).

    Parameters
    ----------
    file: os.PathLike
        Path to the TIFF file.

    Returns
    -------
    bool
        True if the TIFF file is a raw ScanImage TIFF; False otherwise.
    """
    if not file or not isinstance(file, (str, os.PathLike)):
        return False
    if Path(file).suffix.lower() not in (".tif", ".tiff"):
        return False
    try:
        with tifffile.TiffFile(file) as tf:
            if tf.scanimage_metadata is None:
                return False

            page0 = tf.pages.first
            if page0 and 50839 in page0.tags:
                return False

            if isinstance(tf.shaped_metadata, (list, tuple)) and tf.shaped_metadata:
                return False

            return True
    except Exception:
        return False


def get_metadata(
    file,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
    z_step: float | None = None,
):
    """
    Extract metadata from a TIFF file or directory of TIFF files produced by ScanImage.

    This function handles single files, lists of files, or directories containing TIFF files.
    When given a directory, it automatically finds and processes all TIFF files in natural
    sort order. For multiple files, it calculates frames per file accounting for z-planes.

    Parameters
    ----------
    file : os.PathLike, str, or list
        - Single file path: processes that file
        - Directory path: processes all TIFF files in the directory
        - List of file paths: processes all files in the list
    dx : float, optional
        X pixel resolution in micrometers. Overrides extracted value.
    dy : float, optional
        Y pixel resolution in micrometers. Overrides extracted value.
    dz : float, optional
        Z step size in micrometers. Overrides extracted value.
        Also available as ``z_step`` for backward compatibility.
    verbose : bool, optional
        If True, returns extended metadata including all ScanImage attributes. Default is False.
    z_step : float, optional
        Alias for ``dz`` (backward compatibility).

    Returns
    -------
    dict
        A dictionary containing extracted metadata with normalized resolution aliases:
        - dx, dy, dz: canonical resolution values in micrometers
        - pixel_resolution: (dx, dy) tuple
        - voxel_size: (dx, dy, dz) tuple
        - umPerPixX, umPerPixY, umPerPixZ: legacy format
        - PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ: OME format

        For multiple files, also includes:
        - 'frames_per_file': list of frame counts per file (accounting for z-planes)
        - 'total_frames': total frames across all files
        - 'file_paths': list of processed file paths
        - 'tiff_pages_per_file': raw TIFF page counts per file

    Raises
    ------
    ValueError
        If no recognizable metadata is found or no TIFF files found in directory.

    Examples
    --------
    >>> # Single file with z-resolution
    >>> meta = get_metadata("path/to/rawscan_00001.tif", dz=5.0)
    >>> print(f"Voxel size: {meta['voxel_size']}")

    >>> # Directory of files
    >>> meta = get_metadata("path/to/scan_directory/")
    >>> print(f"Files processed: {len(meta['file_paths'])}")
    >>> print(f"Frames per file: {meta['frames_per_file']}")

    >>> # List of specific files
    >>> files = ["scan_00001.tif", "scan_00002.tif", "scan_00003.tif"]
    >>> meta = get_metadata(files, dz=5.0)
    """
    if dz is None and z_step is not None:
        dz = z_step

    if hasattr(file, "metadata"):
        metadata = dict(file.metadata) if file.metadata else {}
        return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)

    if isinstance(file, (list, tuple)):
        if not all(isinstance(f, (str, os.PathLike)) for f in file):
            raise ValueError(
                "All items in the list must be of type str or os.PathLike."
                f"Got: {file} of type {type(file)}"
            )
        file_paths = [Path(f) for f in file]
        metadata = get_metadata_batch(file_paths)
        return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)

    file_path = Path(file)

    if file_path.is_dir():
        if file_path.suffix in [".zarr"]:
            metadata = get_metadata_single(file_path)
            return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)
        tiff_files = get_files(file_path, "tif", sort_ascending=True)
        if not tiff_files:
            raise ValueError(f"No TIFF files found in directory: {file_path}")
        metadata = get_metadata_batch(tiff_files)
        return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)

    if file_path.is_file():
        metadata = get_metadata_single(file_path)
        return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)

    raise ValueError(f"Path does not exist or is not accessible: {file_path}")


def _read_ij_tag_50839(tiff_file) -> dict | None:
    """Parse the IJMetadata tag (50839).

    mbo writes its full metadata JSON here under the ``Info`` key; Fiji writes
    slice ``Labels``. Returns the embedded mbo dict when present, otherwise the
    raw tag dict, or ``None`` when the tag is absent/unparseable.
    """
    try:
        page0 = tiff_file.pages.first
        if not page0 or 50839 not in page0.tags:
            return None
        tag_value = page0.tags[50839].value
        if isinstance(tag_value, bytes):
            return json.loads(tag_value.rstrip(b"\x00").decode("utf-8"))
        if isinstance(tag_value, str):
            return json.loads(tag_value)
        if isinstance(tag_value, dict):
            info = tag_value.get("Info")
            if isinstance(info, str):
                try:
                    return json.loads(info)
                except (json.JSONDecodeError, TypeError):
                    return {k: v for k, v in tag_value.items() if k != "Info"}
            return tag_value
    except (json.JSONDecodeError, TypeError, AttributeError, KeyError):
        return None
    return None


def imagej_hyperstack_counts(tiff_file) -> tuple[int, int, int]:
    """(frames, slices, channels) from ImageJ metadata, each defaulting to 1."""
    ij = tiff_file.imagej_metadata or {}
    return tuple(int(ij.get(k, 1) or 1) for k in ("frames", "slices", "channels"))


def _imagej_to_metadata(tiff_file) -> dict:
    """Deposit ImageJ/Fiji hyperstack source tags for canonical resolution.

    Reads the hyperstack dimension counts (``frames``/``slices``/``channels``),
    voxel spacing (``spacing`` -> ``dz``), frame interval (``finterval`` ->
    ``fs``) and pixel resolution (``XResolution``/``YResolution`` -> ``dx``/
    ``dy``) as raw source keys. The metadata registry resolves them downstream,
    so values are not pre-normalized here.
    """
    ij = dict(tiff_file.imagej_metadata or {})
    page0 = tiff_file.pages.first
    frames, slices, channels = imagej_hyperstack_counts(tiff_file)

    md: dict = {"frames": frames, "slices": slices, "channels": channels}
    md["hyperstack"] = bool(ij.get("hyperstack", False))
    if page0 is not None:
        md["Ly"] = int(page0.shape[-2])
        md["Lx"] = int(page0.shape[-1])
        md["dtype"] = str(page0.dtype)

    spacing = ij.get("spacing")
    if spacing:
        md["spacing"] = float(spacing)
    finterval = ij.get("finterval")
    if finterval:
        md["finterval"] = float(finterval)
    unit = ij.get("unit")
    if unit:
        md["unit"] = unit

    # NOTE: resolution tags are pixels per unit: without a micron unit an inch-scanned or uncalibrated image gives a bogus dx; both µ spellings (U+00B5, U+03BC) occur
    micron_units = {"micron", "microns", "um", "µm", "μm", "micrometer", "micrometers"}
    if page0 is not None and unit and str(unit).lower() in micron_units:
        for tag in ("XResolution", "YResolution"):
            tag_obj = page0.tags.get(tag)
            value = tag_obj.value if tag_obj is not None else None
            if isinstance(value, (tuple, list)) and len(value) == 2 and value[1]:
                md[tag] = value[0] / value[1]
            elif isinstance(value, (int, float)) and value:
                md[tag] = float(value)
    return md


_OME_LENGTH_TO_UM = {
    "pm": 1e-6,
    "nm": 1e-3,
    "µm": 1.0,
    "μm": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
}
_OME_TIME_TO_S = {
    "ns": 1e-9,
    "µs": 1e-6,
    "μs": 1e-6,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
}


def _ome_to_metadata(tiff_file) -> dict:
    """Deposit OME-XML Pixels attributes as raw source keys.

    ``SizeX/Y/Z/T/C``, ``DimensionOrder``, ``PhysicalSize*`` and
    ``TimeIncrement`` resolve through the metadata registry (SizeZ ->
    num_zplanes, PhysicalSizeX -> dx, TimeIncrement -> finterval -> fs).
    Values are kept raw; only units are converted (physical sizes to µm,
    TimeIncrement to seconds) when the XML declares a non-default unit.
    """
    import xml.etree.ElementTree as ET

    xml_text = tiff_file.ome_metadata
    if not xml_text:
        return {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}

    pixels = next((el for el in root.iter() if _ome_local_name(el) == "Pixels"), None)
    if pixels is None:
        return {}

    md: dict = {"is_ome": True}
    page0 = tiff_file.pages.first
    if page0 is not None:
        md["Ly"] = int(page0.shape[-2])
        md["Lx"] = int(page0.shape[-1])
        md["dtype"] = str(page0.dtype)

    attrs = pixels.attrib
    for key in ("SizeX", "SizeY", "SizeZ", "SizeT", "SizeC"):
        if attrs.get(key):
            md[key] = int(attrs[key])
    if attrs.get("DimensionOrder"):
        md["DimensionOrder"] = attrs["DimensionOrder"]

    for key in ("PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ"):
        value = attrs.get(key)
        if not value:
            continue
        scale = _OME_LENGTH_TO_UM.get(attrs.get(f"{key}Unit", "µm"))
        if scale is not None:
            md[key] = float(value) * scale
    value = attrs.get("TimeIncrement")
    if value:
        scale = _OME_TIME_TO_S.get(attrs.get("TimeIncrementUnit", "s"))
        if scale is not None:
            md["TimeIncrement"] = float(value) * scale

    names = [
        el.attrib["Name"]
        for el in pixels
        if _ome_local_name(el) == "Channel" and el.attrib.get("Name")
    ]
    if names:
        md["channel_names"] = names

    image = next((el for el in root.iter() if _ome_local_name(el) == "Image"), None)
    if image is not None:
        if image.attrib.get("Name"):
            md["image_name"] = image.attrib["Name"]
        for el in image:
            if _ome_local_name(el) == "AcquisitionDate" and el.text and el.text.strip():
                md["acquisition_date"] = el.text.strip()
            elif _ome_local_name(el) == "Description" and el.text and el.text.strip():
                md["description"] = el.text.strip()
    return md


def _ops_fallback(file: Path) -> dict:
    """``_metadata_from_ops`` of the nearest ``ops.npy`` above ``file``, else ``{"source": "no_metadata"}``."""
    for parent in Path(file).parents:
        ops_path = parent / "ops.npy"
        if ops_path.exists():
            result = _metadata_from_ops(load_npy(ops_path).item())
            if "plane0" in str(parent).lower():
                result["num_planes"] = 1
            return result
    return {"source": "no_metadata"}


def _processed_tiff_metadata(tiff_file, file: Path) -> dict:
    """The metadata a non-ScanImage tiff carries: shaped metadata, else the ImageJ or OME layout with any embedded mbo JSON layered under it, else tag 50839 or the page description alone."""
    if tiff_file.shaped_metadata:
        return tiff_file.shaped_metadata[0]
    tag_meta = _read_ij_tag_50839(tiff_file)
    if tiff_file.is_imagej:
        meta = _imagej_to_metadata(tiff_file)
        if isinstance(tag_meta, dict):
            layout = {k: meta[k] for k in ("frames", "slices", "channels")}
            meta.update(tag_meta)
            meta.update(layout)
        return meta
    if tiff_file.is_ome:
        meta = _ome_to_metadata(tiff_file)
        if meta:
            if isinstance(tag_meta, dict):
                layout = {k: meta[k] for k in ("SizeT", "SizeZ", "SizeC") if k in meta}
                meta.update(tag_meta)
                meta.update(layout)
            return meta
    if isinstance(tag_meta, dict):
        return tag_meta
    page0 = tiff_file.pages.first
    if page0 and page0.description:
        try:
            meta = json.loads(page0.description)
        except json.JSONDecodeError:
            meta = None
        if isinstance(meta, dict):
            return meta
    raise ValueError(f"No metadata found in {file}.")


def _scanimage_tiff_metadata(tiff_file, file: Path) -> dict:
    """The deposit for a raw ScanImage tiff: the mROI geometry, the interleaved page count and the ``si`` header, through ``clean_scanimage_metadata``."""
    meta = tiff_file.scanimage_metadata
    if not meta:
        logger.info(f"{file} has no scanimage_metadata, trying ops.npy fallback.")
        return _ops_fallback(file)

    from mbo_utilities._writers import _make_json_serializable as _serialize

    si = meta.get("FrameData", {})
    if not si:
        logger.warning(f"No FrameData found in {file}.")
        return {"source": "scanimage_no_framedata"}

    shape = tiff_file.pages[0].shape
    roi_group = meta.get("RoiGroups", {}).get("imagingRoiGroup", {}).get("rois")
    if roi_group is None:
        logger.warning(f"No ROI information found in {file}.")
        return {"source": "scanimage_no_rois"}
    if isinstance(roi_group, dict):
        roi_group = [roi_group]
    num_rois = len(roi_group)

    channel_save = si.get("SI.hChannels.channelSave")
    if channel_save is None or isinstance(channel_save, (int, float)):
        nchannels = 1
    else:
        nchannels = len(channel_save)

    zoom_factor = si.get("SI.hRoiManager.scanZoomFactor")
    uniform_sampling = si.get("SI.hScan2D.uniformSampling", "NA")
    objective_resolution = si.get("SI.objectiveResolution", 1.0)
    frame_rate = si.get("SI.hRoiManager.scanFrameRate")

    fly_to_time = float(si.get("SI.hScan2D.flytoTimePerScanfield", 0))
    line_period = float(si.get("SI.hRoiManager.linePeriod", 1))
    num_fly_to_lines = round(fly_to_time / line_period) if line_period > 0 else 0

    sizes = []
    num_pixel_xys = []
    for roi in roi_group:
        scanfields = roi.get("scanfields")
        if scanfields is None:
            continue
        if isinstance(scanfields, list):
            scanfields = scanfields[0] if scanfields else {}
        size_xy = scanfields.get("sizeXY")
        pixel_res = scanfields.get("pixelResolutionXY")
        if size_xy:
            sizes.append(size_xy)
        if pixel_res:
            num_pixel_xys.append(pixel_res)

    if not sizes or not num_pixel_xys:
        logger.warning(f"Could not extract ROI size/resolution from {file}.")
        return {"source": "scanimage_no_roi_sizes"}

    size_xy = sizes[0]
    num_pixel_xy = num_pixel_xys[0]
    roi_fov_x_um = round(objective_resolution * size_xy[0])
    roi_fov_y_um = round(objective_resolution * size_xy[1])
    pixel_resolution = (
        roi_fov_x_um / num_pixel_xy[0],
        roi_fov_y_um / num_pixel_xy[1],
    )
    roi_width, roi_height = (
        num_pixel_xy[0],
        num_pixel_xy[1],
    )

    metadata = {
        "nchannels": nchannels,
        "num_rois": num_rois,
        "roi": (roi_width, roi_height),  # one strip; fov is every strip side by side
        "fov": (num_rois * roi_width, roi_height),
        "fov_um": (num_rois * roi_fov_x_um, roi_fov_y_um),
        "frame_rate": frame_rate,
        "pixel_resolution": np.round(pixel_resolution, 2),
        "ndim": len(shape),
        "dtype": "int16",
        "size": np.prod(shape),
        "page_height": shape[0],
        "page_width": shape[1],
        "objective_resolution": objective_resolution,
        "zoom_factor": zoom_factor,
        "uniform_sampling": uniform_sampling,
        "num_fly_to_lines": num_fly_to_lines,
        "roi_heights": [px[1] for px in num_pixel_xys],
        "roi_groups": _serialize(roi_group),
        "si": _serialize(si),
    }
    return clean_scanimage_metadata(metadata)


def get_metadata_single(file: Path):
    """The metadata of one file: a zarr, h5 or mesc through its reader; a raw ScanImage tiff from its header; any other tiff from what it embeds (``_processed_tiff_metadata``).

    >>> get_metadata_single(Path("rawscan_00001.tif"))["num_timepoints"]
    5345
    """
    if file.suffix in [".zarr", ".h5", ".mesc"]:
        from mbo_utilities import imread

        return imread(file).metadata

    with tifffile.TiffFile(file) as tiff_file:
        if is_raw_scanimage(file):
            return _scanimage_tiff_metadata(tiff_file, file)
        return _processed_tiff_metadata(tiff_file, file)


def get_metadata_batch(file_paths: list | tuple):
    """
    Extract and aggregate metadata from a list of TIFF files.

    Parameters
    ----------
    file_paths : list of Path
        List of TIFF file paths.

    Returns
    -------
    dict
        Aggregated metadata with per-file frame information.
    """
    if not file_paths:
        raise ValueError("No files provided")

    metadata = get_metadata_single(file_paths[0])
    nchannels = metadata.get("nchannels", 1)

    _t0 = time.perf_counter()
    frames_per_file = [query_tiff_pages(fp) // nchannels for fp in file_paths]
    logger.debug(
        "counted %d frames across %d files in %.2fs",
        sum(frames_per_file),
        len(file_paths),
        time.perf_counter() - _t0,
    )

    total_frames = sum(frames_per_file)
    return metadata | {
        "num_timepoints": total_frames,
        "nframes": total_frames,
        "num_frames": total_frames,
        "frames_per_file": frames_per_file,
        "file_paths": [str(fp) for fp in file_paths],
        "num_files": len(file_paths),
    }


def query_tiff_pages(file_path):
    """
    Get page count for TIFF files for both 64-bit (BigTiff) and 32-bit TIFFS.

    Frames per tiff must be uniform.

    1. Reading first TWO IFD offsets only
    2. Calculating page_size = second_offset - first_offset
    3. Estimating: total_pages = file_size / page_size


    Parameters
    ----------
    file_path : str
        Path to ScanImage TIFF file

    Returns
    -------
    int
        Number of pages
    """
    file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as f:
        header = f.read(8)

        if header[:2] == b"II":
            bo = "<"
        elif header[:2] == b"MM":
            bo = ">"
        else:
            raise ValueError("Not a TIFF file")

        version = struct.unpack(f"{bo}H", header[2:4])[0]

        if version == 42:
            offset_fmt = f"{bo}I"
            offset_size = 4
            tag_count_fmt = f"{bo}H"
            tag_count_size = 2
            tag_size = 12
            first_ifd_offset = struct.unpack(offset_fmt, header[4:8])[0]
            header_size = 8

        elif version == 43:
            offset_fmt = f"{bo}Q"
            offset_size = 8
            tag_count_fmt = f"{bo}Q"
            tag_count_size = 8
            tag_size = 20
            f.seek(8)
            first_ifd_offset = struct.unpack(offset_fmt, f.read(offset_size))[0]
            header_size = 16

        else:
            raise ValueError(f"Unknown TIFF version: {version}")

        f.seek(first_ifd_offset)

        tag_count = struct.unpack(tag_count_fmt, f.read(tag_count_size))[0]
        tag_block = f.read(tag_count * tag_size)

        second_ifd_offset = struct.unpack(offset_fmt, f.read(offset_size))[0]

        if second_ifd_offset == 0:
            return 1

        page_size = second_ifd_offset - first_ifd_offset

        bigtiff = version == 43
        width = height = None
        bits, spp, compression = 8, 1, 1
        for i in range(tag_count):
            tag, value = _ifd_inline_value(
                tag_block[i * tag_size : (i + 1) * tag_size], bo, bigtiff
            )
            if value is None:
                continue
            if tag == 256:
                width = value
            elif tag == 257:
                height = value
            elif tag == 258:
                bits = value
            elif tag == 259:
                compression = value
            elif tag == 277:
                spp = value

        min_page_bytes = width * height * spp * bits // 8 if width and height else None
        if (
            page_size <= 0
            or compression != 1
            or (min_page_bytes is not None and page_size < min_page_bytes)
        ):
            return _count_ifd_chain(
                f,
                first_ifd_offset,
                file_size,
                tag_count_fmt,
                tag_count_size,
                tag_size,
                offset_fmt,
                offset_size,
            )

        data_size = file_size - header_size
        # NOTE: assumes [IFD][data][IFD][data]; writers that cluster IFDs (Femtonics, Bio-Formats OME) would inflate this ~1000x, so the stride is bounded by one page of data above
        num_pages = data_size // page_size

        return int(num_pages)


def _ifd_inline_value(entry: bytes, bo: str, bigtiff: bool) -> tuple[int, int | None]:
    """(tag, value) for a single-count SHORT/LONG/LONG8 IFD entry, else (tag, None)."""
    if bigtiff:
        tag, typ = struct.unpack(f"{bo}HH", entry[:4])
        count = struct.unpack(f"{bo}Q", entry[4:12])[0]
        raw = entry[12:20]
    else:
        tag, typ, count = struct.unpack(f"{bo}HHI", entry[:8])
        raw = entry[8:12]
    if count != 1:
        return tag, None
    if typ == 3:
        return tag, struct.unpack(f"{bo}H", raw[:2])[0]
    if typ == 4:
        return tag, struct.unpack(f"{bo}I", raw[:4])[0]
    if typ == 16 and bigtiff:
        return tag, struct.unpack(f"{bo}Q", raw[:8])[0]
    return tag, None


def _count_ifd_chain(
    f,
    first_ifd_offset,
    file_size,
    tag_count_fmt,
    tag_count_size,
    tag_size,
    offset_fmt,
    offset_size,
) -> int:
    """Exact page count by following next-IFD offsets (2 small reads per page)."""
    count = 0
    offset = first_ifd_offset
    while 0 < offset < file_size and count < 4_000_000:
        f.seek(offset)
        data = f.read(tag_count_size)
        if len(data) < tag_count_size:
            break
        n_tags = struct.unpack(tag_count_fmt, data)[0]
        f.seek(offset + tag_count_size + n_tags * tag_size)
        data = f.read(offset_size)
        if len(data) < offset_size:
            break
        count += 1
        offset = struct.unpack(offset_fmt, data)[0]
    return count


def _json_serializable(value) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _ome_local_name(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _clean_value(x):
    if x is None:
        return None
    if isinstance(x, (np.generic,)):
        x = x.item()
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        x = x.tolist()
    if isinstance(x, float):
        if not np.isfinite(x):
            return None
        return x
    if isinstance(x, (int, bool)):
        return x
    if isinstance(x, str):
        s = x.strip()
        return s if s != "" else None
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            cv = _clean_value(v)
            if cv is not None:
                out.append(cv)
        return out if out else None
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            cv = _clean_value(v)
            if cv is not None:
                out[str(k)] = cv
        return out if out else None
    return x if _json_serializable(x) else None


def _prune_empty(d):
    if not isinstance(d, dict):
        return d
    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            pv = _prune_empty(v)
            if pv and len(pv) > 0:
                d[k] = pv
            else:
                d.pop(k, None)
        elif v in (None, [], ""):
            d.pop(k, None)
    return d


def _collect_si_keys(node):
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("SI."):
                out.append((k, v))
            out.extend(_collect_si_keys(v))
    elif isinstance(node, (list, tuple)):
        for v in node:
            out.extend(_collect_si_keys(v))
    return out


def _nest_into_si(root_dict, dotted_key, value):
    dotted_key = dotted_key.removeprefix("SI.")

    parts = dotted_key.split(".")
    cur = root_dict
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    leaf = parts[-1]
    cur[leaf] = _clean_value(value)


def clean_scanimage_metadata(meta: dict) -> dict:
    """
    Build a JSON-serializable, nicely nested dict from ScanImage metadata.

    - All non-'si' top-level keys are kept after cleaning
    - All 'SI.*' keys (from anywhere) are nested under 'si' with 'SI.' prefix stripped
    - So SI.hChannels.channelSave -> metadata['si']['hChannels']['channelSave']
    """
    result = {}
    for k, v in meta.items():
        if k == "si":
            continue
        cv = _clean_value(v)
        if cv is not None:
            result[k] = cv

    result["si"] = {}

    if isinstance(meta.get("si"), dict):
        for k, v in meta["si"].items():
            if not (isinstance(k, str) and k.startswith("SI.")):
                cv = _clean_value(v)
                if cv is not None:
                    result["si"][k] = cv

    si_pairs = _collect_si_keys(meta)
    for dotted_key, val in si_pairs:
        _nest_into_si(result["si"], dotted_key, val)

    result = _prune_empty(result)

    from .scanimage import (
        detect_stack_type,
        get_frame_rate,
        get_num_color_channels,
        get_num_zplanes,
        get_roi_info,
        get_z_step_size,
    )

    if result.get("si"):
        stack_type = detect_stack_type(result)
        result["stack_type"] = stack_type
        result["lbm_stack"] = stack_type == "lbm"
        result["piezo_stack"] = stack_type == "piezo"
        result["num_color_channels"] = get_num_color_channels(result)

        if "num_zplanes" not in result and "num_planes" not in result:
            num_z = get_num_zplanes(result)
            result["num_zplanes"] = num_z
            result["num_planes"] = num_z

        if stack_type != "lbm":
            dz = get_z_step_size(result)
            if dz is not None:
                result["dz"] = dz

        fs = get_frame_rate(result)
        if fs is not None:
            result["fs"] = fs

        roi_info = get_roi_info(result)
        for k, v in roi_info.items():
            if v is not None and k not in result:
                result[k] = v

    from .params import normalize_metadata

    normalize_metadata(result)

    return result


def default_ops():
    """Default options to run pipeline."""
    return {
        "look_one_level_down": False,
        "fast_disk": [],
        "delete_bin": False,
        "mesoscan": False,
        "bruker": False,
        "bruker_bidirectional": False,
        "h5py": [],
        "h5py_key": "data",
        "nwb_file": "",
        "nwb_driver": "",
        "nwb_series": "",
        "save_path0": "",
        "save_folder": [],
        "subfolders": [],
        "move_bin": False,
        "nplanes": 1,
        "nchannels": 1,
        "functional_chan": 1,
        "tau": 1.3,
        "fs": 10.0,
        "force_sktiff": False,
        "frames_include": -1,
        "multiplane_parallel": False,
        "ignore_flyback": [],
        "preclassify": 0.0,
        "save_mat": False,
        "save_NWB": False,
        "combined": True,
        "aspect": 1.0,
        "do_bidiphase": False,
        "bidiphase": 0,
        "bidi_corrected": False,
        "do_registration": True,
        "two_step_registration": False,
        "keep_movie_raw": False,
        "nimg_init": 300,
        "batch_size": 500,
        "maxregshift": 0.1,
        "align_by_chan": 1,
        "reg_tif": False,
        "reg_tif_chan2": False,
        "subpixel": 10,
        "smooth_sigma_time": 0,
        "smooth_sigma": 1.15,
        "th_badframes": 1.0,
        "norm_frames": True,
        "force_refImg": False,
        "pad_fft": False,
        "nonrigid": True,
        "block_size": [
            128,
            128,
        ],
        "snr_thresh": 1.2,
        "maxregshiftNR": 5,
        "1Preg": False,
        "spatial_hp_reg": 42,
        "pre_smooth": 0,
        "spatial_taper": 40,
        "roidetect": True,
        "spikedetect": True,
        "sparse_mode": True,
        "spatial_scale": 0,
        "connected": True,
        "nbinned": 5000,
        "max_iterations": 20,
        "threshold_scaling": 1.0,
        "max_overlap": 0.75,
        "high_pass": 100,
        "spatial_hp_detect": 25,
        "denoise": False,
        "anatomical_only": 0,
        "diameter": 0,
        "cellprob_threshold": 0.0,
        "flow_threshold": 1.5,
        "spatial_hp_cp": 0,
        "pretrained_model": "cyto",
        "soma_crop": True,
        "neuropil_extract": True,
        "inner_neuropil_radius": 2,
        "min_neuropil_pixels": 350,
        "lam_percentile": 50.0,
        "allow_overlap": False,
        "use_builtin_classifier": False,
        "classifier_path": "",
        "chan2_thres": 0.65,
        "baseline": "maximin",
        "win_baseline": 60.0,
        "sig_baseline": 10.0,
        "prctile_baseline": 8.0,
        "neucoeff": 0.7,
    }


def _build_ome_metadata(
    shape: tuple,
    dtype,
    metadata: dict,
    dims: tuple[str, ...] | None = None,
) -> dict:
    """
    Build comprehensive OME-NGFF v0.5 metadata from ScanImage and other metadata.

    Parameters
    ----------
    shape : tuple
        Shape of the array matching dims order
    dtype : np.dtype
        Data type of the array
    metadata : dict
        Metadata dictionary with optional keys
    dims : tuple[str, ...] | None
        Dimension labels like ("T", "Z", "Y", "X"). If None, defaults to TZYX for 4D.

    Returns
    -------
    dict
        Complete OME-NGFF v0.5 metadata attributes
    """
    from mbo_utilities.arrays.features._dim_labels import infer_dims
    from mbo_utilities.arrays.features._dim_tags import (
        dims_to_ome_axes,
        normalize_dims,
    )

    ndim = len(shape)
    if dims is None:
        dims = infer_dims(ndim)
    else:
        dims = normalize_dims(dims)

    from mbo_utilities.metadata.params import get_param, get_voxel_size

    vs = get_voxel_size(metadata)
    frame_rate = get_param(metadata, "fs", default=1.0)

    time_scale = 1.0 / float(frame_rate) if frame_rate else 1.0

    axes = dims_to_ome_axes(dims)

    scale_values = []
    for dim in dims:
        if dim == "T":
            scale_values.append(time_scale)
        elif dim == "Z":
            scale_values.append(float(vs.dz) if vs.dz else 1.0)
        elif dim == "Y":
            scale_values.append(float(vs.dy))
        elif dim == "X":
            scale_values.append(float(vs.dx))
        else:
            scale_values.append(1.0)

    datasets = [
        {
            "path": "0",
            "coordinateTransformations": [{"type": "scale", "scale": scale_values}],
        }
    ]

    multiscales = [
        {
            "name": metadata.get("name", "volume"),
            "axes": axes,
            "datasets": datasets,
            "type": "none",
            "metadata": {
                "method": "none",
                "description": "Single resolution level (no pyramid)",
            },
        }
    ]

    ome_content = {
        "version": "0.5",
        "multiscales": multiscales,
    }

    result = {"ome": ome_content}

    custom_meta = {}

    if "si" in metadata:
        si = metadata["si"]
        custom_meta["scanimage"] = {
            "version": f"{si.get('VERSION_MAJOR', 'unknown')}.{si.get('VERSION_MINOR', 0)}",
            "imaging_system": si.get("imagingSystem", "unknown"),
            "objective_resolution": si.get(
                "objectiveResolution", metadata.get("objective_resolution")
            ),
            "scan_mode": si.get("hScan2D", {}).get("scanMode", "unknown"),
        }

        if "hBeams" in si:
            custom_meta["scanimage"]["laser_power"] = si["hBeams"].get("powers", 0)
            custom_meta["scanimage"]["power_fraction"] = si["hBeams"].get(
                "powerFractions", 0
            )

        if "hRoiManager" in si:
            roi_mgr = si["hRoiManager"]
            custom_meta["scanimage"]["roi"] = {
                "scan_zoom": roi_mgr.get("scanZoomFactor", metadata.get("zoom_factor")),
                "lines_per_frame": roi_mgr.get("linesPerFrame"),
                "pixels_per_line": roi_mgr.get("pixelsPerLine"),
                "line_period": roi_mgr.get("linePeriod"),
                "bidirectional": si.get("hScan2D", {}).get("bidirectional", True),
            }

    if "roi_groups" in metadata:
        custom_meta["roi_groups"] = metadata["roi_groups"]

    acq_meta = {}
    for key in ["acquisition_date", "experimenter", "description", "specimen"]:
        if key in metadata:
            acq_meta[key] = metadata[key]

    if acq_meta:
        custom_meta["acquisition"] = acq_meta

    microscope_meta = {}
    for key in [
        "objective",
        "emission_wavelength",
        "excitation_wavelength",
        "numerical_aperture",
    ]:
        if key in metadata:
            microscope_meta[key] = metadata[key]

    if microscope_meta:
        custom_meta["microscope"] = microscope_meta

    processing_meta = {}
    for key in ["fix_phase", "phasecorr_method", "use_fft", "register_z"]:
        if key in metadata:
            processing_meta[key] = metadata[key]

    if processing_meta:
        custom_meta["processing"] = processing_meta

    if "file_paths" in metadata or "num_files" in metadata:
        custom_meta["source_files"] = {
            "num_files": metadata.get("num_files"),
            "num_timepoints": metadata.get(
                "num_timepoints", metadata.get("num_frames")
            ),
            "frames_per_file": metadata.get("frames_per_file"),
        }

    for key, value in custom_meta.items():
        if _json_serializable(value):
            result[key] = value
        else:
            logger.debug(f"Skipping non-serializable metadata key: {key}")

    for key, value in metadata.items():
        if (
            key
            not in [
                "pixel_resolution",
                "frame_rate",
                "fs",
                "dz",
                "z_step",
                "name",
                "si",
                "roi_groups",
                "acquisition_date",
                "experimenter",
                "description",
                "specimen",
                "objective",
                "emission_wavelength",
                "excitation_wavelength",
                "numerical_aperture",
                "fix_phase",
                "phasecorr_method",
                "use_fft",
                "register_z",
                "file_paths",
                "num_files",
                "num_timepoints",
                "num_frames",
                "nframes",
                "frames_per_file",
            ]
            and key not in result
        ):
            if _json_serializable(value):
                result[key] = value

    return result
