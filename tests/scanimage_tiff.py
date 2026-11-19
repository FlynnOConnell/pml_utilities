"""Writes small genuine ScanImage BigTIFFs for tests: the v4 static header ScanImage puts before the first IFD, the ROI group JSON, and one uncompressed int16 page per frame and channel.

Every pixel of an mROI strip holds ``page * 8 + strip``, counted from ``first_page`` across files, and every fly-to line holds -1,
so a test can say which raw page and strip any value came from.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

FLY_TO_VALUE = -1


def _matlab_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, (list, tuple)):
        return "[" + " ".join(_matlab_value(v) for v in value) + "]"
    if value == float("inf"):
        return "Inf"
    return repr(value)


def scanimage_header(keys: dict | None = None, **overrides) -> dict:
    """``SI.*`` frame data of a plain single-channel, single-plane acquisition.

    ``overrides`` replace keys written with ``__`` for the dots
    (``hStackManager__numSlices=3``); ``keys`` adds full keys, for names that
    hold ``__`` themselves (``SI.hScan2D.virtualChannelSettings__1.source``).
    """
    header = {
        "SI.VERSION_MAJOR": 2023,
        "SI.objectiveResolution": 100.0,
        "SI.hChannels.channelSave": 1,
        "SI.hRoiManager.scanFrameRate": 30.0,
        "SI.hRoiManager.scanVolumeRate": 30.0,
        "SI.hRoiManager.linePeriod": 1e-4,
        "SI.hRoiManager.mroiEnable": False,
        "SI.hRoiManager.scanZoomFactor": 1,
        "SI.hScan2D.logAverageFactor": 1,
        "SI.hStackManager.enable": False,
        "SI.hStackManager.numSlices": 1,
        "SI.hStackManager.framesPerSlice": 1,
        "SI.hStackManager.stackZStepSize": 1,
        "SI.hStackManager.numVolumes": 1,
        "SI.hStackManager.stackMode": "slow",
        "SI.hFastZ.enable": False,
        "SI.hFastZ.discardFlybackFrames": False,
        "SI.hFastZ.numDiscardFlybackFrames": 0,
    }
    header.update({"SI." + k.replace("__", "."): v for k, v in overrides.items()})
    return header | (keys or {})


def page_shape(roi_heights: list[int], roi_width: int, fly_to_lines: int) -> tuple[int, int]:
    return sum(roi_heights) + fly_to_lines * (len(roi_heights) - 1), roi_width


def _page_pixels(page: int, roi_heights, roi_width, fly_to_lines) -> np.ndarray:
    rows = []
    for strip, height in enumerate(roi_heights):
        if strip:
            rows.append(np.full((fly_to_lines, roi_width), FLY_TO_VALUE, np.int16))
        rows.append(np.full((height, roi_width), page * 8 + strip, np.int16))
    return np.concatenate(rows)


def write_scanimage_tiff(
    path: Path,
    num_pages: int,
    header: dict,
    roi_heights: list[int] = (16,),
    roi_width: int = 12,
    fly_to_lines: int = 0,
    first_page: int = 0,
    size_um: float = 1.0,
) -> Path:
    """Write ``num_pages`` pages of one ScanImage file; ``first_page`` continues the page count of the files before it.

    ``size_um`` is each scanfield's ``sizeXY`` in degrees; times ``SI.objectiveResolution`` it is the strip's width in µm.
    """
    roi_heights = list(roi_heights)
    height, width = page_shape(roi_heights, roi_width, fly_to_lines)
    header = dict(header)
    if len(roi_heights) > 1:
        header["SI.hRoiManager.mroiEnable"] = True
        header.setdefault(
            "SI.hScan2D.flytoTimePerScanfield", fly_to_lines * header["SI.hRoiManager.linePeriod"]
        )
    frame_data = "\n".join(f"{k} = {_matlab_value(v)}" for k, v in header.items()).encode() + b"\0"
    rois = [
        {"scanfields": {"sizeXY": [size_um, size_um * h / roi_width], "pixelResolutionXY": [roi_width, h]}}
        for h in roi_heights
    ]
    roi_json = json.dumps({"RoiGroups": {"imagingRoiGroup": {"rois": rois}}}).encode() + b"\0"

    static = struct.pack("<IIII", 0x07030301, 4, len(frame_data), len(roi_json)) + frame_data + roi_json
    first_ifd = 16 + len(static) + (-len(static)) % 8

    description_len = 128
    software = b"SI.VERSION_MAJOR = 2023".ljust(31, b" ") + b"\0"
    num_tags = 12
    ifd_len = 8 + num_tags * 20 + 8
    pixel_len = height * width * 2
    page_len = ifd_len + description_len + len(software) + pixel_len

    with open(path, "wb") as f:
        f.write(struct.pack("<2sHHHQ", b"II", 43, 8, 0, first_ifd))
        f.write(static)
        f.write(b"\0" * (first_ifd - f.tell()))
        for i in range(num_pages):
            ifd_at = first_ifd + i * page_len
            description_at = ifd_at + ifd_len
            software_at = description_at + description_len
            pixels_at = software_at + len(software)
            next_ifd = ifd_at + page_len if i < num_pages - 1 else 0
            page = first_page + i
            description = f"frameNumbers = {page + 1}\nframeTimestamps_sec = {page / 30:.6f}\n".encode()
            description = description.ljust(description_len - 1, b" ") + b"\0"
            tags = [
                (256, 4, 1, width),
                (257, 4, 1, height),
                (258, 3, 1, 16),
                (259, 3, 1, 1),
                (262, 3, 1, 1),
                (270, 2, description_len, description_at),
                (273, 16, 1, pixels_at),
                (277, 3, 1, 1),
                (278, 4, 1, height),
                (279, 16, 1, pixel_len),
                (305, 2, len(software), software_at),
                (339, 3, 1, 2),
            ]
            f.write(struct.pack("<Q", num_tags))
            for tag, typ, count, value in tags:
                f.write(struct.pack("<HHQQ", tag, typ, count, value))
            f.write(struct.pack("<Q", next_ifd))
            f.write(description)
            f.write(software)
            f.write(_page_pixels(page, roi_heights, roi_width, fly_to_lines).tobytes())
    return path


def write_scanimage_files(
    directory: Path, pages_per_file: list[int], header: dict, stem: str = "scan", **kwargs
) -> list[Path]:
    """Split one acquisition over several files named like ScanImage's ``<stem>_00001_00001.tif``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths, first_page = [], 0
    for i, n in enumerate(pages_per_file):
        path = directory / f"{stem}_00001_{i + 1:05d}.tif"
        paths.append(write_scanimage_tiff(path, n, header, first_page=first_page, **kwargs))
        first_page += n
    return paths


def strip_value(page: int, strip: int = 0) -> int:
    return page * 8 + strip
