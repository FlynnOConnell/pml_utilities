"""Where a recording's ROI work is saved: its folder and its rois zarr."""

from __future__ import annotations

import hashlib
from pathlib import Path

from mbo_utilities.arrays.features._dim_tags import TAG_REGISTRY, DimensionTag, OutputFilename

__all__ = ["ROIS_SUFFIX", "labels_path", "roi_folder", "slice_name"]

ROIS_SUFFIX = ".rois.zarr"


def roi_folder(fpath, root=None) -> Path:
    """The folder ``fpath``'s ROI work is saved in: beside a file or inside a directory, or under ``root`` as ``<name>_<hash of the full path>`` so two ``data_raw.bin`` never meet."""
    base = Path.cwd() if fpath is None else Path(fpath)
    if root is None:
        return base.parent if base.suffix else base
    digest = hashlib.sha1(str(base.resolve()).encode()).hexdigest()[:8]
    return Path(root) / f"{base.name}_{digest}"


def labels_path(fpath, tag: str = "", root=None) -> Path:
    """``fpath``'s rois zarr: ``<stem>.rois.zarr`` in its ``roi_folder``, or ``<stem>_<tag>.rois.zarr`` for one recording of a file holding several (a MESc unit's ``MSession_0_MUnit_3``)."""
    base = Path(fpath)
    stem = base.stem if base.suffix else base.name
    name = f"{stem}_{tag}{ROIS_SUFFIX}" if tag else f"{stem}{ROIS_SUFFIX}"
    return roi_folder(fpath, root) / name


def slice_name(z: int | None = None, c: int | None = None) -> str:
    """One slice's name in the filename vocabulary (``ch02_zplane03``): the
    C tag before the Z tag, 1-based, only the axes given.
    """
    tags = []
    if c is not None:
        tags.append(DimensionTag(TAG_REGISTRY["C"], int(c) + 1, None))
    if z is not None:
        tags.append(DimensionTag(TAG_REGISTRY["Z"], int(z) + 1, None))
    return OutputFilename(tags).build("")
