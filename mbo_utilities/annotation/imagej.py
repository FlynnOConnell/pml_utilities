"""ImageJ / Fiji and plain-file exchange for the ROI store and trace table.

``RoiSet.zip`` (Fiji's ROI Manager) carries every drawn ROI as its polygon;
a label TIFF is the store's label image as an ImageJ hyperstack; a trace
CSV is one column per trace row; ``footprints.npz`` is the dense
``spatial_footprints`` masknmf's seeding reads.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import roifile
import tifffile

from mbo_utilities.annotation.ngff import _label_volume, _plane_axes_ome
from mbo_utilities.annotation.store import RoiLabelStore
from mbo_utilities.annotation.traces import RoiTrace

__all__ = [
    "export_footprints_npz",
    "export_label_tiff",
    "export_roiset",
    "export_traces",
    "import_roiset",
    "mask_outline",
]


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """``(n, 2)`` float32 ``(x, y)`` vertices of a mask's largest outer
    contour, on pixel centres.
    """
    contours, _ = cv2.findContours(
        np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("empty mask")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float32)


def _roi_name(store: RoiLabelStore, index: int) -> str:
    record = store.rois[index]
    name = f"{record.uid:04d}"
    if 0 <= record.class_index < len(store.label_names):
        name += f"-{store.label_names[record.class_index]}"
    return name


def export_roiset(store: RoiLabelStore, path) -> Path:
    """Every drawn ROI as one entry of a ``RoiSet.zip`` Fiji's ROI Manager
    opens: the polygon as drawn, or a promoted mask's outline. ``c`` and
    ``z`` are stored 1-based, as ImageJ counts them (``position`` on a store
    without plane axes); ``t_position`` 0 puts an ROI on every frame.
    """
    path = Path(path)
    rois = []
    for i, record in enumerate(store.rois):
        points = (
            record.polygon
            if record.polygon is not None
            else mask_outline(store.mask(i))
        )
        roi = roifile.ImagejRoi.frompoints(
            np.asarray(points, np.float32),
            name=_roi_name(store, i),
            c=store.roi_c(i),
            z=store.roi_z(i),
            position=None if store.plane_axes else record.plane,
        )
        roi.t_position = 0
        rois.append(roi)
    if path.exists():
        path.unlink()
    roifile.roiwrite(path, rois, mode="w")
    return path


def _roi_plane(store: RoiLabelStore, roi) -> int:
    """The store plane an ImageJ ROI sits on: ``z_position`` / ``c_position``
    of a hyperstack ROI, ``position`` of a plain stack's, both 1-based; 0
    when it carries none.
    """
    pos = {}
    zname, cname = store.axis_name("z"), store.axis_name("c")
    if zname is not None and roi.z_position:
        pos[zname] = int(roi.z_position) - 1
    if cname is not None and roi.c_position:
        pos[cname] = int(roi.c_position) - 1
    if pos:
        return store.plane_of(pos)
    if roi.position:
        return int(np.clip(int(roi.position) - 1, 0, store.nz - 1))
    return 0


def import_roiset(store: RoiLabelStore, path) -> list[int]:
    """Add every ROI of a ``RoiSet.zip`` (or one ``.roi``) to ``store``,
    each filled from its polygon on the plane its positions name, its
    ImageJ name as the note. Returns the new store indices.
    """
    read = roifile.roiread(path)
    rois = read if isinstance(read, list) else [read]
    added = []
    for roi in rois:
        points = np.asarray(roi.coordinates(), np.float32)
        if points.ndim != 2 or len(points) < 3:
            continue
        pts = np.round(points).astype(np.int32)
        pts[:, 0] = pts[:, 0].clip(0, store.nx - 1)
        pts[:, 1] = pts[:, 1].clip(0, store.ny - 1)
        filled = np.zeros((store.ny, store.nx), np.uint8)
        cv2.fillPoly(filled, [pts], 1)
        index = store.add_roi(
            _roi_plane(store, roi), filled.astype(bool), source="imagej", polygon=points
        )
        if index is None:
            continue
        if roi.name:
            store.set_note(index, str(roi.name))
        added.append(index)
    return added


def export_label_tiff(
    store: RoiLabelStore, path, scale: tuple[float | None, ...] | None = None
) -> Path:
    """The store's label image (ROI ``i`` paints ``i + 1``, the smallest on
    top) as an ImageJ hyperstack TIFF, ``slices`` and ``channels`` from the
    store's plane axes (any other plane axis lands on T); ``scale`` is
    ``(dz, dy, dx)`` in um when known.
    """
    path = Path(path)
    axes, plane_shape = _plane_axes_ome(store)
    volume = _label_volume(store).reshape(*plane_shape, store.ny, store.nx)
    lut = np.zeros(int(volume.max()) + 1, np.uint16)
    for i, record in enumerate(store.rois):
        lut[record.uid] = i + 1
    volume = lut[volume]
    letters = []
    for axis in axes:
        name = axis["name"]
        letters.append("Z" if name == "z" else "C" if name == "c" else "T")
    order = [letters.index(k) for k in "TZC" if k in letters]
    volume = np.transpose(volume, order + [len(letters), len(letters) + 1])
    metadata: dict = {"axes": "".join(letters[i] for i in order) + "YX"}
    kwargs = {}
    dz, dy, dx = scale if scale is not None else (None, None, None)
    if dz:
        metadata["spacing"] = float(dz)
    if dy and dx:
        metadata["unit"] = "um"
        kwargs["resolution"] = (1.0 / float(dx), 1.0 / float(dy))
    tifffile.imwrite(path, volume, imagej=True, metadata=metadata, **kwargs)
    return path


def export_traces(rows: list[RoiTrace], path, fs: float | None = None) -> Path:
    """One CSV: a ``time_s`` column when every row shares a rate (``fs``
    when given), else ``frame``; then one column per row named by
    ``RoiTrace.name``, each placed at the frames it covers, empty elsewhere.
    """
    path = Path(path)
    rates = {t.fs for t in rows if t.fs}
    rate = fs or (rates.pop() if len(rates) == 1 else None)
    starts = [t.frames[0] if t.frames else 0 for t in rows]
    steps = [t.frames[2] if t.frames and len(t.frames) > 2 else 1 for t in rows]
    n = max((s + t.n_frames * k for s, k, t in zip(starts, steps, rows)), default=0)
    columns = []
    for start, step, t in zip(starts, steps, rows):
        column = [""] * n
        values = np.asarray(t.F if t.F is not None else [], np.float32)
        for k, v in enumerate(values):
            at = start + k * step
            if at < n:
                column[at] = f"{float(v):.6g}"
        columns.append(column)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s" if rate else "frame"] + [t.name for t in rows])
        for i in range(n):
            first = f"{i / rate:.6g}" if rate else str(i)
            writer.writerow([first] + [c[i] for c in columns])
    return path


def export_footprints_npz(store: RoiLabelStore, path) -> Path:
    """``spatial_footprints`` ``(H, W, n)`` float32, one column per drawn ROI
    (its weights at its pixels), the file masknmf's seeding reads; ``plane``
    and ``uid`` ride along, one per column.
    """
    path = Path(path)
    n = len(store.rois)
    footprints = np.zeros((store.ny, store.nx, n), np.float32)
    for k in range(n):
        ypix, xpix, weights = store.footprint(k)
        footprints[ypix, xpix, k] = weights
    np.savez(
        path,
        spatial_footprints=footprints,
        plane=np.array([r.plane for r in store.rois], np.int64),
        uid=np.array([r.uid for r in store.rois], np.int64),
    )
    return path
