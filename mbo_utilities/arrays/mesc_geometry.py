"""What a Femtonics file says about where its AOD ROIs and images are.

An AOD unit holds several ROIs, each a straight line (line scan) or a
rectangular patch (chessboard, ribbon scan) the scientist drew on a picture,
and each scanned at its *own* depth (the AOD steers the beam to a different
(x, y, z) per ROI within one frame period). Everything here turns the raw
MESc attrs into microns and pixels; the attrs are read directly with h5py:

- A linescan unit's ``CoordinateMapJSON["maps"][0]["driftEndPoints"]``: one
  ``[[x0, x1], [y0, y1], [z0, z1]]`` per ROI (MESc 4.6.2 adds the midpoint,
  ``[[x0, xm, x1], ...]``), in physical microns, in the same order as the
  unit's ROI axis. ``z0 == z1`` on every real ROI checked. This is the
  segment the AOD actually scanned, ``ROI width x pixelSize`` long; the
  protocol's ``guideLine`` is the shorter hand-drawn guide it was fitted to.
  A chessboard, ribbon or multicube unit's ``["contours"]`` instead: the
  closed outline of each patch, one z per patch. A chessboard or multicube
  patch is its four corners, ``[[x0..x3], [y0..y3], [z0..z3]]``; a ribbon
  (MethodType 9) is one long side then the other walked back, 50-150
  points, with the short edges implied by the closing jumps.
- An image unit's ``ReferenceViewportJSON["viewports"][0]``: where its
  ``[0, 0]`` pixel is, its rotation about z and its size (``geomTransTransl``,
  ``geomTransRot``, ``width``, ``height``).
- A Z-stack unit's ``MinZ`` / ``MaxZ`` / ``ZDim`` attrs: the stack's depth
  range (relative to that Z origin) and slice count, uniformly stepped.

Z-origin: the stack's Z origin is ``geomTransTransl[2]``, not the unit's
``LabelingOriginTransl`` attr.

These coordinates are not one frame. A scan and the picture it was drawn on
share one; another unit's differ by whatever moved between them, and
``geomTransTransl`` has no stage term. So ROIs land on their own picture as
stored and on a Z-stack only through a measured shift
(``analysis.mesc_placement``).

Every reader returns ``None`` when its attr is missing (older MESc version,
non-AOD acquisition) so a caller can skip the overlay rather than crash.
"""

from __future__ import annotations

import json

import h5py
import numpy as np

__all__ = [
    "roi_outlines_um",
    "viewport_geometry",
    "zstack_depth_info",
    "z_step_um",
    "um_to_pixels",
    "image_overlays",
    "line_positions",
]


def _outlines_of(unit) -> list[np.ndarray] | None:
    """:func:`roi_outlines_um` of an open unit group."""
    raw = unit.attrs.get("CoordinateMapJSON")
    if not raw:
        return None
    maps = json.loads(raw).get("maps") or []
    if not maps:
        return None
    outlines = maps[0].get("driftEndPoints") or maps[0].get("contours")
    if not outlines:
        return None
    return [np.asarray(seg, dtype=float) for seg in outlines]


def _is_patch(seg) -> bool:
    """A closed outline: a chessboard or multicube patch's four corners, or a
    ribbon's two long sides (one forward, one back); a line has 2 or 3 points.
    """
    return seg.shape[1] >= 4


def _far_end(seg) -> int:
    """Index of the point a length is measured to: a line's last point, the
    end of a patch's first side (corner 1 of 4, a ribbon's first long side).
    """
    return seg.shape[1] // 2 - 1 if _is_patch(seg) else -1


def _viewport_of(unit) -> dict | None:
    """:func:`viewport_geometry` of an open unit group."""
    raw = unit.attrs.get("ReferenceViewportJSON")
    if not raw:
        return None
    viewports = json.loads(raw).get("viewports") or []
    if not viewports:
        return None
    vp = viewports[0]
    transl = tuple(float(v) for v in vp["geomTransTransl"])
    width, height = float(vp["width"]), float(vp["height"])
    # MEScan stamps an RTMC reference unit with a 1 um square at the origin:
    # a placeholder, not where the reference region was scanned
    if transl == (0.0, 0.0, 0.0) and width == 1.0 and height == 1.0:
        return None
    # [x, y, z, w]; FemtoAPI's quatMult is anti-Hamilton, so its q^-1 p q is this active rotation
    _qx, _qy, qz, qw = vp.get("geomTransRot", (0.0, 0.0, 0.0, 1.0))
    return {
        "transl": transl,
        "width": width,
        "height": height,
        "angle": 2.0 * float(np.arctan2(qz, qw)),
    }


def roi_outlines_um(mesc_path, unit_key: str) -> list[np.ndarray] | None:
    """One ``(3, N)`` ``[xs, ys, zs]`` array per ROI, in microns: the ends of
    a line scan's line (``driftEndPoints``, N = 2, or 3 with the midpoint) or
    the closed outline of a patch (``contours``: N = 4 corners, or a ribbon's
    two long sides, in drawing order), in the order of the unit's ROI axis.

    ``None`` if this unit has no ``CoordinateMapJSON`` or neither key inside
    it (present on real AOD units, not guaranteed on others).
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        return None if unit is None else _outlines_of(unit)


def viewport_geometry(mesc_path, unit_key: str) -> dict | None:
    """``{"transl": (x, y, z), "width": um, "height": um, "angle": rad}`` for a
    unit's FOV: a pixel at ``p`` microns from the corner is at
    ``transl + R(angle) p``.

    ``None`` if this unit has no ``ReferenceViewportJSON``, or only the
    1 um placeholder MEScan writes on an RTMC reference unit.
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        return None if unit is None else _viewport_of(unit)


def zstack_depth_info(mesc_path, unit_key: str) -> dict | None:
    """``{"transl_z", "min_z", "max_z", "zdim"}`` for a Z-stack unit's depth axis.

    ``None`` if this unit is missing ``ReferenceViewportJSON``, ``MinZ``,
    ``MaxZ`` or ``ZDim``. ``transl_z`` is ``geomTransTransl[2]`` (module
    docstring: why this and not ``LabelingOriginTransl``).
    """
    vp = viewport_geometry(mesc_path, unit_key)
    if vp is None:
        return None
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None or not {"MinZ", "MaxZ", "ZDim"} <= set(unit.attrs.keys()):
            return None
        # NOTE: every stack seen so far is bottomToTop (0); FemtoAPI negates z otherwise
        if unit.attrs.get("ScanningDirection", 0) != 0:
            raise ValueError(
                f"{unit_key}: slice depths of a topToBottom stack are unverified"
            )
        return {
            "transl_z": vp["transl"][2],
            "min_z": float(unit.attrs["MinZ"]),
            "max_z": float(unit.attrs["MaxZ"]),
            "zdim": int(unit.attrs["ZDim"]),
        }


def z_step_um(depth: dict) -> float:
    return (depth["max_z"] - depth["min_z"]) / max(depth["zdim"] - 1, 1)


def um_to_pixels(
    points_xy_um: np.ndarray, viewport: dict, ny: int, nx: int
) -> np.ndarray:
    """``(N, 2)`` physical ``[x, y]`` microns -> ``(N, 2)`` pixel ``[col, row]``.

    ``geomTransTransl`` is the micron position of the array's ``[0, 0]``
    pixel and rows increase with +y: checked on the 2026-07-30 spine file
    by ranking each line's brightness on the snapshot it was drawn on
    (``BackgroundImagePath``) against random same-shaped lines nearby - 36
    of 40 in-plane lines beat chance this way (median rank 0.76) while the
    mirrored mapping sits at chance (19 of 40, median 0.49). The rotation is
    what puts a 45 degree rig's patches inside their pictures, with their
    edges along the pixel axes.
    """
    cos, sin = np.cos(viewport["angle"]), np.sin(viewport["angle"])
    dx, dy = (np.asarray(points_xy_um, dtype=float) - viewport["transl"][:2]).T
    col = (cos * dx + sin * dy) * nx / viewport["width"]
    row = (cos * dy - sin * dx) * ny / viewport["height"]
    return np.stack([col, row], axis=1)


def image_overlays(
    mesc_path,
    unit_key: str,
    image_key: str,
    image_path=None,
    shift_um=(0.0, 0.0, 0.0),
) -> list[dict]:
    """One multi-ROI unit's outlines on one image, in its pixels.

    ``image_key`` is a snapshot (``MethodType`` 1) or a Z-stack (``MethodType``
    2), in ``image_path`` when another file holds it. Every point is moved by
    ``shift_um`` first: nothing for the picture a scan was drawn on, a
    ``mesc_placement.StackPlacement``'s shift for a Z-stack. A snapshot carries
    every ROI, which is what the MESc GUI draws on it, whatever its depth. A
    Z-stack carries the ROIs inside its field **and its depth range**, each on
    the slice nearest its depth. An ROI scanned above or below the stack is
    left out: the stack holds no picture of it, and drawing it on an edge slice
    puts an outline on tissue it was never scanned in.

    One dict per ROI, in ROI order::

        roi        0-based index on that unit's ROI axis
        kind       ``"line"`` (open polyline) or ``"patch"`` (closed, first
                   point repeated at the end)
        pixels     ``(N, 2)`` ``[col, row]`` on the image
        color      ``(r, g, b)`` the MESc GUI drew it in (``ROIJSON``), or None
        slice      0-based Z-stack slice, None on a snapshot
    """
    image_path = mesc_path if image_path is None else image_path
    vp = viewport_geometry(image_path, image_key)
    depth = zstack_depth_info(image_path, image_key)
    with h5py.File(image_path, "r") as f:
        image = f.get(image_key)
        if vp is None or image is None or "Channel_0" not in image:
            return []
        ny, nx = (int(v) for v in image["Channel_0"].shape[-2:])
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        outlines = None if unit is None else _outlines_of(unit)
        if outlines is None:
            return []
        raw = unit.attrs.get("ROIJSON")
    outlines = [seg + np.asarray(shift_um, dtype=float)[:, None] for seg in outlines]
    rois = (json.loads(raw).get("rois") or []) if raw else []
    # "#AARRGGBB", black where MESc left the colour unset; a hand-edited
    # ROI list no longer pairs with the scan
    colors = [None] * len(outlines)
    if len(rois) == len(outlines):
        colors = [
            None
            if r["color"].upper().endswith("000000")
            else tuple(int(r["color"][i : i + 2], 16) / 255.0 for i in (3, 5, 7))
            for r in rois
        ]
    out = []
    for i, seg in enumerate(outlines):
        k = None
        if depth is not None:
            col, row = um_to_pixels(seg[:2].mean(axis=1)[None], vp, ny, nx)[0]
            z_um = float(seg[2].mean()) - depth["transl_z"]
            k = int(np.round((z_um - depth["min_z"]) / z_step_um(depth)))
            if not (0 <= col <= nx and 0 <= row <= ny and 0 <= k < depth["zdim"]):
                continue
        kind = "patch" if _is_patch(seg) else "line"
        xy = seg[:2].T
        if kind == "patch":
            xy = np.vstack([xy, xy[:1]])
        out.append(
            {
                "roi": i,
                "kind": kind,
                "pixels": um_to_pixels(xy, vp, ny, nx),
                "color": colors[i],
                "slice": k,
            }
        )
    return out


def line_positions(
    mesc_path, unit_key: str, sample_counts: list[int] | None = None
) -> list[dict] | None:
    """Where each of a multi-ROI unit's scanned lines or patches was drawn,
    one dict per ROI in ROI order, or None when the unit has no geometry.

    Stored microns, in the frame of the scan and its picture::

        index       ROI index (the unit's ROI / Z axis position)
        start_um    ``[x, y, z]`` of the first point
        end_um      ``[x, y, z]`` of the last point (a patch: its second corner)
        length_um   XY length first -> last point
        sample_um   microns per pixel along the line, from ``sample_counts``
                    (``mesc_roi_extents[i]["width"]``); None without them
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None:
            return None
        outlines = _outlines_of(unit)
        if not outlines:
            return None
    out = []
    for i, seg in enumerate(outlines):
        seg = np.asarray(seg, dtype=float)
        end = _far_end(seg)
        length = float(np.hypot(*(seg[:2, end] - seg[:2, 0])))
        n = (
            None
            if sample_counts is None or i >= len(sample_counts)
            else int(sample_counts[i])
        )
        out.append(
            {
                "index": i,
                "start_um": [float(v) for v in seg[:, 0]],
                "end_um": [float(v) for v in seg[:, end]],
                "length_um": length,
                "sample_um": (length / n) if n else None,
            }
        )
    return out
