"""Where a Femtonics AOD unit's drawn ROIs sit inside a Z-stack.

An AOD unit holds several ROIs, each a straight line (line scan) or a
rectangular patch (chessboard, ribbon scan) the scientist drew on a
reference Z-stack, and each scanned at its *own* depth (the AOD steers the
beam to a different (x, y, z) per ROI within one frame period). So a unit's
ROIs are scattered across the stack: some slices carry one, some carry
several, most carry none. Everything here turns the raw MESc attrs into the
per-ROI placement a viewer needs to show that truthfully: which slice each
ROI belongs on, how far off that slice it really is, how long it is, and
which end is the start of the kymograph.

Neither ``lab4/convert/aod/aod.py`` nor ``MescArray.metadata`` keeps what's
needed for this: both reduce a ROI's guideline to a centroid + rotation
quaternion (``mesc_centroids`` / ``mesc_rotations``), enough to know
roughly where a ROI is, not its start/end segment. The attrs are read
directly with h5py:

- A linescan unit's ``CoordinateMapJSON["maps"][0]["driftEndPoints"]``: one
  ``[[x0, x1], [y0, y1], [z0, z1]]`` per ROI, in physical microns, in the
  same order as the unit's ROI axis. ``z0 == z1`` on every real ROI checked.
  A chessboard or ribbon unit's ``["contours"]`` instead: the four corners
  of each patch, ``[[x0..x3], [y0..y3], [z0..z3]]``, one z per patch.
- A Z-stack unit's ``ReferenceViewportJSON["viewports"][0]``: the FOV's
  physical placement (``geomTransTransl``, ``width``, ``height``), XY corner
  and the stack's own Z origin, in the same micron frame as the endpoints.
- A Z-stack unit's ``MinZ`` / ``MaxZ`` / ``ZDim`` attrs: the stack's depth
  range (relative to that Z origin) and slice count, uniformly stepped
  (``ZModeDebugString`` was ``"slow"`` on the units checked, a real raster
  Z-stack, not an arbitrarily placed AOD scan).

Z-origin: the stack's Z origin is ``geomTransTransl[2]``, not the unit's
``LabelingOriginTransl`` attr. Using the latter put a real linescan's depth
range entirely outside every candidate Z-stack, which is how the first
overlay landed on a FOV with no visible dendrite.

Every reader returns ``None`` when its attr is missing (older MESc version,
non-AOD acquisition) so a caller can skip the overlay rather than crash.
"""

from __future__ import annotations

import json

import h5py
import numpy as np

__all__ = [
    "roi_outlines_um",
    "linescan_endpoints_um",
    "viewport_geometry",
    "zstack_depth_info",
    "slice_depths_um",
    "roi_slice_indices",
    "roi_placements",
    "slices_with_rois",
    "neighbour_slices",
    "um_to_pixels",
]


# ---------------------------------------------------------------------------
# raw attr readers
# ---------------------------------------------------------------------------


def roi_outlines_um(mesc_path, unit_key: str) -> list[np.ndarray] | None:
    """One ``(3, N)`` ``[xs, ys, zs]`` array per ROI, in microns: the two
    ends of a line scan's line (``driftEndPoints``, N = 2) or the corners of
    a chessboard or ribbon patch (``contours``, N = 4, in drawing order), in
    the order of the unit's ROI axis.

    ``None`` if this unit has no ``CoordinateMapJSON`` or neither key inside
    it (present on real AOD units, not guaranteed on others).
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None or "CoordinateMapJSON" not in unit.attrs:
            return None
        raw = unit.attrs["CoordinateMapJSON"]
        if not raw:
            return None
        doc = json.loads(raw)
        maps = doc.get("maps") or []
        if not maps:
            return None
        outlines = maps[0].get("driftEndPoints") or maps[0].get("contours")
        if not outlines:
            return None
        return [np.asarray(seg, dtype=float) for seg in outlines]


# the line-scan name the viewers import; a line's outline is its two ends
linescan_endpoints_um = roi_outlines_um


def viewport_geometry(mesc_path, unit_key: str) -> dict | None:
    """``{"transl": (x, y, z), "width": um, "height": um}`` for a unit's FOV.

    ``None`` if this unit has no ``ReferenceViewportJSON``.
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None or "ReferenceViewportJSON" not in unit.attrs:
            return None
        raw = unit.attrs["ReferenceViewportJSON"]
        if not raw:
            return None
        doc = json.loads(raw)
        viewports = doc.get("viewports") or []
        if not viewports:
            return None
        vp = viewports[0]
        return {
            "transl": tuple(float(v) for v in vp["geomTransTransl"]),
            "width": float(vp["width"]),
            "height": float(vp["height"]),
        }


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
        return {
            "transl_z": vp["transl"][2],
            "min_z": float(unit.attrs["MinZ"]),
            "max_z": float(unit.attrs["MaxZ"]),
            "zdim": int(unit.attrs["ZDim"]),
        }


# ---------------------------------------------------------------------------
# depth placement
# ---------------------------------------------------------------------------


def _z_step(depth: dict) -> float:
    return (depth["max_z"] - depth["min_z"]) / max(depth["zdim"] - 1, 1)


def slice_depths_um(depth: dict) -> np.ndarray:
    """Depth of every slice, relative to the stack's own Z origin, ``(zdim,)``.

    Slice 0 is at ``min_z``; the slider's slice ``k`` is at
    ``min_z + k * step``. Relative (not absolute) so it reads like the
    numbers in the MESc GUI, which shows depth from the stack's origin.
    """
    return depth["min_z"] + np.arange(depth["zdim"]) * _z_step(depth)


def roi_slice_indices(lines_um: list[np.ndarray], depth: dict) -> np.ndarray:
    """0-based Z-stack slice index per ROI, from each line's mean z.

    Rounds to the nearest slice: the Z slider moves in integer steps, so a
    ROI's line belongs on exactly one slice. Indices are clipped into
    ``[0, zdim - 1]``; a ROI scanned outside the captured depth range still
    gets a (clamped) slice, and :func:`roi_placements` flags it, rather than
    silently vanishing.
    """
    return np.array([p["slice"] for p in roi_placements(lines_um, depth)], dtype=int)


def roi_placements(
    lines_um: list[np.ndarray],
    depth: dict,
    sample_counts: list[int] | None = None,
) -> list[dict]:
    """Everything a viewer needs to place and describe one ROI's outline
    (a line's two ends, a patch's four corners).

    One dict per ROI, in ROI order::

        index        ROI index (the unit's ROI / Z axis position)
        z_um         the line's depth relative to the stack's Z origin
        slice        nearest 0-based slice (clipped into the stack)
        slice_z_um   that slice's depth, same frame as ``z_um``
        dz_um        ``z_um - slice_z_um``: how far the line really sits
                     off the slice it is drawn on (< half a step when in range)
        in_range     False when the ROI was scanned above/below the stack;
                     ``slice`` is then the nearest edge slice and ``dz_um``
                     says by how much it misses
        tilted       True when the points differ in z (never seen on real
                     data; the ROI is then drawn at its mean depth)
        length_um    XY length of the drawn line, or of a patch's first edge
        sample_um    microns per kymograph column, when ``sample_counts``
                     (pixels along each line, ``mesc_roi_extents[i]["width"]``)
                     is given; else ``None``
    """
    step = _z_step(depth)
    depths = slice_depths_um(depth)
    zdim = depth["zdim"]
    out = []
    for i, seg in enumerate(lines_um):
        seg = np.asarray(seg, dtype=float)
        z_um = float(seg[2].mean()) - depth["transl_z"]
        raw_idx = int(np.round((z_um - depth["min_z"]) / step)) if step else 0
        idx = int(np.clip(raw_idx, 0, zdim - 1))
        dxy = seg[:2, 1] - seg[:2, 0]
        length = float(np.hypot(*dxy))
        n = None if sample_counts is None else int(sample_counts[i])
        out.append(
            {
                "index": i,
                "z_um": z_um,
                "slice": idx,
                "slice_z_um": float(depths[idx]),
                "dz_um": z_um - float(depths[idx]),
                "in_range": 0 <= raw_idx < zdim,
                "tilted": not np.allclose(seg[2], seg[2, 0]),
                "length_um": length,
                "sample_um": (length / n) if n else None,
            }
        )
    return out


def slices_with_rois(placements: list[dict]) -> dict[int, list[int]]:
    """``{slice: [roi indices]}`` for the slices that carry at least one line,
    ascending by slice. Most slices of a stack carry none."""
    table: dict[int, list[int]] = {}
    for p in placements:
        table.setdefault(p["slice"], []).append(p["index"])
    return dict(sorted(table.items()))


def neighbour_slices(current: int, occupied) -> tuple[int | None, int | None]:
    """The nearest line-carrying slice strictly below and above ``current``.

    ``occupied`` is any iterable of slice indices (e.g. the keys of
    :func:`slices_with_rois`). Either side is ``None`` when nothing is there,
    so a viewer can grey out its prev/next-depth buttons.
    """
    occ = sorted(set(int(s) for s in occupied))
    below = [s for s in occ if s < current]
    above = [s for s in occ if s > current]
    return (below[-1] if below else None, above[0] if above else None)


# ---------------------------------------------------------------------------
# XY placement
# ---------------------------------------------------------------------------


def um_to_pixels(
    points_xy_um: np.ndarray,
    viewport: dict,
    ny: int,
    nx: int,
    flip_y: bool = False,
) -> np.ndarray:
    """``(N, 2)`` physical ``[x, y]`` microns -> ``(N, 2)`` pixel ``[col, row]``.

    ``geomTransTransl`` is the micron position of the array's ``[0, 0]``
    pixel and rows increase with +y: checked on the 2026-07-30 spine file
    by ranking each line's brightness on the snapshot it was drawn on
    (``BackgroundImagePath``) against random same-shaped lines nearby - 36
    of 40 in-plane lines beat chance this way (median rank 0.76) while the
    mirrored mapping sits at chance (19 of 40, median 0.49). The centre
    reading is ruled out outright: it would put every line outside the
    field. ``flip_y=True`` mirrors rows for a rig that saves the other way
    round (lab4's reference converters flip saved images in Y for display,
    Femtonics.md Sec 5 #1).
    """
    tx, ty, _tz = viewport["transl"]
    px_w = viewport["width"] / nx
    px_h = viewport["height"] / ny
    pts = np.asarray(points_xy_um, dtype=float)
    col = (pts[:, 0] - tx) / px_w
    row = (pts[:, 1] - ty) / px_h
    if flip_y:
        row = ny - row
    return np.stack([col, row], axis=1)
