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
  ``[[x0, x1], [y0, y1], [z0, z1]]`` per ROI (MESc 4.6.2 adds the midpoint,
  ``[[x0, xm, x1], ...]``), in physical microns, in the same order as the
  unit's ROI axis. ``z0 == z1`` on every real ROI checked. This is the
  segment the AOD actually scanned, ``ROI width x pixelSize`` long; the
  protocol's ``guideLine`` is the shorter hand-drawn guide it was fitted to.
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

from mbo_utilities.arrays.mesc import list_mesc_units

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
    "image_overlays",
    "zstack_contents",
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
    return {"transl": transl, "width": width, "height": height}


def roi_outlines_um(mesc_path, unit_key: str) -> list[np.ndarray] | None:
    """One ``(3, N)`` ``[xs, ys, zs]`` array per ROI, in microns: the ends of
    a line scan's line (``driftEndPoints``, N = 2, or 3 with the midpoint) or
    the corners of a chessboard or ribbon patch (``contours``, N = 4, in
    drawing order), in the order of the unit's ROI axis.

    ``None`` if this unit has no ``CoordinateMapJSON`` or neither key inside
    it (present on real AOD units, not guaranteed on others).
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        return None if unit is None else _outlines_of(unit)


# the line-scan name the viewers import; a line's outline is its two ends
linescan_endpoints_um = roi_outlines_um


def viewport_geometry(mesc_path, unit_key: str) -> dict | None:
    """``{"transl": (x, y, z), "width": um, "height": um}`` for a unit's FOV.

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
        return {
            "transl_z": vp["transl"][2],
            "min_z": float(unit.attrs["MinZ"]),
            "max_z": float(unit.attrs["MaxZ"]),
            "zdim": int(unit.attrs["ZDim"]),
        }


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
        length_um    XY length of the scanned line (first to last point), or
                     of a patch's first edge
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
        # a patch is its four corners; a line runs start -> (midpoint) -> end
        end = 1 if seg.shape[1] == 4 else -1
        dxy = seg[:2, end] - seg[:2, 0]
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


def image_overlays(
    mesc_path,
    image_key: str,
    units: list[dict] | None = None,
    plane_tol_um: float = 1.0,
) -> list[dict]:
    """The ROI outlines that belong on one image unit, in its pixels.

    ``image_key`` is a snapshot (``MethodType`` 1) or a Z-stack (``MethodType``
    2). A snapshot carries the ROIs of every multi-ROI unit whose
    ``BackgroundImagePath`` names it, which is what the MESc GUI draws on it:
    all of them, whatever their depth. A Z-stack carries the ROIs of every
    multi-ROI unit whose outlines fall inside its field **and its depth
    range**, each on the slice nearest its depth (:func:`roi_placements`).
    An ROI scanned above or below the stack is left out: the stack holds no
    picture of it, and drawing it on an edge slice puts an outline on tissue
    it was never scanned in.

    One dict per ROI, unit order then ROI order::

        unit       the multi-ROI unit's key, ``MSession_0/MUnit_19``
        munit      its ``MUnit_19``
        roi        0-based index on that unit's ROI axis
        kind       ``"line"`` (open polyline) or ``"patch"`` (closed, 5 points)
        pixels     ``(N, 2)`` ``[col, row]`` on the image
        color      ``(r, g, b)`` the MESc GUI drew it in (``ROIJSON``), or None
        z_um       absolute depth
        dz_um      snapshot: offset from its plane; stack: from the slice it
                   is drawn on
        slice      0-based Z-stack slice, None on a snapshot
        on_plane   snapshot: ``|dz_um| <= plane_tol_um``; stack: always True,
                   an ROI outside the depth range having been left out

    A unit whose outline count differs from its ROI count is left out:
    pairing them by order would be a guess. ``units`` is
    ``list_mesc_units(mesc_path)`` when the caller already holds it.
    """
    if units is None:
        units = list_mesc_units(mesc_path)
    out: list[dict] = []
    with h5py.File(mesc_path, "r") as f:
        image = f.get(image_key)
        if image is None or "Channel_0" not in image:
            return out
        vp = _viewport_of(image)
        if vp is None:
            return out
        ny, nx = (int(v) for v in image["Channel_0"].shape[-2:])
        tx, ty, tz = vp["transl"]
        depth = None
        if {"MinZ", "MaxZ", "ZDim"} <= set(image.attrs.keys()):
            depth = {
                "transl_z": tz,
                "min_z": float(image.attrs["MinZ"]),
                "max_z": float(image.attrs["MaxZ"]),
                "zdim": int(image.attrs["ZDim"]),
            }
        for u in units:
            unit = f.get(u["key"])
            if unit is None or u["key"] == image_key:
                continue
            outlines = _outlines_of(unit)
            if outlines is None or len(outlines) != u["nrois"]:
                continue
            if depth is None:
                bg = unit.attrs.get("BackgroundImagePath", "")
                bg = bg.decode() if isinstance(bg, bytes) else str(bg)
                if bg.strip("/") != image_key.strip("/"):
                    continue
                placements = None
            else:
                placements = roi_placements(outlines, depth)
            raw = unit.attrs.get("ROIJSON")
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
            for i, seg in enumerate(outlines):
                x, y = float(seg[0].mean()), float(seg[1].mean())
                if depth is not None and not (
                    tx <= x <= tx + vp["width"]
                    and ty <= y <= ty + vp["height"]
                    and placements[i]["in_range"]
                ):
                    continue
                kind = "patch" if seg.shape[1] == 4 else "line"
                xy = seg[:2].T
                if kind == "patch":
                    xy = np.vstack([xy, xy[:1]])
                z_um = float(seg[2].mean())
                if placements is None:
                    dz, k, on = z_um - tz, None, abs(z_um - tz) <= plane_tol_um
                else:
                    dz, k, on = placements[i]["dz_um"], placements[i]["slice"], True
                out.append(
                    {
                        "unit": u["key"],
                        "munit": u["munit"],
                        "roi": i,
                        "kind": kind,
                        "pixels": um_to_pixels(xy, vp, ny, nx),
                        "color": colors[i],
                        "z_um": z_um,
                        "dz_um": float(dz),
                        "slice": k,
                        "on_plane": bool(on),
                    }
                )
    return out


def zstack_contents(mesc_path, units: list[dict] | None = None) -> dict[str, list[str]]:
    """Which scans each Z-stack of the file holds: stack key -> the keys of
    every multi-ROI unit with at least one outline inside the stack's field
    and depth range (:func:`image_overlays`), in unit order. A snapshot's
    scans are its ``scans`` entry from ``list_mesc_units``; a stack has no
    such link, only geometry."""
    if units is None:
        units = list_mesc_units(mesc_path)
    return {
        u["key"]: list(dict.fromkeys(r["unit"] for r in image_overlays(mesc_path, u["key"], units)))
        for u in units
        if u["kind"] == "zstack"
    }


def line_positions(mesc_path, unit_key: str, sample_counts: list[int] | None = None) -> list[dict] | None:
    """Where each of a multi-ROI unit's scanned lines or patches sits, one
    dict per ROI in ROI order, or None when the unit has no geometry.

    Everything is in the file's absolute micron frame (the one every
    ``ReferenceViewportJSON`` and ``driftEndPoints`` share)::

        index       ROI index (the unit's ROI / Z axis position)
        start_um    ``[x, y, z]`` of the first point
        end_um      ``[x, y, z]`` of the last point (a patch: its second corner)
        z_um        mean depth of the ROI
        length_um   XY length first -> last point
        sample_um   microns per pixel along the line, from ``sample_counts``
                    (``mesc_roi_extents[i]["width"]``); None without them
        dz_um       depth against the snapshot the lines were drawn on (the
                    unit's ``BackgroundImagePath``); None without one
        stack       the first Z-stack whose field and depth range hold the
                    line (:func:`image_overlays`); ``slice`` its 0-based
                    slice nearest the line, ``slice_dz_um`` the line's offset
                    from it, ``in_stack`` True; all None when no stack was
                    scanned around the line

    The snapshot's plane is where MESc draws every line whatever its depth;
    ``dz_um`` says how far off that plane each one really is.
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None:
            return None
        outlines = _outlines_of(unit)
        if not outlines:
            return None
        raw = unit.attrs.get("BackgroundImagePath")
        path = raw.decode() if isinstance(raw, bytes) else raw
        snapshot = _viewport_of(f[path]) if path and path in f else None
    out = []
    for i, seg in enumerate(outlines):
        seg = np.asarray(seg, dtype=float)
        end = 1 if seg.shape[1] == 4 else -1
        length = float(np.hypot(*(seg[:2, end] - seg[:2, 0])))
        n = None if sample_counts is None or i >= len(sample_counts) else int(sample_counts[i])
        z_um = float(seg[2].mean())
        out.append(
            {
                "index": i,
                "start_um": [float(v) for v in seg[:, 0]],
                "end_um": [float(v) for v in seg[:, end]],
                "z_um": z_um,
                "length_um": length,
                "sample_um": (length / n) if n else None,
                "dz_um": None if snapshot is None else z_um - snapshot["transl"][2],
                "stack": None,
                "slice": None,
                "slice_dz_um": None,
                "in_stack": None,
            }
        )
    # the first Z-stack whose field holds each line: its slice and offset
    units = list_mesc_units(mesc_path)
    key = str(unit_key).strip("/")
    for stack in (u["key"] for u in units if u["kind"] == "zstack"):
        placed = {
            r["roi"]: r for r in image_overlays(mesc_path, stack, units) if r["unit"].strip("/") == key
        }
        for p in out:
            r = placed.get(p["index"])
            if r is not None and p["stack"] is None:
                p.update(stack=stack, slice=r["slice"], slice_dz_um=r["dz_um"], in_stack=r["on_plane"])
    return out
