"""Where a MESc scan's lines or patches sit in a Z-stack, measured from its picture.

The file says part of it: a viewport's tilt and a fast stack's RTMC are stored
and ``mesc_geometry.um_to_voxels`` applies both. What is left is not in the
file. ``geomTransTransl`` leaves the stage out and the tissue drifts 4 to 18 um
per hour between a scan and its stack, so a line's stored position and a
stack's are in different frames: 170 um apart in x after a stage move
(stan110_expt1). The picture a scan was drawn on shares the scan's frame, so
the shift that lands the picture on the stack is the shift its ROIs need.

RTMC holds a line on the tissue through a scan, but it jumps within the scan's
first half second, and on some scans that step took the beam off the tissue: the
lines then sit at their stored position plus the mean RTMC. Nothing in the file
says which scans, so the recording decides::

    position in the stack = stored position + shift (+ mean RTMC during the scan)
                            - mean RTMC of a fast stack

Both the picture and the stack go onto one grid, are band-passed to the scale
of cells and dendrites, and meet by masked normalized cross-correlation
(Padfield 2012), which scores only where they overlap, so a picture larger
than the stack still matches. Under ``NCC_PASS`` the scan is not placed. The
recording then checks the answer without the picture: each ROI's time mean
against the stack under it, with and without RTMC, compared with the same score
at random offsets. The picture is matched slice by slice and by translation
alone, so a picture and a stack whose rotations differ are not placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy import ndimage
from skimage.registration._masked_phase_cross_correlation import (
    cross_correlate_masked,
)

from mbo_utilities import log
from mbo_utilities.analysis.roi_traces import roi_geometry
from mbo_utilities.arrays.mesc_geometry import (
    TILT_TOLERANCE,
    roi_outlines_um,
    um_to_voxels,
    viewport_geometry,
    voxels_to_um,
    zstack_depth_info,
)

logger = log.get("analysis.mesc_placement")

_UNPLACED_WARNED: set[tuple[str, str]] = set()

__all__ = ["NCC_PASS", "P_CONFIRMED", "StackPlacement", "place_scan", "zstack_files"]

# 177 picture and stack pairs on two rigs: 93 of 94 real matches score above it, no match tens of um off does
NCC_PASS = 0.6
P_CONFIRMED = 0.05
GRID_PX = 400
BLUR_PX = 1.0
HIGHPASS_PX = 20.0
MIN_OVERLAP = 0.5
CHECK_BLUR_UM = 0.5
NULL_RANGE_UM = 40.0
NULL_MIN_UM = 8.0
N_NULL = 200


@dataclass(frozen=True)
class StackPlacement:
    """One scan on one Z-stack: ``shift_um`` is what moved between the two
    after the file's own coordinates (``mesc_geometry.um_to_voxels`` of a
    stored ROI moved by it is its place in the stack), ``score`` is the
    picture's match, ``gap_s`` how long after the stack the scan started. ``profile_r`` is the ROIs' time means against the
    stack under them and ``profile_p`` how often random offsets did as well;
    both None when the scan is not placed or has no ROI to check.
    ``with_rtmc`` says the recording put the mean RTMC into ``shift_um``,
    ``channel`` is the 0-based channel it was measured on.
    """

    stack_path: str
    stack_key: str
    shift_um: tuple[float, float, float]
    score: float
    gap_s: float
    profile_r: float | None = None
    profile_p: float | None = None
    with_rtmc: bool = False
    channel: int = 0

    @property
    def placed(self) -> bool:
        return self.score >= NCC_PASS

    @property
    def confirmed(self) -> bool:
        return self.profile_p is not None and self.profile_p < P_CONFIRMED


def zstack_files(path) -> list[Path]:
    """The stack files saved beside a recording: ``<stem>_z.mesc`` or
    ``<stem>_zstack.mesc``, in its folder or one up.
    """
    path = Path(path)
    return [
        folder / f"{path.stem}{suffix}.mesc"
        for folder in (path.parent, path.parent.parent)
        for suffix in ("_z", "_zstack")
        if (folder / f"{path.stem}{suffix}.mesc").is_file()
    ]


def _channel(unit, c: int):
    """``Channel_c`` of an open unit, its last channel when it has fewer."""
    return unit[f"Channel_{min(c, sum(k.startswith('Channel_') for k in unit) - 1)}"]


def _on_grid(images: np.ndarray, viewport: dict, um_per_px: float) -> np.ndarray:
    """``(N, Y, X)`` onto a ``um_per_px`` grid, somata compressed and band-passed."""
    size = round(viewport["width"] / um_per_px), round(viewport["height"] / um_per_px)
    grid = np.stack(
        [
            cv2.resize(im.astype(np.float32), size, interpolation=cv2.INTER_AREA)
            for im in images
        ]
    )
    floor = np.percentile(grid, 1, axis=(1, 2), keepdims=True)
    grid = np.sqrt(np.clip(grid - floor, 0, None))
    return ndimage.gaussian_filter(
        grid, (0, BLUR_PX, BLUR_PX)
    ) - ndimage.gaussian_filter(grid, (0, HIGHPASS_PX, HIGHPASS_PX))


def _match(picture: np.ndarray, stack: np.ndarray) -> tuple[float, float, int, int]:
    """``(score, slice, col, row)``: the fractional slice a band-passed picture
    fits best and where its ``[0, 0]`` pixel lands there, in stack pixels.
    """
    ny, nx = picture.shape
    ones_picture, ones_slice = np.ones_like(picture), np.ones_like(stack[0])
    peaks, row, col = [], 0, 0
    for plane in stack:
        # NOTE: axes must be non-negative; skimage's shape check does not normalize them
        ncc = cross_correlate_masked(
            plane,
            picture,
            ones_slice,
            ones_picture,
            mode="full",
            axes=(0, 1),
            overlap_ratio=MIN_OVERLAP,
        )
        if not peaks or ncc.max() > max(peaks):
            row, col = np.unravel_index(ncc.argmax(), ncc.shape)
        peaks.append(float(ncc.max()))
    k = int(np.argmax(peaks))
    fractional = float(k)
    if 0 < k < len(stack) - 1:
        below, top, above = peaks[k - 1 : k + 2]
        curvature = below - 2 * top + above
        fractional += 0.5 * (below - above) / curvature if curvature else 0.0
    return peaks[k], fractional, int(col) - nx + 1, int(row) - ny + 1


def _roi_points_um(seg: np.ndarray, ny: int, nx: int) -> np.ndarray | None:
    """Microns of every pixel of one ROI, ``(3, ny, nx)``: a line's samples
    run first to last point, a patch's columns corner 0 to 1 and its rows
    corner 0 to 3. None for a ribbon, which is no rectangle.
    """
    if seg.shape[1] > 4:
        return None
    across = (seg[:, 3] - seg[:, 0]) if seg.shape[1] == 4 else np.zeros(3)
    along = (seg[:, 1] if seg.shape[1] == 4 else seg[:, -1]) - seg[:, 0]
    u = (np.arange(nx) + 0.5) / nx
    v = (np.arange(ny) + 0.5) / ny
    return (
        seg[:, 0, None, None]
        + along[:, None, None] * u[None, None, :]
        + across[:, None, None] * v[None, :, None]
    )


def _profile_score(volume, points, means, viewport, depth, offset_um) -> float:
    """Mean r, through Fisher z, of each ROI's time mean with the stack under it."""
    ny, nx = volume.shape[1:]
    scores = []
    for at, mean in zip(points, means):
        flat = at.reshape(3, -1) + offset_um[:, None]
        col, row, k = um_to_voxels(flat.T, viewport, depth, ny, nx).T
        under = ndimage.map_coordinates(
            volume, [k, row - 0.5, col - 0.5], order=1, cval=np.nan
        )
        if np.isfinite(under).all() and under.std() > 0 and mean.std() > 0:
            r = np.corrcoef(under, mean.ravel())[0, 1]
            scores.append(np.arctanh(np.clip(r, -0.99, 0.99)))
    return float(np.tanh(np.mean(scores))) if scores else np.nan


def _check_profiles(mesc, c, volume, viewport, depth, stored, moved):
    """``(with_rtmc, r, null)``: whether a scan's time means fit the stack better
    at ``moved`` than at ``stored``, how well, and the same score at ``N_NULL``
    random xy offsets of the two; ``(False, None, None)`` with no ROI to compare.
    """
    outlines = roi_outlines_um(mesc.filenames[0], mesc.unit_key)
    geometry = roi_geometry(mesc)
    if len(geometry) != len(outlines):
        return False, None, None
    means = mesc.temporal_mean()[c]
    points, profiles = [], []
    for seg, (roi, (ny, nx)) in zip(outlines, sorted(geometry.items())):
        mean = means[roi, :ny, :nx]
        if seg.shape[1] < 4:
            ny, mean = 1, mean.mean(axis=0)
        at = _roi_points_um(seg, ny, nx)
        if at is not None:
            points.append(at)
            profiles.append(mean)
    if not points:
        return False, None, None
    um_per_px = viewport["width"] / volume.shape[-1]
    sigma = CHECK_BLUR_UM / um_per_px
    blurred = ndimage.gaussian_filter(volume, (0, sigma, sigma))
    args = (blurred, points, profiles, viewport, depth)
    at_stored, at_moved = _profile_score(*args, stored), _profile_score(*args, moved)
    observed = np.fmax(at_stored, at_moved)
    if not np.isfinite(observed):
        return False, None, None
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, N_NULL)
    radii = rng.uniform(NULL_MIN_UM, NULL_RANGE_UM, N_NULL)
    offsets = np.stack(
        [radii * np.cos(angles), radii * np.sin(angles), np.zeros(N_NULL)], axis=1
    )
    # NOTE: each offset picks the better of the two too, or picking would flatter p
    null = np.array(
        [
            np.fmax(_profile_score(*args, stored + o), _profile_score(*args, moved + o))
            for o in offsets
        ]
    )
    return bool(at_moved > at_stored or np.isnan(at_stored)), float(observed), null


def place_scan(mesc, stack_path, stack_key: str) -> StackPlacement | None:
    """Measure where ``mesc``'s scan sits in one Z-stack, on the channel its time
    means fit best: the answer must not follow the viewer's channel slider. None
    when the scan links no picture, either unit has no position, or the two
    are rotated differently.
    """
    path = mesc.filenames[0]
    picture_key = next(
        (u["background_unit"] for u in mesc.units if u["key"] == mesc.unit_key), None
    )
    picture_vp = viewport_geometry(path, picture_key) if picture_key else None
    stack_vp = viewport_geometry(stack_path, stack_key)
    depth = zstack_depth_info(stack_path, stack_key)
    if picture_vp is None or stack_vp is None or depth is None:
        return None
    turn = (np.trace(picture_vp["rotation"].T @ stack_vp["rotation"]) - 1) / 2
    if turn < TILT_TOLERANCE:
        if (picture_key, stack_key) not in _UNPLACED_WARNED:
            _UNPLACED_WARNED.add((picture_key, stack_key))
            logger.warning(
                f"{picture_key} and {stack_key} are rotated differently; "
                f"{mesc.unit_key} is not placed"
            )
        return None
    matches = []
    for c in range(mesc.nc):
        with h5py.File(path, "r") as f:
            picture = _channel(f[picture_key], c)[:].mean(axis=0, keepdims=True)
            scan_time = float(f[mesc.unit_key].attrs["MeasurementDatePosix"])
        with h5py.File(stack_path, "r") as f:
            volume = _channel(f[stack_key], c)[:].astype(np.float32)
            gap_s = scan_time - float(f[stack_key].attrs["MeasurementDatePosix"])
        um_per_px = max(
            stack_vp["width"] / GRID_PX,
            stack_vp["width"] / volume.shape[-1],
            picture_vp["width"] / picture.shape[-1],
        )
        stack_grid = _on_grid(volume, stack_vp, um_per_px)
        picture_grid = _on_grid(picture, picture_vp, um_per_px)[0]
        score, fractional, col, row = _match(picture_grid, stack_grid)
        found = voxels_to_um(
            [[col, row, fractional]], stack_vp, depth, *stack_grid.shape[1:]
        )[0]
        stored = found - np.array(picture_vp["transl"])
        matches.append((score, c, stored, volume))
    matches.sort(key=lambda m: m[0], reverse=True)
    placement = {
        "stack_path": str(stack_path),
        "stack_key": stack_key,
        "gap_s": gap_s,
    }
    placed = [m for m in matches if m[0] >= NCC_PASS]
    if not placed:
        score, c, stored, _ = matches[0]
        return StackPlacement(
            **placement, shift_um=tuple(stored.tolist()), score=score, channel=c
        )
    rtmc = mesc.rtmc
    drift = [
        float(np.mean(rtmc[f"{a} total"]["um"])) if f"{a} total" in rtmc else 0.0
        for a in "XYZ"
    ]
    checks = [
        _check_profiles(mesc, c, volume, stack_vp, depth, stored, stored + drift)
        for _, c, stored, volume in placed
    ]
    best = int(np.argmax([-np.inf if r is None else r for _, r, _ in checks]))
    score, c, stored, _ = placed[best]
    with_rtmc, r, _ = checks[best]
    p = None
    if r is not None:
        # NOTE: the offsets repeat per channel, so this is the null of picking a channel too
        null = np.fmax.reduce([n for _, _, n in checks if n is not None])
        null = null[np.isfinite(null)]
        p = float((1 + np.sum(null >= r)) / (1 + len(null)))
    return StackPlacement(
        **placement,
        shift_um=tuple((stored + drift if with_rtmc else stored).tolist()),
        score=score,
        profile_r=r,
        profile_p=p,
        with_rtmc=with_rtmc,
        channel=c,
    )
