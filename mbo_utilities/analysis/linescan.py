"""Photostim events, reference-stack pairing and QC figures for AOD line scans.

A Femtonics AOD line-scan unit is a set of short lines (a few microns, one
per spine or dendritic segment) scanned round-robin at kHz rates, usually
with a photostimulation train somewhere in the run and a reference Z-stack
recorded in the same ``.mesc`` file. Everything a user needs to judge one
run is therefore: where the lines sit on the structure, whether each line
actually crosses a spine, what the kymographs look like, what the traces
did around the stimulus, and whether the stage drifted. This module turns
the extraction outputs (``roi_workflow.extract_linescan_traces``) into that
figure set, numbered so a directory listing reads in that order::

    01a_background_snapshot_lines.png  lines on the snapshot they were drawn on
    01b_reference_zstack_lines.png     lines on the paired Z-stack, slice by slice
    01c_line_zooms.png                 8 um crop around every line, both channels
    02_line_profiles.png            time-averaged intensity along each line
    03a_kymographs_<ch>.png         position x time per ROI, one per channel
    04a_traces_raw.png              F per ROI, stimulus marked
    04b_traces_dfof.png             rolling-baseline dF/F per ROI
    05_stim_response.png            stimulus-aligned dF/F, heatmap + traces
    06_roi_response_metrics.png     per-ROI baseline, peak, noise, SNR
    07_motion_correction.png        RTMC X/Y/Z correction over the run

Stimulus timing comes from the ``PatternSeq_AO1`` curve: the scanner
alternates between the imaging pattern (``mainPatternIndex``) and one or
more stimulation patterns, and every sample whose value is not the imaging
pattern is a stimulation interval (the same rule ``lab4``'s converter and
its reference scripts use to write ``frames.npy``). The pattern's own
geometry in this file is relative (a square about the origin), so the
stimulus *location* is not drawn.

Line placement: a line-scan unit names the raster snapshot it was drawn on
(``BackgroundImagePath``, an MSession_1 image taken seconds before the scan)
and its lines are in the same micron frame as that image's
``ReferenceViewportJSON``. Mapping microns to array pixels with the
translation as the array's (0, 0) corner and rows increasing with +y (no
mirror) was checked on the 2026-07-30 spine file: 36 of the 40 lines within
1 um of their snapshot's plane are brighter than random same-shaped lines
nearby (median rank 0.76), while the mirrored mapping is at chance (19 of
40, median 0.49). ``flip_y`` stays available for a rig that saves the
other way round.

Figure failures log and continue; a figure never aborts an extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays.mesc_geometry import (
    linescan_endpoints_um,
    roi_placements,
    slices_with_rois,
    um_to_pixels,
    viewport_geometry,
    zstack_depth_info,
)

__all__ = [
    "stim_events",
    "background_image",
    "zstack_candidates",
    "pair_reference_zstack",
    "stim_aligned_dfof",
    "response_metrics",
    "plot_linescan_figures",
]

_FIG_BG = "black"
_FIG_FG = "white"
_TRAIN_GAP_MS = 100.0  # pulses closer than this belong to one train
_SMOOTH_S = 0.02  # display / peak smoothing window for kHz traces


# ---------------------------------------------------------------------------
# stimulus timing
# ---------------------------------------------------------------------------


def stim_events(mesc_path, unit_key: str, fs: float, n_frames: int) -> dict | None:
    """Photostimulation intervals of a unit from its ``PatternSeq_AO1`` curve.

    Returns ``None`` when the unit has no protocol, no pattern-sequence
    curve, or never left the imaging pattern. Otherwise::

        frames       sorted frame indices acquired while a stim pattern was
                     active (what lab4 saves as ``frames.npy``)
        pulses_s     start time of every stim interval, seconds from unit start
        onsets_s     first pulse of each train (pulses < 100 ms apart merge)
        durations_s  length of each train, first pulse start to last pulse end
        patterns     0-based indices of the stim patterns used
    """
    from mbo_utilities.arrays.mesc import _parse_curves

    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None or "MultiROIProtocolJSON" not in unit.attrs:
            return None
        try:
            proto = json.loads(unit.attrs["MultiROIProtocolJSON"])
            main = int(proto["protocol"]["scanners"]["mainPatternIndex"])
        except (KeyError, TypeError, ValueError):
            return None
        curves = _parse_curves(unit)
    seq = curves.get("PatternSeq_AO1")
    if seq is None or len(seq["values"]) < 2:
        return None
    ts = np.asarray(seq["timestamps"], dtype=float)  # ms, start of each sample
    vals = np.asarray(seq["values"]).astype(int)
    end_ms = n_frames / fs * 1000.0

    frames: list[int] = []
    pulses: list[tuple[float, float]] = []
    patterns: set[int] = set()
    for i, v in enumerate(vals):
        if v in (0, main):
            continue
        t0 = ts[i]
        t1 = ts[i + 1] if i + 1 < len(ts) else end_ms
        if t0 >= end_ms:
            continue
        f0 = int(np.floor(t0 * fs / 1000.0))
        f1 = max(f0 + 1, int(np.ceil(t1 * fs / 1000.0)))
        frames.extend(range(f0, min(f1, n_frames)))
        pulses.append((t0, t1))
        patterns.add(int(v) - 1)
    if not pulses:
        return None

    onsets, durations = [], []
    train_start, train_end = pulses[0]
    for t0, t1 in pulses[1:]:
        if t0 - train_end > _TRAIN_GAP_MS:
            onsets.append(train_start)
            durations.append(train_end - train_start)
            train_start = t0
        train_end = t1
    onsets.append(train_start)
    durations.append(train_end - train_start)
    return {
        "frames": np.unique(np.asarray(frames, dtype=int)),
        "pulses_s": [t0 / 1000.0 for t0, _ in pulses],
        "onsets_s": [o / 1000.0 for o in onsets],
        "durations_s": [d / 1000.0 for d in durations],
        "patterns": sorted(patterns),
    }


# ---------------------------------------------------------------------------
# reference Z-stack
# ---------------------------------------------------------------------------


def _fov_fraction(lines_um, vp) -> float:
    tx, ty, _ = vp["transl"]
    inside = 0
    for seg in lines_um:
        x, y = float(seg[0].mean()), float(seg[1].mean())
        inside += tx <= x <= tx + vp["width"] and ty <= y <= ty + vp["height"]
    return inside / max(len(lines_um), 1)


def background_image(mesc_path, unit_key: str) -> dict | None:
    """The raster snapshot a line-scan unit's lines were drawn on.

    MESc records it as the unit's ``BackgroundImagePath`` (an MSession_1
    image unit acquired seconds before the scan, at the focal plane the
    lines were placed in). Returns ``{"key", "munit", "transl", "width",
    "height", "z", "nchannels", "shape"}`` or ``None`` when the attr is
    missing, dangling, or the target has no viewport.
    """
    with h5py.File(mesc_path, "r") as f:
        unit = f.get(unit_key)
        if unit is None:
            return None
        raw = unit.attrs.get("BackgroundImagePath")
        path = raw.decode() if isinstance(raw, bytes) else raw
        if not path or path not in f or "Channel_0" not in f[path]:
            return None
        img = f[path]
        vp_raw = img.attrs.get("ReferenceViewportJSON")
        if not vp_raw:
            return None
        try:
            vp = json.loads(vp_raw)["viewports"][0]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        shape = tuple(int(v) for v in img["Channel_0"].shape)
        nchannels = sum(1 for k in img if k.startswith("Channel_"))
        transl = tuple(float(v) for v in vp["geomTransTransl"])
        return {
            "key": path.strip("/"),
            "munit": path.rstrip("/").rsplit("/", 1)[-1],
            "transl": transl,
            "width": float(vp["width"]),
            "height": float(vp["height"]),
            "z": transl[2],
            "nchannels": nchannels,
            "shape": shape,
        }


def _background_planes(mesc_path, bg: dict) -> dict[int, np.ndarray]:
    """``{channel: (Y, X) float}`` of the snapshot (its single frame)."""
    with h5py.File(mesc_path, "r") as f:
        img = f[bg["key"]]
        return {
            c: np.asarray(img[f"Channel_{c}"][0], dtype=np.float32)
            for c in range(bg["nchannels"])
        }


def zstack_candidates(
    mesc_path, unit_key: str, units: list[dict] | None = None, *, min_px_per_line: int = 10,
) -> list[dict]:
    """Every Z-stack of the file scored against a line-scan unit's lines:
    ``xy_fraction`` / ``z_fraction`` of lines inside its field / depth range,
    ``um_per_px``, and ``coarse`` when it puts fewer than ``min_px_per_line``
    pixels along the median line. All stacks, including those holding no
    line at all, so a picker can show why one is unsuitable."""
    from mbo_utilities.arrays.mesc import list_mesc_units

    lines = linescan_endpoints_um(mesc_path, unit_key)
    if not lines:
        return []
    lengths = [float(np.hypot(*(seg[:2, 1] - seg[:2, 0]))) for seg in lines]
    max_um_per_px = float(np.median(lengths)) / max(min_px_per_line, 1)
    units = units if units is not None else list_mesc_units(mesc_path)
    out = []
    for u in units:
        if u["modality_name"] != "zstack":
            continue
        vp = viewport_geometry(mesc_path, u["key"])
        depth = zstack_depth_info(mesc_path, u["key"])
        if vp is None or depth is None:
            continue
        nx = int(u["shape"][-1])
        placements = roi_placements(lines, depth)
        um_per_px = vp["width"] / max(nx, 1)
        out.append(
            {
                "key": u["key"],
                "munit": u["munit"],
                "xy_fraction": _fov_fraction(lines, vp),
                "z_fraction": sum(p["in_range"] for p in placements) / len(placements),
                "um_per_px": um_per_px,
                "coarse": um_per_px > max_um_per_px,
            }
        )
    return out


def pair_reference_zstack(
    mesc_path, unit_key: str, units: list[dict] | None = None, *, min_px_per_line: int = 10,
) -> dict | None:
    """Pick the Z-stack unit of the same file that the lines were drawn on.

    Lines live in absolute microns, so any stack whose field contains them
    can show them; what decides is whether it can *resolve* them. A stack
    is a candidate only when it has at least ``min_px_per_line`` pixels
    along the median line (a 400 um whole-cell stack at 0.8 um/px puts 4
    pixels on a 3 um spine line and is useless as a reference). Among
    candidates holding at least 80 % of the lines in XY the finest pixel
    size wins; failing that, the best XY coverage. Depth coverage is
    reported (``z_fraction``) but does not disqualify a stack: on the
    2026-07-30 file the lines of two dendrites were placed up to 7 um below
    their own stack's bottom slice, and switching to the coarse stack there
    lost the dendrite entirely. Lines outside the depth range are drawn on
    the nearest edge slice and flagged.

    Only when no stack resolves the lines does a coarse one get used, with
    ``"coarse": True`` so the figure can say so. ``None`` when no stack
    contains any line in XY.

    Returns ``{"key", "munit", "xy_fraction", "z_fraction", "um_per_px",
    "coarse"}``.
    """
    cands = [c for c in zstack_candidates(mesc_path, unit_key, units, min_px_per_line=min_px_per_line)
             if c["xy_fraction"] > 0]
    if not cands:
        return None
    fine = [c for c in cands if not c["coarse"]]
    pool = fine or cands
    good = [c for c in pool if c["xy_fraction"] >= 0.8]
    if good:
        return min(good, key=lambda c: c["um_per_px"])
    return max(pool, key=lambda c: (c["xy_fraction"], -c["um_per_px"]))


# ---------------------------------------------------------------------------
# stimulus-aligned responses
# ---------------------------------------------------------------------------


def _smooth(x: np.ndarray, fs: float, window_s: float = _SMOOTH_S) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    n = max(1, int(round(window_s * fs)))
    return uniform_filter1d(x, size=n, axis=-1, mode="nearest") if n > 1 else x


def stim_aligned_dfof(
    F: np.ndarray, fs: float, onsets_s: list[float], pre_s: float = 1.0, post_s: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(n_events, K, n)`` dF/F around each train onset, with ``F0`` the mean
    of the ``pre_s`` window before the onset (the last 50 ms before the
    onset excluded, so a pulse that lands early does not leak into F0).
    Returns ``(dfof, t_s)``; windows that run off either end are NaN-padded."""
    K, T = F.shape
    n_pre, n_post = int(round(pre_s * fs)), int(round(post_s * fs))
    guard = int(round(0.05 * fs))
    t_s = (np.arange(-n_pre, n_post) / fs).astype(np.float32)
    out = np.full((len(onsets_s), K, n_pre + n_post), np.nan, np.float32)
    for e, onset in enumerate(onsets_s):
        c = int(round(onset * fs))
        b0, b1 = max(0, c - n_pre), max(0, c - guard)
        if b1 <= b0:
            continue
        f0 = F[:, b0:b1].mean(axis=1, keepdims=True)
        f0 = np.where(f0 > 0, f0, np.nan)
        a0, a1 = c - n_pre, c + n_post
        s0, s1 = max(a0, 0), min(a1, T)
        out[e, :, s0 - a0 : s1 - a0] = (F[:, s0:s1] - f0) / f0
    return out, t_s


def response_metrics(
    F: np.ndarray,
    fs: float,
    stim: dict | None,
    *,
    pre_s: float = 1.0,
    post_s: float = 2.0,
) -> list[dict]:
    """Per-ROI response numbers, one dict per ROI, for ``stat.npy`` and the
    metrics figure.

    With a stimulus: ``f0`` is the pre-onset mean, ``peak_dfof`` the maximum
    of the 20 ms-smoothed dF/F within ``post_s`` of the first train,
    ``time_to_peak_s`` its latency, ``response_auc`` the integral of dF/F
    over that window (dF/F x s), ``noise_dfof`` the standard deviation of
    the smoothed pre-onset dF/F and ``snr`` their ratio. Without one, ``f0``
    is the 10th percentile of F, the peak is taken over the whole run and
    the noise is a robust (MAD) estimate of the smoothed trace.
    """
    K, T = F.shape
    if stim and stim.get("onsets_s"):
        aligned, t_s = stim_aligned_dfof(F, fs, stim["onsets_s"][:1], pre_s=pre_s, post_s=post_s)
        d = _smooth(np.nan_to_num(aligned[0]), fs)
        pre = t_s < -0.05
        post = (t_s >= 0) & (t_s <= post_s)
        c = int(round(stim["onsets_s"][0] * fs))
        f0 = F[:, max(0, c - int(pre_s * fs)) : max(1, c - int(0.05 * fs))].mean(axis=1)
        peak_idx = np.argmax(np.where(post, d, -np.inf), axis=1)
        peak = d[np.arange(K), peak_idx]
        ttp = t_s[peak_idx]
        auc = np.nansum(np.where(post, d, 0.0), axis=1) / fs
        noise = d[:, pre].std(axis=1)
        mode = "stim"
    else:
        f0 = np.percentile(F, 10, axis=1)
        f0 = np.where(f0 > 0, f0, np.nan)
        d = _smooth((F - f0[:, None]) / f0[:, None], fs)
        peak_idx = np.argmax(d, axis=1)
        peak = d[np.arange(K), peak_idx]
        ttp = peak_idx / fs
        auc = np.full(K, np.nan)
        noise = 1.4826 * np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1)
        mode = "spontaneous"
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(noise > 0, peak / noise, np.nan)
    return [
        {
            "f0": float(f0[i]),
            "peak_dfof": float(peak[i]),
            "time_to_peak_s": float(ttp[i]),
            "response_auc": float(auc[i]),
            "noise_dfof": float(noise[i]),
            "snr": float(snr[i]),
            "response_mode": mode,
        }
        for i in range(K)
    ]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _agg_plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _dark(ax):
    ax.set_facecolor(_FIG_BG)
    ax.tick_params(colors=_FIG_FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_FIG_FG)
    ax.xaxis.label.set_color(_FIG_FG)
    ax.yaxis.label.set_color(_FIG_FG)
    ax.title.set_color(_FIG_FG)


def _save(fig, path):
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=150, facecolor=_FIG_BG, bbox_inches="tight")
    plt.close(fig)


def _roi_colors(K: int) -> np.ndarray:
    from mbo_utilities.annotation.store import CLASS_COLORS

    return np.array([CLASS_COLORS[i % len(CLASS_COLORS)][:3] for i in range(K)], dtype=float)


def _mark_stim(ax, stim, fs=None, text=True):
    if not stim:
        return
    for k, onset in enumerate(stim["onsets_s"]):
        ax.axvline(onset, color="yellow", lw=0.8, alpha=0.8, ls="--")
        if text and k == 0:
            ax.text(
                onset, 1.0, " stim", color="yellow", fontsize=7, va="top", ha="left",
                transform=ax.get_xaxis_transform(),
            )


def _channel_label(channel_names, c: int) -> str:
    try:
        name = str(channel_names[c]).strip().lower()
    except (IndexError, TypeError):
        name = ""
    return name or f"chan{c}"


def _norm_image(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (1, 99.7))
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def _composite(slices_by_channel: dict[int, np.ndarray], channel_names) -> np.ndarray:
    """RGB from a green and a red channel when both exist, else greyscale."""
    if len(slices_by_channel) >= 2:
        names = {c: _channel_label(channel_names, c) for c in slices_by_channel}
        green = next((c for c, n in names.items() if "green" in n), min(slices_by_channel))
        red = next((c for c, n in names.items() if "red" in n and c != green), None)
        if red is None:
            red = next(c for c in slices_by_channel if c != green)
        rgb = np.zeros(slices_by_channel[green].shape + (3,), np.float32)
        rgb[..., 0] = _norm_image(slices_by_channel[red])
        rgb[..., 1] = _norm_image(slices_by_channel[green])
        rgb[..., 2] = rgb[..., 0] * 0.6
        return rgb
    c = next(iter(slices_by_channel))
    return np.repeat(_norm_image(slices_by_channel[c])[..., None], 3, axis=-1)


def _draw_lines(ax, pixel_lines, colors, on, labels=True):
    for i, seg in enumerate(pixel_lines):
        is_on = on[i]
        ax.plot(seg[:, 0], seg[:, 1], color=colors[i], lw=2.0 if is_on else 0.8,
                alpha=1.0 if is_on else 0.35, ls="-" if is_on else "--")
        ax.plot(seg[0, 0], seg[0, 1], "o", color=colors[i], ms=4 if is_on else 2,
                alpha=1.0 if is_on else 0.35)
        if labels and is_on:
            ax.annotate(str(i), seg[0], color=colors[i], fontsize=7,
                        xytext=(3, 3), textcoords="offset points")


def plot_background_snapshot(
    out_dir: Path, mesc_path, unit_key: str, bg: dict, extents, *, flip_y: bool = False,
    save_name: str = "01a_background_snapshot_lines.png",
) -> Path | None:
    """Every line on the raster snapshot it was drawn on (the unit's
    ``BackgroundImagePath``), composite of the channels, with the depth of
    each line relative to the snapshot's focal plane. Lines more than 1 um
    off that plane are dashed: they were placed on a different slice of the
    3D view and need the Z-stack figure."""
    plt = _agg_plt()
    lines = linescan_endpoints_um(mesc_path, unit_key)
    if not lines:
        return None
    n = min(len(lines), len(extents)) if extents else len(lines)
    lines = lines[:n]
    planes = _background_planes(mesc_path, bg)
    ny, nx = planes[0].shape
    vp = {"transl": bg["transl"], "width": bg["width"], "height": bg["height"]}
    pixel_lines = [um_to_pixels(seg[:2].T, vp, ny, nx, flip_y=flip_y) for seg in lines]
    dz = [float(seg[2].mean()) - bg["z"] for seg in lines]
    on = [abs(d) <= 1.0 for d in dz]
    colors = _roi_colors(n)
    names = [f"chan{c}" for c in planes]
    try:
        from mbo_utilities.arrays.mesc import MescArray

        names = MescArray(mesc_path, unit=unit_key).metadata.get("channel_names") or names
    except Exception:
        pass
    ncols = 1 + len(planes)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 6.4), facecolor=_FIG_BG, squeeze=False)
    axes = axes[0]
    axes[0].imshow(_composite(planes, names), interpolation="nearest")
    axes[0].set_title("composite", color=_FIG_FG, fontsize=9)
    for ax, (c, img) in zip(axes[1:], planes.items()):
        ax.imshow(_norm_image(img), cmap="gray", interpolation="nearest")
        ax.set_title(_channel_label(names, c), color=_FIG_FG, fontsize=9)
    for ax in axes:
        ax.set_facecolor(_FIG_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_lines(ax, pixel_lines, colors, on)
        ax.set_xlim(0, nx)
        ax.set_ylim(ny, 0)
    fig.suptitle(f"{unit_key.rsplit('/', 1)[-1]} on snapshot {bg['munit']}", color=_FIG_FG, fontsize=11)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def plot_line_zooms(
    out_dir: Path, mesc_path, unit_key: str, extents, *, bg: dict | None, ref: dict | None,
    flip_y: bool = False, half_um: float = 4.0, save_name: str = "01c_line_zooms.png",
) -> Path | None:
    """An 8 um crop around every line, each channel, locally contrast-
    stretched: the line should cross a bright spine or shaft. The crop comes
    from the snapshot when the line is within 1 um of its plane, else from
    the paired Z-stack's slice at the line's depth (title says which)."""
    from mbo_utilities.arrays.mesc import MescArray

    plt = _agg_plt()
    lines = linescan_endpoints_um(mesc_path, unit_key)
    if not lines or (bg is None and ref is None):
        return None
    n = min(len(lines), len(extents)) if extents else len(lines)
    lines = lines[:n]
    colors = _roi_colors(n)
    sources: list[tuple[str, dict[int, np.ndarray], dict, str]] = []  # per ROI
    bg_planes = _background_planes(mesc_path, bg) if bg is not None else None
    zs = ref_planes = None
    placements = None
    if ref is not None:
        vp_ref = viewport_geometry(mesc_path, ref["key"])
        depth = zstack_depth_info(mesc_path, ref["key"])
        if vp_ref is not None and depth is not None:
            zs = MescArray(mesc_path, unit=ref["key"])
            placements = roi_placements(lines, depth)
            ref_planes = {}
    names = MescArray(mesc_path, unit=unit_key).metadata.get("channel_names") or []
    for i, seg in enumerate(lines):
        # whichever image is nearest in depth: the snapshot's single plane,
        # or the stack's nearest slice (which may be an edge slice several
        # microns away when the line was scanned outside the stack)
        dz_bg = abs(float(seg[2].mean()) - bg["z"]) if bg is not None else np.inf
        dz_zs = abs(placements[i]["dz_um"]) if zs is not None else np.inf
        if bg_planes is not None and dz_bg <= min(dz_zs, 1.0):
            sources.append(("snap", bg_planes, {"transl": bg["transl"], "width": bg["width"], "height": bg["height"]},
                            f"{bg['munit']} {float(seg[2].mean()) - bg['z']:+.1f} um"))
        elif zs is not None and dz_zs <= dz_bg:
            k = placements[i]["slice"]
            if k not in ref_planes:
                ref_planes[k] = {c: np.asarray(zs[0, c, k], dtype=np.float32) for c in range(int(zs.shape[1]))}
            flag = "" if placements[i]["in_range"] else " (outside stack)"
            sources.append(("stack", ref_planes[k], vp_ref,
                            f"{ref['munit']} slice {k + 1} {placements[i]['dz_um']:+.1f} um{flag}"))
        elif bg_planes is not None:
            sources.append(("snap", bg_planes, {"transl": bg["transl"], "width": bg["width"], "height": bg["height"]},
                            f"{bg['munit']} {float(seg[2].mean()) - bg['z']:+.1f} um (off plane)"))
        else:
            sources.append(None)
    nchan = max(len(src[1]) for src in sources if src is not None)
    per_row = 4 if nchan == 1 else 3
    nrows = int(np.ceil(n / per_row))
    fig, axes = plt.subplots(nrows, per_row * nchan, figsize=(2.3 * per_row * nchan, 2.5 * nrows),
                             facecolor=_FIG_BG, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, src in enumerate(sources):
        if src is None:
            continue
        kind, planes, vp, where = src
        ny, nx = planes[0].shape
        px_all = [um_to_pixels(seg[:2].T, vp, ny, nx, flip_y=flip_y) for seg in lines]
        half = half_um / (vp["width"] / nx)
        cx, cy = px_all[i].mean(axis=0)
        c0, c1 = int(max(0, cx - half)), int(min(nx, cx + half))
        r0, r1 = int(max(0, cy - half)), int(min(ny, cy + half))
        for cc, (c, img) in enumerate(sorted(planes.items())):
            ax = axes[i // per_row, (i % per_row) * nchan + cc]
            ax.set_visible(True)
            crop = img[r0:r1, c0:c1]
            if crop.size:
                lo, hi = np.percentile(crop, (2, 99.8))
                ax.imshow(np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", vmin=0, vmax=1,
                          extent=(c0, c1, r1, r0), interpolation="nearest")
            _draw_lines(ax, px_all, [colors[j] if j == i else (1.0, 0.6, 0.0) for j in range(n)],
                        [j == i for j in range(n)], labels=False)
            ax.set_xlim(c0, c1)
            ax.set_ylim(r1, r0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(colors[i])
            ax.set_title(f"ROI {i} {_channel_label(names, c)}\n{where}", color=colors[i], fontsize=7)
    fig.suptitle(f"{unit_key.rsplit('/', 1)[-1]} line zooms ({2 * half_um:.0f} um)", color=_FIG_FG, fontsize=11)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def plot_reference_zstack(
    out_dir: Path, mesc_path, unit_key: str, ref: dict, extents, *, flip_y: bool = False,
    save_name: str = "01b_reference_zstack_lines.png",
) -> Path | None:
    """Lines drawn on the slices of the paired Z-stack that carry them, plus
    a max projection with every line. ROI index at each line's start."""
    from mbo_utilities.arrays.mesc import MescArray

    plt = _agg_plt()
    lines = linescan_endpoints_um(mesc_path, unit_key)
    vp = viewport_geometry(mesc_path, ref["key"])
    depth = zstack_depth_info(mesc_path, ref["key"])
    if not lines or vp is None or depth is None:
        return None
    n = min(len(lines), len(extents)) if extents else len(lines)
    lines = lines[:n]
    widths = [int(e["width"]) for e in extents[:n]] if extents else None
    placements = roi_placements(lines, depth, widths)
    occupied = slices_with_rois(placements)

    zs = MescArray(mesc_path, unit=ref["key"])
    nz, ny, nx = int(zs.shape[2]), int(zs.shape[3]), int(zs.shape[4])
    channels = list(range(int(zs.shape[1])))
    names = zs.metadata.get("channel_names") or []
    pixel_lines = [um_to_pixels(seg[:2].T, vp, ny, nx, flip_y=flip_y) for seg in lines]
    colors = _roi_colors(n)

    slice_ids = sorted(occupied)
    if len(slice_ids) > 11:
        slice_ids = sorted(sorted(slice_ids, key=lambda k: -len(occupied[k]))[:11])
    panels = ["max"] + slice_ids
    ncols = min(4, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.6 * nrows), facecolor=_FIG_BG)
    axes = np.atleast_1d(axes).ravel()

    proj = {c: np.max(np.asarray(zs[0, c, :]), axis=0) for c in channels}
    in_range = [p["in_range"] for p in placements]
    for ax, panel in zip(axes, panels):
        ax.set_facecolor(_FIG_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        if panel == "max":
            ax.imshow(_composite(proj, names), interpolation="nearest")
            ax.set_title("max projection", color=_FIG_FG, fontsize=9)
            here = range(n)
        else:
            k = int(panel)
            img = {c: np.asarray(zs[0, c, k]) for c in channels}
            ax.imshow(_composite(img, names), interpolation="nearest")
            z_um = depth["min_z"] + k * (depth["max_z"] - depth["min_z"]) / max(nz - 1, 1)
            ax.set_title(f"slice {k + 1}  z {z_um:+.1f} um", color=_FIG_FG, fontsize=9)
            here = occupied[k]
        for i in range(n):
            seg = pixel_lines[i]
            on = i in here
            # a line scanned outside the stack's depth range is drawn dashed
            # on the nearest edge slice: its XY is right, its depth is not here
            ls = "-" if in_range[i] else "--"
            ax.plot(seg[:, 0], seg[:, 1], color=colors[i], lw=2.0 if on else 0.8,
                    alpha=1.0 if on else 0.35, ls=ls)
            ax.plot(seg[0, 0], seg[0, 1], "o", color=colors[i], ms=4 if on else 2,
                    alpha=1.0 if on else 0.35)
            if on:
                ax.annotate(f"{i}{'' if in_range[i] else '!'}", seg[0], color=colors[i], fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(0, nx)
        ax.set_ylim(ny, 0)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    coarse = " (COARSE: no stack in this file resolves the lines)" if ref.get("coarse") else ""
    fig.suptitle(f"{unit_key.rsplit('/', 1)[-1]} on z-stack {ref['munit']}{coarse}", color=_FIG_FG, fontsize=11)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def plot_line_profiles(
    out_dir: Path, kymos: dict[int, np.ndarray], extents, channel_names, um_per_px: float | None,
    save_name: str = "02_line_profiles.png",
) -> Path:
    """Time-averaged intensity along every line, one panel per ROI, one
    curve per channel, in the file's converted counts (zero = no photons)
    on a y-axis shared by all ROIs. A spine crossed by the line shows as a
    bump; a flat profile at the background level means the line missed."""
    plt = _agg_plt()
    K = len(extents)
    ncols = min(6, K)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.1 * nrows), facecolor=_FIG_BG,
                             squeeze=False, sharey=True)
    colors = _roi_colors(K)
    chan_style = {}
    for c in kymos:
        label = _channel_label(channel_names, c)
        chan_style[c] = ("lime" if "green" in label else "red" if "red" in label else "white", label)
    profiles = {c: [np.nanmean(k[i, :, : int(extents[i]["width"])], axis=0) for i in range(K)]
                for c, k in kymos.items()}
    top = max(float(np.nanmax(p)) for ps in profiles.values() for p in ps)
    bottom = min(0.0, min(float(np.nanmin(p)) for ps in profiles.values() for p in ps))
    for i in range(K):
        ax = axes.ravel()[i]
        _dark(ax)
        w = int(extents[i]["width"])
        x = np.arange(w) * (um_per_px if um_per_px else 1.0)
        for c in kymos:
            ax.plot(x, profiles[c][i], color=chan_style[c][0], lw=1.2, label=chan_style[c][1])
        ax.set_title(f"ROI {i}", color=colors[i], fontsize=9)
        ax.set_xlim(x[0], x[-1] if w > 1 else 1)
        if i % ncols == 0:
            ax.set_ylabel("F (counts)", fontsize=8)
        if i >= K - ncols:
            ax.set_xlabel("um along line" if um_per_px else "px along line", fontsize=8)
    axes.ravel()[0].set_ylim(bottom, top * 1.08 if top > 0 else 1)
    for ax in axes.ravel()[K:]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, facecolor=_FIG_BG,
                   labelcolor=_FIG_FG, edgecolor=_FIG_FG)
    fig.suptitle("Line profiles", color=_FIG_FG, fontsize=11)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def plot_kymographs(
    out_dir: Path, kymo: np.ndarray, bin_s: float, extents, stim, label: str, save_name: str,
    um_per_px: float | None = None,
) -> Path:
    """Position x time image per ROI (rows), shared time axis, per-ROI
    contrast, stimulus onsets marked."""
    plt = _agg_plt()
    K, nb, W = kymo.shape
    fig, axes = plt.subplots(K, 1, figsize=(14, max(2.0, 0.55 * K + 1.2)), facecolor=_FIG_BG,
                             sharex=True, squeeze=False)
    colors = _roi_colors(K)
    t_end = nb * bin_s
    for i in range(K):
        ax = axes[i, 0]
        w = int(extents[i]["width"])
        img = kymo[i, :, :w].T  # (position, time)
        lo, hi = np.nanpercentile(img, (1, 99.5))
        extent_y = w * um_per_px if um_per_px else w
        ax.imshow(img, aspect="auto", cmap="magma", vmin=lo, vmax=hi,
                  extent=(0, t_end, extent_y, 0), interpolation="nearest")
        ax.set_facecolor(_FIG_BG)
        ax.tick_params(colors=_FIG_FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(colors[i])
        ax.set_ylabel(f"ROI {i}", color=colors[i], fontsize=8, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        _mark_stim(ax, stim, text=(i == 0))
    axes[-1, 0].set_xlabel("time (s)", color=_FIG_FG, fontsize=9)
    axes[-1, 0].set_xlim(0, t_end)
    fig.suptitle(f"Kymographs ({label})", color=_FIG_FG, fontsize=11)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def _stacked_traces(ax, traces: np.ndarray, t_s: np.ndarray, colors, lw=0.5, gap=0.15):
    """Rows offset so each ROI has its own lane; returns the row offsets."""
    K = traces.shape[0]
    traces = np.where(np.isfinite(traces), traces, np.nan)
    p1, p99 = np.nanpercentile(np.nan_to_num(traces), (1, 99), axis=1)
    span = np.nanmedian(p99 - p1)
    span = span if np.isfinite(span) and span > 0 else 1.0
    step = span * (1 + gap)
    offsets = np.arange(K)[::-1] * step
    for i in range(K):
        ax.plot(t_s, traces[i] - p1[i] + offsets[i], color=colors[i], lw=lw)
    ax.set_yticks(offsets)
    ax.set_yticklabels([f"ROI {i}" for i in range(K)], fontsize=7)
    for tick, c in zip(ax.get_yticklabels(), colors):
        tick.set_color(c)
    ax.set_xlim(t_s[0], t_s[-1])
    return offsets, step


def plot_traces(
    out_dir: Path, F: np.ndarray, fs: float, stim, *, kind: str, save_name: str, label: str = "",
) -> Path:
    """Stacked per-ROI traces over the whole run. ``kind`` is ``"raw"`` (F as
    recorded) or ``"dfof"`` (already dF/F, drawn 20 ms-smoothed)."""
    plt = _agg_plt()
    K, T = F.shape
    t_s = np.arange(T) / fs
    data = _smooth(F, fs) if kind == "dfof" else F
    fig, ax = plt.subplots(figsize=(14, max(3.0, 0.42 * K + 1.5)), facecolor=_FIG_BG)
    _dark(ax)
    colors = _roi_colors(K)
    offsets, step = _stacked_traces(ax, data, t_s, colors)
    _mark_stim(ax, stim)
    # scale bar on the right edge
    x_bar = t_s[-1] * 0.995
    ax.plot([x_bar, x_bar], [offsets[-1], offsets[-1] + step / (1.15)], color=_FIG_FG, lw=2)
    unit = "dF/F" if kind == "dfof" else "counts"
    ax.text(x_bar, offsets[-1] + step / 2.3, f" {step / 1.15:.2g} {unit}", color=_FIG_FG, fontsize=7,
            ha="right", va="center", rotation=90)
    ax.set_xlabel("time (s)", fontsize=9)
    title = f"F ({label})" if kind == "raw" else f"dF/F ({label})"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def plot_stim_response(
    out_dir: Path, F: np.ndarray, fs: float, stim: dict, *, pre_s: float = 1.0, post_s: float = 5.0,
    save_name: str = "05_stim_response.png",
) -> Path:
    """Stimulus-aligned dF/F (pre-onset F0): a ROI x time heatmap, every ROI
    overlaid, and the mean across ROIs with its s.e.m. First train only when
    there are several; the others are listed in the title."""
    plt = _agg_plt()
    aligned, t_s = stim_aligned_dfof(F, fs, stim["onsets_s"][:1], pre_s=pre_s, post_s=post_s)
    d = _smooth(np.nan_to_num(aligned[0]), fs)
    K = d.shape[0]
    colors = _roi_colors(K)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor=_FIG_BG, sharex=True,
                             gridspec_kw={"height_ratios": [max(2.0, K * 0.22), 2, 2]})
    for ax in axes:
        _dark(ax)
    lim = np.nanpercentile(np.abs(d), 99.5)
    lim = lim if np.isfinite(lim) and lim > 0 else 0.1
    im = axes[0].imshow(d, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                        extent=(t_s[0], t_s[-1], K - 0.5, -0.5), interpolation="nearest")
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels([f"ROI {i}" for i in range(K)], fontsize=7)
    for tick, c in zip(axes[0].get_yticklabels(), colors):
        tick.set_color(c)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.01)
    cb.ax.tick_params(colors=_FIG_FG, labelsize=7)
    cb.set_label("dF/F", color=_FIG_FG, fontsize=8)
    axes[0].set_title("dF/F per ROI", fontsize=10)

    for i in range(K):
        axes[1].plot(t_s, d[i], color=colors[i], lw=0.6, alpha=0.9)
    axes[1].set_ylabel("dF/F", fontsize=9)
    axes[1].set_title("all ROIs", fontsize=9)

    mean = np.nanmean(d, axis=0)
    sem = np.nanstd(d, axis=0) / np.sqrt(max(K, 1))
    axes[2].fill_between(t_s, mean - sem, mean + sem, color="deepskyblue", alpha=0.3, lw=0)
    axes[2].plot(t_s, mean, color="deepskyblue", lw=1.2)
    axes[2].axhline(0, color=_FIG_FG, lw=0.5, alpha=0.5)
    axes[2].set_ylabel("dF/F", fontsize=9)
    axes[2].set_xlabel("time from stimulus onset (s)", fontsize=9)
    axes[2].set_title("mean +/- sem", fontsize=9)
    for ax in axes:
        ax.axvline(0, color="yellow", lw=0.8, ls="--")
        dur = stim["durations_s"][0]
        if dur > 0:
            ax.axvspan(0, dur, color="yellow", alpha=0.15, lw=0)
    axes[2].set_xlim(t_s[0], t_s[-1])
    n_pulses = sum(1 for p in stim["pulses_s"] if stim["onsets_s"][0] <= p <= stim["onsets_s"][0] + stim["durations_s"][0] + 1e-6)
    fig.suptitle(f"Stimulus response (t = {stim['onsets_s'][0]:.2f} s, {n_pulses} pulses)", color=_FIG_FG, fontsize=11)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def plot_response_metrics(
    out_dir: Path, metrics: list[dict], F_other: dict[int, np.ndarray], channel_names, main_channel: int,
    save_name: str = "06_roi_response_metrics.png",
) -> Path:
    """Per-ROI bars: baseline F on every channel, peak dF/F, latency, noise
    and SNR. Reads the response mode (stimulus or spontaneous) from the
    metrics themselves."""
    plt = _agg_plt()
    K = len(metrics)
    colors = _roi_colors(K)
    x = np.arange(K)
    mode = metrics[0]["response_mode"] if metrics else "spontaneous"
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor=_FIG_BG)
    axes = axes.ravel()
    for ax in axes:
        _dark(ax)

    ax = axes[0]
    ax.bar(x, [m["f0"] for m in metrics], color=colors, label=_channel_label(channel_names, main_channel))
    for c, Fo in F_other.items():
        ax.plot(x, np.percentile(Fo, 10, axis=1), "o", color="red" if "red" in _channel_label(channel_names, c) else "white",
                ms=4, label=f"{_channel_label(channel_names, c)} (10th pct)")
    ax.set_title("F0", fontsize=9)
    ax.legend(fontsize=7, facecolor=_FIG_BG, labelcolor=_FIG_FG, edgecolor=_FIG_FG)

    def _bars(ax, key, title, fmt="{:.2f}"):
        vals = np.array([m[key] for m in metrics], dtype=float)
        ax.bar(x, np.nan_to_num(vals), color=colors)
        ax.set_title(title, fontsize=9)
        top = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, fmt.format(v), color=_FIG_FG, fontsize=6, ha="center",
                        va="bottom" if v >= 0 else "top")
        ax.set_ylim(min(0, np.nanmin(vals)) - 0.1 * top if np.isfinite(vals).any() else 0,
                    (np.nanmax(vals) if np.isfinite(vals).any() else 1) + 0.25 * top)

    _bars(axes[1], "peak_dfof", "peak dF/F")
    _bars(axes[2], "time_to_peak_s", "time to peak (s)", "{:.2f}")
    _bars(axes[3], "noise_dfof", "noise (dF/F)", "{:.3f}")
    _bars(axes[4], "snr", "SNR", "{:.1f}")
    if mode == "stim":
        _bars(axes[5], "response_auc", "AUC 0-2 s", "{:.2f}")
    else:
        axes[5].text(0.5, 0.5, "no stimulus in this run:\npeak and noise are over the whole trace",
                     color=_FIG_FG, ha="center", va="center", fontsize=9, transform=axes[5].transAxes)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(K)], fontsize=7)
        ax.set_xlabel("ROI", fontsize=8)
    fig.suptitle("Response metrics", color=_FIG_FG, fontsize=11)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def plot_motion_correction(
    out_dir: Path, curves: dict, duration_s: float, stim, save_name: str = "07_motion_correction.png",
) -> Path | None:
    """The AOD's real-time motion correction (RTMC) totals in X, Y and Z over
    the run. A drifting trace means the lines were being moved to follow the
    tissue; a jump means a correction the traces may show as a step."""
    plt = _agg_plt()
    axes_data = [(a, curves.get(f"RTMC {a} correction (total)")) for a in ("X", "Y", "Z")]
    axes_data = [(a, c) for a, c in axes_data if c is not None and len(c["timestamps"]) > 1]
    if not axes_data:
        return None
    fig, axes = plt.subplots(len(axes_data), 1, figsize=(12, 2.2 * len(axes_data) + 0.8),
                             facecolor=_FIG_BG, sharex=True, squeeze=False)
    for ax, (name, c) in zip(axes[:, 0], axes_data):
        _dark(ax)
        t = np.asarray(c["timestamps"], dtype=float) / 1000.0
        v = np.asarray(c["values"], dtype=float)
        ax.plot(t, v, color="deepskyblue", lw=0.6)
        ax.set_ylabel(f"{name} (um)", fontsize=9)
        ax.set_title(f"RTMC {name}", fontsize=9)
        _mark_stim(ax, stim, text=(name == "X"))
        ax.set_xlim(0, duration_s)
    axes[-1, 0].set_xlabel("time (s)", fontsize=9)
    fig.suptitle("Motion correction", color=_FIG_FG, fontsize=11)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path


def plot_linescan_figures(
    out_dir: str | Path,
    *,
    mesc_path,
    unit_key: str,
    F_by_channel: dict[int, np.ndarray],
    kymo_by_channel: dict[int, np.ndarray],
    kymo_bin_s: float,
    dfof: np.ndarray | None,
    fs: float,
    extents,
    channel_names,
    main_channel: int,
    stim: dict | None,
    metrics: list[dict],
    curves: dict | None = None,
    reference: dict | None = None,
    flip_y: bool = False,
    logger=None,
) -> list[Path]:
    """Write the whole numbered figure set into ``out_dir``; returns the
    paths written. Each figure is independent: one failing is logged and
    the rest still run."""
    logger = logger or log.get("linescan")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("[0-9][0-9]*_*.png"):
        old.unlink()
    written: list[Path] = []
    um_per_px = None
    try:
        with h5py.File(mesc_path, "r") as f:
            proto = json.loads(f[unit_key].attrs["MultiROIProtocolJSON"])
            main = int(proto["protocol"]["scanners"]["mainPatternIndex"]) - 1
            ps = proto["scanPatterns"]["patterns"][main].get("pixelSize")
            um_per_px = float(ps[0] if isinstance(ps, (list, tuple)) else ps) if ps else None
    except Exception:
        um_per_px = None

    def _try(what, fn, *args, **kwargs):
        try:
            p = fn(*args, **kwargs)
            if p is not None:
                written.append(p)
        except Exception as e:  # a figure never aborts the extraction
            logger.warning(f"linescan figures: {what} failed: {e}")

    F = F_by_channel[main_channel]
    label = _channel_label(channel_names, main_channel)
    bg = None
    try:
        bg = background_image(mesc_path, unit_key)
    except Exception as e:
        logger.warning(f"linescan figures: background image lookup failed: {e}")
    if bg is not None:
        _try("background snapshot", plot_background_snapshot, out_dir, mesc_path, unit_key, bg,
             extents, flip_y=flip_y)
    if reference is not None:
        _try("reference zstack", plot_reference_zstack, out_dir, mesc_path, unit_key, reference,
             extents, flip_y=flip_y)
    if bg is not None or reference is not None:
        _try("line zooms", plot_line_zooms, out_dir, mesc_path, unit_key, extents, bg=bg,
             ref=reference, flip_y=flip_y)
    if kymo_by_channel:
        _try("line profiles", plot_line_profiles, out_dir, kymo_by_channel, extents, channel_names,
             um_per_px)
        for j, c in enumerate(sorted(kymo_by_channel)):
            name = _channel_label(channel_names, c)
            _try(f"kymographs {name}", plot_kymographs, out_dir, kymo_by_channel[c], kymo_bin_s,
                 extents, stim, name, f"03{'abcdefgh'[j]}_kymographs_{name}.png", um_per_px)
    _try("raw traces", plot_traces, out_dir, F, fs, stim, kind="raw", save_name="04a_traces_raw.png",
         label=label)
    if dfof is not None:
        _try("dfof traces", plot_traces, out_dir, dfof, fs, stim, kind="dfof",
             save_name="04b_traces_dfof.png", label=label)
    if stim and stim.get("onsets_s"):
        _try("stim response", plot_stim_response, out_dir, F, fs, stim)
    others = {c: v for c, v in F_by_channel.items() if c != main_channel}
    _try("response metrics", plot_response_metrics, out_dir, metrics, others, channel_names, main_channel)
    if curves:
        _try("motion correction", plot_motion_correction, out_dir, curves, F.shape[1] / fs, stim)
    return written
