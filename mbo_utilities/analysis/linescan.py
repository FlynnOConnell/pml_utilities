"""Where an AOD line-scan unit's lines were drawn: the snapshot behind them, the Z-stack that holds them, the experiment folder's scan, and the trace figure masknmf's QC reuses."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from mbo_utilities.arrays.mesc_geometry import (
    linescan_endpoints_um,
    roi_placements,
    viewport_geometry,
    zstack_depth_info,
)

__all__ = [
    "experiment_linescan_mesc",
    "background_image",
    "zstack_candidates",
    "pair_reference_zstack",
]

_FIG_BG = "black"
_FIG_FG = "white"
_SMOOTH_S = 0.02  # display / peak smoothing window for kHz traces


def experiment_linescan_mesc(path) -> Path | None:
    """The raw line-scan ``.mesc`` of an experiment laid out the way the
    curation notebook's data path is: ``<animal>/<expt>/<expt>/<expt>.mesc``
    with the processed traces in ``<animal>/<expt>/PF`` and the Z-stack in
    ``<animal>/<expt>/<expt>_zstack.mesc``.

    ``path`` may be the experiment folder, its ``PF`` folder, the inner
    ``<expt>/<expt>`` folder, or the ``.mesc`` itself. Returns None when no
    line scan is there (a Data or animal folder, or any other folder), so a
    caller can fall through to whatever else the path may be.
    """
    path = Path(path).expanduser()
    if path.is_file():
        return path if path.suffix.lower() == ".mesc" else None
    if not path.is_dir():
        return None
    if path.name == "PF" or (path.parent.name == path.name and path.parent != path):
        experiment = path.parent
    else:
        experiment = path
    folders = [experiment / experiment.name, experiment]
    for folder in folders:
        named = folder / f"{experiment.name}.mesc"
        if named.is_file():
            return named
    for folder in folders:
        if not folder.is_dir():
            continue
        found = sorted(
            p for p in folder.glob("*.mesc") if not p.stem.lower().endswith("_zstack")
        )
        if found:
            return found[0]
    return None


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


def zstack_candidates(
    mesc_path,
    unit_key: str,
    units: list[dict] | None = None,
    *,
    min_px_per_line: int = 10,
    zstack_path=None,
) -> list[dict]:
    """Every Z-stack of the file scored against a line-scan unit's lines:
    ``xy_fraction`` / ``z_fraction`` of lines inside its field / depth range,
    ``um_per_px``, and ``coarse`` when it puts fewer than ``min_px_per_line``
    pixels along the median line. All stacks, including those holding no
    line at all, so a picker can show why one is unsuitable. ``zstack_path``
    is the file holding the stacks when they were saved separately from the
    line scan (``units`` then lists that file's units).
    """
    from mbo_utilities.arrays.mesc import list_mesc_units

    lines = linescan_endpoints_um(mesc_path, unit_key)
    if not lines:
        return []
    stack_path = mesc_path if zstack_path is None else zstack_path
    lengths = [float(np.hypot(*(seg[:2, -1] - seg[:2, 0]))) for seg in lines]
    max_um_per_px = float(np.median(lengths)) / max(min_px_per_line, 1)
    units = units if units is not None else list_mesc_units(stack_path)
    out = []
    for u in units:
        if u["modality_name"] != "zstack":
            continue
        vp = viewport_geometry(stack_path, u["key"])
        depth = zstack_depth_info(stack_path, u["key"])
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
    mesc_path,
    unit_key: str,
    units: list[dict] | None = None,
    *,
    min_px_per_line: int = 10,
    zstack_path=None,
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
    cands = [
        c
        for c in zstack_candidates(
            mesc_path,
            unit_key,
            units,
            min_px_per_line=min_px_per_line,
            zstack_path=zstack_path,
        )
        if c["xy_fraction"] > 0
    ]
    if not cands:
        return None
    fine = [c for c in cands if not c["coarse"]]
    pool = fine or cands
    good = [c for c in pool if c["xy_fraction"] >= 0.8]
    if good:
        return min(good, key=lambda c: c["um_per_px"])
    return max(pool, key=lambda c: (c["xy_fraction"], -c["um_per_px"]))


def _smooth(x: np.ndarray, fs: float, window_s: float = _SMOOTH_S) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    n = max(1, int(round(window_s * fs)))
    return uniform_filter1d(x, size=n, axis=-1, mode="nearest") if n > 1 else x


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

    return np.array(
        [CLASS_COLORS[i % len(CLASS_COLORS)][:3] for i in range(K)], dtype=float
    )


def _mark_stim(ax, stim, fs=None, text=True):
    if not stim:
        return
    for k, onset in enumerate(stim["onsets_s"]):
        ax.axvline(onset, color="yellow", lw=0.8, alpha=0.8, ls="--")
        if text and k == 0:
            ax.text(
                onset,
                1.0,
                " stim",
                color="yellow",
                fontsize=7,
                va="top",
                ha="left",
                transform=ax.get_xaxis_transform(),
            )


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
    out_dir: Path,
    F: np.ndarray,
    fs: float,
    stim,
    *,
    kind: str,
    save_name: str,
    label: str = "",
) -> Path:
    """Stacked per-ROI traces over the whole run. ``kind`` is ``"raw"`` (F as
    recorded) or ``"dfof"`` (already dF/F, drawn 20 ms-smoothed).
    """
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
    ax.plot(
        [x_bar, x_bar], [offsets[-1], offsets[-1] + step / (1.15)], color=_FIG_FG, lw=2
    )
    unit = "dF/F" if kind == "dfof" else "counts"
    ax.text(
        x_bar,
        offsets[-1] + step / 2.3,
        f" {step / 1.15:.2g} {unit}",
        color=_FIG_FG,
        fontsize=7,
        ha="right",
        va="center",
        rotation=90,
    )
    ax.set_xlabel("time (s)", fontsize=9)
    title = f"F ({label})" if kind == "raw" else f"dF/F ({label})"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = out_dir / save_name
    _save(fig, path)
    return path
