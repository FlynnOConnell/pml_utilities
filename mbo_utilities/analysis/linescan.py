"""The experiment folder's line scan and the trace figure masknmf's QC reuses."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "experiment_linescan_mesc",
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
