"""Traces of a recording's ROIs for slides: loaded, processed and drawn.

:func:`trace_units` lists the units of a ``.mesc`` that have ROIs to trace,
:func:`load_rows` reads one unit's line or patch means and the ROIs drawn on
it (cached under ``~/.mbo/cache``), :func:`process` turns a row into what is
shown (raw, dF/F or z-score, filtered) and :class:`MinMaxPyramid` lets a view
of any span draw about one point per pixel column. :func:`trace_figure` draws
rows one per cell, each in its own colour on its own scale with a labelled
bar, over a time axis; :func:`export_figure` writes PNG, SVG, PDF, the
window as CSV and the settings as JSON. :func:`compute_readouts` measures
rows over a span and :func:`morphology_figure` draws them on their tissue.
The dashboard window (``gui/dashboard_app.py``) is one view of these.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.font_manager
import matplotlib.image
import matplotlib.ticker
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mbo_utilities import log
from mbo_utilities.analysis.filters import FilterSettings, apply_filter
from mbo_utilities.analysis.trace_readouts import (
    correlation_matrix,
    detect_events,
    noise_floor_hz,
    power_spectrum,
    triggered_average,
)
from mbo_utilities.annotation.display import DffSettings, display_trace
from mbo_utilities.annotation.traces import RoiTrace

logger = log.get("dashboard")

__all__ = [
    "FORMATS",
    "FigureOutline",
    "FigureSettings",
    "FigureTrace",
    "MinMaxPyramid",
    "PALETTES",
    "ReadoutSettings",
    "DOMAIN_SOURCE",
    "Readouts",
    "TraceView",
    "bar_label",
    "compute_readouts",
    "cut",
    "export_figure",
    "export_morphology",
    "export_readouts",
    "figure_rgba",
    "load_rows",
    "morphology_figure",
    "nice_bar",
    "process",
    "row_times",
    "trace_figure",
    "trace_units",
]

FORMATS = ("png", "svg", "pdf", "csv")
# Arial where installed (Windows, macOS); Nimbus Sans has its metrics on Linux
FONTS = ["Arial", "Nimbus Sans", "DejaVu Sans"]
PALETTES = ("color", "black")
THEMES = ("light", "dark")
SCALES = ("row", "shared")
KINDS = {"raw": "raw", "dff": "dF/F", "zscore": "z-score"}
# Okabe-Ito on white; on black, the subcell review's brighter set; index k
# is the same hue in both, so a row keeps its colour across themes
COLORS = {
    "light": (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#E69F00",
        "#000000",
    ),
    "dark": (
        "#4e9cff",
        "#ff8c3a",
        "#4cd97b",
        "#ff5c8a",
        "#c38bff",
        "#ffd43b",
        "#ffffff",
    ),
}
INK = {"light": "black", "dark": "white"}
BACKGROUND = {"light": "white", "dark": "black"}
UNDERLAY = {"light": "#c4c4c4", "dark": "#5c5c5c"}
CAPTION = {"light": "#7a7a7a", "dark": "#9a9a9a"}
# a row's height is its range between these percentiles, so one outlier
# sample does not flatten it
RANGE_PERCENTILES = (0.5, 99.5)
LANE_GAP = 0.3
SAVE_KWARGS = {"bbox_inches": "tight", "pad_inches": 0.05}
# text stays text in the SVG and PDF, so PowerPoint and Illustrator can edit it
EDITABLE_TEXT = {"svg.fonttype": "none", "pdf.fonttype": 42}
KIND_UNITS = {"raw": "photons", "dff": "% dF/F", "zscore": "SD"}
# bump when what a cache file holds changes
CACHE_VERSION = 2
DOMAIN_SOURCE = "domains"  # a domain row's source; its lines are extra["lines"]
PYRAMID_FACTOR = 4


@dataclass
class FigureTrace:
    """One trace as plotted: ``y`` at ``t`` seconds on the recording's clock,
    ``under`` (the same samples unfiltered) drawn in grey beneath it when
    given, in ``color`` (a matplotlib colour; None takes the palette's).
    """

    label: str
    t: np.ndarray
    y: np.ndarray
    under: np.ndarray | None = None
    color: str | None = None


@dataclass
class FigureSettings:
    """How :func:`trace_figure` draws.

    ``start_s`` / ``stop_s`` cut the window (None: the traces' ends).
    ``y_unit`` names the vertical bars (``% dF/F``, ``SD``, ``photons``).
    ``scale`` ``row`` fits each row to its own range with its own bar;
    ``shared`` puts every row on one scale with one bar. ``time_axis`` draws
    a time axis in seconds, else a time bar. ``caption`` is one small grey
    line under the figure (the dashboard says what grey and colour are). A
    bar of None is 1, 2 or 5 x 10^n: the time bar near a fifth of the
    window, a vertical one near 60% of its row's range.
    """

    start_s: float | None = None
    stop_s: float | None = None
    y_unit: str = ""
    width_in: float = 8.0
    row_height_in: float = 1.0
    font_pt: float = 12.0
    line_pt: float = 0.75
    time_bar_s: float | None = None
    y_bar: float | None = None
    palette: str = "color"
    theme: str = "light"
    scale: str = "row"
    time_axis: bool = True
    labels: bool = True
    transparent: bool = False
    caption: str = ""
    dpi: int = 300

    def __post_init__(self):
        for name, allowed in (
            ("palette", PALETTES),
            ("theme", THEMES),
            ("scale", SCALES),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"{name} must be one of {allowed}, got {getattr(self, name)!r}"
                )


@dataclass
class TraceView:
    """How a row is shown: ``kind`` (:data:`KINDS`) through ``dff``'s
    baseline, then ``filter``. ``invert`` turns a measured trace over its
    mean before dF/F (an indicator that dims with activity); ``flip``
    negates what is shown.
    """

    kind: str = "dff"
    dff: DffSettings = field(default_factory=DffSettings)
    filter: FilterSettings = field(default_factory=FilterSettings)
    invert: bool = False
    flip: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {tuple(KINDS)}, got {self.kind!r}")


@dataclass
class ReadoutSettings:
    """What :func:`compute_readouts` measures: a spectrum in ``resolution_hz``
    bins, and events of the trigger row ``threshold_sd`` robust SDs past its
    median (``polarity`` -1 for troughs), ``gap_ms`` apart, averaged from
    ``before_ms`` to ``after_ms`` around each. ``skip_s`` leaves out the start
    of a recording when no window is active: a scan opens on a transient on
    every row, which the correlation and the average would take for a signal.
    """

    skip_s: float = 1.0
    resolution_hz: float = 1.0
    threshold_sd: float = 5.0
    gap_ms: float = 5.0
    polarity: int = 1
    before_ms: float = 20.0
    after_ms: float = 20.0


@dataclass
class Readouts:
    """What the plotted rows say over ``span`` seconds, one ``labels`` entry a
    row: ``power`` and each row's ``floors`` (the Hz it reaches the noise at);
    ``correlation`` between the rows and the motion traces after them, named
    by ``correlation_labels``; ``mean`` and ``sem`` of every row around
    ``event_times``, the events of row ``trigger``, at ``lags_ms``.
    """

    labels: list[str]
    span: tuple[float, float]
    frequency: np.ndarray
    power: np.ndarray  ## (n_rows, n_frequencies)
    floors: list[float]
    correlation_labels: list[str]
    correlation: np.ndarray  ## (n_rows + n_motion, n_rows + n_motion)
    trigger: int
    event_times: np.ndarray
    lags_ms: np.ndarray
    mean: np.ndarray  ## (n_rows, n_lags)
    sem: np.ndarray  ## (n_rows, n_lags)


@dataclass
class FigureOutline:
    """One ROI as :func:`morphology_figure` draws it: ``pixels`` ``(n, 2)``
    ``[col, row]`` on the image, in ``color`` (a matplotlib colour).
    """

    label: str
    pixels: np.ndarray
    color: str


def trace_units(path) -> list[dict]:
    """The units of a ``.mesc`` with ROIs to trace and more than one
    timepoint (``list_mesc_units`` rows), in file order.
    """
    from mbo_utilities.arrays.mesc import is_trace_unit, list_mesc_units

    return [
        u
        for u in list_mesc_units(path)
        if is_trace_unit(u) and (u.get("nframes") or 0) > 1
    ]


def _cache_path(path: Path, unit: str, channel: int, rois: Path | None) -> Path:
    """The cache file of one unit's traces: named for the recording, keyed
    by its size and mtime, the unit, the channel and the rois zarr's mtime,
    so an edited file or new ROIs read again.
    """
    from mbo_utilities.preferences import get_mbo_dirs

    stat = path.stat()
    parts = [
        str(path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        unit,
        channel,
        CACHE_VERSION,
    ]
    if rois is not None:
        parts += [str(rois), max(p.stat().st_mtime_ns for p in rois.rglob("*"))]
    digest = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]
    return get_mbo_dirs()["cache"] / "dashboard" / f"{path.stem}_{digest}.npz"


def _drawn_store(path: Path, arr):
    """The ROIs drawn on this unit in the ROI tool, ``(store, rois zarr)``,
    or ``(None, None)`` when none were saved or they were drawn on a frame of
    another size.
    """
    from mbo_utilities.annotation.ngff import RoiFile
    from mbo_utilities.preferences import get_roi_folder
    from mbo_utilities.roi_workflow import labels_path

    tag = str(arr.unit_key).strip("/").replace("/", "_")
    rois = labels_path(path, tag, root=get_roi_folder() or None)
    if not rois.exists():
        return None, None
    store = RoiFile.load(rois).store
    if (store.ny, store.nx) != (arr.ny, arr.nx):
        logger.warning(
            f"{rois.name} was drawn on a {store.ny}x{store.nx} frame, the unit is "
            f"{arr.ny}x{arr.nx}; its ROIs are left out"
        )
        return None, None
    names = [name for name, _ in store.plane_axes]
    store.axis_roles = {
        role: name for role, name in (("z", "ROI"), ("c", "Channel")) if name in names
    }
    return store, rois


def load_rows(
    path, unit: str, channel: int = 0, progress=None, trim_s: float | None = None
) -> list[RoiTrace]:
    """One ``RoiTrace`` per ROI of a ``.mesc`` unit: each line's or patch's
    raw mean in photon counts, then each whole-ROI domain of the unit's layout
    in the domain file (:func:`~mbo_utilities.annotation.domains.domain_means`,
    source :data:`DOMAIN_SOURCE`, its lines in ``extra["lines"]``), then each
    ROI drawn on it in the ROI tool (the mean of its pixels on its ROI), all
    read from ``channel``. Every row starts ``trim_s`` into the recording (None:
    the unit's trim in the domain file) and says so in ``frames``, so it stays
    on the recording's clock.

    The traces are cached under ``~/.mbo/cache/dashboard``; nothing is
    written beside the data. ``progress(i, k, seconds)`` follows each ROI of
    a first read.
    """
    from mbo_utilities.analysis.roi_traces import roi_geometry
    from mbo_utilities.annotation.domains import (
        DomainFile,
        domain_means,
        find_domains,
        trim_seconds,
        trimmed_window,
    )
    from mbo_utilities.lazy_array import base_array
    from mbo_utilities.preferences import get_roi_folder
    from mbo_utilities.reader import imread
    from mbo_utilities.roi_workflow import as_movie, measure_mean, unit_roi_means

    path = Path(path)
    arr = imread(path, unit=unit)
    store, rois = _drawn_store(path, arr)
    cache = _cache_path(path, unit, channel, rois)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            lines, drawn = data["lines"], data["drawn"]
        logger.info(f"dashboard: {unit} traces from {cache}")
    else:
        started = time.perf_counter()
        lines = unit_roi_means(arr, channel=channel, progress=progress)
        drawn = np.zeros((0, lines.shape[1]), np.float32)
        if store is not None and store.rois:
            drawn = np.concatenate(
                [
                    measure_mean(
                        as_movie(arr, z=store.roi_z(i), c=channel),
                        [store.mask(i)],
                        neuropil=False,
                    )[0]
                    for i in range(len(store.rois))
                ]
            )
            conv = arr.metadata.get("mesc_channel_conversion") or []
            if channel < len(conv):
                drawn = drawn * float(conv[channel]["scale"]) + float(
                    conv[channel]["offset"]
                )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lines=lines, drawn=drawn)
        logger.info(
            f"dashboard: read {unit} ({len(lines)} ROIs, {len(drawn)} drawn) in "
            f"{time.perf_counter() - started:.1f} s"
        )
    fs = float(arr.fs or 1.0)
    munit = unit.rsplit("/", 1)[-1]
    source = f"{munit} {'lines' if arr.metadata.get('mesc_layout') == 'packed' else 'patches'}"
    positions = getattr(base_array(arr), "line_positions", None) or []
    rows = []
    for k, trace in enumerate(lines):
        extra = {"line": k, "unit": "photons"}
        if k < len(positions):
            extra.update(positions[k])
        rows.append(
            RoiTrace(
                uid=0,
                member=k,
                source=source,
                label=f"ROI {k}",
                fs=fs,
                z=k,
                c=int(channel),
                F=np.asarray(trace, np.float32),
                extra=extra,
            )
        )
    found = find_domains(path, get_roi_folder() or None)
    layout = (
        None
        if found is None
        else DomainFile.read(found).layout_of(munit.rsplit("_", 1)[-1])
    )
    if layout is not None:
        sizes = {r: h * w for r, (h, w) in roi_geometry(arr).items()}
        means = domain_means(lines, layout.domains, sizes)
        if len(means) < len(layout.domains):
            logger.info(
                f"dashboard: {', '.join(n for n in layout.domains if n not in means)} "
                "pick pixels inside their lines and need a pixel read; left out"
            )
        for name, trace in means.items():
            rois = list(layout.domains[name])
            rows.append(
                RoiTrace(
                    uid=0,
                    member=name,
                    source=DOMAIN_SOURCE,
                    label=name,
                    fs=fs,
                    z=rois[0],
                    c=int(channel),
                    F=trace,
                    extra={"unit": "photons", "lines": rois},
                )
            )
    for i, trace in enumerate(drawn):
        record = store.rois[i]
        rows.append(
            RoiTrace(
                uid=int(record.uid),
                z=store.roi_z(i),
                c=int(channel),
                source="drawn",
                label=record.note or f"drawn {record.uid}",
                fs=fs,
                F=np.asarray(trace, np.float32),
                extra={"unit": "photons"},
            )
        )
    if trim_s is None:
        trim_s = trim_seconds(path, unit, get_roi_folder() or None)
    start, stop = trimmed_window(None, lines.shape[1], round(trim_s * fs))
    if start:
        for row in rows:
            row.F = row.F[start:stop]
            row.frames = (start, stop, 1)
    return rows


def row_times(row: RoiTrace, n: int) -> np.ndarray:
    """Seconds on the recording's clock of a row's ``n`` samples: its own
    rate at its binning from the start of its frame window.
    """
    rate = float(row.fs or 1.0)
    first, step = (row.frames[0], row.frames[2]) if row.frames else (0, 1)
    return (first + np.arange(n) * step * row.frame_average) / rate


def process(
    row: RoiTrace, view: TraceView
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t, shown, unfiltered)`` of one row as ``view`` says: the kind
    through ``display_trace``, then the filter (``shown is unfiltered`` when
    there is none).
    """
    y = np.asarray(
        display_trace(row, view.kind, view.dff, False, view.invert, view.flip),
        np.float64,
    )
    shown = (
        y
        if view.filter.kind == "none"
        else apply_filter(y, float(row.fs or 1.0), view.filter)
    )
    return row_times(row, len(y)), shown, y


class MinMaxPyramid:
    """A uniformly sampled trace's min and max over bins of 4, 16, 64, ...
    samples, so a view of any span draws about two points per pixel column.
    """

    def __init__(self, t0: float, dt: float, y: np.ndarray):
        self.t0, self.dt = float(t0), float(dt)
        self.y = np.asarray(y, np.float32)
        self.levels: list[tuple[np.ndarray, np.ndarray]] = []
        lo = hi = self.y
        while len(lo) > 2 * PYRAMID_FACTOR:
            pad = -len(lo) % PYRAMID_FACTOR
            lo = (
                np.concatenate([lo, np.full(pad, lo[-1])])
                .reshape(-1, PYRAMID_FACTOR)
                .min(1)
            )
            hi = (
                np.concatenate([hi, np.full(pad, hi[-1])])
                .reshape(-1, PYRAMID_FACTOR)
                .max(1)
            )
            self.levels.append((lo, hi))

    def view(self, x0: float, x1: float, columns: int) -> tuple[np.ndarray, np.ndarray]:
        """``(x, y)`` to draw from ``x0`` to ``x1`` seconds over ``columns``
        pixels: the samples when few enough, else each bin's min and max.
        """
        n = len(self.y)
        i0 = int(np.clip(np.floor((x0 - self.t0) / self.dt) - 1, 0, n))
        i1 = int(np.clip(np.ceil((x1 - self.t0) / self.dt) + 2, 0, n))
        span = max(i1 - i0, 1)
        level = 0
        while level < len(self.levels) and span / PYRAMID_FACTOR**level > 2 * max(
            columns, 1
        ):
            level += 1
        if level == 0:
            x = self.t0 + np.arange(i0, i1) * self.dt
            return x, self.y[i0:i1]
        size = PYRAMID_FACTOR**level
        lo, hi = self.levels[level - 1]
        j0, j1 = i0 // size, min(-(-i1 // size), len(lo))
        x = self.t0 + (np.arange(j0, j1) * size + size / 2) * self.dt
        return np.repeat(x, 2), np.column_stack([lo[j0:j1], hi[j0:j1]]).ravel()


def cut(
    traces: list[FigureTrace], start_s: float | None, stop_s: float | None
) -> list[FigureTrace]:
    """Each trace's samples from ``start_s`` to ``stop_s`` seconds."""
    out = []
    for tr in traces:
        keep = np.ones(len(tr.t), bool)
        if start_s is not None:
            keep &= tr.t >= start_s
        if stop_s is not None:
            keep &= tr.t <= stop_s
        under = None if tr.under is None else tr.under[keep]
        out.append(FigureTrace(tr.label, tr.t[keep], tr.y[keep], under, tr.color))
    return out


def nice_bar(span: float, fraction: float = 0.2) -> float:
    """The 1, 2 or 5 x 10^n at or below ``fraction`` of ``span``."""
    target = abs(float(span)) * fraction
    if not target > 0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(target))
    return float(max(m for m in (1.0, 2.0, 5.0) if m * magnitude <= target) * magnitude)


def _font() -> str:
    """The first of ``FONTS`` matplotlib has, so it never warns per text."""
    names = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    return next(f for f in FONTS if f in names)


def bar_label(value: float, unit: str) -> str:
    """``20% dF/F``, ``2 SD``, ``0.5 s``: a percent unit sits on the number."""
    return f"{value:g}{'' if unit.startswith('%') else ' '}{unit}".strip()


def _envelope(t: np.ndarray, y: np.ndarray, n_bins: int):
    """``(t, y)`` cut to its min and max in each of ``n_bins`` bins, in time
    order, so a 1 kHz line scan draws as it looks at the pixels it lands on.
    """
    if len(y) <= 2 * n_bins:
        return t, y
    per = -(-len(y) // n_bins)
    n_bins = -(-len(y) // per)
    ## NOTE: the last bin is padded with its own last sample, never dropped
    pad = n_bins * per - len(y)
    yb = np.concatenate([y, np.full(pad, y[-1])]).reshape(n_bins, per)
    tb = np.concatenate([t, np.full(pad, t[-1])]).reshape(n_bins, per)
    lo, hi = np.nanargmin(yb, axis=1), np.nanargmax(yb, axis=1)
    idx = np.sort(np.stack([lo, hi], axis=1), axis=1)  ## (n_bins, 2)
    rows = np.arange(n_bins)[:, None]
    return tb[rows, idx].ravel(), yb[rows, idx].ravel()


def trace_figure(traces: list[FigureTrace], settings: FigureSettings) -> Figure:
    """The traces in ``settings``' window, one row each top to bottom, each
    in its own colour, named on the left with its vertical bar on the right,
    over a time axis or a time bar.
    """
    shown = [tr for tr in cut(traces, settings.start_s, settings.stop_s) if len(tr.y)]
    if not shown:
        raise ValueError("no samples in the window")
    t0 = min(float(tr.t[0]) for tr in shown)
    t1 = max(float(tr.t[-1]) for tr in shown)
    bounds = np.array(
        [
            np.nanpercentile(tr.y if tr.under is None else tr.under, RANGE_PERCENTILES)
            for tr in shown
        ]
    )  ## (rows, 2)
    ranges = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-12)
    if settings.scale == "shared":
        ranges[:] = ranges.max()
    bars = [settings.y_bar or nice_bar(r, fraction=0.6) for r in ranges]

    theme = settings.theme
    ink = INK[theme]
    n = len(shown)
    height_in = settings.row_height_in * n + 0.6
    fig = Figure(
        figsize=(settings.width_in, height_in),
        dpi=settings.dpi,
        facecolor=BACKGROUND[theme],
    )
    FigureCanvasAgg(fig)
    ## 0.6 in under the rows holds the time axis or bar
    ax = fig.add_axes((0.0, 0.6 / height_in, 1.0, 1.0 - 0.6 / height_in))
    ax.set_facecolor(BACKGROUND[theme])
    palette = COLORS[theme] if settings.palette == "color" else (ink,)
    n_bins = int(settings.width_in * settings.dpi)
    font = {"family": _font(), "size": settings.font_pt}
    small = {"family": _font(), "size": max(settings.font_pt * 0.8, 6.0)}
    lane = 1.0 + LANE_GAP
    colors = [
        palette[k % len(palette)]
        if tr.color is None or settings.palette == "black"
        else tr.color
        for k, tr in enumerate(shown)
    ]
    for k, (tr, low, height, bar, color) in enumerate(
        zip(shown, bounds[:, 0], ranges, bars, colors)
    ):
        base = -k * lane
        if tr.under is not None:
            t, y = _envelope(tr.t, (tr.under - low) / height, n_bins)
            ax.plot(
                t,
                y + base,
                color=UNDERLAY[theme],
                lw=0.6 * settings.line_pt,
                solid_joinstyle="round",
            )
        t, y = _envelope(tr.t, (tr.y - low) / height, n_bins)
        ax.plot(t, y + base, color=color, lw=settings.line_pt, solid_joinstyle="round")
        if settings.scale == "shared" and k < n - 1:
            continue
        ## bars sit just right of the axes, in axes x and data y
        y0 = base + 0.5 - bar / height / 2
        ax.plot(
            [1.008, 1.008],
            [y0, y0 + bar / height],
            transform=ax.get_yaxis_transform(),
            color=ink,
            lw=2 * settings.line_pt,
            solid_capstyle="butt",
            clip_on=False,
        )
        ax.annotate(
            bar_label(bar, settings.y_unit),
            xy=(1.008, base + 0.5),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0.4 * settings.font_pt, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=ink,
            **small,
        )
    if settings.labels:
        for k, tr in enumerate(shown):
            ax.annotate(
                tr.label,
                xy=(0.0, -k * lane + 0.5),
                xycoords=ax.get_yaxis_transform(),
                xytext=(-0.5 * settings.font_pt, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color=colors[k],
                **font,
            )
    ax.set_xlim(t0, t1)
    ax.set_ylim(-(n - 1) * lane - 0.15, 1.15)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    if settings.time_axis:
        ax.spines["bottom"].set_color(ink)
        ax.spines["bottom"].set_linewidth(0.75 * settings.line_pt)
        ax.tick_params(
            axis="x", colors=ink, labelsize=small["size"], width=0.75 * settings.line_pt
        )
        for label in ax.get_xticklabels():
            label.set_family(small["family"])
        ax.set_xlabel("time (s)", color=ink, fontdict=font)
        if settings.time_bar_s is not None:
            ax.xaxis.set_major_locator(
                matplotlib.ticker.MultipleLocator(settings.time_bar_s)
            )
    else:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        time_bar = settings.time_bar_s or nice_bar(t1 - t0)
        bottom = -(n - 1) * lane - 0.12
        ax.plot(
            [t0, t0 + time_bar],
            [bottom, bottom],
            color=ink,
            lw=2 * settings.line_pt,
            solid_capstyle="butt",
            clip_on=False,
        )
        ax.text(
            t0 + time_bar / 2,
            bottom - 0.06,
            bar_label(time_bar, "s"),
            ha="center",
            va="top",
            color=ink,
            fontdict=small,
        )
    if settings.caption:
        fig.text(
            0.99,
            0.005,
            settings.caption,
            ha="right",
            va="bottom",
            color=CAPTION[theme],
            fontdict=small,
        )
    return fig


def figure_rgba(fig: Figure, settings: FigureSettings, dpi: float) -> np.ndarray:
    """``fig`` as its PNG would be at ``dpi``, ``(height, width, 4)`` uint8."""
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, transparent=settings.transparent, **SAVE_KWARGS
    )
    buf.seek(0)
    rgba = matplotlib.image.imread(buf, format="png")  ## (height, width, 4) in [0, 1]
    return np.ascontiguousarray((rgba * 255).round().astype(np.uint8))


def compute_readouts(
    labels: list[str],
    t: np.ndarray,
    traces: np.ndarray,
    fs: float,
    trigger: int,
    span: tuple[float, float],
    settings: ReadoutSettings,
    motion=None,
    motion_filter: FilterSettings | None = None,
) -> Readouts:
    """Measure ``(n_rows, n_samples)`` ``traces`` at ``t`` seconds over ``span``.

    ``motion`` (a ``MotionCorrection``) joins the correlation through
    ``motion_filter``, the filter the rows went through, so both are in one band.

    Raises
    ------
    ValueError
        If ``span`` holds fewer than two samples.
    """
    first, last = np.searchsorted(t, span[0]), np.searchsorted(t, span[1], "right")
    if last - first < 2:
        raise ValueError(f"{span[0]:g} to {span[1]:g} s holds fewer than two samples")
    times, shown = t[first:last], traces[:, first:last]
    frequency, power = power_spectrum(shown, fs, settings.resolution_hz)
    motion_traces = {} if not motion else motion.at(times)
    if motion_filter is not None:
        motion_traces = {
            label: apply_filter(y, fs, motion_filter)
            for label, y in motion_traces.items()
        }
    events = detect_events(
        shown[trigger],
        fs,
        settings.threshold_sd,
        settings.gap_ms,
        settings.polarity,
    )
    before = int(round(settings.before_ms * fs / 1000))
    after = int(round(settings.after_ms * fs / 1000))
    mean, sem, _ = triggered_average(shown, events, before, after)
    return Readouts(
        labels=labels,
        span=span,
        frequency=frequency,
        power=power,
        floors=[noise_floor_hz(frequency, p) for p in power],
        correlation_labels=[
            *labels,
            *(f"{motion.source} {label}" for label in motion_traces),
        ],
        correlation=correlation_matrix(
            np.concatenate(
                [shown, np.array(list(motion_traces.values())).reshape(-1, len(times))]
            )
        ),
        trigger=trigger,
        event_times=times[events],
        lags_ms=np.arange(-before, after + 1) * 1000 / fs,
        mean=mean,
        sem=sem,
    )


def export_readouts(readouts: Readouts, stem: Path | str) -> list[Path]:
    """Write ``<stem>_spectrum.csv``, ``_correlation.csv``, ``_triggered.csv``
    and ``_events.csv``. Returns the paths written.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    r = readouts
    tables = {
        "spectrum": (["frequency_hz", *r.labels], [r.frequency, *r.power]),
        "triggered": (
            [
                "lag_ms",
                *r.labels,
                *(f"{label} sem" for label in r.labels),
            ],
            [r.lags_ms, *r.mean, *r.sem],
        ),
        "events": (["time_s"], [r.event_times]),
        "correlation": (r.correlation_labels, list(r.correlation)),
    }
    written = []
    for name, (header, columns) in tables.items():
        path = stem.with_name(f"{stem.name}_{name}.csv")
        with path.open("w") as f:
            f.write(",".join(header) + "\n")
            np.savetxt(f, np.column_stack(columns), delimiter=",", fmt="%.6g")
        written.append(path)
    return written


def morphology_figure(
    image: np.ndarray,
    outlines: list[FigureOutline],
    um_per_px: float,
    view: tuple[float, float, float, float],
    settings: FigureSettings,
) -> Figure:
    """A ``(Y, X)`` image under ``outlines`` and a scale bar, cropped to
    ``view`` ``(col0, col1, row0, row1)``. Rows grow upward, as the dashboard
    and MESc draw them.
    """
    col0, col1, row0, row1 = view
    width = settings.width_in / 2
    fig = Figure(figsize=(width, width * (row1 - row0) / (col1 - col0)))
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image, cmap="gray", origin="lower", interpolation="nearest")
    for outline in outlines:
        col, row = outline.pixels.T
        ax.plot(col, row, color=outline.color, lw=settings.line_pt * 2)
        if settings.labels:
            ax.annotate(
                outline.label,
                (col[-1], row[-1]),
                xytext=(3, 3),
                textcoords="offset points",
                color=outline.color,
                fontsize=settings.font_pt * 0.75,
                fontfamily=_font(),
                annotation_clip=True,
            )
    bar_um = nice_bar((col1 - col0) * um_per_px)
    bar_px = bar_um / um_per_px
    right = col1 - 0.04 * (col1 - col0)
    low = row0 + 0.05 * (row1 - row0)
    ax.plot([right - bar_px, right], [low, low], color="white", lw=2)
    ax.annotate(
        bar_label(bar_um, "um"),
        (right - bar_px / 2, low),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        color="white",
        fontsize=settings.font_pt * 0.75,
        fontfamily=_font(),
    )
    ax.set_xlim(col0, col1)
    ax.set_ylim(row0, row1)
    ax.set_axis_off()
    return fig


def _save_figure(fig: Figure, path: Path, settings: FigureSettings) -> None:
    """Save ``fig`` with editable text, its format from ``path``'s suffix."""
    with matplotlib.rc_context(EDITABLE_TEXT):
        fig.savefig(
            path, dpi=settings.dpi, transparent=settings.transparent, **SAVE_KWARGS
        )
    if path.suffix == ".svg" and _font() == "Nimbus Sans":
        ## NOTE: Nimbus Sans has Arial's metrics; named Arial, the SVG opens
        ## with Arial in PowerPoint on a machine without Nimbus Sans
        path.write_text(path.read_text().replace("Nimbus Sans", "Arial"))


def export_morphology(
    fig: Figure, settings: FigureSettings, stem, info: dict
) -> list[Path]:
    """Write a :func:`morphology_figure` as ``<stem>.png``, ``.svg``, ``.pdf``
    and ``info`` as ``<stem>.json``. Returns the paths written.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = [stem.with_name(f"{stem.name}.{fmt}") for fmt in ("png", "svg", "pdf")]
    for path in written:
        _save_figure(fig, path, settings)
    path = stem.with_name(f"{stem.name}.json")
    path.write_text(json.dumps(info, indent=2))
    return [*written, path]


def export_figure(
    traces: list[FigureTrace],
    settings: FigureSettings,
    stem: Path | str,
    formats=FORMATS,
    info: dict | None = None,
) -> list[Path]:
    """Write ``<stem>.<format>`` for each of ``formats`` and ``<stem>.json``
    (``settings`` and ``info``, what the figure was made from). The CSV is
    the window's samples: ``time_s`` and one column per trace when they share
    a clock, else ``label,time_s,value`` rows. Returns the paths written.
    """
    unknown = set(formats) - set(FORMATS)
    if unknown:
        raise ValueError(f"formats must be among {FORMATS}, got {sorted(unknown)}")
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    fig = trace_figure(traces, settings)
    for fmt in formats:
        if fmt == "csv":
            continue
        path = stem.with_name(f"{stem.name}.{fmt}")
        _save_figure(fig, path, settings)
        written.append(path)
    if "csv" in formats:
        shown = cut(traces, settings.start_s, settings.stop_s)
        path = stem.with_name(f"{stem.name}.csv")
        shared = all(np.array_equal(tr.t, shown[0].t) for tr in shown)
        with path.open("w") as f:
            if shared:
                f.write(",".join(["time_s", *(tr.label for tr in shown)]) + "\n")
                np.savetxt(
                    f,
                    np.column_stack([shown[0].t, *(tr.y for tr in shown)]),
                    delimiter=",",
                    fmt="%.6g",
                )
            else:
                f.write("label,time_s,value\n")
                for tr in shown:
                    for ti, yi in zip(tr.t, tr.y):
                        f.write(f"{tr.label},{ti:.6g},{yi:.6g}\n")
        written.append(path)
    path = stem.with_name(f"{stem.name}.json")
    path.write_text(
        json.dumps({"settings": asdict(settings), **(info or {})}, indent=2)
    )
    written.append(path)
    logger.info(
        f"dashboard: wrote {stem.name} ({', '.join(p.suffix[1:] for p in written)}) to {stem.parent}"
    )
    return written
