"""The dashboard: a ``.mesc`` unit's ROI traces, processed, windowed and
exported for slides, in a hello_imgui window of its own.

Open a ``.mesc``, pick a unit with ROIs (:func:`~mbo_utilities.dashboard.trace_units`),
and its line or patch means load with any ROIs drawn on it in the ROI tool
(:func:`~mbo_utilities.dashboard.load_rows`, cached after the first read).
The ROIs are on the left, the processing (kind, dF/F baseline, filter),
read-outs, windows and export on the right, and in the middle the ROIs on their
tissue (``gui/dashboard_morphology.py``) beside an overview of the whole
recording over one plot per ROI on a shared time axis, or a read-out of them
(``gui/dashboard_readouts.py``). Every trace is processed on a worker thread
and drawn from a min/max pyramid, so the window never waits on a computation
and draws about two points per pixel column at any zoom.

``mbo dashboard [file.mesc] [--unit MUnit_28]``, the launcher's Dashboard
entry and the MESc tab's Dashboard button open it
(``python -m mbo_utilities.gui.dashboard_app``).
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.colors
import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx, immapp, immvision, implot
from imgui_bundle import portable_file_dialogs as pfd

from mbo_utilities import log
from mbo_utilities.analysis.dff import NOISE_ESTIMATES
from mbo_utilities.analysis.filters import FILTERS, describe
from mbo_utilities.annotation.display import DFF_METHODS
from mbo_utilities.dashboard import (
    COLORS,
    KIND_UNITS,
    KINDS,
    PALETTES,
    SCALES,
    THEMES,
    FigureSettings,
    FigureTrace,
    MinMaxPyramid,
    ReadoutSettings,
    TraceView,
    export_figure,
    figure_rgba,
    load_rows,
    process,
    trace_figure,
    trace_units,
)
from mbo_utilities.gui import _setup
from mbo_utilities.gui.dashboard_morphology import (
    PANEL_WIDTH,
    MorphologyMixin,
    write_morphology,
)
from mbo_utilities.gui.dashboard_readouts import PAGES, ReadoutsMixin
from mbo_utilities.gui.imgui.lines import (
    X_AXIS_HIDDEN,
    line,
    plot_style,
    subplots,
    vec4,
)
from mbo_utilities.gui.launch import launch_window
from mbo_utilities.roi_workflow import roi_folder

logger = log.get("gui.dashboard_app")

__all__ = ["DashboardApp", "launch_dashboard", "open_dashboard"]

TITLE = "Dashboard"
WINDOW_SIZE = (1700, 1000)
LEFT_WIDTH = 230
RIGHT_WIDTH = 360
FIELD_WIDTH = 90
OVERVIEW_HEIGHT = 110
ROW_HEIGHT = 140
PREVIEW_DPI = 100
UNDER_RGB = (0.45, 0.45, 0.45)
WINDOW_RGB = (0.95, 0.85, 0.35)
ERROR = imgui.ImVec4(1.0, 0.45, 0.45, 1.0)
FILTER_LABELS = {
    "none": "none",
    "lowpass": "low-pass",
    "highpass": "high-pass",
    "bandpass": "band-pass",
    "notch": "notch",
    "gaussian": "Gaussian",
    "mean": "moving mean",
}


@dataclass
class RowResult:
    """One row as ``view`` showed it: its samples and their pyramids."""

    view: str
    t: np.ndarray
    shown: np.ndarray
    under: np.ndarray | None
    pyramid: MinMaxPyramid
    pyramid_under: MinMaxPyramid | None


def compute_row(row, view: TraceView, view_key: str) -> RowResult:
    """Process one row on the worker thread and build its pyramids; the
    unfiltered trace is kept only when a filter is on.
    """
    t, shown, raw = process(row, view)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    under = None if shown is raw else raw.astype(np.float32)
    shown = shown.astype(np.float32)
    return RowResult(
        view_key,
        t,
        shown,
        under,
        MinMaxPyramid(float(t[0]), dt, shown),
        None if under is None else MinMaxPyramid(float(t[0]), dt, under),
    )


def rgb(color: str) -> tuple[float, float, float]:
    return matplotlib.colors.to_rgb(color)


def render_preview(traces: list[FigureTrace], settings: FigureSettings) -> np.ndarray:
    """The figure as its PNG would look, ``(height, width, 3)`` uint8 at ``PREVIEW_DPI``."""
    fig = trace_figure(traces, settings)
    return np.ascontiguousarray(figure_rgba(fig, settings, PREVIEW_DPI)[..., :3])


def write_exports(jobs: list[tuple]) -> str:
    """Write each ``(traces, settings, stem, info)`` job's figure files."""
    written = [
        path
        for traces, settings, stem, info in jobs
        for path in export_figure(traces, settings, stem, info=info)
    ]
    folder = written[0].parent if written else ""
    return f"wrote {len(written)} files to {folder}"


class DashboardApp(MorphologyMixin, ReadoutsMixin):
    """One window's state: the file, the unit and its rows, how they are
    shown, the windows and the export settings.
    """

    def __init__(self, path=None, unit: str | None = None, channel: int = 0):
        _setup.setup_imgui()
        self.path: Path | None = None
        self.units: list[dict] = []
        self.unit = unit
        self.channel = int(channel)
        self.rows: list = []
        self.picked: set[tuple] = set()
        self.view = TraceView()
        self.show_under = True
        self.figure = FigureSettings()
        self.windows: list[dict] = []
        self.active = -1
        self.folder = ""
        self.status = ""
        self.error = ""
        self.row_height = ROW_HEIGHT
        self.frames = 0
        self.max_frames: int | None = None
        self.progress = (0, 0)
        self.x_view = (0.0, 1.0)
        self.duration = (0.0, 1.0)
        self._set_x: tuple[float, float] | None = None
        self._loader = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dashboard-load"
        )
        self._worker = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="dashboard-work"
        )
        self._listing: Future | None = None
        self._loading: Future | None = None
        self._results: dict[tuple, RowResult] = {}
        self._pending: dict[tuple, Future] = {}
        self._view_key = ""
        self._exporting: Future | None = None
        self._previewing: Future | None = None
        self._preview: np.ndarray | None = None
        self._preview_new = False
        self._show_preview = False
        self._dialog = None
        self.process_error = ""
        self.page = PAGES[0]
        self._set_page: str | None = None
        self.readout = ReadoutSettings()
        self.readouts = None
        self.readout_rows = None
        self.readout_error = ""
        self.trigger: tuple | None = None
        self.show_events = False
        self._reading: Future | None = None
        self._reading_rows: list = []
        self._readouts_key: tuple | None = None
        self.show_morphology = True
        self.morph_picked_only = False
        self.morph_pad = 0
        self.morph_view = (0.0, 1.0, 0.0, 1.0)
        self.stack_files: list[str] = []
        self.motion = None
        self._morpher = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dashboard-morphology"
        )
        self._morph_on: tuple | None = None
        self._morph_scale = 1.0
        self._morph_textures: dict = {}
        self._stack_dialog = None
        self._reset_morphology()
        if path is not None:
            self.open(path)

    def open(self, path) -> None:
        """List the units of ``path`` with ROIs; a unit given at start opens."""
        self.path = Path(path)
        self.units, self.rows, self._results = [], [], {}
        self.status = f"reading {self.path.name}"
        self._listing = self._loader.submit(trace_units, self.path)

    def choose(self, unit: str, channel: int | None = None) -> None:
        """Load ``unit``'s ROI traces from ``channel`` on the loader thread."""
        self.unit = unit
        if channel is not None:
            self.channel = int(channel)
        self.rows, self._results, self._pending = [], {}, {}
        self.readouts, self._readouts_key, self._reading = None, None, None
        self._reset_morphology()
        self.progress = (0, 0)
        self.status = f"loading {unit.rsplit('/', 1)[-1]}"
        self._loading = self._loader.submit(
            load_rows, self.path, unit, self.channel, self._on_progress
        )

    def _on_progress(self, i, k, seconds) -> None:
        self.progress = (i + 1, k)

    def _unit_info(self) -> dict:
        return next((u for u in self.units if u["key"] == self.unit), {})

    def _poll(self) -> None:
        """Take what the loader and the worker have finished."""
        if self._listing is not None and self._listing.done():
            listing, self._listing = self._listing, None
            if listing.exception() is not None:
                self.error = f"could not read {self.path.name}: {listing.exception()}"
            else:
                self.units = listing.result()
                self.status = f"{len(self.units)} units with ROIs"
                keys = [u["key"] for u in self.units]
                if self.unit is not None:
                    match = [
                        k for k in keys if k == self.unit or k.endswith("/" + self.unit)
                    ]
                    if match:
                        self.choose(match[0])
                    else:
                        self.error = (
                            f"{self.unit} has no ROIs to trace in {self.path.name}"
                        )
                        self.unit = None
        if self._loading is not None and self._loading.done():
            loading, self._loading = self._loading, None
            if loading.exception() is not None:
                self.error = f"loading {self.unit} failed: {loading.exception()!r}"
                logger.error(self.error)
            else:
                self._loaded(loading.result())
        for key, future in list(self._pending.items()):
            if not future.done():
                continue
            del self._pending[key]
            if future.exception() is not None:
                self.process_error = str(future.exception())
            else:
                self._results[key] = future.result()
        if self._exporting is not None and self._exporting.done():
            exporting, self._exporting = self._exporting, None
            if exporting.exception() is not None:
                self.error = f"export failed: {exporting.exception()}"
            else:
                self.error = ""
                self.status = exporting.result()
        if self._previewing is not None and self._previewing.done():
            previewing, self._previewing = self._previewing, None
            if previewing.exception() is not None:
                self.error = f"preview failed: {previewing.exception()}"
            else:
                self._preview, self._preview_new = previewing.result(), True
        self._poll_morphology()
        self._poll_readouts()

    def _loaded(self, rows) -> None:
        self.rows = rows
        self.picked = {r.key for r in rows[:4]}
        self.error = ""
        n = max((r.n_frames for r in rows), default=0)
        fs = float(rows[0].fs or 1.0) if rows else 1.0
        self.duration = (0.0, max(n - 1, 1) / fs)
        self.x_view = self.duration
        self._set_x = self.duration
        self.windows, self.active = [], -1
        self.folder = str(roi_folder(self.path) / "figures")
        drawn = sum(r.source == "drawn" for r in rows)
        self.status = (
            f"{self.unit.rsplit('/', 1)[-1]}: {len(rows) - drawn} ROIs"
            + (f", {drawn} drawn" if drawn else "")
            + f" at {fs:.0f} Hz, {self.duration[1]:.0f} s"
        )

    def _picked_rows(self) -> list:
        return [r for r in self.rows if r.key in self.picked]

    def _schedule(self) -> None:
        """Process every picked row not yet shown as the current view says."""
        key = repr(self.view)
        if key != self._view_key:
            self._view_key = key
            self.process_error = ""
        if self.process_error:
            return
        for row in self._picked_rows():
            have = self._results.get(row.key)
            if (have is not None and have.view == key) or row.key in self._pending:
                continue
            self._pending[row.key] = self._worker.submit(
                compute_row, row, copy.deepcopy(self.view), key
            )

    def _color(self, row, theme: str = "dark") -> str:
        palette = COLORS[theme]
        return palette[self.rows.index(row) % len(palette)]

    def _row_rgb(self, row) -> tuple[float, float, float]:
        return rgb(self._color(row))

    def _y_unit(self) -> str:
        return KIND_UNITS[self.view.kind]

    def draw(self) -> None:
        """One frame of the window."""
        self.frames += 1
        if self.max_frames is not None and self.frames >= self.max_frames:
            hello_imgui.get_runner_params().app_shall_exit = True
        self._poll()
        self._draw_header()
        if self.path is None:
            self._draw_open()
        elif self.rows:
            self._schedule()
            self._schedule_readouts()
            self._keys()
            self._draw_main()
        elif self._loading is not None:
            done, total = self.progress
            imgui.text(f"Reading the ROIs of {self.unit} (channel {self.channel})")
            imgui.progress_bar(
                done / total if total else 0.0,
                imgui.ImVec2(400, 0),
                f"ROI {done}/{total}" if total else "",
            )
        else:
            self._draw_units()
        self._draw_preview()

    def _draw_header(self) -> None:
        if imgui.button("Open .mesc"):
            self._dialog = pfd.open_file(
                "Open a .mesc",
                str(self.path.parent if self.path else Path.home()),
                ["MESc", "*.mesc"],
            )
        if self._dialog is not None and self._dialog.ready():
            picked = self._dialog.result()
            self._dialog = None
            if picked:
                self.unit = None
                self.open(picked[0])
        if self.path is not None:
            imgui.same_line()
            imgui.text(self.path.name)
        if self.units and self.rows:
            keys = [u["key"] for u in self.units]
            imgui.same_line()
            imgui.set_next_item_width(140)
            index = keys.index(self.unit) if self.unit in keys else 0
            changed, index = imgui.combo(
                "##unit", index, [k.rsplit("/", 1)[-1] for k in keys]
            )
            if changed:
                self.choose(keys[index])
            n_channels = int(self._unit_info().get("nchannels") or 1)
            imgui.same_line()
            imgui.set_next_item_width(70)
            changed, c = imgui.combo(
                "channel##ch", self.channel, [str(c) for c in range(n_channels)]
            )
            if changed:
                self.choose(self.unit, c)
        imgui.same_line()
        if self.error:
            imgui.text_colored(ERROR, self.error)
        else:
            imgui.text_disabled(self.status)
        imgui.separator()

    def _draw_open(self) -> None:
        imgui.text("Open a .mesc to plot the ROIs of one of its units.")

    def _draw_units(self) -> None:
        if self._listing is not None:
            imgui.text_disabled(f"reading {self.path.name}...")
            return
        if not self.units:
            imgui.text_disabled(f"{self.path.name} has no units with ROIs to trace.")
            return
        imgui.text("Pick a unit")
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_fixed_fit
        )
        if not imgui.begin_table("##units", 6, flags):
            return
        for name in ("unit", "ROIs", "layout", "rate (Hz)", "length (s)", "comment"):
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        for u in self.units:
            imgui.table_next_row()
            imgui.table_next_column()
            if imgui.selectable(
                f"{u['key'].rsplit('/', 1)[-1]}##{u['key']}",
                False,
                imgui.SelectableFlags_.span_all_columns,
            )[0]:
                self.choose(u["key"])
            for value in (
                u.get("nrois"),
                u.get("kind"),
                f"{u['fs']:.0f}" if u.get("fs") else "",
                f"{u['duration_s']:.1f}" if u.get("duration_s") else "",
                u.get("comment") or "",
            ):
                imgui.table_next_column()
                imgui.text(str(value))
        imgui.end_table()

    def _keys(self) -> None:
        if imgui.get_io().want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.w, False):
            self.add_window(*self.x_view)
        if imgui.is_key_pressed(imgui.Key.f, False):
            self._set_x = self.duration
        if imgui.is_key_pressed(imgui.Key.m, False):
            self.show_morphology = not self.show_morphology

    def _draw_main(self) -> None:
        avail = imgui.get_content_region_avail()
        with imgui_ctx.begin_child("##rois", imgui.ImVec2(LEFT_WIDTH, 0)):
            self._draw_rois()
        imgui.same_line()
        center = (
            avail.x - LEFT_WIDTH - RIGHT_WIDTH - 2 * imgui.get_style().item_spacing.x
        )
        with imgui_ctx.begin_child("##plots", imgui.ImVec2(center, 0)):
            self._draw_plots()
        imgui.same_line()
        with imgui_ctx.begin_child("##side", imgui.ImVec2(RIGHT_WIDTH, 0)):
            self._draw_processing()
            self._draw_readout_settings()
            self._draw_windows()
            self._draw_export()

    def _draw_rois(self) -> None:
        imgui.text("ROIs")
        imgui.same_line()
        if imgui.small_button("all"):
            self.picked = {r.key for r in self.rows}
        imgui.same_line()
        if imgui.small_button("none"):
            self.picked = set()
        drawn_header = False
        for row in self.rows:
            if row.source == "drawn" and not drawn_header:
                imgui.separator_text("drawn")
                drawn_header = True
            r, g, b = rgb(self._color(row))
            imgui.color_button(
                f"##c{row.key}",
                imgui.ImVec4(r, g, b, 1.0),
                imgui.ColorEditFlags_.no_tooltip,
                imgui.ImVec2(12, 12),
            )
            imgui.same_line()
            changed, on = imgui.checkbox(
                f"{row.name}##{row.key}", row.key in self.picked
            )
            if changed:
                self.picked = (
                    (self.picked | {row.key}) if on else (self.picked - {row.key})
                )
            if imgui.is_item_hovered() and "length_um" in (row.extra or {}):
                e = row.extra
                imgui.set_tooltip(
                    f"{e.get('length_um', 0):.1f} um long, a sample every "
                    f"{e.get('sample_um', 0):.2f} um"
                )

    def _draw_plots(self) -> None:
        if self.show_morphology:
            with imgui_ctx.begin_child("##morph", imgui.ImVec2(PANEL_WIDTH, 0)):
                self._draw_morphology()
            imgui.same_line()
        with imgui_ctx.begin_child("##pages", imgui.ImVec2(0, 0)):
            self._draw_pages()

    def _draw_pages(self) -> None:
        if not imgui.begin_tab_bar("##pages_bar"):
            return
        for page, draw, read_out in (
            (PAGES[0], self._draw_traces, False),
            (PAGES[1], self._draw_spectrum, True),
            (PAGES[2], self._draw_correlation, True),
            (PAGES[3], self._draw_triggered, True),
        ):
            flags = imgui.TabItemFlags_.none
            if page == self._set_page:
                flags = imgui.TabItemFlags_.set_selected
            if imgui.begin_tab_item(page, None, flags)[0]:
                self.page = page
                if not self._picked_rows():
                    imgui.text_disabled("Tick ROIs on the left to plot them.")
                elif read_out:
                    with plot_style(grid=True):
                        self._draw_readout_page(draw)
                else:
                    draw()
                imgui.end_tab_item()
        self._set_page = None
        imgui.end_tab_bar()

    def _draw_traces(self) -> None:
        picked = self._picked_rows()
        with plot_style(grid=True):
            self._draw_overview(picked[0])
            height = max(
                len(picked) * self.row_height, imgui.get_content_region_avail().y
            )
            with imgui_ctx.begin_child("##rows_scroll", imgui.ImVec2(0, 0)):
                width = imgui.get_content_region_avail().x
                self._draw_rows(picked, height, width)

    def _draw_overview(self, row) -> None:
        """The whole recording of the first plotted ROI, the windows shaded
        and the view as a box: drag it to pan, click elsewhere to jump.
        """
        result = self._results.get(row.key)
        flags = (
            implot.Flags_.no_title | implot.Flags_.no_legend | implot.Flags_.no_menus
        )
        if not implot.begin_plot(
            "##overview", imgui.ImVec2(-1, OVERVIEW_HEIGHT), flags
        ):
            return
        implot.setup_axes(
            "", "", implot.AxisFlags_.lock, X_AXIS_HIDDEN | implot.AxisFlags_.lock
        )
        implot.setup_axis_limits(implot.ImAxis_.x1, *self.duration, implot.Cond_.always)
        lo, hi = 0.0, 1.0
        if result is not None:
            lo, hi = (float(v) for v in np.nanpercentile(result.shown, (0.1, 99.9)))
            implot.setup_axis_limits(implot.ImAxis_.y1, lo, hi, implot.Cond_.always)
            x, y = result.pyramid.view(
                *self.duration, int(imgui.get_content_region_avail().x)
            )
            line(
                "##overview_line",
                y,
                x=x,
                color=rgb(self._color(row)),
                alpha=0.8,
                legend=False,
            )
        for i, window in enumerate(self.windows):
            spec = implot.Spec(
                fill_color=vec4(WINDOW_RGB, 0.35 if i == self.active else 0.15),
                flags=implot.ItemFlags_.no_legend,
            )
            implot.plot_shaded(
                f"##w{i}",
                np.array([window["start"], window["stop"]], np.float64),
                np.array([lo, lo], np.float64),
                np.array([hi, hi], np.float64),
                spec,
            )
        x0, x1 = self.x_view
        moved, nx0, _, nx1, _, _clicked, _hovered, held = implot.drag_rect(
            0, x0, lo, x1, hi, imgui.ImVec4(1, 1, 1, 0.9), 0, False, False, False
        )
        if moved:
            width = x1 - x0
            center = (nx0 + nx1) / 2
            self._set_x = (center - width / 2, center + width / 2)
        elif (
            implot.is_plot_hovered()
            and imgui.is_mouse_clicked(imgui.MouseButton_.left)
            and not held
        ):
            mouse = implot.get_plot_mouse_pos().x
            width = x1 - x0
            self._set_x = (mouse - width / 2, mouse + width / 2)
        implot.end_plot()

    def _draw_rows(self, picked, height: float, width: float) -> None:
        """One plot per ROI on a shared time axis, each fitting its own y to
        what is in view; the bottom one carries the time axis.
        """
        flags = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_title
        flags |= implot.SubplotFlags_.no_legend
        columns = int(width)
        with subplots("##rows", len(picked), 1, height=height, flags=flags) as ok:
            if not ok:
                return
            for k, row in enumerate(picked):
                self._draw_row(row, k == 0, k == len(picked) - 1, columns)
        self._set_x = None

    def _draw_row(self, row, first: bool, last: bool, columns: int) -> None:
        if not implot.begin_plot(
            f"##row{row.key}", imgui.ImVec2(-1, 0), implot.Flags_.no_title
        ):
            return
        implot.setup_axes(
            "time (s)" if last else "",
            f"{row.name} ({self._y_unit()})",
            implot.AxisFlags_.none if last else X_AXIS_HIDDEN,
            implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit,
        )
        if self._set_x is not None:
            implot.setup_axis_limits(
                implot.ImAxis_.x1, *self._set_x, implot.Cond_.always
            )
        result = self._results.get(row.key)
        limits = implot.get_plot_limits()
        x0, x1 = limits.x.min, limits.x.max
        if first:
            self.x_view = (x0, x1)
        if result is not None:
            if result.pyramid_under is not None and self.show_under:
                x, y = result.pyramid_under.view(x0, x1, columns)
                line("##under", y, x=x, color=UNDER_RGB, legend=False)
            x, y = result.pyramid.view(x0, x1, columns)
            line(f"{row.name}", y, x=x, color=rgb(self._color(row)), legend=False)
            self._draw_event_marks(row, x0, x1)
            if result.view != self._view_key:
                implot.annotation(
                    x0,
                    limits.y.max,
                    vec4(WINDOW_RGB),
                    imgui.ImVec2(4, 4),
                    True,
                    "updating",
                )
        else:
            implot.annotation(
                x0, 0.0, vec4(WINDOW_RGB), imgui.ImVec2(4, 0), True, "processing"
            )
        for i, window in enumerate(self.windows):
            spec = implot.Spec(
                fill_color=vec4(WINDOW_RGB, 0.18 if i == self.active else 0.07),
                flags=implot.ItemFlags_.no_legend | implot.ItemFlags_.no_fit,
            )
            implot.plot_shaded(
                f"##w{i}",
                np.array([window["start"], window["stop"]], np.float64),
                np.array([limits.y.min] * 2, np.float64),
                np.array([limits.y.max] * 2, np.float64),
                spec,
            )
        implot.end_plot()

    def _draw_processing(self) -> None:
        if not imgui.collapsing_header("Processing", imgui.TreeNodeFlags_.default_open):
            return
        v, d, f = self.view, self.view.dff, self.view.filter
        kinds = list(KINDS)
        imgui.set_next_item_width(FIELD_WIDTH * 1.5)
        changed, i = imgui.combo("kind", kinds.index(v.kind), list(KINDS.values()))
        if changed:
            v.kind = kinds[i]
        if v.kind != "raw":
            methods = list(DFF_METHODS)
            imgui.set_next_item_width(FIELD_WIDTH * 1.5)
            changed, i = imgui.combo("baseline", methods.index(d.method), methods)
            if changed:
                d.method = methods[i]
            imgui.set_item_tooltip(
                "F0 of dF/F: maxmin is the rolling max of the rolling min of F smoothed by sigma; "
                "rolling a rolling percentile; gaussian a slow Gaussian; window the mean over a span."
            )
            for label, name in (
                ("window (s)", "window_s"),
                ("sigma (s)", "sigma_s"),
                ("percentile", "percentile"),
                ("gaussian (s)", "gaussian_s"),
            ):
                if self._dff_field_used(d.method, name):
                    imgui.set_next_item_width(FIELD_WIDTH)
                    _, value = imgui.input_float(
                        label, getattr(d, name), 0.0, 0.0, "%g"
                    )
                    setattr(d, name, max(value, 0.0))
            if d.method == "window":
                imgui.set_next_item_width(FIELD_WIDTH)
                _, d.f0_start_s = imgui.input_float(
                    "F0 from (s)", d.f0_start_s, 0.0, 0.0, "%g"
                )
                imgui.set_next_item_width(FIELD_WIDTH)
                _, d.f0_stop_s = imgui.input_float(
                    "F0 to (s)", d.f0_stop_s, 0.0, 0.0, "%g"
                )
        if v.kind == "zscore":
            imgui.set_next_item_width(FIELD_WIDTH)
            _, value = imgui.input_float("slow drift (s)", d.slow_s, 0.0, 0.0, "%g")
            d.slow_s = max(value, 0.0)
            imgui.set_item_tooltip(
                "A Gaussian of this many seconds is taken off first; 0 keeps it."
            )
            noises = list(NOISE_ESTIMATES)
            imgui.set_next_item_width(FIELD_WIDTH)
            changed, i = imgui.combo("noise", noises.index(d.noise), noises)
            if changed:
                d.noise = noises[i]
        _, v.invert = imgui.checkbox("invert", v.invert)
        imgui.set_item_tooltip(
            "Turn the measured trace over its mean before dF/F: for an indicator that dims "
            "with activity (JEDI, ASAP), so events point up."
        )
        imgui.same_line()
        _, v.flip = imgui.checkbox("flip", v.flip)
        imgui.set_item_tooltip("Negate what is shown.")
        imgui.separator_text("filter")
        imgui.set_next_item_width(FIELD_WIDTH * 1.5)
        changed, i = imgui.combo(
            "##filter", FILTERS.index(f.kind), [FILTER_LABELS[k] for k in FILTERS]
        )
        if changed:
            f.kind = FILTERS[i]
        if f.kind in ("highpass", "bandpass"):
            imgui.set_next_item_width(FIELD_WIDTH)
            _, f.low_hz = imgui.input_float("low (Hz)", f.low_hz, 0.0, 0.0, "%g")
        if f.kind in ("lowpass", "bandpass"):
            imgui.set_next_item_width(FIELD_WIDTH)
            _, f.high_hz = imgui.input_float("high (Hz)", f.high_hz, 0.0, 0.0, "%g")
        if f.kind in ("lowpass", "highpass", "bandpass"):
            imgui.set_next_item_width(FIELD_WIDTH)
            _, order = imgui.input_int("order", f.order, 0, 0)
            f.order = min(max(order, 1), 8)
        if f.kind == "notch":
            imgui.set_next_item_width(FIELD_WIDTH)
            _, f.notch_hz = imgui.input_float("line (Hz)", f.notch_hz, 0.0, 0.0, "%g")
            imgui.set_next_item_width(FIELD_WIDTH)
            _, f.notch_q = imgui.input_float("Q", f.notch_q, 0.0, 0.0, "%g")
        if f.kind in ("gaussian", "mean"):
            imgui.set_next_item_width(FIELD_WIDTH)
            _, value = imgui.input_float("width (ms)", f.width_ms, 0.0, 0.0, "%g")
            f.width_ms = max(value, 0.01)
        if f.kind != "none":
            _, self.show_under = imgui.checkbox("unfiltered in grey", self.show_under)
        if self.process_error:
            imgui.text_colored(ERROR, self.process_error)
        busy = len(self._pending)
        if busy:
            imgui.text_disabled(f"processing {busy} ROI{'s' if busy > 1 else ''}...")

    @staticmethod
    def _dff_field_used(method: str, name: str) -> bool:
        return {
            "window_s": method in ("maxmin", "rolling"),
            "sigma_s": method == "maxmin",
            "percentile": method in ("rolling", "percentile"),
            "gaussian_s": method == "gaussian",
        }[name]

    def add_window(self, start: float, stop: float) -> None:
        self.windows.append(
            {
                "name": f"w{len(self.windows) + 1}",
                "start": float(start),
                "stop": float(stop),
            }
        )
        self.active = len(self.windows) - 1

    def _draw_windows(self) -> None:
        if not imgui.collapsing_header("Windows", imgui.TreeNodeFlags_.default_open):
            return
        if imgui.button("Add the view (w)"):
            self.add_window(*self.x_view)
        imgui.same_line()
        if imgui.button("Whole recording (f)"):
            self._set_x = self.duration
        imgui.text_disabled(f"view {self.x_view[0]:.3f} to {self.x_view[1]:.3f} s")
        remove = None
        for i, window in enumerate(self.windows):
            imgui.push_id(i)
            if imgui.radio_button("##active", i == self.active):
                self.active = i
                self._set_x = (window["start"], window["stop"])
            imgui.same_line()
            imgui.set_next_item_width(60)
            _, window["name"] = imgui.input_text("##name", window["name"])
            imgui.same_line()
            imgui.set_next_item_width(75)
            _, window["start"] = imgui.input_float(
                "##start", window["start"], 0.0, 0.0, "%.3f"
            )
            imgui.same_line()
            imgui.set_next_item_width(75)
            _, window["stop"] = imgui.input_float(
                "##stop", window["stop"], 0.0, 0.0, "%.3f"
            )
            imgui.same_line()
            if imgui.small_button("x"):
                remove = i
            imgui.pop_id()
        if remove is not None:
            del self.windows[remove]
            self.active = min(self.active, len(self.windows) - 1)

    def _draw_export(self) -> None:
        if not imgui.collapsing_header("Export", imgui.TreeNodeFlags_.default_open):
            return
        s = self.figure
        for label, name, fmt, low in (
            ("width (in)", "width_in", "%.1f", 1.0),
            ("row height (in)", "row_height_in", "%.2f", 0.2),
            ("font (pt)", "font_pt", "%.0f", 4.0),
            ("line (pt)", "line_pt", "%.2f", 0.1),
        ):
            imgui.set_next_item_width(FIELD_WIDTH)
            _, value = imgui.input_float(label, getattr(s, name), 0.0, 0.0, fmt)
            setattr(s, name, max(value, low))
        imgui.set_next_item_width(FIELD_WIDTH)
        _, bar = imgui.input_float("time bar (s)", s.time_bar_s or 0.0, 0.0, 0.0, "%g")
        s.time_bar_s = bar if bar > 0 else None
        imgui.set_item_tooltip(
            "The time bar, or the axis's tick spacing. 0 picks a round one."
        )
        imgui.set_next_item_width(FIELD_WIDTH)
        _, bar = imgui.input_float("y bar", s.y_bar or 0.0, 0.0, 0.0, "%g")
        s.y_bar = bar if bar > 0 else None
        imgui.set_item_tooltip(
            "Every vertical bar's length. 0 picks a round one per row."
        )
        for label, name, options in (
            ("color", "palette", PALETTES),
            ("background", "theme", THEMES),
            ("scale", "scale", SCALES),
        ):
            imgui.set_next_item_width(FIELD_WIDTH)
            changed, i = imgui.combo(
                label, options.index(getattr(s, name)), list(options)
            )
            if changed:
                setattr(s, name, options[i])
        imgui.set_next_item_width(FIELD_WIDTH)
        _, dpi = imgui.input_int("dpi", s.dpi, 0, 0)
        s.dpi = min(max(dpi, 50), 1200)
        _, s.time_axis = imgui.checkbox("time axis", s.time_axis)
        imgui.same_line()
        _, s.labels = imgui.checkbox("labels", s.labels)
        imgui.same_line()
        _, s.transparent = imgui.checkbox("transparent", s.transparent)
        imgui.set_next_item_width(-1)
        _, self.folder = imgui.input_text("##folder", self.folder)
        imgui.set_item_tooltip("The folder figures are written to.")
        ready = self._ready()
        busy = self._exporting is not None or self._previewing is not None
        imgui.begin_disabled(not ready or busy)
        if imgui.button("Preview"):
            settings = self._settings(*self.x_view)
            self._previewing = self._worker.submit(
                render_preview, self.figure_traces(settings.theme), settings
            )
            self._show_preview = True
        imgui.same_line()
        if imgui.button("Export view"):
            self._exporting = self._worker.submit(
                write_exports, self.exports([("view", *self.x_view)])
            )
        active = (
            self.windows[self.active] if 0 <= self.active < len(self.windows) else None
        )
        imgui.begin_disabled(active is None)
        imgui.same_line()
        if imgui.button("Export window") and active is not None:
            self._exporting = self._worker.submit(
                write_exports,
                self.exports([(active["name"], active["start"], active["stop"])]),
            )
        imgui.same_line()
        if imgui.button("All windows"):
            self._exporting = self._worker.submit(
                write_exports,
                self.exports(
                    [(w["name"], w["start"], w["stop"]) for w in self.windows]
                ),
            )
        imgui.end_disabled()
        imgui.end_disabled()
        if not ready:
            imgui.text_disabled("waiting for the processing to finish")
        imgui.begin_disabled(self._morph_on is None or self._exporting is not None)
        if imgui.button("Export morphology") and self._morph_on is not None:
            self._exporting = self._worker.submit(
                write_morphology, *self.morphology_job()
            )
        imgui.end_disabled()
        imgui.set_item_tooltip(
            "The panel as shown with the picked ROIs and, with a placed stack, "
            "every ROI's position as a CSV."
        )
        imgui.same_line()
        if imgui.small_button("Open folder"):
            self.open_folder()

    def _ready(self) -> bool:
        picked = self._picked_rows()
        return bool(picked) and all(
            (r := self._results.get(row.key)) is not None and r.view == self._view_key
            for row in picked
        )

    def figure_traces(self, theme: str) -> list[FigureTrace]:
        """The plotted rows as the figure draws them, in this window's colours."""
        return [
            FigureTrace(
                row.name,
                self._results[row.key].t,
                self._results[row.key].shown,
                self._results[row.key].under if self.show_under else None,
                self._color(row, theme),
            )
            for row in self._picked_rows()
        ]

    def _settings(self, start: float, stop: float) -> FigureSettings:
        f = self.view.filter
        caption = ""
        if f.kind != "none":
            caption = (
                f"grey: unfiltered   colour: {describe(f)}"
                if self.show_under
                else describe(f)
            )
        return replace(
            self.figure,
            start_s=start,
            stop_s=stop,
            y_unit=self._y_unit(),
            caption=caption,
        )

    def stem(self, label: str, start: float, stop: float) -> str:
        """``<file>_<unit>_<kind>_<window>``, filename-safe."""
        munit = str(self.unit).rsplit("/", 1)[-1]
        window = label if label != "view" else f"{start:.3g}-{stop:.3g}s"
        return re.sub(
            r"[^\w.\-]+", "_", f"{self.path.stem}_{munit}_{self.view.kind}_{window}"
        )

    def exports(self, spans) -> list[tuple]:
        """``(traces, settings, stem, info)`` of each ``(label, start, stop)``
        span, taken on the draw thread so the worker never reads state the
        window is changing.
        """
        jobs = []
        for label, start, stop in spans:
            settings = self._settings(start, stop)
            info = {
                "source": str(self.path),
                "unit": self.unit,
                "channel": self.channel,
                "window": {"name": label, "start_s": start, "stop_s": stop},
                "view": asdict(self.view),
                "unfiltered_in_grey": self.show_under,
                "rows": [r.record() for r in self._picked_rows()],
            }
            jobs.append(
                (
                    self.figure_traces(settings.theme),
                    settings,
                    Path(self.folder) / self.stem(label, start, stop),
                    info,
                )
            )
        return jobs

    def open_folder(self) -> None:
        folder = Path(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        opener = {"win32": "explorer", "darwin": "open"}.get(sys.platform, "xdg-open")
        subprocess.Popen([opener, str(folder)])

    def _draw_preview(self) -> None:
        if not self._show_preview:
            return
        imgui.set_next_window_size(imgui.ImVec2(900, 600), imgui.Cond_.first_use_ever)
        expanded, self._show_preview = imgui.begin("Figure preview", self._show_preview)
        if expanded:
            if self._preview is None:
                imgui.text_disabled("rendering...")
            else:
                h, w = self._preview.shape[:2]
                scale = min(imgui.get_content_region_avail().x / w, 1.0)
                immvision.image_display(
                    "##preview",
                    self._preview,
                    (int(w * scale), int(h * scale)),
                    refresh_image=self._preview_new,
                )
                self._preview_new = False
        imgui.end()

    def runner_params(self) -> hello_imgui.RunnerParams:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = TITLE
        params.app_window_params.window_geometry.size = WINDOW_SIZE
        params.app_window_params.window_geometry.size_auto = False
        params.app_window_params.resizable = True
        params.ini_filename = _setup.get_default_ini_path("dashboard")
        params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.provide_full_screen_window
        )
        params.imgui_window_params.tweaked_theme.theme = (
            hello_imgui.ImGuiTheme_.darcula_darker
        )
        params.callbacks.show_gui = self.draw
        ## NOTE: a texture outliving the GL context segfaults when it is collected
        params.callbacks.before_exit = self._morph_textures.clear
        return params

    def run(self, frames: int | None = None, screenshot=None) -> None:
        """Run the window until it is closed (or ``frames`` frames, for a
        smoke test; ``screenshot`` then saves the last frame).
        """
        self.max_frames = frames
        immvision.use_rgb_color_order()
        addons = immapp.AddOnsParams()
        addons.with_implot = True
        try:
            immapp.run(runner_params=self.runner_params(), add_ons_params=addons)
            if screenshot is not None:
                import imageio.v3 as iio

                iio.imwrite(
                    str(screenshot),
                    np.asarray(hello_imgui.final_app_window_screenshot()),
                )
        finally:
            self._loader.shutdown(wait=False, cancel_futures=True)
            self._worker.shutdown(wait=False, cancel_futures=True)
            self._morpher.shutdown(wait=False, cancel_futures=True)


def open_dashboard(
    path=None,
    unit: str | None = None,
    channel: int = 0,
    frames: int | None = None,
    screenshot=None,
) -> DashboardApp:
    """Run the dashboard on ``path`` (a ``.mesc``; None opens empty), on
    ``unit`` when given, until its window is closed.
    """
    app = DashboardApp(path, unit=unit, channel=channel)
    app.run(frames=frames, screenshot=screenshot)
    return app


def launch_dashboard(path=None, unit: str | None = None) -> tuple[int, str]:
    """Open the dashboard in its own process; returns its pid and the log
    name :class:`~mbo_utilities.gui.launch.LaunchedWindow` watches.
    """
    args = [] if path is None else [str(path)]
    if unit:
        args += ["--unit", str(unit)]
    log_name = f"dashboard_{Path(str(path)).stem if path else 'empty'}"
    return launch_window("mbo_utilities.gui.dashboard_app", args, log_name), log_name


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", type=Path, help="a .mesc file")
    ap.add_argument(
        "--unit",
        default=None,
        help="the unit to open (MUnit_28 or MSession_0/MUnit_28)",
    )
    ap.add_argument(
        "--channel", type=int, default=0, help="the channel ROIs are read from"
    )
    ap.add_argument(
        "--frames", type=int, default=None, help="exit after this many frames"
    )
    ap.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="with --frames: save the last frame",
    )
    args = ap.parse_args(argv)
    open_dashboard(args.path, args.unit, args.channel, args.frames, args.screenshot)


if __name__ == "__main__":
    main()
