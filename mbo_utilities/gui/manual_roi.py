"""Manual ROI drawing, labeling and traces, laid out and keyed like masknmf's curation viewer.

``ManualRoiWidget`` is one view of a :class:`~mbo_utilities.annotation.RoiModel`
(the ``RoiLabelStore``, the ``RoiTraceTable`` and the slider position) over
a ``MboNDViewer``. This module is the glue: construction, the slider and
playhead wiring, ``add_roi``, loading runs as derived sets, promotion and
the rois zarr persistence. Everything else is a mixin in ``gui/roi/`` over
the widget's own attributes; ``gui/roi/_shared`` holds the constants and
keybinds. AGENTS.md §7.6 is the contract.
"""

from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from masknmf.visualization.imgui import (
    RoiOrder,
    TracePlot,
)
from masknmf.visualization.rois import SELECTED_ALPHA

from mbo_utilities import log
from mbo_utilities.annotation import (
    ENGINES,
    UNLABELED,
    DffSettings,
    RoiFile,
    RoiLabelStore,
    RoiModel,
    RoiTrace,
    RoiTraceTable,
)
from mbo_utilities.arrays.features._dim_tags import (
    filename_tags,
)
from mbo_utilities.behavior import behavior_for
from mbo_utilities.gui._top_strip import TopPanel, TopStrip
from mbo_utilities.gui.imgui import (
    LabelSet,
    SummaryImageViewer,
)
from mbo_utilities.gui.imgui.behavior import BehaviorPlot
from mbo_utilities.gui.imgui.motion import MotionPlot
from mbo_utilities.gui.playhead import Playhead, TimeAxis
from mbo_utilities.gui.roi._shared import (
    BEHAVIOR_PLOT_HEIGHT,
    COLOR_BY,
    COLORMAPS,
    CONTOUR_SHAPES,
    DEFAULT_LABEL_NAMES,
    ENGINE_HELP,
    KEYBINDS,
    MIN_ROI_PIXELS,
    MOTION_PANEL_HEIGHT,
    PANEL_HEIGHT,
    POSITION_KEYS,
    ROI_KEYS,
    TRACE_COLUMNS,
    _PlaneOrder,
    help_markdown,
    roi_widgets_available,
    unit_key,
)
from mbo_utilities.gui.roi.curation import CurationMixin
from mbo_utilities.gui.roi.jobs import JobsMixin
from mbo_utilities.gui.roi.keys import KeysMixin
from mbo_utilities.gui.roi.overlay import OverlayMixin
from mbo_utilities.gui.roi.selection import SelectionMixin
from mbo_utilities.gui.roi.trace_table import TraceTableMixin
from mbo_utilities.gui.roi.traces_panel import TracesPanelMixin
from mbo_utilities.gui.roi_runs import (
    DerivedSet,
    RoiRunManager,
    run_dir_complete,
    set_color,
)
from mbo_utilities.gui.slice import Slice, viewer_positions, viewer_roles
from mbo_utilities.lazy_array import base_array
from mbo_utilities.preferences import (
    get_roi_autosave,
    get_roi_folder,
    set_roi_autosave,
    set_roi_folder,
)
from mbo_utilities.results import (
    Results,
    results_pipeline,
)
from mbo_utilities.roi_workflow import (
    RUN_PIPELINE,
    detection_algo,
    labels_path,
    roi_folder,
)

__all__ = [
    "BEHAVIOR_PLOT_HEIGHT",
    "COLOR_BY",
    "COLORMAPS",
    "ENGINE_HELP",
    "KEYBINDS",
    "MOTION_PANEL_HEIGHT",
    "PANEL_HEIGHT",
    "ROI_KEYS",
    "TRACE_COLUMNS",
    "ManualRoiWidget",
    "attach_roi_widget",
    "detach_roi_widget",
    "help_markdown",
    "labels_path",
    "roi_widgets_available",
    "unit_key",
]


class ManualRoiWidget(
    KeysMixin,
    OverlayMixin,
    SelectionMixin,
    JobsMixin,
    TracesPanelMixin,
    TraceTableMixin,
    CurationMixin,
):
    """Freehand ROI painting, labeling and run curation on a ``MboNDViewer``'s first subplot.

    ROI work is saved to ``<stem>.rois.zarr`` in ``fpath``'s ROI folder on
    Save (or as it changes with autosave on) and restored on construction.
    ``store`` and ``runs`` (:meth:`park_runs`) carry the previous widget's
    state across an off/on toggle; ``strip`` is the figure's shared top
    strip, ``host`` the ``PreviewDataWidget`` whose display settings the
    trace plot follows; both are owned here when omitted. ``auto_trace``
    traces every ROI the moment it is drawn.
    """

    def __init__(
        self,
        iw,
        fpath=None,
        label_names=(),
        store=None,
        runs=None,
        strip=None,
        host=None,
        auto_trace=True,
    ):
        self.iw = iw
        self.host = host
        self.figure = iw.figure
        self.fpath = Path(fpath) if fpath is not None else None
        self.root = (
            Path(get_roi_folder()) if get_roi_folder() else None
        )  # None saves beside the data
        self.autosave = get_roi_autosave()
        self._folder_dialog = None
        self._file_dialog: tuple[str, Any] | None = (
            None  # (exchange kind, the pfd dialog)
        )
        self.unit_key = unit_key(
            iw
        )  # a .mesc keeps ROIs, runs and traces per recording
        self.tag = self.unit_key.strip("/").replace("/", "_")
        self.unit = self.unit_key.rsplit("/", 1)[-1]
        self.logger = log.get("gui.manual_roi")
        self.focus_tab = False

        self.subplot = iw.figure[0, 0]
        self.image = iw.graphics[0]
        self.ny, self.nx = self.image.data.value.shape[:2]

        self.roles = viewer_roles(iw)
        self.tdim, self.cdim, self.zdim = (self._axis_for(r) for r in ("t", "c", "z"))
        self._bind_recording()
        # every scrolling dim but time keys its own mask plane; z last, so a z-only store keeps plane == z
        axes = []
        for name in iw.dim_names:
            if name == self.tdim:
                continue
            rr = iw.ndwidget.indices.ref_ranges.get(name)
            n = max(int(rr.stop - rr.start), 1) if rr is not None else 1
            if n > 1:
                axes.append((name, n))
        axes.sort(key=lambda a: a[0] == self.zdim)
        self.plane_axes: tuple[tuple[str, int], ...] = tuple(axes)
        nz = int(np.prod([n for _, n in axes])) if axes else 1

        self._adopted_store = store is not None and (store.nz, store.ny, store.nx) == (
            nz,
            self.ny,
            self.nx,
        )
        if not self._adopted_store:
            store = RoiLabelStore(nz, self.ny, self.nx, min_pixels=MIN_ROI_PIXELS)
        store.plane_axes = self.plane_axes
        store.axis_roles = self.axis_roles
        for name in label_names:
            store.add_label_name(name)
        self.model = RoiModel(store)  # the Process tab reads the same object

        self.selected = -1
        self.selected_derived: tuple[int, int] | None = None
        self._windows: list = []  # spike-average windows the Traces panel started
        self.buffer: list[
            tuple[int, int]
        ] = []  # the group: (si, k), si -1 for a drawn ROI
        self.pixels: OrderedDict[tuple, tuple] = (
            OrderedDict()
        )  # (store plane, row, col) -> trace key, newest first
        self.pixel_group: list[tuple] = []
        self.active_pixel: tuple | None = None
        self.pixel_traces = False
        self.status = "draw (a) a region, add it as an ROI (r)"
        self._save_error: str | None = None
        self._run_error: str | None = None
        self._save_suffix = (
            ""  # "<nz>x<ny>x<nx>" when the saved rois zarr is for another shape
        )
        self.new_label = ""
        self._note_buf = ""
        self.scroll_to_selection = False
        self.follow = False
        self.keybinds_open = False
        self._undo: list[dict] = []  # ctrl+z steps, newest last (_push_undo)
        self._panel_hovered = -2

        self.show_masks = True
        self.opacity = 0.5
        self.masks_by_peak = True
        self.show_selected_masks = True
        self.selected_opacity = SELECTED_ALPHA
        self.show_contours = False
        self.contour_opacity = 0.9
        self.show_selected_contours = True
        self.selected_contour_opacity = 0.7
        self.contour_shape = CONTOUR_SHAPES[0]
        self.traces_this_slice = True

        self._feathers: dict[int, tuple] = {}
        self.rows: list[tuple[int, int]] = []
        self._row_index: dict[tuple[int, int], int] = {}
        self._promoted: dict[tuple[str, int], int] = {}
        self.derived: list[DerivedSet] = []
        self.classes = LabelSet(0, self.store.label_names)
        self.order = _PlaneOrder({"del": np.zeros(0, np.int8)}, self.classes.labels, 0)
        self._footprints: tuple | None = None  # (key, rows, FootprintSet) as drawn last
        self.filter_select = (
            False  # apply: the rows on the switch's side of the range are the group
        )
        self.filter_outside = True

        # the host's playhead and slice when it has them; a widget on a bare viewer feeds its own
        self.playhead = getattr(host, "playhead", None)
        self.slice = getattr(host, "slice", None)
        self._own_slice = self.slice is None
        if self.playhead is None:
            self.playhead = Playhead()
            if host is not None:
                host.playhead = self.playhead
        if self.slice is None:
            self.slice = Slice()
            self.slice.move(*viewer_positions(iw), source=self)
            if host is not None:
                host.slice = self.slice
            iw.ndwidget.indices.add_event_handler(self._on_indices)
        self.model.set_view(self._view_pos())
        self.slice.add_event_handler(self._on_slice, "slice")

        self.overlay = self.subplot.add_image(
            np.zeros((self.ny, self.nx, 4), np.uint8),
            name="manual_roi_overlay",
            alpha_mode="blend",
            offset=(0, 0, 1),
        )
        # NOTE: literal RGBA bytes; auto-ranging off the all-zero start saturates every colour to white
        self.overlay.vmin, self.overlay.vmax = 0, 255
        for tile in self.overlay.world_object.children:
            tile.material.pick_write = False
        # every ROI's contour in one buffer split by NaN rows; screen-space thickness keeps a hairline at any zoom
        self.outline = self.subplot.add_line(
            np.zeros((5, 3), np.float32),
            colors=np.zeros((5, 4), np.float32),
            thickness=1.0,
            size_space="screen",
            name="manual_roi_outline",
            offset=(0, 0, 1.25),
            visible=False,
        )
        self.outline.world_object.material.pick_write = False

        self.region_selector = (
            None  # the drawn polygon (a) until Add ROI (r) keeps it or esc drops it
        )
        self.region_outside = False
        self._region_key = None
        self._region_hits: list | None = None
        self._press: tuple[float, float] | None = (
            None  # a click travelling further is a pan
        )
        self._press_drawn = False
        renderer = self.subplot.renderer
        renderer.add_event_handler(self._pointer_down, "pointer_down")
        renderer.add_event_handler(self._pointer_up, "pointer_up")
        self.summary = SummaryImageViewer(iw.figure, title="Full FOV")

        self.trace_uid = 0  # the last ROI a trace landed for
        self._trace_results: queue.Queue = queue.Queue()
        self._trace_threads: list[threading.Thread] = []
        self.trace_sel: set[tuple] = (
            set()
        )  # the highlighted trace rows: the selection's, or picked in the table
        self.trace_picked = False
        self._trace_stats: dict[tuple, tuple] = {}
        self._trace_display: dict[tuple, np.ndarray] = {}
        self._trace_deflection = (False, False)
        self._trace_window_cache: dict[tuple, np.ndarray] = {}
        self._kind: str | None = (
            None  # a DISPLAY_KINDS name; None lets every row's pipeline pick
        )
        self._flip = False
        self.dff: DffSettings | None = None  # None keeps each pipeline's
        self.trace_subtract = "viewer"  # DEFLECTION_MODES
        self.trace_invert = "viewer"
        self.show_vnoiser = (
            True  # the vnoiser run of the recording on screen, a panel under the traces
        )
        self._vnoiser_kind: str | None = None
        self._vnoiser_lines_key = None
        self._trace_unit: tuple = (None, False)
        self.trace_order = RoiOrder({}, 0)
        self._trace_keys: list[tuple] = []
        self.show_traces = (
            True  # off, selecting only highlights (masknmf's show selected traces)
        )
        self.show_motion = True
        self.show_behavior = True
        self.trace_plot: TracePlot | None = None
        self._motion_panel = False
        self._plot_frames: tuple | None = None
        self._plot_lines_key = None
        self._motion_key = None
        self.pending_traces = (
            None  # drawn in place of "no traces" while a host computes them
        )
        self._fs_value: float | None = None
        self._fs_read = False
        self.color_by = COLOR_BY[0]
        self.color_cmap = COLORMAPS[0]

        self.engine = ENGINES[0]
        self.run_where = "drawn"
        self.run_z: int | None = None
        self.run_c: int | None = None
        self.run_tp: list[int] | None = (
            None  # 0-based frames from parse_timepoint_selection; None every frame
        )
        self.auto_trace = bool(auto_trace)
        self.run_tag = ""  # empty shows DEFAULT_RUN_TAG as the box's hint
        self.manager = RoiRunManager()
        self._registry_extra: list[
            dict
        ] = []  # loaded runs no derived set stands for, saved with the ROIs
        self._saved_runs: list[dict] = []
        self._adopted: set[int] = set()  # pids _adopt_finished_runs already pulled in
        self._adopt_checked = 0.0
        self._results_loaded: set[str] = set()
        self._restoring = False

        # the shared top edge runs the per-frame hook whatever is up; a bare viewer gets its own strip
        self._own_strip = strip is None
        self.strip = TopStrip(iw.figure) if self._own_strip else strip
        self.strip.add_hook(self._frame)
        self._traces_panel = TopPanel(
            "traces", "Traces", self.draw_traces, PANEL_HEIGHT, 11
        )
        self.strip.register(self._traces_panel)

        self._closed = False
        self._restore()
        self._restore_runs(runs)
        self.model.add_event_handler(self._on_rois, "rois")
        self.model.add_event_handler(self._on_traces, "traces")
        self.model.add_event_handler(self._on_view, "view")
        self.playhead.add_event_handler(self._on_playhead, "time")
        self._resync()
        self.refresh_overlay()

    def _bind_recording(self) -> None:
        """The recording's facets for the Traces tab: motion correction, behavior log, line positions (an AOD unit's, by ROI index) and results."""
        # the viewer may hold a bare ndarray, which has none of the LazyArray facets
        arr = base_array(self.iw.data[0]) if self.iw.data else None
        self.motion = MotionPlot(getattr(arr, "motion_correction", None))
        self.behavior = BehaviorPlot(behavior_for(arr) if arr is not None else None)
        self.line_positions = list(getattr(arr, "line_positions", None) or [])
        self.results = getattr(arr, "results", None)

    def _line_position(self, trace: RoiTrace) -> dict:
        """The position of the line a row was read on (its ``line`` in ``extra``, else its z), under the row's own record; empty without geometry."""
        line = trace.extra.get("line")
        if line is None and trace.stands_for_roi:
            line = trace.z
        pos = (
            dict(self.line_positions[int(line)])
            if line is not None and 0 <= int(line) < len(self.line_positions)
            else {}
        )
        pos.update(
            {
                k: v
                for k, v in trace.extra.items()
                if k in POSITION_KEYS and v is not None
            }
        )
        return pos

    def close(self):
        """Take everything back off the figure: panel, overlays, handlers."""
        if self._closed:
            return
        self._closed = True
        self._drop_region()
        renderer = self.subplot.renderer
        renderer.remove_event_handler(self._pointer_down, "pointer_down")
        renderer.remove_event_handler(self._pointer_up, "pointer_up")
        if self._own_slice:
            self.iw.ndwidget.indices.remove_event_handler(self._on_indices)
        self.slice.remove_event_handler(self._on_slice)
        self.playhead.remove_event_handler(self._on_playhead)
        self.model.close()  # the parked store and table must not call into a closed widget
        for graphic in (self.overlay, self.outline):
            self.subplot.delete_graphic(graphic)
        self.summary.close()
        self.summary.cleanup()
        self.strip.remove_hook(self._frame)
        self.strip.unregister("traces")
        if self._own_strip:
            self.strip.close()
        self.strip = None

    def park_runs(self) -> dict:
        """The state the next attach adopts: the live run manager, loaded sets, registry leftovers and the trace table."""
        return {
            "manager": self.manager,
            "derived": self.derived,
            "extra": self._registry_extra,
            "traces": self.traces,
        }

    @property
    def store(self) -> RoiLabelStore:
        return self.model.store

    @store.setter
    def store(self, store: RoiLabelStore) -> None:
        self.model.store = store

    @property
    def traces(self) -> RoiTraceTable:
        return self.model.traces

    @property
    def z(self) -> int:
        """The flat store plane on screen (``model.plane``)."""
        return self.model.plane

    def _axis_for(self, role: str) -> str | None:
        """The viewer's slider name for the array axis ``role`` (t, c, z)."""
        return next((name for name, r in self.roles.items() if r == role), None)

    def axis_label(self, role: str) -> str:
        """The ``role`` axis's name on this data: the slider's own when it is a word (``ROI`` on an AOD unit, ``Zplane``, ``View``), else the letter."""
        name = self._axis_for(role)
        return name if name is not None and len(name) > 1 else role

    def _axis_tag(self, role: str, index: int) -> str:
        """``z3`` / ``ROI 3``: one 1-based position on a named axis."""
        label = self.axis_label(role)
        return f"{label} {index + 1}" if len(label) > 1 else f"{label}{index + 1}"

    @property
    def axis_roles(self) -> dict[str, str]:
        """``{"z": <slider name>, "c": <slider name>}`` for the store."""
        return {
            role: name
            for role, name in (("z", self.zdim), ("c", self.cdim))
            if name is not None
        }

    def _view_pos(self) -> dict[str, int]:
        """The slider position over the dims that key mask planes; a dim a unit switch dropped sits at 0."""
        return {name: self.slice.positions.get(name, 0) for name, _n in self.plane_axes}

    def _plane_pos(self, plane: int) -> dict[str, int]:
        return self.store.plane_pos(plane)

    def _goto_plane(self, plane: int):
        """Move every scroll slider so the viewer shows ``plane``."""
        for name, v in self._plane_pos(plane).items():
            if int(self.iw.indices[name]) != v:
                self.iw.indices[name] = v

    def _plane_label(self, plane: int) -> str:
        return self.store.plane_label(plane)

    def _on_indices(self, _indices):
        """A widget on a bare viewer feeds its own slice and playhead."""
        self.slice.move(*viewer_positions(self.iw), source=self)
        if self.tdim is not None:
            self.playhead.seek(
                self.viewer_axis().seconds(self.current_frame()), source=self
            )

    def _on_slice(self, _event):
        """The sliders sit on another channel or z-plane: the store plane follows."""
        self.model.set_view(self._view_pos())

    def _on_playhead(self, event):
        """Another view moved the playhead: put the viewer's T on that frame."""
        if event.info.get("source") is self or self.tdim is None:
            return
        frame = int(round(self.viewer_axis().units(event.info["seconds"])))
        if frame != self.current_frame():
            self.set_frame(frame)

    def viewer_axis(self) -> TimeAxis:
        """The viewer's T slider on the clock: frames at the host's binning."""
        return TimeAxis.sampled(self.fs(), getattr(self.host, "frame_average", 1) or 1)

    def trace_axis(self, trace: RoiTrace) -> TimeAxis:
        """One row's samples on the clock: its own rate (a results file's
        scan) else the movie's, at its binning, from its frame window.
        """
        first, step = (
            (trace.frames[0], trace.frames[2]) if trace.frames is not None else (0, 1)
        )
        return TimeAxis.sampled(
            trace.fs or self.fs(), trace.frame_average * step, first
        )

    def _on_view(self, _event):
        """The sliders landed on another plane: drop a half-drawn region,
        refilter the table and redraw the overlay.
        """
        if self.region_selector is not None:
            self._drop_region()
        if self.order.plane is not None:
            self.order.plane = self.z
            self.order.rebuild()
        self._motion_key = None
        self.refresh_overlay()

    def _on_rois(self, event):
        """A store mutation: rows, overlay and the autosave follow it. A
        vanished ROI takes its traces and its feather cache with it.
        """
        action = event.info.get("action")
        if action == "tint":
            self.refresh_overlay()
            return
        if action in ("add", "delete", "clear", "restore"):
            live = {r.uid for r in self.store.rois}
            self.traces.prune(live)
            self._feathers = {u: v for u, v in self._feathers.items() if u in live}
        if action != "note":
            self._resync()
        if self.color_by not in (COLOR_BY[0], "peak") and action in (
            "add",
            "delete",
            "clear",
            "restore",
            "class",
        ):
            self.apply_color_by()
        self.refresh_overlay()
        if action != "note":  # a note saves when its box loses focus
            self._autosave()

    def _on_traces(self, _event):
        self._traces_changed()
        if self.color_by == "peak":
            self.apply_color_by()

    def current_frame(self) -> int:
        """The viewer's T index; 0 without a T slider, or while a unit switch has dropped it."""
        if self.tdim is None or self.tdim not in self.iw.dim_names:
            return 0
        return int(self.iw.indices[self.tdim])

    def set_frame(self, frame: int):
        """Move the viewer's t; the trace cursor drag scrubs the movie with this."""
        if self.tdim is None or self.tdim not in self.iw.dim_names:
            return
        movie = self.movie()
        limit = (int(movie.shape[0]) - 1) if movie is not None else 0
        self.iw.indices[self.tdim] = int(np.clip(frame, 0, limit))

    @property
    def labels(self) -> np.ndarray:
        """The plane on screen's label image: ROI ``i`` paints ``i + 1``."""
        return self.store.label_image(self.z)

    @property
    def counts(self) -> list[int]:
        return self.store.counts

    @property
    def n_rois(self) -> int:
        return len(self.store.rois)

    @property
    def drawing(self) -> bool:
        """A region is armed and its polygon is still being placed."""
        return (
            self.region_selector is not None
            and self.region_selector._move_info.mode == "create"
        )

    def _resync(self):
        """Rebuild the rows, label set and table order from the store and the derived sets; drawn rows first, so table ids match store indices."""
        rois = self.store.rois
        self._promoted = {}
        for i, r in enumerate(rois):
            name, _, row = r.source.rpartition(":")
            if name and row.isdigit():
                self._promoted[(name, int(row))] = i
        self.rows = [(-1, i) for i in range(len(rois))]
        planes = [r.plane for r in rois]
        sources = [0] * len(rois)
        areas = [r.area for r in rois]
        oks = [1] * len(rois)
        probs = [np.nan] * len(rois)
        marks = [0] * len(rois)
        for si, s in enumerate(self.derived):
            if not s.visible:
                continue
            for k, (ypix, _xpix, _lam) in enumerate(s.footprints):
                self.rows.append((si, k))
                planes.append(s.z)
                sources.append(1 + si)
                areas.append(int(ypix.size))
                oks.append(1 if s.accepted[k] else 0)
                probs.append(s.prob(k))
                marks.append(int(k in s.discarded))
        self._row_index = {pair: row for row, pair in enumerate(self.rows)}
        if (
            self.selected_derived is not None
            and self.selected_derived not in self._row_index
        ):
            self.selected_derived = None
        self.buffer = [pair for pair in self.buffer if pair in self._row_index]
        labels = np.full(len(self.rows), UNLABELED, np.int64)
        labels[: len(rois)] = [r.class_index for r in rois]
        for row in range(len(rois), len(self.rows)):
            si, k = self.rows[row]
            labels[row] = self.derived[si].classes.get(k, UNLABELED)
        self.classes = LabelSet(len(self.rows), self.store.label_names, labels)
        self.store.label_names = self.classes.names
        planes = np.asarray(planes, np.int64)
        columns = {  # in the filter combo's order
            "label": self.classes.labels,
            "source": np.asarray(sources, np.int64),
            "area": np.asarray(areas, np.int64),
            "peak": self._row_peaks(),
            "ok": np.asarray(oks, np.int64),
        }
        if self.has_prob:
            columns["prob"] = np.asarray(probs, np.float64)
        if self.store.nz > 1:
            columns["z"] = planes
        columns["del"] = np.asarray(marks, np.int8)
        order = self.order
        order.columns = columns
        order.labels = self.classes.labels
        order.n_items = len(self.rows)
        order.planes = planes
        order.sources = columns["source"]
        if order.source is not None and not 0 <= order.source <= len(self.derived):
            order.source = None
        if order.range_column not in columns:
            order.set_range_column("area")
        else:
            # the span follows the rows; limits the user narrowed stay inside it
            span, limits = order.range_span, order.range_limits
            order.set_range_column(order.range_column)
            lo, hi = order.range_span
            if tuple(limits) != tuple(span) and max(limits[0], lo) <= min(
                limits[1], hi
            ):
                order.range_limits = (max(limits[0], lo), min(limits[1], hi))
        order.rebuild()
        self._footprints = None

    def _sync_store_from_classes(self):
        self.store.label_names = tuple(self.classes.names)
        for record, ci in zip(self.store.rois, self.classes.labels):
            record.class_index = int(ci)
        for row in range(self.n_rois, len(self.rows)):
            si, k = self.rows[row]
            ci = int(self.classes.labels[row])
            if ci == UNLABELED:
                self.derived[si].classes.pop(k, None)
            else:
                self.derived[si].classes[k] = ci

    @property
    def has_prob(self) -> bool:
        """Whether a loaded set carries a classifier probability (suite2p's ``iscell[:, 1]``)."""
        return any(s.has_prob for s in self.derived)

    @property
    def columns(self) -> tuple[str, ...]:
        """The ROIs table's columns: the id, then the order's, ``del`` last as in masknmf."""
        return ("id", *self.order.columns)

    def set_drawing(self, on: bool):
        """Arm a region (the polygon's first click lands on the image), or drop it."""
        if on and self.region_selector is None:
            self._start_region()
        elif not on and self.region_selector is not None:
            self._drop_region()

    def add_roi(self, points):
        """Fill a closed polygon (``(x, y)`` points on the image) and store it as the next label on the plane on screen."""
        if len(points) < 3:
            self.status = "a region needs three points"
            return
        drawn = np.asarray(points, np.float32)
        points = np.round(drawn).astype(np.int32)
        points[:, 0] = points[:, 0].clip(0, self.nx - 1)
        points[:, 1] = points[:, 1].clip(0, self.ny - 1)
        filled = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(filled, [points], 1)
        self._push_undo()
        index = self.store.add_roi(self.z, filled.astype(bool), polygon=drawn)
        if index is None:
            self._undo.pop()
            self.status = f"under {MIN_ROI_PIXELS} px, not added"
            return
        self.select_roi(index)
        if self.auto_trace and self.trace_disabled(index) is None:
            self.quick_trace(index)

    def _set_name(self, path: Path) -> str:
        """Display name for a run dir; a per-slice child (a name made of
        filename tags, ``ch01_zplane02``) keeps the run dir it belongs to.
        """
        path = Path(path)
        name = path.name
        tags = {t.definition.label for t in filename_tags(name)}
        child = bool(tags & {"zplane", "ch"}) or (
            name[:1] == "z" and name[1:].isdigit()
        )
        if child or path.parent.suffix == ".zarr":
            name = f"{path.parent.name}/{name}"
        while any(s.name == name for s in self.derived):
            name += "~"
        return name

    def _add_derived(
        self,
        unit,
        path,
        pipeline: str = "",
        *,
        z: int | None = None,
        algo: str = "",
        discarded=(),
        classes=None,
        colors=None,
        iscell=None,
    ) -> DerivedSet | None:
        """Wrap one pixel unit as a derived set keyed by ``path`` (replacing an earlier load of it) and merge its rows' traces.

        ``z`` None takes the unit's ``attrs["z"]``, else its plane number;
        ``iscell`` overrides the unit's own accept / reject flags.
        """
        path = Path(path)
        shape = (
            None
            if unit.image_shape is None
            else tuple(int(v) for v in unit.image_shape)
        )
        if unit.member_kind != "pixel" or shape is None:
            self._run_error = f"{path.name} has no pixel ROIs to show"
            return None
        if shape != (self.ny, self.nx):
            self._run_error = (
                f"{path.name} is {shape[0]}x{shape[1]}, data is {self.ny}x{self.nx}"
            )
            return None
        if z is None:
            z = unit.attrs.get("z")
        if z is None:
            z = int(unit.index) - 1 if unit.kind == "plane" else 0
        if not 0 <= int(z) < self.store.nz:
            self._run_error = (
                f"{path.name} is plane {int(z) + 1}, data has {self.store.nz} plane(s)"
            )
            return None
        promoted_traces: list[RoiTrace] = []
        classes = dict(classes or {})
        colors = dict(colors or {})
        iscell = dict(iscell or {})
        for si, old in enumerate(self.derived):
            if old.path == path:
                discarded = set(discarded) | old.discarded
                classes = {**old.classes, **classes}
                colors = {**old.colors, **colors}
                promoted_traces = [  # rows promoted out of the old load stay with their drawn rois
                    t for t in self.traces.from_source(old.name) if t.stands_for_roi
                ]
                self.unload_set(si)
                break
        kind = str(pipeline or unit.attrs.get("pipeline") or "suite2p")
        s = DerivedSet(
            unit=unit,
            path=path,
            name=self._set_name(path),
            color=set_color(len(self.derived)),
            z=int(z),
            kind=kind,
            algo=str(algo),
            discarded={int(k) for k in discarded},
            classes={int(k): int(v) for k, v in classes.items()},
            colors={int(k): tuple(float(x) for x in v) for k, v in colors.items()},
        )
        for k, on in iscell.items():
            if 0 <= int(k) < len(s):
                s.accepted[int(k)] = bool(on)
        self.derived.append(s)
        if self.store.nz > 1 and len(self.derived) == 1 and self.order.plane is None:
            self.order.plane = (
                self.z
            )  # per-plane results open the ROI table on the plane on screen
        self._merge_run_traces(unit, s.name, path)
        for k in range(len(s)):
            if k not in s.discarded and (s.name, k) not in self._promoted:
                trace = self._member_trace(s, k)
                if trace is not None:
                    self.traces.add(trace)
        for trace in promoted_traces:
            self.traces.add(trace)
        self._resync()
        self.refresh_overlay()
        self._autosave()
        return s

    def _member_trace(self, s: DerivedSet, k: int) -> RoiTrace | None:
        """Component ``k`` of a loaded set as a Traces-tab row, read where
        its plane sits unless the run recorded where it read; None when the
        run carries no traces.
        """
        pos = self._plane_pos(s.z)
        zname, cname = self.store.axis_name("z"), self.store.axis_name("c")
        return s.trace(
            k,
            z=pos.get(zname, 0),
            c=pos.get(cname, 0) if cname else self._channel(),
        )

    def load_run(
        self, path, discarded=(), classes=None, colors=None, iscell=None
    ) -> bool:
        """Read one run (a results file, one unit of one, a plane dir, a masknmf run folder or a vnoiser PF folder) through ``Results.open``.

        A pixel unit loads as a derived set, every row included; a manual
        run's unit whose rows stand for drawn ROIs still in the store merges
        into the trace table by uid. A line unit goes to the Traces tab, one
        row per ROI and one per member line; with a MESc unit on screen only
        the scan that processed it loads, without the member lines. Returns
        True when anything loaded.
        """
        from mbo_utilities.arrays.mesc import MescArray
        from mbo_utilities.results import unit_for_source

        path = Path(path)
        if (
            path.parent.suffix == ".zarr"
            and results_pipeline(path) is None
            and results_pipeline(path.parent) is not None
        ):
            file, only = path.parent, path.name
        else:
            file, only = path, None
        try:
            results = Results.open(file)
        except Exception as e:  # noqa: BLE001 - shown in the status row
            self._run_error = f"could not load {path.name}: {e}"
            return False
        src = base_array(self.iw.data[0]) if self.iw.data else None
        on_screen = (
            unit_for_source(results, src.unit_key)
            if isinstance(src, MescArray)
            else None
        )
        own = self.fpath is not None and roi_folder(self.fpath) in (file, file.parent)
        algo = (
            detection_algo(results.metadata)
            if results.metadata and results.pipeline in ("suite2p", "masknmf")
            else results.pipeline
        )
        loaded = 0
        lines = 0
        for unit in results.units.values():
            if only is not None and unit.name != only:
                continue
            if unit.member_kind == "line" and isinstance(src, MescArray):
                if unit.name != on_screen:
                    continue
                rows = self._line_unit_rows(results, unit, file, with_members=False)
                if rows:
                    self.traces.drop_source(f"{file.name}/{unit.name}")
                    for trace in rows:
                        self.traces.add(trace)
                    loaded += 1
                    lines += 1
                continue
            if unit.member_kind == "pixel" and unit.image_shape is not None:
                uids = unit.attrs.get("uids")
                if (
                    results.pipeline == RUN_PIPELINE
                    and uids
                    and len(uids) == unit.n_rois
                ):
                    name = self._set_name(file)
                    if self._merge_run_traces(unit, name, file):
                        loaded += 1
                        if not any(
                            str(e["path"]) == str(file) for e in self._registry_extra
                        ):
                            self._registry_extra.append({"path": str(file)})
                    continue
                plane_dir = unit.attrs.get("plane_dir")
                key = (
                    Path(plane_dir)
                    if plane_dir and Path(plane_dir).is_dir()
                    else (
                        file
                        if file == roi_folder(self.fpath or "")
                        else file / unit.name
                    )
                    if results_pipeline(file) is None
                    else file / unit.name
                )
                loaded += (
                    self._add_derived(
                        unit,
                        key,
                        results.pipeline,
                        z=0 if own and self.store.nz == 1 else None,
                        algo=algo,
                        discarded=discarded,
                        classes=classes,
                        colors=colors,
                        iscell=iscell,
                    )
                    is not None
                )
                continue
            name = f"{file.name}/{unit.name}"
            rows = self._line_unit_rows(results, unit, file)
            if not rows:
                continue
            self.traces.drop_source(name)
            for trace in rows:
                self.traces.add(trace)
            loaded += 1
            lines += 1
        if not loaded:
            if not self._run_error:
                self._run_error = (
                    f"{file.name} never processed {src.unit_key}"
                    if isinstance(src, MescArray)
                    else f"{file.name} has no unit this view can show"
                )
            return False
        self._results_loaded.add(str(file))
        # line units stand in the trace table only, so the file is listed for the next session
        if lines and not any(str(e["path"]) == str(file) for e in self._registry_extra):
            self._registry_extra.append({"path": str(file)})
        self._autosave()
        self.status = f"loaded {loaded} unit(s) of {file.name}"
        return True

    load_results = load_run

    def _line_unit_rows(
        self, results, unit, file: Path, with_members: bool = True
    ) -> list[RoiTrace]:
        """One trace-table row per ROI of a line unit (``raw`` as F, ``dff`` as norm, the rest under ``kinds``), then one per member line with ``with_members``."""
        name = f"{file.name}/{unit.name}"
        rows: list[RoiTrace] = []
        n = unit.n_rois
        lines = unit.member_kind == "line"
        channel = int(results.source.get("channel") or 0)
        window = results.source.get(
            "frames"
        )  # a frame-window run's traces sit where they were recorded
        frames = None if not window else (int(window[0]), int(window[1]), 1)
        # the reader's geometry places the lines when the shown recording is the scan these results came from
        src = base_array(self.iw.data[0])
        positions = (
            getattr(src, "line_positions", None)
            if lines
            and unit.attrs.get("source_unit") in (None, getattr(src, "unit_key", None))
            else None
        ) or []
        for k, roi in enumerate(unit.roi_names):
            entry = {  # the run's own names, which the spike-average window opens the ROI by
                "label": str(roi),
                "fs": unit.fs,
                "c": channel,
                "frames": frames,
                "kinds": {},
                "extra": {
                    "unit": unit.name,
                    "roi": str(roi),
                    "source_unit": unit.attrs.get("source_unit"),
                },
            }
            members = unit.members[k] if k < len(unit.members) else ()
            if lines and len(members) == 1:
                entry["z"] = int(members[0])
                entry["extra"]["line"] = int(members[0])
                if int(members[0]) < len(positions):
                    entry["extra"].update(positions[int(members[0])])
            elif lines and len(members):
                entry["z"] = int(
                    members[0]
                )  # a domain of several lines is listed on its first
                entry["extra"]["lines"] = [int(m) for m in members]
            for kind, arr in unit.traces.items():
                row = np.asarray(arr[k], np.float32)
                if kind == "raw":
                    entry["F"] = row
                elif kind == "neuropil":
                    entry["Fneu"] = row
                elif kind == "dff":
                    entry["norm"] = row
                else:
                    entry["kinds"][kind] = row
            rows.append(
                RoiTrace(
                    uid=0,
                    member=k,
                    source=name,
                    engine=results.pipeline,
                    path=file,
                    **entry,
                )
            )
        if not with_members:
            return rows
        ids = list(unit.attrs.get("member_ids") or [])
        for kind, arr in unit.member_traces.items():
            for i, row in enumerate(np.asarray(arr, np.float32)):
                member = ids[i] if i < len(ids) else i
                k = unit.member_roi(member)
                if k is None:
                    label = f"{unit.member_kind} {member} ({kind})"
                elif len(unit.members[k]) == 1:
                    label = f"{unit.roi_names[k]} ({kind})"
                else:
                    label = f"{unit.roi_names[k]} {unit.member_kind} {member} ({kind})"
                where = (
                    {"z": int(member), "extra": {"line": int(member)}} if lines else {}
                )
                if lines and int(member) < len(positions):
                    where["extra"].update(positions[int(member)])
                rows.append(
                    RoiTrace(
                        uid=0,
                        member=n + i,
                        source=name,
                        engine=results.pipeline,
                        path=file,
                        label=label,
                        fs=unit.fs,
                        F=row,
                        c=channel,
                        frames=frames,
                        **where,
                    )
                )
        return rows

    def unload_set(self, si: int):
        s = self.derived.pop(si)
        self.traces.drop_source(s.name)
        self._registry_extra = [
            e for e in self._registry_extra if str(e["path"]) != str(s.path)
        ]
        if self.selected_derived is not None:
            osi, k = self.selected_derived
            if osi == si:
                self.selected_derived = None
            elif osi > si:
                self.selected_derived = (osi - 1, k)
        self.buffer = [
            (osi - (1 if osi > si else 0), k) for osi, k in self.buffer if osi != si
        ]
        if self.order.source is not None and self.order.source == 1 + si:
            self.order.source = None
        elif self.order.source is not None and self.order.source > 1 + si:
            self.order.source -= 1
        self._resync()
        self.refresh_overlay()
        self._autosave()

    def promoted_index(self, si: int, k: int) -> int | None:
        """Store index of the drawn ROI promoted from set ``si`` row ``k``."""
        return self._promoted.get((self.derived[si].name, k))

    def _promote(self, si: int, k: int) -> int | None:
        """Copy one derived component into the store; None, with the reason in the status, when it cannot land."""
        s = self.derived[si]
        if k in s.discarded:
            self.status = f"{s.name} row {k} is discarded"
            return None
        if (s.name, k) in self._promoted:
            self.status = f"{s.name} row {k} is already promoted"
            return None
        if len(self.store.rois) >= 65535:
            self.status = "store is full (65535 labels)"
            return None
        ypix, xpix, lam = s.footprints[k]
        index = self.store.add_roi(
            s.z,
            pixels=ypix.astype(np.int64) * self.nx + xpix,
            weights=lam,
            source=f"{s.name}:{k}",
        )
        if index is None:
            self.status = f"{s.name} row {k} is under {MIN_ROI_PIXELS} px"
            return None
        self._promoted[(s.name, k)] = index
        if k in s.classes:
            self.store.set_class(index, s.classes[k])
        trace = self.traces.remove(("member", s.name, k)) or self._member_trace(
            s, k
        )  # becomes the drawn ROI's
        if trace is not None:
            trace.uid, trace.member = self.store.rois[index].uid, None
            self.traces.add(trace)
        return index

    def promote_derived(self, si: int, k: int) -> int | None:
        """Promote one derived component, select it, then step to the next
        promotable derived row in view.
        """
        index = self._promote(si, k)
        if index is None:
            return None
        self.select_roi(index)
        self.refresh_overlay()
        row = self._row_index.get((si, k))
        start = 0
        if row is not None:
            hits = np.flatnonzero(self.order.order == row)
            if len(hits):
                start = int(hits[0]) + 1
        self._select_next_derived(start, skip_promoted=True)
        return index

    def promote_set(self, si: int):
        """Promote every shown component of one set."""
        s = self.derived[si]
        promoted = skipped = 0
        self.store.block_events(True)
        try:
            for k in range(len(s)):
                if k in s.discarded or (s.name, k) in self._promoted:
                    skipped += 1
                    continue
                if self._promote(si, k) is None:
                    skipped += 1
                else:
                    promoted += 1
        finally:
            self.store.block_events(False)
        self._resync()
        self.refresh_overlay()
        self._autosave()
        self.status = f"{s.name}: promoted {promoted} / skipped {skipped}"

    def set_accepted(self, si: int, k: int, on: bool | None = None):
        """Flip (or set) one derived component's accepted flag. The run is
        never written: the flag is kept with the ROIs (the rois zarr's
        ``runs`` entry for the run) as an override on the run's own
        ``iscell``.
        """
        s = self.derived[si]
        s.accepted[k] = (not s.accepted[k]) if on is None else bool(on)
        self._resync()
        self.refresh_overlay()
        self._autosave()
        state = "accepted" if s.accepted[k] else "rejected"
        self.status = f"{s.name} row {k}: {state}"

    @property
    def folder(self) -> Path | None:
        """The folder this dataset's ROI work is saved in (``roi_folder``)."""
        return None if self.fpath is None else roi_folder(self.fpath, self.root)

    def _save_target(self) -> Path:
        tag = "_".join(t for t in (self.tag, self._save_suffix) if t)
        return labels_path(self.fpath, tag, self.root)

    def _restore(self):
        """Adopt the rois zarr saved for this data (the ROI folder, else beside the data): its ROIs, quick traces and run list.

        One saved for data of another shape is left alone and this data saves
        under a name carrying its shape.
        """
        if (
            self.fpath is None or self._adopted_store
        ):  # the parked store is the in-session truth
            return
        found = None
        for target in dict.fromkeys(
            (self._save_target(), labels_path(self.fpath, self.tag))
        ):
            if target.exists():
                found = target
                break
        if found is None:
            return
        try:
            contents = RoiFile.load(found)
        except (OSError, ValueError) as e:
            self.logger.warning(f"could not restore {found}: {e}")
            self.status = f"restore failed: {e}"
            return
        store = contents.store
        if store.shape != self.store.shape:
            self._save_suffix = "x".join(str(v) for v in self.store.shape)
            own = self._save_target()
            self.logger.warning(
                f"{found} is for {store.shape} data, this is {self.store.shape}; "
                f"left alone, saving to {own.name}"
            )
            if own.exists():
                self._restore()
                return
            self.status = (
                f"{found.name} is for other data; starting fresh as {own.name}"
            )
            return
        for name in self.store.label_names:
            store.add_label_name(name)
        store.min_pixels = MIN_ROI_PIXELS
        store.plane_axes = self.plane_axes
        store.axis_roles = self.axis_roles
        self.store = store
        zname, cname = store.axis_name("z"), store.axis_name("c")
        for trace in contents.traces:
            self.traces.add(trace)
            if trace.source == "pixel" and "pixel" in trace.extra:
                pos = {}
                if zname is not None:
                    pos[zname] = trace.z
                if cname is not None:
                    pos[cname] = trace.c
                row, col = trace.extra["pixel"]
                self.pixels[(store.plane_of(pos), int(row), int(col))] = trace.key
        self._saved_runs = list(contents.runs)
        self.status = f"restored {len(store.rois)} ROIs from {found.name}"
        self.refresh_overlay()

    def _restore_runs(self, parked: dict | None):
        """Adopt the previous widget's parked runs, else re-load every surviving run the rois zarr lists."""
        if parked is not None:
            manager = parked.get("manager")
            if manager is not None:
                self.manager = manager
            if self._adopted_store:
                self.model.traces = parked.get("traces") or RoiTraceTable()
                self._traces_changed()
                self.derived = [
                    s
                    for s in (parked.get("derived") or [])
                    if s.shape == (self.ny, self.nx)
                ]
                self._registry_extra = list(parked.get("extra") or [])
                return
        self._restoring = True  # parked sets of another store key its uids; this data's own runs load instead
        try:
            for entry in self._saved_runs:
                path = Path(entry["path"])
                if run_dir_complete(path) and self.load_run(
                    path,
                    discarded=entry.get("discarded", ()),
                    classes=entry.get("classes"),
                    colors=entry.get("colors"),
                    iscell=entry.get("iscell"),
                ):
                    continue
                self._registry_extra.append(entry)
            self._load_array_results()
        finally:
            self._restoring = False

    def _load_array_results(self) -> None:
        """Load the array's own ``results``: one derived set per pixel unit (each on its z, keyed by plane dir), one trace row per domain of a line unit; an array in a run dir with no results shows that dir's ROIs."""
        from mbo_utilities.results import unit_for_source

        results = self.results
        if results is None:
            if self.fpath is None:
                return
            own = roi_folder(self.fpath)
            if run_dir_complete(own) and not any(s.path == own for s in self.derived):
                self.load_run(own)
            return
        file = Path(results.path or self.fpath or results.pipeline)
        src = base_array(self.iw.data[0]) if self.iw.data else None
        shown = unit_for_source(results, str(getattr(src, "unit_key", "") or ""))
        algo = detection_algo(results.metadata) if results.metadata else ""
        for unit in results.units.values():
            if unit.member_kind == "line" and unit.name == shown:
                name = f"{file.name}/{unit.name}"
                if not self.traces.from_source(name):
                    for trace in self._line_unit_rows(
                        results, unit, file, with_members=False
                    ):
                        self.traces.add(trace)
                    self._results_loaded.add(str(file))
                continue
            if unit.member_kind != "pixel" or unit.image_shape is None:
                continue
            if tuple(unit.image_shape) != (self.ny, self.nx):
                continue
            path = Path(
                unit.attrs.get("plane_dir") or (results.path or Path()) / unit.name
            )
            if any(s.path == path for s in self.derived):
                continue
            self._add_derived(  # a single-plane movie is this plane whatever z the unit recorded
                unit,
                path,
                results.pipeline,
                z=0 if self.store.nz == 1 else None,
                algo=algo,
            )

    def _run_entries(self) -> list[dict]:
        """The loaded runs as the rois zarr keeps them: one entry per derived
        set (its marks, labels, colors and accept / reject overrides) plus
        the runs no set stands for.
        """
        loaded = {str(s.path) for s in self.derived}
        entries = []
        for s in self.derived:
            iscell = s.unit.iscell
            own = (
                np.asarray(iscell)[:, 0] > 0
                if iscell is not None and len(iscell) == len(s)
                else np.ones(len(s), bool)
            )
            entries.append(
                {
                    "path": str(s.path),
                    "discarded": s.discarded,
                    "classes": s.classes,
                    "colors": {k: list(v) for k, v in s.colors.items()},
                    "iscell": {
                        k: bool(a)
                        for k, (a, o) in enumerate(zip(s.accepted, own))
                        if a != o
                    },
                }
            )
        blank = {"discarded": set(), "classes": {}, "colors": {}, "iscell": {}}
        entries += [
            {**blank, **e} for e in self._registry_extra if str(e["path"]) not in loaded
        ]
        return entries

    def _source_block(self) -> dict:
        """The rois zarr's record of the recording it annotates."""
        arr = base_array(self.iw.data[0]) if self.iw.data else None
        return {
            "path": str(self.fpath),
            "reader_kwargs": dict(getattr(arr, "reader_kwargs", None) or {}),
            "unit": self.unit_key,
        }

    def _write_rois(self) -> Path:
        """Write the store, the quick traces and the run list to the rois zarr; ``OSError`` when the folder cannot be written."""
        arr = base_array(self.iw.data[0]) if self.iw.data else None
        target = self._save_target()
        RoiFile(target).save(
            self.store,
            self.traces.rows,
            source=self._source_block(),
            runs=self._run_entries(),
            scale=tuple(getattr(arr, name, None) for name in ("dz", "dy", "dx")),
        )
        return target

    def _autosave(self):
        if not self.autosave or self.fpath is None or self._restoring:
            return
        try:
            self._write_rois()
            self._save_error = None
        except OSError as e:
            if self._save_error is None:
                self.logger.warning(f"autosave to {self._save_target()} failed: {e}")
            self._save_error = f"autosave failed: {e}"

    def save(self):
        """Write the drawn ROIs, the quick traces and the loaded runs to the rois zarr in the ROI folder."""
        if self.fpath is None:
            self.status = "no data path to save for"
            return
        try:
            target = self._write_rois()
        except OSError as e:
            self._save_error = f"save failed: {e}; pick another folder"
            return
        self._save_error = None
        self.status = f"saved to {target}"
        self.logger.info(f"saved {self.n_rois} ROIs to {target}")

    def set_autosave(self, on: bool):
        """Turn autosave on or off, remembered for every dataset; on saves everything now."""
        self.autosave = bool(on)
        set_roi_autosave(self.autosave)
        if self.autosave:
            self.save()

    def set_folder(self, root):
        """Save into ``root`` (one folder per dataset under it), or beside the data when None; remembered for every dataset."""
        self.root = Path(root) if root else None
        set_roi_folder(self.root or "")
        if self.autosave:
            self.save()
        else:
            self.status = f"ROIs will be saved to {self.folder}"


def attach_roi_widget(parent: Any, focus: bool = False) -> ManualRoiWidget | None:
    """Turn the ROI widget on for a ``PreviewDataWidget``, adopting the ROIs and runs of an earlier toggle this session."""
    widget = parent.manual_roi
    if widget is not None:
        widget.focus_tab = widget.focus_tab or focus
        return widget
    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
    widget = ManualRoiWidget(
        parent.image_widget,
        fpath,
        label_names=DEFAULT_LABEL_NAMES,
        store=getattr(parent, "_manual_roi_store", None),
        runs=getattr(parent, "_manual_roi_runs", None),
        strip=parent.top_strip,
        host=parent,
    )
    widget.focus_tab = focus
    parent.manual_roi = widget
    return widget


def detach_roi_widget(parent: Any) -> None:
    """Turn the ROI widget off, keeping its store and runs for the next toggle."""
    widget = parent.manual_roi
    if widget is None:
        return
    parent._manual_roi_store = widget.store
    parent._manual_roi_runs = widget.park_runs()
    widget.close()
    parent.manual_roi = None
