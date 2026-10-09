"""Manual ROI drawing, labeling and traces, laid out and keyed like masknmf's curation viewer.

Toggled from ``Widgets > Manual ROI Labeling`` in the preview GUI (``mbo
<path> --widget manualroi`` opens with it on). The ROIs tab of the
right-hand widget (``widgets/tabs.py``) is masknmf's Tools panel: the guide
and keybinds buttons on top, then a Curation tab (OVERLAY, SELECTION,
LABELS, RUN) and an ROIs tab (the filter over the table of drawn and algo
ROIs); the Traces tab holds the trace table. The trace plot is masknmf's
:class:`~masknmf.visualization.imgui.TracePlot`, a panel on the figure's
top strip, with the recording's motion correction as a panel of its own and
its behavior log as a raster row over them. The keys are masknmf's
``DEMIXING`` table plus its 1-9 / 0 / u labeling keys and a few of this
tool's own (:data:`ROI_KEYS`); while the tool is on it takes those keys from
the viewer's global shortcuts. The Process tab's ``ROIs`` pipeline
(``widgets/pipelines/rois.py``) reads this widget's model.

The state is a :class:`~mbo_utilities.annotation.RoiModel`: the
``RoiLabelStore`` (one ``(P, Y, X)`` uint16 label volume, 0 background, ROI
``i`` is ``i + 1``, so masks never overlap; each ROI keeps a persistent
``uid`` and a ``source`` naming where it came from), the
``RoiTraceTable`` (one row per measurement, keyed by ROI uid, z-plane,
channel and engine) and the slider position. The widget subscribes to the
model's events and redraws from them, so the same model drives the
Process tab's ROI pipeline without either widget polling the other.

Draw (a) arms a polygon: click its vertices on the image, click the first
one to close it. While it is there it selects every ROI in view on this
slice whose center is inside it (or outside, per its switch), live as it is
drawn and dragged. Add ROI (r) fills it into the store as a drawn ROI on
the exact slice the viewer shows (every scrolling dim except time keys its
own plane of masks); with ``auto_trace`` on its mean trace is computed at
once. Annotations autosave next to the data as an OME-NGFF-style labels
zarr (``manual_labels.zarr``, see ``mbo_utilities.annotation``) and are
restored from it on relaunch. A file holding several recordings (a
``.mesc``) gets one per recording (``manual_labels_MSession_0_MUnit_3.zarr``;
the run registry and run dirs likewise), and switching recordings swaps the
whole widget, so the ROIs, runs and traces on screen are the shown one's.

Runs read an ROI's mask where it was drawn and its pixels wherever the RUN
section points them (``run_z`` / ``run_c`` / ``run_tp``, "as drawn" by
default), through :meth:`run_rois`, writing ``rois_<tag>/`` beside the
data; a run of the same ROI at the same coordinates with the same engine
replaces its row. Region detection and full-plane suite2p / masknmf runs
load their outputs as derived sets, rows of the table, whose components can
be promoted into the drawn store (y). Delete (d) removes a drawn ROI and
marks an algo one for deletion: it stays listed at the top of the table and
red on the image until it is unmarked. Loaded runs are remembered in a
``roi_runs.json`` sidecar and restored on relaunch.

Masks draw as masknmf's feathered footprints: every ROI at the masks
opacity, the selection and its group at their own with a white rim, each
in the color its trace takes; contours outline every other ROI. Clicking
selects what is under the cursor, a drawn ROI before an algo one; clicking
the selection again, or the background, deselects. Ctrl+click (in the
image or the table) toggles an ROI in the group and shift+click adds it (a
row range in the table): the group's traces share the plot, one color per
member, and class labels apply to every member. With pixel traces on (p) a
click on an empty pixel adds the movie's 5x5 average there to the plot as a
group member. Ctrl+z undoes the last drawn or deleted ROI, mark, pixel
average or deselect. Selecting a listed ROI on another plane jumps every
slider that plane encodes. Only the first subplot is drawable.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr
from fastplotlib.graphics.selectors._polygon import point_in_polygon
from imgui_bundle import (
    icons_fontawesome_6 as fa,
)
from imgui_bundle import imgui
from masknmf.visualization.imgui import (
    CLICK_SLOP,
    GROUP_COLORS,
    RoiOrder,
    TracePlot,
    button_colors,
    draw_help_buttons,
    draw_keybinds_popup,
    draw_range_filter,
    draw_roi_table,
    draw_switch,
    grid,
    help_buttons_width,
    help_mark,
    right_aligned_text,
    section,
    switch_width,
    tooltip,
)
from masknmf.visualization.imgui import THEME as MTHEME
from masknmf.visualization.imgui.keybinds import CLASSIFICATION, DEMIXING, Bind, pressed
from masknmf.visualization.rois import MARKED_COLOR, SELECTED_ALPHA, FootprintSet

from mbo_utilities import log
from mbo_utilities.annotation import (
    DISPLAY_KINDS,
    ENGINES,
    FULL_IMAGE,
    UNLABELED,
    DffSettings,
    LabelsZarr,
    RoiLabelStore,
    RoiModel,
    RoiTrace,
    RoiTraceTable,
    available_kinds,
    display_trace,
    displayed_kind,
    trace_profile,
    y_label,
)
from mbo_utilities.annotation.display import DFF_METHODS
from mbo_utilities.arrays.features._dim_tags import (
    TAG_REGISTRY,
    DimensionTag,
    OutputFilename,
    filename_tags,
)
from mbo_utilities.arrays.features._selection import to_lsp_kwargs
from mbo_utilities.arrays.features._slicing import index_window
from mbo_utilities.behavior import behavior_for
from mbo_utilities.gui import roi_runs
from mbo_utilities.gui._keyboard import claim_keys
from mbo_utilities.gui._theme import THEME, em, label_button, to_vec4
from mbo_utilities.gui._top_strip import TopPanel, TopStrip
from mbo_utilities.gui.imgui import (
    UNLABEL_ALL,
    LabelSet,
    SummaryImageViewer,
    draw_label_editor,
)
from mbo_utilities.gui.imgui.behavior import EPOCH_COLORS, BehaviorPlot
from mbo_utilities.gui.imgui.motion import MOTION_COLORS, MotionPlot
from mbo_utilities.gui.imgui.table import FILTER_ALL
from mbo_utilities.gui.playhead import Playhead, TimeAxis
from mbo_utilities.gui.roi_runs import (
    DerivedSet,
    RoiRun,
    RoiRunManager,
    component_color,
    finished_dirs,
    full_plane_args,
    load_run_registry,
    outline_data,
    registry_path,
    result_traces,
    run_dir_complete,
    save_run_registry,
    set_color,
)
from mbo_utilities.gui.slice import Slice, viewer_positions, viewer_roles
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.lazy_array import base_array
from mbo_utilities.results import results_pipeline, unit_name
from mbo_utilities.roi_workflow import (
    OUT_PREFIX,
    SAVE_NAME,
    PlaneMovie,
    demix_rois,
    detection_algo,
    discover_rois,
    extract_rois,
    feather_mask,
    labels_path,
    load_run_dir,
    roi_trace,
    run_result_from_unit,
)

__all__ = [
    "COLOR_BY",
    "COLORMAPS",
    "ENGINE_HELP",
    "KEYBINDS",
    "ROI_KEYS",
    "ManualRoiWidget",
    "SAVE_NAME",
    "attach_roi_widget",
    "detach_roi_widget",
    "labels_path",
    "roi_widgets_available",
]

# height the Traces panel asks the top strip for (the strip adds the menu
# row and its tab bar on top of this); a motion panel and the behavior
# raster each add their own
PANEL_HEIGHT = 200
MOTION_PANEL_HEIGHT = 300
BEHAVIOR_PLOT_HEIGHT = 140

# how often to look for finished pipeline runs started outside this widget;
# the check reads one sidecar per tracked process, so not every frame
ADOPT_INTERVAL_S = 1.0

# the trace table's columns after its id: (name, sortable, hidden by default)
TRACE_COLUMNS = (
    ("z", True, False),
    ("c", True, False),
    ("source", False, True),
    ("engine", False, True),
    ("frames", True, True),
    ("peak", True, False),
)

# what a row may carry in ``extra`` about the line it was read on
# (``mesc_geometry.line_positions``); a row's own record wins over the recording's
POSITION_KEYS = (
    "start_um",
    "end_um",
    "z_um",
    "length_um",
    "sample_um",
    "dz_um",
    "stack",
    "slice",
    "slice_dz_um",
    "in_stack",
)

# tri-state stage toggles, shared by both pipelines: 0 skip, 1 run, 2 force
_STAGE_NAMES = ("skip", "run", "force")

MIN_ROI_PIXELS = 9
# the box a pixel trace averages around the clicked pixel: masknmf's 5x5
PIXEL_RADIUS = 2
# no seeded class labels: the label set starts empty, the user names their own
DEFAULT_LABEL_NAMES: tuple[str, ...] = ()
# what each extraction engine (annotation.ENGINES) gives back, for tooltips
ENGINE_HELP = {
    "mean": "mean - the raw mean of the pixels in each mask, frame by "
    "frame, plus a neuropil ring around it. No pipeline needed.",
    "suite2p": "suite2p - suite2p's own extractor: F from the mask and "
    "Fneu from its neuropil ring, ready for its neuropil "
    "subtraction.",
    "masknmf": "masknmf - seeded NMF: each mask's demixed signal, with light "
    "from overlapping neurons and background pulled back out.",
}
# where a run reads each ROI's pixels: where it was drawn, or the slice on screen
RUN_WHERE = ("drawn", "screen")

# the tag a run gets when the box is left empty; shown as the box hint
DEFAULT_RUN_TAG = "manual"

# the Curation tab's color by: masknmf's id palette, a store column or the trace peak
COLOR_BY = ("roi id", "class", "z", "c", "area", "peak")
COLORMAPS = ("viridis", "plasma", "turbo", "coolwarm", "tab10")
# contours trace each ROI's border, or ring it without covering it (o)
CONTOUR_SHAPES = ("outline", "circle")

# every grid's captions, so the caption column is one width across the sections and tabs
_CAPTIONS = (
    "masks",
    "contours",
    "sel masks",
    "sel contours",
    "weighting",
    "color by",
    "traces",
    "new label",
    "classes",
    "labeled",
    "engine",
    "where",
    "run",
    "options",
    "on disk",
    "filter",
    "label",
    "source",
    "slice",
    "range",
)

# behind the traces, naming what the lines are: near-black with nothing shown, forest green for a selection
_TRACE_MODES = {
    "normal": (0.02, 0.02, 0.03, 1.0),
    "selection": (0.05, 0.14, 0.08, 1.0),
}
_UNDO_DEPTH = 50  # ctrl+z steps kept

# masknmf's DEMIXING keys, worded for drawn and algo ROIs, with its labeling
# keys and this tool's own; every handler and the keybinds popup read this
ROI_KEYS: Mapping[str, Bind] = {
    **DEMIXING,
    "click": Bind(
        "click",
        "on an ROI: select it, again to deselect; on an empty pixel with pixel traces on: add its 5x5 average to the "
        "plot",
    ),
    "ctrl_click": Bind(
        "ctrl + click",
        "toggle an ROI or pixel average in the group, in the image or the table",
    ),
    "masks": Bind("m", "toggle every ROI's mask", imgui.Key.m),
    "contours": Bind("c", "toggle every other ROI's contour", imgui.Key.c),
    "follow": Bind(
        "f",
        "center the image on the selection and keep following it; labeling then advances",
        imgui.Key.f,
    ),
    "poly": Bind(
        "a",
        "draw: click a polygon on the image, the first point again to close it; it selects the ROIs it holds. Again or "
        "esc drops it, the selection stays",
        imgui.Key.a,
    ),
    "escape": Bind(
        "esc",
        "close the keybinds window, drop the drawn region (the selection stays), else deselect everything and drop the "
        "pixel averages",
        imgui.Key.escape,
    ),
    "select_all": Bind(
        "ctrl + a", "group every ROI the table shows", imgui.Key.a, ctrl=True
    ),
    "help": Bind("h", "this tool's guide", imgui.Key.h),
    "shift_click": Bind(
        "shift + click", "add an ROI to the group; in the table, every row up to it"
    ),
    "pixel_trace": Bind(
        "p",
        "toggle quick pixel trace: a click on an empty pixel adds the movie's 5x5 average to the plot",
        imgui.Key.p,
    ),
    "roi": Bind(
        "r",
        "add an ROI: keep the drawn region as a drawn ROI; with none drawn, start drawing",
        imgui.Key.r,
    ),
    "delete": Bind(
        "d / delete",
        "delete the selected drawn ROI, drop the active pixel average, or mark the selected algo ROIs for deletion "
        "(unmark when all are)",
        (imgui.Key.d, imgui.Key.delete),
    ),
    "undo": Bind(
        "ctrl + z",
        "undo the last drawn or deleted ROI, mark, pixel average or deselect",
        imgui.Key.z,
        ctrl=True,
    ),
    "label": Bind(
        "1-9",
        "assign that label to the selection or the group; with center on, step to the next",
    ),
    "clear": CLASSIFICATION["clear"],
    "unlabeled": CLASSIFICATION["unlabeled"],
    "promote": Bind(
        "y", "promote the selected algo ROI into the drawn ROIs", imgui.Key.y
    ),
    "discard": Bind(
        "n", "mark the selected algo ROI for deletion and step to the next", imgui.Key.n
    ),
    "accept": Bind("x", "accept / reject the selected algo ROI", imgui.Key.x),
    "rings": Bind("o", "contours: each ROI's border, or a ring around it", imgui.Key.o),
    "run": Bind(
        "shift + t",
        "run the selection through the RUN section's engine",
        imgui.Key.t,
        shift=True,
    ),
}
# the keys the viewer's global shortcuts leave alone while the tool is on
_CLAIMED_KEYS = (
    "up_arrow",
    "down_arrow",
    "left_arrow",
    "right_arrow",
    "m",
    "c",
    "p",
    "h",
    "k",
    "o",
)
# the menu bar's keybinds cheat sheet lists the same rows
KEYBINDS = tuple((bind.label, bind.action) for bind in ROI_KEYS.values())

_HELP_STEPS = (
    "Draw (a) and click a polygon's vertices on the image, the first one "
    "again to close it: it selects the ROIs whose centers it holds. Add ROI "
    "(r) keeps it as a drawn ROI; with trace on draw ticked its mean trace "
    "plots at once. Ctrl+Z undoes, d deletes the selection.",
    "Label ROIs with the class buttons or keys 1-9 (0 clears); u jumps to "
    "the next unlabeled one and, with center on (f), labeling steps there on its own.",
    "The Curation tab's RUN section runs the selected, grouped or listed "
    "ROIs through mean (raw mask average), suite2p (suite2p's extractor) or "
    "masknmf (seeded NMF, demixed), reading each mask where it was drawn or "
    "on the slice on screen; shift+t runs the selection.",
    "Every measurement is a row of the Traces tab: which ROI, on which "
    "z-plane and channel, with which engine. Re-running one replaces it.",
    "Detected components arrive as rows of the ROIs table: promote one into "
    "the drawn set (y) or mark it for deletion (d / n). Deleting a promoted "
    "ROI makes its row promotable again.",
    "Click an ROI to plot its traces; ctrl / shift + click group several, "
    "their traces share the plot. With pixel traces on (p), a click on an "
    "empty pixel adds its 5x5 average.",
)
_HELP_FILES = (
    "manual_labels.zarr  the drawn ROIs, autosaved\n"
    "                    (mbo_utilities.annotation.LabelsZarr.load)\n"
    "rois_<tag>/         one run's outputs: stat.npy, F.npy, Fneu.npy,\n"
    "                    iscell.npy, ops.npy, rois.json\n"
    "roi_runs.json       which runs this dataset has loaded"
)


def help_markdown() -> str:
    """This tool's guide, as markdown for the app's help viewer.

    The guide button (h) opens the app's help viewer on this section; the
    keys are not repeated here, the keybinds button (k) lists them.
    """
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(_HELP_STEPS, 1))
    return (
        "## ROI Labeling\n\n"
        "### Workflow\n\n"
        f"{steps}\n\n"
        "### Output files\n\n"
        f"```\n{_HELP_FILES}\n```\n"
    )


def slice_name(z: int | None = None, c: int | None = None) -> str:
    """One slice's name in the filename vocabulary (``ch02_zplane03``): the
    C tag before the Z tag, 1-based, only the axes given.
    """
    tags = []
    if c is not None:
        tags.append(DimensionTag(TAG_REGISTRY["C"], int(c) + 1, None))
    if z is not None:
        tags.append(DimensionTag(TAG_REGISTRY["Z"], int(z) + 1, None))
    return OutputFilename(tags).build("")


def _frames_of(movie) -> dict:
    """The frame fields of a row read through ``movie``: the window when
    the frames are one, else the index list.
    """
    indices = movie.t_indices
    window = index_window(indices)
    if indices is None or window is not None:
        return {"frames": window}
    return {"frames": None, "extra": {"tp_indices": list(indices)}}


def unit_key(iw) -> str:
    """The key of the recording a viewer shows when its file holds several
    (a MESc unit, ``MSession_0/MUnit_3``), else ``""``.
    """
    data = getattr(iw, "data", None)
    arr = base_array(data[0]) if data else None
    return str(getattr(arr, "unit_key", None) or "")


def roi_widgets_available() -> bool:
    """True when the shared imgui widgets import."""
    try:
        import mbo_utilities.gui.imgui  # noqa: F401
    except Exception:
        return False
    return True


def _cleared_note(cleared) -> str:
    """Status tail naming the filters a selection had to drop to show itself."""
    return (
        f" · cleared the {', '.join(cleared)} filter"
        + ("s" if len(cleared) > 1 else "")
        if cleared
        else ""
    )


class _PlaneOrder(RoiOrder):
    """masknmf's ``RoiOrder`` with label, "only this z-plane" and "only this source" filters.

    Rows marked for deletion (``del``) come first, as in masknmf.
    """

    def __init__(self, columns, labels, n_items):
        super().__init__(columns, n_items, pinned="del")
        self.labels = labels
        self.filter_label = FILTER_ALL
        self.plane: int | None = None
        self.planes = np.zeros(0, np.int64)
        self.source: int | None = None  # None = all, 0 = drawn, 1 + si = a set
        self.sources = np.zeros(0, np.int64)

    def rebuild(self):
        self.pinned = "del" if "del" in self.columns else None
        super().rebuild()
        if not len(self.order):
            return
        keep = np.ones(len(self.order), bool)
        if self.filter_label != FILTER_ALL:
            keep &= self.labels[self.order] == self.filter_label
        if self.plane is not None:
            keep &= self.planes[self.order] == self.plane
        if self.source is not None:
            keep &= self.sources[self.order] == self.source
        if keep.all():
            return
        current = self.current
        self.order = self.order[keep]
        hits = np.flatnonzero(self.order == current)
        self.pos = (
            int(hits[0])
            if len(hits)
            else int(min(self.pos, max(len(self.order) - 1, 0)))
        )

    def hidden_by(self, item: int) -> list:
        """Names of the filters that keep ``item`` out of the current view."""
        out = []
        if (
            self.filter_label != FILTER_ALL
            and int(self.labels[item]) != self.filter_label
        ):
            out.append("label")
        if self.hidden(item):
            out.append(self.range_column)
        if self.plane is not None and int(self.planes[item]) != self.plane:
            out.append("plane")
        if self.source is not None and int(self.sources[item]) != self.source:
            out.append("source")
        return out

    def clear_filter(self, name: str):
        if name == "label":
            self.filter_label = FILTER_ALL
        elif name == "plane":
            self.plane = None
        elif name == "source":
            self.source = None
        elif name == self.range_column:
            self.set_range_column(self.range_column)

    def reveal(self, item: int) -> list:
        """Put ``item`` under the cursor, dropping whatever filters hide it; returns those filters."""
        cleared = self.hidden_by(item)
        for name in cleared:
            self.clear_filter(name)
        if cleared:
            self.rebuild()
        self.goto(item)
        return cleared

    def next_unlabeled(self, inclusive: bool = False) -> bool:
        """First unlabeled drawn row after the cursor, wrapping.

        ``inclusive`` starts at the cursor instead of after it: when a label
        filter has just dropped the row that was labeled, the cursor already
        sits on the next candidate and stepping past it would skip one.
        """
        # only drawn rows can take a label, so u never lands on a derived one
        hits = np.flatnonzero(
            (self.labels[self.order] < 0) & (self.sources[self.order] == 0)
        )
        if not len(hits):
            return False
        after = hits[hits >= self.pos] if inclusive else hits[hits > self.pos]
        self.pos = int(after[0] if len(after) else hits[0])
        return True


class ManualRoiWidget:
    """Freehand ROI painting, labeling and run curation on a ``MboNDViewer``.

    Parameters
    ----------
    iw : MboNDViewer
        The viewer to draw on. Only its first subplot is drawable.
    fpath : path-like, optional
        The data path; annotations autosave to ``manual_labels.zarr`` beside
        it, loaded runs are remembered in ``roi_runs.json``, and both are
        restored from there on construction.
    label_names : iterable of str
        Class labels to seed the label set with.
    store : RoiLabelStore, optional
        Adopt this in-memory store instead of starting empty / restoring
        from disk (how ROIs survive an off/on toggle).
    runs : dict, optional
        State from :meth:`park_runs` of the previous widget (how loaded
        runs, traces and live background work survive an off/on toggle).
    strip : TopStrip, optional
        The figure's shared top strip to hang the Traces panel off; it also
        runs the per-frame hook. Omit to own one (standalone use).
    host : PreviewDataWidget, optional
        The widget that owns the viewer's display settings; the trace plot
        follows its window function. Omit when running standalone.
    auto_trace : bool
        Trace every ROI the moment it is drawn (its mean at the slice on
        screen), so drawing a cell shows its trace without another click.
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
        # a file holding several recordings (a .mesc) keeps ROIs, runs and
        # traces per recording: every sidecar is named after the one shown
        self.unit_key = unit_key(iw)
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
        # every scrolling dim except time keys its own mask plane, so masks
        # follow the channel / z / any extra slider; z sits last in the flat
        # order so a z-only store keeps plane == z and old stores restore
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
        # store + trace table + slider position; the Process tab reads the same object
        self.model = RoiModel(store)

        self.selected = -1
        self.selected_derived: tuple[int, int] | None = None
        # the spike-average windows the Traces panel started, until they exit
        self._windows: list = []
        # ctrl / shift click builds a group here; its traces share the plot
        # and label actions apply to every member. Entries are (si, k), si -1 = drawn.
        self.buffer: list[tuple[int, int]] = []
        # pixel averages (p): (store plane, row, col) -> trace key, newest
        # first; those in pixel_group plot with the group, delete drops the active one
        self.pixels: OrderedDict[tuple, tuple] = OrderedDict()
        self.pixel_group: list[tuple] = []
        self.active_pixel: tuple | None = None
        self.pixel_traces = False
        self.status = "draw (a) a region, add it as an ROI (r)"
        self._save_error: str | None = None
        self._run_error: str | None = None
        self._writer: LabelsZarr | None = None
        self.new_label = ""
        self._note_buf = ""
        self.scroll_to_selection = False
        self.follow = False  # center the shown ROI; labeling then advances
        self.keybinds_open = False
        # ctrl+z steps, newest last
        self._undo: list[dict] = []

        # masknmf's OVERLAY: every mask at one opacity, weighted by its trace's
        # peak or its own; the selection and group at their own with a white
        # rim; contours of every other ROI, and of the selection in its color
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
        # the trace table lists the slice on screen; "all slices" lifts it
        self.traces_this_slice = True

        self._feathers: dict[int, tuple] = {}
        self.rows: list[tuple[int, int]] = []
        self._row_index: dict[tuple[int, int], int] = {}
        self._promoted: dict[tuple[str, int], int] = {}
        self.derived: list[DerivedSet] = []
        self.classes = LabelSet(0, self.store.label_names)
        self.order = _PlaneOrder({"del": np.zeros(0, np.int8)}, self.classes.labels, 0)
        # the plane's footprints as drawn last: (key, rows, FootprintSet)
        self._footprints: tuple | None = None
        # the ROIs table's filter: apply makes the rows on the switch's side of
        # the range the group, and keeps it there while the range moves
        self.filter_select = False
        self.filter_outside = True

        # one time and one slice for every view of the recording: the host's
        # when it has them, fed by its handler on the sliders; a widget on a
        # bare viewer makes its own and feeds them itself
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
        # literal RGBA bytes: auto-ranging off the all-zero start saturates every colour to white
        self.overlay.vmin, self.overlay.vmax = 0, 255
        for tile in self.overlay.world_object.children:
            tile.material.pick_write = False
        # the contours: one line, every ROI's path in one buffer split by NaN
        # rows, per-vertex colors; thickness in screen pixels, so a hairline
        # stays one at any zoom
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

        # the drawn region (a): a polygon that selects what it holds until
        # Add ROI (r) keeps it or esc drops it
        self.region_selector = None
        self.region_outside = False
        self._region_key = None
        self._region_hits: list | None = None
        # its bounding box (y0, y1, x0, x1): where Find in region looks
        self.region: tuple[int, int, int, int] | None = None
        # the last press on the image: a click travelling further is a pan
        self._press: tuple[float, float] | None = None
        self._press_drawn = False
        renderer = self.subplot.renderer
        renderer.add_event_handler(self._pointer_down, "pointer_down")
        renderer.add_event_handler(self._pointer_up, "pointer_up")
        self.summary = SummaryImageViewer(iw.figure, title="Full FOV")

        # the last ROI a trace landed for
        self.trace_uid = 0
        self._trace_results: queue.Queue = queue.Queue()
        self._trace_threads: list[threading.Thread] = []
        # the trace table's highlighted rows: the selection's own, or the rows
        # picked there (trace_picked), which the plot then shows instead
        self.trace_sel: set[tuple] = set()
        self.trace_picked = False
        self._trace_stats: dict[tuple, tuple] = {}
        self._trace_display: dict[tuple, np.ndarray] = {}
        self._trace_deflection = (False, False)
        # one entry: the last windowed line, so panning does not recompute it
        self._trace_window_cache: dict[tuple, np.ndarray] = {}
        # what the plot shows of each row (a DISPLAY_KINDS name); None lets
        # every row's pipeline pick, and a kind a row lacks falls back the same way
        self._kind: str | None = None
        # the panel's own dF/F baseline for rows that compute one; None keeps
        # each pipeline's
        self.dff: DffSettings | None = None
        # the trace table's sortable order over the keys it lists
        self.trace_order = RoiOrder({}, 0)
        self._trace_keys: list[tuple] = []
        # plot the selection's traces; off, selecting only highlights (masknmf's show selected traces)
        self.show_traces = True
        # the recording's motion correction as a panel of the plot, its
        # behavior log as a raster row over it
        self.show_motion = True
        self.show_behavior = True
        # masknmf's stacked trace panels, rebuilt when the frames or panels change
        self.trace_plot: TracePlot | None = None
        self._plot_frames: tuple | None = None
        self._plot_lines_key = None
        self._motion_key = None
        # drawn on the tab in place of "no traces" while a host computes them
        self.pending_traces = None
        self._fs_value: float | None = None
        self._fs_read = False
        # the Curation tab's color by: the id palette, else a store column or the trace peak through a colormap
        self.color_by = COLOR_BY[0]
        self.color_cmap = COLORMAPS[0]

        # run_tp is the 0-based frame list parse_timepoint_selection gives; None = every frame
        self.engine = ENGINES[0]
        self.run_where = "drawn"
        self.run_z: int | None = None
        self.run_c: int | None = None
        self.run_tp: list[int] | None = None
        self.auto_trace = bool(auto_trace)
        # empty: the box shows DEFAULT_RUN_TAG as a hint, so what is
        # typed there is the user's own tag and nothing else
        self.run_tag = ""
        self.manager = RoiRunManager()
        self._registry_extra: list[dict] = []
        # pids of finished pipeline runs already pulled in by
        # _adopt_finished_runs, and when it last looked
        self._adopted: set[int] = set()
        self._adopt_checked = 0.0
        # results files (mbo_utilities.results) already loaded, by path
        self._results_loaded: set[str] = set()
        self._restoring = False

        # the top edge is shared (menu row, Signal Quality plot, this Traces
        # panel) and runs the per-frame hook whatever is up; standalone use,
        # tests and a bare viewer, gets its own strip
        self._own_strip = strip is None
        self.strip = TopStrip(iw.figure) if self._own_strip else strip
        self.strip.add_hook(self._frame)
        # kept so the panel can ask for more height once the motion panel shows
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
        """The recording's facets for the Traces tab: its motion correction,
        its behavior log (found beside it on first use) and where each of
        its scanned lines sits (an AOD unit's ``line_positions``, by ROI
        index; empty for anything else).
        """
        data = getattr(self.iw, "data", None)
        arr = base_array(data[0]) if data else None
        self.motion = MotionPlot(getattr(arr, "motion_correction", None))
        self.behavior = BehaviorPlot(behavior_for(arr) if arr is not None else None)
        self.line_positions = list(getattr(arr, "line_positions", None) or [])
        # what the pipeline that wrote the data found (a suite2p run's planes)
        self.results = getattr(arr, "results", None)

    def _line_position(self, trace: RoiTrace) -> dict:
        """Where the line a row was read on sits: the recording's placement
        of that ROI (its ``line`` in ``extra``, else its z on an AOD unit,
        whose Z axis is the ROI index) under the row's own record; empty
        without geometry.
        """
        line = trace.extra.get("line")
        if line is None and (trace.stands_for_roi or trace.source == FULL_IMAGE):
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
        for fn, kind in (
            (self._pointer_down, "pointer_down"),
            (self._pointer_up, "pointer_up"),
        ):
            try:
                renderer.remove_event_handler(fn, kind)
            except (KeyError, ValueError):
                pass
        if self._own_slice:
            try:
                self.iw.ndwidget.indices.remove_event_handler(self._on_indices)
            except (KeyError, ValueError, AttributeError):
                pass
        self.slice.remove_event_handler(self._on_slice)
        self.playhead.remove_event_handler(self._on_playhead)
        # the parked store and table must not keep calling into a closed widget
        self.model.close()
        for graphic in (self.overlay, self.outline):
            try:
                self.subplot.delete_graphic(graphic)
            except (KeyError, ValueError):
                pass
        try:
            self.summary.close()
            self.summary.cleanup()
        except Exception:
            self.logger.debug("summary viewer cleanup failed", exc_info=True)
        self.strip.remove_hook(self._frame)
        self.strip.unregister("traces")
        if self._own_strip:
            self.strip.close()
        self.strip = None

    def park_runs(self) -> dict:
        """State handed to the next attach: the live run manager, loaded
        sets, registry leftovers and the trace table.
        """
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
        """What the ``role`` axis is called on this data: the slider's own
        name when it is a word (``ROI`` on an AOD unit, whose Z axis is the
        line index; ``Zplane``, ``Channel``, ``View``), else the letter.
        """
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
        """The slider position over the dims that key mask planes."""
        # a unit switch (MESc) can drop a scroll dim the store was keyed on;
        # that dim then sits at plane 0 rather than taking the GUI down
        return {name: self.slice.positions.get(name, 0) for name, _n in self.plane_axes}

    def _current_z(self) -> int:
        """Flat store plane for the viewer's scroll position (see
        ``RoiLabelStore.plane_of``).
        """
        return self.store.plane_of(self._view_pos())

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
        if action in ("add", "delete", "clear"):
            live = {r.uid for r in self.store.rois}
            self.traces.prune(live)
            self._feathers = {u: v for u, v in self._feathers.items() if u in live}
        if action != "note":
            self._resync()
        if self.color_by not in (COLOR_BY[0], "peak") and action in (
            "add",
            "delete",
            "clear",
            "class",
        ):
            self.apply_color_by()
        self.refresh_overlay()
        # a note saves when its box loses focus, not per keystroke
        if action != "note":
            self._autosave()

    def _on_traces(self, _event):
        self._traces_changed()
        if self.color_by == "peak":
            self.apply_color_by()

    def apply_color_by(self):
        """Tint every drawn ROI by ``color_by`` through ``color_cmap``: a
        store column, or the peak of its traces; the id palette restores the
        class / hue colors.
        """
        if self.color_by == COLOR_BY[0]:
            self.model.colorize(None)
            return
        if self.color_by == "peak":
            values = {}
            for record in self.store.rois:
                peaks = [
                    self._trace_stat(t.key)[2] for t in self.traces.for_roi(record.uid)
                ]
                if peaks:
                    values[record.uid] = max(peaks)
        else:
            values = self.model.column(self.color_by)
        self.model.colorize(
            values,
            cmap=self.color_cmap,
            categorical=self.color_by in ("class", "z", "c"),
        )

    def set_color_by(self, by: str, cmap: str | None = None):
        if by not in COLOR_BY:
            raise ValueError(f"color by one of {COLOR_BY}, not {by!r}")
        self.color_by = by
        if cmap is not None:
            self.color_cmap = cmap
        self.apply_color_by()
        self.status = (
            "colors: class / group / hue"
            if by == COLOR_BY[0]
            else f"colored by {by} ({self.color_cmap})"
        )

    def current_frame(self) -> int:
        """The viewer's T index; 0 without a T slider (or while a unit switch
        has dropped it, until :meth:`rebind`).
        """
        if self.tdim is None:
            return 0
        try:
            return int(self.iw.indices[self.tdim])
        except KeyError:
            return 0

    def set_frame(self, frame: int):
        """Move the viewer's t; the trace cursor drag scrubs the movie with this."""
        if self.tdim is None or self.tdim not in self.iw.dim_names:
            return
        movie = self.movie()
        limit = (int(movie.shape[0]) - 1) if movie is not None else 0
        self.iw.indices[self.tdim] = int(np.clip(frame, 0, limit))

    @property
    def labels(self) -> np.ndarray:
        """The current z-plane's label image (a view into the store volume)"""
        return self.store.labels[self.z]

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
        """Rebuild the combined rows, label set and table order from the
        store and the loaded derived sets. Drawn rows come first so table
        ids match store indices; promoted rows are recomputed from the
        store's ``source`` strings. Algo rows marked for deletion stay
        listed, ``del`` set, and the table pins them on top.
        """
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
            iscell = s.result.iscell
            scored = iscell is not None and np.ndim(iscell) == 2 and iscell.shape[1] > 1
            for k, stat_row in enumerate(s.result.stat):
                self.rows.append((si, k))
                planes.append(s.result.z)
                sources.append(1 + si)
                areas.append(int(stat_row.get("npix", len(stat_row["ypix"]))))
                oks.append(1 if s.accepted[k] else 0)
                probs.append(float(iscell[k, 1]) if scored else np.nan)
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
        # in the order the filter combo lists them
        columns = {
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
            # the span follows the rows; limits the user narrowed stay, inside it
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
        return any(
            s.result.iscell is not None
            and np.ndim(s.result.iscell) == 2
            and s.result.iscell.shape[1] > 1
            for s in self.derived
        )

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

    def set_region_mode(self, on: bool):
        """The Process tab's region drawing: the same polygon, its bounding box is where Find looks."""
        self.set_drawing(on)

    def add_roi(self, points):
        """Fill a closed polygon (``(x, y)`` points on the image) and store it as the next label on the plane on screen."""
        if len(points) < 3:
            self.status = "a region needs three points"
            return
        points = np.round(np.asarray(points, np.float32)).astype(np.int32)
        points[:, 0] = points[:, 0].clip(0, self.nx - 1)
        points[:, 1] = points[:, 1].clip(0, self.ny - 1)
        filled = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(filled, [points], 1)
        index = self.store.add_roi(self.z, filled.astype(bool))
        if index is None:
            self.status = f"under {MIN_ROI_PIXELS} free px, not added"
            return
        self._push_undo({"kind": "add", "uid": self.store.rois[index].uid})
        self.select_roi(index)
        if self.auto_trace and self.trace_disabled(index) is None:
            self.quick_trace(index)

    def clear_region(self):
        """Drop the drawn region and the box Find looks in."""
        self._drop_region()
        self.region = None

    def _pick(self, row: int, col: int, mods: frozenset = frozenset()):
        """Masknmf's click: a drawn ROI under the cursor, else an algo one.
        Ctrl toggles it in the group, shift adds it; a plain click selects it
        alone, or deselects it when it is already the selection. An empty
        pixel adds its 5x5 average with pixel traces on, else deselects.
        """
        try:
            hit: tuple[int, int] | None = None
            index = self.store.roi_at(self.z, row, col)
            if index >= 0:
                hit = (-1, index)
            elif 0 <= row < self.ny and 0 <= col < self.nx:
                for si, s in enumerate(self.derived):
                    if not s.visible or s.result.z != self.z:
                        continue
                    k = int(s.pick_map[row, col])
                    if k >= 0:
                        hit = (si, k)
                        break
            ctrl = bool(mods & {"Ctrl", "Control"})
            if hit is not None:
                if ctrl:
                    self.buffer_toggle(*hit)
                elif "Shift" in mods:
                    self.buffer_add(*hit)
                elif self._selection_pair() == hit and not self.buffer:
                    self._snapshot()
                    self.select_roi(-1)
                else:
                    self.buffer_clear()
                    self.select_pair(*hit)
                return
            if self.pixel_traces and 0 <= row < self.ny and 0 <= col < self.nx:
                self.add_pixel(row, col, toggle=ctrl)
                return
            if self.buffer or self.pixel_group or self._selection_pair() is not None:
                self._snapshot()
            self.pixel_group = []
            self.active_pixel = None
            self.buffer_clear()
            self.select_roi(-1)
        except Exception as e:  # noqa: BLE001 - surfaced in the status row
            self.logger.exception("pick failed")
            self.status = f"pick failed: {type(e).__name__}: {e}"

    def _row_grouped(self, item) -> bool:
        if isinstance(item, tuple):
            return item in self.pixel_group
        return 0 <= item < len(self.rows) and self.rows[item] in self.buffer

    def _table_select(self, item):
        """A plain table click: the row alone, or nothing when it already is the selection."""
        if isinstance(item, tuple):
            pid = item[1:]
            self.buffer_clear()
            self.select_roi(-1)
            self.pixel_group = [pid]
            self.active_pixel = pid
            self._refresh_group_view()
            return
        if self.rows[item] == self._selection_pair() and not self.buffer:
            self._snapshot()
            self.select_roi(-1)
            return
        self.buffer_clear()
        self.select_row(item)

    def _table_ctrl(self, item):
        if isinstance(item, tuple):
            pid = item[1:]
            if pid in self.pixel_group:
                self.pixel_group.remove(pid)
                self.active_pixel = None
            else:
                self._seed_buffer()
                self.pixel_group.append(pid)
                self.active_pixel = pid
            self._refresh_group_view()
        elif 0 <= item < len(self.rows):
            self.buffer_toggle(*self.rows[item])

    def select_row(self, row: int | None):
        """Route a table-row selection to the right kind."""
        if row is None or not 0 <= row < len(self.rows):
            self.select_roi(-1)
            return
        self.select_pair(*self.rows[row])

    def select_roi(self, index: int | None):
        """Select drawn ROI ``index``; anything out of range clears the
        selection (and any derived one).

        Selecting an ROI on another plane jumps the z slider to it; one with
        a trace shows it in the trace plot.
        """
        self.selected_derived = None
        self.selected = index if index is not None and 0 <= index < self.n_rois else -1
        self.scroll_to_selection = True
        if self.selected < 0:
            self._note_buf = ""
            self.status = f"{self.n_rois} ROIs"
        else:
            record = self.store.rois[self.selected]
            self._note_buf = record.note
            cleared = self.order.reveal(self.selected)
            self.status = f"ROI {self.selected}: {record.area} px" + _cleared_note(
                cleared
            )
            if record.plane != self.z:
                self._goto_plane(record.plane)
            if self.traces.for_roi(record.uid):
                self.trace_uid = record.uid
            if self.follow:
                self._center_on(*self._feather(self.selected)[:2])
        self._sync_trace_sel()
        self.refresh_overlay()

    def select_derived(self, si: int, k: int):
        """Select component ``k`` of derived set ``si`` (clears any drawn
        selection; jumps z to the set's plane).
        """
        if not (
            0 <= si < len(self.derived) and 0 <= k < len(self.derived[si].result.stat)
        ):
            self.select_roi(-1)
            return
        self.selected = -1
        self._note_buf = ""
        self.selected_derived = (si, k)
        self.scroll_to_selection = True
        s = self.derived[si]
        row = self._row_index.get((si, k))
        cleared = self.order.reveal(row) if row is not None else []
        stat_row = s.result.stat[k]
        npix = int(stat_row.get("npix", len(stat_row["ypix"])))
        tail = " · promoted" if (s.name, k) in self._promoted else ""
        tail += " · marked for deletion" if k in s.discarded else ""
        self.status = f"{s.name} row {k}: {npix} px{tail}" + _cleared_note(cleared)
        if s.result.z != self.z:
            self._goto_plane(s.result.z)
        if self.follow:
            self._center_on(stat_row["ypix"], stat_row["xpix"])
        self._sync_trace_sel()
        self.refresh_overlay()

    def in_buffer(self, si: int, k: int) -> bool:
        return (si, k) in self.buffer

    def _seed_buffer(self):
        """A first ctrl / shift click keeps the current selection grouped,
        so 'select one, ctrl+click another' makes a group of two.
        """
        if self.buffer:
            return
        pair = self._selection_pair()
        if pair is not None:
            self.buffer.append(pair)

    def buffer_add(self, si: int, k: int):
        """Add one row to the group and make it the shown one."""
        self._seed_buffer()
        if (si, k) not in self.buffer:
            self.buffer.append((si, k))
        self._after_buffer_change(si, k)

    def buffer_toggle(self, si: int, k: int):
        """Ctrl+click: flip one row's group membership; the cursor moves to another member."""
        self._seed_buffer()
        if (si, k) in self.buffer:
            self.buffer.remove((si, k))
            if self._selection_pair() == (si, k):
                if self.buffer:
                    self.select_pair(*self.buffer[-1])
                else:
                    self.select_roi(-1)
            self._refresh_group_view()
            self.status = f"{len(self.buffer)} in group"
        else:
            self.buffer.append((si, k))
            self._after_buffer_change(si, k)

    def buffer_extend_to(self, item: int):
        """Shift+click in the table: group every row between the cursor and
        ``item``, in the order the table shows.
        """
        hits = np.flatnonzero(self.order.order == item)
        if not len(hits):
            return
        self._seed_buffer()
        a, b = sorted((self.order.pos, int(hits[0])))
        for pos in range(a, b + 1):
            pair = self.rows[int(self.order.order[pos])]
            if pair not in self.buffer:
                self.buffer.append(pair)
        self._after_buffer_change(*self.rows[item])

    def buffer_clear(self):
        if self.buffer or self.pixel_group:
            self.buffer = []
            self.pixel_group = []
            self.filter_select = False
            self._refresh_group_view()

    def _after_buffer_change(self, si: int, k: int):
        n = len(self.buffer)
        self.filter_select = (
            self.filter_select and set(self.buffer) == self._filter_pairs()
        )
        self.select_pair(si, k)
        if n > 1:
            self.status = (
                f"{n} in group · their traces share the plot, labels apply to all"
            )

    def _refresh_group_view(self):
        self._plot_lines_key = None
        self.refresh_overlay()

    def set_group_color(self, rgb: tuple[float, float, float] | None):
        """Give every grouped ROI (or just the selection) an explicit
        display color; None reverts to the class / hue colors.
        """
        targets = list(self.buffer)
        if not targets and self._selection_pair() is not None:
            targets = [self._selection_pair()]
        if not targets:
            return
        rgb255 = None if rgb is None else tuple(int(round(float(v) * 255)) for v in rgb)
        self.store.block_events(True)
        try:
            for si, k in targets:
                if si < 0:
                    self.store.set_color(k, rgb255)
                elif rgb is None:
                    self.derived[si].colors.pop(k, None)
                else:
                    self.derived[si].colors[k] = tuple(float(v) for v in rgb)
        finally:
            self.store.block_events(False)
        self._refresh_group_view()
        self._autosave()
        self._save_registry()
        self.status = (
            f"colored {len(targets)} ROI(s)"
            if rgb is not None
            else f"reset {len(targets)} color(s)"
        )

    def step(self, delta: int):
        """Up / down: the next or previous row of the ROIs table; the image,
        the table and the trace plot land on the same ROI.
        """
        if self.order.step(delta):
            self.buffer_clear()
            self.select_row(self.order.current)

    def step_trace(self, delta: int) -> bool:
        """Move to the next / previous row of the trace table, in the order
        the table shows, and select the ROI behind it.
        """
        rows = self._sorted_trace_rows()
        if not rows:
            return False
        at = next((i for i, key in enumerate(rows) if key in self.trace_sel), None)
        if at is None:
            pos = 0 if delta > 0 else len(rows) - 1
        else:
            pos = int(np.clip(at + delta, 0, len(rows) - 1))
        self.select_trace(rows[pos])
        return True

    def select_trace(self, key):
        """Plot just this trace and select the ROI behind it."""
        self.trace_sel = {key}
        self.trace_picked = True
        self._plot_lines_key = None
        pair = self._key_to_pair(key)
        if pair is None:
            return
        si, k = pair
        if si < 0:
            self.trace_uid = self.store.rois[k].uid
        self.select_pair(si, k)
        self.trace_sel = {key}
        self.trace_picked = True

    def toggle_trace(self, key):
        """Add / remove one trace from the plotted set (ctrl+click)."""
        (self.trace_sel.discard if key in self.trace_sel else self.trace_sel.add)(key)
        self.trace_picked = True
        self._plot_lines_key = None

    def next_unlabeled(self, inclusive: bool = False):
        if self.order.next_unlabeled(inclusive):
            self.select_row(self.order.current)

    def _select_next_derived(
        self,
        start_pos: int,
        skip_promoted: bool = False,
        unlabeled_only: bool = False,
    ):
        """Select the next derived row in view at or after ``start_pos``.

        ``unlabeled_only`` walks past rows that already carry a label and
        wraps once, so labeling a run's components in follow mode lands on
        each one that still needs a label rather than on whatever row
        happens to come next.
        """
        n = len(self.order.order)
        if not n:
            return
        start = max(start_pos, 0)
        span = (
            [(start + i) % n for i in range(n)] if unlabeled_only else range(start, n)
        )
        for pos in span:
            row = int(self.order.order[pos])
            si, k = self.rows[row]
            if si < 0:
                continue
            if skip_promoted and (self.derived[si].name, k) in self._promoted:
                continue
            if unlabeled_only and int(self.classes.labels[row]) != UNLABELED:
                continue
            self.select_row(row)
            return

    def delete_roi(self, index: int, record: bool = True):
        """Drop one drawn ROI and renumber the labels above it; traces of
        every other ROI survive (they are keyed by uid). Ctrl+z brings it back.
        """
        if not 0 <= index < self.n_rois:
            return
        if record:
            roi = self.store.rois[index]
            mask = self.store.labels[roi.plane] == index + 1
            self._push_undo(
                {
                    "kind": "delete",
                    "record": replace(roi),
                    "mask": np.nonzero(mask),
                    "traces": list(self.traces.for_roi(roi.uid)),
                }
            )
        # the store's event resyncs the rows, so the group is renumbered first
        self.buffer = [
            (si, k - (1 if si < 0 and k > index else 0))
            for si, k in self.buffer
            if not (si < 0 and k == index)
        ]
        self.store.delete_roi(index)
        self.select_roi(min(index, self.n_rois - 1))
        self.status = f"deleted ROI {index}"

    def delete_selected(self):
        """Masknmf's Delete: the selected drawn ROI goes, the active pixel
        average is dropped, else the selected algo ROIs are marked for
        deletion (unmarked when all are); marking one steps to the next row.
        """
        if self.selected >= 0 and not any(si >= 0 for si, _k in self.buffer):
            drawn = [k for si, k in self.buffer if si < 0] or [self.selected]
            for k in sorted(drawn, reverse=True):
                self.delete_roi(k)
            return
        if self.active_pixel is not None:
            self._snapshot()
            self.drop_pixel(self.active_pixel)
            self.refresh_overlay()
            return
        pairs = [pair for pair in self.buffer if pair[0] >= 0]
        if not pairs and self.selected_derived is not None:
            pairs = [self.selected_derived]
        if not pairs:
            return
        on = not all(k in self.derived[si].discarded for si, k in pairs)
        follows = None
        if on and len(pairs) == 1:
            row = self._row_index.get(pairs[0])
            view = [int(r) for r in self.order.order]
            if row in view[:-1]:
                follows = view[view.index(row) + 1]
        self.mark(pairs, on)
        if follows is not None:
            self.buffer_clear()
            self.select_row(follows)

    def clear(self):
        """Delete every drawn ROI (loaded runs and their rows stay)."""
        self.buffer = [pair for pair in self.buffer if pair[0] >= 0]
        self.store.clear()
        self.select_roi(-1)
        self.status = "cleared"

    def assign_class(self, class_index: int):
        """Give the selected ROI - drawn or derived - a class label;
        UNLABELED (-1) clears it. With a group of two or more (ctrl / shift
        click) the label lands on every member.
        """
        if len(self.buffer) > 1:
            self.store.block_events(True)
            try:
                for si, k in self.buffer:
                    if si < 0:
                        self.store.set_class(k, class_index)
                    elif class_index == UNLABELED:
                        self.derived[si].classes.pop(k, None)
                    else:
                        self.derived[si].classes[k] = int(class_index)
            finally:
                self.store.block_events(False)
            self._resync()
            name = (
                "unlabeled"
                if class_index == UNLABELED
                else self.classes.names[class_index]
            )
            self.status = f"{len(self.buffer)} ROIs -> {name}"
            self._refresh_group_view()
            self._autosave()
            self._save_registry()
            return
        if self.selected >= 0:
            self.store.set_class(self.selected, class_index)
            # False when a label filter dropped the row as it was labeled -
            # the cursor is then already on the next candidate
            listed = self.order.goto(self.selected)
            self.status = f"ROI {self.selected}: {self.classes.name_of(self.selected)}"
            if self.follow and class_index != UNLABELED:
                self.next_unlabeled(inclusive=not listed)
            return
        if self.selected_derived is None:
            return
        si, k = self.selected_derived
        s = self.derived[si]
        if class_index == UNLABELED:
            s.classes.pop(k, None)
        else:
            s.classes[k] = int(class_index)
        self._resync()
        row = self._row_index.get((si, k))
        if row is not None:
            self.order.goto(row)
            self.status = f"{s.name} row {k}: {self.classes.name_of(row)}"
        self._save_registry()
        if self.follow and class_index != UNLABELED:
            # the next algo row that still needs a label, not just the next
            # one: labeling used to land on rows already labeled
            self._select_next_derived(self.order.pos + 1, unlabeled_only=True)

    label_selected = assign_class

    def unlabel_all(self):
        self.store.block_events(True)
        try:
            for i in range(self.n_rois):
                self.store.set_class(i, UNLABELED)
        finally:
            self.store.block_events(False)
        self.classes.assign(range(self.n_rois), UNLABELED)
        self.order.rebuild()
        self.refresh_overlay()
        self.status = f"cleared {self.n_rois} labels"
        self._autosave()

    def _feather(self, index: int) -> tuple:
        """``(ypix, xpix, lam)`` of one drawn mask, soft-edged; cached by
        uid (a mask's pixels never change once drawn).
        """
        record = self.store.rois[index]
        got = self._feathers.get(record.uid)
        if got is None:
            mask = self.store.labels[record.plane] == index + 1
            w = feather_mask(mask)
            ypix, xpix = np.nonzero(mask)
            got = (ypix.astype(np.int32), xpix.astype(np.int32), w[ypix, xpix])
            self._feathers[record.uid] = got
        return got

    def refresh_overlay(self):
        """Masknmf's masks and contours over the plane on screen, drawn and
        algo ROIs alike: every mask at the masks opacity (weighted by its
        trace's peak, or its own), the selection and the group at the sel
        masks opacity with a white rim, marked rows red; every other ROI's
        contour in white, the selection's and the group's in their color.
        """
        rows = [row for row in range(len(self.rows)) if self._row_plane(row) == self.z]
        key = (self.z, tuple(rows))
        if self._footprints is None or self._footprints[0] != key:
            footprints = FootprintSet([self._row_footprint(row) for row in rows])
            self._footprints = (key, rows, footprints)
        _key, rows, footprints = self._footprints
        footprints.colors = (
            np.array([self._row_rgb(row) for row in rows], np.float32) if rows else None
        )
        peaks = np.asarray(self.order.columns.get("peak", np.zeros(0)), np.float64)
        peaks = (
            peaks[rows] if len(peaks) == len(self.rows) else np.full(len(rows), np.nan)
        )
        finite = np.isfinite(peaks) & (peaks > 0)
        top = float(peaks[finite].max()) if finite.any() else 1.0
        # a row without a trace draws as if at the field's peak
        footprints.peaks = np.where(finite, peaks, top).astype(np.float32)
        position = {self.rows[row]: i for i, row in enumerate(rows)}
        group = self._group_colors()
        picks = [position[pair] for pair in self.buffer if pair in position]
        pair = self._selection_pair()
        selected = position.get(pair) if pair is not None else None
        visible = self.show_masks or self.show_selected_masks
        self.overlay.visible = visible and bool(rows)
        if self.overlay.visible:
            self.overlay.data = footprints.rgba(
                (self.ny, self.nx),
                self.opacity if self.show_masks else 0.0,
                selected if self.show_selected_masks else None,
                [],
                {i: group.get(self.rows[rows[i]], footprints.color(i)) for i in picks}
                if self.show_selected_masks
                else {},
                self.selected_opacity,
                by_peak=self.masks_by_peak,
            )
        highlighted = set(picks) | ({selected} if selected is not None else set())
        comps = []
        for i, row in enumerate(rows):
            on = i in highlighted
            if (
                on
                and not self.show_selected_contours
                or not on
                and not self.show_contours
            ):
                continue
            ypix, xpix, lam = footprints.footprints[i]
            if on:
                comps.append(
                    (
                        ypix,
                        xpix,
                        lam,
                        footprints.color(i),
                        self.selected_contour_opacity,
                    )
                )
            else:
                comps.append((ypix, xpix, lam, (1.0, 1.0, 1.0), self.contour_opacity))
        positions, colors = outline_data(comps, self.contour_shape)
        if not len(positions):
            self.outline.visible = False
            return
        self.outline.data = positions
        self.outline.colors = colors
        self.outline.visible = True

    def _center_on(self, ypix, xpix):
        """Pan the camera onto one mask, holding the zoom the user set.

        Framing every ROI with ``show_rect`` re-zoomed the view on each
        step, always to the same wide rect, so a review pass fought the
        camera the whole way. The view only moves; it widens only when the
        mask cannot fit in it (or when there is no view yet).
        """
        if not len(ypix):
            return
        y0, y1 = float(np.min(ypix)), float(np.max(ypix))
        x0, x1 = float(np.min(xpix)), float(np.max(xpix))
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        cam = self.subplot.camera
        # a little air around the mask, so "does it fit" is not pixel-exact
        need_w, need_h = (x1 - x0) * 1.5, (y1 - y0) * 1.5
        width, height = float(cam.width), float(cam.height)
        if width <= 0 or height <= 0 or need_w > width or need_h > height:
            half = max(need_w, need_h, 40.0) / 2
            cam.show_rect(cx - half, cx + half, cy - half, cy + half)
            return
        pos = cam.local.position
        cam.local.position = (cx, cy, float(pos[2]))

    def _center_selection(self):
        if self.selected >= 0:
            ypix, xpix, _lam = self._feather(self.selected)
            self._center_on(ypix, xpix)
        elif self.selected_derived is not None:
            si, k = self.selected_derived
            row = self.derived[si].result.stat[k]
            self._center_on(row["ypix"], row["xpix"])

    def toggle_follow(self):
        """Center (f): every view on the selection, following it as it moves;
        a label then steps to the next ROI, like masknmf's classification GUI.
        """
        self.follow = not self.follow
        if self.follow:
            self._center_selection()
            self.status = "centering on the selection; labeling advances"
        else:
            self.status = f"{self.n_rois} ROIs"

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
        self, res, discarded=(), classes=None, colors=None
    ) -> DerivedSet | None:
        """Wrap a loaded run as a derived set (replacing an earlier load of
        the same dir) and merge its uid-keyed traces.
        """
        if tuple(res.shape) != (self.ny, self.nx):
            self._run_error = (
                f"{res.path.name} is {res.shape[0]}x{res.shape[1]}, "
                f"data is {self.ny}x{self.nx}"
            )
            return None
        if not 0 <= res.z < self.store.nz:
            self._run_error = (
                f"{res.path.name} is plane {res.z + 1}, "
                f"data has {self.store.nz} plane(s)"
            )
            return None
        promoted_traces: list[RoiTrace] = []
        classes = dict(classes or {})
        colors = dict(colors or {})
        for si, old in enumerate(self.derived):
            if old.result.path == res.path:
                discarded = set(discarded) | old.discarded
                classes = {**old.classes, **classes}
                colors = {**old.colors, **colors}
                # rows promoted out of the old load stay with their drawn rois
                promoted_traces = [
                    t for t in self.traces.from_source(old.name) if t.stands_for_roi
                ]
                self.unload_set(si)
                break
        s = DerivedSet(
            res,
            self._set_name(res.path),
            set_color(len(self.derived)),
            discarded={int(k) for k in discarded},
            classes={int(k): int(v) for k, v in classes.items()},
            colors={int(k): tuple(float(x) for x in v) for k, v in colors.items()},
        )
        self.derived.append(s)
        if self.store.nz > 1 and len(self.derived) == 1 and self.order.plane is None:
            # per-plane results: the ROI table opens on the plane on screen
            self.order.plane = self.z
        self._merge_run_traces(res, s.name)
        for k in range(len(res.stat)):
            if k not in s.discarded and (s.name, k) not in self._promoted:
                trace = self._member_trace(s, k)
                if trace is not None:
                    self.traces.add(trace)
        for trace in promoted_traces:
            self.traces.add(trace)
        self._resync()
        self.refresh_overlay()
        self._save_registry()
        return s

    def _member_trace(self, s: DerivedSet, k: int) -> RoiTrace | None:
        """Component ``k`` of a loaded set as a Traces-tab row, or None when
        the run carries no traces.
        """
        res = s.result
        if res.F is None or not 0 <= k < len(res.F):
            return None
        pos = self._plane_pos(res.z)
        zname, cname = self.store.axis_name("z"), self.store.axis_name("c")
        return RoiTrace(
            uid=0,
            member=k,
            source=s.name,
            z=pos.get(zname, 0) if res.read_z is None else int(res.read_z),
            c=(pos.get(cname, 0) if cname else self._channel())
            if res.read_c is None
            else int(res.read_c),
            engine=res.engine
            or ("masknmf" if res.kind in ("demix", "masknmf") else "suite2p"),
            F=np.asarray(res.F[k], np.float32),
            Fneu=None if res.Fneu is None else np.asarray(res.Fneu[k], np.float32),
            norm=None if res.norm is None else np.asarray(res.norm[k], np.float32),
            kinds={name: np.asarray(v[k], np.float32) for name, v in res.kinds.items()},
            frames=res.frames,
            path=res.path,
        )

    def load_run(self, path, discarded=(), classes=None, colors=None) -> bool:
        """Read one run dir into the widget: extract runs merge their
        traces, everything else loads as a derived set (every row, the
        rejected ones included - curation happens here). A results file
        (``mbo_utilities.results``), one unit inside one, or a masknmf run
        folder (its newest results file) goes through :meth:`load_results`.
        """
        from mbo_utilities.arrays.masknmf_run import is_masknmf_run
        from mbo_utilities.results import newest_results, results_pipeline

        path = Path(path)
        if is_masknmf_run(path) and newest_results(path) is not None:
            return self.load_results(newest_results(path), discarded, classes, colors)
        if results_pipeline(path) is not None or (
            path.parent.suffix == ".zarr" and results_pipeline(path.parent) is not None
        ):
            return self.load_results(path, discarded, classes, colors)
        try:
            res = load_run_dir(path, iscell_only=False, logger=self.logger)
        except Exception as e:  # noqa: BLE001 - shown in the status row
            self._run_error = f"could not load {path.name}: {e}"
            return False
        if (
            self.store.nz == 1
            and res.z != 0
            and self.fpath is not None
            and path == labels_path(self.fpath).parent
        ):
            # the movie on screen IS this plane, whatever z the run recorded
            res = replace(res, z=0)
        if res.kind == "extract":
            self._merge_run_traces(res, self._set_name(path))
            if not any(str(e["path"]) == str(path) for e in self._registry_extra):
                self._registry_extra.append(
                    {"path": str(path), "kind": res.kind, "discarded": []}
                )
            self._save_registry()
            return True
        return self._add_derived(res, discarded, classes, colors) is not None

    def load_results(self, path, discarded=(), classes=None, colors=None) -> bool:
        """Read a results file (AGENTS.md §7.5) into the widget. Pixel units
        (suite2p, masknmf) load as derived sets exactly like a run dir; line
        units (the vnoiser pipeline's scans) go straight to the Traces tab,
        one row per ROI plotting its denoised trace and one per member line
        plotting the line's raw trace. Every row is named by its ROI
        (``roi3``, ``roi3 (raw)``; a line of a multi-line ROI adds itself,
        ``roi3 line 12 (raw)``) and carries the line it was read from on Z
        and the pipeline's channel on C. ``path`` may name one unit inside
        the file (``<file>.zarr/zplane01``). Returns True when anything
        loaded.
        """
        from mbo_utilities.results import Results, results_pipeline

        path = Path(path)
        if path.parent.suffix == ".zarr" and results_pipeline(path) is None:
            file, only = path.parent, path.name
        else:
            file, only = path, None
        try:
            results = Results.read(file)
        except Exception as e:  # noqa: BLE001 - shown in the status row
            self._run_error = f"could not load {file.name}: {e}"
            return False
        loaded = 0
        for unit in results.units.values():
            if only is not None and unit.name != only:
                continue
            if unit.member_kind == "pixel" and unit.image_shape is not None:
                plane_dir = unit.attrs.get("plane_dir")
                key = (
                    Path(plane_dir)
                    if plane_dir and Path(plane_dir).is_dir()
                    else file / unit.name
                )
                res = run_result_from_unit(unit, key, results.pipeline)
                if (
                    self.store.nz == 1
                    and res.z != 0
                    and self.fpath is not None
                    and file.parent == labels_path(self.fpath).parent
                ):
                    # the movie on screen IS this plane, whatever plane the file recorded
                    res = replace(res, z=0)
                loaded += self._add_derived(res, discarded, classes, colors) is not None
                continue
            name = f"{file.name}/{unit.name}"
            rows: list[RoiTrace] = []
            n = unit.n_rois
            lines = unit.member_kind == "line"
            channel = int(results.source.get("channel") or 0)
            # a line's position comes from the reader's geometry, when the
            # shown recording is the scan these results came from
            src = base_array(self.iw.data[0])
            positions = (
                getattr(src, "line_positions", None)
                if lines
                and unit.attrs.get("source_unit")
                in (None, getattr(src, "unit_key", None))
                else None
            ) or []
            for k, roi in enumerate(unit.roi_names):
                # the run's own names, which the spike-average window opens the ROI by
                entry = {
                    "label": str(roi),
                    "fs": unit.fs,
                    "c": channel,
                    "kinds": {},
                    "extra": {"unit": unit.name, "roi": str(roi)},
                }
                members = unit.members[k] if k < len(unit.members) else ()
                if lines and len(members) == 1:
                    entry["z"] = int(members[0])
                    entry["extra"]["line"] = int(members[0])
                    if int(members[0]) < len(positions):
                        entry["extra"].update(positions[int(members[0])])
                # every kind the unit wrote, under the row's field for it
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
                        label = (
                            f"{unit.roi_names[k]} {unit.member_kind} {member} ({kind})"
                        )
                    where = (
                        {"z": int(member), "extra": {"line": int(member)}}
                        if lines
                        else {}
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
                            **where,
                        )
                    )
            if not rows:
                continue
            self.traces.drop_source(name)
            for trace in rows:
                self.traces.add(trace)
            loaded += 1
        if not loaded:
            if not self._run_error:
                self._run_error = f"{file.name} has no unit this view can show"
            return False
        self._results_loaded.add(str(file))
        if any(t.source.startswith(f"{file.name}/") for t in self.traces):
            if not any(str(e["path"]) == str(file) for e in self._registry_extra):
                self._registry_extra.append(
                    {"path": str(file), "kind": results.pipeline, "discarded": []}
                )
        self._save_registry()
        self.status = f"loaded {loaded} unit(s) of {file.name}"
        return True

    def unload_set(self, si: int):
        s = self.derived.pop(si)
        self.traces.drop_source(s.name)
        self._registry_extra = [
            e for e in self._registry_extra if str(e["path"]) != str(s.result.path)
        ]
        if self.selected_derived is not None:
            osi, k = self.selected_derived
            if osi == si:
                self.selected_derived = None
            elif osi > si:
                self.selected_derived = (osi - 1, k)
        self._resync()
        self.refresh_overlay()
        self._save_registry()

    def promoted_index(self, si: int, k: int) -> int | None:
        """Store index of the drawn ROI promoted from set ``si`` row ``k``."""
        return self._promoted.get((self.derived[si].name, k))

    def _promote(self, si: int, k: int) -> int | None:
        """Copy one derived component into the store; None with a status
        message when it cannot land.
        """
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
        stat_row = s.result.stat[k]
        mask = np.zeros((self.ny, self.nx), bool)
        mask[stat_row["ypix"], stat_row["xpix"]] = True
        index = self.store.add_roi(s.result.z, mask, source=f"{s.name}:{k}")
        if index is None:
            self.status = "overlaps existing ROIs, nothing free to claim"
            return None
        self._promoted[(s.name, k)] = index
        if k in s.classes:
            self.store.set_class(index, s.classes[k])
        # the component's trace becomes the drawn ROI's, keyed by its uid
        trace = self.traces.remove(("member", s.name, k)) or self._member_trace(s, k)
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
            for k in range(len(s.result.stat)):
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
        self.refresh_overlay()
        self._autosave()
        self.status = f"{s.name}: promoted {promoted} / skipped {skipped}"

    def set_accepted(self, si: int, k: int, on: bool | None = None):
        """Flip (or set) one derived component's accepted flag, mirrored
        into the run dir's ``iscell.npy``, or a results file's ``rois/iscell``.
        """
        s = self.derived[si]
        s.accepted[k] = (not s.accepted[k]) if on is None else bool(on)
        path = s.result.path / "iscell.npy"
        try:
            if s.result.path.parent.suffix == ".zarr":
                group = zarr.open_group(s.result.path.parent, mode="r+")
                group[f"{s.result.path.name}/rois/iscell"][k, 0] = float(s.accepted[k])
            else:
                n = len(s.result.stat)
                iscell = np.load(path) if path.exists() else np.ones((n, 2), np.float32)
                if len(iscell) != n:
                    iscell = np.ones((n, 2), np.float32)
                iscell[k, 0] = 1.0 if s.accepted[k] else 0.0
                np.save(path, iscell)
        except OSError as e:
            self._save_error = f"iscell save failed: {e}"
        self._resync()
        self.refresh_overlay()
        state = "accepted" if s.accepted[k] else "rejected"
        self.status = f"{s.name} row {k}: {state}"

    def discard_derived(self, si: int, k: int, advance: bool = False):
        """Mark one algo row for deletion (n); ``advance`` steps to the next row."""
        self.mark([(si, k)], True)
        if advance:
            self._select_next_derived(self.order.pos + 1)

    def undiscard_derived(self, si: int, k: int):
        self.mark([(si, k)], False)

    def restore_discarded(self, si: int):
        s = self.derived[si]
        self.mark([(si, k) for k in sorted(s.discarded)], False)

    def _save_target(self) -> Path:
        return labels_path(self.fpath, self.tag)

    def _restore(self):
        """Adopt a previously saved labels zarr next to the data, if any."""
        if self.fpath is None:
            return
        target = self._save_target()
        self._writer = LabelsZarr(target)
        if self._adopted_store:
            # the parked store is the in-session truth; the zarr can be
            # behind it when an autosave failed
            return
        if not target.exists():
            return
        try:
            store = LabelsZarr.load(target)
        except (OSError, ValueError) as e:
            self.logger.warning(f"could not restore {target}: {e}")
            self.status = f"restore failed: {e}"
            return
        if (store.nz, store.ny, store.nx) != (
            self.store.nz,
            self.store.ny,
            self.store.nx,
        ):
            zsize = dict(self.plane_axes).get(self.zdim, 1)
            if (
                (store.ny, store.nx) == (self.store.ny, self.store.nx)
                and not store.plane_axes
                and store.nz == zsize < self.store.nz
            ):
                # a store saved before channels keyed planes: z is last in
                # the flat order, so its planes are the first ones here
                grown = np.zeros(self.store.labels.shape, np.uint16)
                grown[: store.nz] = store.labels
                store = RoiLabelStore(
                    self.store.nz,
                    self.store.ny,
                    self.store.nx,
                    label_names=store.label_names,
                    labels=grown,
                    rois=store.rois,
                    next_uid=store.next_uid,
                )
            else:
                self.logger.warning(
                    f"{target} is {store.labels.shape}, data wants "
                    f"{self.store.labels.shape}; starting fresh"
                )
                self.status = "saved labels do not match this data, starting fresh"
                return
        for name in self.store.label_names:
            store.add_label_name(name)
        store.min_pixels = MIN_ROI_PIXELS
        store.plane_axes = self.plane_axes
        store.axis_roles = self.axis_roles
        self.store = store
        self.status = f"restored {len(store.rois)} ROIs"
        self.refresh_overlay()

    def _restore_runs(self, parked: dict | None):
        """Adopt the previous widget's parked runs, else re-load every
        surviving run dir named in ``roi_runs.json``.
        """
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
                    if tuple(s.result.shape) == (self.ny, self.nx)
                ]
                self._registry_extra = list(parked.get("extra") or [])
                return
            # the parked sets and traces key uids of the previous data's
            # store; fall through to this data's own registry
        self._restoring = True
        try:
            entries = (
                load_run_registry(registry_path(self.fpath, self.tag))
                if self.fpath is not None
                else []
            )
            for entry in entries:
                path = Path(entry["path"])
                if not run_dir_complete(path):
                    # a spawned pipeline may have suffixed the dir name
                    hits = sorted(
                        d
                        for d in path.parent.glob(path.name + "*")
                        if d.is_dir() and run_dir_complete(d)
                    )
                    if hits:
                        path = hits[0]
                        entry = {**entry, "path": str(path)}
                if run_dir_complete(path):
                    if self.load_run(
                        path,
                        discarded=entry.get("discarded", ()),
                        classes=entry.get("classes"),
                        colors=entry.get("colors"),
                    ):
                        continue
                self._registry_extra.append(entry)
            self._load_array_results()
        finally:
            self._restoring = False

    def _load_array_results(self) -> None:
        """Show what the pipeline that wrote the data on screen found: one
        derived set per pixel unit of the array's ``results`` (a suite2p
        volume's planes each land on their own z), keyed by the plane dir the
        unit was read from so curation writes back there. An array with no
        results that sits in a run dir still shows that dir's ROIs.
        """
        results = self.results
        if results is None:
            if self.fpath is None:
                return
            own = labels_path(self.fpath).parent
            if run_dir_complete(own) and not any(
                s.result.path == own for s in self.derived
            ):
                self.load_run(own)
            return
        for unit in results.units.values():
            if unit.member_kind != "pixel" or unit.image_shape is None:
                continue
            if tuple(unit.image_shape) != (self.ny, self.nx):
                continue
            path = Path(
                unit.attrs.get("plane_dir") or (results.path or Path()) / unit.name
            )
            if any(s.result.path == path for s in self.derived):
                continue
            res = run_result_from_unit(unit, path, results.pipeline)
            if results.metadata:
                # the run's ops say which detector made the rows (s2p-sparsery, s2p-cellpose)
                res = replace(res, algo=detection_algo(results.metadata))
            if self.store.nz == 1 and res.z != 0:
                # the movie on screen IS this plane, whatever z the unit recorded
                res = replace(res, z=0)
            self._add_derived(res)

    def _autosave(self):
        if self._writer is None:
            return
        try:
            self._writer.save_dirty(self.store, source_path=self.fpath)
            self._save_error = None
        except OSError as e:
            if self._save_error is None:
                self.logger.warning(f"autosave to {self._writer.path} failed: {e}")
            self._save_error = f"autosave failed: {e}"

    def save(self):
        """Write the full store to ``manual_labels.zarr`` next to the data."""
        target = self._save_target()
        if self._writer is None or self._writer.path != target:
            self._writer = LabelsZarr(target)
        try:
            self._writer.save(self.store, source_path=self.fpath)
        except OSError as e:
            self._save_error = f"save failed: {e}"
            return
        self._save_error = None
        self.status = f"saved to {target.name}"
        self.logger.info(f"saved {self.n_rois} ROIs to {target}")

    def _save_registry(self):
        """Mirror the loaded sets (plus not-yet-loadable entries) into the
        ``roi_runs.json`` sidecar.
        """
        if self.fpath is None or self._restoring:
            return
        loaded = {str(s.result.path) for s in self.derived}
        entries = [
            {
                "path": str(s.result.path),
                "kind": s.result.kind,
                "discarded": s.discarded,
                "classes": s.classes,
                "colors": {k: list(v) for k, v in s.colors.items()},
            }
            for s in self.derived
        ]
        entries += [e for e in self._registry_extra if str(e["path"]) not in loaded]
        try:
            save_run_registry(registry_path(self.fpath, self.tag), entries)
        except OSError as e:
            self._save_error = f"run registry save failed: {e}"

    def open_full_fov(self):
        """Masknmf's summary-image popup over the frame on screen and the labels."""
        frame = np.asarray(self.image.data.value, np.float32)
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=-1)
        images = {"current frame": frame}
        if self.n_rois:
            images["ROI labels"] = self.labels.astype(np.float32)
        self.summary.set_images(images, selected="current frame")
        self.summary.open()

    def _channel(self) -> int:
        return int(self.iw.indices[self.cdim]) if self.cdim is not None else 0

    def movie(
        self, plane: int | None = None, *, z: int | None = None, c: int | None = None
    ) -> PlaneMovie | None:
        """``(T, Y, X)`` view of the viewer's array.

        By default the pixels behind store ``plane`` (the plane on screen
        when None): the z-plane and channel that plane encodes, the
        viewer's channel when channels do not key planes. ``z`` / ``c``
        read another z-plane or channel instead, with the same mask (a
        cell drawn on the structural channel, traced on the functional
        one). None when the array cannot be wrapped.
        """
        pos = self._plane_pos(self.z if plane is None else int(plane))
        zname, cname = self.store.axis_name("z"), self.store.axis_name("c")
        if z is None:
            z = pos.get(zname, 0) if zname is not None else 0
        if c is None:
            c = pos.get(cname, 0) if cname is not None else self._channel()
        try:
            arr = self.iw.data[0]
            nz = PlaneMovie(arr).nz
            return PlaneMovie(arr, z=(int(z) if nz > 1 else 0), c=int(c))
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    def _coords(self, z=None, c=None, frames=None) -> tuple:
        """The read coordinates a run uses: the explicit ones, else what
        ``run_where`` says (the slice on screen, the fixed ``run_z`` /
        ``run_c``, or None for where each ROI was drawn) and ``run_tp``
        (0-based frames, None = every frame).
        """
        if self.run_where == "screen":
            base_z, base_c = self.model.z, self.model.c
        elif self.run_where == "fixed":
            base_z, base_c = self.run_z, self.run_c
        else:
            base_z = base_c = None
        return (
            base_z if z is None else z,
            base_c if c is None else c,
            self.run_tp if frames is None else frames,
        )

    def _where_label(self) -> str:
        """Where a run reads its pixels, as the tooltips say it."""
        if self.run_where == "screen":
            where = f"slice on screen ({self._plane_label(self.z)})"
        elif self.run_where == "fixed":
            parts = []
            if self.run_z is not None:
                parts.append(self._axis_tag("z", self.run_z))
            if self.run_c is not None:
                parts.append(self._axis_tag("c", self.run_c))
            where = " ".join(parts) if parts else "as drawn"
        else:
            where = "as drawn"
        if self.run_tp is not None:
            window = index_window(self.run_tp)
            where += (
                f", frames {window[0] + 1}-{window[1]}"
                + (f"-{window[2]}" if window[2] > 1 else "")
                if window
                else f", {len(self.run_tp)} frames"
            )
        return where

    def _frame_select(self, movie: PlaneMovie | None, tp) -> PlaneMovie | None:
        """``movie`` over the 0-based frames ``tp`` it has; the movie itself
        for None, nothing usable, or every frame.
        """
        if movie is None or tp is None:
            return movie
        nt = int(movie.shape[0])
        kept = [int(t) for t in tp if 0 <= int(t) < nt]
        if not kept or kept == list(range(nt)):
            return movie
        return movie.select(kept)

    @property
    def trace_busy(self) -> bool:
        self._trace_threads = [t for t in self._trace_threads if t.is_alive()]
        return bool(self._trace_threads)

    @property
    def busy(self) -> bool:
        """Anything still working: runs in the manager or trace threads"""
        return self.trace_busy or self.manager.busy

    def has_traces(self) -> bool:
        """Anything the Traces tab could plot"""
        return bool(self.traces)

    def trace_disabled(self, index: int) -> str | None:
        """Why drawn ROI ``index`` cannot be traced right now, or None."""
        movie = self.movie()
        if movie is None or int(movie.shape[0]) < 2:
            return "no (T, Y, X) movie behind this view"
        return None

    def quick_trace(self, index: int):
        """Mean of the ROI's pixels per frame, on a thread tracked as a job."""
        self.trace_rois([index])

    def trace_rois(
        self,
        indices: list[int],
        *,
        z: int | None = None,
        c: int | None = None,
        frames=None,
    ):
        """Mean-trace drawn ROIs on one background job.

        Each mask is read from the z-plane / channel / frame window given
        (else ``run_z`` / ``run_c`` / ``run_tp``, else where it was
        drawn) and lands as a ``mean`` row of the trace table, replacing an
        earlier row at the same coordinates. The ROIs are traced one after
        another on a single thread, so running a long list never spawns a
        thread per ROI.
        """
        z, c, tp = self._coords(z, c, frames)
        work = []
        for target in self.model.targets(indices, z=z, c=c):
            movie = self._frame_select(
                self.movie(target.plane, z=target.z, c=target.c), tp
            )
            if movie is None:
                continue
            mask = self.store.labels[target.plane] == target.index + 1
            work.append((target, mask, feather_mask(mask), movie))
        if not work:
            if indices:
                self.status = "nothing to trace"
            return
        # traces taken at different binnings live on different time bases;
        # record it so the table can say so
        averaged = int(getattr(self.host, "frame_average", 1) or 1)
        description = (
            f"quick trace - ROI {work[0][0].index}"
            if len(work) == 1
            else f"quick trace - {len(work)} ROIs"
        )
        job = get_process_manager().start_job("roi_trace", description)
        self.status = f"{description} started"

        def run():
            frames_done = 0
            for n, (target, mask, weights, movie) in enumerate(work):
                try:
                    y = roi_trace(movie, mask, weights=weights)
                except Exception as error:  # noqa: BLE001 - reported on the job
                    self.logger.exception(f"quick trace - ROI {target.index} failed")
                    job.fail(f"{type(error).__name__}: {error}")
                    self._trace_results.put((target, None, str(error)))
                    return
                frames_done = int(y.size)
                trace = RoiTrace(
                    uid=target.uid,
                    z=target.z,
                    c=target.c,
                    engine="mean",
                    source="quick",
                    F=np.asarray(y, np.float32),
                    frame_average=averaged,
                    **_frames_of(movie),
                )
                self._trace_results.put((target, trace, None))
                job.set_progress((n + 1) / len(work), f"ROI {target.index}")
            job.done(
                f"{frames_done} frames" if len(work) == 1 else f"{len(work)} traces"
            )

        thread = threading.Thread(
            target=run, name=f"roi-trace-{work[0][0].index}", daemon=True
        )
        self._trace_threads.append(thread)
        thread.start()

    def trace_full(self, *, z: int | None = None, c: int | None = None, frames=None):
        """The whole frame as one mask: its mean per frame at the run
        coordinates (the slice on screen when nothing says otherwise), as a
        ``FULL_IMAGE`` row of the trace table, replacing an earlier read of
        the same slice.
        """
        z, c, tp = self._coords(z, c, frames)
        movie = self._frame_select(self.movie(z=z, c=c), tp)
        if movie is None or int(movie.shape[0]) < 2:
            self.status = "no (T, Y, X) movie behind this view"
            return
        averaged = int(getattr(self.host, "frame_average", 1) or 1)
        zz, cc = int(movie.z), int(movie.c)
        label = f"full image {self._axis_tag('z', zz)} {self._axis_tag('c', cc)}"
        job = get_process_manager().start_job("roi_trace", label)
        self.status = f"{label} started"
        mask = np.ones(movie.shape[1:], bool)

        def run():
            try:
                y = roi_trace(movie, mask)
            except Exception as error:  # noqa: BLE001 - reported on the job
                self.logger.exception(f"{label} failed")
                job.fail(f"{type(error).__name__}: {error}")
                self._trace_results.put((None, None, str(error)))
                return
            trace = RoiTrace(
                uid=0,
                member=slice_name(zz, cc),
                source=FULL_IMAGE,
                label=label,
                z=zz,
                c=cc,
                engine="mean",
                F=np.asarray(y, np.float32),
                frame_average=averaged,
                **_frames_of(movie),
            )
            self._trace_results.put((None, trace, None))
            job.done(f"{int(y.size)} frames")

        thread = threading.Thread(target=run, name="roi-trace-full", daemon=True)
        self._trace_threads.append(thread)
        thread.start()

    def _traces_changed(self):
        """Trace rows moved: drop stale stats and picks, redraw the lines, refresh the peaks."""
        self._trace_stats.clear()
        self._trace_display.clear()
        self.trace_sel &= set(self.traces.keys)
        if not self.trace_picked:
            self.trace_sel = set(self._selection_trace_keys())
        self._plot_lines_key = None
        if len(self.order.columns.get("peak", ())) == len(self.rows):
            self.order.columns["peak"] = self._row_peaks()
        self._footprints = None

    def _merge_run_traces(self, res, name: str) -> int:
        """Add one run's rows to the trace table under source ``name``,
        keyed by store uid: ``res.uids`` first, else legacy
        ``store_indices`` mapped through the current store. A run that did
        not record where it read gets the ROI's own z-plane and channel.
        Returns how many rows landed.
        """
        if res.F is None:
            return 0
        uids = res.uids
        if uids is None and res.store_indices is not None:
            uids = np.full(len(res.stat), -1, np.int64)
            for row, i in enumerate(res.store_indices):
                if 0 <= int(i) < self.n_rois:
                    uids[row] = self.store.rois[int(i)].uid
                else:
                    self.logger.info(
                        f"manual_roi: {name} row {row} maps to missing ROI {i}; skipped"
                    )
        if uids is None:
            return 0
        merged = 0
        for trace in result_traces(res, uids):
            index = self.store.uid_index(trace.uid)
            if index is None:
                continue
            trace.source = name
            if res.read_z is None:
                trace.z = self.store.roi_z(index)
            if res.read_c is None:
                trace.c = self.store.roi_c(index)
            self.traces.add(trace)
            self.trace_uid = trace.uid
            merged += 1
        return merged

    def pipeline_for(self, engine: str | None = None) -> str | None:
        """Which pipeline's parameters an engine runs on, or None for one
        that takes none (the numpy mean extractor).
        """
        engine = self.engine if engine is None else engine
        return engine if engine in ("masknmf", "suite2p") else None

    def masknmf_settings(self) -> dict | None:
        """The masknmf parameters set in the Process tab, or None for defaults.

        Runs started here go through the same settings the Process tab edits, so
        there is one place to change them rather than two that disagree.
        """
        return roi_runs.masknmf_settings(self.host)

    def suite2p_detection_settings(self) -> dict | None:
        """The Process tab's suite2p detection section, for unseeded discovery."""
        s2p = getattr(self.host, "s2p", None)
        if s2p is None:
            return None
        try:
            return (s2p.to_dict() or {}).get("detection") or None
        except Exception:
            self.logger.debug("suite2p settings unreadable", exc_info=True)
            return None

    def _pipeline_summary(self, kind: str) -> tuple[str, str]:
        """``(label, tooltip)`` for what a run of ``kind`` would use."""
        if kind == "masknmf":
            instances = getattr(self.host, "_pipeline_instances", None) or {}
            settings = getattr(instances.get("MaskNMF"), "settings", None)
            if settings is None:
                return "masknmf: defaults", (
                    "The Process tab has not built masknmf yet, so this runs on "
                    "its defaults. Open it to set registration, compression "
                    "and demixing parameters."
                )
            from mbo_utilities.gui.widgets.pipelines.masknmf import _collect_modified

            stages = " · ".join(
                f"{name}:{_STAGE_NAMES[int(getattr(section, attr, 1)) % 3]}"
                for name, section, attr in (
                    ("reg", settings.registration, "do_registration"),
                    ("pmd", settings.compression, "do_compression"),
                    ("demix", settings.demixing, "do_demixing"),
                )
            )
            changed = _collect_modified(settings)
            label = (
                f"masknmf: {len(changed)} changed" if changed else "masknmf: defaults"
            )
            detail = "\n".join(f"{n} = {v}  (default {d})" for n, v, d in changed[:12])
            return label, f"{stages}\n{detail}" if detail else stages
        s2p = getattr(self.host, "s2p", None)
        if s2p is None:
            return "suite2p: defaults", (
                "The Process tab has not been opened yet, so this runs on "
                "suite2p's defaults."
            )
        from mbo_utilities.gui.widgets.pipelines.settings import collect_modified_params

        stages = " · ".join(
            f"{name}:{_STAGE_NAMES[int(getattr(s2p, attr, 1)) % 3]}"
            for name, attr in (("reg", "do_registration"), ("detect", "do_detection"))
        )
        try:
            changed = collect_modified_params(
                s2p,
                getattr(self.host, "s2p_db", None),
                getattr(self.host, "s2p_extras", None),
            )
        except Exception:
            changed = []
        label = f"suite2p: {len(changed)} changed" if changed else "suite2p: defaults"
        detail = "\n".join(f"{row[0]} = {row[1]}" for row in changed[:12])
        return label, f"{stages}\n{detail}" if detail else stages

    def open_pipeline_params(self, kind: str):
        """Jump to the Process tab with ``kind`` selected - that is where these
        parameters are edited.
        """
        if self.host is None:
            self.status = "no Process tab to open"
            return
        self.host._selected_pipeline_name = (
            "MaskNMF" if kind == "masknmf" else "Suite2p"
        )
        self.host._force_run_tab = True
        self.status = f"{kind} parameters are in the Process tab"

    @property
    def run_prefix(self) -> str:
        """What a run dir's name starts with: ``rois_``, then the recording's
        tag when the file holds several (``rois_MSession_0_MUnit_3_``).
        """
        return f"{OUT_PREFIX}{self.tag}_" if self.tag else OUT_PREFIX

    def _run_out_dir(self, tag: str) -> Path | None:
        if self.fpath is None:
            self.status = "no data path to write beside"
            return None
        if any(
            r.tag == tag and r.job is not None and not r.finished
            for r in self.manager.runs
        ):
            self.status = f"{self.run_prefix}{tag} is still being written"
            return None
        return labels_path(self.fpath).parent / f"{self.run_prefix}{tag}"

    def _next_find_tag(self) -> str:
        base = labels_path(self.fpath).parent if self.fpath is not None else None
        used = {r.tag for r in self.manager.runs}
        n = 1
        while True:
            tag = f"find{n:02d}"
            if tag not in used and (
                base is None or not (base / f"{self.run_prefix}{tag}").exists()
            ):
                return tag
            n += 1

    def run_rois(
        self,
        indices: list[int],
        tag: str,
        *,
        z: int | None = None,
        c: int | None = None,
        frames=None,
        engine: str | None = None,
    ):
        """Send drawn ROIs through an extraction engine on the viewer's own
        array, writing ``rois_<tag>/`` beside the data.

        Each mask stays on the plane it was drawn on; its pixels come from
        ``z`` / ``c`` / ``frames`` (else the Process tab's ``run_z`` /
        ``run_c`` / ``run_tp``, else where the ROI was drawn). ROIs
        read from more than one z-plane or channel get one child dir each
        (``zplane02``, ``zplane02_ch01``). A finished run's rows replace
        earlier rows at the same coordinates with the same engine. The run
        closes over a store snapshot, so drawing on is safe while it works.
        """
        indices = [int(i) for i in indices if 0 <= int(i) < self.n_rois]
        if not indices:
            self.status = "nothing to run"
            return
        tag = (tag or "").strip() or DEFAULT_RUN_TAG
        out_dir = self._run_out_dir(tag)
        if out_dir is None:
            return
        engine = self.engine if engine is None else str(engine)
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}; one of {ENGINES}")
        z, c, tp = self._coords(z, c, frames)
        store = self.store.snapshot()
        groups: dict[tuple[int, int, int], list[int]] = {}
        for target in self.model.targets(indices, z=z, c=c):
            groups.setdefault((target.plane, target.z, target.c), []).append(
                target.index
            )
        movies = {}
        for key in groups:
            plane, zz, cc = key
            movie = self._frame_select(self.movie(plane, z=zz, c=cc), tp)
            if movie is None:
                self.status = "no (T, Y, X) movie behind this view"
                return
            movies[key] = movie
        many_c = len({cc for _p, _z, cc in groups}) > 1
        dests = {}
        for key in groups:
            _plane, zz, cc = key
            # one child per slice read, named in the filename vocabulary
            child = slice_name(zz, cc if many_c else None)
            dests[key] = out_dir if len(groups) == 1 else out_dir / child
        settings = self.masknmf_settings() if engine == "masknmf" else None
        logger = self.logger

        def fn(job):
            outs = []
            for i, (key, on_plane) in enumerate(groups.items()):
                plane, zz, cc = key
                job.set_progress(i / len(groups), slice_name(zz, cc))
                if engine == "masknmf":
                    out = demix_rois(
                        movies[key],
                        store,
                        on_plane,
                        z=plane,
                        c=cc,
                        out_dir=dests[key],
                        settings=settings,
                        tag=tag,
                        logger=logger,
                    )
                else:
                    out = extract_rois(
                        movies[key],
                        store,
                        on_plane,
                        z=plane,
                        c=cc,
                        out_dir=dests[key],
                        engine=engine,
                        tag=tag,
                        logger=logger,
                    )
                if out is not None:
                    outs.append(Path(out))
            return outs

        run = RoiRun(
            kind="demix" if engine == "masknmf" else "extract",
            tag=tag,
            description=f"{engine}: {len(indices)} ROI(s) -> {out_dir.name}",
            out_root=out_dir,
            planes=sorted({zz + 1 for _p, zz, _c in groups}),
        )
        self.manager.submit(run, fn, heavy=(engine == "masknmf"))
        self._run_error = None
        self.status = f"{run.description} started"

    def selection_indices(self) -> list[int]:
        """Drawn ROIs the selection covers: the group if there is one, else the selected ROI."""
        grouped = [k for si, k in self.buffer if si < 0]
        return grouped or ([self.selected] if self.selected >= 0 else [])

    def run_selection(self, indices: list[int] | None = None):
        """Run the selection through the engine the Process tab's ROIs pipeline is set to.

        One ROI goes to its own ``rois_roiNN/`` as the row button does; a group
        goes to the tab's tag in one job, as picking "selected" there does.
        """
        indices = self.selection_indices() if indices is None else indices
        if not indices:
            self.status = "select an ROI first"
            return
        if len(indices) == 1:
            self.run_roi(indices[0])
        else:
            self.run_rois(indices, self.effective_tag)

    def run_roi(self, index: int):
        """Run one drawn ROI into ``rois_roiNN/`` (the R tag, 1-based)."""
        self.run_rois(
            [index], DimensionTag(TAG_REGISTRY["R"], index + 1, None).to_string()
        )

    @property
    def effective_tag(self) -> str:
        """The tag a run will actually use: what was typed, else the default."""
        return (self.run_tag or "").strip() or DEFAULT_RUN_TAG

    def listed_drawn(self) -> list[int]:
        """Store indices of the drawn ROIs the table currently lists."""
        return [
            self.rows[int(r)][1] for r in self.order.order if self.rows[int(r)][0] < 0
        ]

    def run_in_view(self):
        """Run every drawn ROI the table currently lists."""
        self.run_rois(self.listed_drawn(), self.effective_tag)

    def trace_in_view(self):
        """Quick trace every drawn ROI the table currently lists."""
        self.trace_rois(self.listed_drawn())

    def discover_region(self, engine: str):
        """Detect ROIs inside ``self.region`` on the plane on screen; the
        region is consumed by the submit.
        """
        if self.region is None:
            self.status = "draw a region with r first"
            return
        tag = self._next_find_tag()
        out_dir = self._run_out_dir(tag)
        if out_dir is None:
            return
        box = self.region
        z = self.z
        src = self.movie(z)
        if src is None:
            self.status = "no (T, Y, X) movie behind this view"
            return
        c = src.c
        settings = (
            self.masknmf_settings()
            if engine == "masknmf"
            else self.suite2p_detection_settings()
        )
        logger = self.logger

        def fn(job):
            job.set_progress(0.05, f"{engine} in {box[0]}:{box[1]}, {box[2]}:{box[3]}")
            return discover_rois(
                src,
                box,
                engine=engine,
                z=z,
                c=c,
                out_dir=out_dir,
                settings=settings,
                tag=tag,
                logger=logger,
            )

        run = RoiRun(
            kind="discover",
            tag=tag,
            description=f"find ({engine}) -> {out_dir.name}",
            out_root=out_dir,
            box=box,
            planes=[z + 1],
        )
        self.manager.submit(run, fn, heavy=True)
        self.clear_region()
        self._run_error = None
        self.status = f"{run.description} started"

    def run_full_plane(
        self, kind: str, *, z: int | None = None, c: int | None = None, frames=None
    ):
        """Spawn a full suite2p / masknmf run of one z-plane as a detached
        worker: the plane, channel and frame window the run coordinates
        say (the slice on screen when nothing says otherwise). The Process
        tab's own suite2p / masknmf pipelines cover whole volumes.
        """
        if self.fpath is None:
            self.status = "no data path to run on"
            return
        z, c, tp = self._coords(z, c, frames)
        movie = self.movie(z=z, c=c)
        if movie is None:
            self.status = "no (T, Y, X) movie behind this view"
            return
        # workers take 1-based plane / channel and a 0-based frame list
        lsp = to_lsp_kwargs({"Z": [int(movie.z)], "C": [int(movie.c)]})
        plane = lsp["planes"][0]
        channel = lsp["channels"][0] if movie.nc > 1 else None
        selected = self._frame_select(movie, tp)
        tp_indices = None if selected is movie else selected.t_indices
        try:
            args = full_plane_args(
                kind,
                self.fpath,
                plane,
                self.iw,
                host=self.host,
                channel=channel,
                tp_indices=tp_indices,
            )
        except ValueError as e:
            self.status = str(e)
            return
        tag = slice_name(movie.z, movie.c if channel is not None else None)
        run = RoiRun(kind=kind, tag=tag, description=f"{kind} {tag}", planes=[plane])
        self.manager.spawn(run, kind, args)
        if run.pid is None:
            return
        run.out_dirs = [Path(args["output_dir"]) / unit_name("plane", plane)]
        self._registry_extra.append(
            {"path": str(run.out_dirs[0]), "kind": kind, "discarded": []}
        )
        self._save_registry()
        self.status = f"{run.description} started (pid {run.pid})"

    def _poll_jobs(self):
        """Drain finished traces and runs; called once per frame from the panel."""
        while True:
            try:
                target, trace, error = self._trace_results.get_nowait()
            except queue.Empty:
                break
            index = self.store.uid_index(target.uid) if target is not None else None
            if error is not None:
                shown = (
                    "full image"
                    if target is None
                    else (index if index is not None else f"uid {target.uid}")
                )
                self.status = f"trace for {shown if target is None else 'ROI ' + str(shown)} failed: {error}"
                continue
            if target is not None and index is None:
                continue  # deleted while the trace ran
            self.traces.add(trace)
            if trace.stands_for_roi:
                self.trace_uid = trace.uid
            elif (
                trace.source == FULL_IMAGE
                and self._selection_pair() is None
                and not self.trace_picked
            ):
                # nothing else on the plot: a whole-frame read shows itself
                self.trace_sel = {trace.key}
                self.trace_picked = True
                self._plot_lines_key = None
            self.status = (
                f"{trace.label or 'ROI ' + str(index)}: {trace.n_frames} frames"
            )
        for run, payload in self.manager.poll(get_process_manager()):
            if run.error is not None:
                self._run_error = f"{run.description} failed: {run.error}"
                continue
            if run.kind == "discover" and payload is None:
                self.status = f"{run.description}: nothing found in the region"
                continue
            if run.job is not None:
                if isinstance(payload, (str, Path)):
                    run.out_dirs = [Path(payload)]
                elif payload:
                    run.out_dirs = [Path(o) for o in payload]
                outs = [d for d in run.out_dirs if run_dir_complete(d)]
            else:
                # spawned pipelines may suffix the plane dir name, so
                # resolve the real dirs from disk instead of the guess
                outs = finished_dirs(run.out_root, run.planes) if run.out_root else []
                if outs:
                    guessed = {str(d) for d in run.out_dirs}
                    self._registry_extra = [
                        e for e in self._registry_extra if str(e["path"]) not in guessed
                    ]
                    run.out_dirs = outs
            loaded = sum(self.load_run(d) for d in outs)
            run.loaded = bool(loaded)
            if outs and loaded == len(outs):
                self._run_error = None
                names = ", ".join(d.name for d in outs)
                took = run.job.elapsed_str() if run.job is not None else ""
                self.logger.info(
                    f"roi run done: {names}" + (f" in {took}" if took else "")
                )
                self.status = f"done: {names}"
                # the run browser is gone: its timing and outputs belong on the
                # job, which the status button and process console already show
                if run.job is not None:
                    run.job.status_message = f"{names} · {took}"
            elif not outs:
                self.status = f"{run.description}: nothing written"
                if run.job is not None:
                    run.job.status_message = "nothing written"
        self._adopt_finished_runs()

    def _adopt_finished_runs(self):
        """Load the ROIs of pipeline runs this widget did not start.

        A suite2p / masknmf run launched from the Process tab writes its
        plane dirs beside the data like any other run, but nothing was
        watching for them: its components only showed up after a restart,
        when the widget rebuilds from the registry. Rows arrive with the
        run's own iscell, so accepted and rejected land in the table the
        same way a run started here does.
        """
        now = time.monotonic()
        if now - self._adopt_checked < ADOPT_INTERVAL_S:
            return
        self._adopt_checked = now
        if self.fpath is None:
            return
        root = labels_path(self.fpath).parent
        mine = {r.pid for r in self.manager.runs if r.pid is not None}
        loaded = {str(s.result.path) for s in self.derived} | self._results_loaded
        for info in get_process_manager().get_running():
            if (
                info.pid in mine
                or info.pid in self._adopted
                or info.status != "completed"
                or info.task_type not in ("suite2p", "masknmf", "vnoiser")
            ):
                continue
            self._adopted.add(info.pid)
            args = info.args or {}
            out = args.get("output_dir")
            if not out:
                continue
            out = Path(out)
            if root not in (out, *out.parents) and out not in root.parents:
                continue  # another dataset's run
            if info.task_type == "vnoiser":
                # a zarr-format vnoiser run leaves one results file in the PF folder
                from mbo_utilities.results import newest_results

                found = newest_results(out, "vnoiser")
                dirs = [found] if found is not None and str(found) not in loaded else []
            else:
                dirs = [
                    d
                    for d in finished_dirs(out, args.get("planes"))
                    if str(d) not in loaded
                ]
            # a volume run writes a dir per plane and this widget shows one:
            # the planes it cannot take are the Process tab's business, not
            # an error to put in front of the user here
            held = self._run_error
            took = [d for d in dirs if self.load_run(d)]
            self._run_error = held
            if took:
                names = ", ".join(d.name for d in took)
                self.logger.info(f"adopted {info.task_type} run: {names}")
                self.status = f"loaded {info.task_type}: {names}"
            elif dirs:
                self.logger.debug(
                    f"{info.task_type} run at {out} has no dir for this plane"
                )

    def handle_keys(self):
        """Masknmf's DEMIXING keys with its labeling keys and this tool's own (``ROI_KEYS``)."""
        io = imgui.get_io()
        if io.want_text_input:
            return
        claim_keys(_CLAIMED_KEYS)
        keys = ROI_KEYS
        if pressed(keys["delete"]):
            self.delete_selected()
        if pressed(keys["escape"]):
            if self.keybinds_open:
                self.keybinds_open = False
            elif self.region_selector is not None:
                self._drop_region()
            else:
                self.deselect()
        if pressed(keys["select_all"]):
            self.select_all()
        if pressed(keys["undo"]):
            self.undo()
        stride = 10 if io.key_shift else 1
        if pressed(keys["down"]):
            self.step(stride)
        if pressed(keys["up"]):
            self.step(-stride)
        if pressed(keys["right"]):
            self.set_frame(self.current_frame() + stride)
        if pressed(keys["left"]):
            self.set_frame(self.current_frame() - stride)
        if pressed(keys["masks"]):
            self.set_masks(not self.show_masks)
        if pressed(keys["contours"]):
            self.set_contours(not self.show_contours)
        if pressed(keys["rings"]):
            self.cycle_contour_shape()
        if pressed(keys["follow"]):
            self.toggle_follow()
        if pressed(keys["trace_follow"]) and self.trace_plot is not None:
            self.trace_plot.follow = not self.trace_plot.follow
        if pressed(keys["pixel_trace"]):
            self.set_pixel_traces(not self.pixel_traces)
        if pressed(keys["roi"]):
            if self.region_selector is not None and not self._drawing():
                self._commit_region()
            elif self.region_selector is None:
                self._start_region()
        if pressed(keys["poly"]):
            self._start_region()
        if pressed(keys["help"]):
            self.open_guide()
        if pressed(keys["keybinds"]):
            self.keybinds_open = not self.keybinds_open
        if pressed(keys["unlabeled"]):
            self.next_unlabeled()
        if pressed(keys["run"]):
            self.run_selection()
        if self.selected_derived is not None:
            if pressed(keys["promote"]):
                self.promote_derived(*self.selected_derived)
            if pressed(keys["discard"]):
                self.discard_derived(*self.selected_derived, advance=True)
            if pressed(keys["accept"]):
                self.set_accepted(*self.selected_derived)
        if self._selection_pair() is not None or self.buffer:
            picked = self.classes.hotkey_pressed()
            if picked is not None:
                self.assign_class(picked)

    @property
    def top_tab(self) -> str | None:
        """The top strip's selected panel: ``"traces"`` while the plot is up."""
        return self.strip.active

    def _frame(self):
        """Per-frame work the strip runs whatever tab is on top: background
        jobs, the region, the keys, and the tool's own windows.
        """
        self._poll_jobs()
        self._poll_region()
        self.handle_keys()
        self.keybinds_open = draw_keybinds_popup(
            ROI_KEYS, self.keybinds_open, "ROI keybinds"
        )
        self.summary.draw()

    def draw_rois(self):
        """The ROIs tab: masknmf's Tools panel. The Full FOV button and the
        guide and keybinds buttons, then the Curation and ROIs tabs, each a
        child that scrolls on its own.
        """
        if imgui.button(f"{fa.ICON_FA_IMAGE} Full FOV"):
            self.open_full_fov()
        tooltip(
            "the frame on screen and the ROI labels at full size: zoom, colormap, contrast, pixel values"
        )
        buttons_w = help_buttons_width("ROI Guide")
        imgui.same_line()
        if imgui.get_content_region_avail().x < buttons_w + em(0.6):
            imgui.new_line()
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - buttons_w
        )
        guide, self.keybinds_open = draw_help_buttons(
            False, self.keybinds_open, "ROI Guide"
        )
        if guide:
            self.open_guide()
        if imgui.begin_tab_bar("##roi_tools"):
            if imgui.begin_tab_item("Curation")[0]:
                imgui.begin_child("##roi_curation_tab")
                self._draw_curation()
                imgui.end_child()
                imgui.end_tab_item()
            if imgui.begin_tab_item("ROIs")[0]:
                imgui.begin_child("##roi_table_tab")
                self.draw_tab()
                imgui.end_child()
                imgui.end_tab_item()
            imgui.end_tab_bar()

    def _draw_labels(self, g):
        section("LABELS")
        g.row("new label")
        self.new_label, changed = draw_label_editor(
            self.classes, self.new_label, "_roi"
        )
        if changed:
            self._sync_store_from_classes()
            self._resync()
            self.refresh_overlay()
            self._autosave()
        if self.n_rois:
            done = int((self.classes.labels[: self.n_rois] >= 0).sum())
            g.row("labeled")
            imgui.text_colored(
                to_vec4(THEME.ok if done == self.n_rois else THEME.warn),
                f"{done} / {self.n_rois}",
            )
            help_mark("drawn ROIs with a label; u jumps to the next one without")
        if not self.classes.names:
            return
        g.row("classes")
        imgui.new_line()
        picked = self._draw_label_columns()
        if picked == UNLABEL_ALL:
            self.unlabel_all()
        elif picked is not None:
            self.assign_class(picked)

    def _draw_label_columns(self):
        """The unlabel actions pinned across the top, then one button per
        class, split into columns filled evenly.

        Class buttons carry two columns of their own, the count then the
        name, each left-aligned, so the names line up down the column
        instead of drifting with however wide the counts happen to be.
        """
        if not self.classes.names:
            return None
        picked = self._draw_unlabel_row()
        n = len(self.classes.names)
        gap, hint = em(0.8), em(2.0)
        # as many columns as the width holds at a readable button, three at most
        ncols = max(
            1,
            min(n, 3, int((imgui.get_content_region_avail().x + gap) // (em(8) + gap))),
        )
        per_col = -(-n // ncols)
        col_w = max(
            (imgui.get_content_region_avail().x - gap * (ncols - 1)) / ncols, em(5)
        )
        size = imgui.ImVec2(max(col_w - hint - em(0.7), em(3.0)), 0)
        count_w = max(
            imgui.calc_text_size(self._count_text(i)).x for i in range(n)
        ) + em(0.5)
        for c0 in range(0, n, per_col):
            if c0:
                imgui.same_line(0, gap)
            imgui.begin_group()
            for i in range(c0, min(c0 + per_col, n)):
                with label_button(self.classes.color(i)):
                    if imgui.button(f"##lab{i}", size):
                        picked = i
                self._draw_label_button_text(i, count_w)
                if i < 9:
                    imgui.same_line(0, 4)
                    imgui.text_disabled(f"({i + 1})")
            imgui.end_group()
        return picked

    def _count_text(self, i: int) -> str:
        """The count column of a class button. "n=3", not "(3)": the (1-9)
        that follows the button is the keybind, and a bare count beside it
        read as one too.
        """
        return f"n={self.classes.count(i)}"

    def _draw_label_button_text(self, i: int, count_w: float) -> None:
        """Paint one class button's two text columns over the button just
        drawn. imgui centres a button's own label, which left the names
        starting at a different x on every row.
        """
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        pad = imgui.get_style().frame_padding.x
        y = lo.y + (hi.y - lo.y - imgui.get_text_line_height()) * 0.5
        draw = imgui.get_window_draw_list()
        color = imgui.get_color_u32(imgui.Col_.text)
        # a long name is clipped to its button rather than bleeding into the
        # column beside it
        draw.push_clip_rect(
            imgui.ImVec2(lo.x, lo.y), imgui.ImVec2(hi.x - 1, hi.y), True
        )
        draw.add_text(imgui.ImVec2(lo.x + pad, y), color, self._count_text(i))
        draw.add_text(
            imgui.ImVec2(lo.x + pad + count_w, y), color, self.classes.names[i]
        )
        draw.pop_clip_rect()

    def _draw_unlabel_row(self):
        """Unlabel at the top left, unlabel all at the top right: two small
        buttons that stay put however many classes there are.
        """
        picked = None
        x0 = imgui.get_cursor_pos_x()
        avail = imgui.get_content_region_avail().x
        if imgui.small_button("unlabel##_roi"):
            picked = UNLABELED
        tooltip("Clear the selected ROI's label (0)")
        pad = imgui.get_style().frame_padding.x * 2
        right_w = imgui.calc_text_size("unlabel all").x + pad
        imgui.same_line()
        imgui.set_cursor_pos_x(max(x0 + avail - right_w, imgui.get_cursor_pos_x()))
        with button_colors(MTHEME.danger, MTHEME.danger_hover):
            if imgui.small_button("unlabel all##_roi"):
                picked = UNLABEL_ALL
        tooltip("Clear every label on this plane")
        return picked

    def _status_message(self) -> tuple[tuple, str]:
        if self._save_error is not None:
            return THEME.err, self._save_error
        if self._run_error is not None:
            return THEME.err, self._run_error
        active = self.manager.active
        if active:
            verbs = {"discover": "find", "extract": "extract", "demix": "masknmf"}
            names = ", ".join(
                f"{verbs.get(r.kind, r.kind)} "
                + (f"{self.run_prefix}{r.tag}" if r.job is not None else r.tag)
                for r in active
            )
            return THEME.warn, f"{len(active)} running: {names}"
        return THEME.text_dim, self.status

    def draw_tab(self):
        """Masknmf's Signals tab for ROIs: the filter, then the drawn, algo
        and pixel-average rows in one sortable table, the selection under it.
        """
        if self.order.range_column is not None:
            self._draw_filter()
        footer = imgui.get_frame_height_with_spacing() * 2.5
        if imgui.begin_child("##roi_table", imgui.ImVec2(0, -footer)):
            if self.rows or self.pixels:
                columns = self.columns
                formatters = {
                    name: partial(self._format_cell, name) for name in columns[1:]
                }
                self.scroll_to_selection = draw_roi_table(
                    self.order,
                    columns,
                    formatters,
                    self.scroll_to_selection,
                    table_id="manual_rois",
                    hidden={"ok", "peak"} - ({"ok"} if self.derived else set()),
                    cursor=self._selection_pair() is not None,
                    on_select=self._table_select,
                    is_grouped=self._row_grouped,
                    on_ctrl_select=self._table_ctrl,
                    on_shift_select=self._table_shift,
                    row_color=self._table_color,
                    prefix_rows=[
                        (("px", *pid), f"px {pid[1]},{pid[2]}") for pid in self.pixels
                    ],
                )
            else:
                imgui.text_disabled("no ROIs yet: draw (a) a region and add it (r)")
        imgui.end_child()
        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

    def _trace_label(self, trace: RoiTrace) -> str:
        """Legend text for one row: the engine, then where it was read when
        that is not where the ROI was drawn, then a frame window.
        """
        if not trace.stands_for_roi:
            return trace.name
        parts = [trace.engine]
        index = self.store.uid_index(trace.uid)
        if index is not None:
            if trace.z != self.store.roi_z(index):
                parts.append(self._axis_tag("z", trace.z))
            if trace.c != self.store.roi_c(index):
                parts.append(self._axis_tag("c", trace.c))
        if trace.frames is not None:
            start, stop, step = trace.frames
            parts.append(f"t{start + 1}-{stop}" + (f"-{step}" if step > 1 else ""))
        elif trace.extra.get("tp_indices"):
            parts.append(f"{len(trace.extra['tp_indices'])} frames")
        return " ".join(parts)

    def _selection_trace_keys(self) -> list[tuple]:
        """Trace-table keys of the selection; empty when it has none."""
        if self.selected_derived is not None:
            if self.selected_derived[0] >= len(self.derived):
                return []
            return self._member_keys(*self.selected_derived)
        # mid store event the selection can still name an ROI the store just dropped
        if not 0 <= self.selected < self.n_rois:
            return []
        return [t.key for t in self.traces.for_roi(self.store.rois[self.selected].uid)]

    def _sync_trace_sel(self):
        """Point the trace table's picks at the selection, so the rows it
        highlights are the ROI the image is showing.
        """
        self.trace_sel = set(self._selection_trace_keys())
        self.trace_picked = False
        self._plot_lines_key = None

    def _trace_target(self):
        """``(header, [(label, key), ...])`` the plot shows, or None."""
        got = self._plot_lines()
        if got is None:
            return None
        header, lines = got
        return header, [(label, key) for label, key, _rgb in lines]

    def _binning_tag(self, key) -> str:
        """`` x10`` when a trace was taken at a different frame averaging than
        the data now shows — those traces are on another time base.
        """
        trace = self.traces.get(key)
        taken = int(trace.frame_average if trace is not None else 1) or 1
        now = int(getattr(self.host, "frame_average", 1) or 1)
        return f" x{taken}" if taken != now and taken > 1 else ""

    def fs(self) -> float | None:
        """Raw sampling rate of the recording behind the view in Hz, or None.

        Read once from the array ``imread`` returned, under any frame
        averaging, since every ``TimeAxis`` applies the binning itself;
        without one the trace plot can only offer frame units.
        """
        if not self._fs_read:
            self._fs_read = True
            data = getattr(self.iw, "data", None)
            rate = getattr(base_array(data[0]), "fs", None) if data else None
            self._fs_value = float(rate) if rate else None
        return self._fs_value

    @property
    def kind(self) -> str | None:
        """The kind the plot shows of each row; None lets every row's pipeline pick."""
        return self._kind

    @kind.setter
    def kind(self, kind: str | None) -> None:
        if kind != self._kind:
            self._kind = kind
            self._redisplay()

    def kind_options(self, rows) -> tuple[str, ...]:
        """The kinds on offer for ``rows``: every kind any of them can show,
        in ``DISPLAY_KINDS`` order.
        """
        offered = {kind for trace in rows for kind in available_kinds(trace)}
        return tuple(kind for kind in DISPLAY_KINDS if kind in offered)

    def plot_y_label(self, rows) -> str:
        """The y axis label of what the rows show: one when they agree, else joined."""
        subtract, invert = self.deflection()
        labels = list(
            dict.fromkeys(y_label(t, self.kind, subtract, invert) for t in rows)
        )
        return " / ".join(label for label in labels if label) or DISPLAY_KINDS["dff"]

    def _plotted_rows(self, lines) -> list[RoiTrace]:
        return [
            t for t in (self.traces.get(key) for _label, key in lines) if t is not None
        ]

    def _redisplay(self) -> None:
        """The rows read differently now: drop the cached arrays, redraw the lines."""
        self._trace_display.clear()
        self._trace_stats.clear()
        self._trace_window_cache.clear()
        self._plot_lines_key = None

    def _window_spec(self) -> tuple[str, int]:
        """``(projection, size)`` of the viewer's window function.

        The preview trace gets the same window the image does, so what the
        plot shows is what the frame on screen shows.
        """
        host = self.host
        if host is None:
            return "mean", 1
        try:
            return str(host.proj), max(1, int(host.window_size))
        except Exception:
            return "mean", 1

    def _windowed(self, y):
        """``y`` under the viewer's rolling window; the raw array at size 1."""
        proj, size = self._window_spec()
        if y is None or size <= 1 or y.size < size:
            return y
        cached = self._trace_window_cache.get((id(y), proj, size))
        if cached is not None:
            return cached
        pad = (size - 1) // 2, size // 2
        padded = np.pad(y, pad, mode="edge")
        view = np.lib.stride_tricks.sliding_window_view(padded, size)
        func = {"max": np.max, "std": np.std}.get(proj, np.mean)
        out = np.ascontiguousarray(func(view, axis=-1), np.float32)
        # one entry per plotted array and window, so several lines never
        # recompute each other's window every frame
        if len(self._trace_window_cache) > 256:
            self._trace_window_cache.clear()
        self._trace_window_cache[(id(y), proj, size)] = out
        return out

    def deflection(self) -> tuple[bool, bool]:
        """The viewer's ``(Mean Subtraction, Invert Deflection)``, which the
        traces follow so the plot reads like the image.
        """
        return (
            bool(getattr(self.host, "mean_subtraction", False)),
            bool(getattr(self.host, "invert_deflection", False)),
        )

    def _display(self, key) -> np.ndarray | None:
        """The cached display array for one trace key, in the panel's kind,
        dF/F settings and the viewer's deflection (``annotation.display``);
        None when the row is gone or carries nothing.
        """
        deflection = self.deflection()
        if deflection != self._trace_deflection:
            self._trace_deflection = deflection
            self._redisplay()
        got = self._trace_display.get(key)
        if got is None:
            trace = self.traces.get(key)
            if trace is None:
                return None
            y = display_trace(trace, self.kind, self.dff, *deflection)
            if y is None:
                return None
            got = np.ascontiguousarray(y, np.float32)
            self._trace_display[key] = got
        return got

    def _set_by_name(self, name: str):
        for si, s in enumerate(self.derived):
            if s.name == name:
                return si, s
        return None

    def _key_to_pair(self, key) -> tuple[int, int] | None:
        """``(si, k)`` behind one trace key, or None when the ROI is gone
        (a results file's rows stand for no ROI).
        """
        trace = self.traces.get(key)
        if trace is None:
            return None
        if trace.stands_for_roi:
            index = self.store.uid_index(trace.uid)
            return (-1, index) if index is not None else None
        hit = self._set_by_name(trace.source)
        return (hit[0], int(trace.member)) if hit is not None else None

    def _trace_color(self, key) -> tuple[float, float, float] | None:
        """The mask color of the ROI behind one trace key, so plot lines and
        table rows match the overlay.
        """
        pair = self._key_to_pair(key)
        if pair is None:
            return None
        si, k = pair
        if si < 0:
            return tuple(v / 255.0 for v in self.store.roi_rgb(k))
        return component_color(self.derived[si], k)

    def _trace_shown(self, key) -> tuple[float, str]:
        """``(sort value, display text)`` for a key's roi column."""
        trace = self.traces.get(key)
        if trace is None:
            return float(1 << 30), "?"
        if trace.stands_for_roi:
            index = self.store.uid_index(trace.uid)
            if index is not None:
                return float(index), f"{index}"
            return float((1 << 30) + trace.uid), f"uid {trace.uid}"
        if self._set_by_name(trace.source) is None:
            # rows that stand for no ROI sort after the drawn ones, in table order
            return float((1 << 30) + self.traces.keys.index(key)), trace.name
        index = self._promoted.get((trace.source, trace.member))
        if index is not None:
            return float(index), f"{index}"
        return float((1 << 30) + trace.member), f"{trace.member}"

    def _plot_lines(self):
        """``(header, [(label, key, rgb), ...])`` the plot shows, None for nothing.

        Rows picked in the trace table when they are not the selection's
        own; else, with a group of two or more or any pixel average, one
        line per member in its group color (masknmf's group); else every
        trace of the selected ROI, in its mask color when it has one.
        """
        if not self.show_traces:
            return None
        if self.trace_picked and self.trace_sel:
            keys = [key for key in self.traces.keys if key in self.trace_sel]
            lines = []
            for i, key in enumerate(keys):
                trace = self.traces.get(key)
                _v, shown = self._trace_shown(key)
                label = (
                    f"{shown} · {self._trace_label(trace)}"
                    if trace.stands_for_roi
                    else f"{trace.source} · {shown}"
                )
                rgb = (
                    GROUP_COLORS[i % len(GROUP_COLORS)]
                    if len(keys) > 1
                    else self._trace_color(key)
                )
                lines.append((label, key, rgb))
            return (f"{len(lines)} selected", lines) if lines else None
        members = [*self.buffer, *self.pixel_group]
        if len(members) > 1 or self.pixel_group:
            colors = self._group_colors()
            lines = []
            for member in members:
                key = self._member_line(member)
                if key is None or key not in self.traces:
                    continue
                rgb = colors.get(member, MARKED_COLOR)
                if len(member) == 3:
                    label = f"px {member[1]},{member[2]}"
                elif member[0] < 0:
                    label = f"ROI {member[1]}"
                else:
                    label = f"{self.derived[member[0]].name} {member[1]}"
                lines.append((label, key, rgb))
            return (f"{len(members)} grouped", lines) if lines else None
        keys = [key for key in self._selection_trace_keys() if key in self.traces]
        if not keys:
            return None
        pair = self._selection_pair()
        row = self._row_index.get(pair)
        lines = []
        for i, key in enumerate(keys):
            if len(keys) == 1:
                rgb = self._row_rgb(row) if row is not None else self._trace_color(key)
            else:
                rgb = GROUP_COLORS[i % len(GROUP_COLORS)]
            lines.append((self._trace_label(self.traces.get(key)), key, rgb))
        if self.selected >= 0:
            header = f"ROI {self.selected}"
        else:
            si, k = self.selected_derived
            header = f"{self.derived[si].name} row {k}"
        return header, lines

    def _spike_target(self, rows) -> tuple | None:
        """``(path, unit, roi, c)`` of the run behind the one ROI plotted,
        when that ROI came from a run's results; None otherwise.

        An algo row is ROI ``k`` of the unit its set was read from (a run
        folder or plane dir, or ``<file>.zarr/<unit>``); a results row of the
        trace table carries its file, unit and ROI.
        """
        if self.selected < 0 and self.selected_derived is not None and not self.buffer:
            si, k = self.selected_derived
            path = Path(self.derived[si].result.path)
            if results_pipeline(path.parent) is not None:
                return (path.parent, path.name, str(k), None)
            return (path, None, str(k), None) if path.is_dir() else None
        if len(rows) == 1 and rows[0] is not None and "roi" in (rows[0].extra or {}):
            row = rows[0]
            return (row.path, row.extra["unit"], row.extra["roi"], row.c)
        return None

    def _open_spike_average(self, path, unit, roi, c) -> None:
        from mbo_utilities.gui.launch import LaunchedWindow
        from mbo_utilities.gui.spike_average_viewer import launch_spike_average

        pid, log_name = launch_spike_average(path, unit, roi, c)
        self._windows.append(LaunchedWindow(pid, log_name, "Spike average"))
        self.status = f"spike average of {roi} in its own window (PID {pid})"

    def draw_traces(self):
        """The Traces panel: a row of controls, the behavior raster, then
        masknmf's stacked trace panels (the motion correction over the
        traces) on the viewer's frames, the playhead bound to the viewer's t.
        """
        target = self._plot_lines()
        motion = self.motion if self.motion else None
        behavior = self.behavior if self.behavior else None
        imgui.begin_disabled(motion is None)
        changed, on = imgui.checkbox("MC", self.show_motion and motion is not None)
        if changed:
            self.show_motion = on
        imgui.end_disabled()
        tooltip(
            f"{motion.y_label}: the motion correction the whole recording went through, as a panel over the traces"
            if motion is not None
            else "This recording carries no motion correction."
        )
        imgui.same_line(0, em(0.8))
        imgui.begin_disabled(behavior is None)
        changed, on = imgui.checkbox(
            "Behavior", self.show_behavior and behavior is not None
        )
        if changed:
            self.show_behavior = on
        imgui.end_disabled()
        tooltip(
            f"{behavior.source}: what the animal did during the recording "
            f"({', '.join([*behavior.signals, *behavior.events, *behavior.epochs])}), over the traces, its epochs "
            "shaded behind them"
            if behavior is not None
            else "No behavior log was found for this recording: a file named after its subject and day, beside it or "
            "in a behavior folder."
        )
        rows = (
            []
            if target is None
            else [self.traces.get(key) for _label, key, _rgb in target[1]]
        )
        kinds = self.kind_options(rows)
        if kinds:
            imgui.same_line(0, em(0.8))
            shown = [displayed_kind(t, self.kind) for t in rows]
            current = self.kind if self.kind in kinds else shown[0]
            imgui.set_next_item_width(em(9))
            changed, sel = imgui.combo(
                "##trace_kind", kinds.index(current), list(kinds)
            )
            tooltip(
                "What each row shows, one line per row: dff (its pipeline's dF/F, or one computed here), raw (F "
                "alone), neuropil (Fneu alone), raw - neuropil (F minus the pipeline's share of Fneu), spikes, "
                "denoised, z-score. A row without that kind shows its pipeline's default."
            )
            if changed:
                self.kind = kinds[sel]
            if any(s == "dff" and t.norm is None for t, s in zip(rows, shown)):
                imgui.same_line(0, em(0.4))
                if imgui.small_button("dF/F##dff_settings"):
                    imgui.open_popup("##dff_settings")
                tooltip(
                    "The baseline of a dF/F computed here: from raw - neuropil when the row has a neuropil, from the "
                    "raw trace otherwise"
                )
                self._draw_dff_settings(rows)
        if callable(getattr(self.host, "reference_view", None)):
            imgui.same_line(0, em(0.8))
            if imgui.small_button("Reference image##traces"):
                self.host.reference_view()
            tooltip(
                "The picture this recording's lines or patches were drawn on, with them drawn and the slider's ROI "
                "thick. Click one there to select it."
            )
        spike_target = self._spike_target(rows)
        if spike_target is not None:
            imgui.same_line(0, em(0.8))
            if imgui.small_button("Spike average##traces"):
                self._open_spike_average(*spike_target)
            tooltip(
                "This ROI averaged around its spikes (its run's detected events, the events its curation accepts, "
                "or its trace over a threshold): the movie, the trace and every motion correction, in a window "
                "of its own"
            )
        for window in list(self._windows):
            state = window.poll()
            if state is not None:
                self._windows.remove(window)
                if state:
                    self.status = state
        imgui.same_line(0, em(0.8))
        if target is not None:
            proj, size = self._window_spec()
            window = f" · {proj} {size}" if size > 1 else ""
            imgui.text_disabled(
                (f"{self.unit} · " if self.unit else "")
                + f"{target[0]}, frame {self.current_frame()}{window}"
            )
            tooltip(
                "drag pans, scroll zooms, double-click fits, right-click for autofit and the x unit · shift+scroll "
                "zooms x only, alt+scroll zooms y only"
            )
        elif self.pending_traces is not None:
            self.pending_traces()
        else:
            imgui.text_disabled(
                "No traces shown. Click an ROI with traces, or draw one (a, r) with trace on draw."
            )
        show_motion = motion is not None and self.show_motion
        show_behavior = behavior is not None and self.show_behavior
        if target is None and not show_motion and not show_behavior:
            # nothing to plot: the panel is its row of controls, the plot empty for when it comes back
            self._traces_panel.height = int(imgui.get_frame_height_with_spacing())
            if (
                self.trace_plot is not None
                and self.trace_plot._lines[self.trace_plot.panels[-1]]
            ):
                self._plot_lines_key = None
                self.trace_plot.set(self.trace_plot.panels[-1], [])
                self._plot_mode([])
            return
        self._traces_panel.height = (
            PANEL_HEIGHT
            + (MOTION_PANEL_HEIGHT - PANEL_HEIGHT) * int(show_motion)
            + BEHAVIOR_PLOT_HEIGHT * int(show_behavior)
        )
        if self.strip.collapsed or self.tdim is None:
            return
        self._sync_plot(rows)
        plot = self.trace_plot
        nt = len(plot.x)
        lines_key = (
            None
            if target is None
            else tuple((label, key, rgb) for label, key, rgb in target[1]),
            self.kind,
            repr(self.dff),
            self.deflection(),
            self._window_spec(),
            self._plot_frames,
        )
        if lines_key != self._plot_lines_key:
            self._plot_lines_key = lines_key
            lines = []
            for label, key, rgb in [] if target is None else target[1]:
                y = self._display(key)
                if y is None:
                    continue
                lines.append((label, self._on_frames(key, self._windowed(y), nt), rgb))
            plot.set(plot.panels[-1], lines)
            self._plot_mode(lines)
        plot.frame = self.current_frame()
        if show_behavior:
            self._draw_behavior_row(behavior)
        moved = plot.draw()
        if moved is not None:
            self.playhead.seek(self.viewer_axis().seconds(moved), source="trace_plot")

    def _draw_dff_settings(self, rows) -> None:
        """The popup editing the panel's dF/F baseline; it starts from the
        first plotted row's pipeline settings and can go back to them.
        """
        if not imgui.begin_popup("##dff_settings"):
            return
        if self.dff is None:
            self.dff = replace(trace_profile(rows[0].engine).dff)
        d = self.dff
        imgui.set_next_item_width(em(9))
        changed, idx = imgui.combo(
            "baseline", DFF_METHODS.index(d.method), ["rolling max-min", "percentile"]
        )
        if changed:
            d.method = DFF_METHODS[idx]
        if d.method == "maxmin":
            imgui.set_next_item_width(em(6))
            moved, d.window_s = imgui.input_float(
                "window (s)", d.window_s, 0.5, 5.0, "%.1f"
            )
            d.window_s = max(0.1, d.window_s)
            changed |= moved
            imgui.set_next_item_width(em(6))
            moved, d.sigma_s = imgui.input_float(
                "smoothing (s)", d.sigma_s, 0.01, 0.1, "%.3f"
            )
            d.sigma_s = max(0.0, d.sigma_s)
            changed |= moved
            if not all(t.fs for t in rows):
                imgui.text_disabled("a row without a sampling rate uses the percentile")
        else:
            imgui.set_next_item_width(em(6))
            moved, d.percentile = imgui.slider_float(
                "percentile", d.percentile, 1.0, 50.0, "%.0f"
            )
            changed |= moved
        if imgui.small_button("pipeline defaults"):
            self.dff = None
            changed = True
        if changed:
            self._redisplay()
        imgui.end_popup()

    def _sorted_trace_rows(self) -> list[tuple]:
        """Trace-table keys in the order the table shows them, so stepping
        with the arrows walks what the user sees.
        """
        self._sync_trace_order(self._trace_rows())
        return [self._trace_keys[int(i)] for i in self.trace_order.order]

    def _trace_rows(self) -> list[tuple]:
        """The table's rows, in insertion order: every trace key, or only the
        ones placed on the slice on screen (drawn ROIs, loaded components,
        scanned lines and pixel averages read on the channel and z-plane
        shown; a results file's rows have no slice and always show).
        """
        keys = self.traces.keys
        if not self.traces_this_slice or self.store.nz <= 1:
            return keys
        z, c = self.slice.z, self.slice.c
        out = []
        for key in keys:
            trace = self.traces.get(key)
            placed = trace is not None and (
                trace.stands_for_roi
                or trace.source in (FULL_IMAGE, "pixel")
                or "line" in trace.extra
                or self._set_by_name(trace.source) is not None
            )
            if not placed or (trace.z == z and trace.c == c):
                out.append(key)
        return out

    def _trace_cells(self, key) -> tuple:
        """The text of one row's roi / z / c / engine / source columns."""
        trace = self.traces.get(key)
        _v, shown = self._trace_shown(key)
        if trace is None:
            return (shown, "", "", "", "")
        # a results file's rows have no slice to name
        placed = (
            trace.stands_for_roi
            or trace.source in (FULL_IMAGE, "pixel")
            or "line" in trace.extra
            or self._set_by_name(trace.source) is not None
        )
        if not placed:
            return (shown, "", "", trace.engine, trace.source)
        return (
            shown,
            f"{trace.z + 1}",
            f"{trace.c}",
            trace.engine,
            trace.source + self._binning_tag(key),
        )

    def _trace_stat(self, key) -> tuple[int, float, float, float]:
        """``(frames, mean, peak, snr)`` of the displayed trace,
        cached until the trace sets change; snr is peak over baseline in
        robust sd units.
        """
        got = self._trace_stats.get(key)
        if got is None:
            y = self._display(key)
            f = y if y is not None else np.zeros(0, np.float32)
            if f.size:
                med = float(np.median(f))
                mad = float(np.median(np.abs(f - med)))
                peak = float(f.max())
                snr = (peak - med) / (1.4826 * mad) if mad > 0 else 0.0
                got = (int(f.size), float(f.mean()), peak, snr)
            else:
                got = (0, 0.0, 0.0, 0.0)
            self._trace_stats[key] = got
        return got

    def delete_trace_row(self, key) -> None:
        """Delete one trace row. A drawn ROI keeps its mask (and its other
        rows); an algo component is its trace, so it is marked for deletion.
        """
        self.trace_sel.discard(key)
        pair = self._key_to_pair(key)
        if pair is not None and pair[0] >= 0:
            self.mark([pair], True)
            return
        if self.traces.remove(key) is not None:
            self.status = "deleted trace"

    def draw_trace_table(self, keys=None, table_id: str = "##trace_table"):
        """The Traces tab: every collected trace in masknmf's ROI table;
        click plots one, ctrl+click several, shift+click a range. ``keys``
        narrows the rows to those (the Process tab shows the ROIs it is
        about to run).
        """
        rows = self._trace_rows()
        if keys is not None:
            wanted = set(keys)
            rows = [key for key in rows if key in wanted]
        if not rows:
            imgui.text_disabled(
                "No traces yet. Draw an ROI with trace on draw, run the selection (shift+t), or load a run."
            )
            return
        self._sync_trace_order(rows)
        imgui.text_disabled(f"{len(rows)} traces · {len(self.trace_sel)} picked")
        imgui.same_line(0, em(0.8))
        if imgui.small_button(f"{fa.ICON_FA_LIST_CHECK}##plot_all"):
            self.trace_sel = set(rows)
            self.trace_picked = True
            self._plot_lines_key = None
        tooltip("Plot every listed trace")
        imgui.same_line(0, em(0.4))
        if imgui.small_button(f"{fa.ICON_FA_ERASER}##unplot"):
            self._sync_trace_sel()
        tooltip(
            "Unpick every trace: the plot goes back to the selection; the rows stay"
        )
        imgui.same_line(0, em(0.4))
        picked = [key for key in rows if key in self.trace_sel]
        imgui.begin_disabled(not picked)
        with button_colors(MTHEME.danger, MTHEME.danger_hover):
            delete = imgui.small_button(f"{fa.ICON_FA_TRASH}##delete_picked")
        imgui.end_disabled()
        tooltip(
            f"Delete the {len(picked)} picked trace(s) (a drawn ROI keeps its mask; an algo component is marked for "
            "deletion)"
        )
        if self.store.nz > 1:
            imgui.same_line(0, em(0.8))
            changed, every = imgui.checkbox("all slices", not self.traces_this_slice)
            if changed:
                self.traces_this_slice = not every
            tooltip(
                "List the rows of every channel and z-plane, not only the slice on screen"
            )
        columns = (
            "id",
            *(self._trace_header(name) for name, _sortable, _hidden in TRACE_COLUMNS),
        )
        hidden = {
            self._trace_header(name) for name, _sortable, hide in TRACE_COLUMNS if hide
        }
        if self.store.axis_size("z") <= 1:
            hidden.add(self._trace_header("z"))
        formatters = {
            self._trace_header(name): partial(self._trace_cell, name)
            for name, _sortable, _hidden in TRACE_COLUMNS
        }
        draw_roi_table(
            self.trace_order,
            columns,
            formatters,
            False,
            table_id=table_id,
            hidden=hidden,
            cursor=False,
            on_select=self._trace_table_select,
            is_grouped=lambda i: self._trace_keys[i] in self.trace_sel,
            on_ctrl_select=self._trace_table_ctrl,
            on_shift_select=self._trace_table_shift,
            row_color=lambda i: self._trace_color(self._trace_keys[i]),
            row_label=lambda i: self._trace_shown(self._trace_keys[i])[1],
        )
        picked_one = [key for key in rows if key in self.trace_sel]
        if len(picked_one) == 1:
            trace = self.traces.get(picked_one[0])
            pos = self._line_position(trace) if trace is not None else {}
            if pos.get("start_um") is not None:
                (x0, y0), (x1, y1) = pos["start_um"][:2], pos["end_um"][:2]
                imgui.text_disabled(
                    f"line ({x0:.0f}, {y0:.0f}) -> ({x1:.0f}, {y1:.0f}) um, {pos['length_um']:.1f} um long"
                    + (
                        f", {pos['sample_um']:.2f} um per sample"
                        if pos.get("sample_um")
                        else ""
                    )
                )
        if delete:
            for key in picked:
                self.delete_trace_row(key)
            self.status = f"deleted {len(picked)} traces"

    def _row_peaks(self) -> np.ndarray:
        """The peak of each row's displayed traces, NaN for a row without any."""
        peaks = np.full(len(self.rows), np.nan)
        for row, (si, k) in enumerate(self.rows):
            # mid store event the rows can still name an ROI the store just dropped
            if si < 0 and k >= self.n_rois or si >= len(self.derived):
                continue
            if si < 0:
                keys = [t.key for t in self.traces.for_roi(self.store.rois[k].uid)]
            else:
                keys = self._member_keys(si, k)
            values = [self._trace_stat(key)[2] for key in keys if key in self.traces]
            if values:
                peaks[row] = max(values)
        return peaks

    def _member_keys(self, si: int, k: int) -> list[tuple]:
        """Trace keys of algo row ``k`` of set ``si``: its own row, else its promoted ROI's."""
        s = self.derived[si]
        key = ("member", s.name, k)
        if key in self.traces:
            return [key]
        index = self._promoted.get((s.name, k))
        if index is None:
            return []
        return [t.key for t in self.traces.for_roi(self.store.rois[index].uid)]

    def _format_cell(self, name: str, item) -> str:
        """One cell of the ROIs table; ``item`` is a row, or a pixel average's key."""
        if isinstance(item, tuple):
            trace = self.traces.get(self.pixels.get(item))
            peak = "" if trace is None else f"{self._trace_stat(trace.key)[2]:.3g}"
            side = 2 * PIXEL_RADIUS + 1
            return {"source": "pixel", "area": f"{side * side}", "peak": peak}.get(
                name, ""
            )
        si, k = self.rows[item]
        if name == "label":
            return self.classes.name_of(item) if self.classes.labels[item] >= 0 else ""
        if name == "source":
            # the algorithm, not the run name: the source filter lists run names
            if si < 0:
                return "drawn"
            s = self.derived[si]
            algo = getattr(s.result, "algo", "") or s.name
            return f"{algo} · promoted" if (s.name, k) in self._promoted else algo
        if name == "z":
            z = self.store.rois[k].plane if si < 0 else self.derived[si].result.z
            return self._plane_label(z)
        if name == "ok":
            return "" if si < 0 else ("yes" if self.derived[si].accepted[k] else "no")
        if name == "del":
            return "x" if self.order.columns["del"][item] else ""
        value = self.order.columns[name][item]
        if np.isnan(value):
            return ""
        return f"{int(value)}" if name == "area" else f"{float(value):.3g}"

    def _drawing(self) -> bool:
        """The region has the pointer: its vertices are being placed or dragged."""
        return (
            self.region_selector is not None
            and self.region_selector._move_info.mode is not None
        )

    def _start_region(self):
        """Draw (a): arm a polygon on the image, or drop the one there is (the selection stays)."""
        if self.region_selector is not None:
            self._drop_region()
            return
        self.filter_select = False
        self._snapshot()
        self.region_selector = self.iw.graphics[0].add_polygon_selector(
            fill_color=(0.0, 0.0, 0.0, 0.0),
            edge_color=MTHEME.accent[:3],
            vertex_color=MTHEME.accent[:3],
            edge_thickness=2,
            vertex_size=8,
        )
        self.status = "click on the image to add points; click the first point to close"

    def _poll_region(self):
        """The region selects the ROIs in view on this slice whose centers it
        holds, following it as it is drawn or dragged. Left with under three
        vertices, or once the selection is edited by hand, it is dropped.
        """
        selector = self.region_selector
        if selector is None:
            return
        polygon = np.asarray(selector.selection)[:, :2]
        moving = selector._move_info.mode is not None
        if polygon.shape[0] < 3:
            if not moving:
                self._drop_region()
            return
        x0, y0 = np.floor(polygon.min(axis=0)).astype(int)
        x1, y1 = np.ceil(polygon.max(axis=0)).astype(int)
        self.region = (
            int(np.clip(y0, 0, self.ny - 1)),
            int(np.clip(y1, 1, self.ny)),
            int(np.clip(x0, 0, self.nx - 1)),
            int(np.clip(x1, 1, self.nx)),
        )
        if (
            self._region_hits is not None
            and not moving
            and self.buffer != self._region_hits
        ):
            self._drop_region()
            return
        key = (polygon.tobytes(), self.region_outside, self.order.range_limits, moving)
        if key == self._region_key:
            return
        self._region_key = key
        view = [
            int(row) for row in self.order.order if self._row_plane(int(row)) == self.z
        ]
        inside = [point_in_polygon(self._row_center(row), polygon) for row in view]
        hits = [
            self.rows[row]
            for row, hit in zip(view, inside)
            if hit != self.region_outside
        ]
        if hits != self._region_hits:
            self._region_hits = hits
            self.buffer = list(hits)
            self.selected = -1
            self.selected_derived = None
            if hits:
                si, k = hits[-1]
                if si < 0:
                    self.selected = k
                else:
                    self.selected_derived = (si, k)
                self.order.goto(self._row_index[hits[-1]])
            where = "outside" if self.region_outside else "inside"
            self.status = f"region: {len(hits)} ROI(s) {where}; add it as an ROI (r), esc drops it"
            self.refresh_overlay()

    def _drop_region(self):
        selector, self.region_selector = self.region_selector, None
        self._region_key = None
        self._region_hits = None
        if selector is None:
            return
        if selector._move_info.mode is not None:
            selector._end_move_mode()
        try:
            self.subplot.delete_graphic(selector)
        except (KeyError, ValueError):
            pass

    def _commit_region(self):
        """Add ROI (r): the settled region becomes a drawn ROI on the slice on screen, and the selection."""
        selector = self.region_selector
        if selector is None or selector._move_info.mode is not None:
            return
        points = np.asarray(selector.selection)[:, :2]
        self._drop_region()
        self.buffer_clear()
        self.add_roi(points)

    def _row_plane(self, row: int) -> int:
        si, k = self.rows[row]
        return self.store.rois[k].plane if si < 0 else self.derived[si].result.z

    def _row_footprint(self, row: int) -> tuple:
        """``(ypix, xpix, lam)`` of one table row."""
        si, k = self.rows[row]
        if si < 0:
            return self._feather(k)
        stat_row = self.derived[si].result.stat[k]
        lam = stat_row.get("lam")
        if lam is None:
            lam = np.ones(len(stat_row["ypix"]), np.float32)
        return (
            np.asarray(stat_row["ypix"], np.int32),
            np.asarray(stat_row["xpix"], np.int32),
            np.asarray(lam, np.float32),
        )

    def _row_center(self, row: int) -> tuple[float, float]:
        """``(x, y)`` center of one row's footprint, where the region tests it."""
        ypix, xpix, _lam = self._row_footprint(row)
        if not len(ypix):
            return (-1.0, -1.0)
        return float(np.mean(xpix)) + 0.5, float(np.mean(ypix)) + 0.5

    def _row_rgb(self, row: int) -> tuple[float, float, float]:
        """One row's color in 0-1, shared by its mask, its table row and its
        trace: its group color in a group of two or more, red when marked,
        else its own.
        """
        group = self._group_colors()
        pair = self.rows[row]
        if pair in group:
            return group[pair]
        if self.order.columns["del"][row]:
            return MARKED_COLOR
        si, k = pair
        if si < 0:
            return tuple(v / 255.0 for v in self.store.roi_rgb(k))
        return tuple(float(v) for v in component_color(self.derived[si], k))

    def _group_colors(self) -> dict:
        """One contrasting color per group member, shared by its trace, mask and table row (masknmf's)."""
        members = [*self.buffer, *self.pixel_group]
        if len(members) < 2:
            return {}
        return {m: GROUP_COLORS[i % len(GROUP_COLORS)] for i, m in enumerate(members)}

    def _pointer_down(self, ev):
        if getattr(ev, "button", 1) != 1:
            return
        self._press_drawn = self._drawing()
        self._press = (ev.x, ev.y)

    def _pointer_up(self, ev):
        """A click on the image picks: no drag, no region taking the pointer, no imgui window over it."""
        press, self._press = self._press, None
        if getattr(ev, "button", 1) != 1 or press is None or self._press_drawn:
            return
        if abs(ev.x - press[0]) + abs(ev.y - press[1]) > CLICK_SLOP or self._drawing():
            return
        if (
            imgui.get_current_context() is not None
            and imgui.get_io().want_capture_mouse
        ):
            return
        pos = self.subplot.map_screen_to_world((ev.x, ev.y))
        if pos is None:
            return
        mods = frozenset(getattr(ev, "modifiers", ()) or ())
        self._pick(int(pos[1]), int(pos[0]), mods)

    def _selection_pair(self) -> tuple[int, int] | None:
        if self.selected >= 0:
            return (-1, self.selected)
        return self.selected_derived

    def select_pair(self, si: int, k: int):
        """Select one row by its ``(set, row)`` pair; ``si`` -1 is a drawn ROI."""
        if si < 0:
            self.select_roi(k)
        else:
            self.select_derived(si, k)

    def add_pixel(self, row: int, col: int, toggle: bool = False):
        """A pixel average (p): the movie's 5x5 mean around ``(row, col)`` on
        the slice on screen, joining the group and the plot; ctrl on one
        already grouped takes it out.
        """
        pid = (self.z, int(row), int(col))
        if toggle and pid in self.pixel_group:
            self.pixel_group.remove(pid)
            self.active_pixel = None
            self._plot_lines_key = None
            self.refresh_overlay()
            return
        if pid not in self.pixels:
            key = self._trace_pixel(int(row), int(col))
            if key is None:
                return
            self._snapshot()
            self.pixels[pid] = key
            self.pixels.move_to_end(pid, last=False)
        self._seed_buffer()
        if pid not in self.pixel_group:
            self.pixel_group.append(pid)
        self.active_pixel = pid
        self._plot_lines_key = None
        self.refresh_overlay()

    def _trace_pixel(self, row: int, col: int) -> tuple | None:
        """Start the 5x5 average around one pixel on a background job; returns the row's key."""
        movie = self.movie()
        if movie is None or int(movie.shape[0]) < 2:
            self.status = "no (T, Y, X) movie behind this view"
            return None
        mask = np.zeros(movie.shape[1:], bool)
        mask[
            max(row - PIXEL_RADIUS, 0) : row + PIXEL_RADIUS + 1,
            max(col - PIXEL_RADIUS, 0) : col + PIXEL_RADIUS + 1,
        ] = True
        zz, cc = int(movie.z), int(movie.c)
        label = f"px {row},{col}"
        trace = RoiTrace(
            uid=0,
            member=f"{row},{col} {slice_name(zz, cc)}",
            source="pixel",
            label=label,
            z=zz,
            c=cc,
            engine="mean",
            frame_average=int(getattr(self.host, "frame_average", 1) or 1),
            extra={"pixel": (row, col)},
        )
        job = get_process_manager().start_job("roi_trace", f"pixel average {label}")
        thread = threading.Thread(
            target=self._run_pixel,
            args=(movie, mask, trace, job),
            name="roi-trace-pixel",
            daemon=True,
        )
        self._trace_threads.append(thread)
        thread.start()
        return trace.key

    def _run_pixel(self, movie, mask, trace: RoiTrace, job):
        try:
            y = roi_trace(movie, mask)
        except Exception as error:  # noqa: BLE001 - reported on the job
            self.logger.exception(f"{trace.label} failed")
            job.fail(f"{type(error).__name__}: {error}")
            self._trace_results.put((None, None, str(error)))
            return
        trace.F = np.asarray(y, np.float32)
        trace.frames = _frames_of(movie)["frames"]
        self._trace_results.put((None, trace, None))
        job.done(f"{int(y.size)} frames")

    def drop_pixel(self, pid: tuple):
        """Take one pixel average off the plot, the table and the trace rows."""
        key = self.pixels.pop(pid, None)
        if pid in self.pixel_group:
            self.pixel_group.remove(pid)
        if self.active_pixel == pid:
            self.active_pixel = None
        if key is not None:
            self.traces.remove(key)
        self._plot_lines_key = None

    def set_pixel_traces(self, on: bool):
        """Pixel traces (p); turning them off drops every pixel average."""
        self.pixel_traces = bool(on)
        if on:
            return
        for pid in list(self.pixels):
            self.drop_pixel(pid)
        self.refresh_overlay()

    def _table_shift(self, item):
        if isinstance(item, tuple):
            pid = item[1:]
            self._seed_buffer()
            if pid not in self.pixel_group:
                self.pixel_group.append(pid)
            self.active_pixel = pid
            self._refresh_group_view()
        else:
            self.buffer_extend_to(item)

    def mark(self, pairs, on: bool, record: bool = True):
        """Mark algo rows for deletion, or unmark them: they stay listed,
        pinned at the top of the table and red on the image, until unmarked.
        """
        pairs = [(int(si), int(k)) for si, k in pairs if si >= 0]
        if not pairs:
            return
        if record:
            self._push_undo({"kind": "mark", "pairs": pairs, "on": on})
        for si, k in pairs:
            if on:
                self.derived[si].discarded.add(k)
            else:
                self.derived[si].discarded.discard(k)
        self._resync()
        self.refresh_overlay()
        self._save_registry()
        self.status = f"{'marked' if on else 'unmarked'} {len(pairs)} algo ROI(s)"

    def deselect(self):
        """Esc: drop the selection, the group and the pixel averages."""
        if self.buffer or self.pixels or self._selection_pair() is not None:
            self._snapshot()
        for pid in list(self.pixels):
            self.drop_pixel(pid)
        self.buffer = []
        self.pixel_group = []
        self.filter_select = False
        self.select_roi(-1)

    def select_all(self):
        """Ctrl+a: group every row the table shows."""
        rows = [int(r) for r in self.order.order]
        if not rows:
            return
        self.buffer = [self.rows[r] for r in rows]
        self.select_row(rows[-1])
        self._refresh_group_view()
        self.status = f"{len(rows)} in group"

    def _selection_state(self) -> dict:
        return {
            "selected": self.selected,
            "selected_derived": self.selected_derived,
            "buffer": list(self.buffer),
            "pixel_group": list(self.pixel_group),
            "pixels": OrderedDict(self.pixels),
            "active_pixel": self.active_pixel,
        }

    def _push_undo(self, step: dict):
        step["selection"] = self._selection_state()
        self._undo.append(step)
        del self._undo[:-_UNDO_DEPTH]

    def _snapshot(self):
        """Push the selection for ctrl+z (masknmf's snapshot before a deselect)."""
        self._push_undo({"kind": "selection"})

    def undo(self):
        """Ctrl+z: back one step; a deleted ROI comes back with its traces, a drawn one goes."""
        if not self._undo:
            return
        step = self._undo.pop()
        self._drop_region()
        kind = step["kind"]
        if kind == "add":
            index = self.store.uid_index(step["uid"])
            if index is not None:
                self.delete_roi(index, record=False)
        elif kind == "delete":
            record = step["record"]
            mask = np.zeros((self.ny, self.nx), bool)
            mask[step["mask"]] = True
            min_pixels, self.store.min_pixels = self.store.min_pixels, 1
            try:
                index = self.store.add_roi(record.plane, mask, source=record.source)
            finally:
                self.store.min_pixels = min_pixels
            if index is not None:
                roi = self.store.rois[index]
                roi.uid, roi.class_index, roi.note, roi.color = (
                    record.uid,
                    record.class_index,
                    record.note,
                    record.color,
                )
                for trace in step["traces"]:
                    self.traces.add(trace)
                self._resync()
                self._autosave()
        elif kind == "mark":
            live = [(si, k) for si, k in step["pairs"] if si < len(self.derived)]
            self.mark(live, not step["on"], record=False)
        state = step["selection"]
        for pid in list(self.pixels):
            if pid not in state["pixels"]:
                self.drop_pixel(pid)
        self.pixels = OrderedDict(
            (pid, key)
            for pid, key in state["pixels"].items()
            if key in self.traces or pid in self.pixels
        )
        self.pixel_group = [pid for pid in state["pixel_group"] if pid in self.pixels]
        self.active_pixel = (
            state["active_pixel"] if state["active_pixel"] in self.pixels else None
        )
        self.buffer = [pair for pair in state["buffer"] if pair in self._row_index]
        if state["selected_derived"] in self._row_index:
            self.select_derived(*state["selected_derived"])
        else:
            self.select_roi(state["selected"])
        self._refresh_group_view()
        self.status = f"undone, {len(self._undo)} more"

    def set_masks(self, show: bool):
        """Every mask (m); the selection's has its own switch."""
        self.show_masks = bool(show)
        self.refresh_overlay()

    def set_contours(self, show: bool):
        """Every other ROI's contour (c)."""
        self.show_contours = bool(show)
        self.refresh_overlay()

    def cycle_contour_shape(self):
        """Contours along each ROI's border, or a ring around it (o)."""
        self.contour_shape = CONTOUR_SHAPES[
            (CONTOUR_SHAPES.index(self.contour_shape) + 1) % len(CONTOUR_SHAPES)
        ]
        self.refresh_overlay()
        self.status = f"contours: {self.contour_shape}"

    def open_guide(self):
        """The guide (h): this tool's page of the app's help viewer."""
        from mbo_utilities.gui._help_viewer import ROI_DOC, docs_for

        host = self.host
        if host is None:
            return
        names = [filename for _title, filename in docs_for(host)]
        if ROI_DOC in names:
            host._help_selected_doc = names.index(ROI_DOC)
        host._show_help_popup = True

    def _draw_curation(self):
        """Masknmf's Curation tab: OVERLAY, SELECTION, then this tool's LABELS and RUN."""
        g = grid(_CAPTIONS)
        # sliders: seven tenths of the panel, or what their row has left before the (?) mark
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        slider_w = min(0.7 * imgui.get_window_width(), right - g.cell_x[0] - g.mark_w)
        self._draw_overlay(g, slider_w)
        self._draw_selection()
        self._draw_labels(g)
        self._draw_run(g)
        imgui.spacing()
        imgui.push_text_wrap_pos(0)
        if self.drawing:
            imgui.text_disabled("click to add points; click the first point to close")
        else:
            color, text = self._status_message()
            imgui.text_colored(to_vec4(color), text)
        imgui.pop_text_wrap_pos()
        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

    def _draw_overlay(self, g, slider_w: float):
        section("OVERLAY")
        rows = (
            ("masks", "show_masks", "opacity", "every ROI's mask at this opacity (m)"),
            (
                "sel masks",
                "show_selected_masks",
                "selected_opacity",
                "the selected and grouped masks at this opacity, with a white rim",
            ),
            (
                "contours",
                "show_contours",
                "contour_opacity",
                "every other ROI's contour at this opacity (c; o rings them)",
            ),
            (
                "sel contours",
                "show_selected_contours",
                "selected_contour_opacity",
                "the selected and grouped contours, in their mask's color at this opacity",
            ),
        )
        for caption, show, opacity, tip in rows:
            changed, on = imgui.checkbox(caption, getattr(self, show))
            if changed:
                setattr(self, show, on)
                self.refresh_overlay()
            g.cell(0)
            imgui.set_next_item_width(slider_w)
            changed, value = imgui.slider_float(
                f"##{opacity}", getattr(self, opacity), 0.05, 1.0, "%.2f"
            )
            if changed:
                setattr(self, opacity, value)
                self.refresh_overlay()
            help_mark(tip)
            if caption == "masks":
                g.row("weighting")
                flipped, self.masks_by_peak = draw_switch(
                    "roi_weighting",
                    self.masks_by_peak,
                    "own peak",
                    "signal peak",
                    self.show_masks,
                )
                if flipped:
                    self.refresh_overlay()
                help_mark(
                    "own peak: every mask solid; signal peak: faint where the ROI's trace is weak",
                    g.cell_x[0] + slider_w + em(0.3),
                )
        g.row("color by")
        imgui.set_next_item_width(g.w)
        changed, index = imgui.combo(
            "##color_by", COLOR_BY.index(self.color_by), list(COLOR_BY)
        )
        if changed:
            self.set_color_by(COLOR_BY[index])
        g.cell(1)
        imgui.begin_disabled(self.color_by == COLOR_BY[0])
        imgui.set_next_item_width(g.w)
        changed, index = imgui.combo(
            "##color_cmap", COLORMAPS.index(self.color_cmap), list(COLORMAPS)
        )
        if changed:
            self.set_color_by(self.color_by, COLORMAPS[index])
        imgui.end_disabled()
        help_mark(
            "color the drawn ROIs by their class, z-plane or channel (a color per level), their area or the peak of "
            "their traces (a gradient); roi id: class / hue colors"
        )
        g.row("traces")
        buttons = (
            (
                fa.ICON_FA_ARROWS_LEFT_RIGHT_TO_LINE,
                self.trace_plot is not None and self.trace_plot.follow,
                "trace_follow",
                "Center: keep the current frame in the middle of the traces as the movie plays or the slider moves, "
                "the zoom kept; near either end of the recording the view stops at that end (t)",
            ),
            (
                fa.ICON_FA_EYE_DROPPER,
                self.pixel_traces,
                "pixel_traces",
                "Quick pixel trace: click an empty pixel to add the movie's 5x5 average there to the plot, grouped "
                "with whatever is shown, and to the top of the ROIs table; delete drops it (p)",
            ),
            (
                fa.ICON_FA_CHART_LINE,
                self.show_traces,
                "show_traces",
                "Show selected traces: plot whatever is selected, every trace of one ROI, or one line per group "
                "member. Off, selecting only highlights, however big the selection",
            ),
            (
                fa.ICON_FA_PENCIL,
                self.auto_trace,
                "auto_trace",
                "Trace on draw: trace every ROI the moment it is added, its mean where the RUN section points it",
            ),
        )
        for i, (icon, on, action, tip) in enumerate(buttons):
            if i:
                imgui.same_line(0, g.gap)
            with button_colors(MTHEME.accent, MTHEME.accent, (0.05, 0.05, 0.05), on=on):
                clicked = imgui.button(
                    f"{icon}##roi_traces_{i}", imgui.ImVec2(em(3.2), 0)
                )
            tooltip(tip)
            if not clicked:
                continue
            if action == "trace_follow" and self.trace_plot is not None:
                self.trace_plot.follow = not self.trace_plot.follow
            elif action == "pixel_traces":
                self.set_pixel_traces(not self.pixel_traces)
            elif action == "show_traces":
                self.show_traces = not self.show_traces
                self._plot_lines_key = None
            elif action == "auto_trace":
                self.auto_trace = not self.auto_trace

    def _draw_selection(self):
        """Masknmf's SELECTION: one centered row of icon buttons, the region's side switch under it."""
        section("SELECTION")
        drawing = self.region_selector is not None
        settled = drawing and not self._drawing()
        pair = self._selection_pair()
        derived = [p for p in self.buffer if p[0] >= 0] or (
            [pair] if pair is not None and pair[0] >= 0 else []
        )
        nothing = pair is None and self.active_pixel is None and not self.buffer
        unmark = (
            bool(derived)
            and self.selected < 0
            and self.active_pixel is None
            and all(k in self.derived[si].discarded for si, k in derived)
        )
        gap, avail = em(0.6), imgui.get_content_region_avail().x
        w = min((avail - 6 * gap) / 7, em(3.2))
        size = imgui.ImVec2(w, imgui.get_frame_height() * 1.2)
        imgui.dummy(imgui.ImVec2(0, em(0.4)))
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + max((avail - 7 * w - 6 * gap) / 2, 0)
        )
        with button_colors(
            MTHEME.accent, MTHEME.accent, (0.05, 0.05, 0.05), on=self.follow
        ):
            if imgui.button(f"{fa.ICON_FA_LOCATION_CROSSHAIRS}##roi_center", size):
                self.toggle_follow()
        tooltip(
            "Center: the image on the selected ROI, following it as the selection moves; labeling then advances (f)"
        )
        imgui.same_line(0, gap)
        with button_colors(
            MTHEME.emphasis, MTHEME.emphasis_hover, (0.05, 0.05, 0.05), on=drawing
        ):
            if imgui.button(f"{fa.ICON_FA_DRAW_POLYGON}##roi_draw", size):
                self._start_region()
        tooltip(
            "Draw is on: click again or esc to drop the region, the selection stays (a)"
            if drawing
            else "Draw: a polygon on the image selects every ROI in view whose center is on the switch's side of it, "
            "live as it is drawn and dragged; Add ROI keeps it (a)"
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(not settled)
        if imgui.button(f"{fa.ICON_FA_PLUS}##roi_add", size):
            self._commit_region()
        imgui.end_disabled()
        tooltip(
            "Add ROI: keep the drawn region as a drawn ROI on the slice on screen (r)"
            + ("" if settled else "; draw one first")
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(nothing)
        with button_colors(MTHEME.danger, MTHEME.danger_hover, on=not unmark):
            if imgui.button(f"{fa.ICON_FA_TRASH}##roi_delete", size):
                self.delete_selected()
        imgui.end_disabled()
        key = ROI_KEYS["delete"].label
        tooltip(
            f"Unmark: the {len(derived)} selected algo ROI(s) stay ({key})"
            if unmark
            else f"Delete: the selected drawn ROI, or mark the {len(derived)} selected algo ROI(s) for deletion; a "
            f"pixel average is dropped ({key})"
        )
        imgui.same_line(0, gap)
        promotable = (
            self.selected_derived is not None
            and self.promoted_index(*self.selected_derived) is None
        )
        imgui.begin_disabled(not promotable)
        if imgui.button(f"{fa.ICON_FA_ARROW_UP}##roi_promote", size):
            self.promote_derived(*self.selected_derived)
        imgui.end_disabled()
        tooltip(
            "Promote: copy the selected algo ROI into the drawn ROIs, then step to the next (y)"
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(self.selected_derived is None)
        if imgui.button(f"{fa.ICON_FA_CHECK}##roi_accept", size):
            self.set_accepted(*self.selected_derived)
        imgui.end_disabled()
        tooltip(
            "Accept / reject the selected algo ROI, written to its run's iscell (x)"
        )
        imgui.same_line(0, gap)
        if imgui.button(f"{fa.ICON_FA_FLOPPY_DISK}##roi_save", size):
            self.save()
        tooltip(
            f"Save: the drawn ROIs to {self._save_target()} now; they autosave there after every change"
            if self._writer is not None
            else "Save: the drawn ROIs are kept in memory only; open a file to autosave beside it"
        )
        imgui.dummy(imgui.ImVec2(0, em(0.4)))
        row_w = (
            switch_width("inside", "outside") + em(0.3) + imgui.calc_text_size("(?)").x
        )
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x()
            + max((imgui.get_content_region_avail().x - row_w) / 2, 0)
        )
        flipped, self.region_outside = draw_switch(
            "roi_region", self.region_outside, "inside", "outside", drawing
        )
        if flipped:
            self._region_key = None
        help_mark(
            "which side of the region is selected, lit while it drives the selection"
        )
        if self.selected >= 0 and len(self.buffer) <= 1:
            imgui.set_next_item_width(-1)
            changed, self._note_buf = imgui.input_text_with_hint(
                "##roi_note", "note", self._note_buf
            )
            if changed:
                self.store.set_note(self.selected, self._note_buf)
            if imgui.is_item_deactivated_after_edit():
                self._autosave()

    def _draw_run(self, g):
        """RUN: the engine and where it reads, the run buttons, the tag."""
        section("RUN")
        g.row("engine")
        imgui.set_next_item_width(g.w)
        changed, index = imgui.combo(
            "##roi_engine", ENGINES.index(self.engine), list(ENGINES)
        )
        if changed:
            self.engine = ENGINES[index]
        g.cell(1)
        imgui.set_next_item_width(g.w)
        where = self.run_where if self.run_where in RUN_WHERE else RUN_WHERE[0]
        changed, index = imgui.combo(
            "##roi_where", RUN_WHERE.index(where), ["as drawn", "on screen"]
        )
        if changed:
            self.run_where = RUN_WHERE[index]
        help_mark(f"{ENGINE_HELP[self.engine]}\nReads each mask {self._where_label()}.")
        g.row("run")
        indices = self.selection_indices()
        listed = self.listed_drawn()
        no_path = self.fpath is None
        w = imgui.get_frame_height() * 1.6
        imgui.begin_disabled(no_path or not indices)
        if imgui.button(f"{fa.ICON_FA_PLAY}##roi_run_sel", imgui.ImVec2(w, 0)):
            self.run_selection(indices)
        imgui.end_disabled()
        tooltip(
            "no data path to write beside"
            if no_path
            else f"Run the {len(indices)} selected ROI(s) through {self.engine} -> "
            f"{self.run_prefix}{self.effective_tag}/ (shift+t)"
            if indices
            else "select a drawn ROI first"
        )
        imgui.same_line(0, g.gap)
        imgui.begin_disabled(no_path or not listed)
        if imgui.button(f"{fa.ICON_FA_FORWARD}##roi_run_all", imgui.ImVec2(w, 0)):
            self.run_in_view()
        imgui.end_disabled()
        tooltip(
            f"Run all {len(listed)} drawn ROIs the table lists through {self.engine}"
        )
        imgui.same_line(0, g.gap)
        why = "no drawn ROIs listed" if not listed else self.trace_disabled(listed[0])
        imgui.begin_disabled(why is not None)
        if imgui.button(f"{fa.ICON_FA_CHART_LINE}##roi_trace_all", imgui.ImVec2(w, 0)):
            self.trace_in_view()
        imgui.end_disabled()
        tooltip(
            why
            or f"Quick trace all {len(listed)} listed ROIs: their means ({self._where_label()})"
        )
        imgui.same_line(0, g.gap)
        marked = sum(len(s.discarded) for s in self.derived)
        right_aligned_text(f"{self.n_rois} drawn, {marked} marked")
        g.row("options")
        imgui.set_next_item_width(g.w)
        _changed, self.run_tag = imgui.input_text_with_hint(
            "##roi_run_tag", DEFAULT_RUN_TAG, self.run_tag
        )
        help_mark(f"the run's tag: it writes {self.run_prefix}<tag>/ beside the data")

    def _selection_status(self) -> str:
        """Masknmf's footer: what is grouped or selected, else how to start."""
        if len(self.buffer) > 1 or self.pixel_group:
            drawn = sorted(k for si, k in self.buffer if si < 0)
            algo = [self._row_index[p] for p in self.buffer if p[0] >= 0]
            shown = f"{len(drawn)} drawn" if len(drawn) > 12 else f"drawn {drawn}"
            pixels = [f"{r},{c}" for _z, r, c in self.pixel_group]
            return f"{len(self.buffer) + len(pixels)} grouped: {shown}, algo rows {algo}, pixel avgs {pixels}"
        if self.selected >= 0:
            return f"ROI {self.selected} selected"
        if self.selected_derived is not None:
            si, k = self.selected_derived
            s = self.derived[si]
            return f"{s.name} row {k} selected" + (
                " (marked for deletion)" if k in s.discarded else ""
            )
        return "click an ROI to see its traces; draw (a) a region and add it (r) for a new one"

    def _table_color(self, item):
        if isinstance(item, tuple):
            return self._group_colors().get(item[1:], MARKED_COLOR)
        return self._row_rgb(item)

    def _filter_pairs(self) -> set:
        """The rows on the filter switch's side of the range."""
        order = self.order
        values = np.asarray(order.columns[order.range_column], np.float64)
        inside = (values >= order.range_limits[0]) & (values <= order.range_limits[1])
        side = np.isfinite(values) & ~inside if self.filter_outside else inside
        return {self.rows[int(row)] for row in np.flatnonzero(side)}

    def _draw_filter(self):
        """Masknmf's filter over the table: its column, apply with its side
        switch, the range, then this tool's label, source and slice filters.
        """
        order = self.order
        g = grid(_CAPTIONS)
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        slider_w = min(0.7 * imgui.get_window_width(), right - g.cell_x[0] - g.mark_w)
        g.row("filter")
        imgui.set_next_item_width(g.w)
        glow = self.filter_select
        if glow:
            for color, value in (
                (imgui.Col_.frame_bg, MTHEME.emphasis),
                (imgui.Col_.frame_bg_hovered, MTHEME.emphasis_hover),
                (imgui.Col_.button, MTHEME.emphasis),
                (imgui.Col_.button_hovered, MTHEME.emphasis_hover),
                (imgui.Col_.text, (0.05, 0.05, 0.05)),
            ):
                imgui.push_style_color(color, to_vec4(value))
        opened = imgui.begin_combo("##roi_range_column", order.range_column)
        if glow:
            imgui.pop_style_color(5)
        picked = None
        if opened:
            for name in order.columns:
                if imgui.selectable(name, name == order.range_column)[0]:
                    picked = name
            imgui.end_combo()
        changed = False
        if picked is not None and picked != order.range_column:
            order.set_range_column(picked)
            changed = True
            self.filter_select = False
        tooltip(
            "Filter column: the range below spans it (del is 0 or 1); picking one puts the range back at its full span"
        )
        imgui.same_line(0, g.gap)
        inner = imgui.get_style().item_inner_spacing.x
        need = (
            imgui.get_frame_height()
            + inner
            + imgui.calc_text_size("apply").x
            + g.gap
            + switch_width("inside", "outside")
            + g.mark_w
        )
        if imgui.get_content_region_avail().x < need:
            # a narrow panel: apply and its switch on a line of their own, under the combo
            imgui.new_line()
            imgui.set_cursor_pos_x(g.cell_x[0])
        if self.filter_select and set(self.buffer) != self._filter_pairs():
            self.filter_select = False
        toggled, self.filter_select = imgui.checkbox("apply", self.filter_select)
        if toggled and self.filter_select:
            self._drop_region()
        tooltip(
            "Apply to selection: the rows on the switch's side of the range become the group, as ctrl+a does for "
            "the table, and follow the lines as they move; editing the selection by hand switches this off"
        )
        imgui.same_line(0, g.gap)
        flipped, self.filter_outside = draw_switch(
            "roi_filter", self.filter_outside, "inside", "outside", self.filter_select
        )
        help_mark(
            "which side of the range the filter selects, lit while it drives the selection"
        )
        imgui.same_line(0, g.gap)
        right_aligned_text(f"{len(order.order)} / {order.n_items}")
        g.row("range")
        moved = draw_range_filter(order, "_roi", slider_w)
        tooltip(
            "Range: drag a line to move it, double-click for the full span; the table shows what is inside"
        )
        changed |= moved
        g.row("label")
        names = ["all", "unlabeled", *self.classes.names]
        imgui.set_next_item_width(g.w)
        picked, index = imgui.combo("##roi_label_filter", order.filter_label + 2, names)
        if picked:
            order.filter_label = index - 2
            changed = True
        g.cell(1)
        sources = ["all", "drawn", *(s.name for s in self.derived)]
        current = 0 if order.source is None else order.source + 1
        imgui.set_next_item_width(g.w)
        picked, index = imgui.combo(
            "##roi_source_filter", min(current, len(sources) - 1), sources
        )
        if picked:
            order.source = None if index == 0 else index - 1
            changed = True
        help_mark(
            "only the rows with this label, and from this source: drawn by hand, or a loaded run"
        )
        if self.store.nz > 1:
            g.row("slice")
            on = order.plane is not None
            picked, on = imgui.checkbox(
                f"this plane ({self._plane_label(self.z)} of {self.store.nz})", on
            )
            if picked:
                order.plane = self.z if on else None
                changed = True
        if changed:
            order.rebuild()
        if self.filter_select and (toggled or moved or flipped or changed):
            pairs = self._filter_pairs()
            self.buffer = [
                self.rows[int(row)]
                for row in order.order
                if self.rows[int(row)] in pairs
            ]
            self._refresh_group_view()
        imgui.dummy(imgui.ImVec2(0, em(0.2)))

    def _member_line(self, member) -> tuple | None:
        """The one trace key a group member plots: a pixel average's row,
        an algo row's own, a drawn ROI's newest.
        """
        if len(member) == 3:
            return self.pixels.get(member)
        si, k = member
        keys = (
            self._member_keys(si, k)
            if si >= 0
            else [t.key for t in self.traces.for_roi(self.store.rois[k].uid)]
        )
        return keys[-1] if keys else None

    def _plot_axis(self) -> tuple[int, np.ndarray | None]:
        """``(frames, seconds of each frame or None)``: the viewer's T axis, what the plot spans."""
        nt = 1
        if self.tdim is not None:
            try:
                nt = max(int(self.iw.data[0].shape[0]), 1)
            except (AttributeError, IndexError, TypeError):
                nt = 1
        if not self.fs() or nt < 2:
            return nt, None
        axis = self.viewer_axis()
        return nt, np.arange(nt, dtype=np.float64) / axis.per_second + axis.offset

    def _on_frames(self, key, y: np.ndarray, nt: int) -> np.ndarray:
        """One row's samples where the viewer's frames sit: as they are when
        the row is on the viewer's clock, else interpolated, NaN outside the
        stretch it covers (a frame window, another rate or binning).
        """
        src = self.trace_axis(self.traces.get(key))
        dst = self.viewer_axis()
        if len(y) == nt and src == dst:
            return y
        xs = np.arange(len(y), dtype=np.float64) / src.per_second + src.offset
        at = np.arange(nt, dtype=np.float64) / dst.per_second + dst.offset
        return np.interp(
            at, xs, y.astype(np.float64), left=np.nan, right=np.nan
        ).astype(np.float32)

    def _motion_lines(self, nt: int) -> list:
        """The recording's motion shifts on the plane on screen, on the viewer's frames."""
        motion = self.motion.motion if self.motion else None
        if motion is None:
            return []
        at = np.arange(nt, dtype=np.float64) / self.viewer_axis().per_second
        lines = []
        for label, (t, shift) in motion.traces.items():
            if motion.planes.get(label, self.slice.z) != self.slice.z:
                continue
            y = np.interp(
                at,
                np.asarray(t, np.float64),
                np.asarray(shift, np.float64),
                left=np.nan,
                right=np.nan,
            )
            lines.append(
                (
                    label,
                    y.astype(np.float32),
                    MOTION_COLORS.get(label[0], (0.8, 0.8, 0.8)),
                )
            )
        return lines

    def _sync_plot(self, rows) -> None:
        """Build or reset the trace plot for the frames and panels on screen, then put its lines in."""
        nt, timings = self._plot_axis()
        motion = bool(self.motion) and self.show_motion
        traces_panel = self.plot_y_label(rows) if rows else "traces"
        panels = ((self.motion.y_label,) if motion else ()) + (traces_panel,)
        key = (
            nt,
            None if timings is None else float(timings[-1]),
            panels,
            self.show_behavior,
        )
        if self.trace_plot is None:
            # no autofit, as in masknmf: the zoom set on one ROI's traces is kept while selecting others
            self.trace_plot = TracePlot(panels, nt, timings, autofit=False)
            # seconds whenever the data has a rate; timings equal to the frames are frames
            self.trace_plot._use_time = self.trace_plot._time is not None
        elif key != self._plot_frames:
            self.trace_plot.reset(panels, nt, timings)
            # a rate that stays keeps the unit picked
            if (
                timings is None
                or self._plot_frames is None
                or self._plot_frames[1] is None
            ):
                self.trace_plot._use_time = self.trace_plot._time is not None
        if key != self._plot_frames:
            self._plot_frames = key
            self._motion_key = None
            self._plot_lines_key = None
            if self.behavior and self.show_behavior:
                axis = self.viewer_axis()
                for i, (name, spans) in enumerate(
                    (self.behavior.behavior.epochs or {}).items()
                ):
                    spans = np.asarray(spans, np.float64)
                    if not len(spans):
                        continue
                    frames = np.rint(spans * axis.per_second).astype(np.int64)
                    self.trace_plot.span(
                        name,
                        frames[:, 0],
                        frames[:, 1],
                        EPOCH_COLORS[i % len(EPOCH_COLORS)],
                    )
        if motion and self._motion_key != self.slice.z:
            self._motion_key = self.slice.z
            self.trace_plot.set(panels[0], self._motion_lines(nt))

    def _plot_mode(self, lines) -> None:
        self.trace_plot.background = _TRACE_MODES["selection" if lines else "normal"]

    def _draw_behavior_row(self, behavior: BehaviorPlot) -> None:
        """The behavior raster over the trace panels, its x range the traces'."""
        plot = self.trace_plot
        axis = self.viewer_axis()
        x_per_second = (
            1.0 if plot._use_time and plot._time is not None else axis.per_second
        )
        span = plot._x_span
        moved, held = behavior.draw(
            "##roi_behavior_plot",
            BEHAVIOR_PLOT_HEIGHT - imgui.get_style().item_spacing.y,
            cursor=float(plot.x[plot.frame]),
            cursor_id=2,
            x_per_second=x_per_second,
            x_label="",
            x_axis=False,
            x_limits=span,
        )
        if held and moved is not None:
            self.playhead.seek(float(moved) / x_per_second, source="behavior_plot")

    def _sync_trace_order(self, keys: list[tuple]) -> None:
        """The trace table's order over ``keys``, its sort kept when they change."""
        if keys == self._trace_keys and self.trace_order.n_items == len(keys):
            return
        self._trace_keys = list(keys)
        stats = [self._trace_stat(key) for key in keys]
        cells = [self._trace_cells(key) for key in keys]
        columns = {
            self._trace_header("z"): np.array(
                [float(c[1]) if c[1] else np.nan for c in cells]
            ),
            self._trace_header("c"): np.array(
                [float(c[2]) if c[2] else np.nan for c in cells]
            ),
            "frames": np.array([s[0] for s in stats], np.float64),
            "peak": np.array([s[2] for s in stats], np.float64),
        }
        order = self.trace_order
        order.columns = columns
        order.n_items = len(keys)
        if order.sort_by not in columns:
            order.sort_by = None
        order.rebuild()

    def _trace_header(self, name: str) -> str:
        """A trace-table column's header: the data's own name for the z and c axes (``ROI``, ``Channel``)."""
        return self.axis_label(name) if name in ("z", "c") else name

    def _trace_cell(self, name: str, item: int) -> str:
        key = self._trace_keys[item]
        shown, z_text, c_text, engine, source = self._trace_cells(key)
        if name == "z":
            return z_text
        if name == "c":
            return c_text
        if name == "source":
            return source
        if name == "engine":
            return engine
        n, _mean, peak, _snr = self._trace_stat(key)
        return f"{n}" if name == "frames" else f"{peak:.3g}"

    def _trace_table_select(self, item: int):
        self.select_trace(self._trace_keys[item])

    def _trace_table_ctrl(self, item: int):
        self.toggle_trace(self._trace_keys[item])

    def _trace_table_shift(self, item: int):
        """Every row between the last picked one and ``item``, in the order shown."""
        view = [int(i) for i in self.trace_order.order]
        picked = [
            view.index(i)
            for i, key in enumerate(self._trace_keys)
            if key in self.trace_sel and i in view
        ]
        if item not in view:
            return
        to = view.index(item)
        start = picked[-1] if picked else to
        for pos in range(min(start, to), max(start, to) + 1):
            self.trace_sel.add(self._trace_keys[view[pos]])
        self.trace_picked = True
        self._plot_lines_key = None


def attach_roi_widget(parent: Any, focus: bool = False) -> ManualRoiWidget | None:
    """Turn the ROI widget on for a ``PreviewDataWidget``; ROIs and runs from
    an earlier toggle this session are adopted. Returns None (logged) when it
    cannot be built.
    """
    widget = getattr(parent, "manual_roi", None)
    if widget is not None:
        widget.focus_tab = widget.focus_tab or focus
        return widget
    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
    try:
        widget = ManualRoiWidget(
            parent.image_widget,
            fpath,
            label_names=DEFAULT_LABEL_NAMES,
            store=getattr(parent, "_manual_roi_store", None),
            runs=getattr(parent, "_manual_roi_runs", None),
            strip=getattr(parent, "top_strip", None),
            host=parent,
        )
    except Exception:
        parent.logger.warning("manual ROI widget unavailable", exc_info=True)
        parent.manual_roi = None
        return None
    widget.focus_tab = focus
    parent.manual_roi = widget
    return widget


def detach_roi_widget(parent: Any) -> None:
    """Turn the ROI widget off, keeping its store and runs for the next toggle."""
    widget = getattr(parent, "manual_roi", None)
    if widget is None:
        return
    parent._manual_roi_store = widget.store
    parent._manual_roi_runs = widget.park_runs()
    widget.close()
    parent.manual_roi = None
