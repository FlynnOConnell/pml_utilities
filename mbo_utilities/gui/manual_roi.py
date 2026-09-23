"""Manual ROI drawing + labeling widget for the viewer.

Toggled from ``Widgets > Manual ROI Labeling`` in the preview GUI (``mbo
<path> --widget manualroi`` opens with it on). The ROIs tab of the
right-hand widget (``widgets/tabs.py``) holds the control sections -
NAVIGATE, DRAW, VIEW, LABELS, each gated by its Widgets-menu subwidget
toggle - over a status row and the combined ROI table; the Traces tab holds
the trace table. The trace plot and its controls are a panel on the
figure's top strip, which has the width for it. No tab or panel is ever
selected for the user. Running ROIs is the Process tab's business: its ``ROIs``
pipeline (``widgets/pipelines/rois.py``) reads this widget's model. The
table, label set, stroke capture, overlay compositing and theme are the
shared widgets from ``mbo_utilities.gui.imgui``.

The state is a :class:`~mbo_utilities.annotation.RoiModel`: the
``RoiLabelStore`` (one ``(P, Y, X)`` uint16 label volume, 0 background, ROI
``i`` is ``i + 1``, so masks never overlap; each ROI keeps a persistent
``uid`` and a ``source`` naming where it came from), the
``RoiTraceTable`` (one row per measurement, keyed by ROI uid, z-plane,
channel and engine) and the slider position. The widget subscribes to the
model's events and redraws from them, so the same model drives the
Process tab's ROI pipeline without either widget polling the other.

Arm "Add ROI" (a), drag a closed stroke around a cell, release and the
enclosed pixels become a mask; with ``auto_trace`` on (the default) its
mean trace is computed at once and the Traces panel shows it. A stroke
lands on the exact slice the viewer shows: every scrolling dim except time
(z, channel, any extra slider) keys its own plane of masks, and flipping a
slider swaps the overlays, table filter and traces to that slice's ROIs.
Data without scroll sliders degrades to a single plane. Annotations
autosave next to the data as an OME-NGFF-style labels zarr
(``manual_labels.zarr``, see ``mbo_utilities.annotation``) and are restored
from it on relaunch. A file holding several recordings (a ``.mesc``) gets
one per recording (``manual_labels_MSession_0_MUnit_3.zarr``; the run
registry and run dirs likewise), and switching recordings swaps the whole
widget, so the ROIs, runs and traces on screen are always the shown one's.

Runs read an ROI's mask where it was drawn and its pixels wherever the
Process tab points them (``run_z`` / ``run_c`` / ``run_tp``, "as
drawn" by default), through :meth:`run_rois`, writing ``rois_<tag>/``
beside the data; a run of the same ROI at the same coordinates with the
same engine replaces its row. Region detection (r) and full-plane suite2p
/ masknmf runs load their outputs as derived sets - a second overlay plus
rows in the table - whose components can be promoted into the drawn store
(y) or discarded (n). Loaded runs are remembered in a ``roi_runs.json``
sidecar and restored on relaunch.

Masks draw as thin rings by default (VIEW, or o to cycle): a filled
footprint hides the very pixels you are judging, and at a handful of pixels
per cell there is no rim thin enough to help. "circle" rings each ROI
without covering it, "outline" traces its own border, and "fill" is the old
shaded footprint. The vector modes are line geometry, so the stroke stays a
hairline however far in you zoom.

With drawing off, clicking selects what is under the cursor - a derived
component when its overlay shows there, else the drawn ROI - and clicking
the background clears the selection. Ctrl+click (in the image, the ROI
table, or the trace-plot legend) toggles the ROI in a group buffer and
shift+click adds to it (a row range in the table); class labels and the
group color then apply to every member, so two cells can be grouped and
sent to "soma" in two clicks. Mask, table and trace colors all come from
the same per-ROI color. Selecting a listed ROI on another plane jumps
every slider that plane encodes. Only the first subplot is drawable.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from imgui_bundle import (
    icons_fontawesome_6 as fa,
)
from imgui_bundle import (
    imgui,
    imgui_ctx,
    implot,
)

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
    neuropil_overlay,
    trace_profile,
    y_label,
)
from mbo_utilities.annotation.display import DFF_METHODS
from mbo_utilities.arrays.features._dim_labels import slider_roles
from mbo_utilities.arrays.features._dim_tags import (
    TAG_REGISTRY,
    DimensionTag,
    OutputFilename,
    filename_tags,
)
from mbo_utilities.arrays.features._selection import to_lsp_kwargs
from mbo_utilities.arrays.features._slicing import index_window
from mbo_utilities.gui import roi_runs
from mbo_utilities.gui._imgui_helpers import (
    fit_width,
    right_aligned_text,
    selected_button_style,
    set_tooltip,
    settings_row,
    settings_table,
)
from mbo_utilities.gui._keyboard import claim_arrow_keys
from mbo_utilities.gui._theme import (
    THEME,
    danger_button,
    em,
    label_button,
    to_vec4,
)
from mbo_utilities.gui._top_strip import TopPanel, TopStrip
from mbo_utilities.gui.imgui import (
    UNLABEL_ALL,
    LabelSet,
    RoiOrder,
    RowAction,
    StrokeDrawer,
    SummaryImageViewer,
    draw_label_editor,
    draw_label_filter,
    draw_range_filter,
    draw_roi_table,
)
from mbo_utilities.gui.imgui.lines import plot_style, subplots
from mbo_utilities.gui.imgui.motion import MotionPlot
from mbo_utilities.gui.playhead import Playhead, TimeAxis
from mbo_utilities.gui.roi_runs import (
    MASK_MODES,
    RING_SCALE,
    SELECTED_ALPHA,
    DerivedSet,
    RoiRun,
    RoiRunManager,
    component_color,
    derived_outline,
    derived_rgba,
    feathered_rgba,
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
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.gui.widgets.widget_toggles import sub_enabled
from mbo_utilities.lazy_array import base_array
from mbo_utilities.results import unit_name
from mbo_utilities.roi_workflow import (
    OUT_PREFIX,
    SAVE_NAME,
    PlaneMovie,
    demix_rois,
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
    "ManualRoiWidget",
    "SAVE_NAME",
    "SELECTED_OPACITY",
    "attach_roi_widget",
    "detach_roi_widget",
    "labels_path",
    "roi_widgets_available",
]

# height the Traces panel asks the top strip for (the strip adds the menu
# row and its tab bar on top of this), and with the motion plot under the
# trace; the trace's share of the pair until the splitter between them is dragged
PANEL_HEIGHT = 226
MOTION_PANEL_HEIGHT = 340
TRACE_SHARE = 0.6

# how often to look for finished pipeline runs started outside this widget;
# the check reads one sidecar per tracked process, so not every frame
ADOPT_INTERVAL_S = 1.0
# below this width the tabs collapse to a placeholder line
MIN_TAB_WIDTH = 150

# (name, stretch weight, hidden by default); sort keys are looked up by name
TRACE_COLUMNS = (
    ("id", 1.4, False),
    ("z", 0.7, False),
    ("c", 0.7, False),
    ("source", 1.6, True),
    ("frames", 1.0, True),
    ("peak", 1.0, True),
    ("", 0.6, False),
)

# x axis units for the trace plot; the time ones need a sampling rate
X_UNITS = ("frames", "seconds", "ms")

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
X_AXIS_LABELS = {"frames": "frame", "seconds": "time (s)", "ms": "time (ms)"}

# tri-state stage toggles, shared by both pipelines: 0 skip, 1 run, 2 force
_STAGE_NAMES = ("skip", "run", "force")

MIN_ROI_PIXELS = 9
MIN_REGION_SIDE = 4
SELECTED_OPACITY = SELECTED_ALPHA

MASK_MODE_TIPS = {
    "circle": "A thin ring around each ROI - nothing covers the pixels",
    "outline": "A thin line along the ROI's own border",
    "fill": "The whole footprint, shaded (hides what is under it)",
}
# no seeded class labels: the label set starts empty, the user names their own
DEFAULT_LABEL_NAMES: tuple[str, ...] = ()
COLUMNS = ("id", "label", "source", "ok")
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

# the tag a run gets when the box is left empty; shown as the box hint
DEFAULT_RUN_TAG = "manual"

# VIEW > color by: a store column or the trace peak; "none" is the class / group / hue coloring
COLOR_BY = ("none", "class", "z", "c", "area", "peak")
COLORMAPS = ("viridis", "plasma", "turbo", "coolwarm", "tab10")

# every caption of the ROIs tab's settings tables (one table per section and
# one for the filters), so the control column starts at the same x in each
_CAPTIONS = (
    "in view",
    "labeling",
    "image",
    "draw",
    "edit",
    "auto",
    "masks",
    "show",
    "opacity",
    "stroke",
    "color by",
    "on disk",
    "labeled",
    "new label",
    "classes",
    "filter",
    "slice",
    "range",
)

RUN_ICON = fa.ICON_FA_PLAY
TRACE_ICON = fa.ICON_FA_CHART_LINE
REMOVE_ICON = fa.ICON_FA_XMARK

KEYBINDS = (
    ("a", "arm / disarm ROI drawing"),
    ("r", "arm / disarm region drawing"),
    ("esc", "stop drawing, else clear the region"),
    ("ctrl+z", "undo the last drawn ROI"),
    ("delete", "delete the selected ROI / discard the selected algo one"),
    (
        "up / down",
        "previous / next trace while the Traces panel is up, else ROI in view",
    ),
    ("u", "next unlabeled ROI"),
    ("f", "center the shown ROI; labeling then advances"),
    ("1-9", "label the selected ROI (drawn or algo), then advance"),
    ("0", "clear its label"),
    ("y", "promote the selected algo ROI"),
    ("n", "discard the selected algo ROI"),
    ("x", "accept / reject the selected algo ROI"),
    (
        "t",
        "quick trace the selected ROI: its mean, read where the Process tab points it",
    ),
    ("shift+t", "run the selection through the Process tab's engine"),
    ("b", "toggle the drawn overlay"),
    ("d", "toggle the algo overlay"),
    ("o", "cycle how masks draw: circle / outline / fill"),
    ("click", "select what is under the cursor (drawing off)"),
    ("ctrl+click", "toggle an ROI in the group (image, table, trace legend)"),
    ("shift+click", "add to the group (a row range in the table)"),
    ("esc", "also empties the group"),
)

_HELP_STEPS = (
    "Arm Add ROI (a) and drag a closed stroke around a cell; release fills "
    "it and, with trace on draw ticked, plots its mean trace. Ctrl+Z "
    "undoes, delete removes the selection.",
    "Label ROIs with the class buttons or keys 1-9 (0 clears); u jumps to "
    "the next unlabeled one and labeling steps there on its own.",
    "The Process tab's ROIs pipeline runs the selected, grouped, listed or "
    "all ROIs through mean (raw mask average), suite2p (suite2p's "
    "extractor) or masknmf (seeded NMF, demixed), reading each mask where "
    "it was drawn or on the z-plane / channel / frames you pick there. "
    "The row buttons on the ROIs tab and the t key run one ROI the same way.",
    "Every measurement is a row of the Traces tab: which ROI, on which "
    "z-plane and channel, with which engine. Re-running one replaces it.",
    "Find in region: Draw region (r), drag a box, then suite2p or masknmf "
    "looks for ROIs inside it, unseeded (Process tab > ROIs).",
    "Detected components arrive as an algo overlay and table rows: "
    "promote one into the drawn set (y) or discard it (n). Deleting a "
    "promoted ROI makes its row promotable again.",
    "The Traces tab plots quick traces and run traces per ROI; the Runs "
    "tab lists active, finished, loaded and on-disk runs.",
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

    The ROI panel used to carry its own Help and Keys buttons; there is one
    of each for the whole app now, so the content lives here - beside the
    code it describes - and the viewer renders it as a section. The keys
    are not repeated here; the Keybinds popup lists them.
    """
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(_HELP_STEPS, 1))
    return (
        "## ROI Labeling\n\n"
        "### Workflow\n\n"
        f"{steps}\n\n"
        "### Output files\n\n"
        f"```\n{_HELP_FILES}\n```\n"
    )


_CURSOR_COLOR = imgui.ImVec4(1.0, 0.85, 0.3, 0.9)
# the trace reads over the neuropil under it, which keeps implot's 1.0
_TRACE_WEIGHT = 1.5
_FNEU_COLOR = (0.25, 0.55, 1.0, 1.0)
_fneu_cmap: int | None = None


def _fneu_colormap() -> int:
    """The registered single-blue colormap every neuropil line draws with."""
    global _fneu_cmap
    if _fneu_cmap is None:
        idx = implot.get_colormap_index("mbo_fneu")
        if idx < 0:
            idx = implot.add_colormap(
                "mbo_fneu", np.array([_FNEU_COLOR, _FNEU_COLOR], np.float32)
            )
        _fneu_cmap = int(idx)
    return _fneu_cmap


def _line_colormap(rgb) -> int:
    """A registered single-color colormap for one trace line (this implot
    build has no per-line color argument, so lines take their color from the
    pushed colormap). Looked up by name so a recreated context re-registers.
    """
    key = tuple(int(round(float(v) * 255)) for v in rgb)
    name = "mbo_line_{}_{}_{}".format(*key)
    idx = implot.get_colormap_index(name)
    if idx < 0:
        color = (key[0] / 255.0, key[1] / 255.0, key[2] / 255.0, 1.0)
        idx = implot.add_colormap(name, np.array([color, color], np.float32))
    return int(idx)


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
    """``RoiOrder`` with "only this z-plane" and "only this source" filters."""

    def __init__(self, columns, labels, n_items):
        super().__init__(columns, labels, n_items)
        self.plane: int | None = None
        self.planes = np.zeros(0, np.int64)
        self.source: int | None = None  # None = all, 0 = drawn, 1 + si = a set
        self.sources = np.zeros(0, np.int64)

    def rebuild(self):
        super().rebuild()
        if not len(self.order):
            return
        keep = np.ones(len(self.order), bool)
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
        out = super().hidden_by(item)
        if self.plane is not None and int(self.planes[item]) != self.plane:
            out.append("plane")
        if self.source is not None and int(self.sources[item]) != self.source:
            out.append("source")
        return out

    def clear_filter(self, name: str):
        if name == "plane":
            self.plane = None
        elif name == "source":
            self.source = None
        else:
            super().clear_filter(name)

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

        self.roles = slider_roles(iw.dim_names)
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
        # ctrl / shift click builds a group here; label and color actions
        # then apply to every member. Entries are (si, k), si -1 = drawn.
        self.buffer: list[tuple[int, int]] = []
        self._group_color = (1.0, 0.8, 0.2)
        self._pending_row_action: tuple[str, int, int] | None = None
        self.status = "press Add ROI to start"
        self._save_error: str | None = None
        self._run_error: str | None = None
        self._writer: LabelsZarr | None = None
        self.new_label = ""
        self._note_buf = ""
        self.scroll_to_selection = False
        self.follow = False  # center the shown ROI; labeling then advances

        self.show_masks = True
        self.opacity = 0.45
        self.show_derived = True
        self.derived_opacity = 0.6
        # how masks draw: a ring standing in for each ROI, its own border,
        # or the filled footprint. The vector modes stroke in screen pixels
        # (line_width), the ring in image pixels (ring_scale over the mask's
        # equal-area radius)
        self.mask_mode = MASK_MODES[0]
        self.line_width = 1.0
        self.ring_scale = RING_SCALE

        self._feathers: dict[int, tuple] = {}
        self.rows: list[tuple[int, int]] = []
        self._row_index: dict[tuple[int, int], int] = {}
        self._promoted: dict[tuple[str, int], int] = {}
        self.derived: list[DerivedSet] = []
        self.classes = LabelSet(0, self.store.label_names)
        self.order = _PlaneOrder(
            {"source": np.zeros(0, np.int64)}, self.classes.labels, 0
        )

        # one time for every view of the recording: the host's when it has one
        self.playhead = getattr(host, "playhead", None)
        if self.playhead is None:
            self.playhead = Playhead()
            if host is not None:
                host.playhead = self.playhead
        self.model.set_view(self._view_pos())
        iw.ndwidget.indices.add_event_handler(self._on_indices)

        self.overlay = self.subplot.add_image(
            np.zeros((self.ny, self.nx, 4), np.uint8),
            name="manual_roi_overlay",
            alpha_mode="blend",
            offset=(0, 0, 1),
        )
        self.derived_overlay = self.subplot.add_image(
            np.zeros((self.ny, self.nx, 4), np.uint8),
            name="manual_roi_derived",
            alpha_mode="blend",
            offset=(0, 0, 1.5),
        )
        # literal RGBA bytes: auto-ranging off the all-zero start saturates every colour to white
        for overlay in (self.overlay, self.derived_overlay):
            overlay.vmin, overlay.vmax = 0, 255
            for tile in overlay.world_object.children:
                tile.material.pick_write = False
        self.derived_overlay.visible = False

        # the vector overlays, one line per source. Thickness is in screen
        # pixels, so a hairline stays a hairline at any zoom, and the paths
        # of every ROI ride in one buffer split by NaN rows. They start on
        # a few dummy vertices with an [n, 4] colors array, which makes the
        # colors per-vertex; _set_line reallocates both on every refresh
        self.outline = self.subplot.add_line(
            np.zeros((5, 3), np.float32),
            colors=np.zeros((5, 4), np.float32),
            thickness=self.line_width,
            size_space="screen",
            name="manual_roi_outline",
            offset=(0, 0, 1.25),
            visible=False,
        )
        self.derived_outline = self.subplot.add_line(
            np.zeros((5, 3), np.float32),
            colors=np.zeros((5, 4), np.float32),
            thickness=self.line_width,
            size_space="screen",
            name="manual_roi_derived_outline",
            offset=(0, 0, 1.6),
            visible=False,
        )
        for line in (self.outline, self.derived_outline):
            line.world_object.material.pick_write = False

        self.drawer = StrokeDrawer(self.subplot, self._on_stroke, self._pick)
        self.summary = SummaryImageViewer(iw.figure, title="Full FOV")
        self.region: tuple[int, int, int, int] | None = None
        self.region_mode = False
        self.region_line = None

        # the last ROI a trace landed for: the plot's fallback with nothing selected
        self.trace_uid = 0
        self._trace_results: queue.Queue = queue.Queue()
        self._trace_threads: list[threading.Thread] = []
        self.trace_sel: set[tuple] = set()  # trace-table keys to plot
        self._trace_stats: dict[tuple, tuple] = {}
        self._trace_display: dict[tuple, tuple] = {}
        # one entry: the last windowed line, so panning does not recompute it
        self._trace_window_cache: dict[tuple, np.ndarray] = {}
        self.correct_neuropil = True
        # what the plot shows of each row (a DISPLAY_KINDS name); None lets
        # every row's pipeline pick, and a kind a row lacks falls back the same way
        self._kind: str | None = None
        # the panel's own dF/F baseline for rows that compute one; None keeps
        # each pipeline's
        self.dff: DffSettings | None = None
        self._trace_sort = (0, True)
        self._trace_fit = True
        self._plot_key = None
        # autofit refits the axes whenever the plotted traces change; turn it
        # off to hold a zoomed-in stretch while stepping through ROIs
        self.autofit = True
        self._force_fit = False
        # None until picked: seconds whenever the data has a rate
        self._x_unit: str | None = None
        # the Traces tab shows the trace plot, and the motion plot under it
        # in linked subplots when the recording went through motion
        # correction (MC); the splitter's share is kept between frames
        self.show_trace = True
        self.show_motion = True
        self._motion_linked = False
        self._motion_ratios = implot.SubplotsRowColRatios(
            row_ratios=[TRACE_SHARE, 1.0 - TRACE_SHARE]
        )
        # drawn on the tab in place of "no traces" while a host computes them
        self.pending_traces = None
        self._fs_value: float | None = None
        self._fs_read = False
        # VIEW > color by: a store column or the trace peak through a colormap
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
        # panel) and runs the per-frame hook whatever is up; standalone use —
        # tests, a bare viewer — gets its own strip
        self._own_strip = strip is None
        self.strip = TopStrip(iw.figure) if self._own_strip else strip
        self.strip.add_hook(self._frame)
        # kept so the panel can ask for more height once the motion plot shows
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
        self.refresh_derived_overlay()

    def _bind_recording(self) -> None:
        """The recording's facets for the Traces tab: its motion correction,
        and where each of its scanned lines sits (an AOD unit's
        ``line_positions``, by ROI index; empty for anything else).
        """
        data = getattr(self.iw, "data", None)
        arr = base_array(data[0]) if data else None
        self.motion = MotionPlot(getattr(arr, "motion_correction", None))
        self.line_positions = list(getattr(arr, "line_positions", None) or [])

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
        self.set_drawing(False)
        renderer = self.subplot.renderer
        for fn, kind in (
            (self.drawer._down, "pointer_down"),
            (self.drawer._move, "pointer_move"),
            (self.drawer._up, "pointer_up"),
        ):
            try:
                renderer.remove_event_handler(fn, kind)
            except (KeyError, ValueError):
                pass
        try:
            self.iw.ndwidget.indices.remove_event_handler(self._on_indices)
        except (KeyError, ValueError, AttributeError):
            pass
        self.playhead.remove_event_handler(self._on_playhead)
        # the parked store and table must not keep calling into a closed widget
        self.model.close()
        graphics = [
            self.overlay,
            self.derived_overlay,
            self.outline,
            self.derived_outline,
            self.drawer.line,
        ]
        if self.region_line is not None:
            graphics.append(self.region_line)
        for graphic in graphics:
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
        pos = {}
        for name, _n in self.plane_axes:
            # a unit switch (MESc) can drop a scroll dim the store was keyed
            # on; that dim then sits at plane 0 rather than taking the GUI down
            try:
                pos[name] = int(self.iw.indices[name])
            except KeyError:
                pos[name] = 0
        return pos

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
        self.model.set_view(self._view_pos())
        if self.tdim is not None:
            self.playhead.seek(
                self.viewer_axis().seconds(self.current_frame()), source=self
            )

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

    def plot_axis(self) -> TimeAxis:
        """The trace plot's x axis in the chosen unit."""
        if self.x_unit == "frames":
            return self.viewer_axis()
        return TimeAxis(1000.0 if self.x_unit == "ms" else 1.0)

    def _on_view(self, _event):
        """The sliders landed on another plane: drop a half-drawn stroke,
        refilter the table and swap the overlays.
        """
        if self.drawer.stroke:
            self.drawer.stroke = []
            self.drawer.line.visible = False
        if self.order.plane is not None:
            self.order.plane = self.z
            self.order.rebuild()
        self.refresh_overlay()
        self.refresh_derived_overlay()

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
        if self.color_by not in ("none", "peak") and action in (
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
        """Tint every ROI by ``color_by`` through ``color_cmap`` (VIEW card):
        a store column, or the peak of its traces; "none" restores the class
        / group / hue colors.
        """
        if self.color_by == "none":
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
            if by == "none"
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
        return self.drawer.armed and not self.region_mode

    @property
    def stroke(self) -> list:
        return self.drawer.stroke

    @property
    def stroke_line(self):
        return self.drawer.line

    def _resync(self):
        """Rebuild the combined rows, label set and table order from the
        store and the loaded derived sets. Drawn rows come first so table
        ids match store indices; promoted rows are recomputed from the
        store's ``source`` strings.
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
        oks = [1] * len(rois)
        for si, s in enumerate(self.derived):
            if not s.visible:
                continue
            for k, stat_row in enumerate(s.result.stat):
                if k in s.discarded:
                    continue
                self.rows.append((si, k))
                planes.append(s.result.z)
                sources.append(1 + si)
                oks.append(1 if s.accepted[k] else 0)
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
        columns = {
            "source": np.asarray(sources, np.int64),
            "ok": np.asarray(oks, np.int64),
        }
        if self.store.nz > 1:
            columns["z"] = planes
        self.order.columns = columns
        self.order.labels = self.classes.labels
        self.order.n_items = len(self.rows)
        self.order.planes = planes
        self.order.sources = columns["source"]
        if self.order.source is not None and not 0 <= self.order.source <= len(
            self.derived
        ):
            self.order.source = None
        self.order.rebuild()

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
    def columns(self) -> tuple[str, ...]:
        return COLUMNS + (("z",) if self.store.nz > 1 else ())

    def _formatters(self) -> dict:
        def source(row):
            # the algorithm, not the run name: "suite2p" alone does not say
            # which detector made the component, and the run name is what
            # the source filter above the table already lists
            si, k = self.rows[row]
            if si < 0:
                return "drawn"
            s = self.derived[si]
            algo = getattr(s.result, "algo", "") or s.name
            return f"{algo} · promoted" if (s.name, k) in self._promoted else algo

        def zplane(row):
            si, k = self.rows[row]
            z = self.store.rois[k].plane if si < 0 else self.derived[si].result.z
            return self._plane_label(z)

        def ok(row):
            si, k = self.rows[row]
            if si < 0:
                return ""
            return "yes" if self.derived[si].accepted[k] else "no"

        return {"source": source, "ok": ok, "z": zplane}

    def _arm_mode(self, mode: str):
        """One of "off", "roi", "region" owns the stroke drawer."""
        current = (
            "region" if self.region_mode else ("roi" if self.drawer.armed else "off")
        )
        if mode == current:
            return
        self.region_mode = mode == "region"
        self.drawer.arm(mode != "off")
        if mode == "roi":
            self.status = "drag a closed stroke around a cell"
        elif mode == "region":
            self.status = "drag a box around the region"
        else:
            self.status = f"{self.n_rois} ROIs"

    def set_drawing(self, on: bool):
        """Arm or disarm ROI stroke drawing (lifts the pan binding while armed)."""
        self._arm_mode("roi" if on else "off")

    def set_region_mode(self, on: bool):
        """Arm or disarm region drawing; a finished drag becomes ``self.region``."""
        self._arm_mode("region" if on else "off")

    def _on_stroke(self, stroke):
        # runs inside a renderer pointer event: a raise here would vanish
        # into the event loop and could leave a stored ROI undrawn
        try:
            if self.region_mode:
                self._set_region(stroke)
            else:
                self.add_roi(stroke)
        except Exception as e:  # noqa: BLE001 - surfaced in the status row
            self.logger.exception("stroke handling failed")
            self.status = f"stroke failed: {type(e).__name__}: {e}"
            self._resync()
            self.refresh_overlay()

    def add_roi(self, stroke):
        """Fill a closed stroke and store it as the next label on plane z."""
        if len(stroke) < 3:
            self.status = "stroke too short"
            return
        points = np.round(np.asarray(stroke, np.float32)).astype(np.int32)
        points[:, 0] = points[:, 0].clip(0, self.nx - 1)
        points[:, 1] = points[:, 1].clip(0, self.ny - 1)
        filled = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(filled, [points], 1)
        index = self.store.add_roi(self.z, filled.astype(bool))
        if index is None:
            self.status = f"under {MIN_ROI_PIXELS} free px, not added"
            return
        self.select_roi(index)
        if self.auto_trace and self.trace_disabled(index) is None:
            self.quick_trace(index)

    def _set_region(self, stroke):
        if len(stroke) < 2:
            return
        points = np.asarray(stroke, np.float32)
        y0 = int(np.clip(np.floor(points[:, 1].min()), 0, self.ny - 1))
        x0 = int(np.clip(np.floor(points[:, 0].min()), 0, self.nx - 1))
        y1 = int(np.clip(np.ceil(points[:, 1].max()), y0 + 1, self.ny))
        x1 = int(np.clip(np.ceil(points[:, 0].max()), x0 + 1, self.nx))
        if y1 - y0 < MIN_REGION_SIDE or x1 - x0 < MIN_REGION_SIDE:
            self.status = f"region under {MIN_REGION_SIDE} px per side, ignored"
            return
        self.region = (y0, y1, x0, x1)
        if self.region_line is None:
            self.region_line = self.subplot.add_line(
                np.zeros((5, 3), np.float32),
                colors="cyan",
                thickness=1.5,
                name="roi_region",
                offset=(0, 0, 1.75),
                visible=False,
            )
            self.region_line.world_object.material.pick_write = False
        self.region_line.data = np.array(
            [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0]],
            np.float32,
        )
        self.region_line.visible = True
        self.status = f"region {y1 - y0}x{x1 - x0}"

    def clear_region(self):
        self.region = None
        if self.region_line is not None:
            self.region_line.visible = False

    def _pick(self, row: int, col: int, mods: frozenset = frozenset()):
        """Select what the click shows: a visible derived component first
        (the derived overlay draws on top), else the drawn ROI. Ctrl+click
        toggles it in the group buffer, shift+click adds it; a plain click
        replaces the group with the single selection.
        """
        try:
            hit: tuple[int, int] | None = None
            if self.show_derived and 0 <= row < self.ny and 0 <= col < self.nx:
                for si, s in enumerate(self.derived):
                    if not s.visible or s.result.z != self.z:
                        continue
                    k = int(s.pick_map[row, col])
                    if k >= 0 and k not in s.discarded:
                        hit = (si, k)
                        break
            if hit is None:
                index = self.store.roi_at(self.z, row, col)
                if index >= 0:
                    hit = (-1, index)
            if "Ctrl" in mods:
                if hit is not None:
                    self.buffer_toggle(*hit)
                return
            if "Shift" in mods:
                if hit is not None:
                    self.buffer_add(*hit)
                return
            self.buffer_clear()
            if hit is None:
                self.select_roi(-1)
            elif hit[0] < 0:
                self.select_roi(hit[1])
            else:
                self.select_derived(*hit)
        except Exception as e:  # noqa: BLE001 - surfaced in the status row
            self.logger.exception("pick failed")
            self.status = f"pick failed: {type(e).__name__}: {e}"

    def _row_grouped(self, item: int) -> bool:
        return 0 <= item < len(self.rows) and self.rows[item] in self.buffer

    def _table_select(self, item: int):
        """Plain table click: single selection, group dropped."""
        self.buffer_clear()
        self.select_row(item)

    def _table_ctrl(self, item: int):
        if 0 <= item < len(self.rows):
            self.buffer_toggle(*self.rows[item])

    def select_row(self, row: int | None):
        """Route a table-row selection to the right kind."""
        if row is None or not 0 <= row < len(self.rows):
            self.select_roi(-1)
            return
        si, k = self.rows[row]
        if si < 0:
            self.select_roi(k)
        else:
            self.select_derived(si, k)

    def select_roi(self, index: int | None):
        """Select drawn ROI ``index``; anything out of range clears the
        selection (and any derived one).

        Selecting an ROI on another plane jumps the z slider to it; one with
        a trace shows it in the Traces tab.
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
        self.refresh_derived_overlay()

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
        self.status = f"{s.name} row {k}: {npix} px{tail}" + _cleared_note(cleared)
        if s.result.z != self.z:
            self._goto_plane(s.result.z)
        if self.follow:
            self._center_on(stat_row["ypix"], stat_row["xpix"])
        self._sync_trace_sel()
        self.refresh_overlay()
        self.refresh_derived_overlay()

    def in_buffer(self, si: int, k: int) -> bool:
        return (si, k) in self.buffer

    def _seed_buffer(self):
        """A first ctrl / shift click keeps the current selection grouped,
        so 'select one, ctrl+click another' makes a group of two.
        """
        if self.buffer:
            return
        if self.selected >= 0:
            self.buffer.append((-1, self.selected))
        elif self.selected_derived is not None:
            self.buffer.append(self.selected_derived)

    def buffer_add(self, si: int, k: int):
        """Add one row to the group and make it the shown one."""
        self._seed_buffer()
        if (si, k) not in self.buffer:
            self.buffer.append((si, k))
        self._after_buffer_change(si, k)

    def buffer_toggle(self, si: int, k: int):
        """Ctrl+click: flip one row's group membership."""
        self._seed_buffer()
        if (si, k) in self.buffer:
            self.buffer.remove((si, k))
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
        if self.buffer:
            self.buffer = []
            self._refresh_group_view()

    def _after_buffer_change(self, si: int, k: int):
        n = len(self.buffer)
        if si < 0:
            self.select_roi(k)
        else:
            self.select_derived(si, k)
        if n > 1:
            self.status = f"{n} in group · labels and color apply to all"

    def _refresh_group_view(self):
        self.refresh_overlay()
        self.refresh_derived_overlay()

    def set_group_color(self, rgb: tuple[float, float, float] | None):
        """Give every grouped ROI (or just the selection) an explicit
        display color, in masks, table and traces alike; None reverts to
        the class / hue colors.
        """
        targets = list(self.buffer)
        if not targets:
            if self.selected >= 0:
                targets = [(-1, self.selected)]
            elif self.selected_derived is not None:
                targets = [self.selected_derived]
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
        """Up / down: the next or previous trace while the Traces panel is
        up, else the next or previous ROI. Either way the image, both tables
        and the trace plot land on the same ROI.
        """
        if self.top_tab == "traces" and self.step_trace(delta):
            return
        if self.order.step(delta):
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
        self._trace_fit = True
        pair = self._key_to_pair(key)
        if pair is None:
            return
        si, k = pair
        if si < 0:
            self.trace_uid = self.store.rois[k].uid
            self.select_roi(k)
        else:
            self.select_derived(si, k)

    def toggle_trace(self, key):
        """Add / remove one trace from the plotted set (ctrl+click)."""
        (self.trace_sel.discard if key in self.trace_sel else self.trace_sel.add)(key)
        self._trace_fit = True

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

    def delete_roi(self, index: int):
        """Drop one drawn ROI and renumber the labels above it; traces of
        every other ROI survive (they are keyed by uid).
        """
        if not 0 <= index < self.n_rois:
            return
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
        """Delete the selected drawn ROI, or discard the selected derived one."""
        if self.selected >= 0:
            self.delete_roi(self.selected)
        elif self.selected_derived is not None:
            self.discard_derived(*self.selected_derived, advance=True)

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

    def _set_line(self, line, positions, colors):
        """Push prepared path geometry onto one of the vector overlays.

        Nothing to draw hides the graphic instead: a line has to put its
        vertices somewhere, and an empty buffer is not worth allocating.
        """
        if not len(positions):
            line.visible = False
            return
        line.data = positions
        line.colors = colors
        line.thickness = self.line_width
        line.visible = True

    def refresh_overlay(self):
        """Drawn masks, colored exactly like the imported ones; only the
        table's source column tells them apart.

        One component list feeds either renderer - the feathered fill, or
        the thin paths of a vector mode - and the graphic the mode is not
        using is hidden rather than emptied, so switching back is free.
        Strokes draw opaque: a hairline at the fill's opacity is invisible.
        """
        vector = self.mask_mode != "fill"
        self.overlay.visible = self.show_masks and not vector
        self.outline.visible = self.show_masks and vector
        if not self.show_masks:
            return
        comps = []
        halo = []
        sel = None
        grouped = {k for si, k in self.buffer if si < 0}
        for i, record in enumerate(self.store.rois):
            if record.plane != self.z:
                continue
            ypix, xpix, lam = self._feather(i)
            rgb = np.asarray(self.store.roi_rgb(i), np.float32) / 255.0
            fill = SELECTED_OPACITY if i in grouped else self.opacity
            comps.append((ypix, xpix, lam, rgb, 1.0 if vector else fill))
            if i in grouped:
                halo.append((ypix, xpix))
            if i == self.selected:
                sel = (ypix, xpix, rgb)
                halo.append((ypix, xpix))
        if vector:
            self._set_line(
                self.outline,
                *outline_data(comps, self.mask_mode, halo, self.ring_scale),
            )
        else:
            self.overlay.data = feathered_rgba((self.ny, self.nx), comps, sel)

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
        """Flip review mode: the shown ROI is framed by the camera and a
        label steps to the next one, like masknmf's classification GUI.
        """
        self.follow = not self.follow
        if self.follow:
            self._center_selection()
            self.status = "centering on the shown ROI; labeling advances"
        else:
            self.status = f"{self.n_rois} ROIs"

    def toggle_drawn_overlay(self):
        self.show_masks = not self.show_masks
        self.refresh_overlay()

    def set_mask_mode(self, mode: str):
        """Switch how masks draw, for both overlays at once."""
        if mode not in MASK_MODES or mode == self.mask_mode:
            return
        self.mask_mode = mode
        self.refresh_overlay()
        self.refresh_derived_overlay()
        self.status = f"masks: {mode}"

    def cycle_mask_mode(self):
        self.set_mask_mode(
            MASK_MODES[(MASK_MODES.index(self.mask_mode) + 1) % len(MASK_MODES)]
        )

    def refresh_derived_overlay(self):
        """Recompute the derived overlay for the plane on screen; called on
        select / z / load / discard / promote / toggle / opacity / mode
        changes. Follows the drawn overlay's mask mode, so the two sources
        never draw in two different idioms.
        """
        sets_on_z = [s for s in self.derived if s.result.z == self.z]
        show = self.show_derived and bool(sets_on_z)
        vector = self.mask_mode != "fill"
        self.derived_overlay.visible = show and not vector
        self.derived_outline.visible = show and vector
        if not show:
            return
        selected = None
        if self.selected_derived is not None:
            si, k = self.selected_derived
            if self.derived[si].result.z == self.z:
                selected = (self.derived[si], k)
        grouped = {(id(self.derived[si]), k) for si, k in self.buffer if si >= 0}
        if vector:
            self._set_line(
                self.derived_outline,
                *derived_outline(
                    sets_on_z, self.mask_mode, selected, grouped, self.ring_scale
                ),
            )
        else:
            self.derived_overlay.data = derived_rgba(
                (self.ny, self.nx),
                sets_on_z,
                self.derived_opacity,
                selected,
                grouped=grouped,
            )

    def toggle_derived_overlay(self):
        self.show_derived = not self.show_derived
        self.refresh_derived_overlay()

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
        self._merge_run_traces(res, s.name)
        for k in range(len(res.stat)):
            if k not in s.discarded and (s.name, k) not in self._promoted:
                trace = self._member_trace(s, k)
                if trace is not None:
                    self.traces.add(trace)
        for trace in promoted_traces:
            self.traces.add(trace)
        self._resync()
        self.refresh_derived_overlay()
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
            frames=res.frames,
            path=res.path,
        )

    def load_run(self, path, discarded=(), classes=None, colors=None) -> bool:
        """Read one run dir into the widget: extract runs merge their
        traces, everything else loads as a derived set (every row, the
        rejected ones included - curation happens here). A results file
        (``mbo_utilities.results``), or one unit inside one, goes through
        :meth:`load_results`.
        """
        from mbo_utilities.results import results_pipeline

        path = Path(path)
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
        units (the voltage pipeline's scans) go straight to the Traces tab,
        one row per ROI plotting its denoised trace and one per member line
        plotting the line's raw trace. Every row is named by its ROI
        (``roi3``, ``roi3 (raw)``; a line of a multi-line ROI adds itself,
        ``roi3 line 12 (raw)``) and carries the line it was read from on Z
        and the pipeline's channel on C. ``path`` may name one unit inside
        the file (``<file>.zarr/zplane01``). Returns True when anything
        loaded.
        """
        from mbo_utilities.results import read_results, results_pipeline

        path = Path(path)
        if path.parent.suffix == ".zarr" and results_pipeline(path) is None:
            file, only = path.parent, path.name
        else:
            file, only = path, None
        try:
            results = read_results(file)
        except Exception as e:  # noqa: BLE001 - shown in the status row
            self._run_error = f"could not load {file.name}: {e}"
            return False
        loaded = 0
        for unit in results.units.values():
            if only is not None and unit.name != only:
                continue
            if unit.member_kind == "pixel" and unit.image_shape is not None:
                res = run_result_from_unit(unit, file / unit.name, results.pipeline)
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
                entry = {"label": str(roi), "fs": unit.fs, "c": channel, "kinds": {}}
                members = unit.members[k] if k < len(unit.members) else ()
                if lines and len(members) == 1:
                    entry["z"] = int(members[0])
                    entry["extra"] = {"line": int(members[0])}
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
        self.refresh_derived_overlay()
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
        self.refresh_derived_overlay()
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
        self.refresh_derived_overlay()
        self._autosave()
        self.status = f"{s.name}: promoted {promoted} / skipped {skipped}"

    def set_accepted(self, si: int, k: int, on: bool | None = None):
        """Flip (or set) one derived component's accepted flag, mirrored
        into the run dir's ``iscell.npy``.
        """
        s = self.derived[si]
        s.accepted[k] = (not s.accepted[k]) if on is None else bool(on)
        path = s.result.path / "iscell.npy"
        try:
            n = len(s.result.stat)
            iscell = np.load(path) if path.exists() else np.ones((n, 2), np.float32)
            if len(iscell) != n:
                iscell = np.ones((n, 2), np.float32)
            iscell[k, 0] = 1.0 if s.accepted[k] else 0.0
            np.save(path, iscell)
        except OSError as e:
            self._save_error = f"iscell save failed: {e}"
        self._resync()
        self.refresh_derived_overlay()
        state = "accepted" if s.accepted[k] else "rejected"
        self.status = f"{s.name} row {k}: {state}"

    def discard_derived(self, si: int, k: int, advance: bool = False):
        """Hide one derived component; its trace row goes with it."""
        s = self.derived[si]
        s.discarded.add(int(k))
        if self.selected_derived == (si, k):
            self.selected_derived = None
        self.traces.remove(("member", s.name, int(k)))
        self._resync()
        self.refresh_derived_overlay()
        self._save_registry()
        self.status = f"discarded {s.name} row {k}"
        if advance:
            self._select_next_derived(self.order.pos)

    def undiscard_derived(self, si: int, k: int):
        s = self.derived[si]
        s.discarded.discard(int(k))
        trace = self._member_trace(s, int(k))
        if trace is not None and (s.name, int(k)) not in self._promoted:
            self.traces.add(trace)
        self._resync()
        self.refresh_derived_overlay()
        self._save_registry()

    def restore_discarded(self, si: int):
        s = self.derived[si]
        for k in sorted(s.discarded):
            trace = self._member_trace(s, k)
            if trace is not None and (s.name, k) not in self._promoted:
                self.traces.add(trace)
        s.discarded.clear()
        self._resync()
        self.refresh_derived_overlay()
        self._save_registry()

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
        if self.fpath is None:
            return
        self._restoring = True
        try:
            for entry in load_run_registry(registry_path(self.fpath, self.tag)):
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
            own = labels_path(self.fpath).parent
            if run_dir_complete(own) and not any(
                s.result.path == own for s in self.derived
            ):
                # the data sits in a suite2p / masknmf result dir: show its ROIs
                self.load_run(own)
        finally:
            self._restoring = False

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
        """Trace rows moved: drop stale stats and selections, refit the plot."""
        self._trace_stats.clear()
        self._trace_display.clear()
        self.trace_sel &= set(self._trace_rows())
        self._trace_fit = True

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
                or info.task_type not in ("suite2p", "masknmf", "voltage")
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
            if info.task_type == "voltage":
                # a zarr-format voltage run leaves one results file in the PF folder
                from mbo_utilities.results import newest_results

                found = newest_results(out, "voltage")
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
        io = imgui.get_io()
        if io.want_text_input:
            return
        claim_arrow_keys(("up_arrow", "down_arrow"))
        if imgui.is_key_pressed(imgui.Key.a, False):
            self.set_drawing(not self.drawing)
        if imgui.is_key_pressed(imgui.Key.r, False):
            self.set_region_mode(not self.region_mode)
        if imgui.is_key_pressed(imgui.Key.escape):
            if self.drawer.armed:
                self._arm_mode("off")
            elif self.region is not None:
                self.clear_region()
            elif self.buffer:
                self.buffer_clear()
                self.status = "group cleared"
        if io.key_ctrl and imgui.is_key_pressed(imgui.Key.z, False):
            self.delete_roi(self.n_rois - 1)
        if imgui.is_key_pressed(imgui.Key.delete, False):
            self.delete_selected()
        if imgui.is_key_pressed(imgui.Key.u, False):
            self.next_unlabeled()
        if imgui.is_key_pressed(imgui.Key.f, False):
            self.toggle_follow()
        if imgui.is_key_pressed(imgui.Key.up_arrow):
            self.step(-1)
        if imgui.is_key_pressed(imgui.Key.down_arrow):
            self.step(1)
        if imgui.is_key_pressed(imgui.Key.b, False):
            self.toggle_drawn_overlay()
        if imgui.is_key_pressed(imgui.Key.d, False):
            self.toggle_derived_overlay()
        if imgui.is_key_pressed(imgui.Key.o, False):
            self.cycle_mask_mode()
        if imgui.is_key_pressed(imgui.Key.t, False):
            if imgui.get_io().key_shift:
                self.run_selection()
            elif self.selected >= 0 and self.trace_disabled(self.selected) is None:
                self.quick_trace(self.selected)
        if self.selected_derived is not None:
            if imgui.is_key_pressed(imgui.Key.y, False):
                self.promote_derived(*self.selected_derived)
            if imgui.is_key_pressed(imgui.Key.n, False):
                self.discard_derived(*self.selected_derived, advance=True)
            if imgui.is_key_pressed(imgui.Key.x, False):
                self.set_accepted(*self.selected_derived)
        if self.selected >= 0 or self.selected_derived is not None:
            picked = self.classes.hotkey_pressed()
            if picked is not None:
                self.assign_class(picked)

    @property
    def top_tab(self) -> str | None:
        """The top strip's selected panel: ``"traces"`` while the plot is up."""
        return self.strip.active

    def _frame(self):
        """Per-frame work the strip runs whatever tab is on top: background
        jobs, keyboard handling, and our own floating windows.
        """
        self._poll_jobs()
        self.handle_keys()
        self.summary.draw()

    def draw_rois(self):
        """The ROIs tab: the control sections, each gated by its Widgets-menu
        subwidget toggle and laid out like the Process tab's ROIs pipeline
        (a ``separator_text`` title over one two-column settings table: dim
        captions in a fixed column, controls in the stretch column, counts
        right-aligned), the status row, then the table (:meth:`draw_tab`).
        Narrower than ``MIN_TAB_WIDTH`` the tab collapses to its placeholder
        line.
        """
        sections = [("NAVIGATE", self._draw_navigate)]
        if sub_enabled("manual_roi", "tools"):
            sections.append(("DRAW", self._draw_draw_tools))
        if sub_enabled("manual_roi", "overlay"):
            sections.append(("VIEW", self._draw_view))
        if sub_enabled("manual_roi", "labels"):
            sections.append(("LABELS", self._draw_labels))
        with fit_width("ROI tools", min_width=MIN_TAB_WIDTH) as shown:
            if not shown:
                return
            imgui.spacing()
            for title, draw in sections:
                imgui.separator_text(title)
                draw()
                imgui.spacing()
            self._draw_status()
            self.draw_tab()

    def _draw_navigate(self):
        """Step through the ROIs in view; the keys sit in the tooltips."""
        n = len(self.order.order)
        gap = em(0.6)
        with settings_table("##roi_navigate", _CAPTIONS) as table:
            if not table:
                return
            settings_row("in view")
            if imgui.button("prev", imgui.ImVec2(em(4), 0)):
                self.step(-1)
            set_tooltip("The previous ROI in view (up)", show_mark=False)
            imgui.same_line(0, gap)
            if imgui.button("next", imgui.ImVec2(em(4), 0)):
                self.step(1)
            set_tooltip("The next ROI in view (down)", show_mark=False)
            imgui.same_line(0, gap)
            right_aligned_text(f"{self.order.pos + 1 if n else 0} / {n}")
            settings_row("labeling")
            if imgui.button("next unlabeled", imgui.ImVec2(em(9), 0)):
                self.next_unlabeled()
            set_tooltip("Jump to the next ROI without a label (u)", show_mark=False)
            imgui.same_line(0, gap)
            changed, self.follow = imgui.checkbox("center & advance", self.follow)
            if changed and self.follow:
                self._center_selection()
            set_tooltip(
                "Center the image on the selected ROI; labeling it then steps to the next one (f)",
                show_mark=False,
            )
            settings_row("image")
            if imgui.button("Open full FOV", imgui.ImVec2(em(9), 0)):
                self.open_full_fov()
            set_tooltip(
                "The frame on screen and the ROI labels in the summary-image window",
                show_mark=False,
            )

    def _draw_draw_tools(self):
        """The drawing tools with their keys in tooltips, then the
        trace-on-draw switch. Running what was drawn is the Process tab's
        ROIs pipeline (and the row buttons on the ROIs tab).
        """
        gap = em(0.6)
        nothing = self.selected < 0 and self.selected_derived is None
        have_region = self.region is not None
        with settings_table("##roi_draw", _CAPTIONS) as table:
            if not table:
                return
            settings_row("draw")
            with selected_button_style(self.drawing):
                if imgui.button("Add ROI", imgui.ImVec2(em(7), 0)):
                    self.set_drawing(not self.drawing)
            set_tooltip("Drag a closed stroke around a cell (a)", show_mark=False)
            imgui.same_line(0, gap)
            with selected_button_style(self.region_mode):
                if imgui.button(
                    "Region" if have_region else "Draw region", imgui.ImVec2(em(7), 0)
                ):
                    self.set_region_mode(not self.region_mode)
            if imgui.is_item_hovered():
                y0, y1, x0, x1 = self.region if have_region else (0, 0, 0, 0)
                imgui.set_tooltip(
                    f"Region {y1 - y0}x{x1 - x0} px - drag again to replace it (r); "
                    "Process tab > ROIs looks for cells inside it"
                    if have_region
                    else "Drag a box on the image to mark where to look for cells (r)"
                )
            imgui.same_line(0, gap)
            right_aligned_text(f"{self.n_rois} ROIs")
            settings_row("edit")
            if imgui.button("Undo", imgui.ImVec2(em(6), 0)):
                self.delete_roi(self.n_rois - 1)
            set_tooltip("Remove the ROI drawn last (ctrl+z)", show_mark=False)
            imgui.same_line(0, gap)
            if nothing:
                imgui.begin_disabled()
            if imgui.button(
                "Discard" if self.selected_derived is not None else "Delete",
                imgui.ImVec2(em(6), 0),
            ):
                self.delete_selected()
            if nothing:
                imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(
                    "select an ROI first"
                    if nothing
                    else "Delete the drawn ROI, discard the algo one (del)"
                )
            imgui.same_line(0, gap)
            with danger_button():
                if imgui.button("Clear", imgui.ImVec2(em(6), 0)):
                    self.clear()
            set_tooltip("Delete every drawn ROI", show_mark=False)
            settings_row("auto")
            _changed, self.auto_trace = imgui.checkbox("trace on draw", self.auto_trace)
            set_tooltip(
                "Trace every ROI the moment it is drawn: its mean at the "
                "z-plane / channel the Process tab's ROIs pipeline points at "
                "(where it was drawn, by default), plotted on the Traces panel",
                show_mark=False,
            )

    def _draw_view(self):
        """How masks draw, which overlays show, and the knobs the mode uses.

        The sliders swap with the mode: opacity is a fill idea and stroke
        width a line one, so only the ones that do something are shown.
        """
        vector = self.mask_mode != "fill"
        gap = em(0.6)
        with settings_table("##roi_view", _CAPTIONS) as table:
            if not table:
                return
            settings_row("masks")
            for i, mode in enumerate(MASK_MODES):
                if i:
                    imgui.same_line(0, gap)
                if imgui.radio_button(
                    f"{mode}##mask_mode_{mode}", self.mask_mode == mode
                ):
                    self.set_mask_mode(mode)
                set_tooltip(f"{MASK_MODE_TIPS[mode]} (o cycles)", show_mark=False)
            settings_row("show")
            changed, self.show_masks = imgui.checkbox("drawn", self.show_masks)
            if changed:
                self.refresh_overlay()
            set_tooltip("The ROIs you drew by hand (b)", show_mark=False)
            imgui.same_line(0, gap)
            changed, self.show_derived = imgui.checkbox("algo", self.show_derived)
            if changed:
                self.refresh_derived_overlay()
            set_tooltip(
                "ROIs an algorithm found (find / demix / full-plane runs) (d)",
                show_mark=False,
            )
            if not vector:
                settings_row("opacity")
                imgui.set_next_item_width(em(6.5))
                changed, self.opacity = imgui.slider_float(
                    "##opacity", self.opacity, 0.05, 1.0, "drawn %.2f"
                )
                if changed:
                    self.refresh_overlay()
                imgui.same_line(0, gap)
                imgui.set_next_item_width(em(6.5))
                changed, self.derived_opacity = imgui.slider_float(
                    "##derived_opacity", self.derived_opacity, 0.05, 1.0, "algo %.2f"
                )
                if changed:
                    self.refresh_derived_overlay()
            else:
                settings_row("stroke")
                dirty = False
                imgui.set_next_item_width(em(6.5))
                changed, self.line_width = imgui.slider_float(
                    "##line_width", self.line_width, 0.5, 4.0, "width %.1f px"
                )
                dirty |= changed
                set_tooltip(
                    "Stroke width on screen, whatever the zoom", show_mark=False
                )
                if self.mask_mode == "circle":
                    imgui.same_line(0, gap)
                    imgui.set_next_item_width(em(6.5))
                    changed, self.ring_scale = imgui.slider_float(
                        "##ring_scale", self.ring_scale, 0.6, 4.0, "ring %.2f"
                    )
                    dirty |= changed
                    set_tooltip("Ring radius over the mask's own", show_mark=False)
                if dirty:
                    self.refresh_overlay()
                    self.refresh_derived_overlay()
            settings_row("color by")
            imgui.set_next_item_width(em(6.5))
            changed, sel = imgui.combo(
                "##color_by", COLOR_BY.index(self.color_by), list(COLOR_BY)
            )
            set_tooltip(
                "Color every ROI by a value, like fastplotlib's cmap_transform: its "
                "class, z-plane or channel (one color per level), its area, or the "
                "peak of its traces (a gradient). none: class / group / hue colors.",
                show_mark=False,
            )
            if changed:
                self.set_color_by(COLOR_BY[sel])
            imgui.same_line(0, gap)
            imgui.set_next_item_width(em(6.5))
            if self.color_by == "none":
                imgui.begin_disabled()
            changed, sel = imgui.combo(
                "##color_cmap", COLORMAPS.index(self.color_cmap), list(COLORMAPS)
            )
            set_tooltip(
                "The colormap for the value picked on the left", show_mark=False
            )
            if self.color_by == "none":
                imgui.end_disabled()
            if changed:
                self.set_color_by(self.color_by, COLORMAPS[sel])
            settings_row("on disk")
            if imgui.button("Save", imgui.ImVec2(em(6), 0)):
                self.save()
            imgui.same_line(0, gap)
            self._draw_save_note()

    def _draw_labels(self):
        with settings_table("##roi_labels", _CAPTIONS) as table:
            if not table:
                return
            if self.n_rois:
                # just the count: NAVIGATE already carries "next unlabeled (u)"
                done = int((self.classes.labels[: self.n_rois] >= 0).sum())
                settings_row("labeled")
                imgui.align_text_to_frame_padding()
                imgui.text_colored(
                    to_vec4(THEME.ok if done == self.n_rois else THEME.warn),
                    f"{done} / {self.n_rois}",
                )
            settings_row("new label")
            self.new_label, changed = draw_label_editor(
                self.classes, self.new_label, "_roi"
            )
            if changed:
                self._sync_store_from_classes()
                self.order.rebuild()
                self.refresh_overlay()
                self._autosave()
            if not self.classes.names:
                return
            settings_row("classes")
            picked = self._draw_label_columns()
            if picked == UNLABEL_ALL:
                self.unlabel_all()
            elif picked is not None:
                self.assign_class(picked)

    def _draw_label_columns(self):
        """The unlabel actions pinned across the top, then one button per
        class, split into columns filled evenly.

        Class buttons carry two columns of their own - the count, then the
        name - each left-aligned, so the names line up down the column
        instead of drifting with however wide the counts happen to be.
        """
        picked = None
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
        # a touch narrower than the column: the buttons carried more empty
        # space than the names needed
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
        set_tooltip("Clear the selected ROI's label (0)", show_mark=False)
        pad = imgui.get_style().frame_padding.x * 2
        right_w = imgui.calc_text_size("unlabel all").x + pad
        imgui.same_line()
        imgui.set_cursor_pos_x(max(x0 + avail - right_w, imgui.get_cursor_pos_x()))
        with danger_button():
            if imgui.small_button("unlabel all##_roi"):
                picked = UNLABEL_ALL
        set_tooltip("Clear every label on this plane", show_mark=False)
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

    def _draw_status(self):
        """The status message with right-aligned counts. Help / keybinds live
        on the menu row, beside the Metadata Viewer button.
        """
        color, text = self._status_message()
        imgui.align_text_to_frame_padding()
        imgui.text_colored(to_vec4(color), text)
        imgui.same_line(0, em(1.0))
        avail = imgui.get_content_region_avail().x
        if avail <= em(1):
            return
        m = sum(len(s.result.stat) - len(s.discarded) for s in self.derived)
        counts = (
            f"{self.unit} · " if self.unit else ""
        ) + f"{self.n_rois} drawn · {m} algo"
        with imgui_ctx.begin_child(
            "##roi_counts",
            imgui.ImVec2(avail, em(1.6)),
            imgui.ChildFlags_.none,
            imgui.WindowFlags_.no_scrollbar,
        ):
            width = imgui.calc_text_size(counts).x
            inner = imgui.get_content_region_avail().x
            if width < inner:
                imgui.set_cursor_pos_x(inner - width)
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(counts)

    def _draw_save_note(self):
        if self._writer is None:
            imgui.text_disabled("autosave off")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "ROIs are kept in memory only - open a file to autosave "
                    "beside it, or press Save"
                )
            return
        imgui.text_disabled("Autosaved")
        if not imgui.is_item_hovered():
            return
        imgui.begin_tooltip()
        imgui.text(f"labels zarr: {self._save_target()}")
        imgui.text_colored(
            to_vec4(THEME.code),
            "from mbo_utilities.annotation import LabelsZarr\n"
            f'store = LabelsZarr.load(r"{self._save_target()}")\n'
            "store.labels        # (Z, Y, X) uint16; 0 = bg, ROI i = i + 1\n"
            "store.rois          # per-ROI plane, area, class, note, uid, source",
        )
        imgui.end_tooltip()

    def draw_tab(self):
        """The ROIs tab: filters, the combined drawn + algo table, the
        selection footer and the run-all row pinned under it.
        """
        changed_any = False
        with settings_table("##roi_filters", _CAPTIONS) as table:
            if table:
                settings_row("filter")
                if draw_label_filter(self.order, self.classes, "_roi", em(8)):
                    changed_any = True
                set_tooltip("Filter by label", show_mark=False)
                imgui.same_line(0, em(0.6))
                names = ["all", "drawn", *(s.name for s in self.derived)]
                current = 0 if self.order.source is None else self.order.source + 1
                imgui.set_next_item_width(em(8))
                changed, sel = imgui.combo(
                    "##source_filter", min(current, len(names) - 1), names
                )
                set_tooltip(
                    "Filter by source: drawn by hand, or a loaded run", show_mark=False
                )
                if changed:
                    self.order.source = None if sel == 0 else sel - 1
                    changed_any = True
                imgui.same_line(0, em(0.6))
                right_aligned_text(f"{len(self.order.order)} of {self.order.n_items}")
                if self.store.nz > 1:
                    settings_row("slice")
                    on = self.order.plane is not None
                    changed, on = imgui.checkbox(
                        f"this plane ({self._plane_label(self.z)} of {self.store.nz})",
                        on,
                    )
                    if changed:
                        self.order.plane = self.z if on else None
                        changed_any = True
                if self.order.range_column is not None:
                    settings_row("range")
                    if draw_range_filter(self.order, "_roi"):
                        changed_any = True
        if changed_any:
            self.order.rebuild()

        footer = 3 * imgui.get_frame_height_with_spacing() + 12
        # a few rows of table at least under the control sections; the tab scrolls
        height = max(imgui.get_content_region_avail().y - footer, em(10))
        with imgui_ctx.begin_child("##roi_table", imgui.ImVec2(0, height)):
            if self.rows:
                pos = self.order.pos
                self.scroll_to_selection = draw_roi_table(
                    self.order,
                    self.classes,
                    self.columns,
                    self._formatters(),
                    self.scroll_to_selection,
                    table_id="manual_rois",
                    on_select=self._table_select,
                    actions=self.row_actions,
                    is_grouped=self._row_grouped,
                    on_ctrl_select=self._table_ctrl,
                    on_shift_select=self.buffer_extend_to,
                )
                if self.order.pos != pos and self.order.current is not None:
                    self.select_row(self.order.current)
            else:
                imgui.text_disabled("no ROIs yet")

        pending, self._pending_row_action = self._pending_row_action, None
        if pending is not None:
            _act, si, k = pending
            if si < 0:
                self.delete_roi(k)
            else:
                self.discard_derived(si, k, advance=self.selected_derived == (si, k))

        imgui.separator()
        if len(self.buffer) > 1:
            imgui.text_disabled(f"{len(self.buffer)} ROIs grouped")
            imgui.same_line(0, 8)
            changed, col = imgui.color_edit3(
                "##group_color",
                list(self._group_color),
                imgui.ColorEditFlags_.no_inputs,
            )
            if changed:
                self._group_color = tuple(col)
            imgui.same_line(0, 4)
            if imgui.small_button("color group"):
                self.set_group_color(self._group_color)
            set_tooltip("Give every grouped ROI this color", show_mark=False)
            imgui.same_line(0, 4)
            if imgui.small_button("reset color"):
                self.set_group_color(None)
            set_tooltip("Back to class / hue colors", show_mark=False)
            imgui.same_line(0, 4)
            if imgui.small_button("ungroup"):
                self.buffer_clear()
            set_tooltip("Empty the group (esc)", show_mark=False)
            self._draw_run_selection("run group")
            imgui.text_disabled("label buttons and keys 1-9 apply to the whole group")
        elif self.selected >= 0:
            imgui.set_next_item_width(-1)
            changed, self._note_buf = imgui.input_text_with_hint(
                "##note", "note", self._note_buf
            )
            if changed:
                self.store.set_note(self.selected, self._note_buf)
            if imgui.is_item_deactivated_after_edit():
                self._autosave()
            if imgui.button("Delete selected", imgui.ImVec2(em(9), 0)):
                self.delete_roi(self.selected)
            imgui.same_line(0, em(0.6))
            self._draw_run_selection("run selected")
        elif self.selected_derived is not None:
            si, k = self.selected_derived
            s = self.derived[si]
            promoted = (s.name, k) in self._promoted
            imgui.text_disabled(
                f"{s.name} row {k}" + (" · promoted" if promoted else "")
            )
            if promoted:
                imgui.begin_disabled()
            if imgui.button("Promote", imgui.ImVec2(em(6), 0)):
                self.promote_derived(si, k)
            if promoted:
                imgui.end_disabled()
            imgui.same_line(0, em(0.6))
            if imgui.button("Discard", imgui.ImVec2(em(6), 0)):
                self.discard_derived(si, k, advance=True)
            imgui.same_line(0, em(0.6))
            if imgui.button(
                "Reject" if s.accepted[k] else "Accept", imgui.ImVec2(em(6), 0)
            ):
                self.set_accepted(si, k)
        else:
            imgui.text_disabled("select an ROI to note")
            imgui.begin_disabled()
            imgui.button("Delete selected", imgui.ImVec2(em(9), 0))
            imgui.end_disabled()
        self._draw_run_all()

    def _draw_run_selection(self, label: str):
        """Run the selection through the engine, from where the selection already is."""
        indices = self.selection_indices()
        why = "no data path to write beside" if self.fpath is None else None
        if why is None and not indices:
            why = "nothing selected"
        if why is not None:
            imgui.begin_disabled()
        if imgui.button(f"{RUN_ICON} {label} ({self.engine})", imgui.ImVec2(em(12), 0)):
            self.run_selection(indices)
        if why is not None:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                why
                or f"Run {len(indices)} ROI(s) through {self.engine} ({self._where_label()});"
                f" the rows land in the Traces tab (shift+T)"
            )

    def _draw_run_all(self):
        """The run-all row pinned under the table.

        Right-aligned so it sits under the table's per-row action icons and
        carries the same two: run every listed drawn ROI through the
        engine the Process tab's ROIs pipeline is set to, or quick trace
        them all.
        """
        listed = self.listed_drawn()
        run_label = f"{RUN_ICON} run all"
        trace_label = f"{TRACE_ICON} trace all"
        pad = imgui.get_style().frame_padding.x * 2
        gap = em(0.6)
        width = (
            imgui.calc_text_size(run_label).x
            + imgui.calc_text_size(trace_label).x
            + 2 * pad
            + gap
        )
        avail = imgui.get_content_region_avail().x
        if width < avail:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - width)
        no_run = not listed or self.fpath is None
        if no_run:
            imgui.begin_disabled()
        if imgui.button(run_label):
            self.run_in_view()
        if no_run:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "no data path to write beside"
                if self.fpath is None
                else "no drawn ROIs listed"
                if not listed
                else f"Run all {len(listed)} listed ROIs through {self.engine} "
                f"-> {self.run_prefix}{self.effective_tag}/ "
                f"({self._where_label()})"
            )
        imgui.same_line(0, gap)
        why = "no drawn ROIs listed" if not listed else self.trace_disabled(listed[0])
        if why is not None:
            imgui.begin_disabled()
        if imgui.button(trace_label):
            self.trace_in_view()
        if why is not None:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                why
                or f"Quick trace all {len(listed)} listed ROIs ({self._where_label()})"
            )

    # row actions: callbacks take a table row index and route per kind

    def _act_run(self, row: int):
        si, k = self.rows[row]
        if si < 0:
            self.run_roi(k)

    def _run_disabled(self, row: int) -> str | None:
        si, _k = self.rows[row]
        if si >= 0:
            return "promote first"
        return "no data path to write beside" if self.fpath is None else None

    def _act_trace(self, row: int):
        si, k = self.rows[row]
        if si < 0:
            self.quick_trace(k)

    def _trace_row_disabled(self, row: int) -> str | None:
        si, k = self.rows[row]
        if si >= 0:
            return "promote first"
        return self.trace_disabled(k)

    def _act_remove(self, row: int):
        # mutating mid-table-draw rebuilds the rows the clipper is still
        # iterating; run it once draw_roi_table has returned
        si, k = self.rows[row]
        self._pending_row_action = ("remove", si, k)

    @property
    def row_actions(self) -> tuple[RowAction, ...]:
        return (
            RowAction(
                RUN_ICON,
                f"Run - {self.engine} this ROI ({self._where_label()})",
                self._act_run,
                self._run_disabled,
            ),
            RowAction(
                TRACE_ICON,
                f"Quick trace - mean of this ROI per frame ({self._where_label()})",
                self._act_trace,
                self._trace_row_disabled,
            ),
            RowAction(
                REMOVE_ICON,
                "Remove - delete the drawn ROI, discard the algo one",
                self._act_remove,
            ),
        )

    def _lines_for_uid(self, uid):
        """``(header, [(label, key), ...])`` for one ROI uid, or None."""
        rows = self.traces.for_roi(uid)
        if not rows:
            return None
        index = self.store.uid_index(uid)
        header = f"ROI {index}" if index is not None else f"uid {uid}"
        return header, [(self._trace_label(t), t.key) for t in rows]

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
            si, k = self.selected_derived
            s = self.derived[si]
            key = ("member", s.name, k)
            if key in self.traces:
                return [key]
            index = self._promoted.get((s.name, k))
            if index is None:
                return []
            return [t.key for t in self.traces.for_roi(self.store.rois[index].uid)]
        if self.selected < 0:
            return []
        return [t.key for t in self.traces.for_roi(self.store.rois[self.selected].uid)]

    def _sync_trace_sel(self):
        """Point the plotted set at the selection, so the trace in the top
        panel is the ROI the image is showing. A multi-row (ctrl+click)
        selection that already covers it is left alone.
        """
        keys = set(self._selection_trace_keys())
        if keys and self.trace_sel & keys:
            return
        if self.trace_sel != keys:
            self.trace_sel = keys
            self._trace_fit = True

    def _trace_target(self):
        """``(header, [(label, key), ...])`` for the shown ROI, or None.

        With something selected this is that ROI's traces and nothing else:
        the plot must never show an ROI the image is not showing. Only with
        no selection does it fall back to the last trace collected, then to
        any row that has one.
        """
        if self.selected_derived is not None or self.selected >= 0:
            keys = self._selection_trace_keys()
            if not keys:
                return None
            if self.selected_derived is not None:
                si, k = self.selected_derived
                name = self.derived[si].name
                return f"{name} row {k}", [
                    (self._trace_label(self.traces.get(key)), key) for key in keys
                ]
            return self._lines_for_uid(self.store.rois[self.selected].uid)
        candidates = [self.trace_uid] + [t.uid for t in self.traces if t.stands_for_roi]
        for uid in candidates:
            got = self._lines_for_uid(uid)
            if got is not None:
                return got
        # rows that stand for no drawn ROI (a results file's lines): the first
        for trace in self.traces:
            if not trace.stands_for_roi:
                return trace.source, [(self._trace_label(trace), trace.key)]
        return None

    def _binning_tag(self, key) -> str:
        """`` x10`` when a trace was taken at a different frame averaging than
        the data now shows — those traces are on another time base.
        """
        trace = self.traces.get(key)
        taken = int(trace.frame_average if trace is not None else 1) or 1
        now = int(getattr(self.host, "frame_average", 1) or 1)
        return f" x{taken}" if taken != now and taken > 1 else ""

    def fs(self) -> float | None:
        """Sampling rate of the data behind the view in Hz, or None.

        Read once from the array's metadata; without one the trace plot can
        only offer frame units.
        """
        if not self._fs_read:
            self._fs_read = True
            movie = self.movie()
            meta = getattr(getattr(movie, "arr", None), "metadata", None)
            if meta:
                try:
                    from mbo_utilities.metadata import get_param

                    rate = get_param(dict(meta), "fs")
                    self._fs_value = float(rate) if rate else None
                except Exception:
                    self.logger.debug("no usable fs in metadata", exc_info=True)
        return self._fs_value

    def x_units(self) -> tuple[str, ...]:
        """X axis units on offer: frames always, time only with an ``fs``."""
        return X_UNITS if self.fs() else X_UNITS[:1]

    @property
    def x_unit(self) -> str:
        """The x axis unit: the one picked, else seconds when the data has a rate."""
        if self._x_unit is None:
            return "seconds" if self.fs() else "frames"
        return self._x_unit

    @x_unit.setter
    def x_unit(self, unit: str) -> None:
        self._x_unit = unit

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

    def neuropil_offered(self, rows) -> bool:
        """Whether a plotted row's pipeline measured a neuropil to correct with."""
        return any(
            trace_profile(t.engine).neuropil and t.Fneu is not None for t in rows
        )

    def plot_y_label(self, rows) -> str:
        """The y axis label of what the rows show: one when they agree, else joined."""
        labels = list(dict.fromkeys(y_label(t, self.kind) for t in rows))
        return " / ".join(label for label in labels if label) or DISPLAY_KINDS["dff"]

    def _plotted_rows(self, lines) -> list[RoiTrace]:
        return [
            t for t in (self.traces.get(key) for _label, key in lines) if t is not None
        ]

    def _redisplay(self) -> None:
        """The rows read differently now: drop the cached arrays and refit."""
        self._trace_display.clear()
        self._trace_stats.clear()
        self._trace_window_cache.clear()
        self._trace_fit = True

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

    def _display(self, key) -> tuple:
        """Cached ``(trace, neuropil)`` display arrays for one trace key, in
        the panel's kind and dF/F settings (``annotation.display``).
        """
        got = self._trace_display.get(key)
        if got is None:
            trace = self.traces.get(key)
            if trace is None:
                return None, None
            y = display_trace(trace, self.kind, self.dff, self.correct_neuropil)
            if y is None:
                return None, None
            yneu = neuropil_overlay(trace, self.kind, self.dff)
            if yneu is not None:
                yneu = np.ascontiguousarray(yneu, np.float32)
            got = (np.ascontiguousarray(y, np.float32), yneu)
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
        """``(header, [(label, key), ...])``: the checked trace-table rows,
        else whatever the current selection points at.

        When the checked rows are exactly the selection's own traces — which
        is what selecting an ROI leaves behind — the plot is titled after the
        ROI rather than counted, so it reads as "this is what the image is
        showing".
        """
        if self.trace_sel and self.trace_sel != set(self._selection_trace_keys()):
            lines = []
            for key in sorted(self.trace_sel, key=repr):
                trace = self.traces.get(key)
                if trace is None:
                    continue
                _v, shown = self._trace_shown(key)
                if trace.stands_for_roi:
                    lines.append((f"{shown} · {self._trace_label(trace)}", key))
                else:
                    lines.append((f"{trace.source} · {shown}", key))
            if lines:
                return f"{len(lines)} selected", lines
        return self._trace_target()

    def draw_traces(self):
        """The Traces panel: the trace-table selection (else the shown ROI)
        as pannable, zoomable lines, the cursor bound to the viewer's t, and
        under it, on the same time axis, the motion correction the recording
        went through (``MC``) when it has one.
        """
        target = self._plot_lines()
        motion = self.motion if self.motion else None
        _changed, self.show_trace = imgui.checkbox("Trace", self.show_trace)
        set_tooltip(
            "The selected traces; off gives the motion plot the whole panel.",
            show_mark=False,
        )
        imgui.same_line(0, 12)
        # the whole recording went through its motion correction, so it
        # applies to every trace: always offered, on by default
        if motion is not None:
            _changed, self.show_motion = imgui.checkbox("MC", self.show_motion)
            set_tooltip(
                f"{motion.y_label}: the motion correction the whole recording went "
                "through, so it applies to every trace; drawn under the trace on "
                "the same time axis.",
                show_mark=False,
            )
        else:
            imgui.begin_disabled()
            imgui.checkbox("MC", False)
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip("This recording carries no motion correction.")
        rows = [] if target is None else self._plotted_rows(target[1])
        # the pipelines behind the plotted rows decide what the panel offers
        if self.neuropil_offered(rows):
            imgui.same_line(0, 12)
            changed, self.correct_neuropil = imgui.checkbox(
                "neuropil corrected", self.correct_neuropil
            )
            set_tooltip(
                "subtract the pipeline's neuropil (0.7 x Fneu) from the raw trace before "
                "dF/F; offered by suite2p and the mean engine's ring, never by masknmf",
                show_mark=False,
            )
            if changed:
                self._redisplay()
        kinds = self.kind_options(rows)
        if kinds:
            imgui.same_line(0, 12)
            shown = [displayed_kind(t, self.kind) for t in rows]
            current = self.kind if self.kind in kinds else shown[0]
            imgui.set_next_item_width(em(6.5))
            changed, sel = imgui.combo(
                "##trace_kind", kinds.index(current), list(kinds)
            )
            set_tooltip(
                "What each row shows: raw, its pipeline's dF/F (or one computed here), "
                "denoised, z-score. A row without that kind shows its pipeline's default.",
                show_mark=False,
            )
            if changed:
                self.kind = kinds[sel]
                self._redisplay()
            if any(s == "dff" and t.norm is None for t, s in zip(rows, shown)):
                imgui.same_line(0, 6)
                if imgui.small_button("dF/F##dff_settings"):
                    imgui.open_popup("##dff_settings")
                set_tooltip(
                    "The baseline of a dF/F computed here from a raw trace",
                    show_mark=False,
                )
                self._draw_dff_settings(rows)
        imgui.same_line(0, 12)
        changed, self.autofit = imgui.checkbox("autofit", self.autofit)
        set_tooltip(
            "Refit the axes to whatever is plotted. Turn it off to keep the "
            "stretch you zoomed to while stepping through ROIs; double-click "
            "the plot to fit it once.",
            show_mark=False,
        )
        if changed and self.autofit:
            self._force_fit = True
        imgui.same_line(0, 10)
        units = self.x_units()
        if self.x_unit not in units:
            self.x_unit = units[0]
        imgui.set_next_item_width(em(5.5))
        changed, sel = imgui.combo("##x_unit", units.index(self.x_unit), list(units))
        set_tooltip(
            "X axis units"
            + ("" if self.fs() else " - no fs in the metadata, so frames only"),
            show_mark=False,
        )
        if changed:
            self.x_unit = units[sel]
            # the axis means something else now, so the old range would not
            # show anything sensible
            self._force_fit = True
        # a host showing a file of several recordings offers the picture this
        # one's lines or patches were drawn on (the MESc tab's reference image)
        if callable(getattr(self.host, "reference_view", None)):
            imgui.same_line(0, 12)
            if imgui.small_button("Reference image##traces"):
                self.host.reference_view()
            set_tooltip(
                "The picture this recording's lines or patches were drawn on, with "
                "them drawn and the slider's ROI thick. Click one there to select it.",
                show_mark=False,
            )
        imgui.same_line(0, 12)
        if target is not None:
            header, lines = target
            proj, size = self._window_spec()
            window = f" · {proj} {size}" if size > 1 else ""
            imgui.text_disabled(
                (f"{self.unit} · " if self.unit else "")
                + f"{header}, frame {self.current_frame()}{window}"
            )
            set_tooltip(
                "drag pans, scroll zooms, double-click fits · "
                "shift+scroll zooms x only, alt+scroll zooms y only",
                show_mark=False,
            )
        elif self.pending_traces is not None:
            self.pending_traces()
        else:
            imgui.text_disabled(
                f"No traces yet. Use {TRACE_ICON} on a row of the ROIs tab, or the Process tab's ROIs pipeline."
            )
        show_trace = self.show_trace and target is not None
        show_motion = motion is not None and self.show_motion
        self._traces_panel.height = MOTION_PANEL_HEIGHT if show_motion else PANEL_HEIGHT
        linked = show_trace and show_motion
        if linked != self._motion_linked:
            # in or out of the subplots both plots are new to implot
            self._motion_linked = linked
            self._trace_fit = True
            if motion is not None:
                motion.refit()
        if not show_trace and not show_motion:
            return
        height = max(imgui.get_content_region_avail().y - 4, 60.0)
        # one scope over both plots: they stack in the same panel, so a frame
        # around either would be a box around half of it
        with plot_style():
            if linked:
                link = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_title
                with subplots(
                    "##roi_trace_sub",
                    2,
                    1,
                    height,
                    flags=link,
                    ratios=self._motion_ratios,
                ) as ok:
                    if ok:
                        self._draw_trace_plot(lines, -1.0)
                        self._draw_motion_plot(motion, -1.0)
            elif show_trace:
                self._draw_trace_plot(lines, height)
            else:
                self._draw_motion_plot(motion, height)

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

    def _draw_motion_plot(self, motion: MotionPlot, height: float) -> None:
        """The motion plot in the trace plot's x units with the playhead on
        it; dragging the playhead scrubs the movie.
        """
        plot = self.plot_axis()
        moved, held = motion.draw(
            "##roi_motion_plot",
            height,
            cursor=plot.units(self.playhead.time),
            cursor_id=1,
            x_per_second=plot.per_second,
            x_label=X_AXIS_LABELS[self.x_unit],
        )
        if held and moved is not None:
            self.playhead.seek(plot.seconds(moved), source="motion_plot")

    def _draw_trace_plot(self, lines, height: float) -> None:
        """The trace plot; inside subplots ``height`` is the cell's."""
        if implot.get_current_context() is None:
            implot.create_context()
        key = tuple(label for label, _ in lines)
        if key != self._plot_key:
            self._plot_key = key
            self._trace_fit = True
        fit = (self._trace_fit and self.autofit) or self._force_fit
        self._trace_fit = False
        self._force_fit = False
        if fit:
            implot.set_next_axes_to_fit()
        flags = implot.Flags_.no_title
        if len(lines) <= 1:
            flags |= implot.Flags_.no_legend
        if not implot.begin_plot("##roi_trace_plot", imgui.ImVec2(-1, height), flags):
            return
        try:
            # implot has no axis lock and its OverrideMod swallows input, so shift
            # and alt drop input on the other axis for this frame
            io = imgui.get_io()
            none = implot.AxisFlags_.none
            locked = implot.AxisFlags_.lock
            implot.setup_axes(
                X_AXIS_LABELS[self.x_unit],
                self.plot_y_label(self._plotted_rows(lines)),
                locked if io.key_alt else none,
                locked if io.key_shift else none,
            )
            ctrl = io.key_ctrl
            plot = self.plot_axis()
            for label, tkey in lines:
                y, yneu = self._display(tkey)
                if y is None:
                    continue
                # each row sits where it was recorded
                xscale, xstart = self.trace_axis(self.traces.get(tkey)).on(plot)
                rgb = self._trace_color(tkey)
                if rgb is not None:
                    implot.push_colormap(_line_colormap(rgb))
                implot.plot_line(
                    label,
                    self._windowed(y),
                    xscale=xscale,
                    xstart=xstart,
                    spec=implot.Spec(line_weight=_TRACE_WEIGHT),
                )
                if rgb is not None:
                    implot.pop_colormap()
                if yneu is not None:
                    implot.push_colormap(_fneu_colormap())
                    implot.plot_line(
                        f"{label} Fneu",
                        self._windowed(yneu),
                        xscale=xscale,
                        xstart=xstart,
                    )
                    implot.pop_colormap()
                pair = self._key_to_pair(tkey)
                if pair is not None:
                    # ctrl+click a legend entry: toggle its ROI in the group
                    if (
                        ctrl
                        and implot.is_legend_entry_hovered(label)
                        and imgui.is_mouse_clicked(0)
                    ):
                        self.buffer_toggle(*pair)
                    if implot.begin_legend_popup(label):
                        in_group = pair in self.buffer
                        if imgui.menu_item_simple(
                            "remove from group" if in_group else "add to group"
                        ):
                            self.buffer_toggle(*pair)
                        if imgui.menu_item_simple("select this ROI"):
                            self.buffer_clear()
                            if pair[0] < 0:
                                self.select_roi(pair[1])
                            else:
                                self.select_derived(*pair)
                        implot.end_legend_popup()
            if self.tdim is not None:
                moved, at = implot.drag_line_x(
                    0, plot.units(self.playhead.time), _CURSOR_COLOR, 1.5
                )[:2]
                if moved:
                    self.playhead.seek(plot.seconds(at), source="trace_plot")
        finally:
            implot.end_plot()

    def _sorted_trace_rows(self) -> list[tuple]:
        """Trace-table keys in the order the table shows them, so stepping
        with the arrows walks what the user sees.
        """
        rows = self._trace_rows()
        col, ascending = self._trace_sort

        def sort_key(key):
            n, _mean, peak, _snr = self._trace_stat(key)
            _shown, z, c, engine, source = self._trace_cells(key)
            values = {
                "id": self._trace_shown(key)[0],
                "z": z,
                "c": c,
                "source": source,
                "frames": n,
                "peak": peak,
            }
            return values.get(TRACE_COLUMNS[min(col, len(TRACE_COLUMNS) - 1)][0], 0)

        rows.sort(key=sort_key, reverse=not ascending)
        return rows

    def _trace_rows(self) -> list[tuple]:
        """Every listable trace key: the table's rows, in insertion order."""
        return self.traces.keys

    def _trace_cells(self, key) -> tuple:
        """The text of one row's roi / z / c / engine / source columns."""
        trace = self.traces.get(key)
        _v, shown = self._trace_shown(key)
        if trace is None:
            return (shown, "", "", "", "")
        # a results file's rows have no slice to name
        placed = (
            trace.stands_for_roi
            or trace.source == FULL_IMAGE
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
            y, _ = self._display(key)
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
        rows); an algo component is its trace, so it is discarded.
        """
        self.trace_sel.discard(key)
        pair = self._key_to_pair(key)
        if pair is not None and pair[0] >= 0:
            si, k = pair
            self.discard_derived(si, k, advance=self.selected_derived == (si, k))
            return
        if self.traces.remove(key) is not None:
            self.status = "deleted trace"

    def draw_trace_table(self, keys=None, table_id: str = "##trace_table"):
        """The right bar's Traces tab: every collected trace with stats;
        click selects one, ctrl+click several — the selection is what the
        top strip's Traces panel plots. ``keys`` narrows the rows to those (the Process
        tab shows the ROIs it is about to run).
        """
        rows = self._sorted_trace_rows()
        if keys is not None:
            wanted = set(keys)
            rows = [key for key in rows if key in wanted]
        if not rows:
            imgui.text_disabled(
                f"No traces yet. Use {TRACE_ICON} on a row of the ROIs tab, or run the "
                "Process tab's ROIs pipeline."
            )
            return
        imgui.text_disabled(f"{len(rows)} traces · {len(self.trace_sel)} plotted")
        imgui.same_line(0, 12)
        if imgui.small_button("plot all"):
            self.trace_sel = set(rows)
            self._trace_fit = True
        imgui.same_line(0, 6)
        # "clear" here used to empty the plot, which reads as "delete these"
        # next to a table of traces; the two are separate buttons now
        if imgui.small_button("unplot"):
            self.trace_sel.clear()
            self._trace_fit = True
        set_tooltip("Take every trace off the plot; the rows stay", show_mark=False)
        imgui.same_line(0, 6)
        with danger_button():
            delete_all = imgui.small_button("delete all")
        set_tooltip(
            f"Delete all {len(rows)} traces (drawn ROIs keep their masks; "
            "algo components are discarded)",
            show_mark=False,
        )
        # stretch, not fit-to-content: the tab is a narrow column and
        # fixed-width columns ran off its right edge. frames and peak start
        # hidden — right-click the header to bring them back.
        flags = (
            imgui.TableFlags_.sortable
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.scroll_y
            | imgui.TableFlags_.resizable
            | imgui.TableFlags_.hideable
            | imgui.TableFlags_.sizing_stretch_prop
        )
        avail = imgui.get_content_region_avail()
        if not imgui.begin_table(
            table_id, len(TRACE_COLUMNS), flags, imgui.ImVec2(0, avail.y)
        ):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        stretch = imgui.TableColumnFlags_.width_stretch
        # z only says something on data with more than one plane
        flat = {"z": self.store.axis_size("z") <= 1}
        for i, (name, weight, hidden) in enumerate(TRACE_COLUMNS):
            column_flags = stretch
            if i == 0:
                column_flags |= imgui.TableColumnFlags_.default_sort
            if hidden or flat.get(name, False):
                column_flags |= imgui.TableColumnFlags_.default_hide
            if not name:  # the delete button: nothing to sort or hide
                column_flags |= (
                    imgui.TableColumnFlags_.no_sort | imgui.TableColumnFlags_.no_hide
                )
            # the z and c columns take the data's own axis names (ROI, Channel)
            header = self.axis_label(name) if name in ("z", "c") else name
            imgui.table_setup_column(header, column_flags, weight)
        imgui.table_headers_row()
        set_tooltip("Right-click a header to show or hide columns", show_mark=False)
        specs = imgui.table_get_sort_specs()
        if specs is not None and specs.specs_dirty:
            if specs.specs_count > 0:
                self._trace_sort = (
                    int(specs.specs.column_index),
                    specs.specs.sort_direction == imgui.SortDirection.ascending,
                )
            specs.specs_dirty = False
        ctrl = imgui.get_io().key_ctrl
        pending = None
        for key in rows:
            tag = "_".join(str(part) for part in key)
            imgui.table_next_row()
            imgui.table_next_column()
            shown, z_text, c_text, engine, source = self._trace_cells(key)
            picked = key in self.trace_sel
            rgb = self._trace_color(key)
            if rgb is not None:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*rgb, 1.0))
            clicked, _ = imgui.selectable(
                f"{shown}##tr_{tag}",
                picked,
                imgui.SelectableFlags_.span_all_columns
                # the row spans the delete button's column too; without this
                # it takes the hover and the button never sees a click
                | imgui.SelectableFlags_.allow_overlap,
            )
            if rgb is not None:
                imgui.pop_style_color()
            if clicked:
                if ctrl:
                    self.toggle_trace(key)
                else:
                    self.select_trace(key)
            trace = self.traces.get(key)
            pos = self._line_position(trace) if trace is not None else {}
            if trace is not None and imgui.is_item_hovered():
                # what this row is and how it was extracted, and for a line row
                # where it was collected: the line's ends, length and spacing
                tip = f"{shown}: {trace.name}"
                if engine:
                    tip += f"\nextracted with {engine}"
                if pos.get("start_um") is not None:
                    (x0, y0), (x1, y1) = pos["start_um"][:2], pos["end_um"][:2]
                    tip += (
                        f"\nline ({x0:.0f}, {y0:.0f}) -> ({x1:.0f}, {y1:.0f}) um, "
                        f"{pos['length_um']:.1f} um long"
                        + (
                            f", {pos['sample_um']:.2f} um per sample"
                            if pos.get("sample_um")
                            else ""
                        )
                    )
                imgui.set_tooltip(tip)
            n, _mean, peak, _snr = self._trace_stat(key)
            for text, numeric in (
                (z_text, True),
                (c_text, True),
                (source, False),
                (f"{n}", True),
                (f"{peak:.1f}", True),
            ):
                if not imgui.table_next_column():
                    continue
                if numeric:
                    room = (
                        imgui.get_content_region_avail().x
                        - imgui.calc_text_size(text).x
                    )
                    if room > 0:
                        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + room)
                imgui.text(text)
            if imgui.table_next_column():
                if imgui.small_button(f"{REMOVE_ICON}##del_{tag}"):
                    # deleting mid-draw rebuilds the rows this loop is
                    # walking; do it once the table has ended
                    pending = key
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Delete - this trace (a drawn ROI keeps its mask; "
                        "an algo component is discarded)"
                    )
        imgui.end_table()
        if delete_all:
            for key in rows:
                self.delete_trace_row(key)
            self.status = f"deleted {len(rows)} traces"
        elif pending is not None:
            self.delete_trace_row(pending)


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
