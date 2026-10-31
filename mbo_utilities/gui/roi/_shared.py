"""Constants, keybinds, help text and small helpers shared by the ROI tool's modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui
from masknmf.visualization.imgui import (
    RoiOrder,
)
from masknmf.visualization.imgui.keybinds import CLASSIFICATION, DEMIXING, Bind

from mbo_utilities.annotation import (
    RoiLabelStore,
)
from mbo_utilities.arrays.features._slicing import index_window
from mbo_utilities.gui.imgui.table import FILTER_ALL
from mbo_utilities.lazy_array import base_array
from mbo_utilities.roi_workflow import (
    PlaneMovie,
)

PANEL_HEIGHT = (
    200  # the traces panel's height below the top strip's menu row and tab bar
)
MOTION_PANEL_HEIGHT = 300
BEHAVIOR_PLOT_HEIGHT = 140

ADOPT_INTERVAL_S = 1.0  # the finished-run check reads one progress file per process, so not every frame
DEFLECTION_MODES = ("viewer", "off", "on")
NOISE_LABELS = {
    "sd": "SD",
    "mad": "robust SD (MAD)",
}  # panel names for NOISE_ESTIMATES

TRACE_COLUMNS = (  # (name, sortable, hidden by default), after the id column
    ("z", True, False),
    ("c", True, False),
    ("source", False, True),
    ("engine", False, True),
    ("frames", True, True),
    ("peak", True, False),
)

# a row's ``extra`` keys from ``mesc_geometry.line_positions``; the row's own record wins over the recording's
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

_STAGE_NAMES = ("skip", "run", "force")  # suite2p's 0 / 1 / 2

MIN_ROI_PIXELS = 9
PIXEL_RADIUS = 2  # masknmf's 5x5 pixel average
DEFAULT_LABEL_NAMES: tuple[str, ...] = ()
ENGINE_HELP = {  # tooltips for annotation.ENGINES
    "mean": "mean - the raw mean of the pixels in each mask, frame by "
    "frame, plus a neuropil ring around it. No pipeline needed.",
    "suite2p": "suite2p - suite2p's own extractor: F from the mask and "
    "Fneu from its neuropil ring, ready for its neuropil "
    "subtraction.",
    "masknmf": "masknmf - seeded NMF: each mask's demixed signal, with light "
    "from overlapping neurons and background pulled back out.",
}
RUN_WHERE = ("drawn", "screen")
DEFAULT_RUN_TAG = "manual"
COLOR_BY = (
    "roi id",
    "class",
    "z",
    "c",
    "area",
    "peak",
)  # masknmf's id palette, a store column, or the trace peak
COLORMAPS = ("viridis", "plasma", "turbo", "coolwarm", "tab10")
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

_TRACE_MODES = {  # the plot background: near-black with nothing shown, forest green for a selection
    "normal": (0.02, 0.02, 0.03, 1.0),
    "selection": (0.05, 0.14, 0.08, 1.0),
}
_UNDO_DEPTH = 50

_OWN_KEYS: Mapping[str, Bind] = {
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
# NOTE: masknmf leaves shift open so the arrows read it as a stride; every other key must pin
# shift=False here, or shift+m toggles the masks as well as firing the viewer's own shift+m
_STRIDE_KEYS = ("up", "down", "left", "right")
ROI_KEYS: Mapping[
    str, Bind
] = {  # masknmf's DEMIXING keys and this tool's own; every handler and the popup read it
    name: bind._replace(shift=False)
    if bind.shift is None and name not in _STRIDE_KEYS
    else bind
    for name, bind in {**DEMIXING, **_OWN_KEYS}.items()
}
_CLAIMED_KEYS = (  # the viewer's global shortcuts leave these alone while the tool is on
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
    "<stem>.rois.zarr    the drawn ROIs, quick traces and which runs are\n"
    "                    loaded, on Save or with autosave on\n"
    "                    (mbo_utilities.annotation.RoiFile.load); its\n"
    "                    labels/drawn is an OME-NGFF 0.5 label image\n"
    "<stem>.<tag>.<stamp>.manual.zarr\n"
    "                    one run: a results zarr, one unit per slice read"
)


def help_markdown() -> str:
    """The tool's guide, as markdown for the app's help viewer.

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


def _frames_of(movie) -> dict:
    """The frame fields of a row read through ``movie``: ``frames`` when the indices are a window, else ``tp_indices``."""
    indices = movie.t_indices
    window = index_window(indices)
    if indices is None or window is not None:
        return {"frames": window}
    return {"frames": None, "extra": {"tp_indices": list(indices)}}


def unit_key(iw) -> str:
    """The recording's key when the viewer's file holds several (a MESc unit, ``MSession_0/MUnit_3``), else ``""``."""
    arr = base_array(iw.data[0]) if iw.data else None
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


@dataclass
class _RunSpec:
    """One manual run's inputs and outputs, handed to the job thread."""

    groups: dict[tuple[int, int, int], list[int]]
    movies: dict[tuple[int, int, int], PlaneMovie]
    store: RoiLabelStore
    engine: str
    tag: str
    out_path: Path
    source: Any
    settings: dict | None = None
    many_c: bool = False
    work: Path = field(default_factory=Path)


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
        self.source: int | None = (
            None  # None every source, 0 drawn, 1 + i the i-th derived set
        )
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
        hits = np.flatnonzero(  # only drawn rows take a label
            (self.labels[self.order] < 0) & (self.sources[self.order] == 0)
        )
        if not len(hits):
            return False
        after = hits[hits >= self.pos] if inclusive else hits[hits > self.pos]
        self.pos = int(after[0] if len(after) else hits[0])
        return True
