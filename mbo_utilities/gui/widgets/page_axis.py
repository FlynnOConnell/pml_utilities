"""The Page Axis panel reopens a single-file TIFF with its pages on the axis the user names: time, z-planes or channels.

A plain stack carries no axis labels, ImageJ writes ``slices`` for any stack
and tifffile's ImageJ default writes ``channels``, so the reader's guess is
about the file, not the experiment. The pick is ``imread(..., dims=...)``,
the same ``dims`` a worker gets through ``reader_kwargs``.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import hello_imgui, imgui

from mbo_utilities import log
from mbo_utilities.arrays.tiff import TiffArray
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.lazy_array import base_array
from mbo_utilities.reader import imread, source_reader_kwargs

logger = log.get("gui.page_axis")

PAGE_AXES = {"time": "TYX", "z-planes": "ZYX", "channels": "CYX"}


def page_axis_tiff(parent: Any) -> TiffArray | None:
    """The shown array when it is one TIFF file whose pages all sit on one of T, C or Z, else None."""
    data = parent.image_widget.data
    if not data:
        return None
    arr = base_array(data[0])
    if not isinstance(arr, TiffArray) or len(arr.filenames) != 1:
        return None
    populated = [n for n in (arr.nt, arr.nc, arr.nz) if n > 1]
    return arr if len(populated) == 1 else None


def page_axis_reading(arr: TiffArray) -> str:
    """The ``PAGE_AXES`` key the pages are on now."""
    if arr.nc > 1:
        return "channels"
    if arr.nz > 1:
        return "z-planes"
    return "time"


def reread_pages(parent: Any, reading: str) -> None:
    """Reopen the recording on screen with its pages on another axis.

    The ROI tool is rebuilt around the new array: its store is keyed by
    the planes the pages make, which this changes.
    """
    from mbo_utilities.gui._dialogs import swap_viewer_array
    from mbo_utilities.gui.manual_roi import attach_roi_widget, detach_roi_widget
    from mbo_utilities.gui.run_gui import _wrap_for_viewer

    arr = base_array(parent.image_widget.data[0])
    kwargs = {**source_reader_kwargs(arr), "dims": PAGE_AXES[reading]}
    roi_on = parent.manual_roi is not None
    if roi_on:
        detach_roi_widget(parent)
    reopened = _wrap_for_viewer(imread(arr.source_path, **kwargs), logger)
    swap_viewer_array(parent, reopened)
    if roi_on:
        attach_roi_widget(parent)


class PageAxisWidget(Widget):
    """A combo naming the axis a single-file TIFF's pages are read as."""

    name = "Page axis"
    priority = 56
    toggle_key = "preview"

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return page_axis_tiff(parent) is not None

    def draw(self) -> None:
        arr = page_axis_tiff(self.parent)
        if arr is None:
            return
        readings = list(PAGE_AXES)
        current = readings.index(page_axis_reading(arr))
        imgui.spacing()
        imgui.separator()
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Page Axis")
        imgui.set_next_item_width(hello_imgui.em_size(10))
        changed, picked = imgui.combo("pages are", current, readings)
        set_tooltip(
            "A plain TIFF says nothing about what its pages are. ImageJ stacks "
            "and tifffile's default open as time; pick z-planes for a z-stack "
            "or channels for a multi-channel image. Reopens the file."
        )
        if changed and picked != current:
            reread_pages(self.parent, readings[picked])
