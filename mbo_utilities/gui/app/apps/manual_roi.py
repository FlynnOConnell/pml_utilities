"""Drawing, labeling and tracing ROIs on the viewer by hand."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities import log
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.linescan_viewer import attach_standard_traces
from mbo_utilities.gui.manual_roi import (
    KEYBINDS,
    attach_roi_widget,
    detach_roi_widget,
)
from mbo_utilities.lazy_array import base_array

logger = log.get("gui.app")


def recording_key(arr) -> tuple:
    """Which recording ``arr`` is: its file and the unit within it for a ``.mesc``,
    or the array itself when it lives only in memory.
    """
    if arr.source_path is None:
        return ("memory", id(arr))
    return (str(arr.source_path), getattr(arr, "unit_key", None))


class ManualRoiApp(App):
    """The manual ROI widget: its ROI table and controls, and its trace table.

    Showing the app turns labeling on: the widget is built on the host's
    ``context``, draws its overlay on the viewer, hangs its trace plot off
    the top strip and handles its keys in the strip's frame hook; a line
    scan's per-line traces come with it. Hiding it turns labeling off and
    parks the ROIs and runs on the context. Opening another recording parks
    them under the recording they were drawn on, so switching back (another
    MESc unit and back, or a rebin of the same one) finds them where they
    were, background runs included.
    """

    id = "manual_roi"
    title = "ROIs"
    dock = "right"
    order = 3
    size = 380
    keybinds = KEYBINDS

    def __init__(self):
        super().__init__()
        # the recording on screen, and the ROI state of every one left for another
        self.recording = None
        self.parked: dict[tuple, tuple] = {}

    def available(self, host) -> bool:
        return host.context is not None

    def frame(self, host) -> None:
        context = host.context
        if self.open and context.manual_roi is None:
            self.recording = base_array(host.data)
            if attach_roi_widget(context) is not None:
                try:
                    attach_standard_traces(context)
                except Exception:
                    logger.warning("line-scan traces unavailable", exc_info=True)
        elif not self.open and context.manual_roi is not None:
            self._detach(context)

    def data_changed(self, host) -> None:
        context = host.context
        self._detach(context)
        if self.recording is not None:
            self.parked[recording_key(self.recording)] = (
                context._manual_roi_store,
                context._manual_roi_runs,
            )
        self.recording = base_array(host.data)
        context._manual_roi_store, context._manual_roi_runs = self.parked.pop(
            recording_key(self.recording), (None, None)
        )

    def _detach(self, context) -> None:
        if context.linescan_traces is not None:
            context.linescan_traces.close()
            context.linescan_traces = None
        detach_roi_widget(context)

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        widget = host.context.manual_roi
        if widget is None:
            imgui.text_disabled("Starting manual ROI labeling.")
            return
        if imgui.begin_tab_bar("##manual_roi"):
            if imgui.begin_tab_item("ROIs")[0]:
                widget.draw_rois()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Traces")[0]:
                if widget.has_traces():
                    widget.draw_trace_table()
                else:
                    imgui.text_disabled(
                        "No traces yet: trace a row of the ROIs tab, or run "
                        "the ROIs pipeline from the Process tab."
                    )
                imgui.end_tab_item()
            imgui.end_tab_bar()
