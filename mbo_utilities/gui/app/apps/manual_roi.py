"""Drawing, labeling and tracing ROIs on the viewer by hand."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.manual_roi import (
    KEYBINDS,
    attach_roi_widget,
    detach_roi_widget,
)
from mbo_utilities.lazy_array import base_array


class ManualRoiApp(App):
    """The manual ROI widget: its ROI table and controls, and its trace table.

    Showing the app turns labeling on: the widget is built on the host's
    ``context``, draws its overlay on the viewer, hangs its trace plot off
    the top strip and handles its keys in the strip's frame hook. Hiding it
    turns labeling off and parks the ROIs and runs on the context, so they
    come back when it is shown again. Rebinning the same recording keeps
    them; opening another recording starts from what is saved beside it.
    """

    id = "manual_roi"
    title = "ROIs"
    dock = "right"
    order = 3
    size = 380
    keybinds = KEYBINDS

    def __init__(self):
        super().__init__()
        # the recording the ROIs were drawn on
        self.recording = None

    def available(self, host) -> bool:
        return host.context is not None

    def frame(self, host) -> None:
        context = host.context
        if self.open and context.manual_roi is None:
            attach_roi_widget(context)
            self.recording = base_array(host.data)
        elif not self.open and context.manual_roi is not None:
            detach_roi_widget(context)

    def data_changed(self, host) -> None:
        context = host.context
        detach_roi_widget(context)
        if base_array(host.data) is not self.recording:
            context._manual_roi_store = None
            context._manual_roi_runs = None

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
