"""Suite2p ROI diagnostics."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.diagnostics import DiagnosticsWidget


class DiagnosticsApp(App):
    """The ROI diagnostics table and its plots, over a plane directory.

    It opens on its own load prompt rather than on the host's data: what it
    reads is a suite2p output directory, not the movie on the canvas.
    """

    id = "diagnostics"
    title = "Diagnostics"
    window = True
    order = 80
    window_size = (1100, 700)

    def __init__(self):
        super().__init__()
        self.widget = DiagnosticsWidget()

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        self.widget.draw()
