"""Sending work to another machine: the BioHPC cluster and a cloud GPU."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._cloud import draw_cloud_tab
from mbo_utilities.gui._imgui_helpers import fit_width
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.biohpc import BioHpcPanel


class BiohpcApp(App):
    """The BioHPC sign-in, then transfers, metadata, analysis and jobs on the cluster.

    The panel keeps its session here, so it stays signed in while the
    window is closed.
    """

    id = "biohpc"
    title = "BioHPC"
    window = True
    order = 80
    window_size = (880, 620)

    def __init__(self):
        super().__init__()
        self.panel: BioHpcPanel | None = None

    def available(self, host) -> bool:
        return host.context is not None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        if self.panel is None:
            self.panel = BioHpcPanel(host.context)
        with fit_width():
            self.panel.draw()


class CloudApp(App):
    """A cloud GPU: sign in, upload the open data's folder and run on it."""

    id = "cloud"
    title = "Cloud (GPU)"
    window = True
    order = 81
    window_size = (760, 560)

    def __init__(self):
        super().__init__()
        # the panel's sign-in and run state, kept between frames
        self.state: dict = {}

    def available(self, host) -> bool:
        return host.data is not None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        draw_cloud_tab(self.state, host.data.source_path)
