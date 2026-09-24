"""Sending work to another machine: the BioHPC cluster and a cloud GPU."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._cloud import draw_cloud_tab
from mbo_utilities.gui._imgui_helpers import fit_width
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.biohpc import draw_biohpc_tab
from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

# the BioHPC panel shows a sub-tab only while its entry is on
BIOHPC_TABS = (
    "biohpc",
    "biohpc.transfer",
    "biohpc.metadata",
    "biohpc.analysis",
    "biohpc.jobs",
)


class BiohpcApp(App):
    """The BioHPC sign-in, then transfers, metadata, analysis and jobs on the cluster.

    The panel keeps its session on the host's ``context``, so it stays
    signed in while the window is closed.
    """

    id = "biohpc"
    title = "BioHPC"
    window = True
    order = 80
    window_size = (880, 620)

    def available(self, host) -> bool:
        return host.context is not None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        for key in BIOHPC_TABS:
            set_widget_enabled(key, True, persist=False)
        with fit_width():
            draw_biohpc_tab(host.context)


class CloudApp(App):
    """A cloud GPU: sign in, upload the open data's folder and run on it."""

    id = "cloud"
    title = "Cloud (GPU)"
    window = True
    order = 81
    window_size = (760, 560)

    def available(self, host) -> bool:
        return host.context is not None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        draw_cloud_tab(host.context, host.context.fpath)
