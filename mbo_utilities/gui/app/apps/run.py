"""The Process tab: the registered pipelines, run on the open data."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.pipelines import draw_run_tab, start_preload


class RunApp(App):
    """The pipelines that apply to the open data, each with its settings and run button.

    The pipeline widgets draw themselves against the host's ``context``;
    which one is selected and each widget's state survive opening other
    data, and the context starts the per-dataset defaults over.
    """

    id = "run"
    title = "Process"
    dock = "right"
    order = 2
    size = 380

    def __init__(self):
        super().__init__()
        start_preload()

    def available(self, host) -> bool:
        return host.context is not None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        draw_run_tab(host.context)
