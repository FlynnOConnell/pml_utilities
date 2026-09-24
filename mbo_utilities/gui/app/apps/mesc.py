"""The measurement units of an open ``.mesc`` file, one row each."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.mesc_units import MescTabWidget, mesc_array_of


class AppMescTab(MescTabWidget):
    """The MESc tab, showing a unit by opening it into the host.

    ``host.set_data`` is the one event every app rebuilds from, so the
    viewer swap, the per-unit ROI parking and the line-scan traces all
    follow from it instead of from the tab.
    """

    def _show(self, arr, title: str) -> None:
        self.parent.host.set_data(arr)


class MescApp(App):
    """Every recording in the open ``.mesc`` with its picture, RTMC and comment;
    clicking a row opens that unit, and the picture button shows where its
    lines or patches were drawn.

    The tab outlives the unit on screen (it keeps the units it opened and the
    reference popup) and is dropped when a file that is not a ``.mesc`` opens.
    """

    id = "mesc"
    title = "MESc"
    dock = "right"
    order = 0
    size = 380
    start_open = True

    def __init__(self):
        super().__init__()
        self.tab: AppMescTab | None = None

    def available(self, host) -> bool:
        return host.context is not None and mesc_array_of(host.data) is not None

    def data_changed(self, host) -> None:
        if self.tab is not None and mesc_array_of(host.data) is None:
            self.tab.cleanup()
            self.tab = None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        if self.tab is None:
            self.tab = AppMescTab(host.context)
        self.tab.draw()

    def close(self) -> None:
        if self.tab is not None:
            self.tab.cleanup()
            self.tab = None
