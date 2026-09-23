"""One figure edge, shared by every app that asked for a panel there."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from mbo_utilities.gui.app._app import draw_guarded

if TYPE_CHECKING:
    from mbo_utilities.gui.app._app import App
    from mbo_utilities.gui.app._host import AppHost


class Dock(ImguiWindow):
    """The apps panelled on one edge, drawn as tabs when there is more than one.

    fastplotlib keeps one imgui window per edge, so apps cannot each claim
    one. They register here instead and the dock multiplexes them: it takes
    the edge when the first app opens and gives it back when the last one
    closes, so an edge with nothing on it costs no canvas.
    """

    def __init__(self, host: AppHost, edge: str):
        super().__init__()
        self.host = host
        self.edge = edge

    def showing(self) -> list[App]:
        return [
            app
            for app in self.host.ordered()
            if app.dock == self.edge and app.open and app.available(self.host)
        ]

    def sync(self) -> None:
        """Claim, resize or release the edge to match what is open."""
        apps = self.showing()
        held = self.host.figure.imgui_windows.get(self.edge) is self
        if not apps:
            if held:
                self.host.figure.remove_imgui_window(self.edge)
            return
        want = max(app.size for app in apps)
        if not held:
            self.host.figure.add_imgui_window(self, location=self.edge, size=want, title=None)
        elif self.size != want:
            self.size = want

    def update(self) -> None:
        apps = self.showing()
        if not apps:
            return
        if len(apps) == 1:
            self._body(apps[0])
            return
        if imgui.begin_tab_bar(f"##dock_{self.edge}"):
            for app in apps:
                if imgui.begin_tab_item(f"{app.title}###{app.id}", None, imgui.TabItemFlags_.none)[0]:
                    self._body(app)
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    def _body(self, app: App) -> None:
        draw_guarded(app, self.host)
