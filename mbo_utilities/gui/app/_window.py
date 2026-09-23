"""A floating window for one app, with a close box."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from mbo_utilities.gui.app._app import draw_guarded

if TYPE_CHECKING:
    from mbo_utilities.gui.app._app import App
    from mbo_utilities.gui.app._host import AppHost


class AppWindow(ImguiWindow):
    """One app in a window it can be closed from.

    ``draw`` is overridden rather than ``update`` so the window is opened
    with the app's own ``open`` flag: closing it from the title bar is the
    same act as switching the app off in the Apps menu. A closed app draws
    nothing, so the window stays registered with the figure for the host's
    life and ordering never changes under the user.

    An ``owns_window`` app is handed the frame root instead and opens its
    own window, which is how imgui's own tool windows and anything else
    already calling ``imgui.begin`` is hosted unchanged.
    """

    def __init__(self, host: AppHost, app: App):
        super().__init__()
        self.host = host
        self.app = app

    def draw(self) -> None:
        app = self.app
        if not app.available(self.host):
            return
        if app.owns_window:
            app.draw_window(self.host)
            return
        if not app.open:
            return
        imgui.set_next_window_size(imgui.ImVec2(*app.window_size), imgui.Cond_.first_use_ever)
        expanded, app.open = imgui.begin(f"{app.title}###{app.id}", app.open)
        if expanded:
            draw_guarded(app, self.host)
        imgui.end()
