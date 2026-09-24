"""Hosting an ``imgui_debugger`` Panel as an app."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mbo_utilities.gui.app._app import App

if TYPE_CHECKING:
    from imgui_debugger import Panel

    from mbo_utilities.gui.app._host import AppHost


class PanelApp(App):
    """An ``imgui_debugger`` panel, drawn by the host.

    A panel is already an app in every respect the host cares about: it has
    a visibility flag, a body, a window of its own and a hotkey. So the two
    vocabularies are joined rather than translated -- ``open`` *is*
    ``panel.visible`` -- and the panel keeps opening its own window, which
    is the only way imgui's native tools can be drawn at all.
    """

    owns_window = True
    window = True

    def __init__(self, panel: Panel, order: int = 100, store=None, menu: str = "Debug"):
        self.panel = panel
        self.store = store
        self.menu = menu
        self.id = f"panel_{type(panel).__name__}"
        self.title = panel.title
        self.order = order
        # App.__init__ writes open, which is the panel's visible; keep what
        # the panel was restored with
        self.start_open = panel.visible
        super().__init__()

    @property
    def open(self) -> bool:
        return self.panel.visible

    @open.setter
    def open(self, value: bool) -> None:
        self.panel.visible = bool(value)

    def draw_window(self, host: AppHost) -> None:
        self.panel.render_window()

    def close(self) -> None:
        if self.store is not None:
            self.store.set_panel_state(self.panel.window_id, self.panel.get_state())
