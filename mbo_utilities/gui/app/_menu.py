"""The figure's top edge: the menus that list every app, and what is on each slot."""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING

from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from mbo_utilities import log

if TYPE_CHECKING:
    from mbo_utilities.gui.app._host import AppHost

HEIGHT = 30
# the top menus in order; Debug shows only with debug logging on
MENUS = ("File", "View", "Docs", "Debug")
DOCS_URL = "https://millerbrainobservatory.github.io/mbo_utilities/"


class MenuBar(ImguiWindow):
    """The host's own controls, drawn as a menu bar on the top edge.

    Each app is listed under its ``menu`` with its shortcut beside it, and
    ticking it shows or hides the app. ``draw`` is overridden because an
    imgui menu bar has to be opened in the window itself, and the edge
    windows fastplotlib draws put their contents in a child. The frame's
    shortcuts run here too, since this is the one window drawn every frame.
    """

    def __init__(self, host: AppHost):
        super().__init__()
        self.host = host
        host.figure.add_imgui_window(
            self,
            location="top",
            size=HEIGHT,
            title=None,
            window_flags=(
                imgui.WindowFlags_.no_collapse
                | imgui.WindowFlags_.no_resize
                | imgui.WindowFlags_.no_title_bar
                | imgui.WindowFlags_.no_scrollbar
                | imgui.WindowFlags_.no_bring_to_front_on_focus
                | imgui.WindowFlags_.menu_bar
            ),
        )

    def draw(self) -> None:
        imgui.set_next_window_size((self.width, self.height))
        imgui.set_next_window_pos((self.x, self.y))
        imgui.begin(
            f"##menu_bar{self._id_counter}", p_open=None, flags=self._window_flags
        )
        if imgui.begin_menu_bar():
            for name in MENUS:
                if name != "Debug" or log.debug_enabled():
                    self._menu(name)
            self._slot_menus()
            imgui.text(f"   {self.host.status()}")
            imgui.end_menu_bar()
        imgui.end()
        self.host.keys()

    def _menu(self, name: str) -> None:
        if not imgui.begin_menu(name):
            return
        for app in self.host.ordered():
            if app.menu != name:
                continue
            if imgui.menu_item(
                app.title,
                app.shortcut,
                p_selected=app.open,
                enabled=app.available(self.host),
            )[0]:
                app.open = not app.open
        if name == "Docs":
            imgui.separator()
            if imgui.menu_item("Online docs", "", p_selected=False)[0]:
                webbrowser.open(DOCS_URL)
        imgui.end_menu()

    def _slot_menus(self) -> None:
        for slot, subplot in enumerate(self.host.slots):
            if not imgui.begin_menu(subplot.name or f"slot {slot}"):
                continue
            current = self.host.stage.get(slot)
            if imgui.menu_item("empty", "", p_selected=current is None)[0]:
                self.host.mount(None, slot)
            for app in self.host.ordered():
                if not app.scene:
                    continue
                if imgui.menu_item(
                    app.title,
                    "",
                    p_selected=current == app.id,
                    enabled=app.available(self.host),
                )[0]:
                    self.host.mount(app.id, slot)
            imgui.end_menu()
