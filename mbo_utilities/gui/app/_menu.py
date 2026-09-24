"""The menu row at the top of the strip: the menus that list every app, and each slot's."""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities import log

if TYPE_CHECKING:
    from mbo_utilities.gui.app._host import AppHost

# the top menus in order; Debug shows only with debug logging on
MENUS = ("File", "View", "Docs", "Debug")
DOCS_URL = "https://millerbrainobservatory.github.io/mbo_utilities/"


class MenuBar:
    """The host's own controls, drawn as the menu row of the top strip.

    Each app is listed under its ``menu`` with its shortcut beside it, and
    ticking it shows or hides the app. The strip calls ``draw`` every
    frame before its panels, so the frame's shortcuts run here too.
    """

    def __init__(self, host: AppHost):
        self.host = host

    def draw(self) -> None:
        # a menu bar has to be opened in a window of its own; the strip's body is a child
        with imgui_ctx.begin_child(
            "##menu",
            window_flags=imgui.WindowFlags_.menu_bar,
            child_flags=imgui.ChildFlags_.auto_resize_y
            | imgui.ChildFlags_.always_auto_resize,
        ):
            if imgui.begin_menu_bar():
                for name in MENUS:
                    if name != "Debug" or log.debug_enabled():
                        self._menu(name)
                self._slot_menus()
                imgui.text(f"   {self.host.status()}")
                for app in self.host.ordered():
                    if app.available(self.host):
                        app.draw_menu_bar(self.host)
                imgui.end_menu_bar()
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
