"""Every shortcut on screen, read from the apps that handle them."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App

SECTION = imgui.ImVec4(0.6, 0.8, 1.0, 1.0)
KEY = imgui.ImVec4(0.9, 0.9, 0.5, 1.0)


class KeybindsApp(App):
    """The keybinds sheet: the host's own keys, then each available app's.

    Nothing here is a hand-kept list: an app's ``shortcut`` shows or hides
    it and its ``keybinds`` name the keys its ``on_keys`` handles, so a new
    app's keys appear here with no edit.
    """

    id = "keybinds"
    title = "Keybinds"
    menu = "Docs"
    shortcut = "k"
    window = True
    order = 20
    window_size = (380, 520)

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        flags = (
            imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
        )
        if not imgui.begin_table("##keybinds", 2, flags):
            return
        imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, 110)
        imgui.table_setup_column("what", imgui.TableColumnFlags_.width_stretch)
        sections = [("Window", [("p", "Fold the right panel")])]
        for app in host.ordered():
            if not app.available(host):
                continue
            rows = list(app.keybinds)
            if app.shortcut:
                rows.insert(0, (app.shortcut, f"Show or hide {app.title}"))
            if rows:
                sections.append((app.title, rows))
        for title, rows in sections:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_colored(SECTION, title)
            for chord, what in rows:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_colored(KEY, chord)
                imgui.table_next_column()
                imgui.text(what)
        imgui.end_table()
