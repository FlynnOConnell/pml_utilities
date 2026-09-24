"""The Process Console, and the status button on the menu bar that opens it."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._popups import draw_console_body
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.menu_bar import draw_status_button, process_status


class ConsoleApp(App):
    """System meters, every app's running work, in-process jobs and background
    processes with their logs.

    The status button at the right of the menu bar is coloured by what is
    running and shows or hides the console. What the apps are doing comes
    from each app's ``progress``, so nothing here names another app.
    """

    id = "console"
    title = "Process Console"
    window = True
    order = 60
    window_size = (520, 420)

    def work(self, host) -> list[dict]:
        """Every app's running work, in menu order."""
        return [item for app in host.ordered() for item in app.progress(host)]

    def draw_menu_bar(self, host) -> None:
        text, widest, color = process_status(self.work(host))
        style = imgui.get_style()
        width = imgui.calc_text_size(widest).x + style.frame_padding.x * 2.0
        x = imgui.get_window_width() - width - style.item_spacing.x
        if x > imgui.get_cursor_pos_x():
            imgui.set_cursor_pos_x(x)
        if draw_status_button(text, widest, color):
            self.open = not self.open

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        draw_console_body(self, self.work(host))
