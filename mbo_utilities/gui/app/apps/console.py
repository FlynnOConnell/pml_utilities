"""The Process Console, and the status button on the menu bar that opens it."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._popups import draw_console_body
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.process_manager import get_process_manager


def process_status(progress_items: list) -> tuple[str, str, imgui.ImVec4]:
    """The status button's label, its widest label and its colour.

    Counts spawned processes, in-process jobs and ``progress_items`` (the
    caller's own running work). The widest label fixes the button's width
    while a running percentage changes, so the click target stays put.
    """
    try:
        from imgui_bundle import icons_fontawesome as fa

        icon_idle = fa.ICON_FA_CIRCLE
        icon_running = fa.ICON_FA_SPINNER
        icon_error = fa.ICON_FA_EXCLAMATION_TRIANGLE
        icon_check = fa.ICON_FA_CHECK_CIRCLE
    except (ImportError, AttributeError):
        icon_idle, icon_running, icon_error, icon_check = (
            "\uf111",
            "\uf110",
            "\uf071",
            "\uf058",
        )

    pm = get_process_manager()
    pm.cleanup_finished()
    # in-process jobs count the same as spawned ones, so a click shows up somewhere
    procs = [*pm.get_running(), *pm.get_jobs()]
    running = [p for p in procs if p.is_alive()]
    completed = [p for p in procs if not p.is_alive() and p.status == "completed"]
    errors = [p for p in procs if not p.is_alive() and p.status == "error"]
    n_running = len(running) + sum(1 for i in progress_items if not i.get("done"))

    if errors:
        text = f"{icon_error} Error ({len(errors)})"
        return text, text, imgui.ImVec4(0.8, 0.2, 0.2, 1.0)
    if n_running:
        color = imgui.ImVec4(0.85, 0.45, 0.0, 1.0)
        if progress_items:
            done = sum(i["progress"] for i in progress_items) / len(progress_items)
            return (
                f"{icon_running} Running ({n_running}) {int(done * 100)}%",
                f"{icon_running} Running ({n_running}) 100%",
                color,
            )
        text = f"{icon_running} Running ({n_running})"
        return text, text, color
    green = imgui.ImVec4(0.15, 0.55, 0.15, 1.0)
    if completed:
        word = "task" if len(completed) == 1 else "tasks"
        text = f"{icon_check} Completed {len(completed)} {word}"
        return text, text, green
    text = f"{icon_idle} Console: Idle"
    return text, text, green


def draw_status_button(text: str, widest: str, color: imgui.ImVec4) -> bool:
    """Draw the rounded status button in ``color``; True when it was clicked."""
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)
    imgui.push_style_color(imgui.Col_.button, color)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1, 1, 1, 1))
    imgui.push_style_color(
        imgui.Col_.button_hovered,
        imgui.ImVec4(
            min(color.x + 0.1, 1.0),
            min(color.y + 0.1, 1.0),
            min(color.z + 0.1, 1.0),
            1.0,
        ),
    )
    imgui.push_style_color(imgui.Col_.button_active, color)
    width = imgui.calc_text_size(widest).x + imgui.get_style().frame_padding.x * 2.0
    clicked = imgui.button(f"{text}##process_status", imgui.ImVec2(width, 0.0))
    imgui.pop_style_color(4)
    imgui.pop_style_var()
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
    return clicked


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
