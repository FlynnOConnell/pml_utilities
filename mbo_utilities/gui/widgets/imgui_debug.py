"""The ImGui Debugger window, opened from Widgets > ImGui Debugger.

One floating window of switches: Dear ImGui's metrics/debugger, debug log,
ID stack tool, demo and about windows. Every window here is a floating one,
drawn from ``PreviewDataWidget.draw`` every frame.
"""

from __future__ import annotations

from imgui_bundle import imgui

__all__ = [
    "TOOLS",
    "draw_imgui_debug_menu_item",
    "draw_imgui_debug_windows",
    "open_imgui_debugger",
]

# key -> (label, what it shows)
TOOLS: tuple[tuple[str, str, str], ...] = (
    (
        "metrics",
        "Metrics / Debugger",
        "Every open window, its draw commands, vertex counts and internal state.",
    ),
    (
        "debug_log",
        "Debug Log",
        "A running log of imgui events: active id, focus and window changes.",
    ),
    (
        "id_stack",
        "ID Stack Tool",
        "Hover a widget to see the id path it was registered under; the tool "
        "for two widgets that share an id.",
    ),
    (
        "demo",
        "Demo Window",
        "Dear ImGui's own demo: every widget with its source shown.",
    ),
    (
        "about",
        "About",
        "Version, backend and build flags of the running imgui.",
    ),
)

_open: dict[str, bool] = {"window": False} | {key: False for key, _, _ in TOOLS}


def open_imgui_debugger() -> None:
    """Show the ImGui Debugger window."""
    _open["window"] = True


def draw_imgui_debug_menu_item() -> None:
    """Draw the Widgets-menu entry that opens the window."""
    if imgui.menu_item("ImGui Debugger", "", _open["window"], True)[0]:
        _open["window"] = not _open["window"]


def _draw_window() -> None:
    imgui.set_next_window_size(imgui.ImVec2(300, 0), imgui.Cond_.first_use_ever)
    expanded, _open["window"] = imgui.begin("ImGui Debugger", True)
    if expanded:
        io = imgui.get_io()
        imgui.text_disabled(
            f"{io.framerate:.0f} FPS   {1000.0 / max(io.framerate, 1e-6):.1f} ms/frame"
        )
        imgui.separator()
        imgui.spacing()
        for key, label, help_text in TOOLS:
            changed, value = imgui.checkbox(label, _open[key])
            if imgui.is_item_hovered():
                imgui.set_tooltip(help_text)
            if changed:
                _open[key] = value
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        if imgui.button("Pick item"):
            imgui.debug_start_item_picker()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Click a widget in the GUI; imgui breaks on it and the "
                "Metrics window shows which code drew it."
            )
    imgui.end()


def draw_imgui_debug_windows() -> None:
    """Draw the debugger window and whichever tool windows are on. Call once per frame."""
    if _open["window"]:
        _draw_window()
    if _open["metrics"]:
        _open["metrics"] = bool(imgui.show_metrics_window(True))
    if _open["debug_log"]:
        _open["debug_log"] = bool(imgui.show_debug_log_window(True))
    if _open["id_stack"]:
        _open["id_stack"] = bool(imgui.show_id_stack_tool_window(True))
    if _open["demo"]:
        _open["demo"] = bool(imgui.show_demo_window(True))
    if _open["about"]:
        _open["about"] = bool(imgui.show_about_window(True))
