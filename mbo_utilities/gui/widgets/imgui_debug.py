"""Dear ImGui's own debug tools, as a tab.

The tab and its windows only exist while debug logging is on (the
``MBO_DEBUG`` flag: ``mbo --debug``, the file dialog's or the Options
popup's "Debug logging" checkbox), which is what
``WidgetEntry(debug_only=True)`` gates.

The tool windows are floating windows of imgui's own, so they are drawn
from ``PreviewDataWidget.draw`` every frame rather than from the tab body,
which only runs while the tab is selected.

The style editor is not here: it is File > Style Editor
(``widgets/style_editor.py``), always available and saved under ``~/.mbo``.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.widget_toggles import widget_enabled

__all__ = ["ImguiDebugWidget", "draw_imgui_debug_windows"]

TOGGLE_KEY = "imgui_debug"

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

_open: dict[str, bool] = {key: False for key, _, _ in TOOLS}


class ImguiDebugWidget(Widget):
    """The ImGui tab: switches for Dear ImGui's own debug windows."""

    name = "ImGui Debug"
    tab_label = "ImGui"
    placement = "tab"
    toggle_key = TOGGLE_KEY
    priority = 200

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def draw(self) -> None:
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


def draw_imgui_debug_windows(parent: Any) -> None:
    """Draw whichever imgui tool windows are switched on. Call once per frame."""
    if not widget_enabled(TOGGLE_KEY):
        return

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
