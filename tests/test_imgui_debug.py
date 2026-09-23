"""The ImGui tab: Dear ImGui's own debug windows, only while MBO_DEBUG is on."""

from imgui_bundle import imgui

from mbo_utilities.gui.widgets import imgui_debug
from mbo_utilities.gui.widgets.widget_toggles import (
    get_entry,
    set_widget_enabled,
    widget_enabled,
)


def draw_frames(widget, n=2):
    """Draw ``widget`` and the tool windows for ``n`` frames on a bare context."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1200, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for _ in range(n):
            imgui.new_frame()
            imgui.begin("host")
            widget.draw()
            imgui.end()
            imgui_debug.draw_imgui_debug_windows(None)
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


def test_entry_is_debug_only():
    assert get_entry("imgui_debug").debug_only is True
    assert get_entry("imgui_debug").default is False


def test_tab_stays_off_without_the_flag(monkeypatch):
    monkeypatch.delenv("MBO_DEBUG", raising=False)
    set_widget_enabled("imgui_debug", True, persist=False)
    assert widget_enabled("imgui_debug") is False


def test_tab_follows_the_flag(monkeypatch):
    set_widget_enabled("imgui_debug", True, persist=False)
    monkeypatch.setenv("MBO_DEBUG", "1")
    assert widget_enabled("imgui_debug") is True
    monkeypatch.setenv("MBO_DEBUG", "0")
    assert widget_enabled("imgui_debug") is False


def test_tab_and_tool_windows_draw(monkeypatch):
    monkeypatch.setenv("MBO_DEBUG", "1")
    set_widget_enabled("imgui_debug", True, persist=False)
    for key, _, _ in imgui_debug.TOOLS:
        monkeypatch.setitem(imgui_debug._open, key, True)
    draw_frames(imgui_debug.ImguiDebugWidget(None))
    # every window is still open: nothing closed itself
    assert all(imgui_debug._open[key] for key, _, _ in imgui_debug.TOOLS)


def test_tool_windows_skipped_when_the_tab_is_off(monkeypatch):
    monkeypatch.delenv("MBO_DEBUG", raising=False)
    monkeypatch.setitem(imgui_debug._open, "metrics", True)
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(400, 400)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        imgui.new_frame()
        imgui_debug.draw_imgui_debug_windows(None)
        imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)
