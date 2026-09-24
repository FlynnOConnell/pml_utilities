"""Widgets > ImGui Debugger: a floating window of imgui's own debug tools and
imgui_debugger's inspector; Widgets > Style Editor beside it. No tab, no submenus."""

from imgui_bundle import imgui
from mbo_utilities.gui import widgets
from mbo_utilities.gui.widgets import imgui_debug
from mbo_utilities.gui.widgets.widget_toggles import WIDGET_REGISTRY, draw_widgets_menu


class Host:
    """What the menu and the inspector are handed: a bag of attributes."""

    def __init__(self):
        self.fpath = "session1.tif"
        self._show_biohpc = False
        self._show_cloud = False


def draw_frames(parent, n=2, menu=False):
    """Draw the debug windows (and the Widgets menu, opened) for ``n`` frames."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1200, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for _ in range(n):
            imgui.new_frame()
            if menu:
                imgui.begin("host", None, imgui.WindowFlags_.menu_bar)
                imgui.begin_menu_bar()
                imgui.open_popup("Widgets")
                draw_widgets_menu(parent)
                imgui.end_menu_bar()
                imgui.end()
            imgui_debug.draw_imgui_debug_windows(parent)
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


def test_no_imgui_tab_and_no_debug_gate():
    widgets._discover_widgets()
    assert "imgui_debug" not in {e.key for e in WIDGET_REGISTRY}
    assert "ImGui" not in {cls.tab_label for cls in widgets._WIDGET_CLASSES}
    # every widget's toggle is a menu entry, not a section of one
    keys = {e.key for e in WIDGET_REGISTRY}
    assert all(
        cls.toggle_key is None or cls.toggle_key in keys
        for cls in widgets._WIDGET_CLASSES
    )


def test_every_menu_entry_is_one_toggle():
    assert all(not hasattr(e, "subwidgets") for e in WIDGET_REGISTRY)


def test_the_window_and_every_tool_draw(monkeypatch):
    monkeypatch.delenv("MBO_DEBUG", raising=False)
    host = Host()
    for key in imgui_debug._open:
        monkeypatch.setitem(imgui_debug._open, key, True)
    monkeypatch.setattr(imgui_debug, "_inspector", None)
    imgui_debug.get_inspector(host).visible = True
    draw_frames(host)
    # nothing closed itself, and the inspector looks at the host
    assert all(imgui_debug._open.values())
    assert imgui_debug.get_inspector(host).config.target is host


def test_opening_needs_no_debug_flag(monkeypatch):
    monkeypatch.delenv("MBO_DEBUG", raising=False)
    monkeypatch.setitem(imgui_debug._open, "window", False)
    imgui_debug.open_imgui_debugger()
    assert imgui_debug._open["window"]


def test_the_widgets_menu_draws(monkeypatch):
    monkeypatch.setitem(imgui_debug._open, "window", False)
    draw_frames(Host(), menu=True)
