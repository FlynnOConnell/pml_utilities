"""File > Style Editor: one editor per process, saved under ~/.mbo/imgui."""

import pytest
from imgui_bundle import imgui
from mbo_utilities.gui.widgets import style_editor


@pytest.fixture(autouse=True)
def mbo_home(tmp_path, monkeypatch):
    """Point ~/.mbo at a temp dir and drop the module-level singletons."""
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))
    monkeypatch.setattr(style_editor, "_store", None)
    monkeypatch.setattr(style_editor, "_editor", None)
    monkeypatch.setattr(style_editor, "_last_visible", None)
    yield tmp_path


def draw_frames(n=2):
    """Draw the editor window for ``n`` frames on a bare imgui context."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1200, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        for _ in range(n):
            imgui.new_frame()
            style_editor.draw_style_editor_window(None)
            imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)


def test_store_lives_under_the_mbo_imgui_dir(mbo_home):
    root = style_editor.style_store().root
    assert root.name == "imgui"
    assert root.parent.name == "mbo"


def test_editor_is_a_singleton_and_starts_closed():
    editor = style_editor.get_style_editor()
    assert style_editor.get_style_editor() is editor
    assert editor.visible is False
    assert editor.window_id.endswith("##imgui_debugger_mbo_style")


def test_editor_is_backed_by_the_store():
    assert style_editor.get_style_editor().config.store is style_editor.style_store()


def test_apply_saved_style_is_a_noop_with_nothing_saved():
    assert style_editor.apply_saved_style() == 0


def test_saved_style_is_applied_next_launch(monkeypatch):
    ctx = imgui.create_context()
    try:
        imgui.get_style().frame_rounding = 9.0
        style_editor.get_style_editor().save()
        imgui.get_style().frame_rounding = 0.0
        assert style_editor.apply_saved_style() > 0
        assert imgui.get_style().frame_rounding == 9.0
    finally:
        imgui.destroy_context(ctx)


def test_presets_land_in_the_store():
    ctx = imgui.create_context()
    try:
        style_editor.get_style_editor().save_preset("night")
    finally:
        imgui.destroy_context(ctx)
    store = style_editor.style_store()
    assert store.presets() == ["night"]
    assert (store.root / "styles" / "night.json").is_file()


def test_panel_state_is_written_when_it_opens():
    editor = style_editor.get_style_editor()
    editor.show()
    draw_frames()
    saved = style_editor.style_store().panel_state(editor.window_id)
    assert saved["visible"] is True


def test_closed_editor_draws_nothing():
    draw_frames()
    assert style_editor.get_style_editor().visible is False


def test_the_imgui_debug_tab_has_no_style_entry():
    from mbo_utilities.gui.widgets import imgui_debug

    assert "style" not in {key for key, _, _ in imgui_debug.TOOLS}
    assert "style" not in imgui_debug._open
