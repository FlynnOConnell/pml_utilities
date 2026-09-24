"""The style store under ~/.mbo/imgui, and the app's style editor over it."""

import pytest
from imgui_bundle import imgui
from imgui_debugger import StyleEditor
from mbo_utilities.gui.app.apps.debug import debug_apps
from mbo_utilities.gui.widgets import style_editor


@pytest.fixture(autouse=True)
def mbo_home(tmp_path, monkeypatch):
    """Point ~/.mbo at a temp dir and drop the module-level singletons."""
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))
    monkeypatch.setattr(style_editor, "_store", None)
    yield tmp_path


def editor():
    """The style editor panel the app registers, over the mbo store."""
    return next(
        app.panel for app in debug_apps(None) if isinstance(app.panel, StyleEditor)
    )


def test_store_lives_under_the_mbo_imgui_dir(mbo_home):
    root = style_editor.style_store().root
    assert root.name == "imgui"
    assert root.parent.name == "mbo"


def test_editor_is_backed_by_the_store():
    assert editor().config.store is style_editor.style_store()


def test_apply_saved_style_is_a_noop_with_nothing_saved():
    assert style_editor.apply_saved_style() == 0


def test_saved_style_is_applied_next_launch(monkeypatch):
    ctx = imgui.create_context()
    try:
        imgui.get_style().frame_rounding = 9.0
        editor().save()
        imgui.get_style().frame_rounding = 0.0
        assert style_editor.apply_saved_style() > 0
        assert imgui.get_style().frame_rounding == 9.0
    finally:
        imgui.destroy_context(ctx)


def test_presets_land_in_the_store():
    ctx = imgui.create_context()
    try:
        editor().save_preset("night")
    finally:
        imgui.destroy_context(ctx)
    store = style_editor.style_store()
    assert store.presets() == ["night"]
    assert (store.root / "styles" / "night.json").is_file()
