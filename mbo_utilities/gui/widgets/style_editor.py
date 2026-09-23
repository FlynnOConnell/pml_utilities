"""The imgui style editor, on its own menu item, saved under ``~/.mbo/imgui``.

The editor is ``imgui_debugger.StyleEditor`` backed by a ``ConfigStore`` at
``get_mbo_dirs()["imgui"]``: it writes ``state.json`` a second after the last
slider moves, keeps named presets in ``styles/``, and ``apply_saved_style()``
puts the style back at launch.
"""

from __future__ import annotations

from typing import Any

from imgui_debugger import ConfigStore, StyleEditor, StyleEditorConfig

from mbo_utilities import log
from mbo_utilities.preferences import get_mbo_dirs

__all__ = [
    "style_store",
    "get_style_editor",
    "apply_saved_style",
    "draw_style_menu_item",
    "draw_style_editor_window",
]

logger = log.get("gui.style_editor")

_store: ConfigStore | None = None
_editor: StyleEditor | None = None
_last_visible: bool | None = None


def style_store() -> ConfigStore:
    """The store under ``~/.mbo/imgui`` holding the style, presets and panel state."""
    global _store
    if _store is None:
        _store = ConfigStore(get_mbo_dirs()["imgui"])
    return _store


def get_style_editor() -> StyleEditor:
    """The one style editor for this process, restored to how it was left."""
    global _editor
    if _editor is None:
        _editor = StyleEditor(
            StyleEditorConfig(
                title="Style Editor",
                window_id="mbo_style",
                visible=False,
                window_size=(560, 720),
                store=style_store(),
            )
        )
        _editor.set_state(style_store().panel_state(_editor.window_id))
    return _editor


def apply_saved_style() -> int:
    """Apply the saved style over the built-in one; returns fields applied.

    Called once at startup, right after ``style_imgui_opaque()`` so a saved
    style wins over the shipped theme.
    """
    applied = style_store().apply_style()
    if applied:
        logger.info(f"applied saved imgui style ({applied} fields)")
    return applied


def draw_style_menu_item() -> None:
    """Draw the File-menu entry that opens the editor."""
    get_style_editor().menu_item("Style Editor")


def draw_style_editor_window(parent: Any) -> None:
    """Draw the editor window, saving its panel state when it opens or closes."""
    global _last_visible
    editor = get_style_editor()
    editor.render_window()
    if editor.visible != _last_visible:
        _last_visible = editor.visible
        style_store().set_panel_state(editor.window_id, editor.get_state())
