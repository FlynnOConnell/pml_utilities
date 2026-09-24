"""The imgui style editor, on its own menu item, saved under ``~/.mbo/imgui``.

The editor is ``imgui_debugger.StyleEditor`` backed by a ``ConfigStore`` at
``get_mbo_dirs()["imgui"]``: it writes ``state.json`` a second after the last
slider moves, keeps named presets in ``styles/``, and ``apply_saved_style()``
puts the style back at launch.
"""

from __future__ import annotations

from imgui_debugger import ConfigStore, StyleEditor

from mbo_utilities import log
from mbo_utilities.preferences import get_mbo_dirs

__all__ = [
    "style_store",
    "apply_saved_style",
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


def apply_saved_style() -> int:
    """Apply the saved style over the built-in one; returns fields applied.

    Called once at startup, right after ``style_imgui_opaque()`` so a saved
    style wins over the shipped theme.
    """
    applied = style_store().apply_style()
    if applied:
        logger.info(f"applied saved imgui style ({applied} fields)")
    return applied
