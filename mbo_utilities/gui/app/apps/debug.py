"""Dear ImGui's own debug tools, and the variable inspector, as apps."""

from __future__ import annotations

from imgui_debugger import DebugTools, DemoPanel, StyleEditor

from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.app._panel_app import PanelApp
from mbo_utilities.gui.widgets.style_editor import style_store


def debug_apps(target: object) -> list[App]:
    """The inspector, the style editor and imgui's own tools, one app each.

    ``target`` is what the variable inspector opens on: hand it the host and
    every app, slot and index on screen can be read live. Each panel keeps
    its size and visibility under ``~/.mbo/imgui``, the same store the
    preview window's style editor uses.
    """
    store = style_store()
    tools = DebugTools.default(target, store=store)
    tools.add(DemoPanel())
    tools.load_state()
    # the style editor is the user's, not a debug tool, so it sits under File
    return [
        PanelApp(
            panel,
            order=200 + i,
            store=store,
            menu="File" if isinstance(panel, StyleEditor) else "Debug",
        )
        for i, panel in enumerate(tools)
    ]
