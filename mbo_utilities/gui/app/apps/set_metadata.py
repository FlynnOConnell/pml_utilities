"""Typing the metadata a reader could not find, or correcting what it found."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._metadata_editor import draw_metadata_editor_content
from mbo_utilities.gui.app._app import App


class SetMetadataApp(App):
    """The suggested fields for the open array, and custom keys, set by hand.

    What is typed lands on the host's ``metadata_edits``, which the Metadata
    viewer, Save As and every pipeline read alongside the array's own, and
    is dropped when other data opens.
    """

    id = "set_metadata"
    title = "Set Metadata"
    menu = "File"
    shortcut = "Shift+M"
    window = True
    order = 20
    window_size = (560, 620)

    def available(self, host) -> bool:
        return host.data is not None

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        draw_metadata_editor_content(
            host.metadata_edits, host.data, host.data.source_path
        )
