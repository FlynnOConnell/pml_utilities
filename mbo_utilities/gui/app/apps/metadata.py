"""What the open array says about itself."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._metadata import draw_metadata_inspector
from mbo_utilities.gui.app._app import App


class MetadataApp(App):
    """The metadata inspector over whatever the host has open."""

    id = "metadata"
    title = "Metadata"
    shortcut = "m"
    window = True
    order = 70
    window_size = (720, 560)

    def available(self, host) -> bool:
        return bool(getattr(host.data, "metadata", None))

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        draw_metadata_inspector(dict(host.data.metadata), host.data)
