"""The preferences: render and compute GPU, debug logging, memory log, line-scan traces."""

from __future__ import annotations

from imgui_bundle import imgui

from mbo_utilities.gui._options_popup import draw_options, sync_options
from mbo_utilities.gui.app._app import App


class OptionsApp(App):
    """The options the File menu opens, saved to the preferences as they change.

    Holds the half-typed widget state ``draw_options`` keeps between frames;
    the preferences file is the truth and is read when the app is built.
    """

    id = "options"
    title = "Options"
    menu = "File"
    window = True
    order = 90
    window_size = (460, 380)

    def __init__(self):
        super().__init__()
        sync_options(self)

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        draw_options(self)
