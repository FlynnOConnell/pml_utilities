"""Opening a file into the host."""

from __future__ import annotations

from imgui_debugger import draw_path_popup

from mbo_utilities import imread, log
from mbo_utilities.gui.app._app import App

logger = log.get("gui.app")

NOTE = "any file or folder imread opens; nothing is read until it is shown"


class OpenApp(App):
    """A path prompt that replaces what every other app is looking at.

    Opens the path lazily with ``imread`` and hands the array to
    ``AppHost.set_data``. A path that fails to open keeps the prompt up
    with the reason under it.
    """

    id = "open"
    title = "Open"
    window = True
    owns_window = True
    order = 5

    def __init__(self):
        super().__init__()
        self.path = ""
        self.note = NOTE

    def draw_window(self, host) -> None:
        if not self.open:
            return
        self.open, self.path, confirmed = draw_path_popup(
            "Open",
            self.open,
            self.path,
            "path to a file or folder",
            "Open",
            note=self.note,
        )
        if not confirmed:
            return
        try:
            array = imread(self.path)
        except Exception as error:
            logger.exception(f"could not open {self.path}")
            self.note = f"{type(error).__name__}: {error}"
            self.open = True
            return
        host.set_data(array)
        self.note = NOTE
        self.open = False
