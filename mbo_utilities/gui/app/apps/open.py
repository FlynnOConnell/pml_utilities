"""Opening a file into the host."""

from __future__ import annotations

import numpy as np
from imgui_debugger import draw_path_popup

from mbo_utilities.gui.app._app import App


class OpenApp(App):
    """A path prompt that replaces what every other app is looking at.

    Reads the first ``nt`` timepoints of the first z-plane and colour
    channel into memory, the shape the apps here expect, and hands it to
    ``AppHost.set_data``, which remounts whatever is on the slots.
    """

    id = "open"
    title = "Open"
    window = True
    owns_window = True
    order = 5

    def __init__(self):
        super().__init__()
        self.path = ""
        self.nt = 500
        self.note = "the first z-plane and colour channel are read into memory"

    def draw_window(self, host) -> None:
        if not self.open:
            return
        self.open, self.path, confirmed = draw_path_popup(
            "Open", self.open, self.path, "path to a file or folder", "Open", note=self.note
        )
        if not confirmed:
            return
        from mbo_utilities import imread

        array = imread(self.path)
        host.set_data(np.asarray(array[: min(self.nt, array.shape[0]), 0, 0], dtype=np.float32))
        self.open = False
