"""The IsoView editors: crop, segment and dead pixels, each in a window of its own."""

from __future__ import annotations

from mbo_utilities.arrays.isoview import IsoviewArray
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets import (
    isoview_crop,
    isoview_deadpixel,
    isoview_segment,
)
from mbo_utilities.gui.widgets.pipelines.isoview import (
    maybe_refresh_raw_projections,
    maybe_spawn_raw_projections,
)
from mbo_utilities.lazy_array import base_array

# (module that draws the editor, the prefix its window state lives under, title)
TOOLS = {
    "crop": (isoview_crop, "_iso_crop", "IsoView Crop"),
    "segment": (isoview_segment, "_iso_seg", "IsoView Segment"),
    "deadpixel": (isoview_deadpixel, "_iso_dp", "IsoView Dead Pixels"),
}


class IsoviewToolApp(App):
    """One IsoView editor over the open IsoView tree.

    The editor draws its own window and keeps its state on the host's
    ``context`` (where the IsoView pipeline's buttons open it), so ``open``
    here is kept in step with the editor's own flag: the menu opens or
    closes the editor, and the editor's close box or the pipeline's button
    shows up in the menu.
    """

    window = True
    owns_window = True
    order = 70

    def __init__(self, tool: str):
        self.module, self.prefix, self.title = TOOLS[tool]
        self.id = f"isoview_{tool}"
        super().__init__()
        # open as it was after the last frame, to tell a menu click from the editor
        self._was_open = False

    def available(self, host) -> bool:
        return host.context is not None and isinstance(
            base_array(host.data), IsoviewArray
        )

    def draw_window(self, host) -> None:
        context = host.context
        if self.open != self._was_open:
            if self.open:
                setattr(context, f"_show{self.prefix}_window", True)
            else:
                setattr(context, f"{self.prefix}_window_open", False)
        self.module.draw_window(context)
        self.open = self._was_open = getattr(context, f"{self.prefix}_window_open")


class IsoviewProjectionsJob(App):
    """Raw XY projections of an open raw IsoView tree, written in the background.

    The segment and dead-pixel editors preview from them, and the
    Projections panel appears once the job has written them. Listed in no
    menu: it has nothing to show, only work to start and watch.
    """

    id = "isoview_projections"
    title = "IsoView projections"
    menu = ""

    def available(self, host) -> bool:
        return host.context is not None

    def frame(self, host) -> None:
        maybe_spawn_raw_projections(host.context)
        maybe_refresh_raw_projections(host.context)
