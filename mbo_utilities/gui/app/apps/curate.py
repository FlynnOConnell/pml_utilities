"""Opening the curation window on the open line scan, in its own process."""

from __future__ import annotations

from mbo_utilities.gui._availability import HAS_VNOISER
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.curation_viewer import curation_target, launch_curation_window


class CurateApp(App):
    """The File menu's Curate: the curation window (``mbo curate``) on the open
    ``.mesc`` or ``PF`` folder.

    Nothing is drawn here: choosing it starts the window in its own process,
    which outlives this one, so ``open`` never stays on.
    """

    id = "curate"
    title = "Curate"
    menu = "File"
    order = 30

    def __init__(self):
        # what the curation window can take of the open data, and which data that was
        self.target = None
        self._for = None
        super().__init__()

    @property
    def open(self) -> bool:
        return False

    @open.setter
    def open(self, value: bool) -> None:
        if value and self.target is not None:
            launch_curation_window(self.target)

    def available(self, host) -> bool:
        if self._for is not host.data:
            self._for = host.data
            self.target = curation_target(getattr(host.data, "source_path", None))
        return HAS_VNOISER and self.target is not None
