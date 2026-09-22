"""The mbo log, in a window."""

from __future__ import annotations

import logging

from imgui_bundle import imgui

from mbo_utilities import log
from mbo_utilities.gui.app._app import App
from mbo_utilities.gui.widgets.gui_logger import GuiLogger, GuiLogHandler


class LogApp(App):
    """The Debug panel the preview window has, on its own.

    Shows what an app owning a process-wide resource looks like: the handler
    goes on at construction and comes off in ``close``.
    """

    id = "log"
    title = "Log"
    window = True
    order = 90
    window_size = (820, 420)

    def __init__(self):
        super().__init__()
        self.panel = GuiLogger()
        self.handler = GuiLogHandler(self.panel)
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.handler.setLevel(logging.DEBUG)
        log.attach(self.handler)

    def close(self) -> None:
        for name in log.get_package_loggers():
            logging.getLogger(name).removeHandler(self.handler)

    def draw_canvas(self, host, size: imgui.ImVec2) -> None:
        self.panel.draw()
