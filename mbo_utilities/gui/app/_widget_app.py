"""Hosting a ``gui.widgets`` Widget as an app, until it becomes one."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from mbo_utilities import log
from mbo_utilities.gui.app._app import App

if TYPE_CHECKING:
    from imgui_bundle import imgui

    from mbo_utilities.gui.app._host import AppHost

logger = log.get("gui.app")


class WidgetApp(App):
    """An app that draws an existing ``Widget``.

    A widget already owns its state and answers whether it applies to the
    data, which is most of what an app is. What it does not have is a place
    to be drawn that is not the preview window's control column, so the
    host gives it one and asks its ``is_supported`` once per array, handing
    it the host's ``context`` as its parent. The widget is built on the
    array it was asked about and dropped with it.
    """

    widget_class: ClassVar[Any] = None

    def __init__(self):
        super().__init__()
        self.widget = None
        # is_supported for the open array; None until asked
        self._supported: bool | None = None

    def available(self, host: AppHost) -> bool:
        if host.context is None:
            return False
        if self._supported is None:
            self._supported = self.widget_class.is_supported(host.context)
        return self._supported

    def data_changed(self, host: AppHost) -> None:
        self.close()
        self._supported = None

    def draw_canvas(self, host: AppHost, size: imgui.ImVec2) -> None:
        if self.widget is None:
            self.widget = self.widget_class(host.context)
        self.widget.draw()

    def close(self) -> None:
        if self.widget is not None:
            self.widget.cleanup()
            self.widget = None
