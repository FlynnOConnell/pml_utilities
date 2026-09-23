"""The figure, its docks and the apps drawn on it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mbo_utilities import log
from mbo_utilities.gui.app._app import DOCKS, App
from mbo_utilities.gui.app._dock import Dock
from mbo_utilities.gui.app._menu import MenuBar
from mbo_utilities.gui.app._window import AppWindow

if TYPE_CHECKING:
    from fastplotlib.layouts import ImguiFigure, Subplot

logger = log.get("gui.app")


class AppHost:
    """The one canvas, and the apps that draw on it.

    One canvas carries one ``ImguiFigure``, which carries one imgui context:
    every ``ImguiRenderer`` makes a context of its own and claims the
    canvas's pointer and key events, so a second figure on the same canvas
    would fight the first. The host is therefore built once and apps are
    swapped inside it, never by rebuilding the figure.

    Subplots cannot be added to or removed from a live figure either
    (``Figure.add_subplot`` is not implemented), so the figure is built with
    the slots it will ever have and apps are granted one rather than making
    their own. Swapping a slot is graphics coming out and going in, which is
    supported, and the day subplots become dynamic only ``mount`` changes.

    An app reaches the host for three things: what data is open, where its
    graphics go, and the shared cursor. That list is the contract; an app
    never puts state of its own on the host.
    """

    def __init__(self, figure: ImguiFigure, data: Any = None):
        self.figure = figure
        self.data = data
        self.apps: dict[str, App] = {}
        self.windows: dict[str, AppWindow] = {}
        self.slots: list[Subplot] = list(figure)
        # subplot index -> the id of the app mounted on it
        self.stage: dict[int, str | None] = {}
        # the one time on screen, shared by every app; gui.playhead.Playhead
        # takes this over when the session lands
        self.index = 0
        self.docks = {edge: Dock(self, edge) for edge in DOCKS}
        self.menu = MenuBar(self)
        figure.add_animations(self._frame)

    def register(self, *apps: App) -> None:
        """Add apps to the host, in the order they should be listed."""
        for app in apps:
            if app.id in self.apps:
                raise ValueError(f"an app with id {app.id!r} is already registered")
            self.apps[app.id] = app
            if app.window:
                window = AppWindow(self, app)
                self.windows[app.id] = window
                self.figure.add_imgui_window(window, location="floating")
            logger.debug(f"registered app {app.id}")

    def ordered(self) -> list[App]:
        return sorted(self.apps.values(), key=lambda app: (app.order, app.title))

    def mount(self, app_id: str | None, slot: int = 0) -> None:
        """Put ``app_id`` on subplot ``slot``, taking off whatever was there."""
        if self.stage.get(slot) == app_id:
            return
        for other, mounted in list(self.stage.items()):
            if mounted == app_id and other != slot:
                self.mount(None, other)
        current = self.stage.get(slot)
        if current is not None:
            app = self.apps[current]
            app.unmount(self)
            app.mounted = None
        subplot = self.slots[slot]
        subplot.clear()
        self.stage[slot] = app_id
        if app_id is None:
            logger.debug(f"slot {slot} emptied")
            return
        app = self.apps[app_id]
        app.mounted = subplot
        app.mount(self, subplot)
        subplot.auto_scale()
        logger.debug(f"mounted {app_id} on slot {slot}")

    def set_data(self, data: Any) -> None:
        """Open something else: every mounted app is taken off and put back.

        Per-dataset state lives in what ``mount`` builds, so unmounting is
        what clears it. Nothing keeps a list of what to reset.
        """
        self.data = data
        self.index = 0
        for slot, app_id in list(self.stage.items()):
            if app_id is None:
                continue
            self.mount(None, slot)
            self.mount(app_id, slot)

    def status(self) -> str:
        return f"t {self.index}"

    def close(self) -> None:
        """Take every app off the stage and release what they hold.

        The figure belongs to whoever made it and is left open.
        """
        for slot in list(self.stage):
            self.mount(None, slot)
        for app in self.apps.values():
            app.close()

    def _frame(self, figure) -> None:
        """Run the apps' frame hooks, then let the docks claim their edges.

        A pre-render animation, so this happens before pygfx draws and long
        before any imgui window is drawn: a dock that takes or gives back an
        edge here has the layout right on the same frame.
        """
        for app in list(self.apps.values()):
            if app.available(self):
                app.frame(self)
        for dock in self.docks.values():
            dock.sync()
