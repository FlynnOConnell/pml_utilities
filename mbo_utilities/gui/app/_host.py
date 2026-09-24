"""The figure, its docks and the apps drawn on it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import imgui_bundle
from imgui_bundle import imgui, implot

from mbo_utilities import __version__, log
from mbo_utilities.gui._imgui_helpers import style_imgui_opaque
from mbo_utilities.gui._metadata_editor import MetadataEdits
from mbo_utilities.gui._stats import ZStats, compute_zstats, hydrate_zstats
from mbo_utilities.gui._top_strip import TopStrip
from mbo_utilities.gui.app._app import DOCKS, App
from mbo_utilities.gui.app._dock import Dock
from mbo_utilities.gui.app._keys import pressed
from mbo_utilities.gui.app._menu import MenuBar
from mbo_utilities.gui.app._window import AppWindow
from mbo_utilities.gui.playhead import Playhead, TimeAxis
from mbo_utilities.gui.widgets.style_editor import apply_saved_style
from mbo_utilities.preferences import get_mbo_dirs

if TYPE_CHECKING:
    from fastplotlib.layouts import ImguiFigure, Subplot
    from imgui_debugger import ConfigStore

logger = log.get("gui.app")

# the store key holding which apps were showing
SHOWING = "app_host"


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
    ``slots`` leaves out the subplots something else already draws on, such
    as the viewer's images.

    An app reaches the host for four things: what data is open, where its
    graphics go (the slots, the ``viewer`` showing the data when there is
    one, and the top ``strip``, which runs per-frame hooks and shows full
    width panels as tabs under the menus), the shared position (the playhead, the channel and the z-plane
    on screen), and what is computed or entered once about the open data
    for every app to read (``zstats``, the summary stats of the viewer's
    arrays, and ``metadata_edits``, what the user typed over its metadata).
    That list is the contract; an app never puts state of its own on the
    host.

    With a ``store``, which apps are showing is saved as it changes and put
    back when an app registers, so the app opens the way it was left. An
    app that owns its window (a prompt, an imgui tool) keeps its own.
    """

    def __init__(
        self,
        figure: ImguiFigure,
        data: Any = None,
        slots: list[Subplot] | None = None,
        store: ConfigStore | None = None,
        viewer=None,
    ):
        # the fps overlay's renderer leaves its own context current; use the figure's
        imgui.set_current_context(figure.imgui_renderer.imgui_context)
        if implot.get_current_context() is None:
            implot.create_context()
        style_imgui_opaque()
        apply_saved_style()
        imgui.get_io().set_ini_filename(str(Path(get_mbo_dirs()["imgui"]) / "app.ini"))
        # the emphasis the pipeline settings and the stats plots draw in
        roboto = Path(imgui_bundle.__file__).parent / "assets" / "fonts" / "Roboto"
        self.bold_font = imgui.get_io().fonts.add_font_from_file_ttf(
            str(roboto / "Roboto-Bold.ttf"), 14, imgui.ImFontConfig()
        )

        self.figure = figure
        self.data = data
        self.viewer = viewer
        self.zstats = None if viewer is None else ZStats(viewer, self.bold_font)
        self.metadata_edits = MetadataEdits()
        self.store = store
        self._showing = {} if store is None else dict(store.panel_state(SHOWING))
        self.apps: dict[str, App] = {}
        self.windows: dict[str, AppWindow] = {}
        self.slots: list[Subplot] = list(figure) if slots is None else list(slots)
        # subplot index -> the id of the app mounted on it
        self.stage: dict[int, str | None] = {}
        # the position on screen, shared by every app; indices are 0-based
        self.playhead = Playhead()
        self.channel = 0
        self.zplane = 0
        self.docks = {edge: Dock(self, edge) for edge in DOCKS}
        self.menu = MenuBar(self)
        # the top edge: the menu row, then panels apps hang off it as tabs
        self.strip = TopStrip(figure, draw_menu=self.menu.draw)
        figure.add_animations(self._frame)
        if self.zstats is not None:
            self.compute_stats()

    def register(self, *apps: App) -> None:
        """Add apps to the host, in the order they should be listed."""
        for app in apps:
            if app.id in self.apps:
                raise ValueError(f"an app with id {app.id!r} is already registered")
            self.apps[app.id] = app
            if not app.owns_window:
                app.open = self._showing.get(app.id, app.open)
            if app.window:
                window = AppWindow(self, app)
                self.windows[app.id] = window
                self.figure.add_imgui_window(window, location="floating")
            logger.debug(f"registered app {app.id}")

    def ordered(self) -> list[App]:
        return sorted(self.apps.values(), key=lambda app: (app.order, app.title))

    def compute_stats(self) -> None:
        """The open data's summary stats: cached ones where the store has them,
        the rest computed on a background thread.
        """
        self.zstats.reset()
        hydrated = hydrate_zstats(self.zstats)
        pending = [i for i, done in enumerate(hydrated) if not done]
        for i in pending:
            self.zstats.running[i] = True
        if pending:
            compute_zstats(self.zstats, only=pending)

    def keys(self) -> None:
        """The frame's shortcuts: p folds the right dock, an app's chord shows or
        hides it, then every app handles its own keys.
        """
        if pressed("p"):
            self.docks["right"].collapsed = not self.docks["right"].collapsed
        for app in self.ordered():
            if not app.available(self):
                continue
            if app.shortcut and pressed(app.shortcut):
                app.open = not app.open
            app.on_keys(self)

    def time_axis(self) -> TimeAxis:
        """The open data's T axis on the playhead's clock."""
        return TimeAxis.sampled(None if self.data is None else self.data.fs)

    @property
    def frame(self) -> int:
        """The playhead as an index into the open data's T axis."""
        if self.data is None:
            return 0
        frame = round(self.time_axis().units(self.playhead.time))
        return min(max(frame, 0), self.data.shape[0] - 1)

    def seek_frame(self, frame: int, source=None) -> None:
        """Move the playhead to T index ``frame``."""
        self.playhead.seek(self.time_axis().seconds(frame), source=source)

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
        """Open something else: every app hears it, then mounted apps are remounted.

        Per-dataset state lives in what ``mount`` builds and in what
        ``data_changed`` drops, so nothing keeps a list of what to reset.
        """
        self.data = data
        self.channel = 0
        self.zplane = 0
        self.metadata_edits = MetadataEdits()
        self.playhead.seek(0.0, source=self)
        # the old stats must not reach an app rebuilding for the new data
        if self.zstats is not None:
            self.zstats.reset()
        for app in list(self.apps.values()):
            app.data_changed(self)
        if self.zstats is not None:
            self.compute_stats()
        for slot, app_id in list(self.stage.items()):
            if app_id is None:
                continue
            self.mount(None, slot)
            self.mount(app_id, slot)
        self.figure.canvas.set_title(self.title())

    def title(self) -> str:
        """The window title: the program and what it has open."""
        source = None if self.data is None else self.data.source_path
        name = Path(source).name if source else "untitled"
        return f"Miller Brain Studio v{__version__} - {name}"

    def status(self) -> str:
        if self.data is None:
            return "nothing open"
        if self.data.fs:
            return f"t {self.frame}  {self.playhead.time:.3f} s"
        return f"t {self.frame}"

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
        if self.store is not None:
            showing = {
                app.id: app.open for app in self.apps.values() if not app.owns_window
            }
            if showing != self._showing:
                self._showing = showing
                self.store.set_panel_state(SHOWING, showing)
