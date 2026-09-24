"""One app: a visualization that owns its state and knows how to draw itself."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from imgui_bundle import imgui

from mbo_utilities import log

if TYPE_CHECKING:
    from fastplotlib.layouts import Subplot

    from mbo_utilities.gui.app._host import AppHost

# edges an app can ask for a panel on; the menu bar has the top, the viewer's sliders the bottom
DOCKS = ("left", "right")

logger = log.get("gui.app")

# apps whose body has already been reported as raising
_reported: set[str] = set()


class App:
    """One visualization: its state, its controls and its canvas.

    An app draws through ``draw_options`` and ``draw_canvas`` and nothing
    else. Neither may open an imgui window, because the host decides where
    the pair is drawn: a tab in an edge dock, a floating window with a close
    box, or both at once. ``mount`` puts pygfx graphics into a subplot the
    host grants it, so the same app can own part of the scene as well.

    Every attribute an app needs lives on the instance. The host holds the
    ``open`` flag and the shared position, and nothing else about the app.
    """

    id: ClassVar[str] = ""
    title: ClassVar[str] = ""

    # which edge wants a panel tab for this app, None for no panel
    dock: ClassVar[str | None] = None
    # whether the host gives this app a floating window of its own
    window: ClassVar[bool] = False
    # with window: the app opens that window itself, in draw_window. The
    # placement for code that already calls imgui.begin, imgui's own tool
    # windows among it; options and canvas are not used.
    owns_window: ClassVar[bool] = False
    # whether this app can be mounted onto a subplot
    scene: ClassVar[bool] = False

    # the top menu that lists this app: File, View, Docs or Debug
    menu: ClassVar[str] = "View"
    # the chord that shows or hides this app, written as the menu shows it
    shortcut: ClassVar[str] = ""
    # (chord, what it does) for each key on_keys handles, for the keybinds sheet
    keybinds: ClassVar[tuple[tuple[str, str], ...]] = ()
    # menu and tab order, lower first
    order: ClassVar[int] = 100
    # thickness in pixels this app wants from its dock
    size: ClassVar[int] = 260
    window_size: ClassVar[tuple[int, int]] = (640, 480)
    start_open: ClassVar[bool] = False

    def __init__(self):
        self.open = self.start_open
        self.mounted: Subplot | None = None

    def available(self, host: AppHost) -> bool:
        """Whether this app can do anything with what the host has open."""
        return True

    def frame(self, host: AppHost) -> None:
        """Work to do every frame, before anything is drawn, showing or not."""

    def draw_options(self, host: AppHost) -> None:
        """Controls above the canvas."""

    def draw_canvas(self, host: AppHost, size: imgui.ImVec2) -> None:
        """The body, which fits itself into ``size`` and never asks for more."""

    def draw_window(self, host: AppHost) -> None:
        """For ``owns_window``: draw at the frame root, opening the window.

        Called every frame whatever ``open`` says, because an app that owns
        its window owns the shortcut that opens it too.
        """

    def progress(self, host: AppHost) -> list[dict]:
        """The running work the console lists for this app: ``key``, ``text``,
        ``progress`` from 0 to 1 and ``done``.
        """
        return []

    def draw_menu_bar(self, host: AppHost) -> None:
        """Items at the right of the menu bar, drawn every frame, showing or not."""

    def on_keys(self, host: AppHost) -> None:
        """Handle this app's keys; runs inside every imgui frame, showing or not."""

    def data_changed(self, host: AppHost) -> None:
        """The host opened other data: drop whatever was built on the old."""

    def mount(self, host: AppHost, subplot: Subplot) -> None:
        """Add this app's graphics to ``subplot``."""

    def unmount(self, host: AppHost) -> None:
        """Drop references to the graphics; the host clears the subplot after."""

    def close(self) -> None:
        """Release gpu textures, threads and files."""


def draw_guarded(app: App, host: AppHost) -> None:
    """Draw an app's body, showing what it raised instead of losing the frame.

    The draw boundary, and the only place an app's exception is caught. It
    saves the window, not the imgui stack: a body that raises between
    ``begin_child`` and ``end_child`` still leaves that unbalanced.
    """
    try:
        app.draw_options(host)
        avail = imgui.get_content_region_avail()
        app.draw_canvas(host, imgui.ImVec2(max(avail.x, 1.0), max(avail.y, 1.0)))
    except Exception as error:
        if app.id not in _reported:
            _reported.add(app.id)
            logger.exception(f"{app.id} raised while drawing")
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.4, 0.4, 1.0), f"{type(error).__name__}: {error}"
        )
