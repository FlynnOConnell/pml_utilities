"""A movie on a subplot, with the transport that drives the shared cursor."""

from __future__ import annotations

from imgui_bundle import imgui
from imgui_debugger import MoviePlayer

from mbo_utilities.gui.app._app import App

CMAPS = ("viridis", "gray", "magma", "plasma")


class MovieApp(App):
    """The open movie as an image graphic, scrubbed by a shared MoviePlayer.

    The player owns the transport and the host owns the cursor, so the two
    are reconciled once a frame: whichever of them moved wins, and every
    other app follows the host. The player advances from its own controls,
    so playback runs while they are on screen and stops when they are not.
    """

    id = "movie"
    title = "Movie"
    dock = "right"
    scene = True
    order = 10
    size = 300
    start_open = True

    def __init__(self):
        super().__init__()
        self.graphic = None
        self.player = MoviePlayer()
        self.cmap = "viridis"
        self._shown = -1

    def available(self, host) -> bool:
        return host.data is not None

    def mount(self, host, subplot) -> None:
        self.player.set_movie(host.data)
        self.player.jump_to(host.index)
        self.graphic = subplot.add_image(
            host.data[host.index], name="movie", cmap=self.cmap
        )
        self._shown = host.index

    def unmount(self, host) -> None:
        self.graphic = None
        self._shown = -1

    def frame(self, host) -> None:
        if self.player.t != self._shown:
            host.index = self.player.t
        elif host.index != self.player.t:
            self.player.jump_to(host.index)
        if self.graphic is not None and host.index != self._shown:
            self.graphic.data = host.data[host.index]
        self._shown = host.index

    def draw_options(self, host) -> None:
        self.player.draw(
            slider_width=imgui.get_content_region_avail().x - 12 * imgui.get_font_size()
        )
        imgui.set_next_item_width(-imgui.FLT_MIN)
        changed, index = imgui.combo("##cmap", CMAPS.index(self.cmap), list(CMAPS))
        if changed:
            self.cmap = CMAPS[index]
            if self.graphic is not None:
                self.graphic.cmap = self.cmap
        if self.graphic is None:
            imgui.text_disabled("not on a slot")
            return
        frame = host.data[host.index]
        imgui.text(f"{frame.shape[1]} x {frame.shape[0]}")
        imgui.text(f"{float(frame.min()):.3g} .. {float(frame.max()):.3g}")
