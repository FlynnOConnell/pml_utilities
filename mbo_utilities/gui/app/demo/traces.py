"""Traces read out of the open movie, on a subplot of their own."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mbo_utilities.gui.app._app import App

# top-left corner of the four boxes each trace averages, as fractions of the frame
BOXES = ((0.15, 0.15), (0.15, 0.6), (0.6, 0.15), (0.6, 0.6))
COLORS = ("cyan", "magenta", "yellow", "green")


class TracesApp(App):
    """One line per box, and a cursor at the frame the movie is showing."""

    id = "traces"
    title = "Traces"
    dock = "right"
    scene = True
    order = 20
    size = 280
    start_open = True

    def __init__(self):
        super().__init__()
        self.lines = []
        self.traces = []
        self.cursor = None
        self.offset = 1.0
        self._shown = -1

    def available(self, host) -> bool:
        return host.data is not None

    def mount(self, host, subplot) -> None:
        movie = np.asarray(host.data[:, host.channel, host.zplane])
        nt, ny, nx = movie.shape
        t = np.arange(nt, dtype=np.float32)
        for i, (fy, fx) in enumerate(BOXES):
            y0, x0 = int(fy * ny), int(fx * nx)
            box = movie[:, y0 : y0 + ny // 4, x0 : x0 + nx // 4]
            trace = box.reshape(nt, -1).mean(axis=1).astype(np.float32)
            self.traces.append(trace - trace.mean())
            line = np.column_stack([t, self.traces[i] + i * self.offset])
            self.lines.append(subplot.add_line(line, colors=COLORS[i], name=f"box{i}"))
        span = (len(BOXES) + 1) * self.offset
        self.cursor = subplot.add_line(
            np.array([[host.frame, -span], [host.frame, span]], dtype=np.float32),
            colors="w",
            thickness=2,
            name="cursor",
        )
        self._shown = host.frame

    def unmount(self, host) -> None:
        self.lines = []
        self.traces = []
        self.cursor = None
        self._shown = -1

    def frame(self, host) -> None:
        if self.cursor is not None and host.frame != self._shown:
            self.cursor.data[:, 0] = host.frame
            self._shown = host.frame

    def draw_options(self, host) -> None:
        if self.cursor is None:
            imgui.text_disabled("not on a slot")
            return
        imgui.text(f"{len(self.lines)} boxes")
        imgui.set_next_item_width(-imgui.FLT_MIN)
        changed, value = imgui.slider_float("##offset", self.offset, 0.0, 5.0)
        if changed:
            self.offset = value
            for i, line in enumerate(self.lines):
                line.data[:, 1] = self.traces[i] + i * self.offset
