"""Apps: one canvas, one figure, and visualizations that can be swapped on it.

An :class:`App` owns its own state and draws through ``draw_options`` and
``draw_canvas``. The :class:`AppHost` decides where that pair goes -- a tab in
an edge dock, a floating window, or both -- and grants an app a subplot when it
wants to put graphics in the scene. Nothing is placed by the app, so the same
app can be moved anywhere on the canvas without touching its code.

``mbo app`` opens the host with three sets. The synthetic ones cover the
surfaces: a movie and its traces as pygfx graphics on two subplots, and an
image viewer that puts its own texture in a window. The ported ones are the
preview window's panels drawn here instead -- Open, projections, summary
images, the tile grid, the metadata inspector, suite2p diagnostics and the
log -- and they show themselves only when the open data gives them something
to draw. The third set is ``imgui_debugger``: the variable inspector pointed
at the host, the style editor, and imgui's own metrics, debug log, id stack
and demo windows, which draw their own windows through ``owns_window``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mbo_utilities.gui.app._app import DOCKS, App
from mbo_utilities.gui.app._dock import Dock
from mbo_utilities.gui.app._host import AppHost
from mbo_utilities.gui.app._menu import MenuBar
from mbo_utilities.gui.app._texture import Texture
from mbo_utilities.gui.app._window import AppWindow

__all__ = ["DOCKS", "App", "AppHost", "AppWindow", "Dock", "MenuBar", "Texture", "build_host", "run_app"]


def build_host(data: Any = None, *, apps=None, size: tuple[int, int] = (1400, 900)) -> AppHost:
    """The host: two subplots, every app registered, the movie and traces mounted."""
    import fastplotlib as fpl

    if data is None:
        from mbo_utilities.gui.app.demo import movie_data

        data = movie_data()
    figure = fpl.Figure(shape=(1, 2), names=[["scene", "traces"]], size=size)
    host = AppHost(figure, data=data)
    if apps is None:
        from mbo_utilities.gui.app.apps import debug_apps, ported_apps
        from mbo_utilities.gui.app.demo import demo_apps

        apps = demo_apps() + ported_apps() + debug_apps(host)
    host.register(*apps)
    if "movie" in host.apps:
        host.mount("movie", 0)
    if "traces" in host.apps:
        host.mount("traces", 1)
    return host


def run_app(path=None, *, nt: int = 500, frames: int = 0, size: tuple[int, int] = (1400, 900)) -> AppHost:
    """Open the app host, on ``path`` when given, else on a synthetic movie.

    ``path`` is read as ``(T, Y, X)`` from the first z-plane and colour
    channel, at most ``nt`` timepoints, and held in memory: the demo apps
    reduce over the whole movie when they mount.

    ``frames`` draws that many frames and returns instead of running the
    event loop, for a smoke test on an offscreen canvas.
    """
    import fastplotlib as fpl

    data = None
    if path is not None:
        from mbo_utilities import imread

        array = imread(path)
        data = np.asarray(array[: min(nt, array.shape[0]), 0, 0], dtype=np.float32)
    host = build_host(data, size=size)
    host.figure.show()
    if frames > 0:
        for _ in range(frames):
            host.figure.canvas.force_draw()
        return host
    fpl.loop.run()
    return host
