"""Apps: one canvas, one figure, and visualizations that can be swapped on it.

An :class:`App` owns its own state and draws through ``draw_options`` and
``draw_canvas``. The :class:`AppHost` decides where that pair goes -- a tab in
an edge dock, a floating window, or both -- and grants an app a subplot when it
wants to put graphics in the scene. Nothing is placed by the app, so the same
app can be moved anywhere on the canvas without touching its code.

``mbo app`` opens the host on the viewer: fastplotlib's n-d viewer over the
open array, with a slider for every T, C and Z it has, and the playhead as
the one time every app follows. The ported apps are the preview window's
panels drawn here instead -- Open, projections, summary images, the tile
grid, the metadata inspector, suite2p diagnostics and the log -- and they
show themselves only when the open data gives them something to draw. The
``imgui_debugger`` set is the variable inspector pointed at the host, the
style editor, and imgui's own metrics, debug log, id stack and demo
windows, which draw their own windows through ``owns_window``. The demo
apps in ``app.demo`` exercise the subplot slots on a plain figure.
"""

from __future__ import annotations

from typing import Any

from imgui_debugger import ConfigStore

from mbo_utilities import imread
from mbo_utilities.arrays.features import get_slider_dims
from mbo_utilities.gui._ndviewer import MboNDViewer
from mbo_utilities.gui.app._app import DOCKS, App
from mbo_utilities.gui.app._dock import Dock
from mbo_utilities.gui.app._host import AppHost
from mbo_utilities.gui.app._menu import MenuBar
from mbo_utilities.gui.app._texture import Texture
from mbo_utilities.gui.app._window import AppWindow
from mbo_utilities.gui.app.apps import ViewerApp, debug_apps, ported_apps
from mbo_utilities.gui.app.demo import movie_data
from mbo_utilities.gui.run_gui import (
    _after_show,
    _figure_kwargs_for_here,
    _squeeze_for_viewer,
)
from mbo_utilities.gui.widgets.style_editor import style_store

__all__ = [
    "DOCKS",
    "App",
    "AppHost",
    "AppWindow",
    "Dock",
    "MenuBar",
    "Texture",
    "build_host",
    "run_app",
]


def build_host(
    data: Any = None,
    *,
    apps=None,
    size: tuple[int, int] = (1400, 900),
    store: ConfigStore | None = None,
) -> AppHost:
    """The host on the viewer's figure, every app registered.

    ``data`` is anything ``imread`` opens, a synthetic movie when None. It
    stays lazy: the viewer reads the frames it shows. ``store`` remembers
    which apps were showing.
    """
    array = imread(movie_data() if data is None else data)
    viewer = MboNDViewer(
        _squeeze_for_viewer(array),
        slider_dim_names=getattr(array, "slider_dim_labels", None)
        or get_slider_dims(array),
        cmap="gnuplot2",
        figure_kwargs=_figure_kwargs_for_here(size=size),
    )
    host = AppHost(viewer.figure, data=array, slots=[], store=store, viewer=viewer)
    host.register(ViewerApp(array))
    host.register(*(ported_apps() + debug_apps(host) if apps is None else apps))
    return host


def run_app(path=None, *, frames: int = 0, size: tuple[int, int] = (1400, 900)) -> None:
    """Open the app host on ``path``, else on a synthetic movie, until its window closes.

    ``frames`` draws that many frames and returns instead of running the
    event loop, for a smoke test on an offscreen canvas. Either way the
    host and its viewer are closed before returning: a reader's open files
    and background reads otherwise keep the interpreter from exiting.
    """
    import fastplotlib as fpl

    host = build_host(path, size=size, store=style_store())
    host.figure.show()
    _after_show(host.viewer)
    host.figure.canvas.set_title(host.title())
    if frames > 0:
        for _ in range(frames):
            host.figure.canvas.force_draw()
    else:
        fpl.loop.run()
    host.close()
    host.viewer.close()
