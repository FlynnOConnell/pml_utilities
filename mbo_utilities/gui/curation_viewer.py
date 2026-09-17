"""Standalone vnoiser event curation window.

The curation notebook's dashboard (``vnoiser/notebooks/curation.ipynb``) on
its own, nothing else: no image panels, no Z-stack, no ROI table. The
dashboard is :class:`mbo_utilities.gui.event_curation.EventCurationWidget`
with the recordings table, mode and source controls in a column beside it.
Two hosts draw the same layout:

- :class:`CurationApp`: a hello_imgui desktop window (``mbo scan.mesc``);
- :class:`CurationVis`: a fastplotlib figure whose top edge window is the
  whole canvas. In a notebook the figure is a jupyter_rfb canvas, so the
  dashboard streams to the browser from wherever the kernel runs (a cluster
  node, as the plotly notebook did); offscreen it is what the tests draw.
  ``open_curation_viewer`` picks it in a notebook and displays the canvas.
  On rendercanvas's ``http`` canvas it is what
  :class:`mbo_utilities.gui.curation_server.CurationServer` streams to
  browsers over the network (``mbo curate <path> --serve``).

What opens:

- a ``PF`` folder the voltage pipeline wrote (the folder, its traces file,
  or the experiment folder holding it): every scan / domain trace in it is
  listed and loaded; the source line scan named in its ``pipeline.json``
  (or laid out beside it) brings the scan's RTMC traces;
- a raw line-scan ``.mesc`` with a ``PF`` folder beside it: that folder;
- a raw AOD ``.mesc`` without one: every ROI of every line-scan, chessboard
  or ribbon unit as a raw recording; clicking one runs vnoiser's wavelet denoiser on its
  trace (minutes the first time, cached under ``.curation/cache`` beside the
  file) and curates the result - the full pipeline vnoiser offers.

Usage:
    mbo curate <expt>/PF                a PF folder, or the folder holding it
    mbo curate scan.mesc                a line scan, with or without a PF folder
    python -m mbo_utilities.gui.curation_viewer [path] [--channel 0]
        [--frames N --screenshot out.png]
    vis = open_curation_viewer(path)    in a notebook cell: the canvas shows
                                        in the cell; vis.widget is the dashboard

The image viewer (``mbo scan.mesc``, ``mbo <expt>/PF``) has no curation of
its own: its Curate button (the Voltage pipeline, or File > Curate) runs
:func:`launch_curation_window`, this window in a second process, since one
imgui loop cannot host a second hello_imgui runner. The line-scan + Z-stack
viewer that draws the lines on the stack is ``mbo linescan scan.mesc --view``
(``gui/linescan_viewer.py``).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx, immapp

from mbo_utilities.gui import _setup
from mbo_utilities.gui._availability import HAS_VNOISER
from mbo_utilities.gui._edge_window import EdgeWindow
from mbo_utilities.gui._theme import em
from mbo_utilities.install import VNOISER_HINT
from mbo_utilities.preferences import get_mbo_dirs

__all__ = [
    "CurationApp",
    "CurationVis",
    "PanelHost",
    "curation_target",
    "launch_curation_window",
    "open_curation_viewer",
    "main",
]

# the controls column beside the dashboard
CONTROLS_EM = 26.0
WINDOW_SIZE = (1500, 900)
# pixels of the figure left to pygfx under the dashboard window: nothing is
# drawn there, but the empty subplot's fixed insets (title row, padding,
# ~37 px) come off it and the renderer needs a positive viewport after that
FIGURE_MARGIN = 48
# the smallest canvas the dashboard renders on: under it the window cannot
# leave pygfx a positive viewport (a browser tab shrunk to nothing, or one
# that has not reported its size yet)
MIN_CANVAS = (320, 200)


class PanelHost:
    """Stands in for a figure's ``TopStrip`` when there is no figure.

    The curation widget registers its panel and a frame hook here exactly as
    it would on the strip; the app draws the panel itself, so the tab bar,
    resizing and right-bar sync the strip does are not needed.
    """

    def __init__(self) -> None:
        self.panels: list = []
        self.hooks: list[Callable[[], None]] = []
        self.active: str | None = None
        self.right_tab = ""

    def register(self, panel) -> None:
        self.panels = [p for p in self.panels if p.key != panel.key] + [panel]
        if self.active is None:
            self.active = panel.key

    def unregister(self, key: str) -> None:
        self.panels = [p for p in self.panels if p.key != key]
        if self.active == key:
            self.active = self.panels[0].key if self.panels else None

    def has(self, key: str) -> bool:
        return any(p.key == key for p in self.panels)

    def add_hook(self, fn: Callable[[], None]) -> None:
        if fn not in self.hooks:
            self.hooks.append(fn)

    def remove_hook(self, fn: Callable[[], None]) -> None:
        if fn in self.hooks:
            self.hooks.remove(fn)

    def focus(self, key: str) -> None:
        if self.has(key):
            self.active = key

    def report_right_tab(self, name: str) -> None:
        self.right_tab = name

    def take_right_focus(self, name: str) -> bool:
        return False

    def close(self) -> None:
        self.panels = []
        self.hooks = []
        self.active = None

    def panel(self, key: str):
        return next((p for p in self.panels if p.key == key), None)


def draw_dashboard(widget, host: PanelHost) -> None:
    """One frame of the layout both hosts share: the widget's frame hook,
    the dashboard panel on the left, the controls column on the right."""
    for hook in list(host.hooks):
        hook()
    panel = host.panel("curation")
    avail = imgui.get_content_region_avail()
    controls_w = em(CONTROLS_EM)
    gap = em(0.6)
    with imgui_ctx.begin_child(
        "##curation_dashboard", imgui.ImVec2(max(avail.x - controls_w - gap, em(20)), 0),
    ):
        if panel is not None:
            panel.draw()
    imgui.same_line(0, gap)
    with imgui_ctx.begin_child("##curation_controls", imgui.ImVec2(0, 0)):
        widget.draw_tab()


class _Dashboard:
    """What both hosts share: the widget on a :class:`PanelHost` and the
    ways to point it at data."""

    def __init__(self, path=None, *, channel: int = 0, logger: logging.Logger | None = None,
                 figure=None):
        if not HAS_VNOISER:
            raise SystemExit(f"vnoiser is not installed: {VNOISER_HINT}")
        from mbo_utilities.gui.event_curation import EventCurationWidget

        self.logger = logger or logging.getLogger("curation_viewer")
        self.channel = int(channel)
        self.host = PanelHost()
        parent = SimpleNamespace(
            image_widget=SimpleNamespace(figure=figure),
            logger=self.logger,
            fpath=None if path is None else str(path),
        )
        # data_path "" defers opening to `open`, which knows about raw .mesc
        self.widget = EventCurationWidget(parent, strip=self.host, data_path="" if path is not None else None)
        self.source = None if path is None else Path(path)
        if path is not None:
            self.open(path)

    @property
    def title(self) -> str:
        where = self.source.name if self.source is not None else "vnoiser"
        return f"Event curation - {where}"

    # ------------------------------------------------------------------
    # what to curate
    # ------------------------------------------------------------------

    def open(self, path) -> None:
        """Point the dashboard at ``path``."""
        from mbo_utilities.vnoiser import pf_dir_for_mesc

        path = Path(path).expanduser()
        self.source = path
        if path.is_file() and path.suffix.lower() == ".mesc" and pf_dir_for_mesc(path) is None:
            self.open_raw_mesc(path)
        else:
            # a PF folder (or the folder holding it), and a .mesc with one beside it
            self.widget.scan(str(path))

    def open_raw_mesc(self, mesc_path) -> int:
        """Every ROI of every AOD ROI unit as a raw recording the
        denoiser runs on when clicked. Returns how many."""
        return self.widget.scan_raw_mesc(mesc_path, self.channel)


class CurationApp(_Dashboard):
    """The curation dashboard in its own hello_imgui desktop window.

    Parameters
    ----------
    path : optional
        What to open (see the module docstring); None reopens the last
        data path the widget remembers.
    channel : int
        Channel averaged into a raw line scan's traces.
    """

    def __init__(self, path=None, *, channel: int = 0, logger: logging.Logger | None = None):
        _setup.setup_imgui()
        self.frames = 0
        self.max_frames: int | None = None
        super().__init__(path, channel=channel, logger=logger)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def draw(self) -> None:
        """One frame of the window."""
        self.frames += 1
        if self.max_frames is not None and self.frames >= self.max_frames:
            hello_imgui.get_runner_params().app_shall_exit = True
        draw_dashboard(self.widget, self.host)

    def runner_params(self) -> hello_imgui.RunnerParams:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = self.title
        params.app_window_params.window_geometry.size = WINDOW_SIZE
        params.app_window_params.window_geometry.size_auto = False
        params.app_window_params.resizable = True
        params.ini_filename = _setup.get_default_ini_path("curation_viewer")
        params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.darcula_darker
        params.callbacks.show_gui = self.draw
        return params

    def run(self, frames: int | None = None, screenshot=None) -> None:
        """Run the window until it is closed (or ``frames`` frames, for a
        smoke test; ``screenshot`` then saves the last frame)."""
        self.max_frames = frames
        addons = immapp.AddOnsParams()
        addons.with_implot = True
        addons.with_markdown = False
        try:
            immapp.run(runner_params=self.runner_params(), add_ons_params=addons)
            if screenshot is not None:
                import imageio.v3 as iio

                image = hello_imgui.final_app_window_screenshot()
                iio.imwrite(str(screenshot), np.asarray(image))
                print(f"screenshot -> {screenshot}", flush=True)
        finally:
            self.widget.close()


class _DashboardWindow(EdgeWindow):
    """The figure's top edge window, grown to the whole canvas."""

    def __init__(self, figure, vis: "CurationVis"):
        self.vis = vis
        super().__init__(figure, size=self._want(figure), location="top", title=None)

    @staticmethod
    def _want(figure) -> int:
        try:
            height = float(figure.canvas.get_logical_size()[1])
        except Exception:
            height = 0.0
        return max(int(height) - FIGURE_MARGIN, 100)

    def update(self) -> None:
        want = self._want(self.vis.figure)
        if self.size != want:
            self.size = want
        draw_dashboard(self.vis.widget, self.vis.host)


class CurationVis(_Dashboard):
    """The curation dashboard on a fastplotlib figure: the notebook route.

    ``show()`` returns the canvas; in a notebook that is the jupyter_rfb
    widget the cell displays, streamed from the kernel's machine. The same
    ``show`` / ``close`` contract as ``DataVis``.

    Parameters
    ----------
    path : optional
        What to open (see the module docstring).
    channel : int
        Channel averaged into a raw line scan's traces.
    size : (int, int), optional
        Canvas size in pixels; the notebook default otherwise.
    canvas : optional
        Passed to the figure (``"offscreen"`` for tests); wherever this
        runs decides otherwise.
    """

    def __init__(self, path=None, *, channel: int = 0, size=None, canvas=None,
                 logger: logging.Logger | None = None):
        import fastplotlib as fpl

        from mbo_utilities.gui.run_gui import _figure_kwargs_for_here

        kwargs = _figure_kwargs_for_here(size)
        if canvas is not None:
            # an explicit canvas takes none of the desktop branch's Qt
            # kwargs (present_method="screen" is not a bitmap method)
            kwargs["canvas"] = canvas
            kwargs.pop("canvas_kwargs", None)
        self.figure = fpl.Figure(**kwargs)
        # the one subplot is empty and sits behind the dashboard window: no
        # axes (their update needs a graphic to intersect)
        subplot = self.figure[0, 0]
        subplot.axes.visible = False
        # its "(0, 0)" title would show in the strip left under the dashboard
        subplot.title.visible = False
        self._output = None
        self._shown = False
        self._closed = False
        super().__init__(path, channel=channel, logger=logger, figure=self.figure)
        self.window = _DashboardWindow(self.figure, self)

    def show(self):
        """Show the figure; returns the canvas widget in a notebook."""
        if not self._shown:
            self._output = self.figure.show()
            # the figure's own draw function, guarded (see _draw)
            self.figure.canvas.request_draw(self._draw)
            self._shown = True
        return self._output

    def _draw(self) -> None:
        """The canvas's draw function: the figure's render with two guards a
        streamed canvas needs. A browser reports its size only after it
        connects (the canvas is 1 x 1 until then) and can shrink by hundreds
        of pixels in one step, so the dashboard window is sized to the
        canvas *before* pygfx renders (the window's own update runs after,
        inside the imgui pass) and a canvas under MIN_CANVAS draws nothing.
        The next frame is requested whatever happened: the imgui figure
        only asks for one after a successful render, so one bad frame would
        otherwise end the stream."""
        canvas = self.figure.canvas
        try:
            w, h = canvas.get_logical_size()
            if w < MIN_CANVAS[0] or h < MIN_CANVAS[1]:
                return
            want = self.window._want(self.figure)
            if self.window.size != want:
                self.window.size = want  # resets the figure's layout
            self.figure._render()
        finally:
            canvas.request_draw()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.widget.close()
        try:
            self.figure.close()
        except AttributeError:
            # an offscreen figure has no output widget to close
            self.figure.canvas.close()


def open_curation_viewer(path=None, *, channel: int = 0, run: bool = True, frames=None,
                         screenshot=None, figure: bool | None = None, canvas=None, size=None):
    """Open the curation dashboard on ``path``.

    In a notebook (or with ``figure=True``) it is a :class:`CurationVis` on
    a fastplotlib figure, shown and displayed in the cell; returned so the
    cell can reach ``vis.widget``. Otherwise a :class:`CurationApp` desktop
    window, run until closed (``run=False`` only builds it).
    """
    from mbo_utilities.gui._notebook import display_widget, in_notebook

    if figure is None:
        figure = in_notebook()
    if figure:
        vis = CurationVis(path, channel=channel, canvas=canvas, size=size)
        if run:
            display_widget(vis.show())
        return vis
    app = CurationApp(path, channel=channel)
    if run:
        app.run(frames=frames, screenshot=screenshot)
    return app


def curation_target(path) -> Path | None:
    """What of a viewer's open path the curation window can take: a ``.mesc``
    itself, or the ``PF`` folder a path names (the folder, its traces
    pickle, or the experiment holding it); None for anything else."""
    from mbo_utilities.arrays.pf import pf_dir_of

    if isinstance(path, (list, tuple)):
        path = path[0] if path else None
    if not path:
        return None
    path = Path(str(path))
    if path.is_file() and path.suffix.lower() == ".mesc":
        return path
    return pf_dir_of(path)


def launch_curation_window(path, channel: int = 0) -> int:
    """Open the curation window on ``path`` in its own process, what
    ``mbo curate PATH`` does, and return its pid. The window outlives the
    viewer that opened it; its output goes to ``~/.mbo/logs``."""
    python = sys.executable
    if sys.platform == "win32" and python.endswith("python.exe"):
        # no console window beside the curation window
        pythonw = python[:-10] + "pythonw.exe"
        if Path(pythonw).exists():
            python = pythonw
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = get_mbo_dirs()["logs"] / f"{stamp}_curate_{Path(str(path)).stem}.log"
    cmd = [python, "-m", "mbo_utilities.gui.curation_viewer", str(path), "--channel", str(int(channel))]
    with log_file.open("a", encoding="utf-8") as out:
        if sys.platform == "win32":
            proc = subprocess.Popen(
                cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdin=subprocess.DEVNULL, stdout=out, stderr=out,
            )
        else:
            proc = subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL, stdout=out, stderr=out)
    return proc.pid


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", type=Path,
                    help="a PF folder (or the experiment folder holding it), or a line-scan .mesc "
                         "(default: the last data path)")
    ap.add_argument("--channel", type=int, default=0, help="channel averaged for raw line scans")
    ap.add_argument("--frames", type=int, default=None, help="exit after this many frames")
    ap.add_argument("--screenshot", type=Path, default=None, help="with --frames: save the last frame")
    args = ap.parse_args(argv)
    open_curation_viewer(args.path, channel=args.channel, frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
