"""``mbo`` opens the Studio; heavy imports stay inside the functions so ``--check-install`` is instant."""

import contextlib
import copy
import functools
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import click

from mbo_utilities import log
from mbo_utilities.gui._notebook import display_widget, in_notebook

if sys.platform == "win32":
    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "mbo.utilities.gui.1.0"
    )


def _set_qt_icon():
    """Set the icon on the QApplication and every top-level window, native handles included, once the window is shown."""
    try:
        from PyQt6.QtGui import QIcon
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        return
    from mbo_utilities.gui._setup import get_package_assets_path
    from mbo_utilities.preferences import get_mbo_dirs

    app = QApplication.instance()
    if app is None:
        return
    icon_path = get_package_assets_path() / "app_settings" / "icon.png"
    if not icon_path.exists():
        icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"
    if not icon_path.exists():
        return
    icon = QIcon(str(icon_path))
    app.setWindowIcon(icon)
    for window in app.topLevelWidgets():
        window.setWindowIcon(icon)
        handle = window.windowHandle()
        if handle:
            handle.setIcon(icon)
        app.processEvents()


class SplashScreen:
    """A tkinter splash for desktop-shortcut launches; tkinter ships with every Windows python."""

    def __init__(self):
        self.root = None
        self._closed = False

    def show(self):
        try:
            import tkinter as tk

            self.root = tk.Tk()
        except Exception:
            # no tkinter, or tkinter.TclError without a display
            return
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        width, height = 300, 120
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.configure(bg="#1e1e2e")
        title = tk.Label(
            self.root,
            text="MBO Utilities",
            font=("Segoe UI", 16, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e",
        )
        title.pack(pady=(20, 5))
        self.loading_label = tk.Label(
            self.root,
            text="Loading...",
            font=("Segoe UI", 10),
            fg="#a6adc8",
            bg="#1e1e2e",
        )
        self.loading_label.pack(pady=(0, 10))
        self.progress_frame = tk.Frame(self.root, bg="#1e1e2e")
        self.progress_frame.pack(pady=5)
        self.dots = []
        for _i in range(5):
            dot = tk.Label(
                self.progress_frame,
                text="●",
                font=("Segoe UI", 12),
                fg="#45475a",
                bg="#1e1e2e",
            )
            dot.pack(side=tk.LEFT, padx=3)
            self.dots.append(dot)
        self.current_dot = 0
        self._animate()
        self.root.update()

    def _animate(self):
        if self.root is None or self._closed:
            return
        for i, dot in enumerate(self.dots):
            dot.configure(fg="#89b4fa" if i == self.current_dot else "#45475a")
        self.current_dot = (self.current_dot + 1) % len(self.dots)
        self.root.after(200, self._animate)

    def close(self):
        self._closed = True
        if self.root is not None:
            with contextlib.suppress(Exception):
                self.root.destroy()
            self.root = None


def _check_installation():
    from mbo_utilities.install import check_installation, print_status_cli

    status = check_installation()
    print_status_cli(status)
    return status.all_ok


def _select_file(runner_params: Any | None = None) -> tuple[Any, Any, Any, bool, str]:
    """Run the launcher and return its path, split-rois flag, widget flag, metadata-only flag and mode."""
    from imgui_bundle import hello_imgui, immapp

    from mbo_utilities.gui._setup import get_default_ini_path
    from mbo_utilities.gui.widgets.file_dialog import FileDialog

    dlg = FileDialog()

    from mbo_utilities import __version__

    if runner_params is None:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = (
            f"The Studio v{__version__} – Data Selection"
        )
        params.app_window_params.window_geometry.size = (340, 720)
        params.app_window_params.window_geometry.size_auto = False
        params.app_window_params.resizable = True
    else:
        params = runner_params

    # without an ini path hello_imgui writes <window_title>.ini into the current directory
    if not params.ini_filename:
        params.ini_filename = get_default_ini_path("file_dialog")
    params.callbacks.show_gui = dlg.render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)
    return (
        dlg.selected_path,
        dlg.split_rois,
        dlg.widget_enabled,
        dlg.metadata_only,
        dlg.gui_modes[dlg.selected_mode_index],
    )


def _show_metadata_viewer(metadata: dict) -> None:
    from imgui_bundle import hello_imgui, immapp

    from mbo_utilities import __version__
    from mbo_utilities.gui._metadata import draw_metadata_inspector
    from mbo_utilities.gui._setup import get_default_ini_path

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = f"The Studio v{__version__} – Metadata"
    params.app_window_params.window_geometry.size = (800, 800)
    params.ini_filename = get_default_ini_path("metadata_viewer")
    params.callbacks.show_gui = lambda: draw_metadata_inspector(metadata)

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)


class _SqueezeSingletonDims:
    """A 5D TCZYX array without its size-1 T, C, Z axes, so the viewer gets one
    slider per ``get_slider_dims`` entry.
    """

    def __init__(self, arr):
        self._arr = arr
        full = tuple(arr.shape)
        if len(full) != 5:
            raise ValueError(f"expected 5D TCZYX array, got shape {full}")
        self._kept = [i for i in range(3) if full[i] > 1] + [3, 4]
        self._dropped = [i for i in range(3) if full[i] <= 1]
        self._shape = tuple(full[i] for i in self._kept)
        orig_dims = getattr(arr, "dims", None)
        if orig_dims is not None and len(orig_dims) == 5:
            self._dims = tuple(orig_dims[i] for i in self._kept)
        else:
            self._dims = None

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def dims(self):
        return self._dims

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))
        full_key = [None] * 5
        for d in self._dropped:
            full_key[d] = 0
        it = iter(key)
        for i in self._kept:
            full_key[i] = next(it)
        return self._arr[tuple(full_key)]

    def __len__(self):
        return self._shape[0]

    def __array__(self, dtype=None, copy=None):
        """The first (Y, X) frame, never the whole array."""
        import numpy as np

        arr = np.asarray(self[(0,) * (self.ndim - 2)])
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def astype(self, dtype, *args, **kwargs):
        import numpy as np

        return np.asarray(self).astype(dtype, *args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._arr, name)


class _ScrubTimingProxy:
    """The array the viewer reads, logging each read's time at DEBUG.

    ``_arr`` is the innermost reader, for isinstance checks; reads go through
    ``_wrapped``.
    """

    def __init__(self, arr):
        import numpy as np

        self._wrapped = arr
        inner = arr
        while hasattr(inner, "_arr"):
            inner = inner._arr
        self._arr = inner
        self._np = np
        self._logger = log.get("gui.scrub")

    @property
    def shape(self):
        return self._wrapped.shape

    @property
    def ndim(self):
        return self._wrapped.ndim

    @property
    def dtype(self):
        return self._wrapped.dtype

    @property
    def dims(self):
        return getattr(self._wrapped, "dims", None)

    def __len__(self):
        return len(self._wrapped)

    def __getitem__(self, key):
        import logging

        np = self._np
        # a bare int reads one frame, not every camera and plane of a timepoint
        if isinstance(key, (int, np.integer)) and self._wrapped.ndim > 3:
            key = (
                (int(key),)
                + (0,) * (self._wrapped.ndim - 3)
                + (slice(None), slice(None))
            )
        if not self._logger.isEnabledFor(logging.DEBUG):
            return self._wrapped[key]
        t0 = time.perf_counter()
        out = self._wrapped[key]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        type_name = type(self._arr).__name__
        if isinstance(key, tuple) and key and isinstance(key[0], (int, np.integer)):
            key_str = f"t={int(key[0])}"
        elif isinstance(key, (int, np.integer)):
            key_str = f"t={int(key)}"
        else:
            key_str = repr(key)
        self._logger.debug(f"{type_name}[{key_str}]: {dt_ms:.1f}ms")
        return out

    def __array__(self, dtype=None, copy=None):
        arr = self._np.asarray(self._wrapped)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def astype(self, dtype, *args, **kwargs):
        return self._np.asarray(self).astype(dtype, *args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._wrapped, name)


def _squeeze_for_viewer(arr):
    """Drop the size-1 T, C, Z axes (fastplotlib wants ndim == sliders + 2), then add the scrub timer.

    Frame averaging swaps the viewer's array later and reproduces exactly this wrapping.
    """
    if not hasattr(arr, "shape") or len(arr.shape) != 5:
        out = arr
    elif any(arr.shape[i] == 1 for i in range(3)):
        out = _SqueezeSingletonDims(arr)
    else:
        out = arr
    return _ScrubTimingProxy(out)


_NOTEBOOK_SIZE = (1400, 900)
_PREVIEW_WIDTH = 340  # the PreviewDataWidget on the figure's right edge


def screen_box() -> tuple[int, int] | None:
    """The screen's available work area less the window frame and title bar,
    ``None`` when there is no Qt screen to ask.
    """
    try:
        from PyQt6.QtGui import QGuiApplication
    except ImportError:
        return None
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return None
    avail = screen.availableGeometry()
    return max(400, avail.width() - 40), max(400, avail.height() - 100)


def fit_figure_size(
    box: tuple[int, int],
    image_hw: tuple[int, int],
    grid: tuple[int, int] = (1, 1),
    top: int = 0,
    bottom: int = 0,
    right: int = 0,
    min_width: float = 0.0,
) -> tuple[int, int]:
    """Canvas size whose render area has the images' aspect.

    The image column fills the height of ``box`` (w, h) and the width follows
    from the aspect, so the subplot shows no black band around the image. The
    edge windows (``top`` strip, ``bottom`` sliders, ``right`` widget) and each
    subplot's histogram dock and frame padding are added around it. The window
    is widened only as far as ``min_width`` needs to keep the top strip on one
    row; that is the one place the image gets a margin.
    """
    from mbo_utilities.gui._fpl_config import (
        HISTOGRAM_WIDTH,
        SUBPLOT_PAD_H,
        SUBPLOT_PAD_W,
    )
    from mbo_utilities.gui._top_strip import MIN_RENDER_AREA

    nrows, ncols = grid
    h, w = max(float(image_hw[0]), 1.0), max(float(image_hw[1]), 1.0)
    col_pad = HISTOGRAM_WIDTH + SUBPLOT_PAD_W
    img_h = (box[1] - top - bottom) / nrows - SUBPLOT_PAD_H
    img_w = img_h * w / h
    width = ncols * (img_w + col_pad) + right
    if width > box[0]:
        img_w = (box[0] - right) / ncols - col_pad
        img_h = img_w * h / w
        width = box[0]
    height = nrows * (img_h + SUBPLOT_PAD_H) + top + bottom
    return int(max(width, min_width)), int(max(height, top + bottom + MIN_RENDER_AREA))


def _figure_kwargs_for_here(
    size: tuple[int, int] | None = None, fit: dict | None = None
) -> dict:
    """The canvas and size for wherever this process is running.

    A notebook gets the jupyter canvas: it belongs in the output cell, not
    in a Qt window the browser cannot see, which is what the pyqt6 branch
    would build whenever PyQt6 happens to be installed (napari pulls it in).
    jupyter_rfb streams the same wgpu frames over the kernel, so this is also
    what makes a remote/JupyterHub session work. A notebook cell has no
    screen to measure, so its room is a fixed box with space for the edge
    windows; a desktop window's room is the screen's available work area,
    so launches on shorter monitors (laptops, 1080p with a taskbar) don't run
    past the bottom of the screen.

    ``size`` is used as given. Otherwise ``fit``, the keyword arguments of
    :func:`fit_figure_size`, sizes the canvas to the images within that room,
    and without either the canvas is the room itself, at most 1000 px a side.
    """
    # every viewer asks here right before it builds its figure, so this is where the fpl config lands
    from mbo_utilities.gui import _fpl_config  # noqa: F401

    if in_notebook():
        if size is None:
            size = fit_figure_size(_NOTEBOOK_SIZE, **fit) if fit else _NOTEBOOK_SIZE
        return {"canvas": "jupyter", "size": tuple(size)}

    if os.environ.get("RENDERCANVAS_FORCE_OFFSCREEN"):
        # tests and headless capture: rendercanvas.auto already resolved to
        # the offscreen backend, and a Qt canvas would fight it
        if size is None:
            size = fit_figure_size((1000, 1000), **fit) if fit else (1000, 1000)
        return {"size": tuple(size)}

    if os.environ.get("RENDERCANVAS_BACKEND", "qt").lower() not in ("qt", "pyqt6"):
        # a chosen backend is left to rendercanvas.auto, which honors the variable
        RenderCanvas = None
    else:
        try:
            from rendercanvas.pyqt6 import RenderCanvas
        except (ImportError, RuntimeError):
            # RuntimeError once another qt backend is selected
            RenderCanvas = None

    if size is None:
        box = screen_box() or (1000, 1000)
        size = (
            fit_figure_size(box, **fit)
            if fit
            else (min(1000, box[0]), min(1000, box[1]))
        )

    if RenderCanvas is not None:
        # present_method="screen" renders the wgpu surface directly. The
        # "bitmap" method blits via QPainter inside paintEvent, which
        # re-enters during Qt's resize loop ("Recursive repaint detected",
        # paintEngine==0) and freezes the window on resize.
        return {
            "canvas": "pyqt6",
            "canvas_kwargs": {"present_method": "screen"},
            "size": tuple(size),
        }
    return {"size": tuple(size)}


_MIN_VIEWPORT = 150  # logical px per side of image a subplot keeps; the window stops shrinking there
_NOTEBOOK_CANVASES = ("JupyterRenderCanvas", "AnywidgetRenderCanvas")
_BROWSER_CANVASES = ("HttpRenderCanvas", "BrowserCanvas")


def _clamp_window_to_layout(figure, event=None) -> None:
    """Keep the window at least as big as its edge windows, docks, per-subplot
    imgui windows and a ``_MIN_VIEWPORT`` image need.

    The figure clamps its render area to 1 px, then each subplot subtracts
    its histogram window, docks and spacing from its share, so a canvas
    narrower than the panels gives pygfx a negative viewport and a
    validation error every frame. Registered on the canvas ``resize``
    event so it follows whatever the edges are sized to at the time. Qt and
    glfw windows get a minimum size; a notebook canvas has none, so one
    dragged smaller is set back to it.
    """
    canvas = figure.canvas
    area_w, area_h = figure.get_pygfx_render_area()[2:]
    need_w = need_h = 0.0
    for subplot in figure:
        frame = subplot.frame
        fw, fh = frame.rect[2:]
        vw, vh = frame.viewport.rect[2:]
        # the overhead a subplot carries at any size, scaled back up by its
        # fraction of the render area
        need_w = max(need_w, (fw - vw + _MIN_VIEWPORT) * area_w / fw)
        need_h = max(need_h, (fh - vh + _MIN_VIEWPORT) * area_h / fh)
    edges = figure.imgui_windows
    for side in ("left", "right"):
        need_w += edges[side].size if edges[side] is not None else 0
    for side in ("top", "bottom"):
        need_h += edges[side].size if edges[side] is not None else 0
    min_w, min_h = int(math.ceil(need_w)), int(math.ceil(need_h))
    if hasattr(canvas, "setMinimumSize"):
        current = canvas.minimumSize()
        if (current.width(), current.height()) != (min_w, min_h):
            canvas.setMinimumSize(min_w, min_h)
    elif type(canvas).__name__ == "GlfwRenderCanvas":
        import glfw

        # glfw window sizes are logical on macOS and physical on Windows and X11
        scale = 1.0 if canvas._screen_size_is_logical else canvas.get_pixel_ratio()
        glfw.set_window_size_limits(
            canvas._window,
            int(math.ceil(min_w * scale)),
            int(math.ceil(min_h * scale)),
            glfw.DONT_CARE,
            glfw.DONT_CARE,
        )
    elif event is not None:
        # a notebook or http canvas is 1x1 until the browser reports its first resize
        w, h = canvas.get_logical_size()
        if w < min_w or h < min_h:
            if type(canvas).__name__ in _NOTEBOOK_CANVASES:
                # the css size trait still holds the pre-drag value, so an
                # unchanged value would never reach the browser; an http
                # browser would report the 0 x 0 back as a resize
                canvas.set_logical_size(0, 0)
            canvas.set_logical_size(max(w, min_w), max(h, min_h))


def _hold_min_size(figure) -> None:
    """Keep a Qt, glfw, notebook or browser canvas from shrinking below its layout."""
    canvas = figure.canvas
    if not (
        hasattr(canvas, "setMinimumSize")
        or type(canvas).__name__
        in ("GlfwRenderCanvas", *_NOTEBOOK_CANVASES, *_BROWSER_CANVASES)
    ):
        return
    canvas.add_event_handler(
        functools.partial(_clamp_window_to_layout, figure), "resize"
    )
    _clamp_window_to_layout(figure)


def _after_show(iw) -> None:
    """Window title, icon and minimum size; only meaningful once a desktop
    canvas exists.
    """
    from mbo_utilities import __version__

    canvas = iw.figure.canvas
    if hasattr(canvas, "set_title"):
        canvas.set_title(f"The Studio v{__version__}")
    _set_qt_icon()
    _hold_min_size(iw.figure)


def _create_image_widget(
    data_array,
    widget: bool = True,
    figure_kwargs_override=None,
    show: bool = True,
):
    """Build the viewer with its side widget.

    ``widget`` is ``"preview"`` (the PreviewDataWidget), ``"manualroi"`` (the
    preview plus the ROI tool switched on) or ``"none"``; True and False are
    the old spellings of the first and last. ``figure_kwargs_override``
    replaces the canvas and size picked for this process (headless capture).
    ``show=False`` leaves showing to the caller, which is how ``DataVis``
    separates construction from ``show()``.
    """
    if isinstance(widget, bool) or widget is None:
        widget = "preview" if widget else "none"
    if widget not in ("preview", "manualroi", "none"):
        raise ValueError(
            f"unknown widget {widget!r}, expected one of: preview, manualroi, none"
        )

    # the roi tool is a Widgets-menu toggle the preview widget builds from; it goes on for this
    # session (never persisted) when asked for or when rois or a pipeline's rois sit beside the data
    manual_roi = signal_quality = False
    if widget != "none":
        from mbo_utilities.gui.manual_roi import labels_path
        from mbo_utilities.gui.roi_runs import run_dir_complete
        from mbo_utilities.gui.widgets.widget_toggles import widget_enabled
        from mbo_utilities.lazy_array import base_array
        from mbo_utilities.preferences import get_roi_folder

        src = data_array.source_path
        results = getattr(base_array(data_array), "results", None)
        root = get_roi_folder() or None
        manual_roi = (
            widget == "manualroi"
            or widget_enabled("manual_roi")
            or (
                results is not None
                and any(u.member_kind == "pixel" for u in results.units.values())
            )
            or (
                src is not None
                and (
                    labels_path(src).exists()
                    or labels_path(src, root=root).exists()
                    or run_dir_complete(labels_path(src).parent)
                )
            )
        )
        signal_quality = widget_enabled("signal_quality")

    from mbo_utilities.arrays.features import get_slider_dims

    custom_labels = getattr(data_array, "slider_dim_labels", None)
    if custom_labels:
        slider_dim_names = tuple(custom_labels)
    else:
        slider_dim_names = get_slider_dims(data_array)

    # isoview fluorescence sits in the first few hundred counts; anything else starts at the frame's range
    is_isoview = getattr(data_array, "kind", None) in {
        "raw",
        "corrected",
        "fused",
        "clusterpt",
    }
    graphic_kwargs = {"vmin": 0, "vmax": 1000} if is_isoview else {}

    if hasattr(data_array, "roi_mode") and hasattr(data_array, "iter_rois"):
        arrays = []
        names = []
        base_name = None
        if getattr(data_array, "filenames", None):
            first_file = Path(data_array.filenames[0])
            base_name = first_file.stem
            if base_name in ("data", "data_raw"):
                base_name = first_file.parent.name
            if len(base_name) > 24:
                base_name = base_name[:21] + "..."
        for r in data_array.iter_rois():
            arr = copy.copy(data_array)
            arr.fix_phase = False
            arr.roi = r
            arrays.append(_squeeze_for_viewer(arr))
            names.append(f"ROI {r}" if r else (base_name or "Full Image"))
    else:
        arrays = [_squeeze_for_viewer(data_array)]
        names = None

    from mbo_utilities.gui._ndviewer import MboNDViewer, sliders_height

    if figure_kwargs_override is not None:
        figure_kwargs = figure_kwargs_override
    else:
        from fastplotlib.utils import calculate_figure_shape

        from mbo_utilities.gui._top_strip import (
            MENU_HEIGHT,
            MENU_MIN_WIDTH,
            strip_height,
        )
        from mbo_utilities.gui.manual_roi import PANEL_HEIGHT
        from mbo_utilities.gui.widgets.preview_data import ZSTATS_PANEL_HEIGHT

        rgb = bool(getattr(arrays[0], "rgb", False))
        shape = tuple(arrays[0].shape)
        top, right, min_width = 0, 0, 0.0
        if widget != "none":
            top, right, min_width = MENU_HEIGHT, _PREVIEW_WIDTH, MENU_MIN_WIDTH
        # the strip is as tall as the tab that opens first: the Traces panel, else the Signal Quality plot
        if manual_roi:
            top = strip_height(PANEL_HEIGHT)
        elif signal_quality:
            top = strip_height(ZSTATS_PANEL_HEIGHT)
        figure_kwargs = _figure_kwargs_for_here(
            fit=dict(
                image_hw=shape[-3:-1] if rgb else shape[-2:],
                grid=calculate_figure_shape(len(arrays)),
                top=top,
                bottom=sliders_height(MboNDViewer._n_slider_dims(arrays[0], rgb)),
                right=right,
                min_width=min_width,
            )
        )

    iw = MboNDViewer(
        data=arrays,
        names=names,
        slider_dim_names=slider_dim_names,
        cmap="gnuplot2",
        histogram_widget=True,
        figure_kwargs=figure_kwargs,
        graphic_kwargs=graphic_kwargs,
    )

    if show:
        iw.show()
        _after_show(iw)

    if widget != "none":
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget
        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        if manual_roi:
            set_widget_enabled("manual_roi", True, persist=False)
        gui = PreviewDataWidget(
            iw=iw,
            fpath=data_array.source_path,
            size=_PREVIEW_WIDTH,
        )
        # the EdgeWindow shim registers itself during __init__; add_gui exists only on mbo-fastplotlib
        add_gui = getattr(iw.figure, "add_gui", None)
        if add_gui is not None:
            add_gui(gui)
    return iw


def _is_jupyter() -> bool:
    """Old name for ``in_notebook``; a plain ``ipython`` shell is not one."""
    return in_notebook()


def _run_gui_impl(
    data_in: str | Path | None = None,
    roi: int | tuple[int, ...] | None = None,
    widget: bool | str = True,
    metadata_only: bool = False,
    select_only: bool = False,
    show_splash: bool = False,
    runner_params: Any | None = None,
    mode: str = "Fastplotlib viewer (default)",
    unit: int | str | None = None,
    dims: str | None = None,
    vis: str = "demixing",
    raw_path: str | Path | None = None,
    motion_correction_path: str | Path | None = None,
    qc: bool = False,
):
    """``run_gui`` after the heavy imports."""
    from mbo_utilities.gui import _gpu_cache
    from mbo_utilities.preferences import get_debug_logging, get_gpu_index

    # enumerate_adapters() inside an imgui frame initializes wgpu, which on windows drops the
    # current WGL context and flickers the window (glfw error 65544), so the cache is primed here
    _gpu_cache.prime()
    gpu_index = get_gpu_index()
    adapters = _gpu_cache.get_adapters()
    if 0 <= gpu_index < len(adapters):
        import fastplotlib as fpl

        fpl.select_adapter(adapters[gpu_index])
    # an explicit MBO_DEBUG (mbo --debug / --no-debug) wins over the preference
    if get_debug_logging() and "MBO_DEBUG" not in os.environ:
        log.set_debug(True)

    splash = None
    if show_splash and not in_notebook():
        splash = SplashScreen()
        splash.show()

    try:
        if splash:
            splash.close()
            splash = None

        if data_in is None:
            if in_notebook():
                # the launcher is a desktop window; for a hub session it would open on the kernel's machine
                raise ValueError(
                    "run_gui() needs a path in a notebook: the file picker is a "
                    "desktop window. Pass the file or folder, e.g. "
                    'run_gui("/data/session.tif"), or build DataVis(path).'
                )
            data_in, roi_from_dialog, widget, metadata_only, mode = _select_file(
                runner_params=runner_params
            )
            if not data_in:
                return None
            if roi is None:
                roi = roi_from_dialog

        if select_only:
            return data_in

        # the file dialog hands back a list even for one file
        if isinstance(data_in, (list, tuple)) and len(data_in) == 1:
            data_in = data_in[0]
        from mbo_utilities.arrays.masknmf_run import is_masknmf_run, run_demixing
        from mbo_utilities.masknmf.reg_denoise import is_reg_denoise

        if mode == "Spike-triggered average (run with results)":
            from mbo_utilities.gui.spike_average_viewer import (
                open_spike_average_viewer,
            )

            return open_spike_average_viewer(data_in)
        qc = qc or mode == "Registration QC (MaskNMF run)"
        if (
            qc
            and isinstance(data_in, (str, Path))
            and not (
                is_reg_denoise(data_in)
                or is_masknmf_run(data_in)
                or is_masknmf_run(Path(data_in).parent)
            )
        ):
            raise click.ClickException(
                f"{Path(data_in).name} is not a MaskNMF run folder, its results.hdf5 or a "
                "reg-denoise result; Registration QC (--qc) opens only those"
            )
        if (
            not metadata_only
            and isinstance(data_in, (str, Path))
            and (qc or is_reg_denoise(data_in))
        ):
            from mbo_utilities.gui.reg_denoise_viewer import RegDenoiseViewer

            viewer = RegDenoiseViewer(data_in, raw_path=raw_path)
            output = viewer.show()
            if in_notebook():
                display_widget(output)
                return viewer
            import fastplotlib as fpl

            fpl.loop.run()
            return None
        # a masknmf demixing result, or a run folder that demixed, opens in masknmf's own viewers
        from mbo_utilities.arrays.demixing import has_demixing_results

        if isinstance(data_in, (str, Path)):
            data_in = run_demixing(data_in) or data_in
            from mbo_utilities.masknmf.asap7 import is_asap7_run
            from mbo_utilities.masknmf.params import ASAP7_RESULTS_FILE

            if is_asap7_run(data_in):
                data_in = Path(data_in) / ASAP7_RESULTS_FILE

        if (
            not metadata_only
            and isinstance(data_in, (str, Path))
            and has_demixing_results(data_in)
        ):
            import importlib.util

            from mbo_utilities.install import _MASKNMF_HINT

            if (
                importlib.util.find_spec("masknmf") is None
                or importlib.util.find_spec("torch") is None
            ):
                raise click.ClickException(
                    f"{Path(data_in).name} is a masknmf demixing result and opens in masknmf's "
                    f"viewers, but masknmf (with torch) is not installed here: {_MASKNMF_HINT}"
                )
            from mbo_utilities.gui.masknmf_vis import MasknmfViewers

            viewers = MasknmfViewers(
                data_in,
                raw_path=raw_path,
                motion_correction_path=motion_correction_path,
            )
            output = viewers.open(vis)
            if in_notebook():
                display_widget(output)
                return viewers
            import fastplotlib as fpl

            fpl.loop.run()
            return None
        # a .mesc holding AOD ROI units opens on the first one
        if _is_mesc(data_in):
            if unit is None:
                unit = _first_linescan_unit(data_in)
            mesc_kwargs, proceed = _resolve_mesc_unit(data_in, unit)
            if not proceed:
                return None
            unit = mesc_kwargs.get("unit", unit)

        if mode == "Napari viewer":
            return _launch_napari(data_in, roi, unit, dims)
        return _launch_standard_viewer(data_in, roi, widget, metadata_only, unit, dims)
    finally:
        if splash:
            splash.close()


def _resolve_mesc_unit(data_in, unit):
    """The reader kwargs naming the ``.mesc`` unit to open, and whether to proceed.

    A ``.mesc`` opens on its first unit with no prompt; the MESc tab switches
    between them. ``({}, True)`` for anything else. The flag is always True
    now and stays for the callers that check for a cancelled picker.
    """
    if unit is not None:
        return {"unit": unit}, True
    if not _is_mesc(data_in):
        return {}, True
    from mbo_utilities.arrays.mesc import list_mesc_units

    path = Path(data_in)
    units = list_mesc_units(path)
    if not units:
        return {}, True
    if len(units) > 1:
        log.get("gui.boot").info(
            f"{path.name} holds {len(units)} measurement units; opening "
            f"{units[0]['key']} (switch from the MESc tab)."
        )
    return {"unit": units[0]["key"]}, True


def _is_mesc(path) -> bool:
    if not isinstance(path, (str, Path)):
        return False
    return Path(path).is_file() and Path(path).suffix.lower() == ".mesc"


def _first_linescan_unit(path) -> str | None:
    """The key of the first AOD ROI unit (``ROI_LAYOUTS``) in a ``.mesc``, or None.

    A vnoiser run beside the file wins: the viewer opens on a processed scan.
    """
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, list_mesc_units
    from mbo_utilities.results import ResultsArray, newest_results, results_dir_of

    units = [u for u in list_mesc_units(path) if u.get("kind") in ROI_LAYOUTS]
    path = Path(path)
    for parent in (path.parent.parent, path.parent):
        found = newest_results(parent, "vnoiser", source=path) or results_dir_of(
            parent / "PF"
        )
        if found is None:
            continue
        run = ResultsArray(found, source=False)
        wanted = {
            str(u.attrs.get("source_unit", "")).rsplit("/", 1)[-1]
            for u in run.results.units.values()
        }
        for u in units:
            if u["key"].rsplit("/", 1)[-1] in wanted:
                return u["key"]
        break
    return units[0]["key"] if units else None


class ViewerCancelled(Exception):
    """The user dismissed a picker the viewer needed an answer from."""


def _load_for_viewer(data_in, roi=None, unit=None, dims=None):
    """Open ``data_in`` the way the viewer wants it.

    Resolves the ``.mesc`` unit, reads lazily (``dims`` names a plain
    TIFF's page axis), then wraps the array so the viewer shows it aligned
    (per-plane axial shifts saved with the data) and can toggle scan-phase
    correction on formats without a native control. Raises
    ``ViewerCancelled`` when the unit picker was dismissed.
    """
    from mbo_utilities.arrays import normalize_roi
    from mbo_utilities.reader import imread

    logger = log.get("gui.boot")
    roi = normalize_roi(roi)
    mesc_kwargs, proceed = _resolve_mesc_unit(data_in, unit)
    if not proceed:
        raise ViewerCancelled(f"unit selection cancelled for {data_in}")
    if dims:
        mesc_kwargs["dims"] = dims
    t0 = time.perf_counter()
    data_array = imread(data_in, roi=roi, **mesc_kwargs)
    t_imread = time.perf_counter() - t0
    logger.debug(
        f"[boot] imread: {t_imread * 1000:.0f} ms  "
        f"type={type(data_array).__name__}  "
        f"shape={getattr(data_array, 'shape', None)}  "
        f"dims={getattr(data_array, 'dims', None)}"
    )
    return _wrap_for_viewer(data_array, logger)


def _wrap_for_viewer(data_array, logger):
    """Show the planes aligned when the data carries valid ``plane_shifts``, and give every time series without a native phase control a disabled scan-phase wrap the toggle can flip without re-reading."""
    from mbo_utilities.arrays import with_phasecorr
    from mbo_utilities.arrays._registration import (
        validate_axial_shifts,
        with_axial_shifts,
    )

    nt, _, nz = data_array._shape5d()[:3]
    if validate_axial_shifts(data_array.metadata, nz):
        data_array = with_axial_shifts(data_array)
        logger.info(f"axial alignment applied ({nz} planes)")
    if not hasattr(data_array, "phase_correction") and nt > 1:
        data_array = with_phasecorr(data_array)
        logger.debug("scan-phase correction available (disabled)")
    return data_array


def _launch_standard_viewer(data_in, roi, widget, metadata_only, unit=None, dims=None):
    """Build a ``DataVis`` for the path and show it the way this process can.

    A terminal gets a window and the event loop; a notebook gets the canvas
    in the output cell and the ``DataVis`` back so the cell can close it.
    """
    logger = log.get("gui.boot")

    if metadata_only:
        try:
            data_array = _load_for_viewer(data_in, roi=roi, unit=unit, dims=dims)
        except ViewerCancelled:
            return None
        metadata = data_array.metadata
        if not metadata:
            return None
        _show_metadata_viewer(metadata)
        return None

    from mbo_utilities.gui.data_vis import DataVis

    t1 = time.perf_counter()
    try:
        vis = DataVis(data_in, roi=roi, widget=widget, unit=unit, dims=dims)
    except ViewerCancelled:
        return None
    logger.debug(
        f"[boot] DataVis (imread + widget): {(time.perf_counter() - t1) * 1000:.0f} ms"
    )

    output = vis.show()
    if in_notebook():
        display_widget(output)
        return vis

    import fastplotlib as fpl

    fpl.loop.run()
    return None


class _NapariArray:
    """A lazy array as napari reads it: no ``__array__``, so napari slices through ``__getitem__`` instead of taking the first frame as the whole image.

    ``c_index`` pins C and reports the 5D TCZYX array as 4D TZYX, so a
    single-channel file gets one layer with four axis labels instead of a
    stray one-channel split.
    """

    def __init__(self, arr, c_index: int | None = None):
        self._arr = arr
        self._c_index = c_index
        if c_index is not None:
            full = tuple(arr.shape)
            if len(full) != 5:
                raise ValueError(
                    f"_NapariArray(c_index=...) expects 5D TCZYX input, got shape {full}"
                )
            self._shape = (full[0], full[2], full[3], full[4])
            self._ndim = 4
        else:
            self._shape = tuple(arr.shape)
            self._ndim = arr.ndim

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ndim(self):
        return self._ndim

    def __getitem__(self, key):
        import numpy as np

        if self._c_index is None:
            return np.asarray(self._arr[key])
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < 4:
            key = key + (slice(None),) * (4 - len(key))
        full_key = (key[0], int(self._c_index), key[1], key[2], key[3])
        return np.asarray(self._arr[full_key])


_NAPARI_DEFAULTS = {"colormap": "gnuplot2", "contrast_limits": (0, 400)}


def _prompt_for_dz(default: float = 16.0) -> float | None:
    """Ask for the z-step in µm with a modal Qt dialog over napari's running QApplication; None when cancelled or without Qt."""
    try:
        from qtpy.QtWidgets import QApplication, QInputDialog
    except ImportError:
        return None
    if QApplication.instance() is None:
        return None
    value, ok = QInputDialog.getDouble(
        None,
        "Z-spacing required",
        (
            "No `dz` (z-step) found in source metadata.\n\n"
            "Enter the z-spacing in micrometers (\u00b5m):"
        ),
        float(default),
        0.1,
        1000.0,
        3,
    )
    return float(value) if ok else None


def _apply_napari_defaults(layers) -> None:
    """Set the colormap and contrast on every image layer ``add_image`` or ``open`` returned."""
    import napari

    if layers is None:
        return
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    for layer in layers:
        if isinstance(layer, napari.layers.Image):
            layer.colormap = _NAPARI_DEFAULTS["colormap"]
            layer.contrast_limits = _NAPARI_DEFAULTS["contrast_limits"]


def _launch_napari(data_in, roi=None, unit=None, dims=None):
    logger = log.get("gui.napari")
    try:
        import napari
    except ImportError:
        logger.warning(
            "napari not installed. Please install it using `uv pip install napari pyqt6`"
        )
        return None

    from mbo_utilities.arrays import normalize_roi
    from mbo_utilities.arrays.features._dim_labels import get_dims
    from mbo_utilities.metadata.output import OutputMetadata
    from mbo_utilities.reader import imread

    mesc_kwargs, proceed = _resolve_mesc_unit(data_in, unit)
    if not proceed:
        return None
    if dims:
        mesc_kwargs["dims"] = dims
    arr = imread(data_in, roi=normalize_roi(roi), **mesc_kwargs)
    arr_dims = list(get_dims(arr))
    nz = int(arr.shape[arr_dims.index("Z")]) if "Z" in arr_dims else 1
    viewer = napari.Viewer(ndisplay=3 if nz > 1 else 2)
    logger.info(f"napari viewer launched (ndisplay={viewer.dims.ndisplay})")

    path_str = str(data_in)
    if path_str.endswith(".zarr"):
        try:
            added = viewer.open(path_str, plugin="napari-ome-zarr")
        except Exception as e:
            # the plugin raises its own types for a store it cannot read; the layer is built below instead
            logger.debug(f"napari-ome-zarr failed: {e}")
            added = None
        if added:
            _apply_napari_defaults(added)
            napari.run()
            return None

    # napari rejects axis_labels shorter than the rank, so a size-1 C is pinned by _NapariArray
    # and a real C is napari's channel_axis; either way C leaves the labels and the scale
    channel_axis = None
    squeeze_c_index = None
    if "C" in arr_dims:
        c_idx = arr_dims.index("C")
        if int(arr.shape[c_idx]) > 1:
            channel_axis = c_idx
        else:
            squeeze_c_index = 0

    om = OutputMetadata(
        source=arr.metadata, source_shape=arr.shape, source_dims=arr_dims
    )
    vs = om.voxel_size
    logger.info(f"napari scale source: dx={vs.dx}, dy={vs.dy}, dz={vs.dz}")
    scale = list(om.to_napari_scale(arr_dims))
    # without a dz napari uses 1.0 and squashes the 3D view
    if vs.dz is None and nz > 1:
        user_dz = _prompt_for_dz()
        if user_dz is not None:
            scale[arr_dims.index("Z")] = float(user_dz)
            logger.info(f"napari scale: applied user-entered dz={user_dz} um")
        else:
            logger.warning(
                "napari scale: user cancelled dz prompt; "
                "falling back to dz=1.0 (z-axis will be squashed)"
            )
    axis_labels = list(arr_dims)
    if "C" in arr_dims:
        scale.pop(arr_dims.index("C"))
        axis_labels.pop(arr_dims.index("C"))

    layer_kwargs = {
        **_NAPARI_DEFAULTS,
        "scale": tuple(scale),
        "axis_labels": tuple(axis_labels),
    }
    if channel_axis is not None:
        layer_kwargs["channel_axis"] = channel_axis
    layer_name = Path(path_str).stem
    if hasattr(arr, "roi_mode") and hasattr(arr, "iter_rois"):
        for r in arr.iter_rois():
            r_arr = copy.copy(arr)
            r_arr.fix_phase = False
            r_arr.roi = r
            added = viewer.add_image(
                _NapariArray(r_arr, c_index=squeeze_c_index),
                name=f"ROI {r}" if r else layer_name,
                **layer_kwargs,
            )
            _apply_napari_defaults(added)
    else:
        added = viewer.add_image(
            _NapariArray(arr, c_index=squeeze_c_index), name=layer_name, **layer_kwargs
        )
        _apply_napari_defaults(added)
    napari.run()


def run_gui(
    data_in: str | Path | None = None,
    roi: int | tuple[int, ...] | None = None,
    widget: bool | str = True,
    metadata_only: bool = False,
    select_only: bool = False,
    runner_params: Any | None = None,
    unit: int | str | None = None,
    dims: str | None = None,
    vis: str = "demixing",
    raw_path: str | Path | None = None,
    motion_correction_path: str | Path | None = None,
    qc: bool = False,
):
    """
    Open a GUI to preview data of any supported type.

    ``dims`` names a plain TIFF's page axis (``"ZYX"`` for a z-stack,
    ``"CYX"`` for channels) the way ``imread(..., dims=...)`` does.

    ``qc`` opens Registration-Denoising Quality Control on a MaskNMF run
    folder (or its ``results.hdf5``) or a ``mbo reg-denoise`` result, which
    opens there without it. A run folder whose run demixed opens in masknmf's
    demixing viewer; one that did not opens in the Studio.

    A masknmf demixing result opens in masknmf's own viewer instead:
    ``vis`` picks ``demixing`` (default) or ``classification`` before launch;
    the viewer window is masknmf's as is. ``raw_path`` and
    ``motion_correction_path`` give the demixing viewer its raw panel and
    shift traces; omitted, the files beside the result are used when present.

    The one-call form of ``DataVis``: it builds the viewer, picks the canvas
    and size for wherever it is running, and shows it. In a terminal or
    script it opens a window and runs the event loop until it closes. In a
    notebook it puts the canvas in the output cell and returns the
    ``DataVis`` so the cell can reach the viewer and ``close()`` it. The
    file picker is a desktop window, so a notebook has to pass the path.

    Parameters
    ----------
    data_in : str, Path, optional
        Path to data file or directory. If None, shows file selection dialog.
    roi : int, tuple of int, optional
        ROI index(es) to display. None shows all ROIs for raw files.
    widget : bool or str, default True
        Side widget to attach: ``"preview"`` (PreviewDataWidget, the default),
        ``"manualroi"`` (manual ROI drawing) or ``"none"``. ``True``/``False``
        are the legacy spellings of ``"preview"``/``"none"``.
    metadata_only : bool, default False
        If True, only show metadata inspector (no image viewer).
    select_only : bool, default False
        If True, only show file selection dialog and return the selected path.
        Does not load data or open the image viewer.
    runner_params : Any, optional
        hello_imgui.RunnerParams instance for custom window configuration.
    unit : int or str, optional
        For ``.mesc`` inputs, which measurement unit to open — an index or a
        ``"MSession_0/MUnit_3"`` key. When omitted and the file holds more
        than one unit, a picker dialog is shown.

    Returns
    -------
    DataVis, Path, or None
        In a notebook: the ``DataVis``, already shown in the output cell.
        In standalone: None (runs the event loop until closed).
        With select_only=True: the selected path (str or Path).

    Examples
    --------
    In a notebook:

    >>> import mbo_utilities as mbo
    >>> vis = mbo.run_gui("path/to/data.tif")
    >>> vis.iw.cmap = "viridis"
    >>> vis.close()

    or with the class, the way masknmf's viewers work:

    >>> from mbo_utilities import DataVis
    >>> vis = DataVis("path/to/data.tif", roi=1, widget="none")
    >>> vis.show()

    From a script, ``run_gui`` blocks until the window closes:

    >>> mbo.run_gui("path/to/data.tif")
    >>> path = mbo.run_gui(select_only=True)  # only the picker

    From the command line::

        mbo path/to/data.tif
        mbo path/to/data.tif --roi 1 --no-widget
        mbo path/to/data.tif --widget manualroi
        mbo path/to/data.tif --metadata-only
        mbo --select-only
    """
    # CUDA_VISIBLE_DEVICES has to be set before any torch or cupy import; spawned workers inherit it
    from mbo_utilities.gpu import apply_persisted_compute_gpu

    apply_persisted_compute_gpu()
    return _run_gui_impl(
        data_in=data_in,
        roi=roi,
        widget=widget,
        metadata_only=metadata_only,
        select_only=select_only,
        runner_params=runner_params,
        unit=unit,
        dims=dims,
        vis=vis,
        raw_path=raw_path,
        motion_correction_path=motion_correction_path,
        qc=qc,
    )


if __name__ == "__main__":
    run_gui()
