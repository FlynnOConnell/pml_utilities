"""
CLI entry point for mbo_utilities GUI.

This module is designed for fast startup - heavy imports are deferred until needed.
Operations like --check-install should be near-instant.
"""

import contextlib
import functools
import math
import os
import sys
from pathlib import Path
from typing import Any

import click

from mbo_utilities.gui._notebook import display_widget, in_notebook

# Set AppUserModelID immediately for Windows
try:
    import ctypes
    import sys

    if sys.platform == "win32":
        myappid = "mbo.utilities.gui.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass


def _set_qt_icon():
    """Set the Qt application window icon.

    Must be called AFTER the canvas/window is created and shown.
    Sets the icon on QApplication and all top-level windows including
    native window handles for proper Windows taskbar display.
    """
    try:
        from PyQt6.QtGui import QIcon
        from PyQt6.QtWidgets import QApplication

        from mbo_utilities.gui._setup import get_package_assets_path
        from mbo_utilities.preferences import get_mbo_dirs

        app = QApplication.instance()
        if app is None:
            return

        # try package assets first, then user assets
        icon_path = get_package_assets_path() / "app_settings" / "icon.png"
        if not icon_path.exists():
            icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"

        if not icon_path.exists():
            return

        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)

        # set on all top-level windows including native handles
        for window in app.topLevelWidgets():
            window.setWindowIcon(icon)
            handle = window.windowHandle()
            if handle:
                handle.setIcon(icon)
            app.processEvents()
    except Exception:
        pass


class SplashScreen:
    """Simple splash screen using tkinter (always available on windows)."""

    def __init__(self):
        self.root = None
        self._closed = False

    def show(self):
        """Show splash screen with loading indicator."""
        try:
            import tkinter as tk

            self.root = tk.Tk()
            self.root.overrideredirect(True)  # no title bar
            self.root.attributes("-topmost", True)

            # center on screen
            width, height = 300, 120
            x = (self.root.winfo_screenwidth() - width) // 2
            y = (self.root.winfo_screenheight() - height) // 2
            self.root.geometry(f"{width}x{height}+{x}+{y}")

            # styling
            self.root.configure(bg="#1e1e2e")

            # title
            title = tk.Label(
                self.root,
                text="MBO Utilities",
                font=("Segoe UI", 16, "bold"),
                fg="#89b4fa",
                bg="#1e1e2e",
            )
            title.pack(pady=(20, 5))

            # loading text
            self.loading_label = tk.Label(
                self.root,
                text="Loading...",
                font=("Segoe UI", 10),
                fg="#a6adc8",
                bg="#1e1e2e",
            )
            self.loading_label.pack(pady=(0, 10))

            # simple progress animation
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

        except Exception:
            self.root = None

    def _animate(self):
        """Animate the loading dots."""
        if self.root is None or self._closed:
            return
        try:
            for i, dot in enumerate(self.dots):
                dot.configure(fg="#89b4fa" if i == self.current_dot else "#45475a")
            self.current_dot = (self.current_dot + 1) % len(self.dots)
            self.root.after(200, self._animate)
        except Exception:
            pass

    def close(self):
        """Close the splash screen."""
        self._closed = True
        if self.root is not None:
            with contextlib.suppress(Exception):
                self.root.destroy()
            self.root = None


def _get_version() -> str:
    """Get the current mbo_utilities version."""
    try:
        import mbo_utilities

        return getattr(mbo_utilities, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _check_for_upgrade() -> tuple[str, str | None]:
    """Check pypi for newer version of mbo_utilities (cached for 1 hour).

    returns (current_version, latest_version) or (current_version, None) if check fails.
    """
    import json
    import urllib.request

    current = _get_version()

    # check cache first (1 hour expiry)
    try:
        from mbo_utilities.env_cache import get_cached_pypi_version, update_pypi_cache

        cached = get_cached_pypi_version(max_age_hours=1)
        if cached:
            return current, cached
    except Exception:
        pass

    # fetch from pypi
    try:
        url = "https://pypi.org/pypi/mbo-utilities/json"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            latest = data["info"]["version"]
            # update cache
            try:
                update_pypi_cache(latest)
            except Exception:
                pass
            return current, latest
    except Exception:
        return current, None


def _print_upgrade_status():
    """Print upgrade status to console."""
    current, latest = _check_for_upgrade()

    click.echo(f"Current version: {current}")

    if latest is None:
        click.secho(
            "Could not check for updates (network error or package not on PyPI)",
            fg="yellow",
        )
        return

    click.echo(f"Latest version:  {latest}")

    if current == "unknown":
        click.secho("Could not determine current version", fg="yellow")
    elif current == latest:
        click.secho("You are running the latest version!", fg="green")
    else:
        # simple version comparison (works for semver)
        try:
            from packaging.version import parse

            if parse(current) < parse(latest):
                click.secho("\nUpgrade available! Run:", fg="cyan")
                click.secho(
                    "  uv pip install --upgrade mbo-utilities", fg="cyan", bold=True
                )
                click.echo("  or")
                click.secho(
                    "  pip install --upgrade mbo-utilities", fg="cyan", bold=True
                )
            else:
                click.secho(
                    "You are running a newer version than PyPI (dev build)", fg="green"
                )
        except ImportError:
            # no packaging module, do string comparison
            if current != latest:
                click.secho("\nDifferent version on PyPI. To upgrade:", fg="cyan")
                click.secho(
                    "  uv pip install --upgrade mbo-utilities", fg="cyan", bold=True
                )


def _check_installation():
    """Verify that mbo_utilities and key dependencies are properly installed."""
    from mbo_utilities.install import check_installation, print_status_cli

    status = check_installation()
    print_status_cli(status)
    return status.all_ok


def _select_file(runner_params: Any | None = None) -> tuple[Any, Any, Any, bool, str]:
    """Show file selection dialog and return user choices."""
    from imgui_bundle import hello_imgui, immapp

    from mbo_utilities.gui._setup import get_default_ini_path
    from mbo_utilities.gui.widgets.file_dialog import (
        FileDialog,
    )  # triggers _setup import

    dlg = FileDialog()

    from mbo_utilities import __version__

    if runner_params is None:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = (
            f"Miller Brain Studio v{__version__} – Data Selection"
        )
        params.app_window_params.window_geometry.size = (340, 720)
        params.app_window_params.window_geometry.size_auto = False
        params.app_window_params.resizable = True
    else:
        params = runner_params

    # always route the ini to the mbo settings folder, even when a caller
    # (e.g. scripts/capture_docs.py) supplies its own runner_params; otherwise
    # hello_imgui writes <window_title>.ini into the current directory
    if not params.ini_filename:
        params.ini_filename = get_default_ini_path("file_dialog")

    # always override the gui callback to render our dialog
    params.callbacks.show_gui = dlg.render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)

    # Get selected mode
    mode = dlg.gui_modes[dlg.selected_mode_index]

    return (
        dlg.selected_path,
        dlg.split_rois,
        dlg.widget_enabled,
        dlg.metadata_only,
        mode,
    )


def _show_metadata_viewer(metadata: dict) -> None:
    """Show metadata in an ImGui window."""
    from imgui_bundle import hello_imgui, immapp

    from mbo_utilities import __version__
    from mbo_utilities.gui._metadata import draw_metadata_inspector
    from mbo_utilities.gui._setup import get_default_ini_path

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = (
        f"Miller Brain Studio v{__version__} – Metadata"
    )
    params.app_window_params.window_geometry.size = (800, 800)
    params.ini_filename = get_default_ini_path("metadata_viewer")
    params.callbacks.show_gui = lambda: draw_metadata_inspector(metadata)

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)


class _SqueezeSingletonDims:
    """wraps a 5D TCZYX array, dropping every singleton non-spatial dim.

    fastplotlib's ImageWidget derives its slider count from ndim-2 and
    doesn't handle singleton sliders, so any T/C/Z with size 1 must be
    removed before handoff. get_slider_dims filters by the same
    size>1 predicate; this wrapper keeps the shape in lockstep so
    len(slider_dim_names) always equals wrapped.ndim - 2.
    """

    def __init__(self, arr):
        self._arr = arr
        full = tuple(arr.shape)
        if len(full) != 5:
            raise ValueError(f"expected 5D TCZYX array, got shape {full}")
        # keep non-spatial dims only if size > 1, always keep Y, X
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
        # pad short keys to wrapped ndim
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))
        # rebuild a 5D key: 0 at dropped axes, user key at kept axes
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
        # materialize through __getitem__ so the squeezed shape is honored.
        # without this, np.asarray() / .astype() / etc. fall through
        # __getattr__ to the underlying array and leak the unsqueezed rank.
        import numpy as np

        arr = np.asarray(self[tuple(slice(None) for _ in range(self.ndim))])
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def astype(self, dtype, *args, **kwargs):
        # fastplotlib's TextureArray._fix_data calls data.astype(np.float32).
        # route through __array__ so the cast sees the squeezed shape.
        import numpy as np

        return np.asarray(self).astype(dtype, *args, **kwargs)

    def __getattr__(self, name):
        # IMPORTANT: do not forward numpy-array-protocol methods here
        # (__array__, astype, reshape, etc.) — those must be defined
        # explicitly above so they respect the squeezed shape. this
        # fallback exists only for domain-specific attributes like
        # `metadata`, `stack_type`, `num_beamlets`, etc.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._arr, name)


class _ScrubTimingProxy:
    """Logs per-frame __getitem__ time at DEBUG level.

    Applied by `_squeeze_for_viewer` to every array handed to fastplotlib
    so scrub timing surfaces uniformly across reader types
    (IsoviewArray, LBMArray, Suite2pArray, ZarrArray, …) without
    per-class edits. Zero-cost when DEBUG is filtered out: the
    `isEnabledFor` check short-circuits before perf_counter.

    `_arr` deliberately points to the *innermost* reader (peeling any
    `_SqueezeSingletonDims`) so existing `getattr(x, "_arr", x)` patterns
    in `_dialogs.py` / `preview_data.py` continue to surface a real
    ScanImageArray / TiffArray for isinstance checks. Indexing routes
    through `_wrapped` instead, which preserves the squeeze layer's
    shape transform.

    Must define `__array__` and `astype` explicitly — fastplotlib's
    TextureArray._fix_data calls `.astype(np.float32)`, and without an
    explicit hook here `__getattr__` would silently forward to the
    underlying array and bypass our timing.
    """

    def __init__(self, arr):
        import numpy as np

        from mbo_utilities.log import get as get_logger

        self._wrapped = arr
        # peel every wrapper layer (_SqueezeSingletonDims, AxialShiftView,
        # _ChannelView, ...) so isinstance checks downstream see the real
        # reader. No reader class defines `_arr`, so this terminates there.
        inner = arr
        while hasattr(inner, "_arr"):
            inner = inner._arr
        self._arr = inner
        self._np = np
        self._logger = get_logger("gui.scrub")

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
        # fastplotlib's subsample_array probes the histogram range with a
        # bare `arr[0]`. On a >3D lazy array that pulls every non-leading
        # dim into memory (IsoviewArray TCZYX `arr[0]` = all cameras x all
        # Z-planes, several GB). Collapse a bare-int probe to one
        # representative 2D frame so it reads a single plane.
        if isinstance(key, (int, np.integer)) and self._wrapped.ndim > 3:
            key = (
                (int(key),)
                + (0,) * (self._wrapped.ndim - 3)
                + (slice(None), slice(None))
            )
        if not self._logger.isEnabledFor(logging.DEBUG):
            return self._wrapped[key]
        import time

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
    """Wrap `arr` the way the viewer wants it: singleton non-spatial dims
    dropped, then the scrub timer.

    fastplotlib expects ndim == len(slider_dim_names) + 2, so any T/C/Z of
    size 1 must be squeezed, not just C. Module level because anything that
    swaps the viewer's array afterwards (frame averaging) has to reproduce
    exactly this wrapping.
    """
    if not hasattr(arr, "shape") or len(arr.shape) != 5:
        out = arr
    elif any(arr.shape[i] == 1 for i in range(3)):
        out = _SqueezeSingletonDims(arr)
    else:
        out = arr
    return _ScrubTimingProxy(out)


_NOTEBOOK_SIZE = (1400, 900)
# the PreviewDataWidget on the figure's right edge
_PREVIEW_WIDTH = 300


def screen_box() -> tuple[int, int] | None:
    """The screen's available work area less the window frame and title bar,
    ``None`` when there is no Qt screen to ask.
    """
    try:
        from PyQt6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        avail = screen.availableGeometry()
    except Exception:
        return None
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
    # every viewer asks here right before it builds its figure
    from mbo_utilities.gui import _fpl_config  # noqa: F401

    if in_notebook():
        if size is None:
            size = fit_figure_size(_NOTEBOOK_SIZE, **fit) if fit else _NOTEBOOK_SIZE
        return {"canvas": "jupyter", "size": tuple(size)}

    import os

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
        except (ImportError, RuntimeError):  # RuntimeError if qt is already selected
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


# the smallest render viewport a subplot keeps, per side, when the window is
# dragged narrow; below this the window stops shrinking
_MIN_VIEWPORT = 24


def _clamp_window_to_layout(figure, event=None) -> None:
    """Keep the Qt window at least as big as its edge windows, docks and
    per-subplot imgui windows need.

    The figure clamps its render area to 1 px, then each subplot subtracts
    its histogram window, docks and spacing from its share, so a canvas
    narrower than the panels gives pygfx a negative viewport and a
    validation error every frame. Registered on the canvas ``resize``
    event so it follows whatever the edges are sized to at the time.
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
    current = canvas.minimumSize()
    if (current.width(), current.height()) != (min_w, min_h):
        canvas.setMinimumSize(min_w, min_h)


def _after_show(iw) -> None:
    """Window title, icon and minimum size; only meaningful once a desktop
    canvas exists.
    """
    from mbo_utilities import __version__

    canvas = iw.figure.canvas
    if hasattr(canvas, "set_title"):
        canvas.set_title(f"Miller Brain Studio v{__version__}")
    _set_qt_icon()
    if hasattr(canvas, "setMinimumSize"):
        canvas.add_event_handler(
            functools.partial(_clamp_window_to_layout, iw.figure), "resize"
        )
        _clamp_window_to_layout(iw.figure)


def _create_image_widget(
    data_array,
    widget: bool = True,
    figure_kwargs_override=None,
    show: bool = True,
):
    """Create fastplotlib ImageWidget with an optional side widget.

    `widget` names which one to attach: "preview" (PreviewDataWidget),
    "manualroi" (manual ROI drawing), or "none". True/False are the legacy
    spellings of "preview"/"none".

    `figure_kwargs_override` replaces the auto-selected canvas/size dict (used by
    scripts/capture_docs.py to build an offscreen viewer for headless capture).

    `show=False` builds everything but leaves showing to the caller; that is
    how ``DataVis`` separates construction from ``show()``.
    """
    import copy

    import numpy as np

    if isinstance(widget, bool) or widget is None:
        widget = "preview" if widget else "none"
    if widget not in ("preview", "manualroi", "none"):
        raise ValueError(
            f"unknown widget {widget!r}, expected one of: preview, manualroi, none"
        )

    # drawing needs the windowing controls to see anything, so the ROI
    # ui takes the top strip and right-widget tabs alongside the preview
    # widget, not instead of it. Flip the Widgets-menu toggle on for this session
    # when asked for, or when this data has annotations or pipeline ROIs
    # beside it; the widget builds itself from the toggle. Not persisted — the
    # flag came from the command line or the disk, not the menu.
    manual_roi = signal_quality = False
    if widget != "none":
        from mbo_utilities.gui.manual_roi import labels_path
        from mbo_utilities.gui.roi_runs import run_dir_complete
        from mbo_utilities.gui.widgets.widget_toggles import widget_enabled

        src = data_array.source_path
        manual_roi = (
            widget == "manualroi"
            or widget_enabled("manual_roi")
            or (
                src is not None
                and (
                    labels_path(src).exists()
                    or run_dir_complete(labels_path(src).parent)
                )
            )
        )
        signal_quality = widget_enabled("signal_quality")

    # Determine slider dimension names from array's dims property if available
    from mbo_utilities.arrays.features import get_slider_dims

    custom_labels = getattr(data_array, "slider_dim_labels", None)
    if custom_labels:
        slider_dim_names = tuple(custom_labels)
    else:
        slider_dim_names = get_slider_dims(data_array)

    # window_funcs/window_sizes must match slider_dim_names length
    if slider_dim_names:
        n_sliders = len(slider_dim_names)
        # apply mean to first dim (usually t), None for rest
        window_funcs = (np.mean,) + (None,) * (n_sliders - 1)
        window_sizes = (1,) + (None,) * (n_sliders - 1)
    else:
        window_funcs = None
        window_sizes = None

    def _is_isoview(arr) -> bool:
        """IsoviewArray has a `kind` attribute set to raw/corrected/fused."""
        return getattr(arr, "kind", None) in {"raw", "corrected", "fused", "clusterpt"}

    # Isoview fluorescence sits in the 0..few-hundred-counts range; the
    # generic (-100, 4000) default washes it out completely.
    if _is_isoview(data_array):
        graphic_kwargs = {"vmin": 0, "vmax": 1000}
    else:
        graphic_kwargs = {"vmin": -100, "vmax": 4000}

    # Handle multi-ROI data (duck typing: check for roi_mode attribute)
    if hasattr(data_array, "roi_mode") and hasattr(data_array, "iter_rois"):
        arrays = []
        names = []
        # get name from first filename if available, truncate if too long
        base_name = None
        if hasattr(data_array, "filenames") and data_array.filenames:
            from pathlib import Path

            first_file = Path(data_array.filenames[0])
            base_name = first_file.stem
            # for suite2p arrays (data.bin), use parent folder name instead
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
        # the strip is as tall as the tab that will be selected: the Traces
        # panel registers first, else the Signal Quality plot once its stats
        # are in; the ROI controls are a right-bar tab and need no width
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
        window_funcs=window_funcs,
        window_sizes=window_sizes,
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
        # the EdgeWindow shim registers itself with the figure during
        # __init__; add_gui exists only on mbo-fastplotlib
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
    vis: str = "demixing",
    raw_path: str | Path | None = None,
    motion_correction_path: str | Path | None = None,
):
    """Internal implementation of run_gui with all heavy imports."""
    # Apply persisted Options (GPU adapter, debug logging) before any
    # fastplotlib Figure is created or the launch dialog renders. Also
    # prime the adapter cache so the file_dialog's Options popup doesn't
    # call enumerate_adapters() from inside the imgui frame (that
    # initializes wgpu, which on Windows clears the current WGL context
    # and makes the host window flicker — Glfw Error 65544).
    try:
        from mbo_utilities.gui import _gpu_cache
        from mbo_utilities.preferences import get_debug_logging, get_gpu_index

        _gpu_cache.prime()
        _gpu_idx = get_gpu_index()
        _adapters = _gpu_cache.get_adapters()
        if _gpu_idx >= 0 and 0 <= _gpu_idx < len(_adapters):
            import fastplotlib as fpl

            fpl.select_adapter(_adapters[_gpu_idx])
        # an explicit MBO_DEBUG (mbo --debug / --no-debug) wins over the preference
        if get_debug_logging() and "MBO_DEBUG" not in os.environ:
            from mbo_utilities import log as _mbo_log

            _mbo_log.set_debug(True)
    except Exception:
        pass

    # show splash screen while loading (only for desktop shortcut launches)
    splash = None
    if show_splash and not _is_jupyter():
        splash = SplashScreen()
        splash.show()

    try:
        # Import heavy dependencies only when actually running GUI

        # close splash before showing file dialog
        if splash:
            splash.close()
            splash = None

        # Handle file selection if no path provided
        if data_in is None:
            if in_notebook():
                # the launcher is a desktop window with native dialogs; it
                # would open on the kernel's machine, which for a hub session
                # is not the one the user is looking at
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
            # Use ROI from dialog if not specified in function call
            if roi is None:
                roi = roi_from_dialog

        # If select_only, just return the path
        if select_only:
            return data_in

        # the file dialog hands back a list even for one file
        if isinstance(data_in, (list, tuple)) and len(data_in) == 1:
            data_in = data_in[0]
        # a masknmf demixing result opens in masknmf's own viewers
        from mbo_utilities.arrays.demixing import has_demixing_results

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
        # a .mesc holding AOD ROI units (line scans, chessboard or ribbon
        # patches) opens on the first one with no prompt: the Voltage
        # pipeline follows the unit on screen and offers the other scans
        # there, and its Curate button opens the curation window. A PF or
        # experiment folder opens as a ResultsArray through imread, like a suite2p
        # folder.
        # Other .mesc files prompt for their unit once, here.
        if _is_mesc(data_in):
            if unit is None:
                unit = _first_linescan_unit(data_in)
            mesc_kwargs, proceed = _resolve_mesc_unit(data_in, unit)
            if not proceed:
                return None
            unit = mesc_kwargs.get("unit", unit)

        # Dispatch based on Mode
        # pollen calibration is auto-detected in the fastplotlib viewer via get_viewer_class()
        if mode == "Fastplotlib viewer (default)":
            return _launch_standard_viewer(data_in, roi, widget, metadata_only, unit)
        if mode == "Napari viewer":
            return _launch_napari(data_in, roi, unit)
        return _launch_standard_viewer(data_in, roi, widget, metadata_only, unit)

    finally:
        # ensure splash is closed on any exit path
        if splash:
            splash.close()


def _fmt_duration(seconds):
    """Compact recording length: '' when unknown, seconds under a minute."""
    if not seconds:
        return ""
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


def _resolve_mesc_unit(data_in, unit):
    """Reader kwargs selecting which MUnit of a ``.mesc`` to open.

    Every ``.mesc`` opens straight to its first measurement unit, no prompt.
    A file with more than one MUnit (unrelated scans the operator ran back
    to back) is switched between from the MESc tab
    (``mbo_utilities.gui.widgets.mesc_units.MescTabWidget``), an ImGui
    widget like the rest of the viewer — no Qt involved anywhere in this
    path. An explicit ``unit`` is the deliberate bypass.

    Returns ``({}, True)`` for anything that isn't a ``.mesc``. The second
    element is kept (always True now) so existing callers that check for a
    cancelled picker don't need to change.
    """
    from mbo_utilities.log import get as _get

    logger = _get("gui.boot")
    if unit is not None:
        return {"unit": unit}, True
    try:
        path = Path(data_in)
    except TypeError:
        return {}, True
    if not (path.is_file() and path.suffix.lower() == ".mesc"):
        return {}, True

    from mbo_utilities.arrays.mesc import list_mesc_units

    try:
        units = list_mesc_units(path)
    except Exception as e:
        logger.warning(f"could not list units in {path.name}: {e}")
        return {}, True
    if not units:
        return {}, True  # let imread raise the real "nothing readable" error
    if len(units) > 1:
        logger.info(
            f"{path.name} holds {len(units)} measurement units; opening "
            f"{units[0]['key']} (switch from the MESc tab)."
        )
    return {"unit": units[0]["key"]}, True


def _is_mesc(path) -> bool:
    try:
        p = Path(path)
    except TypeError:
        return False
    return p.is_file() and p.suffix.lower() == ".mesc"


def _mesc_unit(path, unit) -> dict | None:
    """The unit record ``unit`` names in the ``.mesc``: an index, a
    ``MSession_0/MUnit_3`` key or a bare ``MUnit_3``; None picks the first.
    """
    from mbo_utilities.arrays.mesc import list_mesc_units

    try:
        units = list_mesc_units(path)
    except Exception:
        return None
    if not units:
        return None
    if unit is None:
        return units[0]
    if isinstance(unit, int):
        return units[unit] if 0 <= unit < len(units) else None
    return next((u for u in units if unit in (u["key"], u["munit"])), None)


def _is_linescan_unit(path, unit) -> bool:
    """Whether the unit is an AOD line scan (a "packed" unit: one row per
    line, one column per sample along it).
    """
    if unit is None:
        return False
    chosen = _mesc_unit(path, unit)
    return chosen is not None and chosen.get("kind") == "packed"


def _first_linescan_unit(path) -> str | None:
    """The key of the first AOD ROI unit (a line scan, chessboard or ribbon
    scan: ``ROI_LAYOUTS``) in a ``.mesc``, or None. With a PF folder beside
    the file (the voltage pipeline's output) the unit of its first scan wins,
    so the viewer opens on a processed scan.
    """
    from mbo_utilities.arrays.mesc import ROI_LAYOUTS, list_mesc_units
    from mbo_utilities.results import ResultsArray, newest_results, results_dir_of

    try:
        units = [u for u in list_mesc_units(path) if u.get("kind") in ROI_LAYOUTS]
    except Exception:
        return None
    path = Path(path)
    for parent in (path.parent.parent, path.parent):
        found = newest_results(parent, "voltage") or results_dir_of(parent / "PF")
        if found is None:
            continue
        try:
            run = ResultsArray(found, source=False)
        except Exception:
            break
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


def _load_for_viewer(data_in, roi=None, unit=None):
    """Open ``data_in`` the way the viewer wants it.

    Resolves the ``.mesc`` unit, reads lazily, then wraps the array so the
    viewer shows it aligned (per-plane axial shifts saved with the data) and
    can toggle scan-phase correction on formats without a native control.
    Raises ``ViewerCancelled`` when the unit picker was dismissed.
    """
    import time as _time

    from mbo_utilities.arrays import normalize_roi
    from mbo_utilities.log import get as get_logger
    from mbo_utilities.reader import imread

    logger = get_logger("gui.boot")
    roi = normalize_roi(roi)
    # a .mesc holds many unrelated scans; pick one before opening anything
    mesc_kwargs, proceed = _resolve_mesc_unit(data_in, unit)
    if not proceed:
        raise ViewerCancelled(f"unit selection cancelled for {data_in}")
    t0 = _time.perf_counter()
    data_array = imread(data_in, roi=roi, **mesc_kwargs)
    t_imread = _time.perf_counter() - t0
    logger.debug(
        f"[boot] imread: {t_imread * 1000:.0f} ms  "
        f"type={type(data_array).__name__}  "
        f"shape={getattr(data_array, 'shape', None)}  "
        f"dims={getattr(data_array, 'dims', None)}"
    )
    return _wrap_for_viewer(data_array, logger)


def _wrap_for_viewer(data_array, logger):
    """Axial alignment and scan-phase wraps; both transparent to the caller."""
    # if the dataset carries valid per-plane axial shifts (written by
    # imwrite(register_z=True)), show the aligned view by default. transparent
    # wrap: shape/dims/metadata/domain attrs forward to the source; only the
    # (y, x) of each plane is remapped on read.
    try:
        from mbo_utilities.arrays._registration import (
            validate_axial_shifts,
            with_axial_shifts,
        )

        _md = getattr(data_array, "metadata", None)
        _nz = int(data_array._shape5d()[2]) if hasattr(data_array, "_shape5d") else None
        if validate_axial_shifts(_md, _nz):
            try:
                data_array = with_axial_shifts(data_array)
                logger.info(f"axial alignment applied ({_nz} planes)")
            except Exception as e:
                # shifts are present but couldn't be applied: warn so the view
                # isn't silently identical to the unregistered data.
                logger.warning(f"plane_shifts present but not applied: {e}")
    except Exception as e:
        logger.debug(f"axial alignment skipped: {e}")

    # offer scan-phase correction on any time-dimension array that lacks a
    # native phase_correction feature (zarr, h5, npy, mp4, plain tiff). the
    # wrap is transparent (disabled), so the GUI toggle flips it on/off without
    # re-reading from disk, exactly like ScanImageArray's built-in control.
    try:
        if not hasattr(data_array, "phase_correction"):
            from mbo_utilities.arrays import with_phasecorr

            _s5 = data_array._shape5d() if hasattr(data_array, "_shape5d") else None
            if _s5 and int(_s5[0]) > 1:
                data_array = with_phasecorr(data_array)
                logger.debug("scan-phase correction available (disabled)")
    except Exception as e:
        logger.debug(f"phasecorr wrap skipped: {e}")
    return data_array


def _launch_standard_viewer(data_in, roi, widget, metadata_only, unit=None):
    """Build a ``DataVis`` for the path and show it the way this process can.

    A terminal gets a window and the event loop; a notebook gets the canvas
    in the output cell and the ``DataVis`` back so the cell can close it.
    """
    import time as _time

    from mbo_utilities.log import get as get_logger

    logger = get_logger("gui.boot")

    if metadata_only:
        try:
            data_array = _load_for_viewer(data_in, roi=roi, unit=unit)
        except ViewerCancelled:
            return None
        metadata = data_array.metadata
        if not metadata:
            return None
        _show_metadata_viewer(metadata)
        return None

    from mbo_utilities.gui.data_vis import DataVis

    t1 = _time.perf_counter()
    try:
        vis = DataVis(data_in, roi=roi, widget=widget, unit=unit)
    except ViewerCancelled:
        return None
    logger.debug(
        f"[boot] DataVis (imread + widget): {(_time.perf_counter() - t1) * 1000:.0f} ms"
    )

    output = vis.show()
    if in_notebook():
        display_widget(output)
        return vis

    import fastplotlib as fpl

    fpl.loop.run()
    return None


class _NapariArray:
    """Wrapper that prevents napari from eagerly materializing lazy arrays.

    mbo_utilities lazy arrays define __array__() returning the first frame.
    Napari calls np.asarray() on the object when added, triggering materialization
    of just that 2D frame instead of the full ND dataset, breaking the viewer.
    By not defining __array__, we force napari to use __getitem__ to access slices.

    When `c_index` is set, the wrapper presents the underlying 5D TCZYX
    array as a 4D TZYX view by pinning C to that index. This is what
    `_launch_napari` uses when the data is single-channel — passing the
    raw 5D array makes napari see 5 dimensions and reject 4-element
    `axis_labels`, while `channel_axis=1` would create a stray
    one-channel split. Squeezing C here gets us a clean 4D view that
    napari treats as a single (T, Z, Y, X) image layer.
    """

    def __init__(self, arr, c_index: int | None = None):
        self._arr = arr
        self._c_index = c_index
        if c_index is not None:
            # advertise 4D (T, Z, Y, X) by stripping the C axis at index 1
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

        # rebuild a 5D TCZYX key from the user's 4D TZYX key by inserting
        # the pinned c_index at position 1. napari's slicing protocol
        # passes a tuple of slices/ints; pad short keys with full slices
        # the same way numpy does so partial indexes still work.
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < 4:
            key = key + (slice(None),) * (4 - len(key))
        full_key = (key[0], int(self._c_index), key[1], key[2], key[3])
        return np.asarray(self._arr[full_key])


# napari layer defaults — used for every layer added by `_launch_napari`,
# including the napari-ome-zarr plugin path (we set them post-add there).
# If you want to change them globally, change them here, not at every
# add_image call site.
_NAPARI_DEFAULTS = {
    "colormap": "gnuplot2",
    "contrast_limits": (0, 400),
}


def _prompt_for_dz(default: float = 16.0) -> float | None:
    """Pop a Qt input dialog asking the user for the z-spacing.

    Used by `_launch_napari` when the source metadata doesn't supply `dz`
    (LBM TIFFs are the common case — `dz` has to come from the user). The
    napari Viewer is already running so its QApplication is alive; we
    just need a modal QInputDialog on top of it.

    Returns the entered value in micrometers, or None if the user
    cancelled. The dialog uses QInputDialog.getDouble which validates
    range and decimals natively, so we don't need a separate validator.

    Range: 0.1 - 1000.0 µm. Anything outside that is almost certainly a
    typo (sub-100nm pixels are nanoscale; >1mm is not light microscopy).
    """
    try:
        from qtpy.QtWidgets import QApplication, QInputDialog
    except ImportError:
        try:
            from PyQt6.QtWidgets import QApplication, QInputDialog
        except ImportError:
            return None  # no Qt available — caller falls back to default

    # need a QApplication instance to parent the dialog. napari's already
    # running one by the time we get here, but be defensive.
    app = QApplication.instance()
    if app is None:
        return None

    value, ok = QInputDialog.getDouble(
        None,
        "Z-spacing required",
        (
            "No `dz` (z-step) found in source metadata.\n\n"
            "Enter the z-spacing in micrometers (\u00b5m):"
        ),
        float(default),  # initial value
        0.1,  # min
        1000.0,  # max
        3,  # decimals
    )
    return float(value) if ok else None


def _launch_napari(data_in, roi=None, unit=None):
    from mbo_utilities.log import get as get_logger

    logger = get_logger("gui.napari")

    try:
        import napari
    except ImportError:
        logger.warning(
            "napari not installed. Please install it using `uv pip install napari pyqt6`"
        )
        return None

    try:
        from pathlib import Path

        from mbo_utilities.arrays import normalize_roi
        from mbo_utilities.arrays.features._dim_labels import get_dims
        from mbo_utilities.metadata.output import OutputMetadata
        from mbo_utilities.reader import imread

        path_str = str(data_in)

        # a .mesc holds many unrelated scans; pick one before opening anything
        mesc_kwargs, proceed = _resolve_mesc_unit(data_in, unit)
        if not proceed:
            return None

        # Probe the source up front so we can pick the right viewer mode
        # (3D when there's a real Z axis) and log the spatial scale we'll
        # be passing to napari. We re-load below for the actual layer when
        # falling out of the ome-zarr plugin path; the probe is cheap.
        probe_arr = None
        probe_dims = None
        probe_z_size = 1
        try:
            probe_arr = imread(data_in, roi=normalize_roi(roi), **mesc_kwargs)
            probe_dims = get_dims(probe_arr)
            if "Z" in probe_dims:
                probe_z_size = int(probe_arr.shape[probe_dims.index("Z")])
        except Exception as e:
            logger.debug(f"napari probe failed (will still try plugin path): {e}")

        # 3D display when the data actually has multiple z-planes; otherwise
        # 2D so the canvas isn't constantly switching back and forth.
        ndisplay = 3 if probe_z_size > 1 else 2
        viewer = napari.Viewer(ndisplay=ndisplay)
        logger.info(f"napari viewer launched (ndisplay={ndisplay})")

        def _apply_layer_defaults(layer):
            """Force colormap + contrast on a layer, ignoring failures."""
            try:
                layer.colormap = _NAPARI_DEFAULTS["colormap"]
            except Exception:
                pass
            try:
                layer.contrast_limits = _NAPARI_DEFAULTS["contrast_limits"]
            except Exception:
                pass

        # Try napari-ome-zarr plugin first for .zarr files
        loaded = False
        if path_str.endswith(".zarr"):
            try:
                added = viewer.open(path_str, plugin="napari-ome-zarr")
                # `viewer.open` returns the list of layers it created
                if added:
                    for layer in added:
                        _apply_layer_defaults(layer)
                loaded = True
            except Exception as e:
                # OME-Zarr plugin failed, fall back to mbo_utilities
                logger.debug(f"napari-ome-zarr failed: {e}")

        if not loaded:
            # Load via mbo_utilities and add as layer
            try:
                arr = (
                    probe_arr
                    if probe_arr is not None
                    else imread(data_in, roi=normalize_roi(roi), **mesc_kwargs)
                )
                dims = probe_dims if probe_dims is not None else get_dims(arr)

                # 5D TCZYX contract:
                #   - real multi-channel (C > 1) → use napari's channel_axis,
                #     leave the array 5D, napari splits per channel
                #   - singleton C (the common case for LBM) → tell the
                #     wrapper to squeeze C, so napari sees a clean 4D
                #     (T, Z, Y, X) layer with no stray "channels" slider
                # We must NOT just drop C from axis_labels while leaving
                # the underlying array 5D — napari rejects that with
                # `axis_labels must have length ndim=5`.
                channel_axis = None
                squeeze_c_index = None
                if "C" in dims:
                    c_idx = list(dims).index("C")
                    c_size = int(arr.shape[c_idx]) if c_idx < len(arr.shape) else 1
                    if c_size > 1:
                        channel_axis = c_idx
                    else:
                        squeeze_c_index = 0

                # Build scale via OutputMetadata.to_napari_scale, which
                # already pulls dx/dy/dz off the source metadata via the
                # voxel_size feature. Log the resolved values so the user
                # can confirm the metadata was actually picked up rather
                # than silently falling back to (1, 1, 1).
                scale = None
                resolved_dz = None
                try:
                    om = OutputMetadata(
                        source=arr.metadata,
                        source_shape=arr.shape,
                        source_dims=dims,
                    )
                    vs = om.voxel_size
                    logger.info(
                        f"napari scale source: dx={vs.dx}, dy={vs.dy}, dz={vs.dz}"
                    )
                    scale = list(om.to_napari_scale(dims))
                    resolved_dz = vs.dz
                except Exception as e:
                    logger.debug(f"Failed to build scale: {e}")

                # If the source didn't supply a dz and we actually have
                # multiple z-planes, prompt the user. Otherwise napari
                # silently falls back to dz=1.0 and the 3D view is
                # squashed in the wrong proportion. The dialog is the
                # same one the metadata editor would show, so the value
                # entered here is treated as authoritative for this
                # session and written into the array's scale tuple.
                if (
                    resolved_dz is None
                    and "Z" in dims
                    and probe_z_size > 1
                    and scale is not None
                ):
                    user_dz = _prompt_for_dz()
                    if user_dz is not None:
                        z_idx = list(dims).index("Z")
                        if z_idx < len(scale):
                            scale[z_idx] = float(user_dz)
                        logger.info(
                            f"napari scale: applied user-entered dz={user_dz} um"
                        )
                    else:
                        logger.warning(
                            "napari scale: user cancelled dz prompt; "
                            "falling back to dz=1.0 (z-axis will be squashed)"
                        )

                axis_labels = list(dims)
                # Drop the C axis from scale and labels regardless of whether
                # it's a real channel split — napari's `channel_axis` strips
                # it from the displayed dims for us, and for the singleton
                # case we collapsed it to None and want it gone anyway.
                if "C" in dims:
                    c_idx_for_strip = list(dims).index("C")
                    if scale and len(scale) > c_idx_for_strip:
                        scale.pop(c_idx_for_strip)
                    if len(axis_labels) > c_idx_for_strip:
                        axis_labels.pop(c_idx_for_strip)

                layer_name = Path(path_str).stem

                def _add_to_viewer(layer_arr, name):
                    # squeeze the singleton C axis when there isn't a
                    # real channel split — napari then sees a 4D
                    # (T, Z, Y, X) view with rank matching axis_labels.
                    n_arr = _NapariArray(layer_arr, c_index=squeeze_c_index)
                    kwargs = {
                        "name": name,
                        "colormap": _NAPARI_DEFAULTS["colormap"],
                        "contrast_limits": _NAPARI_DEFAULTS["contrast_limits"],
                    }
                    if scale:
                        kwargs["scale"] = tuple(scale)
                    if channel_axis is not None:
                        kwargs["channel_axis"] = channel_axis

                    try:
                        added_layers = viewer.add_image(
                            n_arr, axis_labels=tuple(axis_labels), **kwargs
                        )
                    except TypeError:
                        # Older napari without axis_labels kwarg
                        added_layers = viewer.add_image(n_arr, **kwargs)

                    # add_image returns either a single layer or a list when
                    # channel_axis is set; normalize and re-apply the
                    # defaults in case napari overrode them after add.
                    if added_layers is None:
                        return
                    if not isinstance(added_layers, (list, tuple)):
                        added_layers = [added_layers]
                    for layer in added_layers:
                        _apply_layer_defaults(layer)

                # Handle multi-ROI
                if hasattr(arr, "roi_mode") and hasattr(arr, "iter_rois"):
                    import copy

                    for r in arr.iter_rois():
                        r_arr = copy.copy(arr)
                        r_arr.fix_phase = False
                        r_arr.roi = r
                        name = f"ROI {r}" if r else layer_name
                        _add_to_viewer(r_arr, name)
                else:
                    _add_to_viewer(arr, layer_name)

                loaded = True
            except Exception as e:
                logger.error(f"Failed to load via mbo_utilities: {e}")

        if not loaded:
            # Last resort: let napari try to open it directly
            try:
                added = viewer.open(path_str)
                if added:
                    for layer in added:
                        _apply_layer_defaults(layer)
            except Exception as e:
                logger.error(f"Failed to open via napari default fallback: {e}")

        napari.run()
    except Exception as e:
        logger.error(f"Error launching napari: {e}")


def run_gui(
    data_in: str | Path | None = None,
    roi: int | tuple[int, ...] | None = None,
    widget: bool | str = True,
    metadata_only: bool = False,
    select_only: bool = False,
    runner_params: Any | None = None,
    unit: int | str | None = None,
    vis: str = "demixing",
    raw_path: str | Path | None = None,
    motion_correction_path: str | Path | None = None,
):
    """
    Open a GUI to preview data of any supported type.

    A masknmf demixing result opens in masknmf's own viewer instead:
    ``vis`` picks ``demixing`` (default), ``compression`` or
    ``classification`` before launch; the viewer window is masknmf's as is.
    ``raw_path`` and ``motion_correction_path`` give the demixing viewer its
    raw panel and shift traces (and the compression viewer its raw movie);
    omitted, the files beside the result are used when present.

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
    # resolve compute-GPU policy -> CUDA_VISIBLE_DEVICES before any torch/cupy
    # import or worker spawn; detached workers inherit this environment. Env
    # MBO_GPU overrides the persisted GUI preference.
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
        vis=vis,
        raw_path=raw_path,
        motion_correction_path=motion_correction_path,
    )


if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
