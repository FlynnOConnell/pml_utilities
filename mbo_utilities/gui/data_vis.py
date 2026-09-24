"""``DataVis``: the Miller Brain Studio viewer as an object.

The same viewer ``mbo path/to/data`` opens, built the way masknmf's
``*Vis`` classes are: construct with the data, ``show()`` to put it on
screen, ``close()`` to take it down. ``run_gui`` is the one-call form that
also picks the canvas and size for wherever it is running.

In a notebook, construct and call ``show()`` in the same cell; the canvas is
the cell's output and the kernel's loop drives it, so ``fpl.loop.run()``
must not be called. In a terminal or script, ``show()`` opens a window and
the caller runs ``fpl.loop.run()`` (``run_gui`` does).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["DataVis"]


class DataVis:
    """Preview imaging data of any supported type in the app host.

    Parameters
    ----------
    data : str, Path, array, or sequence of paths
        Anything ``imread`` opens, or an array already in memory.
    roi : int or tuple of int, optional
        ROI index(es) for multi-ROI raw files. None shows the stitched image,
        0 every ROI side by side.
    widget : str, default "preview"
        ``"preview"`` registers every app, ``"manualroi"`` also turns manual
        ROI labeling on, ``"none"`` shows only the viewer.
    unit : int or str, optional
        Which measurement unit of a ``.mesc`` to open. In a terminal the
        picker asks when this is omitted; in a notebook it is required when
        the file holds more than one.
    size : tuple of int, optional
        Canvas size in pixels. Defaults to the screen's work area for a
        desktop window and to (1400, 900) in a notebook.
    figure_kwargs
        Passed on to the figure, e.g. ``canvas="jupyter"``. ``run_gui`` sets
        these itself.

    Examples
    --------
    In a notebook::

        from mbo_utilities import DataVis
        vis = DataVis("path/to/data.tif")
        vis.show()

    then ``vis.close()`` when done. From a script::

        import fastplotlib as fpl
        vis = DataVis("path/to/data.tif")
        vis.show()
        fpl.loop.run()
    """

    def __init__(
        self,
        data: Any,
        roi: int | tuple[int, ...] | None = None,
        widget: bool | str = "preview",
        unit: int | str | None = None,
        size: tuple[int, int] | None = None,
        **figure_kwargs,
    ):
        from mbo_utilities.gui.app import build_host
        from mbo_utilities.gui.run_gui import _load_for_viewer
        from mbo_utilities.gui.widgets.style_editor import style_store

        if isinstance(widget, bool) or widget is None:
            widget = "preview" if widget else "none"
        if widget not in ("preview", "manualroi", "none"):
            raise ValueError(
                f"unknown widget {widget!r}, expected one of: preview, manualroi, none"
            )
        self._source = data
        self._data_array = _load_for_viewer(data, roi=roi, unit=unit)
        self._host = build_host(
            self._data_array,
            apps=[] if widget == "none" else None,
            size=size,
            store=None if widget == "none" else style_store(),
            figure_kwargs=figure_kwargs,
        )
        if widget == "manualroi":
            self._host.apps["manual_roi"].open = True
        self._output = None
        self._shown = False
        self._closed = False

    def show(self, **kwargs):
        """Show the figure. Returns the canvas, which a notebook cell displays
        when it is the last expression.
        """
        from mbo_utilities.gui.run_gui import _after_show

        if not self._shown:
            # offscreen and desktop canvases return None here; only the
            # notebook canvas is a widget worth handing back
            self._output = self._host.figure.show(**kwargs)
            self._shown = True
            _after_show(self._host.viewer)
            self._host.figure.canvas.set_title(self._host.title())
        return self._output

    def close(self) -> None:
        """Stop the apps' threads, release the files they hold and close the figure."""
        if self._closed:
            return
        self._closed = True
        self._host.close()
        self._host.viewer.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def host(self):
        """The ``AppHost``: the apps, the playhead, the open data."""
        return self._host

    @property
    def iw(self):
        """The ``MboNDViewer`` underneath: sliders, cmap, window functions."""
        return self._host.viewer

    @property
    def image_widget(self):
        return self._host.viewer

    @property
    def figure(self):
        return self._host.figure

    @property
    def data(self):
        """The array as the viewer sees it (after axial and phase wraps)."""
        return self._data_array

    @property
    def source(self):
        """What the viewer was built from: a path, paths, or an array."""
        return self._source

    def __repr__(self) -> str:
        src = self._source
        if isinstance(src, (str, Path)):
            what = Path(src).name
        elif isinstance(src, (list, tuple)) and src:
            what = f"{len(src)} files"
        else:
            what = type(src).__name__
        state = "closed" if self._closed else ("shown" if self._shown else "built")
        return f"DataVis({what}, shape={tuple(self._data_array.shape)}, {state})"
