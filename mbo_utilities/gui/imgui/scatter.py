"""A pickable implot scatter plot.

Per-point colours, hover tooltip, click to focus, a ring on the focused
point. ``items`` draws inside a plot someone else opened (markers over a
trace, say); ``draw`` opens its own. Axes are linear; picking works in
pixels so a cluster of points is picked by what is nearest on screen.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from imgui_bundle import imgui, implot

from mbo_utilities.gui.imgui.lines import line_plot, packed_colors, vec4

__all__ = ["ScatterPlot", "nearest_index"]

FOCUS_COLOR = (0.85, 0.15, 0.25, 1.0)
HOVER_COLOR = (1.0, 1.0, 1.0, 0.9)
OUTLINE_COLOR = (0.1, 0.1, 0.1, 0.8)


def nearest_index(px, py, mouse_x: float, mouse_y: float, radius: float) -> int | None:
    """Index of the point nearest the mouse, within ``radius`` pixels, else
    None. NaN points are never picked."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    if px.size == 0:
        return None
    d2 = (px - float(mouse_x)) ** 2 + (py - float(mouse_y)) ** 2
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    i = int(np.argmin(d2))
    return i if d2[i] <= float(radius) ** 2 else None


class ScatterPlot:
    """Points with per-point colours that the user can hover and click.

    Parameters
    ----------
    plot_id : str
        implot id when ``draw`` opens its own plot; also the ring item ids.
    marker_size : float
        Marker diameter in pixels.
    pick_radius : float
        How close, in pixels, the mouse must be to pick a point.
    """

    def __init__(self, plot_id: str, marker_size: float = 6.0, pick_radius: float = 8.0):
        self.plot_id = plot_id
        self.marker_size = float(marker_size)
        self.pick_radius = float(pick_radius)
        self.focus_color = FOCUS_COLOR
        self.hovered: int | None = None
        self._fit = True

    def refit(self) -> None:
        """Fit the axes to the points on the next ``draw``."""
        self._fit = True

    def items(
        self,
        x,
        y,
        colors=None,
        *,
        focused: int | None = None,
        label: str = "points",
        legend: bool = False,
        tooltip: Callable[[int], str] | None = None,
        groups: list[tuple[str, np.ndarray]] | None = None,
    ) -> int | None:
        """Draw inside an open plot; returns the index clicked this frame.

        Parameters
        ----------
        x, y : array
            Point coordinates in plot units.
        colors : array, optional
            ``(n, 4)`` rgba in 0..1 per point; the item colour otherwise.
        focused : int, optional
            Index drawn with a ring.
        label : str
            Legend entry when ``groups`` is not given.
        legend : bool
            Show legend entries.
        tooltip : callable, optional
            ``tooltip(i)`` text for the hovered point; ``i: x, y`` otherwise.
        groups : list of (label, indices), optional
            One legend entry per group instead of one for all points.
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = min(x.size, y.size)
        packed = packed_colors(colors) if colors is not None and n else None
        for name, idx in groups or [(label, None)]:
            self._points(name, x, y, packed, idx, legend)

        hovered = None
        if n and implot.is_plot_hovered():
            hovered = self._pick(x, y)
        self.hovered = hovered
        clicked = hovered is not None and imgui.is_mouse_clicked(0)
        if hovered is not None:
            self._ring(x[hovered], y[hovered], HOVER_COLOR, 1.5, "##hover")
            text = tooltip(hovered) if tooltip else f"{hovered}: {x[hovered]:.3g}, {y[hovered]:.3g}"
            imgui.set_tooltip(text)
        if focused is not None and 0 <= focused < n:
            self._ring(x[focused], y[focused], self.focus_color, 2.5, "##focus")
        return hovered if clicked else None

    def draw(
        self,
        x,
        y,
        colors=None,
        *,
        x_label: str = "",
        y_label: str = "",
        height: float = -1.0,
        width: float = -1.0,
        focused: int | None = None,
        label: str = "points",
        legend: bool = False,
        tooltip: Callable[[int], str] | None = None,
        groups: list[tuple[str, np.ndarray]] | None = None,
    ) -> int | None:
        """Open a plot, draw the points, close it; returns the clicked index."""
        fit, self._fit = self._fit, False
        with line_plot(self.plot_id, x_label, y_label, height, width, fit=fit, legend=legend) as ok:
            if not ok:
                return None
            return self.items(
                x, y, colors,
                focused=focused, label=label, legend=legend, tooltip=tooltip, groups=groups,
            )

    def _points(self, label, x, y, packed, idx, legend) -> None:
        spec = implot.Spec(
            marker=implot.Marker_.circle.value,
            marker_size=self.marker_size,
            marker_line_color=vec4(OUTLINE_COLOR),
            line_weight=0.5,
        )
        if not legend:
            spec.flags = implot.ItemFlags_.no_legend
        if idx is not None:
            idx = np.asarray(idx, dtype=int)
            x, y = x[idx], y[idx]
            if packed is not None:
                packed = packed[idx]
        if packed is not None:
            spec.marker_fill_colors = np.ascontiguousarray(packed, dtype=np.uint32)
        implot.plot_scatter(label, np.ascontiguousarray(x), np.ascontiguousarray(y), spec)

    def _pick(self, x, y) -> int | None:
        limits = implot.get_plot_limits()
        pos, size = implot.get_plot_pos(), implot.get_plot_size()
        xr = limits.x.max - limits.x.min
        yr = limits.y.max - limits.y.min
        if xr <= 0 or yr <= 0 or size.x <= 0 or size.y <= 0:
            return None
        px = pos.x + (x - limits.x.min) / xr * size.x
        py = pos.y + (limits.y.max - y) / yr * size.y
        mouse = imgui.get_mouse_pos()
        return nearest_index(px, py, mouse.x, mouse.y, self.pick_radius)

    def _ring(self, x, y, color, weight, ring_id) -> None:
        spec = implot.Spec(
            marker=implot.Marker_.circle.value,
            marker_size=self.marker_size + 5.0,
            marker_fill_color=imgui.ImVec4(0, 0, 0, 0),
            marker_line_color=vec4(color),
            line_weight=float(weight),
            flags=implot.ItemFlags_.no_legend,
        )
        implot.plot_scatter(
            f"{self.plot_id}{ring_id}",
            np.array([x], dtype=np.float64),
            np.array([y], dtype=np.float64),
            spec,
        )
