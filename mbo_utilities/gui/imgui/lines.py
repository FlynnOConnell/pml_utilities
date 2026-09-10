"""implot line plots for 1D traces.

A plot context plus the pieces a trace panel needs inside it: coloured
lines, reference lines, draggable thresholds. Shared by the event curation
panels; anything plotting traces against time can use it.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from imgui_bundle import imgui, implot

__all__ = [
    "decimate_minmax",
    "drag_hline",
    "drag_vline",
    "hlines",
    "line",
    "line_plot",
    "packed_colors",
    "vec4",
    "vlines",
]


def vec4(color, alpha: float | None = None) -> imgui.ImVec4:
    """An rgb(a) tuple in 0..1 as ImVec4, alpha overridden when given."""
    if isinstance(color, imgui.ImVec4):
        return color if alpha is None else imgui.ImVec4(color.x, color.y, color.z, alpha)
    r, g, b = (float(v) for v in color[:3])
    a = float(color[3]) if len(color) > 3 else 1.0
    return imgui.ImVec4(r, g, b, a if alpha is None else float(alpha))


def packed_colors(colors) -> np.ndarray:
    """``(n, 3|4)`` floats in 0..1 as the uint32 array implot takes for
    per-point colours."""
    rgba = np.asarray(colors, dtype=np.float64)
    if rgba.ndim == 1:
        rgba = rgba.reshape(1, -1)
    if rgba.shape[1] == 3:
        rgba = np.column_stack([rgba, np.ones(len(rgba))])
    u8 = np.clip(np.round(rgba * 255.0), 0, 255).astype(np.uint32)
    return (u8[:, 3] << 24) | (u8[:, 2] << 16) | (u8[:, 1] << 8) | u8[:, 0]


def _f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def decimate_minmax(y, n_bins: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """``(index, value)`` keeping each bin's min and max, so a long trace
    draws with a few thousand points and no peak goes missing. Returns the
    trace itself when it is short enough."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n <= 2 * n_bins:
        return np.arange(n, dtype=np.float64), y
    width = int(np.ceil(n / n_bins))
    n_bins = int(np.ceil(n / width))
    pad = n_bins * width - n
    padded = np.concatenate([y, np.full(pad, np.nan)]) if pad else y
    blocks = padded.reshape(n_bins, width)
    lo = np.nanargmin(blocks, axis=1)
    hi = np.nanargmax(blocks, axis=1)
    base = np.arange(n_bins) * width
    first, second = np.minimum(lo, hi), np.maximum(lo, hi)
    idx = np.empty(2 * n_bins, dtype=np.float64)
    idx[0::2] = base + first
    idx[1::2] = base + second
    return idx, y[idx.astype(int)]


@contextmanager
def line_plot(
    plot_id: str,
    x_label: str = "",
    y_label: str = "",
    height: float = -1.0,
    width: float = -1.0,
    fit: bool = False,
    legend: bool = True,
    flags: int = 0,
):
    """``begin_plot`` / ``end_plot`` with labelled axes; yields whether the
    plot is drawn. Shift locks y while scrolling (zoom x only), alt locks x."""
    if implot.get_current_context() is None:
        implot.create_context()
    if fit:
        implot.set_next_axes_to_fit()
    plot_flags = implot.Flags_.no_title | flags
    if not legend:
        plot_flags |= implot.Flags_.no_legend
    if not implot.begin_plot(plot_id, imgui.ImVec2(width, height), plot_flags):
        yield False
        return
    try:
        io = imgui.get_io()
        none, locked = implot.AxisFlags_.none, implot.AxisFlags_.lock
        implot.setup_axes(
            x_label,
            y_label,
            locked if io.key_alt else none,
            locked if io.key_shift else none,
        )
        yield True
    finally:
        implot.end_plot()


def _spec(color, alpha, weight, legend) -> implot.Spec:
    spec = implot.Spec(line_weight=float(weight))
    if color is not None:
        spec.line_color = vec4(color, alpha)
    if not legend:
        spec.flags = implot.ItemFlags_.no_legend
    return spec


def line(
    label: str,
    y,
    x=None,
    color=None,
    weight: float = 1.0,
    alpha: float | None = None,
    xscale: float = 1.0,
    xstart: float = 0.0,
    legend: bool = True,
) -> None:
    """One line; ``x`` given, else sample index times ``xscale`` from ``xstart``."""
    spec = _spec(color, alpha, weight, legend)
    if x is None:
        implot.plot_line(label, _f64(y), float(xscale), float(xstart), spec)
    else:
        implot.plot_line(label, _f64(x), _f64(y), spec)


def hlines(label: str, ys, color, weight: float = 1.0, legend: bool = True) -> None:
    spec = _spec(color, None, weight, legend)
    spec.flags |= implot.InfLinesFlags_.horizontal
    implot.plot_inf_lines(label, _f64(np.atleast_1d(ys)), spec)


def vlines(label: str, xs, color, weight: float = 1.0, legend: bool = True) -> None:
    spec = _spec(color, None, weight, legend)
    implot.plot_inf_lines(label, _f64(np.atleast_1d(xs)), spec)


def drag_hline(line_id: int, y: float, color, weight: float = 1.5, tag: bool = True) -> tuple[float, bool]:
    """A horizontal line the user can drag. Returns ``(y, held)``: the
    line's position this frame and whether the mouse still holds it."""
    _moved, value, _clicked, _hovered, held = implot.drag_line_y(
        int(line_id), float(y), vec4(color), float(weight), 0, False, False, False
    )
    if tag:
        implot.tag_y(float(value), vec4(color))
    return float(value), bool(held)


def drag_vline(line_id: int, x: float, color, weight: float = 1.5, tag: bool = True) -> tuple[float, bool]:
    """A vertical line the user can drag. Returns ``(x, held)``."""
    _moved, value, _clicked, _hovered, held = implot.drag_line_x(
        int(line_id), float(x), vec4(color), float(weight), 0, False, False, False
    )
    if tag:
        implot.tag_x(float(value), vec4(color))
    return float(value), bool(held)
