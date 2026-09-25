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
    "X_AXIS_HIDDEN",
    "decimate_minmax",
    "drag_hline",
    "drag_vline",
    "hlines",
    "line",
    "line_plot",
    "packed_colors",
    "plot_style",
    "subplots",
    "vec4",
    "vlines",
]

# an upper row of stacked plots: the bottom row's x axis is the one they share
X_AXIS_HIDDEN = (
    implot.AxisFlags_.no_label
    | implot.AxisFlags_.no_tick_labels
    | implot.AxisFlags_.no_tick_marks
)


def vec4(color, alpha: float | None = None) -> imgui.ImVec4:
    """An rgb(a) tuple in 0..1 as ImVec4, alpha overridden when given."""
    if isinstance(color, imgui.ImVec4):
        return (
            color if alpha is None else imgui.ImVec4(color.x, color.y, color.z, alpha)
        )
    r, g, b = (float(v) for v in color[:3])
    a = float(color[3]) if len(color) > 3 else 1.0
    return imgui.ImVec4(r, g, b, a if alpha is None else float(alpha))


def packed_colors(colors) -> np.ndarray:
    """``(n, 3|4)`` floats in 0..1 as the uint32 array implot takes for
    per-point colours.
    """
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
    trace itself when it is short enough.
    """
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


CLEAR = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
# a plot drawn straight onto the panel behind it. implot takes its frame
# colour from imgui's, which `style_imgui_opaque` makes a blue-grey, so every
# plot sat in a blue box; grid lines go by colour rather than by axis flag so
# one scope covers a plot and the subplots around it.
PLOT_COLORS = {
    "frame_bg": CLEAR,
    "plot_bg": CLEAR,
    "plot_border": CLEAR,
    "axis_tick": imgui.ImVec4(0.80, 0.82, 0.86, 0.35),
    "axis_text": imgui.ImVec4(0.78, 0.80, 0.84, 1.00),
    "legend_bg": imgui.ImVec4(0.04, 0.05, 0.06, 0.80),
    "legend_border": CLEAR,
    "legend_text": imgui.ImVec4(0.88, 0.89, 0.92, 1.00),
    "inlay_text": imgui.ImVec4(0.80, 0.82, 0.86, 1.00),
}


@contextmanager
def plot_style(grid: bool = False):
    """Implot colours for a plot with no box of its own: transparent frame,
    background and border, faint ticks and labels, a dim legend, and no grid
    lines unless ``grid``. Wraps ``begin_plot`` or ``begin_subplots``, which
    is where the frame is drawn.
    """
    # the style stack lives on the context, and pushing onto no context is a
    # segfault, not an error: this runs before the plot that would make one
    if implot.get_current_context() is None:
        implot.create_context()
    colors = dict(PLOT_COLORS)
    colors["axis_grid"] = imgui.ImVec4(1.0, 1.0, 1.0, 0.06) if grid else CLEAR
    pushed = 0
    for name, color in colors.items():
        col = getattr(implot.Col_, name, None)
        if col is not None:
            implot.push_style_color(col, color)
            pushed += 1
    implot.push_style_var(implot.StyleVar_.plot_border_size, 0.0)
    try:
        yield
    finally:
        implot.pop_style_var()
        implot.pop_style_color(pushed)


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
    x_flags: int = 0,
):
    """``begin_plot`` / ``end_plot`` with labelled axes; yields whether the
    plot is drawn. Shift locks y while scrolling (zoom x only), alt locks x.
    ``x_flags`` are added to the x axis's (``X_AXIS_HIDDEN`` for a row over
    another plot's axis).
    """
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
            (locked if io.key_alt else none) | x_flags,
            locked if io.key_shift else none,
        )
        yield True
    finally:
        implot.end_plot()


@contextmanager
def subplots(
    plot_id: str,
    rows: int,
    cols: int,
    height: float = -1.0,
    width: float = -1.0,
    flags: int = 0,
    ratios=None,
):
    """``begin_subplots`` / ``end_subplots``; yields whether they are drawn.
    The plots opened inside fill the cells in order, their plot areas
    aligned across rows; ``SubplotFlags_.link_all_x`` shares one time axis.
    ``ratios`` is an ``implot.SubplotsRowColRatios`` the host keeps, so a
    dragged splitter stays where it was left.
    """
    if implot.get_current_context() is None:
        implot.create_context()
    if not implot.begin_subplots(
        plot_id, int(rows), int(cols), imgui.ImVec2(width, height), flags, ratios
    ):
        yield False
        return
    try:
        yield True
    finally:
        implot.end_subplots()


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


def dotted_vline(
    x: float, color, weight: float = 1.0, dash_px: float = 4.0, gap_px: float = 4.0
) -> None:
    """A dotted vertical marker (a reference line such as t = 0) across the
    open plot, drawn on its draw list; implot has no dashed line style.
    """
    pos, size = implot.get_plot_pos(), implot.get_plot_size()
    px = float(implot.plot_to_pixels(float(x), 0.0).x)
    if not pos.x <= px <= pos.x + size.x:
        return
    draw = implot.get_plot_draw_list()
    col = imgui.get_color_u32(vec4(color))
    implot.push_plot_clip_rect()
    y = pos.y
    bottom = pos.y + size.y
    while y < bottom:
        draw.add_line(
            imgui.ImVec2(px, y), imgui.ImVec2(px, min(y + dash_px, bottom)), col, weight
        )
        y += dash_px + gap_px
    implot.pop_plot_clip_rect()


def drag_hline(
    line_id: int, y: float, color, weight: float = 1.5, tag: bool = True
) -> tuple[float, bool]:
    """A horizontal line the user can drag. Returns ``(y, held)``: the
    line's position this frame and whether the mouse still holds it.
    """
    _moved, value, _clicked, _hovered, held = implot.drag_line_y(
        int(line_id), float(y), vec4(color), float(weight), 0, False, False, False
    )
    if tag:
        implot.tag_y(float(value), vec4(color))
    return float(value), bool(held)


def drag_vline(
    line_id: int, x: float, color, weight: float = 1.5, tag: bool = True
) -> tuple[float, bool]:
    """A vertical line the user can drag. Returns ``(x, held)``."""
    _moved, value, _clicked, _hovered, held = implot.drag_line_x(
        int(line_id), float(x), vec4(color), float(weight), 0, False, False, False
    )
    if tag:
        implot.tag_x(float(value), vec4(color))
    return float(value), bool(held)
