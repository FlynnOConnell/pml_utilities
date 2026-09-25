"""A recording's behavior as an implot plot on the trace's time axis.

A recording with a behavior log (``behavior_for(arr)``, a
:class:`~mbo_utilities.behavior.Behavior`) gets it drawn under its traces
the way ``MotionPlot`` draws motion correction. :class:`BehaviorPlot` is one
plot with three layers: the epochs as translucent bands over the full height
(the highlighted regions), the signals as lines over the upper part (the
first on the left axis, the next on the right) and the events as a raster
strip along the bottom, one lane per kind with a tick per event. Bands and
lanes sit on a third, hidden axis locked to lane units, so zooming the
signals never moves them; every layer is a legend entry the legend can hide.
``shade_into`` puts the same bands behind any other plot on the same time
axis, which is how the reward zones show behind the traces.
"""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

from mbo_utilities.behavior import Behavior
from mbo_utilities.gui.imgui.lines import (
    X_AXIS_HIDDEN,
    decimate_minmax,
    drag_vline,
    line_plot,
    vec4,
)

__all__ = ["EPOCH_COLORS", "EVENT_COLORS", "LANE_SHARE", "SIGNAL_COLORS", "BehaviorPlot"]

SIGNAL_COLORS = ((0.95, 0.75, 0.3), (0.45, 0.8, 1.0), (0.8, 0.6, 1.0))
EVENT_COLORS = {
    "lick": (0.75, 0.75, 0.75),
    "reward": (0.3, 0.9, 0.4),
    "lap": (1.0, 0.5, 0.9),
}
OTHER_EVENT_COLOR = (0.9, 0.9, 0.9)
EPOCH_COLORS = ((0.3, 0.9, 0.4), (0.4, 0.6, 1.0), (1.0, 0.7, 0.3), (0.9, 0.4, 0.9))
EPOCH_ALPHA = 0.18
CURSOR_COLOR = (1.0, 0.85, 0.3, 0.9)
# the share of the plot's height the raster strip takes when there are events
LANE_SHARE = 1.0 / 3.0
# a tick spans this much of its lane
TICK_LOW, TICK_HIGH = 0.15, 0.85
LANE_ORDER = ("lick", "reward", "lap")
STRIP_COLOR = (0.5, 0.5, 0.5, 0.5)


def nice_ticks(lo: float, hi: float, n: int = 4) -> np.ndarray:
    """About ``n`` round tick values between ``lo`` and ``hi``: the axis is
    fit wider than the data to leave room for the raster strip, and ticks
    outside the data would read as if the signal went there.
    """
    span = (hi - lo) or 1.0
    raw = span / max(n - 1, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    # a hair of slack either end, so a signal from 0.001 still gets its 0
    first = np.ceil((lo - step * 0.01) / step) * step
    ticks = np.arange(first, hi + step * 0.01, step)
    # ceil of a small negative is -0.0, which implot would label "-0"
    ticks[ticks == 0] = 0.0
    return ticks


class BehaviorPlot:
    """One session's behavior as one plot. ``bool(plot)`` says whether there
    is anything to draw.

    Parameters
    ----------
    behavior : Behavior, optional
        ``behavior_for(arr)``; None or empty draws nothing.
    points : int
        Points each signal is min/max decimated to before drawing.
    """

    def __init__(self, behavior: Behavior | None, points: int = 4000):
        self.behavior = behavior
        self.signals: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        for name, signal in (behavior.signals if behavior else {}).items():
            idx, values = decimate_minmax(signal.values, points)
            ts = np.asarray(signal.t, dtype=np.float64)[idx.astype(int)]
            self.signals[name] = (
                np.ascontiguousarray(ts),
                np.ascontiguousarray(values),
                signal.unit,
            )
        self.events = {
            name: np.asarray(t, dtype=np.float64)
            for name, t in (behavior.events if behavior else {}).items()
        }
        # the known kinds first, in a fixed order, then the rest as logged
        self.lanes = [n for n in LANE_ORDER if n in self.events] + [
            n for n in self.events if n not in LANE_ORDER
        ]
        # a tick is a segment from the lane's floor to its ceiling
        self._lane_ys = {
            name: np.tile([k + TICK_LOW, k + TICK_HIGH], len(self.events[name]))
            for k, name in enumerate(self.lanes)
        }
        # a kind's spans as one polygon: up at each start, down at each stop,
        # in lane units (0 to 1, scaled to the plot's lane total when drawn)
        self._epoch_xs: dict[str, np.ndarray] = {}
        self._epoch_ys: dict[str, np.ndarray] = {}
        for name, spans in (behavior.epochs if behavior else {}).items():
            spans = np.asarray(spans, dtype=np.float64)
            if not len(spans):
                continue
            self._epoch_xs[name] = np.repeat(spans, 2, axis=1).reshape(-1)
            self._epoch_ys[name] = np.tile([0.0, 1.0, 1.0, 0.0], len(spans))
        self.epochs = list(self._epoch_xs)
        self.duration_s = behavior.duration_s if behavior else 0.0
        self.source = behavior.source if behavior else ""
        self._fit = True

    def __bool__(self) -> bool:
        return bool(self.signals or self.events or self.epochs)

    def refit(self) -> None:
        """Fit again on the next draw: a host that moved the plot into or
        out of subplots hands implot a new plot, with no range of its own.
        """
        self._fit = True

    def lane_total(self) -> float:
        """The hidden axis's range: the lanes fill its bottom ``LANE_SHARE``."""
        return len(self.lanes) / LANE_SHARE if self.lanes else 1.0

    def _setup_lanes(self, total: float) -> None:
        """The hidden third axis, locked to ``[0, total]`` lane units."""
        implot.setup_axis(
            implot.ImAxis_.y3,
            None,
            implot.AxisFlags_.no_decorations
            | implot.AxisFlags_.no_menus
            | implot.AxisFlags_.lock,
        )
        implot.setup_axis_limits(implot.ImAxis_.y3, 0.0, total, implot.Cond_.always)

    def _shade(self, x_per_second: float, total: float) -> None:
        """Every kind of epoch as bands over the plot's full height."""
        implot.set_axis(implot.ImAxis_.y3)
        floor = np.zeros(4)
        for i, name in enumerate(self.epochs):
            spec = implot.Spec(line_weight=0.0)
            spec.fill_color = vec4(EPOCH_COLORS[i % len(EPOCH_COLORS)])
            spec.fill_alpha = EPOCH_ALPHA
            xs = self._epoch_xs[name] * x_per_second
            ys = self._epoch_ys[name] * total
            implot.plot_shaded(name, xs, np.resize(floor, len(xs)), ys, spec)
        implot.set_axis(implot.ImAxis_.y1)

    def shade_into(self, x_per_second: float) -> None:
        """Inside another plot on the same time axis, after its axes are set
        up and before its items: the epochs as bands over its full height.
        """
        if not self.epochs:
            return
        self._setup_lanes(1.0)
        self._shade(x_per_second, 1.0)

    def _ticks(self, x_per_second: float) -> None:
        """The raster strip: a lane per event kind, a tick per event, the
        kind's name at the left edge of what is on screen.
        """
        implot.set_axis(implot.ImAxis_.y3)
        for name in self.lanes:
            color = EVENT_COLORS.get(name, OTHER_EVENT_COLOR)
            spec = implot.Spec(line_weight=1.0 if name == "lick" else 2.0)
            spec.line_color = vec4(color)
            spec.flags = implot.LineFlags_.segments
            xs = np.repeat(self.events[name] * x_per_second, 2)
            implot.plot_line(name, xs, self._lane_ys[name], spec)
        limits = implot.get_plot_limits()
        left = float(limits.x.min)
        # the strip's top edge, so the lanes read as a strip under the signals
        spec = implot.Spec(line_weight=1.0)
        spec.line_color = vec4(STRIP_COLOR)
        spec.flags = implot.LineFlags_.segments | implot.ItemFlags_.no_legend
        implot.plot_line(
            "##strip",
            np.array([left, float(limits.x.max)]),
            np.array([float(len(self.lanes))] * 2),
            spec,
        )
        draw = implot.get_plot_draw_list()
        backdrop = imgui.get_color_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.65))
        implot.push_plot_clip_rect()
        for k, name in enumerate(self.lanes):
            size = imgui.calc_text_size(name)
            # a dark backdrop under the name, which otherwise sits on the ticks
            at = implot.plot_to_pixels(left, k + 0.5)
            x0 = at.x + 3.0
            draw.add_rect_filled(
                imgui.ImVec2(x0, at.y - size.y * 0.5 - 1.0),
                imgui.ImVec2(x0 + size.x + 6.0, at.y + size.y * 0.5 + 1.0),
                backdrop,
                2.0,
            )
            implot.push_style_color(
                implot.Col_.inlay_text, vec4(EVENT_COLORS.get(name, OTHER_EVENT_COLOR))
            )
            implot.plot_text(name, left, k + 0.5, imgui.ImVec2(size.x * 0.5 + 6.0, 0.0))
            implot.pop_style_color()
        implot.pop_plot_clip_rect()
        implot.set_axis(implot.ImAxis_.y1)

    def draw(
        self,
        plot_id: str,
        height: float = -1.0,
        cursor: float | None = None,
        cursor_id: int | None = None,
        duration_s: float | None = None,
        x_per_second: float = 1.0,
        x_label: str = "time (s)",
        x_axis: bool = True,
    ) -> tuple[float | None, bool]:
        """The behavior on one plot; inside subplots ``height`` is the cell's.
        The x axis is time in the host's units, ``x_per_second`` of them per
        second; ``x_axis`` False hides it, for a row stacked over another
        plot's. The first draw fits the signals over the part of the height
        above the raster strip and shows the whole recording (``duration_s``,
        the behavior's own extent without), which is also as far as the x
        axis can pan. ``cursor`` marks a time in x units; with ``cursor_id``
        it is draggable, and the moved time and whether it is held come back.
        """
        if implot.get_current_context() is None:
            implot.create_context()
        fit, self._fit = self._fit, False
        duration = float(duration_s) if duration_s is not None else self.duration_s
        x_max = max(duration * x_per_second, 1e-3)
        names = list(self.signals)
        y_label = f"{names[0]} ({self.signals[names[0]][2]})" if names else self.source
        with line_plot(
            plot_id,
            x_label if x_axis else "",
            y_label,
            height=height,
            legend=True,
            x_flags=0 if x_axis else X_AXIS_HIDDEN,
        ) as ok:
            if not ok:
                return cursor, False
            # the lane names sit bottom left, so the legend goes top right
            implot.setup_legend(implot.Location_.north_east)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, 0.0, x_max)
            if fit:
                implot.setup_axis_limits(
                    implot.ImAxis_.x1, 0.0, x_max, implot.Cond_.always
                )
            if len(names) > 1:
                implot.setup_axis(
                    implot.ImAxis_.y2,
                    f"{names[1]} ({self.signals[names[1]][2]})",
                    implot.AxisFlags_.aux_default,
                )
            # the signals fit over the height above the raster strip, their
            # ticks only where the data is
            strip = LANE_SHARE if self.lanes else 0.0
            for i, name in enumerate(names[:2]):
                axis = implot.ImAxis_.y2 if i else implot.ImAxis_.y1
                values = self.signals[name][1]
                lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
                implot.setup_axis_ticks(axis, nice_ticks(lo, hi), None, False)
                if not fit:
                    continue
                span = (hi - lo) or 1.0
                top = hi + 0.05 * span
                bottom = top - 1.1 * span / (1.0 - strip)
                implot.setup_axis_limits(axis, bottom, top, implot.Cond_.always)
            total = self.lane_total()
            self._setup_lanes(total)
            self._shade(x_per_second, total)
            for i, name in enumerate(names[:2]):
                implot.set_axis(implot.ImAxis_.y2 if i else implot.ImAxis_.y1)
                spec = implot.Spec(line_weight=1.5 if i == 0 else 1.0)
                spec.line_color = vec4(
                    SIGNAL_COLORS[i % len(SIGNAL_COLORS)], 0.95 if i == 0 else 0.75
                )
                t, values, _unit = self.signals[name]
                implot.plot_line(name, t * x_per_second, values, spec)
            implot.set_axis(implot.ImAxis_.y1)
            if self.lanes:
                self._ticks(x_per_second)
            if cursor is None:
                return None, False
            if cursor_id is None:
                spec = implot.Spec(line_weight=1.0)
                spec.line_color = vec4(CURSOR_COLOR)
                spec.flags = implot.ItemFlags_.no_legend
                implot.plot_inf_lines("##cursor", np.array([float(cursor)]), spec)
                return float(cursor), False
            return drag_vline(int(cursor_id), float(cursor), CURSOR_COLOR, 1.5)
