"""A recording's motion correction as an implot line plot.

A recording that went through motion correction carries the shifts the
stage applied (``LazyArray.motion_correction``, a
:class:`~mbo_utilities.arrays.features.MotionCorrection`: RTMC for an AOD
scan, a registration's offsets later). :class:`MotionPlot` draws them as one
plot on the same time axis as a trace of that recording: under the trace of
the Traces tab, over the candidate trace of the curation dashboard. The host
opens implot subplots with ``link_all_x`` around both plots, so they share
one time range and their plot areas line up.
"""

from __future__ import annotations

import numpy as np
from imgui_bundle import implot

from mbo_utilities.arrays.features import MotionCorrection
from mbo_utilities.gui.imgui.lines import (
    X_AXIS_HIDDEN,
    decimate_minmax,
    drag_vline,
    line,
    line_plot,
    vlines,
)

__all__ = ["MOTION_COLORS", "MotionPlot"]

MOTION_COLORS = {"X": (0.95, 0.35, 0.35), "Y": (0.35, 0.85, 0.4), "Z": (0.4, 0.55, 1.0)}
CURSOR_COLOR = (1.0, 0.85, 0.3, 0.9)


class MotionPlot:
    """One recording's motion correction as one plot: every trace its own
    line, coloured by axis, the legend hiding any of them. ``bool(plot)``
    says whether the recording has any.

    Parameters
    ----------
    motion : MotionCorrection, optional
        ``arr.motion_correction``; None or empty draws nothing.
    points : int
        Points each trace is min/max decimated to before drawing.
    """

    def __init__(self, motion: MotionCorrection | None, points: int = 4000):
        self.motion = motion
        # decimated once: tens of thousands of points per trace every frame
        # is what made the whole window lag
        self.traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label, (t, shift) in (motion.traces if motion else {}).items():
            idx, values = decimate_minmax(shift, points)
            ts = np.asarray(t, dtype=np.float64)[idx.astype(int)]
            self.traces[label] = (
                np.ascontiguousarray(ts),
                np.ascontiguousarray(values),
            )
        # x in the host's units, converted once per unit change, not per frame
        self._scaled: tuple[float, dict[str, np.ndarray]] = (
            1.0,
            {label: t for label, (t, _v) in self.traces.items()},
        )
        self.duration_s = motion.duration_s if motion else 0.0
        self.y_label = f"{motion.source} shift ({motion.unit})" if motion else ""
        self._fit = True

    def __bool__(self) -> bool:
        return bool(self.traces)

    def refit(self) -> None:
        """Fit again on the next draw: a host that moved the plot into or
        out of subplots hands implot a new plot, with no range of its own.
        """
        self._fit = True

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
        """The traces on one plot; inside subplots ``height`` is the cell's.
        The x axis is time in the host's units, ``x_per_second`` of them per
        second (frames of a movie, ms); ``x_axis`` False hides it, for a row
        stacked over another plot's. The first draw fits y and shows the
        whole recording (``duration_s``, the traces' own extent without),
        which is also as far as the x axis can pan. ``cursor`` marks a time
        in x units; with ``cursor_id`` it is draggable, and the moved time
        and whether it is held come back.
        """
        fit, self._fit = self._fit, False
        duration = float(duration_s) if duration_s is not None else self.duration_s
        x_max = max(duration * x_per_second, 1e-3)
        if fit:
            implot.set_next_axis_to_fit(implot.ImAxis_.y1)
        with line_plot(
            plot_id,
            x_label if x_axis else "",
            self.y_label,
            height=height,
            legend=True,
            x_flags=0 if x_axis else X_AXIS_HIDDEN,
        ) as ok:
            if not ok:
                return cursor, False
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, 0.0, x_max)
            if fit:
                implot.setup_axis_limits(
                    implot.ImAxis_.x1, 0.0, x_max, implot.Cond_.always
                )
            if self._scaled[0] != x_per_second:
                self._scaled = (
                    x_per_second,
                    {label: t * x_per_second for label, (t, _v) in self.traces.items()},
                )
            for label, (_t, v) in self.traces.items():
                r, g, b = MOTION_COLORS.get(label[0], (0.8, 0.8, 0.8))
                line(
                    label, v, x=self._scaled[1][label], color=(r, g, b, 0.9), weight=1.0
                )
            if cursor is None:
                return None, False
            if cursor_id is None:
                vlines("##cursor", [float(cursor)], CURSOR_COLOR, 1.0, legend=False)
                return float(cursor), False
            return drag_vline(int(cursor_id), float(cursor), CURSOR_COLOR, 1.5)
