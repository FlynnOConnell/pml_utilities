"""Real-time motion correction traces as an implot line plot.

A scan MEScan ran with RTMC carries the X/Y/Z shifts it applied
(``MescArray.rtmc``). :class:`RtmcPlot` draws them as one plot on the same
time axis as a trace of that scan: under the F plot of the line-scan viewer's
``Line traces`` tab, over the candidate trace of the curation dashboard. The
host opens implot subplots with ``link_all_x`` around both plots, so they
share one time range and their plot areas line up.
"""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

from mbo_utilities.gui.imgui.lines import decimate_minmax, drag_vline, line, line_plot, vlines

__all__ = ["RTMC_COLORS", "RtmcPlot"]

RTMC_COLORS = {"X": (0.95, 0.35, 0.35), "Y": (0.35, 0.85, 0.4), "Z": (0.4, 0.55, 1.0)}
CURSOR_COLOR = (1.0, 0.85, 0.3, 0.9)


class RtmcPlot:
    """The RTMC traces of one scan as one plot: every trace (X/Y/Z, total
    and intercycle) its own line and checkbox, the totals shown at first.
    ``bool(plot)`` says whether the scan has any.

    Parameters
    ----------
    rtmc : dict
        ``MescArray.rtmc``: ``{"X total": {"t": s, "um": µm}, ...}``.
    points : int
        Points each trace is min/max decimated to before drawing.
    """

    def __init__(self, rtmc: dict[str, dict], points: int = 4000):
        # decimated once: tens of thousands of points per trace every frame
        # is what made the whole window lag
        self.traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label, tr in rtmc.items():
            idx, values = decimate_minmax(tr["um"], points)
            t = np.asarray(tr["t"], dtype=np.float64)[idx.astype(int)]
            self.traces[label] = (np.ascontiguousarray(t), np.ascontiguousarray(values))
        self.show = {label: " total" in label for label in self.traces}
        self.duration_s = max((float(t[-1]) for t, _ in self.traces.values() if len(t)), default=0.0)
        self._fit = True

    def __bool__(self) -> bool:
        return bool(self.traces)

    @property
    def shown(self) -> bool:
        """Whether any trace is ticked, so the plot has something to draw."""
        return any(self.show.values())

    def refit(self) -> None:
        """Fit again on the next draw: a host that moved the plot into or
        out of subplots hands implot a new plot, with no range of its own."""
        self._fit = True

    def draw_checkboxes(self, key: str) -> None:
        """``RTMC (um):`` and one checkbox per trace, on the current line."""
        imgui.text_disabled("RTMC (um):")
        for label in self.traces:
            imgui.same_line(0, 8)
            _changed, self.show[label] = imgui.checkbox(f"{label}##{key}", self.show[label])
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"real-time motion correction applied while the scan ran: {label} "
                    "shift in um, on the same time axis as the trace"
                )

    def draw(
        self,
        plot_id: str,
        height: float = -1.0,
        cursor: float | None = None,
        cursor_id: int | None = None,
        duration_s: float | None = None,
    ) -> tuple[float | None, bool]:
        """The ticked traces on one plot; inside subplots ``height`` is the
        cell's. The first draw fits y and shows the whole recording
        (``duration_s``, the traces' own extent without), which is also as
        far as the x axis can pan. ``cursor`` marks a time; with
        ``cursor_id`` it is draggable, and the moved time and whether it is
        held come back."""
        fit, self._fit = self._fit, False
        duration = float(duration_s) if duration_s is not None else self.duration_s
        if fit:
            implot.set_next_axis_to_fit(implot.ImAxis_.y1)
        with line_plot(plot_id, "time (s)", "RTMC (um)", height=height, legend=True) as ok:
            if not ok:
                return cursor, False
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, 0.0, max(duration, 1e-3))
            if fit:
                implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, max(duration, 1e-3), implot.Cond_.always)
            for label, (t, v) in self.traces.items():
                if not self.show[label]:
                    continue
                r, g, b = RTMC_COLORS[label[0]]
                alpha, weight = (0.9, 1.0) if " total" in label else (0.55, 0.8)
                line(label, v, x=t, color=(r, g, b, alpha), weight=weight)
            if cursor is None:
                return None, False
            if cursor_id is None:
                vlines("##cursor", [float(cursor)], CURSOR_COLOR, 1.0, legend=False)
                return float(cursor), False
            return drag_vline(int(cursor_id), float(cursor), CURSOR_COLOR, 1.5)
