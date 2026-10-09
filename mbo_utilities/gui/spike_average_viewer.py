"""Spike-Triggered Average: one ROI of a run, averaged around its spikes.

Laid out like Registration QC: the averaged movies side by side in an
``NDWidget`` whose slider is the lag from the spike, and masknmf's
``TracePlot`` docked above them on the same lag (the ROI's trace, each
movie's mean, every motion correction, each with its standard error). Over
the panels, the ROI's whole trace with its spikes marked: the run's detected
events, the events a curation file accepts, or the trace's peaks over a
threshold that drags. ROI, trace and spikes are picked there.

The Studio opens it in its own process (:func:`launch_spike_average`:
``python -m mbo_utilities.gui.spike_average_viewer <run> --unit --roi``),
from the Traces tab, the MaskNMF and vnoiser tabs and the launcher;
``mbo spike-average <run>`` from a terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fastplotlib as fpl
import numpy as np
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui, implot
from masknmf.visualization.imgui import TracePlot
from masknmf.visualization.imgui.layout import HANDLE_THICKNESS, draw_edge_handle
from masknmf.visualization.imgui.theme import em, opaque_popups

from mbo_utilities import log
from mbo_utilities.analysis.spike_average import (
    MEAN_F,
    THRESHOLD,
    SpikeSource,
    threshold_peaks,
)
from mbo_utilities.gui._notebook import display_widget, in_notebook
from mbo_utilities.gui.imgui.lines import (
    decimate_minmax,
    drag_hline,
    line,
    line_plot,
    plot_style,
    vec4,
)
from mbo_utilities.gui.imgui.motion import MOTION_COLORS
from mbo_utilities.gui.launch import launch_window
from mbo_utilities.gui.run_gui import _figure_kwargs_for_here, _hold_min_size

logger = log.get("gui.spike_average_viewer")

__all__ = ["SpikeAverageViewer", "launch_spike_average", "open_spike_average_viewer"]

TITLE = "Spike-Triggered Average"
WINDOW_SIZE = (1100, 1000)
TRACES_HEIGHT = 620
SIGNAL_PLOT_EM = 9.0
MOVIE_COLORS = {
    "movie": (0.85, 0.85, 0.85),
    "registered": (0.85, 0.85, 0.85),
    "compressed": (0.35, 0.60, 0.95),
}
TRACE_COLOR = (0.95, 0.75, 0.30)
SIGNAL_COLOR = (0.85, 0.85, 0.85, 1.0)
PEAK_COLOR = (0.16, 0.62, 0.56, 1.0)
THRESHOLD_COLOR = (0.84, 0.15, 0.24, 1.0)
ERROR_COLOR = (0.95, 0.22, 0.30, 1.0)
# a line's standard error, drawn as two fainter lines under one legend entry
SEM_SHADE = 0.45
THRESHOLD_ID = 7


class SpikeAverageViewer:
    """One ROI's spike-triggered averages, re-averaged as the ROI, the trace
    and the spikes change.

    Parameters
    ----------
    source : SpikeSource
        The ROI and the run it came from.
    kind : str, optional
        The trace spikes are found on and averaged; the most processed one
        the ROI has by default.
    spikes : str, optional
        Where the spikes come from: a key of ``source.spikes`` or
        ``"threshold"``; the run's events, else a curation file's, else the
        threshold by default.
    threshold : float, optional
        Peak height that counts as a spike, in the trace's units; three
        standard deviations over its mean when None.
    distance : int
        Fewest frames between two thresholded spikes.
    before, after : int
        Frames averaged before and after each spike.
    """

    def __init__(
        self,
        source: SpikeSource,
        kind: str | None = None,
        spikes: str | None = None,
        threshold: float | None = None,
        distance: int = 5,
        before: int = 10,
        after: int = 10,
        size=WINDOW_SIZE,
    ):
        self.source = source
        self.distance = int(distance)
        self.before, self.after = int(before), int(after)
        self.subtract_mean = True
        self.status = ""
        self._held = False
        self._fit_signal = True
        self.kind = kind if kind in source.traces else next(iter(source.traces))
        self.spikes_from = (
            spikes
            if spikes in (*source.spikes, THRESHOLD)
            else next(iter(source.spikes), THRESHOLD)
        )
        self._plot_signal(threshold)
        self.spikes = self._find_spikes()
        self.average = source.average(self.spikes, self.kind, self.before, self.after)

        lags = self.average.lags
        self.lag = lags / source.fs if source.fs else lags.astype(np.float64)
        step = float(self.lag[1] - self.lag[0])
        names = list(self.average.movies)
        figure_kwargs = _figure_kwargs_for_here(size)
        figure_kwargs["canvas_kwargs"] = {
            **figure_kwargs.get("canvas_kwargs", {}),
            "title": TITLE,
        }
        self.ndw = fpl.NDWidget(
            {"time": (float(self.lag[0]), float(self.lag[-1]) + step, step)},
            shape=(1, len(names)),
            names=names,
            controller_ids=[tuple(names)],
            **figure_kwargs,
        )
        self.images = {
            name: self.ndw[name].add_nd_image(
                movie,
                ["time", "m", "n"],
                ["m", "n"],
                slider_maps={"time": self.lag},
                name=name,
            )
            for name, movie in self.average.movies.items()
        }
        for subplot in self.ndw.figure:
            subplot.tooltip.enabled = False
        self.traces = TracePlot(
            list(self.average.traces), len(lags), self.lag, autofit=False
        )
        self.window = ImguiWindow(update_call=self._draw)
        self.ndw.figure.add_imgui_window(
            self.window, location="top", size=TRACES_HEIGHT, title=TITLE
        )
        self.traces.link(self.ndw.indices)
        self._show(self.average)
        self.seek(self.before)

    @property
    def trace(self) -> np.ndarray:
        return self.source.traces[self.kind]

    @property
    def lag_index(self) -> int:
        """The lag on screen, as an index into the window."""
        return self.traces.frame

    def seek(self, index: int) -> None:
        """Show the lag at ``index`` of the window in every movie."""
        self.ndw.indices.set_dim_index("time", float(self.lag[int(index)]))

    def _plot_signal(self, threshold: float | None) -> None:
        trace = self.trace
        self.threshold = (
            float(trace.mean() + 3.0 * trace.std())
            if threshold is None
            else float(threshold)
        )
        # decimated once: every frame of a minute at a kHz is what lags a plot
        index, self.trace_y = decimate_minmax(trace)
        self.trace_t = index / self.source.fs if self.source.fs else index
        self._fit_signal = True

    def _find_spikes(self) -> np.ndarray:
        if self.spikes_from == THRESHOLD:
            return threshold_peaks(self.trace, self.threshold, self.distance)
        return self.source.spikes[self.spikes_from]

    def update(self) -> None:
        """Average again around the spikes; the averages on screen stay when
        none has a whole window, and the status line says so.
        """
        spikes = self._find_spikes()
        try:
            average = self.source.average(spikes, self.kind, self.before, self.after)
        except ValueError as error:
            self.status = str(error)
            return
        self.status = ""
        self.spikes, self.average = spikes, average
        self._show(average)

    def _show(self, average) -> None:
        for name, movie in average.movies.items():
            shown = movie - movie.mean(axis=0) if self.subtract_mean else movie
            nd = self.images[name]
            nd.data = shown
            nd.graphic.cmap = "gray"
            if self.subtract_mean:
                peak = float(np.abs(shown).max()) or 1.0
                nd.graphic.vmin, nd.graphic.vmax = -peak, peak
            else:
                nd.graphic.vmin, nd.graphic.vmax = float(movie.min()), float(movie.max())
        panels = list(average.traces)
        if tuple(panels) != self.traces.panels:
            self.traces.reset(panels, len(self.lag), self.lag)
        for panel, lines in average.traces.items():
            drawn = []
            for label, (mean, sem) in lines.items():
                color = (
                    MOVIE_COLORS.get(label)
                    or MOTION_COLORS.get(label[0])
                    or (TRACE_COLOR if label == self.kind else None)
                )
                drawn.append((label, mean, color))
                if sem is not None and np.any(sem):
                    faint = None if color is None else tuple(SEM_SHADE * v for v in color)
                    drawn.append((f"{label} sem", mean + sem, faint))
                    drawn.append((f"{label} sem", mean - sem, faint))
            self.traces.set(panel, drawn, fit=True)

    def set_roi(self, roi) -> None:
        """Average another ROI of the same unit, keeping the trace and spike
        choices it also has.
        """
        self.source = self.source.with_roi(roi)
        if self.kind not in self.source.traces:
            self.kind = next(iter(self.source.traces))
        if self.spikes_from not in (*self.source.spikes, THRESHOLD):
            self.spikes_from = next(iter(self.source.spikes), THRESHOLD)
        self._plot_signal(None)
        self.update()

    def set_kind(self, kind: str) -> None:
        """Find and average the spikes on another of the ROI's traces."""
        self.kind = kind
        self._plot_signal(None)
        self.update()

    def set_spikes_from(self, key: str) -> None:
        self.spikes_from = key
        self.update()

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.update()

    def set_subtract_mean(self, subtract: bool) -> None:
        """Show each movie about its mean over the window (the change around
        the spike, limits symmetric about zero) or as averaged.
        """
        self.subtract_mean = bool(subtract)
        self._show(self.average)

    def _draw(self) -> None:
        opaque_popups()
        average, source = self.average, self.source
        rate = f"at {source.fs:.0f} Hz" if source.fs else "(no rate: frames)"
        imgui.text(
            f"{average.label}: {average.n_spikes} spikes, {self.before} frames before "
            f"and {self.after} after {rate}"
        )
        if self.status:
            imgui.text_colored(vec4(ERROR_COLOR), self.status)
        names = source.unit.roi_names
        imgui.set_next_item_width(em(8))
        changed, k = imgui.combo("ROI##sta_roi", names.index(source.roi), names)
        if changed:
            self.set_roi(names[k])
        kinds = list(source.traces)
        imgui.same_line(0, em(1.5))
        imgui.set_next_item_width(em(8))
        changed, k = imgui.combo("trace##sta_kind", kinds.index(self.kind), kinds)
        if changed:
            self.set_kind(kinds[k])
        choices = [*source.spikes, THRESHOLD]
        imgui.same_line(0, em(1.5))
        imgui.set_next_item_width(em(10))
        changed, k = imgui.combo(
            "spikes##sta_spikes", choices.index(self.spikes_from), choices
        )
        if changed:
            self.set_spikes_from(choices[k])
        imgui.same_line(0, em(1.5))
        changed, subtract = imgui.checkbox("subtract mean", self.subtract_mean)
        if changed:
            self.set_subtract_mean(subtract)
        with plot_style():
            fit, self._fit_signal = self._fit_signal, False
            with line_plot(
                "##sta_signal",
                "time (s)" if source.fs else "frame",
                self.kind,
                height=em(SIGNAL_PLOT_EM),
                fit=fit,
                legend=False,
            ) as ok:
                if ok:
                    line("##signal", self.trace_y, x=self.trace_t, color=SIGNAL_COLOR)
                    at = self.spikes / source.fs if source.fs else self.spikes
                    marks = implot.Spec(
                        marker=implot.Marker_.circle.value,
                        marker_size=3.0,
                        marker_fill_color=vec4(PEAK_COLOR),
                        marker_line_color=vec4(PEAK_COLOR),
                    )
                    implot.plot_scatter(
                        "##spikes",
                        np.ascontiguousarray(at, np.float64),
                        np.ascontiguousarray(self.trace[self.spikes], np.float64),
                        marks,
                    )
                    if self.spikes_from == THRESHOLD:
                        threshold, held = drag_hline(
                            THRESHOLD_ID, self.threshold, THRESHOLD_COLOR
                        )
                        if held:
                            self.threshold = threshold
                        elif self._held:
                            self.update()
                        self._held = held
        moved = self.traces.draw(reserve=HANDLE_THICKNESS)
        draw_edge_handle(self.window)
        if moved is not None:
            self.traces.on_frame(moved)

    def show(self):
        """Show the figure; returns the canvas widget in a notebook."""
        output = self.ndw.show()
        _hold_min_size(self.ndw.figure)
        return output

    def close(self) -> None:
        try:
            self.ndw.figure.close()
        except AttributeError:
            # an offscreen figure has no output widget to close
            self.ndw.figure.canvas.close()


def open_spike_average_viewer(
    path,
    unit: str | None = None,
    roi=None,
    c: int | None = None,
    run: bool = True,
    **kwargs,
) -> SpikeAverageViewer:
    """Open one ROI of the run at ``path`` (:meth:`SpikeSource.open`) and show
    its spike-triggered averages (``kwargs`` go to the viewer): in the cell
    in a notebook, else a desktop window run until it is closed (``run=False``
    only shows it).
    """
    viewer = SpikeAverageViewer(SpikeSource.open(path, unit, roi, c), **kwargs)
    if in_notebook():
        display_widget(viewer.show())
    else:
        viewer.show()
        if run:
            fpl.loop.run()
    return viewer


def launch_spike_average(
    path, unit: str | None = None, roi=None, c: int | None = None
) -> tuple[int, str]:
    """Open the window on one ROI of the run at ``path`` in its own process;
    returns its pid and the log name :func:`~mbo_utilities.gui.launch.window_failure`
    takes.
    """
    args = [str(path)]
    for flag, value in (("--unit", unit), ("--roi", roi), ("--channel", c)):
        if value is not None:
            args += [flag, str(value)]
    log_name = f"spike_average_{Path(path).name}"
    return launch_window("mbo_utilities.gui.spike_average_viewer", args, log_name), log_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("path", type=Path, help="a run with results")
    parser.add_argument("--unit", default=None)
    parser.add_argument("--roi", default=None)
    parser.add_argument("--channel", type=int, default=None)
    parsed = parser.parse_args()
    open_spike_average_viewer(parsed.path, parsed.unit, parsed.roi, parsed.channel)
