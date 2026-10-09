"""Spike-triggered average window for an AOD voltage recording.

One fastplotlib figure over a :class:`~mbo_utilities.analysis.spike_average.MasknmfUnit`.
The spikes are the peaks of one of masknmf's demixed traces over a threshold:
the trace is plotted whole, its peaks marked, and the threshold is a line
that drags (three standard deviations over the trace's mean to start);
another signal is a pick away. Around those spikes, the registered and the
compressed movie are averaged, side by side and playing through the window,
over the averaged traces in linked plots (the patch's mean fluorescence, the
AOD's real-time motion correction, masknmf's shifts), each with its standard
error as a band. A spike that is activity shows in the movies and the
fluorescence and leaves the motion plots flat inside their bands.

Usage:
    mbo spike-average <masknmf run folder>
    mbo spike-average scan.mesc --unit MUnit_14 --masknmf run/results.hdf5
    vis = open_spike_average_viewer(run_folder)   in a notebook
"""

from __future__ import annotations

from pathlib import Path

import fastplotlib as fpl
import numpy as np
from imgui_bundle import imgui, implot

from mbo_utilities.analysis.spike_average import (
    MasknmfUnit,
    accepted_events,
    threshold_peaks,
)
from mbo_utilities.gui._edge_window import EdgeWindow
from mbo_utilities.gui._notebook import display_widget, in_notebook
from mbo_utilities.gui._theme import em
from mbo_utilities.gui.imgui.lines import (
    X_AXIS_HIDDEN,
    decimate_minmax,
    drag_hline,
    drag_vline,
    line,
    line_plot,
    plot_style,
    subplots,
    vec4,
    vlines,
)
from mbo_utilities.gui.imgui.motion import MOTION_COLORS
from mbo_utilities.gui.imgui.movie_player import MoviePlayer
from mbo_utilities.gui.run_gui import _figure_kwargs_for_here

__all__ = ["SpikeAverageVis", "SpikeAverageWindow", "open_spike_average_viewer"]

WINDOW_SIZE = (1100, 1000)
TRACES_HEIGHT = 680
SIGNAL_PLOT_EM = 11.0
LINE_COLORS = {"registered": (0.85, 0.85, 0.85), "compressed": (0.35, 0.60, 0.95)}
SIGNAL_COLOR = (0.85, 0.85, 0.85, 1.0)
PEAK_COLOR = (0.16, 0.62, 0.56, 1.0)
THRESHOLD_COLOR = (0.84, 0.15, 0.24, 1.0)
ERROR_COLOR = (0.95, 0.22, 0.30, 1.0)
SPIKE_COLOR = (0.62, 0.62, 0.65, 0.9)
CURSOR_COLOR = (1.0, 0.85, 0.3, 0.9)
SEM_ALPHA = 0.25
THRESHOLD_ID = 7
CURSOR_ID = 11


class SpikeAverageWindow(EdgeWindow):
    """The figure's bottom edge window: the signal the spikes are found on,
    the transport and the averaged traces.
    """

    def __init__(self, figure, vis: SpikeAverageVis):
        self.vis = vis
        super().__init__(
            figure,
            size=TRACES_HEIGHT,
            location="bottom",
            title="Spike-triggered average",
        )

    def update(self) -> None:
        self.vis.draw_traces()


class SpikeAverageVis:
    """A unit's spike-triggered averages on a fastplotlib figure.

    ``show()`` returns the canvas, which a notebook cell displays; the same
    ``show`` / ``close`` contract as ``DataVis``.

    Parameters
    ----------
    unit : MasknmfUnit
        The recording and its masknmf run.
    signal : int
        Which demixed trace the spikes are found on, 0-based.
    threshold : float, optional
        Peak height that counts as a spike, in the trace's units; three
        standard deviations over its mean when None.
    distance : int
        Fewest frames between two spikes.
    spikes : array, optional
        Spikes found elsewhere (curated), as frames from the unit's
        ``first_frame``: shown on the signal in place of its thresholded
        peaks, with no threshold to drag.
    before, after : int
        Frames averaged before and after each spike.
    size : (int, int), optional
        Canvas size in pixels.
    canvas : optional
        Passed to the figure (``"offscreen"`` for tests).
    """

    def __init__(
        self,
        unit: MasknmfUnit,
        signal: int = 0,
        threshold: float | None = None,
        distance: int = 5,
        spikes=None,
        before: int = 10,
        after: int = 10,
        size=WINDOW_SIZE,
        canvas=None,
    ):
        kwargs = _figure_kwargs_for_here(size)
        if canvas is not None:
            kwargs["canvas"] = canvas
            kwargs.pop("canvas_kwargs", None)
        self.unit = unit
        self.distance = int(distance)
        self.before, self.after = int(before), int(after)
        self.curated = spikes is not None
        self.spikes = np.asarray(spikes) if self.curated else None
        self.status = ""
        self._plot_signal(signal, threshold)
        if not self.curated:
            self.spikes = threshold_peaks(self.trace, self.threshold, self.distance)
        self.average = unit.average(self.spikes, self.before, self.after)
        self.lag_ms = self.average.lags * 1000.0 / unit.fs
        names = list(self.average.movies)
        self.figure = fpl.Figure(shape=(1, len(names)), names=[names], **kwargs)
        self.player = MoviePlayer(self.average.movies[names[0]], fps=5.0)
        self.player.jump_to(self.before)
        self.graphics = {
            name: self.figure[name].add_image(
                movie[self.player.t], name=name, cmap="gray"
            )
            for name, movie in self.average.movies.items()
        }
        self._fit = True
        self._held = False
        self.set_subtract_mean(True)
        self.window = SpikeAverageWindow(self.figure, self)
        self._output = None
        self._shown = False

    def _plot_signal(self, signal: int, threshold: float | None) -> None:
        self.signal = int(signal)
        self.trace = self.unit.signals[:, self.signal]
        self.threshold = (
            float(self.trace.mean() + 3.0 * self.trace.std())
            if threshold is None
            else float(threshold)
        )
        # decimated once: every frame of a minute at a kHz is what lags a plot
        index, self.trace_y = decimate_minmax(self.trace)
        self.trace_t = index / self.unit.fs
        self._fit_signal = True

    def set_signal(self, signal: int, threshold: float | None = None) -> None:
        """Find the spikes on another demixed trace, at ``threshold`` or its
        own starting one.
        """
        self._plot_signal(signal, threshold)
        self.detect()

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.detect()

    def detect(self) -> None:
        """Average again around the signal's peaks over the threshold; the
        averages on screen stay when none has a whole window, and the status
        line says so.
        """
        if self.curated:
            return
        spikes = threshold_peaks(self.trace, self.threshold, self.distance)
        try:
            average = self.unit.average(spikes, self.before, self.after)
        except ValueError as error:
            self.status = str(error)
            return
        self.status = ""
        self.spikes, self.average = spikes, average
        self._fit = True
        self.set_subtract_mean(self.subtract_mean)

    def set_subtract_mean(self, subtract: bool) -> None:
        """Show each movie about its mean over the window (the change around
        the spike, limits symmetric about zero) or as averaged.
        """
        self.subtract_mean = bool(subtract)
        self.shown = {}
        for name, movie in self.average.movies.items():
            if self.subtract_mean:
                self.shown[name] = movie - movie.mean(axis=0)
                peak = float(np.abs(self.shown[name]).max())
                limits = (-peak, peak)
            else:
                self.shown[name] = movie
                limits = (float(movie.min()), float(movie.max()))
            self.graphics[name].data = self.shown[name][self.player.t]
            self.graphics[name].vmin, self.graphics[name].vmax = limits

    def seek(self, index: int) -> None:
        """Show the frame at ``index`` of the window in every movie."""
        self.player.jump_to(index)
        for name, graphic in self.graphics.items():
            graphic.data = self.shown[name][self.player.t]

    def draw_traces(self) -> None:
        """The edge window's body: what was averaged, the controls, the
        signal with its spikes and threshold, then one plot per kind of
        averaged trace on a shared lag axis, the frame on screen as a cursor
        that drags.
        """
        average = self.average
        imgui.text(
            f"{average.label}: {average.n_spikes} spikes of signal {self.signal}, "
            f"{self.before} frames before and {self.after} after at {average.fs:.0f} Hz"
        )
        if self.status:
            imgui.text_colored(vec4(ERROR_COLOR), self.status)
        if self.player.draw(slider_width=em(18)):
            self.seek(self.player.t)
        imgui.same_line(0, em(1.5))
        changed, subtract = imgui.checkbox("subtract mean", self.subtract_mean)
        if changed:
            self.set_subtract_mean(subtract)
        if not self.curated:
            imgui.same_line(0, em(1.5))
            imgui.set_next_item_width(em(8))
            changed, signal = imgui.combo(
                "##spike_signal",
                self.signal,
                [f"signal {k}" for k in range(self.unit.signals.shape[1])],
            )
            if changed:
                self.set_signal(signal)
        with plot_style():
            fit, self._fit_signal = self._fit_signal, False
            with line_plot(
                "##spike_signal_plot",
                "time (s)",
                f"signal {self.signal}",
                height=em(SIGNAL_PLOT_EM),
                fit=fit,
                legend=False,
            ) as ok:
                if ok:
                    line("##signal", self.trace_y, x=self.trace_t, color=SIGNAL_COLOR)
                    marks = implot.Spec(
                        marker=implot.Marker_.circle.value,
                        marker_size=3.0,
                        marker_fill_color=vec4(PEAK_COLOR),
                        marker_line_color=vec4(PEAK_COLOR),
                    )
                    implot.plot_scatter(
                        "##spikes",
                        np.ascontiguousarray(self.spikes / self.unit.fs, np.float64),
                        np.ascontiguousarray(self.trace[self.spikes], np.float64),
                        marks,
                    )
                    if not self.curated:
                        threshold, held = drag_hline(
                            THRESHOLD_ID, self.threshold, THRESHOLD_COLOR
                        )
                        if held:
                            self.threshold = threshold
                        elif self._held:
                            self.detect()
                        self._held = held
            fit, self._fit = self._fit, False
            cursor = float(self.lag_ms[self.player.t])
            rows = list(average.traces.items())
            with subplots(
                "##spike_average", len(rows), 1, flags=implot.SubplotFlags_.link_all_x
            ) as drawn:
                if not drawn:
                    return
                for i, (y_label, lines) in enumerate(rows):
                    last = i == len(rows) - 1
                    with line_plot(
                        f"##spike_average_{i}",
                        "time from spike (ms)" if last else "",
                        y_label,
                        fit=fit,
                        x_flags=0 if last else X_AXIS_HIDDEN,
                    ) as ok:
                        if not ok:
                            continue
                        vlines("##spike", [0.0], SPIKE_COLOR, 1.0, legend=False)
                        for label, (mean, sem) in lines.items():
                            color = LINE_COLORS.get(
                                label, MOTION_COLORS.get(label[0], (0.8, 0.8, 0.8))
                            )
                            band = implot.Spec(line_weight=0.0)
                            band.fill_color = vec4(color)
                            band.fill_alpha = SEM_ALPHA
                            band.flags = implot.ItemFlags_.no_legend
                            implot.plot_shaded(
                                f"##{label}_sem",
                                np.ascontiguousarray(self.lag_ms, dtype=np.float64),
                                np.ascontiguousarray(mean - sem, dtype=np.float64),
                                np.ascontiguousarray(mean + sem, dtype=np.float64),
                                band,
                            )
                            line(label, mean, x=self.lag_ms, color=color, weight=1.5)
                        moved, held = drag_vline(
                            CURSOR_ID + i, cursor, CURSOR_COLOR, tag=False
                        )
                        if held:
                            index = int(np.abs(self.lag_ms - moved).argmin())
                            if index != self.player.t:
                                self.seek(index)

    def show(self):
        """Show the figure; returns the canvas widget in a notebook."""
        if not self._shown:
            self._output = self.figure.show()
            self._shown = True
        return self._output

    def close(self) -> None:
        try:
            self.figure.close()
        except AttributeError:
            # an offscreen figure has no output widget to close
            self.figure.canvas.close()


def open_spike_average_viewer(
    path,
    unit: str | None = None,
    masknmf_path=None,
    label_path=None,
    recording: str | None = None,
    signal: int = 0,
    threshold: float | None = None,
    distance: int = 5,
    before: int = 10,
    after: int = 10,
    run: bool = True,
    **kwargs,
) -> SpikeAverageVis:
    """Read a masknmf run and show its spike-triggered averages:
    displayed in the cell in a notebook, else a desktop window run until it
    is closed (``run=False`` only shows it).

    ``path`` is a run folder the MaskNMF pipeline wrote (or its
    ``results.hdf5``), read by :meth:`MasknmfUnit.from_run`; or a ``.mesc``
    with its ``unit`` and the ``masknmf_path`` results hdf5 of a run on it,
    read by :meth:`MasknmfUnit.open` (``kwargs`` go to it).

    The spikes are demixed trace ``signal``'s peaks over ``threshold``.
    With ``label_path``, a curation file, they are the events it accepts
    for ``recording`` (the unit's own id when None) instead.
    """
    path = Path(path)
    if masknmf_path is None:
        source = MasknmfUnit.from_run(path.parent if path.is_file() else path)
    else:
        source = MasknmfUnit.open(path, unit, masknmf_path, **kwargs)
    spikes = None
    if label_path is not None:
        events = accepted_events(label_path)
        if recording is None:
            recording = source.recording
        if recording not in events:
            raise KeyError(
                f"{label_path} accepts no events of {recording}; it holds {sorted(events)}"
            )
        spikes = events[recording] - source.first_frame
        spikes = spikes[(spikes >= 0) & (spikes < source.signals.shape[0])]
    vis = SpikeAverageVis(source, signal, threshold, distance, spikes, before, after)
    if in_notebook():
        display_widget(vis.show())
    else:
        vis.show()
        if run:
            fpl.loop.run()
    return vis
