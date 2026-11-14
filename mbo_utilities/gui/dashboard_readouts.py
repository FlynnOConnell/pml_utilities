"""The dashboard's read-outs: the plotted rows' spectrum, their correlation
with each other and the motion correction, and their average around the
trigger row's events.

They measure what the rows show, so a filter set in Processing is the band of
the correlation, and they run over the active window, else the recording past
its opening transient.
:class:`ReadoutsMixin` holds the drawing, in the manner of ``gui/roi``: it has
no state of its own, every attribute is :class:`~.dashboard_app.DashboardApp`'s.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from imgui_bundle import imgui, implot

from mbo_utilities.analysis.filters import describe
from mbo_utilities.dashboard import Readouts, compute_readouts, export_readouts
from mbo_utilities.gui.imgui.lines import line, vec4, vlines

__all__ = ["PAGES", "ReadoutsMixin", "run_readouts", "write_readouts"]

PAGES = ("Traces", "Spectrum", "Correlation", "Triggered average")
SCALE_WIDTH = 60
EVENT_RGB = (0.95, 0.85, 0.35)
SEM_ALPHA = 0.15


def run_readouts(
    labels, results, fs, trigger, span, settings, motion, motion_filter
) -> Readouts:
    """Stack the rows' samples on a worker thread and measure them."""
    traces = np.stack([r.shown for r in results])
    return compute_readouts(
        labels, results[0].t, traces, fs, trigger, span, settings, motion, motion_filter
    )


def write_readouts(readouts: Readouts, stem) -> str:
    written = export_readouts(readouts, stem)
    return f"wrote {len(written)} files to {written[0].parent}"


class ReadoutsMixin:
    """The read-out pages and their settings."""

    def _readout_span(self) -> tuple[str, float, float]:
        """``(name, start, stop)`` read: the active window, else the recording
        from ``skip_s``.
        """
        if 0 <= self.active < len(self.windows):
            w = self.windows[self.active]
            return w["name"], w["start"], w["stop"]
        start = self.duration[0] + self.readout.skip_s
        return f"the recording from {start:g} s", start, self.duration[1]

    def _trigger_row(self, picked: list):
        """The trigger row, the first plotted one when it is not plotted."""
        return next((r for r in picked if r.key == self.trigger), picked[0])

    def _schedule_readouts(self) -> None:
        """Measure the plotted rows when a read-out wants them and they changed."""
        wanted = self.page != "Traces" or self.show_events
        if not wanted or self._reading is not None or not self._ready():
            return
        picked = self._picked_rows()
        _, *span = self._readout_span()
        trigger = self._trigger_row(picked)
        key = (
            self._view_key,
            tuple(r.key for r in picked),
            tuple(span),
            trigger.key,
            repr(self.readout),
            self.motion is None,
        )
        if key == self._readouts_key:
            return
        self._readouts_key = key
        self.readout_error = ""
        self._reading = self._worker.submit(
            run_readouts,
            [r.name for r in picked],
            [self._results[r.key] for r in picked],
            float(picked[0].fs or 1.0),
            picked.index(trigger),
            tuple(span),
            copy.deepcopy(self.readout),
            self.motion,
            None
            if self.view.filter.kind == "none"
            else copy.deepcopy(self.view.filter),
        )
        self._reading_rows = picked

    def _poll_readouts(self) -> None:
        if self._reading is None or not self._reading.done():
            return
        future, self._reading = self._reading, None
        if future.exception() is not None:
            self.readout_error = str(future.exception())
            return
        self.readouts, self.readout_rows = future.result(), self._reading_rows

    def _draw_readout_page(self, draw) -> None:
        """A read-out page, or why it has nothing to show yet."""
        if self.readout_error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.45, 0.45, 1.0), self.readout_error)
        elif self.readouts is None:
            imgui.text_disabled("measuring...")
        else:
            draw(self.readouts)

    def _plot_size(self) -> imgui.ImVec2:
        """What is left of the page above one caption line."""
        line_height = imgui.get_text_line_height_with_spacing()
        return imgui.ImVec2(-1, imgui.get_content_region_avail().y - line_height)

    def _span_text(self, r: Readouts) -> str:
        return f"{r.span[0]:.3g} to {r.span[1]:.3g} s"

    def _draw_spectrum(self, r: Readouts) -> None:
        flags = implot.Flags_.no_title
        if implot.begin_plot("##spectrum", self._plot_size(), flags):
            fit = implot.AxisFlags_.auto_fit
            unit = self._y_unit()
            implot.setup_axes("frequency (Hz)", f"power ({unit})^2 / Hz", fit, fit)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
            implot.setup_legend(implot.Location_.north_east)
            for k, row in enumerate(self.readout_rows):
                ## NOTE: the DC and Nyquist bins hold half a bin's power, and stretch the log axis
                line(
                    f"{row.name}: to {r.floors[k]:.0f} Hz",
                    r.power[k, 1:-1],
                    x=r.frequency[1:-1],
                    color=self._row_rgb(row),
                )
            implot.end_plot()
        imgui.text_disabled(
            f"{self._span_text(r)}, {self.readout.resolution_hz:g} Hz bins. "
            "The legend gives the Hz each row reaches its noise floor at."
        )

    def _draw_correlation(self, r: Readouts) -> None:
        n = len(r.correlation_labels)
        size = self._plot_size()
        size.x = imgui.get_content_region_avail().x - SCALE_WIDTH
        flags = implot.Flags_.equal | implot.Flags_.no_title | implot.Flags_.no_legend
        implot.push_colormap(implot.Colormap_.rd_bu)
        if implot.begin_plot("##correlation", size, flags):
            lock = implot.AxisFlags_.lock
            implot.setup_axes("", "", lock, lock)
            implot.setup_axes_limits(0, n, 0, n, implot.Cond_.always)
            ticks = (np.arange(n) + 0.5).tolist()
            short = [name.rsplit(" ", 1)[-1] for name in r.correlation_labels]
            implot.setup_axis_ticks(implot.ImAxis_.x1, ticks, short)
            implot.setup_axis_ticks(
                implot.ImAxis_.y1, ticks, r.correlation_labels[::-1]
            )
            implot.plot_heatmap(
                "##r",
                np.ascontiguousarray(r.correlation, np.float64),
                -1.0,
                1.0,
                "%.2f",
                implot.Point(0, 0),
                implot.Point(n, n),
            )
            implot.end_plot()
        imgui.same_line()
        implot.colormap_scale("r", -1.0, 1.0, imgui.ImVec2(SCALE_WIDTH - 8, size.y))
        implot.pop_colormap()
        f = self.view.filter
        band = "unfiltered" if f.kind == "none" else describe(f)
        imgui.text_disabled(f"{self._span_text(r)}, {band}")

    def _draw_triggered(self, r: Readouts) -> None:
        n = len(r.event_times)
        if n == 0:
            imgui.text_disabled("No events past the threshold on the trigger row.")
            return
        flags = implot.Flags_.no_title
        if implot.begin_plot("##triggered", self._plot_size(), flags):
            fit = implot.AxisFlags_.auto_fit
            implot.setup_axes("time from event (ms)", self._y_unit(), fit, fit)
            for k, row in enumerate(self.readout_rows):
                color = self._row_rgb(row)
                name = f"{row.name} (trigger)" if k == r.trigger else row.name
                implot.plot_shaded(
                    f"##sem{k}",
                    np.ascontiguousarray(r.lags_ms),
                    np.ascontiguousarray(r.mean[k] - r.sem[k]),
                    np.ascontiguousarray(r.mean[k] + r.sem[k]),
                    implot.Spec(
                        fill_color=vec4(color, SEM_ALPHA),
                        flags=implot.ItemFlags_.no_legend,
                    ),
                )
                line(name, r.mean[k], x=r.lags_ms, color=color)
            implot.end_plot()
        rate = n / max(r.span[1] - r.span[0], 1e-12)
        imgui.text_disabled(
            f"{n} events, {rate:.2f} Hz, mean and SEM over {self._span_text(r)}"
        )
        imgui.set_item_tooltip(
            "The trigger row's own average is picked on its noise, so it runs high."
        )

    def _draw_event_marks(self, row, x0: float, x1: float) -> None:
        """The trigger row's events in the time on screen."""
        r = self.readouts
        if not self.show_events or r is None or self.readout_rows is None:
            return
        if row.key != self.readout_rows[r.trigger].key:
            return
        first, last = np.searchsorted(r.event_times, (x0, x1))
        if last > first:
            vlines("##events", r.event_times[first:last], EVENT_RGB, legend=False)

    def _draw_readout_settings(self) -> None:
        s = self.readout
        picked = self._picked_rows()
        name, *_ = self._readout_span()
        imgui.text_disabled(f"over {name}")
        if picked:
            trigger = self._trigger_row(picked)
            changed, i = imgui.combo(
                "trigger", picked.index(trigger), [r.name for r in picked]
            )
            if changed:
                self.trigger = picked[i].key
        changed, i = imgui.combo(
            "events are", 0 if s.polarity == 1 else 1, ["peaks", "troughs"]
        )
        if changed:
            s.polarity = 1 if i == 0 else -1
        for label, name, low in (
            ("threshold (SD)", "threshold_sd", 0.1),
            ("gap (ms)", "gap_ms", 0.0),
            ("before (ms)", "before_ms", 0.1),
            ("after (ms)", "after_ms", 0.1),
            ("bins (Hz)", "resolution_hz", 0.01),
            ("skip start (s)", "skip_s", 0.0),
        ):
            _, value = imgui.input_float(label, getattr(s, name), 0.0, 0.0, "%g")
            setattr(s, name, max(value, low))
        imgui.set_item_tooltip(
            "Left out unless a window is active: a scan opens on a transient on every row."
        )
        _, self.show_events = imgui.checkbox(
            "mark events on the trace", self.show_events
        )
        imgui.begin_disabled(self.readouts is None or self._exporting is not None)
        if imgui.button("Save read-outs") and self.readouts is not None:
            munit = str(self.unit).rsplit("/", 1)[-1]
            stem = Path(self.folder) / f"{self.path.stem}_{munit}_{self.view.kind}"
            self._exporting = self._worker.submit(write_readouts, self.readouts, stem)
        imgui.end_disabled()
        imgui.set_item_tooltip("Spectrum, correlation, average and events as CSVs.")
