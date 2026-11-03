"""The slow / fast method's panels of the curation window.

:class:`SlowFastPanelsMixin` is drawn by
:class:`~mbo_utilities.gui.event_curation.EventCurationWidget` while its
method is ``slow_fast``: the trace with its slow events and fast peaks (A),
the detection card (A1 to A3), the focused slow event, the focused peak and
every peak's prominence against its width (B to D), and the Decision tab's
event, detection and files sections. The rules are
:mod:`mbo_utilities.analysis.slow_fast`'s, the labels
:class:`~mbo_utilities.annotation.slow_fast.SlowFastSession`'s; a mixin has
no state of its own, every attribute is the widget's.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from imgui_bundle import imgui

from mbo_utilities.analysis.slow_fast import SlowFastSettings
from mbo_utilities.gui import _theme as theme
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui._theme import card, em, section
from mbo_utilities.gui.imgui.lines import (
    bands,
    dotted_vline,
    drag_hline,
    line,
    line_plot,
)

__all__ = ["SlowFastPanelsMixin", "settings_tag"]

FAST_TRACE_COLOR = (0.85, 0.85, 0.85, 1.0)
SLOW_TRACE_COLOR = (0.35, 0.60, 0.95, 1.0)
SLOW_BAND_COLOR = (0.35, 0.60, 0.95, 1.0)
SLOW_BAND_ALPHA = 0.16
EDGE_COLOR = (0.16, 0.62, 0.56, 1.0)
PEAK_SD_COLOR = (0.84, 0.15, 0.24, 1.0)
PROMINENCE_COLOR = (0.95, 0.68, 0.25, 1.0)
FINE_COLOR = (0.30, 0.47, 0.66, 1.0)
RAW_COLOR = (0.62, 0.62, 0.65, 0.5)
ZERO_COLOR = (0.62, 0.62, 0.65, 0.9)
PEAK_WINDOW_MS = 25.0
# the card's three sliders: (key, setting, short name, low, high, colour, tooltip)
CARD = (
    (
        "sf_slow",
        "slow_sigma_ms",
        "slow ms",
        5.0,
        30.0,
        SLOW_TRACE_COLOR,
        "slow trace Gaussian sigma, ms: 12.7 ms is about a 10 Hz low-pass",
    ),
    (
        "sf_edge",
        "edge_sd",
        "edge SD",
        0.5,
        5.0,
        EDGE_COLOR,
        "slow event edges: where the slow trace crosses this many robust SD (the teal line)",
    ),
    (
        "sf_peak",
        "slow_sd",
        "peak SD",
        1.0,
        10.0,
        PEAK_SD_COLOR,
        "a slow event is kept when the slow trace reaches this many robust SD (the red line)",
    ),
)
# the Detection section's advanced settings: (setting, label, format, tooltip)
ADVANCED = (
    (
        "merge_ms",
        "merge ms",
        "%.0f",
        "slow stretches closer than this merge; 0 never merges",
    ),
    (
        "k_prom",
        "prominence",
        "%.1f",
        "a fast peak's prominence, in noise units (the amber line on D)",
    ),
    ("fast_sigma_ms", "fast ms", "%.1f", "fast trace Gaussian sigma, ms"),
    (
        "base_sigma_ms",
        "base ms",
        "%.1f",
        "Gaussian before the moving-median baseline, ms",
    ),
    ("base_window_s", "base window s", "%.1f", "moving-median baseline window, s"),
    (
        "skip_s",
        "skip s",
        "%.1f",
        "slow events touching the first or last this many seconds are dropped",
    ),
    (
        "fine_sigma_ms",
        "fine ms",
        "%.1f",
        "the light Gaussian of amplitude_fine (panel C), ms; 0 is the raw trace",
    ),
)


def settings_tag(session) -> str:
    """A key that changes whenever the detection does."""
    return repr(sorted(session.settings.to_dict().items()))


class SlowFastPanelsMixin:
    """The slow / fast method's panels on the curation widget."""

    def _slow_fast_items(self, session) -> None:
        """Inside panel A's open plot: the slow events as bands, the fast and
        slow traces, the edge and peak lines (drag either; applied on
        release), the candidates.
        """
        events, s = session.events, session.settings
        bounds = session.slow_bounds()
        bands(bounds[:, 0], bounds[:, 1], SLOW_BAND_COLOR, SLOW_BAND_ALPHA)
        tag = settings_tag(session)
        t = session.t
        t_fast, fast = self._cached_trace(f"fast {tag}", t, events.fast_trace)
        line(f"fast ({s.fast_sigma_ms:g} ms)", fast, x=t_fast, color=FAST_TRACE_COLOR)
        t_slow, slow = self._cached_trace(f"slow {tag}", t, events.slow_trace)
        line(
            f"slow ({s.slow_sigma_ms:g} ms)",
            slow,
            x=t_slow,
            color=SLOW_TRACE_COLOR,
            weight=1.6,
        )
        sd = events.sd
        for line_id, name, color in (
            (1, "edge_sd", EDGE_COLOR),
            (2, "slow_sd", PEAK_SD_COLOR),
        ):
            self._sd_line(session, line_id, name, color, sd)
        self._draw_candidate_points(session, session.times_s, session.peak_values)

    def _sd_line(self, session, line_id: int, name: str, color, sd: float) -> None:
        """A draggable line at ``setting`` x ``sd``; the setting changes on release."""
        if sd <= 0:
            return
        shown = self._line_drags.get(name, getattr(session.settings, name) * sd)
        value, held = drag_hline(line_id, shown, color)
        if held:
            self._line_drags[name] = value
        elif name in self._line_drags:
            session.set_settings(**{name: max(self._line_drags.pop(name) / sd, 0.1)})
            self._panel_keys.clear()

    def _draw_slow_fast_card(self, session, width, height) -> None:
        """A1 to A3: slow sigma, edge SD and peak SD as slider columns."""
        with card("##curation_sliders", "A1 - A3", height, width):
            set_tooltip(
                "Sliders apply on release and detect again. A1 slow ms: the slow "
                "trace's smoothing. A2 edge SD: where a slow event starts and ends. "
                "A3 peak SD: how high it must reach. Fast peaks are counted only "
                "inside slow events. Labels stay on every peak found again. Saved "
                "per recording; Detection in the Decision tab has the rest.",
            )
            counts = (
                f"{session.events.n_slow} slow",
                f"{session.events.n_slow} slow",
                f"{session.n} fast",
            )
            for i, (key, name, short, lo, hi, color, tip) in enumerate(CARD):
                if i:
                    imgui.same_line(0, em(0.4))
                self._slider_column(
                    key,
                    short,
                    getattr(session.settings, name),
                    lo,
                    hi,
                    partial(self._apply_setting, session, name),
                    color,
                    counts[i],
                    tip,
                )

    def _apply_setting(self, session, name: str, value: float) -> None:
        session.set_settings(**{name: float(value)})
        self._panel_keys.clear()

    def _slow_fast_panels(self, session) -> tuple:
        return (
            ("B. Slow event", self._draw_slow_event),
            (f"C. Fast peak (±{PEAK_WINDOW_MS:g} ms)", self._draw_fast_peak),
            ("D. Prominence vs width", self._draw_peak_scatter),
        )

    def _draw_slow_event(self, session, height: float) -> None:
        """The slow event the focused peak sits in, with every peak inside it."""
        info = session.event_info(session.current)
        start, end = info["slow_start_s"], info["slow_end_s"]
        pad = max(0.25 * (end - start), 0.05)
        events, fs = session.events, session.fs
        lo = max(int((start - pad) * fs), 0)
        hi = min(int((end + pad) * fs) + 1, session.trace.size)
        t = np.arange(lo, hi) / fs
        fit = self._fit_for(
            "sf_event", (session.recording_id, info["slow_id"], settings_tag(session))
        )
        with line_plot(
            "##curation_sf_b", "time (s)", "dF/F", height=height, fit=fit, legend=False
        ) as ok:
            if not ok:
                return
            bands([start], [end], SLOW_BAND_COLOR, SLOW_BAND_ALPHA)
            line("fast", events.fast_trace[lo:hi], x=t, color=FAST_TRACE_COLOR)
            line(
                "slow",
                events.slow_trace[lo:hi],
                x=t,
                color=SLOW_TRACE_COLOR,
                weight=1.6,
            )
            inside = np.flatnonzero(events.fast["slow_id"] == info["slow_id"])
            if session.excluded or not inside.size:
                return
            focused = np.flatnonzero(inside == session.current)
            picked = self.event_points.items(
                session.times_s[inside],
                session.peak_values[inside],
                session.colors(inside),
                focused=int(focused[0]) if focused.size else None,
                tooltip=lambda k: f"peak {inside[k] + 1}: {session.label(inside[k])}",
            )
            if picked is not None:
                session.select(int(inside[picked]))

    def _draw_fast_peak(self, session, height: float) -> None:
        """The focused peak: the baseline-subtracted trace, its fine and fast smoothings."""
        events, fs, s = session.events, session.fs, session.settings
        center = int(session.sample_index[session.current])
        half = int(PEAK_WINDOW_MS * fs / 1000)
        lo, hi = max(center - half, 0), min(center + half + 1, session.trace.size)
        t_ms = (np.arange(lo, hi) - center) / fs * 1000
        fit = self._fit_for(
            "sf_peak", (session.recording_id, center, settings_tag(session))
        )
        with line_plot(
            "##curation_sf_c", "time from peak (ms)", "dF/F", height=height, fit=fit
        ) as ok:
            if not ok:
                return
            line("raw", events.x[lo:hi], x=t_ms, color=RAW_COLOR)
            if s.fine_sigma_ms > 0:
                line(
                    f"{s.fine_sigma_ms:g} ms",
                    events.fine_trace[lo:hi],
                    x=t_ms,
                    color=FINE_COLOR,
                    weight=1.2,
                )
            line(
                f"{s.fast_sigma_ms:g} ms",
                events.fast_trace[lo:hi],
                x=t_ms,
                color=FAST_TRACE_COLOR,
                weight=2.0,
            )
            dotted_vline(0.0, ZERO_COLOR)

    def _draw_peak_scatter(self, session, height: float) -> None:
        """Every candidate in view as width against prominence; the amber
        line is the prominence floor (drag it; applied on release).
        """
        fast = session.events.fast
        visible = session.visible
        xs = fast["width_ms"][visible] if session.n else np.zeros(0)
        ys = fast["prominence_noise"][visible] if session.n else np.zeros(0)
        focused = np.flatnonzero(visible == session.current)
        if self._fit_for("sf_scatter", (session.recording_id, settings_tag(session))):
            self.peak_scatter.refit()
        picked = self.peak_scatter.draw(
            xs,
            ys,
            session.colors(visible),
            x_label="half width (ms)",
            y_label="prominence (noise SD)",
            height=height,
            focused=int(focused[0]) if focused.size else None,
            tooltip=lambda k: (
                f"peak {visible[k] + 1}: {session.label(visible[k])}\n"
                f"width {xs[k]:.2f} ms  prominence {ys[k]:.1f}"
            ),
            inside=lambda: self._scatter_inside(session, xs, ys, visible),
            flags=self._box_flags(),
        )
        if picked is not None:
            session.select(int(visible[picked]))

    def _scatter_inside(self, session, xs, ys, visible) -> None:
        self._box_select("scatter", xs, ys, visible, self.peak_scatter)
        self._sd_line(session, 3, "k_prom", PROMINENCE_COLOR, 1.0)

    def _draw_slow_fast_event(self, session) -> None:
        if not session.n:
            imgui.text_disabled(
                "Trace excluded."
                if session.excluded
                else "No fast peaks inside slow events."
            )
            return
        info = session.event_info(session.current)
        imgui.text(f"peak {info['index'] + 1}/{session.n}")
        imgui.text("label: ")
        imgui.same_line(0, 0)
        imgui.text_colored(
            theme.to_vec4(self._label_color(info["label"])), info["label"]
        )
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(
            f"time {info['time_s']:.3f} s · amplitude {info['amplitude']:.4f} · "
            f"prominence {info['prominence_noise']:.1f} noise · width {info['width_ms']:.2f} ms"
        )
        imgui.text_wrapped(
            f"slow event {info['slow_id'] + 1}: {info['slow_start_s']:.3f}-{info['slow_end_s']:.3f} s, "
            f"peak {info['slow_amplitude_sd']:.1f} SD, {info['slow_n_fast']} fast"
        )
        imgui.pop_style_color()

    def _draw_detection(self, session) -> None:
        """The rest of the settings, the trace's exclusion and a reset."""
        section("Detection")
        changed, excluded = imgui.checkbox("exclude trace", session.excluded)
        if changed:
            session.set_excluded(excluded)
            self._panel_keys.clear()
        set_tooltip(
            "a trace too noisy to curate: no candidates, kept in the file as excluded",
            show_mark=False,
        )
        imgui.same_line(0, em(1.0))
        if imgui.small_button("defaults##sf_defaults"):
            session.set_settings(**SlowFastSettings().to_dict())
            self._panel_keys.clear()
        set_tooltip("the published run's settings for this trace", show_mark=False)
        imgui.same_line(0, em(0.6))
        if imgui.small_button("use for new##sf_new"):
            self.slow_fast_settings = SlowFastSettings(**session.settings.to_dict())
        set_tooltip(
            "traces with nothing saved open with this trace's settings", show_mark=False
        )
        if not imgui.collapsing_header("advanced##sf_advanced"):
            return
        for name, label, fmt, tip in ADVANCED:
            imgui.set_next_item_width(em(6))
            _changed, value = imgui.input_float(
                f"{label}##sf_{name}", getattr(session.settings, name), 0.0, 0.0, fmt
            )
            if imgui.is_item_deactivated_after_edit() and value >= 0:
                self._apply_setting(session, name, value)
            set_tooltip(tip, show_mark=False)

    def _draw_slow_fast_files(self, session) -> None:
        yes, no, unlabeled = session.counts()
        imgui.text_disabled(f"yes {yes} · no {no} · unlabeled {unlabeled}")
        imgui.text_disabled(
            f"{session.events.n_slow} slow events · {session.n} fast peaks"
        )
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(f"curation file: {session.label_path}")  # the path is long
        imgui.pop_style_color()
