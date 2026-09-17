"""The RTMC plot: a scan's motion-correction traces on a plot that shares a
trace's time axis, drawn alone, in linked subplots, and as the line-scan
viewer's ``Line traces`` tab. One frame each on a bare imgui context."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

pytest.importorskip("imgui_bundle")
from imgui_bundle import imgui, implot  # noqa: E402


def _rtmc(n: int = 20000) -> dict:
    t = np.arange(n) / 1000.0
    return {
        "X total": {"t": t, "um": np.sin(t)},
        "Y total": {"t": t, "um": np.cos(t)},
        "X intercycle": {"t": t, "um": 0.1 * t},
    }


def _frame(body, frames: int = 2) -> None:
    """Draw ``body`` in a window on a bare imgui context, no renderer."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(900, 700)
    # imgui 1.92 builds fonts lazily once a renderer claims texture support
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    plot_ctx = implot.create_context()
    try:
        for _ in range(frames):
            imgui.new_frame()
            imgui.set_next_window_size(imgui.ImVec2(800, 600))
            imgui.begin("host")
            body()
            imgui.end()
            imgui.end_frame()
    finally:
        implot.destroy_context(plot_ctx)
        imgui.destroy_context(ctx)


def draw_alone(plot, seen: list) -> None:
    plot.draw_checkboxes("alone")
    seen.append(plot.draw("##rtmc", 200.0, cursor=1.0, duration_s=20.0))


def draw_linked(plot, ratios, seen: list) -> None:
    from mbo_utilities.gui.imgui.lines import line, line_plot, subplots

    link = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_title
    with subplots("##pair", 2, 1, 400.0, flags=link, ratios=ratios) as ok:
        assert ok
        seen.append(plot.draw("##rtmc", cursor=1.0, cursor_id=5, duration_s=20.0))
        with line_plot("##trace", "time (s)", "F", fit=True) as ok:
            assert ok
            line("f", np.sin(np.arange(2000) / 100.0), x=np.arange(2000) / 100.0)


def test_the_plot_draws_alone_and_in_linked_subplots():
    from mbo_utilities.gui.imgui.rtmc import RtmcPlot

    plot = RtmcPlot(_rtmc())
    seen = []
    _frame(partial(draw_alone, plot, seen))
    assert seen == [(1.0, False), (1.0, False)]
    ratios = implot.SubplotsRowColRatios(row_ratios=[0.3, 0.7])
    seen.clear()
    _frame(partial(draw_linked, plot, ratios, seen))
    assert seen == [(1.0, False), (1.0, False)]
    assert list(ratios.row_ratios) == [pytest.approx(0.3), pytest.approx(0.7)]


class FakeOverlay:
    def __init__(self):
        self.fs = 1000.0
        self.selected = 0
        self.t_index = 10
        self.colors = np.ones((2, 4))
        self.moved = []

    def goto_time(self, t):
        self.moved.append(t)


class FakeStrip:
    def __init__(self):
        self.panels = []

    def register(self, panel):
        self.panels.append(panel)

    def unregister(self, key):
        self.panels = [p for p in self.panels if p.key != key]


def test_line_traces_panel_draws_f_over_the_motion_plot():
    from mbo_utilities.gui.linescan_viewer import TRACES_PANEL_HEIGHT, TRACES_RTMC_PANEL_HEIGHT, LineTracesPanel

    strip = FakeStrip()
    traces = np.random.default_rng(0).random((2, 5000)).astype(np.float32)
    panel = LineTracesPanel(None, FakeOverlay(), traces, strip, False, rtmc=_rtmc())
    assert [p.key for p in strip.panels] == ["line_traces"]
    assert strip.panels[0].height == TRACES_RTMC_PANEL_HEIGHT
    assert panel.rtmc and panel.rtmc.shown
    _frame(panel.draw_tab)
    panel._show_f = False
    _frame(panel.draw_tab)
    panel.rtmc.show = dict.fromkeys(panel.rtmc.show, False)
    _frame(panel.draw_tab)
    panel.close()
    assert strip.panels == []
    plain = LineTracesPanel(None, FakeOverlay(), traces, FakeStrip(), False)
    assert not plain.rtmc and plain.strip.panels[0].height == TRACES_PANEL_HEIGHT
    _frame(plain.draw_tab)
