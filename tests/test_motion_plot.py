"""The motion plot: a recording's motion correction on a plot that shares a
trace's time axis, drawn alone, in linked subplots, and as the line-scan
viewer's ``Traces`` tab. One frame each on a bare imgui context."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

pytest.importorskip("imgui_bundle")
from imgui_bundle import imgui, implot  # noqa: E402

from mbo_utilities.arrays.features import MotionCorrection  # noqa: E402


def _motion(n: int = 20000) -> MotionCorrection:
    t = np.arange(n) / 1000.0
    return MotionCorrection("RTMC", "um", {"X": (t, np.sin(t)), "Y": (t, np.cos(t)), "Z layer 3": (t, 0.1 * t)})


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
    seen.append(plot.draw("##motion", 200.0, cursor=1.0, duration_s=20.0))


def draw_in_frames(plot, seen: list) -> None:
    # a movie's frame axis: 100 frames per second, the cursor at frame 50
    seen.append(plot.draw("##motion", 200.0, cursor=50.0, duration_s=20.0, x_per_second=100.0, x_label="frame"))


def draw_linked(plot, ratios, seen: list) -> None:
    from mbo_utilities.gui.imgui.lines import line, line_plot, subplots

    link = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_title
    with subplots("##pair", 2, 1, 400.0, flags=link, ratios=ratios) as ok:
        assert ok
        seen.append(plot.draw("##motion", cursor=1.0, cursor_id=5, duration_s=20.0))
        with line_plot("##trace", "time (s)", "F", fit=True) as ok:
            assert ok
            line("f", np.sin(np.arange(2000) / 100.0), x=np.arange(2000) / 100.0)


def test_the_model_carries_one_trace_per_axis_and_its_extent():
    motion = _motion()
    assert motion and sorted(motion.traces) == ["X", "Y", "Z layer 3"]
    assert motion.duration_s == pytest.approx(19.999)
    assert not MotionCorrection("suite2p", "px") and MotionCorrection("suite2p", "px").duration_s == 0.0


def test_the_plot_decimates_and_labels_by_source_and_unit():
    from mbo_utilities.gui.imgui.motion import MotionPlot

    plot = MotionPlot(_motion(), points=1000)
    assert plot and sorted(plot.traces) == ["X", "Y", "Z layer 3"]
    # a min and a max per bin
    assert plot.traces["X"][0].shape == plot.traces["X"][1].shape == (2000,)
    assert plot.duration_s == pytest.approx(19.999)
    assert plot.y_label == "RTMC shift (um)"
    assert not MotionPlot(None) and not MotionPlot(MotionCorrection("suite2p", "px"))


def test_the_plot_draws_alone_in_frames_and_in_linked_subplots():
    from mbo_utilities.gui.imgui.motion import MotionPlot

    plot = MotionPlot(_motion())
    seen = []
    _frame(partial(draw_alone, plot, seen))
    assert seen == [(1.0, False), (1.0, False)]
    seen.clear()
    _frame(partial(draw_in_frames, plot, seen))
    assert seen == [(50.0, False), (50.0, False)]
    ratios = implot.SubplotsRowColRatios(row_ratios=[0.3, 0.7])
    seen.clear()
    _frame(partial(draw_linked, plot, ratios, seen))
    assert seen == [(1.0, False), (1.0, False)]
    assert list(ratios.row_ratios) == [pytest.approx(0.3), pytest.approx(0.7)]


class FakeOverlay:
    def __init__(self):
        from mbo_utilities.gui.playhead import Playhead

        self.playhead = Playhead()
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


def test_line_traces_panel_is_the_traces_tab_with_the_motion_plot_under_the_trace():
    from mbo_utilities.gui.linescan_viewer import TRACES_MOTION_PANEL_HEIGHT, TRACES_PANEL_HEIGHT, LineTracesPanel

    strip = FakeStrip()
    traces = np.random.default_rng(0).random((2, 5000)).astype(np.float32)
    panel = LineTracesPanel(None, FakeOverlay(), traces, strip, False, motion=_motion())
    assert [(p.key, p.label) for p in strip.panels] == [("traces", "Traces")]
    assert strip.panels[0].height == TRACES_MOTION_PANEL_HEIGHT
    assert panel.motion and panel.show_trace and panel.show_motion
    # the x axis spans the recording, not just the 5 s of traces on disk
    assert panel.duration_s == pytest.approx(19.999)
    _frame(panel.draw_tab)
    panel.show_trace = False
    _frame(panel.draw_tab)
    panel.show_motion = False
    _frame(panel.draw_tab)
    panel.close()
    assert strip.panels == []
    plain = LineTracesPanel(None, FakeOverlay(), traces, FakeStrip(), False)
    assert not plain.motion and plain.strip.panels[0].height == TRACES_PANEL_HEIGHT
    assert plain.duration_s == pytest.approx(4.999)
    _frame(plain.draw_tab)
