"""The motion plot: a recording's motion correction on a plot that shares a
trace's time axis, drawn alone, in linked subplots, and as the line-scan
viewer's ``Traces`` tab. One frame each on a bare imgui context.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

pytest.importorskip("imgui_bundle")
from imgui_bundle import imgui, implot  # noqa: E402
from mbo_utilities.arrays.features import MotionCorrection  # noqa: E402


def _motion(n: int = 20000) -> MotionCorrection:
    t = np.arange(n) / 1000.0
    return MotionCorrection(
        "RTMC",
        "um",
        {"X": (t, np.sin(t)), "Y": (t, np.cos(t)), "Z layer 3": (t, 0.1 * t)},
    )


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
    seen.append(
        plot.draw(
            "##motion",
            200.0,
            cursor=50.0,
            duration_s=20.0,
            x_per_second=100.0,
            x_label="frame",
        )
    )


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
    assert (
        not MotionCorrection("suite2p", "px")
        and MotionCorrection("suite2p", "px").duration_s == 0.0
    )


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


def record_line(seen: list, label, *_args, **_kwargs) -> None:
    seen.append(label)


def draw_plane(plot, z) -> None:
    plot.draw("##motion", 200.0, z=z)


def test_a_stage_run_plane_by_plane_draws_only_the_plane_on_screen(monkeypatch):
    from mbo_utilities.gui.imgui import motion as motion_module
    from mbo_utilities.gui.imgui.motion import MotionPlot

    t = np.arange(100) / 10.0
    plot = MotionPlot(
        MotionCorrection(
            "suite2p",
            "px",
            {"X a": (t, t), "Y a": (t, -t), "X b": (t, 2 * t), "Z": (t, t)},
            planes={"X a": 0, "Y a": 0, "X b": 1},
        )
    )
    drawn = []
    monkeypatch.setattr(motion_module, "line", partial(record_line, drawn))
    _frame(partial(draw_plane, plot, 0), frames=1)
    # a trace no plane claims belongs to all of them
    assert drawn == ["X a", "Y a", "Z"]
    drawn.clear()
    _frame(partial(draw_plane, plot, 1), frames=1)
    assert drawn == ["X b", "Z"]
    drawn.clear()
    _frame(partial(draw_plane, plot, None), frames=1)
    assert drawn == ["X a", "Y a", "X b", "Z"]
