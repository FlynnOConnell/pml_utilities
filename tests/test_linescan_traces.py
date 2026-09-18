"""A line-scan unit on the standard viewer (``mbo file.mesc``): its per-ROI
traces land on the ROI widget's Traces tab as one external trace set, the
plotted row follows the ROI slider and the slider follows a picked row, and
the set goes when the traces are closed."""

from __future__ import annotations

import logging
import traceback

import numpy as np
import pytest

pytest.importorskip("imgui_bundle")

from tests.test_pf_array import write_mesc  # noqa: E402

FIGURE_SIZE = (640, 480)


class _Parent:
    """What ``attach_standard_traces`` reads off a ``PreviewDataWidget``."""

    def __init__(self, iw, roi):
        self.image_widget = iw
        self.manual_roi = roi
        self.top_strip = roi.tools_window
        self.logger = logging.getLogger("test_linescan_traces")
        self.fpath = None


class _DrawTraces:
    """Draws the Traces tab body on the strip's frames, keeping any error."""

    def __init__(self, roi):
        self.roi = roi
        self.errors: list[str] = []

    def __call__(self, *_args) -> None:
        try:
            self.roi.draw_traces()
        except Exception:
            self.errors.append(traceback.format_exc())


@pytest.fixture
def viewer(tmp_path):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = MescArray(write_mesc(tmp_path / "scan.mesc", munits=(35,), frames=40), unit="MUnit_35")
    # the launch path names the sliders after the array (run_gui)
    iw = MboNDViewer(data=display_wrap(arr), slider_dim_names=arr.slider_dim_labels, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    roi = ManualRoiWidget(iw, fpath=None)
    yield _Parent(iw, roi), arr
    roi.close()
    iw.close()


def test_line_traces_are_one_external_set_following_the_roi_slider(viewer):
    from mbo_utilities.gui.linescan_viewer import attach_standard_traces

    parent, arr = viewer
    roi = parent.manual_roi
    traces = attach_standard_traces(parent)
    assert traces is not None and parent.linescan_traces is traces
    assert traces.roi_dim is not None
    # while the job runs the tab shows its progress in place of "no traces"
    assert roi.pending_traces == traces.draw_pending
    if traces.job.idle:
        traces.job.start()
    traces.job.wait()
    traces()
    name = "MUnit_35 lines"
    ts = roi.trace_sets[name]
    assert ts.external and ts.kind == traces.job.source
    assert sorted(ts.data) == list(range(7))
    assert ts.data[0]["label"] == "ROI 0" and ts.data[0]["fs"] == pytest.approx(arr.fs)
    assert ts.data[0]["F"].shape == (40,)
    assert roi.pending_traces is None and roi.focus_traces
    assert roi.trace_sel == {("uid", name, 0)}
    # an external row stands for no drawn ROI
    assert roi._key_to_pair(("uid", name, 0)) is None

    # the ROI slider picks the plotted row
    parent.image_widget.indices[traces.roi_dim] = 3
    traces()
    assert roi.trace_sel == {("uid", name, 3)}
    # and a row picked in the trace table moves the slider
    roi.select_trace(("uid", name, 5))
    traces()
    assert int(parent.image_widget.indices[traces.roi_dim]) == 5
    assert roi.trace_sel == {("uid", name, 5)}

    draw = _DrawTraces(roi)
    roi.tools_window._update_calls[:] = [draw]
    for _ in range(2):
        parent.image_widget.figure.canvas.draw()
    assert not draw.errors, draw.errors[0]
    header, lines = roi._plot_lines()
    assert lines == [(f"{name} · ROI 5", ("uid", name, 5))]

    traces.close()
    assert name not in roi.trace_sets and roi.pending_traces is None
    assert roi.trace_sel == set()


def test_nothing_attaches_without_the_roi_widget_or_for_another_unit(viewer, tmp_path):
    from mbo_utilities.gui.linescan_viewer import attach_standard_traces

    parent, _arr = viewer
    parent.manual_roi = None
    assert attach_standard_traces(parent) is None
    parent.image_widget.data[0] = np.zeros((4, 8, 8), np.float32)
    assert attach_standard_traces(parent) is None
