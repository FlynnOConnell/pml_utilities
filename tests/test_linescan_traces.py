"""A line-scan unit on the standard viewer (``mbo file.mesc``): its per-ROI
traces land on the ROI widget's Traces tab as one external trace set, the
plotted row follows the ROI slider and the slider follows a picked row, and
the set goes when the traces are closed.
"""

from __future__ import annotations

import logging
import traceback

import numpy as np
import pytest

pytest.importorskip("imgui_bundle")

from tests.test_results_array import write_mesc  # noqa: E402

FIGURE_SIZE = (640, 480)


class _Parent:
    """What ``attach_standard_traces`` reads off a ``PreviewDataWidget``."""

    def __init__(self, iw, roi):
        self.image_widget = iw
        self.manual_roi = roi
        self.top_strip = roi.strip
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

    arr = MescArray(
        write_mesc(tmp_path / "scan.mesc", munits=(35,), frames=40), unit="MUnit_35"
    )
    # the launch path names the sliders after the array (run_gui)
    iw = MboNDViewer(
        data=display_wrap(arr),
        slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": FIGURE_SIZE},
    )
    iw.show()
    roi = ManualRoiWidget(iw, fpath=None)
    yield _Parent(iw, roi), arr
    roi.close()
    iw.close()


@pytest.fixture
def run_viewer(tmp_path):
    """The ``viewer`` unit with a vnoiser run of it beside the file."""
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    from tests.test_results_array import write_pf

    mesc = write_mesc(tmp_path / "scan.mesc", munits=(35,), frames=40)
    write_pf(tmp_path / "PF", source=mesc, units={"35": "MSession_0/MUnit_35"})
    arr = MescArray(mesc, unit="MUnit_35")
    iw = MboNDViewer(
        data=display_wrap(arr),
        slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": FIGURE_SIZE},
    )
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
    rows = roi.traces.from_source(name)
    assert sorted(t.member for t in rows) == list(range(7))
    # measured line means, shown with the mean engine's profile; where they came from rides in extra
    assert all(
        not t.stands_for_roi
        and t.engine == "mean"
        and t.extra["from"] == traces.job.source
        for t in rows
    )
    first = roi.traces.get(("member", name, 0))
    assert first.label == "ROI 0" and first.fs == pytest.approx(arr.fs)
    assert first.F.shape == (40,)
    # each row is its line on the unit's ROI axis; the synthetic file has no scan geometry
    assert [t.z for t in sorted(rows, key=lambda t: t.member)] == list(range(7))
    assert all(
        t.c == 0 and t.extra == {"line": t.member, "from": traces.job.source}
        for t in rows
    )
    assert roi._trace_cells(first.key)[:3] == ("ROI 0", "1", "0")
    assert roi.pending_traces is None
    assert roi.trace_sel == {("member", name, 0)}
    # a line row stands for no drawn ROI
    assert roi._key_to_pair(("member", name, 0)) is None

    # the ROI slider picks the plotted row
    parent.image_widget.indices[traces.roi_dim] = 3
    traces()
    assert roi.trace_sel == {("member", name, 3)}
    # and a row picked in the trace table moves the slider
    roi.select_trace(("member", name, 5))
    traces()
    assert int(parent.image_widget.indices[traces.roi_dim]) == 5
    assert roi.trace_sel == {("member", name, 5)}

    draw = _DrawTraces(roi)
    roi.strip._update_calls[:] = [draw]
    for _ in range(2):
        parent.image_widget.figure.canvas.draw()
    assert not draw.errors, draw.errors[0]
    header, lines = roi._plot_lines()
    assert [(label, key) for label, key, _rgb in lines] == [
        (f"{name} · ROI 5", ("member", name, 5))
    ]

    traces.close()
    assert name not in roi.traces.sources() and roi.pending_traces is None
    assert roi.trace_sel == set()


def write_chessboard_mesc(path, frames=8):
    """A ``.mesc`` with one chessboard unit, ``MUnit_9``: 3 patches of 6 x 8 px
    tiled along X at three depths, drawn on a snapshot 2 um above the first.
    """
    import json

    import h5py

    with h5py.File(path, "w") as f:
        u = f.create_group("MSession_0").create_group("MUnit_9")
        u.attrs.update(
            {
                "MethodType": 8,
                "VecChannelsSize": 1,
                "TStepInMs": 5.0,
                "MeasurementDatePosix": 1,
                "Comment": "soma",
                "BackgroundImagePath": "/MSession_1/MUnit_9",
            }
        )
        u.attrs["MultiROIProtocolJSON"] = json.dumps(
            {
                "protocol": {"scanners": {"mainPatternIndex": 1}},
                "scanPatterns": {
                    "patterns": [
                        {
                            "centerPoints": [
                                [4.0, 14.0, 24.0],
                                [3.0, 3.0, 3.0],
                                [-50.0, -52.0, -54.0],
                            ],
                            "pixelSizeX": 1.0,
                            "rotation": [0, 0, 0, 1],
                        }
                    ]
                },
            }
        )
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {
                "maps": [
                    {
                        "measurementROIs": [],
                        "contours": [
                            [[x, x + 8, x + 8, x], [0.0, 0.0, 6.0, 6.0], [z] * 4]
                            for x, z in ((0.0, -50.0), (10.0, -52.0), (20.0, -54.0))
                        ],
                    }
                ]
            }
        )
        u.create_dataset(
            "Channel_0",
            data=np.arange(frames * 6 * 24, dtype=np.uint16).reshape(frames, 6, 24),
        )
        snap = f.create_group("MSession_1").create_group("MUnit_9")
        snap.attrs.update(
            {
                "MethodType": 1,
                "VecChannelsSize": 1,
                "TStepInMs": 10.0,
                "MeasurementDatePosix": 1,
            }
        )
        snap.attrs["ReferenceViewportJSON"] = json.dumps(
            {
                "viewports": [
                    {
                        "geomTransTransl": [0.0, 0.0, -48.0],
                        "width": 64.0,
                        "height": 64.0,
                    }
                ]
            }
        )
        snap.create_dataset("Channel_0", data=np.zeros((1, 8, 8), np.uint16))
    return path


@pytest.fixture
def patch_viewer(tmp_path):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = MescArray(write_chessboard_mesc(tmp_path / "chess.mesc"), unit="MUnit_9")
    iw = MboNDViewer(
        data=display_wrap(arr),
        slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": FIGURE_SIZE},
    )
    iw.show()
    roi = ManualRoiWidget(iw, fpath=None)
    yield _Parent(iw, roi), arr
    roi.close()
    iw.close()


def test_patch_traces_attach_for_a_chessboard_unit(patch_viewer):
    """A chessboard (or ribbon) unit gets its per-patch mean traces the way a
    line scan gets its lines: computed in the background as soon as the unit
    is shown, one row per patch carrying its depth, and a dF/F the panel
    computes from the raw mean over a configurable baseline.
    """
    from mbo_utilities.annotation import DffSettings, available_kinds, display_trace
    from mbo_utilities.gui.linescan_viewer import attach_standard_traces

    parent, arr = patch_viewer
    roi = parent.manual_roi
    assert arr.metadata["mesc_layout"] == "tiled" and arr.shape == (8, 1, 3, 6, 8)
    traces = attach_standard_traces(parent)
    assert traces is not None and traces.name == "MUnit_9 patches"
    if traces.job.idle:
        traces.job.start()
    traces.job.wait()
    traces()
    rows = roi.traces.from_source("MUnit_9 patches")
    assert sorted(t.member for t in rows) == [0, 1, 2]
    first = roi.traces.get(("member", "MUnit_9 patches", 0))
    # the patch's own pixel mean per frame, placed by the scan geometry
    assert np.allclose(first.F, arr[:, 0, 0].reshape(8, -1).mean(axis=1))
    assert first.extra["z_um"] == -50.0 and first.extra["dz_um"] == pytest.approx(-2.0)
    # the label is the ROI alone; where it sits is a hover, not a number in its name
    assert first.label == "ROI 0"
    # a dF/F over a configurable baseline comes from the raw mean on the panel
    assert available_kinds(first) == ("dff", "raw")
    dff = display_trace(first, "dff", DffSettings(method="percentile", percentile=10.0))
    assert dff is not None and dff.shape == (8,)
    traces.close()
    assert "MUnit_9 patches" not in roi.traces.sources()


def test_nothing_attaches_without_the_roi_widget_or_for_another_unit(viewer, tmp_path):
    from mbo_utilities.gui.linescan_viewer import attach_standard_traces

    parent, _arr = viewer
    parent.manual_roi = None
    assert attach_standard_traces(parent) is None
    parent.image_widget.data[0] = np.zeros((4, 8, 8), np.float32)
    assert attach_standard_traces(parent) is None


def test_lines_with_geometry_carry_their_position(tmp_path):
    """A file whose lines have scanned segments and a snapshot: each row
    says where its line sits and how far off the snapshot plane it is.
    """
    import json

    import h5py
    from mbo_utilities.arrays.mesc_geometry import line_positions

    path = write_mesc(tmp_path / "scan.mesc", munits=(35,), frames=8)
    with h5py.File(path, "a") as f:
        unit = f["MSession_0/MUnit_35"]
        maps = json.loads(unit.attrs["CoordinateMapJSON"])
        # 7 lines, 2 um long along x, at depths -100 .. -94 um
        maps["maps"][0]["driftEndPoints"] = [
            [[10.0 + i, 12.0 + i], [20.0, 20.0], [-100.0 + i, -100.0 + i]]
            for i in range(7)
        ]
        unit.attrs["CoordinateMapJSON"] = json.dumps(maps)
        unit.attrs["BackgroundImagePath"] = "/MSession_1/MUnit_35"
        snap = f.create_group("MSession_1/MUnit_35")
        snap.attrs.update({"MethodType": 1, "VecChannelsSize": 1})
        snap.attrs["ReferenceViewportJSON"] = json.dumps(
            {
                "viewports": [
                    {
                        "geomTransTransl": [0.0, 0.0, -97.0],
                        "width": 64.0,
                        "height": 64.0,
                    }
                ]
            }
        )
        snap.create_dataset("Channel_0", data=np.zeros((1, 8, 8), np.uint16))
    rows = line_positions(path, "MSession_0/MUnit_35", [4] * 7)
    assert [r["index"] for r in rows] == list(range(7))
    assert rows[0]["start_um"] == [10.0, 20.0, -100.0] and rows[0]["end_um"] == [
        12.0,
        20.0,
        -100.0,
    ]
    assert rows[0]["length_um"] == pytest.approx(2.0) and rows[0][
        "sample_um"
    ] == pytest.approx(0.5)
    assert rows[0]["z_um"] == -100.0 and rows[0]["dz_um"] == pytest.approx(-3.0)
    assert rows[3]["dz_um"] == pytest.approx(0.0) and rows[6]["dz_um"] == pytest.approx(
        3.0
    )
    # no Z-stack in this file holds the lines
    assert rows[0]["stack"] is None and rows[0]["slice"] is None
    # no geometry at all: None, so a caller draws nothing rather than guessing
    assert line_positions(path, "MSession_1/MUnit_35") is None
    assert line_positions(path, "MSession_0/MUnit_99") is None


def test_a_vnoiser_run_of_the_unit_loads_its_domains_and_the_slider_picks_them(
    run_viewer,
):
    """Opening a scan that vnoiser ran on shows the run's domains on the
    Traces tab without a click: one row per domain carrying every kind the
    run wrote, picked by the ROI slider for any line the domain holds, the
    recording's own line mean for a line no domain holds.
    """
    from mbo_utilities.annotation import available_kinds, display_trace
    from mbo_utilities.gui.linescan_viewer import attach_standard_traces

    parent, arr = run_viewer
    roi = parent.manual_roi
    assert arr.results is not None and arr.results.pipeline == "vnoiser"
    rows = roi.traces.from_source("PF/scan35")
    # the domains alone: the lines' raw means are the recording's own rows
    assert sorted(t.label for t in rows) == ["basal1", "soma"]
    soma = next(t for t in rows if t.label == "soma")
    assert soma.engine == "vnoiser" and soma.extra["lines"] == [0, 1, 2]
    assert available_kinds(soma)[0] == "denoised"
    np.testing.assert_allclose(
        display_trace(soma, "denoised", flip=True), -display_trace(soma, "denoised")
    )

    traces = attach_standard_traces(parent)
    if traces.job.idle:
        traces.job.start()
    traces.job.wait()
    traces()
    assert roi.trace_sel == {soma.key}
    basal = next(t for t in rows if t.label == "basal1")
    parent.image_widget.indices[traces.roi_dim] = 4
    traces()
    assert roi.trace_sel == {basal.key}
    # line 6 is in no domain: its own mean
    parent.image_widget.indices[traces.roi_dim] = 6
    traces()
    assert roi.trace_sel == {("member", "MUnit_35 lines", 6)}
    # picking a domain in the table moves the slider to its first line
    roi.select_trace(soma.key)
    traces()
    assert int(parent.image_widget.indices[traces.roi_dim]) == 0
    traces.close()
