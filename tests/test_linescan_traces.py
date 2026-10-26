"""A line-scan unit on the standard viewer (``mbo file.mesc``): its per-ROI
traces land on the ROI widget's Traces tab as one external trace set, the
plotted row follows the ROI slider and the slider follows a picked row, and
the set goes when the traces are closed.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

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


class _DrawWithSettings(_DrawTraces):
    """Draws the Traces tab body with its Trace settings popup open."""

    def __call__(self, *_args) -> None:
        from imgui_bundle import imgui

        imgui.open_popup("##trace_settings")
        super().__call__()


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
        t.c == 0
        and t.extra
        == {"line": t.member, "from": traces.job.source, "unit": "photons"}
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
    assert available_kinds(first) == ("dff", "zscore", "raw")
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


def test_a_vnoiser_run_of_the_unit_is_a_panel_of_its_own(run_viewer):
    """Opening a scan that vnoiser ran on loads the run's domains without a
    click, one row per domain carrying every kind the run wrote. The ROI
    slider keeps plotting the recording's own line; the domain holding that
    line is the vnoiser panel's, and a line no domain holds leaves it empty.
    Picking a domain in the trace table moves the slider to its first line.
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
    basal = next(t for t in rows if t.label == "basal1")
    assert soma.engine == "vnoiser" and soma.extra["lines"] == [0, 1, 2]
    assert soma.extra["source_unit"] == "MSession_0/MUnit_35"
    assert available_kinds(soma)[0] == "denoised"
    np.testing.assert_allclose(
        display_trace(soma, "denoised", flip=True), -display_trace(soma, "denoised")
    )

    traces = attach_standard_traces(parent)
    if traces.job.idle:
        traces.job.start()
    traces.job.wait()
    traces()
    name = "MUnit_35 lines"
    assert roi.trace_sel == {("member", name, 0)}
    holding, every, run = roi._vnoiser_rows()
    assert run is not None and run.name == "PF"
    assert holding == [soma] and sorted(t.label for t in every) == ["basal1", "soma"]
    parent.image_widget.indices[traces.roi_dim] = 4
    traces()
    assert roi.trace_sel == {("member", name, 4)}
    assert roi._vnoiser_rows()[0] == [basal]
    # line 6 is in no domain: its own mean, and nothing in the vnoiser panel
    parent.image_widget.indices[traces.roi_dim] = 6
    traces()
    assert roi.trace_sel == {("member", name, 6)}
    assert roi._vnoiser_rows()[0] == []
    # the plot: the recording's line in the traces panel, the domain in a panel of its own under it
    parent.image_widget.indices[traces.roi_dim] = 1
    traces()
    draw = _DrawTraces(roi)
    roi.strip._update_calls[:] = [draw]
    for _ in range(2):
        parent.image_widget.figure.canvas.draw()
    assert not draw.errors, draw.errors[0]
    plot = roi.trace_plot
    assert plot.panels[-1] == "vnoiser denoised"
    assert roi._traces_panel_name() == plot.panels[-2]
    assert [label for label, _y, _rgb in plot._lines[plot.panels[-2]]] == [
        f"{name} · ROI 1"
    ]
    assert [label for label, _y, _rgb in plot._lines["vnoiser denoised"]] == ["soma"]
    # the archive's PF run wrote only the denoised trace; a kind it lacks shows that
    roi._vnoiser_kind = "zscore"
    for _ in range(2):
        parent.image_widget.figure.canvas.draw()
    assert not draw.errors, draw.errors[0]
    assert roi.trace_plot.panels[-1] == "vnoiser denoised"
    roi.show_vnoiser = False
    for _ in range(2):
        parent.image_widget.figure.canvas.draw()
    assert not any(p.startswith("vnoiser") for p in roi.trace_plot.panels)
    # picking a domain in the table moves the slider to its first line
    roi.select_trace(soma.key)
    traces()
    assert int(parent.image_widget.indices[traces.roi_dim]) == 0
    traces.close()


def test_loading_a_run_takes_the_shown_units_domains_where_it_ran(run_viewer):
    """Loading a run by hand (the vnoiser tab's Load into Traces, a finished
    run adopted) takes the domains of the scan that processed the unit on
    screen and nothing of the other scans; a run over a frame window plots
    over that window.
    """
    from mbo_utilities.results import Results

    parent, arr = run_viewer
    roi = parent.manual_roi
    pf = Path(arr.filenames[0]).parent / "PF"
    roi.traces.drop_source("PF/scan35")
    assert roi.load_run(pf)
    assert sorted(roi.traces.sources()) == ["PF/scan35"]
    assert sorted(t.label for t in roi.traces.from_source("PF/scan35")) == [
        "basal1",
        "soma",
    ]
    results = Results.open(pf)
    results.source["frames"] = [10, 30]
    rows = roi._line_unit_rows(results, results.units["scan35"], pf, with_members=False)
    assert all(t.frames == (10, 30, 1) for t in rows)
    assert roi.trace_axis(rows[0]).offset == pytest.approx(10 / rows[0].fs)


@pytest.fixture
def single_patch_viewer(tmp_path):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    from tests.test_voltage_traces import _single_patch_mesc

    path = tmp_path / "cell.mesc"
    green, _red = _single_patch_mesc(path, n_frames=200)
    arr = MescArray(path, unit="MUnit_4")
    iw = MboNDViewer(
        data=display_wrap(arr),
        slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": FIGURE_SIZE},
    )
    iw.show()
    roi = ManualRoiWidget(iw, fpath=None)
    yield _Parent(iw, roi), arr, green
    roi.close()
    iw.close()


def test_a_single_patch_scan_is_one_row_of_its_frame(single_patch_viewer):
    """A one-patch chessboard scan has no ROI axis; its frame is its one
    ROI, and its row is the frame's mean in photons.
    """
    from mbo_utilities.gui.linescan_viewer import attach_standard_traces

    parent, arr, _green = single_patch_viewer
    roi = parent.manual_roi
    assert arr.metadata["mesc_layout"] == "frames"
    traces = attach_standard_traces(parent)
    assert traces is not None and traces.roi_dim is None
    if traces.job.idle:
        traces.job.start()
    traces.job.wait()
    assert traces.job.error is None, traces.job.error
    traces()
    rows = roi.traces.from_source(traces.name)
    assert len(rows) == 1 and rows[0].extra["unit"] == "photons"
    scale, offset = (
        arr.metadata["mesc_channel_conversion"][0][k] for k in ("scale", "offset")
    )
    expected = arr[:, 0, 0].reshape(arr.shape[0], -1).mean(axis=1) * scale + offset
    np.testing.assert_allclose(rows[0].F, expected, rtol=1e-5)
    assert roi.trace_sel == {rows[0].key}
    traces.close()


def test_a_vnoiser_run_reaches_its_panel_from_the_scan(single_patch_viewer):
    """The whole path on a single-patch scan: the scan's own row plots in
    the traces panel with the panel's dF/F and inversion, a vnoiser run of
    it (traces only) is loaded the way a finished run is adopted, and its
    domain plots in the vnoiser panel in any kind the run wrote.
    """
    from mbo_utilities.gui.linescan_viewer import attach_standard_traces
    from mbo_utilities.results import newest_results
    from mbo_utilities.vnoiser.params import VnoiserSettings
    from mbo_utilities.vnoiser.pipeline import run_vnoiser_pipeline

    parent, arr, _green = single_patch_viewer
    roi = parent.manual_roi
    traces = attach_standard_traces(parent)
    if traces.job.idle:
        traces.job.start()
    traces.job.wait()
    traces()
    assert roi._vnoiser_rows() == ([], [], None)
    settings = VnoiserSettings()
    settings.runtime.denoise = False
    run_vnoiser_pipeline(
        arr.filenames[0], domains={"cell": {"rois": [0], "pixels": "bright"}}, settings=settings
    )
    found = newest_results(Path(arr.filenames[0]).parent, "vnoiser")
    assert found is not None and roi.load_run(found)
    holding, every, run = roi._vnoiser_rows()
    assert run == found and [t.label for t in holding] == ["cell"] == [t.label for t in every]
    assert roi.kind_options(every) == ("dff", "zscore", "raw")

    draw = _DrawWithSettings(roi)
    roi.strip._update_calls[:] = [draw]
    for _ in range(3):
        parent.image_widget.figure.canvas.draw()
    assert not draw.errors, draw.errors[0]
    plot = roi.trace_plot
    assert plot.panels == ("dF/F (%)", "vnoiser dF/F (%)")
    roi._vnoiser_kind = "zscore"
    roi.trace_invert = "on"
    roi.dff = None
    for _ in range(3):
        parent.image_widget.figure.canvas.draw()
    assert not draw.errors, draw.errors[0]
    assert roi.deflection() == (False, True)
    assert roi.trace_plot.panels == ("dF/F (%), inverted", "vnoiser z-score")
    vlines = roi.trace_plot._lines["vnoiser z-score"]
    assert [label for label, _y, _rgb in vlines] == ["cell"]
    assert np.isfinite(vlines[0][1]).any()
    traces.close()
