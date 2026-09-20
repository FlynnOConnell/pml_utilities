"""The outer view of a MESc unit: its quick projections and the images it
was scanned on, with its ROIs drawn.

Fixture: a Z-stack (11 slices, 2 um apart, slice k filled with k) whose
field holds a four-line scan (6 frames, two lines on slice 3, one on slice
7, one below the stack) drawn on a snapshot focused 4 um under the stack's
origin, so two lines are on the snapshot's plane and two are not.
"""

from __future__ import annotations

import json
import traceback
from functools import partial

import h5py
import numpy as np
import pytest

TRANSL = (100.0, 200.0, -50.0)
SNAP_TRANSL = (100.0, 200.0, -54.0)
LINES = [
    [[110, 130], [210, 210], [-54, -54]],
    [[120, 120], [204, 228], [-46, -46]],
    [[105, 135], [220, 230], [-53.9, -53.9]],
    [[112, 118], [212, 212], [-70, -70]],
]


@pytest.fixture(scope="module")
def mesc_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("outer") / "outer.mesc"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")
        z = s.create_group("MUnit_0")
        z.attrs.update(
            {"MethodType": 2, "VecChannelsSize": 1, "TStepInMs": 1.0,
             "MeasurementDatePosix": 0, "Comment": "zstack",
             "MinZ": -10.0, "MaxZ": 10.0, "ZDim": 11}
        )
        z.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": list(TRANSL), "width": 40.0, "height": 32.0}]}
        )
        stack = np.broadcast_to(np.arange(11, dtype=np.uint16)[:, None, None], (11, 64, 80))
        z.create_dataset("Channel_0", data=np.ascontiguousarray(stack))

        ls = s.create_group("MUnit_1")
        ls.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 2.0,
             "MeasurementDatePosix": 1, "Comment": "linescan",
             "BackgroundImagePath": "/MSession_0/MUnit_4"}
        )
        boxes = [
            {"lowerLeftFramePix": [2 * i + 1, 1], "upperRightFramePix": [2 * i + 2, 1]}
            for i in range(4)
        ]
        ls.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": boxes, "driftEndPoints": LINES}]}
        )
        # packed: one raw frame holding the 6 timepoints of every line as rows
        ls.create_dataset("Channel_0", data=rng.integers(0, 1000, (1, 6, 8)).astype(np.uint16))

        snap = s.create_group("MUnit_4")
        snap.attrs.update({"MethodType": 1, "VecChannelsSize": 1, "TStepInMs": 1.0,
                           "MeasurementDatePosix": 0, "ImageRoleDebugString": "background"})
        snap.attrs["ReferenceViewportJSON"] = json.dumps(
            {"viewports": [{"geomTransTransl": list(SNAP_TRANSL), "width": 40.0, "height": 32.0}]}
        )
        snap.create_dataset("Channel_0", data=np.full((1, 64, 80), 7, np.uint16))
    return path


def open_unit(path, key):
    from mbo_utilities.arrays.mesc import MescArray

    return MescArray(path, unit=key)


def test_unit_projections_sample_a_long_recording():
    from mbo_utilities.arrays.numpy import NumpyArray
    from mbo_utilities.gui.mesc_outer_view import unit_projections

    data = np.arange(10 * 2 * 2, dtype=np.float32).reshape(10, 1, 1, 2, 2)
    arr = NumpyArray(data, dims="TCZYX")
    full = unit_projections(arr, 0, 0)
    np.testing.assert_allclose(full["mean"], data[:, 0, 0].mean(axis=0))
    np.testing.assert_allclose(full["max"], data[:, 0, 0].max(axis=0))
    np.testing.assert_allclose(full["std"], data[:, 0, 0].std(axis=0))
    # 12 samples of 4-pixel frames: three frames, evenly spaced
    sampled = unit_projections(arr, 0, 0, max_elements=12)
    np.testing.assert_allclose(sampled["mean"], data[::4, 0, 0].mean(axis=0))
    # one frame is no projection
    assert unit_projections(NumpyArray(data[:1], dims="TCZYX"), 0, 0) == {}


def test_a_one_row_slice_projects_every_roi():
    from mbo_utilities.arrays.numpy import NumpyArray
    from mbo_utilities.gui.mesc_outer_view import unit_projections

    # a line scan's plane: one line per ROI on Z, one row tall
    data = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 1, 3, 1, 4)
    out = unit_projections(NumpyArray(data, dims="TCZYX"), 0, 1)
    assert out["mean"].shape == (3, 4)
    np.testing.assert_allclose(out["mean"], data[:, 0, :, 0].mean(axis=0))


def test_outer_images_of_a_line_scan(mesc_path):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.mesc_outer_view import outer_images

    mesc = MescArray(mesc_path, unit=1)
    try:
        images = outer_images(mesc, partial(open_unit, mesc_path), 0, 0)
    finally:
        mesc.close()
    by_key = {im.key: im for im in images}
    assert [im.key for im in images] == [
        "MUnit_1 mean", "MUnit_1 max", "MUnit_1 std",
        "MUnit_4 snapshot",
        "MUnit_0 stack mean", "MUnit_0 stack max", "MUnit_0 slice 4", "MUnit_0 slice 8",
    ]
    # the unit's own projections follow its lines-by-samples rule
    mesc = MescArray(mesc_path, unit=1)
    try:
        block = np.asarray(mesc[:, 0], np.float32)
        _nt, _nc, nz, ny, nx = mesc.shape
        expected = block.reshape(-1, nz * ny, nx) if ny == 1 and nz > 1 else block[:, 0]
    finally:
        mesc.close()
    np.testing.assert_allclose(by_key["MUnit_1 mean"].image, expected.mean(axis=0))
    np.testing.assert_allclose(by_key["MUnit_1 max"].image, expected.max(axis=0))
    assert by_key["MUnit_1 mean"].records == [] and by_key["MUnit_1 mean"].slice is None
    # the snapshot, with the lines on its plane solid
    snap = by_key["MUnit_4 snapshot"]
    assert snap.image.shape == (64, 80) and float(snap.image[0, 0]) == 7.0
    assert [r["roi"] for r in snap.records] == [0, 1, 2, 3]
    assert [r["on_plane"] for r in snap.records] == [True, False, True, False]
    assert snap.slice is None
    # the stack: slice k holds the value k, so the projections are known
    np.testing.assert_allclose(by_key["MUnit_0 stack mean"].image, 5.0)
    np.testing.assert_allclose(by_key["MUnit_0 stack max"].image, 10.0)
    assert by_key["MUnit_0 slice 4"].slice == 3 and float(by_key["MUnit_0 slice 4"].image[0, 0]) == 3.0
    assert by_key["MUnit_0 slice 8"].slice == 7
    assert [r["slice"] for r in by_key["MUnit_0 slice 4"].records] == [3, 7, 3, 0]


def test_the_reader_places_its_lines(mesc_path):
    """``MescArray.line_positions``: each line's depth against the snapshot
    it was drawn on, read once; None for a unit without ROIs."""
    from mbo_utilities.arrays.mesc import MescArray

    scan = MescArray(mesc_path, unit=1)
    stack = MescArray(mesc_path, unit=0)
    try:
        positions = scan.line_positions
        assert [p["index"] for p in positions] == [0, 1, 2, 3]
        np.testing.assert_allclose([p["dz_um"] for p in positions], [0.0, 8.0, 0.1, -16.0])
        # and on the Z-stack holding them: the slice each sits on, one below it
        assert [p["stack"] for p in positions] == ["MSession_0/MUnit_0"] * 4
        assert [p["slice"] for p in positions] == [3, 7, 3, 0]
        assert [p["in_stack"] for p in positions] == [True, True, True, False]
        assert scan.line_positions is positions
        assert stack.line_positions is None
    finally:
        scan.close()
        stack.close()


def test_trace_rows_on_a_line_scan_know_their_line(mesc_path):
    """On a line-scan viewer every row finds its line through the reader: a
    results row by its line, a drawn ROI's quick trace by the ROI slider it
    was read on; the depth cell is the offset from the snapshot."""
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    from mbo_utilities.annotation import RoiTrace
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = MescArray(mesc_path, unit=1)
    iw = MboNDViewer(
        data=display_wrap(arr), slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": (640, 480)},
    )
    iw.show()
    widget = ManualRoiWidget(iw, fpath=None, auto_trace=False)
    try:
        assert [p["index"] for p in widget.line_positions] == [0, 1, 2, 3]
        results_row = widget.traces.add(RoiTrace(
            uid=0, member=1, source="results", engine="voltage", label="roi1",
            F=np.zeros(6, np.float32), z=1, c=0, extra={"line": 1},
        ))
        drawn_row = widget.traces.add(RoiTrace(uid=5, F=np.zeros(6, np.float32), z=2, c=0))
        assert widget._line_position(results_row)["dz_um"] == 8.0
        assert widget._trace_cells(results_row.key)[5] == "+8.0 um"
        assert widget._line_position(drawn_row)["slice"] == 3
        assert widget._trace_cells(drawn_row.key)[5] == "+0.1 um"
        # a row's own record wins over the recording's
        results_row.extra["dz_um"] = -2.5
        assert widget._trace_cells(results_row.key)[5] == "-2.5 um"
    finally:
        widget.close()
        iw.close()
        arr.close()


def test_a_stack_or_snapshot_has_only_its_own_projections(mesc_path):
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui.mesc_outer_view import outer_images

    stack = MescArray(mesc_path, unit=0)
    snap = MescArray(mesc_path, unit="MUnit_4")
    try:
        # a one-frame stack over depth has no time projection and nothing holds it
        assert outer_images(stack, partial(open_unit, mesc_path), 0, 0) == []
        assert outer_images(snap, partial(open_unit, mesc_path), 0, 0) == []
    finally:
        stack.close()
        snap.close()


class ViewerHost:
    """A viewer-hosting parent: what the outer view reads."""

    def __init__(self, iw):
        self.image_widget = iw


def test_outer_view_popup_draws_and_highlights_the_sliders_roi(mesc_path):
    """On a real figure: the popup builds the images for the shown line scan,
    draws without raising, and colours the lines with the slider's ROI thick."""
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    from mbo_utilities.arrays.mesc import MescArray
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui._top_strip import TopStrip
    from mbo_utilities.gui.mesc_outer_view import GHOST_ALPHA, ON_THICKNESS, SELECTED_THICKNESS, OuterView, z_slider
    from mbo_utilities.gui.widgets.mesc_units import display_wrap

    arr = MescArray(mesc_path, unit=1)
    iw = MboNDViewer(
        data=display_wrap(arr), slider_dim_names=arr.slider_dim_labels,
        figure_kwargs={"size": (640, 480)},
    )
    iw.show()
    strip = TopStrip(iw.figure)
    host = ViewerHost(iw)
    view = OuterView(host)
    errors = []

    def guarded(*_args):
        try:
            strip.update()
            view.draw()
        except Exception:
            errors.append(traceback.format_exc())

    try:
        assert view.open(arr, partial(open_unit, mesc_path))
        assert view.is_open and view.unit == "MUnit_1"
        assert view.viewer.current_key == "MUnit_4 snapshot"
        strip._update_calls[:] = [guarded]
        for _ in range(3):
            iw.figure.canvas.draw()
        assert not errors, errors[0]
        # every line placed on the snapshot draws solid, the slider's one thick
        lines = view.contours("MUnit_4 snapshot")
        assert [t for _p, _c, t in lines] == [SELECTED_THICKNESS, ON_THICKNESS, ON_THICKNESS, ON_THICKNESS]
        assert [c[3] for _p, c, _t in lines] == [1.0, 1.0, 1.0, 1.0]
        assert all(p.shape == (2, 2) for p, _c, _t in lines)
        # on a stack slice only the lines on it are solid; one below the stack never is
        slice4 = view.contours("MUnit_0 slice 4")
        assert [t for _p, _c, t in slice4] == [SELECTED_THICKNESS, 1.0, ON_THICKNESS, 1.0]
        assert [c[3] for _p, c, _t in slice4] == [1.0, GHOST_ALPHA, 1.0, GHOST_ALPHA]
        # clicking a line in the popup selects its ROI: the slider (found by
        # position on a line scan) moves and the highlight follows
        pts = np.asarray(view.images["MUnit_4 snapshot"].records[2]["pixels"], float)
        # a quarter of the way along: the midpoint sits on the vertical line 2 as well
        near = pts[0] + 0.25 * (pts[1] - pts[0])
        assert view.pick("MUnit_4 snapshot", float(near[1]), float(near[0])) == 2
        assert int(iw.indices[z_slider(iw.dim_names)]) == 2
        assert [t for _p, _c, t in view.contours("MUnit_4 snapshot")] == [ON_THICKNESS, ON_THICKNESS, SELECTED_THICKNESS, ON_THICKNESS]
        assert view.pick("MUnit_4 snapshot", -40.0, -40.0) is None
        assert view.pick("MUnit_1 mean", 0.0, 0.0) is None
        # the same unit again is not rebuilt; another slider position is
        built = view._built
        assert view.open(arr, partial(open_unit, mesc_path)) and view._built is built
        host._mesc_overlay_ghosts = False
        assert len(view.contours("MUnit_4 snapshot")) == 4
        assert len(view.contours("MUnit_0 slice 4")) == 2
        assert view.contours("MUnit_1 mean") == []
        view.close()
        assert not view.is_open
    finally:
        view.cleanup()
        iw.close()
        arr.close()
